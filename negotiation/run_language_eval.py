"""Compare language models on reading supplier messages and writing replies.

Examples (from the repository root):
  python negotiation/run_language_eval.py --llm rule --output runs/language/rule.json
  python negotiation/run_language_eval.py --llm claude --output runs/language/claude.json      # ANTHROPIC_API_KEY
  python negotiation/run_language_eval.py --llm local --llm-model <name> --llm-url http://127.0.0.1:8000/v1 \
      --output runs/language/local.json                                                      # vLLM / Ollama / SGLang
  python negotiation/run_language_eval.py --compare runs/language/*.json

The most important number is `unsafe_pass`: readings that passed every code check but carry the
wrong intent or terms. Those would reach the policy without a person looking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.language import language_model_for, read_supplier_message, write_reply  # noqa: E402
from negotiation_agent.language_cases import Q, READ_CASES, WRITE_CASES  # noqa: E402


def evaluate(lm) -> dict:
    rows, latencies = [], []
    for case in READ_CASES:
        started = time.perf_counter()
        reading = read_supplier_message(lm, case.text, Q, case.previous, case.buyer)
        latencies.append(time.perf_counter() - started)
        intent_ok = reading.intent == case.intent or {reading.intent, case.intent} == {"question", "other"} or (case.intent == "offer" and case.expect_status == "needs_review"
                                                       and reading.status == "needs_review")
        terms_ok = case.terms is None or reading.terms == case.terms
        passed = reading.status == "ok"
        same_intent = reading.intent == case.intent \
            or {reading.intent, case.intent} == {"question", "other"} \
            or (case.intent == "accept" and reading.intent == "offer" and reading.terms == case.buyer)
        correct_pass = passed and same_intent and (case.terms is None or reading.terms == case.terms)
        rows.append({"case_id": case.case_id, "tags": list(case.tags), "expect_status": case.expect_status,
                     "status": reading.status, "intent": reading.intent, "expect_intent": case.intent,
                     "terms": asdict(reading.terms) if reading.terms else None,
                     "expect_terms": asdict(case.terms) if case.terms else None,
                     "intent_ok": intent_ok, "terms_ok": terms_ok,
                     "unsafe_pass": passed and not correct_pass,
                     "missed_review": case.expect_status == "needs_review" and passed,
                     "needless_review": case.expect_status == "ok" and not passed,
                     "reasons": list(reading.reasons), "says_final": reading.says_final,
                     "other_demands": list(reading.other_demands)})
    writes = []
    for case in WRITE_CASES:
        started = time.perf_counter()
        draft = write_reply(lm, case.action, case.terms, case.template, Q, case.supplier_message)
        writes.append({"case_id": case.case_id, "source": draft.source, "text": draft.text,
                       "reasons": list(draft.reasons), "rejected_text": draft.rejected_text,
                       "seconds": round(time.perf_counter() - started, 3)})
    n = len(rows)
    expect_ok = [r for r in rows if r["expect_status"] == "ok"]
    summary = {
        "read_cases": n,
        "unsafe_pass": sum(r["unsafe_pass"] for r in rows),
        "missed_review": sum(r["missed_review"] for r in rows),
        "needless_review": sum(r["needless_review"] for r in rows),
        "automated_correctly": sum(1 for r in expect_ok if r["status"] == "ok" and not r["unsafe_pass"]),
        "automatable_cases": len(expect_ok),
        "intent_accuracy": round(sum(r["intent_ok"] for r in rows) / n, 3),
        "terms_accuracy_where_labelled": round(sum(r["terms_ok"] for r in rows if r["expect_terms"])
                                               / max(1, sum(1 for r in rows if r["expect_terms"])), 3),
        "final_flag_on_final_cases": sum(1 for r in rows if "final" in r["tags"] and r["says_final"]),
        "read_latency_median_s": round(statistics.median(latencies), 3),
        "drafts_from_model": sum(1 for w in writes if w["source"] == "model"),
        "drafts_rejected_by_guard": sum(1 for w in writes if w["rejected_text"] is not None),
        "write_cases": len(writes),
    }
    return {"language_model": lm.describe(), "is_model": lm.is_model, "summary": summary,
            "reads": rows, "writes": writes}


def compare(paths: list[str]) -> None:
    keys = ["unsafe_pass", "missed_review", "needless_review", "automated_correctly", "intent_accuracy",
            "terms_accuracy_where_labelled", "final_flag_on_final_cases", "read_latency_median_s",
            "drafts_from_model", "drafts_rejected_by_guard"]
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [r["language_model"].get("model") or r["language_model"]["kind"] for r in results]
    print("metric".ljust(32) + "".join(n[:22].ljust(24) for n in names))
    for k in keys:
        print(k.ljust(32) + "".join(str(r["summary"][k]).ljust(24) for r in results))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["rule", "claude", "openai", "local", "auto"], default="rule")
    p.add_argument("--llm-model")
    p.add_argument("--llm-url")
    p.add_argument("--output")
    p.add_argument("--compare", nargs="+")
    args = p.parse_args(argv)
    if args.compare:
        compare(args.compare)
        return 0
    lm = language_model_for(args.llm, args.llm_model, args.llm_url)
    result = evaluate(lm)
    for r in result["reads"]:
        flag = "UNSAFE" if r["unsafe_pass"] else ("miss" if r["missed_review"] else ("extra" if r["needless_review"] else "ok"))
        print(f"{r['case_id']:<20} {flag:<7} status={r['status']:<12} intent={r['intent']} terms={r['terms']}")
    for w in result["writes"]:
        print(f"[write {w['case_id']}] {w['source']}: {w['text']}" + (f"  (guard: {w['reasons']})" if w["reasons"] else ""))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
