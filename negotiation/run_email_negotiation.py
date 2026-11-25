"""Full negotiations where the supplier writes free text and the language layer reads it.

Compares each case with the same case negotiated through structured fields, so any
difference in outcome comes from reading and writing.

  python negotiation/run_email_negotiation.py --llm rule --cases 60 --output runs/language/email_rule.json
  python negotiation/run_email_negotiation.py --llm claude --cases 30 --output runs/language/email_claude.json
  python negotiation/run_email_negotiation.py --llm local --llm-model <name> --llm-url http://127.0.0.1:8000/v1 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent.backends import KeywordBackend  # noqa: E402
from negotiation_agent.email_loop import EmailSupplier  # noqa: E402
from negotiation_agent.evaluation import make_cases  # noqa: E402
from negotiation_agent.language import language_model_for  # noqa: E402
from negotiation_agent.policy import PolicyConfig  # noqa: E402
from negotiation_agent.session import TERMINAL, NegotiationSession, SimulatedSupplier  # noqa: E402
from negotiation_agent.simulator import default_buyer_private, initial_state  # noqa: E402
from negotiation_agent.system1 import gate_for  # noqa: E402


def negotiate(case, private, backend, supplier) -> NegotiationSession:
    session = NegotiationSession(initial_state(case.case_id, case.hidden), private, backend, supplier,
                                 PolicyConfig(), gate=gate_for(backend))
    for _ in range(80):
        if session.status in TERMINAL or session.status == "needs_review":
            break
        if session.status == "awaiting_approval":
            session.approve(session.pending.request_id, "evaluation-approver")
            continue
        session.step()
    return session


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["rule", "claude", "openai", "local", "auto"], default="rule")
    p.add_argument("--llm-model")
    p.add_argument("--llm-url")
    p.add_argument("--cases", type=int, default=60)
    p.add_argument("--output")
    args = p.parse_args(argv)
    lm = language_model_for(args.llm, args.llm_model, args.llm_url)
    private = default_buyer_private()
    rows, outcomes, drafts = [], Counter(), Counter()
    for case in make_cases("dev", args.cases):
        base = negotiate(case, private, KeywordBackend(), SimulatedSupplier(case.hidden))
        email = EmailSupplier(case.hidden, lm, case.seed)
        run = negotiate(case, private, KeywordBackend(), email)
        for e in email.log:
            outcomes[e["outcome"]] += 1
            drafts[e["draft_source"]] += 1
        b, r = base.outcome(), run.outcome()
        rows.append({"case_id": case.case_id, "structured": {"status": base.status, "saving": b["realized_saving_vs_alternative"]},
                     "text": {"status": run.status, "saving": r["realized_saving_vs_alternative"],
                              "authority_violations": r.get("authority_violations", 0)},
                     "same_outcome": base.status == run.status and b["agreement"] == r["agreement"],
                     "turns": email.log})
    n = len(rows)
    summary = {
        "cases": n, "language_model": lm.describe(),
        "same_outcome_as_structured": sum(r["same_outcome"] for r in rows),
        "supplier_messages": sum(outcomes.values()),
        "automated_correct": outcomes["automated_correct"], "automated_wrong": outcomes["automated_wrong"],
        "person_typed": outcomes["person_typed"],
        "mean_saving_structured": round(sum(r["structured"]["saving"] or 0 for r in rows) / n),
        "mean_saving_text": round(sum(r["text"]["saving"] or 0 for r in rows) / n),
        "authority_violations_text": sum(r["text"]["authority_violations"] or 0 for r in rows),
        "buyer_drafts": dict(drafts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
