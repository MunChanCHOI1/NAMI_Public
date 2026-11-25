"""Candidate sensitivity probe: does an acceptance predictor separate counter-offers?

For each synthetic public negotiation state the probe asks, per candidate, whether the
supplier would accept it unchanged in its next response, then compares the predictions
with what the hidden-parameter simulator actually does for that candidate.

Candidate ladders are placed from public information only (relative to the supplier's
latest offer), so their position carries no hint of the hidden acceptance threshold.

Run through `negotiation/run_probe.py`. Development seeds only (1000-1999).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from . import metrics as M
from .backends import ConstantBackend, HeuristicBackend, KeywordBackend, SystemOneBackend, ZoroHttpBackend, ZoroNativeBackend
from .contracts import BuyerPrivate, Candidate, PublicState, SupplierHidden, Terms
from .economics import HardConstraintViolation, classify_authority, saving_vs_alternative
from .provenance import (backend_execution_config, json_safe, provenance, reasoning_requested, request_sha256,
                         reset_response, response_snapshot, validate_predictions)
from .requests import FINAL_OFFER_QID, VARIANTS, build_acceptance_request, final_offer_question, public_view
from .simulator import (PRICE_STEP, default_buyer_private, is_last_chance, make_supplier,
                        state_after_scripted_rounds, would_accept)

DEV_SEED_BASE = 1000
LADDERS = ("price", "payment", "delivery", "mixed")
ORDERED_LADDERS = ("price", "payment", "delivery")   # one condition varies; rank is meaningful
# (price cut, delivery days or None for the supplier's quoted lead time, payment days).
# Which of these the supplier prefers depends on its hidden financing and expedite costs.
MIXED = ((0.035, None, 60), (0.05, None, 0), (0.02, 14, 60), (0.04, 14, 0), (0.065, None, 0))
PRICE_CUTS = (0.08, 0.06, 0.04, 0.025, 0.01)     # deepest cut first: least favourable for the supplier
PAYMENT_DAYS = (60, 45, 30, 15, 0)
DELIVERY_DAYS = (10, 14, 18, 21)
PAYMENT_LADDER_CUT, DELIVERY_LADDER_CUT = 0.045, 0.015   # chosen so that no two questions share terms
PAYMENT_FAST_ID, PAYMENT_CONTROL_ID = "Y5", "Y1"   # pay on delivery vs the quoted 60 days


@dataclass(frozen=True)
class ProbeState:
    state_id: str
    state: PublicState            # public
    hidden: SupplierHidden        # evaluator only
    twin_of: str | None = None    # neutral-message state with identical numbers


def _spec():
    """(style, payment_sensitive, scripted rounds, feasible). Reveal variants are added where they differ."""
    rows = [(style, sensitive, depth, True)
            for style in ("boulware", "linear", "deadline")
            for sensitive in (False, True)
            for depth in (1, 2)]
    rows += [("linear", False, 0, True), ("boulware", True, 0, True),
             ("linear", False, 1, False), ("boulware", True, 1, False)]
    return rows


def build_states() -> list[ProbeState]:
    states = []
    for index, (style, sensitive, depth, feasible) in enumerate(_spec()):
        seed = DEV_SEED_BASE + index
        tag = f"{style[:1]}{'p' if sensitive else 'n'}{depth}{'' if feasible else 'x'}"
        neutral_id = f"dev-{seed}-{tag}"
        neutral = make_supplier(seed, style, sensitive, reveal=False, feasible=feasible)
        states.append(ProbeState(neutral_id, state_after_scripted_rounds(neutral_id, neutral, depth), neutral))
        revealed = make_supplier(seed, style, sensitive, reveal=True, feasible=feasible)
        if revealed != neutral and feasible and depth > 0:
            revealed_id = neutral_id + "-said"
            states.append(ProbeState(revealed_id, state_after_scripted_rounds(revealed_id, revealed, depth),
                                     revealed, twin_of=neutral_id))
    return states


def build_ladder(state: PublicState, ladder: str) -> list[Candidate]:
    """Ordered ladders run from least to most favourable for the supplier; `mixed` has no order."""
    base = state.last_offer("supplier")
    assert base is not None
    floor_price = lambda cut: int(base.terms.unit_price * (1 - cut) // PRICE_STEP * PRICE_STEP)  # noqa: E731
    if ladder == "price":
        return [Candidate(f"P{i + 1}", replace(base.terms, unit_price=floor_price(cut)))
                for i, cut in enumerate(PRICE_CUTS)]
    if ladder == "payment":
        fixed = replace(base.terms, unit_price=floor_price(PAYMENT_LADDER_CUT))
        return [Candidate(f"Y{i + 1}", replace(fixed, payment_days=d)) for i, d in enumerate(PAYMENT_DAYS)]
    if ladder == "delivery":
        fixed = replace(base.terms, unit_price=floor_price(DELIVERY_LADDER_CUT))
        return [Candidate(f"D{i + 1}", replace(fixed, delivery_days=d)) for i, d in enumerate(DELIVERY_DAYS)]
    if ladder == "mixed":
        return [Candidate(f"M{i + 1}", Terms(floor_price(cut), delivery or base.terms.delivery_days, payment))
                for i, (cut, delivery, payment) in enumerate(MIXED)]
    raise ValueError(f"unknown ladder {ladder!r}")


def evaluator_facts(probe: ProbeState, candidate: Candidate, private: BuyerPrivate) -> dict:
    """Ground truth and our economics for one candidate. Never passed to a backend."""
    try:
        saving = saving_vs_alternative(candidate.terms, probe.state.quantity, private)
        authority = classify_authority(candidate.terms, probe.state.quantity, private)
    except HardConstraintViolation:
        saving, authority = 0, "forbidden"
    return {"truth": would_accept(probe.state, probe.hidden, candidate.terms),
            "saving": saving, "authority": authority}


def run(backend, ladders=LADDERS, max_states: int | None = None, variant_states: int = 4,
        log=lambda message: None, on_call=lambda call: None, source_provenance: dict | None = None) -> dict:
    source_provenance = source_provenance or provenance({
        "backend": backend.kind, **backend.describe().get("configuration", {}),
        "probe": {"ladders": list(ladders), "max_states": max_states, "variant_states": variant_states},
    })
    private = default_buyer_private()
    rule = HeuristicBackend()
    states = build_states()[:max_states]
    records, variant_records, final_records, calls, errors = [], [], [], [], []
    plan = [(probe, ladder, "base") for probe in states for ladder in ladders]
    plan += [(probe, "price", variant) for probe in states[:variant_states]
             for variant in VARIANTS if variant != "base"]
    candidate_requests = len(plan)
    final_supported = getattr(backend, "supports_final_offer", backend.is_model)
    if final_supported:
        plan += [(probe, "final_offer", "base") for probe in states]
    complete = True
    interrupted = False
    for number, (probe, ladder, variant) in enumerate(plan, 1):
        final_question = ladder == "final_offer"
        candidates = [] if final_question else build_ladder(probe.state, ladder)
        request = ({"state": public_view(probe.state), "questions": {FINAL_OFFER_QID: final_offer_question()}}
                   if final_question else build_acceptance_request(probe.state, candidates, variant))
        group = probe.twin_of or probe.state_id
        call = {"state_id": probe.state_id, "group_id": group, "ladder": ladder, "variant": variant,
                "questions": len(request["questions"]), "request_sha256": request_sha256(request),
                "request": json_safe(request)}
        started = time.perf_counter()
        predicted = None
        try:
            reset_response(backend)
            predicted = backend.predict(probe.state, candidates, request)
            # Validate the entire response before committing any candidate row.
            values = validate_predictions(predicted, request["questions"])
            call.update(status="ok", predictions=values)
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - retain failed attempts, no invented values
            interrupted = isinstance(error, KeyboardInterrupt)
            call.update(status="interrupted" if interrupted else "error",
                        error=f"{type(error).__name__}: {error}")
            if predicted is not None:
                call["invalid_predictions"] = json_safe(predicted)
            errors.append(call)
            complete = False
        call["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        call["response"] = response_snapshot(backend)
        calls.append(call)
        on_call(call)
        if "error" in call:
            log(f"[{number}/{len(plan)}] {probe.state_id} {ladder}/{variant} FAILED {call['error']}")
            if interrupted:
                break
            continue
        if final_question:
            final_records.append({"state_id": probe.state_id, "group_id": group,
                                  "p": values[FINAL_OFFER_QID], "truth": is_last_chance(probe.state, probe.hidden)})
        else:
            rule_predicted = rule.predict(probe.state, candidates, request) if variant == "base" else {}
            for rank, candidate in enumerate(candidates):
                record = {"state_id": probe.state_id, "group_id": group, "ladder": ladder, "variant": variant,
                          "candidate_id": candidate.candidate_id, "rank": rank,
                          "terms": asdict(candidate.terms), "p": values[candidate.candidate_id]}
                if variant == "base":
                    record.update(evaluator_facts(probe, candidate, private))
                    record["p_rule"] = rule_predicted[candidate.candidate_id]
                    record["p_flat"] = 0.5
                    records.append(record)
                else:
                    variant_records.append(record)
        log(f"[{number}/{len(plan)}] {probe.state_id} {ladder}/{variant} {call['latency_ms']} ms")

    result = {
        "schema_version": 2,
        "protocol_version": "20261116_grouped_candidates_and_separate_final_offer",
        "experiment": "candidate_sensitivity_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.describe(),
        "provenance": source_provenance,
        "evidence_class": ("real_model_probe" if backend.is_model else "cpu_reference_not_model_evidence"),
        "complete": complete, "interrupted": interrupted,
        "scope": {"synthetic_supplier_simulator": True, "development_seeds_only": True,
                  "states": len(states), "planned_requests": len(plan), "ladders": list(ladders),
                  "supplier_groups": len({p.twin_of or p.state_id for p in states}),
                  "planned_candidate_requests": candidate_requests,
                  "planned_final_offer_requests": len(states) if final_supported else 0,
                  "python": platform.python_version()},
        "calls": calls,
        "errors": errors,
    }
    final_calls = [c for c in calls if c["ladder"] == "final_offer"]
    result["final_offer_probe"] = {
        "supported": final_supported,
        "complete": len(final_records) == len(states) if final_supported else None,
        "status": ("complete" if len(final_records) == len(states) else "partial") if final_supported else "unsupported",
        "note": "Separate diagnostic; not part of the candidate route thresholds.",
        "questions_scored": len(final_records), "attempted_requests": len(final_calls),
        "metrics": {"auc": M.auc(final_records), **M.calibration(final_records)} if final_records else None,
        "records": [{k: v for k, v in r.items() if k != "truth"} for r in final_records],
    }
    if records:
        twins = {p.state_id: p.twin_of for p in states if p.twin_of}
        latencies = sorted(c["latency_ms"] for c in calls if c["status"] == "ok" and c["ladder"] != "final_offer")
        summary = {
            "questions_scored": len(records),
            "discrimination": M.discrimination(records),
            "reference_rule_discrimination": M.discrimination(records, key="p_rule"),
            "by_subset": {
                name: {"questions": len(subset), "model": M.discrimination(subset),
                       "rule": M.discrimination(subset, key="p_rule")}
                for name, subset in (
                    ("mixed_sets", [r for r in records if r["ladder"] == "mixed"]),
                    ("states_with_informative_statement", [r for r in records if r["state_id"] in twins]),
                ) if subset
            },
            "spread": M.spread(records),
            "monotonicity": M.monotonicity([r for r in records if r["ladder"] in ORDERED_LADDERS]),
            "calibration": M.calibration(records),
            "decision_value": M.decision_value(records),
            "message_sensitivity": M.message_sensitivity(records, twins, PAYMENT_FAST_ID, PAYMENT_CONTROL_ID),
            "invariance": {"meaningful_for_this_backend": backend.is_model,
                           **M.invariance(records, variant_records)},
            "latency_ms_per_request": {"min": latencies[0], "median": latencies[len(latencies) // 2],
                                       "max": latencies[-1]} if latencies else None,
        }
        result["metrics"] = summary
        result["verdict"] = M.verdict(summary, backend.is_model, complete)
    result["records"] = [{k: v for k, v in r.items() if k not in ("truth", "saving", "authority", "p_rule", "p_flat")}
                         for r in records + variant_records]
    result["evaluator_only"] = [{k: r[k] for k in ("state_id", "group_id", "ladder", "candidate_id", "truth", "saving",
                                                   "authority", "p_rule")} for r in records]
    result["final_offer_evaluator_only"] = final_records
    return result


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("constant", "heuristic", "keyword", "zoro", "zoro-http", "systemone"), default="heuristic")
    parser.add_argument("--original-checkpoint")
    parser.add_argument("--exported-checkpoint")
    parser.add_argument("--official-source-dir", default=".cache/trida-stack")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--decision-temperature", type=float)
    parser.add_argument("--url", help="base URL of an official decide server, e.g. http://localhost:30001")
    parser.add_argument("--model", default="zoro-1")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--reasoning", action="store_true",
                        help="zoro-http only: the server writes its reasoning before each decision. The reasoning "
                             "is a sampled generation, so a reasoning run is ONE sample, not a repeatable "
                             "estimate; it is recorded as its own configuration (default: off)")


def make_backend(args):
    reasoning = reasoning_requested(args)   # SystemExit unless the backend is zoro-http
    if args.backend == "constant":
        return ConstantBackend()
    if args.backend == "heuristic":
        return HeuristicBackend()
    if args.backend == "keyword":
        return KeywordBackend()
    if args.backend == "zoro-http":
        if not args.url:
            raise SystemExit("--url is required for --backend zoro-http")
        # The keyword is passed only when set, so a default call is exactly the former one.
        return ZoroHttpBackend(args.url, args.official_source_dir, args.decision_temperature,
                               args.request_timeout, **({"reasoning": True} if reasoning else {}))
    if args.backend == "systemone":
        if not args.url:
            raise SystemExit("--url is required for --backend systemone")
        return SystemOneBackend(args.url, args.model, args.request_timeout)
    if not args.original_checkpoint or not args.exported_checkpoint:
        raise SystemExit("--backend zoro needs --original-checkpoint and --exported-checkpoint (docs/GPU_RUNBOOK.md)")
    return ZoroNativeBackend(args.original_checkpoint, args.exported_checkpoint, args.official_source_dir,
                             device=args.device, dtype=args.dtype, max_input_tokens=args.max_input_tokens,
                             decision_temperature=args.decision_temperature)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate sensitivity probe for the negotiation agent.")
    add_backend_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ladders", nargs="+", choices=LADDERS, default=list(LADDERS))
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--variant-states", type=int, default=4, help="states that also get wording variants")
    parser.add_argument("--quick", action="store_true", help="12 states: price/mixed plus separate final-offer questions")
    parser.add_argument("--no-records", action="store_true", help="write the summary without per-candidate rows")
    args = parser.parse_args(argv)
    if args.quick:
        args.ladders, args.max_states, args.variant_states = ["price", "mixed"], 12, 0
    if args.max_states is not None and args.max_states < 1 or args.variant_states < 0:
        parser.error("--max-states must be positive and --variant-states must be non-negative")
    settings = {**backend_execution_config(args),
                "probe": {"ladders": args.ladders, "max_states": args.max_states,
                          "variant_states": args.variant_states, "final_offer_separate": True}}
    source_provenance = provenance(settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calls_path = args.output.with_suffix(args.output.suffix + ".calls.jsonl")
    result = {"schema_version": 2, "experiment": "candidate_sensitivity_probe", "complete": False,
              "status": "initializing", "provenance": source_provenance, "calls_file": str(calls_path)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with calls_path.open("w", encoding="utf-8") as stream:
        def save_call(call):
            stream.write(json.dumps(call, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        try:
            backend = make_backend(args)
        except (Exception, KeyboardInterrupt) as error:  # preserve initialization failure without model claims
            result.update(status="initialization_failed", errors=[{"stage": "backend_initialization",
                          "error": f"{type(error).__name__}: {error}"}])
        else:
            result = run(backend, tuple(args.ladders), args.max_states, args.variant_states,
                         log=lambda message: print(message, file=sys.stderr, flush=True), on_call=save_call,
                         source_provenance=source_provenance)
            result["backend"] = backend.describe()
            result["calls_file"] = str(calls_path)
    if args.no_records:
        for key in ("records", "evaluator_only", "final_offer_evaluator_only", "calls"):
            result.pop(key, None)
        result.get("final_offer_probe", {}).pop("records", None)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verdict = result.get("verdict", {})
    print(f"{args.output}  complete={result['complete']}  route={verdict.get('route')}  "
          f"model_evidence={verdict.get('counts_as_model_evidence')}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
