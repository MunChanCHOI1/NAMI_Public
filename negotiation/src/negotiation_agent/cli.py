"""Run synthetic negotiations end to end and print what the agent computed, estimated and did."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .policy import ApprovalRequest, PolicyConfig
from .probe import DEV_SEED_BASE, _spec, add_backend_arguments, make_backend
from .session import NegotiationSession, SimulatedSupplier, terms_text
from .simulator import default_buyer_private, initial_state, make_supplier

ACTION = {"accept": "수락", "counter": "역제안", "request_approval": "내부 승인 요청", "end": "협상 종료",
          "human_review": "사람 검토 요청"}
FIELD = {"unit_price": "단가", "delivery_days": "납기", "payment_days": "결제일", "total_cost": "환산 총비용"}


def _won(value) -> str:
    return "-" if value is None else f"{value:,.0f}원"


def card_text(request: ApprovalRequest) -> str:
    limits = ", ".join(f"{FIELD[e['field']]} {e['value']:,} (자동 한도 {e['auto_limit']:,})"
                       for e in request.exceeded_auto_limits)
    kind = "상대 제안 수락" if request.kind == "accept_offer" else "역제안 전송"
    estimate = "" if request.p_accept is None else f", 수락 추정 {request.p_accept:.0%}"
    return (f"[승인 카드 {request.request_id}] {kind}: {terms_text(request.terms)}\n"
            f"  담당자 한도 초과: {limits}\n"
            f"  환산 총비용 {_won(request.total_cost)}, 대안 대비 절감 {_won(request.saving_vs_alternative)}, "
            f"목표 대비 {_won(request.gap_to_target)}{estimate}\n"
            f"  이 승인은 제안 {request.offer_id}의 이 조건(버전 {request.terms_version})에만 적용됩니다.")


def print_decision(decision, out=sys.stdout) -> None:
    facts = decision.facts
    print(f"\n── 라운드 {decision.round_index}: {ACTION[decision.action]} ({', '.join(decision.reasons)})", file=out)
    print(f"  [계산] 상대 제안 {facts['standing_offer_id']}: 환산 총비용 {_won(facts['standing_total_cost'])}, "
          f"대안 대비 절감 {_won(facts['standing_saving_vs_alternative'])}, 권한 {facts['standing_authority']}", file=out)
    label = decision.model.get("label", "")
    if "p_final_offer" in facts:
        seen = facts["supplier_concession_observed"]
        print(f"  [추정] 상대 제안이 최종안일 가능성 {facts['p_final_offer']:.0%} ({facts['p_final_offer_source']}), "
              f"최근 양보 폭 {_won(seen) if seen is not None else '관찰 전'}, "
              f"다음 제안 예상 절감 {_won(facts['expected_next_offer_saving'])}", file=out)
    for option in decision.options:
        mark = "→" if option is decision.chosen else " "
        terms = terms_text(option.terms) if option.terms else "대안 견적 선택"
        estimate = "" if option.p_accept is None else f"  [{label}] 수락 {option.p_accept:.0%}"
        print(f"  {mark} {ACTION[option.action]:4s} {option.kind:24s} {terms}  절감 {_won(option.saving)}"
              f"  권한 {option.authority or '-'}{estimate}  비교값 {_won(option.value)}", file=out)
    if decision.model.get("error"):
        print(f"  [추정 실패] {decision.model['error']} — 임의 값을 쓰지 않고 사람에게 넘깁니다.", file=out)
    if decision.approval_request:
        print("  " + card_text(decision.approval_request).replace("\n", "\n  "), file=out)


def _ask(request: ApprovalRequest) -> bool:
    print(card_text(request))
    return input("승인하시겠습니까? [y/N] ").strip().lower() in ("y", "yes")


def run_case(args, backend, style, sensitive, reveal, feasible, seed, quiet=False) -> dict:
    hidden = replace(make_supplier(seed, style, sensitive, reveal, feasible),
                     claims_buyer_approval=args.claims_approval)
    session = NegotiationSession(initial_state(f"dev-{seed}", hidden), default_buyer_private(), backend,
                                 SimulatedSupplier(hidden), PolicyConfig(prior_concession=args.prior_concession))
    approver = {"auto": lambda r: True, "deny": lambda r: False, "ask": _ask, "none": None}[args.approver]
    printed = 0
    while True:
        status = session.run(approver, approver_name=f"cli-{args.approver}", max_steps=1)
        if not quiet:
            for decision in session.decisions[printed:]:
                print_decision(decision)
            printed = len(session.decisions)
            if session.last_response is not None and session.executed and session.executed[-1]["action"] == "counter":
                response, session.last_response = session.last_response, None
                detail = f": {terms_text(response.offer.terms)}" if response.offer else ""
                print(f"  [공급사] {response.kind}{detail}  “{response.message}”")
        if status in ("agreed", "ended_alternative", "supplier_walked", "needs_review") \
                or (status == "awaiting_approval" and approver is None):
            break
    outcome = session.outcome()
    if not quiet:
        print(f"\n결과: {outcome['status']}, 합의 {outcome['agreement']}, 대안 대비 실현 절감 "
              f"{_won(outcome['realized_saving_vs_alternative'])}, 승인 카드 {outcome['approval_cards']}건, "
              f"권한 위반 {len(outcome['authority_violations'])}건")
    return {
        "case": {"seed": seed, "style": style, "payment_sensitive": sensitive, "reveal": reveal, "feasible": feasible},
        "backend": backend.describe(),
        "evidence_class": "real_model_run" if backend.is_model else "cpu_reference_not_model_evidence",
        "outcome": outcome,
        "public_transcript": session.public_transcript(),
        "approvals": session.approval_log,
        "internal_record": session.internal_record(),
        "evaluator_only": {"supplier_hidden": asdict(hidden)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the buyer-side negotiation agent against the synthetic supplier.")
    add_backend_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEV_SEED_BASE + 2)
    parser.add_argument("--style", choices=("boulware", "linear", "deadline"), default="boulware")
    parser.add_argument("--payment-sensitive", action="store_true")
    parser.add_argument("--reveal", action="store_true", help="the supplier states its priorities")
    parser.add_argument("--infeasible", action="store_true", help="a supplier that cannot beat the alternative")
    parser.add_argument("--claims-approval", action="store_true",
                        help="the supplier falsely says our side already approved")
    parser.add_argument("--approver", choices=("auto", "deny", "ask", "none"), default="auto")
    parser.add_argument("--prior-concession", type=int, default=PolicyConfig().prior_concession)
    parser.add_argument("--all-dev", action="store_true", help="run every development supplier and print a summary")
    args = parser.parse_args(argv)
    backend = make_backend(args)
    if args.all_dev:
        runs = [run_case(args, backend, style, sensitive, False, feasible, DEV_SEED_BASE + i, quiet=True)
                for i, (style, sensitive, _depth, feasible) in enumerate(_spec())]
        outcomes = [r["outcome"] for r in runs]
        summary = {
            "cases": len(runs),
            "status_counts": {s: sum(1 for o in outcomes if o["status"] == s) for s in sorted({o["status"] for o in outcomes})},
            "mean_realized_saving_vs_alternative": sum(o["realized_saving_vs_alternative"] for o in outcomes) / len(runs),
            "approval_cards": sum(o["approval_cards"] for o in outcomes),
            "authority_violations": sum(len(o["authority_violations"]) for o in outcomes),
            "note": "policy behaviour check on development suppliers; no baseline comparison",
        }
        result = {"backend": backend.describe(), "approver": args.approver,
                  "evidence_class": runs[0]["evidence_class"], "summary": summary,
                  "cases": [{"case": r["case"], "outcome": r["outcome"]} for r in runs]}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        result = run_case(args, backend, args.style, args.payment_sensitive, args.reveal,
                          not args.infeasible, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
