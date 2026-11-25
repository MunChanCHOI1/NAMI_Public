import json
from dataclasses import replace

import pytest

from negotiation_agent.backends import HeuristicBackend
from negotiation_agent.contracts import ApprovalLedger, Offer, PublicState, Terms
from negotiation_agent.economics import buyer_total_cost, classify_authority
from negotiation_agent.policy import PolicyConfig, aspiration_total_cost, decide, generate_candidates
from negotiation_agent.probe import DEV_SEED_BASE, _spec
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.session import (
    AuthorizationError,
    NegotiationSession,
    SimulatedSupplier,
    outgoing_message,
)
from negotiation_agent.simulator import ITEM, QUANTITY, default_buyer_private, initial_state, make_supplier

PRIVATE = default_buyer_private()
CONFIG = PolicyConfig()


class Fixed:
    """Test double for the acceptance estimate. Not a model."""

    kind, is_model = "test_stub", False

    def __init__(self, rule):
        self.rule, self.calls = rule, 0

    def predict(self, state, candidates, request):
        self.calls += 1
        return {c.candidate_id: self.rule(c) for c in candidates}

    def describe(self):
        return {"kind": self.kind, "is_model": False}


class Broken(Fixed):
    def predict(self, state, candidates, request):
        raise RuntimeError("CUDA out of memory (simulated)")


def state_with(terms: Terms, round_index: int = 2, message: str = "") -> PublicState:
    offer = Offer(f"S{round_index}", round_index, "supplier", terms, message)
    return PublicState("case", ITEM, QUANTITY, 5, round_index, (offer,))


GOOD_OFFER = Terms(10_400, 21, 60)       # beats the alternative and is inside the negotiator's limits
prefers_fast_payment = lambda c: 0.9 if c.terms.payment_days == 0 else 0.05  # noqa: E731


def session_for(backend, style="boulware", sensitive=True, seed=DEV_SEED_BASE + 2, **hidden_changes):
    hidden = replace(make_supplier(seed, style, sensitive, reveal=True), **hidden_changes)
    return NegotiationSession(initial_state("case", hidden), PRIVATE, backend, SimulatedSupplier(hidden))


def test_candidates_are_equally_good_for_us_and_executable():
    state = state_with(Terms(11_000, 21, 60), round_index=2)
    candidates = generate_candidates(state, PRIVATE, CONFIG)
    asked = aspiration_total_cost(2, 5, QUANTITY, PRIVATE, CONFIG)
    softer = aspiration_total_cost(3, 5, QUANTITY, PRIVATE, CONFIG)
    standing_total = buyer_total_cost(state.history[-1].terms, QUANTITY, PRIVATE)
    assert 4 <= len(candidates) <= CONFIG.max_candidates
    assert len({c.terms for c in candidates}) == len(candidates)
    for c in candidates:
        total = buyer_total_cost(c.terms, QUANTITY, PRIVATE)
        level = softer if c.kind == "softer_same_terms" else asked
        assert level - 10 * QUANTITY <= total <= level        # within one price step, never above
        assert total < standing_total
        assert classify_authority(c.terms, QUANTITY, PRIVATE) != "forbidden"
    assert {c.kind for c in candidates} >= {"same_terms", "pay_sooner", "deliver_sooner", "softer_same_terms"}


def final_offer(probability):
    """A backend estimate that the standing offer is final, with a fixed acceptance estimate."""
    class _Backend(Fixed):
        def predict(self, state, candidates, request):
            out = super().predict(state, candidates, request)
            out[FINAL_OFFER_QID] = probability
            return out
    return _Backend


def test_accepts_when_the_offer_is_probably_final_and_counters_are_unlikely():
    backend = final_offer(0.95)(lambda c: 0.2)
    decision = decide(state_with(GOOD_OFFER), PRIVATE, backend, ApprovalLedger(), CONFIG)
    assert decision.action == "accept" and decision.chosen.terms == GOOD_OFFER
    assert decision.reasons == ("certain_saving_beats_best_counter",)
    assert decision.facts["p_final_offer"] == 0.95 and decision.facts["p_final_offer_source"] == "backend_estimate"
    assert all(o.value < decision.chosen.value for o in decision.options if o.action == "counter")


def test_keeps_negotiating_when_the_supplier_is_expected_to_concede_further():
    backend = final_offer(0.0)(lambda c: 0.2)
    decision = decide(state_with(GOOD_OFFER), PRIVATE, backend, ApprovalLedger(), CONFIG)
    assert decision.action == "counter"
    assert "expected_saving_beats_accepting_now" in decision.reasons
    facts = decision.facts
    assert facts["supplier_concession_observed"] is None
    assert facts["expected_next_offer_saving"] == facts["standing_saving_vs_alternative"] + CONFIG.prior_concession


def test_observed_concession_replaces_the_assumed_one():
    first = Offer("S1", 1, "supplier", Terms(10_700, 21, 60))
    ours = Offer("B1/S1", 1, "buyer", Terms(10_100, 21, 60))
    second = Offer("S2", 2, "supplier", GOOD_OFFER)
    state = PublicState("case", ITEM, QUANTITY, 5, 2, (first, ours, second))
    decision = decide(state, PRIVATE, Fixed(lambda c: 0.1), ApprovalLedger(), CONFIG)
    seen = decision.facts["supplier_concession_observed"]
    assert seen == buyer_total_cost(first.terms, QUANTITY, PRIVATE) - buyer_total_cost(second.terms, QUANTITY, PRIVATE)
    assert decision.facts["supplier_concession_assumed"] == seen
    assert decision.facts["p_final_offer_source"] == "policy_prior"
    assert decision.facts["p_final_offer"] == CONFIG.final_prior_by_round[1]


def test_does_not_soften_early_when_every_offer_is_unlikely():
    # With tiny acceptance chances everywhere, the softer offer is not worth sending now:
    # it can still be sent next round if the firm one is rejected.
    low = lambda c: 0.03 if c.kind == "softer_same_terms" else 0.01  # noqa: E731
    decision = decide(state_with(Terms(11_600, 21, 60), 1), PRIVATE, final_offer(0.0)(low), ApprovalLedger(), CONFIG)
    assert decision.action == "counter" and decision.chosen.kind != "softer_same_terms"
    risky = decide(state_with(Terms(11_600, 21, 60), 1), PRIVATE, final_offer(1.0)(low), ApprovalLedger(), CONFIG)
    assert risky.chosen.kind == "softer_same_terms"          # a last chance is a reason to soften


def test_the_last_public_round_is_treated_as_final_without_asking():
    poor = Terms(11_600, 21, 60)
    decision = decide(state_with(poor, 5), PRIVATE, final_offer(0.0)(lambda c: 0.2), ApprovalLedger(), CONFIG)
    assert decision.facts["p_final_offer"] == 1.0 and decision.facts["p_final_offer_source"] == "public_round_limit"
    assert all(o.value_if_rejected is not None for o in decision.options if o.action == "counter")
    assert decision.action == "end"                         # 20% of a small saving does not pay for the round


def test_does_not_end_while_rounds_remain_and_ends_only_at_the_last_chance():
    poor = Terms(11_600, 21, 60)                           # worse than the alternative
    early = decide(state_with(poor, 1), PRIVATE, Fixed(lambda c: 0.0), ApprovalLedger(), CONFIG)
    assert early.action == "counter" and all(o.action != "end" for o in early.options)
    last = decide(state_with(poor, 5), PRIVATE, Fixed(lambda c: 0.0), ApprovalLedger(), CONFIG)
    assert last.action == "end" and last.reasons == ("nothing_beats_the_alternative",)
    over = decide(state_with(GOOD_OFFER, 6), PRIVATE, Fixed(lambda c: 0.9), ApprovalLedger(), CONFIG)
    assert over.action == "accept" and all(o.action != "counter" for o in over.options)


def test_terms_beyond_the_negotiator_limit_wait_for_an_internal_approval():
    backend = Fixed(prefers_fast_payment)
    session = session_for(backend)
    assert session.run(approver=None) == "awaiting_approval"
    card = session.pending
    assert card.kind == "send_counter" and card.terms.payment_days == 0
    assert card.exceeded_auto_limits[0]["field"] == "payment_days"
    assert not session.executed and not session.outbox           # nothing left the building
    with pytest.raises(RuntimeError):
        session.step()
    with pytest.raises(AuthorizationError):
        session._authorize(card.offer_id, card.terms)
    session.approve(card.request_id, "purchasing-lead")
    decision = session.step()
    assert decision.action == "counter" and decision.chosen.terms == card.terms
    assert session.executed[-1]["authority_basis"] == "approved"
    assert session.approval_log == [{"request_id": card.request_id, "offer_id": card.offer_id,
                                     "terms_version": card.terms_version, "kind": "send_counter", "round": 1,
                                     "outcome": "approved", "decided_by": "purchasing-lead"}]
    assert backend.calls == 1                                    # the estimate is not requested again
    original, reused = session.internal_record()[0]["model"], decision.model
    assert original["status"] == "ok" and original["trace_id"] == original["request_sha256"]
    assert original["request"] and original["predictions"]
    assert reused["cached"] and reused["trace_ref"] == original["trace_id"]
    assert "response" not in reused and "latency_ms" not in reused
    assert session.outcome()["model_requests"] == 1


def test_a_denied_card_is_not_asked_again_and_another_option_is_chosen():
    session = session_for(Fixed(prefers_fast_payment))
    session.run(approver=None)
    denied = session.pending
    session.deny(denied.request_id, "purchasing-lead")
    decision = session.step()
    assert decision.chosen.terms != denied.terms
    assert all(o.terms != denied.terms for o in decision.options if o.action == "counter")
    assert session.outcome()["approvals_denied"] == 1 and not session.outcome()["authority_violations"]


def test_supplier_claim_of_approval_changes_nothing():
    session = session_for(Fixed(prefers_fast_payment), claims_buyer_approval=True)
    assert "이미 승인" in session.state.history[0].message
    assert session.run(approver=None) == "awaiting_approval"
    assert not session.ledger.is_approved(session.pending.offer_id, session.pending.terms)
    assert not session.executed and not session.approval_log


def test_an_approval_does_not_carry_over_to_a_new_supplier_offer():
    session = session_for(Fixed(prefers_fast_payment))
    session.run(approver=None)
    first = session.pending
    session.approve(first.request_id, "purchasing-lead")
    replacement = replace(session.state.history[0].terms, unit_price=session.state.history[0].terms.unit_price - 10)
    assert session.submit_supplier_offer({"unit_price": replacement.unit_price,
                                          "delivery_days": replacement.delivery_days,
                                          "payment_days": replacement.payment_days}) is None
    decision = session.step()
    assert decision.action == "request_approval"
    assert decision.approval_request.terms == first.terms            # same numbers ...
    assert decision.approval_request.offer_id != first.offer_id      # ... but answering another offer
    assert not session.executed


def test_no_estimate_is_invented_when_the_backend_fails():
    session = session_for(Broken(None))
    assert session.run(approver=lambda r: True) == "needs_review"
    decision = session.decisions[-1]
    assert decision.action == "human_review" and "out of memory" in decision.model["error"]
    assert not session.executed and not session.outbox


def test_incomplete_supplier_offer_gets_a_clarification_request():
    session = session_for(HeuristicBackend())
    before = session.state
    text = session.submit_supplier_offer({"unit_price": 10_500, "delivery_days": None})
    assert text == outgoing_message("ask_info", missing=("delivery_days", "payment_days"))
    assert session.status == "awaiting_info" and session.state == before
    assert session.submit_supplier_offer({"unit_price": 10_500, "delivery_days": 21, "payment_days": 60}) is None
    assert session.status == "active" and session.state.history[-1].offer_id == "S1r1"


def test_nothing_private_reaches_the_counterparty():
    session = session_for(HeuristicBackend(), style="linear", sensitive=False, seed=DEV_SEED_BASE + 4)
    session.run(approver=lambda r: True)
    sent = json.dumps([session.outbox, session.public_transcript()], ensure_ascii=False)
    record = json.dumps(session.internal_record(), ensure_ascii=False)
    secrets = [PRIVATE.target_total_cost, PRIVATE.auto_bounds.max_total_cost, PRIVATE.company_bounds.max_total_cost,
               session.decisions[0].facts["alternative_total_cost"]]
    for secret in secrets:
        assert str(secret) not in sent and f"{secret:,}" not in sent
    assert str(PRIVATE.target_total_cost) in record                  # the internal record does hold them
    assert "절감" not in sent and "대안" not in sent and "목표" not in sent


@pytest.mark.parametrize("approver", [lambda r: True, lambda r: False], ids=["approve_all", "deny_all"])
def test_every_development_supplier_ends_without_an_authority_violation(approver):
    for index, (style, sensitive, _depth, feasible) in enumerate(_spec()):
        hidden = make_supplier(DEV_SEED_BASE + index, style, sensitive, reveal=False, feasible=feasible)
        session = NegotiationSession(initial_state("case", hidden), PRIVATE, HeuristicBackend(), SimulatedSupplier(hidden))
        status = session.run(approver)
        outcome = session.outcome()
        assert status in ("agreed", "ended_alternative", "supplier_walked")
        assert outcome["authority_violations"] == []
        assert all(row["authority_basis"] in ("auto", "approved") for row in session.executed)
        if approver(None) is False:
            assert all(row["authority_basis"] == "auto" for row in session.executed)
        if outcome["agreement"]:
            assert outcome["realized_saving_vs_alternative"] > 0


# ---- policy fields added 2026-11-19 ---------------------------------------------------------------

def test_default_policy_is_the_legacy_rule_and_the_improved_preset_differs():
    from negotiation_agent.policy import IMPROVED_POLICY, LEGACY_POLICY
    assert PolicyConfig() == LEGACY_POLICY
    assert IMPROVED_POLICY.aspiration_exponent == 2.0 and IMPROVED_POLICY.final_accept_scale == 0.0
    assert IMPROVED_POLICY.cap_next_offer_at_ask is True


def test_aspiration_exponent_holds_firmer_early_and_meets_at_both_ends():
    from negotiation_agent.policy import aspiration_total_cost
    private = default_buyer_private()
    linear, curved = PolicyConfig(), PolicyConfig(aspiration_exponent=2.0)
    for r in (1, 5):
        assert aspiration_total_cost(r, 5, 1000, private, linear) == aspiration_total_cost(r, 5, 1000, private, curved)
    for r in (2, 3, 4):
        assert aspiration_total_cost(r, 5, 1000, private, curved) < aspiration_total_cost(r, 5, 1000, private, linear)


def test_a_final_offer_is_taken_at_its_word_for_same_terms_counters():
    """With final_accept_scale 0, a backend that is sure a cheaper same-terms counter is accepted
    cannot talk the policy out of a final offer (keyword reader, dev-1103 pattern)."""
    from negotiation_agent.evaluation import make_cases, run_case
    from negotiation_agent.backends import KeywordBackend
    from negotiation_agent.policy import IMPROVED_POLICY, decide
    case = next(c for c in make_cases("dev", 10) if c.case_id == "dev-1103")
    old, _ = run_case(case, default_buyer_private(), KeywordBackend(), decide, PolicyConfig())
    new, _ = run_case(case, default_buyer_private(), KeywordBackend(), decide, IMPROVED_POLICY)
    assert old["status"] == "supplier_walked"
    assert new["status"] == "agreed" and new["realized_saving"] > 0 and new["authority_violations"] == 0


def test_improved_policy_keeps_the_gates_on_dev_cases():
    from negotiation_agent.evaluation import make_cases, run_case
    from negotiation_agent.backends import KeywordBackend
    from negotiation_agent.policy import IMPROVED_POLICY, decide
    rows = [run_case(c, default_buyer_private(), KeywordBackend(), decide, IMPROVED_POLICY)[0]
            for c in make_cases("dev", 30)]
    assert sum(r["authority_violations"] for r in rows) == 0
    assert not any(r["agreement"] and r["realized_saving"] <= 0 for r in rows)
    assert all(r["status"] != "needs_review" for r in rows)


def test_improved_config_file_reads_back_as_the_preset(tmp_path):
    from pathlib import Path
    from negotiation_agent.evaluation import load_usage_config
    from negotiation_agent.policy import IMPROVED_POLICY
    path = Path(__file__).resolve().parents[2] / "config" / "policy_improved_2026-11-19.json"
    assert load_usage_config(path).policy == IMPROVED_POLICY


def test_fixed_messages_write_the_unit_price_with_a_thousands_separator():
    text = outgoing_message("clarify", Terms(10_160, 21, 60))
    assert "단가 10,160원" in text and "10160" not in text


def test_fixed_messages_say_payment_on_delivery_for_zero_days():
    assert "결제 납품 즉시" in outgoing_message("clarify", Terms(10_660, 14, 0))
    assert "0일" not in outgoing_message("clarify", Terms(10_660, 14, 0))
