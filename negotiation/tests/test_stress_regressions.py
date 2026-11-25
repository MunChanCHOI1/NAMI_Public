"""Regressions for defects found by negotiation/stress_virtual.py (virtual adversarial suppliers)."""

from negotiation_agent.backends import KeywordBackend
from negotiation_agent.contracts import Offer, PublicState, SupplierResponse, Terms
from negotiation_agent.economics import classify_authority
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.session import NegotiationSession
from negotiation_agent.simulator import ITEM, QUANTITY, default_buyer_private
from negotiation_agent.system1 import RuleReader, System1Gate, number_check, offer_number_check

SENT = Terms(10_300, 14, 60)


def test_number_check_catches_swapped_delivery_and_payment():
    # Both numbers appear in the sent terms, but under the other field's label.
    assert number_check("확정합니다. 납기 60일, 결제 14일로 하겠습니다.", SENT, 1000)["mismatches"] == ["60일", "14일"]
    assert number_check("단가 10300원, 납기 14일, 결제 납품 후 60일 그대로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("납기 14일 결제 60일로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("결제는 납품 후 60일, 납기 14일로 확정합니다.", SENT, 1000)["ok"]
    held = offer_number_check("납기 60일, 결제 14일에 10,800원 드리겠습니다.", Terms(10_800, 14, 60), SENT, 1000)
    assert held["mismatches"] == ["60일", "14일"]


def test_number_check_reads_amounts_written_in_korean_numerals():
    assert number_check("확정합니다. 단가는 만 이천원으로 하겠습니다.", SENT, 1000)["mismatches"] == ["금액 만 이천원"]
    assert not number_check("1만 2천원으로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("단가 1만 300원 그대로 확정합니다.", SENT, 1000)["ok"]       # no false "300원"
    assert number_check("만 삼백원, 총 천삼십만원으로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("원가 부담이 있지만 그대로 확정합니다.", SENT, 1000)["ok"]   # '원가' is not an amount


def _session(counterparty, gate=True):
    state = PublicState("stress-test", ITEM, QUANTITY, 5, 1, (Offer("S1", 1, "supplier", Terms(11_000, 21, 60)),))
    return NegotiationSession(state, default_buyer_private(), KeywordBackend(), counterparty, PolicyConfig(),
                              gate=System1Gate(RuleReader()) if gate else None)


class _Swapper:
    def respond(self, state, buyer_terms):
        return SupplierResponse("accept", message=f"확정합니다. 납기 {buyer_terms.payment_days}일, "
                                                  f"결제 {buyer_terms.delivery_days}일로 하겠습니다.")

    def confirm(self, state, buyer_terms):
        return self.respond(state, buyer_terms)


def test_field_swapping_acceptance_is_not_recorded_as_agreement():
    session = _session(_Swapper())
    assert session.run(lambda request: True) == "needs_review"
    assert session.agreement is None


class _CounterWithoutTerms:
    def respond(self, state, buyer_terms):
        return SupplierResponse("counter", offer=None, message="다시 검토하겠습니다.")


def test_counter_without_terms_stops_for_review_instead_of_crashing():
    session = _session(_CounterWithoutTerms())
    assert session.run(lambda request: True) == "needs_review"
    assert session.review_reason == "supplier_reply_without_terms"
    assert session.agreement is None and session.state.current_round == 1


class _SlowConceder:
    """Concedes 10-20 KRW a round from 10,880 and accepts any counter at or above 10,600."""

    PRICES = {2: 10_880, 3: 10_860, 4: 10_850, 5: 10_840, 6: 10_840}

    def respond(self, state, buyer_terms):
        if buyer_terms.unit_price >= 10_600:
            return SupplierResponse("accept", message="제안하신 조건으로 진행하겠습니다.")
        nxt = state.current_round + 1
        offer = Offer(f"S{nxt}", nxt, "supplier", Terms(self.PRICES[nxt], buyer_terms.delivery_days,
                                                         buyer_terms.payment_days))
        return SupplierResponse("counter", offer=offer)

    def confirm(self, state, buyer_terms):
        return SupplierResponse("accept", message="네, 그대로 확정합니다.")


def test_denied_counters_fall_back_to_a_counter_inside_own_authority():
    # Before the fix: in round 3 every composition sat above the negotiator's own price limit, the
    # approver denied them, and the agent ended with two rounds left. Now it can still counter at
    # its own limit, and that counter is executed without any approval.
    session = _session(_SlowConceder())
    status = session.run(lambda request: False)
    assert status == "agreed"
    assert session.ledger._approvals == {}
    assert classify_authority(session.agreement, QUANTITY, default_buyer_private()) == "auto"
    assert session.outcome()["authority_violations"] == []
    kinds = [o.kind for d in session.decisions for o in d.options if d.chosen is o]
    assert "within_own_limit" in kinds


def test_implausible_offer_stops_for_a_person():
    from negotiation_agent.contracts import Terms
    from negotiation_agent.session import implausible_terms
    from negotiation_agent.simulator import default_buyer_private
    private = default_buyer_private()
    assert implausible_terms(Terms(1, 21, 60), private)
    assert implausible_terms(Terms(private.alternative.unit_price, 21, 100_000), private)
    assert implausible_terms(Terms(private.alternative.unit_price, 21, 60), private) == []
