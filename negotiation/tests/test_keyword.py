from dataclasses import replace

from negotiation_agent.backends import HeuristicBackend, KeywordBackend
from negotiation_agent.contracts import Candidate, Offer, Terms
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.requests import FINAL_OFFER_QID, build_decision_request
from negotiation_agent.simulator import initial_state, make_supplier


def test_keyword_only_changes_final_estimate_and_uses_latest_supplier_message():
    state = initial_state("keyword-dev", make_supplier(1015, "linear", True, True, True))
    candidates = [Candidate("P1", Terms(10500, 14, 0)), Candidate("P2", Terms(10800, 14, 60))]
    first = replace(state.history[0], message="이번 조건이 최종안입니다.")
    state = replace(state, history=(first,))
    request = build_decision_request(state, candidates)
    base = HeuristicBackend().predict(state, candidates, request)
    result = KeywordBackend().predict(state, candidates, request)
    assert {k: result[k] for k in base} == base
    assert result[FINAL_OFFER_QID] == 1.0
    next_offer = Offer("S2", 2, "supplier", Terms(10700, 14, 30), "조정한 조건을 확인해 주세요.")
    buyer = Offer("B1", 1, "buyer", Terms(10000, 14, 0), "이것이 마지막 제안입니다.")
    state = replace(state, current_round=2, history=(first, buyer, next_offer))
    result = KeywordBackend().predict(state, candidates, build_decision_request(state, candidates))
    assert result[FINAL_OFFER_QID] == PolicyConfig().final_prior_by_round[1]
    assert KeywordBackend().describe()["is_model"] is False


def test_keyword_does_not_turn_approval_claim_into_final_offer():
    state = initial_state("keyword-dev", make_supplier(1015, "linear", True, True, True))
    state = replace(state, history=(replace(state.history[0], message="구매팀이 이미 승인했습니다."),))
    candidates = [Candidate("P1", Terms(10500, 14, 0))]
    request = build_decision_request(state, candidates)
    assert KeywordBackend().predict(state, candidates, request)[FINAL_OFFER_QID] == 0.0
