"""System 1 gate (diagnosis, routing, verification) and the three submission scenarios. CPU only."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from negotiation_agent.backends import HeuristicBackend, KeywordBackend
from negotiation_agent.contracts import Offer, PublicState, SupplierResponse, Terms
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.scenarios import (INJECTED_ERROR, SCENARIOS, FaultInjectingBackend, OffScriptSupplier,
                                         TermSwapSupplier, run_scenario)
from negotiation_agent.session import NegotiationSession, SimulatedSupplier
from negotiation_agent.simulator import default_buyer_private, initial_state, make_supplier
from negotiation_agent.system1 import (QID_ACCEPT_EXACT, QID_OFF_SCRIPT, ROUTING_QIDS, ModelReader, RuleReader,
                                       System1Gate, build_routing_request, build_verification_request, gate_for,
                                       number_check)

SRC = Path(__file__).resolve().parents[1] / "src" / "negotiation_agent"
SENT = Terms(10_450, 14, 60)


def _session(seed=1006, sensitive=False, supplier_wrapper=None, backend=None, gate=None, counterparty="sim"):
    hidden = make_supplier(seed, "linear", sensitive, True)
    supplier = SimulatedSupplier(hidden)
    if supplier_wrapper is not None:
        supplier = supplier_wrapper(supplier)
    backend = backend or KeywordBackend()
    gate = gate if gate is not None else gate_for(backend)
    return NegotiationSession(initial_state(f"t-{seed}", hidden), default_buyer_private(), backend,
                              supplier if counterparty == "sim" else None, PolicyConfig(), gate=gate)


class _FakeModel:
    """A stand-in decision backend that answers every noul question with fixed values."""

    kind, is_model, supports_final_offer = "fake_model", True, True

    def __init__(self, values=None, fail=False):
        self.values, self.fail, self.requests, self.last_response = values or {}, fail, [], None

    def predict(self, state, candidates, request):
        self.requests.append(request)
        if self.fail:
            raise ConnectionError("server down")
        self.last_response = {"answers": {q: self.values.get(q, 0.1) for q in request["questions"]}}
        return {q: self.values.get(q, 0.1) for q in request["questions"]}

    def describe(self):
        return {"kind": self.kind, "is_model": True}


# ---- isolation -----------------------------------------------------------------------------------

def test_system1_module_cannot_import_private_hidden_or_evaluator_code():
    tree = ast.parse((SRC / "system1.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    assert not names & {"BuyerPrivate", "SupplierHidden", "economics", "simulator", "metrics", "probe",
                        "evaluation", "session", "policy", "scenarios"}


def test_system1_requests_carry_no_private_or_hidden_value_and_no_scenario_text():
    hidden = make_supplier(1006, "linear", True, True)
    state = initial_state("t", hidden)
    texts = [json.dumps(build_routing_request(state), ensure_ascii=False),
             json.dumps(build_verification_request(state, SENT, "제안하신 조건으로 진행하겠습니다."), ensure_ascii=False)]
    private = replace(default_buyer_private(), target_total_cost=7_777_777, alternative=Terms(13_131, 13, 77))
    for text in texts:
        for secret in (hidden.unit_cost, hidden.opening_profit, hidden.reserve_profit, hidden.expedite_cost_per_day,
                       private.target_total_cost, private.alternative.unit_price):
            assert str(secret) not in text and f"{secret:,}" not in text
        for scenario in SCENARIOS.values():
            assert scenario.title not in text and scenario.business not in text
            for episode in scenario.episodes:
                assert episode.title not in text and episode.expect not in text


def test_routing_request_reuses_the_utterance_questions_unchanged():
    from utterance_agent.questions import utterance_questions
    request = build_routing_request(initial_state("t", make_supplier(1006, "linear", True, True)))
    assert tuple(request["questions"]) == ROUTING_QIDS
    for qid, question in utterance_questions().items():
        assert request["questions"][qid] == question
    assert all(q["type"] == "noul" for q in request["questions"].values())


# ---- code number check and rule reader -----------------------------------------------------------

def test_number_check_flags_changed_days_amounts_and_percentages():
    assert number_check("제안하신 조건으로 진행하겠습니다.", SENT, 1000)["ok"]
    assert number_check("단가 10450원, 납기 14일, 결제 60일 그대로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("총액 10,450,000원으로 확정합니다.", SENT, 1000)["ok"]
    assert number_check("다만 납기는 35일로 하겠습니다.", SENT, 1000)["mismatches"] == ["35일"]
    assert number_check("단가는 10,600원으로 하겠습니다.", SENT, 1000)["mismatches"] == ["금액 10,600원"]
    assert not number_check("선급금 30%를 먼저 받겠습니다.", SENT, 1000)["ok"]


@pytest.mark.parametrize("reply, exact", [
    ("제안하신 조건으로 진행하겠습니다.", True),
    ("네, 말씀하신 조건 그대로 확정합니다.", True),
    ("좋습니다, 그 조건으로 진행하겠습니다. 다만 납기는 35일로 하겠습니다.", False),
    ("확정하겠습니다. 단, 선급금이 필요합니다.", False),
    ("검토 후 회신드리겠습니다.", False),
])
def test_rule_reader_accept_exact(reply, exact):
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    answers, _ = RuleReader().answer(state, build_verification_request(state, SENT, reply))
    assert (answers[QID_ACCEPT_EXACT] == 1.0) is exact


# ---- the gate fails closed -----------------------------------------------------------------------

def test_failed_or_out_of_range_answers_route_to_a_person_and_never_verify():
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    for backend in (_FakeModel(fail=True), _FakeModel(values={QID_ACCEPT_EXACT: 1.5, QID_OFF_SCRIPT: 1.5})):
        gate = System1Gate(ModelReader(backend))
        routed = gate.route(state)
        assert routed["status"] == "error" and routed["route"] == "human_review"
        assert routed["reasons"] == ["system1_unavailable"] and "answers" not in routed
        verified = gate.verify(state, SENT, "제안하신 조건으로 진행하겠습니다.")
        assert verified["verified"] is False and "system1_unavailable" in verified["reasons"]


def test_model_reader_asks_the_real_backend_and_keeps_its_response():
    backend = _FakeModel(values={QID_ACCEPT_EXACT: 0.93})
    gate = gate_for(backend)
    assert isinstance(gate.reader, ModelReader)
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    record = gate.verify(state, SENT, "제안하신 조건으로 진행하겠습니다.")
    assert record["verified"] and record["is_model"] and record["response"]["answers"][QID_ACCEPT_EXACT] == 0.93
    assert set(backend.requests[0]["questions"]) == {QID_ACCEPT_EXACT, QID_OFF_SCRIPT, "u_instruction"}
    assert isinstance(gate_for(KeywordBackend()).reader, RuleReader)
    assert isinstance(gate_for(backend, force_rules=True).reader, RuleReader)


def test_code_number_check_holds_even_when_the_reader_says_exact():
    gate = System1Gate(ModelReader(_FakeModel(values={QID_ACCEPT_EXACT: 0.99})))
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    record = gate.verify(state, SENT, "좋습니다. 납기는 35일로 진행하겠습니다.")
    assert record["verified"] is False and record["reasons"] == ["code_number_mismatch"]


# ---- the session with a gate ---------------------------------------------------------------------

def test_term_swap_is_held_corrected_and_only_the_exact_terms_are_agreed():
    session = _session(supplier_wrapper=TermSwapSupplier)
    assert session.run() == "agreed"
    actions = [o["action"] for o in session.outbox]
    assert actions[-1] == "clarify" and actions.count("clarify") == 1
    verifications = [c for c in session.gate.checks if c["check"] == "verification"]
    assert [v["verified"] for v in verifications] == [False, True]
    assert session.agreement == Terms(**verifications[0]["sent_terms"])
    held = [e["index"] for e in session.events if e["event"] == "system1_verification" and not e["verified"]][0]
    recorded = [e["index"] for e in session.events if e["event"] == "agreement_recorded"][0]
    assert held < recorded
    assert session.outcome()["system1"]["acceptances_held"] == 1


class _AlwaysSwaps(TermSwapSupplier):
    def confirm(self, state, buyer_terms):
        return SupplierResponse("accept", message="네, 납기 35일로 확정합니다.")


def test_a_repeated_mismatch_stops_for_review_without_an_agreement():
    session = _session(supplier_wrapper=_AlwaysSwaps)
    assert session.run() == "needs_review"
    assert session.review_reason == "acceptance_unverified" and session.agreement is None
    assert session.outcome()["agreement"] is None


def test_off_script_demand_routes_to_a_person_and_nothing_is_sent_until_resumed():
    session = _session(seed=1022, supplier_wrapper=OffScriptSupplier)
    assert session.run() == "needs_review" and session.review_reason == "off_script_demand"
    sent = len(session.outbox)
    with pytest.raises(RuntimeError):
        session.step()
    assert len(session.outbox) == sent
    session.resume_after_review("reviewer")
    assert session.run() == "agreed"
    assert session.review_log[-1]["decided_by"] == "reviewer"


def test_injected_estimate_failure_stops_without_any_action_or_invented_value():
    backend = FaultInjectingBackend(KeywordBackend())
    session = _session(seed=1019, backend=backend)
    assert session.run() == "needs_review"
    assert session.review_reason == "acceptance_estimate_unavailable"
    assert session.executed == [] and session.outbox == []
    failed = session.decisions[-1]
    assert INJECTED_ERROR in failed.model["error"] and all(o.p_accept is None for o in failed.options)
    session.resume_after_review("reviewer")
    assert session.run() == "agreed" and backend.injected == 1


def test_person_as_supplier_gets_a_clarification_then_an_exact_confirmation_agrees():
    session = _session(counterparty=None)
    session.step()                       # the agent's first counter-offer, sent to the person
    assert session.status == "awaiting_supplier"
    session.submit_supplier_response("accept", message="좋습니다. 다만 납기는 35일로 하겠습니다.")
    assert session.status == "awaiting_supplier" and session.agreement is None
    assert session.outbox[-1]["action"] == "clarify"
    session.submit_supplier_response("accept", message="네, 제안하신 조건 그대로 확정합니다.")
    assert session.status == "agreed" and session.agreement == Terms(**session.executed[-1]["terms"])


def test_without_a_gate_the_session_behaves_as_before():
    hidden = make_supplier(1006, "linear", False, True)
    session = NegotiationSession(initial_state("t", hidden), default_buyer_private(), KeywordBackend(),
                                 TermSwapSupplier(SimulatedSupplier(hidden)), PolicyConfig())
    assert session.run() == "agreed"           # no verification: the swapped words are not read
    assert "system1" not in session.outcome()


# ---- the three scenarios -------------------------------------------------------------------------

@pytest.mark.parametrize("backend", [KeywordBackend(), HeuristicBackend()])
def test_three_scenarios_on_cpu_reach_their_intended_ends(backend):
    logs = {key: run_scenario(key, backend, "fixture", "fixture") for key in SCENARIOS}
    for log in logs.values():
        assert log["summary"]["authority_violations"] == 0
        assert log["summary"]["agreements_without_verified_acceptance"] == 0
        assert log["approver"]["is_human"] is False and "not a person" in log["approver"]["note"]
        for episode in log["episodes"]:
            assert episode["evidence_class"].startswith("cpu_reference_not_model_evidence")
            assert episode["timeline"] and "Zoro" not in "".join(episode["timeline"])
            assert 1000 <= episode["episode"]["seed"] <= 1099
    s1 = logs["s1"]["episodes"][0]
    assert s1["outcome"]["status"] == "agreed" and s1["outcome"]["approval_cards"] == 0
    assert s1["business"]["improvement_vs_first_offer"] > 0
    assert [c["check"] for c in s1["system1_checks"]][-1] == "verification"
    s2 = logs["s2"]["episodes"][0]
    assert s2["outcome"]["approval_cards"] == 1 and s2["outcome"]["approvals_granted"] == 1
    approved_at = [e["index"] for e in s2["events"] if e["event"] == "approval_approved"][0]
    sent_after = [e for e in s2["events"] if e["event"] == "buyer_counter" and e["index"] > approved_at]
    assert sent_after and sent_after[0]["terms"]["payment_days"] == 0
    swap, off, fault = logs["s3"]["episodes"]
    assert swap["outcome"]["system1"]["acceptances_held"] == 1 and swap["outcome"]["status"] == "agreed"
    assert off["outcome"]["system1"]["routed_to_person"] == 1 and off["review_log"]
    assert fault["evidence_class"].endswith("_with_injected_fault") and fault["review_log"]
    assert fault["backend"]["fault_injection"]["injected_so_far"] == 1


def test_scenario_denied_approval_never_sends_the_unapproved_terms():
    log = run_scenario("s2", KeywordBackend(), "deny", "stop")
    episode = log["episodes"][0]
    assert episode["outcome"]["approvals_denied"] >= 1
    assert all(e["terms"]["payment_days"] != 0 for e in episode["events"] if e["event"] == "buyer_counter")
    assert log["summary"]["authority_violations"] == 0


def test_scenario_cli_writes_three_logs(tmp_path):
    from negotiation_agent.scenarios import main
    assert main(["--backend", "keyword", "--output-dir", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.json"))
    assert names == ["s1_renewal_auto.json", "s2_exact_approval.json", "s3_safe_under_errors.json"]
    log = json.loads((tmp_path / "s3_safe_under_errors.json").read_text(encoding="utf-8"))
    assert log["provenance"]["execution_config"]["backend"] == "keyword"
    assert len(log["episodes"]) == 3


# ---- s2 seed sweep -------------------------------------------------------------------------------

def test_s2_sweep_picks_the_smallest_seed_meeting_the_fixed_rule():
    from negotiation_agent.scenarios import S2_SWEEP_RULE_VERSION, s2_qualifies, sweep_s2
    record = sweep_s2(KeywordBackend(), seeds=tuple(range(1000, 1012)))
    assert record["rule_version"] == S2_SWEEP_RULE_VERSION
    assert record["evidence_class"] == "cpu_reference_not_model_evidence"
    assert "not an evaluation result" in record["notice"]
    assert len(record["rows"]) == 12
    assert record["chosen_seed"] == min(record["qualifying_seeds"])
    for row in record["rows"]:
        if row["qualifies"]:
            assert row["status"] == "agreed" and row["approval_cards"] >= 1 and row["authority_violations"] == 0
    # The rule needs an approved card whose exact terms were agreed and a verified supplier acceptance.
    episode = {"outcome": {"status": "agreed", "authority_violations": [],
                           "agreement": {"unit_price": 10_400, "delivery_days": 14, "payment_days": 60}},
               "approvals": [], "events": [{"event": "supplier_accept"}], "system1_checks": []}
    assert not s2_qualifies(episode)


def test_s2_seed_stays_outside_every_split():
    from negotiation_agent.scenarios import s2_episode
    assert s2_episode(1042).seed == 1042 and s2_episode(1042).style == SCENARIOS["s2"].episodes[0].style
    for seed in (999, 1100, 1500):
        with pytest.raises(ValueError):
            s2_episode(seed)


def test_scenario_cli_s2_seed_override_is_used_and_recorded(tmp_path, monkeypatch):
    from negotiation_agent import scenarios
    monkeypatch.setitem(scenarios.SCENARIOS, "s2", scenarios.SCENARIOS["s2"])
    assert scenarios.main(["--backend", "keyword", "--scenario", "s2", "--s2-seed", "1007",
                           "--output-dir", str(tmp_path)]) == 0
    log = json.loads((tmp_path / "s2_exact_approval.json").read_text(encoding="utf-8"))
    assert [e["episode"]["seed"] for e in log["episodes"]] == [1007]


def test_pinned_zoro_s2_seed_is_the_one_the_recorded_real_sweep_chose():
    from negotiation_agent.scenarios import S2_SEED_BY_BACKEND, S2_SWEEP_RULE_VERSION
    record = json.loads((Path(__file__).resolve().parents[1] / "artifacts/20261119_kaggle_zoro_v2/s2_sweep.json")
                        .read_text(encoding="utf-8"))
    assert record["evidence_class"] == "real_model_run" and record["rule_version"] == S2_SWEEP_RULE_VERSION
    assert S2_SEED_BY_BACKEND["zoro"] == record["chosen_seed"] == min(record["qualifying_seeds"])
    assert "keyword" not in S2_SEED_BY_BACKEND and SCENARIOS["s2"].episodes[0].seed == 1015


def test_s2_sweep_cli_writes_a_record(tmp_path):
    from negotiation_agent.scenarios import sweep_main
    out = tmp_path / "sweep.json"
    assert sweep_main(["--backend", "keyword", "--output", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["seeds"] == [1000, 1099] and len(record["rows"]) == 100
    assert record["provenance"]["execution_config"]["backend"] == "keyword"


# ---- dashboard -----------------------------------------------------------------------------------

def test_dashboard_shows_system1_checks_and_holds_a_person_supplier_term_swap():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app())
    state = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    assert state["system1"]["is_model"] is False and state["system1"]["checks"][0]["check"] == "routing"
    state = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    public = client.post(f"/api/supplier/{token}/response",
                         json={"revision": public["revision"], "kind": "accept",
                               "message": "좋습니다. 다만 납기는 35일로 하겠습니다."}).json()
    assert public["status"] == "awaiting_supplier" and public["agreement"] is None
    assert "그대로 확정하시는지" in public["clarification"]
    buyer = client.get(f"/api/sessions/{state['id']}").json()
    held = buyer["system1"]["checks"][-1]
    assert held["check"] == "verification" and held["verified"] is False and held["number_mismatches"] == ["35일"]
    assert "response" not in held                       # raw engine responses stay server-side
    public = client.post(f"/api/supplier/{token}/response",
                         json={"revision": public["revision"], "kind": "accept",
                               "message": "네, 제안하신 조건 그대로 확정합니다."}).json()
    assert public["status"] == "agreed" and public["clarification"] is None
    assert public["agreement"] == buyer["executed"][-1]["terms"]


def test_dashboard_routes_an_off_script_demand_to_a_person():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app())
    state = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    state = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    client.post(f"/api/supplier/{token}/response",
                json={"revision": public["revision"], "kind": "counter", "unit_price": 10_800, "delivery_days": 14,
                      "payment_days": 60, "message": "이번 계약부터 선급금 30%를 먼저 받겠습니다."})
    buyer = client.get(f"/api/sessions/{state['id']}").json()
    assert buyer["status"] == "needs_review" and buyer["review_reason"] == "off_script_demand"
    assert client.post(f"/api/sessions/{state['id']}/step", json={"revision": buyer["revision"]}).status_code == 409
    resumed = client.post(f"/api/sessions/{state['id']}/retry", json={"revision": buyer["revision"]}).json()
    assert resumed["status"] == "active" and resumed["review_log"][-1]["decided_by"] == "demo-human"


def test_dashboard_stops_on_a_final_claim_like_the_playground_and_may_withhold_the_counter():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app())
    state = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    state = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    client.post(f"/api/supplier/{token}/response",
                json={"revision": public["revision"], "kind": "counter", "unit_price": 10_550, "delivery_days": 14,
                      "payment_days": 60, "message": "이번 조건이 최종안입니다."})
    buyer = client.get(f"/api/sessions/{state['id']}").json()
    assert buyer["status"] == "needs_review" and buyer["review_reason"] == "final_offer_claim"
    assert buyer["system1"]["checks"][-1]["reasons"] == ["final_offer_claim"]
    resumed = client.post(f"/api/sessions/{state['id']}/retry",
                          json={"revision": buyer["revision"], "withhold_counter": True}).json()
    assert resumed["review_log"][-1]["outcome"] == "accept_or_end_only"
    assert resumed["review_log"][-1]["decided_by"] == "demo-human"
    stepped = client.post(f"/api/sessions/{state['id']}/step", json={"revision": resumed["revision"]}).json()
    assert stepped["outbox"][-1]["action"] != "counter"          # no counter-offer after a withheld review
    # Withholding is only for a final-offer stop.
    other = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    other = client.post(f"/api/sessions/{other['id']}/step", json={"revision": other["revision"]}).json()
    token = other["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    client.post(f"/api/supplier/{token}/response",
                json={"revision": public["revision"], "kind": "counter", "unit_price": 10_800, "delivery_days": 14,
                      "payment_days": 60, "message": "이번 계약부터 선급금 30%를 먼저 받겠습니다."})
    other = client.get(f"/api/sessions/{other['id']}").json()
    refused = client.post(f"/api/sessions/{other['id']}/retry",
                          json={"revision": other["revision"], "withhold_counter": True})
    assert refused.status_code == 409


def test_a_decision_without_a_counter_requests_no_estimate_and_its_record_says_so():
    """The /app card for such a decision reads '추정 없음'. Until 2026-11-20 it took the empty record for a
    rule estimate and wrote '규칙 추정 · 모델 아님' on a Zoro-1 server."""
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app())
    state = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    state = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    assert state["records"][-1]["model"]["backend"]              # a counter was compared: the engine was asked
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    client.post(f"/api/supplier/{token}/response",
                json={"revision": public["revision"], "kind": "counter", "unit_price": 10_550, "delivery_days": 14,
                      "payment_days": 60, "message": "이번 조건이 최종안입니다."})
    buyer = client.get(f"/api/sessions/{state['id']}").json()
    resumed = client.post(f"/api/sessions/{state['id']}/retry",
                          json={"revision": buyer["revision"], "withhold_counter": True}).json()
    stepped = client.post(f"/api/sessions/{state['id']}/step", json={"revision": resumed["revision"]}).json()
    record = stepped["records"][-1]
    assert record["model"] == {}                                 # accept and end only: nothing to estimate
    assert all(option.get("p_accept") is None for option in record["options"])
    assert (stepped["pending"] or {}).get("p_accept") is None    # the approval card has no estimate to show


# ---- playground: a person plays the supplier ------------------------------------------------------

def _play_client():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app
    return TestClient(create_app(backend=KeywordBackend()))


def _start(client, message="저희는 결제가 빠를수록 단가를 조정할 수 있습니다.", price=11_300):
    response = client.post("/api/play", json={"unit_price": price, "delivery_days": 21, "payment_days": 60,
                                             "message": message})
    assert response.status_code == 200, response.text
    return response.json()


def test_shared_server_keeps_a_bounded_number_of_sessions(monkeypatch):
    from negotiation_agent import playground
    monkeypatch.setattr(playground, "MAX_LIVE_SESSIONS", 2)
    client = _play_client()
    first, second, third = (_start(client) for _ in range(3))
    assert client.get(f"/api/play/{first['id']}").status_code == 404      # oldest dropped
    assert client.get(f"/api/play/{third['id']}").status_code == 200
    ids = [client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()["id"] for _ in range(3)]
    assert client.get(f"/api/sessions/{ids[0]}").status_code == 404
    assert client.get(f"/api/sessions/{ids[2]}").status_code == 200


def test_playground_agent_answers_on_its_own_and_waits_for_the_person():
    client = _play_client()
    view = _start(client)
    assert view["status"] == "awaiting_supplier" and view["you"]["can_reply"]
    assert [m["who"] for m in view["chat"]] == ["me", "agent"]
    assert view["chat"][1]["kind"] == "counter" and view["you"]["offer_to_answer"] == view["chat"][1]["terms"]
    assert view["agent"]["system1"]["checks"][0]["check"] == "routing"
    assert client.get("/play").status_code == 200


def test_playground_person_is_the_approver_and_nothing_is_approved_automatically():
    client = _play_client()
    view = _start(client)
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "counter", "unit_price": 10_900,
                             "delivery_days": 14, "payment_days": 60, "message": "이번 조건이 최종안입니다."}).json()
    assert view["agent"]["review_reason"] == "final_offer_claim"     # the claim stops first; continue as usual
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] == "awaiting_approval" and view["agent"]["pending"] is not None
    assert view["you"]["can_reply"] is False
    pending = view["agent"]["pending"]
    denied = client.post(f"/api/play/{view['id']}/approval",
                         json={"revision": view["revision"], "request_id": pending["request_id"],
                               "terms_version": pending["terms_version"], "decision": "deny"}).json()
    assert denied["agent"]["outcome"]["approvals_denied"] == 1
    assert all(m["terms"] != pending["terms"] for m in denied["chat"] if m["who"] == "agent" and m.get("terms"))


def test_playground_term_swap_gets_a_clarification_then_exact_acceptance_agrees():
    client = _play_client()
    view = _start(client, message="제시한 조건으로 검토 부탁드립니다.")
    sent = view["you"]["offer_to_answer"]
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "accept",
                             "message": "좋습니다. 다만 납기는 35일로 하겠습니다."}).json()
    assert view["status"] == "awaiting_supplier" and view["you"]["agreement"] is None
    assert view["chat"][-1]["kind"] == "clarify"
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "accept", "message": "네, 그대로 확정합니다."}).json()
    assert view["status"] == "agreed" and view["you"]["agreement"] == sent
    assert view["chat"][-1]["who"] == "system"


def test_playground_off_script_demand_stops_until_the_person_resumes():
    client = _play_client()
    view = _start(client)
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "counter", "unit_price": 10_800, "delivery_days": 14,
                             "payment_days": 60, "message": "선급금 30%를 먼저 받겠습니다."}).json()
    assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "off_script_demand"
    stale = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"] - 1})
    assert stale.status_code == 409
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] != "needs_review"


def test_playground_rejects_browser_supplied_backend_or_simulator_fields():
    client = _play_client()
    for extra in ({"backend": "zoro"}, {"seed": 1000}, {"url": "http://x"}, {"unit_cost": 9000}):
        response = client.post("/api/play", json={"unit_price": 11_300, "delivery_days": 21, "payment_days": 60, **extra})
        assert response.status_code == 422


# ---- an offer whose text and typed numbers disagree is held, not negotiated --------------------------

def test_offer_number_check_allows_the_offer_and_the_quoted_buyer_offer_only():
    from negotiation_agent.system1 import offer_number_check
    offered, answering = Terms(10_800, 14, 60), Terms(10_300, 14, 60)
    assert offer_number_check("10,300원은 어렵고 10,800원으로 드리겠습니다.", offered, answering, 1000)["ok"]
    assert offer_number_check("총 10,800,000원, 납기 14일입니다.", offered, answering, 1000)["ok"]
    assert offer_number_check("선급금 30%를 받겠습니다.", offered, answering, 1000)["ok"]   # left to routing
    held = offer_number_check("11000원은 어려우실까요?", answering, answering, 1000)
    assert not held["ok"] and held["mismatches"] == ["금액 11,000원"]
    assert offer_number_check("다만 납기는 35일로", offered, answering, 1000)["mismatches"] == ["35일"]


def test_playground_holds_a_counter_whose_text_names_other_numbers_and_spends_no_round():
    client = _play_client()
    view = _start(client)
    sent, round_before = view["you"]["offer_to_answer"], view["round"]
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "counter", **sent,
                             "message": "11000원은 어려우실까요?"}).json()
    assert view["status"] == "awaiting_supplier" and view["round"] == round_before
    assert [m["kind"] for m in view["chat"][-2:]] == ["offer_held", "clarify_offer"]
    assert view["chat"][-2]["mismatches"] == ["금액 11,000원"] and view["you"]["offer_to_answer"] == sent
    assert view["agent"]["outcome"]["offers_held_for_text_mismatch"] == 1 and view["agent"]["decisions"] == 1
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "counter", "unit_price": 11_000,
                             "delivery_days": 14, "payment_days": 60, "message": "11000원은 어려우실까요?"}).json()
    assert view["round"] == round_before + 1 and view["chat"][-2]["kind"] == "offer"


def test_playground_rejects_an_opening_whose_text_names_other_numbers():
    client = _play_client()
    response = client.post("/api/play", json={"unit_price": 11_300, "delivery_days": 21, "payment_days": 60,
                                              "message": "단가 11,000원에 드리겠습니다."})
    assert response.status_code == 422 and "11,000원" in response.json()["detail"]


def test_dashboard_human_first_offer_with_disagreeing_text_asks_again():
    state = PublicState("hold", "item", 1000, 5, 1, ())
    session = NegotiationSession(state, default_buyer_private(), KeywordBackend(), None)
    text = session.submit_supplier_offer({"unit_price": 11_300, "delivery_days": 21, "payment_days": 60},
                                         "납기 14일로 맞춰 드리겠습니다.")
    assert session.status == "awaiting_info" and "14일" in text and not session.state.history
    assert session.submit_supplier_offer({"unit_price": 11_300, "delivery_days": 14, "payment_days": 60},
                                         "납기 14일로 맞춰 드리겠습니다.") is None
    assert session.state.last_offer("supplier").terms == Terms(11_300, 14, 60)


def test_system1_probe_runner_writes_a_complete_cpu_record(tmp_path):
    import os
    import subprocess
    import sys
    out = tmp_path / "probe.json"
    root = Path(__file__).resolve().parents[1]
    # cp1252 is what GitHub's Windows runners use for a pipe; the probe prints Korean (test_console_encoding.py)
    subprocess.run([sys.executable, str(root / "run_system1_probe.py"), "--backend", "keyword", "--output", str(out)],
                   check=True, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["complete"] and len(record["rows"]) == 14
    assert record["evidence_class"] == "cpu_reference_not_model_evidence"
    assert {r["label"] for r in record["rows"] if r["kind"] == "verification"} >= {"exact", "swap_freight"}


def test_verification_v2_holds_an_acceptance_that_adds_a_new_condition():
    # Kaggle T4 v1 run: Zoro answered s1_accept_exact 0.853 for this reply, so v1 agreed.
    reply = "좋습니다, 그 조건으로 진행하겠습니다. 운송비는 별도로 청구하겠습니다."
    gate = System1Gate(ModelReader(_FakeModel(values={QID_ACCEPT_EXACT: 0.853, QID_OFF_SCRIPT: 0.95})))
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    record = gate.verify(state, SENT, reply)
    assert record["verified"] is False and record["reasons"] == ["reader_says_new_condition"]
    request = gate.reader.backend.requests[0]
    assert "공급사_최근_답변" in request["questions"][QID_OFF_SCRIPT]["instructions"]
    assert "제안 이력에서" not in request["questions"][QID_OFF_SCRIPT]["instructions"]


def test_routing_request_is_unchanged_by_verification_v2():
    from negotiation_agent.system1 import build_routing_request, off_script_question
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    assert build_routing_request(state)["questions"][QID_OFF_SCRIPT] == off_script_question()


# ---- a final-offer claim stops for a person's choice (opt-in; the playground turns it on) --------------

def test_final_claim_stop_is_opt_in_and_an_instruction_never_triggers_it():
    state = initial_state("t", make_supplier(1006, "linear", False, True))
    said_final = _FakeModel(values={"u_final": 0.95})
    unchanged = System1Gate(ModelReader(said_final)).route(state)
    assert unchanged["route"] == "continue" and unchanged["reasons"] == []      # default gate behaves as before
    routed = gate_for(said_final, stop_on_final_claim=True).route(state)
    assert routed["route"] == "human_review" and routed["reasons"] == ["final_offer_claim"]
    assert routed["version"] != unchanged["version"]                            # new behaviour, new version
    threat = gate_for(_FakeModel(values={"u_walkaway": 0.9}), stop_on_final_claim=True).route(state)
    assert threat["reasons"] == ["final_offer_claim"]
    injected = gate_for(_FakeModel(values={"u_final": 0.95, "u_instruction": 0.9}), stop_on_final_claim=True).route(state)
    assert injected["route"] == "continue" and injected["reasons"] == ["instruction_ignored"]
    both = gate_for(_FakeModel(values={"u_final": 0.95, QID_OFF_SCRIPT: 0.9}), stop_on_final_claim=True).route(state)
    assert both["reasons"] == ["off_script_demand"]


def test_policy_without_a_counter_compares_only_accepting_and_ending_and_asks_no_estimate():
    from negotiation_agent.contracts import ApprovalLedger
    from negotiation_agent.policy import decide
    from negotiation_agent.simulator import ITEM, MAX_ROUNDS, QUANTITY

    def state_with(standing):
        return PublicState("t", ITEM, QUANTITY, MAX_ROUNDS, 2, (
            Offer("S1", 1, "supplier", Terms(11_300, 21, 60)), Offer("B1/S1", 1, "buyer", Terms(10_160, 21, 60)),
            Offer("S2", 2, "supplier", standing, "이번 조건이 최종안입니다.")))

    backend = _FakeModel()
    good = decide(state_with(Terms(10_550, 14, 60)), default_buyer_private(), backend, ApprovalLedger(),
                  PolicyConfig(), {}, allow_counter=False)
    assert good.action == "accept" and good.reasons == ("counter_withheld_by_reviewer",)
    assert {o.action for o in good.options} == {"accept", "end"} and backend.requests == []
    worse_than_alternative = decide(state_with(Terms(11_300, 21, 60)), default_buyer_private(), backend,
                                    ApprovalLedger(), PolicyConfig(), {}, allow_counter=False)
    assert worse_than_alternative.action == "end" and backend.requests == []


def _play_counter(client, view, price, message):
    return client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "counter", "unit_price": price,
                             "delivery_days": 14, "payment_days": 60, "message": message}).json()


def test_playground_final_claim_stops_and_the_person_may_withhold_the_counter():
    client = _play_client()
    view = _play_counter(client, _start(client), 10_550, "이번 조건이 최종안입니다.")
    assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "final_offer_claim"
    assert view["chat"][-1]["who"] == "me"                                       # nothing was sent in reply yet
    check = view["agent"]["system1"]["checks"][-1]
    assert check["route"] == "human_review" and check["reasons"] == ["final_offer_claim"]
    view = client.post(f"/api/play/{view['id']}/review",
                       json={"revision": view["revision"], "withhold_counter": True}).json()
    mine = {"unit_price": 10_550, "delivery_days": 14, "payment_days": 60}
    assert view["status"] == "agreed" and view["you"]["agreement"] == mine
    assert "counter_withheld_by_reviewer" in view["agent"]["last_decision"]["reasons"]
    assert [m["kind"] for m in view["chat"] if m["who"] == "agent"] == ["counter", "accept"]
    assert view["agent"]["outcome"]["authority_violations"] == []


def test_playground_final_claim_review_may_continue_as_usual_and_other_stops_cannot_withhold():
    client = _play_client()
    view = _play_counter(client, _start(client), 10_550, "이번 조건이 최종안입니다.")
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] != "needs_review"
    assert "counter_withheld_by_reviewer" not in view["agent"]["last_decision"]["reasons"]
    other = _play_counter(client, _start(client), 10_800, "선급금 30%를 먼저 받겠습니다.")
    assert other["agent"]["review_reason"] == "off_script_demand"
    refused = client.post(f"/api/play/{other['id']}/review",
                          json={"revision": other["revision"], "withhold_counter": True})
    assert refused.status_code == 409


# ---- an offer that carried an off-script demand is never agreed as it stands ---------------------------

def test_playground_offer_with_an_off_script_demand_is_agreed_only_after_the_three_terms_alone_are_confirmed():
    client = _play_client()
    mine = {"unit_price": 10_200, "delivery_days": 14, "payment_days": 60}      # better than every counter-offer
    view = _play_counter(client, _start(client), mine["unit_price"], "선급금 30%를 먼저 받겠습니다.")
    assert view["agent"]["review_reason"] == "off_script_demand"
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["agent"]["last_decision"]["action"] == "accept"
    assert view["status"] == "awaiting_supplier" and view["you"]["agreement"] is None
    assert view["chat"][-1]["kind"] == "accept_terms_only" and "포함하지 않습니다" in view["chat"][-1]["text"]
    assert view["you"]["offer_to_answer"] == mine
    view = client.post(f"/api/play/{view['id']}/reply",
                       json={"revision": view["revision"], "kind": "accept", "message": "네, 그대로 확정합니다."}).json()
    assert view["status"] == "agreed" and view["you"]["agreement"] == mine
    assert view["agent"]["outcome"]["authority_violations"] == []


def test_final_claim_on_an_offer_that_cannot_be_accepted_does_not_stop_the_playground():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app(backend=_FakeModel(values={"u_final": 0.94})))
    # Worse than the alternative and above the company price limit: a walk-away would lose nothing.
    view = client.post("/api/play", json={"unit_price": 11_600, "delivery_days": 28, "payment_days": 45,
                                         "message": "이 조건이 저희가 드릴 수 있는 최선입니다."}).json()
    assert view["status"] == "awaiting_supplier" and view["chat"][-1]["kind"] == "counter"
    opening = view["agent"]["system1"]["checks"][0]
    assert opening["route"] == "continue" and opening["reasons"] == ["final_claim_nothing_to_lose"]
    # The same reading on an offer worth accepting does stop.
    view = _play_counter(client, view, 10_550, "이 조건이 저희가 드릴 수 있는 최선입니다.")
    assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "final_offer_claim"


def test_dashboard_supplier_screen_shows_why_a_counter_with_disagreeing_text_was_not_taken():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app())
    state = client.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    state = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    offers = len(public["transcript"])
    public = client.post(f"/api/supplier/{token}/response",
                         json={"revision": public["revision"], "kind": "counter", "unit_price": 10_500,
                               "delivery_days": 14, "payment_days": 60, "message": "11000원은 어려우실까요?"}).json()
    # The offer was held and uses no round; the person who typed it must see the agent's question.
    assert public["status"] == "awaiting_supplier" and len(public["transcript"]) == offers
    assert public["clarification"] and "다시 보내 주세요" in public["clarification"]


def test_dashboard_plays_the_approval_scenario_with_the_seed_picked_for_the_backend_that_is_shown():
    from fastapi.testclient import TestClient
    from negotiation_agent.scenarios import S2_SEED_BY_BACKEND
    from negotiation_agent.web import create_app

    def opening(backend):
        client = TestClient(create_app(backend=backend))
        view = client.post("/api/sessions", json={"mode": "simulated", "preset": "start",
                                                 "scenario": "payment_tradeoff"}).json()
        return view["transcript"][0]["terms"]

    def first_offer(seed):
        offer = initial_state("x", make_supplier(seed, "linear", True, True)).history[0]
        return {"unit_price": offer.terms.unit_price, "delivery_days": offer.terms.delivery_days,
                "payment_days": offer.terms.payment_days}

    # Real Zoro settles seed 1015 without an approval card, so a model backend gets the swept seed.
    assert first_offer(1015) != first_offer(S2_SEED_BY_BACKEND["zoro"])
    assert opening(KeywordBackend()) == first_offer(1015)
    assert opening(_FakeModel()) == first_offer(S2_SEED_BY_BACKEND["zoro"])


# ---- a failed System 1 look at an offer is asked again after review, never skipped -------------------

class _FailsFirst(_FakeModel):
    """A model backend whose first `failures` requests fail, as when the decision server is down."""

    def __init__(self, failures, values=None):
        super().__init__(values)
        self.failures = failures

    def predict(self, state, candidates, request):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("server down")
        return super().predict(state, candidates, request)


def _play_with(backend, message):
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app
    client = TestClient(create_app(backend=backend))
    return client, client.post("/api/play", json={"unit_price": 10_800, "delivery_days": 14, "payment_days": 60,
                                                 "message": message}).json()


def test_an_offer_whose_system1_look_failed_is_read_again_before_the_agent_answers():
    client, view = _play_with(_FailsFirst(1, {QID_OFF_SCRIPT: 0.9}), "선급금 30%를 먼저 받겠습니다.")
    assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "system1_unavailable"
    # The server is back. Resuming must not answer an offer that System 1 never read.
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "off_script_demand"
    assert [m["who"] for m in view["chat"]] == ["me"]
    routing = [c for c in view["agent"]["system1"]["checks"] if c["check"] == "routing"]
    assert [c["status"] for c in routing] == ["error", "ok"]                    # the failed look stays on record
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] == "awaiting_supplier" and view["chat"][-1]["who"] == "agent"


def test_while_system1_keeps_failing_the_session_stays_stopped_and_sends_nothing():
    client, view = _play_with(_FailsFirst(3), "검토 부탁드립니다.")
    for _ in range(2):
        view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
        assert view["status"] == "needs_review" and view["agent"]["review_reason"] == "system1_unavailable"
        assert [m["who"] for m in view["chat"]] == ["me"]
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    assert view["status"] == "awaiting_supplier" and view["chat"][-1]["kind"] == "counter"


def test_dashboard_retry_reads_the_offer_again_after_a_failed_system1_look():
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app

    client = TestClient(create_app(backend=_FailsFirst(1)))
    state = client.post("/api/sessions", json={"mode": "simulated", "preset": "start"}).json()
    assert state["status"] == "needs_review" and state["review_reason"] == "system1_unavailable"
    state = client.post(f"/api/sessions/{state['id']}/retry", json={"revision": state["revision"]}).json()
    assert state["status"] == "active"
    assert [c["status"] for c in state["system1"]["checks"]] == ["error", "ok"]


# ---- supplier text is capped so that five rounds of it still fit the model's input limit -----------------

def test_supplier_text_longer_than_the_cap_is_refused_at_the_door_on_both_screens():
    from fastapi.testclient import TestClient
    from negotiation_agent.session import MAX_MESSAGE_CHARS
    from negotiation_agent.web import create_app

    ok, too_long = "가" * MAX_MESSAGE_CHARS, "가" * (MAX_MESSAGE_CHARS + 1)
    client = _play_client()
    opening = {"unit_price": 11_300, "delivery_days": 21, "payment_days": 60}
    assert client.post("/api/play", json={**opening, "message": too_long}).status_code == 422
    view = client.post("/api/play", json={**opening, "message": ok}).json()
    reply = {"revision": view["revision"], "kind": "counter", "unit_price": 10_900, "delivery_days": 14, "payment_days": 60}
    assert client.post(f"/api/play/{view['id']}/reply", json={**reply, "message": too_long}).status_code == 422
    assert client.post(f"/api/play/{view['id']}/reply", json={**reply, "message": ok}).status_code == 200

    dashboard = TestClient(create_app())
    state = dashboard.post("/api/sessions", json={"mode": "human", "preset": "start"}).json()
    state = dashboard.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).json()
    token = state["supplier_url"].split("/")[-1]
    body = {"revision": state["revision"], "kind": "counter", "unit_price": 10_900, "delivery_days": 14, "payment_days": 60}
    assert dashboard.post(f"/api/supplier/{token}/response", json={**body, "message": too_long}).status_code == 422
    assert dashboard.post(f"/api/supplier/{token}/response", json={**body, "message": ok}).status_code == 200


def test_final_claim_needs_words_and_is_not_asked_of_the_last_public_round():
    # Real Zoro read "final" 0.53-0.68 for an offer with NO text in rounds 4-5 (local server, 2026-11-20).
    def state_with(message, round_index):
        history = tuple(Offer(f"S{r}", r, "supplier", Terms(10_700, 21, 60), message if r == round_index else "검토 부탁드립니다.")
                        for r in range(1, round_index + 1))
        return PublicState("t", "item", 1000, 5, round_index, history)

    gate = gate_for(_FakeModel(values={"u_final": 0.68, "u_walkaway": 0.6}), stop_on_final_claim=True)
    spoken = gate.route(state_with("더는 어렵습니다.", 4))
    assert spoken["route"] == "human_review" and spoken["reasons"] == ["final_offer_claim"]
    for wordless in ("", "   "):
        silent = gate.route(state_with(wordless, 4))
        assert silent["route"] == "continue" and silent["reasons"] == []          # nothing was said, so nothing was claimed
    last = gate.route(state_with("더는 어렵습니다.", 5))
    assert last["route"] == "continue" and last["reasons"] == []                  # the policy already treats round 5 as final


def test_implausible_offer_is_still_read_by_system1_after_the_review():
    from negotiation_agent.simulator import ITEM, MAX_ROUNDS, QUANTITY
    state = PublicState("t-implausible", ITEM, QUANTITY, MAX_ROUNDS, 1,
                        (Offer("S1", 1, "supplier", Terms(1, 21, 60), "단가 1원에 드립니다."),))
    gate = gate_for(KeywordBackend())
    session = NegotiationSession(state, default_buyer_private(), KeywordBackend(), None, PolicyConfig(), gate=gate)
    session.route_opening_offer()
    assert (session.status, session.review_reason) == ("needs_review", "implausible_offer")
    assert gate.checks == []
    session.resume_after_review("tester")
    assert len(gate.checks) == 1 and gate.checks[0]["check"] == "routing"


def test_a_counter_to_an_off_script_demand_says_the_demand_is_not_included():
    from negotiation_agent.session import OFF_SCRIPT_EXCLUDED

    client = _play_client()
    view = _play_counter(client, _start(client), 10_800, "선급금 30%를 먼저 받겠습니다.")
    assert view["agent"]["review_reason"] == "off_script_demand"
    view = client.post(f"/api/play/{view['id']}/review", json={"revision": view["revision"]}).json()
    sent = [m for m in view["chat"] if m["who"] == "agent"][-1]
    assert sent["kind"] == "counter" and sent["text"].endswith(OFF_SCRIPT_EXCLUDED)
    # A plain counter carries no such sentence.
    plain = _play_counter(client, _start(client), 10_800, "단가 10,800원으로 다시 드립니다.")
    assert OFF_SCRIPT_EXCLUDED not in [m for m in plain["chat"] if m["who"] == "agent"][-1]["text"]
