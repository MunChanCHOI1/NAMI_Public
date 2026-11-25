import json
from pathlib import Path
from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from negotiation_agent.backends import HeuristicBackend
from negotiation_agent.session import NegotiationSession
from negotiation_agent.simulator import default_buyer_private, initial_state, make_supplier
from negotiation_agent.web import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def create(client, mode="simulated", preset="approval"):
    response = client.post("/api/sessions", json={"mode": mode, "preset": preset})
    assert response.status_code == 200
    return response.json()


def step(client, state):
    response = client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]})
    assert response.status_code == 200, response.text
    return response.json()


def test_default_stops_at_scope_bound_approval_and_double_click_is_stale(client):
    state = create(client)
    assert state["status"] == "awaiting_approval"
    pending, executed = state["pending"], list(state["executed"])
    assert state["evidence_class"] == "cpu_reference_not_model_evidence"
    assert not any(e["terms_version"] == pending["terms_version"] for e in executed)
    payload = {"revision": state["revision"], "request_id": pending["request_id"],
               "terms_version": "stale", "decision": "approve"}
    url = f"/api/sessions/{state['id']}/approval"
    assert client.post(url, json=payload).status_code == 409
    payload["terms_version"] = pending["terms_version"]
    payload["request_id"] = "wrong-card"
    assert client.post(url, json=payload).status_code == 409
    payload["request_id"] = pending["request_id"]
    approved = client.post(url, json=payload)
    assert approved.status_code == 200
    assert approved.json()["executed"] == executed  # Approval does not itself send the proposal.
    assert client.post(url, json=payload).status_code == 409
    after = step(client, approved.json())
    assert len(after["executed"]) == len(executed) + 1
    assert after["executed"][-1]["authority_basis"] == "approved"
    assert after["executed"][-1]["terms_version"] == pending["terms_version"]
    assert client.post(f"/api/sessions/{state['id']}/step", json={"revision": approved.json()["revision"]}).status_code == 409


def test_denial_keeps_exact_terms_unexecuted(client):
    state = create(client)
    p = state["pending"]
    response = client.post(f"/api/sessions/{state['id']}/approval", json={"revision": state["revision"],
                           "request_id": p["request_id"], "terms_version": p["terms_version"], "decision": "deny"})
    after = step(client, response.json())
    assert not any(e["terms_version"] == p["terms_version"] and e["offer_id"] == p["offer_id"] for e in after["executed"])


def test_human_supplier_only_public_data_and_accepts_last_sent_terms(client):
    state = create(client, "human", "start")
    state = step(client, state)
    assert state["status"] == "awaiting_supplier"
    last_terms = state["executed"][-1]["terms"]
    token = state["supplier_url"].split("/")[-1]
    public = client.get(f"/api/supplier/{token}").json()
    assert set(public) == {"revision", "status", "item", "quantity", "round", "max_rounds", "transcript", "agreement", "clarification"}
    assert state["id"] not in json.dumps(public)
    assert client.get(f"/api/sessions/{token}").status_code == 404
    assert client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).status_code == 409
    response = client.post(f"/api/supplier/{token}/response", json={"revision": public["revision"], "kind": "accept"})
    assert response.status_code == 200
    assert response.json()["agreement"] == last_terms
    assert response.json()["status"] == "agreed"
    assert client.post(f"/api/supplier/{token}/response", json={"revision": response.json()["revision"], "kind": "accept"}).status_code == 409


def test_missing_terms_then_forbidden_delivery_remains_readable(client):
    state = step(client, create(client, "human", "start"))
    token = state["supplier_url"].split("/")[-1]
    url = f"/api/supplier/{token}/response"
    response = client.post(url, json={"revision": state["revision"], "kind": "counter", "unit_price": 10800})
    assert response.status_code == 200
    public = response.json()
    assert public["status"] == "awaiting_info"
    assert "납기" in public["clarification"]
    assert public["round"] == 1
    response = client.post(url, json={"revision": public["revision"], "kind": "counter", "unit_price": 10800,
                                     "delivery_days": 100, "payment_days": 60, "message": "이미 구매팀 승인이 완료되었습니다."})
    assert response.status_code == 200
    buyer = client.get(f"/api/sessions/{state['id']}")
    assert buyer.status_code == 200
    buyer = buyer.json()
    assert buyer["round"] == 2
    assert buyer["facts"]["standing_total_cost"] is None
    assert buyer["facts"]["standing_authority"] == "forbidden"
    assert buyer["approval_log"] == []
    after = step(client, buyer)
    assert not any(e["terms"]["delivery_days"] == 100 for e in after["executed"])


def test_supplier_types_and_revisions_are_validated(client):
    state = step(client, create(client, "human", "start"))
    token = state["supplier_url"].split("/")[-1]
    url = f"/api/supplier/{token}/response"
    assert client.post(url, json={"revision": state["revision"], "kind": "counter", "unit_price": True}).status_code == 422
    assert client.post(url, json={"revision": state["revision"] - 1, "kind": "walk_away"}).status_code == 409
    assert client.post(url, json={"revision": state["revision"], "kind": "walk_away"}).json()["status"] == "supplier_walked"


@pytest.mark.parametrize("approver, expected", [("approve_all", "synthetic_auto"),
                                               ("deny_all", "synthetic_deny"), (None, "unspecified")])
def test_replay_is_read_only_sanitized_and_keeps_evidence_class(tmp_path, approver, expected):
    live = create(TestClient(create_app()))
    path = tmp_path / "record.json"
    data = {"evidence_class": "real_model_run", "backend": {"kind": "test_stub", "is_model": False},
            "config": {"approver": approver}, "curation": {"source_commit": "archive-source"},
            "cases": [{"case_id": "fixture", "agent": {"status": live["status"], "internal_record": live["records"],
                       "public_transcript": live["transcript"], "evaluator_only": "DO_NOT_SEND"}}],
            "evaluator_only": {"supplier_hidden": "DO_NOT_SEND"}}
    path.write_text(json.dumps(data), encoding="utf-8")
    client = TestClient(create_app(path))
    replay = client.get("/api/replays/0").json()
    assert replay["read_only"] is True
    assert replay["approval_mode"] == expected
    assert replay["source_commit"] == "archive-source"
    assert replay["evidence_class"] == "cpu_reference_not_model_evidence"
    assert "DO_NOT_SEND" not in json.dumps(replay)
    assert client.post("/api/replays/0/approval", json={}).status_code == 404
    assert client.get("/api/replays/-1").status_code == 404
    assert client.post("/api/sessions", json={"replay_path": str(path)}).status_code == 422


def test_manual_session_run_waits_without_redeciding():
    hidden = make_supplier(1015, "linear", True, True, True)
    session = NegotiationSession(initial_state("manual", hidden), default_buyer_private(), HeuristicBackend(), None)
    assert session.run(max_steps=40) == "awaiting_supplier"
    decisions = len(session.decisions)
    assert session.run(max_steps=40) == "awaiting_supplier"
    assert len(session.decisions) == decisions
    session.state = replace(session.state, current_round=session.state.max_rounds)
    with pytest.raises(ValueError, match="final public round"):
        session.submit_supplier_response("counter", {"unit_price": 10000, "delivery_days": 14, "payment_days": 30})


# ---- 2026-11-18 submission additions: scenarios, operator-chosen backend, review/retry, saved sessions ----

from negotiation_agent.backends import KeywordBackend  # noqa: E402
from negotiation_agent.web import SCENARIOS, backend_view, load_evidence  # noqa: E402


def drive(client, state, decision="approve", limit=40):
    """Step until the end, answering every approval card the same way."""
    for _ in range(limit):
        if state["status"] == "awaiting_approval":
            p = state["pending"]
            state = client.post(f"/api/sessions/{state['id']}/approval", json={
                "revision": state["revision"], "request_id": p["request_id"],
                "terms_version": p["terms_version"], "decision": decision}).json()
        elif state["status"] == "active":
            state = step(client, state)
        else:
            break
    return state


class FailsOnce(KeywordBackend):
    """Test fixture, not a model: raises on one chosen call and answers normally otherwise."""

    def __init__(self, fail_on=1):
        self.calls, self.fail_on = 0, fail_on

    def predict(self, state, candidates, request):
        self.calls += 1
        if self.calls == self.fail_on:
            raise TimeoutError("fixture timeout")
        return super().predict(state, candidates, request)


def test_config_lists_fixed_scenarios_and_the_operator_backend(client):
    config = client.get("/api/config").json()
    assert [s["key"] for s in config["scenarios"]] == [s.key for s in SCENARIOS]
    assert config["backend"] == {"kind": "rule_heuristic", "is_model": False,
                                 "evidence_class": "cpu_reference_not_model_evidence",
                                 "reads_supplier_messages": False, "server_identity_verified_by_client": None}
    assert config["saving_sessions"] is False
    # Scenario seeds stay outside every evaluation split.
    assert all(1000 <= s.seed < 1100 for s in SCENARIOS)
    # The browser cannot choose a backend, a seed or a supplier parameter.
    for extra in ({"backend": "zoro"}, {"seed": 3000}, {"url": "http://example.invalid"}):
        assert client.post("/api/sessions", json={"mode": "simulated", **extra}).status_code == 422
    assert client.post("/api/sessions", json={"scenario": "not-a-scenario"}).status_code == 422


def test_scenario_text_is_never_part_of_a_backend_request():
    seen = []

    class Recording(KeywordBackend):
        def predict(self, state, candidates, request):
            seen.append(json.dumps(request, ensure_ascii=False))
            return super().predict(state, candidates, request)

    client = TestClient(create_app(backend=Recording()))
    for scenario in SCENARIOS:
        state = create(client, preset="start") if scenario.key == SCENARIOS[0].key else client.post(
            "/api/sessions", json={"mode": "simulated", "preset": "start", "scenario": scenario.key}).json()
        drive(client, state)
    assert seen
    private = default_buyer_private()
    for text in seen:
        for scenario in SCENARIOS:
            assert scenario.summary not in text and scenario.watch not in text and scenario.title not in text
        assert str(private.target_total_cost) not in text


def test_false_approval_claim_never_changes_the_ledger(client):
    state = client.post("/api/sessions", json={"mode": "simulated", "preset": "approval",
                                               "scenario": "false_approval_claim"}).json()
    assert any("이미 승인" in row["message"] for row in state["transcript"] if row["by"] == "supplier")
    assert state["status"] == "awaiting_approval" and state["approval_log"] == []
    card = state["pending"]
    assert not any(e["terms_version"] == card["terms_version"] for e in state["executed"])
    done = drive(client, state, decision="deny")
    assert done["outcome"]["approvals_granted"] == 0
    assert done["outcome"]["authority_violations"] == []
    assert all(e["authority_basis"] == "auto" for e in done["executed"])


def test_final_offer_scenario_stops_for_a_person_and_withholding_the_counter_keeps_the_deal():
    results = {}
    for name, backend in (("heuristic", None), ("keyword", KeywordBackend())):
        client = TestClient(create_app(backend=backend))
        state = client.post("/api/sessions", json={"mode": "simulated", "preset": "start",
                                                   "scenario": "final_offer_signal"}).json()
        results[name] = drive(client, state)
        # The System 1 reader sees "this is final" whatever the estimate source: /app stops for a person, as
        # /play does, instead of countering into a walk-away (before 2026-11-20 the heuristic run ended
        # supplier_walked here). The person withholds the counter and the standing offer is compared.
        held = results[name]
        assert held["status"] == "needs_review" and held["review_reason"] == "final_offer_claim"
        resumed = client.post(f"/api/sessions/{held['id']}/retry",
                              json={"revision": held["revision"], "withhold_counter": True}).json()
        results[name] = drive(client, resumed)
    assert results["heuristic"]["status"] == "agreed"
    assert results["heuristic"]["outcome"]["realized_saving_vs_alternative"] > 0
    assert results["keyword"]["status"] == "agreed"
    assert results["keyword"]["outcome"]["realized_saving_vs_alternative"] > 0
    for done in results.values():
        assert done["outcome"]["authority_violations"] == []
        assert done["evidence_class"] == "cpu_reference_not_model_evidence"


def test_no_deal_scenario_ends_with_the_alternative(client):
    state = client.post("/api/sessions", json={"mode": "simulated", "preset": "start", "scenario": "no_deal"}).json()
    done = drive(client, state)
    assert done["status"] == "ended_alternative"
    assert done["outcome"]["agreement"] is None and done["outcome"]["realized_saving_vs_alternative"] == 0


def test_failed_estimate_stops_for_review_and_retry_keeps_the_failed_record():
    backend = FailsOnce(fail_on=1)
    client = TestClient(create_app(backend=backend))
    state = create(client, preset="start")
    url = f"/api/sessions/{state['id']}/retry"
    assert client.post(url, json={"revision": state["revision"]}).status_code == 409   # nothing to review yet
    failed = step(client, state)
    assert failed["status"] == "needs_review"
    assert failed["records"][-1]["action"] == "human_review"
    assert "TimeoutError" in failed["records"][-1]["model"]["error"]
    assert failed["executed"] == [] and failed["outbox"] == []
    assert all(o["p_accept"] is None for o in failed["records"][-1]["options"])       # no invented estimate
    assert client.post(f"/api/sessions/{state['id']}/step", json={"revision": failed["revision"]}).status_code == 409
    assert client.post(url, json={"revision": failed["revision"] - 1}).status_code == 409
    resumed = client.post(url, json={"revision": failed["revision"]}).json()
    assert resumed["status"] == "active" and resumed["executed"] == []
    assert resumed["review_log"] == [{"round": 1, "decision_index": 0, "outcome": "retry_requested",
                                      "decided_by": "demo-human"}]
    after = step(client, resumed)
    assert [r["action"] for r in after["records"]] == ["human_review", "counter"]
    assert backend.calls == 2


def test_backend_view_marks_http_identity_unverified_and_hides_paths():
    class Http:
        kind, is_model = "official_decide_http", True

        def describe(self):
            return {"configuration": {"official_source_dir": "/secret/path"},
                    "last_response_metadata": {"served_model_ids": ["snapshot-766c"], "endpoint_origin": "http://10.0.0.5:30001"}}

    view = backend_view(Http())
    assert view["server_identity_verified_by_client"] is False
    assert view["evidence_class"] == "real_model_run"
    assert view["served_model_ids_reported_by_server"] == ["snapshot-766c"]
    assert "/secret/path" not in json.dumps(view) and "10.0.0.5" not in json.dumps(view)


def test_saved_sessions_replay_read_only_without_hidden_parameters(tmp_path):
    client = TestClient(create_app(save_dir=tmp_path))
    assert client.get("/api/config").json()["saving_sessions"] is True
    done = drive(client, create(client, preset="start"))
    files = sorted(tmp_path.glob("session-*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["evidence_class"] == "cpu_reference_not_model_evidence"
    assert saved["approver"] == "human_dashboard" and saved["complete"] is True
    assert saved["outcome"] == done["outcome"]
    assert "supplier_hidden" in saved["evaluator_only"]            # kept in the operator's file only
    replayed = TestClient(create_app(tmp_path)).get("/api/replays/0").json()
    assert replayed["read_only"] is True and replayed["approval_mode"] == "human_dashboard"
    assert replayed["case_id"].startswith("payment_tradeoff-")
    assert [r["action"] for r in replayed["records"]] == [r["action"] for r in done["records"]]
    assert "unit_cost" not in json.dumps(replayed) and "reserve_profit" not in json.dumps(replayed)


def test_human_session_record_has_no_simulated_supplier(tmp_path):
    client = TestClient(create_app(save_dir=tmp_path))
    state = step(client, create(client, "human", "start"))
    assert state["scenario"] is None
    saved = json.loads(next(tmp_path.glob("session-*.json")).read_text(encoding="utf-8"))
    assert saved["case"]["counterparty"] == "person_typed_supplier"
    assert "evaluator_only" not in saved and saved["complete"] is False


def test_product_workspace_page_and_asked_questions_carry_only_public_data(client):
    page = client.get("/app")
    assert page.status_code == 200 and "workspace.js" in page.text
    state = step(client, create(client, preset="start"))
    asked = state["records"][-1]["model"]["asked"]
    assert len(asked) == state["records"][-1]["model"]["questions"]
    text = json.dumps(asked, ensure_ascii=False)
    private = default_buyer_private()
    for value in (private.target_total_cost, private.auto_bounds.max_total_cost, private.auto_bounds.max_unit_price):
        assert str(value) not in text and f"{value:,}" not in text
    assert "request" not in state["records"][-1]["model"]        # the raw request body stays server-side


def test_evidence_panel_reads_committed_summaries_only(client):
    evidence = client.get("/api/evidence").json()
    assert evidence == load_evidence()
    dev = evidence["development"]
    assert dev["evidence_class"] == "real_model_run" and dev["split"] == "dev" and dev["heldout_used"] is False
    by_key = {p["key"]: p for p in dev["policies"]}
    assert by_key["reference_true_acceptance"]["evaluator_only"] is True
    assert all(p["authority_violations"] == 0 for p in dev["policies"])
    against_rule = next(d for d in dev["differences"] if d["against"] == "rule_policy")
    assert against_rule["ci90"][0] < 0 < against_rule["ci90"][1]      # shown as uncertain, not as a win
    assert evidence["candidate_probe"]["route"].startswith("C_or_D")
    assert load_evidence(Path("/nonexistent-evidence-root")) == {"available": False}


# ---- 2026-11-20 product screen: typed supplier offers and buyer criteria set on the screen ----

from negotiation_agent.web import BuyerSettings, buyer_private, buyer_settings_view  # noqa: E402

CUSTOM = {"target_unit_price": 9_800, "max_unit_price": 10_100, "alternative_unit_price": 10_500,
          "max_delivery_days": 14, "min_payment_days": 45}


def supplier_send(client, state, **body):
    token = state["supplier_url"].rsplit("/", 1)[1]
    response = client.post(f"/api/supplier/{token}/response", json={"revision": state["revision"], **body})
    assert response.status_code == 200, response.text
    return client.get(f"/api/sessions/{state['id']}").json()


def test_default_settings_round_trip_to_the_stated_assumptions(client):
    defaults = client.get("/api/config").json()["buyer_defaults"]
    settings = {k: v for k, v in defaults.items() if k != "alternative_terms"}
    assert buyer_private(BuyerSettings(**settings)) == default_buyer_private()
    assert buyer_settings_view(default_buyer_private()) == defaults


def test_typed_negotiation_starts_empty_and_takes_the_opening_offer_from_the_person(client):
    state = client.post("/api/sessions", json={"mode": "human", "preset": "blank"}).json()
    assert state["status"] == "awaiting_supplier" and state["transcript"] == []
    assert state["facts"]["standing_offer_id"] is None
    assert client.post(f"/api/sessions/{state['id']}/step", json={"revision": state["revision"]}).status_code == 409
    state = supplier_send(client, state, kind="counter", unit_price=11_200, delivery_days=14, payment_days=60,
                          message="이번 달 안에 확정하면 이 조건 드립니다.")
    assert state["status"] == "active" and [r["offer_id"] for r in state["transcript"]] == ["S1"]
    state = step(client, state)
    assert state["status"] == "awaiting_supplier" and state["transcript"][-1]["by"] == "buyer"
    state = supplier_send(client, state, kind="counter", unit_price=10_900, delivery_days=14, payment_days=60,
                          message="이 조건으로 안 되면 이번 건은 접겠습니다.")
    assert [r["offer_id"] for r in state["transcript"]] == ["S1", "B1/S1", "S2"]
    # a blank start is for a person-typed supplier only
    assert client.post("/api/sessions", json={"mode": "simulated", "preset": "blank"}).status_code == 422


def test_buyer_settings_drive_authority_and_never_reach_the_backend_request():
    seen = []

    class Recording(KeywordBackend):
        def predict(self, state, candidates, request):
            seen.append(json.dumps(request, ensure_ascii=False))
            return super().predict(state, candidates, request)

    client = TestClient(create_app(backend=Recording()))
    state = client.post("/api/sessions", json={"mode": "simulated", "preset": "start", "buyer": CUSTOM}).json()
    assert {k: state["buyer_settings"][k] for k in CUSTOM} == CUSTOM
    assert state["auto_bounds"] == {"max_unit_price": 10_100, "max_delivery_days": 14, "min_payment_days": 45,
                                    "max_total_cost": 10_100_000}
    assert state["facts"]["target_total_cost"] == 9_800_000
    done = drive(client, state)
    assert done["outcome"]["authority_violations"] == []
    for row in done["executed"]:
        terms = row["terms"]
        if row["authority_basis"] == "auto":
            assert terms["unit_price"] <= 10_100 and terms["delivery_days"] <= 14 and terms["payment_days"] >= 45
    assert seen
    # Candidate unit prices are our own proposals and may equal a setting; the private totals never appear.
    company_total = buyer_private(BuyerSettings(**CUSTOM)).company_bounds.max_total_cost
    for text in seen:
        for value in (9_800_000, 10_100_000, company_total):
            assert str(value) not in text
        assert "target" not in text and "alternative" not in text and "대안" not in text


def test_buyer_settings_are_validated(client):
    for bad in ({**CUSTOM, "target_unit_price": 10_200},            # target above the negotiator limit
                {**CUSTOM, "max_unit_price": 10_500},               # limit not below the alternative quote
                {**CUSTOM, "max_delivery_days": 31},                # beyond the hard 30-day requirement
                {**CUSTOM, "seed": 3000}):                          # unknown field
        assert client.post("/api/sessions", json={"buyer": bad}).status_code == 422


def test_finished_deal_reports_against_baselines_fixed_at_the_start(client):
    state = client.post("/api/sessions", json={"mode": "simulated", "preset": "start",
                                               "scenario": "within_authority"}).json()
    first = next(row for row in state["transcript"] if row["by"] == "supplier")["terms"]
    done = drive(client, state)
    o = done["outcome"]
    assert done["status"] == "agreed" and o["first_offer_terms"] == first
    # Improvement over taking the first quote = saving now - saving the first quote would have given.
    assert o["saving_vs_first_offer"] > 0
    assert o["human_touches"] == len(done["approval_log"]) + len(done["review_log"])
    assert o["engine_time_ms"] >= 0
    no_deal = drive(client, client.post("/api/sessions", json={"mode": "simulated", "preset": "start",
                                                               "scenario": "no_deal"}).json())
    assert no_deal["outcome"]["saving_vs_first_offer"] == 0


def test_a_first_quote_past_the_hard_delivery_limit_gets_a_reply_not_a_500():
    """31+ days can never be accepted; the outcome summary must still build (it once raised)."""
    from fastapi.testclient import TestClient
    from negotiation_agent.web import create_app
    client = TestClient(create_app())
    view = client.post("/api/play", json={"unit_price": 11300, "delivery_days": 45, "payment_days": 60, "message": ""})
    assert view.status_code == 200
    assert view.json()["chat"][-1]["who"] == "agent"
    assert view.json()["chat"][-1]["terms"]["delivery_days"] <= 30
