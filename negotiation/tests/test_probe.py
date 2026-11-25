import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from negotiation_agent.backends import ConstantBackend, HeuristicBackend, SystemOneBackend
from negotiation_agent.probe import build_ladder, build_states, main, run
from negotiation_agent.requests import build_acceptance_request
from negotiation_agent.simulator import is_last_chance, would_accept


class _Stub:
    """Test double standing in for a model. It is handed the answers; it proves nothing about a model."""

    kind, is_model = "test_stub", True

    def __init__(self, fail_on=None, noise=0.0):
        self.truth = {(p.state_id, c.terms): would_accept(p.state, p.hidden, c.terms)
                      for p in build_states() for ladder in ("price", "payment", "delivery", "mixed")
                      for c in build_ladder(p.state, ladder)}
        self.final_truth = {p.state_id: is_last_chance(p.state, p.hidden) for p in build_states()}
        self.fail_on, self.calls = fail_on, 0

    def predict(self, state, candidates, request):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("CUDA out of memory (simulated)")
        out = {c.candidate_id: 0.9 if self.truth[(state.case_id, c.terms)] else 0.1 for c in candidates}
        if "final_offer" in request["questions"]:
            out["final_offer"] = 0.9 if self.final_truth[state.case_id] else 0.1
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True}


def test_flat_output_fails_the_gates_even_though_it_is_monotone():
    result = run(ConstantBackend(), variant_states=0)
    checks = result["verdict"]["checks"]
    assert all(v["non_decreasing_rate"] == 1.0 for v in result["metrics"]["monotonicity"].values())
    assert all(v["responsive_rate"] == 0.0 for v in result["metrics"]["monotonicity"].values())
    assert not checks["responsive"] and not checks["ranking_ok"]
    assert result["verdict"]["route"].startswith("C_or_D")
    assert result["verdict"]["counts_as_model_evidence"] is False


def test_rule_baseline_is_never_reported_as_model_evidence():
    result = run(HeuristicBackend(), variant_states=0)
    assert result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["verdict"]["counts_as_model_evidence"] is False
    assert result["verdict"]["checks"]["ranking_beats_rule"] is False
    value = result["metrics"]["decision_value"]
    assert value["mean_saving_always_max_saving"] < value["mean_saving_rule_heuristic"] < value["mean_saving_oracle"]
    assert result["metrics"]["by_subset"]["mixed_sets"]["rule"]["within_state_pair_accuracy"] < 1.0


def test_an_informed_predictor_reaches_route_a():
    result = run(_Stub(), variant_states=0)
    assert result["complete"] and result["verdict"]["route"] == "A_probability_expected_value"
    assert result["metrics"]["questions_scored"] == 24 * 19
    assert result["final_offer_probe"]["complete"] and result["final_offer_probe"]["questions_scored"] == 24
    assert result["final_offer_probe"]["metrics"]["auc"] == 1.0
    assert result["scope"]["supplier_groups"] == 16


def test_backend_failure_is_recorded_and_blocks_the_verdict():
    result = run(_Stub(fail_on=2), ladders=("price", "mixed"), max_states=4, variant_states=0)
    assert result["complete"] is False and len(result["errors"]) == 1
    assert "out of memory" in result["errors"][0]["error"]
    assert result["metrics"]["questions_scored"] == (4 * 2 - 1) * 5     # the failed request has no rows
    assert result["verdict"]["counts_as_model_evidence"] is False


def test_cli_writes_a_labelled_summary(tmp_path):
    out = tmp_path / "probe.json"
    assert main(["--backend", "heuristic", "--quick", "--no-records", "--output", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scope"]["states"] == 12 and "records" not in data and "evaluator_only" not in data
    assert data["scope"]["development_seeds_only"] is True


def test_systemone_backend_uses_the_current_envelope():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])).decode("utf-8"))
            seen["path"], seen["body"] = self.path, body
            answers = {qid: {"type": "noul", "noul": 0.25} for qid in body["questions"]}
            payload = json.dumps({"model": "served", "answers": answers,
                                  "usage": {"input_tokens": 10, "output_tokens": 0}}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        probe = build_states()[0]
        candidates = build_ladder(probe.state, "price")
        request = build_acceptance_request(probe.state, candidates)
        backend = SystemOneBackend(f"http://127.0.0.1:{server.server_port}")
        predicted = backend.predict(probe.state, candidates, request)
    finally:
        server.shutdown()
        server.server_close()
    assert seen["path"] == "/v1/systemone"
    assert set(seen["body"]) == {"model", "state", "questions"}       # no deprecated "request" wrapper
    assert predicted == {c.candidate_id: 0.25 for c in candidates}
    assert backend.describe()["served_model"] == "served"
