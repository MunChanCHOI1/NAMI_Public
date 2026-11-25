import json
from types import SimpleNamespace

import pytest

from negotiation_agent import metrics as M, probe as P, provenance as V
from negotiation_agent.backends import CalibratedBackend, ConstantBackend, ZoroNativeBackend


def test_native_response_keeps_full_evidence_and_clears_after_failure():
    request = {"state": {}, "questions": {"final_offer": {"type": "noul"}}}
    response = {"answers": {"final_offer": {"type": "noul", "noul": 0.7}},
                "raw_probabilities": {"final_offer": {"false": 0.3, "true": 0.7}},
                "raw_logits": {"final_offer": {"false": 1.0, "true": 2.0}},
                "metadata": {"request_sha256": V.request_sha256(request), "head_file_sha256": "head-hash",
                             "question_option_order": {"final_offer": ["false", "true"]},
                             "input_tokens_per_question": [123], "backend_latency_ms": 5.0}}
    native = ZoroNativeBackend.__new__(ZoroNativeBackend)
    native.configuration = {"max_input_tokens": 2048}
    native._backend = SimpleNamespace(decide=lambda value: response)
    assert native.predict(None, [], request) == {"final_offer": 0.7}
    saved = V.response_snapshot(native)
    response["metadata"]["input_tokens_per_question"][0] = 999
    assert saved["metadata"]["input_tokens_per_question"] == [123]
    assert native.describe()["last_response_metadata"]["head_file_sha256"] == "head-hash"
    assert saved["raw_probabilities"]["final_offer"]["true"] == 0.7

    def fail(value):
        raise RuntimeError("simulated model error")

    native._backend.decide = fail
    with pytest.raises(RuntimeError):
        native.predict(None, [], request)
    assert V.response_snapshot(native) is None and native.last_metadata == {}
    assert saved["metadata"]["question_option_order"] == {"final_offer": ["false", "true"]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, "missing"])
def test_malformed_predictions_are_atomic_failed_calls(bad):
    class Broken(ConstantBackend):
        supports_final_offer = False

        def predict(self, state, candidates, request):
            values = super().predict(state, candidates, request)
            last = candidates[-1].candidate_id  # early rows must not be partially committed
            if bad == "missing":
                del values[last]
            else:
                values[last] = bad
            self.last_response = {"returned": dict(values)}
            return values

    saved = []
    result = P.run(Broken(), ladders=("price",), max_states=1, variant_states=0, on_call=saved.append)
    assert not result["complete"] and result["records"] == []
    assert len(saved) == 1 and saved[0]["status"] == "error"
    assert saved[0]["request_sha256"] == V.request_sha256(saved[0]["request"])
    assert saved[0]["response"] is not None and saved[0]["latency_ms"] >= 0
    json.dumps(result, allow_nan=False)  # invalid response values remain explicit but writable


def test_one_supplier_group_has_no_calibration_or_confidence_interval():
    result = P.run(ConstantBackend(), ladders=("price",), max_states=1, variant_states=0)
    assert result["complete"]
    cal = result["metrics"]["calibration"]
    assert cal["brier_skill_platt"] is None and cal["platt_cv"]["reason"] == "fewer_than_two_groups"
    assert result["metrics"]["decision_value"]["gain_vs_rule_ci90_by_group"] is None
    assert not result["verdict"]["checks"]["value_vs_rule_ok"]
    final = result["final_offer_probe"]
    assert final["questions_scored"] == 1 and final["metrics"]["platt_cv"]["available"] is False


def test_platt_leaves_both_statement_twins_out(monkeypatch):
    records = [{"state_id": name, "group_id": group, "p": p, "truth": truth}
               for name, group, p, truth in (("a", "A", .1, False), ("a-said", "A", .2, False),
                                              ("b", "B", .8, True), ("b-said", "B", .9, True))]
    training = []

    def fit(xs, ys):
        training.append((xs, ys))
        return 1.0, 0.0

    monkeypatch.setattr(M, "_fit_platt", fit)
    assert M.platt_cross_validated(records) == pytest.approx([.1, .2, .8, .9])
    assert training == [([M._logit(.8), M._logit(.9)], [1.0, 1.0]),
                        ([M._logit(.1), M._logit(.2)], [0.0, 0.0])]


def test_bootstrap_never_separates_twins(monkeypatch):
    records = [{"state_id": state, "group_id": group}
               for state, group in (("a", "A"), ("a-said", "A"), ("b", "B"), ("b-said", "B"))]
    values = {"a": 100.0, "a-said": -100.0, "b": 200.0, "b-said": -200.0}
    monkeypatch.setattr(M, "one_step_choice_value", lambda rows, key:
                        values if key == "p" else {k: 0.0 for k in values})
    result = M.decision_value(records, resamples=100)
    assert result["groups"] == 2 and result["bootstrap_unit"] == "supplier_group"
    assert result["gain_vs_rule_ci90_by_group"] == [0.0, 0.0]


def test_acceptance_calibration_does_not_change_final_offer():
    calibrated = CalibratedBackend(ConstantBackend(.2), a=2.0, b=1.0)
    request = {"questions": {"candidate": {}, "final_offer": {}}}
    result = calibrated.predict(None, [], request)
    assert result["final_offer"] == .2 and result["candidate"] != .2
    assert calibrated.describe()["calibration"]["applies_to"] == "acceptance_questions_only"
    assert V.response_snapshot(calibrated)["uncalibrated_predictions"]["candidate"] == .2


def test_probe_persists_initialization_failure_with_source_snapshot(tmp_path, monkeypatch):
    def fail(args):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(P, "make_backend", fail)
    output = tmp_path / "failed.json"
    assert P.main(["--output", str(output)]) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "initialization_failed" and not result["complete"]
    assert "src/sessionguard/backends/zoro.py" in result["provenance"]["input_sha256"]
    assert result["errors"][0]["stage"] == "backend_initialization"


def test_probe_flushes_public_calls_and_keeps_final_questions_separate(tmp_path):
    output = tmp_path / "probe.json"
    assert P.main(["--backend", "constant", "--max-states", "1", "--ladders", "price",
                   "--variant-states", "0", "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    calls = [json.loads(row) for row in output.with_suffix(".json.calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert calls == result["calls"] and len(calls) == 2
    assert len(calls[0]["request"]["questions"]) == 5
    assert list(calls[1]["request"]["questions"]) == ["final_offer"]
    assert result["metrics"]["questions_scored"] == 5
    assert result["final_offer_probe"]["questions_scored"] == 1
    assert "truth" not in json.dumps(calls) and "hidden" not in json.dumps(calls)


def test_provenance_binds_shared_sources_and_checkpoint_metadata(tmp_path):
    for name in V.SHARED_INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    package = tmp_path / "negotiation/src/negotiation_agent"
    package.mkdir(parents=True)
    (package / "policy.py").write_text("# policy", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    codec = source / "decide_request.py"
    codec.write_text("# codec", encoding="utf-8")
    pins = {"official_source_files": {"inference/zoro-1/decide_request.py": V.file_sha256(codec)}}
    (tmp_path / "config/model_pins.json").write_text(json.dumps(pins), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"not-a-real-model")
    settings = {"official_source_dir": str(source), "original_checkpoint": str(checkpoint)}
    before = V.provenance(settings, root=tmp_path)
    assert before["official_source_files"]["inference/zoro-1/decide_request.py"]["matches_pin"]
    assert before["checkpoint_metadata_sha256"]["original_checkpoint"]["temperature.json"] is None
    assert "model.safetensors" not in before["checkpoint_metadata_sha256"]["original_checkpoint"]
    (tmp_path / "src/sessionguard/contracts.py").write_text("# changed", encoding="utf-8")
    config.write_text('{"changed":true}', encoding="utf-8")
    after = V.provenance(settings, root=tmp_path)
    assert before["input_sha256"] != after["input_sha256"]
    assert before["checkpoint_metadata_sha256"] != after["checkpoint_metadata_sha256"]


def test_http_configuration_keeps_route_but_excludes_credentials():
    args = SimpleNamespace(backend="systemone", url="https://user:secret@example.com:8443/prefix/?token=secret#frag",
                           model="zoro-1", request_timeout=150)
    settings = V.backend_execution_config(args)
    assert settings["endpoint_base_url"] == "https://example.com:8443/prefix"
    assert settings["request_timeout"] == 150 and "secret" not in json.dumps(settings)
