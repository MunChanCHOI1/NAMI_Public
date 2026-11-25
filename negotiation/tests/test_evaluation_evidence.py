"""CPU doubles exercise saved evidence and throwaway held-out identities, never real weights/eval seeds."""

import argparse
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from negotiation_agent import evaluation as E
from negotiation_agent import provenance as V
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import request_sha256
from negotiation_agent.requests import FINAL_OFFER_QID


class RecordedBackend:
    kind, is_model, supports_final_offer = "recorded_test_double", True, True

    def __init__(self, fail_after=None, interrupt=False, invalid=None, before_call=lambda: None):
        self.calls = 0
        self.last_response = None
        self.fail_after, self.interrupt, self.invalid, self.before_call = fail_after, interrupt, invalid, before_call

    def predict(self, state, candidates, request):
        self.before_call()
        self.calls += 1
        if self.interrupt:
            raise KeyboardInterrupt("test interruption")
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("test backend failure")
        predictions = {qid: 0.5 for qid in request["questions"]}
        if self.invalid == "missing_final":
            predictions.pop(FINAL_OFFER_QID)
        elif self.invalid == "nan":
            predictions[FINAL_OFFER_QID] = float("nan")
        self.last_response = {
            "raw_probabilities": {qid: {"true": p, "false": 1 - p} for qid, p in predictions.items()},
            "metadata": {"request_sha256": request_sha256(request), "sequence": self.calls,
                         "option_order": list(request["questions"]), "dtype": "test_only",
                         "load_metadata": {"synthetic": True}},
        }
        return predictions

    def describe(self):
        return {"kind": self.kind, "is_model": True}


def attempts(result):
    return [record["model"] for case in result["cases"] for record in case["agent"]["internal_record"]
            if "trace_id" in record["model"]]


def test_evaluation_retains_each_public_request_response_and_decision_without_hidden_labels():
    backend, streamed = RecordedBackend(), []
    result = E.evaluate("dev", backend, count=2, on_decision=streamed.append)
    saved = attempts(result)
    assert result["complete"] and len(saved) == backend.calls
    assert result["policies"]["agent"]["process"]["model_requests"] == backend.calls
    assert len([r for r in streamed if r["policy"] == "agent"]) == sum(
        len(c["agent"]["internal_record"]) for c in result["cases"]
    )
    for call in saved:
        assert call["status"] == "ok" and call["latency_ms"] >= 0
        assert request_sha256(call["request"]) == call["request_sha256"] == call["trace_id"]
        assert call["response"]["metadata"]["request_sha256"] == call["request_sha256"]
        assert call["response"]["metadata"]["load_metadata"] == {"synthetic": True}
        assert call["predictions"][FINAL_OFFER_QID] == 0.5
        encoded = json.dumps(call["request"])
        assert all(name not in encoded for name in ("target_total_cost", "reserve_profit", "unit_cost", "supplier_hidden"))
    backend.last_response["metadata"]["load_metadata"]["synthetic"] = False
    assert all(call["response"]["metadata"]["load_metadata"]["synthetic"] for call in saved)
    assert "src/sessionguard/backends/zoro.py" in result["provenance"]["input_sha256"]
    assert "config/model_pins.json" in result["provenance"]["input_sha256"]


def test_failed_attempt_has_no_stale_response_and_is_counted():
    backend = RecordedBackend(fail_after=1)
    result = E.evaluate("dev", backend, count=2)
    saved = attempts(result)
    assert not result["complete"] and saved[0]["response"]["metadata"]["sequence"] == 1
    failures = [call for call in saved if call["status"] == "error"]
    assert failures and all(call["response"] is None and "test backend failure" in call["error"] for call in failures)
    assert len(saved) == backend.calls == result["policies"]["agent"]["process"]["model_requests"]


@pytest.mark.parametrize("invalid", ["missing_final", "nan"])
def test_invalid_final_estimate_preserves_response_without_scoring_or_caching_it(invalid):
    result = E.evaluate("dev", RecordedBackend(invalid=invalid), count=1)
    assert not result["complete"] and result["policies"]["agent"]["needs_review"] == 1
    call = attempts(result)[0]
    assert call["status"] == "error" and call["response"] is not None
    assert "invalid_predictions" in call and "predictions" not in call
    assert "agent" not in result["evaluator_only"]["acceptance_predictions"]
    json.dumps(result, allow_nan=False)


def test_interrupt_retains_the_attempt_and_limits_comparisons_to_started_cases():
    backend, streamed = RecordedBackend(interrupt=True), []
    result = E.evaluate("dev", backend, count=3, on_decision=streamed.append)
    assert result["interrupted"] and not result["complete"]
    assert result["config"]["cases"] == 3 and len(result["cases"]) == 1
    assert backend.calls == 1 and attempts(result)[0]["status"] == "interrupted"
    assert streamed[0]["decision"]["model"]["error"] == "KeyboardInterrupt: test interruption"
    assert all(summary["cases"] == 1 for summary in result["policies"].values())


def test_cli_flushes_decisions_before_the_next_call_and_preserves_model_outputs(tmp_path, monkeypatch):
    output = tmp_path / "model.json"
    decisions = output.with_suffix(".json.decisions.jsonl")
    observed = []
    backend = RecordedBackend()

    def before_call():
        if backend.calls:
            rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
            observed.append(rows)
            assert rows and rows[-1]["decision"]["model"]

    backend.before_call = before_call
    monkeypatch.setattr(E, "_build_backend", lambda *_: backend)
    argv = ["evaluate", "--split", "dev", "--cases", "2", "--backend", "zoro", "--output", str(output)]
    assert E.main(argv) == 0
    assert observed
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"] and result["decisions_file"] == str(decisions)
    assert len(attempts(result)) == backend.calls
    original = output.read_bytes(), decisions.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):
        E.main(argv)
    assert (output.read_bytes(), decisions.read_bytes()) == original


def test_cli_saves_initialization_failure_without_claiming_model_execution(tmp_path, monkeypatch):
    def fail(*_):
        raise RuntimeError("test loading failure")

    monkeypatch.setattr(E, "_build_backend", fail)
    output = tmp_path / "failure.json"
    assert E.main(["evaluate", "--backend", "zoro", "--output", str(output)]) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "initialization_failed" and not result["complete"]
    assert result["error"] == "RuntimeError: test loading failure"
    assert result.get("evidence_class") != "real_model_run"
    assert output.with_suffix(".json.decisions.jsonl").read_text(encoding="utf-8") == ""


def test_cli_preserves_completed_model_call_when_execution_after_decision_fails(tmp_path, monkeypatch):
    backend = RecordedBackend()

    def fail(*_):
        raise RuntimeError("test supplier execution failure")

    monkeypatch.setattr(E, "_build_backend", lambda *_: backend)
    monkeypatch.setattr(E.SimulatedSupplier, "respond", fail)
    output = tmp_path / "execution_failure.json"
    assert E.main(["evaluate", "--cases", "1", "--output", str(output)]) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "evaluation_failed" and "supplier execution failure" in result["error"]
    records = [json.loads(line) for line in output.with_suffix(".json.decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    saved = [row["decision"]["model"] for row in records if "trace_id" in row["decision"]["model"]]
    assert len(saved) == backend.calls > 0
    assert saved[-1]["status"] == "ok" and saved[-1]["response"]["metadata"]["sequence"] == backend.calls


@pytest.fixture()
def history(tmp_path, monkeypatch):
    monkeypatch.setitem(E.SPLIT_SEED_BASE, "eval", 9000)
    return tmp_path / "history"


@pytest.mark.parametrize("split", ["dev", "calibration", "eval"])
def test_oversized_split_is_rejected_before_generating_any_supplier(split, monkeypatch):
    def forbidden_generation(*_args, **_kwargs):
        pytest.fail("an out-of-range request must be rejected before supplier generation")

    monkeypatch.setattr(E, "make_supplier", forbidden_generation)
    with pytest.raises(ValueError, match="split seed ranges"):
        E.make_cases(split, E.MAX_CASES_PER_SPLIT + 1)


@pytest.mark.parametrize("next_count", [29, 31])
@pytest.mark.parametrize("legacy", [False, True])
def test_holdout_rejects_overlapping_counts_including_legacy_markers(tmp_path, history, next_count, legacy):
    E.record_exploratory(30, history)
    marker = next(history.glob("*.json"))
    if legacy:
        old = json.loads(marker.read_text(encoding="utf-8"))
        old.pop("case_sha256")
        old["schema_version"] = "1.0"
        marker.write_text(json.dumps(old), encoding="utf-8")
    manifest = tmp_path / "freeze.json"
    E.freeze({"cases": next_count, "backend": "heuristic"}, manifest)
    with pytest.raises(ValueError, match="overlapping cases"):
        E.reserve_heldout({"cases": next_count, "backend": "heuristic"}, manifest, next_count, history)
    assert not manifest.with_name(manifest.name + ".consumed.json").exists()
    assert len(list(history.glob("*.json"))) == 1


def test_case_id_rename_does_not_restore_unseen_status(tmp_path, history, monkeypatch):
    E.record_exploratory(2, history)
    make_cases = E.make_cases
    monkeypatch.setattr(E, "make_cases", lambda split, count: [
        replace(case, case_id="renamed-" + case.case_id) for case in make_cases(split, count)
    ])
    config, manifest = {"backend": "heuristic"}, tmp_path / "freeze.json"
    E.freeze(config, manifest)
    with pytest.raises(ValueError, match="overlapping cases"):
        E.reserve_heldout(config, manifest, 2, history)


def test_unverifiable_legacy_history_cannot_silently_be_ignored(tmp_path, history):
    E.record_exploratory(2, history)
    marker = next(history.glob("*.json"))
    old = json.loads(marker.read_text(encoding="utf-8"))
    old.pop("case_sha256")
    old["dataset_sha256"] = "0" * 64
    marker.write_text(json.dumps(old), encoding="utf-8")
    config, manifest = {"backend": "heuristic"}, tmp_path / "freeze.json"
    E.freeze(config, manifest)
    with pytest.raises(ValueError, match="legacy evaluation history"):
        E.reserve_heldout(config, manifest, 3, history)


def test_freeze_tracks_http_path_and_timeout_without_credentials(tmp_path, history):
    manifest = tmp_path / "freeze.json"
    base = ["--backend", "systemone", "--url", "https://user:secret@example.test/first?token=hidden",
            "--cases", "2"]
    assert E.main(["freeze", *base, "--output", str(manifest)], history_dir=history) == 0
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    config = frozen["execution_config"]
    assert config["endpoint_base_url"] == "https://example.test/first" and config["request_timeout"] == 120
    assert "secret" not in manifest.read_text(encoding="utf-8") and "hidden" not in manifest.read_text(encoding="utf-8")
    for changed in ({**config, "endpoint_base_url": "https://example.test/second"},
                    {**config, "request_timeout": 121}):
        with pytest.raises(ValueError, match="changed"):
            E.reserve_heldout(changed, manifest, 2, history)
    assert not history.exists()


def test_freeze_detects_checkpoint_metadata_and_shared_adapter_changes(tmp_path, history, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = checkpoint / "config.json"
    metadata.write_text('{"version": 1}', encoding="utf-8")
    config = {"backend": "zoro", "original_checkpoint": str(checkpoint)}
    manifest = tmp_path / "freeze.json"
    frozen = E.freeze(config, manifest)
    assert frozen["provenance"]["checkpoint_metadata_sha256"]["original_checkpoint"]["config.json"]
    metadata.write_text('{"version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        E.reserve_heldout(config, manifest, 2, history)
    metadata.write_text('{"version": 1}', encoding="utf-8")
    actual = E.provenance

    def changed_source(settings):
        snapshot = actual(settings)
        snapshot["input_sha256"]["src/sessionguard/backends/zoro.py"] = "changed"
        return snapshot

    monkeypatch.setattr(E, "provenance", changed_source)
    with pytest.raises(ValueError, match="changed"):
        E.reserve_heldout(config, manifest, 2, history)
    assert not history.exists()


# ---- guard corrections (2026-11-18; CPU doubles only, not executed on a GPU or against a real server) ----

HTTP_ARGS = {"zoro-http": ["--backend", "zoro-http", "--url", "http://127.0.0.1:30001", "--decision-temperature", "1"],
             "systemone": ["--backend", "systemone", "--url", "http://127.0.0.1:30001"],
             "zoro": ["--backend", "zoro"]}


def test_one_real_model_tuple_drives_the_recorded_label():
    assert E.REAL_MODEL_BACKENDS is V.REAL_MODEL_BACKENDS == ("zoro", "zoro-http", "systemone")
    parser = argparse.ArgumentParser()
    E._common(parser)
    for name in ("constant", "heuristic", "keyword", "zoro", "zoro-http", "systemone"):
        args = parser.parse_args([*HTTP_ARGS.get(name, ["--backend", name]), "--output", "unused.json"])
        recorded = E.execution_config(args, PolicyConfig(), None)
        assert recorded["real_model_requested"] is (name in E.REAL_MODEL_BACKENDS), name
    assert E.execution_config(SimpleNamespace(backend="heuristic", cases=30, approver="auto"), PolicyConfig(), None) == {
        "backend": "heuristic", "policy": json.loads(json.dumps(asdict(PolicyConfig()))),
        "buyer_private": asdict(E.default_buyer_private()),
        "cases": 30, "approver": "auto", "calibration": None, "real_model_requested": False}


@pytest.mark.parametrize("name", ["zoro-http", "systemone", "zoro"])
@pytest.mark.parametrize("existing", ["both", "result_only", "decisions_only"])
def test_existing_real_model_output_is_never_overwritten(tmp_path, monkeypatch, name, existing):
    output = tmp_path / "model.json"
    decisions = output.with_suffix(".json.decisions.jsonl")
    built = []

    def build(*_):
        built.append(1)
        return RecordedBackend()

    monkeypatch.setattr(E, "_build_backend", build)
    argv = ["evaluate", "--split", "dev", "--cases", "1", *HTTP_ARGS[name], "--output", str(output)]
    assert E.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"] and result["provenance"]["execution_config"]["real_model_requested"] is True
    assert result["provenance"]["execution_config"]["backend"] == name
    if name == "zoro-http":
        assert result["provenance"]["execution_config"]["reasoning"] is False
    if existing == "result_only":
        decisions.unlink()
    elif existing == "decisions_only":
        output.unlink()
    kept = {path: path.read_bytes() for path in (output, decisions) if path.exists()}
    with pytest.raises(SystemExit, match="already exists"):
        E.main(argv)
    assert built == [1]                                    # refused before a second backend was built
    assert {path: path.read_bytes() for path in (output, decisions) if path.exists()} == kept
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(path.name for path in kept)


def test_cpu_reference_output_may_still_be_rewritten(tmp_path):
    output = tmp_path / "reference.json"
    argv = ["evaluate", "--split", "dev", "--cases", "1", "--output", str(output)]
    assert E.main(argv) == 0 and E.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["provenance"]["execution_config"]["real_model_requested"] is False


def test_freeze_records_the_reasoning_flag_and_refuses_a_changed_or_ignored_one(tmp_path, history):
    plain, reasoned = tmp_path / "plain.json", tmp_path / "reasoned.json"
    base = [*HTTP_ARGS["zoro-http"], "--cases", "2"]
    assert E.main(["freeze", *base, "--output", str(plain)], history_dir=history) == 0
    assert E.main(["freeze", *base, "--reasoning", "--output", str(reasoned)], history_dir=history) == 0
    config = json.loads(plain.read_text(encoding="utf-8"))["execution_config"]
    with_flag = json.loads(reasoned.read_text(encoding="utf-8"))["execution_config"]
    assert config["reasoning"] is False and with_flag["reasoning"] is True
    assert config["real_model_requested"] is with_flag["real_model_requested"] is True
    assert with_flag == {**config, "reasoning": True}       # the flag is the only recorded difference
    assert "src/sessionguard/backends/zoro_http.py" in json.loads(plain.read_text(encoding="utf-8"))["input_sha256"]
    for frozen, changed in ((plain, with_flag), (reasoned, config)):
        with pytest.raises(ValueError, match="changed"):
            E.reserve_heldout(changed, frozen, 2, history)
    ignored = tmp_path / "ignored.json"
    for other in (["--backend", "zoro"], ["--backend", "systemone", "--url", "http://127.0.0.1:30001"], []):
        with pytest.raises(SystemExit, match="only valid with --backend zoro-http"):
            E.main(["freeze", *other, "--reasoning", "--cases", "2", "--output", str(ignored)], history_dir=history)
        with pytest.raises(SystemExit, match="only valid with --backend zoro-http"):
            E.main(["evaluate", *other, "--reasoning", "--cases", "2", "--output", str(ignored)], history_dir=history)
    assert not ignored.exists() and not ignored.with_suffix(".json.decisions.jsonl").exists()
    assert not history.exists()


@pytest.mark.parametrize("command", ["evaluate", "freeze", "sweep"])
@pytest.mark.parametrize("fitted_on", ["dev", "eval", None, "Calibration"])
def test_map_not_fitted_on_the_calibration_split_is_refused_everywhere(tmp_path, monkeypatch, history, command, fitted_on):
    def forbidden(*_args, **_kwargs):
        pytest.fail("a refused calibration file must not reach a backend, a freeze or a history marker")

    for name in ("_build_backend", "freeze", "reserve_heldout", "record_exploratory", "evaluate", "sweep"):
        monkeypatch.setattr(E, name, forbidden)
    platt = {"method": "platt", "a": 1.0, "b": 0.0, "n": 10, "source": "test"}
    if fitted_on is not None:
        platt["fitted_on_split"] = fitted_on
    path, output = tmp_path / "platt.json", tmp_path / "out.json"
    path.write_text(json.dumps(platt), encoding="utf-8")
    with pytest.raises(SystemExit, match="not fitted on the calibration split"):
        E.main([command, "--cases", "2", "--calibration", str(path), "--output", str(output)], history_dir=history)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["platt.json"] and not history.exists()


def test_map_fitted_on_the_calibration_split_is_still_frozen_and_applied(tmp_path, history):
    platt = {"method": "platt", "a": 0.9, "b": 0.2, "n": 10, "fitted_on_split": "calibration", "source": "test"}
    path, manifest, output = tmp_path / "platt.json", tmp_path / "freeze.json", tmp_path / "dev.json"
    path.write_text(json.dumps(platt), encoding="utf-8")
    common = ["--cases", "2", "--calibration", str(path)]
    assert E.main(["freeze", *common, "--output", str(manifest)], history_dir=history) == 0
    assert json.loads(manifest.read_text(encoding="utf-8"))["execution_config"]["calibration"] == platt
    assert E.main(["evaluate", "--split", "dev", *common, "--output", str(output)], history_dir=history) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["backend"]["calibration"] == {"method": "platt", "a": 0.9, "b": 0.2, "fitted_on": "test",
                                                "applies_to": "acceptance_questions_only"}
    assert not history.exists()
