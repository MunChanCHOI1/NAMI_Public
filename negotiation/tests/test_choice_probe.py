"""Direct-choice diagnostics use public requests; CPU doubles are never real-model evidence."""

import copy
import json
from dataclasses import replace

import pytest

from negotiation_agent import choice_probe as C
from negotiation_agent.backends import ZoroNativeBackend
from negotiation_agent.probe import build_ladder, build_states
from negotiation_agent.provenance import request_sha256


def response_for(request, choice=None):
    keys = list(request["questions"][C.QID]["criteria"])
    choice = choice or keys[0]
    values = {key: 0.8 if key == choice else 0.2 / (len(keys) - 1) for key in keys}
    return {"answers": {C.QID: {"type": "choice", "choice": choice, "probabilities": values}},
            "raw_probabilities": {C.QID: values}, "raw_logits": {C.QID: values},
            "metadata": {"request_sha256": request_sha256(request), "question_option_order": {C.QID: keys},
                         "load_metadata": {"test_only": True}}}


class TypedDouble:
    kind, is_model = "typed_cpu_test_double", False

    def __init__(self, fail_at=None, interrupt_at=None):
        self.calls = 0
        self.fail_at, self.interrupt_at = fail_at, interrupt_at
        self.last_response = None

    def decide_typed(self, request):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic call failure")
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt("synthetic interruption")
        self.last_response = response_for(request)
        return self.last_response

    def describe(self):
        return {"kind": self.kind, "is_model": self.is_model}


def test_fixed_plan_asks_all_24_states_in_both_orders_without_private_or_hidden_inputs():
    experiment = C.prepare_experiment()
    assert len(experiment["calls"]) == 48 and len(experiment["evaluator_only"]) == 24
    assert len({row["group_id"] for row in experiment["evaluator_only"]}) == 16
    for first, reverse in zip(experiment["calls"][::2], experiment["calls"][1::2]):
        assert first["state_id"] == reverse["state_id"]
        assert first["request"]["state"] == reverse["request"]["state"]
        left = first["request"]["questions"][C.QID]["criteria"]
        right = reverse["request"]["questions"][C.QID]["criteria"]
        assert len(left) == 5 and left == right and list(left) == list(reversed(right))
        assert first["request_sha256"] != reverse["request_sha256"]
        serialized = json.dumps(first["request"])
        for forbidden in ("state_id", "group_id", "truth", "reserve_profit", "target_total_cost", "auto_bounds", "style"):
            assert forbidden not in serialized
    probe = build_states()[0]
    changed = replace(probe, hidden=replace(probe.hidden, reserve_profit=probe.hidden.reserve_profit + 1_000_000))
    assert C.prepare_experiment([probe])["calls"] == C.prepare_experiment([changed])["calls"]


@pytest.mark.parametrize("bad", ["missing", "nan", "sum", "choice", "argmax", "hash", "order"])
def test_invalid_typed_choice_is_rejected_atomically(bad):
    probe = build_states()[0]
    request = C.build_choice_request(probe.state, build_ladder(probe.state, "mixed"))
    response = response_for(request)
    values = response["raw_probabilities"][C.QID]
    if bad == "missing":
        values.pop("M5")
    elif bad == "nan":
        values["M1"] = float("nan")
    elif bad == "sum":
        values["M1"] = 0.9
    elif bad == "choice":
        response["answers"][C.QID]["choice"] = "not_a_candidate"
    elif bad == "argmax":
        response["answers"][C.QID]["choice"] = "M2"
    elif bad == "hash":
        response["metadata"]["request_sha256"] = "stale"
    else:
        response["metadata"]["question_option_order"][C.QID].reverse()
    with pytest.raises(ValueError):
        C.validate_choice(response, request)


def test_native_typed_method_preserves_choice_response_and_clears_failed_metadata():
    class Adapter:
        fail = False

        def decide(self, request):
            if self.fail:
                raise RuntimeError("adapter failed")
            return response_for(request)

    backend = ZoroNativeBackend.__new__(ZoroNativeBackend)
    backend._backend = Adapter()
    probe = build_states()[0]
    request = C.build_choice_request(probe.state, build_ladder(probe.state, "mixed"))
    response = backend.decide_typed(request)
    assert response["answers"][C.QID]["choice"] == "M1"
    assert backend.last_metadata["question_option_order"][C.QID] == ["M1", "M2", "M3", "M4", "M5"]
    backend._backend.fail = True
    with pytest.raises(RuntimeError):
        backend.decide_typed(request)
    assert backend.last_response is None and backend.last_metadata == {}


def test_primary_is_mean_of_per_state_and_not_min_of_order_means():
    experiment = {"evaluator_only": [
        {"state_id": sid, "group_id": sid, "truth": {"yes": True, "no": False}, "informative": True,
         "heuristic_choice": "no", "uniform_expected_hit": 0.5} for sid in ("a", "b")
    ]}
    calls = [{"state_id": sid, "order": order, "status": "ok", "choice": choice}
             for sid, pair in (("a", ("yes", "no")), ("b", ("no", "yes")))
             for order, choice in zip(C.ORDERS, pair)]
    summary = C.summarize(experiment, calls, True)
    assert summary["primary"]["both_orders_hit_rate"] == 0
    assert summary["secondary"]["informative_hit_rate_by_order"] == {"forward": 0.5, "reverse": 0.5}
    assert summary["primary"]["uniform_expected_hit_rate"] == 0.5  # one semantic choice reused, not independent draws
    assert summary["conclusion"] == "direct_choice_improvement_not_demonstrated"


def test_group_bootstrap_keeps_twins_together_and_does_not_infer_from_one_group(monkeypatch):
    rows = [{"group_id": group, "both_orders_hit": hit, "heuristic_hit": baseline}
            for group, hit, baseline in (("a", 1, 0), ("a", 1, 0), ("b", 0, 1))]

    class Alternate:
        def __init__(self, seed):
            self.index = 0

        def choice(self, keys):
            value = keys[self.index % len(keys)]
            self.index += 1
            return value

    monkeypatch.setattr(C.random, "Random", Alternate)
    result = C.paired_group_interval(rows, resamples=20)
    assert result["states"] == 3 and result["groups"] == 2
    assert result["mean_gain"] == pytest.approx(1 / 3) and result["ci90"] == pytest.approx([1 / 3, 1 / 3])
    assert C.paired_group_interval(rows[:2])["ci90"] is None


def test_failed_call_keeps_evidence_without_stale_response_or_partial_state_success():
    experiment = C.prepare_experiment(build_states()[:2])
    backend, journal = TypedDouble(fail_at=2), []
    result = C.run(backend, experiment, on_call=journal.append)
    assert not result["complete"] and len(journal) == 4 and len(result["errors"]) == 1
    assert journal[1]["status"] == "error" and journal[1]["response"] is None
    assert journal[0]["response"]["metadata"]["load_metadata"] == {"test_only": True}
    assert result["metrics"]["secondary"]["completed_state_pairs"] == 1
    assert result["metrics"]["conclusion"] == "incomplete_do_not_interpret"
    assert result["counts_as_model_evidence"] is False


def test_interrupt_is_journalled_and_stops_without_retry():
    backend, journal = TypedDouble(interrupt_at=2), []
    result = C.run(backend, C.prepare_experiment(build_states()[:2]), on_call=journal.append)
    assert result["interrupted"] and not result["complete"] and backend.calls == 2
    assert journal[-1]["status"] == "interrupted" and journal[-1]["request"]


def test_manifest_exists_before_loading_and_results_are_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "choice.json"
    calls = output.with_suffix(".json.calls.jsonl")
    manifest = output.with_suffix(".json.manifest.json")

    def load(args):
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        assert frozen["model_execution_started"] is False and len(frozen["planned_calls"]) == 48
        assert frozen["protocol"]["primary_metric"] == "mean_per_state(hit_forward AND hit_reverse)"
        assert frozen["provenance"]["input_sha256"]["negotiation/src/negotiation_agent/choice_probe.py"]
        assert output.exists() and calls.exists()
        return TypedDouble()

    monkeypatch.setattr(C, "make_backend", load)
    argv = ["--original-checkpoint", str(tmp_path / "orig"), "--exported-checkpoint", str(tmp_path / "export"),
            "--output", str(output)]
    assert C.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"] and len(result["calls"]) == 48
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 48
    saved = [path.read_bytes() for path in (manifest, output, calls)]
    with pytest.raises(SystemExit):
        C.main(argv)
    assert saved == [path.read_bytes() for path in (manifest, output, calls)]


def test_model_initialization_failure_still_preserves_manifest_and_error(tmp_path, monkeypatch):
    def fail(args):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(C, "make_backend", fail)
    output = tmp_path / "failed.json"
    assert C.main(["--original-checkpoint", "unused", "--exported-checkpoint", "unused", "--output", str(output)]) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "initialization_failed" and not result["complete"]
    assert "synthetic load failure" in result["error"] and result.get("counts_as_model_evidence") is not True
    assert output.with_suffix(".json.manifest.json").exists()


def test_noul_reference_requires_identical_public_state_candidates_and_pins(tmp_path):
    experiment = C.prepare_experiment(build_states()[:2])
    from negotiation_agent.requests import build_acceptance_request

    reference = {"experiment": "candidate_sensitivity_probe", "complete": True, "errors": [],
                 "backend": {"is_model": True, "gpu_name": "different_test_gpu"},
                 "provenance": {"model_pins": {"test_pin": "same"}}, "records": [], "calls": []}
    for probe, scoring in zip(build_states()[:2], experiment["evaluator_only"]):
        candidates = build_ladder(probe.state, "mixed")
        request = build_acceptance_request(probe.state, candidates)
        predicted = {c.candidate_id: 0.5 for c in candidates}
        reference["calls"].append({"state_id": probe.state_id, "ladder": "mixed", "variant": "base", "status": "ok",
                                   "request": request, "request_sha256": request_sha256(request), "predictions": predicted})
        reference["records"].extend({"state_id": probe.state_id, "ladder": "mixed", "variant": "base", **row, "p": 0.5}
                                    for row in scoring["candidates"])
    path = tmp_path / "noul.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    matched = C.load_noul_reference(path, experiment, {"model_pins": {"test_pin": "same"}})
    assert matched["matched_state_count"] == 2
    result = C.run(TypedDouble(), experiment)
    comparison = C.supplementary_comparison(matched, result)
    assert comparison["affects_primary_conclusion"] is False
    changed = copy.deepcopy(reference)
    changed["records"][0]["terms"]["unit_price"] += 10
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="terms"):
        C.load_noul_reference(path, experiment, {"model_pins": {"test_pin": "same"}})
    path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError, match="pins"):
        C.load_noul_reference(path, experiment, {"model_pins": {"test_pin": "different"}})
