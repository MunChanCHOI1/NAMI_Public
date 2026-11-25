import json

import pytest

from negotiation_agent import evaluation as E
from negotiation_agent.backends import CalibratedBackend, HeuristicBackend
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.probe import DEV_SEED_BASE, _spec


class StubModel:
    """Test double flagged as a model so that the rule policy is run as a separate baseline."""

    kind, is_model = "test_stub", True

    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, state, candidates, request):
        if self.fail:
            raise RuntimeError("CUDA out of memory (simulated)")
        return {c.candidate_id: 0.5 for c in candidates}

    def describe(self):
        return {"kind": self.kind, "is_model": True}


@pytest.fixture()
def protocol(tmp_path, monkeypatch):
    """Protocol tests run on throwaway seeds and a temporary history, never the real eval set."""
    monkeypatch.setitem(E.SPLIT_SEED_BASE, "eval", 9000)
    history = tmp_path / "history"

    def run(*argv):
        return E.main(list(argv), history_dir=history)

    return run, tmp_path, history


def test_splits_are_disjoint_deterministic_and_mixed():
    seeds = {split: {c.seed for c in E.make_cases(split, 30)} for split in E.SPLIT_SEED_BASE}
    assert not seeds["dev"] & seeds["calibration"] and not seeds["dev"] & seeds["eval"]
    assert not seeds["calibration"] & seeds["eval"]
    assert not seeds["dev"] & {DEV_SEED_BASE + i for i in range(len(_spec()))}     # probe states are separate
    assert E.make_cases("dev", 12) == E.make_cases("dev", 12)
    private = E.default_buyer_private()
    feasible = [E.attainable_saving(c, private, 1000) > 0 for c in E.make_cases("dev", 30)]
    assert any(feasible) and not all(feasible)


def test_development_run_reports_every_policy_under_the_same_gate():
    result = E.evaluate("dev", HeuristicBackend(), count=12)
    assert result["evaluation_mode"] == "development" and result["complete"]
    assert result["evidence_class"] == "cpu_reference_not_model_evidence" and result["agent_is_rule_policy"]
    assert set(result["policies"]) == {"agent", "fixed_concession", "midpoint", E.REFERENCE_POLICY}
    for name, summary in result["policies"].items():
        assert summary["authority_violations"] == 0, name
        assert summary["agreements_not_better_than_alternative"] == 0, name
        assert summary["infeasible_cases"]["ended_without_agreement_rate"] == 1.0, name
    assert "hidden supplier information" in result["policies"][E.REFERENCE_POLICY]["note"]
    estimates = result["policies"]["agent"]["acceptance_estimates"]
    assert estimates["n"] == len(result["evaluator_only"]["acceptance_predictions"]["agent"]) > 0
    assert "acceptance_estimates" not in result["policies"]["fixed_concession"]
    assert set(result["agent_minus"]) == {"fixed_concession", "midpoint", E.REFERENCE_POLICY}
    public_rows = json.dumps(result["cases"])
    assert "unit_cost" not in public_rows and "reserve_profit" not in public_rows


def test_baselines_cannot_bypass_approval_when_every_card_is_denied():
    result = E.evaluate("dev", HeuristicBackend(), count=12, approve=False)
    assert result["config"]["approver"] == "deny_all"
    assert all(s["authority_violations"] == 0 for s in result["policies"].values())


def test_candidate_truth_comes_from_copies_and_matches_what_happened_to_the_sent_offer(monkeypatch):
    monkeypatch.setattr(E.policy.time, "perf_counter", lambda: 0.0)  # deterministic evidence, excluding wall-clock noise
    private = E.default_buyer_private()
    for case in E.make_cases("dev", 8):
        row, records = E.run_case(case, private, HeuristicBackend(), E.policy.decide, PolicyConfig())
        again, records_again = E.run_case(case, private, HeuristicBackend(), E.policy.decide, PolicyConfig())
        assert (row, records) == (again, records_again)
        assert all(0.0 <= r["p"] <= 1.0 and isinstance(r["truth"], bool) for r in records)
        if row["status"] == "agreed" and row["counters_sent"]:
            assert row["realized_saving"] > 0


def test_model_backend_gets_the_rule_policy_as_a_separate_baseline():
    result = E.evaluate("dev", StubModel(), count=6)
    assert result["evidence_class"] == "real_model_run" and not result["agent_is_rule_policy"]
    assert "rule_policy" in result["policies"] and "rule_policy" in result["agent_minus"]


def test_backend_failure_makes_the_run_incomplete_instead_of_inventing_values():
    result = E.evaluate("dev", StubModel(fail=True), count=4)
    assert result["complete"] is False
    assert result["policies"]["agent"]["needs_review"] == 4
    assert result["policies"]["agent"]["mean_realized_saving"] == 0
    assert result["policies"]["fixed_concession"]["complete"] is True


def test_paired_difference_of_identical_runs_is_zero():
    rows = E.evaluate("dev", HeuristicBackend(), count=6)["cases"]
    flat = [{"case_id": r["case_id"], "realized_saving": r["agent"]["realized_saving"]} for r in rows]
    difference = E.paired_difference(flat, flat)
    assert difference["mean_difference"] == 0 and difference["ci90"] == [0, 0] and difference["ties"] == 6


def test_heldout_needs_a_matching_freeze_and_is_recorded_once(protocol):
    run, tmp, history = protocol
    out = tmp / "eval.json"
    with pytest.raises(SystemExit, match="freeze-manifest"):
        run("evaluate", "--split", "eval", "--cases", "4", "--output", str(out))
    assert not history.exists()

    manifest = tmp / "freeze.json"
    assert run("freeze", "--cases", "4", "--output", str(manifest)) == 0
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    assert frozen["evaluation_executed"] is False and frozen["input_sha256"]
    assert not history.exists()                                   # freezing runs no evaluation case

    with pytest.raises(SystemExit, match="changed"):               # a setting differs from the freeze
        run("evaluate", "--split", "eval", "--cases", "4", "--prior-concession", "90000",
            "--freeze-manifest", str(manifest), "--output", str(out))
    assert not history.exists()

    assert run("evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", str(manifest),
               "--output", str(out)) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["evaluation_mode"] == "heldout" and result["protocol"]["mode"] == "heldout"
    markers = list(history.glob("*.json"))
    assert len(markers) == 1 and markers[0].stem == result["protocol"]["dataset_sha256"]
    assert json.loads(markers[0].read_text(encoding="utf-8"))["first_recorded_mode"] == "heldout"

    with pytest.raises(SystemExit, match="already has an evaluation attempt"):
        run("evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", str(manifest), "--output", str(out))
    second = tmp / "freeze2.json"
    run("freeze", "--cases", "4", "--output", str(second))
    with pytest.raises(SystemExit, match="does not make the set unseen"):
        run("evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", str(second), "--output", str(out))

    assert run("evaluate", "--split", "eval", "--cases", "4", "--evaluation-mode", "exploratory",
               "--output", str(out)) == 0
    rerun = json.loads(out.read_text(encoding="utf-8"))
    assert rerun["evaluation_mode"] == "exploratory" and rerun["protocol"]["history_created"] is False
    assert len(list(history.glob("*.json"))) == 1


def test_an_exploratory_first_use_also_consumes_the_set(protocol):
    run, tmp, history = protocol
    out = tmp / "eval.json"
    assert run("evaluate", "--split", "eval", "--cases", "4", "--evaluation-mode", "exploratory",
               "--output", str(out)) == 0
    marker = next(history.glob("*.json"))
    assert json.loads(marker.read_text(encoding="utf-8"))["first_recorded_mode"] == "exploratory"
    manifest = tmp / "freeze.json"
    run("freeze", "--cases", "4", "--output", str(manifest))
    with pytest.raises(SystemExit, match="does not make the set unseen"):
        run("evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", str(manifest), "--output", str(out))


def test_development_runs_never_touch_the_holdout_history(protocol):
    run, tmp, history = protocol
    assert run("evaluate", "--split", "dev", "--cases", "4", "--no-records", "--output", str(tmp / "dev.json")) == 0
    assert run("sweep", "--cases", "4", "--grid", "0", "300000", "--output", str(tmp / "sweep.json")) == 0
    assert not history.exists()
    data = json.loads((tmp / "dev.json").read_text(encoding="utf-8"))
    assert "cases" not in data and "evaluator_only" not in data
    sweep = json.loads((tmp / "sweep.json").read_text(encoding="utf-8"))
    assert sweep["split"] == "dev" and sweep["parameter"] == "prior_concession"
    assert sweep["selected_value"] in (0, 300_000)


def test_calibration_is_fitted_only_on_the_calibration_split(protocol):
    run, tmp, history = protocol
    dev, cal, fitted = tmp / "dev.json", tmp / "cal.json", tmp / "platt.json"
    run("evaluate", "--split", "dev", "--cases", "6", "--output", str(dev))
    with pytest.raises(SystemExit, match="calibration split"):
        run("fit-calibration", "--input", str(dev), "--output", str(fitted))
    run("evaluate", "--split", "calibration", "--cases", "12", "--output", str(cal))
    assert run("fit-calibration", "--input", str(cal), "--output", str(fitted)) == 0
    platt = json.loads(fitted.read_text(encoding="utf-8"))
    assert platt["fitted_on_split"] == "calibration" and platt["brier_after_in_sample"] <= platt["brier_before"]
    out = tmp / "dev_calibrated.json"
    assert run("evaluate", "--split", "dev", "--cases", "6", "--calibration", str(fitted), "--output", str(out)) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["backend"]["calibration"]["method"] == "platt"
    assert "rule_policy" in result["policies"]                    # the uncalibrated rule stays as a baseline
    with pytest.raises(ValueError, match="uncalibrated"):          # a calibrated run cannot be refitted
        E.fit_calibration({**json.loads(cal.read_text(encoding="utf-8")), "backend": result["backend"]})
    assert not history.exists()


def test_calibrated_backend_keeps_order_and_range():
    backend = CalibratedBackend(HeuristicBackend(), a=0.9, b=1.3)
    case = E.make_cases("dev", 1)[0]
    state = E.initial_state(case.case_id, case.hidden)
    candidates = E.policy.generate_candidates(state, E.default_buyer_private(), PolicyConfig())
    request = E.policy.build_decision_request(state, candidates)
    raw = HeuristicBackend().predict(state, candidates, request)
    mapped = backend.predict(state, candidates, request)
    assert all(0.0 < v < 1.0 for v in mapped.values())
    assert sorted(raw, key=raw.get) == sorted(mapped, key=mapped.get)
