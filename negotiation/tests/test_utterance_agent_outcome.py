"""CPU tests of the negotiation comparison, the outcome rule, the report and the demo (track B).

NOT EXECUTED ON A GPU as of 2026-11-18. Every "model" here is the CPU stand-in or a test double
flagged as a model only so that the labelling paths of a real-model run can be exercised. No number
in this file is model evidence.

Case use: negotiations run on the smoke block (utt-6000..6009), on dev-1100..1109 and on scenario
seed 1073 only. The outcome rule, the adoption decision and the stopping rule are tested with
invented rows and invented result files. No case of the fit, bench or test block is negotiated or
scored (the protocol only hashes each block's case list); the one test of the test-block path
replaces that block's cases by two smoke cases.
"""

import importlib.util
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from negotiation_agent import evaluation, policy, web
from negotiation_agent.estimate_sources import ComposedBackend
from negotiation_agent.evaluation import make_cases
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.requests import FINAL_OFFER_QID, build_decision_request
from negotiation_agent.simulator import default_buyer_private, initial_state

from utterance_agent import PROTOCOL_VERSION
from utterance_agent import bench as B
from utterance_agent import cases as C
from utterance_agent import experiment as X
from utterance_agent import outcome as O
from utterance_agent import wording as W
from utterance_agent.interpreter import FIXED_HALF, Thresholds, UtteranceFinalBackend

NEGOTIATION = Path(__file__).resolve().parents[1]
PRIVATE = default_buyer_private()
PINS = "final_limit=L1,deadline_pressure=D1,negated_keyword=K1,neutral=N1"
ROW_FIELDS = ("status", "agreement", "realized_saving", "net_saving", "rounds_used", "counters_sent",
              "approval_cards", "authority_violations")
SMOKE_ARMS = {"paraphrase": "prior,keyword,label_reader,true_final,utt_gate_fit,model_final,m0,model_final_gate",
              "adversarial": "prior,keyword,label_reader,true_final,utt_gate_fit",
              "default": "prior,keyword,utt_gate_fit"}


def _load_demo():
    spec = importlib.util.spec_from_file_location("utterance_demo_under_test", NEGOTIATION / "utterance_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEMO = _load_demo()


@pytest.fixture(autouse=True)
def _private_registry(tmp_path, monkeypatch):
    """No test may write the checkout's real first-use registry."""
    monkeypatch.setattr(X, "REGISTRY_PATH", tmp_path / "registry.json")


def _stand_in(state: Path, *extra: str) -> list[str]:
    return ["--state-dir", str(state), "--backend", "cpu-stand-in", *extra]


def _run(state: Path, style: str, arms: str, *extra: str, **seams) -> int:
    return X.main(["run", "--style", style, "--block", "smoke", "--arms", arms, *_stand_in(state, *extra)], **seams)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A rehearsal state directory up to the fit: plan, smoke bench through the store, fit --rehearsal."""
    root = tmp_path_factory.mktemp("uo")
    state, registry = root / "s", root / "registry.json"
    assert X.main(["plan", "--state-dir", str(state)], registry_path=registry) == 0
    assert X.main(["bench", "--block", "smoke", *_stand_in(state)], registry_path=registry) == 0
    assert X.main(["fit", "--state-dir", str(state), "--rehearsal"], registry_path=registry) == 0
    return SimpleNamespace(state=state, registry=registry)


@pytest.fixture(scope="module")
def rehearsal(fitted, tmp_path_factory):
    """The whole CPU rehearsal: the fitted directory, then every runnable arm of each style, then the report."""
    root = tmp_path_factory.mktemp("ur")
    state, registry = root / "s", root / "registry.json"
    shutil.copytree(fitted.state, state)
    codes = {style: _run(state, style, arms, "--budget-minutes", "30", registry_path=registry)
             for style, arms in SMOKE_ARMS.items()}
    codes["report"] = X.main(["report", "--state-dir", str(state)], registry_path=registry)
    return SimpleNamespace(state=state, registry=registry, codes=codes)


def _copy(source, tmp_path) -> Path:
    target = tmp_path / "c"
    shutil.copytree(source.state, target)
    return target


def _files(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file()}


def _attempts(state: Path, style: str, arm: str, block: str = "smoke") -> list[dict]:
    return X.attempts(O.arm_folder(state, block, style, arm))


def _result(state: Path, style: str, arm: str, index: int = 0) -> dict:
    return _attempts(state, style, arm)[index]["result"]


class ModelDouble(X.UtteranceCpuStandIn):
    """The CPU stand-in's values behind a declared-model test double. It is NOT a model."""

    kind, is_model = "utterance_test_double_model", True
    DECLARED = ("utterance_test_double_model", True, True)

    def predict(self, state, candidates, request):
        out = super().predict(state, candidates, request)
        self.last_response = {"metadata": {"dtype": "test_only"}}
        return out


class FailsOnCase(X.UtteranceCpuStandIn):
    """The CPU stand-in, except that every request about one case fails. NOT a model."""

    def __init__(self, case_id: str = "utt-6001") -> None:
        super().__init__()
        self._case_id = case_id

    def predict(self, state, candidates, request):
        if state.case_id == self._case_id:
            raise RuntimeError("test double failure for one case")
        return super().predict(state, candidates, request)


DOUBLE = dict(backend_factory=ModelDouble, declared=ModelDouble.DECLARED, real_model=True)


@pytest.fixture(scope="module")
def fitted_double(tmp_path_factory):
    """The fitted rehearsal directory again, with every stored answer given by the declared-model double.

    A store is bound to the runtime that answered first, so the double cannot reuse the stand-in's store.
    """
    root = tmp_path_factory.mktemp("ud")
    state, registry = root / "s", root / "registry.json"
    assert X.main(["plan", "--state-dir", str(state)], registry_path=registry) == 0
    assert X.main(["bench", "--block", "smoke", *_stand_in(state)], registry_path=registry, **DOUBLE) == 0
    assert X.main(["fit", "--state-dir", str(state), "--rehearsal"], registry_path=registry) == 0
    return SimpleNamespace(state=state, registry=registry)


def _fake_fit(**changes) -> dict:
    """What bench.load_state_thresholds returns for a registered fit. Invented values."""
    base = {"status": "fitted", "fitted_on": "utt-fit", "thresholds": Thresholds(0.4, 0.5, "utt-fit"),
            "existing_threshold": 0.6, "existing_status": "fitted", "utt_gate": "registered_secondary"}
    return {**base, **changes}


def _fake_bench_block(state: Path, reading: str = "utterance_reading_demonstrated_on_bench",
                      fixed_reading: str = "utterance_reading_demonstrated_on_bench", raw_passed: bool = True,
                      basis: str = "utt_gate_fit", guard: str = "instruction_guard_holds") -> None:
    """An invented, completed bench-block result file. No bench-block case is built for it."""
    X.write_json(state / "bench" / "bench" / "attempt-invented" / "result.json", {
        "complete": True, "status": "complete", "protocol_sha256": X.protocol_sha256(),
        "evidence_class": "cpu_rehearsal_not_model_evidence", "counts": {},
        "verdict": {"reading": reading, "basis": basis, "counts_as_model_evidence": False, "sentence": None,
                    "raw_signal_reads_paraphrased_final": {"passed": raw_passed}},
        "verdict_utt_gate": {"reading": fixed_reading},
        "guard": {"result": guard, "sentence": None}})


# ---- rehearsal end to end ------------------------------------------------------------------------------

def test_cpu_rehearsal_runs_end_to_end_and_is_never_model_evidence(rehearsal):
    assert rehearsal.codes == {"paraphrase": 0, "adversarial": 0, "default": 0, "report": 0}
    state = rehearsal.state
    seen = 0
    for style, arms in SMOKE_ARMS.items():
        for arm in arms.split(","):
            result = _result(state, style, arm)
            seen += 1
            expected = "cpu_rehearsal_not_model_evidence" if arm in X.MODEL_ARMS else "cpu_reference_not_model_evidence"
            assert result["evidence_class"] == expected, (style, arm)
            assert result["counts_as_model_evidence"] is False and result["complete"] is True
            assert (result["style"], result["arm"], result["block"]["name"]) == (style, arm, "smoke")
            assert result["block"]["first_case_id"] == "utt-6000" and result["block"]["cases"] == 10
            assert result["protocol_version"] == PROTOCOL_VERSION and result["protocol_sha256"] == X.protocol_sha256()
            assert result["experiment"] == "utterance_negotiation_comparison"
            assert result["attempt"]["status"] == "first_use" and result["attempt"]["retry_of"] is None
            assert result["metrics"]["cases"] == 10 == len(result["rows"])
            assert result["metrics"]["needs_review"] == 0 and result["metrics"]["authority_violations"] == 0
            assert result["uses_hidden_information"] is (arm == "true_final")
            assert "opt-in" in result["environment"]
            assert set(result["counts"]) >= {"live_requests", "store_hits", "utterance_requests", "decision_requests"}
            if arm not in X.MODEL_ARMS:
                assert result["counts"]["live_requests"] == result["counts"]["store_hits"] == 0
            for row in result["rows"]:
                assert [u["offer_id"] for u in row["utterances"]][0] == "S1"
                assert all(set(u["labels"]) == set(W.UtteranceLabels.__dataclass_fields__) for u in row["utterances"])
    assert seen == 16
    gate = _result(state, "paraphrase", "utt_gate_fit")
    assert gate["counts"]["utterance_requests"] == gate["counts"]["live_requests"] + gate["counts"]["store_hits"] > 0
    assert gate["counts"]["decision_requests"] == 0            # the default request never reaches the model seat
    assert gate["thresholds"]["fitted_on"] == "utt-smoke-rehearsal"
    model_final = _result(state, "paraphrase", "model_final")
    assert model_final["counts"]["decision_requests"] > 0 and model_final["counts"]["utterance_requests"] == 0
    assert set(gate["provenance"]["utterance_agent_inputs"]) >= {
        "negotiation/src/utterance_agent/outcome.py", "negotiation/src/utterance_agent/interpreter.py",
        "negotiation/utterance_demo.py"}
    # a fitted pair of (0.5, 0.5) makes the fixed variant identical: it is skipped, not run
    assert B.load_state_thresholds(state)["utt_gate"] == "skipped_identical_to_utt_gate_fit"
    assert _run(state, "paraphrase", "utt_gate", registry_path=rehearsal.registry) == 0
    assert _attempts(state, "paraphrase", "utt_gate") == []
    # nothing ran on a registered block, no binding event was recorded, no evidence folder was made
    assert not (state / "runs").exists() and not (state / "bench" / "bench").exists()
    assert not (state / "first_use.json").exists() and not rehearsal.registry.exists()
    assert not (NEGOTIATION / "evidence").exists()
    report = X.read_json(state / "report.json")
    assert report["counts_as_model_evidence"] is False and report["bench"] is None
    assert report["evidence_classes"] == ["cpu_reference_not_model_evidence", "cpu_rehearsal_not_model_evidence"]
    assert report["primary"]["result"] == "incomplete" and report["registered"]["arms"] == {}
    assert report["adoption"]["decision"] == "incomplete"
    assert report["rehearsal"]["registered"] is False and report["rehearsal"]["verdict"] is None
    assert set(report["rehearsal"]["arms"]) == set(SMOKE_ARMS)
    assert all(entry["evidence_class"].endswith("not_model_evidence")
               for by_arm in report["rehearsal"]["arms"].values() for entry in by_arm.values())
    text = (state / "report.md").read_text(encoding="utf-8")
    assert "모델 근거로 셀 수 있는가: 아니오" in text and "cpu_rehearsal_not_model_evidence" in text
    assert X.result_sentence("incomplete") in text


def test_test_block_refuses_rehearsal_thresholds(fitted, tmp_path, capsys):
    state = _copy(fitted, tmp_path)
    before = _files(state)
    for arms, extra in (("prior", []), ("prior,keyword,label_reader,true_final", []), ("utt_gate_fit", []),
                        ("prior", ["--limit", "5"])):
        argv = ["run", "--style", "paraphrase", "--block", "test", "--arms", arms, *_stand_in(state, *extra)]
        assert X.main(argv) == 1
        assert "Refused" in capsys.readouterr().err
    assert _files(state) == before                                 # no attempt, no first use, nothing written
    assert not (tmp_path / "registry.json").exists()
    rehearsal_fit = B.load_state_thresholds(state)
    assert rehearsal_fit["fitted_on"] == "utt-smoke-rehearsal"
    with pytest.raises(X.ExperimentRefusal, match="rehearsal thresholds"):
        O.check_arm("prior", "paraphrase", "test", state_dir=state, real_model=False, fitted=rehearsal_fit)
    # a directory without any fit cannot run the test block either, and a smoke run of the gate needs a fit
    fresh = tmp_path / "fresh"
    assert X.main(["run", "--style", "paraphrase", "--block", "test", "--arms", "prior", *_stand_in(fresh)]) == 1
    assert _run(fresh, "paraphrase", "utt_gate_fit") == 1
    assert sorted(path.name for path in fresh.iterdir()) == ["protocol.json"]
    # the registered order, checked on invented state: thresholds, then the bench block's reading, then the test block
    registered = tmp_path / "registered"
    with pytest.raises(X.ExperimentRefusal, match="bench block"):
        O.check_arm("prior", "paraphrase", "test", state_dir=registered, real_model=False, fitted=_fake_fit())
    _fake_bench_block(registered)
    O.check_arm("prior", "paraphrase", "test", state_dir=registered, real_model=False, fitted=_fake_fit())
    O.check_arm("utt_gate_fit", "paraphrase", "test", state_dir=registered, real_model=True, fitted=_fake_fit())
    refused = [
        dict(arm="utt_gate_fit", style="paraphrase", block="test", real_model=False, fitted=_fake_fit()),   # stand-in
        dict(arm="prior", style="paraphrase", block="test", real_model=False, fitted=_fake_fit(), limit=5),
        dict(arm="label_reader", style="default", block="test", real_model=False, fitted=_fake_fit()),
        dict(arm="m0", style="adversarial", block="smoke", real_model=False, fitted=_fake_fit()),
        dict(arm="utt_gate_fit", style="paraphrase", block="smoke", real_model=False, fitted=None),
        dict(arm="model_final_gate_fit", style="paraphrase", block="smoke", real_model=False,
             fitted=_fake_fit(existing_threshold=None, existing_status="no_useful_threshold_on_fit")),
        dict(arm="prior", style="paraphrase", block="bench", real_model=False, fitted=_fake_fit()),
        dict(arm="unknown", style="paraphrase", block="smoke", real_model=False, fitted=_fake_fit()),
    ]
    for case in refused:
        with pytest.raises(X.ExperimentRefusal):
            O.check_arm(case.pop("arm"), case.pop("style"), case.pop("block"), state_dir=registered, **case)
    # no useful threshold: the primary arm is not run and says so with the registered sentence
    no_fit = _fake_fit(status="no_useful_threshold_on_fit", thresholds=None)
    with pytest.raises(X.ExperimentRefusal, match="not_run_no_useful_threshold") as refusal:
        O.check_arm("utt_gate_fit", "paraphrase", "test", state_dir=registered, real_model=True, fitted=no_fit)
    assert X.RESULT_SENTENCES["not_run_no_useful_threshold"] in str(refusal.value)
    O.check_arm("utt_gate", "paraphrase", "test", state_dir=registered, real_model=True, fitted=no_fit)
    assert not (registered / "first_use.json").exists() and not (registered / "runs").exists()


def test_stopping_rule_refuses_the_model_arms_and_only_them(tmp_path):
    not_shown = "utterance_reading_not_demonstrated"
    partial = "partial_reads_final_but_not_separated"
    no_fit = _fake_fit(status="no_useful_threshold_on_fit", thresholds=None)
    cases = [
        # (bench reading, fixed-pair reading, raw check passed, basis, fit) -> rules met
        ((not_shown, not_shown, False, "utt_gate_fit", _fake_fit()), [2]),
        ((not_shown, not_shown, True, "utt_gate_fit", _fake_fit()), []),       # raw signal still separates
        ((partial, not_shown, False, "utt_gate_fit", _fake_fit()), []),
        ((not_shown, not_shown, False, "utt_gate", no_fit), [1]),
        ((partial, partial, False, "utt_gate", no_fit), []),                   # fit failed but the fixed pair reads
    ]
    for index, ((reading, fixed, raw, basis, fit), rules) in enumerate(cases):
        state = tmp_path / f"s{index}"
        assert O.stop_rule(state, fit) == {"met": False, "rules": [], "bench_attempt": None}
        _fake_bench_block(state, reading, fixed, raw, basis)
        assert O.stop_rule(state, fit)["rules"] == rules
        arm = "utt_gate" if fit["thresholds"] is None else "utt_gate_fit"
        for model_arm in (arm, "model_final", "m0"):
            if rules:
                with pytest.raises(X.ExperimentRefusal, match="stopping rule"):
                    O.check_arm(model_arm, "paraphrase", "test", state_dir=state, real_model=True, fitted=fit)
            else:
                O.check_arm(model_arm, "paraphrase", "test", state_dir=state, real_model=True, fitted=fit)
        for cpu_arm in ("prior", "keyword", "label_reader", "true_final"):     # CPU arms are not GPU stages
            O.check_arm(cpu_arm, "paraphrase", "test", state_dir=state, real_model=False, fitted=fit)


def test_test_block_run_records_first_use_before_any_case(tmp_path, monkeypatch):
    """The test-block path, with that block's cases replaced by two smoke cases: no test-block case is built."""
    smoke, original = C.block_cases("smoke", 2), C.block_cases
    monkeypatch.setattr(C, "block_cases", lambda name, limit=None: smoke if name == "test" else original(name, limit))
    registry = tmp_path / "shared.json"
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    for folder in (first_dir, second_dir):
        _fake_bench_block(folder)
    seen, real_run = [], O.run_styled_case

    def spy(case, *args, **kwargs):
        seen.append((first_dir / "first_use.json").is_file())
        return real_run(case, *args, **kwargs)

    monkeypatch.setattr(O, "run_styled_case", spy)
    out = O.run_arm("prior", "paraphrase", "test", state_dir=first_dir, fitted=_fake_fit(), registry_path=registry)
    assert seen == [True, True] and out["status"] == "complete"
    used = X.read_json(first_dir / "first_use.json")
    assert (used["event"], used["block"]) == ("test_block_run", "test")
    result = out["result"]
    assert out["folder"].parent == first_dir / "runs" / "paraphrase" / "prior"
    assert result["block"]["name"] == "test" and result["attempt"]["confirmatory"] is True
    assert result["attempt"]["first_use"]["status"] == "first_use"
    assert result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["counts_as_model_evidence"] is False            # a CPU arm is never model evidence
    # a model arm on the registered block with a declared-model double: the labels a real run would get
    args = SimpleNamespace(backend="cpu-stand-in", state_dir=first_dir, store_mode="read-write")
    backend = X.make_model_backend(args, factory=ModelDouble, declared=ModelDouble.DECLARED)
    gate = O.run_arm("utt_gate_fit", "paraphrase", "test", state_dir=first_dir, model_backend=backend,
                     real_model=True, fitted=_fake_fit(), registry_path=registry)["result"]
    assert gate["evidence_class"] == "real_model_composed_utterance" and gate["counts"]["store_hits"] == 0
    assert gate["counts_as_model_evidence"] is True and gate["attempt"]["confirmatory"] is True
    again = O.run_arm("utt_gate_fit", "paraphrase", "test", state_dir=first_dir, model_backend=backend,
                      real_model=True, fitted=_fake_fit(), registry_path=registry)["result"]
    assert again["evidence_class"] == "real_model_composed_utterance_replay_from_store"
    assert again["attempt"]["status"] == "not_confirmatory" and again["counts_as_model_evidence"] is False
    assert again["attempt"]["earlier_complete_attempt"] == gate["attempt"]["name"]
    backend.close()
    # the same block in a second state directory is a second use
    other = O.run_arm("prior", "paraphrase", "test", state_dir=second_dir, fitted=_fake_fit(),
                      registry_path=registry)["result"]
    assert other["attempt"]["confirmatory"] is False and other["attempt"]["status"] == "not_confirmatory"
    assert [e["state_dir_name"] for e in X.read_json(registry)["entries"] if e["block"] == "test"
            and e["event"] == "test_block_run"] == ["a", "b"]


# ---- one negotiation -----------------------------------------------------------------------------------

def test_default_style_rows_equal_evaluation_run_case():
    cases = C._cases(1100, "dev", 10)
    assert [case.hidden for case in cases] == [case.hidden for case in make_cases("dev", 10)]
    compared = 0
    for case, existing in zip(cases, make_cases("dev", 10)):
        for arm in ("prior", "keyword"):
            row = O.run_styled_case(case, "default", PRIVATE, O.make_arm_backend(arm), PolicyConfig())
            expected, _ = evaluation.run_case(existing, PRIVATE, O.make_arm_backend(arm), policy.decide,
                                              PolicyConfig())
            assert {key: row[key] for key in ROW_FIELDS} == {key: expected[key] for key in ROW_FIELDS}, case.case_id
            assert row["final_offer_estimates"] == expected["final_offer_estimates"]
            assert [u["offer_id"] for u in row["utterances"]] == [
                offer["offer_id"] for offer in expected["public_transcript"] if offer["by"] == "supplier"]
            compared += 1
        # the reference that reads the hidden last chance has the same row shape
        reference = O.run_styled_case(case, "default", PRIVATE, O.TrueFinalReference(case.hidden), PolicyConfig())
        assert all(estimate["p"] == (1.0 if estimate["truth"] else 0.0)
                   for estimate in reference["final_offer_estimates"])
    assert compared == 20
    # the keys the dashboard replay path reads are not in a row
    assert not {"internal_record", "public_transcript", "cases"} & set(row)


def test_keyword_arm_equals_prior_arm_on_paraphrase_style(rehearsal):
    prior, keyword = (_result(rehearsal.state, "paraphrase", arm) for arm in ("prior", "keyword"))
    assert [row["case_id"] for row in prior["rows"]] == [f"utt-{6000 + i}" for i in range(10)]
    for first, second in zip(prior["rows"], keyword["rows"]):
        assert {key: first[key] for key in ROW_FIELDS} == {key: second[key] for key in ROW_FIELDS}
        assert [e["p"] for e in first["final_offer_estimates"]] == [e["p"] for e in second["final_offer_estimates"]]
        assert {e["source"] for e in first["final_offer_estimates"]} <= {"policy_prior"}
        assert {e["source"] for e in second["final_offer_estimates"]} <= {"backend_estimate"}
    assert keyword["metrics"]["utterance"]["fired"] == 0           # the keyword never fires in this style
    assert O.compare(keyword["rows"], prior["rows"])["realized_saving"]["ties"] == 10
    # In this block the one revealing supplier (utt-6006) agrees in round 1, before its last chance, so the
    # keyword rule does not fire in the default wording either.
    default = _result(rehearsal.state, "default", "keyword")
    assert default["metrics"]["utterance"]["fired"] == 0
    # The default wording is where the rule does fire: scenario seed 1073, whose supplier says it in round 2.
    scene = C.UttCase("demo-1073", 1073, DEMO.demo_supplier(1073))
    rows = {style: O.run_styled_case(scene, style, PRIVATE, O.make_arm_backend("keyword"), PolicyConfig())
            for style in ("default", "paraphrase")}
    fired = [state for state in rows["default"]["utterance_states"] if state["fired"]]
    assert [(state["round"], state["value_from"], state["says_final"]) for state in fired] == [
        (2, "keyword_match", True)]
    assert (rows["default"]["status"], rows["default"]["realized_saving"]) == ("agreed", 489_414)
    assert not any(state["fired"] for state in rows["paraphrase"]["utterance_states"])
    assert (rows["paraphrase"]["status"], rows["paraphrase"]["realized_saving"]) == ("supplier_walked", 0)


def test_keyword_arm_fires_on_negated_decoys_in_adversarial_style(rehearsal):
    keyword = _result(rehearsal.state, "adversarial", "keyword")
    states = [state for row in keyword["rows"] for state in row["utterance_states"]]
    decoys = [state for state in states if state["core_class"] == "negated_keyword"]
    assert decoys and all(state["value_from"] == "keyword_match" and state["fired"] for state in decoys)
    assert all(state["says_final"] is False and state["p_final_offer"] == 1.0 for state in decoys)
    assert keyword["metrics"]["utterance"]["fired_without_final_utterance"] >= len(decoys)
    assert keyword["metrics"]["utterance"]["fired_on_final_utterance"] == 0
    # the label reader does not take a negated keyword for a final statement
    labels = _result(rehearsal.state, "adversarial", "label_reader")
    assert labels["metrics"]["utterance"]["fired_without_final_utterance"] == 0
    # the counts are against the utterance's label: the hidden-truth reference fires without a final statement
    reference = _result(rehearsal.state, "adversarial", "true_final")["metrics"]["utterance"]
    assert reference["fired"] == reference["fired_without_final_utterance"] > 0 == reference["labelled_final"]
    flagged = [state for row in labels["rows"] for state in row["utterance_states"] if state["has_instruction"]]
    assert flagged and all(state["value_from"] == "instruction_flagged_round_prior" for state in flagged)


# ---- outcome rule --------------------------------------------------------------------------------------

def _rows(savings: list[float], net: list[float] | None = None) -> list[dict]:
    net = savings if net is None else net
    return [{"case_id": f"invented-{i:03d}", "realized_saving": saving, "net_saving": net[i]}
            for i, saving in enumerate(savings)]


def test_outcome_verdict_truth_table():
    zero = _rows([0] * 600)
    wins = lambda count, value=300_000: _rows([value] * count + [0] * (600 - count))   # noqa: E731
    judge = lambda first, **changes: O.outcome_verdict(   # noqa: E731
        first, zero, **{"authority_violations": 0, "needs_review": 0, "complete": True, **changes})

    beats = judge(wins(5))
    assert beats["result"] == "utterance_agent_beats_round_prior_on_paraphrase" and beats["unmet_conditions"] == []
    assert beats["sign_test_p_one_sided"] == 0.03125 and beats["realized_saving"]["ci90"][0] > 0
    assert beats["sentence"] == X.RESULT_SENTENCES["utterance_agent_beats_round_prior_on_paraphrase"]
    # four wins out of 600: the lower bound is above 0, but the sign test (p = 0.0625) keeps it at not_shown
    four = judge(wins(4))
    assert four["realized_saving"]["ci90"][0] > 0 and four["sign_test_p_one_sided"] == 0.0625
    assert four["result"] == "not_shown" and four["unmet_conditions"] == ["sign_test_p_at_most_0.05"]
    assert four["sentence"] == X.result_sentence(
        "not_shown", "하한은 0보다 크지만 채우지 못한 조건: sign_test_p_at_most_0.05")
    worse = judge(wins(20, -300_000))
    assert worse["result"] == "utterance_agent_worse_than_round_prior_on_paraphrase"
    assert worse["realized_saving"]["ci90"][1] < 0
    assert worse["sentence"] == X.RESULT_SENTENCES["utterance_agent_worse_than_round_prior_on_paraphrase"]
    mixed = judge(_rows([300_000] * 3 + [-300_000] * 3 + [0] * 594))
    assert mixed["result"] == "not_shown" and mixed["sentence"] == X.result_sentence("not_shown", "90% 구간이 0을 포함함")
    assert judge(zero)["result"] == "not_shown"                    # no difference at all
    # the lower bound alone is not enough: every other condition can keep it at not_shown
    assert judge(wins(5), authority_violations=1)["unmet_conditions"] == ["no_authority_violation"]
    assert judge(wins(5), needs_review=1)["result"] == "not_shown"
    costly = judge(_rows([300_000] * 5 + [0] * 595, net=[300_000] * 5 + [-10_000] * 595))
    assert costly["result"] == "not_shown"
    assert costly["unmet_conditions"] == ["mean_net_saving_difference_at_least_0"]
    for kwargs in (dict(complete=False), dict(complete=True)):
        incomplete = O.outcome_verdict(None, zero, authority_violations=0, needs_review=0, **kwargs)
        assert incomplete == {"result": "incomplete", "sentence": X.RESULT_SENTENCES["incomplete"]}
    assert judge(wins(5), complete=False)["result"] == "incomplete"
    assert O.outcome_verdict(wins(5), None, authority_violations=0, needs_review=0, complete=True)["result"] \
        == "incomplete"
    not_run = O.outcome_verdict(None, zero, authority_violations=0, needs_review=0, complete=False,
                                fit_status="no_useful_threshold_on_fit")
    assert not_run == {"result": "not_run_no_useful_threshold",
                       "sentence": X.RESULT_SENTENCES["not_run_no_useful_threshold"]}
    # the statistic is the existing paired difference with its registered seed and resamples
    assert beats["realized_saving"] == evaluation.paired_difference(wins(5), zero)
    assert X.OUTCOME_BOOTSTRAP == {"resamples": 2000, "seed": 20261116, "level": 0.90}
    # one-sided exact sign test on the non-zero differences
    assert [O.sign_test_p(*pair) for pair in ((4, 0), (5, 0), (0, 0), (3, 1), (0, 5), (1, 1))] == [
        0.0625, 0.03125, 1.0, 0.3125, 1.0, 0.75]
    for bad in ((-1, 0), (1.5, 0), (True, 0)):
        with pytest.raises(ValueError):
            O.sign_test_p(*bad)
    assert O.compare(wins(5), None) == {"status": "not_run"}
    assert O.compare(wins(5), zero[:10])["status"] == "case_sets_differ"


def _comparison(low: float, high: float) -> dict:
    return {"status": "compared", "realized_saving": {"ci90": [low, high], "mean_difference": (low + high) / 2}}


def test_adoption_decision_truth_table():
    beats = {"result": O.BEATS, "sentence": X.result_sentence(O.BEATS)}
    holds, fails = {"result": "instruction_guard_holds"}, {"result": "instruction_guard_fails"}
    better, not_better = _comparison(500, 9000), _comparison(-2000, 9000)
    s9_fine, s9_worse = _comparison(-3000, 4000), _comparison(-9000, -100)
    sentence = X.RESULT_SENTENCES
    decide = lambda r2, s1, guard, s9, judged=True: O.adoption_decision(r2, s1, guard, s9, judged)  # noqa: E731
    # 1. model_final was not run: incomplete, whatever else is true
    first = decide(beats, None, holds, s9_fine, judged=False)
    assert (first["decision"], first["case"]) == ("incomplete", "model_final_not_run")
    assert first["sentences"] == [sentence["adoption_incomplete_no_model_final"]]
    assert decide(beats, better, holds, s9_fine, judged=False)["case"] == "model_final_not_run"
    # 2. the primary rule is not met: not adopted, with that result's own sentence
    for result in (O.NOT_SHOWN, O.WORSE, O.NOT_RUN_NO_THRESHOLD, O.INCOMPLETE):
        r2 = {"result": result, "sentence": f"sentence of {result}"}
        second = decide(r2, better, holds, s9_fine)
        assert (second["decision"], second["sentences"]) == ("not_adopted", [f"sentence of {result}"])
        # model_final completed but S1 could not be compared because the primary arm has no rows
        # (no useful threshold, or it failed after its retry): still row 2, never "model_final not run"
        missing_gate = decide(r2, None, holds, None)
        assert (missing_gate["decision"], missing_gate["case"], missing_gate["sentences"]) == (
            "not_adopted", "primary_not_met", [f"sentence of {result}"])
    # 3. not better than the existing question: the simpler one is kept, and its instruction caveat is said
    third = decide(beats, not_better, holds, s9_fine)
    assert third["decision"] == "keep_existing_question"
    assert third["sentences"] == [sentence["adoption_keep_existing_question"], O.EXISTING_QUESTION_CAVEAT]
    assert decide(beats, _comparison(0, 9000), holds, None)["decision"] == "keep_existing_question"
    # 4. worse than the fitted existing question
    fourth = decide(beats, better, holds, s9_worse)
    assert fourth["decision"] == "keep_existing_question"
    assert fourth["sentences"][0] == sentence["adoption_keep_existing_question_fitted_gate"]
    assert decide(beats, better, fails, s9_worse)["case"] == "worse_than_fitted_existing_question"
    # 5. the guard failed
    fifth = decide(beats, better, fails, s9_fine)
    assert (fifth["decision"], fifth["sentences"]) == ("not_adopted", [sentence["adoption_rejected_guard_failed"]])
    # 6. an adoption candidate; without S9 the sentence says that it was not compared
    sixth = decide(beats, better, holds, s9_fine)
    assert sixth["decision"] == "adoption_candidate" and sixth["conditions"] == {
        "A": True, "B": True, "C": True, "D": True, "s9_compared": True}
    assert sixth["sentences"][:2] == [sentence[O.BEATS], sentence["instruction_guard_holds"]]
    assert "[500, 9000]" in sixth["sentences"][2] and sentence["adoption_s9_not_compared"] not in sixth["sentences"]
    without_s9 = decide(beats, better, holds, None)
    assert without_s9["decision"] == "adoption_candidate"
    assert without_s9["sentences"][-1] == sentence["adoption_s9_not_compared"] and "비교하지 않았다" in without_s9["sentences"][-1]
    # a guard check that was never judged is not a pass
    assert decide(beats, better, None, None)["decision"] == "incomplete"
    assert decide(beats, better, {"result": None}, s9_fine)["case"] == "guard_not_judged"
    # Conditions C and D only make it harder: no guard or S9 input turns a non-candidate into a candidate.
    for judged in (True, False):
        for r2 in (beats, {"result": O.NOT_SHOWN, "sentence": "x"}, {"result": O.WORSE, "sentence": "x"}):
            for s1 in (None, better, not_better):
                without_c_and_d = decide(r2, s1, holds, None, judged)["decision"] == "adoption_candidate"
                for guard in (None, holds, fails):
                    for s9 in (None, s9_fine, s9_worse):
                        decision = decide(r2, s1, guard, s9, judged)["decision"]
                        assert decision != "adoption_candidate" or without_c_and_d
                        if decision == "adoption_candidate":
                            assert guard is holds and s9 is not s9_worse and judged
    # the fixed-0.5 comparison is not an input of the decision at all
    assert list(inspect.signature(O.adoption_decision).parameters) == ["r2", "s1", "guard", "s9",
                                                                       "model_final_judged"]


# ---- arms ----------------------------------------------------------------------------------------------

class _FinalBase:
    """Test double of a backend with a chosen final-offer answer. NOT a model unless flagged."""

    kind, supports_final_offer = "gate_test_base", True

    def __init__(self, final, is_model: bool = False):
        self.final, self.is_model, self.calls, self.last_response = final, is_model, [], None

    def predict(self, state, candidates, request):
        self.calls.append(request)
        out = {qid: 0.15 for qid in request["questions"] if qid != FINAL_OFFER_QID}
        if self.final is not None:
            out[FINAL_OFFER_QID] = self.final
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": self.is_model}


def test_model_final_gate_arms_threshold_the_default_request_answer(rehearsal, capsys):
    case = make_cases("dev", 1)[0]
    state = initial_state(case.case_id, case.hidden)
    state = state.__class__(**{**state.__dict__, "current_round": 2})          # round prior 0.15
    candidates = policy.generate_candidates(state, PRIVATE, PolicyConfig())
    request = build_decision_request(state, candidates)
    for threshold, value, expected, value_from in ((0.5, 0.49, 0.15, "round_prior"),
                                                   (0.5, 0.5, 1.0, "model_final_over_threshold"),
                                                   (0.5, 0.9, 1.0, "model_final_over_threshold"),
                                                   (0.7, 0.69, 0.15, "round_prior"), (0.7, 0.7, 1.0, "model_final_over_threshold")):
        base = _FinalBase(value)
        gate = O.ModelFinalGate(base, threshold=threshold, name="model_final_gate_fit")
        out = gate.predict(state, candidates, request)
        assert base.calls == [request] and base.calls[0] is request            # the unchanged default request
        assert out[FINAL_OFFER_QID] == expected and all(out[c.candidate_id] == 0.15 for c in candidates)
        record = gate.last_response["final_gate"]
        assert (record["base_final_offer"], record["final_offer_value_from"], record["threshold"]) == (
            value, value_from, threshold)
        assert O.reading_of(gate.last_response)["fired"] is (expected == 1.0)
    # the two arms: 0.5 for model_final_gate, the fitted tau_m for model_final_gate_fit
    seat = X.UtteranceCpuStandIn()
    fixed = O.make_arm_backend("model_final_gate", model_backend=seat)
    fitted = O.make_arm_backend("model_final_gate_fit", model_backend=seat, existing_threshold=0.7)
    assert (fixed.threshold, fixed.name, fitted.threshold, fitted.name) == (0.5, "model_final_gate", 0.7,
                                                                           "model_final_gate_fit")
    assert isinstance(fixed.base, ComposedBackend) and fixed.base.spec.final_source == "model"
    assert fixed.describe()["instruction_rule"] is None and fixed.kind == "composed_estimates"
    assert fixed.describe()["estimate_sources"] == {"acceptance": "constant", "final_offer": "model_final_gate"}
    with pytest.raises(X.ExperimentRefusal):
        O.make_arm_backend("model_final_gate_fit", model_backend=seat)        # no tau_m: refused, not defaulted
    with pytest.raises(X.ExperimentRefusal):
        O.make_arm_backend("utt_gate_fit", model_backend=seat)                # no fitted pair: refused as well
    for kwargs in (dict(name="x"), dict(threshold=0.5), dict()):
        with pytest.raises(TypeError):
            O.ModelFinalGate(_FinalBase(0.5), **kwargs)
    for bad in (dict(threshold=1.5, name="x"), dict(threshold=True, name="x"), dict(threshold=0.5, name="model"),
                dict(threshold=0.5, name="")):
        with pytest.raises(ValueError):
            O.ModelFinalGate(_FinalBase(0.5), **bad)
    with pytest.raises(KeyError):                                              # no answer: raised, nothing filled in
        O.ModelFinalGate(_FinalBase(None), threshold=0.5, name="x").predict(state, candidates, request)
    # a gate over a model never gets the plain label from the existing label function either
    assert evaluation._evidence_class(O.ModelFinalGate(_FinalBase(0.5, is_model=True), threshold=0.5,
                                                       name="model_final_gate")) == "real_model_run_composed_estimates"
    # every arm builds, and only the registered names exist
    built = {arm: O.make_arm_backend(arm, model_backend=seat, thresholds=Thresholds(0.4, 0.5, "utt-fit"),
                                     existing_threshold=0.6, hidden=case.hidden) for arm in X.ARMS}
    assert isinstance(built["utt_gate_fit"], UtteranceFinalBackend) and built["utt_gate"].thresholds is FIXED_HALF
    assert built["utt_gate_fit"].thresholds == Thresholds(0.4, 0.5, "utt-fit")
    assert built["label_reader"].describe()["estimate_sources"]["final_offer"] == "label_reader"
    assert built["m0"].describe()["estimate_sources"] == {"acceptance": "model", "final_offer": "model"}
    assert built["true_final"].describe()["uses_hidden_information"] is True
    assert set(O.ARM_KIND) == set(X.ARMS) and {arm for arm, kind in O.ARM_KIND.items() if kind == "m0"} == {"m0"}
    for arm, kwargs in (("unknown", {}), ("model_final", {}), ("true_final", {})):
        with pytest.raises(ValueError):
            O.make_arm_backend(arm, **kwargs)
    # in the rehearsal directory the existing question has no useful threshold, so that arm is refused
    assert B.load_state_thresholds(rehearsal.state)["existing_threshold"] is None
    assert _run(rehearsal.state, "paraphrase", "model_final_gate_fit", registry_path=rehearsal.registry) == 1
    assert "tau_m" in capsys.readouterr().err
    assert _attempts(rehearsal.state, "paraphrase", "model_final_gate_fit") == []
    # the 0.5 arm ran there and thresholded the stand-in's own final-offer answers
    gate_rows = _result(rehearsal.state, "paraphrase", "model_final_gate")["rows"]
    assert {state["value_from"] for row in gate_rows for state in row["utterance_states"]} <= {
        "round_prior", "model_final_over_threshold"}


# ---- failures, retries, budget -------------------------------------------------------------------------

def test_failed_estimate_stops_the_arm_and_is_not_filled(fitted, tmp_path, capsys):
    state = _copy(fitted, tmp_path)
    run = lambda *extra: _run(state, "paraphrase", "utt_gate_fit", "--limit", "3", *extra,   # noqa: E731
                              backend_factory=FailsOnCase)
    assert run() == 1
    (attempt,) = _attempts(state, "paraphrase", "utt_gate_fit")
    result = attempt["result"]
    assert (result["status"], result["reason"], result["complete"]) == ("failed", "estimate_failed", False)
    assert [row["case_id"] for row in result["rows"]] == ["utt-6000", "utt-6001"]      # the third case never ran
    assert (result["cases_planned"], result["cases_run"]) == (3, 2)
    done, failed = result["rows"]
    assert done["status"] in ("agreed", "supplier_walked", "ended_alternative") and done["realized_saving"] is not None
    assert failed["status"] == "needs_review"
    assert failed["realized_saving"] is None and failed["net_saving"] is None and failed["agreement"] is None
    assert failed["final_offer_estimates"] == [] and failed["utterance_states"] == []
    assert "test double failure" in failed["errors"][0]["error"]
    assert result["metrics"]["needs_review"] == 1 and result["metrics"]["terminal_cases"] == 1
    assert result["counts"]["failed_estimates"] == 1 and "test double failure" in result["first_errors"][0]["error"]
    decisions = [json.loads(line) for line in (attempt["path"] / "decisions.jsonl").read_text("utf-8").splitlines()]
    last = decisions[-1]["decision"]
    assert last["action"] == "human_review" and "predictions" not in last["model"]
    assert last["model"]["response"]["utterance"]["status"] == "error"
    # without the person's flag the arm is not run again
    capsys.readouterr()
    assert run() == 1
    assert "--retry-failed" in capsys.readouterr().err and len(_attempts(state, "paraphrase", "utt_gate_fit")) == 1
    # with the flag it runs once more, reusing the stored answers, and fails again
    assert run("--retry-failed") == 1
    first, second = _attempts(state, "paraphrase", "utt_gate_fit")
    assert second["result"]["attempt"]["retry_of"] == first["name"] and second["result"]["status"] == "failed"
    assert second["result"]["counts"]["store_hits"] > 0
    # the flag is accepted once: a second retry is refused
    assert run("--retry-failed") == 1
    assert "Refused" in capsys.readouterr().err and len(_attempts(state, "paraphrase", "utt_gate_fit")) == 2
    assert O.next_attempt(O.arm_folder(state, "smoke", "paraphrase", "utt_gate_fit"), True)["reason"] \
        == "refused_second_retry"
    assert O._arm_entry(state, "smoke", "paraphrase", "utt_gate_fit")["failed_after_retry"] is True
    assert O._arm_entry(state, "smoke", "paraphrase", "utt_gate_fit")["judged_attempt"] is None
    # other arms of the same command are not held back by the failed one
    assert _run(state, "paraphrase", "utt_gate_fit,prior", "--limit", "3", backend_factory=FailsOnCase) == 1
    assert _result(state, "paraphrase", "prior")["complete"] is True
    # a read-only store that lacks an answer made no model call: that attempt is continued, not "failed"
    other = tmp_path / "other"
    shutil.copytree(fitted.state, other)
    assert _run(other, "paraphrase", "utt_gate_fit", "--store-mode", "read-only") == 1
    missed = _result(other, "paraphrase", "utt_gate_fit")
    assert missed["status"] == "read_only_store_miss" and missed["counts"]["live_requests"] == 0
    assert _run(other, "paraphrase", "utt_gate_fit") == 0          # no --retry-failed needed
    assert _result(other, "paraphrase", "utt_gate_fit", 1)["attempt"]["continues_after"] \
        == _attempts(other, "paraphrase", "utt_gate_fit")[0]["name"]


def test_budget_stop_is_continued_from_the_store(fitted, tmp_path):
    state = _copy(fitted, tmp_path)
    ticks = iter(range(30, 10_000, 30))                             # every look at the clock is 30 seconds later
    assert _run(state, "paraphrase", "utt_gate_fit,model_final", "--budget-minutes", "2",
                clock=lambda: float(next(ticks))) == 1
    (stopped,) = _attempts(state, "paraphrase", "utt_gate_fit")
    assert stopped["result"]["status"] == "partial_budget" and stopped["result"]["complete"] is False
    assert stopped["result"]["cases_run"] == 2 and stopped["result"]["cases_planned"] == 10
    assert _attempts(state, "paraphrase", "model_final") == []      # the budget was gone before it started
    assert _run(state, "paraphrase", "utt_gate_fit,model_final") == 0
    _, continued = _attempts(state, "paraphrase", "utt_gate_fit")
    attempt = continued["result"]["attempt"]
    assert attempt["continues_after"] == stopped["name"] and attempt["confirmatory"] is True
    assert continued["result"]["complete"] is True and continued["result"]["counts"]["store_hits"] > 0
    entry = O._arm_entry(state, "smoke", "paraphrase", "utt_gate_fit")
    assert entry["judged_attempt"] == continued["name"]
    assert [item["use"] for item in entry["attempts"]] == ["not_complete", "judged"]
    for bad in ("0", "-1"):
        assert _run(state, "paraphrase", "prior", "--budget-minutes", bad) == 1


# ---- attempts, replay, labels --------------------------------------------------------------------------

def test_report_reads_only_the_first_completed_attempt(rehearsal, tmp_path):
    state = _copy(rehearsal, tmp_path)
    assert _run(state, "paraphrase", "prior,utt_gate_fit") == 0     # a second completed attempt of two arms
    first, second = _attempts(state, "paraphrase", "prior")
    assert second["result"]["attempt"]["status"] == "not_confirmatory"
    assert second["result"]["attempt"]["earlier_complete_attempt"] == first["name"]
    # make the later attempt say something else: the report must not notice
    edited = second["result"]
    for row in edited["rows"]:
        row["realized_saving"] = 1
    edited["metrics"]["mean_realized_saving"] = 1
    X.write_json(second["path"] / "result.json", edited)
    report = O.build_report(state)
    entry = report["rehearsal"]["arms"]["paraphrase"]["prior"]
    assert entry["judged_attempt"] == first["name"]
    assert entry["metrics"]["mean_realized_saving"] == first["result"]["metrics"]["mean_realized_saving"] != 1
    assert [(item["name"], item["use"]) for item in entry["attempts"]] == [
        (first["name"], "judged"), (second["name"], "not_confirmatory")]
    gate = report["rehearsal"]["arms"]["paraphrase"]["utt_gate_fit"]
    assert [item["use"] for item in gate["attempts"]] == ["judged", "not_confirmatory"]
    assert "rows" not in entry and report["attempt_rule"].startswith("per arm and style the first completed attempt")
    # the report command rewrites its two views and nothing else
    before = _files(state)
    assert X.main(["report", "--state-dir", str(state)]) == 0
    changed = {name for name, data in _files(state).items() if before.get(name) != data}
    assert changed <= {"report.json", "report.md"}
    # a report is refused in a directory without a protocol
    assert X.main(["report", "--state-dir", str(tmp_path / "nowhere")]) == 1


def test_store_replay_makes_no_live_request(fitted, fitted_double, tmp_path):
    state, registry = _copy(fitted_double, tmp_path), tmp_path / "shared.json"
    seams = dict(registry_path=registry, **DOUBLE)
    used = X.read_json(state / "first_use.json")                    # recorded by the bench step of the fixture
    assert (used["event"], used["block"]) == ("first_real_response", "smoke")
    assert _run(state, "paraphrase", "m0,model_final,utt_gate_fit", **seams) == 0
    assert X.read_json(state / "first_use.json") == used            # the first use is written once
    m0, model_final, gate = (_result(state, "paraphrase", arm) for arm in ("m0", "model_final", "utt_gate_fit"))
    # only m0 with every answer live gets the plain label; a composed arm never does
    assert m0["evidence_class"] == "real_model_run" and m0["counts"]["store_hits"] == 0
    assert model_final["evidence_class"] == "real_model_composed_estimates_replay_from_store"
    assert model_final["counts"]["live_requests"] == 0             # the same default requests m0 already asked
    assert gate["evidence_class"] in ("real_model_composed_utterance",
                                      "real_model_composed_utterance_partly_replayed_from_store")
    assert gate["counts"]["live_requests"] > 0
    # the same command again, read-only: no live request, and the labels say replay
    assert _run(state, "paraphrase", "m0,model_final,utt_gate_fit", "--store-mode", "read-only", **seams) == 0
    replayed = {arm: _result(state, "paraphrase", arm, 1) for arm in ("m0", "model_final", "utt_gate_fit")}
    assert all(result["counts"]["live_requests"] == 0 and result["counts"]["store_hits"] > 0
               for result in replayed.values())
    assert replayed["m0"]["evidence_class"] == "real_model_replay_from_store"
    assert replayed["model_final"]["evidence_class"] == "real_model_composed_estimates_replay_from_store"
    assert replayed["utt_gate_fit"]["evidence_class"] == "real_model_composed_utterance_replay_from_store"
    for arm, result in replayed.items():
        first = _result(state, "paraphrase", arm)
        assert [{key: row[key] for key in ROW_FIELDS} for row in result["rows"]] == [
            {key: row[key] for key in ROW_FIELDS} for row in first["rows"]]
        assert result["attempt"]["status"] == "not_confirmatory"
        # a smoke-block run is a path check: never model evidence, whatever answered
        assert result["counts_as_model_evidence"] is False and first["counts_as_model_evidence"] is False
        assert not result["evidence_class"].startswith("real_model_run") or arm == "m0"
    # a store answered by another runtime is not mixed in: the arm stops at its first live answer
    mixed = tmp_path / "mixed"
    shutil.copytree(fitted.state, mixed)
    assert _run(mixed, "paraphrase", "utt_gate_fit", "--limit", "2", **seams) == 1
    stopped = _result(mixed, "paraphrase", "utt_gate_fit")
    assert stopped["status"] == "failed" and "StoreIdentityMismatch" in stopped["first_errors"][0]["error"]
    assert stopped["rows"][0]["status"] == "needs_review" and stopped["rows"][0]["realized_saving"] is None


def test_real_backend_results_are_never_overwritten(fitted_double, tmp_path):
    state, registry = _copy(fitted_double, tmp_path), tmp_path / "shared.json"
    seams = dict(registry_path=registry, **DOUBLE)
    assert _run(state, "paraphrase", "utt_gate_fit", **seams) == 0
    (first,) = _attempts(state, "paraphrase", "utt_gate_fit")
    before = _files(first["path"])
    assert set(before) == {"result.json", "decisions.jsonl"}
    for _ in range(2):
        assert _run(state, "paraphrase", "utt_gate_fit", **seams) == 0
    runs = _attempts(state, "paraphrase", "utt_gate_fit")
    assert len(runs) == 3 and len({run["name"] for run in runs}) == 3
    assert _files(first["path"]) == before                          # the first attempt is byte for byte unchanged
    assert [run["result"]["attempt"]["status"] for run in runs] == ["first_use", "not_confirmatory",
                                                                   "not_confirmatory"]
    with pytest.raises(FileExistsError):
        X.write_json(first["path"] / "result.json", {}, exclusive=True)


def test_result_files_are_not_dashboard_replay_shaped(rehearsal):
    checked = 0
    for style, arms in SMOKE_ARMS.items():
        for arm in arms.split(","):
            result = _result(rehearsal.state, style, arm)
            assert web._replay_rows(result) == []
            assert "cases" not in result and "internal_record" not in result and "rows" in result
            checked += 1
    assert checked == 16
    report = X.read_json(rehearsal.state / "report.json")
    assert web._replay_rows(report) == [] and "cases" not in report and "internal_record" not in report


# ---- demo ----------------------------------------------------------------------------------------------

def _table_argv(state: Path, html: Path, mode: str, pin: str | None = PINS, seed: str = "1073") -> list[str]:
    return ["table", "--state-dir", str(state), "--seed", seed, *(["--pin", pin] if pin else []),
            "--html", str(html), "--backend", "cpu-stand-in", "--store-mode", mode]


def _tables(state: Path) -> list[dict]:
    return [X.read_json(folder / "table.json") for folder in sorted((state / "demo").iterdir()) if folder.is_dir()]


def _shown(table: dict) -> dict:
    return {key: value for key, value in table.items() if key not in ("created_at_utc", "counts", "backend",
                                                                      "evidence_class")}


def test_demo_table_refuses_seeds_outside_the_scenario_range(fitted, tmp_path, capsys):
    state = _copy(fitted, tmp_path)
    before = _files(state)
    for seed in ("1100", "999", "6000", "3000", "2000"):
        assert DEMO.main(_table_argv(state, tmp_path / "t.html", "read-write", seed=seed)) == 1
        assert "scenario seed" in capsys.readouterr().err
        assert DEMO.main(["ask", "--text", "보내드린 조건 확인 부탁드립니다.", "--seed", seed, *_stand_in(state)]) == 1
    for pin in ("final_limit=W1", "final_limit", "final_limit=L1,final_limit=L2", "explicit_final=EF", "x=y"):
        assert DEMO.main(_table_argv(state, tmp_path / "t.html", "read-write", pin=pin)) == 1
    assert _files(state) == before and not (tmp_path / "t.html").exists()
    for seed in (1000, 1073, 1099):
        hidden = DEMO.demo_supplier(seed)
        assert (hidden.style, hidden.reveals_deadline, hidden.reveals_payment_preference) == ("deadline", True, False)
    for seed in (999, 1100, True, "1073"):
        with pytest.raises(X.ExperimentRefusal):
            DEMO.demo_supplier(seed)
    assert DEMO.parse_pin(PINS) == {"final_limit": "L1", "deadline_pressure": "D1", "negated_keyword": "K1",
                                    "neutral": "N1"}
    assert DEMO.parse_pin(None) == {}
    # a directory without a fit shows no table: the gate it would show does not exist yet
    fresh = tmp_path / "fresh"
    assert DEMO.main(_table_argv(fresh, tmp_path / "t.html", "read-write")) == 1
    assert not (fresh / "demo").exists() and not (fresh / "store").exists()


def test_demo_table_is_rebuilt_read_only_without_live_requests(fitted, tmp_path):
    state, page = _copy(fitted, tmp_path), tmp_path / "table.html"
    # read-only before anything was asked: the cells say that no answer is stored, and hold no value
    assert DEMO.main(_table_argv(state, page, "read-only")) == 1
    (empty,) = _tables(state)
    assert empty["counts"]["live_requests"] == 0 and empty["counts"]["empty_cells"] > 0
    cell = empty["columns"]["D"]["cell"]
    assert cell["reading"] == {"status": "no_stored_answer", "text": "저장된 답 없음"}
    assert cell["existing_question"] == {"status": "no_stored_answer", "text": "저장된 답 없음"}
    assert cell["keyword"]["text"] == "안 걸림 → 사전값"            # the keyword row needs no answer
    assert empty["columns"]["B"]["outcomes"]["utt_gate_fit"]["status"] == "needs_review"
    assert empty["columns"]["B"]["outcomes"]["utt_gate_fit"]["realized_saving"] is None
    assert "저장된 답 없음" in page.read_text(encoding="utf-8")
    # read-write asks what is missing; read-only then rebuilds the same table without a live request
    assert DEMO.main(_table_argv(state, page, "read-write")) == 0
    assert DEMO.main(_table_argv(state, page, "read-only")) == 0
    _, asked, rebuilt = _tables(state)
    assert asked["counts"]["live_requests"] > 0 and asked["counts"]["empty_cells"] == 0
    assert rebuilt["counts"] == {"live_requests": 0, "live_failed_requests": 0, "empty_cells": 0,
                                 "store_hits": asked["counts"]["live_requests"] + asked["counts"]["store_hits"]}
    assert _shown(rebuilt) == _shown(asked)
    for table in (asked, rebuilt):
        assert table["evidence_class"] == "cpu_rehearsal_not_model_evidence"
        assert table["counts_as_model_evidence"] is False and table["real_model_backend"] is False
        # rehearsal thresholds run the utt_gate_fit arm but are never shown under that name
        assert table["gate"]["arm"] == "utt_gate_fit" and table["gate"]["fitted_on"] == "utt-smoke-rehearsal"
        assert table["gate"]["name"] == "utt_gate_fit_rehearsal"
    # the pinned sentences: D1 in round 1 and L1 in round 2 of column B
    rounds = {cell["round"]: cell for cell in rebuilt["columns"]["B"]["rounds"]}
    assert rounds[1]["utterance"] == W.bank_sentence("D1")[1] and rounds[2]["utterance"] == W.bank_sentence("L1")[1]
    assert rebuilt["focus_round"] == 2 and rounds[2]["labels"]["says_final"] is True
    # without pins the draw decides: this seed's round 1 is another sentence of the same class
    assert DEMO.main(_table_argv(state, tmp_path / "unpinned.html", "read-write", pin=None)) == 0
    unpinned = {cell["round"]: cell for cell in _tables(state)[-1]["columns"]["B"]["rounds"]}
    assert unpinned[1]["utterance"] == W.bank_sentence("D2")[1] != rounds[1]["utterance"]
    assert unpinned[1]["labels"]["core_class"] == rounds[1]["labels"]["core_class"] == "deadline_pressure"
    # the scene of the design document: the CPU references part ways on the default wording
    outcomes_a, outcomes_b = rebuilt["columns"]["A"]["outcomes"], rebuilt["columns"]["B"]["outcomes"]
    assert (outcomes_a["prior"]["status"], outcomes_a["prior"]["realized_saving"]) == ("supplier_walked", 0)
    assert (outcomes_a["keyword"]["status"], outcomes_a["keyword"]["realized_saving"]) == ("agreed", 489_414)
    assert (outcomes_b["keyword"]["status"], outcomes_b["keyword"]["realized_saving"]) == ("supplier_walked", 0)
    assert [step["action"] for step in outcomes_a["keyword"]["actions"]] == ["counter", "accept"]
    # one self-contained file that carries its label
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "cpu_rehearsal_not_model_evidence" in text
    assert DEMO.REHEARSAL_BANNER in text
    for external in ("http://", "https://", "<script", "<link", "src=", "@import", "url("):
        assert external not in text
    # every attempt keeps its own table file; only the HTML page is replaced
    assert len({folder.name for folder in (state / "demo").iterdir()}) == 4
    assert not (state / "first_use.json").exists()                 # a CPU rehearsal is no binding event


def test_demo_table_has_the_instruction_column_and_the_existing_question_row(fitted, fitted_double, tmp_path):
    state, page = _copy(fitted, tmp_path), tmp_path / "table.html"
    assert DEMO.main(_table_argv(state, page, "read-write")) == 0
    assert DEMO.main(_table_argv(state, page, "read-only")) == 0
    table = _tables(state)[-1]
    assert table["counts"]["live_requests"] == 0
    columns = table["columns"]
    assert set(columns) == {"A", "B", "C", "D", "neutral"}
    instruction = columns["D"]["cell"]
    assert instruction["utterance"] == f"{W.bank_sentence('N1')[1]} {W.bank_sentence('IF')[1]}"
    assert instruction["labels"]["has_instruction"] is True and instruction["labels"]["says_final"] is False
    assert instruction["reading"]["value_from"] == "instruction_flagged_round_prior"
    assert instruction["reading"]["text"] == "지시 감지: 발언으로 추정을 올리지 않음"
    assert instruction["reading"]["final_offer_value"] == instruction["round_prior"] == 0.15
    negated = columns["C"]["cell"]
    assert negated["utterance"] == W.bank_sentence("K1")[1] and negated["keyword"]["fires"] is True
    assert negated["reading"]["value_from"] == "round_prior" and negated["labels"]["says_final"] is False
    assert columns["neutral"]["cell"]["utterance"] == W.bank_sentence("N1")[1]
    assert {columns[key]["cell"]["round"] for key in ("C", "D", "neutral")} == {table["focus_round"]}
    # the existing-question row: one value for every shown state, replayed without a live request
    cells = [cell for key in ("A", "B") for cell in columns[key]["rounds"]] + [
        columns[key]["cell"] for key in ("C", "D", "neutral")]
    assert len(cells) == 7
    for cell in cells:
        assert cell["existing_question"]["status"] == "ok" and 0.0 <= cell["existing_question"]["final_offer"] <= 1.0
        assert cell["reading"]["status"] == "ok" and set(cell["reading"]["answers"]) == {
            "u_final", "u_walkaway", "u_concession", "u_instruction"}
    header, rows = DEMO.table_rows(table)
    assert header == ["", "A: 기본 문구", "B: 같은 뜻, 다른 표현", "C: 음성 대조 (키워드는 있지만 뜻은 반대)", "D: 지시문"]
    labels = [row[0] for row in rows]
    assert labels == ["2라운드 공급사 발언", "키워드 규칙", "CPU 대역(모델 아님) 기존 질문 하나",
                      "CPU 대역(모델 아님) 발언 판독", "정책의 선택과 결과"]
    assert rows[1][1:] == ["걸림 → 1.0", "안 걸림 → 사전값", "걸림 → 1.0", "안 걸림 → 사전값"]
    assert all(text.startswith("final_offer=") for text in rows[2][1:])
    assert rows[3][4].endswith("지시 감지: 발언으로 추정을 올리지 않음")
    assert rows[4][3] == rows[4][4] == "판독만 표시한다. 협상 결과는 표시하지 않는다."
    # a declared-model double gets the model's row names and a replay label, and is still not evidence
    real = tmp_path / "real"
    shutil.copytree(fitted_double.state, real)
    seams = dict(registry_path=tmp_path / "shared.json", **DOUBLE)
    assert DEMO.main(_table_argv(real, page, "read-write"), **seams) == 0
    assert DEMO.main(_table_argv(real, page, "read-only"), **seams) == 0
    asked, replayed = _tables(real)
    assert asked["evidence_class"].startswith("real_model_composed_utterance")
    assert replayed["evidence_class"] == "real_model_composed_utterance_replay_from_store"
    assert replayed["counts_as_model_evidence"] is False and asked["counts_as_model_evidence"] is False
    assert DEMO.table_rows(replayed)[1][3][0] == "Zoro-1 발언 판독"
    assert DEMO.REHEARSAL_BANNER not in page.read_text(encoding="utf-8")


def test_demo_and_ask_refuse_the_same_backends(fitted, tmp_path, capsys):
    state, built = _copy(fitted, tmp_path), []
    before = _files(state)
    checkpoints = ["--original-checkpoint", "a", "--exported-checkpoint", "b"]
    refused = (["--backend", "zoro", *checkpoints],                                    # no temperature
               ["--backend", "zoro", *checkpoints, "--decision-temperature", "0.7"],
               ["--backend", "zoro", "--decision-temperature", "1"])                   # read-write, no checkpoints
    for extra in refused:
        table = ["table", "--state-dir", str(state), "--seed", "1073", *extra]
        ask = ["ask", "--text", "문장", "--seed", "1073", "--state-dir", str(state), *extra]
        for argv in (table, ask):
            assert DEMO.main(argv, backend_factory=lambda: built.append(1)) == 1
            assert "Refused" in capsys.readouterr().err
            namespace = X.build_demo_parser().parse_args(argv)
            with pytest.raises(X.ExperimentRefusal):
                X.check_arguments(namespace)                        # the experiment's own refusal, unchanged
    for extra in (["--backend", "zoro-http"], ["--backend", "systemone"], ["--backend", "cpu-stand-in", "--reasoning"],
                  ["--backend", "cpu-stand-in", "--url", "http://localhost:1"], []):
        for command in (["table"], ["ask", "--text", "문장"]):
            with pytest.raises(SystemExit):
                DEMO.main([*command, "--seed", "1073", "--state-dir", str(state), *extra])
    assert not built and _files(state) == before
    assert not (tmp_path / "registry.json").exists()


def test_ask_writes_its_answer_apart_from_the_store(fitted, tmp_path, capsys):
    state = _copy(fitted, tmp_path)
    store_before = _files(state / "store")
    sentence = W.bank_sentence("W2")[1]
    assert DEMO.main(["ask", "--text", sentence, "--seed", "1073", *_stand_in(state)]) == 0
    (answer,) = [X.read_json(path) for path in sorted((state / "ask").rglob("answer.json"))]
    assert answer["status"] == "ok" and answer["text"] == sentence
    assert answer["answers"] == {"u_final": 0.0, "u_walkaway": 1.0, "u_concession": 0.0, "u_instruction": 0.0}
    assert answer["value_from"] == "utterance_final" and answer["final_offer_value"] == 1.0
    assert answer["evidence_class"] == "cpu_rehearsal_not_model_evidence"
    assert answer["counts_as_model_evidence"] is False and answer["in_report"] is False
    assert _files(state / "store") == store_before                  # the experiment store is untouched
    # a sentence the stand-in cannot read is shown as a failure, with no value
    capsys.readouterr()
    assert DEMO.main(["ask", "--text", "은행에 없는 문장입니다.", "--seed", "1073", *_stand_in(state)]) == 1
    assert "판독 실패: 사람 검토로 넘김 (ValueError)" in capsys.readouterr().out
    failed = [X.read_json(path) for path in sorted((state / "ask").rglob("answer.json"))][-1]
    assert failed["status"] == "error" and failed["verdict"] == "판독 실패: 사람 검토로 넘김"
    assert "answers" not in failed and "final_offer_value" not in failed
    for text in ("", "   "):
        assert DEMO.main(["ask", "--text", text, "--seed", "1073", *_stand_in(state)]) == 1
    # with a declared-model double the label is the single-sentence demo label and the use is recorded
    assert DEMO.main(["ask", "--text", sentence, "--seed", "1073", *_stand_in(state)], backend_factory=ModelDouble,
                     real_model=True, registry_path=tmp_path / "shared.json") == 0
    real = [X.read_json(path) for path in sorted((state / "ask").rglob("answer.json"))][-1]
    assert real["evidence_class"] == "real_model_single_utterance_demo" and real["counts_as_model_evidence"] is False
    used = X.read_json(state / "first_use.json")
    assert (used["event"], used["block"]) == ("first_real_response", "ask")
    assert _files(state / "store") == store_before
    # the report does not read ask/
    assert X.main(["report", "--state-dir", str(state)]) == 0
    assert "ask" not in json.dumps(X.read_json(state / "report.json")["rehearsal"])


class AlwaysFails(ModelDouble):
    """A declared-model test double whose every call raises. NOT a model; it never answers."""

    def predict(self, state, candidates, request):
        raise RuntimeError("invented failure of every call")


FAILING = dict(backend_factory=AlwaysFails, declared=ModelDouble.DECLARED, real_model=True)


def test_real_runs_and_tables_without_any_answer_are_labelled_as_such(fitted_double, tmp_path):
    """Section 9: a failed call is an attempt, not an answer; no answer means no real-model label."""
    state = _copy(fitted_double, tmp_path)
    seams = dict(registry_path=tmp_path / "shared.json", **FAILING)
    assert _run(state, "paraphrase", "utt_gate_fit", **seams) == 1
    (attempt,) = _attempts(state, "paraphrase", "utt_gate_fit")
    result = attempt["result"]
    assert result["status"] == "failed" and result["cases_run"] == 1
    assert result["counts"]["live_requests"] == result["counts"]["live_failed_requests"] >= 1
    assert result["evidence_class"] in ("real_model_requested_no_model_answer",
                                        "real_model_composed_utterance_replay_from_store")
    page = tmp_path / "t.html"
    assert DEMO.main(_table_argv(state, page, "read-write"), **seams) == 1
    table = _tables(state)[-1]
    assert table["counts"]["live_requests"] == table["counts"]["live_failed_requests"] > 0
    assert table["counts"]["store_hits"] == 0 and table["counts"]["empty_cells"] > 0
    assert table["evidence_class"] == "real_model_requested_no_model_answer"
    assert table["columns"]["D"]["cell"]["reading"]["status"] == "error"


def _bench_block_from_smoke(state: Path, *, stop: bool) -> dict:
    """An invented bench-block result: the smoke bench's own items and readers, with a chosen verdict.

    No bench-block case is built or asked; only the file a stopping-rule check and --from-bench read.
    """
    smoke = X.first_complete_attempt(state / "bench" / "smoke")["result"]
    result = json.loads(json.dumps(smoke))
    result["readers"]["utt_gate_fit"] = result["readers"]["utt_gate"]
    result["verdict"].update(basis="utt_gate_fit", reading="utterance_reading_not_demonstrated" if stop else
                             "utterance_reading_demonstrated_on_bench",
                             raw_signal_reads_paraphrased_final={"passed": not stop})
    X.write_json(state / "bench" / "bench" / "attempt-invented" / "result.json", result)
    return result


def test_demo_table_obeys_the_stopping_rule_and_shows_the_bench_answers(fitted_double, tmp_path, capsys):
    """Section 8.3: once the stopping rule is met G-demo is not run; the table shows the bench answers."""
    state, page = _copy(fitted_double, tmp_path), tmp_path / "t.html"
    bench = _bench_block_from_smoke(state, stop=True)
    assert O.stop_rule(state, B.load_state_thresholds(state))["rules"] == [2]
    built, registry = [], tmp_path / "shared.json"

    def factory():
        built.append(1)
        return ModelDouble()

    seams = dict(backend_factory=factory, declared=ModelDouble.DECLARED, real_model=True, registry_path=registry)
    store_before, first_use = _files(state / "store"), X.read_json(state / "first_use.json")
    assert DEMO.main(_table_argv(state, page, "read-write"), **seams) == 1
    err = capsys.readouterr().err
    assert "stopping rule [2]" in err and "--from-bench" in err
    assert not (state / "demo").exists() and not page.exists() and not built
    # the "R1 not demonstrated" table: no backend, no request, no first use, bench answers in the model cells
    assert DEMO.main([*_table_argv(state, page, "read-write"), "--from-bench"], **seams) == 0
    assert not built and not registry.exists() and _files(state / "store") == store_before
    assert X.read_json(state / "first_use.json") == first_use
    (table,) = _tables(state)
    assert table["model_cells_from"]["source"] == "bench_block_answers"
    assert table["model_cells_from"]["attempt"] == "attempt-invented"
    assert table["model_cells_from"]["stopping_rule"]["met"] is True
    assert table["evidence_class"] == bench["evidence_class"] == "real_model_utterance_bench"
    assert table["counts"]["live_requests"] == table["counts"]["store_hits"] == 0
    assert table["counts"]["empty_cells"] == 0 and table["counts_as_model_evidence"] is False
    outcomes = table["columns"]["B"]["outcomes"]
    assert set(outcomes) == {"prior", "keyword", "utt_gate_fit"}
    assert outcomes["utt_gate_fit"]["status"] == "not_run_stopping_rule" and outcomes["utt_gate_fit"]["actions"] == []
    rounds = {cell["round"]: cell for cell in table["columns"]["B"]["rounds"]}
    l1 = W.bank_sentence("L1")[1]
    expected = [item for item in bench["items"] if item["message"] == l1 and item["status"] == "ok"]
    assert rounds[2]["utterance"] == l1 and rounds[2]["reading"]["status"] == "from_bench"
    assert rounds[2]["reading"]["states"] == len(expected) > 0
    assert rounds[2]["reading"]["mean_answers"]["u_final"] == pytest.approx(
        sum(item["answers"]["u_final"] for item in expected) / len(expected))
    for key in ("C", "D", "neutral"):
        assert table["columns"][key]["cell"]["reading"]["status"] == "from_bench", key
    # the default-wording deadline sentence was never a bench item: shown as such, nothing invented
    first_a = table["columns"]["A"]["rounds"][0]
    assert first_a["reading"] == {"status": "not_in_bench", "text": DEMO.NOT_IN_BENCH}
    assert table["counts"]["cells_not_in_bench"] >= 2
    text = page.read_text(encoding="utf-8")
    assert "중단 규칙(8.3절)에 따라 G-demo를 실행하지 않았다" in text and "이 사례의 상태가 아님" in text
    # without a completed bench-block result there is nothing to show
    fresh = _copy(fitted_double, tmp_path / "f")
    assert DEMO.main([*_table_argv(fresh, page, "read-write"), "--from-bench"], **seams) == 1
    assert not (fresh / "demo").exists()
    # a bench reading that does not meet the stopping rule does not block the demo
    open_state = _copy(fitted_double, tmp_path / "o")
    _bench_block_from_smoke(open_state, stop=False)
    assert O.stop_rule(open_state, B.load_state_thresholds(open_state))["met"] is False
    assert DEMO.main(_table_argv(open_state, page, "read-write"), **seams) == 0 and built


def test_ask_refuses_read_only_and_demo_locations_in_protected_folders(fitted, tmp_path, monkeypatch, capsys):
    state = _copy(fitted, tmp_path)
    before, built = _files(state), []
    argv = ["ask", "--text", W.bank_sentence("W2")[1], "--seed", "1073",
            *_stand_in(state, "--store-mode", "read-only")]
    assert DEMO.main(argv, backend_factory=lambda: built.append(1), real_model=True,
                     registry_path=tmp_path / "shared.json") == 1
    assert "read-only is refused" in capsys.readouterr().err
    assert not built and _files(state) == before and not (tmp_path / "shared.json").exists()
    # --state-dir and --html inside negotiation/evidence or negotiation/artifacts: refused, nothing created
    root = tmp_path / "repo"
    monkeypatch.setattr(X, "REPOSITORY_ROOT", root)
    evidence = root / "negotiation" / "evidence"
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_probe_result(), ensure_ascii=False), encoding="utf-8")
    for argv in (_table_argv(state, evidence / "t.html", "read-write"),
                 _table_argv(evidence / "s", tmp_path / "t.html", "read-write"),
                 ["ask", "--text", "문장", "--seed", "1073", *_stand_in(root / "negotiation" / "artifacts" / "s")],
                 ["probe-table", "--probe-result", str(probe), "--html", str(evidence / "p.html")]):
        assert DEMO.main(argv) == 1
        assert "inside negotiation/" in capsys.readouterr().err
    assert not (root / "negotiation").exists() and _files(state) == before


def _probe_result(**changes) -> dict:
    """An invented result file of the earlier wording probe. No value here comes from a model."""
    shifts = {"explicit_final": 0.123456, "paraphrase_final": 0.234567, "negated_final": -0.345678,
              "open_to_more": -0.456789, "embedded_instruction": 0.567891}
    base = {
        "schema_version": 1, "experiment": "utterance_wording_probe", "complete": True,
        "protocol_version": "invented", "created_at_utc": "2026-11-18T00:00:00+00:00",
        "evidence_class": "real_model_probe", "backend": {"gpu_name": "Invented GPU", "dtype": "fp16"},
        "scope": {"states": 16, "variants_per_state": 9, "planned_requests": 144},
        "metrics": {"neutral_mean_estimate": 0.4321,
                    "by_class": {name: {"states": 16, "mean_shift": value, "share_of_states_up": 1.0,
                                        "mean_shift_ci90_by_state": [value - 0.0111, value + 0.0222]}
                                 for name, value in shifts.items()}},
        "verdict": {"reading": "invented_reading", "warnings": {"embedded_instruction_moves_estimate": True}},
    }
    return {**base, **changes}


def test_probe_table_copies_the_probe_result_without_recomputing(tmp_path, capsys):
    source, page = tmp_path / "probe.json", tmp_path / "probe.html"
    data = _probe_result()
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    table = DEMO.build_probe_table(source)
    assert [row["class"] for row in table["rows"]] == ["explicit_final", "paraphrase_final", "negated_final",
                                                      "open_to_more", "embedded_instruction"]
    for row in table["rows"]:
        copied = data["metrics"]["by_class"][row["class"]]
        assert row["model_mean_shift"] == copied["mean_shift"]                 # the file's value, digit for digit
        assert row["model_ci90"] == copied["mean_shift_ci90_by_state"] and row["states"] == 16
    assert table["neutral_mean_estimate"] == 0.4321 and table["evidence_class"] == "real_model_probe"
    assert (table["gpu_name"], table["dtype"], table["states"], table["variants_per_state"]) == (
        "Invented GPU", "fp16", 16, 9)
    assert table["reading"] == "invented_reading" and table["is_real_model_probe"] is True
    # the keyword column is the literal CPU baseline on the probe's own states: misses paraphrases, fires on negation
    keyword = {row["class"]: row["keyword_mean_shift"] for row in table["rows"]}
    assert round(keyword["explicit_final"], 3) == round(keyword["negated_final"], 3) == 0.803
    assert keyword["paraphrase_final"] == keyword["open_to_more"] == keyword["embedded_instruction"] == 0.0
    assert table["keyword_reference"]["evidence_class"] == "cpu_reference_not_model_evidence"
    assert DEMO.main(["probe-table", "--probe-result", str(source), "--html", str(page)]) == 0
    printed = capsys.readouterr().out
    assert "등급 real_model_probe" in printed and "Invented GPU" in printed and "+0.123 [+0.112, +0.146]" in printed
    text = page.read_text(encoding="utf-8")
    assert "real_model_probe" in text and "+0.568" in text and "정답률이 아니라 표현에 대한 반응이다" in text
    for external in ("http://", "https://", "<script", "<link", "src="):
        assert external not in text
    # another label is shown as it is, in the title
    other = tmp_path / "cpu.json"
    other.write_text(json.dumps(_probe_result(evidence_class="cpu_reference_not_model_evidence")), encoding="utf-8")
    relabelled = DEMO.build_probe_table(other)
    assert relabelled["is_real_model_probe"] is False
    assert "등급 cpu_reference_not_model_evidence" in DEMO._probe_title(relabelled)
    # a file of another experiment, an incomplete result, or no file at all is refused
    refused = {"other": _probe_result(experiment="full_negotiation_evaluation"),
               "partial": _probe_result(complete=False),
               "no_metrics": {key: value for key, value in _probe_result().items() if key != "metrics"}}
    for name, content in refused.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        with pytest.raises(X.ExperimentRefusal):
            DEMO.build_probe_table(path)
        assert DEMO.main(["probe-table", "--probe-result", str(path), "--html", str(tmp_path / f"{name}.html")]) == 1
        assert not (tmp_path / f"{name}.html").exists()
    assert DEMO.main(["probe-table", "--probe-result", str(tmp_path / "missing.json")]) == 1
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert DEMO.main(["probe-table", "--probe-result", str(tmp_path / "broken.json")]) == 1


def test_track_b_files_say_they_never_ran_on_a_gpu_and_keep_the_fixed_wording():
    for path in (NEGOTIATION / "utterance_demo.py", NEGOTIATION / "src" / "utterance_agent" / "outcome.py"):
        assert "NOT EXECUTED ON A GPU as of 2026-11-18" in path.read_text(encoding="utf-8"), path.name
    # the three sentences a viewer sees for a missing answer, a failed reading and a flagged instruction
    assert DEMO.NO_STORED_ANSWER == "저장된 답 없음" and DEMO.READ_FAILED == "판독 실패: 사람 검토로 넘김"
    assert DEMO.INSTRUCTION_VERDICT == "지시 감지: 발언으로 추정을 올리지 않음"
    assert DEMO.SCENARIO_SEEDS == (1000, 1099)
    # both new files are among the inputs a result records
    assert {"negotiation/utterance_demo.py", "negotiation/src/utterance_agent/outcome.py"} <= set(
        X.package_input_hashes())
