"""CPU tests of the tuning runner.

NOT executed on a GPU as of 2026-11-18 (CPU reference backends and test doubles only). Every
"model" here is a test double built on the CPU keyword reference, flagged as a model only so
that the labelling paths of a real-model run can be exercised. No number in this file is model
evidence. Blocks are shortened with the runner's rehearsal option so the tests run in seconds.
"""

import ast
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from negotiation_agent import evaluation as E, policy
from negotiation_agent import tune
from negotiation_agent.backends import HeuristicBackend, KeywordBackend
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec, round_prior
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.prediction_store import PredictionStore
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.simulator import default_buyer_private

REHEARSAL = "d0=3,select=6,confirm=8,cal=20"
IDENTITY = {"backend": "tuning_test_double", "note": "test double built on the keyword reference; not a model"}
SPLIT_LITERAL = "ev" + "al"          # spelled in two parts so this file can be scanned as well
FORBIDDEN_NAMES = {"HISTORY_DIR", "reserve_heldout", "record_exploratory", "freeze", "evaluation_identity",
                   "evaluation_case_identities", "SPLIT_SEED_BASE"}
EVIDENCE_DIR = tune.REPOSITORY_ROOT / "negotiation" / "evidence"


class ModelDouble:
    """KeywordBackend values behind a declared-model test double. It is NOT a model."""

    kind, is_model, supports_final_offer = "tuning_test_double_model", True, True

    def __init__(self, interrupt_at=None, fail_at=(), final=None, dtype="test_only"):
        self.calls = self.answered = 0
        self.interrupt_at, self.fail_at, self.final, self.dtype = interrupt_at, set(fail_at), final, dtype
        self.last_response = None
        self._base = KeywordBackend()

    def predict(self, state, candidates, request):
        self.last_response = None
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt("test interruption")
        if self.calls in self.fail_at:
            raise RuntimeError("test backend failure")
        out = self._base.predict(state, candidates, request)
        if self.final is not None:
            out[FINAL_OFFER_QID] = self.final(state)
        self.answered += 1
        self.last_response = {"metadata": {"dtype": self.dtype, "sequence": self.calls}}
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True, "test_double": True}


def cli(state, *argv, double=None, factory=None, real=True, clock=None):
    """One command through tune.main with the double in the model's place. Returns (exit code, double)."""
    double = double if double is not None else ModelDouble()
    extra = {} if clock is None else {"clock": clock}
    code = tune.main([*argv, "--state-dir", str(state), "--backend", "keyword"],
                     backend_factory=factory or (lambda: double), declared=(ModelDouble.kind, True, True),
                     real_model=real, store_requested=IDENTITY, **extra)
    return code, double


def planned(tmp_path, name="state", rehearsal=REHEARSAL):
    state = tmp_path / name
    assert cli(state, "plan", "--rehearsal-blocks", rehearsal)[0] == 0
    return state


def attempts(state, stage, name="*"):
    """(folder, result) of every attempt of a stage, oldest first per configuration."""
    folders = sorted(Path(state, "runs", stage).glob(f"{name}-*/attempt-*"))
    return [(folder, json.loads((folder / "result.json").read_text(encoding="utf-8"))) for folder in folders]


def results(state, stage, name="*"):
    return [result for _, result in attempts(state, stage, name)]


def complete_rows(state, stage):
    """config name -> per-case outcome of its complete attempt.

    Keyed by name: the id of a Platt arm contains the hash of the calibrate result file it was
    fitted from, which holds timestamps and so differs between two state directories.
    """
    out = {}
    for result in results(state, stage):
        if result["complete"] and result.get("confirm_use") != "not_confirmatory":
            assert result["config"]["name"] not in out
            out[result["config"]["name"]] = [(r["case_id"], r["status"], r["realized_saving"], r["net_saving"])
                                             for r in result["rows"]]
    return out


def read(state, name):
    return json.loads(Path(state, name).read_text(encoding="utf-8"))


def store_entries(state):
    return len(PredictionStore(Path(state) / "store", IDENTITY, mode="read-only"))


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """One whole pipeline with the double: plan, all, (copy), confirm, board."""
    evidence_before = EVIDENCE_DIR.exists()
    splits, original = [], tune.make_cases

    def recording(split, count=E.DEFAULT_CASES):
        splits.append(split)
        return original(split, count)

    tune.make_cases = recording
    try:
        root = tmp_path_factory.mktemp("tune")
        state, double = root / "state", ModelDouble()
        assert cli(state, "plan", "--rehearsal-blocks", REHEARSAL, double=double)[0] == 0
        assert double.calls == 0 and not (state / "store").exists()        # planning builds and calls nothing
        assert cli(state, "run", "--stage", "all", double=double)[0] == 0
        after_all = root / "after_all"
        shutil.copytree(state, after_all)
        answered_after_all = double.answered
        assert cli(state, "run", "--stage", "confirm", double=double)[0] == 0
        assert cli(state, "board", double=double)[0] == 0
    finally:
        tune.make_cases = original
    return SimpleNamespace(state=state, after_all=after_all, double=double, splits=splits,
                           answered_after_all=answered_after_all, evidence_before=evidence_before,
                           board=read(state, "leaderboard.json"), after_all_board=read(after_all, "leaderboard.json"))


# ---- protocol and blocks -----------------------------------------------------------------------------------

def test_registered_blocks_are_slices_of_the_single_prefix_stream():
    dev = E.make_cases("dev", 450)
    assert tune.block_cases("d0") == dev[0:30]
    assert tune.block_cases("select") == E.make_cases("dev", 150)[30:150] == dev[30:150]
    assert tune.block_cases("confirm") == dev[150:450]
    assert tune.block_cases("cal") == E.make_cases("calibration", 60)
    ids = lambda name: (tune.block_cases(name)[0].case_id, tune.block_cases(name)[-1].case_id)  # noqa: E731
    assert ids("d0") == ("dev-1100", "dev-1129") and ids("select") == ("dev-1130", "dev-1249")
    assert ids("confirm") == ("dev-1250", "dev-1549") and ids("cal") == ("calibration-2000", "calibration-2059")
    # A block is never reseeded at its own offset: the split's stream decides style and flags, and it differs
    # from what a fresh stream would draw for the same seeds.
    assert all(case.seed == 1100 + 30 + i for i, case in enumerate(tune.block_cases("select")))
    assert [c.hidden for c in tune.block_cases("select")[:5]] != [c.hidden for c in E.make_cases("dev", 5)]
    # The reserve is disjoint from every block, and no block names a split other than the two allowed ones.
    assert max(stop for split, _, stop in tune.BLOCKS.values() if split == "dev") == tune.RESERVE["start"] == 450
    assert {split for split, _, _ in tune.BLOCKS.values()} == {"dev", "calibration"}
    assert set(tune.ALLOWED_SPLITS) == {"dev", "calibration"}


def test_protocol_document_fixes_the_family_rules_and_exposure():
    protocol = tune.build_protocol()
    assert protocol["protocol_version"] == "tune-2026-11-18.v1" and protocol["registered_protocol"] is True
    assert [c["name"] for c in protocol["candidates"]] == ["M0", "M1", "M2", "M3", "F1", "F2", "M4", "M5"]
    assert {b: (v["first_case_id"], v["last_case_id"], v["cases"]) for b, v in protocol["blocks"].items()} == {
        "d0": ("dev-1100", "dev-1129", 30), "select": ("dev-1130", "dev-1249", 120),
        "confirm": ("dev-1250", "dev-1549", 300), "cal": ("calibration-2000", "calibration-2059", 60)}
    assert all(len(v["content_sha256"]) == 64 for v in protocol["blocks"].values())
    assert protocol["bootstrap"] == {"resamples": 2000, "seed": 20261116, "level": 0.90}
    assert "the real model has seen dev-1100..1129 only" in protocol["known_exposure"]
    assert "had not been run with the real model" in protocol["known_exposure"]
    assert protocol["verdict_vocabulary"] == ["improvement_confirmed", "not_confirmed", "incomplete"]
    assert set(protocol["verdict_vocabulary"]) <= set(protocol["allowed_result_sentences"])
    assert protocol["golden_request"]["sha256"] == tune.golden_request_sha256()
    references = [r["name"] for r in protocol["references"]]
    assert references == ["rule_policy", "constant_policy_prior", "constant_keyword", "rule_keyword",
                          "fixed_concession", "midpoint", "reference_true_acceptance"]
    assert all(r["selectable"] is False and r["label"] == "cpu_reference_not_model_evidence"
               for r in protocol["references"])
    # Every candidate's model-free twin is one of the references (deduplicated, zero model calls).
    assert {c["model_free_twin"] for c in protocol["candidates"]} == {"constant_policy_prior", "constant_keyword"}
    assert all(not config.uses_model() for config in tune.reference_configs())
    # The document holds hashes of the case contents, never the hidden contents themselves.
    text = json.dumps(protocol, ensure_ascii=False)
    assert "reserve_profit" not in text and "unit_cost" not in text and "target_total_cost" not in text


def test_plan_writes_the_protocol_once_and_refuses_another_digest(tmp_path, capsys):
    state = tmp_path / "state"
    built = []
    code, _ = cli(state, "plan", "--rehearsal-blocks", REHEARSAL, factory=lambda: built.append(1))
    out = capsys.readouterr().out
    assert code == 0 and not built and "no model call was made" in out
    assert "ESTIMATES, not measurements" in out and "(estimate, upper bound)" in out
    assert "REHEARSAL BLOCKS" in out and tune.KNOWN_EXPOSURE in out and "Golden request hash check" in out
    assert sorted(p.name for p in state.iterdir()) == ["protocol.json"]
    first = (state / "protocol.json").read_bytes()
    document = json.loads(first)
    assert document["protocol_sha256"] == tune._protocol_digest_of_file(document)
    # Planning again with the same blocks changes nothing; other blocks are a different digest and are refused.
    assert cli(state, "plan", "--rehearsal-blocks", REHEARSAL)[0] == 0
    assert cli(state, "plan")[0] == 0                         # follows the recorded (rehearsal) protocol
    assert cli(state, "plan", "--rehearsal-blocks", "d0=3,select=7,confirm=8,cal=20")[0] == 1
    assert "protocol digest differs" in capsys.readouterr().err
    assert (state / "protocol.json").read_bytes() == first
    # A protocol file edited by hand no longer matches this code.
    document["selection_rule"] = "pick whatever looks best afterwards"
    (state / "protocol.json").write_text(json.dumps(document), encoding="utf-8")
    assert cli(state, "plan")[0] == 1 and cli(state, "board")[0] == 1
    assert cli(state, "run", "--stage", "smoke", factory=lambda: built.append(1))[0] == 1 and not built
    # A registered plan is a different digest than a rehearsal plan.
    registered = tmp_path / "registered"
    assert cli(registered, "plan")[0] == 0
    assert read(registered, "protocol.json")["registered_protocol"] is True
    assert read(registered, "protocol.json")["protocol_sha256"] != json.loads(first)["protocol_sha256"]
    assert cli(registered, "plan", "--rehearsal-blocks", REHEARSAL)[0] == 1
    # A directory with other content and no protocol is not adopted.
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("x", encoding="utf-8")
    assert cli(stray, "plan")[0] == 1 and not (stray / "protocol.json").exists()


def test_plan_estimates_are_requests_times_seconds_and_marked_as_estimates():
    rows = {row["stage"]: row for row in tune.plan_estimates(tune.build_protocol())}
    assert rows["regress"]["estimated_requests_upper_bound"] == 77
    assert rows["select"]["estimated_requests_upper_bound"] == 8 * 308      # a full pass per candidate
    assert rows["select"]["estimated_minutes_upper_bound"] == round(8 * 308 * 1.75 / 60, 1)
    assert all(set(row) == {"stage", "cases", "model_configurations", "estimated_requests_upper_bound",
                            "estimated_minutes_upper_bound"} for row in rows.values())


def test_golden_request_hash_is_checked_before_anything_is_written(tmp_path, monkeypatch, capsys):
    assert tune.golden_request_sha256() == "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
    assert tune.check_golden_request() == tune.GOLDEN_REQUEST["sha256"]
    monkeypatch.setitem(tune.GOLDEN_REQUEST, "sha256", "0" * 64)
    state = tmp_path / "state"
    assert cli(state, "plan")[0] == 1 and not state.exists()
    assert "request contract changed" in capsys.readouterr().err


def test_command_line_refusals_happen_before_any_file_or_backend(tmp_path, capsys):
    state = tmp_path / "state"
    plain = lambda *argv: tune.main([*argv, "--state-dir", str(state)])      # noqa: E731 - the real CLI path
    assert plain("run", "--stage", "smoke") == 1                           # no silent backend default
    assert "--backend is required" in capsys.readouterr().err
    assert plain("run", "--stage", "smoke", "--backend", "heuristic") == 1
    assert "final-offer question" in capsys.readouterr().err
    assert plain("run", "--stage", "smoke", "--backend", "zoro") == 1
    assert "--decision-temperature 1" in capsys.readouterr().err
    assert plain("run", "--stage", "smoke", "--backend", "zoro", "--decision-temperature", "0.7") == 1
    capsys.readouterr()
    assert plain("run", "--stage", "smoke", "--backend", "zoro", "--decision-temperature", "1") == 1
    assert "--original-checkpoint" in capsys.readouterr().err
    # This protocol version tunes with the native adapter only: an HTTP client cannot bind a server identity.
    for backend in ("zoro-http", "systemone"):
        assert plain("run", "--stage", "smoke", "--backend", backend, "--decision-temperature", "1",
                     "--url", "http://127.0.0.1:1") == 1
        assert "native adapter only" in capsys.readouterr().err
    for folder in ("negotiation/evidence/holdout_history", "negotiation/evidence", "negotiation/artifacts/x"):
        protected = tune.REPOSITORY_ROOT / folder
        assert tune.main(["plan", "--state-dir", str(protected)]) == 1
        assert "held-out history or curated evidence" in capsys.readouterr().err
        assert not EVIDENCE_DIR.exists() and not (tune.REPOSITORY_ROOT / "negotiation/artifacts/x").exists()
    assert plain("run", "--stage", "smoke", "--backend", "keyword", "--reasoning") == 1
    assert "--reasoning is only valid" in capsys.readouterr().err
    assert plain("plan", "--backend", "zoro", "--rehearsal-blocks", REHEARSAL) == 1
    assert "refused for real-model backends" in capsys.readouterr().err
    assert plain("plan", "--rehearsal-blocks", "d0=1") == 1 and plain("plan", "--rehearsal-blocks", "tune=3") == 1
    assert plain("plan", "--rehearsal-blocks", "select=121") == 1
    assert plain("board") == 1 and plain("fit-map") == 1                   # nothing planned yet
    assert not state.exists()


def test_a_state_directory_is_bound_to_one_store_identity(tmp_path, capsys):
    state = tmp_path / "state"
    plain = lambda *argv: tune.main([*argv, "--state-dir", str(state)])      # noqa: E731
    assert plain("plan", "--rehearsal-blocks", REHEARSAL) == 0
    assert plain("run", "--stage", "smoke", "--backend", "keyword") == 0
    bound = read(state, "store_identity.json")
    assert bound["backend_choice"] == "keyword" and bound["real_model_backend"] is False
    assert bound["requested"]["cpu_stand_in_not_a_model"] is True
    assert "://" not in json.dumps(bound["requested"]) and str(tmp_path) not in json.dumps(bound["requested"])
    capsys.readouterr()
    assert plain("run", "--stage", "smoke", "--backend", "constant") == 1
    assert "bound to store identity" in capsys.readouterr().err
    assert sorted(p.name for p in (state / "store").iterdir()) == [
        f"{bound['identity12']}.identity.json", f"{bound['identity12']}.jsonl"]
    # The stand-in through the real command line is labelled as a rehearsal, never as a model run.
    smoke = results(state, "smoke")
    assert [r["evidence_class"] for r in smoke] == ["cpu_rehearsal_not_model_evidence"] * 2
    first = json.loads((attempts(state, "smoke")[0][0] / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["decision"]["model"]["is_model"] is False
    assert first["decision"]["model"]["label"] == "규칙 추정(모델 아님)"
    assert first["decision"]["model"]["backend"] == "composed_estimates"
    assert first["decision"]["model"]["request_sha256"] == tune.GOLDEN_REQUEST["sha256"]


def test_only_a_real_model_backend_on_the_registered_blocks_can_count_as_model_evidence(tmp_path, flow, capsys):
    """The flag is about the labels only: here the "real model" is still the test double."""
    registered = tmp_path / "registered"
    assert cli(registered, "plan")[0] == 0 and cli(registered, "run", "--stage", "smoke")[0] == 0
    assert cli(registered, "board")[0] == 0
    assert read(registered, "leaderboard.json")["counts_as_model_evidence"] is True
    assert "실제 모델 backend" in (registered / "leaderboard.md").read_text(encoding="utf-8")
    stand_in = tmp_path / "stand_in"
    assert cli(stand_in, "plan")[0] == 0 and cli(stand_in, "run", "--stage", "smoke", real=False)[0] == 0
    assert cli(stand_in, "board")[0] == 0
    assert read(stand_in, "leaderboard.json")["counts_as_model_evidence"] is False
    assert "모델 근거가 아니다" in (stand_in / "leaderboard.md").read_text(encoding="utf-8")
    assert [r["evidence_class"] for r in results(stand_in, "smoke")] == ["cpu_rehearsal_not_model_evidence"] * 2
    # Shortened blocks are a rehearsal whatever the backend is.
    assert flow.board["counts_as_model_evidence"] is False and flow.board["confirm"]["counts_as_model_evidence"] is False
    assert read(flow.state, "selected_config.json")["selected_from"]["counts_as_model_evidence"] is False
    assert "리허설의 것이며 모델 근거가 아니다" in (flow.state / "leaderboard.md").read_text(encoding="utf-8")
    copy = tmp_path / "copy"
    shutil.copytree(flow.after_all, copy)
    capsys.readouterr()
    assert cli(copy, "run", "--stage", "all")[0] == 0
    assert "counts_as_model_evidence=False" in capsys.readouterr().out


def test_real_model_identity_holds_pins_and_adapter_hash_but_no_location():
    args = SimpleNamespace(backend="zoro", dtype="fp16", max_input_tokens=2048, decision_temperature=1.0,
                           reasoning=False, model="zoro-1", original_checkpoint="/content/x",
                           exported_checkpoint="/content/y", url=None)
    requested = tune.Tuner(Path("unused"), args).requested_identity()
    pins = json.loads((tune.REPOSITORY_ROOT / "config" / "model_pins.json").read_text(encoding="utf-8"))
    assert requested["backend_kind"] == "zoro_native" and requested["adapter_file"] == "src/sessionguard/backends/zoro.py"
    assert requested["model_revision"] == pins["model_revision"]
    assert requested["source_revision"] == pins["source_revision"]
    assert requested["adapter_sha256"] == hashlib.sha256(
        (tune.REPOSITORY_ROOT / "src/sessionguard/backends/zoro.py").read_bytes()).hexdigest()
    assert requested["requests_sha256"] == hashlib.sha256(
        (tune.REPOSITORY_ROOT / "negotiation/src/negotiation_agent/requests.py").read_bytes()).hexdigest()
    assert (requested["dtype"], requested["max_input_tokens"], requested["decision_temperature"],
            requested["reasoning"]) == ("fp16", 2048, 1.0, False)
    assert "/content" not in json.dumps(requested)
    PredictionStore(Path("unused-store"), requested, mode="read-only")      # accepted as a store identity
    other = tune.Tuner(Path("unused"), SimpleNamespace(**{**vars(args), "dtype": "bf16"})).requested_identity()
    http = tune.Tuner(Path("unused"), SimpleNamespace(**{**vars(args), "backend": "zoro-http", "reasoning": True,
                                                         "url": "http://localhost:30001"})).requested_identity()
    assert http["adapter_file"] == "src/sessionguard/backends/zoro_http.py" and http["reasoning"] is True
    assert "localhost" not in json.dumps(http)
    assert len({tune.identity_digest(r) for r in (requested, other, http)}) == 3
    assert tune.Tuner(Path("unused"), args).declared == ("zoro_native", True, True)
    assert tune.Tuner(Path("unused"), args).real_model is True


# ---- one whole pipeline ------------------------------------------------------------------------------------

def test_pipeline_runs_every_stage_and_labels_each_result(flow):
    state = flow.state
    assert set(flow.splits) <= {"dev", "calibration"} and "calibration" in flow.splits
    names = sorted(p.name for p in state.iterdir())
    assert names == ["boards", "confirm_first_use.json", "leaderboard.json", "leaderboard.md", "maps",
                     "protocol.json", "runs", "selected_config.json", "selection.json", "store",
                     "store_identity.json"]
    smoke = results(state, "smoke", "M0")
    assert [r["smoke"]["pass"] for r in smoke] == [1, 2] and smoke[1]["smoke"]["verdict"] == "passed"
    assert [r["block"]["cases"] for r in smoke] == [2, 2] and smoke[0]["block"]["subset"] == "first 2 cases"
    assert smoke[0]["evidence_class"] == "real_model_run" and smoke[0]["counts"]["store_hits"] == 0
    assert smoke[1]["evidence_class"] == "real_model_replay_from_store" and smoke[1]["counts"]["live_requests"] == 0
    regress, = results(state, "regress", "M0")
    assert regress["evidence_class"] == "real_model_run_partly_replayed_from_store"
    assert regress["counts"]["store_hits"] == smoke[0]["counts"]["live_requests"] > 0 < regress["counts"]["live_requests"]
    calibrate, = results(state, "calibrate", "M0")
    assert calibrate["purpose"] == "map_fitting_only" and calibrate["block"]["split"] == "calibration"
    for result in results(state, "select") + results(state, "confirm") + [regress, calibrate] + smoke:
        counts = result["counts"]
        assert result["status"] == "complete" and result["complete"] is True
        assert result["protocol_sha256"] == read(state, "protocol.json")["protocol_sha256"]
        assert result["provenance"]["execution_config"]["tuning"]["config"]["id"] == result["config"]["id"]
        assert counts["decisions"] == sum(counts[k] for k in ("live_requests", "store_hits", "session_cache_hits",
                                                              "cpu_estimates", "no_request", "no_answer"))
        assert "model_requests" not in result["summary"]["process"]
        assert all("internal_record" not in row and "model_requests" not in row for row in result["rows"])
        if result["config"]["role"] == "reference":
            assert result["evidence_class"] == "cpu_reference_not_model_evidence" and result["store_identity"] is None
            assert counts["live_requests"] == counts["store_hits"] == 0
        else:
            assert result["evidence_class"].startswith("real_model_") and result["backend"]["kind"] == "composed_estimates"
            assert result["store_identity"]["identity12"] == read(state, "store_identity.json")["identity12"]
            assert result["store_identity"]["observed"] == {"dtype": "test_only"}
            assert result["estimate_sources"] == result["backend"]["estimate_sources"]
    # Every live request was answered once and stored once; nothing else called the double.
    live = sum(r["counts"]["live_requests"] for stage in tune.STAGES for r in results(state, stage))
    assert live == store_entries(state)
    # ... plus the one unstored runtime check before the first confirm use.
    assert flow.double.answered == flow.double.calls == live + 1
    assert read(state, "store_identity.json")["requested"] == IDENTITY
    # decisions.jsonl holds one line per decision of the attempt.
    folder, result = attempts(state, "select", "M0")[0]
    lines = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == result["counts"]["decisions"] and {line["case_id"] for line in lines} == {
        row["case_id"] for row in result["rows"]}
    assert all(line["decision"]["model"]["label"] == "모델 추정" for line in lines if line["decision"]["model"])


def test_running_every_stage_creates_no_evidence_directory(flow):
    assert EVIDENCE_DIR.exists() == flow.evidence_before
    if not flow.evidence_before:
        assert not EVIDENCE_DIR.exists()
    assert not any("evidence" in p.parts for p in flow.state.rglob("*"))


def test_module_never_names_the_held_out_split_or_its_history():
    for path in (Path(tune.__file__), tune.REPOSITORY_ROOT / "negotiation" / "run_tune.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(rf"\b{SPLIT_LITERAL}\b", node.value), (path.name, node.value[:60])
                assert "holdout" not in node.value and "evidence/" not in node.value
            if isinstance(node, ast.ImportFrom):
                assert not {alias.name for alias in node.names} & FORBIDDEN_NAMES
            if isinstance(node, ast.Name):
                assert node.id not in FORBIDDEN_NAMES
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_NAMES
    with pytest.raises(tune.TuneRefusal):                     # a block naming any other split is refused
        original = dict(tune.BLOCKS)
        try:
            tune.BLOCKS["select"] = (SPLIT_LITERAL, 0, 5)
            tune.block_cases("select")
        finally:
            tune.BLOCKS.clear()
            tune.BLOCKS.update(original)


def test_default_configuration_reproduces_the_plain_backend_and_the_default_requests(tmp_path):
    """Registered D0 block, keyword stand-in through the real command line: M0 == the plain keyword agent."""
    state = tmp_path / "state"
    plain = lambda *argv: tune.main([*argv, "--state-dir", str(state), "--backend", "keyword"])  # noqa: E731
    assert plain("plan") == 0 and plain("run", "--stage", "regress") == 0
    result, = results(state, "regress", "M0")
    private, config = default_buyer_private(), PolicyConfig()
    direct = [E.run_case(case, private, KeywordBackend(), policy.decide, config)[0] for case in E.make_cases("dev", 30)]
    assert [(r["case_id"], r["status"], r["agreement"], r["realized_saving"], r["net_saving"]) for r in result["rows"]] \
        == [(r["case_id"], r["status"], r["agreement"], r["realized_saving"], r["net_saving"]) for r in direct]
    assert result["block"]["cases"] == 30 and result["evidence_class"] == "cpu_rehearsal_not_model_evidence"
    # Two cases with the same public history ask the same request: the second one is a store hit, so the
    # session count of requests is live requests plus hits, and only the live ones are stored.
    counts = result["counts"]
    assert counts["live_requests"] + counts["store_hits"] == sum(r["model_requests"] for r in direct)
    store = PredictionStore(state / "store", read(state, "store_identity.json")["requested"], mode="read-only")
    assert tune.GOLDEN_REQUEST["sha256"] in store and len(store) == counts["live_requests"]
    # The comparison with the recorded real-model run is reported, and marked as not meaningful for a stand-in.
    regression = result["regression"]
    reference = {c["case_id"]: c["agent"] for c in json.loads(
        (tune.REPOSITORY_ROOT / tune.REGRESSION_REFERENCE).read_text(encoding="utf-8"))["case_outcomes"]}
    differing = [r["case_id"] for r in direct if (reference[r["case_id"]]["status"], reference[r["case_id"]][
        "realized_saving"]) != (r["status"], r["realized_saving"])]
    assert regression["cases_compared"] == 30 and regression["meaningful"] is False
    assert [d["case_id"] for d in regression["differing"]] == differing and regression["cases_differing"] == len(differing)
    assert regression["reference_backend"]["kind"] == "zoro_native" and "stand-in" in regression["note"]


def test_regression_is_meaningful_only_for_a_real_model_identity(flow):
    regress, = results(flow.state, "regress", "M0")
    assert regress["regression"]["meaningful"] is True and regress["regression"]["cases_compared"] == 3
    assert "environment drift" in regress["regression"]["note"]
    assert regress["purpose"] == "regression_on_seen_cases_only"
    assert not any(row["id"].startswith("M0") and row["cases"] == 3 for row in flow.board["blocks"]["select"]["rows"])


def test_reference_rows_equal_the_plain_cpu_policies(flow):
    cases = tune.block_cases("select", {"select": 6})
    private, config = default_buyer_private(), PolicyConfig()
    by_name = complete_rows(flow.state, "select")
    expect = lambda backend, fn=policy.decide: [  # noqa: E731
        (r["case_id"], r["status"], r["realized_saving"], r["net_saving"])
        for r in (E.run_case(case, private, backend, fn, config)[0] for case in cases)]
    assert by_name["rule_policy"] == expect(HeuristicBackend())
    assert by_name["fixed_concession"] == expect(None, E.BASELINES["fixed_concession"])
    assert by_name["midpoint"] == expect(None, E.BASELINES["midpoint"])
    assert by_name["constant_keyword"] == expect(ComposedBackend(None, EstimateSpec("constant", 0.15, None, "keyword")))
    assert by_name["reference_true_acceptance"] == [
        (r["case_id"], r["status"], r["realized_saving"], r["net_saving"])
        for r in (E.run_case(case, private, E.TrueAcceptanceReference(case.hidden), policy.decide, config)[0]
                  for case in cases)]


# ---- interruption, budget, failure -------------------------------------------------------------------------

def test_interrupted_run_continues_from_the_store_and_ends_like_an_uninterrupted_one(tmp_path, flow, capsys):
    state = planned(tmp_path)
    code, first = cli(state, "run", "--stage", "all", double=ModelDouble(interrupt_at=12))
    assert code == 130 and first.calls == 12 and first.answered == 11
    assert "status=interrupted" in capsys.readouterr().out
    stopped = [r for stage in tune.STAGES for r in results(state, stage) if r["status"] == "interrupted"]
    assert len(stopped) == 1 and stopped[0]["complete"] is False
    assert not (state / "selection.json").exists() and store_entries(state) == 11
    # The interrupted request reached the backend, so it counts as live; it was not stored or filled in.
    assert stopped[0]["counts"]["store_backend"]["stored_answers"] == stopped[0]["counts"]["live_requests"] > 0
    assert stopped[0]["counts"]["live_failed_requests"] == 1
    assert stopped[0]["cases_run"] < stopped[0]["cases_planned"] and stopped[0]["needs_review_cases"]
    interrupted_bytes = {f: (f / "result.json").read_bytes() for stage in tune.STAGES for f, _ in attempts(state, stage)}
    code, second = cli(state, "run", "--stage", "all")
    assert code == 0
    # Every unique request reached the backend exactly once in total, the interrupted one included only once.
    assert first.answered + second.answered == store_entries(state) == flow.answered_after_all
    assert second.calls == second.answered
    assert complete_rows(state, "select") == complete_rows(flow.after_all, "select")
    assert complete_rows(state, "calibrate") == complete_rows(flow.after_all, "calibrate")
    assert read(state, "selection.json")["selected"]["name"] == read(flow.after_all, "selection.json")["selected"]["name"]
    # The interrupted attempt is still there, unchanged, and the rerun says what it continues.
    for folder, content in interrupted_bytes.items():
        assert (folder / "result.json").read_bytes() == content
    stage = stopped[0]["stage"]
    again = results(state, stage, "M0")
    assert [r["status"] for r in again][-2:] == ["interrupted", "complete"]
    assert again[-1]["continues_after"] == again[-2]["attempt"]
    assert again[-1]["counts"]["store_hits"] >= stopped[0]["counts"]["store_backend"]["stored_answers"]


def test_budget_stop_is_partial_and_blocks_selection_until_a_rerun(tmp_path, flow, capsys):
    state = planned(tmp_path)
    code, first = cli(state, "run", "--stage", "all", "--max-live-requests", "20")
    assert code == 0 and "status=partial_budget" in capsys.readouterr().out
    partial = [r for stage in tune.STAGES for r in results(state, stage) if r["status"] == "partial_budget"]
    assert len(partial) == 1 and partial[0]["complete"] is False and partial[0]["cases_run"] < partial[0]["cases_planned"]
    assert 20 <= first.answered <= 24                      # stops before the next case, not inside one
    assert not (state / "selection.json").exists() and not (state / "selected_config.json").exists()
    board = read(state, "leaderboard.json")                 # written for the operator; it selects nothing
    assert board["selection"]["status"] == "blocked" and board["blocks"]["select"]["candidates_terminal"] is False
    assert {b["candidate"] for b in board["selection"]["blocking"]} == {"M0", "M1", "M2", "M3", "F1", "F2", "M4", "M5"}
    code, second = cli(state, "run", "--stage", "all")
    assert code == 0 and first.answered + second.answered == flow.answered_after_all
    assert complete_rows(state, "select") == complete_rows(flow.after_all, "select")
    assert read(state, "selection.json")["selected"]["name"] == read(flow.after_all, "selection.json")["selected"]["name"]
    # A wall-clock budget works the same way: here every clock reading is 40 s later than the one before.
    ticks = iter(range(0, 10**6, 40))
    timed = planned(tmp_path, "timed")
    code, _ = cli(timed, "run", "--stage", "all", "--budget-minutes", "1", clock=lambda: next(ticks))
    assert code == 0 and "status=partial_budget" in capsys.readouterr().out
    assert not (timed / "selection.json").exists()


def test_failed_estimate_is_terminal_until_a_person_asks_for_a_retry(tmp_path, capsys):
    probe = planned(tmp_path, "probe")
    _, counter = cli(probe, "run", "--stage", "smoke")
    cli(probe, "run", "--stage", "regress", double=counter)
    state = planned(tmp_path)
    code, bad = cli(state, "run", "--stage", "all", double=ModelDouble(fail_at={counter.calls + 3}))
    assert code == 1 and "status=failed" in capsys.readouterr().out
    (folder, failed), = attempts(state, "calibrate", "M0")
    assert (failed["status"], failed["reason"], failed["complete"]) == ("failed", "estimate_failed", False)
    assert len(failed["needs_review_cases"]) == 1 and failed["counts"]["failed_estimates"] == 1
    # The attempt stops at the failed estimate: no later case may ask the same request again on its own.
    assert failed["cases_run"] < failed["cases_planned"]
    assert failed["needs_review_cases"] == [failed["rows"][-1]["case_id"]]
    assert failed["first_errors"][0]["error"].startswith("RuntimeError: test backend failure")
    assert failed["errors_by_type"] == {"RuntimeError": 1} and failed["counts"]["live_failed_requests"] == 1
    assert store_entries(state) == bad.answered == bad.calls - 1   # the failed answer was not stored or filled in
    failed_bytes = (folder / "result.json").read_bytes()
    lines = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    broken, = [line for line in lines if "error" in line["decision"]["model"]]
    assert broken["decision"]["action"] == "human_review" and "predictions" not in broken["decision"]["model"]
    block = dict(broken["decision"]["model"]["response"]["prediction_store"])
    assert isinstance(block.pop("latency_ms"), float)
    assert block == {"hit": False, "identity": read(state, "store_identity.json")["identity12"], "stored": False}
    # No map without the complete calibrate result, and no select stage after it.
    assert cli(state, "fit-map")[0] == 1 and "its state is failed" in capsys.readouterr().err
    assert not (state / "maps").exists() and not (state / "runs" / "select").exists()
    # Running again does not ask the backend again on its own.
    code, idle = cli(state, "run", "--stage", "all")
    assert code == 1 and idle.calls == 0 and len(attempts(state, "calibrate", "M0")) == 1
    assert "not rerun without --retry-failed" in capsys.readouterr().err
    # A person's decision: the retry is a new attempt that names the failed one.
    code, retry = cli(state, "run", "--stage", "calibrate", "--retry-failed")
    assert code == 0 and retry.calls >= 1          # the request that had failed, then the cases after it
    first, second = attempts(state, "calibrate", "M0")
    assert first[0] == folder and (folder / "result.json").read_bytes() == failed_bytes
    assert second[1]["status"] == "complete" and second[1]["retry_of"] == failed["attempt"]
    assert second[1]["retry_failed_flag"] is True
    assert cli(state, "run", "--stage", "all")[0] == 0 and (state / "selection.json").exists()
    # --retry-failed does not rerun what is already complete.
    before = len(list((state / "runs").glob("*/*/attempt-*")))
    code, idle = cli(state, "run", "--stage", "all", "--retry-failed")
    assert code == 0 and idle.calls == 0 and len(list((state / "runs").glob("*/*/attempt-*"))) == before


def test_failed_candidate_waits_for_a_person_before_a_selection_is_written(tmp_path, capsys):
    probe = planned(tmp_path, "probe")
    counter = ModelDouble()
    for stage in ("smoke", "regress", "calibrate"):
        assert cli(probe, "run", "--stage", stage, double=counter)[0] == 0
    state = planned(tmp_path)
    code, bad = cli(state, "run", "--stage", "all", double=ModelDouble(fail_at={counter.calls + 2}))
    assert code == 1 and "status=failed" in capsys.readouterr().out
    (_, m0), = attempts(state, "select", "M0")
    assert (m0["status"], m0["reason"]) == ("failed", "estimate_failed") and len(m0["needs_review_cases"]) == 1
    # The stage stopped at the failed estimate: no other candidate asked that request on its own.
    assert [r["config"]["name"] for r in results(state, "select")] == ["M0"] and m0["cases_run"] < m0["cases_planned"]
    assert read(state, "leaderboard.json")["selection"]["status"] == "blocked"
    # A person runs the stage again without a retry: the others run, M0 stays failed.
    code, again = cli(state, "run", "--stage", "all")
    assert code == 1 and again.calls > 0 and len(attempts(state, "select", "M0")) == 1
    # Every other candidate and reference ran; `all` wrote the leaderboard but did not select on its own.
    assert all(r["complete"] for r in results(state, "select") if r["config"]["name"] != "M0")
    board = read(state, "leaderboard.json")
    assert board["blocks"]["select"]["candidates_terminal"] is True and board["selection"]["status"] == "blocked"
    assert "waits for a person" in board["selection"]["reason"]
    assert board["selection"]["blocking"] == [{"candidate": "M0", "state": "failed", "reason": "estimate_failed"}]
    assert not (state / "selection.json").exists()
    row = next(r for r in board["blocks"]["select"]["rows"] if r["name"] == "M0")
    assert row["mean_realized_saving"] is None and row["needs_review"] == 1 and row["selectable"] is False
    assert all(d["minus_M0"]["realized_saving"] is None and d["minus_M0"]["reason"]
               for d in board["blocks"]["select"]["differences"].values())
    code, idle = cli(state, "run", "--stage", "all")               # running again changes nothing
    assert code == 1 and idle.calls == 0 and not (state / "selection.json").exists()
    assert cli(state, "run", "--stage", "confirm")[0] == 1 and not (state / "confirm_first_use.json").exists()
    # Path 1, a person accepts the failure: an explicit board selects among the other candidates.
    accepted = tmp_path / "accepted"
    shutil.copytree(state, accepted)
    assert cli(accepted, "board")[0] == 0
    selection = read(accepted, "selection.json")
    listed = {c["name"]: c for c in selection["candidates"]}
    assert (listed["M0"]["status"], listed["M0"]["reason"], listed["M0"]["eligible"]) == ("failed", "estimate_failed", False)
    assert selection["selected"]["name"] != "M0" and listed[selection["selected"]["name"]]["eligible"] is True
    # A later retry of M0 is shown on the board but does not change the recorded selection.
    selection_bytes = (accepted / "selection.json").read_bytes()
    assert cli(accepted, "run", "--stage", "select", "--retry-failed")[0] == 0 and cli(accepted, "board")[0] == 0
    assert (accepted / "selection.json").read_bytes() == selection_bytes
    assert next(r for r in read(accepted, "leaderboard.json")["blocks"]["select"]["rows"] if r["name"] == "M0")[
        "status"] == "complete"
    # Path 2, a person asks for the retry first: then M0 is a candidate like the others.
    code, retry = cli(state, "run", "--stage", "all", "--retry-failed")
    assert code == 0 and retry.calls == 0          # another candidate had asked the same request in the meantime
    retried = attempts(state, "select", "M0")[-1][1]
    assert retried["complete"] and retried["retry_of"] == m0["attempt"] and retried["counts"]["live_requests"] == 0
    assert {c["name"]: c["eligible"] for c in read(state, "selection.json")["candidates"]}["M0"] is True


def test_attempt_left_unfinished_by_a_killed_process_is_continued_and_kept(tmp_path):
    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "smoke")[0] == 0
    m0 = results(state, "smoke", "M0")[0]["config"]["id"]
    leftover = state / "runs" / "regress" / m0 / "attempt-20261118T000000.000000Z-dead"
    leftover.mkdir(parents=True)
    killed = json.dumps({"status": "initializing", "complete": False, "attempt": leftover.name})
    (leftover / "result.json").write_text(killed, encoding="utf-8")
    (leftover / "decisions.jsonl").write_text('{"case_id": "dev-1100", "decision": {"mod', encoding="utf-8")
    assert cli(state, "board")[0] == 0                                         # an unfinished attempt is readable
    assert read(state, "leaderboard.json")["other_stages"]["regress"]["status"] == "incomplete"
    assert cli(state, "run", "--stage", "regress")[0] == 0
    first, second = attempts(state, "regress", "M0")
    assert first[0] == leftover and (leftover / "result.json").read_text(encoding="utf-8") == killed
    assert second[1]["status"] == "complete" and second[1]["continues_after"] == leftover.name


def test_backend_that_cannot_be_built_is_not_a_terminal_failure(tmp_path, capsys):
    state = planned(tmp_path)

    def no_weights():
        raise RuntimeError("no weights in this environment")

    assert cli(state, "run", "--stage", "all", factory=no_weights)[0] == 1
    assert "status=initialization_failed" in capsys.readouterr().out
    (_, result), = attempts(state, "smoke", "M0")
    assert result["status"] == "initialization_failed" and result["cases_run"] == 1 and result["complete"] is False
    assert result["evidence_class"] == "real_model_requested_no_model_answer" and result["counts"]["no_answer"] == 1
    assert not (state / "store" / f"{read(state, 'store_identity.json')['identity12']}.jsonl").exists()

    def refused():
        raise SystemExit("--url is required for --backend zoro-http")

    assert cli(state, "run", "--stage", "smoke", factory=refused)[0] == 1     # recorded, not a process exit
    assert results(state, "smoke", "M0")[-1]["status"] == "initialization_failed"
    assert cli(state, "run", "--stage", "smoke")[0] == 0                       # no --retry-failed needed
    assert [r["status"] for r in results(state, "smoke", "M0")] == ["initialization_failed"] * 2 + ["complete"] * 2


def test_answers_of_another_runtime_stop_the_stage_instead_of_being_mixed(tmp_path, capsys):
    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "smoke", double=ModelDouble(dtype="fp16"))[0] == 0
    entries = store_entries(state)
    code, other = cli(state, "run", "--stage", "regress", double=ModelDouble(dtype="bf16"))
    assert code == 1 and "status=store_identity_mismatch" in capsys.readouterr().out
    assert other.calls == 1 and store_entries(state) == entries              # stopped at the first live answer
    (_, result), = attempts(state, "regress", "M0")
    assert result["status"] == "store_identity_mismatch" and result["complete"] is False


def test_smoke_fails_when_the_second_pass_reaches_the_backend(tmp_path, capsys):
    class Forgetful(PredictionStore):
        def get(self, request_sha256):
            return None                       # a store that cannot replay

    state = planned(tmp_path)
    original = tune.PredictionStore
    tune.PredictionStore = Forgetful
    try:
        code, double = cli(state, "run", "--stage", "all")
    finally:
        tune.PredictionStore = original
    assert code == 1 and "status=smoke_failed" in capsys.readouterr().out
    first, second = results(state, "smoke", "M0")
    assert second["smoke"]["verdict"] == "failed" and "live requests" in second["smoke"]["reasons"][0]
    assert not (state / "runs" / "regress").exists()


def test_read_only_store_replays_without_building_and_a_miss_blocks_selection(tmp_path, flow, capsys):
    def never():
        raise AssertionError("a read-only replay must not build the backend")

    replay = tmp_path / "replay"
    shutil.copytree(flow.state, replay)
    before = len(list((replay / "runs").glob("*/*/attempt-*")))
    assert cli(replay, "run", "--stage", "all", "--store-mode", "read-only", factory=never)[0] == 0
    assert len(list((replay / "runs").glob("*/*/attempt-*"))) == before      # everything was already complete

    # A fresh state replays nothing: every request is a miss, nothing is appended, nothing is built.
    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "smoke", "--store-mode", "read-only", factory=never)[0] == 1
    (_, missed), = attempts(state, "smoke", "M0")
    assert (missed["status"], missed["reason"]) == ("failed", "estimate_failed")
    assert missed["counts"]["read_only_store_misses"] == missed["counts"]["failed_estimates"] == 1
    assert missed["counts"]["live_requests"] == 0 and not (state / "store").exists()
    assert cli(state, "run", "--stage", "smoke", "--store-mode", "read-only", factory=never)[0] == 1
    assert len(attempts(state, "smoke", "M0")) == 1                           # not asked again on its own

    # SELECT replayed from a store that lacks its answers: the candidates fail, and that blocks selection.
    partial = planned(tmp_path, "partial")
    for stage in ("smoke", "regress", "calibrate"):
        assert cli(partial, "run", "--stage", stage)[0] == 0
    assert cli(partial, "fit-map")[0] == 0
    assert cli(partial, "run", "--stage", "select", "--store-mode", "read-only", factory=never)[0] == 1
    assert cli(partial, "board")[0] == 0
    board = read(partial, "leaderboard.json")
    assert board["selection"]["status"] == "blocked" and not (partial / "selection.json").exists()
    # The stage stops at the first candidate whose answer is missing; the others were not run.
    assert "read_only_store_miss" in {b["reason"] for b in board["selection"]["blocking"]}


# ---- calibration split and map fitting -----------------------------------------------------------------------

def test_calibration_result_is_for_map_fitting_only_and_never_on_the_leaderboard(tmp_path, flow, capsys):
    state = planned(tmp_path)
    cli(state, "run", "--stage", "smoke")
    capsys.readouterr()
    assert cli(state, "run", "--stage", "calibrate")[0] == 0
    captured = capsys.readouterr()
    progress = [line for line in captured.err.splitlines() if line.startswith("calibrate ")]
    assert progress and not any("saving" in line for line in progress + captured.out.splitlines())
    calibrate, = results(flow.state, "calibrate", "M0")
    assert "mean_realized_saving" not in calibrate["summary"] and "feasible_cases" not in calibrate["summary"]
    assert "acceptance_estimates" in calibrate["summary"] and calibrate["acceptance_predictions"]
    assert all({"p_model_raw", "round_prior", "truth", "composed_source", "value_from"} <= set(r)
               for r in calibrate["final_offer_records"])
    for board in (flow.board, flow.after_all_board):
        text = json.dumps(board, ensure_ascii=False)
        assert "calibration-20" not in text
        assert set(board["blocks"]) <= {"select", "confirm"}
        assert set(board["other_stages"]["calibrate"]) == {"status", "purpose", "note"}
        assert all(row["attempt"] is None or "/calibrate/" not in row["attempt"]
                   for table in board["blocks"].values() for row in table["rows"])
    markdown = (flow.state / "leaderboard.md").read_text(encoding="utf-8")
    assert "calibration-20" not in markdown and "맵 적합 전용" in markdown


def test_fit_map_accepts_only_the_complete_calibrate_result_of_the_default(tmp_path, flow, capsys):
    calibrate, = results(flow.state, "calibrate", "M0")
    maps = tune.fit_maps(calibrate, "f" * 64)
    assert maps["fitted_on_split"] == "calibration" and maps["platt"]["fitted_on_split"] == "calibration"
    # The Platt entry is exactly what evaluation.fit_calibration returns for the same records.
    shim = {"experiment": "full_negotiation_evaluation", "split": "calibration", "backend": calibrate["backend"],
            "evaluator_only": {"acceptance_predictions": {"agent": calibrate["acceptance_predictions"]}}}
    assert maps["platt"] == E.fit_calibration(shim, source="f" * 64)
    assert set(maps["platt"]) == {"method", "a", "b", "n", "fitted_on_split", "source", "backend_kind",
                                  "brier_before", "brier_after_in_sample"}
    temperature = maps["temperature_equivalent"]
    assert temperature["temperature"] == pytest.approx(1 / temperature["a"]) and "never applied" in temperature["note"]
    select, = results(flow.state, "select", "M0")
    regress, = results(flow.state, "regress", "M0")
    other, = results(flow.state, "select", "M1")
    for wrong, text in ((select, "not a calibrate-stage result"), (regress, "not a calibrate-stage result"),
                        ({**calibrate, "status": "failed", "complete": False}, "not complete"),
                        ({**calibrate, "status": "partial_budget", "complete": False}, "not complete"),
                        ({**calibrate, "config": other["config"]}, "not the default M0"),
                        ({**calibrate, "block": select["block"]}, "not run on the calibration split"),
                        ({"experiment": "full_negotiation_evaluation", "split": "calibration"}, "not a tuning result")):
        with pytest.raises(ValueError, match=text):
            tune.fit_maps(wrong, "f" * 64)
    # One fit per state directory, and none before the calibrate result exists.
    maps_file, = (flow.state / "maps").iterdir()
    assert maps_file.name == f"calibration-{read(flow.state, 'store_identity.json')['identity12']}.json"
    content = maps_file.read_bytes()
    written = json.loads(content)
    assert written["source"]["purpose"] == "map_fitting_only" and written["source"]["config_id"].startswith("M0-")
    assert written["platt"]["source"] == written["source"]["result_sha256"] == hashlib.sha256(
        (flow.state / written["source"]["result_file"]).read_bytes()).hexdigest()
    copy = tmp_path / "copy"
    shutil.copytree(flow.state, copy)
    assert cli(copy, "fit-map")[0] == 0 and "already fitted" in capsys.readouterr().out
    assert (copy / "maps" / maps_file.name).read_bytes() == content
    state = planned(tmp_path)
    assert cli(state, "fit-map")[0] == 1
    cli(state, "run", "--stage", "smoke")
    assert cli(state, "fit-map")[0] == 1 and "its state is not_run" in capsys.readouterr().err
    assert not (state / "maps").exists()


def test_shrink_weight_is_zero_for_noise_and_one_for_the_truth():
    truths = [True, False, False, True, False, False, False, True, False, False, False, False]
    record = lambda p, truth, prior=0.45: {"p_model_raw": p, "round_prior": prior, "truth": truth}  # noqa: E731
    # Noise: symmetric around the prior and unrelated to the truth. Any weight only adds variance.
    noise = [record(0.05 if i % 2 else 0.85, truth) for truth in truths for i in range(2)]
    fitted = tune.fit_final_offer_shrink(noise)
    assert fitted["weight"] == 0.0 and fitted["brier_shrunk"] == fitted["brier_prior"] < fitted["brier_model"]
    assert [row["weight"] for row in fitted["grid"]] == [i / 10 for i in range(11)]
    assert fitted["n"] == 24 and fitted["positives"] == 6
    # The truth itself: full weight on the model value.
    exact = tune.fit_final_offer_shrink([record(1.0 if truth else 0.0, truth) for truth in truths])
    assert exact["weight"] == 1.0 and exact["brier_shrunk"] == exact["brier_model"] == 0.0 < exact["brier_prior"]
    # A model value equal to the prior makes every weight equal: the tie goes to the smaller weight.
    tie = tune.fit_final_offer_shrink([record(0.45, truth) for truth in truths])
    assert tie["weight"] == 0.0 and len({row["brier"] for row in tie["grid"]}) == 1
    # In between, the grid value with the lowest Brier score is chosen, computed with the run-time formula.
    mixed = [record(0.9 if truth else (0.6 if i % 3 else 0.1), truth, 0.15) for i, truth in enumerate(truths)]
    middle = tune.fit_final_offer_shrink(mixed)
    scores = {w: tune.M.brier([tune.shrink(r["p_model_raw"], r["round_prior"], w) for r in mixed], truths)
              for w in tune.SHRINK_GRID}
    assert middle["weight"] == min(scores, key=lambda w: (scores[w], w)) and 0.0 < middle["weight"] < 1.0
    # Too few positives or negatives: no weight, with the reason.
    for records in ([record(0.5, False)] * 10 + [record(0.9, True)], [record(0.5, True)] * 10 + [record(0.1, False)],
                    []):
        unavailable = tune.fit_final_offer_shrink(records)
        assert unavailable["weight"] is None and "fewer than 2" in unavailable["unavailable_reason"]


def test_arms_resolve_from_the_maps_and_record_terminal_states():
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # noqa: E731
    pending = {s.arm.name: s for s in tune.resolve_arms(None)}
    assert {n: s.status for n, s in pending.items()} == {
        "M0": "resolved", "M1": "resolved", "M2": "resolved", "M3": "pending", "F1": "pending", "F2": "pending",
        "M4": "pending", "M5": "pending"}
    assert pending["M3"].config_id == "M3-pending" and pending["M5"].config_id == "M5-pending"
    m0 = pending["M0"].config
    assert m0.id == "M0-" + hashlib.sha256(canonical(
        {"policy": tune.policy_json(), "estimates": EstimateSpec().to_json()}).encode()).hexdigest()[:12]
    assert tune.policy_json()["final_prior_by_round"] == [0.0, 0.15, 0.45, 0.5]
    platt = {"method": "platt", "a": 1.6, "b": -0.3, "n": 40, "fitted_on_split": "calibration", "source": "0" * 64}
    full = {s.arm.name: s for s in tune.resolve_arms({"platt": platt, "final_offer_shrink": {"weight": 0.3}})}
    assert all(s.status == "resolved" for s in full.values())
    assert full["M3"].config.spec == EstimateSpec(final_source="shrink", final_shrink_weight=0.3)
    assert full["F1"].config.spec == EstimateSpec("constant", 0.15, None, "shrink", 0.3)
    assert full["F2"].config.spec == EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.3)
    assert full["M4"].config.spec == EstimateSpec(acceptance_platt=platt)
    assert full["M5"].config.spec == EstimateSpec(acceptance_platt=platt, final_source="keyword")
    assert len({s.config_id for s in full.values()}) == 8
    assert all(re.fullmatch(rf"{name}-[0-9a-f]{{12}}", s.config_id) for name, s in full.items())
    other = {s.arm.name: s for s in tune.resolve_arms({"platt": {**platt, "a": 1.7},
                                                       "final_offer_shrink": {"weight": 0.3}})}
    assert other["M4"].config_id != full["M4"].config_id and other["M3"].config_id == full["M3"].config_id
    zero = {s.arm.name: s for s in tune.resolve_arms({"platt": None, "final_offer_shrink": {"weight": 0.0}})}
    assert (zero["M3"].status, zero["F1"].status, zero["F2"].status) == (
        "skipped_identical_to_M1", "skipped_identical_to_twin", "skipped_identical_to_twin")
    assert (zero["M4"].status, zero["M4"].reason, zero["M4"].config_id) == ("failed", "map_unavailable",
                                                                           "M4-map_unavailable")
    assert all(s.terminal_without_run for name, s in zero.items() if name in ("M3", "F1", "F2", "M4", "M5"))
    missing = {s.arm.name: s for s in tune.resolve_arms({"platt": platt, "final_offer_shrink": {"weight": None}})}
    assert all(missing[name].status == "failed" and missing[name].reason == "map_unavailable"
               for name in ("M3", "F1", "F2"))
    # A map fitted anywhere else is refused by the spec itself.
    with pytest.raises(ValueError):
        tune.resolve_arms({"platt": {**platt, "fitted_on_split": "dev"}, "final_offer_shrink": {"weight": 0.3}})


def test_zero_weight_arms_really_are_identical_to_what_they_are_skipped_for():
    cases = E.make_cases("dev", 12)
    private, config, table = default_buyer_private(), PolicyConfig(), PolicyConfig().final_prior_by_round
    noisy = lambda: ModelDouble(final=lambda state: 0.9 if state.current_round % 2 else 0.2)  # noqa: E731

    def outcome(spec):
        backend = ComposedBackend(noisy() if spec.uses_model() else None, spec, table)
        return [(r["status"], r["agreement"], r["realized_saving"], r["rounds_used"])
                for r in (E.run_case(case, private, backend, policy.decide, config)[0] for case in cases)]

    zero = {s.arm.name: s.config.spec for s in tune.resolve_arms({"platt": None, "final_offer_shrink": {"weight": 0.0}})
            if s.config is not None}
    assert outcome(zero["M3"]) == outcome(EstimateSpec(final_source="policy_prior"))            # M1
    assert outcome(zero["F1"]) == outcome(zero["F1"].twin())
    assert outcome(zero["F2"]) == outcome(zero["F2"].twin())
    assert outcome(zero["M3"]) != outcome(EstimateSpec())            # the noisy final value does matter otherwise


def test_zero_shrink_weight_is_recorded_as_skipped_arms_and_selection_still_happens(flow):
    """In the shared pipeline the double's final-offer value never differs from the round prior on CAL."""
    maps_file, = (flow.state / "maps").iterdir()
    maps = json.loads(maps_file.read_text(encoding="utf-8"))
    shrink_fit = maps["final_offer_shrink"]
    assert shrink_fit["weight"] == 0.0 and shrink_fit["positives"] >= 2
    assert len({row["brier"] for row in shrink_fit["grid"]}) == 1            # a tie over the whole grid
    assert {n: (maps["arms"][n]["status"], maps["arms"][n]["reason"]) for n in ("M3", "F1", "F2")} == {
        "M3": ("skipped_identical_to_M1", "final_offer_shrink_weight_is_zero"),
        "F1": ("skipped_identical_to_twin", "final_offer_shrink_weight_is_zero"),
        "F2": ("skipped_identical_to_twin", "final_offer_shrink_weight_is_zero")}
    assert maps["arms"]["M4"]["status"] == maps["arms"]["M5"]["status"] == "resolved"
    for name in ("M3", "F1", "F2"):
        assert not results(flow.state, "select", name) and not results(flow.state, "confirm", name)
    rows = {row["name"]: row for row in flow.after_all_board["blocks"]["select"]["rows"] if row["role"] == "candidate"}
    assert rows["M3"]["status"] == "skipped_identical_to_M1" and rows["M3"]["mean_realized_saving"] is None
    assert rows["F1"]["selectable"] is False and rows["M4"]["status"] == "complete"
    assert flow.after_all_board["blocks"]["select"]["candidates_terminal"] is True
    assert flow.after_all_board["selection"]["status"] == "selected"
    selection = read(flow.after_all, "selection.json")
    assert {c["name"]: c["status"] for c in selection["candidates"]}["F2"] == "skipped_identical_to_twin"
    assert selection["maps_sha256"] == hashlib.sha256(maps_file.read_bytes()).hexdigest()


def test_shrink_arms_run_with_the_fitted_weight_and_are_labelled_as_hybrids(tmp_path):
    """A double whose final-offer value is a fifth of the round prior: the fitted weight is inside (0, 1)."""
    table = PolicyConfig().final_prior_by_round
    state = planned(tmp_path)
    double = ModelDouble(final=lambda s: 0.2 * round_prior(s, table))
    assert cli(state, "run", "--stage", "all", double=double)[0] == 0
    maps = json.loads(next((state / "maps").iterdir()).read_text(encoding="utf-8"))
    weight = maps["final_offer_shrink"]["weight"]
    assert 0.0 < weight < 1.0 and all(arm["status"] == "resolved" for arm in maps["arms"].values())
    assert maps["final_offer_shrink"]["brier_shrunk"] < min(maps["final_offer_shrink"]["brier_model"],
                                                           maps["final_offer_shrink"]["brier_prior"])
    for name, sources in (("M3", {"acceptance": "model", "final_offer": "shrink"}),
                          ("F1", {"acceptance": "constant", "final_offer": "shrink"}),
                          ("F2", {"acceptance": "constant", "final_offer": "keyword_else_shrink"})):
        result, = results(state, "select", name)
        assert result["complete"] and result["estimate_sources"] == sources
        assert result["config"]["estimates"]["final_shrink_weight"] == weight
        assert result["config"]["id"] == maps["arms"][name]["config_id"]
        # The model is still asked (for the final-offer value), so this is a model row, labelled as a hybrid.
        assert result["evidence_class"].startswith("real_model_") and result["store_identity"] is not None
        assert result["backend"]["model_estimates_used"] == (["acceptance", "final_offer"] if name == "M3"
                                                             else ["final_offer"])
        shrunk = [r for r in result["final_offer_records"] if r["value_from"] == "shrink"]
        assert shrunk and all(r["composed_source"] == sources["final_offer"] and r["policy_source"] == "backend_estimate"
                              and r["p_used"] == tune.shrink(r["p_model_raw"], r["round_prior"], weight)
                              and r["p_model_raw"] == 0.2 * r["round_prior"] for r in shrunk)
    board = read(state, "leaderboard.json")
    assert board["selection"]["status"] == "selected" and all(
        row["status"] == "complete" for row in board["blocks"]["select"]["rows"])
    # M0's stored answers serve the constant-acceptance arms too: the request is the same.
    f1, = results(state, "select", "F1")
    assert f1["counts"]["store_hits"] > 0


def test_arms_whose_map_cannot_be_fitted_fail_as_map_unavailable_and_do_not_block_selection(tmp_path):
    state = planned(tmp_path, rehearsal="d0=2,select=3,confirm=3,cal=1")      # one CAL case: no final-offer positives
    assert cli(state, "run", "--stage", "all")[0] == 0
    maps = json.loads(next((state / "maps").iterdir()).read_text(encoding="utf-8"))
    assert maps["final_offer_shrink"]["weight"] is None and "fewer than 2" in maps["final_offer_shrink"]["unavailable_reason"]
    assert {n: (maps["arms"][n]["status"], maps["arms"][n]["reason"]) for n in ("M3", "F1", "F2")} == {
        n: ("failed", "map_unavailable") for n in ("M3", "F1", "F2")}
    board = read(state, "leaderboard.json")
    rows = {row["name"]: row for row in board["blocks"]["select"]["rows"] if row["role"] == "candidate"}
    assert rows["M3"]["id"] == "M3-map_unavailable" and rows["M3"]["status"] == "failed"
    assert not results(state, "select", "M3") and board["selection"]["status"] == "selected"
    selected = {c["name"]: c for c in read(state, "selection.json")["candidates"]}
    assert selected["F1"]["eligible"] is False and selected["F1"]["reason"] == "map_unavailable"
    # No Platt map without both outcomes among the acceptance records: those arms fail the same way.
    calibrate, = results(state, "calibrate", "M0")
    one_sided = {**calibrate, "acceptance_predictions": [{**r, "truth": False} for r in calibrate["acceptance_predictions"]]}
    fitted = tune.fit_maps(one_sided, "0" * 64)
    assert fitted["platt"] is None and fitted["temperature_equivalent"] is None
    assert "no usable acceptance records" in fitted["platt_unavailable_reason"]
    arms = {s.arm.name: s for s in tune.resolve_arms(fitted)}
    assert all((arms[n].status, arms[n].reason) == ("failed", "map_unavailable") for n in ("M4", "M5"))


def test_result_is_written_as_initializing_before_the_backend_is_built(tmp_path):
    state = planned(tmp_path)
    seen = []

    def factory():
        (folder, result), = attempts(state, "smoke", "M0")
        seen.append((result["status"], result["complete"], sorted(result["provenance"]), folder.name,
                     (folder / "decisions.jsonl").exists()))
        return ModelDouble()

    assert cli(state, "run", "--stage", "smoke", factory=factory)[0] == 0
    (status, complete, keys, name, journal), = seen             # built once, lazily, for both passes
    assert (status, complete, journal) == ("initializing", False, True)
    assert {"git_commit", "input_sha256", "model_pins", "execution_config"} <= set(keys)
    assert re.fullmatch(r"attempt-\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{4}", name)
    first, second = results(state, "smoke", "M0")
    assert first["attempt"] == name and first["status"] == "complete"
    assert first["provenance"]["input_sha256"]["negotiation/src/negotiation_agent/tune.py"] == hashlib.sha256(
        Path(tune.__file__).read_bytes()).hexdigest()
    assert "negotiation/run_tune.py" in first["provenance"]["input_sha256"]


# ---- leaderboard, selection, verdict -------------------------------------------------------------------------

def test_leaderboard_rows_and_differences_come_from_the_complete_attempts(flow):
    for stage, board in (("select", flow.board), ("confirm", flow.board)):
        table = board["blocks"][stage]
        rows = {row["id"]: row for row in table["rows"]}
        by_id = {r["config"]["id"]: r for r in results(flow.state, stage) if r["complete"]}
        m0 = next(i for i in rows if i.startswith("M0-"))
        rule = next(i for i in rows if i.startswith("rule_policy-"))
        assert set(by_id) == {i for i, row in rows.items() if row["status"] == "complete"}
        for identifier, result in by_id.items():
            row, case_rows = rows[identifier], result["rows"]
            feasible = [r for r in case_rows if r["feasible"]]
            assert row["mean_realized_saving"] == sum(r["realized_saving"] for r in case_rows) / len(case_rows)
            assert row["mean_net_saving"] == sum(r["net_saving"] for r in case_rows) / len(case_rows)
            assert (row["feasible_agreements"], row["feasible_cases"]) == (
                sum(1 for r in feasible if r["status"] == "agreed"), len(feasible))
            assert row["supplier_walk_aways"] == sum(1 for r in case_rows if r["status"] == "supplier_walked")
            assert row["approval_cards"] == sum(r["approval_cards"] for r in case_rows)
            assert (row["authority_violations"], row["needs_review"]) == (0, 0)
            assert (row["live_requests"], row["store_hits"], row["live_seconds"]) == (
                result["counts"]["live_requests"], result["counts"]["store_hits"], result["counts"]["live_seconds"])
            assert row["evidence_class"] == result["evidence_class"] and row["cases"] == len(case_rows)
            assert row["estimate_sources"] == result["estimate_sources"]
            for label, other in (("minus_M0", m0), ("minus_rule_policy", rule)):
                difference = table["differences"][identifier][label]
                if other == identifier:
                    assert difference["realized_saving"] is None and difference["reason"] == "the same configuration"
                    continue
                assert difference["realized_saving"] == E.paired_difference(case_rows, by_id[other]["rows"])
                assert difference["net_saving"] == E.paired_difference(case_rows, by_id[other]["rows"], "net_saving")
                assert len(difference["realized_saving"]["ci90"]) == 2
        references = [row for row in table["rows"] if row["role"] == "reference"]
        assert references and all(row["evidence_class"] == "cpu_reference_not_model_evidence"
                                  and row["selectable"] is False and row["live_requests"] == 0 for row in references)
    markdown = (flow.state / "leaderboard.md").read_text(encoding="utf-8")
    for heading in ("# 협상 에이전트 사용 설정 튜닝 리더보드", "## 선택 구간(SELECT)", "## 선택 결과", "## 확인 구간(CONFIRM)",
                    "## 판정", "## 그 밖의 단계", "리허설 블록", "CPU 기준(선택 불가)"):
        assert heading in markdown
    assert len(list((flow.state / "boards").iterdir())) >= 2          # every board ever written is kept


def test_comparison_is_null_with_a_reason_unless_both_runs_are_complete_over_the_same_cases():
    rows = [{"case_id": f"dev-{i}", "realized_saving": 1000 * i, "net_saving": 900 * i} for i in range(6)]
    same = tune.Tuner.difference(rows, rows, "x")
    assert same["reason"] is None and same["realized_saving"] == E.paired_difference(rows, rows)
    assert tune.Tuner.difference(rows, None, "x") == {"comparison": "x", "realized_saving": None, "net_saving": None,
                                                     "reason": "one of the two runs is not complete"}
    for other in (rows[:5], list(reversed(rows)), rows[:5] + [{**rows[5], "case_id": "dev-99"}]):
        assert tune.Tuner.difference(rows, other, "x")["reason"] == "the two runs do not cover the same cases"


def test_selection_rule_takes_the_highest_mean_and_ties_go_to_the_earlier_candidate(flow):
    row = lambda name, mean, ok=True: {"name": name, "mean_realized_saving": mean, "selectable": ok}  # noqa: E731
    pick = lambda rows: (tune.choose_candidate(rows) or {}).get("name")  # noqa: E731
    assert pick([row("M0", 100.0), row("M1", 100.0), row("M2", 100.0)]) == "M0"            # M0 wins all ties
    assert pick([row("M0", 100.0), row("M1", 100.5), row("M2", 100.5)]) == "M1"            # the earlier of a tie
    assert pick([row("M0", 100.0), row("M1", None, False), row("M5", 90.0)]) == "M0"
    assert pick([row("M0", 100.0), row("M1", 500.0, False), row("M2", 99.0)]) == "M0"      # not eligible: skipped
    assert pick([row("M0", None, False), row("M1", None, False)]) is None
    # The written selection follows that rule over the real rows, and only candidates are in it.
    board, selection = flow.after_all_board, read(flow.after_all, "selection.json")
    candidates = [r for r in board["blocks"]["select"]["rows"] if r["role"] == "candidate"]
    assert [r["name"] for r in candidates] == [a.name for a in tune.CANDIDATES]
    assert all(r["selectable"] == (r["status"] == "complete" and r["authority_violations"] == 0
                                   and r["needs_review"] == 0) for r in candidates)
    best = max(r["mean_realized_saving"] for r in candidates if r["selectable"])
    expected = next(r for r in candidates if r["selectable"] and r["mean_realized_saving"] == best)
    assert selection["selected"]["id"] == expected["id"] == board["selection"]["selected"]["id"]
    assert selection["selected"]["is_default"] == (expected["name"] == "M0")
    assert [c["name"] for c in selection["candidates"]] == [a.name for a in tune.CANDIDATES]
    assert selection["rule"] == tune.SELECTION_RULE and selection["block"]["first_case_id"] == "dev-1130"
    assert not any(r["selectable"] for r in board["blocks"]["select"]["rows"] if r["role"] == "reference")


def test_selected_config_file_is_what_a_freeze_would_read(flow):
    selection, document = read(flow.state, "selection.json"), read(flow.state, "selected_config.json")
    assert set(document) == {"policy", "estimates", "selected_from"}
    assert document["policy"] == tune.policy_json() and document["estimates"] == selection["selected"]["estimates"]
    spec = EstimateSpec.from_json(document["estimates"])
    spec.validate()
    assert PolicyConfig(**{**document["policy"], "final_prior_by_round": tuple(document["policy"][
        "final_prior_by_round"])}) == PolicyConfig()
    origin = document["selected_from"]
    assert origin["selection_sha256"] == selection["selection_sha256"] and origin["config_id"] == selection["selected"]["id"]
    assert origin["rehearsal_blocks"] == {"cal": 20, "confirm": 8, "d0": 3, "select": 6}
    assert origin["selected_on"] == "dev-1130..dev-1135" and "no held-out case" in origin["notice"]
    load = getattr(E, "load_usage_config", None)
    if load is not None:                                   # the reader of this file, when it is present
        usage = load(flow.state / "selected_config.json")
        assert usage.policy == PolicyConfig() and usage.estimates == spec


def test_selection_is_written_once_and_a_changed_file_is_refused(tmp_path, flow, capsys):
    state = tmp_path / "state"
    shutil.copytree(flow.after_all, state)
    selection_bytes, config_bytes = (state / "selection.json").read_bytes(), (state / "selected_config.json").read_bytes()
    assert cli(state, "board")[0] == 0 and cli(state, "run", "--stage", "all")[0] == 0
    assert (state / "selection.json").read_bytes() == selection_bytes
    assert (state / "selected_config.json").read_bytes() == config_bytes
    (state / "selected_config.json").unlink()               # a crash between the two files is repaired
    assert cli(state, "board")[0] == 0 and (state / "selected_config.json").read_bytes() == config_bytes
    selection = json.loads(selection_bytes)
    other = next(c for c in selection["candidates"] if c["eligible"] and c["id"] != selection["selected"]["id"])
    selection["selected"]["id"], selection["selected"]["name"] = other["id"], other["name"]
    (state / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    capsys.readouterr()
    assert cli(state, "board")[0] == 1 and cli(state, "run", "--stage", "confirm")[0] == 1
    assert "selection.json does not match" in capsys.readouterr().err
    assert not (state / "confirm_first_use.json").exists()


def test_confirm_needs_a_selection_and_is_used_once(tmp_path, flow, capsys):
    state = planned(tmp_path)
    code, double = cli(state, "run", "--stage", "confirm")
    assert code == 1 and double.calls == 0 and "confirm needs selection.json" in capsys.readouterr().err
    assert not (state / "confirm_first_use.json").exists() and not (state / "runs").exists()

    # The first-use marker exists before the first confirm call, also when no call was made yet.
    state = tmp_path / "confirm"
    shutil.copytree(flow.after_all, state)
    assert cli(state, "run", "--stage", "confirm", "--rerun-not-confirmatory")[0] == 1      # nothing to rerun yet
    assert not (state / "confirm_first_use.json").exists()
    code, double = cli(state, "run", "--stage", "confirm", "--max-live-requests", "0")
    assert code == 0 and double.calls == 1                  # only the unstored runtime check before the marker
    marker_bytes = (state / "confirm_first_use.json").read_bytes()
    marker, selection = json.loads(marker_bytes), read(state, "selection.json")
    assert marker["selection_sha256"] == selection["selection_sha256"]
    assert marker["block_content_sha256"] == read(state, "protocol.json")["blocks"]["confirm"]["content_sha256"]
    assert marker["block"]["first_case_id"] == "dev-1250" and marker["selected"]["id"] == selection["selected"]["id"]
    assert not results(state, "confirm")
    # An unfinished first use is continued, not refused, and stays the first use.
    capsys.readouterr()
    assert cli(state, "run", "--stage", "confirm")[0] == 0
    assert "continuing the unfinished first use" in capsys.readouterr().err
    assert (state / "confirm_first_use.json").read_bytes() == marker_bytes
    first_use = results(state, "confirm")
    assert first_use and all(r["confirm_use"] == "first_use" and r["complete"] for r in first_use)
    assert complete_rows(state, "confirm") == complete_rows(flow.state, "confirm")
    names = [r["config"]["name"] for r in first_use]
    assert "M0" in names and selection["selected"]["name"] in names and "rule_policy" in names
    assert tune.twin_reference(EstimateSpec.from_json(selection["selected"]["estimates"])).name in names
    # A second confirm is refused and runs nothing.
    count = len(attempts(state, "confirm"))
    code, double = cli(state, "run", "--stage", "confirm")
    assert code == 1 and double.calls == 0 and "second confirm is refused" in capsys.readouterr().err
    assert len(attempts(state, "confirm")) == count
    verdict = read(state, "leaderboard.json")["confirm"]        # written by the confirm run itself
    assert verdict["verdict"] in tune.VERDICTS and verdict["counts_as_model_evidence"] is False   # rehearsal blocks
    # An explicit rerun is labelled not_confirmatory and changes neither the marker nor the verdict.
    code, double = cli(state, "run", "--stage", "confirm", "--rerun-not-confirmatory")
    assert code == 0 and double.calls == 0                                      # every answer came from the store
    reruns = [r for r in results(state, "confirm") if r["confirm_use"] == "not_confirmatory"]
    assert len(reruns) == count and all(r["evidence_class"] in ("real_model_replay_from_store",
                                                                "cpu_reference_not_model_evidence") for r in reruns)
    assert (state / "confirm_first_use.json").read_bytes() == marker_bytes
    assert cli(state, "board")[0] == 0
    board = read(state, "leaderboard.json")
    assert board["confirm"] == verdict and board["blocks"]["confirm"]["not_confirmatory_reruns"] == count
    assert all("not_confirmatory" not in (row["attempt"] or "") for row in board["blocks"]["confirm"]["rows"])
    first_attempts = {r["attempt"] for r in first_use}
    assert {row["attempt"].rsplit("/", 1)[1] for row in board["blocks"]["confirm"]["rows"]} == first_attempts


def test_unfinished_first_use_is_not_continued_after_the_source_changed(tmp_path, flow, monkeypatch, capsys):
    state = tmp_path / "state"
    shutil.copytree(flow.after_all, state)
    assert cli(state, "run", "--stage", "confirm", "--max-live-requests", "0")[0] == 0
    monkeypatch.setattr(tune, "input_hashes", lambda: {"negotiation/src/negotiation_agent/tune.py": "changed"})
    code, double = cli(state, "run", "--stage", "confirm")
    assert code == 1 and double.calls == 0 and "source files changed" in capsys.readouterr().err
    assert cli(state, "run", "--stage", "confirm", "--rerun-not-confirmatory")[0] == 0
    assert {r["confirm_use"] for r in results(state, "confirm")} == {"not_confirmatory"}
    assert read(state, "leaderboard.json")["confirm"]["verdict"] == "incomplete"      # reruns never count


def test_confirm_verdict_vocabulary_and_rules(flow):
    confirm = flow.board["confirm"]
    assert confirm["verdict"] in tune.VERDICTS and confirm["sentence"] == tune.RESULT_SENTENCES[confirm["verdict"]]
    assert confirm["secondary"]["model_contribution"]["verdict"] in ("model_contribution_shown",
                                                                     "model_contribution_not_shown")
    assert confirm["secondary"]["rule_policy"]["label"] == "not_confirmatory_reference_seen_in_design"
    assert confirm["primary"]["comparison"] == "S_minus_M0" and confirm["default"]["name"] == "M0"
    assert flow.after_all_board["confirm"] is None and "confirm" not in flow.after_all_board["blocks"]
    cases = [f"dev-{1250 + i}" for i in range(40)]
    base = [{"case_id": c, "realized_saving": 100_000 + 1_000 * (i % 7), "net_saving": 80_000 + 1_000 * (i % 7)}
            for i, c in enumerate(cases)]
    shifted = lambda gain, net: [{**r, "realized_saving": r["realized_saving"] + gain + 100 * (i % 3),  # noqa: E731
                                  "net_saving": r["net_saving"] + net} for i, r in enumerate(base)]
    better = tune.Tuner.difference(shifted(5_000, 5_000), base, "S_minus_M0")
    assert better["realized_saving"]["ci90"][0] > 0
    assert tune.primary_verdict(better, False, True, 0)[0] == "improvement_confirmed"
    assert tune.primary_verdict(better, True, True, 0) == ("not_confirmed",
                                                         "default kept: the selected configuration is M0")
    assert tune.primary_verdict(better, False, False, 0)[0] == "incomplete"
    assert tune.primary_verdict(better, False, True, 1)[0] == "not_confirmed"          # an authority violation
    costly = tune.Tuner.difference(shifted(5_000, -1), base, "S_minus_M0")
    assert tune.primary_verdict(costly, False, True, 0) == ("not_confirmed", "the mean net-saving difference is negative")
    noisy = [{**r, "realized_saving": r["realized_saving"] + (60_000 if i % 2 else -55_000)} for i, r in enumerate(base)]
    unclear = tune.Tuner.difference(noisy, base, "S_minus_M0")
    assert unclear["realized_saving"]["mean_difference"] > 0 >= unclear["realized_saving"]["ci90"][0]
    assert tune.primary_verdict(unclear, False, True, 0)[0] == "not_confirmed"
    same = tune.Tuner.difference(base, base, "S_minus_M0")                               # lower bound exactly 0
    assert tune.primary_verdict(same, False, True, 0)[0] == "not_confirmed"
    missing = tune.Tuner.difference(base, base[:-1], "S_minus_M0")
    assert tune.primary_verdict(missing, False, True, 0)[0] == "incomplete"
    assert all(tune.primary_verdict(d, k, c, v)[0] in tune.VERDICTS
               for d in (better, costly, unclear, same, missing) for k in (True, False) for c in (True, False)
               for v in (0, 2))


# ---- pack and restore ----------------------------------------------------------------------------------------

def test_pack_and_restore_round_trip_and_continue(tmp_path, flow, capsys):
    archive = tmp_path / "state.zip"
    assert cli(flow.state, "pack", "--output", str(archive))[0] == 0
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(tune.MANIFEST_NAME))
        files = {p.relative_to(flow.state).as_posix(): p for p in flow.state.rglob("*") if p.is_file()}
        assert {e["path"] for e in manifest["files"]} == set(files) == set(bundle.namelist()) - {tune.MANIFEST_NAME}
        assert all((e["size"], e["sha256"]) == (len(files[e["path"]].read_bytes()), hashlib.sha256(
            files[e["path"]].read_bytes()).hexdigest()) for e in manifest["files"])
        assert manifest["protocol_sha256"] == read(flow.state, "protocol.json")["protocol_sha256"]
    restored = tmp_path / "restored"
    assert cli(restored, "restore", "--input", str(archive))[0] == 0
    assert {p.relative_to(restored).as_posix(): p.read_bytes() for p in restored.rglob("*") if p.is_file()} == {
        name: path.read_bytes() for name, path in files.items()}
    # The restored state is complete: nothing is run again, nothing reaches the backend.
    count = len(list((restored / "runs").glob("*/*/attempt-*")))
    code, double = cli(restored, "run", "--stage", "all")
    assert code == 0 and double.calls == 0 and len(list((restored / "runs").glob("*/*/attempt-*"))) == count
    assert read(restored, "leaderboard.json")["confirm"] == flow.board["confirm"]
    assert cli(restored, "run", "--stage", "confirm")[0] == 1                   # the single use travels with it
    # Restoring the same archive again changes nothing; packing over a file needs --replace.
    capsys.readouterr()
    assert cli(restored, "restore", "--input", str(archive))[0] == 0
    assert "written=0" in capsys.readouterr().out
    assert cli(flow.state, "pack", "--output", str(archive))[0] == 1
    assert cli(flow.state, "pack", "--output", str(archive), "--replace")[0] == 0
    assert cli(flow.state, "pack", "--output", str(flow.state / "inside.zip"))[0] == 1
    assert not (flow.state / "inside.zip").exists() and not list(tmp_path.glob("*.tmp-*"))
    assert cli(tmp_path / "nothing", "pack", "--output", str(tmp_path / "x.zip"))[0] == 1
    # An interrupted state can be moved and finished elsewhere.
    half = planned(tmp_path, "half")
    cli(half, "run", "--stage", "all", "--max-live-requests", "20")
    assert cli(half, "pack", "--output", str(tmp_path / "half.zip"))[0] == 0
    moved = tmp_path / "moved"
    assert cli(moved, "restore", "--input", str(tmp_path / "half.zip"))[0] == 0
    code, double = cli(moved, "run", "--stage", "all")
    assert code == 0 and store_entries(moved) == flow.answered_after_all
    assert complete_rows(moved, "select") == complete_rows(flow.after_all, "select")


def _rewrite(source, target, change):
    with zipfile.ZipFile(source) as bundle, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        members = {name: bundle.read(name) for name in bundle.namelist()}
        change(members)
        for name, data in members.items():
            out.writestr(name, data)


def test_restore_refuses_tampered_archives_and_conflicting_targets(tmp_path, flow, capsys):
    archive = tmp_path / "state.zip"
    assert cli(flow.after_all, "pack", "--output", str(archive))[0] == 0
    store_name = next(n for n in zipfile.ZipFile(archive).namelist() if n.startswith("store/") and n.endswith(".jsonl"))

    def flip(members):
        members["selection.json"] = members["selection.json"].replace(b'"M', b'"X', 1)

    def grow(members):
        members[store_name] += b'{"extra": 1}\n'

    def extra(members):
        members["runs/unlisted.json"] = b"{}"

    def drop(members):
        del members["selection.json"]

    def escape(members):
        manifest = json.loads(members[tune.MANIFEST_NAME])
        manifest["files"].append({"path": "../outside.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()})
        members[tune.MANIFEST_NAME] = json.dumps(manifest).encode()
        members["../outside.json"] = b"{}"

    def no_manifest(members):
        del members[tune.MANIFEST_NAME]

    for number, change in enumerate((flip, grow, extra, drop, escape, no_manifest)):
        broken, target = tmp_path / f"broken{number}.zip", tmp_path / f"target{number}"
        _rewrite(archive, broken, change)
        assert cli(target, "restore", "--input", str(broken))[0] == 1, change.__name__
        assert "Refused" in capsys.readouterr().err and not target.exists()
    assert not (tmp_path / "outside.json").exists()
    not_a_zip = tmp_path / "plain.zip"
    not_a_zip.write_text("not a zip", encoding="utf-8")
    assert cli(tmp_path / "t", "restore", "--input", str(not_a_zip))[0] == 1
    # A non-empty target with another protocol digest is refused and left as it was.
    other = planned(tmp_path, "other", rehearsal="d0=3,select=5,confirm=8,cal=20")
    before = (other / "protocol.json").read_bytes()
    assert cli(other, "restore", "--input", str(archive))[0] == 1
    assert "different protocol digest" in capsys.readouterr().err
    assert sorted(p.name for p in other.iterdir()) == ["protocol.json"] and (other / "protocol.json").read_bytes() == before
    # The same protocol but different recorded content: refused as a whole, nothing merged.
    diverged = planned(tmp_path, "diverged")
    cli(diverged, "run", "--stage", "smoke", double=ModelDouble(dtype="another_runtime"))
    listing = sorted(p.relative_to(diverged).as_posix() for p in diverged.rglob("*") if p.is_file())
    assert cli(diverged, "restore", "--input", str(archive))[0] == 1
    assert "nothing was restored" in capsys.readouterr().err
    assert sorted(p.relative_to(diverged).as_posix() for p in diverged.rglob("*") if p.is_file()) == listing
    # Only the regenerated leaderboard views may differ; the target keeps its own copy of them.
    same = tmp_path / "same"
    shutil.copytree(flow.after_all, same)
    (same / "leaderboard.md").write_text("local view\n", encoding="utf-8")
    assert cli(same, "restore", "--input", str(archive))[0] == 0
    assert "kept_existing_views=['leaderboard.md']" in capsys.readouterr().out
    assert (same / "leaderboard.md").read_text(encoding="utf-8") == "local view\n"


# ---- review fixes of 2026-11-18 (CPU doubles only) ---------------------------------------------------------

def test_first_confirm_use_is_not_spent_on_a_runtime_that_cannot_continue_it(tmp_path, flow, capsys):
    state = tmp_path / "confirm"
    shutil.copytree(flow.after_all, state)
    entries = store_entries(state)
    code, other = cli(state, "run", "--stage", "confirm", double=ModelDouble(dtype="another_runtime"))
    assert code == 1 and other.calls == 1 and "CONFIRM was NOT used" in capsys.readouterr().err
    assert not (state / "confirm_first_use.json").exists() and not results(state, "confirm")
    assert store_entries(state) == entries                       # the check's answer is never stored
    code, idle = cli(state, "run", "--stage", "confirm", "--store-mode", "read-only")
    assert code == 1 and idle.calls == 0 and "read-only is refused" in capsys.readouterr().err
    assert not (state / "confirm_first_use.json").exists()

    def broken():
        raise RuntimeError("no weights here")

    assert cli(state, "run", "--stage", "confirm", factory=broken)[0] == 1
    assert "CONFIRM was NOT used" in capsys.readouterr().err and not (state / "confirm_first_use.json").exists()


def test_latency_excludes_building_the_backend_and_the_build_time_is_reported_apart(tmp_path):
    import time as _time
    state = planned(tmp_path)

    def slow_factory():
        _time.sleep(0.4)
        return ModelDouble()

    assert cli(state, "run", "--stage", "smoke", factory=slow_factory)[0] == 0
    first = results(state, "smoke", "M0")[0]
    assert first["counts"]["store_backend"]["backend_build_seconds"] >= 0.4
    assert first["counts"]["live_latency_ms"]["max"] < 300 and first["counts"]["live_seconds"] < 0.3


def test_edited_maps_or_selection_files_are_refused(tmp_path, flow, capsys):
    state = tmp_path / "edited"
    shutil.copytree(flow.after_all, state)
    maps_file = next((state / "maps").glob("calibration-*.json"))
    original = maps_file.read_text(encoding="utf-8")
    maps = json.loads(original)
    maps["final_offer_shrink"]["weight"] = 0.37                  # "trying another weight" by hand
    maps_file.write_text(json.dumps(maps), encoding="utf-8")
    assert cli(state, "board")[0] == 1 and "hand-written or edited maps file is refused" in capsys.readouterr().err
    assert cli(state, "run", "--stage", "select")[0] == 1
    maps_file.write_text(original, encoding="utf-8")
    assert cli(state, "board")[0] == 0
    # A selection that names estimates outside the registered candidate, with its digests recomputed.
    selection = read(state, "selection.json")
    selection["selected"]["estimates"] = {**selection["selected"]["estimates"], "acceptance_source": "constant",
                                          "acceptance_constant": 0.9, "acceptance_platt": None}
    spec = EstimateSpec.from_json(selection["selected"]["estimates"])
    selection["selected"]["id"] = tune.Config(selection["selected"]["name"], "candidate", "estimates", "", spec).id
    selection["selection_sha256"] = tune._digest({k: v for k, v in selection.items()
                                                  if k not in ("selection_sha256", "created_at_utc")})
    (state / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    code, double = cli(state, "run", "--stage", "confirm")
    assert code == 1 and double.calls == 0 and "edited selection is refused" in capsys.readouterr().err
    assert not (state / "confirm_first_use.json").exists()
    # A state without any calibrate result does not accept a maps file at all.
    bare = planned(tmp_path, "bare")
    assert cli(bare, "run", "--stage", "smoke")[0] == 0
    (bare / "maps").mkdir()
    (bare / "maps" / maps_file.name).write_text(original, encoding="utf-8")
    assert cli(bare, "run", "--stage", "select")[0] == 1
    assert "no complete calibrate result" in capsys.readouterr().err


def test_maps_can_be_declared_unavailable_only_after_a_retried_failure(tmp_path, capsys):
    probe = planned(tmp_path, "probe")
    _, counter = cli(probe, "run", "--stage", "smoke")
    cli(probe, "run", "--stage", "regress", double=counter)
    before_cal = counter.calls

    class AlwaysFailsOnCalibration(ModelDouble):
        def predict(self, state, candidates, request):
            if self.calls >= before_cal:
                self.calls += 1
                raise RuntimeError("non-finite hidden states (test)")
            return super().predict(state, candidates, request)

    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "all", double=AlwaysFailsOnCalibration())[0] == 1
    capsys.readouterr()
    assert cli(state, "fit-map", "--declare-unavailable", "--reason", "x")[0] == 1      # not retried yet
    assert "failed again" in capsys.readouterr().err
    assert cli(state, "run", "--stage", "calibrate", "--retry-failed",
               double=AlwaysFailsOnCalibration(fail_at=range(1, 999)))[0] == 1
    assert cli(state, "fit-map", "--declare-unavailable")[0] == 1                       # a reason is required
    assert cli(state, "fit-map", "--declare-unavailable", "--reason", "FP16 non-finite on one request")[0] == 0
    maps = json.loads(next((state / "maps").glob("calibration-*.json")).read_text(encoding="utf-8"))
    assert maps["platt"] is None and maps["final_offer_shrink"]["weight"] is None
    assert maps["declared_unavailable"]["reason"] == "FP16 non-finite on one request"
    assert len(maps["declared_unavailable"]["calibrate_attempts"]) == 2
    assert {name: arm["status"] for name, arm in maps["arms"].items()} == {
        name: "failed" for name in ("M3", "F1", "F2", "M4", "M5")}
    assert cli(state, "fit-map", "--declare-unavailable", "--reason", "again")[0] == 1  # decided once
    # M0, M1 and M2 go on; the five arms that need a map are terminal, so a selection can be written.
    assert cli(state, "run", "--stage", "select")[0] == 0 and cli(state, "board")[0] == 0
    selection = read(state, "selection.json")
    listed = {c["name"]: (c["status"], c["reason"]) for c in selection["candidates"]}
    assert selection["selected"]["name"] in ("M0", "M1", "M2")
    assert all(listed[name] == ("failed", "map_unavailable") for name in ("M3", "F1", "F2", "M4", "M5"))


def test_smoke_retry_is_recorded_and_sources_are_readable_in_the_table(tmp_path, flow):
    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "smoke", double=ModelDouble(fail_at={1}))[0] == 1
    failed, = results(state, "smoke", "M0")
    assert failed["status"] == "failed" and failed["evidence_class"] == "real_model_requested_no_model_answer"
    assert failed["counts"]["live_requests"] == 0 and failed["counts"]["live_failed_requests"] == 1
    assert cli(state, "run", "--stage", "smoke", "--retry-failed")[0] == 0
    retried = results(state, "smoke", "M0")[1]
    assert retried["retry_of"] == failed["attempt"] and retried["retry_failed_flag"] is True
    rows = {r["name"]: r for r in read(flow.after_all, "leaderboard.json")["blocks"]["select"]["rows"]}
    assert rows["M0"]["estimate_sources_display"] == {"acceptance": "model", "final_offer": "model"}
    if rows["M4"]["status"] == "complete":
        assert rows["M4"]["estimate_sources_display"]["acceptance"] == "model+platt(calibration)"
    assert rows["constant_keyword"]["estimate_sources_display"]["acceptance"] == "constant(0.15)"
    text = (flow.after_all / "leaderboard.md").read_text(encoding="utf-8")
    assert "constant(0.15)" in text


# ---- second review round of 2026-11-18: interruption, store writes, one runner, source versions -------------

def test_interrupt_outside_the_backend_call_is_an_interruption_not_a_failed_estimate(tmp_path, monkeypatch, capsys):
    state = planned(tmp_path)
    original, seen = policy.response_snapshot, {"n": 0}

    def snapshot(backend):
        seen["n"] += 1
        if seen["n"] == 3:
            raise KeyboardInterrupt("between the answer and its record")
        return original(backend)

    monkeypatch.setattr(policy, "response_snapshot", snapshot)
    code, _ = cli(state, "run", "--stage", "all")
    assert code == 130 and "status=interrupted" in capsys.readouterr().out
    last = [r for stage in tune.STAGES for r in results(state, stage)][-1]
    assert last["status"] == "interrupted" and last["counts"]["failed_estimates"] == 0
    monkeypatch.setattr(policy, "response_snapshot", original)
    assert cli(state, "run", "--stage", "smoke")[0] == 0           # continues; no --retry-failed is needed
    assert not (state / tune.LOCK_NAME).exists()


def test_a_failed_store_write_stops_the_stage_before_any_further_model_call(tmp_path, monkeypatch, capsys):
    state = planned(tmp_path)
    original, seen = PredictionStore._append, {"n": 0}

    def append(self, line):
        seen["n"] += 1
        if seen["n"] == 3:
            raise OSError(28, "No space left on device")
        return original(self, line)

    monkeypatch.setattr(PredictionStore, "_append", append)
    code, double = cli(state, "run", "--stage", "all")
    captured = capsys.readouterr()
    assert code == 1 and "status=store_write_failed" in captured.out and double.calls == 3
    assert "no further model call was made" in captured.err
    last = [r for stage in tune.STAGES for r in results(state, stage)][-1]
    assert last["status"] == "store_write_failed" and "No space left" in last["first_errors"][0]["error"]
    monkeypatch.setattr(PredictionStore, "_append", original)
    assert cli(state, "run", "--stage", "smoke")[0] == 0           # not a terminal failure of the configuration


def test_one_runner_per_state_directory(tmp_path, capsys):
    import os
    state = planned(tmp_path)
    host = os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "")
    lock = state / tune.LOCK_NAME
    lock.write_text(json.dumps({"pid": os.getpid(), "host": host, "started_at_utc": "x"}), encoding="utf-8")
    code, double = cli(state, "run", "--stage", "smoke")
    assert code == 1 and double.calls == 0 and "another run is using this state directory" in capsys.readouterr().err
    assert lock.exists()                                             # a live owner's lock is left alone
    if os.name == "posix":
        lock.write_text(json.dumps({"pid": 2 ** 22 + 54321, "host": host, "started_at_utc": "x"}), encoding="utf-8")
        assert cli(state, "run", "--stage", "smoke")[0] == 0        # left by a killed process: taken over
    else:
        lock.unlink()
        assert cli(state, "run", "--stage", "smoke")[0] == 0
    assert not lock.exists()
    archive = tmp_path / "state.zip"
    lock.write_text("{}", encoding="utf-8")
    assert cli(state, "pack", "--output", str(archive))[0] == 0
    assert tune.LOCK_NAME not in zipfile.ZipFile(archive).namelist()


def test_candidates_run_under_different_source_files_block_the_selection(tmp_path, flow):
    state = tmp_path / "mixed"
    shutil.copytree(flow.after_all, state)
    (state / "selection.json").unlink()
    (state / "selected_config.json").unlink()
    folder, result = attempts(state, "select", "M1")[0]
    result["provenance"]["input_sha256"]["negotiation/src/negotiation_agent/estimate_sources.py"] = "0" * 64
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert cli(state, "board")[0] == 0
    board = read(state, "leaderboard.json")
    assert board["selection"]["status"] == "blocked" and not (state / "selection.json").exists()
    assert {"mixed_source_versions"} <= {b["state"] for b in board["selection"]["blocking"]}
    assert read(flow.after_all, "selection.json")["input_sha256_digest"]


def test_unfinished_first_confirm_use_is_not_continued_after_its_answers_left_the_store(tmp_path, flow, capsys):
    state = tmp_path / "confirm"
    shutil.copytree(flow.after_all, state)
    code, _ = cli(state, "run", "--stage", "confirm", "--max-live-requests", "6")
    assert code == 0 and (state / "confirm_first_use.json").exists()
    partial = [r for r in results(state, "confirm") if r["status"] == "partial_budget"]
    assert partial and partial[0]["counts"]["live_requests"] > 0
    store_file = next((state / "store").glob("*.jsonl"))
    lines = store_file.read_text(encoding="utf-8").splitlines()
    store_file.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")      # one recorded answer removed
    capsys.readouterr()
    code, double = cli(state, "run", "--stage", "confirm")
    assert code == 1 and double.calls == 0 and "no longer in the store" in capsys.readouterr().err
    assert len([r for r in results(state, "confirm")]) == len(partial)


def test_exclusive_files_are_complete_or_absent_and_unreadable_state_is_a_refusal(tmp_path, capsys):
    target = tmp_path / "x" / "file.json"
    tune._write_json(target, {"a": 1}, exclusive=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(FileExistsError):
        tune._write_json(target, {"a": 2}, exclusive=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["file.json"]            # no staging file is left
    state = planned(tmp_path)
    assert cli(state, "run", "--stage", "smoke")[0] == 0
    (state / "store_identity.json").write_text("", encoding="utf-8")            # as a killed process could leave it
    capsys.readouterr()
    assert cli(state, "run", "--stage", "regress")[0] == 1
    assert "Refused: a state file could not be read or written" in capsys.readouterr().err

