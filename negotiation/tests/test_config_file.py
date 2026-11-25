"""CPU tests of `--config-file` for `evaluate` and `freeze`.

NOT executed on a GPU as of 2026-11-18 (CPU reference backends and test doubles only): every
"model" here is a test double built on the CPU keyword reference. The savings asserted below
are evaluator-side CPU stand-ins on synthetic suppliers, not model evidence.

Held-out protocol tests run on throwaway seeds and a temporary history directory, never on
the real evaluation set.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from negotiation_agent import evaluation as E
from negotiation_agent.backends import HeuristicBackend, KeywordBackend
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec, shrink
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import request_sha256
from negotiation_agent.requests import FINAL_OFFER_QID

GOLDEN_DEV_1100_ROUND_1 = "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
DEV30_ARTIFACT = Path(E.PACKAGE_ROOT) / "artifacts" / "20261116_colab" / "dev30_summary.json"
DEFAULT_POLICY = json.loads(json.dumps(asdict(PolicyConfig())))
PLATT = {"method": "platt", "a": 0.9, "b": 0.2, "n": 10, "fitted_on_split": "calibration", "source": "test"}

IDENTITY = EstimateSpec()
CONSTANT_PRIOR = EstimateSpec("constant", 0.15, None, "policy_prior")
CONSTANT_KEYWORD = EstimateSpec("constant", 0.15, None, "keyword")
MODEL_KEYWORD = EstimateSpec("model", None, None, "keyword")
MODEL_SHRINK = EstimateSpec("model", None, None, "shrink", 0.25)
ZORO = ["--backend", "zoro"]


class CountingModel:
    """KeywordBackend values behind a declared-model test double. It is NOT a model."""

    kind, is_model, supports_final_offer = "counting_model_test_double", True, True

    def __init__(self):
        self.calls, self.closed, self.last_response = 0, 0, None
        self._base = KeywordBackend()

    def predict(self, state, candidates, request):
        self.calls += 1
        out = self._base.predict(state, candidates, request)
        self.last_response = {"metadata": {"request_sha256": request_sha256(request), "sequence": self.calls},
                              "answers": dict(out)}
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True}

    def close(self):
        self.closed += 1


class CountingRule(HeuristicBackend):
    """The CPU rule reference with a call counter."""

    def __init__(self):
        self.calls = 0

    def predict(self, state, candidates, request):
        self.calls += 1
        return super().predict(state, candidates, request)


def write_config(path, spec=IDENTITY, policy=None, **extra):
    """The shape the tuning runner writes: {"policy": ..., "estimates": ..., extra keys}."""
    estimates = spec.to_json() if isinstance(spec, EstimateSpec) else spec
    content = {"policy": DEFAULT_POLICY if policy is None else policy, "estimates": estimates, **extra}
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_saving(result, name="agent"):
    return result["policies"][name]["mean_realized_saving"]


def outcomes(result, name="agent"):
    return [(case["case_id"], case[name]["status"], case[name]["realized_saving"], case[name]["net_saving"],
             case[name]["rounds_used"], case[name]["approval_cards"]) for case in result["cases"]]


def attempts(result, name="agent"):
    return [record["model"] for case in result["cases"] for record in case[name]["internal_record"]
            if "trace_id" in record["model"]]


def forbid(monkeypatch, *names):
    def forbidden(*_args, **_kwargs):
        pytest.fail("a refused configuration must not reach a backend, a freeze or a history marker")

    for name in names:
        monkeypatch.setattr(E, name, forbidden)


@pytest.fixture()
def run(tmp_path):
    """The command line with a temporary history directory; results are returned parsed."""
    history = tmp_path / "history"

    def call(*argv):
        return E.main([str(a) for a in argv], history_dir=history)

    call.history = history
    return call


@pytest.fixture()
def throwaway_heldout(monkeypatch):
    """Throwaway seeds stand in for the held-out set, exactly as test_evaluation.py does."""
    monkeypatch.setitem(E.SPLIT_SEED_BASE, "eval", 9000)


# ---- without the flag nothing changes ---------------------------------------------------------------

def test_default_cpu_dev30_run_is_unchanged_without_the_flag(tmp_path, run):
    output = tmp_path / "dev.json"
    assert run("evaluate", "--split", "dev", "--output", output) == 0
    result = read(output)
    # CPU reference values (rule-based stand-ins on synthetic suppliers; not model evidence).
    assert round(mean_saving(result), 2) == 284_263.67
    assert round(mean_saving(result, "fixed_concession"), 2) == 260_145.5
    assert round(mean_saving(result, E.REFERENCE_POLICY), 2) == 348_003.93
    assert result["agent_is_rule_policy"] and "rule_policy" not in result["policies"]
    assert result["backend"]["kind"] == HeuristicBackend.kind
    assert result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["provenance"]["execution_config"] == {
        "backend": "heuristic", "policy": DEFAULT_POLICY, "buyer_private": asdict(E.default_buyer_private()),
        "cases": 30, "approver": "auto", "calibration": None, "real_model_requested": False}
    first = attempts(result)[0]
    assert result["cases"][0]["case_id"] == "dev-1100"
    assert first["request_sha256"] == request_sha256(first["request"]) == GOLDEN_DEV_1100_ROUND_1
    assert FINAL_OFFER_QID in first["request"]["questions"]
    # Case by case against the curated run of 2026-11-16, whose CPU rows predate this option.
    recorded = read(DEV30_ARTIFACT)["case_outcomes"]
    keys = ("status", "realized_saving", "net_saving", "rounds_used", "approval_cards", "authority_violations")
    for case, before in zip(result["cases"], recorded, strict=True):
        assert case["case_id"] == before["case_id"]
        for now, then in (("agent", "rule_policy"), ("fixed_concession", "fixed_concession"),
                          ("midpoint", "midpoint"), (E.REFERENCE_POLICY, E.REFERENCE_POLICY)):
            assert {k: case[now][k] for k in keys} == before[then], (case["case_id"], now)


def test_freeze_without_the_flag_records_no_usage_config_key(tmp_path, run):
    manifest = tmp_path / "freeze.json"
    assert run("freeze", "--cases", "4", "--output", manifest) == 0
    assert read(manifest)["execution_config"] == {
        "backend": "heuristic", "policy": DEFAULT_POLICY, "buyer_private": asdict(E.default_buyer_private()),
        "cases": 4, "approver": "auto", "calibration": None, "real_model_requested": False}
    args = SimpleNamespace(backend="heuristic", cases=4, approver="auto")
    assert "usage_config" not in E.execution_config(args, PolicyConfig(), None)
    assert "usage_config" not in E.execution_config(args, PolicyConfig(), None, None)


@pytest.mark.parametrize("argv", [["sweep", "--cases", "2", "--grid", "0"],
                                  ["fit-calibration", "--input", "unused.json"]])
def test_the_flag_exists_only_on_evaluate_and_freeze(tmp_path, run, monkeypatch, capsys, argv):
    forbid(monkeypatch, "_build_backend", "sweep", "fit_calibration", "load_usage_config")
    config = write_config(tmp_path / "config.json")
    with pytest.raises(SystemExit) as refused:
        run(*argv, "--config-file", config, "--output", tmp_path / "out.json")
    assert refused.value.code == 2 and "--config-file" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]


# ---- what is recorded --------------------------------------------------------------------------------

def test_the_whole_file_and_its_hashes_are_embedded_in_the_execution_config(tmp_path, run):
    config = write_config(tmp_path / "selected_config.json", MODEL_SHRINK,
                          operator_note={"selection_sha256": "ab" * 32, "protocol": "tune-test"}, note="메모")
    output = tmp_path / "dev.json"
    assert run("evaluate", "--split", "dev", "--cases", "3", "--backend", "keyword",
               "--config-file", config, "--output", output) == 0
    recorded = read(output)["provenance"]["execution_config"]
    usage = recorded["usage_config"]
    assert set(usage) == {"origin", "file_sha256", "content", "estimates", "config_sha256"}
    assert usage["origin"] == "operator_file_origin_unverified"          # no selected_from: nothing to check
    assert usage["file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert usage["content"] == read(config)                              # extra keys included
    assert usage["content"]["operator_note"]["protocol"] == "tune-test" and usage["content"]["note"] == "메모"
    assert usage["estimates"] == MODEL_SHRINK.to_json()
    from negotiation_agent.policy import LEGACY_POLICY_FIELDS   # left out of the digest at legacy defaults
    digest_policy = {k: v for k, v in DEFAULT_POLICY.items() if k not in LEGACY_POLICY_FIELDS}
    canonical = json.dumps({"policy": digest_policy, "estimates": MODEL_SHRINK.to_json()}, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert usage["config_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert recorded["policy"] == DEFAULT_POLICY and recorded["calibration"] is None
    assert recorded["backend"] == "keyword" and recorded["real_model_requested"] is False
    assert str(tmp_path) not in json.dumps(usage)                         # the content, not where it was read


def test_formatting_changes_the_file_hash_but_not_the_config_hash_and_extra_keys_are_not_used(tmp_path):
    first = write_config(tmp_path / "a.json", MODEL_SHRINK)
    second = tmp_path / "b.json"
    second.write_text(json.dumps({"estimates": {"final_shrink_weight": 0.25, "final_source": "shrink"},
                                  "policy": {}, "comment": "defaults left out"}), encoding="utf-8")
    one, two = E.load_usage_config(first), E.load_usage_config(second)
    assert one.policy == two.policy == PolicyConfig() and one.estimates == two.estimates == MODEL_SHRINK
    assert one.record["config_sha256"] == two.record["config_sha256"]
    assert one.record["file_sha256"] != two.record["file_sha256"]
    assert one.record["estimates"] == two.record["estimates"] == MODEL_SHRINK.to_json()
    other = E.load_usage_config(write_config(tmp_path / "c.json", MODEL_KEYWORD))
    assert other.record["config_sha256"] != one.record["config_sha256"]


def test_policy_fields_are_restored_and_missing_ones_keep_their_defaults(tmp_path):
    policy = {"prior_concession": 300_000, "final_prior_by_round": [0, 0.2, 1]}
    loaded = E.load_usage_config(write_config(tmp_path / "config.json", CONSTANT_KEYWORD, policy))
    assert loaded.policy == PolicyConfig(prior_concession=300_000, final_prior_by_round=(0.0, 0.2, 1.0))
    assert isinstance(loaded.policy.final_prior_by_round, tuple)
    assert all(isinstance(p, float) for p in loaded.policy.final_prior_by_round)
    assert loaded.policy.reserve_margin == PolicyConfig().reserve_margin
    assert loaded.estimates == CONSTANT_KEYWORD


def test_a_file_saved_with_a_byte_order_mark_is_read(tmp_path):
    config = tmp_path / "config.json"
    config.write_bytes(b"\xef\xbb\xbf" + json.dumps({"policy": DEFAULT_POLICY,
                                                    "estimates": CONSTANT_PRIOR.to_json()}).encode("utf-8"))
    loaded = E.load_usage_config(config, "heuristic")
    assert loaded.estimates == CONSTANT_PRIOR
    assert loaded.record["file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_execution_config_refuses_a_second_policy_or_map_and_does_not_share_the_record(tmp_path):
    usage = E.load_usage_config(write_config(tmp_path / "config.json", CONSTANT_PRIOR))
    args = SimpleNamespace(backend="heuristic", cases=4, approver="auto")
    with pytest.raises(ValueError, match="its own policy"):
        E.execution_config(args, PolicyConfig(prior_concession=1), None, usage)
    with pytest.raises(ValueError, match="its own policy"):
        E.execution_config(args, usage.policy, dict(PLATT), usage)
    settings = E.execution_config(args, usage.policy, None, usage)
    settings["usage_config"]["content"]["policy"]["prior_concession"] = -1
    assert usage.record["content"]["policy"]["prior_concession"] == PolicyConfig().prior_concession
    assert json.loads(json.dumps(settings, allow_nan=False)) == settings      # reads back from a manifest unchanged


# ---- running with a configuration -------------------------------------------------------------------

def test_identity_config_reproduces_the_plain_backend_through_the_command_line(tmp_path, run):
    plain, composed = tmp_path / "plain.json", tmp_path / "composed.json"
    config = write_config(tmp_path / "config.json", IDENTITY)
    common = ["evaluate", "--split", "dev", "--cases", "12", "--backend", "keyword"]
    assert run(*common, "--output", plain) == 0
    assert run(*common, "--config-file", config, "--output", composed) == 0
    before, after = read(plain), read(composed)
    assert before["backend"]["kind"] == KeywordBackend.kind
    assert after["backend"]["kind"] == ComposedBackend.kind == "composed_estimates"
    assert after["backend"]["estimate_sources"] == {"acceptance": "model", "final_offer": "model"}
    assert after["backend"]["inner"]["kind"] == KeywordBackend.kind
    # A composition over a CPU stand-in is not labelled as a model run.
    assert after["evidence_class"] == before["evidence_class"] == "cpu_reference_not_model_evidence"
    for name in before["policies"]:
        assert outcomes(after, name) == outcomes(before, name), name
    sent_before, sent_after = attempts(before), attempts(after)
    assert [a["request_sha256"] for a in sent_after] == [a["request_sha256"] for a in sent_before]
    assert [a["predictions"] for a in sent_after] == [a["predictions"] for a in sent_before]
    assert sent_after[0]["request_sha256"] == GOLDEN_DEV_1100_ROUND_1
    assert all(a["response"]["composition"]["status"] == "ok" for a in sent_after)
    assert after["config"]["policy"] == before["config"]["policy"] == DEFAULT_POLICY
    assert "usage_config" not in before["provenance"]["execution_config"]


@pytest.mark.parametrize("spec, expected", [(CONSTANT_PRIOR, 300_035), (CONSTANT_KEYWORD, 324_233)])
def test_a_model_free_config_makes_zero_backend_calls(tmp_path, run, monkeypatch, spec, expected):
    built = []

    def build(_args):
        built.append(CountingRule())
        return built[-1]

    monkeypatch.setattr(E, "make_backend", build)
    output = tmp_path / "dev.json"
    assert run("evaluate", "--split", "dev", "--config-file", write_config(tmp_path / "config.json", spec),
               "--output", output) == 0
    result = read(output)
    assert len(built) == 1 and built[0].calls == 0
    assert result["complete"] and result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["backend"]["kind"] == "composed_estimates" and result["backend"]["is_model"] is False
    assert result["backend"]["inner"] is None and result["backend"]["model_estimates_used"] == []
    assert result["backend"]["estimate_sources"] == spec.estimate_sources()
    # CPU stand-in values on dev-1100..1129 (not model evidence), the same as the composition's own tests.
    assert round(mean_saving(result)) == expected
    assert round(mean_saving(result, "rule_policy"), 2) == 284_263.67     # the rule baseline still runs
    sent = attempts(result)
    assert sent[0]["request_sha256"] == GOLDEN_DEV_1100_ROUND_1           # the request itself is unchanged
    assert all(FINAL_OFFER_QID in a["request"]["questions"] for a in sent)
    assert all(a["is_model"] is False and a["response"]["composition"]["raw_model_predictions"] is None for a in sent)
    source = {"policy_prior": "policy_prior", "keyword": "backend_estimate"}[spec.final_source]
    facts = [r["facts"] for case in result["cases"] for r in case["agent"]["internal_record"]]
    assert {f["p_final_offer_source"] for f in facts if "p_final_offer_source" in f} <= {source, "public_round_limit"}


def test_a_composed_model_run_keeps_the_rule_baseline_and_is_never_overwritten(tmp_path, run, monkeypatch):
    built = []

    def build(_args):
        built.append(CountingModel())
        return built[-1]

    monkeypatch.setattr(E, "make_backend", build)
    output = tmp_path / "model.json"
    decisions = output.with_suffix(".json.decisions.jsonl")
    argv = ["evaluate", "--split", "dev", "--cases", "6", *ZORO,
            "--config-file", write_config(tmp_path / "config.json", MODEL_KEYWORD), "--output", output]
    assert run(*argv) == 0
    result = read(output)
    # A hybrid (model acceptance + keyword final offer) is never labelled as a plain model run.
    assert result["complete"] and result["evidence_class"] == "real_model_run_composed_estimates"
    assert result["estimate_composition"]["estimate_sources"] == {"acceptance": "model", "final_offer": "keyword"}
    assert not result["agent_is_rule_policy"]
    assert "rule_policy" in result["policies"] and "rule_policy" in result["agent_minus"]
    assert result["backend"]["kind"] == "composed_estimates" and result["backend"]["is_model"] is True
    assert result["backend"]["inner"] == {"kind": CountingModel.kind, "is_model": True}
    assert result["backend"]["model_estimates_used"] == ["acceptance"]
    sent = attempts(result)
    assert len(built) == 1 and built[0].calls == len(sent) > 0
    for call in sent:
        block = call["response"]["composition"]
        assert call["backend"] == "composed_estimates" and call["is_model"] is True
        assert block["estimate_sources"] == {"acceptance": "model", "final_offer": "keyword"}
        assert block["final_offer_value_from"] in ("keyword_match", "round_prior")
        assert call["response"]["metadata"]["request_sha256"] == call["request_sha256"]   # inner evidence kept
    recorded = result["provenance"]["execution_config"]
    assert recorded["real_model_requested"] is True and recorded["usage_config"]["estimates"] == MODEL_KEYWORD.to_json()
    kept = output.read_bytes(), decisions.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):
        run(*argv)
    assert len(built) == 1 and (output.read_bytes(), decisions.read_bytes()) == kept


def test_the_files_policy_drives_the_decision_and_the_composition(tmp_path, run, monkeypatch):
    monkeypatch.setattr(E, "make_backend", lambda _args: CountingModel())
    table = [0.05, 0.3, 0.6, 0.9]
    policy = {**DEFAULT_POLICY, "prior_concession": 300_000, "final_prior_by_round": table}
    config = write_config(tmp_path / "config.json", MODEL_SHRINK, policy)
    output = tmp_path / "model.json"
    assert run("evaluate", "--split", "dev", "--cases", "6", *ZORO, "--config-file", config, "--output", output) == 0
    result = read(output)
    assert result["config"]["policy"] == policy == result["provenance"]["execution_config"]["policy"]
    assert result["backend"]["prior_table"] == table
    checked = 0
    for case in result["cases"]:
        for record in case["agent"]["internal_record"]:
            model = record["model"]
            if "trace_id" not in model:
                continue
            block, limit = model["response"]["composition"], record["facts"]["rounds_left_after_counter"] == 0
            prior = 1.0 if limit else table[min(record["round"], len(table)) - 1]
            assert block["round_prior"] == prior
            value = shrink(block["raw_model_predictions"][FINAL_OFFER_QID], prior, 0.25)
            assert model["predictions"][FINAL_OFFER_QID] == value
            if not limit:
                assert record["facts"]["p_final_offer"] == value
                assert record["facts"]["p_final_offer_source"] == "backend_estimate"
                checked += 1
            if record["facts"]["supplier_concession_observed"] is None:
                assert record["facts"]["supplier_concession_assumed"] == 300_000
    assert checked > 0
    # The same constants reach every baseline of the run, as --prior-concession always did.
    plain = E.evaluate("dev", CountingModel(), PolicyConfig(prior_concession=300_000, final_prior_by_round=tuple(table)),
                       count=6)
    for name in ("rule_policy", "fixed_concession", "midpoint", E.REFERENCE_POLICY):
        assert outcomes(result, name) == outcomes(plain, name), name


def test_a_map_inside_the_file_equals_the_calibration_option(tmp_path, run):
    platt = tmp_path / "platt.json"
    platt.write_text(json.dumps(PLATT), encoding="utf-8")
    config = write_config(tmp_path / "config.json", EstimateSpec("model", None, dict(PLATT), "model"))
    by_option, by_file = tmp_path / "option.json", tmp_path / "file.json"
    common = ["evaluate", "--split", "dev", "--cases", "12", "--backend", "keyword"]
    assert run(*common, "--calibration", platt, "--output", by_option) == 0
    assert run(*common, "--config-file", config, "--output", by_file) == 0
    option, file = read(by_option), read(by_file)
    assert outcomes(file) == outcomes(option)
    assert [a["predictions"] for a in attempts(file)] == [a["predictions"] for a in attempts(option)]
    assert file["backend"]["acceptance_platt_applied"] is True and "calibration" not in file["backend"]
    assert file["provenance"]["execution_config"]["calibration"] is None
    assert file["provenance"]["execution_config"]["usage_config"]["estimates"]["acceptance_platt"] == PLATT


def test_a_map_is_fitted_only_on_an_unmapped_backend_acceptance_estimate(tmp_path, run):
    fitted = tmp_path / "platt.json"
    for name, spec, allowed in (("identity", IDENTITY, True), ("keyword_final", MODEL_KEYWORD, True),
                                ("constant", CONSTANT_KEYWORD, False),
                                ("mapped", EstimateSpec("model", None, dict(PLATT), "model"), False)):
        output = tmp_path / f"{name}.json"
        assert run("evaluate", "--split", "calibration", "--cases", "12", "--backend", "keyword",
                   "--config-file", write_config(tmp_path / f"{name}.config.json", spec), "--output", output) == 0
        if allowed:
            assert run("fit-calibration", "--input", output, "--output", fitted) == 0
            assert read(fitted)["fitted_on_split"] == "calibration"
            assert read(fitted)["backend_kind"] == "composed_estimates"
            fitted.unlink()
        else:
            with pytest.raises(SystemExit, match="uncalibrated run whose acceptance estimate is the backend's own"):
                run("fit-calibration", "--input", output, "--output", fitted)
            assert not fitted.exists()
    assert not run.history.exists()


# ---- freeze and the held-out protocol ---------------------------------------------------------------

def test_freeze_with_a_config_then_a_matching_evaluate_on_the_dev_split(tmp_path, run):
    config = write_config(tmp_path / "selected_config.json", CONSTANT_KEYWORD,
                          {**DEFAULT_POLICY, "reserve_margin": 40_000}, operator_note={"rule": "test"})
    manifest, output = tmp_path / "freeze.json", tmp_path / "dev.json"
    common = ["--cases", "6", "--config-file", config]
    assert run("freeze", *common, "--output", manifest) == 0
    frozen = read(manifest)
    assert frozen["evaluation_executed"] is False
    assert frozen["execution_config"]["usage_config"]["content"] == read(config)
    assert frozen["execution_config"]["usage_config"]["file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert frozen["execution_config"]["policy"]["reserve_margin"] == 40_000
    assert frozen["provenance"]["execution_config"] == frozen["execution_config"]
    assert not run.history.exists()                                # freezing runs no case
    assert run("evaluate", "--split", "dev", *common, "--output", output) == 0
    result = read(output)
    assert result["complete"] and result["evaluation_mode"] == "development"
    assert result["provenance"]["execution_config"] == frozen["execution_config"]
    assert result["config"]["policy"]["reserve_margin"] == 40_000
    assert not run.history.exists()


def test_heldout_run_needs_the_frozen_file_unchanged(tmp_path, run, throwaway_heldout):
    config = write_config(tmp_path / "config.json", CONSTANT_KEYWORD)
    original = config.read_bytes()
    manifest, output = tmp_path / "freeze.json", tmp_path / "heldout.json"
    assert run("freeze", "--cases", "4", "--config-file", config, "--output", manifest) == 0
    heldout = ["evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", manifest, "--output", output]

    with pytest.raises(SystemExit, match="changed"):               # the file was left out
        run(*heldout)
    write_config(config, CONSTANT_PRIOR)                           # another estimate source
    with pytest.raises(SystemExit, match="changed"):
        run(*heldout, "--config-file", config)
    write_config(config, CONSTANT_KEYWORD, {**DEFAULT_POLICY, "prior_concession": 90_000})   # another constant
    with pytest.raises(SystemExit, match="changed"):
        run(*heldout, "--config-file", config)
    write_config(config, CONSTANT_KEYWORD, note="an extra key added after the freeze")
    with pytest.raises(SystemExit, match="changed"):
        run(*heldout, "--config-file", config)
    config.write_text(json.dumps(json.loads(original.decode("utf-8"))), encoding="utf-8")    # only the layout
    assert config.read_bytes() != original
    with pytest.raises(SystemExit, match="changed"):               # the frozen bytes are what was pinned
        run(*heldout, "--config-file", config)
    assert not run.history.exists() and not output.exists()
    assert not manifest.with_name(manifest.name + ".consumed.json").exists()

    config.write_bytes(original)
    assert run(*heldout, "--config-file", config) == 0
    result = read(output)
    assert result["evaluation_mode"] == "heldout" and result["protocol"]["mode"] == "heldout"
    assert result["provenance"]["execution_config"] == read(manifest)["execution_config"]
    assert result["backend"]["estimate_sources"] == CONSTANT_KEYWORD.estimate_sources()
    assert len(list(run.history.glob("*.json"))) == 1
    with pytest.raises(SystemExit, match="already has an evaluation attempt"):
        run(*heldout, "--config-file", config)


def test_a_freeze_without_a_config_does_not_admit_a_run_with_one(tmp_path, run, throwaway_heldout):
    manifest, output = tmp_path / "freeze.json", tmp_path / "heldout.json"
    assert run("freeze", "--cases", "4", "--output", manifest) == 0
    config = write_config(tmp_path / "config.json", CONSTANT_PRIOR)
    with pytest.raises(SystemExit, match="changed"):
        run("evaluate", "--split", "eval", "--cases", "4", "--freeze-manifest", manifest,
            "--config-file", config, "--output", output)
    assert not run.history.exists() and not output.exists()


# ---- refusals ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["evaluate", "freeze"])
@pytest.mark.parametrize("extra, message", [
    (["--calibration", "PLATT"], "cannot be combined with --calibration"),
    (["--prior-concession", "90000"], "cannot be combined with --prior-concession"),
    (["--prior-concession", "90000", "--calibration", "PLATT"], "cannot be combined with"),
])
def test_a_second_source_of_policy_or_map_is_a_conflict(tmp_path, run, monkeypatch, command, extra, message):
    forbid(monkeypatch, "_build_backend", "freeze", "reserve_heldout", "record_exploratory", "evaluate",
           "load_usage_config", "_load_calibration")
    platt = tmp_path / "platt.json"
    platt.write_text(json.dumps(PLATT), encoding="utf-8")
    config = write_config(tmp_path / "config.json", CONSTANT_PRIOR)
    extra = [str(platt) if item == "PLATT" else item for item in extra]
    with pytest.raises(SystemExit, match=message):
        run(command, "--cases", "2", "--config-file", config, *extra, "--output", tmp_path / "out.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "platt.json"]
    assert not run.history.exists()


def test_the_default_prior_concession_written_out_is_not_a_conflict(tmp_path, run):
    config = write_config(tmp_path / "config.json", CONSTANT_PRIOR)
    manifest = tmp_path / "freeze.json"
    assert run("freeze", "--cases", "2", "--prior-concession", PolicyConfig().prior_concession,
               "--config-file", config, "--output", manifest) == 0
    assert read(manifest)["execution_config"]["policy"] == DEFAULT_POLICY


def _bad(**changes):
    content = {"policy": dict(DEFAULT_POLICY), "estimates": CONSTANT_PRIOR.to_json()}
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if not field:
            content[section] = value
        else:
            content[section][field] = value
    return json.dumps(content)


BAD_FILES = {
    "not_json": ("{not json", [], "not a JSON file"),
    "utf16_from_a_shell_redirect": (_bad().encode("utf-16"), [], "not a JSON file"),
    "top_level_list": ("[1, 2]", [], "one JSON object"),
    "nan_value": ('{"policy": {}, "estimates": {}, "note": NaN}', [], "not a JSON file"),
    "missing_estimates": (json.dumps({"policy": {}}), [], "missing top-level keys"),
    "missing_policy": (json.dumps({"estimates": {}}), [], "missing top-level keys"),
    "policy_not_an_object": (_bad(policy=[1]), [], "must be a JSON object"),
    "unknown_policy_field": (_bad(policy__aggressiveness=3), [], "unknown policy fields"),
    "policy_bool": (_bad(policy__prior_concession=True), [], "must be an integer"),
    "policy_float": (_bad(policy__reserve_margin=30000.5), [], "must be an integer"),
    "policy_negative": (_bad(policy__prior_concession=-1), [], "must be an integer"),
    "no_candidates": (_bad(policy__max_candidates=0), [], "at least 1"),
    "prior_table_not_a_list": (_bad(policy__final_prior_by_round=0.5), [], "non-empty list of probabilities"),
    "prior_table_empty": (_bad(policy__final_prior_by_round=[]), [], "non-empty list of probabilities"),
    "prior_table_out_of_range": (_bad(policy__final_prior_by_round=[0.0, 1.5]), [], "non-empty list of probabilities"),
    "estimates_not_an_object": (_bad(estimates="model"), [], "must be a JSON object"),
    "unknown_estimates_key": (_bad(estimates__temperature=2.0), [], "unknown estimate spec keys"),
    "unknown_source": (_bad(estimates__final_source="oracle"), [], "final_source must be one of"),
    "constant_without_a_value": (_bad(estimates__acceptance_constant=None), [], "needs acceptance_constant"),
    "shrink_without_a_weight": (_bad(estimates__acceptance_source="model", estimates__acceptance_constant=None,
                                     estimates__final_source="shrink"), ["--backend", "keyword"],
                                "needs final_shrink_weight"),
    "map_fitted_on_the_development_split": (
        _bad(estimates={"acceptance_platt": {"a": 1.0, "b": 0.0, "fitted_on_split": "dev"}}),
        ["--backend", "keyword"], "fitted on the calibration split"),
    "map_without_a_split": (_bad(estimates={"acceptance_platt": {"a": 1.0, "b": 0.0}}), ["--backend", "keyword"],
                            "fitted on the calibration split"),
    "final_estimate_the_rule_backend_lacks": (_bad(estimates=IDENTITY.to_json()), [], "does not supply"),
    "shrink_over_the_rule_backend": (_bad(estimates=MODEL_SHRINK.to_json()), ["--backend", "heuristic"],
                                     "does not supply"),
    "model_free_with_the_native_model": (_bad(), ZORO, "uses no backend estimate"),
    "model_free_with_the_decision_server": (
        _bad(), ["--backend", "zoro-http", "--url", "http://127.0.0.1:30001"], "uses no backend estimate"),
    "model_free_with_the_legacy_server": (
        _bad(estimates=EstimateSpec("rule", None, None, "keyword").to_json()),
        ["--backend", "systemone", "--url", "http://127.0.0.1:30001"], "uses no backend estimate"),
}


@pytest.mark.parametrize("command", ["evaluate", "freeze"])
@pytest.mark.parametrize("name", sorted(BAD_FILES))
def test_an_unusable_file_is_refused_before_anything_is_written(tmp_path, run, monkeypatch, throwaway_heldout,
                                                                command, name):
    forbid(monkeypatch, "_build_backend", "make_backend", "freeze", "reserve_heldout", "record_exploratory", "evaluate")
    content, backend, message = BAD_FILES[name]
    config = tmp_path / "config.json"
    config.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    split = ["--split", "eval", "--evaluation-mode", "exploratory"] if command == "evaluate" else []
    with pytest.raises(SystemExit, match="--config-file") as refused:
        run(command, *split, "--cases", "2", *backend, "--config-file", config, "--output", tmp_path / "out.json")
    assert message in str(refused.value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json"]
    assert not run.history.exists()


@pytest.mark.parametrize("command", ["evaluate", "freeze"])
def test_a_missing_file_is_refused_before_anything_is_written(tmp_path, run, monkeypatch, command):
    forbid(monkeypatch, "_build_backend", "make_backend", "freeze", "reserve_heldout", "record_exploratory", "evaluate")
    with pytest.raises(SystemExit, match="--config-file.*FileNotFoundError"):
        run(command, "--cases", "2", "--config-file", tmp_path / "absent.json", "--output", tmp_path / "out.json")
    assert list(tmp_path.iterdir()) == [] and not run.history.exists()


def test_model_configs_are_accepted_for_every_real_backend_choice_at_freeze(tmp_path, run):
    # Freezing builds no backend, so the recorded settings can be checked without weights or a server.
    config = write_config(tmp_path / "config.json", MODEL_SHRINK)
    for index, backend in enumerate((ZORO, ["--backend", "zoro-http", "--url", "http://127.0.0.1:30001"],
                                     ["--backend", "systemone", "--url", "http://127.0.0.1:30001"])):
        manifest = tmp_path / f"freeze{index}.json"
        assert run("freeze", "--cases", "2", *backend, "--config-file", config, "--output", manifest) == 0
        recorded = read(manifest)["execution_config"]
        assert recorded["real_model_requested"] is True and recorded["backend"] == backend[1]
        assert recorded["usage_config"]["estimates"] == MODEL_SHRINK.to_json()


def test_a_backend_that_cannot_be_composed_is_closed_and_the_failure_is_recorded(tmp_path, run, monkeypatch):
    class NoFinal(CountingModel):
        supports_final_offer = False

    built = []

    def build(_args):
        built.append(NoFinal())
        return built[-1]

    monkeypatch.setattr(E, "make_backend", build)
    output = tmp_path / "failure.json"
    assert run("evaluate", "--cases", "2", *ZORO, "--config-file", write_config(tmp_path / "config.json", MODEL_SHRINK),
               "--output", output) == 1
    result = read(output)
    assert result["status"] == "initialization_failed" and not result["complete"]
    assert "does not supply" in result["error"] and "evidence_class" not in result
    assert built[0].closed == 1 and built[0].calls == 0


# ---- wiring -----------------------------------------------------------------------------------------

def test_build_backend_wraps_only_when_a_usage_config_is_given(tmp_path, monkeypatch):
    inner = CountingModel()
    monkeypatch.setattr(E, "make_backend", lambda _args: inner)
    args = SimpleNamespace(backend="zoro")
    assert E._build_backend(args, None) is inner and E._build_backend(args, None, None) is inner
    usage = E.load_usage_config(write_config(tmp_path / "config.json", MODEL_SHRINK,
                                             {"final_prior_by_round": [0.1, 0.2]}), "zoro")
    composed = E._build_backend(args, None, usage)
    assert isinstance(composed, ComposedBackend) and composed.inner is inner
    assert composed.spec == MODEL_SHRINK and composed.prior_table == (0.1, 0.2)
    assert composed.is_model and composed.kind == "composed_estimates"


def test_backend_choices_that_answer_the_final_offer_question():
    assert [name for name in ("constant", "heuristic", "keyword", "zoro", "zoro-http", "systemone")
            if E.backend_supplies_final_offer(name)] == ["constant", "keyword", "zoro", "zoro-http", "systemone"]


def test_headers_say_the_code_has_not_run_on_a_gpu():
    assert "NOT executed on a GPU as of 2026-11-18" in Path(__file__).read_text(encoding="utf-8")[:400]
    source = Path(E.__file__).read_text(encoding="utf-8")
    section = source[source.index("# ---- usage configuration file (--config-file)"):]
    assert "NOT executed on a GPU as of 2026-11-18" in section[:400]


# ---- origin of a configuration file (review fix of 2026-11-18) ---------------------------------------------

def test_a_file_from_the_tuning_runner_is_checked_against_the_selection_it_names(tmp_path, run, capsys):
    """Uses the runner's own selected_config.json from a CPU rehearsal (keyword stand-in, shortened blocks)."""
    from negotiation_agent import tune
    state = tmp_path / "state"
    blocks = "d0=3,select=6,confirm=8,cal=20"
    assert tune.main(["plan", "--state-dir", str(state), "--rehearsal-blocks", blocks]) == 0
    assert tune.main(["run", "--stage", "all", "--state-dir", str(state), "--backend", "keyword"]) == 0
    selected = state / "selected_config.json"
    content = read(selected)
    assert content["selected_from"]["counts_as_model_evidence"] is False
    capsys.readouterr()
    # CPU backend: accepted, and the origin says the estimates equal the selection the file names.
    output = tmp_path / "dev.json"
    assert run("evaluate", "--split", "dev", "--cases", "3", "--backend", "keyword",
               "--config-file", selected, "--output", output) == 0
    usage = read(output)["provenance"]["execution_config"]["usage_config"]
    assert usage["origin"] == "tuning_selection_digest_matches"
    assert content["selected_from"]["config_id"].endswith("-" + usage["config_sha256"][:12])
    assert "origin=tuning_selection_digest_matches" in capsys.readouterr().err
    # A rehearsal selection is refused for a real-model run, at evaluate and at freeze, before anything is written.
    for command in ("evaluate", "freeze"):
        target = tmp_path / f"{command}-real.json"
        extra = ["--split", "dev"] if command == "evaluate" else []
        with pytest.raises(SystemExit, match="CPU rehearsal or on shortened blocks"):
            run(command, *extra, *ZORO, "--config-file", selected, "--output", target)
        assert not target.exists()
    # Edited after the selection: the named selection no longer matches the estimates.
    edited = tmp_path / "edited.json"
    changed = json.loads(selected.read_text(encoding="utf-8"))
    changed["policy"]["reserve_margin"] = changed["policy"]["reserve_margin"] + 5_000
    edited.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="edited after the selection"):
        run("freeze", "--backend", "keyword", "--config-file", edited, "--output", tmp_path / "edited-freeze.json")
    # The same content without selected_from is an operator-written file: accepted, origin unverified.
    del changed["selected_from"]
    edited.write_text(json.dumps(changed), encoding="utf-8")
    assert run("freeze", "--backend", "keyword", "--config-file", edited, "--output", tmp_path / "operator-freeze.json") == 0
    frozen = read(tmp_path / "operator-freeze.json")
    assert "operator_file_origin_unverified" in json.dumps(frozen)

