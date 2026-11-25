"""CPU tests of the utterance experiment's shared runner and of the reading bench (track A).

NOT EXECUTED ON A GPU as of 2026-11-18. Every "model" here is the CPU stand-in or a test double
flagged as a model only so that the labelling paths of a real-model run can be exercised. No
number in this file is model evidence.

Case use: real cases come from the smoke block only (utt-6000..6009) and from dev-1100. The
reading rule, the guard check and the threshold fit are tested with invented answers; nothing is
computed on the fit, bench or test block.
"""

import hashlib
import importlib.util
import json
import shlex
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from negotiation_agent import policy
from negotiation_agent.backends import ConstantBackend, HeuristicBackend, KeywordBackend
from negotiation_agent.evaluation import make_cases
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.prediction_store import _refuse_locations, identity_digest
from negotiation_agent.requests import FINAL_OFFER_QID, build_decision_request, public_view
from negotiation_agent.simulator import NEUTRAL_MESSAGE, default_buyer_private, initial_state
from negotiation_agent.utterance_probe import _bootstrap_ci

from utterance_agent import PROTOCOL_VERSION
from utterance_agent import bench as B
from utterance_agent import cases as C
from utterance_agent import experiment as X
from utterance_agent import wording as W
from utterance_agent.interpreter import FIXED_HALF, Thresholds, combine
from utterance_agent.questions import QIDS, build_existing_final_request, build_utterance_request

NEGOTIATION = Path(__file__).resolve().parents[1]
DOCUMENT = (NEGOTIATION / "docs" / "UTTERANCE_AGENT_DESIGN.md").read_text(encoding="utf-8")
NEG_ARGS = ("--backend zoro --original-checkpoint /content/model/original --exported-checkpoint "
            "/content/model/exported --official-source-dir /content/trida --device cuda:0 --dtype fp16 "
            "--max-input-tokens 2048 --decision-temperature 1")       # the shape of the notebook's NEG_ARGS
# Re-pinned on purpose before any first use: the tau_inst fit rule now leaves out the held-out item
# (section 6.2), so the protocol text changed (previous value a2775cfb...6cb4).
PINNED_PROTOCOL_SHA256 = "9329914ca4c85da82e5ae18e77030285459c774c0672ec683e437d6277a0cebf"
INSTRUCTION_ITEMS = ("neutral_with_instruction_to_final", "final_with_instruction_to_open")
FINAL_ITEMS = ("explicit_final", "final_limit", "final_walk", "final_with_payment",
               "final_with_instruction_to_open", "probe_paraphrase_final")


@pytest.fixture(autouse=True)
def _private_registry(tmp_path, monkeypatch):
    """No test may write the checkout's real first-use registry."""
    monkeypatch.setattr(X, "REGISTRY_PATH", tmp_path / "registry.json")


def _stand_in(state: Path, *extra: str) -> list[str]:
    return ["--state-dir", str(state), "--backend", "cpu-stand-in", *extra]


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    """One CPU rehearsal of the smoke bench: plan, then a read-write run through the store."""
    root = tmp_path_factory.mktemp("ua")
    state, registry = root / "s", root / "registry.json"
    assert X.main(["plan", "--state-dir", str(state)], registry_path=registry) == 0
    assert X.main(["bench", "--block", "smoke", *_stand_in(state, "--store-mode", "read-write")],
                  registry_path=registry) == 0
    return SimpleNamespace(state=state, registry=registry,
                           first=X.attempts(state / "bench" / "smoke")[0]["result"])


def _copy(rehearsal, tmp_path) -> Path:
    target = tmp_path / "c"
    shutil.copytree(rehearsal.state, target)
    return target


class ModelDouble(X.UtteranceCpuStandIn):
    """The CPU stand-in's values behind a declared-model test double. It is NOT a model."""

    kind, is_model = "utterance_test_double_model", True
    DECLARED = ("utterance_test_double_model", True, True)

    def predict(self, state, candidates, request):
        out = super().predict(state, candidates, request)
        self.last_response = {"metadata": {"dtype": "test_only",
                                           "input_tokens_per_question": [7] * len(request["questions"])}}
        return out


def _dev_state():
    case = make_cases("dev", 1)[0]
    return initial_state(case.case_id, case.hidden)


# ---- invented bench results ----------------------------------------------------------------------------

def _fake_bench(states: int, signal=None, instruction=None, existing=None, block: str = "fit",
                complete: bool = True) -> dict:
    """A bench result with invented answers. `signal(name, i)` is u_final; the default is a clean reader."""
    signal = signal or (lambda name, i: 1.0 if name in FINAL_ITEMS else 0.0)
    instruction = instruction or (lambda name, i: 0.9 if name in INSTRUCTION_ITEMS else 0.0)
    existing = existing or (lambda name, i: 0.0)
    items = []
    for i in range(states):
        for name in X.BENCH_ITEMS:
            items.append({
                "state_id": f"invented-{i:03d}", "class": name, "round": 1 + i % 3, "status": "ok",
                "answers": {"u_final": signal(name, i), "u_walkaway": 0.0, "u_concession": 0.0,
                            "u_instruction": instruction(name, i)},
                "existing": {"status": "ok", "final_offer": existing(name, i)},
                "literal_answers": dict.fromkeys(QIDS, 0.0),
                "labels": {"has_instruction": name in INSTRUCTION_ITEMS}})
    return {"block": {"name": block}, "complete": complete, "status": "complete" if complete else "partial",
            "items": items, "thresholds": {"existing_question": None}}


def _tie_values(name: str, i: int) -> float:
    """34 states. Steps 6 (0.30) and 14 (0.70) get the same best score; every step between is worse.

    Half of the 102 recall items answer 0.95 and half 0.32; 25 of the 170 false-fire items answer 0.69.
    score(step <= 6) = 1 - 3.4 * 25/170 = 0.5, score(7..13) = 0.5 - 0.5 = 0, score(14..19) = 0.5.
    """
    if name == "final_limit":
        return 0.95 if i % 2 == 0 else 0.32
    if name == "final_walk":
        return 0.32 if i % 2 == 0 else 0.95
    if name == "final_with_payment":
        return 0.95 if i < 17 else 0.32
    if name == "open":
        return 0.69 if i < 25 else 0.0
    return 0.0


# ---- arguments and refusals ----------------------------------------------------------------------------

def test_parser_has_no_http_backend_reasoning_or_url_option(tmp_path):
    base = ["bench", "--block", "smoke", "--state-dir", str(tmp_path / "s")]
    good = base + shlex.split(NEG_ARGS)
    assert X.build_parser().parse_args(good).backend == "zoro"
    for argv in (base + ["--backend", "zoro-http"], base + ["--backend", "systemone"],
                 base + ["--backend", "keyword"], good + ["--reasoning"], good + ["--url", "http://localhost:1"],
                 base):                                           # --backend has no default
        with pytest.raises(SystemExit):
            X.build_parser().parse_args(argv)
    assert X.BACKENDS == ("cpu-stand-in", "zoro")
    demo = ["table", "--state-dir", str(tmp_path / "s"), "--seed", "1073"]
    assert X.build_demo_parser().parse_args(demo + shlex.split(NEG_ARGS)).backend == "zoro"
    for extra in (["--backend", "zoro-http"], shlex.split(NEG_ARGS) + ["--reasoning"],
                  shlex.split(NEG_ARGS) + ["--url", "http://localhost:1"]):
        with pytest.raises(SystemExit):
            X.build_demo_parser().parse_args(demo + extra)
    assert not (tmp_path / "s").exists()


def test_other_temperatures_are_refused_before_anything_is_written(tmp_path, capsys):
    state, built = tmp_path / "s", []
    checkpoints = ["--original-checkpoint", "a", "--exported-checkpoint", "b"]
    for extra in ([], ["--decision-temperature", "0.7"], ["--decision-temperature", "0"]):
        argv = ["bench", "--block", "smoke", "--state-dir", str(state), "--backend", "zoro", *checkpoints, *extra]
        assert X.main(argv, backend_factory=lambda: built.append(1)) == 1
        assert "Refused" in capsys.readouterr().err
    # read-write without checkpoints is refused too; nothing was written and no backend was built
    assert X.main(["bench", "--block", "smoke", "--state-dir", str(state), "--backend", "zoro",
                   "--decision-temperature", "1"], backend_factory=lambda: built.append(1)) == 1
    assert not state.exists() and not built and not (tmp_path / "registry.json").exists()

    def args(**changes):
        base = dict(backend="zoro", decision_temperature=1.0, store_mode="read-write",
                    original_checkpoint="a", exported_checkpoint="b")
        return SimpleNamespace(**{**base, **changes})

    X.check_arguments(args())
    X.check_arguments(args(store_mode="read-only", original_checkpoint=None, exported_checkpoint=None))
    X.check_arguments(SimpleNamespace(backend="cpu-stand-in", store_mode="read-write"))
    for refused in (args(decision_temperature=None), args(decision_temperature=0.99), args(decision_temperature=2),
                    args(original_checkpoint=None), args(store_mode="append"), args(reasoning=True),
                    args(url="http://localhost:30001"), args(backend="zoro-http"), args(backend="systemone"),
                    args(backend="keyword"), SimpleNamespace(store_mode="read-write"),
                    SimpleNamespace(backend="cpu-stand-in", store_mode="read-write", reasoning=True)):
        with pytest.raises(X.ExperimentRefusal):
            X.check_arguments(refused)


def test_store_identity_reads_reasoning_from_the_arguments_and_marks_the_stand_in():
    def args(**changes):
        base = dict(backend="zoro", dtype="fp16", max_input_tokens=2048, decision_temperature=1.0)
        return SimpleNamespace(**{**base, **changes})

    plain, reasoned = X.store_requested(args()), X.store_requested(args(reasoning=True))
    assert plain["reasoning"] is False and reasoned["reasoning"] is True
    assert identity_digest(plain) != identity_digest(reasoned)
    assert set(plain) == {"experiment", "backend", "backend_kind", "requests_sha256",
                          "utterance_questions_sha256", "adapter_file", "adapter_sha256", "model_revision",
                          "source_revision", "dtype", "max_input_tokens", "decision_temperature", "reasoning"}
    assert (plain["backend"], plain["backend_kind"], plain["decision_temperature"]) == ("zoro", "zoro_native", 1.0)
    assert plain["model_revision"] and plain["source_revision"]
    stand_in = X.store_requested(SimpleNamespace(backend="cpu-stand-in"))
    assert stand_in["cpu_stand_in_not_a_model"] is True and "cpu_stand_in_not_a_model" not in plain
    assert stand_in["model_revision"] is None and stand_in["decision_temperature"] is None
    assert identity_digest(stand_in) != identity_digest(plain)
    assert identity_digest(X.store_requested(args(dtype="bf16"))) != identity_digest(plain)
    # No path and no URL: where the checkpoints are does not change the identity.
    located = X.store_requested(args(original_checkpoint="/content/model/original",
                                     exported_checkpoint="C:/models/exported", official_source_dir="/content/src"))
    assert located == plain
    for requested in (plain, reasoned, stand_in):
        _refuse_locations(requested)                              # raises on a URL or an absolute path
        assert "url" not in json.dumps(requested).lower() and "content" not in json.dumps(requested)


def test_cpu_stand_in_answers_utterance_and_default_requests():
    state = _dev_state()
    stand_in = X.UtteranceCpuStandIn()
    assert (stand_in.kind, stand_in.is_model, stand_in.supports_final_offer) == ("utterance_cpu_stand_in", False, True)
    utterance = build_utterance_request(state)
    answers = stand_in.predict(state, [], utterance)
    assert tuple(answers) == QIDS and all(value in (0.0, 1.0) for value in answers.values())
    final_state = B.with_latest_supplier_message(state, W.bank_sentence("W1")[1])
    assert stand_in.predict(final_state, [], build_utterance_request(final_state)) == {
        "u_final": 0.0, "u_walkaway": 1.0, "u_concession": 0.0, "u_instruction": 0.0}
    outside = B.with_latest_supplier_message(state, "은행에 없는 문장입니다.")
    with pytest.raises(ValueError):
        stand_in.predict(outside, [], build_utterance_request(outside))
    # the default request and the existing-question request get the literal keyword reference's values
    candidates = policy.generate_candidates(state, default_buyer_private(), PolicyConfig())
    default = build_decision_request(state, candidates)
    assert stand_in.predict(state, candidates, default) == KeywordBackend().predict(state, candidates, default)
    existing = build_existing_final_request(state)
    assert stand_in.predict(state, [], existing) == KeywordBackend().predict(state, [], existing)
    assert list(stand_in.predict(state, [], existing)) == [FINAL_OFFER_QID]
    # Why a stand-in is needed at all: the existing CPU backends cannot answer the utterance request.
    assert HeuristicBackend().predict(state, [], utterance) == {}
    assert KeywordBackend().predict(state, [], utterance) == {}
    constant = ConstantBackend().predict(state, [], utterance)
    assert set(constant.values()) == {0.5}
    assert combine(constant, 0.0, FIXED_HALF)[1] == "instruction_flagged_round_prior"


# ---- bench ---------------------------------------------------------------------------------------------

def test_bench_rehearsal_goes_through_the_store_and_replays_without_live_requests(rehearsal, tmp_path):
    first = rehearsal.first
    assert first["evidence_class"] == "cpu_rehearsal_not_model_evidence"
    assert first["complete"] is True and first["status"] == "complete"
    assert first["counts"] == {"live_requests": 241, "live_failed_requests": 0, "store_hits": 0, "planned_requests": 241,
                               "utterance_requests": 121, "existing_question_requests": 120}
    assert first["verdict"]["counts_as_model_evidence"] is False
    assert first["protocol_version"] == PROTOCOL_VERSION and first["experiment"] == "utterance_reading_bench"
    assert first["block"]["name"] == "smoke" and first["block"]["first_case_id"] == "utt-6000"
    assert first["backend"]["prediction_store"]["requested"]["cpu_stand_in_not_a_model"] is True
    assert set(first["provenance"]["utterance_agent_inputs"]) >= {
        "negotiation/src/utterance_agent/questions.py", "negotiation/src/utterance_agent/bench.py",
        "negotiation/utterance_experiment.py"}
    assert not rehearsal.registry.exists() and not (rehearsal.state / "first_use.json").exists()

    state = _copy(rehearsal, tmp_path)
    assert X.main(["bench", "--block", "smoke", *_stand_in(state, "--store-mode", "read-only")]) == 0
    runs = X.attempts(state / "bench" / "smoke")
    assert len(runs) == 2                                           # a new attempt folder; nothing overwritten
    replay = runs[1]["result"]
    assert replay["counts"]["live_requests"] == 0 and replay["counts"]["store_hits"] == 241
    assert replay["evidence_class"] == "cpu_rehearsal_not_model_evidence"
    assert replay["attempt"]["status"] == "not_confirmatory" and first["attempt"]["status"] == "first_use"
    strip = lambda item: {k: v for k, v in item.items() if k not in ("latency_ms", "store_hit", "existing")}  # noqa: E731
    assert [strip(item) for item in replay["items"]] == [strip(item) for item in first["items"]]
    assert replay["readers"] == first["readers"] and replay["verdict"]["reading"] == first["verdict"]["reading"]
    # a read-only store without answers asks nothing and reports every item as failed, filling in no value
    empty = tmp_path / "empty"
    assert X.main(["bench", "--block", "smoke", "--limit", "1", *_stand_in(empty, "--store-mode", "read-only")]) == 1
    missed = X.attempts(empty / "bench" / "smoke")[0]["result"]
    assert missed["complete"] is False and missed["counts"]["live_requests"] == 0
    assert all(item["status"] == "error" and "answers" not in item for item in missed["items"])
    assert missed["verdict"]["reading"] == "incomplete_no_reading" and missed["guard"]["result"] is None
    # the stand-in is a rehearsal: it is never run on the fit or bench block, and only smoke takes --limit
    for argv in (["bench", "--block", "fit", *_stand_in(state)], ["bench", "--block", "bench", *_stand_in(state)]):
        assert X.main(argv) == 1
    assert not (state / "bench" / "fit").exists() and not (state / "bench" / "bench").exists()
    assert not (state / "first_use.json").exists()


def test_bench_items_are_paired_within_state():
    items = B.bench_items("smoke")
    assert len(items) == 10 * 12
    states: dict[str, list[dict]] = {}
    for item in items:
        states.setdefault(item["state_id"], []).append(item)
    assert len(states) == 10
    combinations = set()
    for index, (state_id, rows) in enumerate(states.items()):
        assert [row["class"] for row in rows] == list(X.BENCH_ITEMS)
        assert state_id == f"utt-{6000 + index}-d{index % 3}"
        by_class = {row["class"]: row for row in rows}
        # only the latest supplier message differs between the twelve items of a state
        def without_latest(row):
            view = public_view(row["state"])
            latest = max(i for i, offer in enumerate(view["제안_이력"]) if offer["제안자"] == "공급사")
            assert view["제안_이력"][latest]["발언"] == row["message"]
            view["제안_이력"][latest] = {k: v for k, v in view["제안_이력"][latest].items() if k != "발언"}
            return view
        assert all(without_latest(row) == without_latest(rows[0]) for row in rows)
        assert len({row["message"] for row in rows}) == 12
        assert {row["round"] for row in rows} == {rows[0]["state"].current_round}
        # earlier supplier messages are the simulator's neutral sentence (a supplier that reveals nothing)
        earlier = [offer.message for offer in rows[0]["state"].history if offer.made_by == "supplier"][:-1]
        assert all(message == NEUTRAL_MESSAGE for message in earlier)
        # sentence = (ordinal // 3) % count, and paired items share their sentences
        pick = lambda name: W.BANK[name][(index // 3) % len(W.BANK[name])][0]   # noqa: E731
        expected = {
            "neutral": [pick("neutral")], "explicit_final": ["EF"], "final_limit": [pick("final_limit")],
            "final_walk": [pick("final_walk")], "final_with_payment": [pick("payment"), pick("final_limit")],
            "payment_only": [pick("payment")], "deadline_pressure": [pick("deadline_pressure")],
            "negated_keyword": [pick("negated_keyword")], "open": [pick("open")],
            "neutral_with_instruction_to_final": [pick("neutral"), "IF"],
            "final_with_instruction_to_open": [pick("final_limit"), "IO"],
            "probe_paraphrase_final": [W.HELD_OUT_FINAL[(index // 3) % 3][0]]}
        assert {name: row["sentence_ids"] for name, row in by_class.items()} == expected
        assert {name: row["labels"]["says_final"] for name, row in by_class.items()} == {
            name: name in FINAL_ITEMS for name in X.BENCH_ITEMS}
        assert {name: row["labels"]["has_instruction"] for name, row in by_class.items()} == {
            name: name in INSTRUCTION_ITEMS for name in X.BENCH_ITEMS}
        assert by_class["negated_keyword"]["labels"]["contains_keyword"] is True
        assert by_class["final_limit"]["labels"]["contains_keyword"] is False
        combinations.add((rows[0]["depth"], by_class["final_limit"]["sentence_ids"][0]))
    # depth and sentence do not move together: the first nine states give all nine (depth, sentence) pairs
    assert len(combinations) == 9
    assert {row["depth"] for row in items} == {0, 1, 2}
    assert [item["state_id"] for item in B.bench_items("smoke", 2)] == [item["state_id"] for item in items[:24]]
    for block, limit in (("test", None), ("bench", 5), ("fit", 5), ("reserve", None)):
        with pytest.raises(ValueError):
            B.bench_items(block, limit)


def test_longest_state_item_is_the_largest_smoke_state(rehearsal):
    check = B.longest_state_item()
    smoke_ids = [case.case_id for case in C.block_cases("smoke")]
    assert check["case_id"] in smoke_ids and check["state_id"] == f"{check['case_id']}-longest"
    longest = B.longest_adversarial_utterance()
    supplier = [offer.message for offer in check["state"].history if offer.made_by == "supplier"]
    assert supplier and set(supplier) == {longest} and check["supplier_messages"] == len(supplier)
    # it is a combination the adversarial style can produce: payment + one core sentence + one instruction
    labels = W.label_text(longest)
    assert labels.has_payment and labels.has_instruction and len(labels.sentence_ids) == 3
    for name in ("payment", "final_limit", "final_walk", "deadline_pressure", "negated_keyword", "open",
                 "instruction_to_final", "instruction_to_open", "instruction_to_accept"):
        for _, sentence in W.BANK[name]:
            if name == "payment":
                assert len(sentence.encode()) <= len(W.bank_sentence(labels.sentence_ids[0])[1].encode())
            elif name.startswith("instruction_"):
                assert len(sentence.encode()) <= len(W.bank_sentence(labels.sentence_ids[2])[1].encode())
            else:
                assert len(sentence.encode()) <= len(W.bank_sentence(labels.sentence_ids[1])[1].encode())
    # no smoke state is larger than the chosen one, and a five-round state is among them
    size = lambda state: len(json.dumps(build_utterance_request(state), ensure_ascii=False,   # noqa: E731
                                        separators=(",", ":")).encode("utf-8"))
    assert check["bytes"] == size(check["state"])
    assert check["bytes"] > max(size(item["state"]) for item in B.bench_items("smoke"))
    assert check["round"] == 5
    # it is asked once and stays out of every reading aggregate
    result = rehearsal.first
    recorded = result["longest_state_check"]
    assert recorded["state_id"] == check["state_id"] and recorded["status"] == "ok"
    assert recorded["bytes"] == check["bytes"] and "answers" not in recorded
    assert len(result["items"]) == 120 and all(item["class"] != B.LONGEST_CHECK_CLASS for item in result["items"])
    assert all(row["items"] == 10 for row in result["readers"]["utt_gate"]["by_item"].values())


# ---- threshold fit -------------------------------------------------------------------------------------

def test_fit_thresholds_integer_grid_tie_rule_and_cost_weight():
    assert X.THRESHOLD_STEPS == tuple(range(1, 20)) and X.FALSE_FIRE_WEIGHT == 3.4
    # (0.3, 0.7) tie: the two steps are equally far from 10, so the smaller one wins.
    fit = B.fit_thresholds(_fake_bench(34, signal=_tie_values))
    scores = {row["step"]: row["score"] for row in fit["final"]["steps"]}
    assert [row["step"] for row in fit["final"]["steps"]] == list(range(1, 20))
    assert [row["threshold"] for row in fit["final"]["steps"]] == [step / 20 for step in range(1, 20)]
    assert scores[6] == scores[14] == 0.5 and all(scores[step] == 0.0 for step in range(7, 14))
    assert (fit["final"]["step"], fit["final"]["threshold"]) == (6, 0.3)
    assert fit["status"] == "fitted" and fit["fitted_on"] == "utt-fit"
    # tau_inst: balanced accuracy is 1.0 for every step in 1..18; the step nearest 10 is 10 itself
    assert (fit["instruction"]["step"], fit["instruction"]["threshold"], fit["instruction"]["balanced_accuracy"]) \
        == (10, 0.5, 1.0)
    assert fit["thresholds"] == {"final": 0.3, "instruction": 0.5, "fitted_on": "utt-fit"}
    assert fit["utt_gate"] == "registered_secondary"

    # Cost weight: (recall 0.9, false fire 0.3) has the higher balanced accuracy, (0.6, 0.05) the higher score.
    def costly(name, i):          # 20 states: 60 recall items, 100 false-fire items
        if name in B.PARAPHRASED_FINAL:
            j = i + 20 * B.PARAPHRASED_FINAL.index(name)
            return 0.95 if j < 36 else (0.42 if j < 54 else 0.0)
        if name in B.NON_FINAL:
            j = i + 20 * B.NON_FINAL.index(name)
            return 0.95 if j < 5 else (0.42 if j < 30 else 0.0)
        return 0.0
    fit = B.fit_thresholds(_fake_bench(20, signal=costly))
    rows = {row["step"]: row for row in fit["final"]["steps"]}
    assert (rows[8]["recall"], rows[8]["false_fire"]) == (0.9, 0.3)
    assert (rows[10]["recall"], rows[10]["false_fire"]) == (0.6, 0.05)
    assert rows[8]["score"] == pytest.approx(0.9 - 3.4 * 0.3) and rows[10]["score"] == pytest.approx(0.6 - 3.4 * 0.05)
    assert (rows[8]["recall"] + 1 - rows[8]["false_fire"]) / 2 > (rows[10]["recall"] + 1 - rows[10]["false_fire"]) / 2
    assert fit["final"]["step"] == 10                      # steps 9..19 tie on the score; 10 is nearest to 10

    # A final sentence that gets flagged as an instruction does not fire, so it counts as a miss.
    flagged = lambda name, i: 0.9 if name in INSTRUCTION_ITEMS or (name == "final_walk" and i < 10) else 0.0  # noqa: E731
    fit = B.fit_thresholds(_fake_bench(20, instruction=flagged))
    assert fit["instruction"]["threshold"] == 0.5
    assert {row["recall"] for row in fit["final"]["steps"]} == {50 / 60}
    assert fit["final"]["best_score"] == pytest.approx(50 / 60)

    # No threshold beats "never fires": the best score is 0, so there is no threshold.
    fit = B.fit_thresholds(_fake_bench(12, signal=lambda name, i: 0.0))
    assert fit["status"] == "no_useful_threshold_on_fit" and fit["thresholds"] is None
    assert fit["final"]["step"] is None and fit["final"]["best_score"] == 0.0
    assert fit["instruction"]["threshold"] == 0.5          # tau_inst is still reported
    worse = B.fit_thresholds(_fake_bench(12, signal=lambda name, i: 1.0))      # fires on everything: 1 - 3.4 < 0
    assert worse["status"] == "no_useful_threshold_on_fit" and worse["final"]["best_score"] == pytest.approx(-2.4)

    # A clean reader fits (0.5, 0.5): the fixed variant would be the same run and is skipped.
    clean = B.fit_thresholds(_fake_bench(12))
    assert clean["thresholds"] == {"final": 0.5, "instruction": 0.5, "fitted_on": "utt-fit"}
    assert clean["utt_gate"] == "skipped_identical_to_utt_gate_fit"
    assert B.fit_thresholds(_fake_bench(12, block="smoke"))["fitted_on"] == "utt-smoke-rehearsal"
    for refused in (_fake_bench(12, block="bench"), _fake_bench(12, block="test"), _fake_bench(12, complete=False)):
        with pytest.raises(X.ExperimentRefusal):
            B.fit_thresholds(refused)


def test_held_out_item_never_moves_the_fit_or_an_aggregate():
    """Section 6.2: the item outside the bank is reported on its own, never fitted on and never mixed in."""
    base = _fake_bench(12)
    odd = _fake_bench(
        12, signal=lambda name, i: 0.0 if name == B.HELD_OUT_ITEM else (1.0 if name in FINAL_ITEMS else 0.0),
        instruction=lambda name, i: 0.6 if name == B.HELD_OUT_ITEM else (0.9 if name in INSTRUCTION_ITEMS else 0.0))
    # Before the fix the held-out answers entered tau_inst as negatives and moved it from 0.5 to 0.65.
    assert B.fit_thresholds(odd) == B.fit_thresholds(base)
    assert B.fit_thresholds(odd)["instruction"]["threshold"] == 0.5
    for thresholds in (FIXED_HALF, Thresholds(0.5, 0.5, "utt-fit")):
        assert B.gate_minus_literal(odd["items"], thresholds) == B.gate_minus_literal(base["items"], thresholds)
    # ... and it is still reported, separately
    summary = B.reader_summary(odd["items"], FIXED_HALF)
    assert summary["held_out"][B.HELD_OUT_ITEM]["fire_rate"] == 0.0
    assert summary["held_out"][B.HELD_OUT_ITEM]["flag_rate"] == 1.0
    assert summary["recall"] == B.reader_summary(base["items"], FIXED_HALF)["recall"] == 1.0


def test_thresholds_file_is_refit_on_load_and_an_edited_file_is_refused(rehearsal, tmp_path, capsys):
    state = _copy(rehearsal, tmp_path)
    assert B.load_state_thresholds(state) is None
    assert X.main(["fit", "--state-dir", str(state)]) == 1                 # no fit-block bench exists
    # The bench block is read with thresholds fitted on the fit block. Without a fit it is refused
    # before a first use is recorded and before any item is built (real_model=True only gets the
    # command past the stand-in refusal; nothing is asked).
    bench_block = ["bench", "--block", "bench", *_stand_in(state, "--store-mode", "read-only")]
    assert X.main(bench_block, real_model=True) == 1
    assert "fitted on the fit block" in capsys.readouterr().err
    assert X.main(["fit", "--state-dir", str(state), "--rehearsal"]) == 0
    assert X.main(bench_block, real_model=True) == 1                       # rehearsal thresholds never run it
    assert "rehearsal thresholds" in capsys.readouterr().err
    assert not (state / "bench" / "bench").exists() and not (state / "first_use.json").exists()
    assert not (tmp_path / "registry.json").exists()
    path = state / "thresholds.json"
    document = X.read_json(path)
    assert document["fit"]["fitted_on"] == "utt-smoke-rehearsal" and document["source"]["block"] == "smoke"
    loaded = B.load_state_thresholds(state)
    assert loaded["thresholds"] == Thresholds(0.5, 0.5, "utt-smoke-rehearsal") and loaded["status"] == "fitted"
    assert B.load_thresholds(path, rehearsal.first) == loaded["thresholds"]
    before = path.read_bytes()
    assert X.main(["fit", "--state-dir", str(state), "--rehearsal"]) == 0   # kept, not rewritten
    assert path.read_bytes() == before and "kept unchanged" in capsys.readouterr().out
    # the next bench is read with the fitted pair
    assert X.main(["bench", "--block", "smoke", *_stand_in(state, "--store-mode", "read-only")]) == 0
    latest = X.attempts(state / "bench" / "smoke")[-1]["result"]
    assert latest["verdict"]["basis"] == "utt_gate_fit"
    assert latest["thresholds"]["utt_gate_fit"] == loaded["thresholds"].to_json()
    assert rehearsal.first["verdict"]["basis"] == "utt_gate" and rehearsal.first["readers"]["utt_gate_fit"] is None
    # an edited threshold is refused wherever the file is read
    for edit in (lambda d: d["fit"]["thresholds"].update(final=0.45),
                 lambda d: d["fit"]["final"].update(step=9, threshold=0.45),
                 lambda d: d["fit"].update(fitted_on="utt-fit"),
                 lambda d: d["source"].update(bench_result_sha256="0" * 64)):
        edited = json.loads(before)
        edit(edited)
        path.write_text(json.dumps(edited, ensure_ascii=False, indent=2), encoding="utf-8")
        with pytest.raises(X.ExperimentRefusal):
            B.load_state_thresholds(state)
        with pytest.raises(X.ExperimentRefusal):
            B.load_thresholds(path, rehearsal.first)
        count = len(X.attempts(state / "bench" / "smoke"))
        assert X.main(["bench", "--block", "smoke", *_stand_in(state, "--store-mode", "read-only")]) == 1
        assert len(X.attempts(state / "bench" / "smoke")) == count
        assert X.main(["fit", "--state-dir", str(state), "--rehearsal"]) == 1
    path.write_bytes(before)
    assert B.load_state_thresholds(state)["fitted_on"] == "utt-smoke-rehearsal"
    # answers that differ from the ones the fit was made from are refused as well
    other = json.loads(json.dumps(rehearsal.first))
    other["items"][0]["answers"]["u_instruction"] = 0.2
    with pytest.raises(X.ExperimentRefusal):
        B.load_thresholds(path, other)


# ---- reading rule R1 -----------------------------------------------------------------------------------

def _reading(result, thresholds=FIXED_HALF):
    return B.reading_verdict(result, thresholds)


def test_reading_rule_truth_table():
    everywhere = lambda *names: (lambda name, i: 1.0 if name in FINAL_ITEMS or name in names else 0.0)  # noqa: E731
    clean = _reading(_fake_bench(60))
    assert clean["reading"] == "utterance_reading_demonstrated_on_bench" and clean["failed_checks"] == []
    assert clean["warnings"] == {"final_sentences_flagged_as_instruction": False,
                                 "instruction_moves_signal": False, "instruction_not_detected": False}
    assert clean["sentence"] == X.RESULT_SENTENCES["utterance_reading_demonstrated_on_bench"]
    assert set(clean["checks"]) == {"reads_paraphrased_final", "separates_negation", "not_fooled_by_schedule",
                                    "not_fooled_by_payment"} and set(clean["levels"]) == {"recall_level",
                                                                                           "false_fire_level"}
    # C1 passes and something else fails: always partial, with the failed names
    partial = {
        ("negated_keyword",): ["separates_negation", "false_fire_level"],
        ("deadline_pressure",): ["not_fooled_by_schedule", "false_fire_level"],
        ("payment_only",): ["not_fooled_by_payment", "false_fire_level"],
        ("open",): ["false_fire_level"],
        ("negated_keyword", "deadline_pressure", "payment_only", "open"):
            ["separates_negation", "not_fooled_by_schedule", "not_fooled_by_payment", "false_fire_level"],
    }
    for names, failed in partial.items():
        verdict = _reading(_fake_bench(60, signal=everywhere(*names)))
        assert verdict["reading"] == "partial_reads_final_but_not_separated", names
        assert verdict["failed_checks"] == failed
        assert verdict["sentence"].endswith(": " + ", ".join(failed) + ".")
        assert "<" not in verdict["sentence"]

    # 3 of 60 states fire on one of the two final sentences: C1 passes on the interval alone, L1 stops it.
    rare = lambda name, i: 1.0 if name == "final_limit" and i < 3 else 0.0   # noqa: E731
    verdict = _reading(_fake_bench(60, signal=rare))
    check = verdict["checks"]["reads_paraphrased_final"]
    assert check["mean"] == pytest.approx(0.025) and check["passed"] is True
    assert check["ci90"] == pytest.approx([0.5 / 60, 0.05]) and check["ci90"] == _bootstrap_ci([0.5] * 3 + [0.0] * 57)
    assert verdict["levels"]["recall_level"] == {"value": pytest.approx(3 / 180), "min": 0.5, "passed": False}
    assert verdict["reading"] == "partial_reads_final_but_not_separated"
    assert verdict["failed_checks"] == ["recall_level"]
    # every other check fails while C1 still passes
    all_fail = lambda name, i: 1.0 if (name == "final_limit" and i < 3) or name in (   # noqa: E731
        "negated_keyword", "deadline_pressure", "payment_only") else 0.0
    verdict = _reading(_fake_bench(60, signal=all_fail))
    assert verdict["reading"] == "partial_reads_final_but_not_separated"
    assert verdict["failed_checks"] == ["separates_negation", "not_fooled_by_schedule", "not_fooled_by_payment",
                                        "recall_level", "false_fire_level"]
    # the level boundaries are inclusive: recall 0.5 and false fire 0.10 pass
    boundary = lambda name, i: 1.0 if (name in FINAL_ITEMS and i % 2 == 0) or (   # noqa: E731
        name == "open" and i < 30) else 0.0
    verdict = _reading(_fake_bench(60, signal=boundary))
    assert verdict["levels"]["recall_level"]["value"] == 0.5 and verdict["levels"]["false_fire_level"]["value"] == 0.1
    assert verdict["reading"] == "utterance_reading_demonstrated_on_bench"
    over = lambda name, i: 1.0 if name in FINAL_ITEMS or (name == "open" and i < 31) else 0.0   # noqa: E731
    assert _reading(_fake_bench(60, signal=over))["failed_checks"] == ["false_fire_level"]

    # C1 fails: not demonstrated, whatever the other checks say
    assert _reading(_fake_bench(60, signal=lambda name, i: 0.0))["reading"] == "utterance_reading_not_demonstrated"
    fires_on_all = _reading(_fake_bench(60, signal=everywhere("neutral")))
    assert fires_on_all["reading"] == "utterance_reading_not_demonstrated"
    assert fires_on_all["levels"]["recall_level"]["passed"] is True
    assert fires_on_all["sentence"] == X.RESULT_SENTENCES["utterance_reading_not_demonstrated"]
    # not complete: no reading
    incomplete = _reading(_fake_bench(60, complete=False))
    assert incomplete["reading"] == "incomplete_no_reading" and incomplete["failed_checks"] == []

    # the three warnings
    flagged = _reading(_fake_bench(60, instruction=lambda name, i: 0.9 if name in INSTRUCTION_ITEMS or (
        name == "final_limit" and i < 20) else 0.0))
    assert flagged["warnings"]["final_sentences_flagged_as_instruction"] is True      # 20 of 180 > 0.10
    assert flagged["levels"]["recall_level"]["value"] == pytest.approx(160 / 180)      # flagged ones did not fire
    at_limit = _reading(_fake_bench(60, instruction=lambda name, i: 0.9 if name in INSTRUCTION_ITEMS or (
        name == "final_limit" and i < 18) else 0.0))
    assert at_limit["warnings"]["final_sentences_flagged_as_instruction"] is False    # exactly 0.10 is not over
    moved = _reading(_fake_bench(60, signal=lambda name, i: 0.3 if name == B.GUARD_ITEM else (
        1.0 if name in FINAL_ITEMS else 0.0)))
    assert moved["warnings"]["instruction_moves_signal"] is True and moved["reading"] == clean["reading"]
    unseen = _reading(_fake_bench(60, instruction=lambda name, i: 0.1))
    assert unseen["warnings"]["instruction_not_detected"] is True
    # the same answers read with other thresholds give the other variant's verdict
    weak = lambda name, i: 0.4 if name in FINAL_ITEMS else 0.0   # noqa: E731
    assert _reading(_fake_bench(60, signal=weak))["reading"] == "utterance_reading_not_demonstrated"
    assert _reading(_fake_bench(60, signal=weak), Thresholds(0.35, 0.5, "utt-fit"))["reading"] \
        == "utterance_reading_demonstrated_on_bench"
    # run_bench reports the fixed pair when there is no fit, and says so
    summary = B.summarize(_fake_bench(60, signal=weak, block="bench"), None, None, "no_useful_threshold_on_fit")
    assert (summary["verdict"]["basis"], summary["verdict"]["basis_reason"]) == ("utt_gate",
                                                                                 "no_useful_threshold_on_fit")
    assert summary["verdict"]["registered_block"] is True and summary["readers"]["utt_gate_fit"] is None
    fitted = B.summarize(_fake_bench(60, signal=weak, block="bench"), Thresholds(0.35, 0.5, "utt-fit"), None)
    assert fitted["verdict"]["basis"] == "utt_gate_fit"
    assert fitted["verdict"]["reading"] == "utterance_reading_demonstrated_on_bench"
    assert fitted["verdict_utt_gate"]["reading"] == "utterance_reading_not_demonstrated"


def test_bench_asks_the_existing_question_once_per_item_and_keeps_it_out_of_r1():
    class Counting(X.UtteranceCpuStandIn):
        def __init__(self):
            super().__init__()
            self.asked = []

        def predict(self, state, candidates, request):
            self.asked.append((tuple(request["questions"]), state.last_offer("supplier").message))
            return super().predict(state, candidates, request)

    backend = Counting()
    result = B.run_bench(backend, "smoke", limit=2)
    assert result["complete"] is True and len(result["items"]) == 24
    utterance = [message for questions, message in backend.asked if questions == QIDS]
    existing = [message for questions, message in backend.asked if questions == (FINAL_OFFER_QID,)]
    assert len(backend.asked) == 2 * 24 + 1 and len(existing) == 24 and len(utterance) == 24 + 1
    assert existing == [item["message"] for item in result["items"]]          # exactly one per item, in order
    assert utterance[:-1] == existing                                          # the last one is the token check
    assert result["counts"] == {"live_requests": 49, "live_failed_requests": 0, "store_hits": 0, "planned_requests": 49,
                                "utterance_requests": 25, "existing_question_requests": 24}
    for item in result["items"]:
        assert item["status"] == "ok" and item["existing"]["status"] == "ok"
        assert item["request_sha256"] != item["existing"]["request_sha256"]
        assert set(item["answers"]) == set(QIDS) and 0.0 <= item["existing"]["final_offer"] <= 1.0
        assert "state" not in item and "request" not in item
    assert result["evidence_class"] == "cpu_rehearsal_not_model_evidence"

    # Changing every answer to the existing question changes neither R1 nor tau_inst nor tau_final.
    changed = json.loads(json.dumps(result))
    for item in changed["items"]:
        item["existing"]["final_offer"] = 0.95 if item["labels"]["says_final"] else 0.05
    for thresholds in (FIXED_HALF, Thresholds(0.3, 0.7, "utt-fit")):
        assert B.reading_verdict(changed, thresholds) == B.reading_verdict(result, thresholds)
    before, after = B.fit_thresholds(result), B.fit_thresholds(changed)
    assert (after["instruction"], after["final"], after["thresholds"], after["status"]) == (
        before["instruction"], before["final"], before["thresholds"], before["status"])
    assert after["existing_question"] != before["existing_question"]
    summary = B.summarize(changed, FIXED_HALF, 0.5)
    assert summary["verdict"]["reading"] == B.summarize(result, FIXED_HALF, None)["verdict"]["reading"]
    assert summary["readers"]["existing_question"]["at_half"]["recall"] == 1.0
    assert result["readers"]["existing_question"]["at_half"]["recall"] == 0.0   # the keyword rule misses paraphrases
    assert result["readers"]["existing_question"]["at_half"]["fire_rate_by_item"]["negated_keyword"] == 1.0
    assert result["readers"]["literal"]["by_item"]["negated_keyword"]["fire_rate"] == 1.0
    assert result["readers"]["literal"]["recall"] == 0.0 and result["readers"]["utt_gate"]["recall"] == 1.0
    assert set(result["readers"]["existing_question"]["neutral_by_round"]) == {"1", "2"}


def test_existing_question_threshold_uses_the_same_grid_score_and_items(tmp_path):
    result = _fake_bench(34, existing=_tie_values)
    fit = B.fit_thresholds(result)
    existing = fit["existing_question"]
    assert (existing["status"], existing["step"], existing["threshold"]) == ("fitted", 6, 0.3)
    assert [row["step"] for row in existing["steps"]] == list(range(1, 20))
    # the very same numbers as the gate's own grid on the same values: one grid, one tie rule, one score
    mirrored = B.fit_thresholds(_fake_bench(34, signal=_tie_values))["final"]
    assert existing["steps"] == mirrored["steps"] and existing["best_score"] == mirrored["best_score"]
    assert existing["arm"] == "model_final_gate_fit"
    assert fit["thresholds"] == {"final": 0.5, "instruction": 0.5, "fitted_on": "utt-fit"}   # independent of tau_m
    # no instruction rule for the existing question: an instruction answer does not change tau_m
    flagged = _fake_bench(34, existing=_tie_values, instruction=lambda name, i: 0.9)
    assert B.fit_thresholds(flagged)["existing_question"] == existing
    assert B.fit_thresholds(flagged)["status"] == "no_useful_threshold_on_fit"   # the gate itself never fires now
    # its status is independent: no useful tau_m while the gate's thresholds are fitted
    none = B.fit_thresholds(_fake_bench(34))
    assert none["existing_question"]["status"] == "no_useful_threshold_on_fit"
    assert none["existing_question"]["threshold"] is None and none["status"] == "fitted"

    # written once, refitted on every load, refused when edited
    state = tmp_path / "s"
    X.ensure_protocol(state, create=True)
    folder = state / "bench" / "fit" / "attempt-invented"
    stored = {**result, "protocol_sha256": X.protocol_sha256()}
    X.write_json(folder / "result.json", stored)
    attempt = X.first_complete_attempt(state / "bench" / "fit")
    assert attempt is not None and attempt["name"] == "attempt-invented"
    path, document, written = B.write_thresholds(state, attempt)
    assert written and document["source"] == {"block": "fit", "attempt": "bench/fit/attempt-invented",
                                              "bench_result_sha256": B.bench_result_sha256(stored),
                                              "evidence_class": None}
    assert B.load_existing_threshold(path, stored) == 0.3
    assert B.load_thresholds(path, stored) == Thresholds(0.5, 0.5, "utt-fit")
    loaded = B.load_state_thresholds(state)
    assert (loaded["existing_threshold"], loaded["existing_status"], loaded["fitted_on"]) == (0.3, "fitted", "utt-fit")
    assert B.write_thresholds(state, attempt)[2] is False             # the same fit is kept, not rewritten
    original = path.read_text(encoding="utf-8")
    edited = json.loads(original)
    edited["fit"]["existing_question"].update(step=7, threshold=0.35)
    path.write_text(json.dumps(edited), encoding="utf-8")
    for load in (B.load_existing_threshold, B.load_thresholds):
        with pytest.raises(X.ExperimentRefusal):
            load(path, stored)
    with pytest.raises(X.ExperimentRefusal):
        B.load_state_thresholds(state)
    with pytest.raises(X.ExperimentRefusal):                          # another fit never replaces the file
        B.write_thresholds(state, attempt)
    path.write_text(original, encoding="utf-8")
    assert B.load_existing_threshold(path, stored) == 0.3


def test_instruction_guard_truth_table():
    def guarded(fires: int, complete: bool = True):
        """The gate fires on the guard item in `fires` of 60 states (the flag is missing there)."""
        return _fake_bench(
            60, signal=lambda name, i: 1.0 if name in FINAL_ITEMS or name == B.GUARD_ITEM else 0.0,
            instruction=lambda name, i: 0.0 if (name == B.GUARD_ITEM and i < fires) or name not in INSTRUCTION_ITEMS
            else 0.9,
            existing=lambda name, i: 0.8 if name == B.GUARD_ITEM else 0.0, block="bench", complete=complete)

    holds = B.guard_verdict(guarded(6), FIXED_HALF, 0.5)
    assert holds["result"] == "instruction_guard_holds" and holds["instruction_fire_rate"] == 0.1   # exactly 0.10
    assert holds["sentence"] == X.RESULT_SENTENCES["instruction_guard_holds"]
    assert holds["signal_over_threshold_rate"] == 1.0          # without the guard it would have fired everywhere
    assert holds["flag_rate"] == pytest.approx(54 / 60)
    assert holds["existing_question_fire_rate"] == 1.0         # the existing question has no guard
    fails = B.guard_verdict(guarded(7), FIXED_HALF, 0.5)
    assert fails["result"] == "instruction_guard_fails" and fails["instruction_fire_rate"] == pytest.approx(7 / 60)
    assert fails["sentence"] == X.RESULT_SENTENCES["instruction_guard_fails"]
    assert B.guard_verdict(guarded(0), FIXED_HALF)["result"] == "instruction_guard_holds"
    assert B.guard_verdict(guarded(0), FIXED_HALF)["existing_question_fire_rate"] is None    # no tau_m known
    assert B.guard_verdict(guarded(60), FIXED_HALF)["instruction_fire_rate"] == 1.0
    # not judged when the bench is not complete
    unjudged = B.guard_verdict(guarded(0, complete=False), FIXED_HALF, 0.5)
    assert unjudged["result"] is None and unjudged["judged"] is False and unjudged["sentence"] is None
    # the guard follows the thresholds it is given
    assert B.guard_verdict(guarded(60), Thresholds(0.5, 0.0, "utt-fit"))["result"] == "instruction_guard_holds"
    # I1 does not change R1: the reading is the same whether the guard holds or fails
    for fires in (6, 7, 60):
        reading = B.reading_verdict(guarded(fires), FIXED_HALF)
        assert reading["reading"] == "utterance_reading_demonstrated_on_bench" and reading["failed_checks"] == []
        summary = B.summarize(guarded(fires), FIXED_HALF, 0.5)
        assert summary["verdict"]["reading"] == reading["reading"]
        assert summary["guard"]["result"] == ("instruction_guard_holds" if fires == 6 else "instruction_guard_fails")
    # the item with a true final sentence and a to-open instruction is not part of I1
    other = _fake_bench(60, instruction=lambda name, i: 0.0, block="bench")
    assert B.guard_verdict(other, FIXED_HALF)["result"] == "instruction_guard_holds"
    assert X.GUARD_LEVELS == {"instruction_fire_max": 0.10} and X.R1_LEVELS["false_fire_max"] == 0.10


# ---- labels --------------------------------------------------------------------------------------------

def test_only_m0_all_live_gets_the_plain_label():
    counts = ((5, 0), (5, 3), (0, 3), (0, 0))
    plain = [(kind, live, hits) for kind in X.EVIDENCE_KINDS for live, hits in counts
             if X.evidence_class(kind, real_model=True, live=live, hits=hits) == "real_model_run"]
    assert plain == [("m0", 5, 0)]
    for kind in X.EVIDENCE_KINDS:
        for real in (True, False):
            for live, hits in counts:
                label = X.evidence_class(kind, real_model=real, live=live, hits=hits)
                if kind != "m0":
                    assert not label.startswith("real_model_run"), (kind, real, live, hits)
                if not real or kind == "cpu_arm":
                    assert label.endswith("not_model_evidence")
    assert X.evidence_class("m0", real_model=True, live=5, hits=3) == "real_model_run_partly_replayed_from_store"
    assert X.evidence_class("m0", real_model=True, live=0, hits=3) == "real_model_replay_from_store"
    expected = {"utterance_arm": "real_model_composed_utterance", "composed_arm": "real_model_composed_estimates",
                "bench": "real_model_utterance_bench"}
    for kind, base in expected.items():
        assert X.evidence_class(kind, real_model=True, live=5, hits=0) == base
        assert X.evidence_class(kind, real_model=True, live=5, hits=3) == base + "_partly_replayed_from_store"
        assert X.evidence_class(kind, real_model=True, live=0, hits=3) == base + "_replay_from_store"
        assert X.evidence_class(kind, real_model=True, live=0, hits=0) == "real_model_requested_no_model_answer"
        assert X.evidence_class(kind, real_model=False, live=5, hits=0) == "cpu_rehearsal_not_model_evidence"
    assert X.evidence_class("ask", real_model=True, live=1, hits=0) == "real_model_single_utterance_demo"
    assert X.evidence_class("cpu_arm", real_model=True, live=0, hits=0) == "cpu_reference_not_model_evidence"
    assert X.evidence_class("m0", real_model=False, live=5, hits=0) == "cpu_rehearsal_not_model_evidence"
    with pytest.raises(ValueError):
        X.evidence_class("model_final", real_model=True, live=1, hits=0)
    # every model arm of the protocol maps to a kind that cannot be plain, except m0
    assert set(X.MODEL_ARMS) - {"m0"} == {"utt_gate_fit", "utt_gate", "model_final", "model_final_gate",
                                          "model_final_gate_fit"}


class AlwaysFails(ModelDouble):
    """A declared-model test double whose every call raises. NOT a model; it never answers."""

    def predict(self, state, candidates, request):
        raise RuntimeError("invented failure of every call")


def test_a_real_bench_without_any_answer_is_not_labelled_as_model_evidence(tmp_path):
    """Section 9: failed calls are attempts, not answers; a run with no answer says so."""
    state = tmp_path / "s"
    argv = ["bench", "--block", "smoke", "--limit", "1", *_stand_in(state, "--store-mode", "read-write")]
    assert X.main(argv, backend_factory=AlwaysFails, declared=ModelDouble.DECLARED, real_model=True,
                  registry_path=tmp_path / "r.json") == 1
    result = X.attempts(state / "bench" / "smoke")[0]["result"]
    assert result["complete"] is False and not any(item["status"] == "ok" for item in result["items"])
    assert result["counts"]["live_requests"] == result["counts"]["live_failed_requests"] == 25
    assert result["counts"]["store_hits"] == 0
    assert result["evidence_class"] == "real_model_requested_no_model_answer"
    # the counting helper: answers, not attempts
    assert X.count_delta({"live_requests": 2, "live_answers": 1, "store_hits": 0},
                         {"live_requests": 7, "live_answers": 3, "store_hits": 4}) == {
        "live_requests": 5, "live_answers": 2, "live_failed_requests": 3, "store_hits": 4}


# ---- first use and protocol binding --------------------------------------------------------------------

def test_first_use_is_recorded_before_a_bench_or_test_block_run(rehearsal, tmp_path, monkeypatch):
    registry, first_dir, second_dir = tmp_path / "shared-registry.json", tmp_path / "a", tmp_path / "b"
    first = X.record_first_use(first_dir, "bench_block_run", "invented-bench-block", registry_path=registry)
    assert first["created"] is True and first["confirmatory"] is True and first["status"] == "first_use"
    recorded = X.read_json(first_dir / "first_use.json")
    assert (recorded["event"], recorded["block"], recorded["protocol_version"], recorded["protocol_sha256"]) == (
        "bench_block_run", "invented-bench-block", PROTOCOL_VERSION, X.protocol_sha256())
    # the design document is not bound by the digest; its hash at the first use is kept beside it
    assert recorded["design_document"] == "negotiation/docs/UTTERANCE_AGENT_DESIGN.md"
    assert recorded["design_document_sha256"] == hashlib.sha256(
        (NEGOTIATION / "docs" / "UTTERANCE_AGENT_DESIGN.md").read_bytes()).hexdigest()
    # the first event wins; a later event of the same directory does not rewrite the file
    later = X.record_first_use(first_dir, "test_block_run", "invented-test-block", registry_path=registry)
    assert later["created"] is False and later["confirmatory"] is True
    assert X.read_json(first_dir / "first_use.json") == recorded
    # the same block in another state directory is a second use
    second = X.record_first_use(second_dir, "bench_block_run", "invented-bench-block", registry_path=registry)
    assert second["confirmatory"] is False and second["status"] == "not_confirmatory"
    assert [use["state_dir_name"] for use in second["earlier_uses_elsewhere"]] == ["a"]
    assert X.record_first_use(second_dir, "test_block_run", "another-invented-block",
                              registry_path=registry)["confirmatory"] is True
    entries = X.read_json(registry)["entries"]
    assert [(e["block"], e["state_dir_name"]) for e in entries] == [
        ("invented-bench-block", "a"), ("invented-test-block", "a"), ("invented-bench-block", "b"),
        ("another-invented-block", "b")]
    X.record_first_use(first_dir, "bench_block_run", "invented-bench-block", registry_path=registry)
    assert len(X.read_json(registry)["entries"]) == 4               # a continued use adds no entry
    with pytest.raises(ValueError):
        X.record_first_use(first_dir, "something_else", "invented-bench-block", registry_path=registry)

    # A real-model response: first_use.json exists before the backend is even built.
    state, built = tmp_path / "real", []

    def factory():
        assert (state / "first_use.json").is_file()
        built.append(1)
        return ModelDouble()

    argv = ["bench", "--block", "smoke", "--limit", "2", *_stand_in(state, "--store-mode", "read-write")]
    assert X.main(argv, backend_factory=factory, declared=ModelDouble.DECLARED, real_model=True,
                  registry_path=registry) == 0
    assert built == [1]
    used = X.read_json(state / "first_use.json")
    assert (used["event"], used["block"]) == ("first_real_response", "smoke")
    result = X.attempts(state / "bench" / "smoke")[0]["result"]
    assert result["evidence_class"] == "real_model_utterance_bench" and result["complete"] is True
    assert result["verdict"]["counts_as_model_evidence"] is False      # a reduced smoke run is never evidence
    assert result["longest_state_check"]["input_tokens_per_question"] == [7, 7, 7, 7]
    assert result["attempt"]["first_use"]["status"] == "first_use"
    # the result keeps the description of the backend that answered, taken before it was closed
    assert result["backend"]["prediction_store"]["inner_built"] is True
    assert result["backend"]["inner"]["kind"] == "utterance_test_double_model"
    replay = [*argv[:-1], "read-only"]
    assert X.main(replay, backend_factory=factory, declared=ModelDouble.DECLARED, real_model=True,
                  registry_path=registry) == 0
    assert built == [1]                                                # a full replay builds no backend
    replayed = X.attempts(state / "bench" / "smoke")[1]["result"]
    assert replayed["evidence_class"] == "real_model_utterance_bench_replay_from_store"
    assert replayed["counts"]["live_requests"] == 0
    # a CPU rehearsal of the smoke block is no binding event
    assert not (rehearsal.state / "first_use.json").exists()

    # After the first use, another protocol digest is refused in that directory.
    changed = dict(W.BANK)
    changed["neutral"] = (("N1", "보내드린 조건을 확인해 주십시오."),) + W.BANK["neutral"][1:]
    with monkeypatch.context() as patch:
        patch.setattr(W, "BANK", changed)
        with pytest.raises(X.ExperimentRefusal):
            X.record_first_use(first_dir, "bench_block_run", "invented-bench-block", registry_path=registry)
        with pytest.raises(X.ExperimentRefusal):
            X.check_protocol_binding(state)
        assert X.main(argv, backend_factory=factory, declared=ModelDouble.DECLARED, real_model=True,
                      registry_path=registry) == 1
    assert len(X.attempts(state / "bench" / "smoke")) == 2 and built == [1]
    X.check_protocol_binding(state)                                    # the registered protocol still fits


def test_protocol_digest_binds_the_state_directory(tmp_path, monkeypatch):
    state = tmp_path / "s"
    assert X.main(["plan", "--state-dir", str(state)]) == 0
    document = X.read_json(state / "protocol.json")
    assert document["protocol_version"] == PROTOCOL_VERSION == "utterance-2026-11-18.v1"
    assert document["protocol_sha256"] == X.protocol_sha256() == X._protocol_digest_of_file(document)
    X.check_protocol_binding(state)
    assert X.main(["plan", "--state-dir", str(state)]) == 0           # unchanged: kept
    assert X.read_json(state / "protocol.json") == document
    # the protocol is the same on every call and holds no location
    assert X.build_protocol() == X.build_protocol()
    text = json.dumps(X.build_protocol(), ensure_ascii=False)
    assert str(tmp_path) not in text and str(NEGOTIATION.parent) not in text
    protocol = X.build_protocol()
    assert protocol["golden_requests"] == {
        "default": {"case_id": "dev-1100", "round": 1,
                    "sha256": "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"},
        "utterance": {"case_id": "dev-1100", "round": 1,
                      "sha256": "ab23444d19a7818572e2c786727ecadd42ba0f1d1a74d4d883f0a23bdfc4343d"}}
    assert X.golden_request_hashes() == {name: row["sha256"] for name, row in protocol["golden_requests"].items()}
    assert protocol["arms"]["names"] == list(X.ARMS) and len(X.ARMS) == 10
    assert protocol["bench"]["items"] == list(X.BENCH_ITEMS) and len(X.BENCH_ITEMS) == 12
    assert {name: block["cases"] for name, block in protocol["cases"]["blocks"].items()} == {
        "smoke": 10, "fit": 30, "bench": 60, "test": 600}
    assert (protocol["styles"]["decoy_rate"], protocol["styles"]["instruction_rate"]) == (0.5, 0.25)
    assert protocol["fit_rule"]["false_fire_weight"] == 3.4 and protocol["result_sentences"] == X.RESULT_SENTENCES
    assert "rehearsal_limit" not in text                              # a rehearsal reduction is not protocol
    # Both tracks must run under ONE protocol digest (section 14 of the design document). The value is
    # pinned here so that a later change of a registered rule, sentence or block is a visible decision:
    # before the first use it means updating this line on purpose, after it a new protocol version.
    assert X.protocol_sha256() == PINNED_PROTOCOL_SHA256

    # A protocol with a changed bank sentence cannot be used in the existing directory.
    changed = dict(W.BANK)
    changed["final_limit"] = (("L1", "이 가격이 저희 한도입니다."),) + W.BANK["final_limit"][1:]
    with monkeypatch.context() as patch:
        patch.setattr(W, "BANK", changed)
        assert X.protocol_sha256() != document["protocol_sha256"]
        with pytest.raises(X.ExperimentRefusal):
            X.check_protocol_binding(state)
        assert X.main(["bench", "--block", "smoke", *_stand_in(state)]) == 1
        assert X.main(["plan", "--state-dir", str(state)]) == 1
        assert X.main(["fit", "--state-dir", str(state), "--rehearsal"]) == 1
        assert not (state / "bench").exists() and not (state / "store").exists()
        assert X.read_json(state / "protocol.json") == document       # the recorded protocol is never rewritten
    X.check_protocol_binding(state)
    # an edited protocol file is refused as well
    edited = dict(document, protocol_version="utterance-2026-11-18.v2")
    (state / "protocol.json").write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(X.ExperimentRefusal):
        X.check_protocol_binding(state)
    with pytest.raises(X.ExperimentRefusal):
        X.check_protocol_binding(tmp_path / "nowhere")


# ---- pack / unpack -------------------------------------------------------------------------------------

def _files(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in sorted(folder.rglob("*"))
            if path.is_file()}


def test_pack_roundtrip_and_a_tampered_archive_is_refused(rehearsal, tmp_path):
    state = _copy(rehearsal, tmp_path)
    assert X.main(["fit", "--state-dir", str(state), "--rehearsal"]) == 0
    archive = tmp_path / "state.zip"
    assert X.main(["pack", "--state-dir", str(state), "--out", str(archive)]) == 0
    with zipfile.ZipFile(archive) as packed:
        names = packed.namelist()
        manifest = json.loads(packed.read(X.MANIFEST_NAME))
    assert manifest["protocol_sha256"] == X.protocol_sha256() and manifest["kind"] == "utterance_experiment_state"
    assert sorted(names) == sorted([X.MANIFEST_NAME, *_files(state)])
    assert {"protocol.json", "thresholds.json"} <= set(names) and any(n.startswith("store/") for n in names)
    restored = tmp_path / "restored"
    assert X.main(["unpack", "--zip", str(archive), "--state-dir", str(restored)]) == 0
    assert _files(restored) == _files(state)
    # the unpacked directory works: thresholds verify and the bench replays without a live request
    assert B.load_state_thresholds(restored)["fitted_on"] == "utt-smoke-rehearsal"
    assert X.main(["bench", "--block", "smoke", *_stand_in(restored, "--store-mode", "read-only")]) == 0
    assert X.attempts(restored / "bench" / "smoke")[-1]["result"]["counts"]["live_requests"] == 0
    again = X.unpack(archive, tmp_path / "second")
    assert again["written"] == len(_files(state)) and again["matches_this_code"] is True
    assert X.unpack(archive, tmp_path / "second") == {**again, "written": 0, "identical": again["written"]}
    # refusals that change nothing
    assert X.main(["pack", "--state-dir", str(state), "--out", str(archive)]) == 1            # never overwritten
    assert X.main(["pack", "--state-dir", str(state), "--out", str(state / "inside.zip")]) == 1
    assert X.main(["pack", "--state-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "x.zip")]) == 1
    assert not (state / "inside.zip").exists() and not (tmp_path / "x.zip").exists()

    def rebuilt(target: Path, change) -> Path:
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
            members = {name: source.read(name) for name in source.namelist()}
            change(members)
            for name, data in members.items():
                out.writestr(name, data)
        return target

    def edit_result(members):
        name = next(n for n in members if n.endswith("result.json"))
        members[name] = members[name].replace(b'"complete": true', b'"complete": false')

    def edit_manifest(members):
        document = json.loads(members[X.MANIFEST_NAME])
        document["files"] = document["files"][1:]
        members[X.MANIFEST_NAME] = json.dumps(document).encode("utf-8")

    tampered = {
        "content": edit_result,
        "threshold": lambda m: m.update({"thresholds.json": m["thresholds.json"].replace(b"0.5", b"0.4")}),
        "extra": lambda m: m.update({"store/extra.jsonl": b"{}\n"}),
        "missing": lambda m: m.pop("thresholds.json"),
        "manifest": edit_manifest,
        "no_manifest": lambda m: m.pop(X.MANIFEST_NAME),
        "protocol": lambda m: m.update({"protocol.json": m["protocol.json"].replace(b"utterance-2026", b"utterance-2027")}),
    }
    for label, change in tampered.items():
        target = tmp_path / f"unpacked-{label}"
        with pytest.raises(X.ExperimentRefusal):
            X.unpack(rebuilt(tmp_path / f"{label}.zip", change), target)
        assert not target.exists(), label
        assert X.main(["unpack", "--zip", str(tmp_path / f"{label}.zip"), "--state-dir", str(target)]) == 1
    with pytest.raises(X.ExperimentRefusal):
        X.unpack(tmp_path / "registry.json", tmp_path / "not-a-zip")
    # a directory that already holds other content is not merged into
    other = tmp_path / "other"
    other.mkdir()
    (other / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(X.ExperimentRefusal):
        X.unpack(archive, other)
    assert _files(other) == {"protocol.json": b"{}"}
    conflicting = tmp_path / "conflicting"
    shutil.copytree(restored, conflicting)
    (conflicting / "thresholds.json").write_text("{}", encoding="utf-8")
    with pytest.raises(X.ExperimentRefusal):
        X.unpack(archive, conflicting)


def _consistent_archive(target: Path, extra: dict[str, bytes]) -> Path:
    """A ZIP whose manifest matches every member: protocol.json plus `extra`. Only the names are wrong."""
    document = {**X.build_protocol(), "created_at_utc": "2026-11-18T00:00:00+00:00"}
    document["protocol_sha256"] = X._protocol_digest_of_file(document)
    members = {"protocol.json": json.dumps(document, ensure_ascii=False).encode("utf-8"), **extra}
    manifest = {"schema_version": 1, "kind": X.MANIFEST_KIND, "protocol_sha256": document["protocol_sha256"],
                "files": [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in members.items()]}
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(X.MANIFEST_NAME, json.dumps(manifest))
        for name, data in members.items():
            archive.writestr(name, data)
    return target


def test_unpack_refuses_names_that_collide_on_windows_and_writes_nothing(tmp_path):
    """Names that one file system keeps apart and Windows maps to one file are refused before any write."""
    cases = {
        "case": {"notes/a.json": b"1", "notes/A.json": b"2"},
        "dot": {"notes/a.json": b"1", "notes/a.json.": b"2"},
        "space": {"notes/a.json": b"1", "notes/a.json ": b"2"},
        "device": {"notes/CON.json": b"1"},
        "colon": {"notes/a:b.json": b"1"},
        "star": {"notes/a*.json": b"1"},
    }
    for label, extra in cases.items():
        archive = _consistent_archive(tmp_path / f"{label}.zip", extra)
        target = tmp_path / f"dest-{label}"
        with pytest.raises(X.ExperimentRefusal):
            X.unpack(archive, target)
        assert not target.exists(), label                          # protocol.json was not written either
    # a well-formed archive of the same shape unpacks
    good = X.unpack(_consistent_archive(tmp_path / "good.zip", {"notes/a.json": b"1"}), tmp_path / "dest-good")
    assert good["written"] == 2 and _files(tmp_path / "dest-good") == {
        "notes/a.json": b"1", "protocol.json": (tmp_path / "dest-good" / "protocol.json").read_bytes()}


def test_a_failed_write_during_unpack_rolls_back_every_file(tmp_path, monkeypatch):
    archive = _consistent_archive(tmp_path / "a.zip", {"notes/a.json": b"1", "store/b.jsonl": b"2"})
    target = tmp_path / "dest"
    real_open = Path.open

    def failing_open(self, mode="r", *args, **kwargs):
        if self.name == "b.jsonl" and "x" in mode:
            raise OSError("invented disk failure")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError):
        X.unpack(archive, target)
    monkeypatch.undo()
    assert not target.exists()                       # the files and the folders made for them are rolled back
    assert X.unpack(archive, target)["written"] == 3                 # and a later unpack starts clean


def test_state_and_output_locations_in_protected_folders_are_refused(tmp_path, monkeypatch, capsys):
    """negotiation/evidence and negotiation/artifacts are never created or written by this experiment."""
    root = tmp_path / "repo"
    monkeypatch.setattr(X, "REPOSITORY_ROOT", root)             # even a missed refusal could only write in tmp
    evidence, artifacts = root / "negotiation" / "evidence", root / "Negotiation" / "Artifacts"
    state = tmp_path / "s"
    assert X.main(["plan", "--state-dir", str(state)]) == 0
    argvs = (["plan", "--state-dir", str(evidence / "utt")],
             ["plan", "--state-dir", str(artifacts / "utt")],          # letter case does not matter
             ["bench", "--block", "smoke", *_stand_in(evidence / "utt")],
             ["fit", "--state-dir", str(evidence), "--rehearsal"],
             ["report", "--state-dir", str(evidence / "utt")],
             ["pack", "--state-dir", str(state), "--out", str(evidence / "x.zip")],
             ["unpack", "--zip", str(tmp_path / "x.zip"), "--state-dir", str(artifacts / "u")])
    for argv in argvs:
        assert X.main(argv) == 1
        assert "inside negotiation/" in capsys.readouterr().err
    assert not (root / "negotiation").exists() and not (root / "Negotiation").exists()
    with pytest.raises(X.ExperimentRefusal):
        X.refuse_protected_location(evidence / "a" / "b", "--html")
    X.refuse_protected_location(root / "negotiation" / "evidence-other", "--state-dir")   # a sibling is fine
    X.refuse_protected_location(root / "runs" / "negotiation" / "utterance" / "v1", "--state-dir")


# ---- documented commands and sentences -----------------------------------------------------------------

def _section_13_9_commands() -> list[str]:
    section = DOCUMENT.split("### 13.9", 1)[1].split("\n## 14.", 1)[0]
    block = section.split("```", 2)[1]
    return [line for line in block.splitlines() if line.strip()]


def test_documented_colab_commands_parse_and_match_the_design_document(tmp_path):
    documented = _section_13_9_commands()
    assert list(X.COLAB_COMMANDS) == documented and len(documented) == 17
    scripts = []
    for line in X.COLAB_COMMANDS:
        assert f"\n{line}\n" in DOCUMENT
        argv = shlex.split(line.replace("<backend>", NEG_ARGS))
        assert argv[0] == "python" and argv[1] in ("negotiation/utterance_experiment.py",
                                                    "negotiation/utterance_demo.py")
        assert (NEGOTIATION.parent / argv[1]).is_file() or argv[1].endswith("utterance_demo.py")   # demo: track B
        parser = X.build_parser() if argv[1].endswith("utterance_experiment.py") else X.build_demo_parser()
        args = parser.parse_args(argv[2:])
        assert args.state_dir == Path("runs/negotiation/utterance/v1")
        assert args.state_dir.name == X.STATE_DIR_NAME
        if hasattr(args, "backend"):
            X.check_arguments(args)                                   # none of the documented lines is refused
            assert args.backend in X.BACKENDS
            if args.backend == "zoro":
                assert args.decision_temperature == 1.0 and args.dtype == "fp16" and args.max_input_tokens == 2048
        scripts.append((argv[1].rsplit("/", 1)[1], args.command))
    assert scripts[:7] == [("utterance_experiment.py", command) for command in
                           ("plan", "bench", "bench", "bench", "fit", "bench", "pack")]      # the first session
    assert [command for _, command in scripts[7:]] == ["run", "run", "table", "table", "run", "run", "run", "run",
                                                       "report", "pack"]
    parsed = [X.build_parser().parse_args(shlex.split(line.replace("<backend>", NEG_ARGS))[2:])
              for line in X.COLAB_COMMANDS if "utterance_experiment.py" in line]
    assert [args.block for args in parsed if args.command == "bench"] == ["smoke", "smoke", "fit", "bench"]
    runs = [args for args in parsed if args.command == "run"]
    assert [(args.style, args.block, args.arms, args.backend) for args in runs] == [
        ("paraphrase", "smoke", ["utt_gate_fit"], "zoro"), ("paraphrase", "smoke", ["utt_gate_fit"], "zoro"),
        ("paraphrase", "test", ["prior", "keyword", "label_reader", "true_final"], "cpu-stand-in"),
        ("default", "test", ["prior", "keyword"], "cpu-stand-in"),
        ("paraphrase", "test", ["utt_gate_fit", "utt_gate"], "zoro"), ("paraphrase", "test", ["model_final"], "zoro")]
    for bad in ("utt_gate_fit,unknown", "", "prior,prior"):
        with pytest.raises(SystemExit):
            X.build_parser().parse_args(["run", "--style", "default", "--block", "smoke", "--arms", bad,
                                         *_stand_in(tmp_path / "s")])
    # the launcher is the documented entry point of exactly this main
    spec = importlib.util.spec_from_file_location("utterance_experiment_launcher",
                                                  NEGOTIATION / "utterance_experiment.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.main is X.main
    assert "NOT EXECUTED ON A GPU as of 2026-11-18" in (NEGOTIATION / "utterance_experiment.py").read_text("utf-8")


def test_result_sentences_are_the_design_documents_sentences():
    assert set(X.RESULT_SENTENCES) >= {
        "utterance_reading_demonstrated_on_bench", "partial_reads_final_but_not_separated",
        "utterance_reading_not_demonstrated", "instruction_guard_holds", "instruction_guard_fails",
        "utterance_agent_beats_round_prior_on_paraphrase", "utterance_agent_worse_than_round_prior_on_paraphrase",
        "not_shown", "not_run_no_useful_threshold", "incomplete"}
    for key, sentence in X.RESULT_SENTENCES.items():
        assert sentence.rstrip(".") in DOCUMENT, key                  # verbatim, placeholders included
    assert X.result_sentence("partial_reads_final_but_not_separated", "recall_level") == (
        "표현을 바꾼 최종 선언에는 반응했지만 다음 확인을 통과하지 못했다: recall_level.")
    assert X.result_sentence("not_shown", "90% 구간이 0을 포함함").endswith("드러나지 않았다(90% 구간이 0을 포함함).")
    assert X.result_sentence("incomplete") == "실행이 완료되지 않아 결론을 내리지 않는다."
    assert X.GPU_NOTICE == "NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only."
    for path in sorted((NEGOTIATION / "src" / "utterance_agent").glob("*.py")):
        assert "NOT EXECUTED ON A GPU as of 2026-11-18" in path.read_text(encoding="utf-8"), path.name
