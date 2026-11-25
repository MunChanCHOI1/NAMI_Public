"""CPU tests of the combine rule, the thresholds, the three readers (track A) and the policy-path
wrapper `UtteranceFinalBackend` (track B).

NOT EXECUTED ON A GPU as of 2026-11-18. Every "model" here is a test double; no number in this
file is model evidence. Cases: dev-1100..1129 (D0) only.
"""

import math
from dataclasses import replace

import pytest

from negotiation_agent import evaluation, policy
from negotiation_agent.contracts import ApprovalLedger
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec
from negotiation_agent.evaluation import make_cases
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import request_sha256, reset_response
from negotiation_agent.requests import FINAL_OFFER_QID, build_decision_request
from negotiation_agent.session import NegotiationSession, SimulatedSupplier
from negotiation_agent.simulator import FINAL_MESSAGE, default_buyer_private, initial_state
from negotiation_agent.utterance_probe import with_latest_supplier_message

from utterance_agent import interpreter as I
from utterance_agent import wording as W
from utterance_agent.questions import QIDS, build_utterance_request

HALF = I.FIXED_HALF


def _state(message: str | None = None):
    case = make_cases("dev", 1)[0]
    state = initial_state(case.case_id, case.hidden)
    return state if message is None else with_latest_supplier_message(state, message)


def _answers(final=0.0, walk=0.0, concession=0.0, instruction=0.0) -> dict:
    return {"u_final": final, "u_walkaway": walk, "u_concession": concession, "u_instruction": instruction}


def test_combine_truth_table():
    prior = 0.15
    # no utterance: nothing was asked
    assert I.combine(None, prior, HALF) == (prior, "no_utterance_round_prior", False, False)
    # says final through either question
    assert I.combine(_answers(final=0.9), prior, HALF) == (1.0, "utterance_final", True, False)
    assert I.combine(_answers(walk=0.9), prior, HALF) == (1.0, "utterance_final", True, False)
    # exactly at a threshold counts (every comparison is >=)
    assert I.combine(_answers(final=0.5), prior, HALF) == (1.0, "utterance_final", True, False)
    assert I.combine(_answers(final=0.49), prior, HALF) == (prior, "round_prior", False, False)
    assert I.combine(_answers(final=1.0, instruction=0.5), prior, HALF) == (
        prior, "instruction_flagged_round_prior", True, True)
    assert I.combine(_answers(final=1.0, instruction=0.49), prior, HALF)[1] == "utterance_final"
    # a flag without a final signal still returns the prior and records that it would not have fired
    assert I.combine(_answers(instruction=0.9), prior, HALF) == (prior, "instruction_flagged_round_prior", False, True)
    # the concession answer never changes the estimate
    for concession in (0.0, 1.0):
        assert I.combine(_answers(final=0.9, concession=concession), prior, HALF)[0] == 1.0
        assert I.combine(_answers(final=0.1, concession=concession), prior, HALF)[0] == prior
    # other thresholds move both gates
    strict = I.Thresholds(0.8, 0.3, "utt-fit")
    assert I.combine(_answers(final=0.79), prior, strict)[1] == "round_prior"
    assert I.combine(_answers(walk=0.8), prior, strict)[1] == "utterance_final"
    assert I.combine(_answers(final=0.9, instruction=0.3), prior, strict)[1] == "instruction_flagged_round_prior"
    # combine() never produces the wrapper-only value
    produced = {I.combine(a, prior, HALF)[1] for a in (None, _answers(), _answers(final=1.0),
                                                       _answers(instruction=1.0))}
    assert produced == set(I.VALUE_FROM) - {"public_round_limit_not_asked"}


def test_thresholds_validate_and_roundtrip():
    fitted = I.Thresholds(0.35, 0.6, "utt-fit")
    fitted.validate()
    assert I.Thresholds.from_json(fitted.to_json()) == fitted
    assert HALF.to_json() == {"final": 0.5, "instruction": 0.5, "fitted_on": None}
    assert I.Thresholds.from_json(HALF.to_json()) == HALF
    assert I.Thresholds.from_json({"final": 0.4, "instruction": 0.5, "fitted_on": "utt-smoke-rehearsal"}).fitted_on \
        == "utt-smoke-rehearsal"
    for bad in (I.Thresholds(1.2, 0.5, None), I.Thresholds(0.5, -0.1, None), I.Thresholds(math.nan, 0.5, None),
                I.Thresholds(0.5, math.inf, None), I.Thresholds(True, 0.5, None), I.Thresholds("0.5", 0.5, None),
                I.Thresholds(0.5, 0.5, "somewhere-else")):
        with pytest.raises(ValueError):
            bad.validate()
    for data in ({"final": 0.5, "instruction": 0.5, "fitted_on": None, "extra": 1},   # unknown key
                 {"final": 0.5, "instruction": 0.5},                                  # missing key
                 {"final": 2.0, "instruction": 0.5, "fitted_on": None},               # out of range
                 [0.5, 0.5, None]):
        with pytest.raises(ValueError):
            I.Thresholds.from_json(data)


def test_thresholds_have_no_default_and_the_unfitted_half_enters_by_name_only():
    with pytest.raises(TypeError):
        I.Thresholds()
    with pytest.raises(TypeError):
        I.Thresholds(0.5, 0.5)                       # fitted_on must be stated as well
    assert (HALF.final, HALF.instruction, HALF.fitted_on) == (0.5, 0.5, None)
    with pytest.raises(TypeError):
        I.combine(_answers(), 0.15)                  # combine() has no default thresholds either


class _Recorder:
    """Test double: records the requests it gets. NOT a model."""

    kind, is_model = "interpreter_test_double", True

    def __init__(self, answers):
        self.answers, self.calls, self.last_response = answers, [], "stale"

    def predict(self, state, candidates, request):
        self.calls.append((state, candidates, request))
        self.last_response = {"metadata": {"sequence": len(self.calls)}}
        return dict(self.answers)

    def describe(self):
        return {"kind": self.kind, "is_model": True}


def test_model_reader_asks_the_four_questions_once_and_validates():
    state = _state()
    backend = _Recorder({**_answers(final=0.7, instruction=0.2), "something_else": 0.3})
    reader = I.ModelReader(backend)
    answers, digest, response = reader.read(state)
    assert answers == _answers(final=0.7, instruction=0.2) and tuple(answers) == QIDS
    assert len(backend.calls) == 1
    asked_state, candidates, request = backend.calls[0]
    assert asked_state is state and candidates == [] and request == build_utterance_request(state)
    assert digest == request_sha256(build_utterance_request(state))
    assert response == {"metadata": {"sequence": 1}}
    assert reader.is_model is True and reader.describe()["question_set"] == "utterance-questions-2026-11-18.v1"
    # A missing or out-of-range answer is raised, never replaced by a value.
    for broken in ({"u_final": 0.5}, {**_answers(), "u_walkaway": 1.5}, {**_answers(), "u_final": math.nan}):
        with pytest.raises((KeyError, ValueError)):
            I.ModelReader(_Recorder(broken)).read(state)

    class Failing(_Recorder):
        def predict(self, state, candidates, request):
            raise RuntimeError("test backend failure")

    with pytest.raises(RuntimeError, match="test backend failure"):
        I.ModelReader(Failing({})).read(state)


def test_literal_reader_is_the_keyword_test_in_reader_shape():
    reader = I.LiteralReader()
    assert reader.is_model is False
    answers, digest, response = reader.read(_state(FINAL_MESSAGE))
    assert answers == _answers(final=1.0) and digest is None and response is None
    negated = W.bank_sentence("K1")[1]              # a keyword that says the opposite still fires: literal test
    assert reader.read(_state(negated))[0] == _answers(final=1.0)
    for sentence_id in ("L1", "W1", "N1", "O1"):    # paraphrased final sentences carry no keyword
        assert reader.read(_state(W.bank_sentence(sentence_id)[1]))[0] == _answers()
    # the fixed pair turns that into the keyword rule's own behaviour
    assert I.combine(reader.read(_state(FINAL_MESSAGE))[0], 0.15, HALF)[:2] == (1.0, "utterance_final")
    assert I.combine(reader.read(_state(W.bank_sentence("L1")[1]))[0], 0.15, HALF)[:2] == (0.15, "round_prior")


def test_label_reader_answers_from_the_labels_and_refuses_unknown_sentences():
    reader = I.LabelReader(W.label_text)
    text = lambda *ids: " ".join(W.bank_sentence(i)[1] for i in ids)   # noqa: E731
    expected = {
        text("N1"): _answers(),
        text("L2"): _answers(final=1.0),
        text("W3"): _answers(walk=1.0),
        text("EF"): _answers(final=1.0),
        text("D1"): _answers(),
        text("K2"): _answers(concession=1.0),
        text("O2"): _answers(concession=1.0),
        text("P1"): _answers(concession=1.0),
        text("P2", "L1"): _answers(final=1.0, concession=1.0),
        text("N1", "IF"): _answers(instruction=1.0),
        text("L1", "IO"): _answers(final=1.0, instruction=1.0),
        W.HELD_OUT_FINAL[0][1]: _answers(final=1.0),
    }
    for message, answers in expected.items():
        assert reader.read(_state(message)) == (answers, None, None), message
    assert reader.is_model is False
    with pytest.raises(ValueError):
        reader.read(_state("은행에 없는 문장입니다."))
    with pytest.raises(ValueError):                  # no utterance at all is not the reader's case to answer
        state = _state()
        reader.read(replace(state, history=(replace(state.history[0], message=""),)))
    # The reader depends on the injected function only.
    seen = []

    class Labels:
        core_class, has_payment, has_instruction = "final_walk", False, True

    assert I.LabelReader(lambda message: seen.append(message) or Labels()).read(_state("아무 문장"))[0] == _answers(
        walk=1.0, instruction=1.0)
    assert seen == ["아무 문장"]


# ---- policy-path wrapper (track B) ---------------------------------------------------------------------

PRIVATE = default_buyer_private()


def _constant(final_source: str = "policy_prior") -> ComposedBackend:
    return ComposedBackend(None, EstimateSpec("constant", 0.15, None, final_source))


def _default_request(state):
    candidates = policy.generate_candidates(state, PRIVATE, PolicyConfig())
    return candidates, build_decision_request(state, candidates)


class _Base:
    """Test double of a base backend: records what it is given and answers every question. NOT a model."""

    kind, supports_final_offer = "wrapper_test_base", True

    def __init__(self, is_model: bool = False):
        self.is_model, self.calls, self.last_response = is_model, [], None

    def predict(self, state, candidates, request):
        self.calls.append((state, candidates, request))
        self.last_response = {"base_sequence": len(self.calls)}
        out = {qid: round(0.2 + 0.1 * index, 3) for index, qid in enumerate(request["questions"])}
        out[FINAL_OFFER_QID] = 0.77
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": self.is_model}


class _Reader:
    """Test double of a reader: fixed answers, or a failure. NOT a model unless flagged for a label test."""

    kind = "wrapper_test_reader"

    def __init__(self, answers=None, is_model: bool = False, error: Exception | None = None):
        self.answers, self.is_model, self.error, self.calls = answers, is_model, error, 0

    def read(self, state):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.answers), "reader-request-digest", {"reader_sequence": self.calls}

    def describe(self):
        return {"kind": self.kind, "is_model": self.is_model}


def _wrapper(base=None, reader=None, thresholds=HALF, name="utt_gate"):
    return I.UtteranceFinalBackend(base or _constant(), reader or _Reader(_answers()), thresholds=thresholds,
                                   name=name)


def test_wrapper_passes_the_default_request_object_unchanged():
    state = _state()
    candidates, request = _default_request(state)
    before = request_sha256(request)
    base = _Base()
    wrapper = _wrapper(base, _Reader(_answers(final=0.9)))
    wrapper.predict(state, candidates, request)
    (seen_state, seen_candidates, seen_request), = base.calls
    assert seen_request is request and seen_state is state and seen_candidates is candidates
    assert request_sha256(request) == before
    assert before == "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"   # dev-1100 round 1
    assert set(request["questions"]) == {c.candidate_id for c in candidates} | {FINAL_OFFER_QID}
    # reset_response reaches the base through the wrapper, like any existing wrapper
    reset_response(wrapper)
    assert wrapper.last_response is None and base.last_response is None


def test_wrapper_replaces_only_final_offer():
    state = _state()
    candidates, request = _default_request(state)
    base = _Base()
    expected = _Base().predict(state, candidates, request)
    for answers, final, value_from in ((_answers(final=0.9), 1.0, "utterance_final"),
                                       (_answers(walk=0.5), 1.0, "utterance_final"),
                                       (_answers(final=0.2, concession=1.0), 0.0, "round_prior")):
        wrapper = _wrapper(base, _Reader(answers))
        out = wrapper.predict(state, candidates, request)
        assert {qid: out[qid] for qid in out if qid != FINAL_OFFER_QID} == {
            qid: value for qid, value in expected.items() if qid != FINAL_OFFER_QID}
        assert out[FINAL_OFFER_QID] == final and list(out) == list(request["questions"])
        record = wrapper.last_response["utterance"]
        assert record["final_offer_value_from"] == value_from and record["final_offer_value"] == final
        assert record["base_final_offer"] == 0.77              # the base's own value is recorded, never used
        assert record["answers"] == answers and record["request_sha256"] == "reader-request-digest"
        assert record["round_prior"] == 0.0 and record["rule"] == "utt_gate" and record["status"] == "ok"
        assert record["thresholds"] == HALF.to_json() and record["latest_utterance"] == state.history[0].message
        assert record["reader"] == {"kind": "wrapper_test_reader", "is_model": False}
        assert wrapper.last_response["base_sequence"] >= 1     # the base's evidence is kept beside it
    # a request without the final-offer question gets no final-offer value
    without = {"state": request["state"], "questions": {k: v for k, v in request["questions"].items()
                                                         if k != FINAL_OFFER_QID}}
    assert FINAL_OFFER_QID not in _wrapper(_constant(), _Reader(_answers(final=1.0))).predict(state, candidates, without)
    described = _wrapper().describe()
    assert set(described) == {"kind", "wrapper", "is_model", "estimate_sources", "acceptance_platt_applied",
                              "model_estimates_used", "spec_sha256", "question_set", "thresholds", "prior_table",
                              "reader", "inner"}
    assert described["estimate_sources"] == {"acceptance": "constant", "final_offer": "utt_gate"}
    assert described["wrapper"] == "utterance_final_estimate" and described["prior_table"] == [0.0, 0.15, 0.45, 0.5]
    assert _wrapper().describe()["spec_sha256"] == described["spec_sha256"]
    assert _wrapper(thresholds=I.Thresholds(0.3, 0.5, "utt-fit"), name="utt_gate_fit").describe()["spec_sha256"] \
        != described["spec_sha256"]


def test_literal_reader_wrapper_equals_keyword_composition_on_d0():
    """The wrapper adds nothing but its reader: with the literal reader it IS the keyword composition."""
    kept = ("status", "agreement", "realized_saving", "net_saving", "rounds_used", "counters_sent",
            "approval_cards", "authority_violations")
    fired = 0
    for case in make_cases("dev", 30):
        wrapped, _ = evaluation.run_case(case, PRIVATE, _wrapper(_constant(), I.LiteralReader(), name="literal"),
                                         policy.decide, PolicyConfig())
        composed, _ = evaluation.run_case(case, PRIVATE, _constant("keyword"), policy.decide, PolicyConfig())
        assert {key: wrapped[key] for key in kept} == {key: composed[key] for key in kept}, case.case_id
        assert wrapped["final_offer_estimates"] == composed["final_offer_estimates"], case.case_id
        fired += sum(1 for record in wrapped["final_offer_estimates"] if record["p"] == 1.0)
    assert fired > 0                                           # the comparison is not vacuous


def test_reader_failure_stops_for_human_review_and_fills_nothing():
    state = _state()
    wrapper = _wrapper(_constant(), _Reader(error=RuntimeError("reader test failure")))
    decision = policy.decide(state, PRIVATE, wrapper, ApprovalLedger(), PolicyConfig())
    assert decision.action == "human_review" and decision.reasons == ("acceptance_estimate_unavailable",)
    assert decision.model["status"] == "error" and "reader test failure" in decision.model["error"]
    assert "predictions" not in decision.model and "p_final_offer" not in decision.facts
    record = decision.model["response"]["utterance"]
    assert record["status"] == "error" and record["answers"] is None and record["final_offer_value"] is None
    assert record["request_sha256"] == request_sha256(build_utterance_request(state))
    # an out-of-range answer of a real reader path is a failure too, not a clipped value
    bad = I.ModelReader(_Recorder({**_answers(), "u_final": 1.5}))
    assert policy.decide(state, PRIVATE, _wrapper(_constant(), bad), ApprovalLedger(),
                         PolicyConfig()).action == "human_review"
    # in a session the negotiation stops for a person
    case = make_cases("dev", 1)[0]
    session = NegotiationSession(state, PRIVATE, wrapper, SimulatedSupplier(case.hidden), PolicyConfig(),
                                 policy.decide)
    assert session.run(approver=lambda request: True) == "needs_review" and session.agreement is None

    # a failing base is a failure as well: the reader is then never asked
    class FailingBase(_Base):
        def predict(self, state, candidates, request):
            raise RuntimeError("base test failure")

    reader = _Reader(_answers(final=1.0))
    assert policy.decide(state, PRIVATE, _wrapper(FailingBase(), reader), ApprovalLedger(),
                         PolicyConfig()).action == "human_review"
    assert reader.calls == 0


def test_any_reader_with_a_bad_answer_stops_for_human_review():
    """The wrapper validates every reader's output, not only ModelReader's (section 5)."""
    state = _state()
    candidates, request = _default_request(state)
    nan = float("nan")
    bad_answers = ({**_answers(walk=0.9), "u_final": nan},            # NaN would be ignored by max() ...
                   {**_answers(final=0.9), "u_walkaway": nan},        # ... or win it, depending on the order
                   {**_answers(), "u_final": 2.0, "u_instruction": -1.0},
                   {"u_final": 1.0, "u_walkaway": 0.0, "u_concession": 0.0})   # a question missing
    for answers in bad_answers:
        wrapper = _wrapper(_constant(), _Reader(answers))
        with pytest.raises((ValueError, KeyError)):
            wrapper.predict(state, candidates, request)
        record = wrapper.last_response["utterance"]
        assert record["status"] == "error" and record["final_offer_value"] is None and record["answers"] is None
        decision = policy.decide(state, PRIVATE, _wrapper(_constant(), _Reader(answers)), ApprovalLedger(),
                                 PolicyConfig())
        assert decision.action == "human_review" and "p_final_offer" not in decision.facts


def test_no_utterance_asks_nothing_and_uses_round_prior():
    state = _state()
    silent = replace(state, history=(replace(state.history[0], message=""),))
    candidates, request = _default_request(silent)
    reader = _Reader(error=RuntimeError("must not be asked"))
    wrapper = _wrapper(_constant(), reader)
    out = wrapper.predict(silent, candidates, request)
    record = wrapper.last_response["utterance"]
    assert reader.calls == 0 and wrapper.reader_calls == 0
    assert out[FINAL_OFFER_QID] == 0.0 == record["round_prior"]
    assert record["final_offer_value_from"] == "no_utterance_round_prior"
    assert (record["latest_utterance"], record["request_sha256"], record["answers"]) == (None, None, None)
    later = replace(silent, current_round=3)                   # the prior is the policy's own table value
    later_candidates, later_request = _default_request(later)
    assert wrapper.predict(later, later_candidates, later_request)[FINAL_OFFER_QID] == 0.45


def test_instruction_flag_falls_back_to_round_prior():
    state = replace(_state(), current_round=2)
    candidates, request = _default_request(state)
    wrapper = _wrapper(_constant(), _Reader(_answers(final=1.0, instruction=0.5)))
    assert wrapper.predict(state, candidates, request)[FINAL_OFFER_QID] == 0.15
    record = wrapper.last_response["utterance"]
    assert record["final_offer_value_from"] == "instruction_flagged_round_prior"
    assert record["instruction_flag"] is True and record["says_final"] is True and record["signal_final"] == 1.0
    below = _wrapper(_constant(), _Reader(_answers(final=1.0, instruction=0.49)))
    assert below.predict(state, candidates, request)[FINAL_OFFER_QID] == 1.0
    strict = _wrapper(_constant(), _Reader(_answers(final=1.0, instruction=0.2)),
                      thresholds=I.Thresholds(0.5, 0.2, "utt-fit"), name="utt_gate_fit")
    assert strict.predict(state, candidates, request)[FINAL_OFFER_QID] == 0.15


def test_wrapper_never_gets_the_plain_real_model_label():
    for base, reader in ((_constant(), _Reader(_answers(), is_model=True)),
                         (_Base(is_model=True), _Reader(_answers(), is_model=True)),
                         (_Base(is_model=True), _Reader(_answers()))):
        wrapper = _wrapper(base, reader)
        assert wrapper.is_model is True and wrapper.kind == "composed_estimates"
        assert evaluation._evidence_class(wrapper) == "real_model_run_composed_estimates"
        composition = evaluation._composition(wrapper)
        assert composition["estimate_sources"]["final_offer"] == "utt_gate"
        assert set(composition) == {"estimate_sources", "acceptance_platt_applied", "model_estimates_used",
                                    "spec_sha256"}
    assert evaluation._evidence_class(_wrapper()) == "cpu_reference_not_model_evidence"
    assert _wrapper(_constant(), _Reader(_answers(), is_model=True)).describe()["model_estimates_used"] == [
        "utterance_reading"]
    # the one name the existing label function reads as "the model's own estimate" cannot be used
    with pytest.raises(ValueError):
        _wrapper(name="model")


def test_last_public_round_asks_nothing_and_is_decided_by_the_round_limit():
    state = _state()
    last = replace(state, current_round=state.max_rounds)
    reader = _Reader(error=RuntimeError("must not be asked on the last public round"))
    wrapper = _wrapper(_constant(), reader)
    decision = policy.decide(last, PRIVATE, wrapper, ApprovalLedger(), PolicyConfig())
    assert decision.action != "human_review" and decision.model["status"] == "ok"
    assert decision.facts["p_final_offer_source"] == "public_round_limit" and decision.facts["p_final_offer"] == 1.0
    record = decision.model["response"]["utterance"]
    assert record["final_offer_value_from"] == "public_round_limit_not_asked"
    assert record["final_offer_value"] == 1.0 and record["request_sha256"] is None
    assert reader.calls == 0
    # one round earlier the same reader is asked and its failure stops the negotiation
    earlier = replace(state, current_round=state.max_rounds - 1)
    assert policy.decide(earlier, PRIVATE, wrapper, ApprovalLedger(), PolicyConfig()).action == "human_review"
    assert reader.calls == 1


def test_wrapper_requires_explicit_thresholds_and_name():
    base, reader = _constant(), _Reader(_answers())
    with pytest.raises(TypeError):
        I.UtteranceFinalBackend(base, reader)
    with pytest.raises(TypeError):
        I.UtteranceFinalBackend(base, reader, thresholds=HALF)
    with pytest.raises(TypeError):
        I.UtteranceFinalBackend(base, reader, name="utt_gate")
    with pytest.raises(TypeError):
        I.UtteranceFinalBackend(base, reader, HALF, "utt_gate")            # keyword-only: no positional slip
    with pytest.raises(TypeError):
        I.UtteranceFinalBackend(base, reader, thresholds=(0.5, 0.5), name="utt_gate")
    with pytest.raises(TypeError):
        I.Thresholds()
    for bad in (I.Thresholds(1.5, 0.5, None), I.Thresholds(0.5, 0.5, "elsewhere")):
        with pytest.raises(ValueError):
            I.UtteranceFinalBackend(base, reader, thresholds=bad, name="utt_gate")
    with pytest.raises(ValueError):
        I.UtteranceFinalBackend(base, reader, thresholds=HALF, name="")
    # the unfitted pair enters by its name only
    assert I.UtteranceFinalBackend(base, reader, thresholds=I.FIXED_HALF, name="utt_gate").thresholds is I.FIXED_HALF
