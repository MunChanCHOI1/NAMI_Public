"""CPU tests of the estimate composition.

NOT executed on a GPU as of 2026-11-18 (CPU reference backends and test doubles only): every
"model" here is a test double built on the CPU keyword reference. The savings asserted below
are evaluator-side CPU stand-ins on synthetic suppliers, not model evidence.
"""

import ast
import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from negotiation_agent import evaluation as E, policy
from negotiation_agent import estimate_sources as S
from negotiation_agent.backends import CalibratedBackend, HeuristicBackend, KeywordBackend
from negotiation_agent.contracts import ApprovalLedger
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec, round_prior, says_final
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import request_sha256, reset_response
from negotiation_agent.requests import FINAL_OFFER_QID, build_decision_request
from negotiation_agent.simulator import default_buyer_private

PRIVATE = default_buyer_private()
CONFIG = PolicyConfig()
TABLE = CONFIG.final_prior_by_round
GOLDEN_DEV_1100_ROUND_1 = "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
MODULE = Path(S.__file__)

CONSTANT_PRIOR = EstimateSpec("constant", 0.15, None, "policy_prior")
CONSTANT_KEYWORD = EstimateSpec("constant", 0.15, None, "keyword")
RULE_PRIOR = EstimateSpec("rule", None, None, "policy_prior")
RULE_KEYWORD = EstimateSpec("rule", None, None, "keyword")
PLATT = {"a": 1.7, "b": -0.4, "fitted_on_split": "calibration", "source_sha256": "0" * 64}


class StandInModel:
    """KeywordBackend values behind a declared-model test double. It is NOT a model."""

    kind, is_model, supports_final_offer = "stand_in_model_test_double", True, True

    def __init__(self, final=None, fail=None, on_call=lambda: None, extra_response=None):
        self.calls, self.seen, self.last_response = 0, [], None
        self.final, self.fail, self.on_call, self.extra_response = final, fail, on_call, extra_response or {}
        self.closed = 0
        self._base = KeywordBackend()

    def predict(self, state, candidates, request):
        self.on_call()
        self.calls += 1
        self.seen.append((state, list(candidates), json.dumps(request, ensure_ascii=False)))
        if self.fail == "error":
            raise RuntimeError("test backend failure")
        if self.fail == "interrupt":
            raise KeyboardInterrupt("test interruption")
        out = self._base.predict(state, candidates, request)
        if self.final is not None:
            out[FINAL_OFFER_QID] = self.final(state) if callable(self.final) else self.final
        if self.fail == "nan_final":
            out[FINAL_OFFER_QID] = float("nan")
        self.last_response = {"metadata": {"request_sha256": request_sha256(request), "sequence": self.calls},
                              "answers": dict(out), **self.extra_response}
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True}

    def close(self):
        self.closed += 1


def decide(backend, state, config=CONFIG):
    return policy.decide(state, PRIVATE, backend, ApprovalLedger(), config)


def same_decision(first, second):
    return (first.action, first.options, first.facts, first.reasons, first.approval_request) == \
        (second.action, second.options, second.facts, second.reasons, second.approval_request)


@pytest.fixture(scope="module")
def reached():
    """Public states (with their candidates and requests) the policy reaches on the 30 dev cases."""
    recorder = StandInModel()
    for case in E.make_cases("dev", 30):
        E.run_case(case, PRIVATE, recorder, policy.decide, CONFIG)
    states = [(state, candidates, json.loads(text)) for state, candidates, text in recorder.seen]
    rounds = {state.current_round for state, _, _ in states}
    assert rounds == {1, 2, 3, 4, 5}                      # includes the public last round
    assert sum(says_final(state) for state, _, _ in states) >= 3
    return states


def run_cases(backend_factory, count, config=CONFIG):
    return [E.run_case(case, PRIVATE, backend_factory(), policy.decide, config, True)[0]
            for case in E.make_cases("dev", count)]


# ---- spec ------------------------------------------------------------------------------------------

def test_default_spec_is_todays_agent_and_valid():
    spec = EstimateSpec()
    spec.validate()
    assert (spec.acceptance_source, spec.final_source) == ("model", "model")
    assert spec.uses_model() and spec.model_estimates_used() == ["acceptance", "final_offer"]
    assert S.ACCEPTANCE_SOURCES == ("model", "rule", "constant")
    assert S.FINAL_SOURCES == ("model", "policy_prior", "keyword", "shrink", "keyword_else_shrink")
    assert S.TWIN_ACCEPTANCE_CONSTANT == 0.15


@pytest.mark.parametrize("spec, uses, estimates", [
    (EstimateSpec("model", None, None, "policy_prior"), True, ["acceptance"]),
    (EstimateSpec("model", None, None, "keyword"), True, ["acceptance"]),
    (EstimateSpec("model", None, None, "shrink", 0.3), True, ["acceptance", "final_offer"]),
    (EstimateSpec("constant", 0.15, None, "shrink", 0.3), True, ["final_offer"]),
    (EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.3), True, ["final_offer"]),
    (EstimateSpec("rule", None, None, "model"), True, ["final_offer"]),
    (EstimateSpec("model", None, dict(PLATT), "keyword"), True, ["acceptance"]),
    (CONSTANT_PRIOR, False, []), (CONSTANT_KEYWORD, False, []), (RULE_PRIOR, False, []), (RULE_KEYWORD, False, []),
])
def test_which_estimates_come_from_the_model(spec, uses, estimates):
    spec.validate()
    assert spec.uses_model() is uses and spec.model_estimates_used() == estimates


@pytest.mark.parametrize("spec", [
    EstimateSpec(acceptance_source="zoro"),
    EstimateSpec(final_source="prior"),
    EstimateSpec(acceptance_source="constant"),                                  # constant needs its value
    EstimateSpec(acceptance_source="constant", acceptance_constant=1.2),
    EstimateSpec(acceptance_source="constant", acceptance_constant=-0.1),
    EstimateSpec(acceptance_source="constant", acceptance_constant=float("nan")),
    EstimateSpec(acceptance_source="constant", acceptance_constant=True),
    EstimateSpec(acceptance_source="model", acceptance_constant=0.15),           # a value without its source
    EstimateSpec(acceptance_source="rule", acceptance_constant=0.15),
    EstimateSpec(acceptance_source="rule", acceptance_platt=dict(PLATT)),        # a map only maps model values
    EstimateSpec(acceptance_source="constant", acceptance_constant=0.15, acceptance_platt=dict(PLATT)),
    EstimateSpec(acceptance_platt={"a": 1.0, "fitted_on_split": "calibration"}),
    EstimateSpec(acceptance_platt={"a": 1.0, "b": float("inf"), "fitted_on_split": "calibration"}),
    EstimateSpec(acceptance_platt={"a": "1", "b": 0.0, "fitted_on_split": "calibration"}),
    EstimateSpec(acceptance_platt={"a": 1.0, "b": 0.0}),                         # where it was fitted is unknown
    EstimateSpec(acceptance_platt={"a": 1.0, "b": 0.0, "fitted_on_split": "dev"}),
    EstimateSpec(acceptance_platt={"a": 1.0, "b": 0.0, "fitted_on_split": "calibration", "extra": {1, 2}}),
    EstimateSpec(acceptance_platt=[1.0, 0.0]),
    EstimateSpec(final_source="shrink"),                                         # shrink needs its weight
    EstimateSpec(final_source="keyword_else_shrink"),
    EstimateSpec(final_source="shrink", final_shrink_weight=1.01),
    EstimateSpec(final_source="shrink", final_shrink_weight=-0.01),
    EstimateSpec(final_source="shrink", final_shrink_weight=float("nan")),
    EstimateSpec(final_source="model", final_shrink_weight=0.5),                 # a weight without its source
    EstimateSpec(final_source="keyword", final_shrink_weight=0.5),
    EstimateSpec(final_source="policy_prior", final_shrink_weight=0.0),
])
def test_inconsistent_specs_are_refused_by_validate_and_by_the_backend(spec):
    with pytest.raises(ValueError):
        spec.validate()
    with pytest.raises(ValueError):
        ComposedBackend(StandInModel(), spec)


def test_boundary_values_are_accepted():
    for spec in (EstimateSpec("constant", 0.0, None, "shrink", 0.0), EstimateSpec("constant", 1, None, "shrink", 1),
                 EstimateSpec("model", None, {"a": 1, "b": 0, "fitted_on_split": "calibration"}, "model")):
        spec.validate()


@pytest.mark.parametrize("spec, twin", [
    (EstimateSpec(), CONSTANT_PRIOR),                                                              # M0
    (EstimateSpec("model", None, None, "policy_prior"), CONSTANT_PRIOR),                           # M1
    (EstimateSpec("model", None, None, "keyword"), CONSTANT_KEYWORD),                              # M2
    (EstimateSpec("model", None, None, "shrink", 0.4), CONSTANT_PRIOR),                            # M3
    (EstimateSpec("constant", 0.15, None, "shrink", 0.4), CONSTANT_PRIOR),                         # F1
    (EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.4), CONSTANT_KEYWORD),          # F2
    (EstimateSpec("model", None, dict(PLATT), "model"), CONSTANT_PRIOR),                           # M4
    (EstimateSpec("model", None, dict(PLATT), "keyword"), CONSTANT_KEYWORD),                       # M5
    (EstimateSpec("model", None, None, "keyword_else_shrink", 0.4), CONSTANT_KEYWORD),
    (EstimateSpec("rule", None, None, "model"), RULE_PRIOR),
    (EstimateSpec("rule", None, None, "keyword_else_shrink", 1.0), RULE_KEYWORD),
    (EstimateSpec("constant", 0.3, None, "model"), EstimateSpec("constant", 0.3, None, "policy_prior")),
    (CONSTANT_PRIOR, CONSTANT_PRIOR), (CONSTANT_KEYWORD, CONSTANT_KEYWORD),
    (RULE_PRIOR, RULE_PRIOR), (RULE_KEYWORD, RULE_KEYWORD),
])
def test_twin_table(spec, twin):
    result = spec.twin()
    assert result == twin
    result.validate()
    assert not result.uses_model() and result.model_estimates_used() == []
    assert result.acceptance_platt is None and result.final_shrink_weight is None
    assert result.twin() == result


def test_twin_of_a_template_without_fitted_values_needs_no_validation():
    assert EstimateSpec(final_source="shrink").twin() == CONSTANT_PRIOR
    assert EstimateSpec("constant", 0.15, None, "keyword_else_shrink").twin() == CONSTANT_KEYWORD


def test_json_round_trip_unknown_keys_and_digest():
    specs = [EstimateSpec(), CONSTANT_PRIOR, CONSTANT_KEYWORD, RULE_KEYWORD,
             EstimateSpec("model", None, None, "shrink", 0.4),
             EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.4),
             EstimateSpec("model", None, dict(PLATT), "keyword"),
             EstimateSpec(final_source="shrink")]          # an unresolved template round-trips too
    for spec in specs:
        data = spec.to_json()
        assert set(data) == {"acceptance_source", "acceptance_constant", "acceptance_platt", "final_source",
                             "final_shrink_weight"}
        through_text = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
        restored = EstimateSpec.from_json(through_text)
        assert restored == spec and restored.to_json() == data
        assert restored.digest() == spec.digest() and len(spec.digest()) == 64
        assert hash(restored) == hash(spec)
    assert len({spec.digest() for spec in specs}) == len(specs)
    assert len(set(specs)) == len(specs)                   # usable for de-duplication, also with a map
    # Pinned: a change of the canonical form would silently rename every stored configuration.
    assert EstimateSpec().digest() == "b3c45cd7c6ec52341f2abb3066cb98639bb7ebc9de6538ff3476f04a8ed5ba03"
    assert CONSTANT_PRIOR.digest() == "4cd72983d6da82e59e6e8e1b0550da721048afe96e35f495d02216576833d364"
    canonical = json.dumps(EstimateSpec().to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert EstimateSpec().digest() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # 1 and 1.0 are the same configuration; missing keys take the defaults.
    assert EstimateSpec("constant", 1, None, "shrink", 0).digest() == \
        EstimateSpec("constant", 1.0, None, "shrink", 0.0).digest()
    assert EstimateSpec.from_json({}) == EstimateSpec()
    assert EstimateSpec.from_json({"acceptance_source": "constant", "acceptance_constant": 0.15,
                                   "final_source": "policy_prior"}) == CONSTANT_PRIOR
    for bad in ({"acceptance_source": "model", "temperature": 2.0}, {"policy": {}}, ["model"], None,
                {"acceptance_source": 3}, {"acceptance_constant": "0.15"}, {"final_shrink_weight": True},
                {"acceptance_platt": "map.json"}):
        with pytest.raises(ValueError):
            EstimateSpec.from_json(bad)


def test_to_json_and_the_spec_do_not_share_the_map_object():
    platt = dict(PLATT)
    spec = EstimateSpec("model", None, platt, "model")
    digest = spec.digest()
    spec.to_json()["acceptance_platt"]["a"] = 99.0
    assert spec.digest() == digest
    backend = ComposedBackend(StandInModel(), spec)
    backend.describe()["spec"]["acceptance_platt"]["a"] = 99.0
    assert backend.describe()["spec"]["acceptance_platt"]["a"] == PLATT["a"]


# ---- public-round prior and keyword test -------------------------------------------------------------

def test_round_prior_matches_the_policy_table_and_the_keyword_backend(reached):
    for state, candidates, request in reached:
        expected = 1.0 if state.current_round >= state.max_rounds else TABLE[min(state.current_round, len(TABLE)) - 1]
        assert round_prior(state, TABLE) == expected
        keyword = KeywordBackend().predict(state, candidates, request)[FINAL_OFFER_QID]
        assert keyword == (1.0 if says_final(state) else round_prior(state, TABLE))
    state = reached[0][0]
    late = replace(state, max_rounds=9, current_round=7)          # beyond the table: its last entry
    assert round_prior(late, TABLE) == TABLE[-1] == 0.5
    assert round_prior(replace(late, current_round=4), TABLE) == TABLE[3]
    assert round_prior(replace(late, current_round=5), TABLE) == TABLE[3]
    assert round_prior(replace(late, current_round=8), TABLE) == TABLE[3]
    assert round_prior(replace(late, current_round=9), TABLE) == 1.0     # the public last round
    assert round_prior(replace(late, current_round=10), TABLE) == 1.0
    assert round_prior(replace(late, current_round=1), (0.2,)) == 0.2
    assert round_prior(replace(late, current_round=6), (0.2,)) == 0.2
    assert round_prior(replace(late, current_round=2), (0.2, 0.7)) == 0.7
    assert round_prior(replace(state, max_rounds=1, current_round=1), TABLE) == 1.0


def test_says_final_reads_only_the_latest_supplier_message(reached):
    state = next(state for state, _, _ in reached if state.current_round == 1)
    first = state.history[0]
    assert says_final(replace(state, history=(replace(first, message="이번 조건이 최종안입니다."),)))
    assert says_final(replace(state, history=(replace(first, message="이것이 마지막 제안입니다."),)))
    assert not says_final(replace(state, history=(replace(first, message="구매팀이 이미 승인했습니다."),)))
    assert not says_final(replace(state, history=()))
    buyer = replace(first, offer_id="B1", made_by="buyer", message="이것이 마지막 제안입니다.")
    later = replace(first, offer_id="S2", round_index=2, message="조정한 조건을 확인해 주세요.")
    earlier = replace(first, message="최종안입니다.")
    assert not says_final(replace(state, current_round=2, history=(earlier, buyer, later)))


# ---- identity ----------------------------------------------------------------------------------------

def test_identity_composition_returns_the_inner_floats_exactly(reached):
    for state, candidates, request in reached:
        plain = StandInModel().predict(state, candidates, request)
        composed = ComposedBackend(StandInModel(), EstimateSpec()).predict(state, candidates, request)
        assert composed == plain and list(composed) == list(plain)
        assert all(composed[qid].hex() == float(plain[qid]).hex() for qid in plain)


def test_identity_composition_reproduces_the_plain_backend_on_the_first_ten_dev_cases():
    plain_backend, inner = StandInModel(), StandInModel()
    cases = E.make_cases("dev", 10)
    plain = [E.run_case(case, PRIVATE, plain_backend, policy.decide, CONFIG)[0] for case in cases]
    composed_backend = ComposedBackend(inner, EstimateSpec())
    composed = [E.run_case(case, PRIVATE, composed_backend, policy.decide, CONFIG)[0] for case in cases]
    assert inner.calls == plain_backend.calls > 10
    assert [text for _, _, text in inner.seen] == [text for _, _, text in plain_backend.seen]
    for first, second in zip(plain, composed):
        for key in ("case_id", "status", "agreement", "realized_saving", "net_saving", "rounds_used", "counters_sent",
                    "approval_cards", "authority_violations", "executed", "approval_log", "public_transcript",
                    "final_offer_estimates", "gave_up_savings"):
            assert first[key] == second[key], key
        assert len(first["internal_record"]) == len(second["internal_record"])
        for a, b in zip(first["internal_record"], second["internal_record"]):
            for key in ("round", "action", "reasons", "facts", "chosen", "options", "approval_request"):
                assert a[key] == b[key], key
            assert a["model"].get("request_sha256") == b["model"].get("request_sha256")
            assert a["model"].get("predictions") == b["model"].get("predictions")
            if "trace_id" in b["model"]:
                assert a["model"]["backend"] == "stand_in_model_test_double"
                assert b["model"]["backend"] == "composed_estimates" and b["model"]["is_model"] is True


# ---- the request is never changed ----------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    EstimateSpec(), EstimateSpec("model", None, None, "policy_prior"), EstimateSpec("model", None, None, "keyword"),
    EstimateSpec("model", None, None, "shrink", 0.3), EstimateSpec("constant", 0.15, None, "shrink", 0.3),
    EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.3),
    EstimateSpec("model", None, dict(PLATT), "model"), EstimateSpec("model", None, dict(PLATT), "keyword"),
])
def test_inner_backend_sees_the_default_request_byte_for_byte(spec):
    inner = StandInModel()
    backend = ComposedBackend(inner, spec)
    rows = [E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)[0] for case in E.make_cases("dev", 10)]
    traced = [record["model"] for row in rows for record in row["internal_record"] if "trace_id" in record["model"]]
    assert inner.calls == len(traced) == len(inner.seen) > 0        # exactly one inner call per request
    for (state, candidates, text), model in zip(inner.seen, traced):
        default = build_decision_request(state, candidates)
        assert text == json.dumps(default, ensure_ascii=False)
        assert request_sha256(json.loads(text)) == request_sha256(default) == model["request_sha256"]
        assert FINAL_OFFER_QID in default["questions"]              # the question stays even when unused
    assert request_sha256(json.loads(inner.seen[0][2])) == GOLDEN_DEV_1100_ROUND_1


def test_predict_does_not_mutate_the_request(reached):
    state, candidates, request = reached[0]
    for spec in (EstimateSpec(), CONSTANT_PRIOR, RULE_KEYWORD, EstimateSpec("model", None, dict(PLATT), "shrink", 0.5)):
        before = copy.deepcopy(request)
        ComposedBackend(StandInModel(), spec).predict(state, candidates, request)
        assert request == before and request_sha256(request) == request_sha256(before)


# ---- final-offer sources in real policy runs -----------------------------------------------------------

def test_policy_prior_omits_the_value_and_the_policy_uses_its_own_table(reached):
    used = 0
    for state, candidates, request in reached:
        backend = ComposedBackend(None, RULE_PRIOR)
        assert FINAL_OFFER_QID not in backend.predict(state, candidates, request)
        composed, plain = decide(ComposedBackend(None, RULE_PRIOR), state), decide(HeuristicBackend(), state)
        assert same_decision(composed, plain)
        last = state.current_round >= state.max_rounds
        assert composed.facts["p_final_offer_source"] == ("public_round_limit" if last else "policy_prior")
        assert composed.facts["p_final_offer"] == (1.0 if last else TABLE[min(state.current_round, len(TABLE)) - 1])
        block = composed.model["response"]["composition"]
        assert block["final_offer_value_from"] == "omitted_policy_uses_its_prior"
        assert FINAL_OFFER_QID not in block["predictions"] and FINAL_OFFER_QID not in composed.model["predictions"]
        # With a model for acceptance the answer to the final-offer question is still demanded, then left unused.
        inner = StandInModel(final=0.93)
        modelled = decide(ComposedBackend(inner, EstimateSpec("model", None, None, "policy_prior")), state)
        assert same_decision(modelled, plain) and inner.calls == 1
        assert modelled.model["response"]["composition"]["raw_model_predictions"][FINAL_OFFER_QID] == 0.93
        used += 1
    assert used == len(reached) > 50


def test_keyword_source_equals_the_keyword_backend_on_the_same_states(reached):
    matched = 0
    for state, candidates, request in reached:
        composed, plain = decide(ComposedBackend(None, RULE_KEYWORD), state), decide(KeywordBackend(), state)
        assert same_decision(composed, plain)
        last = state.current_round >= state.max_rounds
        assert composed.facts["p_final_offer_source"] == ("public_round_limit" if last else "backend_estimate")
        expected = 1.0 if says_final(state) or last else TABLE[min(state.current_round, len(TABLE)) - 1]
        assert composed.facts["p_final_offer"] == expected
        assert composed.model["predictions"][FINAL_OFFER_QID] == expected
        block = composed.model["response"]["composition"]
        assert block["final_offer_value_from"] == ("keyword_match" if says_final(state) else "round_prior")
        assert block["says_final"] is says_final(state) and block["round_prior"] == round_prior(state, TABLE)
        # A model's own final-offer value is not used by this source, whatever it says.
        modelled = decide(ComposedBackend(StandInModel(final=0.02), EstimateSpec("model", None, None, "keyword")), state)
        assert same_decision(modelled, plain)
        matched += says_final(state)
    assert matched >= 3


@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
def test_shrink_and_keyword_else_shrink_values_in_policy_runs(reached, weight):
    p_model = lambda state: 0.9 - 0.1 * state.current_round    # noqa: E731 - a recognisable stand-in value
    kinds = set()
    for state, candidates, request in reached:
        last, prior = state.current_round >= state.max_rounds, round_prior(state, TABLE)
        shrunk = weight * p_model(state) + (1 - weight) * prior
        for source, expected, origin in (
            ("shrink", shrunk, "shrink"),
            ("keyword_else_shrink", 1.0 if says_final(state) else shrunk, "keyword_match" if says_final(state) else "shrink"),
        ):
            inner = StandInModel(final=p_model)
            backend = ComposedBackend(inner, EstimateSpec("constant", 0.15, None, source, weight))
            decision = decide(backend, state)
            assert inner.calls == 1 and decision.model["status"] == "ok"
            assert decision.model["predictions"][FINAL_OFFER_QID] == expected
            assert decision.facts["p_final_offer_source"] == ("public_round_limit" if last else "backend_estimate")
            assert decision.facts["p_final_offer"] == (1.0 if last else expected)
            block = decision.model["response"]["composition"]
            assert block["final_offer_value_from"] == origin
            assert block["raw_model_predictions"][FINAL_OFFER_QID] == p_model(state)
            assert all(decision.model["predictions"][c.candidate_id] == 0.15 for c in candidates)
            kinds.add(origin)
        if weight == 1.0:      # the model value itself
            assert shrunk == p_model(state)
            assert same_decision(decide(ComposedBackend(StandInModel(final=p_model),
                                                        EstimateSpec("constant", 0.15, None, "shrink", 1.0)), state),
                                 decide(ComposedBackend(StandInModel(final=p_model),
                                                        EstimateSpec("constant", 0.15, None, "model")), state))
        if weight == 0.0:      # the public-round prior itself: the same choice as the policy's own prior
            assert shrunk == prior
            own = decide(ComposedBackend(None, CONSTANT_PRIOR), state)
            zero = decide(ComposedBackend(StandInModel(final=p_model),
                                          EstimateSpec("constant", 0.15, None, "shrink", 0.0)), state)
            assert (zero.action, zero.options, zero.reasons) == (own.action, own.options, own.reasons)
            assert zero.facts["p_final_offer"] == own.facts["p_final_offer"]
    assert kinds == {"shrink", "keyword_match"}


def test_last_round_and_table_index_edges_in_policy_runs(reached):
    last = [item for item in reached if item[0].current_round >= item[0].max_rounds]
    assert last
    for state, candidates, request in last:
        for backend in (ComposedBackend(None, CONSTANT_PRIOR), ComposedBackend(None, CONSTANT_KEYWORD),
                        ComposedBackend(StandInModel(final=0.1), EstimateSpec("constant", 0.15, None, "shrink", 0.5)),
                        ComposedBackend(StandInModel(final=0.1), EstimateSpec("constant", 0.15, None, "model"))):
            decision = decide(backend, state)
            assert decision.facts["p_final_offer_source"] == "public_round_limit"
            assert decision.facts["p_final_offer"] == 1.0
            assert decision.model["response"]["composition"]["round_prior"] == 1.0
        assert ComposedBackend(None, CONSTANT_KEYWORD).predict(state, candidates, request)[FINAL_OFFER_QID] == 1.0
        shrunk = ComposedBackend(StandInModel(final=0.1), EstimateSpec("constant", 0.15, None, "shrink", 0.5))
        assert shrunk.predict(state, candidates, request)[FINAL_OFFER_QID] == 0.5 * 0.1 + 0.5 * 1.0

    # Beyond the table but before the public round limit: the table's last entry, as the policy itself reads it.
    short = PolicyConfig(final_prior_by_round=(0.2, 0.7))
    checked = set()
    for state, _, _ in reached:
        if says_final(state) or state.current_round >= state.max_rounds:
            continue
        for config, table in ((CONFIG, TABLE), (short, short.final_prior_by_round)):
            own = decide(ComposedBackend(None, CONSTANT_PRIOR, table), state, config)
            keyword = decide(ComposedBackend(None, CONSTANT_KEYWORD, table), state, config)
            expected = table[min(state.current_round, len(table)) - 1]
            assert own.facts["p_final_offer_source"] == "policy_prior"
            assert keyword.facts["p_final_offer_source"] == "backend_estimate"
            assert own.facts["p_final_offer"] == keyword.facts["p_final_offer"] == expected
            assert (own.action, own.options) == (keyword.action, keyword.options)
            checked.add((state.current_round, len(table)))
    assert {(3, 2), (4, 2), (4, 4), (1, 4)} <= checked          # rounds 3 and 4 clamp to the short table's end


def test_default_prior_table_is_the_policy_default_and_a_custom_one_is_used():
    assert ComposedBackend(None, CONSTANT_KEYWORD).prior_table == PolicyConfig().final_prior_by_round
    assert ComposedBackend(None, CONSTANT_KEYWORD, [0.2, 0.7]).prior_table == (0.2, 0.7)
    for bad in ((), (0.2, 1.5), (0.2, float("nan")), ("0.2",), (True,)):
        with pytest.raises(ValueError):
            ComposedBackend(None, CONSTANT_KEYWORD, bad)


# ---- acceptance sources --------------------------------------------------------------------------------

def test_rule_and_constant_acceptance_values(reached):
    for state, candidates, request in reached[:20]:
        rule = HeuristicBackend().predict(state, candidates, request)
        composed = ComposedBackend(None, RULE_PRIOR).predict(state, candidates, request)
        assert composed == rule
        constant = ComposedBackend(None, EstimateSpec("constant", 0.4, None, "policy_prior")).predict(
            state, candidates, request)
        assert constant == {c.candidate_id: 0.4 for c in candidates}
        mixed = ComposedBackend(StandInModel(final=0.25), EstimateSpec("rule", None, None, "model")).predict(
            state, candidates, request)
        assert mixed == {**rule, FINAL_OFFER_QID: 0.25}


def test_platt_map_is_applied_exactly_like_the_calibrated_backend(reached):
    a, b = PLATT["a"], PLATT["b"]
    for state, candidates, request in reached:
        calibrated = CalibratedBackend(StandInModel(), a, b).predict(state, candidates, request)
        composed_backend = ComposedBackend(StandInModel(), EstimateSpec("model", None, dict(PLATT), "model"))
        composed = composed_backend.predict(state, candidates, request)
        assert composed == calibrated                       # acceptance mapped, final offer passed through
        raw = StandInModel().predict(state, candidates, request)
        for c in candidates:
            p = min(1 - 1e-4, max(1e-4, raw[c.candidate_id]))
            assert composed[c.candidate_id] == 1 / (1 + math.exp(-(a * math.log(p / (1 - p)) + b)))
        assert composed[FINAL_OFFER_QID] == raw[FINAL_OFFER_QID]
        block = composed_backend.last_response["composition"]
        assert block["acceptance_platt_applied"] is True and block["raw_model_predictions"] == raw
        with_keyword = ComposedBackend(StandInModel(final=0.02), EstimateSpec("model", None, dict(PLATT), "keyword"))
        assert with_keyword.predict(state, candidates, request) == calibrated     # the stand-in is the keyword rule
    assert S.platt_map(0.0, 1.0, 0.0) == pytest.approx(1e-4) and S.platt_map(1.0, 1.0, 0.0) == pytest.approx(1 - 1e-4)
    identity = ComposedBackend(StandInModel(), EstimateSpec("model", None, {"a": 1.0, "b": 0.0,
                                                                              "fitted_on_split": "calibration"}, "model"))
    assert identity.describe()["acceptance_platt_applied"] is True


# ---- model-free specs ----------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [CONSTANT_PRIOR, CONSTANT_KEYWORD, RULE_PRIOR, RULE_KEYWORD])
def test_model_free_specs_never_call_the_inner_backend(spec):
    inner = StandInModel(fail="error")          # would fail every decision if it were called
    for backend in (ComposedBackend(inner, spec), ComposedBackend(None, spec)):
        assert backend.is_model is False and backend.uses_inner is False
        rows = [E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)[0] for case in E.make_cases("dev", 10)]
        assert all(row["status"] in ("agreed", "ended_alternative", "supplier_walked") for row in rows)
        for row in rows:
            for record in row["internal_record"]:
                model = record["model"]
                if "trace_id" not in model:
                    continue
                assert model["backend"] == "composed_estimates" and model["is_model"] is False
                assert model["label"] == "규칙 추정(모델 아님)"
                assert set(model["response"]) == {"composition"}
                block = model["response"]["composition"]
                assert block["raw_model_predictions"] is None and block["model_estimates_used"] == []
                assert block["estimate_sources"] == {"acceptance": spec.acceptance_source,
                                                     "final_offer": spec.final_source}
                assert block["predictions"] == model["predictions"]
        description = backend.describe()
        assert description["inner"] is None and description["is_model"] is False
    assert inner.calls == 0 and inner.seen == []


def test_cpu_regression_values_on_dev_1100_to_1129():
    """Evaluator-side CPU stand-ins on synthetic suppliers; not model evidence."""
    def mean(spec):
        rows = run_cases(lambda: ComposedBackend(None, spec), 30)
        assert [row["case_id"] for row in rows] == [f"dev-{seed}" for seed in range(1100, 1130)]
        assert all(row["authority_violations"] == 0 for row in rows)
        return sum(row["realized_saving"] for row in rows) / len(rows)

    assert round(mean(CONSTANT_PRIOR)) == 300_035
    assert round(mean(CONSTANT_KEYWORD)) == 324_233
    assert mean(CONSTANT_PRIOR) == pytest.approx(9_001_036 / 30) and mean(CONSTANT_KEYWORD) == pytest.approx(9_726_977 / 30)
    # The composed rule equals the existing rule policy (284,263.67), so the default behaviour is untouched.
    plain = run_cases(HeuristicBackend, 30)
    assert mean(RULE_PRIOR) == sum(row["realized_saving"] for row in plain) / 30 == pytest.approx(284_263.67, abs=0.01)


# ---- construction refusals -----------------------------------------------------------------------------

class NoFinalStandIn(StandInModel):
    """A stand-in that declares it gives no final-offer estimate, like the rule backend."""

    supports_final_offer = False

    def predict(self, state, candidates, request):
        out = super().predict(state, candidates, request)
        out.pop(FINAL_OFFER_QID)
        return out


class Undeclared:
    """No `supports_final_offer` attribute: the declaration falls back to `is_model`."""

    kind = "undeclared_test_double"

    def __init__(self, is_model):
        self.is_model = is_model

    def describe(self):
        return {"kind": self.kind, "is_model": self.is_model}


def test_a_source_the_inner_backend_cannot_supply_is_refused_at_construction(reached):
    needs_model_final = (EstimateSpec(), EstimateSpec("model", None, None, "shrink", 0.5),
                         EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.5),
                         EstimateSpec("rule", None, None, "model"))
    for spec in needs_model_final:
        for inner in (HeuristicBackend(), NoFinalStandIn(), Undeclared(False)):
            with pytest.raises(ValueError, match="final-offer"):
                ComposedBackend(inner, spec)
        ComposedBackend(Undeclared(True), spec)        # a model is assumed to answer every question
        with pytest.raises(ValueError, match="inner backend is required"):
            ComposedBackend(None, spec)
    for spec in (EstimateSpec("model", None, None, "policy_prior"), EstimateSpec("model", None, None, "keyword")):
        with pytest.raises(ValueError, match="inner backend is required"):
            ComposedBackend(None, spec)
        # Acceptance alone can come from a backend without a final-offer estimate.
        state, candidates, request = reached[0]
        inner = NoFinalStandIn()
        out = ComposedBackend(inner, spec).predict(state, candidates, request)
        assert inner.calls == 1 and set(out) - {FINAL_OFFER_QID} == {c.candidate_id for c in candidates}


def test_composition_over_a_cpu_stand_in_is_not_labelled_as_a_model():
    for spec in (EstimateSpec(), EstimateSpec("model", None, None, "keyword"),
                 EstimateSpec("constant", 0.15, None, "shrink", 0.5)):
        stand_in = ComposedBackend(KeywordBackend(), spec)       # a CPU reference in the model's place
        assert spec.uses_model() and stand_in.uses_inner and stand_in.is_model is False
        assert stand_in.describe()["is_model"] is False and stand_in.describe()["inner"]["kind"] == "rule_keyword"
        assert ComposedBackend(StandInModel(), spec).is_model is True
    state = E.initial_state("dev-1100", E.make_cases("dev", 1)[0].hidden)
    decision = decide(ComposedBackend(KeywordBackend(), EstimateSpec()), state)
    assert decision.model["is_model"] is False and decision.model["label"] == "규칙 추정(모델 아님)"
    assert decide(ComposedBackend(StandInModel(), EstimateSpec()), state).model["label"] == "모델 추정"


def test_evaluate_labels_a_composed_run_by_what_actually_answered():
    """The existing evaluation labels evidence from `is_model`; a composition must not turn a rule into a model."""
    snapshot = {"notice": "test snapshot; no git call"}
    for backend, evidence in ((ComposedBackend(None, CONSTANT_KEYWORD), "cpu_reference_not_model_evidence"),
                              (ComposedBackend(KeywordBackend(), EstimateSpec()), "cpu_reference_not_model_evidence"),
                              # a declared-model test double, as in the other suites: plain only when
                              # both estimates are the model's own unmapped values
                              (ComposedBackend(StandInModel(), EstimateSpec()), "real_model_run"),
                              (ComposedBackend(StandInModel(), EstimateSpec("model", None, None, "keyword")),
                               "real_model_run_composed_estimates")):
        result = E.evaluate("dev", backend, count=2, source_provenance=snapshot)
        assert result["complete"] and result["evidence_class"] == evidence
        assert result["backend"]["kind"] == "composed_estimates" and result["agent_is_rule_policy"] is False
        assert "rule_policy" in result["policies"]       # the rule baseline still runs beside a composed agent


# ---- evidence: responses and descriptions ----------------------------------------------------------------

def test_supports_final_offer_follows_the_final_source():
    assert ComposedBackend(StandInModel(), EstimateSpec("model", None, None, "policy_prior")).supports_final_offer is False
    assert ComposedBackend(None, CONSTANT_PRIOR).supports_final_offer is False
    for spec in (EstimateSpec(), EstimateSpec("model", None, None, "keyword"), EstimateSpec("model", None, None, "shrink", 0.2),
                 EstimateSpec("model", None, None, "keyword_else_shrink", 0.2)):
        assert ComposedBackend(StandInModel(), spec).supports_final_offer is True
    assert ComposedBackend(None, CONSTANT_KEYWORD).supports_final_offer is True


def test_describe_never_reports_a_hybrid_as_the_inner_kind():
    spec = EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.4)
    inner = StandInModel()
    backend = ComposedBackend(inner, spec)
    assert ComposedBackend.kind == backend.kind == "composed_estimates" != inner.kind
    description = backend.describe()
    assert description == {
        "kind": "composed_estimates", "is_model": True,
        "estimate_sources": {"acceptance": "constant", "final_offer": "keyword_else_shrink"},
        "acceptance_platt_applied": False,
        "spec": spec.to_json(), "spec_sha256": spec.digest(), "prior_table": [0.0, 0.15, 0.45, 0.5],
        "model_estimates_used": ["final_offer"], "inner": {"kind": "stand_in_model_test_double", "is_model": True},
    }
    json.dumps(description, allow_nan=False)
    for other in (EstimateSpec(), CONSTANT_PRIOR, RULE_KEYWORD, EstimateSpec("model", None, dict(PLATT), "keyword")):
        described = ComposedBackend(StandInModel(), other).describe()
        assert described["kind"] == "composed_estimates"
        assert (described["inner"] is None) == (not other.uses_model())
    # CalibratedBackend copies the inner kind; the composition must not.
    assert CalibratedBackend(StandInModel(), 1.0, 0.0).kind == "stand_in_model_test_double"


def test_response_keeps_the_inner_evidence_at_the_top_level_and_names_every_source(reached):
    state, candidates, request = next(item for item in reached if says_final(item[0]) and item[0].current_round < 5)
    store_block = {"prediction_store": {"hit": False, "identity": "abcdef012345"}}
    spec = EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.4)
    seen_inside = []
    backend = ComposedBackend(StandInModel(final=0.3, extra_response=store_block,
                                           on_call=lambda: seen_inside.append(backend.last_response)), spec)
    backend.last_response = {"stale": True}
    out = backend.predict(state, candidates, request)
    assert seen_inside == [None]                              # cleared at the start of every predict
    response = backend.last_response
    assert response["prediction_store"] == store_block["prediction_store"]
    assert response["metadata"]["request_sha256"] == request_sha256(request)
    block = response["composition"]
    assert set(block) == {"spec", "spec_sha256", "estimate_sources", "acceptance_platt_applied",
                          "model_estimates_used", "raw_model_predictions", "round_prior", "says_final",
                          "final_offer_value_from", "predictions", "status"}
    assert block["spec"] == spec.to_json() and block["spec_sha256"] == spec.digest()
    assert block["estimate_sources"] == {"acceptance": "constant", "final_offer": "keyword_else_shrink"}
    assert block["model_estimates_used"] == ["final_offer"] and block["status"] == "ok"
    assert block["raw_model_predictions"][FINAL_OFFER_QID] == 0.3
    assert block["says_final"] is True and block["final_offer_value_from"] == "keyword_match"
    assert block["predictions"] == out and out[FINAL_OFFER_QID] == 1.0
    # The model's acceptance values stay visible as raw values, never as what the policy used.
    assert all(out[c.candidate_id] == 0.15 != block["raw_model_predictions"][c.candidate_id] for c in candidates)
    json.dumps(response, allow_nan=False)
    decision = decide(ComposedBackend(StandInModel(final=0.3, extra_response=store_block), spec), state)
    assert decision.model["response"]["prediction_store"]["hit"] is False
    assert decision.model["response"]["composition"]["predictions"] == decision.model["predictions"]


@pytest.mark.parametrize("failure, status", [("error", "error"), ("interrupt", "interrupted"), ("nan_final", "error")])
@pytest.mark.parametrize("spec", [EstimateSpec(), EstimateSpec("model", None, None, "policy_prior"),
                                  EstimateSpec("model", None, None, "keyword"),
                                  EstimateSpec("constant", 0.15, None, "shrink", 0.5)])
def test_a_failed_model_answer_stops_for_review_without_a_composed_value(reached, failure, status, spec):
    state, candidates, request = reached[0]
    inner = StandInModel(fail=failure)
    backend = ComposedBackend(inner, spec)
    decision = decide(backend, state)
    assert decision.action == "human_review" and decision.reasons == ("acceptance_estimate_unavailable",)
    assert decision.model["status"] == status and "predictions" not in decision.model
    assert inner.calls == 1                                   # never retried
    response = decision.model["response"]
    if failure == "nan_final":
        # Also when this composition would not have used the final-offer value: an invalid answer is not half-used.
        block = response["composition"]
        assert block["status"] == "invalid_model_predictions" and block["predictions"] is None
        assert block["raw_model_predictions"][FINAL_OFFER_QID] == {"invalid_nonfinite_float": "nan"}
        assert response["metadata"]["sequence"] == 1
    else:
        assert response is None
    row, _ = E.run_case(E.make_cases("dev", 1)[0], PRIVATE, ComposedBackend(StandInModel(fail=failure), spec),
                        policy.decide, CONFIG)
    assert row["status"] == "needs_review" and row["interrupted"] is (failure == "interrupt")
    assert row["realized_saving"] == 0


def test_a_missing_or_out_of_range_model_answer_is_refused(reached):
    state, candidates, request = reached[0]

    class Dropping(StandInModel):
        def __init__(self, drop=None, **changes):
            super().__init__()
            self.drop, self.changes = drop, changes

        def predict(self, state, candidates, request):
            out = super().predict(state, candidates, request)
            if self.drop:
                out.pop(self.drop)
            out.update(self.changes)
            return out

    for inner in (Dropping(drop=FINAL_OFFER_QID), Dropping(drop=candidates[0].candidate_id),
                  Dropping(**{candidates[0].candidate_id: 1.5}), Dropping(**{FINAL_OFFER_QID: -0.2})):
        for spec in (EstimateSpec(), EstimateSpec("model", None, None, "keyword"),
                     EstimateSpec("constant", 0.15, None, "shrink", 0.5)):
            backend = ComposedBackend(inner, spec)
            with pytest.raises((KeyError, ValueError)):
                backend.predict(state, candidates, request)
            assert backend.last_response["composition"]["predictions"] is None
            assert decide(ComposedBackend(inner, spec), state).action == "human_review"


def test_reset_response_and_close_are_safe_with_and_without_an_inner_backend():
    free = ComposedBackend(None, CONSTANT_PRIOR)
    free.last_response = {"stale": True}
    reset_response(free)
    assert free.last_response is None
    free.close()
    inner = StandInModel()
    inner.last_response = {"stale": True}
    backend = ComposedBackend(inner, EstimateSpec())
    backend.last_response = {"stale": True}
    reset_response(backend)
    assert backend.last_response is None and inner.last_response is None
    backend.close()
    assert inner.closed == 1
    unused = StandInModel()
    ComposedBackend(unused, CONSTANT_PRIOR).close()      # an unused backend the caller built is still closed
    assert unused.closed == 1
    ComposedBackend(HeuristicBackend(), RULE_PRIOR).close()   # no close() on the inner backend


# ---- isolation -----------------------------------------------------------------------------------------

def test_no_private_or_hidden_value_reaches_the_response_or_the_inner_backend():
    hidden_fields = ("unit_cost", "reserve_profit", "opening_profit", "deadline_round", "expedite_cost_per_day",
                     "supplier_hidden", "target_total_cost", "approval_cost", "company_bounds")
    secrets = [PRIVATE.target_total_cost, PRIVATE.company_bounds.max_total_cost, PRIVATE.auto_bounds.max_total_cost]
    assert secrets == [10_200_000, 10_845_000, 10_600_000]

    def leaves(value):
        """Every key and scalar of a JSON value; probabilities are compared as numbers, not as digit strings."""
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from leaves(item)
        else:
            yield value

    responses = 0
    for spec in (EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.4), CONSTANT_KEYWORD, RULE_PRIOR):
        for case in E.make_cases("dev", 10):
            inner = StandInModel()
            backend = ComposedBackend(inner, spec)
            row, _ = E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)
            secret_values = secrets + [case.hidden.unit_cost, case.hidden.reserve_profit, case.hidden.opening_profit]
            for record in row["internal_record"]:
                response = record["model"].get("response")
                if response is None:
                    continue
                responses += 1
                found = list(leaves(response))
                assert not any(isinstance(leaf, str) and any(name in leaf for name in hidden_fields) for leaf in found)
                numbers = [leaf for leaf in found if isinstance(leaf, (int, float)) and not isinstance(leaf, bool)]
                assert not any(number == value for number in numbers for value in secret_values)
                assert all(0.0 <= number <= 1.0 for number in numbers if not isinstance(number, int))
            assert (inner.calls > 0) == spec.uses_model()
            for _, _, text in inner.seen:          # what the inner backend was given: the public request text
                assert not any(name in text for name in hidden_fields)
                assert not any(str(value) in text or f"{value:,}" in text for value in secret_values)
            json.dumps(backend.describe(), allow_nan=False)
    assert responses > 60


def _tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_module_imports_no_private_type_and_names_no_evaluation_split():
    tree, names = _tree(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    assert not names & {"BuyerPrivate", "SupplierHidden", "economics", "simulator", "metrics", "probe", "evaluation",
                        "session", "prediction_store"}
    top_level = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "policy" not in top_level                     # PolicyConfig is read lazily, for the default table only
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "eval" not in literals and not any("evidence/" in text or "holdout" in text for text in literals)


def test_headers_say_where_the_code_has_run():
    # The module has run on a GPU (2026-11-18 Colab T4 tuning, 2026-11-19 local final reader); these
    # tests have not. Each header must say which, and point to the record of the GPU runs.
    module = " ".join(ast.get_docstring(ast.parse(MODULE.read_text(encoding="utf-8"))).split())
    assert "test doubles" in module and "not evidence about the real model" in module
    assert "2026-11-18 Colab T4 tuning run" in module and "PLAN.md section 16.1" in module
    tests = ast.get_docstring(ast.parse(Path(__file__).read_text(encoding="utf-8")))
    assert "NOT executed on a GPU as of 2026-11-18" in tests
    assert "CPU" in tests and "test doubles" in tests


def test_spec_fields_are_exactly_the_documented_ones():
    # The last three were added on 2026-11-19 for final-reader-2026-11-19.v1 (module docstring); their
    # defaults are never written by to_json, so every earlier configuration keeps its digest.
    assert list(asdict(EstimateSpec())) == ["acceptance_source", "acceptance_constant", "acceptance_platt",
                                            "final_source", "final_shrink_weight", "final_threshold",
                                            "final_guard", "request_scope"]
    assert S.shrink(0.8, 0.15, 0.25) == 0.25 * 0.8 + (1 - 0.25) * 0.15
