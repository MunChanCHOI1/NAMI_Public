"""CPU tests of the final-offer reader (protocol final-reader-2026-11-19.v1, Z1 + Z2).

NOT executed on a GPU (CPU reference backends and test doubles only). Every "model" here is a test
double whose final-offer answer is a fixed function of the public state; nothing asserted below is
model evidence. The GPU run of the same protocol is recorded in negotiation/docs/PLAN.md section 16.
"""

import argparse
import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from negotiation_agent import evaluation as E, policy, reader_check as RC, simulator as SIM
from negotiation_agent import estimate_sources as S
from negotiation_agent.contracts import ApprovalLedger
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec, round_prior, says_final
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import request_sha256
from negotiation_agent.requests import (FINAL_OFFER_QID, build_decision_request, build_final_only_request,
                                        final_offer_question, public_view)
from negotiation_agent.simulator import default_buyer_private

PRIVATE = default_buyer_private()
CONFIG = PolicyConfig()
TABLE = CONFIG.final_prior_by_round
READER = RC.READER
GOLDEN_FULL_DEV_1100_ROUND_1 = "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
# The final-only request the 2026-11-19 GPU connection smoke sent for the same state.
GOLDEN_FINAL_ONLY_DEV_1100_ROUND_1 = "e83cb5694bbbb89a2f83c3f8b913f59bbfee9fc60ae6e41aa9e25c5bd2613bca"


class FinalReading:
    """A declared-model test double: acceptance 0.5, final-offer answer `final(state)`. It is NOT a model."""

    kind, is_model, supports_final_offer = "final_reading_test_double", True, True

    def __init__(self, final):
        self.final, self.calls, self.seen, self.last_response = final, 0, [], None

    def predict(self, state, candidates, request):
        self.calls += 1
        self.seen.append((state, list(candidates), json.dumps(request, ensure_ascii=False, sort_keys=True)))
        out = {qid: 0.5 for qid in request["questions"] if qid != FINAL_OFFER_QID}
        if FINAL_OFFER_QID in request["questions"]:
            out[FINAL_OFFER_QID] = float(self.final(state))
        self.last_response = {"metadata": {"request_sha256": request_sha256(request), "sequence": self.calls},
                              "answers": dict(out)}
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": True}

    def close(self):
        pass


def keyword_like(state):
    """The literal test as a probability: a reading that agrees with the keyword everywhere."""
    return 0.9 if says_final(state) else 0.1


def supplier_offers(state):
    return sum(1 for offer in state.history if offer.made_by == "supplier")


def first_state(case_id="dev-1100"):
    case = next(c for c in E.make_cases("dev", 30) if c.case_id == case_id)
    return SIM.initial_state(case.case_id, case.hidden)


@pytest.fixture(scope="module")
def later():
    """A public state the policy reached with at least three supplier offers, and its candidates."""
    inner = FinalReading(lambda state: 0.3)
    backend = ComposedBackend(inner, READER)
    for case in E.make_cases("dev", 30):
        E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)
    found = [(state, candidates) for state, candidates, _ in inner.seen if supplier_offers(state) >= 3]
    assert found
    return found[0]


def with_messages(state, finals):
    """The same state with each supplier message replaced: the literal final sentence where finals[i]."""
    history, k = [], 0
    for offer in state.history:
        if offer.made_by == "supplier":
            history.append(replace(offer, message=SIM.FINAL_MESSAGE if finals[k] else SIM.NEUTRAL_MESSAGE))
            k += 1
        else:
            history.append(offer)
    assert k == len(finals)
    return replace(state, history=tuple(history))


def composition(backend):
    return backend.last_response["composition"]


# ---- what stays exactly as it was ----------------------------------------------------------------------

def test_default_request_and_earlier_specs_are_unchanged():
    state = first_state()
    candidates = policy.generate_candidates(state, PRIVATE, CONFIG)
    assert request_sha256(build_decision_request(state, candidates)) == GOLDEN_FULL_DEV_1100_ROUND_1
    assert EstimateSpec().digest() == "b3c45cd7c6ec52341f2abb3066cb98639bb7ebc9de6538ff3476f04a8ed5ba03"
    assert EstimateSpec("constant", 0.15, None, "policy_prior").digest() == \
        "4cd72983d6da82e59e6e8e1b0550da721048afe96e35f495d02216576833d364"
    for spec in (EstimateSpec(), RC.CONFIGS["K"], RC.CONFIGS["P"]):
        assert list(spec.to_json()) == ["acceptance_source", "acceptance_constant", "acceptance_platt",
                                        "final_source", "final_shrink_weight"]
    assert S.FINAL_SOURCES == ("model", "policy_prior", "keyword", "shrink", "keyword_else_shrink")
    assert "model_threshold" not in S.FINAL_SOURCES            # not a tune-2026-11-18.v1 candidate source


def test_final_only_request_is_the_state_and_the_same_final_question():
    state = first_state()
    candidates = policy.generate_candidates(state, PRIVATE, CONFIG)
    only, full = build_final_only_request(state), build_decision_request(state, candidates)
    assert only == {"state": public_view(state), "questions": {FINAL_OFFER_QID: final_offer_question()}}
    assert only["state"] == full["state"] and only["questions"][FINAL_OFFER_QID] == full["questions"][FINAL_OFFER_QID]
    assert len(full["questions"]) == len(candidates) + 1 > 1
    assert request_sha256(only) == GOLDEN_FINAL_ONLY_DEV_1100_ROUND_1


# ---- the specification ---------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [
    EstimateSpec("constant", 0.15, None, "model_threshold"),                                   # no threshold
    EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=0.0),
    EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=1.2),
    EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=float("nan")),
    EstimateSpec("constant", 0.15, None, "keyword", final_threshold=0.7),                      # threshold elsewhere
    EstimateSpec("constant", 0.15, None, "policy_prior", final_guard=True),                    # nothing to guard
    EstimateSpec("constant", 0.15, None, "model", final_guard=True),
    EstimateSpec("constant", 0.15, None, "keyword_else_shrink", 0.3, final_guard=True),         # not a guarded source
    EstimateSpec("constant", 0.15, None, "keyword", final_guard=1),                            # not a bool
    EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=0.7, request_scope="final"),
    EstimateSpec("model", None, None, "model_threshold", final_threshold=0.7, request_scope="final_offer_only"),
    EstimateSpec("constant", 0.15, None, "keyword", request_scope="final_offer_only"),         # asks nothing
    EstimateSpec("constant", 0.15, None, "policy_prior", request_scope="final_offer_only"),
    EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=0.7, final_guard=True),  # full scope
])
def test_inconsistent_reader_specs_are_refused(spec):
    with pytest.raises(ValueError):
        spec.validate()
    with pytest.raises(ValueError):
        ComposedBackend(FinalReading(keyword_like), spec)


def test_valid_reader_specs():
    for spec in (READER, replace(READER, final_threshold=1.0), replace(READER, final_guard=False),
                 EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=0.7),
                 EstimateSpec("rule", None, None, "model_threshold", final_threshold=0.5, request_scope="final_offer_only"),
                 EstimateSpec("constant", 0.15, None, "keyword", final_guard=True)):
        spec.validate()
    assert READER.uses_model() and READER.model_estimates_used() == ["final_offer"]


def test_twin_of_the_reader_is_the_guarded_keyword_and_asks_nothing():
    twin = READER.twin()
    assert twin == EstimateSpec("constant", 0.15, None, "keyword", final_guard=True) == RC.CONFIGS["T"]
    assert twin.request_scope == "full" and not twin.uses_model()
    assert EstimateSpec("constant", 0.15, None, "model_threshold", final_threshold=0.7).twin() == RC.CONFIGS["K"]
    assert ComposedBackend(None, twin).is_model is False


def test_json_round_trip_of_the_new_keys():
    for spec in (READER, READER.twin(), replace(READER, final_threshold=1)):
        restored = EstimateSpec.from_json(json.loads(json.dumps(spec.to_json())))
        assert restored == spec and restored.digest() == spec.digest()
    assert READER.to_json() == {"acceptance_source": "constant", "acceptance_constant": 0.15, "acceptance_platt": None,
                                "final_source": "model_threshold", "final_shrink_weight": None,
                                "final_threshold": 0.7, "final_guard": True, "request_scope": "final_offer_only"}
    assert replace(READER, final_threshold=1).digest() == replace(READER, final_threshold=1.0).digest()
    for key, bad in (("final_guard", "yes"), ("final_guard", 1), ("final_threshold", "0.7"), ("request_scope", 3)):
        with pytest.raises(ValueError):
            EstimateSpec.from_json({**READER.to_json(), key: bad})


# ---- the threshold and the guard -----------------------------------------------------------------------

def ask(backend, state):
    return backend.predict(state, [], build_final_only_request(state))[FINAL_OFFER_QID]


def by_offer_count(values):
    """A reading that depends only on how many supplier offers the state holds (values[0] for the first)."""
    return lambda state: values[supplier_offers(state) - 1]


def test_threshold_reads_at_or_above_as_final_and_otherwise_uses_the_round_prior(later):
    state, _ = later
    prior = round_prior(state, TABLE)
    assert prior < 1.0
    for reading, value, source in ((0.70, 1.0, "model_threshold_match"), (0.95, 1.0, "model_threshold_match"),
                                   (0.6999, prior, "round_prior"), (0.0, prior, "round_prior")):
        backend = ComposedBackend(FinalReading(lambda s, r=reading: r), replace(READER, final_guard=False))
        assert ask(backend, state) == value
        block = composition(backend)
        assert block["final_offer_value_from"] == source and block["raw_model_predictions"][FINAL_OFFER_QID] == reading
        assert block["final_threshold"] == 0.7 and block["guard"]["enabled"] is False


def test_guard_disbelieves_a_supplier_whose_earlier_final_claim_was_followed_by_another_offer(later):
    state, _ = later
    n = supplier_offers(state)
    earlier = [s for _, s in S.earlier_supplier_states(state)]
    assert len(earlier) == n - 1 and [supplier_offers(s) for s in earlier] == list(range(1, n))
    prior = round_prior(state, TABLE)

    def run(values, ask_earlier=True, spec=READER):
        backend = ComposedBackend(FinalReading(by_offer_count(values)), spec)
        if ask_earlier:
            for previous in earlier:
                ask(backend, previous)
        return ask(backend, state), composition(backend)

    value, block = run([0.9] + [0.2] * (n - 2) + [0.9])
    assert value == prior and block["final_offer_value_from"] == "guard_claim_falsified"
    assert block["guard"] == {"enabled": True, "earlier_claims": 1, "earlier_readings_unknown": 0, "claim_disbelieved": True}

    value, block = run([0.2] * (n - 1) + [0.9])                      # no earlier claim: believed
    assert value == 1.0 and block["final_offer_value_from"] == "model_threshold_match"
    assert block["guard"]["earlier_claims"] == 0 and block["guard"]["claim_disbelieved"] is False

    value, block = run([0.9] * (n - 1) + [0.3])                      # no claim now: nothing to disbelieve
    assert value == prior and block["final_offer_value_from"] == "round_prior"
    assert block["guard"]["earlier_claims"] == n - 1 and block["guard"]["claim_disbelieved"] is False

    value, block = run([0.9] * n, ask_earlier=False)                 # readings never received are not invented
    assert value == 1.0 and block["guard"]["earlier_readings_unknown"] == n - 1 and block["guard"]["earlier_claims"] == 0

    value, block = run([0.9] * n, spec=replace(READER, final_guard=False))
    assert value == 1.0 and block["guard"]["enabled"] is False


def test_keyword_guard_reads_the_earlier_messages_with_the_same_literal_test(later):
    state, _ = later
    n = supplier_offers(state)
    prior = round_prior(state, TABLE)
    guarded, plain = ComposedBackend(None, RC.CONFIGS["T"]), ComposedBackend(None, RC.CONFIGS["K"])

    claimed_twice = with_messages(state, [True] + [False] * (n - 2) + [True])
    assert S.keyword_claims_falsified(claimed_twice) == 1
    assert ask(guarded, claimed_twice) == prior
    assert composition(guarded)["final_offer_value_from"] == "guard_claim_falsified"
    assert composition(guarded)["guard"]["claim_disbelieved"] is True
    assert ask(plain, claimed_twice) == 1.0                         # K is unchanged: no guard
    assert composition(plain)["final_offer_value_from"] == "keyword_match" and "guard" not in composition(plain)

    first_claim = with_messages(state, [False] * (n - 1) + [True])
    assert S.keyword_claims_falsified(first_claim) == 0
    assert ask(guarded, first_claim) == 1.0 and composition(guarded)["final_offer_value_from"] == "keyword_match"


def test_earlier_states_are_prefixes_with_the_round_of_that_offer(later):
    state, _ = later
    for offer, earlier in S.earlier_supplier_states(state):
        assert earlier.history[-1] == offer and earlier.current_round == offer.round_index
        assert earlier.history == state.history[: len(earlier.history)]
        assert (earlier.case_id, earlier.item, earlier.quantity, earlier.max_rounds) == \
            (state.case_id, state.item, state.quantity, state.max_rounds)


def test_every_earlier_reading_is_found_again_in_policy_runs():
    """The guard's lookup must hit for every earlier offer of a negotiation the policy actually ran."""
    backend = ComposedBackend(FinalReading(keyword_like), READER)
    guards = []
    for case in E.make_cases("dev", 30):
        row, _ = E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)
        for record in row["internal_record"]:
            block = ((record.get("model") or {}).get("response") or {}).get("composition") or {}
            if "guard" in block:
                guards.append(block["guard"])
    assert len(guards) > 30 and all(g["earlier_readings_unknown"] == 0 for g in guards)
    assert any(g["earlier_claims"] == 0 for g in guards)


@pytest.mark.parametrize("world", ["standard", "bluff", "negation"])
def test_a_reading_that_agrees_with_the_keyword_gives_exactly_the_twin(world):
    """R with a keyword-shaped reading and T (guarded keyword) must decide identically, guard included."""
    cases = E.make_cases("dev", 30) if world == "standard" else RC.world_cases(40)
    with RC.patched_messages(world):
        rows = {}
        for name, backend in (("R", ComposedBackend(FinalReading(keyword_like), READER)),
                              ("T", ComposedBackend(None, RC.CONFIGS["T"]))):
            rows[name] = [E.run_case(case, PRIVATE, backend, policy.decide, CONFIG)[0] for case in cases]
    sources = {}
    for r, t in zip(rows["R"], rows["T"]):
        assert (r["status"], r["realized_saving"], r["public_transcript"]) == \
            (t["status"], t["realized_saving"], t["public_transcript"])
        for record in r["internal_record"]:
            block = ((record.get("model") or {}).get("response") or {}).get("composition") or {}
            if block:
                source = block["final_offer_value_from"]
                sources[source] = sources.get(source, 0) + 1
    assert sources
    if world == "bluff":
        assert sources.get("guard_claim_falsified", 0) > 0          # the guard was exercised


# ---- the policy's request scope and cache --------------------------------------------------------------

def test_final_only_scope_asks_one_question_and_reuses_an_answer_only_for_the_same_candidates():
    state = first_state()
    inner = FinalReading(lambda s: 0.3)
    backend, cache = ComposedBackend(inner, READER), {}
    first = policy.decide(state, PRIVATE, backend, ApprovalLedger(), CONFIG, cache)
    assert inner.calls == 1 and json.loads(inner.seen[0][2]) == build_final_only_request(state)
    assert first.model["questions"] == 1 and first.model["request_scope"] == "final_offer_only"
    assert first.model["request_sha256"] == GOLDEN_FINAL_ONLY_DEV_1100_ROUND_1 and first.model["is_model"] is True

    second = policy.decide(state, PRIVATE, backend, ApprovalLedger(), CONFIG, cache)
    assert inner.calls == 1 and second.model.get("cached") is True and second.action == first.action
    assert second.options == first.options

    candidates = policy.generate_candidates(state, PRIVATE, CONFIG)
    assert len(candidates) >= 2
    ledger = ApprovalLedger()
    ledger.deny(policy.counter_offer_id(state, state.last_offer("supplier")), candidates[0].terms)
    third = policy.decide(state, PRIVATE, backend, ledger, CONFIG, cache)
    assert inner.calls == 2 and not third.model.get("cached") and len(cache) == 2
    assert third.model["request_sha256"] == first.model["request_sha256"]   # the same public request
    assert all(key.startswith(GOLDEN_FINAL_ONLY_DEV_1100_ROUND_1 + ":") for key in cache)


def test_full_scope_cache_key_is_still_the_plain_request_digest():
    state = first_state()
    cache = {}
    decision = policy.decide(state, PRIVATE, ComposedBackend(FinalReading(lambda s: 0.3),
                                                             EstimateSpec("constant", 0.15, None, "model")),
                             ApprovalLedger(), CONFIG, cache)
    assert list(cache) == [GOLDEN_FULL_DEV_1100_ROUND_1] and "request_scope" not in decision.model


def test_a_final_only_composition_refuses_any_other_request():
    state = first_state()
    candidates = policy.generate_candidates(state, PRIVATE, CONFIG)
    backend = ComposedBackend(FinalReading(lambda s: 0.3), READER)
    with pytest.raises(ValueError):
        backend.predict(state, candidates, build_decision_request(state, candidates))


def test_reader_acceptance_values_are_the_constant_for_every_candidate():
    state = first_state()
    candidates = policy.generate_candidates(state, PRIVATE, CONFIG)
    backend = ComposedBackend(FinalReading(lambda s: 0.3), READER)
    out = backend.predict(state, candidates, build_final_only_request(state))
    assert {c.candidate_id: out[c.candidate_id] for c in candidates} == {c.candidate_id: 0.15 for c in candidates}


def test_reader_runs_are_labelled_by_what_answered():
    assert E._evidence_class(ComposedBackend(FinalReading(keyword_like), READER)) == "real_model_run_composed_estimates"
    from negotiation_agent.backends import KeywordBackend
    assert E._evidence_class(ComposedBackend(KeywordBackend(), READER)) == "cpu_reference_not_model_evidence"
    assert E._evidence_class(ComposedBackend(None, READER.twin())) == "cpu_reference_not_model_evidence"


# ---- message worlds ------------------------------------------------------------------------------------

def test_world_seeds_stay_outside_every_registered_range():
    for split, base in E.SPLIT_SEED_BASE.items():
        last = base + E.MAX_CASES_PER_SPLIT - 1
        assert any(low <= base and last <= high for low, high in RC._RESERVED_SEEDS), split
        for seed in (base, last):
            with pytest.raises(RC.ReaderCheckRefusal):
                RC.world_cases(1, base=seed)
    for base, count in ((1000, 1), (1099, 1), (3899, 1), (3500, 1000)):      # the scenario range; a span into a split
        with pytest.raises(RC.ReaderCheckRefusal):
            RC.world_cases(count, base=base)
    assert RC.world_cases(1, base=3900)[0].case_id == "world-3900"
    with pytest.raises(RC.ReaderCheckRefusal):
        RC.world_cases(0)
    cases = RC.world_cases(400)
    assert cases[0].case_id == "world-7600" and cases[-1].case_id == "world-7999"
    assert {c.split for c in cases} == {"stress_outside_splits"}
    assert [c.hidden for c in RC.world_cases(5)] == [c.hidden for c in cases[:5]]     # deterministic


def test_patched_messages_restores_the_simulator_even_after_an_error():
    original = SIM._message
    with RC.patched_messages("standard"):
        assert SIM._message is original
    with pytest.raises(RuntimeError):
        with RC.patched_messages("bluff"):
            assert SIM._message is not original
            raise RuntimeError("test")
    assert SIM._message is original
    with pytest.raises(ValueError):
        RC.message_world("unknown")


def test_worlds_change_only_the_sentences_of_the_suppliers_they_name():
    cases = RC.world_cases(120)
    changed = {world: 0 for world in RC.WORLDS}
    for case in cases:
        hidden = case.hidden
        for round_index in range(1, 6):
            standard = SIM._message(hidden, round_index)
            for world in RC.WORLDS:
                text = RC.message_world(world)(hidden, round_index)
                if world == "standard" or text == standard:
                    assert text == standard
                    continue
                changed[world] += 1
                if world == "paraphrase":
                    assert hidden.reveals_deadline and round_index >= hidden.deadline_round
                    assert not any(word in text for word in ("최종안", "마지막 제안"))
                else:
                    assert hidden.reveals_payment_preference and not hidden.reveals_deadline
                if world == "negation":
                    assert round_index < hidden.deadline_round and RC.NEGATION in text
                if world == "injection":
                    assert RC.INSTRUCTION in text
                assert not any(str(v) in text for v in (hidden.unit_cost, hidden.reserve_profit))
    assert all(changed[world] > 0 for world in RC.WORLDS if world != "standard")


def test_the_paraphrase_is_chosen_by_the_public_round_only():
    hidden = next(c.hidden for c in RC.world_cases(120) if c.hidden.reveals_deadline)
    other = replace(hidden, unit_cost=hidden.unit_cost + 50, reserve_profit=hidden.reserve_profit + 10_000,
                    opening_profit=hidden.opening_profit + 10_000)
    world = RC.message_world("paraphrase")
    assert all(world(hidden, r) == world(other, r) for r in range(1, 6))
    early = replace(hidden, deadline_round=1)
    used = {p for r in range(1, 6) for p in RC.PARAPHRASES if p in world(early, r)}
    assert used == set(RC.PARAPHRASES)


def test_answers_at_the_public_last_round_are_counted_as_unused():
    def record(source, used):
        return {"facts": {"p_final_offer_source": used},
                "model": {"response": {"composition": {"final_offer_value_from": source}}}}
    row = {"internal_record": [record("model_threshold_match", "backend_estimate"),
                               record("model_threshold_match", "public_round_limit"),
                               record("round_prior", "backend_estimate"), {"facts": {}, "model": {}}]}
    assert RC._final_sources(row) == {"model_threshold_match": 1, "unused_public_round_limit": 1, "round_prior": 1}


def test_a_policy_that_reads_no_message_has_the_same_outcome_in_every_world():
    cases = RC.world_cases(30)
    outcomes = {}
    for world in RC.WORLDS:
        with RC.patched_messages(world):
            rows = [E.run_case(case, PRIVATE, ComposedBackend(None, RC.CONFIGS["P"]), policy.decide, CONFIG)[0]
                    for case in cases]
        outcomes[world] = [(r["status"], r["realized_saving"],
                            [{k: v for k, v in o.items() if k != "message"} if isinstance(o, dict) else o
                             for o in r["public_transcript"]]) for r in rows]
    assert all(outcomes[world] == outcomes["standard"] for world in RC.WORLDS)


# ---- the runner ----------------------------------------------------------------------------------------

def run_main(argv, capsys):
    status = RC.main(argv)
    return status, capsys.readouterr()


@pytest.mark.parametrize("extra", [
    ["--backend", "zoro"],                                                      # no explicit temperature
    ["--backend", "zoro", "--decision-temperature", "0.5"],
    ["--backend", "zoro", "--decision-temperature", "1"],                       # no checkpoints
    ["--backend", "zoro-http", "--url", "http://localhost:1"],                  # cannot bind the server
    ["--backend", "heuristic"],
    ["--backend", "systemone"],
    ["--backend", "keyword", "--world-cases", "401"],
    ["--backend", "keyword", "--world-cases", "0"],
    ["--backend", "keyword", "--parts", "identity"],                            # identity needs dev
])
def test_runner_refuses_before_creating_anything(tmp_path, capsys, extra):
    out = tmp_path / "r"
    status, captured = run_main(extra + ["--output-dir", str(out)], capsys)
    assert status == 2 and "Refused" in captured.err and not out.exists()


def test_runner_refuses_an_existing_directory_and_nothing_to_resume(tmp_path, capsys):
    existing = tmp_path / "e"
    existing.mkdir()
    assert run_main(["--backend", "keyword", "--output-dir", str(existing)], capsys)[0] == 2
    assert list(existing.iterdir()) == []
    assert run_main(["--backend", "keyword", "--resume", "--output-dir", str(tmp_path / "n")], capsys)[0] == 2


def test_requested_identity_holds_no_path():
    args = argparse.Namespace(backend="zoro", dtype="fp16", max_input_tokens=2048, decision_temperature=1.0,
                              original_checkpoint="C:/secret/original", exported_checkpoint="C:/secret/exported")
    identity = RC.requested_identity(args)
    assert identity["protocol"] == RC.PROTOCOL_VERSION and identity["decision_temperature"] == 1.0
    assert "secret" not in json.dumps(identity) and ":/" not in json.dumps(identity)
    assert RC.requested_identity(argparse.Namespace(backend="keyword"))["cpu_stand_in_not_a_model"] is True


def test_cpu_rehearsal_end_to_end_and_resume_from_the_store(tmp_path, capsys):
    out = tmp_path / "c"
    argv = ["--backend", "keyword", "--output-dir", str(out), "--world-cases", "4"]
    status, _ = run_main(argv, capsys)
    assert status == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == RC.PROTOCOL_VERSION and manifest["world_cases"] == 4
    assert manifest["configs"]["R"] == READER.to_json() and manifest["threshold"] == 0.7
    [result_path] = out.glob("result-*.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["complete"] is True and result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert [c["case_id"] for c in result["smoke"]["calls"]] == list(RC.SMOKE_CASES)
    assert result["smoke"]["calls"][0]["request_sha256"] == GOLDEN_FINAL_ONLY_DEV_1100_ROUND_1
    assert set(result["dev"]["summary"]) == {"R", "T", "K", "P", "rule"}
    assert result["identity"]["states"] == RC.IDENTITY_STATES
    assert result["identity"]["reading"] == "final_only_matches_full_request"   # the keyword ignores the scope
    assert list(result["worlds"]["by_world"]) == list(RC.WORLDS)
    assert result["worlds"]["label"].startswith("exploratory")
    # A keyword stand-in reads no paraphrase, so neither reading can be shown by it.
    assert result["worlds"]["overall_reading"] == "reader_robustness_not_shown"
    assert result["worlds"]["model_contribution_reading"] == "model_reading_contribution_not_shown"
    assert all(w["R_minus_T"]["mean_difference"] == 0 for w in result["worlds"]["by_world"].values())
    [journal] = out.glob("cases-*.jsonl")
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 * RC.DEV_CASES + 4 * 4 * len(RC.WORLDS)
    first_store = result["backend"]["prediction_store"]
    assert first_store["live_requests"] > 0 and first_store["inner_built"] is True

    assert run_main(argv, capsys)[0] == 2                                       # never overwritten
    changed = ["--backend", "keyword", "--output-dir", str(out), "--world-cases", "5", "--resume"]
    assert run_main(changed, capsys)[0] == 2                                    # a different manifest
    assert run_main(argv + ["--resume"], capsys)[0] == 0
    results = sorted(out.glob("result-*.json"))
    assert len(results) == 2
    again = json.loads(next(p for p in results if p != result_path).read_text(encoding="utf-8"))
    store = again["backend"]["prediction_store"]
    assert again["complete"] is True and store["live_requests"] == 0 and store["inner_built"] is False
    assert store["store_hits"] > 0
    assert again["dev"]["summary"] == result["dev"]["summary"] and again["worlds"] == result["worlds"]


def test_runner_module_names_no_evaluation_split():
    tree = ast.parse(Path(RC.__file__).read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "eval" not in literals and "calibration" not in literals
    assert not any("evidence/" in text or "holdout_history" in text for text in literals)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "make_cases"]
    assert calls and all(isinstance(c.args[0], ast.Constant) and c.args[0].value == "dev" for c in calls)
