import json

from negotiation_agent.backends import ConstantBackend, HeuristicBackend, KeywordBackend
from negotiation_agent.probe import build_states
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.utterance_probe import (CLASSES, KEYWORDS, VARIANTS, base_states, build_request, main, run,
                                               summarize, verdict, with_latest_supplier_message)


def test_variants_change_only_the_latest_supplier_message():
    state_id, state = base_states()[4]
    baseline = build_request(state)
    for _cls, _variant, message in VARIANTS:
        request = build_request(with_latest_supplier_message(state, message))
        assert list(request["questions"]) == [FINAL_OFFER_QID]
        assert request["questions"] == baseline["questions"]
        before, after = baseline["state"], request["state"]
        assert before["거래"] == after["거래"] and before["진행"] == after["진행"]
        rows = list(zip(before["제안_이력"], after["제안_이력"]))
        latest = max(i for i, (row, _) in enumerate(rows) if row["제안자"] == "공급사")
        for index, (old, new) in enumerate(rows):
            if index == latest:
                assert {k: v for k, v in old.items() if k != "발언"} == {k: v for k, v in new.items() if k != "발언"}
                assert new["발언"] == message
            else:
                assert old == new


def test_requests_carry_no_hidden_or_private_value():
    hidden = {p.state_id: p.hidden for p in build_states()}
    for state_id, state in base_states():
        text = json.dumps(build_request(with_latest_supplier_message(state, VARIANTS[2][2])), ensure_ascii=False)
        h = hidden[state_id]
        for secret in (h.unit_cost, h.opening_profit, h.reserve_profit, h.expedite_cost_per_day, 10_200_000):
            assert str(secret) not in text
        assert h.style not in text


def test_variant_wording_matches_its_class():
    by_class = {}
    for cls, _variant, message in VARIANTS:
        by_class.setdefault(cls, []).append(message)
    assert set(by_class) == {"neutral", *CLASSES}
    has = lambda message: any(word in message for word in KEYWORDS)  # noqa: E731
    assert all(has(m) for m in by_class["explicit_final"] + by_class["negated_final"])
    assert not any(has(m) for m in by_class["neutral"] + by_class["paraphrase_final"] + by_class["open_to_more"]
                   + by_class["embedded_instruction"])
    assert len({variant for _, variant, _ in VARIANTS}) == len(VARIANTS)


def test_keyword_reference_misses_paraphrases_and_fires_on_negation():
    result = run(KeywordBackend())
    assert result["complete"] and result["evidence_class"] == "cpu_reference_not_model_evidence"
    assert result["scope"]["planned_requests"] == 16 * len(VARIANTS) == len(result["calls"])
    assert result["scope"]["ground_truth_scored"] is False
    by_class = result["metrics"]["by_class"]
    assert by_class["explicit_final"]["share_of_states_up"] == 1.0
    assert by_class["paraphrase_final"]["mean_shift"] == 0.0
    assert by_class["open_to_more"]["mean_shift"] == 0.0
    assert by_class["negated_final"]["mean_shift"] == by_class["explicit_final"]["mean_shift"] > 0
    assert by_class["explicit_minus_negated"]["mean_gap"] == 0.0
    verdict_ = result["verdict"]
    assert verdict_["reading"] == "partial_explicit_statement_only"
    assert verdict_["checks"] == {"explicit_final_raises_estimate": True, "paraphrase_final_raises_estimate": False,
                                  "negation_separated_from_statement": False}
    assert verdict_["counts_as_model_evidence"] is False


def test_constant_gives_no_sensitivity_and_heuristic_is_unsupported_without_filled_values():
    constant = run(ConstantBackend(), max_states=3)
    assert constant["verdict"]["reading"] == "wording_sensitivity_not_demonstrated"
    assert all(row["mean_shift"] == 0.0 for name, row in constant["metrics"]["by_class"].items() if name in CLASSES)
    heuristic = run(HeuristicBackend(), max_states=3)
    assert heuristic["supported"] is False and heuristic["complete"] is False
    assert heuristic["calls"] == [] and heuristic["state_shifts"] == {}
    assert heuristic["verdict"]["reading"] == "incomplete_no_reading"


def test_failed_request_is_kept_and_blocks_the_reading():
    class Fails(KeywordBackend):
        calls = 0

        def predict(self, state, candidates, request):
            Fails.calls += 1
            if Fails.calls == 2:
                raise RuntimeError("fixture failure")
            return super().predict(state, candidates, request)

    result = run(Fails(), max_states=2)
    assert result["complete"] is False and len(result["errors"]) == 1
    assert "predictions" not in result["errors"][0]
    # The state whose explicit-final answer failed has no shift for that class; nothing is filled in.
    failed_state = result["errors"][0]["state_id"]
    assert "explicit_final" not in result["state_shifts"][failed_state]
    assert result["verdict"]["reading"] == "incomplete_no_reading"


def test_reading_requires_interval_lower_bounds_above_zero():
    reads = {f"s{i}": {"explicit_final": 0.3 + i * 0.01, "paraphrase_final": 0.2 + i * 0.01,
                       "negated_final": -0.05, "open_to_more": -0.1, "embedded_instruction": 0.02} for i in range(8)}
    good = verdict(summarize(reads), is_model=True, complete=True)
    assert good["reading"] == "wording_sensitivity_demonstrated_on_dev_states" and good["counts_as_model_evidence"]
    assert good["warnings"]["embedded_instruction_moves_estimate"] is False
    noisy = {f"s{i}": {"explicit_final": 0.3, "paraphrase_final": 0.4 if i % 2 else -0.5,
                       "negated_final": 0.0, "open_to_more": 0.0, "embedded_instruction": 0.4} for i in range(8)}
    weak = verdict(summarize(noisy), is_model=True, complete=True)
    assert weak["reading"] == "partial_explicit_statement_only"
    assert weak["warnings"]["embedded_instruction_moves_estimate"] is True
    one_state = verdict(summarize({"s0": reads["s0"]}), is_model=True, complete=True)
    assert one_state["reading"] == "wording_sensitivity_not_demonstrated"   # no interval from one state


def test_cli_writes_summary_and_request_journal(tmp_path):
    output = tmp_path / "utterance.json"
    assert main(["--backend", "keyword", "--quick", "--no-records", "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"] and "calls" not in result and result["scope"]["states"] == 6
    journal = (tmp_path / "utterance.json.calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(journal) == 6 * len(VARIANTS)
    assert main(["--backend", "heuristic", "--quick", "--output", str(tmp_path / "unsupported.json")]) == 1
