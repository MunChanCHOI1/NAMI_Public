import ast
import json
from dataclasses import replace
from pathlib import Path

from negotiation_agent.contracts import TermBounds, Terms
from negotiation_agent.probe import build_ladder, build_states, run
from negotiation_agent.backends import HeuristicBackend
from negotiation_agent.requests import VARIANTS, build_acceptance_request
from negotiation_agent.simulator import default_buyer_private

SRC = Path(__file__).resolve().parents[1] / "src" / "negotiation_agent"


def _imported_names(module: str) -> set[str]:
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_request_builder_and_backends_cannot_import_private_or_hidden_types():
    for module in ("requests.py", "backends.py"):
        names = _imported_names(module)
        assert not names & {"BuyerPrivate", "SupplierHidden", "economics", "simulator", "metrics", "probe"}, module


def test_store_and_estimate_sources_stay_off_private_hidden_and_evaluator_code():
    """Both wrap every model call and one writes the store, so they get the same import scan and more."""
    for module in ("estimate_sources.py", "prediction_store.py"):
        names = _imported_names(module)
        assert not names & {"BuyerPrivate", "SupplierHidden", "economics", "simulator", "metrics", "probe",
                            "evaluation", "session", "tune"}, module


def test_request_contains_no_private_or_hidden_value():
    probe = build_states()[3]
    request = build_acceptance_request(probe.state, build_ladder(probe.state, "mixed"))
    text = json.dumps(request, ensure_ascii=False)
    hidden = probe.hidden
    for secret in (hidden.unit_cost, hidden.opening_profit, hidden.reserve_profit, hidden.expedite_cost_per_day):
        assert str(secret) not in text and f"{secret:,}" not in text
    assert hidden.style not in text
    # Our private numbers use sentinel values that cannot collide with a quoted price.
    private = replace(default_buyer_private(), target_total_cost=7_777_777, alternative=Terms(13_131, 13, 77),
                      auto_bounds=TermBounds(12_121, 29, 31, 8_888_888))
    for secret in (private.target_total_cost, private.alternative.unit_price, private.auto_bounds.max_unit_price,
                   private.auto_bounds.max_total_cost):
        assert str(secret) not in text and f"{secret:,}" not in text
    assert set(request) == {"state", "questions"}
    assert set(request["state"]) == {"거래", "진행", "제안_이력"}


def test_each_candidate_is_an_independent_noul_question():
    probe = build_states()[0]
    candidates = build_ladder(probe.state, "price")
    for variant in VARIANTS:
        request = build_acceptance_request(probe.state, candidates, variant)
        assert list(request["questions"]) == [c.candidate_id for c in candidates]
        for candidate, question in zip(candidates, request["questions"].values()):
            assert question["type"] == "noul" and set(question["criteria"]) == {"true", "false"}
            assert f"단가 {candidate.terms.unit_price}원" in question["instructions"]
            others = {f"{c.terms.unit_price}원" for c in candidates} - {f"{candidate.terms.unit_price}원"}
            assert not any(other in question["instructions"] for other in others)
    base = build_acceptance_request(probe.state, candidates)["questions"]["P1"]
    inverted = build_acceptance_request(probe.state, candidates, "inverted")["questions"]["P1"]
    assert base["criteria"]["true"] == inverted["criteria"]["false"]


def test_result_keeps_evaluator_facts_out_of_the_public_records():
    result = run(HeuristicBackend(), ladders=("price",), max_states=3, variant_states=0)
    assert all(not {"truth", "saving", "authority"} & set(row) for row in result["records"])
    assert all({"truth", "saving", "authority"} <= set(row) for row in result["evaluator_only"])


def test_no_state_asks_the_same_terms_twice_and_twins_get_identical_candidates():
    states = {p.state_id: p for p in build_states()}
    assert len(states) == 24
    for probe in states.values():
        terms = [c.terms for ladder in ("price", "payment", "delivery", "mixed")
                 for c in build_ladder(probe.state, ladder)]
        assert len(terms) == len(set(terms)) == 19
        if probe.twin_of:
            twin = states[probe.twin_of]
            assert build_ladder(probe.state, "mixed") == build_ladder(twin.state, "mixed")
            assert probe.hidden.unit_cost == twin.hidden.unit_cost
