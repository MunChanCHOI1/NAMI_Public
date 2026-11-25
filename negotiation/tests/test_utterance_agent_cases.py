"""CPU tests of the utterance experiment's case blocks.

NOT EXECUTED ON A GPU as of 2026-11-18. These tests look at seeds, slices and source text only;
no saving and no reading is computed on any block.
"""

import ast
import re
from pathlib import Path

import pytest

from negotiation_agent import provenance
from negotiation_agent.evaluation import make_cases

from utterance_agent import cases as C

NEGOTIATION = Path(__file__).resolve().parents[1]
PACKAGE = NEGOTIATION / "src" / "utterance_agent"
SPLIT_LITERAL = "ev" + "al"          # spelled in two parts so this file can be scanned as well
FORBIDDEN_NAMES = {"HISTORY_DIR", "reserve_heldout", "record_exploratory", "freeze", "evaluation_identity",
                   "evaluation_case_identities", "SPLIT_SEED_BASE"}
OTHER_RANGES = ((1000, 1099), (1100, 1999), (2000, 2899), (3000, 3899))   # scenarios and the three splits


def test_blocks_are_disjoint_slices_outside_every_split_and_scenario_range():
    assert (C.SEED_BASE, C.PREFIX) == (6000, "utt")
    assert C.BLOCKS == {"smoke": (0, 10), "fit": (10, 40), "bench": (40, 100), "test": (100, 700)}
    assert (C.RESERVE["start"], C.RESERVE["stop"], C.RESERVE["case_ids"]) == (700, 900, "utt-6700..utt-6899")
    slices = sorted(C.BLOCKS.values()) + [(C.RESERVE["start"], C.RESERVE["stop"])]
    assert slices[0][0] == 0 and slices[-1][1] == 900
    assert all(first[1] == second[0] for first, second in zip(slices, slices[1:]))   # contiguous, no overlap
    assert C.BLOCKS["test"][1] - C.BLOCKS["test"][0] == 600
    assert C.RESERVE["stop"] - C.RESERVE["start"] == 200
    seen: set[int] = set()
    for name, (start, stop) in C.BLOCKS.items():
        cases = C.block_cases(name)
        assert len(cases) == stop - start
        seeds = [case.seed for case in cases]
        assert seeds == list(range(6000 + start, 6000 + stop))
        assert [case.case_id for case in cases] == [f"utt-{seed}" for seed in seeds]
        assert not seen & set(seeds)
        seen |= set(seeds)
        assert all(6000 <= seed <= 6899 for seed in seeds)
        assert not any(low <= seed <= high for seed in seeds for low, high in OTHER_RANGES)
    assert (C.block_cases("fit")[0].case_id, C.block_cases("bench")[0].case_id) == ("utt-6010", "utt-6040")
    assert (C.block_cases("test")[0].case_id, C.block_cases("test")[-1].case_id) == ("utt-6100", "utt-6699")
    # a block is a slice of ONE stream: the smoke block is the prefix of every longer draw
    assert C._cases(6000, "utt", 40)[:10] == C.block_cases("smoke")
    assert C.block_content_sha256(C.block_cases("smoke")) == C.block_content_sha256(C._cases(6000, "utt", 10))
    assert C.block_content_sha256(C.block_cases("smoke")) != C.block_content_sha256(C.block_cases("fit"))


def test_case_draw_order_matches_make_cases():
    ours, theirs = C._cases(1100, "dev", 30), make_cases("dev", 30)
    assert [case.hidden for case in ours] == [case.hidden for case in theirs]
    assert [(case.case_id, case.seed) for case in ours] == [(case.case_id, case.seed) for case in theirs]


def test_case_factory_refuses_other_bases_and_larger_counts():
    for base in (2000, 3000, 1000, 1101, 5999, 6001, 0, True):
        with pytest.raises(ValueError):
            C._cases(base, "x", 1)
    with pytest.raises(ValueError):
        C._cases(1100, "dev", 31)
    assert len(C._cases(1100, "dev", 30)) == 30
    for count in (0, -1, 901, 1.5, True):
        with pytest.raises(ValueError):
            C._cases(6000, "utt", count)
    with pytest.raises(ValueError):
        C.block_cases("reserve")


def test_only_the_smoke_block_accepts_a_limit():
    assert C.REHEARSAL_BLOCKS == ("smoke",)
    assert [case.case_id for case in C.block_cases("smoke", 3)] == ["utt-6000", "utt-6001", "utt-6002"]
    for name in ("fit", "bench", "test"):
        with pytest.raises(ValueError):
            C.block_cases(name, 5)
    for limit in (0, 11, -1, True):
        with pytest.raises(ValueError):
            C.block_cases("smoke", limit)


def _sources() -> list[Path]:
    launchers = [NEGOTIATION / "utterance_experiment.py", NEGOTIATION / "utterance_demo.py"]
    return sorted(PACKAGE.glob("*.py")) + [path for path in launchers if path.is_file()]


def test_package_never_names_the_held_out_split_or_its_history():
    sources = _sources()
    assert {path.name for path in sources} >= {"__init__.py", "questions.py", "interpreter.py", "wording.py",
                                               "cases.py", "experiment.py", "bench.py", "utterance_experiment.py"}
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(rf"\b{SPLIT_LITERAL}\b", node.value), (path.name, node.value[:60])
                assert "holdout" not in node.value and "evidence/" not in node.value, (path.name, node.value[:60])
            if isinstance(node, ast.ImportFrom):
                assert not {alias.name for alias in node.names} & FORBIDDEN_NAMES, path.name
            if isinstance(node, ast.Name):
                assert node.id not in FORBIDDEN_NAMES, path.name
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN_NAMES, path.name
    for base in (3000, 2000):                         # the other splits' base seeds are refused outright
        with pytest.raises(ValueError):
            C._cases(base, "x", 1)


def test_new_files_are_outside_the_hashed_inputs():
    hashed = provenance.input_hashes()
    assert hashed                                     # the existing inputs are still found
    assert not any("utterance_agent" in name or name.endswith("utterance_experiment.py")
                   or name.endswith("utterance_demo.py") for name in hashed)
    # the launchers do not match the hashed launcher pattern either
    assert all(not path.name.startswith("run_") for path in _sources())
    assert "negotiation/src/negotiation_agent/utterance_probe.py" in hashed   # the older probe stays hashed
