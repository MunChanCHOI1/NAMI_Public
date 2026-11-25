"""Cases of the utterance experiment: one seed range of its own, cut into fixed blocks.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only.

Seeds 6000-6899 with the `utt-` case prefix belong to this experiment (section 7.2 of the
design document; the registration line in negotiation/AGENTS.md waits for the user's approval).
The range is outside every split and outside the dashboard scenario range. Cases are drawn the
way the existing splits draw theirs: ONE sequential random stream from the base seed, so a block
is a slice of the prefix and never a reseeded range.

The factory accepts two base seeds only: 6000 (this experiment) and 1100 (at most 30 cases, for
the test that checks the draw order against the development split). Any other number is refused,
because a bare integer argument would otherwise reproduce another split's suppliers.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

from negotiation_agent.contracts import SupplierHidden
from negotiation_agent.simulator import make_supplier

SEED_BASE = 6000
PREFIX = "utt"
MAX_CASES = 900
ALLOWED_BASES: dict[int, int | None] = {6000: None, 1100: 30}   # base seed -> case limit (None: MAX_CASES)
BLOCKS: dict[str, tuple[int, int]] = {"smoke": (0, 10), "fit": (10, 40), "bench": (40, 100), "test": (100, 700)}
RESERVE = {"start": 700, "stop": 900, "case_ids": "utt-6700..utt-6899",
           "rule": "reserved for a later protocol version; never touched by this one"}
REHEARSAL_BLOCKS = ("smoke",)   # the only block whose case count may be reduced
BLOCK_USE = {
    "smoke": "path checks and rehearsals only; never enters a comparison",
    "fit": "reading bench for fitting the thresholds; savings are never computed on it",
    "bench": "reading bench for the registered reading rule; run only after the protocol is bound",
    "test": "negotiation outcome comparison; run only after the protocol is bound",
}


@dataclass(frozen=True)
class UttCase:
    case_id: str
    seed: int
    hidden: SupplierHidden      # evaluator and simulator only


def _cases(base: int, prefix: str, count: int) -> list[UttCase]:
    """The first `count` cases of the stream that starts at `base`.

    Draw order per case, from one random.Random(base): style, payment sensitivity, reveal,
    feasibility; then make_supplier(base + i, ...). Identical to the existing split factory.
    """
    if isinstance(base, bool) or base not in ALLOWED_BASES:
        raise ValueError(f"base seed {base!r} is not allowed; this factory serves {sorted(ALLOWED_BASES)} only")
    limit = ALLOWED_BASES[base] or MAX_CASES
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= limit:
        raise ValueError(f"case count for base seed {base} must be in 1..{limit}")
    rng = random.Random(base)
    cases = []
    for index in range(count):
        style = rng.choice(("boulware", "linear", "deadline"))
        sensitive, reveal, feasible = rng.random() < 0.5, rng.random() < 0.5, rng.random() >= 1 / 6
        hidden = make_supplier(base + index, style, sensitive, reveal, feasible)
        cases.append(UttCase(f"{prefix}-{base + index}", base + index, hidden))
    return cases


def block_cases(name: str, limit: int | None = None) -> list[UttCase]:
    """The cases of one block. `limit` (the first N cases) is a rehearsal option of the smoke block only."""
    if name not in BLOCKS:
        raise ValueError(f"unknown block {name!r}; blocks are {sorted(BLOCKS)}")
    start, stop = BLOCKS[name]
    cases = _cases(SEED_BASE, PREFIX, stop)[start:stop]
    if limit is None:
        return cases
    if name not in REHEARSAL_BLOCKS:
        raise ValueError(f"block {name!r} is always run at its registered size; only {REHEARSAL_BLOCKS} "
                         "accept a reduced case count")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= len(cases):
        raise ValueError(f"limit must be in 1..{len(cases)}")
    return cases[:limit]


def block_content_sha256(cases: list[UttCase]) -> str:
    """Identity of a block's content: case ids and hidden supplier parameters, canonical JSON."""
    payload = {"identity_schema_version": "1.0",
               "cases": [{"case_id": c.case_id, "supplier_hidden": asdict(c.hidden)} for c in cases]}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
