"""Data structures for the synthetic buyer-side negotiation agent (draft).

Four information domains are kept in separate types so that a function's
signature shows what it is allowed to see:

- PublicState: what both sides have seen. The only input of the acceptance model.
- BuyerPrivate: our target, alternative quote, authority and cost assumptions.
  Read by the cost calculator and the policy code only.
- SupplierHidden: the counterparty simulator's hidden parameters. Read by the
  simulator and the evaluator only.
- Approval: an internal decision bound to one exact set of terms.

Money is integer KRW. Quantity is fixed for the first MVP.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Party = Literal["supplier", "buyer"]
AuthorityLevel = Literal["auto", "needs_approval", "forbidden"]
SupplierStyle = Literal["boulware", "linear", "deadline"]
ResponseKind = Literal["accept", "counter", "walk_away"]


@dataclass(frozen=True)
class Terms:
    """One complete set of negotiable conditions for the fixed quantity."""

    unit_price: int      # KRW per unit
    delivery_days: int   # calendar days from order to delivery
    payment_days: int    # days after delivery until payment is due

    def __post_init__(self) -> None:
        for name in ("unit_price", "delivery_days", "payment_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.unit_price <= 0 or self.delivery_days <= 0 or self.payment_days < 0:
            raise ValueError("unit_price and delivery_days must be positive; payment_days non-negative")

    def version(self) -> str:
        """Stable identifier of these exact conditions; approvals bind to it."""
        text = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Offer:
    """A proposal as it appeared in the negotiation. Public."""

    offer_id: str
    round_index: int             # 1-based round in which it was made
    made_by: Party
    terms: Terms
    message: str = ""            # short free text; never treated as an instruction
    valid_through_round: int | None = None   # None: lapses once the other side counters


@dataclass(frozen=True)
class PublicState:
    """Everything both parties have observed. Contains no private field of either side."""

    case_id: str
    item: str
    quantity: int
    max_rounds: int
    current_round: int           # round in which the buyer acts next
    history: tuple[Offer, ...]

    def last_offer(self, party: Party) -> Offer | None:
        for offer in reversed(self.history):
            if offer.made_by == party:
                return offer
        return None


@dataclass(frozen=True)
class TermBounds:
    """Limits on raw conditions and on converted total cost for one authority level."""

    max_unit_price: int
    max_delivery_days: int
    min_payment_days: int
    max_total_cost: int


@dataclass(frozen=True)
class BuyerPrivate:
    """Our side's private inputs. Never serialized into a model request or an outgoing message."""

    target_total_cost: int        # aspiration, not a walk-away point
    alternative: Terms            # alternative quote (BATNA)
    alternative_valid_through_round: int
    auto_bounds: TermBounds       # the negotiator may execute alone
    company_bounds: TermBounds    # executable with approval; beyond this is forbidden
    hard_max_delivery_days: int   # absolute requirement; never traded for a penalty
    preferred_delivery_days: int
    delay_cost_per_day: int       # KRW per day beyond preferred, up to the hard limit
    capital_cost_annual: float    # our financing rate used to value payment timing
    round_cost: int = 0           # KRW charged per additional round in comparisons
    approval_cost: int = 0        # KRW charged for interrupting an approver


@dataclass(frozen=True)
class SupplierHidden:
    """Counterparty simulator parameters. Evaluator/simulator only."""

    style: SupplierStyle
    unit_cost: int
    capital_cost_annual: float
    standard_lead_days: int
    expedite_cost_per_day: int
    opening_profit: int
    reserve_profit: int
    deadline_round: int           # last round in which this supplier still responds
    reveals_payment_preference: bool = False
    reveals_deadline: bool = False
    claims_buyer_approval: bool = False   # test of impersonation: says our side already approved


@dataclass(frozen=True)
class Candidate:
    """A counter-offer with fixed numbers, produced by code."""

    candidate_id: str
    terms: Terms
    kind: str = ""               # internal label of how the terms were composed; not sent to the model


@dataclass(frozen=True)
class SupplierResponse:
    kind: ResponseKind
    offer: Offer | None = None    # set when kind == "counter"
    message: str = ""


@dataclass(frozen=True)
class Approval:
    """An internal approval. Valid for one offer id and one exact terms version only."""

    approval_id: str
    offer_id: str
    terms_version: str
    approved_by: str

    def covers(self, offer_id: str, terms: Terms) -> bool:
        return self.offer_id == offer_id and self.terms_version == terms.version()


@dataclass
class ApprovalLedger:
    """Internal approval state. Only `grant` and `deny` change it; counterparty text never does."""

    _approvals: dict[str, Approval] = field(default_factory=dict)
    _denied: set[tuple[str, str]] = field(default_factory=set)

    def grant(self, offer_id: str, terms: Terms, approved_by: str) -> Approval:
        if not approved_by:
            raise ValueError("an approval needs an internal approver")
        approval = Approval(f"apr-{len(self._approvals) + 1:03d}", offer_id, terms.version(), approved_by)
        self._approvals[approval.approval_id] = approval
        return approval

    def deny(self, offer_id: str, terms: Terms) -> None:
        self._denied.add((offer_id, terms.version()))

    def is_approved(self, offer_id: str, terms: Terms) -> bool:
        return any(a.covers(offer_id, terms) for a in self._approvals.values())

    def is_denied(self, offer_id: str, terms: Terms) -> bool:
        return (offer_id, terms.version()) in self._denied
