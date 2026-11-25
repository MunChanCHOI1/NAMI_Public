"""Baseline buyer policies without any acceptance estimate.

They use the same cost conversion, limits, offer ids and approval gate as the agent
(`policy.finalize`), and the same concession schedule where one is needed, so that a
difference in results comes from how the next move is chosen and nothing else.
"""

from __future__ import annotations

from dataclasses import replace

from .contracts import ApprovalLedger, BuyerPrivate, PublicState, Terms
from .economics import alternative_total_cost, price_for_total_cost
from .policy import (
    Decision,
    Option,
    PolicyConfig,
    _valued,
    aspiration_total_cost,
    counter_offer_id,
    finalize,
    standing_facts,
)


def _same_terms_counter(state, private, ledger, standing, ask_total_cost, standing_total, kind):
    base = standing.terms
    delivery = min(base.delivery_days, private.hard_max_delivery_days)
    price = min(price_for_total_cost(ask_total_cost, delivery, base.payment_days, state.quantity, private),
                private.company_bounds.max_unit_price)
    if price <= 0:
        return None
    terms = Terms(price, delivery, base.payment_days)
    authority, total, saving = _valued(terms, state.quantity, private)
    offer_id = counter_offer_id(state, standing)
    if authority == "forbidden" or ledger.is_denied(offer_id, terms):
        return None
    if standing_total is not None and total >= standing_total:
        return None
    return Option("counter", offer_id, terms, kind, authority, total, saving, None, float(saving),
                  approved=ledger.is_approved(offer_id, terms))


def _threshold_policy(state: PublicState, private: BuyerPrivate, ledger: ApprovalLedger, config: PolicyConfig,
                      ask_total_cost: float, kind: str) -> Decision:
    """Accept an offer that meets our current ask (or any gain at the last chance); otherwise ask."""
    standing, facts, authority, total, saving = standing_facts(state, private, config)
    rounds_left = state.max_rounds - state.current_round
    acceptable = (authority != "forbidden" and saving is not None and saving > 0
                  and not ledger.is_denied(standing.offer_id, standing.terms))
    accept = Option("accept", standing.offer_id, standing.terms, "standing_offer", authority, total, saving, None,
                    float(saving), approved=ledger.is_approved(standing.offer_id, standing.terms)) if acceptable else None
    counter = (_same_terms_counter(state, private, ledger, standing, ask_total_cost, total, kind)
               if state.current_round <= state.max_rounds else None)
    end = Option("end", None, None, "take_alternative", None, facts["alternative_total_cost"], 0, None, 0.0)
    options = [o for o in (accept, counter, end) if o is not None]
    facts = {**facts, "ask_total_cost": round(ask_total_cost)}
    if accept and (total <= ask_total_cost or rounds_left <= 0 or counter is None):
        return finalize(state, private, accept, options, facts, ["offer_meets_current_ask_or_last_chance"])
    if counter:
        return finalize(state, private, counter, options, facts, ["ask_by_schedule"])
    return finalize(state, private, end, options, facts, ["nothing_beats_the_alternative"])


def fixed_concession(state, private, backend, ledger, config=PolicyConfig(), prediction_cache=None) -> Decision:
    """Concede a fixed share per round: ask for this round's scheduled total cost on the supplier's terms."""
    # The baseline keeps the straight-line schedule whatever the agent's aspiration shape is.
    ask = aspiration_total_cost(state.current_round, state.max_rounds, state.quantity, private,
                                replace(config, aspiration_exponent=1.0))
    return _threshold_policy(state, private, ledger, config, ask, "fixed_concession")


def midpoint(state, private, backend, ledger, config=PolicyConfig(), prediction_cache=None) -> Decision:
    """Split the difference between our previous ask and the supplier's current offer."""
    ours = state.last_offer("buyer")
    previous = _valued(ours.terms, state.quantity, private)[1] if ours else None
    previous = float(previous) if previous is not None else float(private.target_total_cost)
    standing = state.last_offer("supplier")
    standing_total = _valued(standing.terms, state.quantity, private)[1] if standing else None
    reference = float(standing_total) if standing_total is not None else float(alternative_total_cost(state.quantity, private))
    reservation = alternative_total_cost(state.quantity, private) - config.reserve_margin
    ask = max(previous, min((previous + reference) / 2, reservation))
    return _threshold_policy(state, private, ledger, config, ask, "midpoint")
