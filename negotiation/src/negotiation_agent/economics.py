"""Deterministic cost conversion, authority checks and expected-value comparison.

All numbers here are computed by code from BuyerPrivate. No model output is used
except the acceptance probability passed into `counter_expected_saving`.
"""

from __future__ import annotations

from .contracts import AuthorityLevel, BuyerPrivate, TermBounds, Terms


class HardConstraintViolation(ValueError):
    """The terms break an absolute requirement and have no cost equivalent."""


def buyer_total_cost(terms: Terms, quantity: int, private: BuyerPrivate) -> int:
    """Converted total cost of meeting the same demand, valued at the delivery date.

    purchase amount
      - financing benefit of paying later (amount * annual rate * payment_days / 365)
      + delay cost for each day beyond the preferred delivery date.
    A delivery beyond the hard limit is not priced; it raises instead.
    """
    if terms.delivery_days > private.hard_max_delivery_days:
        raise HardConstraintViolation(
            f"delivery in {terms.delivery_days} days exceeds the hard limit of {private.hard_max_delivery_days}"
        )
    amount = terms.unit_price * quantity
    financing_benefit = amount * private.capital_cost_annual * terms.payment_days / 365
    delay = private.delay_cost_per_day * max(0, terms.delivery_days - private.preferred_delivery_days)
    return round(amount - financing_benefit + delay)


def alternative_total_cost(quantity: int, private: BuyerPrivate) -> int:
    return buyer_total_cost(private.alternative, quantity, private)


def saving_vs_alternative(terms: Terms, quantity: int, private: BuyerPrivate) -> int:
    """Positive when these terms beat the alternative quote."""
    return alternative_total_cost(quantity, private) - buyer_total_cost(terms, quantity, private)


def _within(terms: Terms, total_cost: int, bounds: TermBounds) -> bool:
    return (
        terms.unit_price <= bounds.max_unit_price
        and terms.delivery_days <= bounds.max_delivery_days
        and terms.payment_days >= bounds.min_payment_days
        and total_cost <= bounds.max_total_cost
    )


def classify_authority(terms: Terms, quantity: int, private: BuyerPrivate) -> AuthorityLevel:
    """auto: executable alone. needs_approval: inside company policy only. forbidden: never executed."""
    if terms.delivery_days > private.hard_max_delivery_days:
        return "forbidden"
    total = buyer_total_cost(terms, quantity, private)
    if _within(terms, total, private.auto_bounds):
        return "auto"
    if _within(terms, total, private.company_bounds):
        return "needs_approval"
    return "forbidden"


def counter_expected_saving(
    acceptance_probability: float,
    saving_if_accepted: float,
    value_if_rejected: float,
    extra_cost: float = 0.0,
) -> float:
    """p * value on acceptance + (1 - p) * value after rejection - extra cost.

    `value_if_rejected` is the caller's estimate of what a rejection still leaves (see
    policy.decide). It is an approximation, not a solved multi-round game.
    """
    if not 0.0 <= acceptance_probability <= 1.0:
        raise ValueError("acceptance_probability must be within [0, 1]")
    p = acceptance_probability
    return p * saving_if_accepted + (1.0 - p) * value_if_rejected - extra_cost


def auto_limit_exceedances(terms: Terms, quantity: int, private: BuyerPrivate) -> list[dict]:
    """Which of the negotiator's own limits these terms exceed. Shown on the approval card."""
    bounds, out = private.auto_bounds, []
    if terms.unit_price > bounds.max_unit_price:
        out.append({"field": "unit_price", "value": terms.unit_price, "auto_limit": bounds.max_unit_price})
    if terms.delivery_days > bounds.max_delivery_days:
        out.append({"field": "delivery_days", "value": terms.delivery_days, "auto_limit": bounds.max_delivery_days})
    if terms.payment_days < bounds.min_payment_days:
        out.append({"field": "payment_days", "value": terms.payment_days, "auto_limit": bounds.min_payment_days})
    if terms.delivery_days <= private.hard_max_delivery_days:
        total = buyer_total_cost(terms, quantity, private)
        if total > bounds.max_total_cost:
            out.append({"field": "total_cost", "value": total, "auto_limit": bounds.max_total_cost})
    return out


def price_for_total_cost(total_cost: float, delivery_days: int, payment_days: int, quantity: int,
                         private: BuyerPrivate, price_step: int = 10) -> int:
    """Highest unit price, on the price grid, whose converted total cost does not exceed `total_cost`."""
    delay = private.delay_cost_per_day * max(0, delivery_days - private.preferred_delivery_days)
    factor = quantity * (1 - private.capital_cost_annual * payment_days / 365)
    return int((total_cost - delay) / factor // price_step * price_step)
