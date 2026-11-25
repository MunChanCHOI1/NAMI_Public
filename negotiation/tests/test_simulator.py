from dataclasses import replace

from negotiation_agent.simulator import (
    aspiration,
    initial_state,
    make_supplier,
    respond,
    state_after_scripted_rounds,
    supplier_profit,
    would_accept,
)


def test_aspiration_falls_to_reserve_by_the_deadline():
    for style in ("boulware", "linear", "deadline"):
        hidden = make_supplier(7, style, payment_sensitive=True, reveal=False)
        values = [aspiration(r, hidden) for r in range(1, hidden.deadline_round + 1)]
        assert values == sorted(values, reverse=True)
        assert values[0] == hidden.opening_profit and values[-1] == hidden.reserve_profit


def test_respond_is_pure_so_candidates_can_be_checked_independently():
    hidden = make_supplier(11, "linear", payment_sensitive=False, reveal=False)
    state = state_after_scripted_rounds("case", hidden, 1)
    before = state
    base = state.last_offer("supplier").terms
    first = [would_accept(state, hidden, replace(base, unit_price=base.unit_price - cut)) for cut in (0, 200, 900)]
    second = [would_accept(state, hidden, replace(base, unit_price=base.unit_price - cut)) for cut in (0, 200, 900)]
    assert first == second and state == before
    assert first[0] is True and first[-1] is False      # its own offer is accepted, a deep cut is not


def test_counter_offer_keeps_buyer_terms_and_meets_the_asked_profit():
    hidden = make_supplier(3, "boulware", payment_sensitive=True, reveal=False)
    state = initial_state("case", hidden)
    asked = replace(state.last_offer("supplier").terms, unit_price=9_000, payment_days=0)
    response = respond(state, hidden, asked)
    assert response.kind == "counter"
    assert (response.offer.terms.delivery_days, response.offer.terms.payment_days) == (asked.delivery_days, 0)
    assert supplier_profit(response.offer.terms, state.quantity, hidden) >= aspiration(2, hidden)


def test_supplier_walks_away_after_its_deadline():
    hidden = make_supplier(5, "deadline", payment_sensitive=False, reveal=False)
    state = replace(initial_state("case", hidden), current_round=hidden.deadline_round)
    low = replace(state.last_offer("supplier").terms, unit_price=hidden.unit_cost)
    assert respond(state, hidden, low).kind == "walk_away"


def test_statement_twins_differ_only_in_messages():
    neutral = make_supplier(21, "linear", payment_sensitive=True, reveal=False)
    said = make_supplier(21, "linear", payment_sensitive=True, reveal=True)
    a = state_after_scripted_rounds("x", neutral, 2)
    b = state_after_scripted_rounds("x", said, 2)
    assert [o.terms for o in a.history] == [o.terms for o in b.history]
    assert [o.message for o in a.history] != [o.message for o in b.history]
