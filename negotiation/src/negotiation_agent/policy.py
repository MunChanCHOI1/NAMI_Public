"""Buyer policy for one round: code computes and decides, the model only estimates acceptance.

Order of work, matching the confirmed rules in docs/PLAN.md:
1. Code values the supplier's standing offer (converted total cost, saving, authority).
2. Code builds counter-offer candidates with fixed numbers. Every candidate at the
   current aspiration is worth the same to us; they differ in how the terms are composed.
3. One model request asks, per candidate, whether the supplier would accept it unchanged.
4. Code compares accepting now, each counter-offer and ending, and picks one.
5. Terms beyond the negotiator's own limits become an approval request instead of an action.

The model never produces a number that is executed, and it never sees BuyerPrivate.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .contracts import ApprovalLedger, BuyerPrivate, Candidate, Offer, PublicState, Terms
from .economics import (
    HardConstraintViolation,
    alternative_total_cost,
    auto_limit_exceedances,
    buyer_total_cost,
    classify_authority,
    counter_expected_saving,
    price_for_total_cost,
)
from .requests import FINAL_OFFER_QID, build_decision_request, build_final_only_request
from .provenance import json_safe, request_sha256, reset_response, response_snapshot, validate_predictions

SOFTER_MARGIN = 20_000   # KRW; equal-value compositions differ only by price rounding

Action = Literal["accept", "counter", "request_approval", "end", "human_review"]


@dataclass(frozen=True)
class PolicyConfig:
    # Assumed supplier concession (KRW of our converted total cost) before one has been observed.
    prior_concession: int = 150_000
    # Assumed chance that the supplier's standing offer is final, by round, when the backend gives
    # no estimate. Reflects an assumption about hidden deadlines; the last public round is always final.
    final_prior_by_round: tuple[float, ...] = (0.0, 0.15, 0.45, 0.5)
    reserve_margin: int = 30_000   # smallest saving vs the alternative we still ask for in the last round
    max_candidates: int = 5
    # Added 2026-11-19 (dev split only). Defaults keep the earlier rule; see IMPROVED_POLICY_FIELDS below.
    # Shape of the aspiration path from target (round 1) to reservation (last round): progress ** exponent.
    # 1.0 is the original straight line; 2.0 holds firmer early and concedes faster late.
    aspiration_exponent: float = 1.0
    # If the standing offer is final, how likely the supplier still accepts a counter-offer that keeps its
    # delivery and payment terms but costs it more (a lower price), relative to the backend's estimate.
    # 0.0: a final offer is taken at its word for same-terms counters. None keeps the original treatment
    # (the acceptance estimate is used as if independent of finality) for every counter-offer.
    final_accept_scale: float | None = None
    # The same, for a counter-offer that changes delivery or payment terms. Such terms can be worth more
    # to the supplier than its own final offer, so the estimate is only halved; None keeps it unscaled.
    final_accept_scale_recomposed: float | None = None
    # Cap the expected next supplier offer at the saving we would ask for next round: a supplier is not
    # expected to offer more than our own ask, however large its last concession was. False: no cap.
    cap_next_offer_at_ask: bool = False


# The class defaults stay on the decision rule before 2026-11-19: every golden request hash, stored
# real-model answer and frozen protocol (tune-2026-11-18.v1, utterance experiments) was built on it.
LEGACY_POLICY_FIELDS = {"aspiration_exponent": 1.0, "final_accept_scale": None, "final_accept_scale_recomposed": None,
                        "cap_next_offer_at_ask": False}
LEGACY_POLICY = PolicyConfig(**LEGACY_POLICY_FIELDS)
# Selected 2026-11-19 on dev-1100..1299 and confirmed once on dev-1300..1599 (CPU stand-ins only; never
# the eval split, never a real model). Opt in with PolicyConfig(**IMPROVED_POLICY_FIELDS) or a config file.
IMPROVED_POLICY_FIELDS = {"aspiration_exponent": 2.0, "final_accept_scale": 0.0, "final_accept_scale_recomposed": 0.5,
                          "cap_next_offer_at_ask": True}
IMPROVED_POLICY = PolicyConfig(**IMPROVED_POLICY_FIELDS)


@dataclass(frozen=True)
class Option:
    """One thing we could do this round, with the numbers behind it. Internal."""

    action: Literal["accept", "counter", "end"]
    offer_id: str | None
    terms: Terms | None
    kind: str
    authority: str | None
    total_cost: int | None
    saving: int | None
    p_accept: float | None          # model estimate; None for accept and end
    value: float                    # certain saving, or expected saving after costs
    value_if_rejected: float | None = None   # what a rejection is assumed to leave, before the final-offer risk
    approved: bool = False


def observed_concession(state: PublicState, private: BuyerPrivate) -> int | None:
    """How much the supplier's latest offer improved on its previous one, in our converted total cost."""
    totals = [_valued(o.terms, state.quantity, private)[1] for o in state.history if o.made_by == "supplier"]
    if len(totals) < 2 or totals[-1] is None or totals[-2] is None:
        return None
    return max(0, totals[-2] - totals[-1])


@dataclass(frozen=True)
class ApprovalRequest:
    """The approval card. Bound to one offer id and one exact set of terms."""

    request_id: str
    kind: Literal["accept_offer", "send_counter"]
    offer_id: str
    round_index: int
    terms: Terms
    terms_version: str
    exceeded_auto_limits: tuple[dict, ...]
    total_cost: int
    saving_vs_alternative: int
    gap_to_target: int
    p_accept: float | None


@dataclass(frozen=True)
class Decision:
    round_index: int
    action: Action
    chosen: Option | None
    options: tuple[Option, ...]
    facts: dict[str, Any]
    reasons: tuple[str, ...]
    approval_request: ApprovalRequest | None = None
    model: dict[str, Any] = field(default_factory=dict)


def aspiration_total_cost(round_index: int, max_rounds: int, quantity: int, private: BuyerPrivate,
                          config: PolicyConfig) -> float:
    """The total cost we ask for in a round: from our target toward just below the alternative."""
    reservation = alternative_total_cost(quantity, private) - config.reserve_margin
    progress = 1.0 if max_rounds <= 1 else min(1.0, max(0.0, (round_index - 1) / (max_rounds - 1)))
    progress = progress ** config.aspiration_exponent
    return private.target_total_cost + (reservation - private.target_total_cost) * progress


def _valued(terms: Terms, quantity: int, private: BuyerPrivate) -> tuple[str, int | None, int | None]:
    try:
        total = buyer_total_cost(terms, quantity, private)
    except HardConstraintViolation:
        return "forbidden", None, None
    return classify_authority(terms, quantity, private), total, alternative_total_cost(quantity, private) - total


def generate_candidates(state: PublicState, private: BuyerPrivate, config: PolicyConfig) -> list[Candidate]:
    """Counter-offers with fixed numbers, all executable under company policy.

    Four compositions are priced to the same converted total cost (this round's aspiration),
    so choosing among them costs us nothing; one softer offer is priced at the next round's.
    """
    standing = state.last_offer("supplier")
    assert standing is not None
    base, quantity = standing.terms, state.quantity
    _, standing_total, _ = _valued(base, quantity, private)
    now = aspiration_total_cost(state.current_round, state.max_rounds, quantity, private, config)
    softer = aspiration_total_cost(state.current_round + 1, state.max_rounds, quantity, private, config)
    delivery = min(base.delivery_days, private.hard_max_delivery_days)
    faster = min(delivery, private.preferred_delivery_days)
    on_delivery = min(base.payment_days, private.company_bounds.min_payment_days)
    usual = max(base.payment_days, private.auto_bounds.min_payment_days)
    specs = [
        ("same_terms", now, delivery, base.payment_days),
        ("pay_sooner", now, delivery, on_delivery),
        ("deliver_sooner", now, faster, base.payment_days),
        ("pay_and_deliver_sooner", now, faster, on_delivery),
        ("usual_payment", now, delivery, usual),
        ("softer_same_terms", softer, delivery, base.payment_days),
    ]
    out: list[Candidate] = []
    seen: set[Terms] = set()
    for kind, total_cost, days, payment in specs:
        price = min(price_for_total_cost(total_cost, days, payment, quantity, private),
                    private.company_bounds.max_unit_price)
        if price <= 0:
            continue
        terms = Terms(price, days, payment)
        authority, total, _ = _valued(terms, quantity, private)
        if terms in seen or authority == "forbidden":
            continue
        if standing_total is not None and total >= standing_total:
            continue   # not better for us than what is already offered
        seen.add(terms)
        out.append(Candidate("", terms, kind))
    if len(out) > config.max_candidates:
        softer_offers = [c for c in out if c.kind == "softer_same_terms"]
        others = [c for c in out if c.kind != "softer_same_terms"]
        out = others[: config.max_candidates - len(softer_offers)] + softer_offers
    out = [Candidate(f"C{i + 1}", c.terms, c.kind) for i, c in enumerate(out)]
    return out


def own_limit_candidate(state: PublicState, private: BuyerPrivate, config: PolicyConfig,
                        existing: list[Terms]) -> Candidate | None:
    """This round's aspiration clamped to the negotiator's own (auto) limits, if that still beats the standing offer."""
    standing = state.last_offer("supplier")
    assert standing is not None
    quantity, bounds = state.quantity, private.auto_bounds
    _, standing_total, _ = _valued(standing.terms, quantity, private)
    now = aspiration_total_cost(state.current_round, state.max_rounds, quantity, private, config)
    days = min(standing.terms.delivery_days, bounds.max_delivery_days, private.hard_max_delivery_days)
    payment = max(standing.terms.payment_days, bounds.min_payment_days)
    price = min(price_for_total_cost(now, days, payment, quantity, private), bounds.max_unit_price)
    if price <= 0:
        return None
    terms = Terms(price, days, payment)
    authority, total, _ = _valued(terms, quantity, private)
    if authority != "auto" or terms in existing or (standing_total is not None and total >= standing_total):
        return None
    return Candidate("", terms, "within_own_limit")


def _candidate_set_sha256(candidates: list[Candidate]) -> str:
    """Identity of the candidate ids and their exact terms, for caching answers to a final-only request."""
    rows = [[c.candidate_id, asdict(c.terms)] for c in candidates]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _survives(offer: Offer, state: PublicState) -> bool:
    return offer.valid_through_round is not None and offer.valid_through_round > state.current_round


def decide(state: PublicState, private: BuyerPrivate, backend, ledger: ApprovalLedger,
           config: PolicyConfig = PolicyConfig(), prediction_cache: dict | None = None,
           allow_counter: bool = True) -> Decision:
    """Choose this round's action. Does not change the state or the ledger.

    allow_counter=False is a reviewer's recorded choice for one round (session.resume_after_review):
    no counter-offer is built or estimated, so only accepting the standing offer and ending are compared.
    """
    standing, facts, authority, total, saving = standing_facts(state, private, config)
    quantity = state.quantity
    rounds_left = state.max_rounds - state.current_round      # rounds remaining after a counter-offer now
    can_counter = allow_counter and state.current_round <= state.max_rounds
    options: list[Option] = []
    if authority != "forbidden" and saving is not None and saving > 0 \
            and not ledger.is_denied(standing.offer_id, standing.terms):
        approved = ledger.is_approved(standing.offer_id, standing.terms)
        cost = private.approval_cost if authority == "needs_approval" and not approved else 0
        options.append(Option("accept", standing.offer_id, standing.terms, "standing_offer", authority,
                              total, saving, None, float(saving - cost), approved=approved))

    model: dict[str, Any] = {}
    candidates = generate_candidates(state, private, config) if can_counter else []
    offer_id = counter_offer_id(state, standing)
    kept = [c for c in candidates if not ledger.is_denied(offer_id, c.terms)]
    if len(kept) < len(candidates):
        # An approver refused a counter. Without this, a round in which every composition sits above
        # the negotiator's own price limit leaves nothing to send and the talks end early although a
        # counter inside our own authority is still possible (stress harness: deny-all approver).
        # Only added after a denial, so requests without one are unchanged.
        fallback = own_limit_candidate(state, private, config, [c.terms for c in kept])
        if fallback is not None and not ledger.is_denied(offer_id, fallback.terms):
            kept.append(Candidate(f"C{len(candidates) + 1}", fallback.terms, fallback.kind))
    candidates = kept
    if candidates:
        # "final_offer_only" is opted into by a composition that takes no acceptance value from the
        # model (estimate_sources); every other backend gets the default request, unchanged.
        scope = getattr(backend, "request_scope", "full")
        if scope == "final_offer_only":
            request = build_final_only_request(state)
        elif scope == "full":
            request = build_decision_request(state, candidates)
        else:
            raise ValueError(f"unknown request scope {scope!r}")
        digest = request_sha256(request)
        # A final-only request does not name the candidates, so a cached answer is reused only for the
        # same candidate set (after a denial the set shrinks and the composition answers again).
        cache_key = digest if scope == "full" else f"{digest}:{_candidate_set_sha256(candidates)}"
        model = {"backend": backend.kind, "is_model": backend.is_model, "request_sha256": digest,
                 "questions": len(request["questions"]),
                 "label": "모델 추정" if backend.is_model else "규칙 추정(모델 아님)"}
        if scope != "full":
            model["request_scope"] = scope
        try:
            if prediction_cache is not None and cache_key in prediction_cache:
                cached = prediction_cache[cache_key]
                predicted = cached["predictions"]
                model.update(cached=True, trace_ref=cached["trace_ref"])
            else:
                model.update(trace_id=digest, request=json_safe(request))
                started = time.perf_counter()
                predicted = None
                reset_response(backend)
                try:
                    predicted = backend.predict(state, candidates, request)
                    required = [c.candidate_id for c in candidates]
                    if getattr(backend, "supports_final_offer", False) or FINAL_OFFER_QID in predicted:
                        required.append(FINAL_OFFER_QID)
                    predicted = validate_predictions(predicted, required)
                    model.update(status="ok", predictions=json_safe(predicted))
                except (Exception, KeyboardInterrupt) as error:
                    model["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                    if predicted is not None:
                        model["invalid_predictions"] = json_safe(predicted)
                    raise
                finally:
                    model["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                    model["response"] = response_snapshot(backend)
                if prediction_cache is not None:
                    prediction_cache[cache_key] = {"predictions": predicted, "trace_ref": digest}
            values = {c.candidate_id: float(predicted[c.candidate_id]) for c in candidates}
            estimated_final = predicted.get(FINAL_OFFER_QID)
        except (Exception, KeyboardInterrupt) as error:  # retain failed/interrupted attempts without invented estimates
            model["error"] = f"{type(error).__name__}: {error}"
            return Decision(state.current_round, "human_review", None, tuple(options), facts,
                            ("acceptance_estimate_unavailable",), None, model)

        # Chance that the standing offer is final, i.e. that a rejected counter-offer ends the talks.
        if rounds_left <= 0:
            p_final, source = 1.0, "public_round_limit"
        elif estimated_final is not None:
            p_final, source = float(estimated_final), "backend_estimate"
        else:
            table = config.final_prior_by_round
            p_final, source = table[min(state.current_round, len(table)) - 1], "policy_prior"
        concession = observed_concession(state, private)
        assumed = concession if concession is not None else config.prior_concession
        next_offer = (saving + assumed) if saving is not None else None
        if config.cap_next_offer_at_ask and next_offer is not None:
            ask_next = aspiration_total_cost(min(state.current_round + 1, state.max_rounds), state.max_rounds,
                                             quantity, private, config)
            next_offer = min(next_offer, max(saving, alternative_total_cost(quantity, private) - round(ask_next)))
        facts.update({"p_final_offer": p_final, "p_final_offer_source": source,
                      "supplier_concession_observed": concession, "supplier_concession_assumed": assumed,
                      "expected_next_offer_saving": next_offer})

        scored = []
        for c in candidates:
            c_authority, c_total, c_saving = _valued(c.terms, quantity, private)
            scored.append((c, c_authority, c_total, c_saving, values[c.candidate_id]))
        for c, c_authority, c_total, c_saving, p in scored:
            # What a rejection leaves if the talks go on. The supplier is assumed to keep conceding,
            # so (1) its next offer should improve by the concession seen so far, and (2) a clearly
            # softer offer that would be accepted now can still be sent, and accepted, next round.
            # (2) is the reason not to soften ahead of schedule unless the talks may end.
            softer = max(((q - p) / (1.0 - p) * s_j for _, _, _, s_j, q in scored
                          if s_j < c_saving - SOFTER_MARGIN and q > p and p < 1.0), default=0.0)
            after = max(0.0, float(next_offer) if next_offer is not None else 0.0, softer)
            if_rejected = (1.0 - p_final) * after
            if _survives(standing, state) and saving is not None:
                if_rejected = max(if_rejected, float(saving))
            approved = ledger.is_approved(offer_id, c.terms)
            extra = private.round_cost + (private.approval_cost if c_authority == "needs_approval" and not approved else 0)
            same_composition = (c.terms.delivery_days, c.terms.payment_days) == \
                (standing.terms.delivery_days, standing.terms.payment_days)
            scale = config.final_accept_scale if same_composition else config.final_accept_scale_recomposed
            if scale is None or p_final <= 0.0:
                value = counter_expected_saving(p, float(c_saving), if_rejected, extra)
            else:
                # Mix the two cases explicitly: if the offer is final, a same-terms lower price is
                # accepted with the scaled chance and a rejection leaves only what survives; if not,
                # the estimate is used as is and a rejection leaves the continuation value.
                keeps = max(0.0, float(saving)) if _survives(standing, state) and saving is not None else 0.0
                p_if_final = min(1.0, p * scale)
                value = (p_final * counter_expected_saving(p_if_final, float(c_saving), keeps)
                         + (1.0 - p_final) * counter_expected_saving(p, float(c_saving), max(after, keeps))
                         - extra)
            options.append(Option("counter", offer_id, c.terms, c.kind, c_authority, c_total, c_saving,
                                  p, value, after, approved))

    # Ending is considered only when nothing is left to try, so that an approximate value
    # never ends a negotiation that can still continue.
    counters = [o for o in options if o.action == "counter"]
    if rounds_left <= 0 or not counters:
        options.append(Option("end", None, None, "take_alternative", None,
                              facts["alternative_total_cost"], 0, None, 0.0))
    priority = {"accept": 2, "counter": 1, "end": 0}
    chosen = max(options, key=lambda o: (o.value, priority[o.action]))

    reasons: list[str] = []
    best_counter = max(counters, key=lambda o: o.value, default=None)
    if chosen.action == "accept":
        reasons.append("certain_saving_beats_best_counter" if best_counter
                       else "no_valid_counter_offer" if allow_counter else "counter_withheld_by_reviewer")
    elif chosen.action == "counter":
        reasons.append("highest_expected_saving")
        if any(o.action == "accept" for o in options):
            reasons.append("expected_saving_beats_accepting_now")
    else:
        reasons.append("nothing_beats_the_alternative")

    return finalize(state, private, chosen, options, facts, reasons, model)


def finalize(state: PublicState, private: BuyerPrivate, chosen: Option, options, facts: dict,
             reasons, model: dict | None = None) -> Decision:
    """Turn a chosen option into a Decision. Terms beyond the negotiator's own limits become a card.

    Every policy, including the baselines, ends here, so none of them can skip the approval gate.
    """
    model = model or {}
    reasons = list(reasons)
    if chosen.action in ("accept", "counter") and chosen.authority == "needs_approval" and not chosen.approved:
        card = ApprovalRequest(
            request_id=f"req-{state.current_round}-{chosen.terms.version()}",
            kind="accept_offer" if chosen.action == "accept" else "send_counter",
            offer_id=chosen.offer_id, round_index=state.current_round, terms=chosen.terms,
            terms_version=chosen.terms.version(),
            exceeded_auto_limits=tuple(auto_limit_exceedances(chosen.terms, state.quantity, private)),
            total_cost=chosen.total_cost, saving_vs_alternative=chosen.saving,
            gap_to_target=chosen.total_cost - private.target_total_cost, p_accept=chosen.p_accept,
        )
        return Decision(state.current_round, "request_approval", chosen, tuple(options), facts,
                        tuple(reasons + ["exceeds_negotiator_limit"]), card, model)
    return Decision(state.current_round, chosen.action, chosen, tuple(options), facts, tuple(reasons), None, model)


def standing_facts(state: PublicState, private: BuyerPrivate, config: PolicyConfig) -> tuple[Offer, dict, str, int | None, int | None]:
    """The code-computed view of the supplier's standing offer, shared by every policy."""
    standing = state.last_offer("supplier")
    if standing is None:
        raise ValueError("no supplier offer to respond to")
    authority, total, saving = _valued(standing.terms, state.quantity, private)
    facts = {
        "standing_offer_id": standing.offer_id,
        "standing_total_cost": total,
        "standing_saving_vs_alternative": saving,
        "standing_authority": authority,
        "alternative_total_cost": alternative_total_cost(state.quantity, private),
        "target_total_cost": private.target_total_cost,
        "rounds_left_after_counter": max(0, state.max_rounds - state.current_round),
        "expected_value_is_approximation": True,
    }
    return standing, facts, authority, total, saving


def counter_offer_id(state: PublicState, standing: Offer) -> str:
    """Our offer id names the supplier offer it answers, so an approval cannot carry over."""
    return f"B{state.current_round}/{standing.offer_id}"
