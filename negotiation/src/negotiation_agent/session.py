"""One negotiation from the buyer's side: policy decisions, approval gate, execution, record.

The session is the only place where an action takes effect. Before anything is sent or
accepted it re-checks, in code, that the exact terms are inside the negotiator's own
limits or carry an internal approval for that offer id and terms version. Nothing the
counterparty writes can create or change an approval.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Protocol

from .contracts import ApprovalLedger, BuyerPrivate, Offer, PublicState, SupplierHidden, SupplierResponse, Terms
from .economics import HardConstraintViolation, classify_authority, saving_vs_alternative
from .policy import ApprovalRequest, Decision, PolicyConfig, counter_offer_id, decide
from . import simulator
from .system1 import offer_number_check

TERMINAL = ("agreed", "ended_alternative", "supplier_walked")
# A supplier's free text is capped so that five rounds of it still fit the smallest model input limit in use:
# the native adapter refuses a question row above --max-input-tokens (2,048 on the Brev server) and never
# truncates, so an over-long history fails every later request and the session can never continue.
# Counted with the Zoro-1 tokenizer on 2026-11-20 (five supplier messages plus our longest replies, largest
# request = acceptance verification): about 1,670 tokens at 200 characters, 1,980 at 300, 4,190 at 1,000.
MAX_MESSAGE_CHARS = 200
REQUIRED_FIELDS = ("unit_price", "delivery_days", "payment_days")


class Counterparty(Protocol):
    def respond(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse: ...

    # Optional: answer a request to confirm the exact terms after an unclear acceptance.
    # def confirm(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse: ...


CONFIRMED_MESSAGE = "네, 말씀하신 조건 그대로 확정합니다."


class SimulatedSupplier:
    """Rule-based supplier. The hidden parameters stay inside this object."""

    def __init__(self, hidden: SupplierHidden) -> None:
        self._hidden = hidden

    def respond(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse:
        return simulator.respond(state, self._hidden, buyer_terms)

    def confirm(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse:
        """The rule supplier only ever accepted the exact terms, so it confirms them unchanged."""
        return SupplierResponse("accept", message=CONFIRMED_MESSAGE)


class AuthorizationError(RuntimeError):
    """An action was about to take effect without authority. Never expected; always a bug."""


def terms_text(terms: Terms) -> str:
    pay = "납품 즉시" if terms.payment_days == 0 else f"납품 후 {terms.payment_days}일"
    return f"단가 {terms.unit_price:,}원, 납기 {terms.delivery_days}일, 결제 {pay}"


def heard_before(message: str, earlier: list[str], threshold: float = 0.85) -> bool:
    """True when a supplier message repeats one they already sent (numbers and spacing ignored).

    Suppliers often paste the same line every round; answering it again each time reads as an echo.
    """
    def norm(text: str) -> str:
        return re.sub(r"[\d,.\s]+", "", text or "")
    now = norm(message)
    return bool(now) and any(SequenceMatcher(None, now, norm(e)).ratio() >= threshold for e in earlier if norm(e))


OFF_SCRIPT_EXCLUDED = "말씀하신 단가·납기·결제 밖의 요구는 이번 제안에 포함하지 않습니다."


def outgoing_message(action: str, terms: Terms | None = None, missing: tuple[str, ...] = (),
                     mismatches: tuple[str, ...] = ()) -> str:
    """Text sent to the counterparty. Built from the chosen public terms only."""
    if action == "counter":
        return f"다음 조건을 제안드립니다: {terms_text(terms)}."
    if action == "accept":
        return f"제시하신 조건({terms_text(terms)})으로 확정하겠습니다."
    if action == "accept_terms_only":
        return (f"제시하신 조건 중 {terms_text(terms)}은 받아들이겠습니다. 단가·납기·결제 밖의 요구는 이 합의에 "
                "포함하지 않습니다. 이 세 조건만으로 확정하시는지 답변 부탁드립니다.")
    if action == "ask_info":
        names = {"unit_price": "단가", "delivery_days": "납기", "payment_days": "결제일"}
        return "다음 항목을 숫자로 확인 부탁드립니다: " + ", ".join(names[m] for m in missing) + "."
    if action == "end":
        return "이번 건은 진행하지 않기로 했습니다. 검토에 감사드립니다."
    if action == "clarify_offer":
        return (f"보내주신 문장의 {', '.join(mismatches)}과 함께 적어 주신 조건({terms_text(terms)})이 서로 다릅니다. "
                "어느 조건으로 제안하시는지 숫자와 문장을 맞춰 다시 보내 주세요. 이 제안은 아직 받지 않았습니다.")
    if action == "clarify":
        return f"보내주신 답변의 조건이 저희 제안과 다릅니다. {terms_text(terms)} 그대로 확정하시는지 확인 부탁드립니다."
    raise ValueError(f"no outgoing message for {action!r}")


def parse_incoming_terms(raw: dict) -> tuple[Terms | None, tuple[str, ...]]:
    """A counterparty's raw offer -> complete terms, or the fields that are missing or unusable."""
    bad = tuple(f for f in REQUIRED_FIELDS
                if isinstance(raw.get(f), bool) or not isinstance(raw.get(f), int)
                or raw[f] < (0 if f == "payment_days" else 1))
    if bad:
        return None, bad
    return Terms(raw["unit_price"], raw["delivery_days"], raw["payment_days"]), ()


PLAUSIBLE_PRICE_RATIO = (0.5, 3.0)   # unit price vs our alternative quote's unit price
MAX_PLAUSIBLE_PAYMENT_DAYS = 180
MAX_PLAUSIBLE_DELIVERY_DAYS = 365


def implausible_terms(terms: Terms, private: BuyerPrivate) -> list[str]:
    """Supplier terms so far from any normal quote that a person should look before anything is decided."""
    low, high = PLAUSIBLE_PRICE_RATIO
    ref = private.alternative.unit_price
    odd = []
    if not low * ref <= terms.unit_price <= high * ref:
        odd.append(f"unit_price {terms.unit_price} outside {low}-{high}x of {ref}")
    if terms.payment_days > MAX_PLAUSIBLE_PAYMENT_DAYS:
        odd.append(f"payment_days {terms.payment_days} > {MAX_PLAUSIBLE_PAYMENT_DAYS}")
    if terms.delivery_days > MAX_PLAUSIBLE_DELIVERY_DAYS:
        odd.append(f"delivery_days {terms.delivery_days} > {MAX_PLAUSIBLE_DELIVERY_DAYS}")
    return odd


class NegotiationSession:
    def __init__(self, state: PublicState, private: BuyerPrivate, backend, counterparty: Counterparty | None,
                 config: PolicyConfig = PolicyConfig(), decide_fn=decide, gate=None,
                 max_confirmations: int = 1, writer=None) -> None:
        self.state, self._private, self._backend = state, private, backend
        # Optional language writer (language.make_reply_writer): rephrases our counter/accept/end text.
        # Code still picks the action and the terms; a draft that fails the code check is replaced by the template.
        self.writer = writer
        self._counterparty, self._config, self._decide = counterparty, config, decide_fn
        self.ledger = ApprovalLedger()
        self.status = "active"
        self.pending: ApprovalRequest | None = None
        self.agreement: Terms | None = None
        self.decisions: list[Decision] = []
        self.approval_log: list[dict] = []
        self.executed: list[dict] = []          # every action that took effect, for the audit
        self.outbox: list[dict] = []            # every text sent to the counterparty
        self.last_response: SupplierResponse | None = None
        self._cache: dict = {}
        self._waiting_offer: Offer | None = None  # None counterparty means a person answers asynchronously.
        self.review_log: list[dict] = []        # a person's decisions to re-ask after a failed estimate
        # Optional System 1 gate (system1.py). It can only stop: route to a person or hold an acceptance.
        self.gate = gate
        self.max_confirmations = max_confirmations
        self.review_reason: str | None = None
        self.events: list[dict] = []            # ordered record of what happened, for the scenario logs
        self._confirmations = 0
        self.offers_held: list[dict] = []       # person-typed offers whose text and numbers disagreed
        self._off_script_offer_id: str | None = None      # standing offer whose text added an off-script demand
        self._counter_withheld_round: int | None = None   # a reviewer chose accept-or-end for this round
        self._routing_unanswered = False                  # the System 1 look at the standing offer failed

    def _event(self, kind: str, **data) -> None:
        self.events.append({"index": len(self.events), "round": self.state.current_round, "event": kind, **data})

    # ---- approval gate -----------------------------------------------------------------------

    def approve(self, request_id: str, approver: str) -> None:
        request = self._pending(request_id)
        self.ledger.grant(request.offer_id, request.terms, approver)
        self._close(request, "approved", approver)

    def deny(self, request_id: str, approver: str) -> None:
        request = self._pending(request_id)
        self.ledger.deny(request.offer_id, request.terms)
        self._close(request, "denied", approver)

    def _pending(self, request_id: str) -> ApprovalRequest:
        if self.pending is None or self.pending.request_id != request_id:
            raise ValueError("no pending approval request with this id")
        return self.pending

    def _close(self, request: ApprovalRequest, outcome: str, approver: str) -> None:
        self._event("approval_" + outcome, request_id=request.request_id, offer_id=request.offer_id,
                    terms_version=request.terms_version, decided_by=approver)
        self.approval_log.append({"request_id": request.request_id, "offer_id": request.offer_id,
                                  "terms_version": request.terms_version, "kind": request.kind,
                                  "round": request.round_index, "outcome": outcome, "decided_by": approver})
        self.pending, self.status = None, "active"

    def _authorize(self, offer_id: str, terms: Terms) -> str:
        """Final check in code immediately before an action takes effect."""
        try:
            level = classify_authority(terms, self.state.quantity, self._private)
        except HardConstraintViolation:
            level = "forbidden"
        if level == "auto":
            return "auto"
        if level == "needs_approval" and self.ledger.is_approved(offer_id, terms):
            return "approved"
        raise AuthorizationError(f"{offer_id} with terms {terms.version()} is {level} and has no approval")

    # ---- human review after a failed estimate ----------------------------------------------------

    def resume_after_review(self, reviewer: str, withhold_counter: bool = False) -> None:
        """A person saw the failed estimate and asks for a new one. Nothing is sent or decided here.

        The failed attempt stays in the decision record. No value is filled in for it: the
        next step asks the backend again and, if that fails too, stops for review again.

        After a final-offer claim the reviewer may also withhold the counter-offer for this round. The
        policy then compares only accepting the standing offer and ending; authority is checked as always.

        A failed System 1 look at the standing offer is likewise asked again here, not skipped: the offer
        is never answered unread. If the look fails again, or now finds a reason to stop, the session
        stops for review again. The caller holds the backend lock, as for a step.
        """
        if self.status != "needs_review":
            raise RuntimeError(f"nothing to review while {self.status}")
        if not reviewer:
            raise ValueError("a review needs an internal reviewer")
        if withhold_counter and self.review_reason != "final_offer_claim":
            raise ValueError("a counter-offer can be withheld only after a final-offer claim")
        self.review_log.append({"round": self.state.current_round, "decision_index": len(self.decisions) - 1,
                                "outcome": "accept_or_end_only" if withhold_counter else "retry_requested",
                                "decided_by": reviewer})
        self._event("review_resumed", decided_by=reviewer, review_reason=self.review_reason,
                    **({"withhold_counter": True} if withhold_counter else {}))
        if withhold_counter:
            self._counter_withheld_round = self.state.current_round
        self.status, self.review_reason = "active", None
        if self._routing_unanswered:
            self._route_standing_offer()

    def _needs_review(self, reason: str, **data) -> None:
        self.status, self.review_reason = "needs_review", reason
        self._event("needs_review", reason=reason, **data)

    # ---- System 1 gate -------------------------------------------------------------------------

    def _route_new_offer(self) -> None:
        """A new supplier offer is in the history. With a gate, diagnose it and maybe route to a person."""
        offer = self.state.last_offer("supplier")
        self._event("supplier_offer", offer_id=offer.offer_id, terms=asdict(offer.terms), message=offer.message)
        self._counter_withheld_round = None        # a reviewer's choice applied to the previous offer only
        odd = implausible_terms(offer.terms, self._private)
        if odd:
            # Far outside any normal quote: most likely a typo or a misread, never a windfall to execute.
            # The System 1 look still runs when the reviewer resumes, so the offer is never answered unread.
            self._routing_unanswered = self.gate is not None
            self._needs_review("implausible_offer", checks=odd)
            return
        if self.gate is not None:
            self._route_standing_offer()

    def _route_standing_offer(self) -> None:
        """The System 1 look at the standing supplier offer. Run again after a failed look is reviewed."""
        offer = self.state.last_offer("supplier")
        check = self.gate.route(self.state, self._standing_offer_can_be_accepted())
        self._event("system1_routing", check_index=len(self.gate.checks) - 1, route=check["route"],
                    reasons=check["reasons"], diagnosis=check.get("diagnosis"))
        self._off_script_offer_id = offer.offer_id if "off_script_demand" in check["reasons"] else None
        self._routing_unanswered = check["status"] != "ok"
        if check["route"] == "human_review":
            self._needs_review(check["reasons"][0])

    def _standing_offer_can_be_accepted(self) -> bool:
        """Would ending the talks now lose something: an executable standing offer that beats the alternative."""
        terms = self.state.last_offer("supplier").terms
        return (classify_authority(terms, self.state.quantity, self._private) != "forbidden"
                and saving_vs_alternative(terms, self.state.quantity, self._private) > 0)

    def route_opening_offer(self) -> None:
        """Give the offer the session started with the same System 1 look as every later offer."""
        if self.state.history and not self.events:
            self._route_new_offer()

    def _accept_reply(self, state_after: PublicState, terms: Terms, reply: str) -> None:
        """The supplier says it accepts the exact terms we sent. Agree only once that is verified."""
        self._event("supplier_accept", terms=asdict(terms), message=reply)
        if self.gate is None:
            self.agreement, self.status = terms, "agreed"
            return
        check = self.gate.verify(state_after, terms, reply)
        self._event("system1_verification", check_index=len(self.gate.checks) - 1, verified=check["verified"],
                    reasons=check["reasons"])
        if check["verified"]:
            self.agreement, self.status = terms, "agreed"
            self._event("agreement_recorded", terms=asdict(terms), terms_version=terms.version())
            return
        if "system1_unavailable" in check["reasons"]:
            self._needs_review("system1_unavailable")
            return
        if self._confirmations >= self.max_confirmations:
            self._needs_review("acceptance_unverified")
            return
        self._confirmations += 1
        self._send("clarify", terms)                      # asking again commits to nothing
        if self._counterparty is None:
            self._waiting_offer = Offer(f"B{state_after.current_round}c{self._confirmations}",
                                        state_after.current_round, "buyer", terms)
            self.status = "awaiting_supplier"
            return
        confirm = getattr(self._counterparty, "confirm", None)
        if confirm is None:
            self._needs_review("acceptance_unverified")
            return
        response = confirm(state_after, terms)
        self.last_response = response
        if response.kind == "accept":
            self._accept_reply(state_after, terms, response.message)
        else:
            self._event("supplier_" + response.kind, message=response.message)
            self._needs_review("acceptance_unverified")

    # ---- incoming offers from a person ---------------------------------------------------------

    def _hold_if_text_disagrees(self, terms: Terms, message: str, answering: Terms | None) -> str | None:
        """Code check: an amount or day count in the text that is not in the typed terms holds the offer.

        The offer is not added to the history and uses no round; the agent asks which terms are meant.
        """
        check = offer_number_check(message, terms, answering, self.state.quantity)
        if check["ok"]:
            return None
        self.offers_held.append({"round": self.state.current_round, "terms": asdict(terms), "message": message,
                                 "mismatches": check["mismatches"]})
        self._event("supplier_offer_held", terms=asdict(terms), message=message, mismatches=check["mismatches"])
        text = outgoing_message("clarify_offer", terms, mismatches=tuple(check["mismatches"]))
        self.outbox.append({"round": self.state.current_round, "action": "clarify_offer", "text": text})
        self._event("buyer_clarify_offer", terms=None, text=text, mismatches=check["mismatches"])
        return text

    def submit_supplier_offer(self, raw: dict, message: str = "") -> str | None:
        """Add a supplier offer typed by a person. Returns a clarification request if it is incomplete."""
        if self._waiting_offer is not None:
            return self.submit_supplier_response("counter", raw, message)
        if self.status not in ("active", "awaiting_info"):
            raise RuntimeError(f"cannot take an offer while {self.status}")
        terms, missing = parse_incoming_terms(raw)
        if terms is None:
            self.status = "awaiting_info"
            text = outgoing_message("ask_info", missing=missing)
            self.outbox.append({"round": self.state.current_round, "action": "ask_info", "text": text})
            self._event("buyer_ask_info", terms=None, text=text, missing=list(missing))
            return text
        held = self._hold_if_text_disagrees(terms, message, None)
        if held is not None:
            self.status = "awaiting_info"
            return held
        same_round = sum(1 for o in self.state.history
                         if o.made_by == "supplier" and o.round_index == self.state.current_round)
        offer_id = f"S{self.state.current_round}" + (f"r{same_round}" if same_round else "")
        offer = Offer(offer_id, self.state.current_round, "supplier", terms, message)
        self.state = replace(self.state, history=self.state.history + (offer,))
        self.pending, self.status = None, "active"     # a new offer supersedes any card for the old one
        self._route_new_offer()
        return None

    def submit_supplier_response(self, kind: str, raw: dict | None = None, message: str = "") -> str | None:
        """Receive a person's answer to the exact buyer offer already sent; never execute a buyer action."""
        if self._counterparty is not None or self._waiting_offer is None \
                or self.status not in ("awaiting_supplier", "awaiting_info"):
            raise RuntimeError("no buyer offer is waiting for a supplier response")
        if kind not in ("accept", "counter", "walk_away"):
            raise ValueError("unknown supplier response")
        offer = None
        if kind == "counter":
            if self.state.current_round >= self.state.max_rounds:
                raise ValueError("the final public round cannot be extended")
            terms, missing = parse_incoming_terms(raw or {})
            if terms is None:
                self.status = "awaiting_info"
                text = outgoing_message("ask_info", missing=missing)
                self.outbox.append({"round": self.state.current_round, "action": "ask_info", "text": text})
                self._event("buyer_ask_info", terms=None, text=text, missing=list(missing))
                return text
            held = self._hold_if_text_disagrees(terms, message, self._waiting_offer.terms)
            if held is not None:
                self.status = "awaiting_supplier"         # the buyer offer still waits for an answer
                return held
            next_round = self.state.current_round + 1
            offer = Offer(f"S{next_round}", next_round, "supplier", terms, message)
            self.state = replace(self.state, history=self.state.history + (offer,), current_round=next_round)
            self.status = "active"
        waiting, self._waiting_offer = self._waiting_offer, None
        self.last_response = SupplierResponse(kind, offer, message)
        if kind == "counter":
            self._route_new_offer()
        elif kind == "accept":
            self._accept_reply(self.state, waiting.terms, message)
        else:
            self._event("supplier_walk_away", message=message)
            self.status = "supplier_walked"
        return None

    # ---- one step ------------------------------------------------------------------------------

    def step(self) -> Decision:
        """Decide and, when authority allows, act. While a card is pending nothing is re-decided."""
        if self.status in TERMINAL or self.status in ("awaiting_approval", "needs_review", "awaiting_info", "awaiting_supplier"):
            raise RuntimeError(f"no step possible while {self.status}")
        if self._counter_withheld_round == self.state.current_round:
            decision = self._decide(self.state, self._private, self._backend, self.ledger, self._config, self._cache,
                                    allow_counter=False)
        else:
            decision = self._decide(self.state, self._private, self._backend, self.ledger, self._config, self._cache)
        self.decisions.append(decision)
        chosen = decision.chosen
        self._event("decision", decision_index=len(self.decisions) - 1, action=decision.action,
                    reasons=list(decision.reasons),
                    chosen=({"action": chosen.action, "offer_id": chosen.offer_id,
                             "terms": asdict(chosen.terms) if chosen.terms else None, "authority": chosen.authority,
                             "saving": chosen.saving, "p_accept": chosen.p_accept} if chosen else None),
                    model_status=decision.model.get("status"), model_error=decision.model.get("error"))
        if decision.action == "request_approval":
            self.pending, self.status = decision.approval_request, "awaiting_approval"
        elif decision.action == "human_review":
            self._needs_review(decision.reasons[0] if decision.reasons else "human_review")
        elif decision.action == "end":
            self._send("end", None)
            self.status = "ended_alternative"
        elif decision.action == "accept" and decision.chosen.offer_id == self._off_script_offer_id:
            # The offer's text added a demand outside the three terms, so accepting it would agree on
            # something unclear. Send the three terms back as ours and agree only on an exact acceptance.
            self._counter(decision, "accept_terms_only")
        elif decision.action == "accept":
            option = decision.chosen
            basis = self._authorize(option.offer_id, option.terms)
            self._record(option.offer_id, "accept", option.terms, basis)
            self._send("accept", option.terms)
            self.agreement, self.status = option.terms, "agreed"
        else:
            self._counter(decision)
        return decision

    def _counter(self, decision: Decision, message: str = "counter") -> None:
        option = decision.chosen
        basis = self._authorize(option.offer_id, option.terms)
        self._record(option.offer_id, message, option.terms, basis)
        text = self._send(message, option.terms)
        before = self.state
        offer_id = option.offer_id if option.action == "counter" else counter_offer_id(before, before.last_offer("supplier"))
        ours = Offer(offer_id, before.current_round, "buyer", option.terms, text)
        if self._counterparty is None:
            self.state = replace(before, history=before.history + (ours,))
            self._waiting_offer, self.status = ours, "awaiting_supplier"
            return
        response = self._counterparty.respond(before, option.terms)
        history = before.history + (ours,)
        self.last_response = response
        if response.kind == "accept":
            self.state = replace(before, history=history)
            self._accept_reply(self.state, option.terms, response.message)
        elif response.kind == "walk_away":
            self.state = replace(before, history=history)
            self._event("supplier_walk_away", message=response.message)
            self.status = "supplier_walked"
        elif response.kind == "counter" and isinstance(response.offer, Offer) and response.offer.made_by == "supplier":
            self.state = replace(before, history=history + (response.offer,), current_round=before.current_round + 1)
            self._route_new_offer()
        else:
            # A counter without terms (or an unknown reply) is not an offer: stop for a person instead
            # of crashing on the missing offer or deciding against an old one.
            self.state = replace(before, history=history)
            self._event("supplier_reply_unusable", reply_kind=response.kind, message=response.message)
            self._needs_review("supplier_reply_without_terms")

    def _send(self, action: str, terms: Terms | None) -> str:
        template = outgoing_message(action, terms)
        text, source, notes = template, "template", ()
        if self.writer is not None:
            last = self.state.last_offer("supplier")
            message = last.message if last else ""
            earlier = [o.message for o in self.state.history if o.made_by == "supplier" and o is not last]
            if message and heard_before(message, earlier):
                message = ""        # already acknowledged once: answer with the terms, not the same echo again
            try:
                text, source, notes = self.writer(action, terms, template, message, self.state.current_round,
                                                  earlier=tuple(e["text"] for e in self.outbox[-3:]))
            except Exception as error:   # the writer must never stop a negotiation
                text, source, notes = template, "template", (f"writer_error:{type(error).__name__}",)
        last = self.state.last_offer("supplier")
        if action == "counter" and last is not None and last.offer_id == self._off_script_offer_id:
            # Countering an offer that carried a demand outside price, delivery and payment: say plainly that
            # the demand is not part of our offer, so a later "yes" cannot be read as agreeing to it.
            text = f"{text} {OFF_SCRIPT_EXCLUDED}"
        entry = {"round": self.state.current_round, "action": action, "text": text}
        if self.writer is not None:
            entry["writer"] = source
            if notes:
                entry["writer_notes"] = list(notes)
        self.outbox.append(entry)
        self._event("buyer_" + action, terms=asdict(terms) if terms else None, text=text,
                    **({"writer": source} if self.writer is not None else {}))
        return text

    def _record(self, offer_id: str, action: str, terms: Terms, basis: str) -> None:
        self.executed.append({"round": self.state.current_round, "offer_id": offer_id, "action": action,
                              "terms": asdict(terms), "terms_version": terms.version(), "authority_basis": basis})

    def run(self, approver: Callable[[ApprovalRequest], bool] | None = None, approver_name: str = "approver",
            max_steps: int = 40) -> str:
        """Run to an end state, or stop where a person is needed. `approver` answers approval cards."""
        for _ in range(max_steps):
            if self.status in TERMINAL or self.status in ("needs_review", "awaiting_info", "awaiting_supplier"):
                break
            if self.status == "awaiting_approval":
                if approver is None:
                    break
                request = self.pending
                (self.approve if approver(request) else self.deny)(request.request_id, approver_name)
                continue
            self.step()
        return self.status

    # ---- records -------------------------------------------------------------------------------

    def public_transcript(self) -> list[dict]:
        """What the counterparty has seen. No private value of ours appears here."""
        rows = [{"round": o.round_index, "by": o.made_by, "offer_id": o.offer_id,
                 "terms": asdict(o.terms), "message": o.message} for o in self.state.history]
        if self.status in ("agreed", "ended_alternative") and self.outbox and self.outbox[-1]["action"] in ("accept", "end"):
            rows.append({"round": self.outbox[-1]["round"], "by": "buyer", "offer_id": None,
                         "terms": asdict(self.agreement) if self.agreement else None,
                         "message": self.outbox[-1]["text"]})
        return rows

    def outcome(self) -> dict[str, Any]:
        """Result relative to the alternative quote, plus an audit of every action that took effect."""
        saving = saving_vs_alternative(self.agreement, self.state.quantity, self._private) if self.agreement else 0
        requested = sum(1 for d in self.decisions if d.action == "request_approval")
        violations = []
        for row in self.executed:
            terms = Terms(**row["terms"])
            level = classify_authority(terms, self.state.quantity, self._private)
            if level == "forbidden" or (level == "needs_approval" and not self.ledger.is_approved(row["offer_id"], terms)):
                violations.append(row)
        result = {
            "status": self.status,
            "agreement": asdict(self.agreement) if self.agreement else None,
            "realized_saving_vs_alternative": saving,
            "rounds_used": self.state.current_round,
            "model_requests": sum(1 for d in self.decisions if "trace_id" in d.model),
            "approval_cards": requested,
            "approvals_granted": sum(1 for a in self.approval_log if a["outcome"] == "approved"),
            "approvals_denied": sum(1 for a in self.approval_log if a["outcome"] == "denied"),
            "authority_violations": violations,
            "offers_held_for_text_mismatch": len(self.offers_held),
            **self._deal_report(saving),
        }
        if self.gate is not None:
            checks = self.gate.checks
            result["system1"] = {
                "checks": len(checks),
                "routing_checks": sum(1 for c in checks if c["check"] == "routing"),
                "routed_to_person": sum(1 for c in checks if c.get("route") == "human_review"),
                "verification_checks": sum(1 for c in checks if c["check"] == "verification"),
                "acceptances_held": sum(1 for c in checks if c["check"] == "verification" and not c["verified"]),
                "failed_answers": sum(1 for c in checks if c["status"] == "error"),
                "review_reason": self.review_reason,
            }
        return result

    def _deal_report(self, saving: int) -> dict[str, Any]:
        """Per-deal figures a buying team can check: the baseline is fixed by the first supplier quote."""
        first = next((o for o in self.state.history if o.made_by == "supplier"), None)
        try:
            first_saving = saving_vs_alternative(first.terms, self.state.quantity, self._private) if first else None
        except HardConstraintViolation:
            first_saving = None      # a first quote we could never take (e.g. 31+ days) has no cost to compare
        latency = [d.model.get("latency_ms") for d in self.decisions] + \
                  [c.get("latency_ms") for c in (self.gate.checks if self.gate is not None else [])]
        return {
            "first_offer_terms": asdict(first.terms) if first else None,
            # What negotiating added over taking the supplier's first quote as it stood (0 without a deal).
            "saving_vs_first_offer": (saving - first_saving) if (self.agreement and first_saving is not None) else 0,
            "human_touches": len(self.approval_log) + len(self.review_log),
            "engine_time_ms": round(sum(x for x in latency if isinstance(x, (int, float))), 1),
        }

    def internal_record(self) -> list[dict]:
        """Decision records with our private numbers. For our own screen and the evaluator only."""
        rows = []
        for d in self.decisions:
            rows.append({
                "round": d.round_index, "action": d.action, "reasons": list(d.reasons), "facts": d.facts,
                "model": d.model,
                "chosen": _option_view(d.chosen) if d.chosen else None,
                "options": [_option_view(o) for o in d.options],
                "approval_request": ({**asdict(d.approval_request), "exceeded_auto_limits":
                                      list(d.approval_request.exceeded_auto_limits)} if d.approval_request else None),
            })
        return rows


def _option_view(option) -> dict:
    view = asdict(option)
    view["value"] = round(option.value, 1)
    return view
