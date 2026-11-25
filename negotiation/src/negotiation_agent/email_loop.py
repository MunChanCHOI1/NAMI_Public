"""Negotiate through free text: the supplier writes Korean messages, the language layer reads them.

The rule supplier decides as before; its answer is rendered as a varied Korean message without
any structured fields, and the buyer side only gets the text. The language layer reads the text
into terms, code checks the reading, and anything unclear goes to a person (here: the evaluator
plays that person and types the true terms, which is counted). Every buyer counter-offer is also
phrased by the writer and guarded by code. All data is synthetic.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from .contracts import Offer, PublicState, SupplierHidden, SupplierResponse, Terms
from .language import LanguageModel, read_supplier_message, write_reply
from .session import SimulatedSupplier, outgoing_message


def render_counter(rng: random.Random, offer: Terms, previous: Terms | None, buyer: Terms, hint: str) -> str:
    p, d, pay = offer.unit_price, offer.delivery_days, offer.payment_days
    pay_text = "납품 즉시 결제" if pay == 0 else f"납품 후 {pay}일 결제"
    styles = ["full", "email", "quote_buyer"]
    if previous is not None and (d, pay) == (previous.delivery_days, previous.payment_days):
        styles.append("price_only")
    if d % 7 == 0:
        styles.append("weeks")
    style = rng.choice(styles)
    if style == "full":
        body = f"단가 {p:,}원, 납기 {d}일, {pay_text} 조건으로 다시 제안드립니다."
    elif style == "email":
        body = f"안녕하세요, 검토 결과 공유드립니다.\n- 단가: {p:,}원\n- 납기: {d}일\n- 결제: {pay_text}\n확인 부탁드립니다."
    elif style == "quote_buyer":
        body = f"말씀하신 {buyer.unit_price:,}원은 어렵고, {p:,}원까지는 맞춰 드릴 수 있습니다. 납기 {d}일, {pay_text}입니다."
    elif style == "price_only":
        body = f"단가를 {p:,}원으로 조정해 드리겠습니다. 나머지 조건은 기존과 같습니다."
    else:
        body = f"{d // 7}주 안에 납품 가능하고, 단가는 {p:,}원, 결제는 {'납품 즉시' if pay == 0 else f'납품 후 {pay}일'}로 부탁드립니다."
    return f"{body} {hint}".strip()


ACCEPT_TEXTS = ("제안하신 조건으로 진행하겠습니다.", "네, 그 조건으로 확정하겠습니다.", "좋습니다. 말씀하신 대로 진행하시죠.")
WALK_TEXTS = ("이 조건으로는 진행하기 어렵습니다.", "검토해 봤지만 이번 건은 진행하기 어렵겠습니다.")


class EmailSupplier:
    """Counterparty whose answers reach the buyer only as text, read back by the language layer."""

    def __init__(self, hidden: SupplierHidden, lm: LanguageModel, seed: int) -> None:
        self._inner = SimulatedSupplier(hidden)
        self.lm, self.rng = lm, random.Random(seed)
        self.log: list[dict] = []

    def confirm(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse:
        return self._inner.confirm(state, buyer_terms)

    def respond(self, state: PublicState, buyer_terms: Terms) -> SupplierResponse:
        last_supplier = state.last_offer("supplier")
        template = outgoing_message("counter", buyer_terms)
        draft = write_reply(self.lm, "counter", buyer_terms, template, state.quantity,
                            last_supplier.message if last_supplier else "")
        truth = self._inner.respond(state, buyer_terms)
        if truth.kind == "counter":
            text = render_counter(self.rng, truth.offer.terms, last_supplier.terms if last_supplier else None,
                                  buyer_terms, truth.message)
        elif truth.kind == "accept":
            text = self.rng.choice(ACCEPT_TEXTS + (f"네, {_plain(buyer_terms)}로 확정하겠습니다.",))
        else:
            text = self.rng.choice(WALK_TEXTS)
        reading = read_supplier_message(self.lm, text, state.quantity,
                                        last_supplier.terms if last_supplier else None, buyer_terms)
        read_kind = {"offer": "counter", "accept": "accept", "walk_away": "walk_away"}.get(reading.intent or "")
        if reading.intent == "offer" and reading.terms == buyer_terms:
            read_kind = "accept"          # the supplier restated our terms: an acceptance
        correct = read_kind == truth.kind and (truth.kind != "counter" or reading.terms == truth.offer.terms)
        entry = {"round": state.current_round, "buyer_terms": asdict(buyer_terms), "buyer_draft": draft.text,
                 "draft_source": draft.source, "draft_guard": list(draft.reasons), "supplier_text": text,
                 "truth_kind": truth.kind, "truth_terms": asdict(truth.offer.terms) if truth.offer else None,
                 "reading": reading.to_dict(), "correct": correct}
        if reading.status != "ok" or read_kind is None:
            entry["outcome"] = "person_typed"        # a person resolves it; the true answer is used
            self.log.append(entry)
            return SupplierResponse(truth.kind, offer=_with_text(truth.offer, text), message=text)
        entry["outcome"] = "automated_correct" if correct else "automated_wrong"
        self.log.append(entry)
        if read_kind == "counter":
            offer = Offer(f"S{state.current_round + 1}", state.current_round + 1, "supplier", reading.terms, text)
            return SupplierResponse("counter", offer=offer, message=text)
        return SupplierResponse(read_kind, message=text)


def _plain(t: Terms) -> str:
    pay = "납품 즉시 결제" if t.payment_days == 0 else f"납품 후 {t.payment_days}일 결제"
    return f"단가 {t.unit_price:,}원, 납기 {t.delivery_days}일, {pay}"


def _with_text(offer: Offer | None, text: str) -> Offer | None:
    if offer is None:
        return None
    return Offer(offer.offer_id, offer.round_index, offer.made_by, offer.terms, text, offer.valid_through_round)
