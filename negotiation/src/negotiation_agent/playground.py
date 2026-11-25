"""Negotiation playground: a person plays the supplier, the buyer agent answers on its own.

The page /play is a tester's view. The left side is what a supplier would see; the right side
shows the agent's inside (its decision, System 1 checks and the buyer's private numbers) so the
tester can see why it answered as it did. A real supplier would never see that right side.

The person also acts as the buyer company's internal approver: an approval card waits for them,
exactly as on the dashboard. Nothing is approved automatically.

The person's own offers are typed terms, the same as the dashboard's "human supplier" input.
The browser cannot choose the backend, a file path, a URL, a seed or any simulator parameter.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .contracts import Offer, PublicState, Terms
from .economics import HardConstraintViolation, alternative_total_cost, buyer_total_cost, classify_authority
from .session import MAX_MESSAGE_CHARS, TERMINAL, NegotiationSession
from .simulator import ITEM, MAX_ROUNDS, QUANTITY, default_buyer_private
from .system1 import gate_for, offer_number_check

PERSON_NEEDED = ("awaiting_supplier", "awaiting_info", "awaiting_approval", "needs_review") + TERMINAL
MAX_AGENT_STEPS = 12


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpeningInput(_Payload):
    unit_price: StrictInt = Field(ge=1, le=1_000_000)
    delivery_days: StrictInt = Field(ge=1, le=3650)
    payment_days: StrictInt = Field(ge=0, le=3650)
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class RevisionInput(_Payload):
    revision: StrictInt = Field(ge=0)


class ReplyInput(RevisionInput):
    kind: Literal["accept", "counter", "walk_away"]
    unit_price: StrictInt | None = Field(default=None, ge=1, le=1_000_000)
    delivery_days: StrictInt | None = Field(default=None, ge=1, le=3650)
    payment_days: StrictInt | None = Field(default=None, ge=0, le=3650)
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class ReviewInput(RevisionInput):
    withhold_counter: bool = False     # after a final-offer claim: compare only accepting and ending


class PlayApprovalInput(RevisionInput):
    request_id: str = Field(max_length=100)
    terms_version: str = Field(max_length=100)
    decision: Literal["approve", "deny"]


def _terms(t: Terms | dict | None) -> dict | None:
    if t is None:
        return None
    return asdict(t) if isinstance(t, Terms) else dict(t)


class PlaySession:
    """One negotiation where the counterparty is the person at the keyboard."""

    def __init__(self, session_id: str, opening: OpeningInput, backend, backend_lock: Lock,
                 system1_rules: bool = False, writer=None) -> None:
        terms = Terms(opening.unit_price, opening.delivery_days, opening.payment_days)
        state = PublicState(f"play-{session_id[:6]}", ITEM, QUANTITY, MAX_ROUNDS, 1,
                            (Offer("S1", 1, "supplier", terms, opening.message.strip()),))
        self.id, self.backend, self.backend_lock = session_id, backend, backend_lock
        self.private = default_buyer_private()
        # The person at the keyboard is also the reviewer, so a final-offer claim stops for their choice.
        self.session = NegotiationSession(state, self.private, backend, None,
                                          gate=gate_for(backend, force_rules=system1_rules,
                                                        stop_on_final_claim=True),
                                          writer=writer)
        self.revision, self.lock = 0, Lock()
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with backend_lock:
            self.session.route_opening_offer()
            self.advance()

    def advance(self) -> None:
        """Let the agent act until a person is needed. Caller holds the backend lock."""
        s = self.session
        for _ in range(MAX_AGENT_STEPS):
            if s.status != "active":
                break
            s.step()

    # ---- views -----------------------------------------------------------------------------

    def chat(self) -> list[dict]:
        """The conversation as both sides saw it, from the ordered session events."""
        rows = []
        for e in self.session.events:
            kind, base = e["event"], {"round": e["round"], "index": e["index"]}
            if kind == "supplier_offer":
                rows.append({**base, "who": "me", "kind": "offer", "terms": e["terms"], "text": e["message"]})
            elif kind == "supplier_offer_held":
                rows.append({**base, "who": "me", "kind": "offer_held", "terms": e["terms"], "text": e["message"],
                             "mismatches": e["mismatches"]})
            elif kind == "supplier_accept":
                rows.append({**base, "who": "me", "kind": "accept", "terms": e["terms"], "text": e["message"]})
            elif kind == "supplier_walk_away":
                rows.append({**base, "who": "me", "kind": "walk_away", "text": e["message"]})
            elif kind.startswith("buyer_"):
                rows.append({**base, "who": "agent", "kind": kind[len("buyer_"):], "terms": e.get("terms"),
                             "text": e["text"]})
            elif kind == "agreement_recorded":
                rows.append({**base, "who": "system", "kind": "agreed", "terms": e["terms"],
                             "text": "합의가 기록되었습니다."})
        return rows

    def view(self) -> dict:
        s = self.session
        waiting = s._waiting_offer
        standing = s.state.last_offer("supplier")
        try:
            standing_total = buyer_total_cost(standing.terms, s.state.quantity, self.private)
            standing_authority = classify_authority(standing.terms, s.state.quantity, self.private)
        except HardConstraintViolation:
            standing_total, standing_authority = None, "forbidden"
        alternative = alternative_total_cost(s.state.quantity, self.private)
        from .web import _records, backend_view, system1_view   # lazy: web imports this module
        records = _records(s.internal_record())
        return {
            "id": self.id, "revision": self.revision, "status": s.status, "round": s.state.current_round,
            "max_rounds": s.state.max_rounds, "item": s.state.item, "quantity": s.state.quantity,
            "chat": self.chat(),
            "you": {   # what a supplier would see and may do
                "can_reply": s.status in ("awaiting_supplier", "awaiting_info"),
                "can_counter": s.status in ("awaiting_supplier", "awaiting_info")
                               and s.state.current_round < s.state.max_rounds,
                "offer_to_answer": _terms(waiting.terms) if waiting is not None else None,
                "agreement": _terms(s.agreement),
            },
            "agent": {  # tester only: never part of a supplier's screen
                "backend": backend_view(self.backend),
                "review_reason": s.review_reason,
                "pending": ({**asdict(s.pending), "exceeded_auto_limits": list(s.pending.exceeded_auto_limits)}
                            if s.pending else None),
                "last_decision": records[-1] if records else None,
                "decisions": len(records),
                "system1": system1_view(s),
                "private": {"target_total_cost": self.private.target_total_cost,
                            "alternative": asdict(self.private.alternative),
                            "alternative_total_cost": alternative,
                            "auto_bounds": asdict(self.private.auto_bounds),
                            "company_bounds": asdict(self.private.company_bounds),
                            "hard_max_delivery_days": self.private.hard_max_delivery_days},
                "standing": {"offer_id": standing.offer_id, "terms": asdict(standing.terms),
                             "total_cost": standing_total, "authority": standing_authority,
                             "saving_vs_alternative": alternative - standing_total if standing_total is not None else None},
                "outcome": s.outcome(),
            },
        }

    def record(self) -> dict:
        s = self.session
        try:
            described = self.backend.describe()
        except Exception as error:   # keep the record even when the backend cannot describe itself
            described = {"kind": self.backend.kind, "describe_error": f"{type(error).__name__}: {error}"}
        return {
            "schema": "negotiation_playground_session_v1", "case_id": s.state.case_id,
            "created_at_utc": self.created_at, "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": described,
            "evidence_class": "real_model_run" if self.backend.is_model else "cpu_reference_not_model_evidence",
            "counterparty": "person_playing_supplier", "approver": "person_in_playground",
            "scope": "interactive synthetic playground; one person-played negotiation, not an evaluation",
            "status": s.status, "outcome": s.outcome(), "chat": self.chat(), "events": s.events,
            "system1_checks": s.gate.checks if s.gate is not None else [],
            "approvals": s.approval_log, "reviews": s.review_log, "internal_record": s.internal_record(),
        }


MAX_LIVE_SESSIONS = 200   # memory bound for a shared server; the oldest session is dropped first


def remember_bounded(store: dict, key, value, cap: int | None = None) -> list:
    """Insert and drop the oldest entries beyond cap; returns the dropped values."""
    cap = MAX_LIVE_SESSIONS if cap is None else cap
    store[key] = value
    dropped = []
    while len(store) > cap:
        dropped.append(store.pop(next(iter(store))))
    return dropped


def install_playground(app: FastAPI, static_dir: Path, backend, backend_lock: Lock,
                       save_dir: Path | None = None, system1_rules: bool = False, writer=None) -> None:
    sessions: dict[str, PlaySession] = {}

    def save(entry: PlaySession) -> None:
        if save_dir is None:
            return
        import json
        save_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(entry.record(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (save_dir / f"play-{entry.id}.json").write_text(text, encoding="utf-8")

    def entry_for(session_id: str, revision: int | None = None) -> PlaySession:
        if session_id not in sessions:
            raise HTTPException(404, "체험 세션이 없습니다. 새로 시작해 주세요.")
        entry = sessions[session_id]
        if revision is not None and revision != entry.revision:
            raise HTTPException(409, "화면보다 세션이 먼저 바뀌었습니다. 새로고침 후 다시 시도하세요.")
        return entry

    def act(entry: PlaySession, change) -> dict:
        try:
            with backend_lock:
                change()
                entry.advance()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        entry.revision += 1
        save(entry)
        return entry.view()

    @app.get("/play")
    def play_page():
        return FileResponse(static_dir / "play.html")

    @app.post("/api/play")
    def play_start(payload: OpeningInput):
        # The opening offer is checked like every later one, before a session exists.
        check = offer_number_check(payload.message, Terms(payload.unit_price, payload.delivery_days,
                                                          payload.payment_days), None, QUANTITY)
        if not check["ok"]:
            raise HTTPException(422, f"문장의 {', '.join(check['mismatches'])}과 입력한 조건이 다릅니다. "
                                     "숫자와 문장을 맞춰 다시 보내 주세요.")
        session_id = uuid4().hex
        entry = PlaySession(session_id, payload, backend, backend_lock, system1_rules, writer)
        remember_bounded(sessions, session_id, entry)
        save(entry)
        return entry.view()

    @app.get("/api/play/{session_id}")
    def play_state(session_id: str):
        entry = entry_for(session_id)
        with entry.lock:
            return entry.view()

    @app.post("/api/play/{session_id}/reply")
    def play_reply(session_id: str, payload: ReplyInput):
        entry = entry_for(session_id)
        with entry.lock:
            entry_for(session_id, payload.revision)
            raw = payload.model_dump()
            return act(entry, lambda: entry.session.submit_supplier_response(payload.kind, raw, payload.message.strip()))

    @app.post("/api/play/{session_id}/approval")
    def play_approval(session_id: str, payload: PlayApprovalInput):
        entry = entry_for(session_id)
        with entry.lock:
            entry_for(session_id, payload.revision)
            pending = entry.session.pending
            if pending is None or pending.request_id != payload.request_id or pending.terms_version != payload.terms_version:
                raise HTTPException(409, "승인 카드 또는 조건 버전이 일치하지 않습니다.")
            action = entry.session.approve if payload.decision == "approve" else entry.session.deny
            return act(entry, lambda: action(payload.request_id, "playground-person"))

    @app.post("/api/play/{session_id}/review")
    def play_review(session_id: str, payload: ReviewInput):
        entry = entry_for(session_id)
        with entry.lock:
            entry_for(session_id, payload.revision)
            return act(entry, lambda: entry.session.resume_after_review("playground-person",
                                                                        payload.withhold_counter))
