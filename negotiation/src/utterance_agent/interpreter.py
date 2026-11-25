"""Turn the four utterance answers into one final-offer estimate.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only, with CPU readers
and test doubles. Nothing here is evidence about the real model.

Track A: `Thresholds`, `combine` and the three readers the bench, the threshold fit and the CPU
stand-in use. Track B: the policy-path wrapper `UtteranceFinalBackend`, which keeps a base
backend's acceptance estimates, asks a reader about the latest supplier utterance and replaces
only the final-offer estimate. The default request object is handed to the base untouched.

The module sees the public state and model answers only. It does not import the sentence bank:
`LabelReader` receives the labelling function from its caller.

Rule (section 5 of the design document), in this order:
    no utterance                        -> round prior, "no_utterance_round_prior"
    u_instruction >= instruction        -> round prior, "instruction_flagged_round_prior"
    max(u_final, u_walkaway) >= final   -> 1.0,         "utterance_final"
    otherwise                           -> round prior, "round_prior"
`u_concession` is recorded and reported but never used for the estimate in this version.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from negotiation_agent.contracts import PublicState
from negotiation_agent.estimate_sources import round_prior, says_final
from negotiation_agent.provenance import (json_safe, request_sha256, reset_response, response_snapshot,
                                          validate_predictions)
from negotiation_agent.requests import FINAL_OFFER_QID

from . import PROTOCOL_VERSION
from .questions import QIDS, QUESTION_SET_VERSION, build_utterance_request, latest_supplier_utterance

FITTED_ON = (None, "utt-fit", "utt-smoke-rehearsal")   # the rehearsal value never runs the test block
VALUE_FROM = ("utterance_final", "round_prior", "instruction_flagged_round_prior", "no_utterance_round_prior",
              "public_round_limit_not_asked")           # the last one is set by the wrapper, not by combine()


def _is_probability(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and 0.0 <= value <= 1.0)


@dataclass(frozen=True)
class Thresholds:
    """The two gates of the rule. There is deliberately no default: an unfitted 0.5 must be asked for."""

    final: float                 # applied to max(u_final, u_walkaway)
    instruction: float           # applied to u_instruction
    fitted_on: str | None        # None (the fixed 0.5 pair), "utt-fit", or "utt-smoke-rehearsal"

    def validate(self) -> None:
        if not _is_probability(self.final) or not _is_probability(self.instruction):
            raise ValueError("thresholds must be finite numbers in [0, 1]")
        if self.fitted_on not in FITTED_ON:
            raise ValueError(f"fitted_on must be one of {FITTED_ON}")

    def to_json(self) -> dict:
        return {"final": float(self.final), "instruction": float(self.instruction), "fitted_on": self.fitted_on}

    @classmethod
    def from_json(cls, data: dict) -> "Thresholds":
        """Inverse of to_json. Unknown and missing keys are rejected; the result is validated."""
        if not isinstance(data, dict):
            raise ValueError("thresholds must be a JSON object")
        keys = {"final", "instruction", "fitted_on"}
        if set(data) != keys:
            raise ValueError(f"thresholds need exactly the keys {sorted(keys)}; got {sorted(data)}")
        made = cls(data["final"], data["instruction"], data["fitted_on"])
        made.validate()
        return made


# The second, unfitted variant (utt_gate). An unfitted 0.5 enters only through this name.
FIXED_HALF = Thresholds(0.5, 0.5, None)


def combine(answers: dict[str, float] | None, prior: float,
            thresholds: Thresholds) -> tuple[float, str, bool, bool]:
    """Section 5: (value, value_from, says_final, instruction_flag). Pure; every comparison is >=.

    `answers` is None when there was no utterance to ask about. A flagged utterance never
    raises the estimate; whether it would have said "final" is still returned for the record.
    """
    if answers is None:
        return prior, "no_utterance_round_prior", False, False
    signal = max(answers["u_final"], answers["u_walkaway"])
    says = signal >= thresholds.final
    if answers["u_instruction"] >= thresholds.instruction:
        return prior, "instruction_flagged_round_prior", says, True
    if says:
        return 1.0, "utterance_final", True, False
    return prior, "round_prior", False, False


class ModelReader:
    """Asks the four questions through any backend with predict(state, candidates, request)."""

    kind = "utterance_model_reader"

    def __init__(self, backend) -> None:
        self.backend = backend

    @property
    def is_model(self) -> bool:
        return bool(self.backend.is_model)

    def read(self, state: PublicState) -> tuple[dict[str, float], str, dict | None]:
        """(answers, request hash, response snapshot). A failure is raised unchanged; nothing is filled in."""
        request = build_utterance_request(state)
        reset_response(self.backend)
        predicted = self.backend.predict(state, [], request)
        answers = validate_predictions(predicted, QIDS)
        return answers, request_sha256(request), response_snapshot(self.backend)

    def describe(self) -> dict:
        return {"kind": self.kind, "is_model": self.is_model, "question_set": QUESTION_SET_VERSION,
                "backend": self.backend.describe()}


class LiteralReader:
    """CPU reference: the literal keyword test in the shape of a reader. Not a model."""

    kind, is_model = "utterance_literal_reader", False

    def read(self, state: PublicState) -> tuple[dict[str, float], None, None]:
        answers = dict.fromkeys(QIDS, 0.0)
        answers["u_final"] = 1.0 if says_final(state) else 0.0
        return answers, None, None

    def describe(self) -> dict:
        return {"kind": self.kind, "is_model": False,
                "note": "literal substring test on the latest supplier message; not language understanding"}


class LabelReader:
    """CPU stand-in: answers from the sentence bank's labels. A sentence outside the bank raises ValueError.

    Not a model. `label_text` is injected so this module never imports the bank.
    """

    kind, is_model = "utterance_bank_label_reader", False

    def __init__(self, label_text: Callable[[str], Any]) -> None:
        self._label_text = label_text

    def read(self, state: PublicState) -> tuple[dict[str, float], None, None]:
        labels = self._label_text(latest_supplier_utterance(state))
        answers = dict.fromkeys(QIDS, 0.0)
        if labels.core_class in ("final_limit", "explicit_final", "held_out_final"):
            answers["u_final"] = 1.0
        if labels.core_class == "final_walk":
            answers["u_walkaway"] = 1.0
        if labels.core_class in ("open", "negated_keyword") or labels.has_payment:
            answers["u_concession"] = 1.0
        if labels.has_instruction:
            answers["u_instruction"] = 1.0
        return answers, None, None

    def describe(self) -> dict:
        return {"kind": self.kind, "is_model": False,
                "note": "reads the labels of the synthetic sentence bank; an upper bound for reading, not a model"}


# ---- policy-path wrapper (track B) ---------------------------------------------------------------------

WRAPPER_NAME = "utterance_final_estimate"
NOT_ASKED_LAST_ROUND = "public_round_limit_not_asked"
# A name an existing label function would read as "the model's own final-offer estimate".
_RESERVED_NAMES = ("model",)


def _base_identity(base) -> dict:
    """The stable part of a base backend's description: no counter, no timing, no location."""
    described = base.describe()
    return {key: described.get(key) for key in ("kind", "is_model", "estimate_sources",
                                                "acceptance_platt_applied", "spec_sha256")}


class UtteranceFinalBackend:
    """Keeps the base backend's acceptance estimates and replaces only `final_offer` by the utterance reading.

    The policy and the session accept any object with kind, is_model, supports_final_offer, predict and
    describe; this wrapper keeps exactly that contract. `kind` is "composed_estimates", so the existing
    label function can never give it the plain real-model label. `thresholds` and `name` have no default:
    an unfitted 0.5 enters only as FIXED_HALF.

    Per call: the base answers the unchanged default request; on the last public round, or when the
    supplier said nothing, the reader is not asked; otherwise the reader's four answers go through
    `combine`. A reader failure, or a reader answer that is missing, not finite or outside [0, 1], is
    raised: the policy stops for human review and no value is filled in.
    """

    kind = "composed_estimates"
    supports_final_offer = True

    def __init__(self, base, reader, *, thresholds: Thresholds, name: str,
                 prior_table: tuple[float, ...] | None = None) -> None:
        if not isinstance(thresholds, Thresholds):
            raise TypeError("thresholds must be a Thresholds value; the unfitted pair is FIXED_HALF")
        thresholds.validate()
        if not isinstance(name, str) or not name or name in _RESERVED_NAMES:
            raise ValueError("name must be a non-empty configuration name other than 'model'")
        if prior_table is None:
            from negotiation_agent.policy import PolicyConfig   # lazy: no private type at module level
            prior_table = PolicyConfig().final_prior_by_round
        prior_table = tuple(prior_table)
        if not prior_table or any(not _is_probability(p) for p in prior_table):
            raise ValueError("prior_table must be a non-empty sequence of probabilities")
        self.base, self.reader = base, reader
        self.inner = base            # provenance.reset_response reaches the base through this name
        self.thresholds, self.name, self.prior_table = thresholds, name, prior_table
        self.is_model = bool(reader.is_model or base.is_model)
        self.last_response: dict[str, Any] | None = None
        self.reader_calls = 0        # utterance requests handed to the reader, failed ones included

    def _record(self, state: PublicState, prior: float, **values: Any) -> dict[str, Any]:
        record = {"status": "ok", "protocol_version": PROTOCOL_VERSION, "question_set": QUESTION_SET_VERSION,
                  "rule": self.name, "thresholds": self.thresholds.to_json(),
                  "latest_utterance": latest_supplier_utterance(state) or None, "request_sha256": None,
                  "answers": None, "signal_final": None, "says_final": False, "instruction_flag": False,
                  "round_prior": prior, "final_offer_value": None, "final_offer_value_from": None,
                  "base_final_offer": None,
                  "reader": {"kind": self.reader.kind, "is_model": bool(self.reader.is_model)},
                  "reader_response": None}
        record.update(values)
        return record

    def predict(self, state: PublicState, candidates: list, request: dict) -> dict[str, float]:
        self.last_response = None
        answered = self.base.predict(state, candidates, request)       # the unchanged request object
        snapshot = response_snapshot(self.base)     # before the reader runs: both may share one backend
        acceptance_ids = [qid for qid in request["questions"] if qid != FINAL_OFFER_QID]
        acceptance = validate_predictions(answered, acceptance_ids)
        base_final = answered.get(FINAL_OFFER_QID)
        prior = round_prior(state, self.prior_table)
        answers, digest, reader_response, says, flag = None, None, None, False, False
        if state.current_round >= state.max_rounds:
            # The policy uses 1.0 on the last public round whatever a backend says, so nothing is asked.
            value, value_from = prior, NOT_ASKED_LAST_ROUND
        else:
            if latest_supplier_utterance(state):
                self.reader_calls += 1
                try:
                    answers, digest, reader_response = self.reader.read(state)
                    # Whatever reader this is, a missing, NaN or out-of-range answer raises here and the
                    # policy stops for human review (section 5); combine() never sees it.
                    answers = validate_predictions(answers, QIDS)
                except BaseException as error:
                    self.last_response = {**(snapshot or {}), "utterance": self._record(
                        state, prior, status="error", error=f"{type(error).__name__}: {error}",
                        request_sha256=request_sha256(build_utterance_request(state)),
                        base_final_offer=base_final,
                        reader_response=response_snapshot(getattr(self.reader, "backend", None)))}
                    raise
            value, value_from, says, flag = combine(answers, prior, self.thresholds)
        out = dict(acceptance)
        if FINAL_OFFER_QID in request["questions"]:
            out[FINAL_OFFER_QID] = value
        out = validate_predictions(out, list(out))
        self.last_response = {**(snapshot or {}), "utterance": self._record(
            state, prior, request_sha256=digest, answers=dict(answers) if answers is not None else None,
            signal_final=max(answers["u_final"], answers["u_walkaway"]) if answers is not None else None,
            says_final=says, instruction_flag=flag, final_offer_value=value, final_offer_value_from=value_from,
            base_final_offer=base_final, reader_response=json_safe(reader_response))}
        return out

    def describe(self) -> dict[str, Any]:
        base = self.base.describe()
        acceptance = (base.get("estimate_sources") or {}).get("acceptance") or self.base.kind
        used = ["acceptance"] if "acceptance" in (base.get("model_estimates_used") or ()) or (
            "model_estimates_used" not in base and self.base.is_model) else []
        if self.reader.is_model:
            used.append("utterance_reading")
        identity = {"question_set": QUESTION_SET_VERSION, "thresholds": self.thresholds.to_json(),
                    "name": self.name, "prior_table": list(self.prior_table), "base": _base_identity(self.base)}
        text = json.dumps(json_safe(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        return {"kind": self.kind, "wrapper": WRAPPER_NAME, "is_model": self.is_model,
                "estimate_sources": {"acceptance": acceptance, "final_offer": self.name},
                "acceptance_platt_applied": bool(base.get("acceptance_platt_applied")),
                "model_estimates_used": used,
                "spec_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "question_set": QUESTION_SET_VERSION, "thresholds": self.thresholds.to_json(),
                "prior_table": list(self.prior_table), "reader": self.reader.describe(), "inner": base}

    def close(self) -> None:
        closed: list[Any] = []
        for target in (self.base, getattr(self.reader, "backend", None)):
            close = getattr(target, "close", None)
            if target is not None and callable(close) and not any(target is other for other in closed):
                closed.append(target)
                close()
