"""Choose, per estimate, where the value the policy uses comes from.

Verified on CPU with the CPU reference backends and test doubles; those tests never use a GPU
and are not evidence about the real model. Executed on a GPU since: the tune-2026-11-18.v1
compositions in the 2026-11-18 Colab T4 tuning run (selection stage only; see
negotiation/docs/UTTERANCE_AGENT_DESIGN.md section 11), and the final-reader composition once
on a local RTX 5060 Laptop on 2026-11-19 (negotiation/docs/PLAN.md section 16.1).

A composition is a deterministic CPU transform applied AFTER the backend answered the
unchanged public request. The request itself is never edited: the `final_offer` question
stays in it even when a configuration does not use its answer, so a stored answer for one
request serves every composition.

Sources:
- acceptance: "model" (the inner backend's value, optionally Platt-mapped with a map fitted
  on the calibration split), "rule" (HeuristicBackend, a CPU reference) or "constant".
- final offer: "model", "policy_prior" (the value is omitted, so the policy uses its own
  table and records `policy_prior`), "keyword" (KeywordBackend's literal test, else the
  public-round prior), "shrink" (w * model + (1 - w) * public-round prior) or
  "keyword_else_shrink".

Everything here reads the public state and the candidates only. A constant, rule or keyword
value is never reported as a plain model answer: every response carries a `composition`
block naming the source of each estimate.

Added 2026-11-19 for the final-offer reader (protocol `final-reader-2026-11-19.v1`, not part of
`tune-2026-11-18.v1`, whose candidate family is unchanged). Written and verified on CPU, then
executed once on a local GPU the same day (PLAN.md section 16.1):
- final source "model_threshold": the model's final-offer answer counts as "the supplier says this is
  final" only at or above `final_threshold`; below it the public-round prior is used.
- `final_guard`: a supplier whose earlier offer was read as final and who then made another offer has
  shown the claim was not true; its later claims are not believed (the round prior is used instead).
  For "keyword" the earlier offers are re-read with the same literal test; for "model_threshold" the
  earlier readings are the answers this backend received for those exact earlier public states.
- `request_scope` "final_offer_only": the policy sends the state with the final-offer question alone
  (no acceptance question), so the model is asked one question per decision. Allowed only when no
  acceptance value comes from the model. This is the one case where the request differs from the
  default; the default request and its hash are unchanged.
New keys are written by `to_json` only when they differ from their defaults, so every earlier
configuration keeps its digest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from typing import Any

from .backends import HeuristicBackend, KeywordBackend, _sigmoid
from .contracts import Candidate, PublicState
from .provenance import json_safe, request_sha256, response_snapshot, validate_predictions
from .requests import FINAL_OFFER_QID, REQUEST_SCOPES, build_final_only_request

ACCEPTANCE_SOURCES = ("model", "rule", "constant")
FINAL_SOURCES = ("model", "policy_prior", "keyword", "shrink", "keyword_else_shrink")   # tune-2026-11-18.v1
LATER_FINAL_SOURCES = ("model_threshold",)        # final-reader-2026-11-19.v1; not a tune v1 candidate source
ALL_FINAL_SOURCES = FINAL_SOURCES + LATER_FINAL_SOURCES
GUARDED_FINAL_SOURCES = ("keyword", "model_threshold")
TWIN_ACCEPTANCE_CONSTANT = 0.15

_FINAL_NEEDS_MODEL = ("model", "shrink", "keyword_else_shrink", "model_threshold")
_FINAL_NEEDS_WEIGHT = ("shrink", "keyword_else_shrink")
_PLATT_SPLIT = "calibration"
_PLATT_CLAMP = 1e-4   # the same clamp as CalibratedBackend


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class EstimateSpec:
    """Where the acceptance estimate and the final-offer estimate come from.

    Construction does not validate, so a caller can hold a template whose shrink weight or
    Platt map is fitted later. `validate()` must pass before the spec is used;
    `ComposedBackend` calls it.
    """

    acceptance_source: str = "model"
    acceptance_constant: float | None = None     # required iff acceptance_source == "constant"
    acceptance_platt: dict | None = None         # {"a", "b", "fitted_on_split": "calibration", ...}; model only
    final_source: str = "model"
    final_shrink_weight: float | None = None     # required iff final_source in ("shrink", "keyword_else_shrink")
    # Added 2026-11-19 (final-reader-2026-11-19.v1). Defaults leave every earlier spec and digest unchanged.
    final_threshold: float | None = None         # required iff final_source == "model_threshold"
    final_guard: bool = False                    # only with final_source "keyword" or "model_threshold"
    request_scope: str = "full"                  # "final_offer_only": no acceptance question is sent

    def __hash__(self) -> int:   # the Platt map is a dict; hash the canonical content instead
        return hash(_canonical(json_safe(self.to_json())))

    def validate(self) -> None:
        """Raise ValueError on any inconsistent combination."""
        if self.acceptance_source not in ACCEPTANCE_SOURCES:
            raise ValueError(f"acceptance_source must be one of {ACCEPTANCE_SOURCES}")
        if self.final_source not in ALL_FINAL_SOURCES:
            raise ValueError(f"final_source must be one of {ALL_FINAL_SOURCES}")
        if self.final_source == "model_threshold":
            if not _is_number(self.final_threshold) or not 0.0 < self.final_threshold <= 1.0:
                raise ValueError("final_source 'model_threshold' needs final_threshold in (0, 1]")
        elif self.final_threshold is not None:
            raise ValueError("final_threshold is only allowed with final_source 'model_threshold'")
        if not isinstance(self.final_guard, bool):
            raise ValueError("final_guard must be true or false")
        if self.final_guard and self.final_source not in GUARDED_FINAL_SOURCES:
            raise ValueError(f"final_guard is only allowed with final_source in {GUARDED_FINAL_SOURCES}")
        if self.request_scope not in REQUEST_SCOPES:
            raise ValueError(f"request_scope must be one of {REQUEST_SCOPES}")
        if self.request_scope == "final_offer_only":
            if self.acceptance_source == "model":
                raise ValueError("request_scope 'final_offer_only' sends no acceptance question, so the "
                                 "acceptance value cannot come from the model")
            if self.final_source not in _FINAL_NEEDS_MODEL:
                raise ValueError("request_scope 'final_offer_only' is only for a final-offer value read from the model")
        if self.final_guard and self.final_source == "model_threshold" and self.request_scope != "final_offer_only":
            raise ValueError("final_guard with 'model_threshold' needs request_scope 'final_offer_only': an earlier "
                             "reading is found again by its exact public request, which only that scope can rebuild")
        if self.acceptance_source == "constant":
            if not _is_number(self.acceptance_constant) or not 0.0 <= self.acceptance_constant <= 1.0:
                raise ValueError("acceptance_source 'constant' needs acceptance_constant in [0, 1]")
        elif self.acceptance_constant is not None:
            raise ValueError("acceptance_constant is only allowed with acceptance_source 'constant'")
        if self.acceptance_platt is not None:
            if self.acceptance_source != "model":
                raise ValueError("acceptance_platt is only allowed with acceptance_source 'model'")
            platt = self.acceptance_platt
            if not isinstance(platt, dict) or not _is_number(platt.get("a")) or not _is_number(platt.get("b")):
                raise ValueError("acceptance_platt needs finite numbers 'a' and 'b'")
            if platt.get("fitted_on_split") != _PLATT_SPLIT:
                raise ValueError("acceptance_platt must have been fitted on the calibration split")
            try:
                _canonical(platt)
            except (TypeError, ValueError) as error:
                raise ValueError("acceptance_platt must be plain JSON data") from error
        if self.final_source in _FINAL_NEEDS_WEIGHT:
            if not _is_number(self.final_shrink_weight) or not 0.0 <= self.final_shrink_weight <= 1.0:
                raise ValueError(f"final_source '{self.final_source}' needs final_shrink_weight in [0, 1]")
        elif self.final_shrink_weight is not None:
            raise ValueError("final_shrink_weight is only allowed with final_source 'shrink' or 'keyword_else_shrink'")

    def uses_model(self) -> bool:
        return self.acceptance_source == "model" or self.final_source in _FINAL_NEEDS_MODEL

    def model_estimates_used(self) -> list[str]:
        used = []
        if self.acceptance_source == "model":
            used.append("acceptance")
        if self.final_source in _FINAL_NEEDS_MODEL:
            used.append("final_offer")
        return used

    def estimate_sources(self) -> dict[str, str]:
        return {"acceptance": self.acceptance_source, "final_offer": self.final_source}

    def twin(self) -> "EstimateSpec":
        """The model-free twin: what this configuration becomes when no model value is used."""
        acceptance, constant = self.acceptance_source, self.acceptance_constant
        if acceptance == "model":
            acceptance, constant = "constant", TWIN_ACCEPTANCE_CONSTANT
        final = {"model": "policy_prior", "shrink": "policy_prior", "keyword_else_shrink": "keyword",
                 "model_threshold": "keyword"}.get(self.final_source, self.final_source)
        # The literal keyword stands in for the model's reading; a guard stays a guard. The twin sends the
        # default request because it asks the model nothing.
        return EstimateSpec(acceptance_source=acceptance, acceptance_constant=constant, acceptance_platt=None,
                            final_source=final, final_shrink_weight=None,
                            final_guard=bool(self.final_guard and final in GUARDED_FINAL_SOURCES))

    def to_json(self) -> dict:
        number = lambda value: float(value) if _is_number(value) else value  # noqa: E731 - 1 and 1.0 hash alike
        data = {
            "acceptance_source": self.acceptance_source,
            "acceptance_constant": number(self.acceptance_constant),
            "acceptance_platt": copy.deepcopy(self.acceptance_platt),
            "final_source": self.final_source,
            "final_shrink_weight": number(self.final_shrink_weight),
        }
        # Written only when set, so a spec without them has exactly the earlier form and digest.
        if self.final_threshold is not None:
            data["final_threshold"] = number(self.final_threshold)
        if self.final_guard is not False:
            data["final_guard"] = self.final_guard
        if self.request_scope != "full":
            data["request_scope"] = self.request_scope
        return data

    @classmethod
    def from_json(cls, data: dict) -> "EstimateSpec":
        """Inverse of to_json. Unknown keys are rejected; missing keys take the defaults.

        This does not check that the combination is consistent: call `validate()`.
        """
        if not isinstance(data, dict):
            raise ValueError("an estimate spec must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown estimate spec keys: {unknown}")
        values = {key: copy.deepcopy(value) for key, value in data.items()}
        for key in ("acceptance_source", "final_source", "request_scope"):
            if key in values and not isinstance(values[key], str):
                raise ValueError(f"{key} must be a string")
        if "final_guard" in values and not isinstance(values["final_guard"], bool):
            raise ValueError("final_guard must be true or false")
        for key in ("acceptance_constant", "final_shrink_weight", "final_threshold"):
            if values.get(key) is not None:
                if not _is_number(values[key]):
                    raise ValueError(f"{key} must be a finite number or null")
                values[key] = float(values[key])
        if values.get("acceptance_platt") is not None and not isinstance(values["acceptance_platt"], dict):
            raise ValueError("acceptance_platt must be an object or null")
        return cls(**values)

    def digest(self) -> str:
        """sha256 of the canonical to_json()."""
        return hashlib.sha256(_canonical(self.to_json()).encode("utf-8")).hexdigest()


def round_prior(state: PublicState, prior_table) -> float:
    """The public-round prior that the standing offer is final, exactly as the policy reads its table."""
    return 1.0 if state.current_round >= state.max_rounds else prior_table[min(state.current_round, len(prior_table)) - 1]


def says_final(state: PublicState) -> bool:
    """KeywordBackend's literal test on the latest supplier offer's message. Not language understanding."""
    latest = state.last_offer("supplier")
    return bool(latest and any(word in latest.message for word in KeywordBackend.FINAL_KEYWORDS))


def earlier_supplier_states(state: PublicState):
    """(offer, public state as the buyer saw it then) for every supplier offer before the latest one."""
    indexes = [i for i, offer in enumerate(state.history) if offer.made_by == "supplier"]
    for i in indexes[:-1]:
        offer = state.history[i]
        yield offer, replace(state, history=state.history[: i + 1], current_round=offer.round_index)


def keyword_claims_falsified(state: PublicState) -> int:
    """Earlier supplier offers that the literal test reads as final. Each was followed by another offer."""
    return sum(1 for _, earlier in earlier_supplier_states(state) if says_final(earlier))


def shrink(p_model: float, prior: float, weight: float) -> float:
    """w * p_model + (1 - w) * prior, in exactly this order so a fit and a run agree bit for bit."""
    return weight * p_model + (1 - weight) * prior


def platt_map(p: float, a: float, b: float) -> float:
    """sigmoid(a * logit(p) + b) with p clamped to [1e-4, 1 - 1e-4]; identical to CalibratedBackend."""
    p = min(1 - _PLATT_CLAMP, max(_PLATT_CLAMP, float(p)))
    return _sigmoid(a * math.log(p / (1 - p)) + b)


class ComposedBackend:
    """Wraps a backend and replaces chosen estimates by CPU values, leaving the request unchanged.

    `kind` is always "composed_estimates": a hybrid never takes the inner backend's kind.
    `is_model` is true only when the spec uses a model value AND the inner backend is a model,
    so a composition over a CPU stand-in is not labelled as a model estimate.
    """

    kind = "composed_estimates"

    def __init__(self, inner, spec: EstimateSpec, prior_table: tuple[float, ...] | None = None) -> None:
        spec.validate()
        if prior_table is None:
            from .policy import PolicyConfig   # lazy, like KeywordBackend: no private type at module level
            prior_table = PolicyConfig().final_prior_by_round
        prior_table = tuple(prior_table)
        if not prior_table or any(not _is_number(p) or not 0.0 <= p <= 1.0 for p in prior_table):
            raise ValueError("prior_table must be a non-empty sequence of probabilities")
        self.uses_inner = spec.uses_model()
        if self.uses_inner:
            if inner is None:
                raise ValueError("this estimate spec uses the model; an inner backend is required")
            inner_final = bool(getattr(inner, "supports_final_offer", inner.is_model))
            if spec.final_source in _FINAL_NEEDS_MODEL and not inner_final:
                raise ValueError(f"final_source '{spec.final_source}' needs a final-offer estimate, "
                                 "which the inner backend does not supply")
            self._inner_final = inner_final
        else:
            self._inner_final = False
        self.inner, self.spec, self.prior_table = inner, spec, prior_table
        self.is_model = bool(self.uses_inner and inner.is_model)
        self.supports_final_offer = spec.final_source != "policy_prior"
        self.last_response: dict[str, Any] | None = None
        self._rule = HeuristicBackend() if spec.acceptance_source == "rule" else None
        self._spec_json, self._spec_sha256 = spec.to_json(), spec.digest()
        # The policy reads this to decide which request it sends (requests.REQUEST_SCOPES).
        self.request_scope = spec.request_scope
        # model_threshold: the final-offer answer received for each exact final-only request, so a later
        # decision can tell whether an earlier offer of the same negotiation was read as final.
        self._readings: dict[str, float] = {}

    def _composition(self, state: PublicState, **values: Any) -> dict[str, Any]:
        return {"spec": copy.deepcopy(self._spec_json), "spec_sha256": self._spec_sha256,
                "estimate_sources": self.spec.estimate_sources(),
                "acceptance_platt_applied": self.spec.acceptance_platt is not None,
                "model_estimates_used": self.spec.model_estimates_used(),
                "round_prior": round_prior(state, self.prior_table), "says_final": says_final(state), **values}

    def _model_claims_falsified(self, state: PublicState) -> tuple[int, int]:
        """(earlier offers read as final, earlier offers with no recorded reading). Never asks the model."""
        claims = unknown = 0
        for _, earlier in earlier_supplier_states(state):
            reading = self._readings.get(request_sha256(build_final_only_request(earlier)))
            if reading is None:
                unknown += 1
            elif reading >= float(self.spec.final_threshold):
                claims += 1
        return claims, unknown

    def predict(self, state: PublicState, candidates: list[Candidate], request: dict) -> dict[str, float]:
        self.last_response = None
        spec, questions = self.spec, list(request["questions"])
        if self.request_scope == "final_offer_only":
            if questions != [FINAL_OFFER_QID]:
                raise ValueError("a final_offer_only composition must receive the final-offer question alone")
            # No acceptance question was sent: the acceptance value of each candidate comes from this
            # composition (constant or rule), never from the model.
            acceptance_ids = [c.candidate_id for c in candidates]
        else:
            acceptance_ids = [qid for qid in questions if qid != FINAL_OFFER_QID]
        snapshot, raw = None, None
        if self.uses_inner:
            try:
                answered = self.inner.predict(state, candidates, request)   # the unchanged request
            finally:
                snapshot = response_snapshot(self.inner)
                self.last_response = snapshot    # a failed call keeps the inner evidence, never a composed value
            # Every answer the inner backend owes is checked, also one this composition does not use.
            required = list(questions) if self._inner_final else \
                acceptance_ids + ([FINAL_OFFER_QID] if FINAL_OFFER_QID in answered and FINAL_OFFER_QID in questions else [])
            try:
                raw = validate_predictions(answered, required)
            except Exception:
                self.last_response = {**(snapshot or {}), "composition": self._composition(
                    state, status="invalid_model_predictions",
                    raw_model_predictions=json_safe(dict(answered)) if isinstance(answered, dict) else None,
                    final_offer_value_from=None, predictions=None)}
                raise

        if spec.acceptance_source == "model":
            if spec.acceptance_platt is None:
                acceptance = {qid: raw[qid] for qid in acceptance_ids}
            else:
                a, b = float(spec.acceptance_platt["a"]), float(spec.acceptance_platt["b"])
                acceptance = {qid: platt_map(raw[qid], a, b) for qid in acceptance_ids}
        elif spec.acceptance_source == "rule":
            ruled = self._rule.predict(state, candidates, request)
            acceptance = {qid: ruled[qid] for qid in acceptance_ids}
        else:
            acceptance = {qid: float(spec.acceptance_constant) for qid in acceptance_ids}

        out = dict(acceptance)
        value_from = "omitted_policy_uses_its_prior"
        extra: dict[str, Any] = {}
        if FINAL_OFFER_QID not in questions:
            value_from = "not_asked"
        elif spec.final_source == "model":
            out[FINAL_OFFER_QID], value_from = raw[FINAL_OFFER_QID], "model"
        elif spec.final_source == "model_threshold":
            prior, reading = round_prior(state, self.prior_table), float(raw[FINAL_OFFER_QID])
            if self.request_scope == "final_offer_only":
                self._readings[request_sha256(request)] = reading
            claimed = reading >= float(spec.final_threshold)
            claims, unknown = self._model_claims_falsified(state) if spec.final_guard else (0, 0)
            disbelieved = bool(claimed and spec.final_guard and claims > 0)
            if claimed and not disbelieved:
                out[FINAL_OFFER_QID], value_from = 1.0, "model_threshold_match"
            else:
                out[FINAL_OFFER_QID] = prior
                value_from = "guard_claim_falsified" if disbelieved else "round_prior"
            extra = {"final_threshold": float(spec.final_threshold),
                     "guard": {"enabled": spec.final_guard, "earlier_claims": claims,
                               "earlier_readings_unknown": unknown, "claim_disbelieved": disbelieved}}
        elif spec.final_source != "policy_prior":
            prior = round_prior(state, self.prior_table)
            if spec.final_source in ("keyword", "keyword_else_shrink") and says_final(state):
                claims = keyword_claims_falsified(state) if spec.final_guard else 0
                if claims:
                    out[FINAL_OFFER_QID], value_from = prior, "guard_claim_falsified"
                else:
                    out[FINAL_OFFER_QID], value_from = 1.0, "keyword_match"
                if spec.final_guard:
                    extra = {"guard": {"enabled": True, "earlier_claims": claims, "earlier_readings_unknown": 0,
                                       "claim_disbelieved": bool(claims)}}
            elif spec.final_source == "keyword":
                out[FINAL_OFFER_QID], value_from = prior, "round_prior"
            else:
                out[FINAL_OFFER_QID] = shrink(raw[FINAL_OFFER_QID], prior, float(spec.final_shrink_weight))
                value_from = "shrink"
        out = validate_predictions(out, list(out))
        self.last_response = {**(snapshot or {}), "composition": self._composition(
            state, status="ok", raw_model_predictions=dict(raw) if raw is not None else None,
            final_offer_value_from=value_from, predictions=dict(out), **extra)}
        return out

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "is_model": self.is_model,
                "estimate_sources": self.spec.estimate_sources(),
                "acceptance_platt_applied": self.spec.acceptance_platt is not None,
                "spec": copy.deepcopy(self._spec_json), "spec_sha256": self._spec_sha256,
                "prior_table": list(self.prior_table),
                "model_estimates_used": self.spec.model_estimates_used(),
                "inner": self.inner.describe() if self.uses_inner else None}

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()
