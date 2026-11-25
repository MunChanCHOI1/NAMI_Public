"""Acceptance predictors. Every backend receives public information only.

Kinds, which every result file must keep apart:
- mock_constant / rule_heuristic: CPU references. They are NOT model evidence.
- zoro_native: the repository's native Transformers adapter with the pinned real
  Zoro-1 weights and original pointer head. Not the official SGLang path.
- official_decide_http: the pinned decision server reached over POST /v1/decide.
- systemone_http: the separate legacy POST /v1/systemone contract.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from .contracts import Candidate, PublicState
from .provenance import backend_execution_config, json_safe, response_snapshot, validate_predictions
from .requests import FINAL_OFFER_QID


class AcceptanceBackend(Protocol):
    kind: str
    is_model: bool

    def predict(self, state: PublicState, candidates: list[Candidate], request: dict) -> dict[str, float]:
        """Return P(supplier accepts unchanged) per question id of `request`."""

    def describe(self) -> dict[str, Any]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


class ConstantBackend:
    """Same value for every candidate. Exists to show that the gates reject it."""

    kind = "mock_constant"
    is_model = False
    supports_final_offer = True

    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def predict(self, state, candidates, request):
        return {qid: self.value for qid in request["questions"]}

    def describe(self):
        return {"kind": self.kind, "is_model": False, "value": self.value}


class HeuristicBackend:
    """Model-free estimate from the public history: the baseline a model has to beat.

    Converts every offer to a supplier-side price equivalent with fixed, public
    assumptions (not the hidden parameters), estimates the supplier's concession per
    round from its own offers, and scores how far the candidate asks beyond that.
    """

    kind = "rule_heuristic"
    is_model = False
    supports_final_offer = False  # the policy explicitly supplies its separate round prior
    ASSUMED_SUPPLIER_RATE = 0.10        # annual financing rate assumed for any supplier
    ASSUMED_EXPEDITE_PER_DAY = 15_000   # KRW per day faster than the first quoted lead time
    DEFAULT_CONCESSION = 0.02           # used when only one supplier offer is visible
    SCALE = 0.012

    def _equivalent(self, terms, quantity, base_delivery) -> float:
        amount = terms.unit_price * quantity
        financing = amount * self.ASSUMED_SUPPLIER_RATE * terms.payment_days / 365
        expedite = self.ASSUMED_EXPEDITE_PER_DAY * max(0, base_delivery - terms.delivery_days)
        return amount - financing - expedite

    def predict(self, state, candidates, request):
        supplier_offers = [o for o in state.history if o.made_by == "supplier"]
        base_delivery = supplier_offers[0].terms.delivery_days
        values = [self._equivalent(o.terms, state.quantity, base_delivery) for o in supplier_offers]
        last = values[-1]
        if len(values) >= 2:
            steps = [(a - b) / a for a, b in zip(values, values[1:])]
            concession = max(0.0, sum(steps) / len(steps))
        else:
            concession = self.DEFAULT_CONCESSION
        out = {}
        for candidate in candidates:
            gap = (last - self._equivalent(candidate.terms, state.quantity, base_delivery)) / last
            out[candidate.candidate_id] = _sigmoid((concession - gap) / self.SCALE)
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": False,
                "assumed_supplier_rate": self.ASSUMED_SUPPLIER_RATE,
                "assumed_expedite_per_day": self.ASSUMED_EXPEDITE_PER_DAY}


class KeywordBackend(HeuristicBackend):
    """A deliberately literal public-message baseline; not language understanding or calibrated probability.

    Acceptance estimates are unchanged. Only the latest supplier message can set the
    final-offer estimate; missing keywords use the policy's default public-round prior.
    """

    kind = "rule_keyword"
    supports_final_offer = True
    FINAL_KEYWORDS = ("최종안", "마지막 제안")

    def predict(self, state, candidates, request):
        out = super().predict(state, candidates, request)
        if FINAL_OFFER_QID in request["questions"]:
            from .policy import PolicyConfig
            latest = state.last_offer("supplier")
            priors = PolicyConfig().final_prior_by_round
            prior = 1.0 if state.current_round >= state.max_rounds else priors[min(state.current_round - 1, len(priors) - 1)]
            out[FINAL_OFFER_QID] = 1.0 if latest and any(word in latest.message for word in self.FINAL_KEYWORDS) else prior
        return out

    def describe(self):
        return {**super().describe(), "final_offer_keywords": list(self.FINAL_KEYWORDS),
                "keyword_final_value": 1.0, "no_keyword_source": "default_policy_public_round_prior",
                "note": "literal substring baseline; does not handle negation or claims of authorization"}


class ZoroNativeBackend:
    """Real pinned Zoro-1 through sessionguard's native adapter. Needs the GPU environment.

    Not executed in the authoring environment. One full forward per question.
    """

    kind = "zoro_native"
    is_model = True
    supports_final_offer = True

    def __init__(self, original_checkpoint: str, exported_checkpoint: str, official_source_dir: str,
                 device: str = "cuda:0", dtype: str = "auto", max_input_tokens: int = 2048,
                 decision_temperature: float | None = None) -> None:
        from sessionguard.backends.zoro import DecisionBackend  # lazy: imports torch on construction

        self.configuration = {
            "original_checkpoint": str(Path(original_checkpoint).expanduser().resolve()),
            "exported_checkpoint": str(Path(exported_checkpoint).expanduser().resolve()),
            "official_source_dir": str(Path(official_source_dir).expanduser().resolve()), "device": device, "dtype": dtype,
            "max_input_tokens": max_input_tokens, "decision_temperature": decision_temperature,
        }
        self.last_response: dict[str, Any] | None = None
        self._backend = DecisionBackend(
            original_checkpoint=original_checkpoint, exported_checkpoint=exported_checkpoint,
            official_source_dir=official_source_dir, device=device, dtype=dtype,
            max_input_tokens=max_input_tokens, decision_temperature=decision_temperature,
        )
        self.last_metadata: dict[str, Any] = {}

    def decide_typed(self, request):
        """Return the official typed response, including choice questions, without inventing acceptance values."""
        self.last_response = None
        self.last_metadata = {}
        response = self._backend.decide(request)
        self.last_response = json_safe(response)
        self.last_metadata = self.last_response["metadata"]
        return self.last_response

    def predict(self, state, candidates, request):
        response = self.decide_typed(request)
        return {qid: float(p["true"]) for qid, p in response["raw_probabilities"].items()}

    def describe(self):
        keep = ("model_id", "model_revision", "source_revision", "backend", "dtype", "device", "gpu_name",
                "decision_temperature", "temperature_source", "official_sglang_parity_verified",
                "head_loaded", "mock", "gpu_execution_performed", "runtime_versions",
                "peak_cuda_memory_allocated_bytes")
        return {"kind": self.kind, "is_model": True,
                "configuration": dict(self.configuration),
                **{k: self.last_metadata.get(k) for k in keep},
                "last_response_metadata": json_safe(self.last_metadata)}


class ZoroHttpBackend:
    """Reuse the checked official client; server provenance must be recorded separately.

    `reasoning=True` asks the server to reason before deciding. As of 2026-11-18 that option
    was exercised from this project against an injected transport only: no real server, no GPU.
    """

    kind = "official_decide_http"
    is_model = True
    supports_final_offer = True

    def __init__(self, url: str, official_source_dir: str, decision_temperature: float | None = None,
                 timeout: float = 180.0, reasoning: bool = False) -> None:
        from sessionguard.backends.zoro_http import HttpDecisionBackend

        # With reasoning the server generates text before deciding: each answer is one
        # sample of a sampled generation, and the client reserves room for that text.
        self.reasoning = bool(reasoning)
        self._backend = HttpDecisionBackend(url, official_source_dir,
                                            reasoning=self.reasoning,
                                            decision_temperature=decision_temperature,
                                            timeout_seconds=timeout)
        self.configuration = backend_execution_config(SimpleNamespace(
            backend="zoro-http", url=url, official_source_dir=official_source_dir,
            decision_temperature=decision_temperature, request_timeout=timeout,
            reasoning=self.reasoning))
        self.last_response = None

    def decide_typed(self, request):
        self.last_response = None
        self.last_response = json_safe(self._backend.decide(request))
        return self.last_response

    def predict(self, state, candidates, request):
        response = self.decide_typed(request)
        return {qid: float(response["answers"][qid]["noul"]) for qid in request["questions"]}

    def describe(self):
        described = {"kind": self.kind, "is_model": True, "configuration": dict(self.configuration),
                     "last_response_metadata": (self.last_response or {}).get("metadata", {}),
                     "server_identity_verified_by_client": False,
                     "notice": "HTTP identity is unverified; record the actual server command, checkpoint and SGLang version."}
        if self.reasoning:   # added only for a reasoning run; the default description is unchanged
            described["reasoning"] = {
                "requested": True,
                "notice": "Reason-then-decide: every answer follows one sampled generation, so this run is a "
                          "single sample and is a separate configuration from the default run.",
            }
        return described

    def close(self):
        self._backend.close()


class SystemOneBackend:
    """An official decide server (serve.py --mode decide) over POST /v1/systemone.

    Authored from the upstream README; verified only against a local stub server.
    The server rounds probabilities to four decimals.
    """

    kind = "systemone_http"
    is_model = True
    supports_final_offer = True

    def __init__(self, url: str, model: str = "zoro-1", timeout: float = 120.0) -> None:
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.last_response_meta: dict[str, Any] = {}
        self.last_response: dict[str, Any] | None = None

    def predict(self, state, candidates, request):
        self.last_response = None
        self.last_response_meta = {}
        body = json.dumps({"model": self.model, **request}, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}, method="POST"
        )
        started = time.perf_counter()
        with urllib.request.urlopen(http_request, timeout=self.timeout) as stream:  # noqa: S310 - operator-supplied URL
            payload = json.loads(stream.read().decode("utf-8"))
        self.last_response = json_safe(payload)
        answers = payload["answers"]
        missing = set(request["questions"]) - set(answers)
        if missing:
            raise ValueError(f"server returned no answer for {sorted(missing)}")
        self.last_response_meta = {
            "served_model": payload.get("model"), "usage": payload.get("usage"),
            "timing_ms": payload.get("timing_ms"),
            "client_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        return {qid: float(answers[qid]["noul"]) for qid in request["questions"]}

    def describe(self):
        settings = backend_execution_config(SimpleNamespace(backend="systemone", url=self.url, model=self.model,
                                                             request_timeout=self.timeout))
        return {"kind": self.kind, "is_model": True, "configuration": settings,
                "requested_model": self.model, **self.last_response_meta}


class CalibratedBackend:
    """Applies a Platt map fitted on the calibration split: sigmoid(a * logit(p) + b)."""

    def __init__(self, inner, a: float, b: float, source: str = "") -> None:
        self.inner, self.a, self.b, self.source = inner, float(a), float(b), source
        self.kind, self.is_model = inner.kind, inner.is_model
        self.supports_final_offer = getattr(inner, "supports_final_offer", inner.is_model)
        self.last_response: dict[str, Any] | None = None

    def predict(self, state, candidates, request):
        self.last_response = None
        try:
            predicted = self.inner.predict(state, candidates, request)
        finally:
            self.last_response = response_snapshot(self.inner)
        predicted = validate_predictions(predicted, predicted)
        out = {}
        for qid, p in predicted.items():
            if qid == FINAL_OFFER_QID:
                out[qid] = p  # acceptance calibration was not fitted on this different event
                continue
            p = min(1 - 1e-4, max(1e-4, float(p)))
            out[qid] = _sigmoid(self.a * math.log(p / (1 - p)) + self.b)
        self.last_response = {**(self.last_response or {}),
                              "acceptance_calibration": {"a": self.a, "b": self.b, "source": self.source},
                              "uncalibrated_predictions": predicted, "predictions": out}
        return out

    def describe(self):
        return {**self.inner.describe(), "calibration": {"method": "platt", "a": self.a, "b": self.b,
                                                         "fitted_on": self.source,
                                                         "applies_to": "acceptance_questions_only"}}
