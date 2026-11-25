"""HTTP client for the official Zoro-1 decision server (``POST /v1/decide``).

The pinned Trida source documents ``serve_decide.py``: one in-process SGLang
engine that answers typed decisions and, with ``reasoning=true``, writes its
reasoning before deciding. This client only talks to such a server. It does not
load weights, and it cannot prove which checkpoint or engine the server runs:
every response therefore records the served model id as *reported by the
endpoint* and ``server_identity_verified_by_client: False``.

Authoring validation: CPU tests against a fake transport only. No real server,
GPU, or SGLang engine was contacted while this module was written.

The official codec silently cuts an oversized state on the server. This client
rejects such a request first, using the pinned codec's own renderer and the
fact that a tokenizer never produces more tokens than UTF-8 bytes.
Generated reasoning is checked again after the response arrives; an unsafe
expanded state is rejected before any session action, not before server encoding.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from sessionguard.contracts import InputOverflowError, ordered_json_sha256

from .zoro import (
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_RELATIVE_PATHS,
    SOURCE_SHA256,
    TRIDA_REVISION,
    _module_from_file,
    sha256_file,
)

# with_reasoning() appends up to max_reasoning_tokens=768 generated tokens to the
# state. Re-tokenizing that text is not guaranteed to give the same count, so the
# allowance is deliberately generous. It is a margin, not a proof.
REASONING_STATE_ALLOWANCE = 1600


def load_official_codec(source_dir: str | Path):
    """Import only the hash-verified request codec; it needs neither torch nor SGLang."""
    source_dir = Path(source_dir)
    name = "decide_request.py"
    path = source_dir / name
    if not path.is_file():
        path = source_dir / OFFICIAL_RELATIVE_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned official codec not found under {source_dir}. Run scripts/fetch_official_sources.py first."
        )
    if sha256_file(path) != SOURCE_SHA256[name]:
        raise ValueError(f"{name} does not match pinned Trida source {TRIDA_REVISION}.")
    return _module_from_file("_sessionguard_zoro_http_codec", path)


def endpoint_origin(endpoint: str) -> str:
    """Validate the endpoint and return scheme://host[:port] without credentials or query."""
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("--endpoint must look like http://host:port")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("--endpoint must not contain credentials, a query, or a fragment.")
    if parts.path not in {"", "/"}:
        raise ValueError("--endpoint is the server root; /v1/decide is appended by the client.")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")


class HttpDecisionBackend:
    """Sequential client of one official decision endpoint; no local model."""

    name = "zoro-http"

    def __init__(
        self,
        endpoint: str,
        official_source_dir: str | Path,
        *,
        reasoning: bool = False,
        decision_temperature: float | None = None,
        timeout_seconds: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.origin = endpoint_origin(endpoint)
        self.codec = load_official_codec(official_source_dir)
        self.reasoning = bool(reasoning)
        if decision_temperature is not None and (
            not math.isfinite(decision_temperature) or decision_temperature <= 0
        ):
            raise ValueError("decision_temperature must be finite and positive.")
        self.temperature = decision_temperature
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("The request timeout must be between 0 and 3600 seconds.")
        self._client = httpx.Client(base_url=self.origin, timeout=timeout_seconds, transport=transport)
        try:
            response = self._client.get("/v1/models")
            response.raise_for_status()
            listed = response.json()["data"]
            self.served_model_ids = [str(item["id"]) for item in listed]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            self._client.close()
            raise RuntimeError(
                f"No decision server answered GET {self.origin}/v1/models "
                f"({type(error).__name__}: {error}). Start the official server first."
            ) from error

    def close(self) -> None:
        self._client.close()

    def _state_bound(self, request: dict[str, Any], *, reserve_reasoning: bool = True) -> int:
        """Upper bound of the state's token count; never an estimate below the true count."""
        text = self.codec.render(request["state"])
        escaped = self.codec._SPECIAL_RE.sub(r"<¦\1¦>", text)
        bound = 1 + len(escaped.encode("utf-8"))  # one state delimiter + at most one token per byte
        if self.reasoning and reserve_reasoning:
            bound += REASONING_STATE_ALLOWANCE
        if bound > self.codec.MAX_STATE:
            raise InputOverflowError(
                f"State may reach {bound} tokens (UTF-8 byte bound"
                f"{' plus the reasoning allowance' if self.reasoning and reserve_reasoning else ''}); "
                f"the official server may cut it at {self.codec.MAX_STATE}. No truncated decision is accepted."
            )
        return bound

    def _checked_answers(self, request: dict[str, Any], body: Any) -> dict[str, Any]:
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise ValueError("The server response has no answers object.")
        answers = body["answers"]
        if set(answers) != set(request["questions"]):
            raise ValueError("The server did not answer exactly the questions that were asked.")
        for qid, question in request["questions"].items():
            answer = answers[qid]
            if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
                raise ValueError(f"Answer {qid!r} has the wrong type.")
            qtype = question.get("type")
            if qtype == "noul":
                value = answer.get("noul")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Answer {qid!r} has an invalid noul probability.")
                continue
            probabilities = answer.get("probabilities")
            if qtype == "choice":
                keys = list(question["criteria"])
                if answer.get("choice") not in keys:
                    raise ValueError(f"Answer {qid!r} names an option that was not offered.")
            elif qtype == "score":
                criteria = question["criteria"]
                keys = [str(i) for i in range(len(criteria))]
                value = answer.get("score")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= len(criteria) - 1:
                    raise ValueError(f"Answer {qid!r} has an invalid score.")
                if answer.get("legend") != [self.codec.render(item) for item in criteria]:
                    raise ValueError(f"Answer {qid!r} has an incorrect score legend.")
            else:
                raise ValueError(f"Unsupported question type: {qtype!r}")
            if not isinstance(probabilities, dict) or list(probabilities) != keys:
                raise ValueError(f"Answer {qid!r} does not report every option in request order.")
            if probabilities is not None:
                values = list(probabilities.values())
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       or v < 0 or v > 1 for v in values):
                    raise FloatingPointError(f"Answer {qid!r} has non-finite or out-of-range probabilities.")
                # The official formatter rounds each value to four decimals.
                if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-4 * len(values) + 1e-9):
                    raise FloatingPointError(f"Answer {qid!r} probabilities do not sum to one.")
        return answers

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(request, dict) or set(request) - {"state", "questions"}:
            raise ValueError("This client sends text-only {state, questions} requests.")
        state_bound = self._state_bound(request)
        payload: dict[str, Any] = {"request": request, "reasoning": self.reasoning}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        try:
            response = self._client.post("/v1/decide", json=payload)
        except httpx.HTTPError as error:
            raise RuntimeError(f"Decision request failed: {type(error).__name__}: {error}") from error
        if response.status_code == 400:
            detail = response.text[:600]
            if "too long" in detail or "shorten" in detail:
                raise InputOverflowError(f"The server rejected an oversized question row: {detail}")
            raise ValueError(f"The server rejected the request: {detail}")
        if response.status_code != 200:
            raise RuntimeError(f"Decision server returned HTTP {response.status_code}: {response.text[:600]}")
        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise ValueError("The server response is not JSON.") from error
        answers = self._checked_answers(request, body)
        expanded_bound = None
        if self.reasoning:
            reasoning = body.get("reasoning")
            if not isinstance(reasoning, str):
                raise ValueError("Reasoning was requested but the server returned none.")
            # The server has already encoded this state. Reject an unsafe result
            # before the runner can act; the initial margin alone is not proof.
            expanded_bound = self._state_bound(
                self.codec.with_reasoning(request, reasoning), reserve_reasoning=False,
            )
        result: dict[str, Any] = {
            "answers": answers,
            "raw_probabilities": {
                qid: dict(answer["probabilities"]) for qid, answer in answers.items()
                if isinstance(answer.get("probabilities"), dict)
            },
            "metadata": {
                "request_sha256": ordered_json_sha256(request),
                "backend": "official_decide_http",
                "endpoint_origin": self.origin,
                "expected_model_id": MODEL_ID,
                "expected_model_revision": MODEL_REVISION,
                "served_model_ids": list(self.served_model_ids),
                # serve_decide.py reports the checkpoint directory name; a snapshot
                # path therefore ends in its revision. This is a hint, not a proof.
                "served_model_id_matches_pinned_revision": MODEL_REVISION in self.served_model_ids,
                "server_identity_verified_by_client": False,
                "source_revision": TRIDA_REVISION,
                "client_codec_sha256": SOURCE_SHA256["decide_request.py"],
                "official_sglang_parity_verified": False,
                "mock": False, "local_model_loaded": False,
                "reasoning_requested": self.reasoning,
                "generation_used": self.reasoning,
                "decision_temperature": self.temperature,
                "temperature_source": (
                    "explicit_override sent to the server; not independently calibrated"
                    if self.temperature is not None
                    else "server default (its checkpoint temperature.json or the official fallback); not observed by the client"
                ),
                "question_option_order": {
                    qid: list(q["criteria"]) for qid, q in request["questions"].items()
                    if isinstance(q.get("criteria"), dict)
                },
                "state_token_upper_bound": state_bound,
                "state_token_limit": self.codec.MAX_STATE,
                "state_with_reasoning_token_upper_bound": expanded_bound,
                "probability_precision": "rounded to four decimals by the official formatter",
                "server_timing_ms": body.get("timing_ms"),
                "server_reasoning_ms": body.get("reasoning_ms"),
                "backend_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "probability_notice": "Probabilities rank allowed actions; they are not calibrated attack probabilities.",
            },
        }
        if self.reasoning:
            result["reasoning"] = reasoning
        return result
