"""Public runtime contracts. Evaluation labels are deliberately absent."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

ACTIONS = ("inspect_session", "inspect_job_context", "keep_session", "revoke_session", "escalate")
INSPECTION_ACTIONS = ACTIONS[:2]
TERMINAL_ACTIONS = ACTIONS[2:]


class InputOverflowError(ValueError):
    """The complete request exceeds a limit; evidence must not be truncated."""


def ordered_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionSeed:
    session_id: str
    principal: str
    tenant_id: str
    status: Literal["active", "revoked"] = "active"
    can_checkout: bool = True


@dataclass(frozen=True)
class RuntimeScenario:
    """Fixture projection for the runner; case_id is logging metadata only."""

    case_id: str
    initial_state: dict[str, Any]
    tool_results: dict[str, dict[str, Any]]
    session_seeds: tuple[SessionSeed, ...]

    @property
    def target_session_id(self) -> str:
        return str(self.initial_state["target_session_id"])


class DecisionBackend(Protocol):
    """Accept the bare typed-decision body: {state, questions}."""

    name: str

    def decide(self, request: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class EpisodeResult:
    case_id: str
    backend_name: str
    stop_reason: str = "running"
    terminal_action: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    revealed_actions: list[str] = field(default_factory=list)
    decision_count: int = 0
    inspection_count: int = 0
    repeated_inspection_count: int = 0
    initial_probe: dict[str, int] = field(default_factory=dict)
    final_probe: dict[str, int] = field(default_factory=dict)
    mutation_effect: str = "not_requested"
    verification_status: str = "not_run"
    verification_details: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
