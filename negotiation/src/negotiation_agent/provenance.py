"""Small, CPU-only evidence helpers shared by negotiation probes and evaluations.

Snapshots contain public requests, model responses and local source/config hashes.
They never read credentials, model weights, buyer-private limits or supplier labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_INPUTS = (
    "src/sessionguard/__init__.py", "src/sessionguard/contracts.py",
    "src/sessionguard/backends/__init__.py", "src/sessionguard/backends/zoro.py",
    "src/sessionguard/backends/zoro_http.py",
    "config/model_pins.json", "pyproject.toml", "requirements-cpu.lock.txt", "requirements-gpu.txt",
)
# Backend choices whose answers come from the real decision model (native adapter or a
# decision server). One definition: evidence labels and the no-overwrite guard both use it.
REAL_MODEL_BACKENDS = ("zoro", "zoro-http", "systemone")
REASONING_BACKEND = "zoro-http"   # the only backend whose wire contract has reason-then-decide
CHECKPOINT_METADATA = (
    "config.json", "temperature.json", "tokenizer_config.json", "tokenizer.json",
    "model.safetensors.index.json", "export_metadata.json", "export_manifest.json", "head.pt", "jev_head.pt",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_sha256(request: dict) -> str:
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """Hash negotiation code plus the shared native adapter and pinned configuration."""
    paths = list((root / "negotiation" / "src" / "negotiation_agent").glob("*.py"))
    paths += list((root / "negotiation").glob("run_*.py"))
    paths += [root / name for name in SHARED_INPUTS]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}


def reasoning_requested(args: Any) -> bool:
    """The --reasoning flag, refused for every backend that cannot honour it.

    Recorded settings and the built backend both go through this check, so a freeze or a
    result can never carry a reasoning flag that the backend silently ignored.
    """
    requested = bool(getattr(args, "reasoning", False))
    if requested and args.backend != REASONING_BACKEND:
        raise SystemExit(f"--reasoning is only valid with --backend {REASONING_BACKEND}")
    return requested


def backend_execution_config(args: Any) -> dict[str, Any]:
    """Record explicit backend settings without URL credentials or query parameters."""
    reasoning = reasoning_requested(args)
    settings: dict[str, Any] = {"backend": args.backend}
    if args.backend == "zoro":
        for key in ("original_checkpoint", "exported_checkpoint", "official_source_dir", "device", "dtype",
                    "max_input_tokens", "decision_temperature"):
            value = getattr(args, key, None)
            if value is not None and key in ("original_checkpoint", "exported_checkpoint", "official_source_dir"):
                value = str(Path(value).expanduser().resolve())
            settings[key] = str(value) if isinstance(value, Path) else value
    elif args.backend == "zoro-http":
        from sessionguard.backends.zoro_http import endpoint_origin
        settings.update(endpoint_origin=endpoint_origin(args.url),
                        official_source_dir=str(Path(args.official_source_dir).expanduser().resolve()),
                        decision_temperature=args.decision_temperature,
                        request_timeout=args.request_timeout, reasoning=reasoning)
    elif args.backend == "systemone":
        parts = urlsplit(getattr(args, "url", "") or "")
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        origin = (
            f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")
        ) if parts.scheme and host else None
        settings.update(served_model=args.model, endpoint_origin=origin,
                        endpoint_base_url=(origin + parts.path.rstrip("/")) if origin else None,
                        request_timeout=getattr(args, "request_timeout", 120.0))
    return settings


def provenance(settings: dict | None = None, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Capture source identity before model loading; do not claim unobserved execution."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=10, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    pins = json.loads((root / "config" / "model_pins.json").read_text(encoding="utf-8"))
    source = (settings or {}).get("official_source_dir")
    official = {}
    if source:
        directory = Path(source)
        if not directory.is_absolute():
            directory = root / directory
        for relative, expected in pins["official_source_files"].items():
            path = directory / relative
            if not path.is_file():
                path = directory / Path(relative).name  # notebook-flat source layout
            actual = file_sha256(path) if path.is_file() else None
            official[relative] = {"expected_sha256": expected, "actual_sha256": actual,
                                  "matches_pin": actual == expected}
    checkpoints = {}
    for key in ("original_checkpoint", "exported_checkpoint"):
        location = (settings or {}).get(key)
        if location:
            directory = Path(location)
            if not directory.is_absolute():
                directory = root / directory
            checkpoints[key] = {name: file_sha256(directory / name) if (directory / name).is_file() else None
                                for name in CHECKPOINT_METADATA}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "input_sha256": input_hashes(root),
        "model_pins": pins, "execution_config": dict(settings or {}),
        "official_source_files": official,
        "checkpoint_metadata_sha256": checkpoints,
        "notice": "Source/config snapshot only; execution is evidenced by each recorded response.",
    }


def json_safe(value: Any) -> Any:
    """Copy evidence, making malformed non-finite responses explicit and JSON-writable."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def response_snapshot(backend: Any) -> dict | None:
    """Copy the last response immediately after predict(), including a wrapped native response.

    Native/HTTP backends clear last_response before each attempt, so a failed call
    cannot accidentally acquire the preceding request's response.
    """
    response = getattr(backend, "last_response", None)
    return json_safe(response) if response is not None else None


def reset_response(backend: Any) -> None:
    """Clear evidence before a new attempt; wrappers clear the underlying backend as well."""
    if hasattr(backend, "last_response"):
        backend.last_response = None
    if hasattr(backend, "inner"):
        reset_response(backend.inner)


def validate_predictions(predicted: dict, question_ids) -> dict[str, float]:
    """Require every requested estimate to be finite and in [0, 1] before scoring any row."""
    values = {qid: float(predicted[qid]) for qid in question_ids}
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in values.values()):
        raise ValueError("backend returned a non-finite or out-of-range probability")
    return values
