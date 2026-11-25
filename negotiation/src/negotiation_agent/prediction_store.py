"""Persistent, append-only store of successful backend answers.

NOT EXECUTED ON A GPU as of 2026-11-18. This module was authored and verified on CPU
only, with reference backends and test doubles. No real-model answer has been written
or replayed through it yet; nothing here is model evidence.

Purpose: an interrupted or repeated configuration must not call the model again for a
request that already has a successful answer. The store keeps one JSON line per
successful LIVE call, keyed by `provenance.request_sha256(request)`:

    {request_sha256, request, predictions, response, latency_ms, recorded_at_utc}

Rules this module keeps:
- It sees the public request and the backend's answer only. It imports no buyer-private
  or supplier-hidden type and receives none of their values.
- Only a complete, finite, in-range answer is stored. A failed, invalid or interrupted
  attempt stores nothing, is never filled in and is never retried here.
- The line is written, flushed and fsynced before `predict` returns.
- Lines are only ever appended; no byte of the file is rewritten or removed. A truncated
  trailing line (a crash in the middle of a write) is skipped and counted; the next
  append first closes it with a marker line naming its sha256, so it stays recognisable
  as a fragment on every later load. Any other unreadable or inconsistent line is refused.
- A replayed answer is labelled as a store hit in `last_response["prediction_store"]`;
  the wrapper reports the declared kind of the wrapped backend, not a new kind.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .provenance import json_safe, request_sha256, reset_response, response_snapshot, validate_predictions

MODES = ("read-write", "read-only")
RECORD_KEYS = ("request_sha256", "request", "predictions", "response", "latency_ms", "recorded_at_utc")
OBSERVED_KEYS = ("dtype", "gpu_name", "head_file_sha256")
_FRAGMENT_KEY = "truncated_fragment_sha256"
_ABSOLUTE_PATH = re.compile(r"^(?:[/\\~]|[A-Za-z]:[/\\])")


class StoreMiss(RuntimeError):
    """A read-only store has no answer for this request."""


class StoreConflict(RuntimeError):
    """The same request key is recorded with different predictions."""


class StoreIdentityMismatch(RuntimeError):
    """The store's files belong to another identity, or a live response disagrees with it."""


class StoreWriteFailed(RuntimeError):
    """An earlier write to the store failed (for example a full disk). No model call is made after that."""


class StoreCorrupt(RuntimeError):
    """A stored line is unreadable, incomplete, or does not hash to its own key."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def identity_digest(requested: dict) -> str:
    """sha256 of the canonical JSON of the requested identity (key order does not matter)."""
    return hashlib.sha256(_canonical(requested).encode("utf-8")).hexdigest()


def observed_from_response(response: dict | None) -> dict:
    """What a live response says about the runtime that produced it; absent fields stay absent."""
    metadata = response.get("metadata") if isinstance(response, dict) else None
    if not isinstance(metadata, dict):
        return {}
    observed = {key: metadata[key] for key in OBSERVED_KEYS if key in metadata}
    versions = metadata.get("runtime_versions")
    if isinstance(versions, dict) and "torch" in versions:
        observed["runtime_versions"] = {"torch": versions["torch"]}
    return json.loads(_canonical(json_safe(observed)))


def _refuse_locations(value: Any, where: str = "requested") -> None:
    """The identity must be the same on every machine: no URL and no absolute filesystem path."""
    if isinstance(value, dict):
        for key, item in value.items():
            _refuse_locations(key, where)
            _refuse_locations(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _refuse_locations(item, where)
    elif isinstance(value, str) and ("://" in value or _ABSOLUTE_PATH.match(value)):
        raise ValueError(f"{where}: a store identity must not contain a URL or an absolute path")


def _validated(predictions: dict, request: dict) -> dict[str, float]:
    """Every question of the request needs a finite estimate in [0, 1]; a partial answer is not stored."""
    try:
        return validate_predictions(predictions, request["questions"])
    except KeyError as error:
        raise ValueError(f"backend returned no estimate for question {error.args[0]!r}; "
                         "an incomplete answer is not stored") from error


def _fsync_directory(directory: Path) -> None:
    try:
        handle = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # not available on this platform (for example Windows)
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


class PredictionStore:
    """One identity's answers in `<directory>/<identity[:12]>.jsonl` plus `<identity[:12]>.identity.json`.

    A directory may hold several identities side by side. `read-only` never creates or
    changes a file. Two processes must not write the same identity at the same time.
    """

    def __init__(self, directory: Path, requested: dict, mode: str = "read-write") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not isinstance(requested, dict) or not requested:
            raise ValueError("requested identity must be a non-empty dict")
        _refuse_locations(requested)
        self.directory, self.mode = Path(directory), mode
        self.requested: dict = json.loads(_canonical(requested))
        self.identity: str = identity_digest(self.requested)
        self.id12: str = self.identity[:12]
        self.path = self.directory / f"{self.id12}.jsonl"
        self.identity_path = self.directory / f"{self.id12}.identity.json"
        self.observed: dict | None = None
        self.created_at_utc: str | None = None
        self.skipped_truncated_lines: int = 0
        self.ignored_duplicate_lines: int = 0
        self._records: dict[str, dict] = {}
        self._pending_fragments: list[bytes] = []   # unfinished end of the file, closed on the next append
        self._needs_newline = False
        self._write_failed = False
        self._open_identity()
        self._load_lines()

    # ---- identity file -------------------------------------------------------------------------

    def _identity_document(self) -> dict:
        return {"identity": self.identity, "requested": self.requested, "observed": self.observed,
                "created_at_utc": self.created_at_utc}

    def _open_identity(self) -> None:
        if not self.identity_path.exists():
            if self.path.exists():
                raise StoreIdentityMismatch(f"{self.path.name} has no identity file; refusing to use it")
            if self.mode == "read-only":
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            self.created_at_utc = _utc_now()
            try:
                with open(self.identity_path, "x", encoding="utf-8") as stream:
                    json.dump(self._identity_document(), stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(self.directory)
                return
            except FileExistsError:
                pass  # created by someone else in the meantime: verify it like any existing file
        try:
            document = json.loads(self.identity_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise StoreIdentityMismatch(f"{self.identity_path.name} is unreadable: {error}") from error
        if not isinstance(document, dict) or document.get("identity") != self.identity \
                or not isinstance(document.get("requested"), dict) \
                or identity_digest(document["requested"]) != self.identity:
            raise StoreIdentityMismatch(
                f"{self.identity_path.name} does not describe the requested identity {self.id12}")
        observed = document.get("observed")
        if observed is not None and not isinstance(observed, dict):
            raise StoreIdentityMismatch(f"{self.identity_path.name} has an unreadable observed block")
        self.observed, self.created_at_utc = observed, document.get("created_at_utc")

    def bind_observed(self, observed: dict) -> None:
        """Bind the first live response's runtime; a later different value is refused before any write."""
        if not isinstance(observed, dict):
            raise ValueError("observed must be a dict")
        observed = json.loads(_canonical(json_safe(observed)))
        if self.observed is not None:
            if observed != self.observed:
                raise StoreIdentityMismatch(
                    f"live response reports {_canonical(observed)} but store {self.id12} is bound to "
                    f"{_canonical(self.observed)}; answers from different runtimes are not mixed")
            return
        self._require_writable()
        previous, self.observed = self.observed, observed
        temporary = self.identity_path.with_name(f"{self.identity_path.name}.tmp-{os.getpid()}")
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(self._identity_document(), stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.identity_path)
        except BaseException:
            self.observed = previous
            temporary.unlink(missing_ok=True)
            raise
        _fsync_directory(self.directory)

    # ---- lines ---------------------------------------------------------------------------------

    def _check_record(self, record: Any, where: str) -> dict:
        if not isinstance(record, dict) or set(record) != set(RECORD_KEYS):
            raise StoreCorrupt(f"{where}: not a prediction record")
        request, predictions, response = record["request"], record["predictions"], record["response"]
        if not isinstance(request, dict) or not isinstance(request.get("questions"), dict):
            raise StoreCorrupt(f"{where}: the stored request has no questions")
        try:
            digest = request_sha256(request)
        except ValueError as error:
            raise StoreCorrupt(f"{where}: the stored request cannot be hashed: {error}") from error
        if digest != record["request_sha256"]:
            raise StoreCorrupt(f"{where}: the stored request does not hash to its key")
        if not isinstance(predictions, dict) or set(predictions) != set(request["questions"]):
            raise StoreCorrupt(f"{where}: the stored predictions do not answer the stored questions")
        try:
            values = validate_predictions(predictions, request["questions"])
        except (TypeError, ValueError) as error:
            raise StoreCorrupt(f"{where}: {error}") from error
        latency = record["latency_ms"]
        if (response is not None and not isinstance(response, dict)) or isinstance(latency, bool) \
                or not isinstance(latency, (int, float)) or not isinstance(record["recorded_at_utc"], str):
            raise StoreCorrupt(f"{where}: malformed response, latency or timestamp")
        return {**record, "predictions": values}

    def _load_lines(self) -> None:
        """Read every line. Only a region left at the end of the file by an interrupted write is tolerated.

        Writes only append, so an interrupted write can leave unreadable bytes at the end of
        the file and nowhere else. Such lines are skipped and counted. The next writer closes
        them with a marker line listing their sha256 values, which keeps them recognisable
        once more lines follow. An unreadable line that no marker closes and that is not at
        the end, or a readable line that is not a consistent record, is refused.
        """
        if not self.path.exists():
            return
        pieces = self.path.read_bytes().split(b"\n")
        tail = pieces.pop()             # b"" when the file ends with a newline
        self._needs_newline = bool(tail)
        pending: list[bytes] = []
        for number, raw in enumerate(pieces + ([tail] if tail else []), 1):
            where = f"{self.path.name} line {number}"
            try:
                value = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                pending.append(raw)
                continue
            if isinstance(value, dict) and _FRAGMENT_KEY in value:
                if not pending or value[_FRAGMENT_KEY] != [hashlib.sha256(p).hexdigest() for p in pending]:
                    raise StoreCorrupt(f"{where}: fragment marker does not match the lines before it")
                self.skipped_truncated_lines += len(pending)
                pending = []
                continue
            if pending:
                raise StoreCorrupt(f"{where}: preceded by an unreadable line that is not a truncated trailing line")
            record = self._check_record(value, where)
            known = self._records.get(record["request_sha256"])
            if known is None:
                self._records[record["request_sha256"]] = record
            elif known["predictions"] == record["predictions"]:
                self.ignored_duplicate_lines += 1
            else:
                raise StoreConflict(f"{where}: request {record['request_sha256'][:12]} is already recorded "
                                    "with different predictions")
        # Whatever is still pending is the unfinished end of the file: no answer was served from it.
        self.skipped_truncated_lines += len(pending)
        self._pending_fragments = pending

    def _require_writable(self) -> None:
        if self.mode != "read-write":
            raise RuntimeError("prediction store is read-only")
        if self._write_failed:
            raise StoreWriteFailed("an earlier write to this prediction store failed; free space and start the "
                                   "command again (the store is reopened then)")

    @property
    def write_failed(self) -> bool:
        return self._write_failed

    def _append(self, line: str) -> None:
        data = b"\n" if self._needs_newline else b""
        if self._pending_fragments:
            marker = {_FRAGMENT_KEY: [hashlib.sha256(p).hexdigest() for p in self._pending_fragments],
                      "truncated_fragment_bytes": [len(p) for p in self._pending_fragments],
                      "closed_at_utc": _utc_now()}
            data += json.dumps(marker, separators=(",", ":")).encode("utf-8") + b"\n"
        data += line.encode("utf-8") + b"\n"
        created = not self.path.exists()
        try:
            with open(self.path, "ab") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self._write_failed = True   # the file may now end in a fragment; a reopen handles it
            raise
        self._needs_newline, self._pending_fragments = False, []
        if created:
            _fsync_directory(self.directory)

    def get(self, request_sha256: str) -> dict | None:
        """The stored record for this request key, or None. The caller gets its own copy."""
        record = self._records.get(request_sha256)
        return copy.deepcopy(record) if record is not None else None

    def put(self, request: dict, predictions: dict[str, float], response: dict | None,
            latency_ms: float) -> dict:
        """Store one successful live answer durably and return the stored record.

        Raises without writing a line when the answer is incomplete or invalid, when the key
        already holds different predictions, or when the response's runtime differs from the
        one this store is bound to. An equal answer for a known key is not written again.
        """
        self._require_writable()
        key = request_sha256(request)
        values = _validated(predictions, request)
        known = self._records.get(key)
        if known is not None:
            if known["predictions"] != values:
                raise StoreConflict(f"request {key[:12]} is already recorded with different predictions")
            return copy.deepcopy(known)
        record = {"request_sha256": key, "request": request, "predictions": values,
                  "response": json_safe(response) if response is not None else None,
                  "latency_ms": float(latency_ms), "recorded_at_utc": _utc_now()}
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stored = self._check_record(json.loads(line), "new record")   # what a later load will see
        self.bind_observed(observed_from_response(response))
        self._append(line)
        self._records[key] = stored
        return copy.deepcopy(stored)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, request_sha256: object) -> bool:
        return request_sha256 in self._records


class StoredPredictionBackend:
    """Answer from the store when the request is known; otherwise ask the wrapped backend once and store it.

    `factory()` builds the wrapped backend on the first miss, so a full replay loads no
    weights. `kind`, `is_model` and `supports_final_offer` are the declared values of the
    wrapped backend and are checked against it when it is built.
    """

    def __init__(self, store: PredictionStore, factory: Callable[[], Any], *, kind: str, is_model: bool,
                 supports_final_offer: bool) -> None:
        if not callable(factory):
            raise TypeError("factory must be a callable that builds the wrapped backend")
        self.store, self._factory = store, factory
        self.kind, self.is_model, self.supports_final_offer = kind, bool(is_model), bool(supports_final_offer)
        self.inner: Any = None                       # built lazily; provenance.reset_response tolerates None
        self.last_response: dict | None = None
        self.live_requests = 0       # calls made to the wrapped backend, including failed ones
        self.stored_answers = 0      # live answers written to the store by this object
        self.store_hits = 0
        self.store_misses = 0        # read-only misses; no backend call was made for them
        self.live_seconds = 0.0      # time spent inside the wrapped backend's predict, failures included
        self.factory_seconds = 0.0
        self._factory_error: str | None = None

    def _built(self) -> Any:
        if self.inner is not None:
            return self.inner
        if self._factory_error is not None:
            raise RuntimeError(f"the wrapped backend could not be built earlier ({self._factory_error}); "
                               "it is not built again in this process")
        started = time.perf_counter()
        try:
            inner = self._factory()
        except Exception as error:
            self._factory_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.factory_seconds += time.perf_counter() - started
        actual = (getattr(inner, "kind", None), bool(getattr(inner, "is_model", False)),
                  bool(getattr(inner, "supports_final_offer", getattr(inner, "is_model", False))))
        declared = (self.kind, self.is_model, self.supports_final_offer)
        if actual != declared:
            self._factory_error = (f"declared (kind, is_model, supports_final_offer)={declared} "
                                   f"but the built backend has {actual}")
            if hasattr(inner, "close"):
                inner.close()
            raise ValueError(self._factory_error)
        self.inner = inner
        return inner

    def predict(self, state, candidates, request) -> dict[str, float]:
        self.last_response = None
        block = {"hit": False, "identity": self.store.id12}
        record = self.store.get(request_sha256(request))
        if record is not None:
            self.store_hits += 1
            self.last_response = {**(record["response"] or {}), "prediction_store": {
                "hit": True, "identity": self.store.id12,
                "first_recorded_at_utc": record["recorded_at_utc"], "first_latency_ms": record["latency_ms"]}}
            return record["predictions"]
        if self.store.mode != "read-write":
            self.store_misses += 1
            raise StoreMiss(f"read-only prediction store {self.store.id12} has no answer for this request")
        if self.store.write_failed:
            # The answer could not be kept: asking the model would spend GPU time on a value thrown away.
            raise StoreWriteFailed("an earlier write to this prediction store failed; no model call was made. "
                                   "Free space and start the command again")
        inner = self._built()
        reset_response(inner)
        self.live_requests += 1
        response, elapsed = None, 0.0
        try:
            started = time.perf_counter()
            try:
                predicted = inner.predict(state, candidates, request)
            finally:
                elapsed = time.perf_counter() - started
                self.live_seconds += elapsed
                response = response_snapshot(inner)
            record = self.store.put(request, predicted, response, round(elapsed * 1000, 3))
        except BaseException:
            # Keep what the backend returned for this attempt, marked as not stored. No value is filled in.
            self.last_response = {**(response or {}), "prediction_store": {
                **block, "stored": False, "latency_ms": round(elapsed * 1000, 3)}}
            raise
        self.stored_answers += 1
        # `latency_ms` is the time inside the wrapped predict only: it excludes building the backend
        # (the checkpoint load happens in `_built()` above, on the first miss).
        self.last_response = {**(response or {}), "prediction_store": {
            **block, "latency_ms": round(elapsed * 1000, 3)}}
        return record["predictions"]

    def check_runtime(self, state, candidates, request) -> dict:
        """One live call whose answer is NOT stored: it only compares the runtime with the stored one.

        Raises StoreIdentityMismatch when the answering runtime differs from the one the stored
        answers came from, and whatever the backend raises when it cannot answer. Used before a
        step that must not start on a runtime that cannot continue it.
        """
        if self.store.mode != "read-write":
            raise RuntimeError("a runtime check needs a read-write store and a live backend")
        inner = self._built()
        reset_response(inner)
        self.live_requests += 1
        started = time.perf_counter()
        try:
            predicted = inner.predict(state, candidates, request)
        finally:
            self.live_seconds += time.perf_counter() - started
        validate_predictions(predicted, request["questions"])
        observed = observed_from_response(response_snapshot(inner))
        self.store.bind_observed(observed)
        return observed

    def describe(self) -> dict:
        return {
            "kind": self.kind, "is_model": self.is_model, "supports_final_offer": self.supports_final_offer,
            "prediction_store": {
                "identity": self.store.identity, "identity12": self.store.id12, "mode": self.store.mode,
                "requested": copy.deepcopy(self.store.requested), "observed": copy.deepcopy(self.store.observed),
                "entries": len(self.store), "skipped_truncated_lines": self.store.skipped_truncated_lines,
                "live_requests": self.live_requests, "stored_answers": self.stored_answers,
                "store_hits": self.store_hits, "store_misses": self.store_misses,
                "live_seconds": round(self.live_seconds, 3), "inner_built": self.inner is not None,
            },
            "inner": self.inner.describe() if self.inner is not None else None,
        }

    def close(self) -> None:
        inner, self.inner = self.inner, None
        if inner is not None and hasattr(inner, "close"):
            inner.close()
