"""CPU-only tests of the prediction store. NOT executed on a GPU as of 2026-11-18.

Every "model" here is a test double built on the keyword rule; nothing in this file is
model evidence. Only development cases are used.
"""

import ast
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from negotiation_agent import evaluation as E
from negotiation_agent import policy
from negotiation_agent import prediction_store as PS
from negotiation_agent.backends import HeuristicBackend, KeywordBackend
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.prediction_store import (
    PredictionStore,
    StoreConflict,
    StoreCorrupt,
    StoredPredictionBackend,
    StoreIdentityMismatch,
    StoreMiss,
    identity_digest,
    observed_from_response,
)
from negotiation_agent.provenance import request_sha256, reset_response
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.simulator import default_buyer_private

GOLDEN_DEV_1100_ROUND_1 = "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
REQUESTED = {"backend": "recorded_test_double", "model_revision": "test-revision", "dtype": "fp16",
             "max_input_tokens": 2048, "decision_temperature": 1.0, "reasoning": False}
DECLARED = {"kind": "recorded_test_double", "is_model": True, "supports_final_offer": True}
MODULE = Path(PS.__file__)


class Double(KeywordBackend):
    """Keyword-rule stand-in that reports itself as a model and returns native-style metadata."""

    kind, is_model, supports_final_offer = "recorded_test_double", True, True

    def __init__(self, fail_on=(), mode="error", gpu_name="Test GPU", with_response=True):
        self.calls, self.closed = 0, False
        self.fail_on, self.mode, self.gpu_name, self.with_response = set(fail_on), mode, gpu_name, with_response
        self.last_response = None
        self.seen: list[str] = []

    def predict(self, state, candidates, request):
        self.last_response = None
        self.calls += 1
        self.seen.append(request_sha256(request))
        failing = self.calls in self.fail_on
        if failing and self.mode == "interrupt":
            raise KeyboardInterrupt("test interruption")
        if failing and self.mode == "error":
            raise RuntimeError("test backend failure")
        out = super().predict(state, candidates, request)
        if failing and self.mode == "nan":
            out[FINAL_OFFER_QID] = float("nan")
        if failing and self.mode == "out_of_range":
            out[FINAL_OFFER_QID] = 1.5
        if failing and self.mode == "missing_final":
            out.pop(FINAL_OFFER_QID)
        if self.with_response:
            self.last_response = {
                "raw_probabilities": {qid: {"true": p, "false": 1 - p} for qid, p in out.items()
                                      if p == p},
                "metadata": {"request_sha256": request_sha256(request), "sequence": self.calls,
                             "dtype": "fp16", "gpu_name": self.gpu_name, "head_file_sha256": "a" * 64,
                             "runtime_versions": {"torch": "0.0-test", "transformers": "0.0-test"}},
            }
        return out

    def close(self):
        self.closed = True


def forbidden_factory():
    pytest.fail("a full replay must not build the wrapped backend")


def wrap(store, inner=None, factory=None, **declared):
    if factory is None:
        factory = (lambda: inner) if inner is not None else forbidden_factory
    return StoredPredictionBackend(store, factory, **{**DECLARED, **declared})


def run_cases(backend, count=4, start=0):
    private = default_buyer_private()
    return [E.run_case(case, private, backend, policy.decide, PolicyConfig(), approve=True)[0]
            for case in E.make_cases("dev", start + count)[start:]]


def comparable(rows):
    """Decisions and outcomes, without timing and without the evidence block that labels a replay."""
    keep = ("case_id", "status", "agreement", "realized_saving", "net_saving", "rounds_used", "counters_sent",
            "approval_cards", "approvals_denied", "authority_violations", "feasible", "capture", "target_met",
            "gave_up_savings", "final_offer_estimates", "public_transcript", "executed", "approval_log",
            "interrupted")
    out = []
    for row in rows:
        decisions = [{"round": r["round"], "action": r["action"], "reasons": r["reasons"], "facts": r["facts"],
                      "chosen": r["chosen"], "options": r["options"], "approval_request": r["approval_request"],
                      "request_sha256": r["model"].get("request_sha256"), "status": r["model"].get("status"),
                      "predictions": r["model"].get("predictions")} for r in row["internal_record"]]
        out.append({**{key: row[key] for key in keep}, "decisions": decisions})
    return out


def models(rows):
    return [r["model"] for row in rows for r in row["internal_record"] if "trace_id" in r["model"]]


def lines(store):
    return [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines()]


def a_request(tag="one"):
    return {"state": {"거래": {"품목": tag}}, "questions": {"C1": {"type": "noul"}, FINAL_OFFER_QID: {"type": "noul"}}}


ANSWER = {"C1": 0.25, FINAL_OFFER_QID: 0.5}


# ---- identity -----------------------------------------------------------------------------------


def test_identity_digest_is_the_canonical_json_sha256_and_ignores_key_order():
    requested = {"backend": "zoro", "dtype": "fp16", "설명": "한글", "nested": {"b": 1, "a": 2}}
    canonical = '{"backend":"zoro","dtype":"fp16","nested":{"a":2,"b":1},"설명":"한글"}'
    assert identity_digest(requested) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert identity_digest(dict(reversed(list(requested.items())))) == identity_digest(requested)
    assert identity_digest({**requested, "dtype": "bf16"}) != identity_digest(requested)


def test_store_files_are_named_by_identity_and_several_identities_share_a_directory(tmp_path):
    first = PredictionStore(tmp_path / "store", REQUESTED)
    second = PredictionStore(tmp_path / "store", {**REQUESTED, "dtype": "bf16"})
    assert first.identity == identity_digest(REQUESTED) and first.id12 == first.identity[:12]
    assert first.path.name == f"{first.id12}.jsonl" and first.identity_path.name == f"{first.id12}.identity.json"
    first.put(a_request(), ANSWER, None, 1.0)
    assert len(first) == 1 and len(second) == 0 and not second.path.exists()
    document = json.loads(first.identity_path.read_text(encoding="utf-8"))
    assert set(document) == {"identity", "requested", "observed", "created_at_utc"}
    assert document["identity"] == first.identity and document["requested"] == REQUESTED
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == sorted(
        [first.path.name, first.identity_path.name, second.identity_path.name])
    assert len(PredictionStore(tmp_path / "store", {**REQUESTED, "dtype": "bf16"})) == 0
    assert len(PredictionStore(tmp_path / "store", dict(reversed(list(REQUESTED.items()))))) == 1


@pytest.mark.parametrize("bad", [
    {"backend": "zoro-http", "url": "https://user:secret@example.test/v1/decide"},
    {"backend": "zoro", "checkpoint": "/content/models/zoro"},
    {"backend": "zoro", "checkpoint": "C:\\models\\zoro"},
    {"backend": "zoro", "sources": ["~/trida-stack"]},
    {"backend": "zoro", "files": {"/abs/zoro.py": "0" * 64}},
    {}, ["backend"],
])
def test_identity_refuses_urls_absolute_paths_and_empty_values(tmp_path, bad):
    with pytest.raises(ValueError):
        PredictionStore(tmp_path / "store", bad)
    assert not (tmp_path / "store").exists()


def test_relative_source_names_and_revisions_are_allowed_in_the_identity(tmp_path):
    requested = {**REQUESTED, "adapter_sha256": {"src/sessionguard/backends/zoro.py": "0" * 64},
                 "model_id": "organisation/zoro-1"}
    assert PredictionStore(tmp_path, requested).requested == requested
    with pytest.raises(ValueError, match="mode"):
        PredictionStore(tmp_path, requested, mode="append")


def test_observed_from_response_keeps_only_runtime_fields_that_are_present():
    response = {"metadata": {"dtype": "fp16", "gpu_name": None, "head_file_sha256": "b" * 64, "device": "cuda:0",
                             "runtime_versions": {"torch": "2.5.1", "transformers": "4.57"}}}
    assert observed_from_response(response) == {"dtype": "fp16", "gpu_name": None, "head_file_sha256": "b" * 64,
                                                "runtime_versions": {"torch": "2.5.1"}}
    assert observed_from_response({"metadata": {"backend": "official_decide_http"}}) == {}
    assert observed_from_response(None) == {} and observed_from_response({"answers": {}}) == {}


# ---- put / get ----------------------------------------------------------------------------------


def test_put_writes_one_durable_line_and_get_returns_a_private_copy(tmp_path, monkeypatch):
    store = PredictionStore(tmp_path, REQUESTED)
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(PS.os, "fsync", lambda fd: (synced.append(store.path.read_bytes() if store.path.exists()
                                                                  else b""), real_fsync(fd))[1])
    request = a_request()
    record = store.put(request, {"C1": 1, FINAL_OFFER_QID: 0.5, "extra": 0.9}, {"metadata": {"dtype": "fp16"}}, 12.3456)
    assert set(record) == set(PS.RECORD_KEYS) == set(lines(store)[0])
    assert record["request_sha256"] == request_sha256(request) and record["request"] == request
    assert record["predictions"] == {"C1": 1.0, FINAL_OFFER_QID: 0.5} and record["latency_ms"] == 12.3456
    assert synced and synced[-1].endswith(b"\n") and json.loads(synced[-1].splitlines()[-1]) == record
    assert store.path.read_bytes().count(b"\n") == 1 and len(store) == 1
    assert request_sha256(request) in store and "missing" not in store
    copy = store.get(record["request_sha256"])
    copy["predictions"]["C1"] = 0.0
    assert store.get(record["request_sha256"])["predictions"]["C1"] == 1.0
    assert store.get("0" * 64) is None
    reloaded = PredictionStore(tmp_path, REQUESTED, mode="read-only")
    assert reloaded.get(record["request_sha256"]) == record and reloaded.observed == {"dtype": "fp16"}


@pytest.mark.parametrize("predictions", [
    {"C1": 0.25}, {"C1": float("nan"), FINAL_OFFER_QID: 0.5}, {"C1": 0.25, FINAL_OFFER_QID: 1.01},
    {"C1": -0.1, FINAL_OFFER_QID: 0.5}, {"C1": float("inf"), FINAL_OFFER_QID: 0.5},
])
def test_put_stores_nothing_for_an_incomplete_or_invalid_answer(tmp_path, predictions):
    store = PredictionStore(tmp_path, REQUESTED)
    with pytest.raises(ValueError):
        store.put(a_request(), predictions, {"metadata": {"dtype": "fp16"}}, 1.0)
    assert len(store) == 0 and not store.path.exists() and store.observed is None


def test_equal_duplicate_is_ignored_and_a_different_answer_for_the_same_request_is_refused(tmp_path):
    store = PredictionStore(tmp_path, REQUESTED)
    first = store.put(a_request(), ANSWER, None, 1.0)
    assert store.put(a_request(), dict(ANSWER), None, 9.0) == first
    with pytest.raises(StoreConflict):
        store.put(a_request(), {**ANSWER, "C1": 0.26}, None, 1.0)
    assert len(lines(store)) == 1 and len(store) == 1
    # The same on load: an equal duplicate line is ignored, a different one is refused.
    line = store.path.read_text(encoding="utf-8")
    store.path.write_text(line + line, encoding="utf-8")
    reloaded = PredictionStore(tmp_path, REQUESTED)
    assert len(reloaded) == 1 and reloaded.ignored_duplicate_lines == 1
    changed = json.loads(line)
    changed["predictions"]["C1"] = 0.75
    store.path.write_text(line + json.dumps(changed, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(StoreConflict, match="different predictions"):
        PredictionStore(tmp_path, REQUESTED, mode="read-only")


def test_read_only_store_never_writes_and_never_creates_files(tmp_path):
    missing = PredictionStore(tmp_path / "absent", REQUESTED, mode="read-only")
    assert len(missing) == 0 and not (tmp_path / "absent").exists()
    with pytest.raises(RuntimeError, match="read-only"):
        missing.put(a_request(), ANSWER, None, 1.0)
    with pytest.raises(RuntimeError, match="read-only"):
        missing.bind_observed({"dtype": "fp16"})
    assert not (tmp_path / "absent").exists()


# ---- truncated, tampered and conflicting files ----------------------------------------------------


def filled(tmp_path, count=3):
    store = PredictionStore(tmp_path, REQUESTED)
    for index in range(count):
        store.put(a_request(f"request-{index}"), ANSWER, None, 1.0)
    return store


def test_truncated_last_line_is_skipped_counted_and_closed_by_the_next_append(tmp_path):
    store = filled(tmp_path)
    complete = store.path.read_bytes()
    cut = complete[: len(complete) - 40]          # the third record loses its end and its newline
    store.path.write_bytes(cut)
    for mode in ("read-only", "read-only"):
        reloaded = PredictionStore(tmp_path, REQUESTED, mode=mode)
        assert len(reloaded) == 2 and reloaded.skipped_truncated_lines == 1
        assert reloaded.get(request_sha256(a_request("request-2"))) is None
    assert store.path.read_bytes() == cut         # reading changes nothing
    writer = PredictionStore(tmp_path, REQUESTED)
    assert len(writer) == 2 and writer.skipped_truncated_lines == 1 and store.path.read_bytes() == cut
    writer.put(a_request("request-2"), ANSWER, None, 2.0)
    writer.put(a_request("request-3"), ANSWER, None, 2.0)
    after = store.path.read_bytes()
    assert after.startswith(cut + b"\n") and after.endswith(b"\n")     # appended only; no byte removed
    again = PredictionStore(tmp_path, REQUESTED, mode="read-only")
    assert len(again) == 4 and again.skipped_truncated_lines == 1
    assert again.get(request_sha256(a_request("request-2")))["latency_ms"] == 2.0


def test_every_interrupted_write_of_a_closing_append_stays_loadable(tmp_path):
    store = filled(tmp_path, 2)
    fragment = store.path.read_bytes() + b'{"request_sha256":"abc","request":{"st'
    store.path.write_bytes(fragment)
    PredictionStore(tmp_path, REQUESTED).put(a_request("request-2"), ANSWER, None, 1.0)
    healed = store.path.read_bytes()
    assert len(PredictionStore(tmp_path, REQUESTED, mode="read-only")) == 3
    for size in range(len(fragment), len(healed) + 1):
        store.path.write_bytes(healed[:size])
        reloaded = PredictionStore(tmp_path, REQUESTED)
        assert len(reloaded) == (3 if size >= len(healed) - 1 else 2), size
        assert reloaded.skipped_truncated_lines >= 1
        reloaded.put(a_request("request-9"), ANSWER, None, 1.0)
        final = PredictionStore(tmp_path, REQUESTED, mode="read-only")
        assert request_sha256(a_request("request-9")) in final and len(final) == len(reloaded)
        assert store.path.read_bytes().startswith(healed[:size])


def test_complete_last_record_without_newline_is_kept_and_the_next_line_starts_cleanly(tmp_path):
    store = filled(tmp_path, 2)
    store.path.write_bytes(store.path.read_bytes().rstrip(b"\n"))
    reloaded = PredictionStore(tmp_path, REQUESTED)
    assert len(reloaded) == 2 and reloaded.skipped_truncated_lines == 0
    reloaded.put(a_request("request-2"), ANSWER, None, 1.0)
    assert len(PredictionStore(tmp_path, REQUESTED, mode="read-only")) == 3 and len(lines(store)) == 3


def rewrite_line(store, index, change):
    rows = store.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(rows[index])
    change(record)
    rows[index] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    store.path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("change", [
    lambda r: r["request"]["state"]["거래"].update(품목="다른 품목"),
    lambda r: r["request"]["questions"]["C1"].update(instructions="changed"),
    lambda r: r.update(request_sha256="0" * 64),
    lambda r: r["predictions"].update(C1=1.5),
    lambda r: r["predictions"].pop(FINAL_OFFER_QID),
    lambda r: r["predictions"].update(C9=0.5),
    lambda r: r.pop("recorded_at_utc"),
    lambda r: r.update(note="added"),
    lambda r: r.update(response="text"),
])
def test_tampered_line_is_refused(tmp_path, change):
    store = filled(tmp_path)
    rewrite_line(store, 1, change)
    for mode in PS.MODES:
        with pytest.raises(StoreCorrupt):
            PredictionStore(tmp_path, REQUESTED, mode=mode)


def test_unreadable_line_in_the_middle_or_a_wrong_marker_is_refused(tmp_path):
    store = filled(tmp_path)
    rows = store.path.read_bytes().splitlines()
    store.path.write_bytes(b"\n".join([rows[0], rows[1][:50], rows[2]]) + b"\n")
    with pytest.raises(StoreCorrupt, match="unreadable"):
        PredictionStore(tmp_path, REQUESTED)
    marker = json.dumps({"truncated_fragment_sha256": ["0" * 64]}).encode()
    store.path.write_bytes(b"\n".join([rows[0], rows[1][:50], marker, rows[2]]) + b"\n")
    with pytest.raises(StoreCorrupt, match="marker"):
        PredictionStore(tmp_path, REQUESTED)
    store.path.write_bytes(b"\n".join([rows[0], marker, rows[2]]) + b"\n")
    with pytest.raises(StoreCorrupt, match="marker"):
        PredictionStore(tmp_path, REQUESTED)
    good = json.dumps({"truncated_fragment_sha256": [hashlib.sha256(rows[1][:50]).hexdigest()]}).encode()
    store.path.write_bytes(b"\n".join([rows[0], rows[1][:50], good, rows[2]]) + b"\n")
    reloaded = PredictionStore(tmp_path, REQUESTED)
    assert len(reloaded) == 2 and reloaded.skipped_truncated_lines == 1


def test_identity_file_must_describe_the_requested_identity(tmp_path):
    store = filled(tmp_path, 1)
    original = store.identity_path.read_text(encoding="utf-8")
    document = json.loads(original)
    store.identity_path.write_text(json.dumps({**document, "requested": {**REQUESTED, "dtype": "bf16"}}),
                                   encoding="utf-8")
    with pytest.raises(StoreIdentityMismatch):
        PredictionStore(tmp_path, REQUESTED)
    store.identity_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreIdentityMismatch):
        PredictionStore(tmp_path, REQUESTED, mode="read-only")
    store.identity_path.unlink()
    for mode in PS.MODES:
        with pytest.raises(StoreIdentityMismatch, match="no identity file"):
            PredictionStore(tmp_path, REQUESTED, mode=mode)
    assert not store.identity_path.exists()
    store.identity_path.write_text(original, encoding="utf-8")
    assert len(PredictionStore(tmp_path, REQUESTED)) == 1


def test_bind_observed_binds_once_and_refuses_a_different_runtime_before_any_write(tmp_path):
    store = PredictionStore(tmp_path, REQUESTED)
    assert json.loads(store.identity_path.read_text(encoding="utf-8"))["observed"] is None
    store.bind_observed({"dtype": "fp16", "gpu_name": "Tesla T4"})
    store.bind_observed({"gpu_name": "Tesla T4", "dtype": "fp16"})
    document = json.loads(store.identity_path.read_text(encoding="utf-8"))
    assert document["observed"] == {"dtype": "fp16", "gpu_name": "Tesla T4"} and document["identity"] == store.identity
    before = store.identity_path.read_bytes()
    with pytest.raises(StoreIdentityMismatch, match="different runtimes"):
        store.bind_observed({"dtype": "fp16", "gpu_name": "Other GPU"})
    with pytest.raises(StoreIdentityMismatch):
        store.put(a_request(), ANSWER, {"metadata": {"dtype": "fp16", "gpu_name": "Other GPU"}}, 1.0)
    assert store.identity_path.read_bytes() == before and not store.path.exists() and len(store) == 0
    store.put(a_request(), ANSWER, {"metadata": {"dtype": "fp16", "gpu_name": "Tesla T4"}}, 1.0)
    reopened = PredictionStore(tmp_path, REQUESTED)
    assert reopened.observed == {"dtype": "fp16", "gpu_name": "Tesla T4"} and len(reopened) == 1
    assert [p.name for p in tmp_path.iterdir() if "tmp" in p.name] == []


def test_failed_write_stores_nothing_in_memory_and_requires_a_reopen(tmp_path, monkeypatch):
    store = filled(tmp_path, 1)

    def failing_fsync(_fd):
        raise OSError("test disk failure")

    monkeypatch.setattr(PS.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk failure"):
        store.put(a_request("request-1"), ANSWER, None, 1.0)
    monkeypatch.undo()
    assert len(store) == 1 and store.get(request_sha256(a_request("request-1"))) is None
    with pytest.raises(RuntimeError, match="reopen"):
        store.put(a_request("request-2"), ANSWER, None, 1.0)


# ---- the backend wrapper in real sessions ----------------------------------------------------------


def test_second_pass_makes_no_backend_call_and_repeats_decisions_and_outcomes(tmp_path):
    inner = Double()
    backend = wrap(PredictionStore(tmp_path, REQUESTED), inner)
    assert backend.inner is None and backend.describe()["inner"] is None
    first = run_cases(backend)
    first_calls = models(first)
    assert inner.calls == len(first_calls) == len(set(inner.seen)) > 4
    assert inner.seen[0] == GOLDEN_DEV_1100_ROUND_1          # the wrapper passes the default request unchanged
    assert backend.live_requests == backend.stored_answers == inner.calls and backend.store_hits == 0
    assert backend.live_seconds > 0 and len(backend.store) == inner.calls
    stored = lines(backend.store)
    assert [row["request_sha256"] for row in stored] == inner.seen
    for call, row in zip(first_calls, stored):
        assert call["backend"] == "recorded_test_double" and call["is_model"] is True and call["status"] == "ok"
        block = dict(call["response"]["prediction_store"])
        assert isinstance(block.pop("latency_ms"), float)       # time inside the wrapped predict only
        assert block == {"hit": False, "identity": backend.store.id12}
        assert call["response"]["metadata"]["request_sha256"] == call["request_sha256"] == row["request_sha256"]
        assert call["predictions"] == row["predictions"] and call["request"] == row["request"]
        assert row["response"]["metadata"]["sequence"] and "prediction_store" not in row["response"]
        assert row["latency_ms"] >= 0 and row["recorded_at_utc"].endswith("+00:00")

    # Same process, same wrapper: every request is now a hit.
    again = run_cases(backend)
    assert inner.calls == len(first_calls) and backend.store_hits == len(first_calls)
    assert comparable(again) == comparable(first)

    # New process stand-in: a fresh store object read from disk and a factory that must not run.
    replay = wrap(PredictionStore(tmp_path, REQUESTED, mode="read-only"))
    second = run_cases(replay)
    assert comparable(second) == comparable(first)
    assert replay.inner is None and replay.live_requests == 0 and replay.live_seconds == 0.0
    assert replay.store_hits == len(first_calls) and replay.store_misses == 0
    for call, row in zip(models(second), stored):
        block = call["response"]["prediction_store"]
        assert block == {"hit": True, "identity": replay.store.id12,
                         "first_recorded_at_utc": row["recorded_at_utc"], "first_latency_ms": row["latency_ms"]}
        assert call["response"]["metadata"] == row["response"]["metadata"]
        assert call["backend"] == "recorded_test_double" and call["is_model"] is True
    described = replay.describe()
    assert described["kind"] == "recorded_test_double" and described["is_model"] is True
    assert described["prediction_store"]["store_hits"] == len(first_calls)
    assert described["prediction_store"]["live_requests"] == 0 and described["inner"] is None
    assert described["prediction_store"]["observed"]["gpu_name"] == "Test GPU"
    assert described["prediction_store"]["requested"] == REQUESTED
    assert len(lines(backend.store)) == len(first_calls)       # a replay appends nothing


def test_wrapped_run_equals_the_plain_backend_run(tmp_path):
    plain = run_cases(Double())
    stored = run_cases(wrap(PredictionStore(tmp_path, REQUESTED), Double()))
    assert comparable(stored) == comparable(plain)
    assert [row["realized_saving"] for row in stored] == [row["realized_saving"] for row in plain]


def test_backend_without_response_evidence_still_gets_the_store_block(tmp_path):
    backend = wrap(PredictionStore(tmp_path, REQUESTED), Double(with_response=False))
    first = models(run_cases(backend, 2))
    assert all(set(call["response"]) == {"prediction_store"} and
               {k: v for k, v in call["response"]["prediction_store"].items() if k != "latency_ms"}
               == {"hit": False, "identity": backend.store.id12} for call in first)
    assert all(row["response"] is None for row in lines(backend.store)) and backend.store.observed == {}
    second = models(run_cases(wrap(PredictionStore(tmp_path, REQUESTED, mode="read-only")), 2))
    assert all(set(call["response"]) == {"prediction_store"} and call["response"]["prediction_store"]["hit"]
               for call in second)


@pytest.mark.parametrize("mode", ["error", "nan", "out_of_range", "missing_final", "interrupt"])
def test_failed_invalid_or_interrupted_estimate_is_not_stored_and_stops_for_review(tmp_path, mode):
    inner = Double(fail_on={3}, mode=mode)
    backend = wrap(PredictionStore(tmp_path, REQUESTED), inner)
    rows = run_cases(backend, 3)
    calls = models(rows)
    failed = [call for call in calls if call["status"] != "ok"]
    assert len(failed) == 1 and failed[0]["status"] == ("interrupted" if mode == "interrupt" else "error")
    assert "predictions" not in failed[0]
    review = [row for row in rows if row["status"] == "needs_review"]
    assert len(review) == 1 and review[0]["internal_record"][-1]["action"] == "human_review"
    assert review[0]["internal_record"][-1]["chosen"] is None
    assert review[0]["interrupted"] is (mode == "interrupt")
    # The failed attempt is a live request that left no line and no value.
    block = dict(failed[0]["response"]["prediction_store"])
    assert isinstance(block.pop("latency_ms"), float)
    assert block == {"hit": False, "identity": backend.store.id12, "stored": False}
    assert ("metadata" in failed[0]["response"]) is (mode not in ("error", "interrupt"))
    assert failed[0]["request_sha256"] not in backend.store
    assert backend.live_requests == inner.calls == len(calls) and backend.stored_answers == inner.calls - 1
    assert len(backend.store) == len(lines(backend.store)) == inner.calls - 1
    assert all(row["request_sha256"] != failed[0]["request_sha256"] for row in lines(backend.store))

    # A person re-runs: only the requests without a stored answer reach the backend.
    reference = run_cases(Double(), 3)
    healthy = Double()
    resumed = wrap(PredictionStore(tmp_path, REQUESTED), healthy)
    assert comparable(run_cases(resumed, 3)) == comparable(reference)
    unique = len({call["request_sha256"] for call in models(reference)})
    assert (inner.calls - 1) + healthy.calls == unique == len(resumed.store)
    assert resumed.store_hits == inner.calls - 1 and resumed.live_requests == healthy.calls


def test_runtime_change_between_sessions_is_refused_before_anything_is_written(tmp_path):
    run_cases(wrap(PredictionStore(tmp_path, REQUESTED), Double(gpu_name="Tesla T4")), 1)
    store = PredictionStore(tmp_path, REQUESTED)
    before = store.path.read_bytes(), store.identity_path.read_bytes()
    inner = Double(gpu_name="Other GPU")
    backend = wrap(store, inner)
    rows = run_cases(backend, 1, start=1)
    assert rows[0]["status"] == "needs_review" and inner.calls == 1
    call = models(rows)[0]
    assert call["status"] == "error" and call["error"].startswith("StoreIdentityMismatch")
    assert call["response"]["prediction_store"]["stored"] is False
    assert (store.path.read_bytes(), store.identity_path.read_bytes()) == before
    # Answers already stored under the first runtime are still replayed.
    replay = wrap(PredictionStore(tmp_path, REQUESTED, mode="read-only"))
    assert run_cases(replay, 1)[0]["status"] != "needs_review" and replay.store_hits > 0


def test_read_only_miss_stops_for_review_without_a_line_or_a_backend(tmp_path):
    run_cases(wrap(PredictionStore(tmp_path, REQUESTED), Double()), 1)
    store = PredictionStore(tmp_path, REQUESTED, mode="read-only")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    backend = wrap(store)
    rows = run_cases(backend, 2)
    assert rows[0]["status"] != "needs_review" and rows[1]["status"] == "needs_review"
    missed = rows[1]["internal_record"][-1]
    assert missed["action"] == "human_review" and missed["chosen"] is None
    assert missed["model"]["status"] == "error" and missed["model"]["error"].startswith("StoreMiss")
    assert missed["model"]["response"] is None and "predictions" not in missed["model"]
    assert backend.store_misses == 1 and backend.live_requests == 0 and backend.inner is None
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    empty = wrap(PredictionStore(tmp_path / "absent", REQUESTED, mode="read-only"))
    assert run_cases(empty, 1)[0]["status"] == "needs_review" and not (tmp_path / "absent").exists()
    with pytest.raises(StoreMiss):
        empty.predict(None, [], a_request())


def test_store_lines_hold_public_requests_only(tmp_path):
    backend = wrap(PredictionStore(tmp_path, REQUESTED), Double())
    run_cases(backend, 6)
    text = backend.store.path.read_text(encoding="utf-8") + backend.store.identity_path.read_text(encoding="utf-8")
    assert not any(number in text for number in ("10200000", "10845000", "10600000", "10,200,000", "10,845,000"))
    text = re.sub(r"[0-9a-f]{64}", "", text)      # digests cannot carry a value; keep short numbers from matching them
    private = default_buyer_private()
    secrets = [private.target_total_cost, private.auto_bounds.max_total_cost, private.company_bounds.max_total_cost,
               private.delay_cost_per_day, private.approval_cost]
    assert {10_200_000, 10_600_000, 10_845_000} <= set(secrets)
    for case in E.make_cases("dev", 6):
        secrets += [case.hidden.opening_profit, case.hidden.reserve_profit, case.hidden.expedite_cost_per_day]
    for secret in secrets:
        assert str(secret) not in text and f"{secret:,}" not in text, secret
    hidden_fields = {"target_total_cost", "alternative", "auto_bounds", "company_bounds", "delay_cost_per_day",
                     "approval_cost", "round_cost", "unit_cost", "capital_cost_annual", "standard_lead_days",
                     "expedite_cost_per_day", "opening_profit", "reserve_profit", "deadline_round", "style",
                     "reveals_deadline", "reveals_payment_preference", "supplier_hidden", "hidden", "truth"}
    assert not any(f'"{name}"' in text for name in hidden_fields)

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    for row in lines(backend.store):
        assert set(row) == set(PS.RECORD_KEYS) and set(row["request"]) == {"state", "questions"}
        assert set(row["request"]["state"]) == {"거래", "진행", "제안_이력"}
        assert not hidden_fields & set(keys(row))


def test_partial_answer_of_a_backend_without_final_offer_estimate_is_not_stored(tmp_path):
    backend = wrap(PredictionStore(tmp_path, {"backend": "rule_heuristic"}), HeuristicBackend(),
                   kind="rule_heuristic", is_model=False, supports_final_offer=False)
    rows = run_cases(backend, 1)
    assert rows[0]["status"] == "needs_review" and len(backend.store) == 0 and not backend.store.path.exists()
    assert "no estimate for question 'final_offer'" in models(rows)[0]["error"]


# ---- wrapper contract --------------------------------------------------------------------------------


def test_declared_values_are_the_wrapped_backends_and_are_checked_when_it_is_built(tmp_path):
    backend = wrap(PredictionStore(tmp_path, REQUESTED), Double())
    assert (backend.kind, backend.is_model, backend.supports_final_offer) == ("recorded_test_double", True, True)
    built = []

    def factory():
        built.append(Double())
        return built[-1]

    for wrong in ({"kind": "zoro_native"}, {"is_model": False}, {"supports_final_offer": False}):
        built.clear()
        mislabelled = wrap(PredictionStore(tmp_path / "wrong", REQUESTED), factory=factory, **wrong)
        with pytest.raises(ValueError, match="declared"):
            mislabelled.predict(None, [], a_request())
        with pytest.raises(RuntimeError, match="not built again"):
            mislabelled.predict(None, [], a_request())
        assert len(built) == 1 and built[0].closed and built[0].calls == 0 and mislabelled.inner is None
        assert len(mislabelled.store) == 0
    with pytest.raises(TypeError):
        StoredPredictionBackend(PredictionStore(tmp_path, REQUESTED), Double(), **DECLARED)


def test_factory_runs_once_lazily_and_a_failed_build_is_not_repeated(tmp_path):
    built = []

    def factory():
        built.append(Double())
        return built[-1]

    backend = wrap(PredictionStore(tmp_path, REQUESTED), factory=factory)
    assert built == []
    run_cases(backend, 2)
    assert len(built) == 1 and backend.inner is built[0] and backend.describe()["inner"]["kind"] == Double.kind
    backend.close()
    assert built[0].closed and backend.inner is None
    wrap(PredictionStore(tmp_path / "unused", REQUESTED)).close()      # nothing built, nothing to close

    attempts = []

    def failing():
        attempts.append(1)
        raise RuntimeError("test loading failure")

    broken = wrap(PredictionStore(tmp_path / "broken", REQUESTED), factory=failing)
    rows = run_cases(broken, 2)
    assert [row["status"] for row in rows] == ["needs_review", "needs_review"] and attempts == [1]
    errors = [call["error"] for call in models(rows)]
    assert "test loading failure" in errors[0] and "not built again" in errors[1]
    assert all(call["response"] is None for call in models(rows))
    assert broken.live_requests == 0 and len(broken.store) == 0

    # An interrupted build is the operator's stop, not a failed build: it is reported and may be tried again.
    loads = []

    def interrupted_once():
        loads.append(1)
        if len(loads) == 1:
            raise KeyboardInterrupt("test interruption while loading")
        return Double()

    stopped = wrap(PredictionStore(tmp_path / "stopped", REQUESTED), factory=interrupted_once)
    first, second = run_cases(stopped, 2)
    assert first["interrupted"] and first["status"] == "needs_review" and second["status"] != "needs_review"
    assert len(loads) == 2 and len(stopped.store) == stopped.live_requests > 0


def test_reset_response_is_safe_before_the_wrapped_backend_exists(tmp_path):
    backend = wrap(PredictionStore(tmp_path, REQUESTED), Double())
    backend.last_response = {"stale": True}
    reset_response(backend)
    assert backend.last_response is None and backend.inner is None
    case = E.make_cases("dev", 1)[0]
    state = E.initial_state(case.case_id, case.hidden)
    candidates = policy.generate_candidates(state, default_buyer_private(), PolicyConfig())
    request = policy.build_decision_request(state, candidates)
    predicted = backend.predict(state, candidates, request)
    assert predicted == KeywordBackend().predict(state, candidates, request)
    assert backend.inner.last_response is not None
    reset_response(backend)
    assert backend.last_response is None and backend.inner.last_response is None
    assert backend.predict(state, candidates, request) == predicted and backend.store_hits == 1
    assert backend.inner.calls == 1 and backend.inner.last_response is None


# ---- module rules ------------------------------------------------------------------------------------


def test_module_says_it_was_not_run_on_a_gpu_and_imports_no_private_or_hidden_type():
    source = MODULE.read_text(encoding="utf-8")
    header = ast.get_docstring(ast.parse(source))
    assert "NOT EXECUTED ON A GPU as of 2026-11-18" in header and "test doubles" in header
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    forbidden = {"BuyerPrivate", "SupplierHidden", "contracts", "economics", "simulator", "metrics", "probe",
                 "evaluation", "policy", "session"}
    assert not names & forbidden
    assert "eval" not in {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant)}
