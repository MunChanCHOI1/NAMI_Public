"""Transport contracts only; these tests never use real Zoro weights or a GPU.

The --reasoning tests below were written on 2026-11-18 against an injected transport.
Nothing here has been executed against a real decision server or on a GPU.
"""
import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from negotiation_agent import evaluation, policy, probe
from negotiation_agent import provenance as V
from negotiation_agent.backends import ZoroHttpBackend
from negotiation_agent.provenance import REPOSITORY_ROOT, request_sha256
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.simulator import default_buyer_private
from sessionguard.backends import zoro_http
from sessionguard.contracts import InputOverflowError


REASONING_TEXT = "합성 추론 문장"
PINNED_CODEC = REPOSITORY_ROOT / ".cache/trida-stack/inference/zoro-1/decide_request.py"


def inject_transport(monkeypatch, seen, response):
    """Replace only the HTTP transport; the client code under test is the real one."""
    def server(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "reported-checkpoint"}]})
        assert request.url.path == "/v1/decide"
        sent = json.loads(request.content)
        seen.append(sent)
        body = dict(response)
        if sent.get("reasoning"):          # as serve_decide.py does: reasoning text only when asked
            body.setdefault("reasoning", REASONING_TEXT)
            body.setdefault("reasoning_ms", 40.0)
        return httpx.Response(200, json=body)

    original = zoro_http.HttpDecisionBackend
    monkeypatch.setattr(zoro_http, "HttpDecisionBackend", lambda *a, **kw: original(
        *a, **kw, transport=httpx.MockTransport(server)))


@pytest.fixture
def wire(monkeypatch):
    """(seen payloads, mutable server response) behind a fake codec and an injected transport."""
    seen = []
    response = {"answers": {"C1": {"type": "noul", "noul": 0.1234},
                             "final_offer": {"type": "noul", "noul": 0.9}}}
    monkeypatch.setattr(zoro_http, "load_official_codec", lambda _: SimpleNamespace(
        render=lambda x: json.dumps(x, ensure_ascii=False), MAX_STATE=7552,
        with_reasoning=lambda request, text: {**request, "state": {**request["state"], "reasoning": text}},
        _SPECIAL_RE=re.compile(r"<\|([A-Za-z0-9_]+)\|>")))
    inject_transport(monkeypatch, seen, response)
    return seen, response


@pytest.fixture
def official(wire):
    seen, response = wire
    backend = ZoroHttpBackend("http://127.0.0.1:30001", ".cache/trida-stack", 1.0)
    yield backend, seen, response
    backend.close()


def cli_args(*extra):
    parser = argparse.ArgumentParser()
    probe.add_backend_arguments(parser)
    return parser.parse_args(list(extra))


HTTP_FLAGS = ("--backend", "zoro-http", "--url", "http://localhost:30001",
              "--decision-temperature", "1", "--request-timeout", "600")


def request():
    return {"state": {"public_message": "합성 공개 제안"}, "questions": {
        "C1": {"type": "noul", "question": "수락?"},
        "final_offer": {"type": "noul", "question": "최종안?"}}}


def test_negotiation_uses_official_envelope_and_keeps_identity_unverified(official):
    backend, seen, _ = official
    value = backend.predict(None, None, request())
    assert value == {"C1": 0.1234, "final_offer": 0.9}
    assert seen == [{"request": request(), "reasoning": False, "temperature": 1.0}]
    meta = backend.last_response["metadata"]
    assert meta["request_sha256"] == request_sha256(request())
    assert meta["served_model_ids"] == ["reported-checkpoint"]
    assert not meta["server_identity_verified_by_client"]
    assert not meta["official_sglang_parity_verified"]


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, "0.3"])
def test_bad_probability_never_reuses_previous_response(official, bad):
    backend, _, response = official
    backend.predict(None, None, request())
    response["answers"]["C1"]["noul"] = bad
    with pytest.raises(ValueError):
        backend.predict(None, None, request())
    assert backend.last_response is None


def test_oversized_public_state_is_rejected_before_http(official):
    backend, seen, _ = official
    data = request()
    data["state"] = "가" * 3000
    with pytest.raises(InputOverflowError):
        backend.predict(None, None, data)
    assert seen == []


def test_cli_routes_official_backend_and_records_temperature(monkeypatch):
    parser = argparse.ArgumentParser()
    probe.add_backend_arguments(parser)
    args = parser.parse_args(["--backend", "zoro-http", "--url", "http://localhost:30001",
                              "--decision-temperature", "1", "--request-timeout", "600"])
    monkeypatch.setattr(probe, "ZoroHttpBackend", lambda *items: items)
    assert probe.make_backend(args) == ("http://localhost:30001", ".cache/trida-stack", 1.0, 600.0)


# ---- --reasoning (default off): one sampled generation per answer, zoro-http only ------------------------

def test_reasoning_changes_only_the_envelope_flag_and_keeps_the_generated_text(wire):
    seen, _ = wire
    plain = ZoroHttpBackend("http://127.0.0.1:30001", ".cache/trida-stack", 1.0)
    reasoned = ZoroHttpBackend("http://127.0.0.1:30001", ".cache/trida-stack", 1.0, reasoning=True)
    try:
        assert plain.predict(None, None, request()) == reasoned.predict(None, None, request())
        assert seen == [{"request": request(), "reasoning": False, "temperature": 1.0},
                        {"request": request(), "reasoning": True, "temperature": 1.0}]
        assert "reasoning" not in plain.last_response
        assert reasoned.last_response["reasoning"] == REASONING_TEXT
        meta = reasoned.last_response["metadata"]
        assert meta["reasoning_requested"] and meta["generation_used"]
        assert meta["request_sha256"] == request_sha256(request())       # the request itself is not rewritten
        assert meta["state_with_reasoning_token_upper_bound"] > plain.last_response["metadata"]["state_token_upper_bound"]
        assert not plain.last_response["metadata"]["reasoning_requested"]
        # describe(): the default description is exactly what it was before the flag existed.
        assert plain.describe()["configuration"]["reasoning"] is False
        assert set(plain.describe()) == {"kind", "is_model", "configuration", "last_response_metadata",
                                         "server_identity_verified_by_client", "notice"}
        described = reasoned.describe()
        assert described["configuration"]["reasoning"] is True
        assert described["reasoning"]["requested"] is True and "single sample" in described["reasoning"]["notice"]
        assert described["kind"] == plain.describe()["kind"] == "official_decide_http"
    finally:
        plain.close()
        reasoned.close()


def test_reasoning_requested_without_returned_text_is_an_error_not_a_plain_answer(wire):
    seen, response = wire
    backend = ZoroHttpBackend("http://127.0.0.1:30001", ".cache/trida-stack", 1.0, reasoning=True)
    try:
        backend.predict(None, None, request())
        response["reasoning"] = None                                     # the server now returns no text
        with pytest.raises(ValueError, match="Reasoning was requested"):
            backend.predict(None, None, request())
        assert backend.last_response is None and len(seen) == 2
    finally:
        backend.close()


def test_cli_reasoning_flag_reaches_the_wire_and_the_recorded_configuration(wire):
    seen, _ = wire
    args = cli_args(*HTTP_FLAGS, "--reasoning")
    assert args.reasoning is True and cli_args(*HTTP_FLAGS).reasoning is False
    recorded = V.backend_execution_config(args)
    assert recorded["reasoning"] is True
    backend = probe.make_backend(args)
    try:
        backend.predict(None, None, request())
        assert seen == [{"request": request(), "reasoning": True, "temperature": 1.0}]
        assert backend.describe()["configuration"] == recorded
    finally:
        backend.close()


def test_cli_without_the_flag_sends_and_records_exactly_what_it_did_before(wire):
    seen, _ = wire
    args = cli_args(*HTTP_FLAGS)
    assert V.backend_execution_config(args) == {
        "backend": "zoro-http", "endpoint_origin": "http://localhost:30001",
        "official_source_dir": str(Path(".cache/trida-stack").expanduser().resolve()),
        "decision_temperature": 1.0, "request_timeout": 600.0, "reasoning": False}
    backend = probe.make_backend(args)
    try:
        backend.predict(None, None, request())
        assert seen == [{"request": request(), "reasoning": False, "temperature": 1.0}]
        assert "reasoning" not in backend.describe()
    finally:
        backend.close()


def test_cli_passes_the_keyword_only_when_the_flag_is_set(monkeypatch):
    monkeypatch.setattr(probe, "ZoroHttpBackend", lambda *items, **options: (items, options))
    expected = ("http://localhost:30001", ".cache/trida-stack", 1.0, 600.0)
    assert probe.make_backend(cli_args(*HTTP_FLAGS)) == (expected, {})
    assert probe.make_backend(cli_args(*HTTP_FLAGS, "--reasoning")) == (expected, {"reasoning": True})


@pytest.mark.parametrize("flags", [
    ("--backend", "constant"), ("--backend", "heuristic"), ("--backend", "keyword"), (),
    ("--backend", "zoro", "--original-checkpoint", "a", "--exported-checkpoint", "b"),
    ("--backend", "systemone", "--url", "http://localhost:30001"),
])
def test_reasoning_is_refused_for_every_other_backend_before_anything_is_built_or_recorded(flags, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("no backend may be built when --reasoning is refused")

    for name in ("ConstantBackend", "HeuristicBackend", "KeywordBackend", "ZoroNativeBackend",
                 "SystemOneBackend", "ZoroHttpBackend"):
        monkeypatch.setattr(probe, name, forbidden)
    args = cli_args(*flags, "--reasoning")
    with pytest.raises(SystemExit, match="only valid with --backend zoro-http"):
        probe.make_backend(args)
    with pytest.raises(SystemExit, match="only valid with --backend zoro-http"):
        V.backend_execution_config(args)          # a freeze or a result cannot record an ignored flag


def test_help_text_says_a_reasoning_run_is_one_sample():
    parser = argparse.ArgumentParser()
    probe.add_backend_arguments(parser)
    text = " ".join(parser.format_help().split())
    assert "--reasoning" in text and "zoro-http only" in text
    assert "sampled generation" in text and "ONE sample" in text and "default: off" in text


def test_official_http_client_is_part_of_the_hashed_inputs():
    assert "src/sessionguard/backends/zoro_http.py" in V.SHARED_INPUTS
    hashes = V.input_hashes()
    assert hashes["src/sessionguard/backends/zoro_http.py"] == V.file_sha256(Path(zoro_http.__file__))
    assert len(V.SHARED_INPUTS) == len(set(V.SHARED_INPUTS))


@pytest.mark.skipif(not PINNED_CODEC.is_file(), reason="pinned official sources are not present in this checkout")
def test_default_first_request_passes_the_pinned_codec_unchanged_with_and_without_reasoning(monkeypatch):
    """Real pinned request codec (no fake), real default request; only the transport is injected."""
    seen, response = [], {"answers": {}}
    inject_transport(monkeypatch, seen, response)
    case = evaluation.make_cases("dev", 1)[0]

    class Capture:
        kind, is_model, supports_final_offer = "capture_test_double", True, True

        def __init__(self):
            self.requests = []

        def predict(self, state, candidates, request):
            self.requests.append(json.loads(json.dumps(request)))
            raise RuntimeError("request captured; no estimate is produced")

        def describe(self):
            return {"kind": self.kind, "is_model": True}

    capture = Capture()
    evaluation.run_case(case, default_buyer_private(), capture, policy.decide, policy.PolicyConfig())
    first = capture.requests[0]
    assert request_sha256(first) == "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"
    assert FINAL_OFFER_QID in first["questions"]
    response["answers"] = {qid: {"type": "noul", "noul": 0.25} for qid in first["questions"]}
    bounds = {}
    for reasoning in (False, True):
        backend = ZoroHttpBackend("http://127.0.0.1:30001", str(REPOSITORY_ROOT / ".cache/trida-stack"), 1.0,
                                  reasoning=reasoning)
        try:
            values = backend.predict(None, None, first)
            meta = backend.last_response["metadata"]
        finally:
            backend.close()
        assert set(values) == set(first["questions"])
        assert seen[-1] == {"request": first, "reasoning": reasoning, "temperature": 1.0}
        assert meta["state_token_limit"] == 7552 and meta["reasoning_requested"] is reasoning
        assert meta["state_token_upper_bound"] < meta["state_token_limit"]
        bounds[reasoning] = meta["state_token_upper_bound"]
    # Room for the generated text is reserved before sending and still fits the official state limit.
    assert bounds[True] - bounds[False] == zoro_http.REASONING_STATE_ALLOWANCE
