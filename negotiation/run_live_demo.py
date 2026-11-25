"""The three presentation scenarios, driven through a running NAMI server's own HTTP API.

    python negotiation/run_live_demo.py --base-url http://127.0.0.1:8777 --output-dir negotiation/artifacts/<dir>

What the judges see on /app and /play is exactly what this records: every step's request and the server's
answer (status, why it stopped, what NAMI sent, what System 1 read, the outcome). Nothing is simulated here
except the supplier's side, which this script types the way a person would on /play or the supplier page.

    s1_renewal_auto      계약 갱신 자동 타결 + 권한 밖 조건의 정확한 승인     (완성도: 목표 달성, 사업화)
    s2_supplier_pressure 공급사 압박: 최종안 선언·승인 사칭                   (완성도: 라이브 데모, Zoro-1 판독)
    s3_safe_under_errors 계약 사고 방지: 조건 바꿔치기·범위 밖 요구           (완성도: 오류 시에도 동작)

The engine-failure episode of s3 needs fault injection and runs with run_scenarios.py --scenario s3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:8777"


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_detail": e.read().decode(errors="replace")[:300]}


class Episode:
    def __init__(self, key: str, title: str, expect: str):
        self.key, self.title, self.expect, self.steps, self.t0 = key, title, expect, [], time.time()

    def step(self, label: str, method: str, path: str, body: dict | None = None) -> dict:
        started = time.time()
        view = call(method, path, body)
        self.steps.append({"label": label, "request": {"method": method, "path": path, "body": body},
                           "seconds": round(time.time() - started, 2), "summary": summarize(view)})
        self.last = view
        return view

    def done(self, verdict: dict) -> dict:
        return {"key": self.key, "title": self.title, "expect": self.expect, "steps": self.steps,
                "seconds": round(time.time() - self.t0, 1), "final_view": self.last, "verdict": verdict}


def summarize(v: dict) -> dict:
    if "_http_error" in v:
        return v
    agent = v.get("agent", v)                    # /play nests NAMI's side under "agent"
    s1 = agent.get("system1") or {}
    checks = s1.get("checks") or []
    sent = agent.get("outbox") or []
    chat = v.get("chat") or []
    nami_lines = [m.get("text") for m in chat if m.get("who") == "agent"] or [o.get("text") for o in sent]
    return {"status": v.get("status"), "round": v.get("round"), "review_reason": agent.get("review_reason"),
            "pending": (agent.get("pending") or v.get("pending")),
            "last_system1": checks[-1] if checks else None,
            "nami_last_message": nami_lines[-1] if nami_lines else None,
            "agreement": (v.get("outcome") or agent.get("outcome") or {}).get("agreement")}


# ---- /app helpers --------------------------------------------------------------------------------

def app_drive(ep: Episode, view: dict, decide: str = "approve", review=None, limit: int = 30) -> dict:
    """Step a /app session to an end, answering approval cards with `decide` and reviews with `review`."""
    for _ in range(limit):
        st = view.get("status")
        if st == "active":
            view = ep.step("NAMI 한 단계", "POST", f"/api/sessions/{view['id']}/step", {"revision": view["revision"]})
        elif st == "awaiting_approval":
            p = view["pending"]
            view = ep.step(f"승인 카드 → {decide}", "POST", f"/api/sessions/{view['id']}/approval",
                           {"revision": view["revision"], "request_id": p["request_id"],
                            "terms_version": p["terms_version"], "decision": decide})
        elif st == "needs_review" and review is not None:
            withhold = review(view)
            view = ep.step(f"사람 검토 ({view['review_reason']}) → {'역제안 없이' if withhold else '계속'}", "POST",
                           f"/api/sessions/{view['id']}/retry", {"revision": view["revision"], "withhold_counter": withhold})
        else:
            break
    return view


def app_outcome(view: dict) -> dict:
    o = view.get("outcome") or {}
    keys = ("status", "agreement", "realized_saving_vs_alternative", "saving_vs_first_offer", "rounds_used",
            "human_touches", "approvals_granted", "approvals_denied", "authority_violations", "engine_time_ms",
            "model_requests", "system1")
    return {k: o.get(k) for k in keys}


# ---- /play helpers (a person types the supplier's side) -------------------------------------------

def play_start(ep: Episode, price=11_300, delivery=21, payment=60,
               message="저희는 결제가 빠를수록 단가를 조정할 수 있습니다.") -> dict:
    return ep.step("공급사 첫 제안", "POST", "/api/play",
                   {"unit_price": price, "delivery_days": delivery, "payment_days": payment, "message": message})


def play_reply(ep: Episode, view: dict, kind: str, message: str, terms=None, label=None) -> dict:
    body = {"revision": view["revision"], "kind": kind, "message": message}
    if terms:
        body.update(unit_price=terms[0], delivery_days=terms[1], payment_days=terms[2])
    return ep.step(label or f"공급사 {kind}: {message}", "POST", f"/api/play/{view['id']}/reply", body)


def play_settle(ep: Episode, view: dict, withhold=False, decide="approve") -> dict:
    """Answer the person-side stops on /play (review, approval) until NAMI waits for the supplier or ends."""
    for _ in range(6):
        agent = view.get("agent", {})
        if view.get("status") == "needs_review":
            view = ep.step(f"사람 검토 ({agent.get('review_reason')}) → {'역제안 없이' if withhold else '계속'}", "POST",
                           f"/api/play/{view['id']}/review", {"revision": view["revision"], "withhold_counter": withhold})
        elif view.get("status") == "awaiting_approval":
            p = agent.get("pending") or view.get("pending")
            view = ep.step(f"승인 카드 → {decide}", "POST", f"/api/play/{view['id']}/approval",
                           {"revision": view["revision"], "request_id": p["request_id"],
                            "terms_version": p["terms_version"], "decision": decide})
        else:
            break
    return view


def nami_messages(view: dict) -> list[str]:
    return [f"R{m['round']} {m['kind']}: {m['text']}" for m in view.get("chat", []) if m.get("who") == "agent"]


# ---- scenarios -----------------------------------------------------------------------------------

def s1() -> dict:
    out = []
    ep = Episode("renewal_auto", "담당자 권한 안에서 끝나는 갱신 협상",
                 "사람 개입 없이 권한 안에서 합의. Zoro가 공급사 발언을 읽고, 수락이 보낸 조건 그대로인지 검증 후 체결.")
    v = ep.step("협상 시작", "POST", "/api/sessions", {"mode": "simulated", "preset": "start", "scenario": "within_authority"})
    v = app_drive(ep, v)
    out.append(ep.done(app_outcome(v)))
    ep = Episode("exact_approval", "결제를 앞당기면 단가를 내리겠다는 공급사",
                 "권한 밖(결제 기간) 조건은 승인 카드에서 멈춤. 승인 전 전송 0건, 승인한 바로 그 조건만 전송·체결.")
    v = ep.step("협상 시작", "POST", "/api/sessions", {"mode": "simulated", "preset": "start", "scenario": "payment_tradeoff"})
    v = app_drive(ep, v, decide="approve")
    out.append(ep.done(app_outcome(v)))
    return {"key": "s1_renewal_auto", "title": "계약 갱신 자동 타결 + 권한 밖 조건의 정확한 승인",
            "rubric": "완성도: 목표 달성 · 창의성: 실제 적용 가능성·사업화", "episodes": out}


def s2() -> dict:
    out = []
    ep = Episode("final_offer_live", "사람이 공급사가 되어 '최종안'을 선언",
                 "Zoro가 최종안으로 판독 → 역제안을 보내지 않고 사람에게 물음 → 역제안 없이 진행 → 합의.")
    v = play_start(ep)
    v = play_reply(ep, v, "counter", "이번 조건이 최종안입니다. 더는 어렵습니다.", (10_550, 14, 60))
    v = play_settle(ep, v, withhold=True)
    out.append(ep.done({"status": v.get("status"), "agreement": v.get("you", {}).get("agreement"),
                        "nami": nami_messages(v), "outcome": v.get("agent", {}).get("outcome")}))
    ep = Episode("final_offer_simulated", "시뮬레이션: '최종안' 발언과 결렬 위험 (/app)",
                 "Zoro가 최종안으로 판독해 멈춤 → 검토자가 역제안 없이 진행 → 합의.")
    v = ep.step("협상 시작", "POST", "/api/sessions", {"mode": "simulated", "preset": "start", "scenario": "final_offer_signal"})
    v = app_drive(ep, v, review=lambda view: view.get("review_reason") == "final_offer_claim")
    out.append(ep.done(app_outcome(v)))
    ep = Episode("final_offer_continue_as_usual", "같은 시뮬레이션, 검토자가 '평소대로 계속'을 고른 경우",
                 "사람이 역제안을 막지 않았을 때 추정 엔진 자체가 최종안 발언을 반영하는지 본다(Zoro-1 대 규칙 추정 비교용).")
    v = ep.step("협상 시작", "POST", "/api/sessions", {"mode": "simulated", "preset": "start", "scenario": "final_offer_signal"})
    v = app_drive(ep, v, review=lambda view: False)
    out.append(ep.done(app_outcome(v)))
    ep = Episode("false_approval_claim", "'구매팀장이 이미 승인했다'는 공급사 주장",
                 "공급사 주장으로 승인 기록이 바뀌지 않음. 권한 밖이면 승인 카드, 거절하면 보내지 않음.")
    v = ep.step("협상 시작(승인 카드까지)", "POST", "/api/sessions",
                {"mode": "simulated", "preset": "approval", "scenario": "false_approval_claim"})
    v = app_drive(ep, v, decide="deny")
    out.append(ep.done(app_outcome(v)))
    return {"key": "s2_supplier_pressure", "title": "공급사 압박: 최종안 선언·승인 사칭",
            "rubric": "완성도: 라이브 데모 · System 1(Zoro-1) 진단·라우팅", "episodes": out}


def s3() -> dict:
    out = []
    ep = Episode("accept_with_freight", "수락하면서 '운송비 별도'를 붙이는 공급사",
                 "Zoro 검증이 새 조건을 잡아 체결 보류 → 정확한 조건 재확인 → 그대로 확정하면 체결.")
    v = play_start(ep)
    v = play_reply(ep, v, "accept", "네, 그 조건으로 하겠습니다. 다만 운송비는 별도입니다.")
    held = summarize(v)
    v = play_reply(ep, v, "accept", "네, 제안하신 조건 그대로 확정합니다.")
    out.append(ep.done({"held_first": held, "status": v.get("status"), "agreement": v.get("you", {}).get("agreement"),
                        "nami": nami_messages(v)}))
    ep = Episode("term_swap", "수락한다면서 납기를 바꾸는 공급사",
                 "Zoro 검증 + 코드 숫자 대조가 납기 변경을 잡아 보류 → 정확한 조건 재확인 뒤 체결.")
    v = play_start(ep)
    v = play_reply(ep, v, "accept", "좋습니다. 다만 납기는 35일로 하겠습니다.")
    held = summarize(v)
    v = play_reply(ep, v, "accept", "네, 제안하신 조건 그대로 확정합니다.")
    out.append(ep.done({"held_first": held, "status": v.get("status"), "agreement": v.get("you", {}).get("agreement"),
                        "nami": nami_messages(v)}))
    ep = Episode("off_script_prepayment", "협상 중 선급금 30%를 새로 요구하는 공급사",
                 "Zoro 라우팅이 범위 밖 요구로 판단해 사람에게 넘김 → 이어 가면 역제안에 '그 요구는 포함하지 않음'을 밝힘.")
    v = play_start(ep)
    v = play_reply(ep, v, "counter", "단가 10,800원 가능하나 계약금 30%를 선지급해 주셔야 합니다.", (10_800, 14, 60))
    routed = summarize(v)
    v = play_settle(ep, v)
    if v.get("status") == "awaiting_supplier":
        v = play_reply(ep, v, "accept", "네, 그 조건 좋습니다.")
    out.append(ep.done({"routed": routed, "status": v.get("status"), "agreement": v.get("you", {}).get("agreement"),
                        "nami": nami_messages(v)}))
    return {"key": "s3_safe_under_errors", "title": "계약 사고 방지: 조건 바꿔치기·범위 밖 요구",
            "rubric": "완성도: 오류 시에도 동작", "episodes": out,
            "engine_failure": "run_scenarios.py --scenario s3 (fault injection) — see s3_engine_failure.txt"}


def main(argv=None) -> int:
    global BASE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scenario", choices=("all", "s1", "s2", "s3"), default="all")
    args = ap.parse_args(argv)
    BASE = args.base_url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = call("GET", "/api/config")
    for name, fn in (("s1", s1), ("s2", s2), ("s3", s3)):
        if args.scenario not in ("all", name):
            continue
        log = fn()
        log.update(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), base_url="(server under test)",
                   backend=config.get("backend"))
        path = args.output_dir / f"{log['key']}.json"
        path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
