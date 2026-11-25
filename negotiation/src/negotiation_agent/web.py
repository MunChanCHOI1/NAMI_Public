"""Synthetic demo API; policy and authority remain in NegotiationSession.

The operator chooses the estimate backend, the replay files and the save directory on the
command line. The browser can choose only among the fixed synthetic scenarios below; it
never supplies a file path, a model URL or a supplier parameter. It may set our own side's
criteria (target, negotiator limits, alternative quote) within validated bounds; those stay in
BuyerPrivate and never reach a backend request or an outgoing message.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .backends import HeuristicBackend
from .contracts import BuyerPrivate, PublicState, TermBounds
from .economics import HardConstraintViolation, alternative_total_cost, buyer_total_cost, classify_authority
from .scenarios import S2_SEED_BY_BACKEND
from .session import MAX_MESSAGE_CHARS, NegotiationSession, SimulatedSupplier
from .simulator import ITEM, MAX_ROUNDS, QUANTITY, default_buyer_private, initial_state, make_supplier
from .system1 import QID_ACCEPT_EXACT, QID_OFF_SCRIPT, gate_for

STATIC = Path(__file__).resolve().parents[2] / "static"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
HTTP_KINDS = ("official_decide_http", "systemone_http")


@dataclass(frozen=True)
class Scenario:
    """A fixed synthetic supplier for the demo. The text is for the audience, never for the backend."""

    key: str
    title: str
    summary: str
    watch: str
    seed: int
    style: str
    payment_sensitive: bool
    reveal: bool
    feasible: bool = True
    claims_approval: bool = False


# Seeds 1000-1099 are outside the development (1100+), calibration (2000+) and evaluation (3000+)
# splits, so replaying a scenario never touches an evaluation case.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("payment_tradeoff", "결제 조건 교환과 내부 승인",
             "자금 회전이 급한 공급사가 결제를 앞당기면 단가를 더 조정할 수 있다고 말합니다.",
             "담당자 한도를 넘는 조건(예: 납품 즉시 결제)은 에이전트가 보내지 않고 승인 카드를 만듭니다.",
             1015, "linear", True, True),
    Scenario("false_approval_claim", "‘이미 승인받았다’는 공급사 주장",
             "공급사가 매 발언에 귀사 구매팀장이 이미 승인했다고 적습니다.",
             "상대 발언은 승인 장부를 바꾸지 못합니다. 한도를 넘는 조건에는 카드가 그대로 나옵니다.",
             1000, "boulware", True, True, claims_approval=True),
    Scenario("final_offer_signal", "‘최종안’ 발언과 결렬 위험",
             "마감이 가까운 공급사가 2라운드에 이번 조건이 최종안이라고 말합니다.",
             "발언을 읽지 않는 규칙 추정은 역제안을 보내 결렬됩니다. 발언을 읽는 추정은 수락 쪽으로 기울 수 있습니다.",
             1073, "deadline", False, True),
    Scenario("no_deal", "대안 견적을 넘지 못하는 공급사",
             "원가가 높아 대안 견적보다 유리한 조건을 낼 수 없는 공급사입니다.",
             "에이전트는 대안보다 불리한 조건에 합의하지 않고 대안 견적으로 끝냅니다.",
             1001, "linear", False, True, feasible=False),
    Scenario("within_authority", "담당자 권한 안에서 끝나는 거래",
             "여유 폭이 있는 공급사와 일반 결제 조건으로 협상합니다.",
             "한도 안의 조건은 승인 카드 없이 실행하고 실행 직전에 권한을 다시 확인합니다.",
             1001, "linear", False, True),
)
SCENARIO_BY_KEY = {s.key: s for s in SCENARIOS}
DEFAULT_SCENARIO = SCENARIOS[0].key
# Seed 1015 was picked with the CPU reference. Real Zoro settles it inside the negotiator's limits, so the
# approval card this scenario promises never appears (seen on a Brev L40S and a local server, 2026-11-20).
# A model backend plays the same supplier settings with the seed the recorded real-Zoro sweep picked for
# the approval scenario (scenarios.S2_SEED_BY_BACKEND). The sweep ran the native backend only.
MODEL_SCENARIO_SEEDS = {"payment_tradeoff": S2_SEED_BY_BACKEND["zoro"]}


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuyerSettings(Payload):
    """Our side's criteria set by a person on the screen. Unit prices are per item."""

    target_unit_price: StrictInt = Field(ge=1_000, le=100_000)
    max_unit_price: StrictInt = Field(ge=1_000, le=100_000)          # negotiator may execute alone up to here
    alternative_unit_price: StrictInt = Field(ge=1_000, le=100_000)  # quote we already hold; never accept worse
    max_delivery_days: StrictInt = Field(ge=1, le=30)                # 30 days is the hard requirement
    min_payment_days: StrictInt = Field(ge=0, le=120)

    @model_validator(mode="after")
    def _ordered(self):
        if not self.target_unit_price <= self.max_unit_price < self.alternative_unit_price:
            raise ValueError("목표 단가 ≤ 담당자 단가 한도 < 대안 견적 단가 순서여야 합니다.")
        return self


def buyer_private(settings: BuyerSettings | None, quantity: int = QUANTITY) -> BuyerPrivate:
    """Map the screen's criteria onto BuyerPrivate. No settings gives the stated default assumptions.

    The defaults round-trip exactly: alternative 10,900 -> company limit 10,845,000 and so on.
    The negotiator's limits never exceed the company's, so `classify_authority` stays ordered.
    """
    base = default_buyer_private()
    if settings is None:
        return base
    with_alternative = replace(base, alternative=replace(base.alternative, unit_price=settings.alternative_unit_price))
    company_total = alternative_total_cost(quantity, with_alternative) // 5_000 * 5_000
    company = replace(base.company_bounds, max_unit_price=settings.alternative_unit_price, max_total_cost=company_total)
    auto = TermBounds(settings.max_unit_price, settings.max_delivery_days, settings.min_payment_days,
                      min(settings.max_unit_price * quantity, company_total))
    return replace(with_alternative, target_total_cost=settings.target_unit_price * quantity,
                   auto_bounds=auto, company_bounds=company)


def buyer_settings_view(private: BuyerPrivate, quantity: int = QUANTITY) -> dict:
    return {"target_unit_price": private.target_total_cost // quantity,
            "max_unit_price": private.auto_bounds.max_unit_price,
            "alternative_unit_price": private.alternative.unit_price,
            "max_delivery_days": private.auto_bounds.max_delivery_days,
            "min_payment_days": private.auto_bounds.min_payment_days,
            "alternative_terms": asdict(private.alternative)}


class CreateSession(Payload):
    mode: Literal["simulated", "human"] = "simulated"
    # blank: a person types the supplier's opening offer too (human mode only)
    preset: Literal["approval", "start", "blank"] = "approval"
    scenario: str = Field(default=DEFAULT_SCENARIO, max_length=40)
    buyer: BuyerSettings | None = None


class Mutation(Payload):
    revision: StrictInt = Field(ge=0)


class ReviewMutation(Mutation):
    withhold_counter: bool = False     # after a final-offer claim: compare only accepting and ending


class ApprovalInput(Mutation):
    request_id: str = Field(max_length=100)
    terms_version: str = Field(max_length=100)
    decision: Literal["approve", "deny"]


class SupplierInput(Mutation):
    kind: Literal["accept", "counter", "walk_away"]
    unit_price: StrictInt | None = Field(default=None, ge=1, le=1_000_000)
    delivery_days: StrictInt | None = Field(default=None, ge=1, le=3650)
    payment_days: StrictInt | None = Field(default=None, ge=0, le=3650)
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


def _model_view(model: dict) -> dict:
    # The browser needs the estimate's identity, not raw engine responses or server paths.
    keys = ("backend", "is_model", "label", "request_sha256", "latency_ms", "error", "trace_id", "cached",
            "questions", "status")
    view = {k: model[k] for k in keys if k in model}
    # The question wording that was actually sent. A request is built from public state and candidate
    # terms only (requests.py), so the text carries no buyer-private limit or supplier-hidden value.
    questions = (model.get("request") or {}).get("questions")
    if isinstance(questions, dict):
        view["asked"] = [{"id": str(qid)[:40], "text": str(q.get("instructions", ""))[:600]}
                         for qid, q in list(questions.items())[:12] if isinstance(q, dict)]
    return view


def _records(records: list[dict]) -> list[dict]:
    keys = ("round", "action", "reasons", "facts", "chosen", "options", "approval_request")
    return [{**{k: row[k] for k in keys if k in row}, "model": _model_view(row.get("model", {}))}
            for row in records]


def system1_view(session: NegotiationSession) -> dict | None:
    """What the System 1 gate asked and decided, for the screen. Raw engine responses stay server-side."""
    gate = session.gate
    if gate is None:
        return None
    checks = []
    for index, c in enumerate(gate.checks):
        answers = c.get("answers") or {}
        item = {"index": index, "check": c["check"], "round": c["round"], "is_model": c["is_model"],
                "reader": c["reader"], "status": c["status"], "latency_ms": c.get("latency_ms"),
                "reasons": c.get("reasons", []), "error": c.get("error")}
        if c["check"] == "routing":
            item.update(route=c.get("route"), final=answers.get("u_final"), walkaway=answers.get("u_walkaway"),
                        concession=answers.get("u_concession"), instruction=answers.get("u_instruction"),
                        off_script=answers.get(QID_OFF_SCRIPT))
        else:
            item.update(verified=c.get("verified"), accept_exact=answers.get(QID_ACCEPT_EXACT),
                        new_condition=answers.get(QID_OFF_SCRIPT), reply=c.get("reply"),
                        number_mismatches=c["code_number_check"]["mismatches"])
        checks.append(item)
    return {"is_model": gate.reader.is_model, "reader": gate.reader.kind, "threshold": gate.threshold,
            "checks": checks, "review_reason": session.review_reason}


def backend_view(backend) -> dict:
    """What the screen may say about the estimate source. No path, URL or raw response."""
    http = backend.kind in HTTP_KINDS
    view = {"kind": backend.kind, "is_model": bool(backend.is_model),
            "evidence_class": "real_model_run" if backend.is_model else "cpu_reference_not_model_evidence",
            "reads_supplier_messages": bool(backend.is_model) or backend.kind == "rule_keyword",
            # An HTTP client cannot prove which weights or engine answered.
            "server_identity_verified_by_client": False if http else None}
    if http:
        try:
            meta = backend.describe().get("last_response_metadata") or {}
        except Exception:  # describing must never break the screen
            meta = {}
        served = meta.get("served_model_ids")
        if isinstance(served, list):
            view["served_model_ids_reported_by_server"] = [str(item)[:120] for item in served[:5]]
    return view


def _replay_files(paths) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        files += sorted(path.glob("*.json")) if path.is_dir() else [path]
    return files


def _replay_rows(data: dict) -> list[dict]:
    backend = data.get("backend", {})
    real = str(data.get("evidence_class")).startswith("real_model_run") and backend.get("is_model") is True
    approver = (data.get("config", {}).get("approver") or data.get("approver")
                or data.get("provenance", {}).get("execution_config", {}).get("approver"))
    approval_mode = {"auto": "synthetic_auto", "approve_all": "synthetic_auto",
                     "deny": "synthetic_deny", "deny_all": "synthetic_deny",
                     "human_dashboard": "human_dashboard"}.get(approver, "unspecified")
    candidates = [(row.get("case_id", str(i)), row["agent"]) for i, row in enumerate(data.get("cases", []))
                  if "agent" in row]
    if "internal_record" in data:
        candidates = [(str(data.get("case_id") or data.get("case", {}).get("seed", "record")), data)]
    result = []
    for case_id, row in candidates:
        records = row.get("internal_record", [])
        if not records:
            continue
        outcome = row.get("outcome", row)
        outcome_keys = ("status", "agreement", "rounds_used", "realized_saving_vs_alternative", "realized_saving")
        result.append({"case_id": case_id, "evidence_class": str(data.get("evidence_class")) if real else "cpu_reference_not_model_evidence",
                       "backend": {k: backend[k] for k in ("kind", "is_model") if k in backend},
                       "records": _records(records), "transcript": row.get("public_transcript", []),
                       "outcome": {k: outcome[k] for k in outcome_keys if k in outcome},
                       "approval_log": row.get("approval_log", row.get("approvals", [])),
                       "executed": row.get("executed", []), "approval_mode": approval_mode,
                       "source_commit": (data.get("provenance", {}).get("git_commit")
                                         or data.get("curation", {}).get("source_commit")),
                       "complete": data.get("complete", True), "read_only": True})
    return result


def load_replays(path) -> list[dict]:
    """Only an operator's CLI path can load a file. Never expose evaluator_only or arbitrary JSON.

    `path` is one file, one directory of saved sessions, or a list of either.
    """
    files = _replay_files(path)
    if not files:
        return []
    result: list[dict] = []
    for file in files:
        result += _replay_rows(json.loads(file.read_text(encoding="utf-8")))
    if not result:
        raise ValueError("replay requires a negotiation record or evaluation cases with internal_record")
    return result


POLICY_LABELS = {
    "agent": ("Zoro 추정 에이전트", False),
    "rule_policy": ("규칙 추정 에이전트", False),
    "fixed_concession": ("고정 양보율", False),
    "midpoint": ("중간값", False),
    "reference_true_acceptance": ("참고 상한 · 숨은 정답 사용", True),
}


def load_evidence(root: Path = ARTIFACTS) -> dict:
    """Curated numbers from the committed evidence files. Read-only; nothing is recomputed or rerun."""
    def read(relative: str):
        path = root / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    out: dict = {"available": False}
    dev = read("20261116_colab/dev30_summary.json")
    if dev:
        policies = []
        for key, (label, evaluator_only) in POLICY_LABELS.items():
            row = dev.get("policies", {}).get(key)
            if not row:
                continue
            feasible = row.get("feasible_cases", {})
            policies.append({
                "key": key, "label": label, "evaluator_only": evaluator_only,
                "mean_realized_saving": row.get("mean_realized_saving"),
                "mean_net_saving": row.get("mean_net_saving_after_process_costs"),
                "feasible_n": feasible.get("n"),
                "feasible_agreements": (round(feasible["agreement_rate"] * feasible["n"])
                                        if feasible.get("n") and feasible.get("agreement_rate") is not None else None),
                "supplier_walked": feasible.get("supplier_walked"),
                "approval_cards": row.get("process", {}).get("approval_cards"),
                "authority_violations": row.get("authority_violations"),
                "agreements_not_better_than_alternative": row.get("agreements_not_better_than_alternative"),
            })
        differences = []
        for key, row in dev.get("agent_minus", {}).items():
            saving = row.get("realized_saving", {})
            differences.append({"against": key, "label": POLICY_LABELS.get(key, (key, False))[0],
                                "evaluator_only": POLICY_LABELS.get(key, (key, False))[1],
                                "mean_difference": saving.get("mean_difference"), "ci90": saving.get("ci90"),
                                "wins": saving.get("wins"), "ties": saving.get("ties"), "losses": saving.get("losses")})
        backend = dev.get("backend", {})
        out["available"] = True
        out["development"] = {
            "evidence_class": dev.get("evidence_class"), "split": dev.get("split"),
            "cases": dev.get("config", {}).get("cases"), "trials": dev.get("curation", {}).get("trials"),
            "approver": dev.get("config", {}).get("approver"), "scope": dev.get("scope"),
            "backend": {k: backend.get(k) for k in ("kind", "model_id", "gpu_name", "dtype", "decision_temperature")},
            "requests": dev.get("actual_requests"), "questions": dev.get("actual_questions"),
            "median_latency_ms": dev.get("policies", {}).get("agent", {}).get("process", {})
                                    .get("latency_ms_per_request", {}).get("median"),
            "policies": policies, "differences": differences,
            "source_sha256": dev.get("curation", {}).get("source_sha256"),
            "heldout_used": False,
        }
    probe = read("20261116_colab/probe_summary.json")
    if probe:
        metrics = probe.get("metrics", {})
        discrimination = metrics.get("discrimination", {})
        rule = metrics.get("reference_rule_discrimination", {})
        out["candidate_probe"] = {
            "route": probe.get("verdict", {}).get("route"), "checks": probe.get("verdict", {}).get("checks"),
            "questions": probe.get("actual_questions"), "states": probe.get("scope", {}).get("states"),
            "auc_across_states": discrimination.get("auc_across_states"),
            "within_state_pair_accuracy": discrimination.get("within_state_pair_accuracy"),
            "rule_within_state_pair_accuracy": rule.get("within_state_pair_accuracy"),
            "gain_vs_rule_mean": metrics.get("decision_value", {}).get("gain_vs_rule_mean"),
            "gain_vs_rule_ci90": metrics.get("decision_value", {}).get("gain_vs_rule_ci90_by_group"),
        }
    sglang = read("20261117_kaggle_sglang/verification.json")
    if sglang:
        comparison = sglang.get("comparison", {})
        out["official_sglang"] = {
            "platform": sglang.get("platform"), "date": sglang.get("verified_at_date"),
            "sglang_version": comparison.get("official_server_evidence", {}).get("environment", {}).get("sglang"),
            "probe_questions": comparison.get("probe", {}).get("matched_questions"),
            "probe_max_abs_difference": comparison.get("probe", {}).get("maximum_absolute_probability_difference"),
            "development_cases": comparison.get("development", {}).get("cases"),
            "same_decision_paths": comparison.get("development", {}).get("same_decision_paths"),
            "same_outcomes": comparison.get("development", {}).get("same_outcomes"),
            "scope": comparison.get("scope"),
        }
    return out


class LiveSession:
    def __init__(self, mode: str, preset: str, scenario: Scenario, backend, backend_lock: Lock,
                 system1_rules: bool = False, buyer: BuyerSettings | None = None, writer=None):
        hidden = replace(make_supplier(scenario.seed, scenario.style, scenario.payment_sensitive,
                                       scenario.reveal, scenario.feasible),
                         claims_buyer_approval=scenario.claims_approval)
        self._hidden = hidden if mode == "simulated" else None
        self.private = buyer_private(buyer)
        self.backend, self.backend_lock = backend, backend_lock
        state = (PublicState(f"demo-typed-{uuid4().hex[:8]}", ITEM, QUANTITY, MAX_ROUNDS, 1, ())
                 if mode == "human" and preset == "blank" else initial_state(f"demo-{scenario.seed}", hidden))
        # The System 1 gate asks the same backend (a model) or the rule reader (a CPU reference).
        self.session = NegotiationSession(state, self.private, backend,
                                          None if mode == "human" else SimulatedSupplier(hidden),
                                          gate=gate_for(backend, force_rules=system1_rules,
                                                        stop_on_final_claim=True),
                                          writer=writer)
        self.mode, self.scenario, self.revision, self.lock = mode, scenario, 0, Lock()
        self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.supplier_token = uuid4().hex if mode == "human" else None
        with backend_lock:
            self.session.route_opening_offer()   # the opening offer gets the same System 1 look
            if mode == "simulated" and preset == "approval" and self.session.status == "active":
                self.session.run(max_steps=10)  # Stops at the first real approval gate. Never auto-approves.

    def view(self, session_id: str) -> dict:
        s = self.session
        standing = s.state.last_offer("supplier")
        total = None
        if standing is not None:
            try:
                total = buyer_total_cost(standing.terms, s.state.quantity, self.private)
            except HardConstraintViolation:
                total = None
        alternative = alternative_total_cost(s.state.quantity, self.private)
        facts = {"standing_offer_id": standing.offer_id if standing else None, "standing_total_cost": total,
                 "standing_saving_vs_alternative": alternative - total if total is not None else None,
                 "standing_authority": (classify_authority(standing.terms, s.state.quantity, self.private)
                                        if standing else None),
                 "target_total_cost": self.private.target_total_cost, "alternative_total_cost": alternative}
        backend = backend_view(self.backend)
        scenario = ({"key": self.scenario.key, "title": self.scenario.title, "summary": self.scenario.summary,
                     "watch": self.scenario.watch} if self.mode == "simulated" else None)
        return {"id": session_id, "revision": self.revision, "mode": self.mode, "status": self.status(),
                "item": s.state.item, "quantity": s.state.quantity, "round": s.state.current_round,
                "max_rounds": s.state.max_rounds, "evidence_class": backend["evidence_class"],
                "backend": backend, "scenario": scenario,
                "records": _records(s.internal_record()), "transcript": s.public_transcript(),
                "outcome": s.outcome(), "pending": asdict(s.pending) if s.pending else None,
                "facts": facts, "auto_bounds": asdict(self.private.auto_bounds),
                "approval_log": s.approval_log, "review_log": s.review_log,
                "executed": s.executed, "outbox": s.outbox, "system1": system1_view(s),
                "review_reason": s.review_reason,
                "supplier_url": f"/supplier/{self.supplier_token}" if self.supplier_token else None,
                "buyer_settings": buyer_settings_view(self.private),
                "read_only": False}

    def has_supplier_offer(self) -> bool:
        return self.session.state.last_offer("supplier") is not None

    def status(self) -> str:
        """A typed negotiation waits for the supplier's opening offer before anything can be decided."""
        return self.session.status if self.has_supplier_offer() else "awaiting_supplier"

    def public_view(self) -> dict:
        s = self.session
        status = self.status() if self.status() in ("awaiting_supplier", "awaiting_info", "agreed", "supplier_walked", "ended_alternative") else "awaiting_buyer"
        return {"revision": self.revision, "status": status, "item": s.state.item, "quantity": s.state.quantity,
                "round": s.state.current_round, "max_rounds": s.state.max_rounds,
                "transcript": s.public_transcript(), "agreement": asdict(s.agreement) if s.agreement else None,
                "clarification": s.outbox[-1]["text"] if s.outbox and (
                    s.status == "awaiting_info"
                    or (s.status == "awaiting_supplier"
                        and s.outbox[-1]["action"] in ("clarify", "clarify_offer"))) else None}

    def record(self, session_id: str) -> dict:
        """The session as a file an operator can replay later. Server-side only."""
        s = self.session
        try:
            described = self.backend.describe()
        except Exception as error:  # keep the record even when the backend cannot describe itself
            described = {"kind": self.backend.kind, "is_model": self.backend.is_model,
                         "describe_error": f"{type(error).__name__}: {error}"}
        counterparty = "simulated_supplier" if self.mode == "simulated" else "person_typed_supplier"
        record = {
            "schema": "negotiation_web_session_v1", "case_id": f"{self.scenario.key}-{session_id[:6]}",
            "case": {"seed": self.scenario.seed, "scenario": self.scenario.key if self.mode == "simulated" else None,
                     "mode": self.mode, "counterparty": counterparty},
            "created_at_utc": self.created_at,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": described,
            "evidence_class": "real_model_run" if self.backend.is_model else "cpu_reference_not_model_evidence",
            "approver": "human_dashboard",
            "scope": "interactive synthetic demo session; one run, not an evaluation",
            "complete": s.status in ("agreed", "ended_alternative", "supplier_walked"),
            "buyer_settings": buyer_settings_view(self.private),
            "status": s.status, "outcome": s.outcome(), "public_transcript": s.public_transcript(),
            "approvals": s.approval_log, "reviews": s.review_log, "executed": s.executed,
            "internal_record": s.internal_record(),
            "system1_checks": s.gate.checks if s.gate is not None else [], "events": s.events,
        }
        if self._hidden is not None:
            record["evaluator_only"] = {"supplier_hidden": asdict(self._hidden)}
        return record


def create_app(replay_path=None, backend=None, save_dir: Path | None = None,
               system1_rules: bool = False, writer=None) -> FastAPI:
    app = FastAPI(title="Negotiation synthetic demo", docs_url=None, redoc_url=None)
    sessions: dict[str, LiveSession] = {}
    supplier_sessions: dict[str, LiveSession] = {}
    session_ids: dict[int, str] = {}
    replays = load_replays(replay_path)
    backend = backend if backend is not None else HeuristicBackend()
    backend_lock = Lock()     # one model, one request at a time, whichever session asks
    evidence = load_evidence()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    from .playground import install_playground
    install_playground(app, STATIC, backend, backend_lock, (save_dir / "play") if save_dir else None, system1_rules,
                       writer)

    def save(entry: LiveSession) -> None:
        if save_dir is None:
            return
        session_id = session_ids[id(entry)]
        save_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(entry.record(session_id), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (save_dir / f"session-{session_id}.json").write_text(text, encoding="utf-8")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/app")
    def workspace():
        """Product prototype: a buyer's negotiation inbox on the same session API as the dashboard."""
        # Revalidate every load so a recording never shows a stale page; assets carry a version query.
        return FileResponse(STATIC / "workspace.html", headers={"Cache-Control": "no-cache"})

    @app.get("/supplier/{token}")
    def supplier_page(token: str):
        if token not in supplier_sessions:
            raise HTTPException(404, "공급사 시연 세션이 없습니다.")
        return FileResponse(STATIC / "supplier.html")

    @app.get("/api/config")
    def config():
        return {"backend": backend_view(backend),
                "scenarios": [{"key": s.key, "title": s.title, "summary": s.summary, "watch": s.watch}
                              for s in SCENARIOS],
                "default_scenario": DEFAULT_SCENARIO, "replays": len(replays),
                "buyer_defaults": buyer_settings_view(default_buyer_private()), "quantity": QUANTITY,
                "evidence_available": evidence["available"], "saving_sessions": save_dir is not None}

    @app.get("/api/evidence")
    def evidence_summary():
        return evidence

    @app.get("/api/replays")
    def replay_list():
        return {"items": [{"index": i, "case_id": r["case_id"], "evidence_class": r["evidence_class"],
                           "steps": len(r["records"])} for i, r in enumerate(replays)]}

    @app.get("/api/replays/{index}")
    def replay(index: int):
        if index < 0 or index >= len(replays):
            raise HTTPException(404, "연결된 실행 기록이 없습니다.")
        return replays[index]

    @app.post("/api/sessions")
    def create(payload: CreateSession):
        if payload.scenario not in SCENARIO_BY_KEY:
            raise HTTPException(422, "알 수 없는 시연 시나리오입니다.")
        if payload.preset == "blank" and payload.mode != "human":
            raise HTTPException(422, "빈 협상은 사람이 공급사 역할을 할 때만 만들 수 있습니다.")
        scenario = SCENARIO_BY_KEY[payload.scenario if payload.mode == "simulated" else DEFAULT_SCENARIO]
        if payload.mode == "simulated" and backend.is_model and scenario.key in MODEL_SCENARIO_SEEDS:
            scenario = replace(scenario, seed=MODEL_SCENARIO_SEEDS[scenario.key])
        entry, session_id = LiveSession(payload.mode, payload.preset, scenario, backend, backend_lock,
                                        system1_rules=system1_rules, buyer=payload.buyer, writer=writer), uuid4().hex
        from .playground import remember_bounded
        for old in remember_bounded(sessions, session_id, entry):
            session_ids.pop(id(old), None)
            if old.supplier_token:
                supplier_sessions.pop(old.supplier_token, None)
        session_ids[id(entry)] = session_id
        if entry.supplier_token:
            supplier_sessions[entry.supplier_token] = entry
        save(entry)
        return entry.view(session_id)

    def entry_for(session_id: str) -> LiveSession:
        if session_id not in sessions:
            raise HTTPException(404, "구매자 시연 세션이 없습니다.")
        return sessions[session_id]

    def check_revision(entry: LiveSession, revision: int):
        if revision != entry.revision:
            raise HTTPException(409, "화면보다 세션이 먼저 변경되었습니다. 새로고침 후 확인하세요.")

    @app.get("/api/sessions/{session_id}")
    def state(session_id: str):
        entry = entry_for(session_id)
        with entry.lock:
            return entry.view(session_id)

    @app.post("/api/sessions/{session_id}/step")
    def step(session_id: str, payload: Mutation):
        entry = entry_for(session_id)
        with entry.lock:
            check_revision(entry, payload.revision)
            if not entry.has_supplier_offer():
                raise HTTPException(409, "공급사 제안이 아직 없습니다.")
            try:
                with backend_lock:
                    entry.session.step()
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc
            entry.revision += 1
            save(entry)
            return entry.view(session_id)

    @app.post("/api/sessions/{session_id}/retry")
    def retry(session_id: str, payload: ReviewMutation):
        """A person saw a failed estimate and asks again. The failed attempt stays on record."""
        entry = entry_for(session_id)
        with entry.lock:
            check_revision(entry, payload.revision)
            try:
                with backend_lock:   # a failed System 1 look at the offer is asked again here
                    entry.session.resume_after_review("demo-human", payload.withhold_counter)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc
            entry.revision += 1
            save(entry)
            return entry.view(session_id)

    @app.post("/api/sessions/{session_id}/approval")
    def approval(session_id: str, payload: ApprovalInput):
        entry = entry_for(session_id)
        with entry.lock:
            check_revision(entry, payload.revision)
            pending = entry.session.pending
            if pending is None or pending.request_id != payload.request_id or pending.terms_version != payload.terms_version:
                raise HTTPException(409, "승인 카드 또는 조건 버전이 일치하지 않습니다.")
            action = entry.session.approve if payload.decision == "approve" else entry.session.deny
            action(payload.request_id, "demo-human")
            entry.revision += 1
            save(entry)
            return entry.view(session_id)

    @app.get("/api/supplier/{token}")
    def supplier_state(token: str):
        if token not in supplier_sessions:
            raise HTTPException(404, "공급사 시연 세션이 없습니다.")
        with supplier_sessions[token].lock:
            return supplier_sessions[token].public_view()

    @app.post("/api/supplier/{token}/response")
    def supplier_response(token: str, payload: SupplierInput):
        if token not in supplier_sessions:
            raise HTTPException(404, "공급사 시연 세션이 없습니다.")
        entry = supplier_sessions[token]
        with entry.lock:
            check_revision(entry, payload.revision)
            try:
                with backend_lock:   # the System 1 gate may ask the model about this reply
                    if payload.kind == "counter" and not entry.has_supplier_offer():
                        entry.session.submit_supplier_offer(payload.model_dump(), payload.message)   # typed opening offer
                    else:
                        entry.session.submit_supplier_response(payload.kind, payload.model_dump(), payload.message)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(409, str(exc)) from exc
            entry.revision += 1
            save(entry)
            return entry.public_view()

    return app


def main(argv=None):
    from .probe import add_backend_arguments, make_backend

    parser = argparse.ArgumentParser(
        description="합성 협상 대시보드. 추정 backend·기록 파일·저장 폴더는 운영자가 여기서만 지정합니다.")
    add_backend_arguments(parser)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1",
                        help="기본은 이 PC에서만 접속. 같은 네트워크의 다른 기기(공급사 역할)를 받으려면 0.0.0.0")
    parser.add_argument("--replay", type=Path, action="append",
                        help="읽기 전용으로 보여 줄 기록 JSON 또는 저장 폴더. 여러 번 지정 가능")
    parser.add_argument("--save-dir", type=Path,
                        help="실시간 세션을 판단마다 저장할 폴더 (예: runs/negotiation/web). 나중에 --replay로 재생")
    parser.add_argument("--system1-rules", action="store_true",
                        help="모델 backend여도 System 1 판단(진단·라우팅·검증)은 규칙 판독기로 답함. 느린 GPU에서 대기 시간을 줄일 때")
    parser.add_argument("--llm", choices=["none", "natural", "claude", "openai", "local", "auto"], default="none",
                        help="우리 측 답장(역제안·수락·종료) 문장. 숫자와 행동은 코드가 정하고 검사합니다. "
                             "natural = 코드가 쓰는 자연스러운 문장, auto = Claude(ANTHROPIC_API_KEY가 있을 때) → "
                             "--llm-url 로컬 서버 → natural, none = 기존 고정 문구")
    parser.add_argument("--llm-model", help="openai/local 모델 이름 (예: Qwen/Qwen3-8B)")
    parser.add_argument("--llm-url", help="OpenAI 호환 서버 주소 (예: http://127.0.0.1:8000/v1)")
    parser.add_argument("--open", metavar="PATH",
                        help="서버가 뜬 뒤 이 PC의 기본 브라우저로 열 경로 (예: /play)")
    args = parser.parse_args(argv)
    backend = make_backend(args)
    view = backend_view(backend)
    print(f"추정 backend: {view['kind']} ({'모델 호출' if view['is_model'] else 'CPU 참조, 모델 근거 아님'})")
    if view["server_identity_verified_by_client"] is False:
        print("주의: 외부 판단 서버의 가중치·엔진은 이 프로그램이 검증하지 못합니다. 서버 명령·체크포인트·버전을 따로 기록하세요.")
    if args.host != "127.0.0.1":
        print(f"주의: {args.host}에서 접속을 받습니다. 인증이 없는 합성 시연이므로 신뢰하는 네트워크에서만 사용하세요.")
    import uvicorn
    address = f"http://127.0.0.1:{args.port}"
    print(f"대시보드: {address}/   ·   협상 체험(내가 공급사): {address}/play")
    if args.open:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(address + args.open)).start()
    writer = None
    if args.llm != "none":
        from .language import language_model_for, make_reply_writer
        from .simulator import QUANTITY
        lm = None if args.llm == "natural" else language_model_for(args.llm, args.llm_model, args.llm_url)
        writer = make_reply_writer(lm, QUANTITY)
        print(f"답장 문장: {lm.describe() if lm else 'natural (code-written)'}")
    uvicorn.run(create_app(args.replay, backend, args.save_dir, args.system1_rules, writer),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
