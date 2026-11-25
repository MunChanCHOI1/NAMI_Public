"""Full-negotiation evaluation on synthetic suppliers: agent vs baselines, with a held-out protocol.

Splits are separated by seed range and by what they may be used for:
- dev: choose policy constants (for example the assumed first-round concession).
- calibration: fit the probability map only.
- eval: one frozen held-out run. A recorded freeze is required, and the first local
  attempt on the eval content is written to evidence/holdout_history, exploratory or not.

Everything measured here describes the rule-based simulator, not real suppliers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import baselines, metrics as M, policy
from .backends import CalibratedBackend, ConstantBackend, HeuristicBackend, KeywordBackend
from .contracts import BuyerPrivate, SupplierHidden, Terms
from .economics import HardConstraintViolation, buyer_total_cost, classify_authority, saving_vs_alternative
from .estimate_sources import ComposedBackend, EstimateSpec
from .policy import PolicyConfig
from .probe import add_backend_arguments, make_backend
from .provenance import REAL_MODEL_BACKENDS, backend_execution_config, input_hashes, json_safe, provenance
from .session import TERMINAL, NegotiationSession, SimulatedSupplier
from .requests import FINAL_OFFER_QID
from .simulator import (
    _price_for_profit,
    default_buyer_private,
    initial_state,
    is_last_chance,
    make_supplier,
    would_accept,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]          # negotiation/
SPLIT_SEED_BASE = {"dev": 1100, "calibration": 2000, "eval": 3000}
DEFAULT_CASES = 30
MAX_CASES_PER_SPLIT = 900  # dev 1100..1999, calibration 2000..2899, eval 3000..3899
HISTORY_DIR = PACKAGE_ROOT / "evidence" / "holdout_history"
BASELINES: dict[str, Callable] = {"fixed_concession": baselines.fixed_concession, "midpoint": baselines.midpoint}


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    seed: int
    hidden: SupplierHidden      # evaluator and simulator only


def make_cases(split: str, count: int = DEFAULT_CASES) -> list[Case]:
    """Deterministic suppliers for a split. About one in six cannot beat the alternative quote."""
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_CASES_PER_SPLIT:
        raise ValueError(f"case count must be in 1..{MAX_CASES_PER_SPLIT}; split seed ranges must not overlap")
    base = SPLIT_SEED_BASE[split]
    rng = random.Random(base)
    cases = []
    for i in range(count):
        style = rng.choice(("boulware", "linear", "deadline"))
        sensitive, reveal, feasible = rng.random() < 0.5, rng.random() < 0.5, rng.random() >= 1 / 6
        hidden = make_supplier(base + i, style, sensitive, reveal, feasible)
        cases.append(Case(f"{split}-{base + i}", split, base + i, hidden))
    return cases


def attainable_saving(case: Case, private: BuyerPrivate, quantity: int) -> int:
    """Evaluator-only upper bound: our best saving if the supplier went down to its reserve.

    Searched over the corner compositions (delivery 14 or the standard lead time, payment
    on delivery, 30 or 60 days) that company policy allows. Zero means no deal can beat the alternative.
    """
    best = 0
    for delivery in (private.preferred_delivery_days, case.hidden.standard_lead_days):
        for payment in (0, 30, 60):
            price = _price_for_profit(case.hidden.reserve_profit, delivery, payment, quantity, case.hidden)
            terms = Terms(price, delivery, payment)
            try:
                if classify_authority(terms, quantity, private) != "forbidden":
                    best = max(best, saving_vs_alternative(terms, quantity, private))
            except HardConstraintViolation:
                continue
    return best


class TrueAcceptanceReference:
    """Evaluator-only reference: answers with what the simulator would really do.

    It reads the hidden supplier parameters, so it is not a backend an agent could have. It
    shows how the same policy performs when the acceptance estimate is perfect, which separates
    "the estimate is weak" from "the decision rule is weak".
    """

    kind, is_model = "evaluator_reference_true_acceptance", False

    def __init__(self, hidden: SupplierHidden) -> None:
        self._hidden = hidden

    def predict(self, state, candidates, request):
        out = {c.candidate_id: 1.0 if would_accept(state, self._hidden, c.terms) else 0.0 for c in candidates}
        out[FINAL_OFFER_QID] = 1.0 if is_last_chance(state, self._hidden) else 0.0
        return out

    def describe(self):
        return {"kind": self.kind, "is_model": False, "uses_hidden_information": True}


REFERENCE_POLICY = "reference_true_acceptance"


def run_case(case: Case, private: BuyerPrivate, backend, decide_fn, config: PolicyConfig,
             approve: bool = True, on_decision=lambda record: None) -> tuple[dict, list[dict]]:
    """One negotiation. Returns the outcome row and evaluator-only acceptance records.

    For every candidate the policy scored, the simulator is asked from a copy of the state
    what it would have done. Only the candidate actually sent affects the negotiation.
    """
    hidden = case.hidden
    session = NegotiationSession(initial_state(case.case_id, hidden), private, backend,
                                 SimulatedSupplier(hidden), config, decide_fn)
    quantity = session.state.quantity
    predictions, finals, seen, gave_up, latencies = [], [], set(), [], []
    counters_sent = 0
    for _ in range(80):
        if session.status in TERMINAL or session.status == "needs_review":
            break
        if session.status == "awaiting_approval":
            request = session.pending
            (session.approve if approve else session.deny)(request.request_id, "evaluation-approver")
            continue
        before, executed_before = session.state, len(session.executed)
        decisions_before = len(session.decisions)
        try:
            decision = session.step()
        finally:
            # The model may have completed before an authorization/simulator error.
            # Preserve that decision even when step() cannot return normally.
            for record in session.internal_record()[decisions_before:]:
                on_decision(record)
        if "latency_ms" in decision.model:
            latencies.append(decision.model["latency_ms"])
        digest = decision.model.get("request_sha256")
        if digest and digest not in seen and "error" not in decision.model:
            seen.add(digest)
            for option in decision.options:
                if option.action == "counter" and option.p_accept is not None:
                    predictions.append({"state_id": case.case_id, "round": decision.round_index,
                                        "request_sha256": digest, "terms": asdict(option.terms),
                                        "kind": option.kind, "p": option.p_accept,
                                        "truth": would_accept(before, hidden, option.terms)})
            if decision.facts.get("p_final_offer_source") in ("backend_estimate", "policy_prior"):
                finals.append({"state_id": case.case_id, "round": decision.round_index,
                               "source": decision.facts["p_final_offer_source"],
                               "p": decision.facts["p_final_offer"], "truth": is_last_chance(before, hidden)})
        if len(session.executed) > executed_before and session.executed[-1]["action"] == "counter":
            counters_sent += 1
            given_up = decision.facts["standing_saving_vs_alternative"] or 0
            if given_up > 0 and session.last_response is not None and session.last_response.kind != "accept":
                gave_up.append(given_up)
    outcome = session.outcome()
    saving = outcome["realized_saving_vs_alternative"]
    best = attainable_saving(case, private, quantity)
    cards = outcome["approval_cards"]
    row = {
        "case_id": case.case_id, "status": session.status, "agreement": outcome["agreement"],
        "realized_saving": saving,
        "net_saving": saving - private.approval_cost * cards - private.round_cost * counters_sent,
        "rounds_used": outcome["rounds_used"], "counters_sent": counters_sent,
        "approval_cards": cards, "approvals_denied": outcome["approvals_denied"],
        "authority_violations": len(outcome["authority_violations"]),
        "model_requests": outcome["model_requests"], "latencies_ms": latencies,
        "feasible": best > 0, "attainable_saving": best,
        "capture": (saving / best) if best > 0 else None,
        "target_met": bool(session.agreement and buyer_total_cost(session.agreement, quantity, private)
                           <= private.target_total_cost),
        "gave_up_savings": gave_up, "final_offer_estimates": finals,
        "internal_record": session.internal_record(), "public_transcript": session.public_transcript(),
        "executed": session.executed, "approval_log": session.approval_log,
        "interrupted": any(d.model.get("status") == "interrupted" for d in session.decisions),
    }
    return row, predictions


def _composition(backend) -> dict | None:
    """Estimate sources of a composed backend, or None for a plain one."""
    if getattr(backend, "kind", None) != "composed_estimates":
        return None
    described = backend.describe()
    return {"estimate_sources": described.get("estimate_sources"),
            "acceptance_platt_applied": bool(described.get("acceptance_platt_applied")),
            "model_estimates_used": described.get("model_estimates_used"),
            "spec_sha256": described.get("spec_sha256")}


def _evidence_class(backend) -> str:
    """A hybrid, mapped or partly model-free configuration never carries the plain real-model label."""
    if not backend.is_model:
        return "cpu_reference_not_model_evidence"
    composition = _composition(backend)
    plain = composition is None or (composition["estimate_sources"] == {"acceptance": "model", "final_offer": "model"}
                                    and not composition["acceptance_platt_applied"])
    return "real_model_run" if plain else "real_model_run_composed_estimates"


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict], predictions: list[dict]) -> dict:
    feasible = [r for r in rows if r["feasible"]]
    infeasible = [r for r in rows if not r["feasible"]]
    latencies = sorted(x for r in rows for x in r["latencies_ms"])
    gave_up = [(g, r["realized_saving"]) for r in rows for g in r["gave_up_savings"]]
    out = {
        "cases": len(rows),
        "complete": all(r["status"] in TERMINAL for r in rows),
        "needs_review": sum(1 for r in rows if r["status"] == "needs_review"),
        "mean_realized_saving": _mean(r["realized_saving"] for r in rows),
        "mean_net_saving_after_process_costs": _mean(r["net_saving"] for r in rows),
        "authority_violations": sum(r["authority_violations"] for r in rows),
        "agreements_not_better_than_alternative": sum(1 for r in rows if r["agreement"] and r["realized_saving"] <= 0),
        "feasible_cases": {
            "n": len(feasible),
            "agreement_rate": _mean(1.0 if r["status"] == "agreed" else 0.0 for r in feasible),
            "supplier_walked": sum(1 for r in feasible if r["status"] == "supplier_walked"),
            "mean_capture_of_attainable": _mean(r["capture"] for r in feasible),
            "target_met_rate": _mean(1.0 if r["target_met"] else 0.0 for r in feasible),
        },
        "infeasible_cases": {
            "n": len(infeasible),
            "ended_without_agreement_rate": _mean(1.0 if r["status"] != "agreed" else 0.0 for r in infeasible),
            "mean_rounds_used": _mean(r["rounds_used"] for r in infeasible),
            "approval_cards": sum(r["approval_cards"] for r in infeasible),
        },
        "process": {
            "mean_rounds_used": _mean(r["rounds_used"] for r in rows),
            "approval_cards": sum(r["approval_cards"] for r in rows),
            "model_requests": sum(r["model_requests"] for r in rows),
            "latency_ms_per_request": ({"median": latencies[len(latencies) // 2], "max": latencies[-1]}
                                       if latencies else None),
        },
        "recovery_after_rejected_counter": {
            "instances": len(gave_up),
            "final_saving_over_saving_given_up": (sum(f for _, f in gave_up) / sum(g for g, _ in gave_up)
                                                  if gave_up else None),
        },
    }
    finals = [f for r in rows for f in r["final_offer_estimates"]]
    if finals:
        truths = [f["truth"] for f in finals]
        base_rate = sum(truths) / len(truths)
        reference = M.brier([base_rate] * len(truths), truths)
        raw = M.brier([f["p"] for f in finals], truths)
        out["final_offer_estimates"] = {
            "n": len(finals), "source": sorted({f["source"] for f in finals}), "base_rate": base_rate,
            "auc": M.auc(finals), "brier": raw, "brier_base_rate": reference,
            "brier_skill": (1 - raw / reference) if reference > 0 else None,
            "note": "rounds before the public last round, on states this policy reached",
        }
    if predictions:
        truths = [p["truth"] for p in predictions]
        base_rate = sum(truths) / len(truths)
        reference = M.brier([base_rate] * len(truths), truths)
        raw = M.brier([p["p"] for p in predictions], truths)
        out["acceptance_estimates"] = {
            "n": len(predictions), "base_rate": base_rate, "auc": M.auc(predictions),
            "brier": raw, "brier_base_rate": reference,
            "brier_skill": (1 - raw / reference) if reference > 0 else None,
            **M.reliability(predictions),
            "note": "scored only on states this policy reached",
        }
    return out


def paired_difference(first: list[dict], second: list[dict], key: str = "realized_saving",
                      seed: int = 20261116, resamples: int = 2000) -> dict:
    """Mean per-case difference (first - second) with a bootstrap 90% interval over cases."""
    other = {r["case_id"]: r[key] for r in second}
    diffs = [r[key] - other[r["case_id"]] for r in first]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    return {"metric": key, "cases": len(diffs), "mean_difference": sum(diffs) / len(diffs),
            "ci90": [means[int(0.05 * resamples)], means[int(0.95 * resamples) - 1]],
            "wins": sum(1 for d in diffs if d > 0), "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0)}


def evaluate(split: str, backend, config: PolicyConfig = PolicyConfig(), count: int = DEFAULT_CASES,
             approve: bool = True, mode: str | None = None, protocol: dict | None = None,
             log=lambda message: None, on_decision=lambda record: None,
             source_provenance: dict | None = None) -> dict:
    source_provenance = source_provenance or provenance({
        "backend": backend.kind, **backend.describe().get("configuration", {}),
        "policy": asdict(config), "cases": count, "approver": "auto" if approve else "deny",
    })
    private = default_buyer_private()
    cases = make_cases(split, count)
    policies: dict[str, tuple[Any, Callable]] = {"agent": (backend, policy.decide)}
    agent_is_rule = not backend.is_model and backend.kind == HeuristicBackend.kind and not isinstance(backend, CalibratedBackend)
    if not agent_is_rule:
        policies["rule_policy"] = (HeuristicBackend(), policy.decide)
    for name, fn in BASELINES.items():
        policies[name] = (None, fn)
    policies[REFERENCE_POLICY] = ("per_case_reference", policy.decide)
    rows: dict[str, list[dict]] = {}
    predictions: dict[str, list[dict]] = {}
    interrupted = False
    for name, (policy_backend, fn) in policies.items():
        rows[name], predictions[name] = [], []
        for number, case in enumerate(cases, 1):
            used = TrueAcceptanceReference(case.hidden) if name == REFERENCE_POLICY else policy_backend
            row, scored = run_case(case, private, used, fn, config, approve,
                                   on_decision=lambda record: on_decision({
                                       "policy": name, "case_id": case.case_id, "decision": record,
                                   }))
            rows[name].append(row)
            predictions[name].extend(scored)
            log(f"{name} [{number}/{len(cases)}] {case.case_id} {row['status']} saving={row['realized_saving']}")
            if row["interrupted"]:
                # Keep the completed/partial agent cases and compare CPU references on those same cases.
                interrupted = True
                cases = cases[:number]
                break
    predictions[REFERENCE_POLICY] = []        # its "estimates" are the answers; nothing to score
    summaries = {name: summarize(rows[name], predictions[name]) for name in policies}
    summaries[REFERENCE_POLICY]["note"] = ("evaluator-only reference using hidden supplier information; "
                                           "an upper reference for this decision rule, not a competitor")
    reference = [n for n in policies if n != "agent"]
    result = {
        "schema_version": 2,
        "experiment": "full_negotiation_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "evaluation_mode": mode or {"dev": "development", "calibration": "calibration"}.get(split, "exploratory"),
        "protocol": protocol or {},
        "provenance": source_provenance,
        "backend": backend.describe(),
        "evidence_class": _evidence_class(backend),
        "agent_is_rule_policy": agent_is_rule,
        **({"estimate_composition": _composition(backend)} if _composition(backend) is not None else {}),
        "complete": not interrupted and all(s["complete"] for s in summaries.values()),
        "interrupted": interrupted,
        "config": {"policy": asdict(config), "cases": count, "approver": "approve_all" if approve else "deny_all",
                   "python": platform.python_version()},
        "scope": "synthetic rule-based suppliers; not evidence about real negotiations",
        "policies": summaries,
        "agent_minus": {name: {"realized_saving": paired_difference(rows["agent"], rows[name]),
                               "net_saving": paired_difference(rows["agent"], rows[name], "net_saving")}
                        for name in reference},
        "cases": [{"case_id": case.case_id,
                   **{name: {k: rows[name][i][k] for k in ("status", "agreement", "realized_saving", "net_saving",
                                                         "rounds_used", "approval_cards", "authority_violations",
                                                         "model_requests", "latencies_ms", "internal_record",
                                                         "public_transcript", "executed", "approval_log")}
                      for name in policies}}
                  for i, case in enumerate(cases)],
        "evaluator_only": {
            "cases": [{"case_id": c.case_id, "supplier_hidden": asdict(c.hidden),
                       "feasible": rows["agent"][i]["feasible"],
                       "attainable_saving": rows["agent"][i]["attainable_saving"]} for i, c in enumerate(cases)],
            "acceptance_predictions": {name: values for name, values in predictions.items() if values},
        },
    }
    return result


# ---- dev-only tuning and calibration ----------------------------------------------------------------

SWEEPABLE = {"prior_concession": int, "reserve_margin": int}


def sweep(backend, parameter: str, grid, count: int = DEFAULT_CASES, base: PolicyConfig = PolicyConfig(),
          log=lambda message: None) -> dict:
    """Development split only: run the agent under each value of one policy constant."""
    if parameter not in SWEEPABLE:
        raise ValueError(f"parameter must be one of {sorted(SWEEPABLE)}")
    private, cases, table = default_buyer_private(), make_cases("dev", count), []
    for value in grid:
        config = replace(base, **{parameter: SWEEPABLE[parameter](value)})
        rows = [run_case(case, private, backend, policy.decide, config)[0] for case in cases]
        summary = summarize(rows, [])
        table.append({"value": SWEEPABLE[parameter](value), "mean_realized_saving": summary["mean_realized_saving"],
                      "mean_net_saving": summary["mean_net_saving_after_process_costs"],
                      "agreement_rate_feasible": summary["feasible_cases"]["agreement_rate"],
                      "supplier_walked": summary["feasible_cases"]["supplier_walked"]})
        log(f"{parameter}={value} mean_net_saving={table[-1]['mean_net_saving']:.0f}")
    best = max(table, key=lambda r: (r["mean_net_saving"], -r["value"]))
    return {"experiment": "policy_constant_sweep", "parameter": parameter, "split": "dev", "cases": count,
            "backend": backend.describe(),
            "evidence_class": "real_model_run" if backend.is_model else "cpu_reference_not_model_evidence",
            "table": table, "selected_value": best["value"],
            "selection_rule": "highest mean net saving on the development split; ties go to the lower value",
            "note": "a policy constant chosen on development cases; freeze it before the held-out run"}


def fit_calibration(result: dict, source: str = "") -> dict:
    """Fit a Platt map from a calibration-split run. Refuses any other split."""
    if result.get("experiment") != "full_negotiation_evaluation" or result.get("split") != "calibration":
        raise ValueError("calibration can only be fitted on a result of the calibration split")
    described = result.get("backend", {})
    if "calibration" in described:
        raise ValueError("fit the map on an uncalibrated run")
    if described.get("kind") == ComposedBackend.kind and (
            described.get("estimate_sources", {}).get("acceptance") != "model"
            or described.get("acceptance_platt_applied")):
        # A --config-file run: only the unmapped acceptance estimate of the wrapped backend may be fitted.
        raise ValueError("fit the map on an uncalibrated run whose acceptance estimate is the backend's own")
    records = result["evaluator_only"]["acceptance_predictions"].get("agent")
    if not records or len({r["truth"] for r in records}) < 2:
        raise ValueError("the calibration run has no usable acceptance records for the agent")
    xs, ys = [M._logit(r["p"]) for r in records], [1.0 if r["truth"] else 0.0 for r in records]
    a, b = M._fit_platt(xs, ys)
    fitted = [1 / (1 + pow(2.718281828459045, -max(-40.0, min(40.0, a * x + b)))) for x in xs]
    truths = [r["truth"] for r in records]
    return {"method": "platt", "a": a, "b": b, "n": len(records), "fitted_on_split": "calibration",
            "source": source, "backend_kind": result["backend"].get("kind"),
            "brier_before": M.brier([r["p"] for r in records], truths),
            "brier_after_in_sample": M.brier(fitted, truths)}


# ---- usage configuration file (--config-file) ----------------------------------------------------------
# NOT executed on a GPU as of 2026-11-18: this path was verified on CPU only, with the CPU reference
# backends and test doubles. A file read here says how the estimates are USED; it is not model evidence.

USAGE_CONFIG_KEYS = ("policy", "estimates")
# What each CPU --backend choice declares about the final-offer estimate, known without building it.
# The real-model choices (zoro, zoro-http, systemone) all answer the final-offer question.
_CPU_BACKEND_CLASSES = {"constant": ConstantBackend, "heuristic": HeuristicBackend, "keyword": KeywordBackend}


@dataclass(frozen=True)
class UsageConfig:
    """One --config-file, read once: the same bytes are hashed, parsed and embedded.

    `record` is what `execution_config` stores under `usage_config`, so `freeze` pins the whole
    file and the held-out run refuses a file whose bytes or meaning changed.
    """

    policy: PolicyConfig
    estimates: EstimateSpec
    record: dict


def _policy_from_json(data: Any) -> PolicyConfig:
    """PolicyConfig from a JSON object. Unknown fields are refused; missing fields keep their defaults."""
    if not isinstance(data, dict):
        raise ValueError('"policy" must be a JSON object of PolicyConfig fields')
    defaults = PolicyConfig()
    unknown = sorted(set(data) - {f.name for f in fields(PolicyConfig)})
    if unknown:
        raise ValueError(f"unknown policy fields: {unknown}")
    values: dict[str, Any] = {}
    for name, value in data.items():
        if name == "aspiration_exponent":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 10.0:
                raise ValueError(f"policy field {name} must be a number in (0, 10]")
            values[name] = float(value)
        elif name == "cap_next_offer_at_ask":
            if not isinstance(value, bool):
                raise ValueError(f"policy field {name} must be true or false")
            values[name] = value
        elif name in ("final_accept_scale", "final_accept_scale_recomposed"):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not 0.0 <= value <= 1.0):
                raise ValueError(f"policy field {name} must be null or a number in [0, 1]")
            values[name] = None if value is None else float(value)
        elif isinstance(getattr(defaults, name), tuple):          # final_prior_by_round: JSON has no tuple
            if not isinstance(value, list) or not value or any(
                    isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0 for p in value):
                raise ValueError(f"policy field {name} must be a non-empty list of probabilities")
            values[name] = tuple(float(p) for p in value)
        else:
            minimum = 1 if name == "max_candidates" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"policy field {name} must be an integer of at least {minimum}")
            values[name] = value
    return PolicyConfig(**values)


def backend_supplies_final_offer(backend_name: str) -> bool:
    """Whether a --backend choice answers the final-offer question, without building the backend."""
    cls = _CPU_BACKEND_CLASSES.get(backend_name)
    return True if cls is None else bool(getattr(cls, "supports_final_offer", cls.is_model))


def load_usage_config(path: Path, backend_name: str | None = None) -> UsageConfig:
    """Read {"policy": {...}, "estimates": EstimateSpec.to_json()}; extra top-level keys are kept, not used.

    Raises ValueError for anything that could not run, so a refusal happens before a freeze is
    written, before an output file exists and before a held-out attempt is recorded.
    """
    raw = path.read_bytes()
    try:
        content = json.loads(raw.decode("utf-8-sig"))
        # Non-finite numbers cannot be written to a manifest; a value must read back unchanged.
        stable = json.loads(json.dumps(content, ensure_ascii=False, allow_nan=False)) == content
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError(f"not a JSON file that can be recorded: {error}") from error
    if not stable or not isinstance(content, dict):
        raise ValueError("the file must hold one JSON object that reads back unchanged")
    missing = [key for key in USAGE_CONFIG_KEYS if key not in content]
    if missing:
        raise ValueError(f"missing top-level keys: {missing}")
    config = _policy_from_json(content["policy"])
    spec = EstimateSpec.from_json(content["estimates"])
    spec.validate()
    if backend_name is not None and "final_offer" in spec.model_estimates_used() \
            and not backend_supplies_final_offer(backend_name):
        raise ValueError(f"final_source '{spec.final_source}' needs the backend's final-offer estimate, "
                         f"which --backend {backend_name} does not supply")
    if backend_name in REAL_MODEL_BACKENDS and not spec.uses_model():
        # Such a run would load the model, never ask it, and still be recorded as a real-model request.
        raise ValueError(f"this configuration uses no backend estimate, so --backend {backend_name} would "
                         "never be asked; run it with a CPU backend (the default) instead")
    # Fields added on 2026-11-19 are left out at their legacy defaults, so earlier digests are unchanged.
    legacy = policy.LEGACY_POLICY_FIELDS
    resolved = {"policy": json.loads(json.dumps({k: v for k, v in asdict(config).items()
                                                 if not (k in legacy and v == legacy[k])})),
                "estimates": spec.to_json()}
    canonical = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Where the file says it comes from is checked as far as it can be; the rest is recorded as unverified.
    origin = "operator_file_origin_unverified"
    selected_from = content.get("selected_from")
    if selected_from is not None:
        if not isinstance(selected_from, dict):
            raise ValueError("selected_from must be the object written by the tuning runner")
        if not str(selected_from.get("config_id", "")).endswith("-" + config_sha256[:12]):
            raise ValueError("the file says it is tuning selection "
                             f"{selected_from.get('config_id')!r}, but its policy and estimates give "
                             f"{config_sha256[:12]}: it was edited after the selection. Remove selected_from to "
                             "use it as an operator-written file")
        if backend_name in REAL_MODEL_BACKENDS and selected_from.get("counts_as_model_evidence") is not True:
            raise ValueError("this configuration was selected in a CPU rehearsal or on shortened blocks "
                             "(selected_from.counts_as_model_evidence is not true); its maps were not fitted on "
                             f"the real model, so it is refused for --backend {backend_name}")
        origin = "tuning_selection_digest_matches"
    record = {
        # "tuning_selection_digest_matches": the estimates equal the selection the file names (the selection
        # itself is not re-verified here). A map's fitted_on_split is the file's declaration, not a check.
        "origin": origin,
        "file_sha256": hashlib.sha256(raw).hexdigest(),          # the file's bytes, formatting included
        "content": content,                                      # the whole file, extra keys included
        "estimates": spec.to_json(),                             # as resolved: defaults filled in
        # Canonical {policy, estimates} as resolved; the same for two files that mean the same.
        "config_sha256": config_sha256,
    }
    return UsageConfig(config, spec, record)


# ---- freeze and held-out history (same rules as SessionGuard) -----------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_config(args, config: PolicyConfig, calibration: dict | None,
                     usage_config: UsageConfig | None = None) -> dict:
    model_backend = args.backend in REAL_MODEL_BACKENDS
    settings = {
        **backend_execution_config(args),
        "policy": json.loads(json.dumps(asdict(config))),   # as it will read back from the manifest
        "buyer_private": asdict(default_buyer_private()),
        "cases": args.cases, "approver": args.approver,
        "calibration": json_safe(calibration),
        "real_model_requested": model_backend,
    }
    if usage_config is not None:     # no key at all without --config-file: earlier freezes keep matching
        if calibration is not None or config != usage_config.policy:
            raise ValueError("a usage config carries its own policy and probability map")
        settings["usage_config"] = copy.deepcopy(usage_config.record)
    return settings


def _write(path: Path, value: Any, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def freeze(config: dict, output: Path) -> dict:
    snapshot = provenance(config)
    payload = {"schema_version": "2.0", "created_at": _now(), "git_commit": snapshot["git_commit"],
               "input_sha256": snapshot["input_sha256"], "execution_config": config,
               "provenance": snapshot, "evaluation_executed": False,
               "notice": "Freeze records configuration; it does not run held-out cases and does not prove "
                         "that development evaluation was completed."}
    _write(output, payload, exclusive=True)
    return payload


def evaluation_identity(count: int) -> str:
    """Identify the eval content (suppliers and our private inputs) independently of code formatting."""
    identity = {"identity_schema_version": "1.0",
                "cases": [{"case_id": c.case_id, "hidden": asdict(c.hidden)} for c in make_cases("eval", count)],
                "buyer_private": asdict(default_buyer_private())}
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluation_case_identities(count: int) -> dict[str, str]:
    """Content identities survive count changes and case-id renaming."""
    private = asdict(default_buyer_private())
    identities = {}
    for case in make_cases("eval", count):
        content = {"supplier_hidden": asdict(case.hidden), "buyer_private": private}
        text = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        identities[case.case_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return identities


def _history_event(count: int, mode: str, history_dir: Path) -> tuple[Path, dict]:
    dataset = evaluation_identity(count)
    event = {"schema_version": "2.0", "started_at": _now(), "dataset_sha256": dataset, "case_count": count,
             "case_sha256": evaluation_case_identities(count),
             "first_recorded_mode": mode,
             "notice": "A local evaluation attempt was recorded. This does not prove completed runs or absence "
                       "of earlier exposure in another checkout. Keep this history with the experiment."}
    return history_dir / f"{dataset}.json", event


def _used_case_hashes(history_dir: Path) -> set[str]:
    used = set()
    for marker in sorted(history_dir.glob("*.json")):
        event = json.loads(marker.read_text(encoding="utf-8"))
        hashes = event.get("case_sha256")
        if hashes is None:
            # Older markers recorded only the complete list hash. Reconstruct only
            # when that exact historical content is still reproducible.
            count = event.get("case_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 1 \
                    or event.get("dataset_sha256") != evaluation_identity(count):
                raise ValueError(f"Cannot verify the cases in legacy evaluation history {marker}. "
                                 "Preserve the history and use exploratory mode.")
            hashes = evaluation_case_identities(count)
        if not isinstance(hashes, dict) or not hashes or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        ):
            raise ValueError(f"Cannot verify the cases in evaluation history {marker}.")
        used.update(hashes.values())
    return used


def record_exploratory(count: int, history_dir: Path = HISTORY_DIR) -> dict:
    marker, event = _history_event(count, "exploratory", history_dir)
    try:
        _write(marker, event, exclusive=True)
        created = True
    except FileExistsError:
        created = False
    return {"mode": "exploratory", "dataset_sha256": event["dataset_sha256"], "history_marker": str(marker),
            "history_created": created,
            "notice": "Exploratory evaluation; do not report it as an untouched held-out result."}


def reserve_heldout(config: dict, manifest: Path, count: int, history_dir: Path = HISTORY_DIR) -> dict:
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    current = provenance(config)
    frozen_snapshot = frozen.get("provenance", {})
    identity_fields = ("input_sha256", "model_pins", "official_source_files", "checkpoint_metadata_sha256")
    if frozen.get("input_sha256") != input_hashes() or frozen.get("execution_config") != config \
            or any(frozen_snapshot.get(key) != current[key] for key in identity_fields):
        raise ValueError("Frozen files or execution settings changed. Re-freeze before the first evaluation, "
                         "or use --evaluation-mode exploratory for a consumed set.")
    consumed = manifest.with_name(manifest.name + ".consumed.json")
    if consumed.exists():
        raise ValueError("This freeze already has an evaluation attempt. Use --evaluation-mode exploratory.")
    marker, event = _history_event(count, "heldout", history_dir)
    if set(event["case_sha256"].values()) & _used_case_hashes(history_dir):
        raise ValueError("This evaluation set already has a recorded attempt for overlapping cases. "
                         "Use --evaluation-mode exploratory; a new freeze does not make the set unseen.")
    event["freeze_sha256"] = _sha256_file(manifest)
    try:
        _write(marker, event, exclusive=True)
    except FileExistsError as error:
        raise ValueError("This evaluation set already has a recorded attempt, even under another freeze. "
                         "Use --evaluation-mode exploratory; a new freeze does not make the set unseen.") from error
    _write(consumed, event, exclusive=True)
    return {"mode": "heldout", "dataset_sha256": event["dataset_sha256"], "history_marker": str(marker),
            "attempt_marker": str(consumed), "freeze_manifest_sha256": event["freeze_sha256"],
            "notice": "First recorded attempt for this set in this checkout; not proof of complete non-exposure."}


# ---- command line -----------------------------------------------------------------------------------

def _load_calibration(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _build_backend(args, calibration: dict | None, usage_config: UsageConfig | None = None):
    backend = make_backend(args)
    if usage_config is not None:
        # The composition answers from the unchanged request; a model-free one never calls `backend`.
        try:
            return ComposedBackend(backend, usage_config.estimates, usage_config.policy.final_prior_by_round)
        except BaseException:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
            raise
    return CalibratedBackend(backend, calibration["a"], calibration["b"], calibration.get("source", "")) \
        if calibration else backend


def _common(parser: argparse.ArgumentParser, config_file: bool = False) -> None:
    add_backend_arguments(parser)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--prior-concession", type=int, default=PolicyConfig().prior_concession)
    parser.add_argument("--approver", choices=("auto", "deny"), default="auto")
    parser.add_argument("--calibration", type=Path, help="Platt map written by fit-calibration")
    if config_file:
        parser.add_argument("--config-file", type=Path,
                            help='usage configuration: JSON {"policy": {PolicyConfig fields}, "estimates": '
                                 "{estimate sources}}, for example the selected_config.json of a tuning run. "
                                 "The whole file is recorded and frozen. Not allowed together with "
                                 "--calibration or a non-default --prior-concession (CPU-verified only; "
                                 "not executed on a GPU as of 2026-11-18)")
    parser.add_argument("--output", type=Path, required=True)


def _usage_config_argument(args) -> UsageConfig | None:
    """Read --config-file, or return None when the command has no such flag or it was not given.

    Every refusal is a SystemExit raised before a freeze, an output file, a backend or a
    held-out history marker exists.
    """
    path = getattr(args, "config_file", None)
    if path is None:
        return None
    if args.calibration is not None:
        raise SystemExit("--config-file cannot be combined with --calibration: "
                         "the file's estimates.acceptance_platt is the only probability map of such a run")
    if args.prior_concession != PolicyConfig().prior_concession:
        raise SystemExit("--config-file cannot be combined with --prior-concession: "
                         "the file's policy.prior_concession is the value of such a run")
    try:
        return load_usage_config(path, args.backend)
    except (OSError, ValueError) as error:
        raise SystemExit(f"--config-file {path}: {type(error).__name__}: {error}") from error


def main(argv: list[str] | None = None, history_dir: Path = HISTORY_DIR) -> int:
    parser = argparse.ArgumentParser(description="Full-negotiation evaluation for the negotiation agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate", help="run the agent and the baselines on one split")
    _common(run, config_file=True)
    run.add_argument("--split", choices=tuple(SPLIT_SEED_BASE), default="dev")
    run.add_argument("--evaluation-mode", choices=("heldout", "exploratory"), default="heldout")
    run.add_argument("--freeze-manifest", type=Path)
    run.add_argument("--no-records", action="store_true", help="omit per-case rows and evaluator-only data")
    frz = commands.add_parser("freeze", help="record the configuration before the held-out run")
    _common(frz, config_file=True)
    swp = commands.add_parser("sweep", help="development split: compare values of one policy constant")
    _common(swp)
    swp.add_argument("--param", choices=sorted(SWEEPABLE), default="prior_concession")
    swp.add_argument("--grid", type=float, nargs="+", default=[0, 50_000, 150_000, 300_000, 600_000])
    fit = commands.add_parser("fit-calibration", help="fit a Platt map from a calibration-split result")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731

    if args.command == "fit-calibration":
        try:
            result = fit_calibration(json.loads(args.input.read_text(encoding="utf-8")), source=_sha256_file(args.input))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        _write(args.output, result)
        print(f"{args.output}  a={result['a']:.3f} b={result['b']:.3f} n={result['n']}")
        return 0

    if not 1 <= args.cases <= MAX_CASES_PER_SPLIT:
        parser.error(f"--cases must be in 1..{MAX_CASES_PER_SPLIT}; split seed ranges must not overlap")
    usage = _usage_config_argument(args)        # evaluate and freeze only; None without --config-file
    calibration = _load_calibration(args.calibration)
    # A map fitted on anything but the calibration split is refused before it can be frozen or applied.
    if calibration and calibration.get("fitted_on_split") != "calibration":
        raise SystemExit("the calibration file was not fitted on the calibration split")
    config = usage.policy if usage else PolicyConfig(prior_concession=args.prior_concession)
    if usage:
        log(f"usage config {usage.record['config_sha256'][:12]} (file sha256 {usage.record['file_sha256'][:12]}): "
            f"policy constants and estimate sources {usage.estimates.estimate_sources()} come from the file; "
            f"origin={usage.record['origin']} (a map's fitted_on_split is the file's declaration, not a check)")
    if args.command == "freeze":
        freeze(execution_config(args, config, calibration, usage), args.output)
        print(f"{args.output}  frozen; no evaluation case was run")
        return 0
    if args.command == "sweep":
        result = sweep(_build_backend(args, calibration), args.param, args.grid, args.cases, config, log)
        _write(args.output, result)
        print(f"{args.output}  {args.param} selected_value={result['selected_value']}")
        return 0

    decisions_path = args.output.with_suffix(args.output.suffix + ".decisions.jsonl")
    if args.backend in REAL_MODEL_BACKENDS and (args.output.exists() or decisions_path.exists()):
        raise SystemExit("Real-model evidence output already exists. Choose a new --output path.")
    settings = execution_config(args, config, calibration, usage)
    source_provenance = provenance(settings)
    protocol, mode = {}, None
    if args.split == "eval":
        if args.evaluation_mode == "heldout":
            if not args.freeze_manifest:
                raise SystemExit("a held-out run needs --freeze-manifest (run `freeze` first), "
                                 "or label the run with --evaluation-mode exploratory")
            try:
                protocol = reserve_heldout(settings, args.freeze_manifest, args.cases, history_dir)
            except ValueError as error:
                raise SystemExit(str(error)) from error
        else:
            protocol = record_exploratory(args.cases, history_dir)
        mode = protocol["mode"]
    result = {"schema_version": 2, "experiment": "full_negotiation_evaluation", "split": args.split,
              "complete": False, "status": "initializing", "protocol": protocol,
              "provenance": source_provenance, "decisions_file": str(decisions_path)}
    _write(args.output, result)
    with decisions_path.open("w", encoding="utf-8") as stream:
        def save_decision(record):
            stream.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        try:
            backend = _build_backend(args, calibration, usage)
        except (Exception, KeyboardInterrupt) as error:
            result.update(status="initialization_failed", error=f"{type(error).__name__}: {error}")
        else:
            try:
                result = evaluate(args.split, backend, config, args.cases,
                                  approve=args.approver == "auto", mode=mode, protocol=protocol, log=log,
                                  on_decision=save_decision, source_provenance=source_provenance)
            except (Exception, KeyboardInterrupt) as error:
                result.update(status="evaluation_failed", error=f"{type(error).__name__}: {error}")
            result["decisions_file"] = str(decisions_path)
    if args.no_records:
        result.pop("cases", None)
        result.pop("evaluator_only", None)
    _write(args.output, result)
    if "policies" not in result:
        log(f"{args.output}  complete=False {result.get('error')}")
        return 1
    agent = result["policies"]["agent"]
    print(f"{args.output}  split={args.split} mode={result['evaluation_mode']} complete={result['complete']} "
          f"agent_mean_saving={agent['mean_realized_saving']:.0f} violations={agent['authority_violations']}"
          + (f" evidence_class={result['evidence_class']} "
             f"estimate_sources={result['estimate_composition']['estimate_sources']}"
             if "estimate_composition" in result else ""))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
