"""E2: the negotiation comparison under the opt-in utterance styles, the outcome rule R2, and the report.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only, with the CPU stand-in,
model-free arms and test doubles. No number produced here so far is model evidence.

Design: negotiation/docs/UTTERANCE_AGENT_DESIGN.md, sections 8 (use of the test block), 8.2 (arms, R2,
secondary comparisons), 8.3 (stopping rule), 9 (labels), 11.5 (adoption decision), 13.6.

What this module keeps:
- Every arm runs the same cases, the same policy (`policy.decide`, default constants) and the same
  buyer assumptions; only the source of the final-offer estimate differs. Approval cards are
  approved automatically, as in the existing evaluator.
- A failed estimate stops the arm. No value is filled in. Running the arm again is a person's
  recorded decision (`--retry-failed`), accepted once.
- Per arm and style only the FIRST completed attempt of a state directory is read by a verdict;
  every later attempt is written to a new folder and marked not_confirmatory.
- The test block runs only at its registered size, only with thresholds fitted on the fit block and
  only after the bench block's reading is on record; its first use is recorded before it happens.
- A composed arm never carries the plain real-model label; a CPU stand-in run is a rehearsal.
- Result files use the top-level key `rows`, never the keys the dashboard replay path reads.

This is evaluator-side code: it reads hidden supplier parameters and sentence labels. Neither
reaches a request.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

from negotiation_agent import policy
from negotiation_agent.estimate_sources import ComposedBackend, EstimateSpec, round_prior
from negotiation_agent.evaluation import paired_difference
from negotiation_agent.policy import PolicyConfig
from negotiation_agent.provenance import json_safe, response_snapshot, validate_predictions
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.session import TERMINAL, NegotiationSession
from negotiation_agent.simulator import default_buyer_private, is_last_chance

from . import GPU_NOTICE, PROTOCOL_VERSION
from . import bench as B
from . import cases as C
from . import experiment as X
from . import wording as W
from .interpreter import FIXED_HALF, LabelReader, ModelReader, Thresholds, UtteranceFinalBackend

# ---- fixed names ---------------------------------------------------------------------------------------

BEATS = "utterance_agent_beats_round_prior_on_paraphrase"
WORSE = "utterance_agent_worse_than_round_prior_on_paraphrase"
NOT_SHOWN = "not_shown"
INCOMPLETE = "incomplete"
NOT_RUN_NO_THRESHOLD = "not_run_no_useful_threshold"
NO_USEFUL_THRESHOLD = "no_useful_threshold_on_fit"
SKIPPED_IDENTICAL = "skipped_identical_to_utt_gate_fit"
READING_NOT_DEMONSTRATED = "utterance_reading_not_demonstrated"
PRIMARY_ARM, CONTROL_ARM, PRIMARY_STYLE = "utt_gate_fit", "prior", "paraphrase"
ARM_KIND = {"prior": "cpu_arm", "keyword": "cpu_arm", "label_reader": "cpu_arm", "true_final": "cpu_arm",
            "utt_gate_fit": "utterance_arm", "utt_gate": "utterance_arm",
            "model_final": "composed_arm", "model_final_gate": "composed_arm",
            "model_final_gate_fit": "composed_arm", "m0": "m0"}
DEFAULT_REQUEST_ARMS = ("model_final", "model_final_gate", "model_final_gate_fit", "m0")   # ask the default request
MODEL_FINAL_GATE_HALF = 0.5          # the fixed secondary arm decided before the results
# Folder of an arm's attempts. The registered block keeps the documented layout; the smoke block
# (rehearsals and path checks) has a folder of its own, so its attempts never count for a verdict.
FOLDER_BY_BLOCK = {"test": "runs", "smoke": "rehearsal"}
# Values of `final_offer_value_from` that mean "the estimate was raised to 1.0 by this arm's reading".
FIRE_VALUES = ("utterance_final", "keyword_match", "model_final_over_threshold", "hidden_last_chance")
DECOY_CLASSES = ("negated_keyword", "open")
APPROVER = "synthetic-auto-approver"
CONTINUABLE = ("partial_budget", "interrupted", "read_only_store_miss", "runner_error", "unreadable")
ENVIRONMENT = ("opt-in supplier utterance style on the rule-based simulator: only the message strings differ "
               "from the default simulator")
SCOPE = ("synthetic rule-based suppliers and code-written sentences of a development-only seed range; not a "
         "held-out result and not evidence about real negotiations")
# Section 11.5, verbatim: said together with either "keep the existing question" outcome.
EXISTING_QUESTION_CAVEAT = "기존 질문은 지시문에 +0.498 움직였고(측정 1회) 지시를 감지하는 답이 없다."
NOT_SHOWN_CONTAINS_ZERO = "90% 구간이 0을 포함함"
NOT_SHOWN_UNMET = "하한은 0보다 크지만 채우지 못한 조건: "


# ---- reference and gate backends -----------------------------------------------------------------------

def _prior_table(prior_table) -> tuple[float, ...]:
    table = tuple(PolicyConfig().final_prior_by_round if prior_table is None else prior_table)
    if not table or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0 for p in table):
        raise ValueError("prior_table must be a non-empty sequence of probabilities")
    return table


class TrueFinalReference:
    """Evaluator-only reference: constant acceptance plus the hidden truth of "this offer is the last".

    It reads the hidden supplier parameters, so it is not a backend an agent could have. It also
    knows a last chance the supplier never announced, which no reading of utterances can.
    """

    kind, is_model, supports_final_offer = "evaluator_reference_true_final", False, True

    def __init__(self, hidden, acceptance: float = X.ACCEPTANCE_CONSTANT) -> None:
        self._hidden, self._acceptance = hidden, float(acceptance)
        self.last_response: dict | None = None

    def predict(self, state, candidates, request) -> dict[str, float]:
        last = is_last_chance(state, self._hidden)
        out = {qid: self._acceptance for qid in request["questions"] if qid != FINAL_OFFER_QID}
        if FINAL_OFFER_QID in request["questions"]:
            out[FINAL_OFFER_QID] = 1.0 if last else 0.0
        self.last_response = {"reference": {
            "final_offer_value_from": "hidden_last_chance" if last else "hidden_not_last_chance"}}
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "is_model": False, "uses_hidden_information": True,
                "acceptance_constant": self._acceptance}


class ModelFinalGate:
    """1.0 when the base's own final-offer answer is at or above a threshold, else the round prior.

    The base answers the unchanged default request; its acceptance estimates pass through. There is
    no instruction rule: the existing question gives no instruction answer. `threshold` and `name`
    have no default: 0.5 for model_final_gate, the fitted tau_m for model_final_gate_fit.
    """

    kind = "composed_estimates"
    supports_final_offer = True

    def __init__(self, base, *, threshold: float, name: str, prior_table=None) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite number in [0, 1]")
        if not isinstance(name, str) or not name or name == "model":
            raise ValueError("name must be a non-empty configuration name other than 'model'")
        if not bool(getattr(base, "supports_final_offer", base.is_model)):
            raise ValueError("the base backend gives no final-offer estimate to put a threshold on")
        self.base, self.inner = base, base
        self.threshold, self.name, self.prior_table = float(threshold), name, _prior_table(prior_table)
        self.is_model = bool(base.is_model)
        self.last_response: dict | None = None

    def predict(self, state, candidates, request) -> dict[str, float]:
        self.last_response = None
        answered = self.base.predict(state, candidates, request)       # the unchanged request object
        snapshot = response_snapshot(self.base)
        out = validate_predictions(answered, [qid for qid in request["questions"] if qid != FINAL_OFFER_QID])
        record: dict[str, Any] = {"rule": self.name, "threshold": self.threshold, "base_final_offer": None,
                                  "round_prior": round_prior(state, self.prior_table),
                                  "final_offer_value": None, "final_offer_value_from": "not_asked"}
        if FINAL_OFFER_QID in request["questions"]:
            value = validate_predictions(answered, [FINAL_OFFER_QID])[FINAL_OFFER_QID]
            fired = value >= self.threshold
            out[FINAL_OFFER_QID] = 1.0 if fired else record["round_prior"]
            record.update(base_final_offer=value, final_offer_value=out[FINAL_OFFER_QID],
                          final_offer_value_from="model_final_over_threshold" if fired else "round_prior")
        self.last_response = {**(snapshot or {}), "final_gate": record}
        return out

    def describe(self) -> dict:
        base = self.base.describe()
        identity = {"name": self.name, "threshold": self.threshold, "prior_table": list(self.prior_table),
                    "base": {key: base.get(key) for key in ("kind", "is_model", "estimate_sources", "spec_sha256")}}
        return {"kind": self.kind, "wrapper": "model_final_gate", "is_model": self.is_model,
                "estimate_sources": {"acceptance": (base.get("estimate_sources") or {}).get("acceptance")
                                     or self.base.kind, "final_offer": self.name},
                "acceptance_platt_applied": bool(base.get("acceptance_platt_applied")),
                "model_estimates_used": ["final_offer"] if self.is_model else [],
                "spec_sha256": X._digest(json_safe(identity)), "threshold": self.threshold,
                "prior_table": list(self.prior_table), "instruction_rule": None, "inner": base}

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


def _constant(final_source: str, inner=None) -> ComposedBackend:
    return ComposedBackend(inner, EstimateSpec("constant", X.ACCEPTANCE_CONSTANT, None, final_source))


def make_arm_backend(arm: str, *, model_backend=None, thresholds: Thresholds | None = None,
                     existing_threshold: float | None = None, hidden=None):
    """The backend of one arm. A model arm needs `model_backend` (the stand-in or the real model seat).

    A missing fitted value is refused, never replaced by a default: utt_gate_fit needs the fitted
    pair, model_final_gate_fit the fitted tau_m, true_final the case's hidden parameters.
    """
    if arm not in X.ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {X.ARMS}")
    if arm in X.MODEL_ARMS and model_backend is None:
        raise ValueError(f"arm {arm} needs a model backend")
    if arm == "prior":
        return _constant("policy_prior")
    if arm == "keyword":
        return _constant("keyword")
    if arm == "label_reader":
        # The label reader answers 0.0 or 1.0, so the fixed pair reads it exactly; no fitted value is involved.
        return UtteranceFinalBackend(_constant("policy_prior"), LabelReader(W.label_text), thresholds=FIXED_HALF,
                                     name="label_reader")
    if arm == "true_final":
        if hidden is None:
            raise ValueError("arm true_final is built per case from that case's hidden parameters")
        return TrueFinalReference(hidden)
    if arm == "utt_gate_fit":
        if thresholds is None:
            raise X.ExperimentRefusal("arm utt_gate_fit needs the thresholds fitted on the fit block; none is "
                                      "available in this state directory")
        return UtteranceFinalBackend(_constant("policy_prior"), ModelReader(model_backend), thresholds=thresholds,
                                     name="utt_gate_fit")
    if arm == "utt_gate":
        return UtteranceFinalBackend(_constant("policy_prior"), ModelReader(model_backend), thresholds=FIXED_HALF,
                                     name="utt_gate")
    if arm == "model_final":
        return _constant("model", model_backend)
    if arm == "model_final_gate":
        return ModelFinalGate(_constant("model", model_backend), threshold=MODEL_FINAL_GATE_HALF,
                              name="model_final_gate")
    if arm == "model_final_gate_fit":
        if existing_threshold is None:
            raise X.ExperimentRefusal("arm model_final_gate_fit needs the fitted tau_m of the existing question; "
                                      "the fit has none (no fit yet, or no useful threshold)")
        return ModelFinalGate(_constant("model", model_backend), threshold=existing_threshold,
                              name="model_final_gate_fit")
    return ComposedBackend(model_backend, EstimateSpec())            # m0: the current agent unchanged


# ---- one negotiation -----------------------------------------------------------------------------------

def reading_of(response: dict | None) -> dict:
    """What an arm's response says about how the final-offer value was made: value_from, flag, fired."""
    response = response if isinstance(response, dict) else {}
    for key in ("utterance", "final_gate", "composition", "reference"):
        block = response.get(key)
        if isinstance(block, dict) and "final_offer_value_from" in block:
            value_from = block["final_offer_value_from"]
            return {"value_from": value_from, "fired": value_from in FIRE_VALUES,
                    "instruction_flag": block.get("instruction_flag") if key == "utterance" else None,
                    "utterance_request": key == "utterance" and block.get("request_sha256") is not None}
    return {"value_from": None, "fired": False, "instruction_flag": None, "utterance_request": False}


def run_styled_case(case, style: str, private, backend, config: PolicyConfig, pinned: dict | None = None, *,
                    on_decision: Callable[[dict], None] = lambda record: None,
                    on_step: Callable[[Any, Any], None] = lambda before, decision: None) -> dict:
    """One negotiation against a styled supplier, to an end state or to a failed estimate.

    The loop, the automatic approval and the row are those of the existing evaluator's case runner;
    the supplier is a StyledSupplier instead. The row adds `utterances` (labels of every supplier
    offer) and `utterance_states` (per judged state: the label of the standing utterance and how
    the arm's final-offer value was made). A non-terminal case carries no saving.
    """
    hidden = case.hidden
    state, first = W.styled_initial_state(case.case_id, hidden, style, case.seed, pinned)
    supplier = W.StyledSupplier(hidden, style, case.seed, pinned)
    supplier.labels[state.history[0].offer_id] = first
    session = NegotiationSession(state, private, backend, supplier, config, policy.decide)
    finals, readings, seen, counters_sent = [], [], set(), 0
    for _ in range(80):
        if session.status in TERMINAL or session.status == "needs_review":
            break
        if session.status == "awaiting_approval":
            session.approve(session.pending.request_id, APPROVER)
            continue
        before, executed_before, decisions_before = session.state, len(session.executed), len(session.decisions)
        try:
            decision = session.step()
        finally:
            for record in session.internal_record()[decisions_before:]:
                on_decision(record)
        on_step(before, decision)
        digest = decision.model.get("request_sha256")
        if digest and digest not in seen and "error" not in decision.model:
            seen.add(digest)
            source = decision.facts.get("p_final_offer_source")
            truth = is_last_chance(before, hidden)
            if source in ("backend_estimate", "policy_prior"):
                finals.append({"state_id": case.case_id, "round": decision.round_index, "source": source,
                               "p": decision.facts["p_final_offer"], "truth": truth})
            standing = before.last_offer("supplier")
            labels = supplier.labels[standing.offer_id]
            readings.append({"round": decision.round_index, "offer_id": standing.offer_id,
                             "core_class": labels.core_class, "says_final": labels.says_final,
                             "has_instruction": labels.has_instruction,
                             "instruction_direction": labels.instruction_direction,
                             "is_last_chance": truth, "policy_source": source,
                             "p_final_offer": decision.facts.get("p_final_offer"),
                             **reading_of(decision.model.get("response"))})
        if len(session.executed) > executed_before and session.executed[-1]["action"] == "counter":
            counters_sent += 1
    outcome = session.outcome()
    terminal = session.status in TERMINAL
    saving, cards = outcome["realized_saving_vs_alternative"], outcome["approval_cards"]
    errors = [{"round": d.round_index, "request_sha256": d.model.get("request_sha256"), "error": d.model["error"]}
              for d in session.decisions if "error" in d.model]
    return {
        "case_id": case.case_id, "status": session.status, "agreement": outcome["agreement"],
        "realized_saving": saving if terminal else None,
        "net_saving": (saving - private.approval_cost * cards - private.round_cost * counters_sent) if terminal
        else None,
        "rounds_used": outcome["rounds_used"], "counters_sent": counters_sent, "approval_cards": cards,
        "authority_violations": len(outcome["authority_violations"]),
        "final_offer_estimates": finals,
        "utterances": [{"offer_id": offer.offer_id, "round": offer.round_index,
                        "labels": supplier.labels[offer.offer_id].to_json()}
                       for offer in session.state.history if offer.made_by == "supplier"],
        "utterance_states": readings, "errors": errors,
        "interrupted": any(d.model.get("status") == "interrupted" for d in session.decisions),
    }


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def arm_metrics(rows: list[dict]) -> dict:
    """Summary of an arm's rows. Savings are averaged over the cases that reached an end state."""
    ended = [row for row in rows if row["status"] in TERMINAL]
    judged = [r for row in rows for r in row["utterance_states"] if r["policy_source"] != "public_round_limit"]
    return {
        "cases": len(rows), "terminal_cases": len(ended),
        "mean_realized_saving": _mean(row["realized_saving"] for row in ended),
        "mean_net_saving": _mean(row["net_saving"] for row in ended),
        "agreed": sum(1 for row in rows if row["status"] == "agreed"),
        "supplier_walked": sum(1 for row in rows if row["status"] == "supplier_walked"),
        "ended_alternative": sum(1 for row in rows if row["status"] == "ended_alternative"),
        "needs_review": sum(1 for row in rows if row["status"] == "needs_review"),
        "authority_violations": sum(row["authority_violations"] for row in rows),
        "utterance": {
            "states": len(judged),
            "labelled_final": sum(1 for r in judged if r["says_final"]),
            "fired": sum(1 for r in judged if r["fired"]),
            # Both counts are against the utterance's label, not against the hidden truth: an arm that does
            # not read the utterance (true_final, model_final*) can fire rightly without a final statement.
            "fired_on_final_utterance": sum(1 for r in judged if r["fired"] and r["says_final"]),
            "fired_without_final_utterance": sum(1 for r in judged if r["fired"] and not r["says_final"]),
            "instruction_flags": sum(1 for r in judged if r["instruction_flag"] is True),
            "not_asked_last_round": sum(1 for row in rows for r in row["utterance_states"]
                                        if r["policy_source"] == "public_round_limit"),
            "note": "judged states before the last public round; labels are those of the standing utterance",
        },
    }


# ---- which runs may start ------------------------------------------------------------------------------

def arm_folder(state_dir: Path, block: str, style: str, arm: str) -> Path:
    return Path(state_dir) / FOLDER_BY_BLOCK[block] / style / arm


def stop_rule(state_dir: Path, fitted: dict | None) -> dict:
    """Section 8.3: whether the bench block's reading stops the GPU stages after it. Reads only.

    1. the fit found no useful threshold and R1 with the fixed pair is "not demonstrated";
    2. R1 with the fitted pair is "not demonstrated" and the paraphrased-final check on raw signal
       differences has no lower bound above 0 either.
    """
    attempt = X.first_complete_attempt(Path(state_dir) / "bench" / "bench")
    if attempt is None:
        return {"met": False, "rules": [], "bench_attempt": None}
    result = attempt["result"]
    primary, fixed = result.get("verdict") or {}, result.get("verdict_utt_gate") or {}
    rules = []
    if (fitted or {}).get("status") == NO_USEFUL_THRESHOLD and fixed.get("reading") == READING_NOT_DEMONSTRATED:
        rules.append(1)
    raw = primary.get("raw_signal_reads_paraphrased_final") or {}
    if primary.get("basis") == PRIMARY_ARM and primary.get("reading") == READING_NOT_DEMONSTRATED \
            and not raw.get("passed"):
        rules.append(2)
    return {"met": bool(rules), "rules": rules, "bench_attempt": attempt["name"]}


def check_arm(arm: str, style: str, block: str, *, state_dir: Path, real_model: bool, fitted: dict | None,
              limit: int | None = None) -> None:
    """Raise ExperimentRefusal for a run that must not start. Reads state files only; writes nothing."""
    if arm not in X.ARMS or style not in W.STYLES or block not in X.RUN_BLOCKS:
        raise X.ExperimentRefusal(f"unknown arm, style or block: {arm!r}, {style!r}, {block!r}")
    if limit is not None and block not in C.REHEARSAL_BLOCKS:
        raise X.ExperimentRefusal(f"--limit is a rehearsal option of the smoke block; block {block} always runs "
                                  "at its registered size")
    allowed = X.ARMS_BY_STYLE[style]["always"] + X.ARMS_BY_STYLE[style]["if_time"]
    if arm not in allowed:
        raise X.ExperimentRefusal(f"arm {arm} is not registered for the {style} style; its arms are "
                                  f"{', '.join(allowed)}")
    model_arm = arm in X.MODEL_ARMS
    if block == "test":
        if model_arm and not real_model:
            raise X.ExperimentRefusal(f"--backend cpu-stand-in is a rehearsal and rehearsals draw from the smoke "
                                      f"block only; arm {arm} runs on the test block with the real model")
        if fitted is None:
            raise X.ExperimentRefusal("the test block runs only after the thresholds are fixed; run "
                                      "`bench --block fit`, `fit` and `bench --block bench` first")
        if fitted["fitted_on"] != "utt-fit":
            raise X.ExperimentRefusal("thresholds.json holds rehearsal thresholds fitted on the smoke block; they "
                                      "never run the test block")
        if X.first_complete_attempt(Path(state_dir) / "bench" / "bench") is None:
            raise X.ExperimentRefusal("the test block runs only after the bench block's reading is on record; "
                                      "run `bench --block bench` first")
    if arm == PRIMARY_ARM:
        if fitted is None:
            raise X.ExperimentRefusal("arm utt_gate_fit needs fitted thresholds; run `bench` and `fit` first")
        if fitted["status"] == NO_USEFUL_THRESHOLD:
            raise X.ExperimentRefusal(f"{NOT_RUN_NO_THRESHOLD}: "
                                      + X.result_sentence(NOT_RUN_NO_THRESHOLD))
    if arm == "model_final_gate_fit" and (fitted is None or fitted["existing_threshold"] is None):
        raise X.ExperimentRefusal("arm model_final_gate_fit needs the fitted tau_m of the existing question; the "
                                  "fit has none (no fit yet, or no useful threshold)")
    if model_arm and real_model:
        stop = stop_rule(state_dir, fitted)
        if stop["met"]:
            raise X.ExperimentRefusal(f"stopping rule {stop['rules']} of the design is met by bench attempt "
                                      f"{stop['bench_attempt']}: the reading was not demonstrated, so the "
                                      "model arms of the negotiation comparison are not run")


def _attempt_status(attempt: dict, digest: str) -> str:
    result = attempt["result"]
    if not isinstance(result, dict) or result.get("protocol_sha256") != digest:
        return "unreadable"
    return str(result.get("status"))


def next_attempt(parent: Path, retry_failed: bool) -> dict:
    """What a new attempt in this folder would be: {"action", "link", "earlier_complete", "reason"}.

    action "run": link is {} (first attempt, or a rerun after a completed one), {"continues_after"}
    (the last attempt stopped at the budget, an interrupt, a read-only store miss or a runner
    error: stored answers are reused and it counts as the same attempt) or {"retry_of"} (the last
    attempt failed on an estimate and a person passed --retry-failed).
    action "skip": the last attempt failed and no retry was asked for, or the one retry was used.
    """
    digest = X.protocol_sha256()
    runs = X.attempts(parent)
    complete = X.first_complete_attempt(parent)
    plan = {"action": "run", "link": {}, "earlier_complete": complete["name"] if complete else None,
            "reason": None}
    if complete is not None or not runs:
        return plan
    last = runs[-1]
    status = _attempt_status(last, digest)
    if status != "failed":
        plan["link"] = {"continues_after": last["name"]}
        return plan
    retried = any(isinstance(a["result"], dict) and (a["result"].get("attempt") or {}).get("retry_of")
                  for a in runs)
    if retried:
        return {**plan, "action": "skip", "reason": "refused_second_retry"}
    if not retry_failed:
        return {**plan, "action": "skip", "reason": "skipped_failed"}
    plan["link"] = {"retry_of": last["name"]}
    return plan


# ---- one arm -------------------------------------------------------------------------------------------

def _block_info(block: str, cases: list, limit: int | None) -> dict:
    start, stop = C.BLOCKS[block]
    return {"name": block, "cases": len(cases), "registered_cases": stop - start,
            "first_case_id": cases[0].case_id, "last_case_id": cases[-1].case_id,
            "content_sha256": C.block_content_sha256(cases), "rehearsal_limit": limit}


def run_arm(arm: str, style: str, block: str, *, state_dir: Path, model_backend=None, real_model: bool = False,
            fitted: dict | None = None, limit: int | None = None, budget_minutes: float | None = None,
            retry_failed: bool = False, store_mode: str = "read-write", registry_path: Path | None = None,
            provenance: dict | None = None, clock: Callable[[], float] = time.monotonic,
            started: float | None = None, log: Callable[[str], None] = lambda message: None) -> dict:
    """Run one arm on one block in a new attempt folder. Returns {"status", "result", "folder"}.

    The test block's first use (and the first real-model response) is recorded before anything
    runs. The arm stops at a failed estimate, at an interrupt or once the budget is used up; the
    stored answers of a stopped attempt are reused by the next one. Nothing is overwritten.
    """
    state_dir = Path(state_dir)
    check_arm(arm, style, block, state_dir=state_dir, real_model=real_model, fitted=fitted, limit=limit)
    if arm == "utt_gate" and fitted is not None and fitted["utt_gate"] == SKIPPED_IDENTICAL:
        log(f"{style} {arm}: the fitted pair is (0.5, 0.5); {SKIPPED_IDENTICAL}")
        return {"status": SKIPPED_IDENTICAL, "result": None, "folder": None}
    parent = arm_folder(state_dir, block, style, arm)
    plan = next_attempt(parent, retry_failed)
    if plan["action"] == "skip":
        log(f"{style} {arm}: " + ("failed earlier on an estimate; it is not run again without --retry-failed"
                                  if plan["reason"] == "skipped_failed" else
                                  "Refused: failed again after its one retry; the comparison stays incomplete"))
        return {"status": plan["reason"], "result": None, "folder": None}
    started = clock() if started is None else started
    budget_reached = lambda: budget_minutes is not None and clock() - started >= budget_minutes * 60   # noqa: E731
    if budget_reached():
        log(f"{style} {arm}: the budget was used up before it started")
        return {"status": "partial_budget", "result": None, "folder": None}
    cases = C.block_cases(block, limit)
    model_arm = arm in X.MODEL_ARMS
    thresholds = fitted["thresholds"] if fitted else None
    existing_threshold = fitted["existing_threshold"] if fitted else None
    shared = make_arm_backend(arm, model_backend=model_backend, thresholds=thresholds,
                              existing_threshold=existing_threshold) if arm != "true_final" else None
    use = None
    if real_model and model_arm and store_mode == "read-write":
        use = X.record_first_use(state_dir, "first_real_response", block, registry_path=registry_path)
    if block == "test":
        use = X.record_first_use(state_dir, "test_block_run", block, registry_path=registry_path)

    folder = X.new_attempt_folder(parent)
    private, config = default_buyer_private(), PolicyConfig()
    before = X.request_counts(model_backend) if model_arm else {"live_requests": 0, "live_answers": 0, "store_hits": 0}
    rows: list[dict] = []
    counts = {"decisions": 0, "decision_requests": 0, "failed_estimates": 0, "read_only_store_misses": 0}
    first_errors: list[dict] = []
    status: str | None = None
    reason: str | None = None
    error: str | None = None
    wall_started = time.perf_counter()
    try:
        with (folder / "decisions.jsonl").open("x", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                if budget_reached():
                    status = "partial_budget"
                    break

                def save(record: dict, case_id: str = case.case_id) -> None:
                    line = json_safe({"case_id": case_id, "decision": record})
                    stream.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    model = record.get("model") or {}
                    counts["decisions"] += 1
                    if arm in DEFAULT_REQUEST_ARMS and "trace_id" in model:
                        counts["decision_requests"] += 1
                    if "error" in model and not str(model["error"]).startswith("KeyboardInterrupt"):
                        counts["failed_estimates"] += 1
                        if str(model["error"]).startswith("StoreMiss"):
                            counts["read_only_store_misses"] += 1
                        if len(first_errors) < 5:
                            first_errors.append({"case_id": case_id, "round": record.get("round"),
                                                 "request_sha256": model.get("request_sha256"),
                                                 "error": str(model["error"])[:2000]})

                backend = TrueFinalReference(case.hidden) if arm == "true_final" else shared
                row = run_styled_case(case, style, private, backend, config, on_decision=save)
                rows.append(row)
                if model_arm:
                    log(f"{style} {arm} [{number}/{len(cases)}] {case.case_id} {row['status']}")
                if row["interrupted"]:
                    status = "interrupted"
                elif row["status"] == "needs_review":
                    # A failed estimate stops for a person: no later case may ask again in this attempt.
                    misses_only = all(item["error"].startswith("StoreMiss") for item in row["errors"])
                    status, reason = ("read_only_store_miss", "no_stored_answer_no_model_call") \
                        if misses_only and row["errors"] else ("failed", "estimate_failed")
                elif row["status"] not in TERMINAL:
                    status, reason = "failed", "case_not_terminal"
                if status is not None:
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as failure:                     # noqa: BLE001 - keep the attempt record, never a value
        status, error = "runner_error", f"{type(failure).__name__}: {failure}"
    if status is None:
        status = "complete" if len(rows) == len(cases) else "failed"
    complete = status == "complete"
    delta = X.count_delta(before, X.request_counts(model_backend) if model_arm else before)
    live, answered, hits = delta["live_requests"], delta["live_answers"], delta["store_hits"]
    reader = getattr(shared, "reader", None)
    confirmatory = plan["earlier_complete"] is None and (use is None or use["confirmatory"])
    registered = block == "test" and limit is None
    result: dict[str, Any] = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": X.protocol_sha256(),
        "experiment": "utterance_negotiation_comparison", "created_at_utc": X._utc_now(),
        "style": style, "arm": arm, "arm_spec": X.ARM_SPECS[arm], "block": _block_info(block, cases, limit),
        "environment": ENVIRONMENT,
        "backend": shared.describe() if shared is not None else TrueFinalReference(cases[0].hidden).describe(),
        "evidence_class": X.evidence_class(ARM_KIND[arm], real_model=real_model, live=answered, hits=hits),
        "uses_hidden_information": arm == "true_final",
        "thresholds": {"utt_gate_fit": thresholds.to_json() if thresholds else None,
                       "utt_gate": FIXED_HALF.to_json(), "existing_question": existing_threshold,
                       "model_final_gate": MODEL_FINAL_GATE_HALF,
                       "fit_status": fitted["status"] if fitted else "thresholds_not_fitted_yet",
                       "fitted_on": fitted["fitted_on"] if fitted else None},
        "attempt": {"name": folder.name, "confirmatory": confirmatory,
                    "status": "first_use" if confirmatory else X.NOT_CONFIRMATORY,
                    "retry_of": plan["link"].get("retry_of"), "continues_after": plan["link"].get("continues_after"),
                    "earlier_complete_attempt": plan["earlier_complete"], "first_use": use},
        "complete": complete, "status": status, "reason": reason,
        "counts": {"live_requests": live, "live_failed_requests": live - answered, "store_hits": hits,
                   "utterance_requests": int(getattr(shared, "reader_calls", 0)) if isinstance(reader, ModelReader)
                   else 0, **counts},
        "cases_planned": len(cases), "cases_run": len(rows),
        "metrics": arm_metrics(rows),
        "rows": rows,
        "counts_as_model_evidence": bool(real_model and model_arm and registered and complete and confirmatory),
        "decisions_file": "decisions.jsonl", "wall_seconds": round(time.perf_counter() - wall_started, 3),
        "scope": SCOPE, "provenance": provenance or {},
    }
    if error is not None:
        result["error"] = error
    if first_errors:
        result["first_errors"] = first_errors
    X.write_json(folder / "result.json", result, exclusive=True)
    text = (f"{style} {arm}: status={status}" + (f" reason={reason}" if reason else "")
            + f" cases={len(rows)}/{len(cases)} live={live} store_hits={hits} evidence={result['evidence_class']}"
            + f" attempt={result['attempt']['status']}")
    if first_errors and not complete:
        text += f" first_error={first_errors[0]['error'][:300]!r}"
    log(text)
    return {"status": status, "result": result, "folder": folder}


# ---- statistics and verdicts ---------------------------------------------------------------------------

def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact binomial p for "at least this many wins" among the non-zero differences (p = 1/2)."""
    if isinstance(wins, bool) or isinstance(losses, bool) or not isinstance(wins, int) \
            or not isinstance(losses, int) or wins < 0 or losses < 0:
        raise ValueError("wins and losses are non-negative integers")
    total = wins + losses
    if total == 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2 ** total


def _difference(first: list[dict], second: list[dict], key: str = "realized_saving") -> dict:
    return paired_difference(first, second, key, seed=X.OUTCOME_BOOTSTRAP["seed"],
                             resamples=X.OUTCOME_BOOTSTRAP["resamples"])


def compare(first: list[dict] | None, second: list[dict] | None) -> dict:
    """Paired comparison first - second on the same cases: both savings, win/tie/loss and the sign test."""
    if not first or not second:
        return {"status": "not_run"}
    if [row["case_id"] for row in first] != [row["case_id"] for row in second]:
        return {"status": "case_sets_differ"}
    saving, net = _difference(first, second), _difference(first, second, "net_saving")
    return {"status": "compared", "cases": saving["cases"], "realized_saving": saving, "net_saving": net,
            "sign_test_p_one_sided": sign_test_p(saving["wins"], saving["losses"])}


def outcome_verdict(first_rows: list[dict] | None, second_rows: list[dict] | None, *, authority_violations: int,
                    needs_review: int, complete: bool, fit_status: str | None = None) -> dict:
    """Outcome rule R2 for (first - second) realized saving: beats, worse, not_shown, incomplete.

    `complete` says that both first completed attempts exist. When the fit found no useful threshold
    and the primary arm was therefore not run, the result is not_run_no_useful_threshold.
    """
    if fit_status == NO_USEFUL_THRESHOLD and not first_rows:
        return {"result": NOT_RUN_NO_THRESHOLD, "sentence": X.result_sentence(NOT_RUN_NO_THRESHOLD)}
    if not complete or not first_rows or not second_rows:
        return {"result": INCOMPLETE, "sentence": X.result_sentence(INCOMPLETE)}
    saving, net = _difference(first_rows, second_rows), _difference(first_rows, second_rows, "net_saving")
    low, high = saving["ci90"]
    p_value = sign_test_p(saving["wins"], saving["losses"])
    conditions = {"ci90_lower_bound_above_0": low > 0,
                  "mean_net_saving_difference_at_least_0": net["mean_difference"] >= 0,
                  f"sign_test_p_at_most_{X.SIGN_TEST_ALPHA}": p_value <= X.SIGN_TEST_ALPHA,
                  "no_authority_violation": authority_violations == 0, "no_needs_review": needs_review == 0}
    unmet = [name for name, met in conditions.items() if not met]
    detail = None
    if not unmet:
        result = BEATS
    elif high < 0:
        result = WORSE
    else:
        result = NOT_SHOWN
        detail = NOT_SHOWN_CONTAINS_ZERO if low <= 0 else NOT_SHOWN_UNMET + ", ".join(unmet)
    return {"result": result, "sentence": X.result_sentence(result, detail), "conditions": conditions,
            "unmet_conditions": unmet, "realized_saving": saving, "net_saving": net,
            "sign_test_p_one_sided": p_value, "authority_violations": authority_violations,
            "needs_review": needs_review}


def _interval(comparison: dict | None) -> list[float] | None:
    if not isinstance(comparison, dict):
        return None
    block = comparison.get("realized_saving") if "realized_saving" in comparison else comparison
    return block.get("ci90") if isinstance(block, dict) else None


def adoption_decision(r2: dict, s1: dict | None, guard: dict | None, s9: dict | None,
                      model_final_judged: bool) -> dict:
    """Section 11.5, read from the top; the first matching case is the decision.

    `model_final_judged` says whether model_final has a first completed attempt: only its absence is
    the first row ("model_final not run"). `s1` is None whenever S1 could not be compared, which also
    happens when the primary arm has no rows (not run for lack of a useful threshold, or failed); that
    case falls through to row 2 ("A not met") with R2's own sentence. `s9` is None when
    model_final_gate_fit was not run. The fixed-0.5 comparison (S10) is deliberately not an input.
    Conditions C (the guard) and D (S9) can only make the decision harder.
    """
    s1_interval, s9_interval = _interval(s1), _interval(s9)
    condition_a = r2.get("result") == BEATS
    condition_b = s1_interval is not None and s1_interval[0] > 0
    guard_result = (guard or {}).get("result")
    condition_c = guard_result == "instruction_guard_holds"
    d_hit = s9_interval is not None and s9_interval[1] < 0
    conditions = {"A": condition_a, "B": condition_b, "C": condition_c, "D": not d_hit,
                  "s9_compared": s9_interval is not None}
    sentences: list[str]
    if not model_final_judged:
        case, decision = "model_final_not_run", "incomplete"
        sentences = [X.result_sentence("adoption_incomplete_no_model_final")]
    elif not condition_a:
        case, decision = "primary_not_met", "not_adopted"
        sentences = [r2.get("sentence") or X.result_sentence(INCOMPLETE)]
    elif not condition_b:
        case, decision = "not_better_than_existing_question", "keep_existing_question"
        sentences = [X.result_sentence("adoption_keep_existing_question"), EXISTING_QUESTION_CAVEAT]
    elif d_hit:
        case, decision = "worse_than_fitted_existing_question", "keep_existing_question"
        sentences = [X.result_sentence("adoption_keep_existing_question_fitted_gate"), EXISTING_QUESTION_CAVEAT]
    elif guard_result is None:
        # Not a row of the table: the guard check was never judged. Never more favourable than "not C".
        case, decision = "guard_not_judged", "incomplete"
        sentences = [X.result_sentence(INCOMPLETE)]
    elif not condition_c:
        case, decision = "guard_failed", "not_adopted"
        sentences = [X.result_sentence("adoption_rejected_guard_failed")]
    else:
        case, decision = "adoption_candidate", "adoption_candidate"
        sentences = [X.result_sentence(BEATS), X.result_sentence("instruction_guard_holds"),
                     f"S1 90% 구간: [{s1_interval[0]:.0f}, {s1_interval[1]:.0f}]원"]
        if s9_interval is None:
            sentences.append(X.result_sentence("adoption_s9_not_compared"))
    return {"decision": decision, "case": case, "conditions": conditions, "sentences": sentences,
            "s1_ci90": s1_interval, "s9_ci90": s9_interval,
            "note": "what may be said at the selection meeting; the choice itself is the user's"}


# ---- report --------------------------------------------------------------------------------------------

def _arm_entry(state_dir: Path, block: str, style: str, arm: str) -> dict | None:
    """One arm's attempts: the first completed one is judged, later ones are not_confirmatory."""
    parent = arm_folder(state_dir, block, style, arm)
    runs = X.attempts(parent)
    if not runs:
        return None
    judged = X.first_complete_attempt(parent)
    digest = X.protocol_sha256()
    listed, after = [], False
    for attempt in runs:
        is_judged = judged is not None and attempt["name"] == judged["name"]
        listed.append({"name": attempt["name"], "status": _attempt_status(attempt, digest),
                       "use": "judged" if is_judged else (X.NOT_CONFIRMATORY if after else "not_complete")})
        after = after or is_judged
    entry: dict[str, Any] = {"attempts": listed, "judged_attempt": judged["name"] if judged else None,
                             "rows": None}
    if judged is not None:
        result = judged["result"]
        entry.update(rows=result["rows"], evidence_class=result["evidence_class"], metrics=result["metrics"],
                     counts=result["counts"], confirmatory=bool(result["attempt"]["confirmatory"]),
                     counts_as_model_evidence=bool(result["counts_as_model_evidence"]),
                     uses_hidden_information=bool(result.get("uses_hidden_information")))
    else:
        entry["failed_after_retry"] = next_attempt(parent, True)["reason"] == "refused_second_retry"
    return entry


def _states(rows: list[dict] | None, where: Callable[[dict], bool]) -> list[tuple[dict, dict]]:
    return [(row, r) for row in rows or [] for r in row["utterance_states"]
            if r["policy_source"] != "public_round_limit" and where(r)]


def _share_fired(rows: list[dict] | None, where: Callable[[dict], bool]) -> dict:
    states = _states(rows, where)
    fired = sum(1 for _, r in states if r["fired"])
    return {"states": len(states), "fired": fired, "share": fired / len(states) if states else None}


def _mean_difference_on(case_ids: set[str], first: list[dict] | None, second: list[dict] | None) -> float | None:
    if not first or not second:
        return None
    other = {row["case_id"]: row["realized_saving"] for row in second}
    return _mean(row["realized_saving"] - other[row["case_id"]] for row in first
                 if row["case_id"] in case_ids and row["case_id"] in other)


def recovered_share(gate: list[dict] | None, prior: list[dict] | None, keyword_default: list[dict] | None,
                    prior_default: list[dict] | None) -> dict:
    """S3: (gate - prior, paraphrase) / (keyword - prior, default), resampling the same cases together.

    No ratio is given when the denominator's 90% interval contains 0.
    """
    if not gate or not prior or not keyword_default or not prior_default:
        return {"status": "not_run"}
    ids = [row["case_id"] for row in gate]
    if any([row["case_id"] for row in rows] != ids for rows in (prior, keyword_default, prior_default)):
        return {"status": "case_sets_differ"}
    numerator = [a["realized_saving"] - b["realized_saving"] for a, b in zip(gate, prior)]
    denominator = [a["realized_saving"] - b["realized_saving"] for a, b in zip(keyword_default, prior_default)]
    bottom = _difference(keyword_default, prior_default)
    out = {"status": "compared", "numerator_mean": _mean(numerator), "denominator_mean": _mean(denominator),
           "denominator_ci90": bottom["ci90"], "ratio": None, "ratio_ci90": None}
    if bottom["ci90"][0] <= 0 <= bottom["ci90"][1]:
        out["note"] = "비율 미정: the denominator's 90% interval contains 0"
        return out
    rng = random.Random(X.OUTCOME_BOOTSTRAP["seed"])
    count, ratios = len(ids), []
    for _ in range(X.OUTCOME_BOOTSTRAP["resamples"]):
        picks = [rng.randrange(count) for _ in range(count)]
        low = sum(denominator[i] for i in picks)
        if low != 0:
            ratios.append(sum(numerator[i] for i in picks) / low)
    ratios.sort()
    out["ratio"] = out["numerator_mean"] / out["denominator_mean"]
    if ratios:
        out["ratio_ci90"] = [ratios[int(0.05 * len(ratios))], ratios[max(0, int(0.95 * len(ratios)) - 1)]]
    return out


def _block_report(state_dir: Path, block: str, fitted: dict | None) -> dict:
    """Arms, the primary comparison and the secondary comparisons of one block."""
    entries: dict[str, dict[str, dict]] = {}
    for style in W.STYLES:
        for arm in X.ARMS:
            entry = _arm_entry(state_dir, block, style, arm)
            if entry is not None:
                entries.setdefault(style, {})[arm] = entry

    def rows(style: str, arm: str) -> list[dict] | None:
        return (entries.get(style, {}).get(arm) or {}).get("rows")

    def metric(style: str, arm: str) -> dict | None:
        return (entries.get(style, {}).get(arm) or {}).get("metrics")

    gate, prior = rows(PRIMARY_STYLE, PRIMARY_ARM), rows(PRIMARY_STYLE, CONTROL_ARM)
    fixed = rows(PRIMARY_STYLE, "utt_gate")
    fixed_note = None
    if fixed is None and fitted is not None and fitted["utt_gate"] == SKIPPED_IDENTICAL:
        fixed, fixed_note = gate, SKIPPED_IDENTICAL
    gate_metrics = metric(PRIMARY_STYLE, PRIMARY_ARM) or {}
    gate_entry = entries.get(PRIMARY_STYLE, {}).get(PRIMARY_ARM) or {}
    primary = outcome_verdict(
        gate, prior, authority_violations=int(gate_metrics.get("authority_violations", 0)),
        needs_review=int(gate_metrics.get("needs_review", 0)), complete=bool(gate and prior),
        fit_status=fitted["status"] if fitted else None)
    primary["comparison"] = f"{PRIMARY_ARM} - {CONTROL_ARM}, {PRIMARY_STYLE} style, realized saving"
    primary["failed_after_one_retry"] = bool(gate_entry.get("failed_after_retry"))

    adv_gate, adv_prior = rows("adversarial", PRIMARY_ARM), rows("adversarial", CONTROL_ARM)
    adv_keyword, adv_existing = rows("adversarial", "keyword"), rows("adversarial", "model_final_gate_fit")
    is_decoy = lambda r: r["core_class"] in DECOY_CLASSES                                        # noqa: E731
    final_with_instruction = lambda r: r["says_final"] and r["has_instruction"]                  # noqa: E731
    to_final_on_non_final = lambda r: not r["says_final"] and r["instruction_direction"] == "to_final"   # noqa: E731

    def decoys(arm_rows: list[dict] | None) -> dict:
        out = _share_fired(arm_rows, is_decoy)
        hit = {row["case_id"] for row, r in _states(arm_rows, is_decoy) if r["fired"]}
        total = _mean_difference_on(hit, arm_rows, adv_prior)
        out["loss_per_decoy_fire_vs_prior"] = (-total * len(hit) / out["fired"]) if total is not None and out["fired"] \
            else None
        return out

    flagged_final = _states(adv_gate, final_with_instruction)
    flagged_cases = {row["case_id"] for row, _ in flagged_final}
    default_prior = rows("default", CONTROL_ARM) or prior      # the prior arm reads no message: same in every style
    secondary = {
        "S1": {"what": "utt_gate_fit - model_final (paraphrase); co-primary",
               **compare(gate, rows(PRIMARY_STYLE, "model_final"))},
        "S2": {"what": "utt_gate_fit - m0 (paraphrase)", **compare(gate, rows(PRIMARY_STYLE, "m0"))},
        "S3": {"what": "share of the keyword rule's default-style gain recovered under paraphrase",
               **recovered_share(gate, prior, rows("default", "keyword"), default_prior),
               "final_utterance_states": (gate_metrics.get("utterance") or {}).get("labelled_final"),
               "final_utterance_states_fired": (gate_metrics.get("utterance") or {}).get("fired_on_final_utterance")},
        "S4": {"what": "adversarial style: utt_gate_fit - keyword and - prior; decoys",
               "gate_minus_keyword": compare(adv_gate, adv_keyword), "gate_minus_prior": compare(adv_gate, adv_prior),
               "decoys": {"utt_gate_fit": decoys(adv_gate), "keyword": decoys(adv_keyword)}},
        "S5": {"what": "the primary comparison, S1 and S2 with the fixed 0.5 pair (utt_gate)", "note": fixed_note,
               "gate_minus_prior": compare(fixed, prior),
               "gate_minus_model_final": compare(fixed, rows(PRIMARY_STYLE, "model_final")),
               "gate_minus_m0": compare(fixed, rows(PRIMARY_STYLE, "m0"))},
        "S6": {"what": "reading errors (gate - label_reader) and what utterances cannot tell "
                       "(label_reader - true_final)",
               "gate_minus_label_reader": compare(gate, rows(PRIMARY_STYLE, "label_reader")),
               "label_reader_minus_true_final": compare(rows(PRIMARY_STYLE, "label_reader"),
                                                        rows(PRIMARY_STYLE, "true_final"))},
        "S7": {"what": "adversarial style: cost of the instruction rule on true final utterances",
               "status": "compared" if adv_gate else "not_run",
               "final_utterances_with_instruction": len(flagged_final),
               "fell_back_to_round_prior": sum(1 for _, r in flagged_final
                                               if r["value_from"] == "instruction_flagged_round_prior"),
               "cases": len(flagged_cases),
               "mean_saving_difference_vs_prior": _mean_difference_on(flagged_cases, adv_gate, adv_prior),
               "mean_saving_difference_vs_true_final": _mean_difference_on(flagged_cases, adv_gate,
                                                                           rows("adversarial", "true_final"))},
        "S8": {"what": "utt_gate_fit - keyword (default style)",
               **compare(rows("default", PRIMARY_ARM), rows("default", "keyword"))},
        "S9": {"what": "utt_gate_fit - model_final_gate_fit (paraphrase); an upper bound below 0 drops the new "
                       "questions", **compare(gate, rows(PRIMARY_STYLE, "model_final_gate_fit"))},
        "S10": {"what": "the fixed-0.5 existing-question arm; reported only, never an adoption input",
                "gate_minus_model_final_gate": compare(gate, rows(PRIMARY_STYLE, "model_final_gate")),
                "model_final_gate_minus_prior": compare(rows(PRIMARY_STYLE, "model_final_gate"), prior)},
        "S11": {"what": "adversarial style: fire share on non-final utterances carrying the to-final instruction",
                "utt_gate_fit": _share_fired(adv_gate, to_final_on_non_final),
                "model_final_gate_fit": _share_fired(adv_existing, to_final_on_non_final),
                "model_final_gate_fit_minus_prior": compare(adv_existing, adv_prior)},
    }
    arms = {style: {arm: {key: value for key, value in entry.items() if key != "rows"}
                    for arm, entry in by_arm.items()} for style, by_arm in entries.items()}
    return {"block": block, "arms": arms, "primary": primary, "secondary": secondary}


def build_report(state_dir: Path) -> dict:
    """Every reading of a state directory, from the first completed attempts only.

    R1 and I1 come from the bench block's first completed attempt; R2, S1-S11 and the adoption
    decision from the test block. Smoke-block runs are listed apart as a rehearsal: paired
    differences on ten cases check the path and carry no verdict.
    """
    state_dir = Path(state_dir)
    X.check_protocol_binding(state_dir)
    fitted = B.load_state_thresholds(state_dir)
    bench_attempt = X.first_complete_attempt(state_dir / "bench" / "bench")
    bench = None
    if bench_attempt is not None:
        result = bench_attempt["result"]
        bench = {"attempt": bench_attempt["name"], "evidence_class": result["evidence_class"],
                 "reading": result["verdict"], "reading_utt_gate": result["verdict_utt_gate"],
                 "guard": result["guard"], "counts": result["counts"],
                 "counts_as_model_evidence": bool(result["verdict"]["counts_as_model_evidence"])}
    registered = _block_report(state_dir, "test", fitted)
    rehearsal = _block_report(state_dir, "smoke", fitted)
    rehearsal["primary"] = {key: rehearsal["primary"].get(key) for key in
                            ("realized_saving", "net_saving", "sign_test_p_one_sided")}
    rehearsal.update(registered=False, verdict=None,
                     label="smoke_block_rehearsal_path_check_only_no_registered_verdict")
    primary, secondary = registered["primary"], registered["secondary"]
    s1 = secondary["S1"] if secondary["S1"]["status"] == "compared" else None
    s9 = secondary["S9"] if secondary["S9"]["status"] == "compared" else None
    model_final_entry = registered["arms"].get(PRIMARY_STYLE, {}).get("model_final") or {}
    adoption = adoption_decision(primary, s1, bench["guard"] if bench else None, s9,
                                 model_final_judged=bool(model_final_entry.get("judged_attempt")))
    primary_entry = registered["arms"].get(PRIMARY_STYLE, {}).get(PRIMARY_ARM) or {}
    labels = sorted({entry["evidence_class"] for block in (registered, rehearsal)
                     for by_arm in block["arms"].values() for entry in by_arm.values()
                     if entry.get("judged_attempt")} | ({bench["evidence_class"]} if bench else set()))
    sentences = [primary["sentence"]]
    if bench is not None:
        sentences = [s for s in (bench["reading"].get("sentence"), bench["guard"].get("sentence")) if s] + sentences
    return {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": X.protocol_sha256(),
        "experiment": "utterance_experiment_report", "created_at_utc": X._utc_now(), "gpu_notice": GPU_NOTICE,
        "state_dir_name": state_dir.name, "registered_state_dir_name": X.STATE_DIR_NAME,
        "thresholds": ({"status": fitted["status"], "fitted_on": fitted["fitted_on"],
                        "utt_gate_fit": fitted["thresholds"].to_json() if fitted["thresholds"] else None,
                        "existing_question": fitted["existing_threshold"],
                        "existing_question_status": fitted["existing_status"], "utt_gate": fitted["utt_gate"]}
                       if fitted else None),
        "bench": bench, "registered": registered, "rehearsal": rehearsal,
        "primary": primary, "adoption": adoption, "sentences": sentences + adoption["sentences"],
        "evidence_classes": labels,
        # Only a judged primary-arm attempt that itself counts (real model, registered size, complete,
        # confirmatory) makes the primary reading model evidence.
        "counts_as_model_evidence": bool(primary_entry.get("counts_as_model_evidence") is True
                                         and primary.get("result") in (BEATS, WORSE, NOT_SHOWN)),
        "environment": ENVIRONMENT, "scope": SCOPE,
        "attempt_rule": "per arm and style the first completed attempt is judged; later attempts are "
                        "not_confirmatory",
    }


def _won(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}원"


def _comparison_line(name: str, comparison: dict) -> str:
    if comparison.get("status") != "compared":
        return f"- {name}: 실행하지 않음"
    saving = comparison["realized_saving"]
    return (f"- {name}: 평균 차이 {_won(saving['mean_difference'])}, 90% 구간 [{saving['ci90'][0]:,.0f}, "
            f"{saving['ci90'][1]:,.0f}]원, 승 {saving['wins']} / 무 {saving['ties']} / 패 {saving['losses']}, "
            f"부호 검정 한쪽 p = {comparison['sign_test_p_one_sided']:.4f}")


def report_markdown(report: dict) -> str:
    """The report as a short Korean document. Every number is copied from the report dict."""
    lines = [f"# 발언 판독 실험 보고 ({report['protocol_version']})", "",
             f"- 상태 폴더: `{report['state_dir_name']}` (등록된 폴더 이름: `{report['registered_state_dir_name']}`)",
             f"- 프로토콜 해시: `{report['protocol_sha256']}`",
             f"- 근거 등급: {', '.join(report['evidence_classes']) or '결과 없음'}",
             f"- 모델 근거로 셀 수 있는가: {'예' if report['counts_as_model_evidence'] else '아니오'}",
             "- 판정에는 구성·스타일마다 첫 완료 attempt만 썼다. 그 뒤의 실행은 `not_confirmatory`다.", ""]
    bench = report["bench"]
    lines += ["## 판독 벤치 (R1, I1)", ""]
    if bench is None:
        lines += ["- U-BENCH 벤치의 완료된 결과가 없다.", ""]
    else:
        lines += [f"- R1: `{bench['reading']['reading']}` (기준 `{bench['reading'].get('basis')}`, "
                  f"등급 `{bench['evidence_class']}`)",
                  f"- I1: `{bench['guard']['result']}`"]
        lines += [f"- {s}" for s in (bench["reading"].get("sentence"), bench["guard"].get("sentence")) if s] + [""]
    primary = report["primary"]
    lines += ["## 협상 결과 비교 (R2, U-TEST)", "", f"- 1차 결과: `{primary['result']}`", f"- {primary['sentence']}"]
    if "realized_saving" in primary:
        lines.append(_comparison_line("utt_gate_fit − prior (paraphrase)", {
            "status": "compared", "realized_saving": primary["realized_saving"],
            "sign_test_p_one_sided": primary["sign_test_p_one_sided"]}))
    lines.append("")
    for style, by_arm in report["registered"]["arms"].items():
        lines += [f"### 구성별 결과: `{style}`", "", "| 구성 | 등급 | 사례 | 평균 실현 절감 | 평균 순절감 | 판정에 쓴 attempt |",
                  "|---|---|---|---|---|---|"]
        for arm, entry in by_arm.items():
            metrics = entry.get("metrics") or {}
            lines.append(f"| `{arm}` | `{entry.get('evidence_class', '완료된 실행 없음')}` | {metrics.get('cases', '—')} | "
                         f"{_won(metrics.get('mean_realized_saving'))} | {_won(metrics.get('mean_net_saving'))} | "
                         f"{entry.get('judged_attempt') or '—'} |")
        lines.append("")
    secondary = report["registered"]["secondary"]
    lines += ["## 2차 비교", "", _comparison_line("S1 utt_gate_fit − model_final", secondary["S1"]),
              _comparison_line("S2 utt_gate_fit − m0", secondary["S2"]),
              _comparison_line("S8 utt_gate_fit − keyword (default)", secondary["S8"]),
              _comparison_line("S9 utt_gate_fit − model_final_gate_fit", secondary["S9"]), ""]
    adoption = report["adoption"]
    lines += ["## 채택 판단 (11.5절)", "", f"- 판단: `{adoption['decision']}` (경우: `{adoption['case']}`)"]
    lines += [f"- {sentence}" for sentence in adoption["sentences"]]
    lines += ["- 고르는 것은 사용자다.", ""]
    rehearsal = report["rehearsal"]
    if rehearsal["arms"]:
        lines += ["## 리허설 (U-SMOKE, 경로 확인 전용)", "",
                  "- 사례 10건 이하의 경로 확인이다. 판정이 없고 비교에 넣지 않는다.", ""]
        for style, by_arm in rehearsal["arms"].items():
            for arm, entry in by_arm.items():
                metrics = entry.get("metrics") or {}
                lines.append(f"- `{style}` / `{arm}`: 등급 `{entry.get('evidence_class', '완료된 실행 없음')}`, "
                             f"사례 {metrics.get('cases', '—')}, 평균 실현 절감 {_won(metrics.get('mean_realized_saving'))}")
        lines.append("")
    lines += ["## 한계", "", "- 규칙 기반 합성 공급사와 코드가 만든 문장에 대한 결과다. 실제 공급사 발언에 대한 근거가 아니다.",
              "- 보류 평가 결과가 아니다.", ""]
    return "\n".join(lines)


# ---- commands ------------------------------------------------------------------------------------------

def command_run(args: Any, *, backend_factory: Callable[[], Any] | None = None,
                declared: tuple[str, bool, bool] | None = None, real_model: bool | None = None,
                registry_path: Path | None = None, clock: Callable[[], float] = time.monotonic) -> int:
    """`run`: the requested arms on one block and style. Exit 0 done, 1 refused or not complete, 130 interrupted.

    Every refusal that depends on the state directory is raised for all requested arms before a
    first use is recorded or an attempt folder exists.
    """
    real = X.real_model_requested(args) if real_model is None else bool(real_model)
    state_dir = Path(args.state_dir)
    if args.limit is not None and args.block not in C.REHEARSAL_BLOCKS:
        raise X.ExperimentRefusal(f"--limit is a rehearsal option of the smoke block; block {args.block} always "
                                  "runs at its registered size")
    if args.limit is not None and (isinstance(args.limit, bool) or not isinstance(args.limit, int) or args.limit < 1):
        raise X.ExperimentRefusal("--limit must be a positive whole number")
    if args.budget_minutes is not None and not args.budget_minutes > 0:
        raise X.ExperimentRefusal("--budget-minutes must be positive")
    X.check_golden_requests()
    X.ensure_protocol(state_dir, create=True)
    fitted = B.load_state_thresholds(state_dir)
    for arm in args.arms:
        check_arm(arm, args.style, args.block, state_dir=state_dir, real_model=real, fitted=fitted,
                  limit=args.limit)
    # Checked after the refusals above so that their messages come first. (The protocol's block content
    # hashes already build every block's case list; nothing is negotiated or read from a case here.)
    try:
        C.block_cases(args.block, args.limit)
    except ValueError as error:
        raise X.ExperimentRefusal(str(error)) from error
    started = clock()
    source = X.run_provenance(args)
    log = lambda message: print(message, file=sys.stderr, flush=True)   # noqa: E731
    model_backend = None
    if any(arm in X.MODEL_ARMS for arm in args.arms):
        model_backend = X.make_model_backend(args, factory=backend_factory, declared=declared)
    statuses = []
    try:
        for arm in args.arms:
            outcome = run_arm(arm, args.style, args.block, state_dir=state_dir, model_backend=model_backend,
                              real_model=real, fitted=fitted, limit=args.limit,
                              budget_minutes=args.budget_minutes, retry_failed=args.retry_failed,
                              store_mode=args.store_mode, registry_path=registry_path, provenance=source,
                              clock=clock, started=started, log=log)
            statuses.append(outcome["status"])
            result = outcome["result"]
            if result is None:
                print(f"{args.style} {arm}  block={args.block}  not run: {outcome['status']}")
            else:
                saving = result["metrics"]["mean_realized_saving"]
                print(f"{outcome['folder'] / 'result.json'}  style={args.style} arm={arm} block={args.block} "
                      f"complete={result['complete']} status={result['status']} "
                      f"evidence_class={result['evidence_class']} live_requests={result['counts']['live_requests']} "
                      f"store_hits={result['counts']['store_hits']} attempt={result['attempt']['status']} "
                      f"counts_as_model_evidence={result['counts_as_model_evidence']}"
                      + (f" mean_realized_saving={saving:.0f}" if result["complete"] and saving is not None else ""))
            if outcome["status"] == "interrupted":
                break
    finally:
        if model_backend is not None:
            model_backend.close()
    if "interrupted" in statuses:
        return 130
    return 0 if all(status in ("complete", SKIPPED_IDENTICAL) for status in statuses) else 1


def command_report(args: Any) -> int:
    """`report`: write report.json and report.md from the first completed attempts. No model call."""
    state_dir = Path(args.state_dir)
    X.ensure_protocol(state_dir, create=False)
    report = build_report(state_dir)
    X.write_json(state_dir / "report.json", report)
    (state_dir / "report.md").write_text(report_markdown(report), encoding="utf-8", newline="\n")
    bench = report["bench"]
    print(f"{state_dir / 'report.json'}  primary={report['primary']['result']} "
          f"reading={bench['reading']['reading'] if bench else None} "
          f"guard={bench['guard']['result'] if bench else None} adoption={report['adoption']['decision']} "
          f"evidence_classes={','.join(report['evidence_classes']) or 'none'} "
          f"counts_as_model_evidence={report['counts_as_model_evidence']}")
    for sentence in report["sentences"]:
        print(sentence)
    return 0


def report(state_dir: Path) -> dict:
    """The report of a state directory (see build_report)."""
    return build_report(state_dir)
