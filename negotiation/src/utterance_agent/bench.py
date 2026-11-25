"""E1: the utterance-reading bench, the threshold fit, the reading rule R1 and the guard check I1.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only, with the CPU stand-in
and invented answers. No number produced here so far is model evidence.

Design: negotiation/docs/UTTERANCE_AGENT_DESIGN.md, sections 5 (fit), 8.1 (bench, R1, I1), 13.6.

One state per supplier of a block; only the latest supplier message changes between the twelve
items of a state, so every comparison is paired within a state. Each item sends two requests: the
four utterance questions, and the existing final-offer question alone. The existing question's
answers fit its own threshold and are reported next to the new reading; they never enter R1.

"Fire" always means the gate's actual output: `combine()` returned 1.0. An utterance flagged as
carrying an instruction does not fire, whatever its signal.

This is evaluator-side code: it reads the hidden supplier parameters to build states and the
sentence labels to score answers. Neither reaches a request.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from negotiation_agent.provenance import request_sha256, reset_response, response_snapshot, validate_predictions
from negotiation_agent.requests import FINAL_OFFER_QID
from negotiation_agent.simulator import MAX_ROUNDS, state_after_scripted_rounds
from negotiation_agent.utterance_probe import with_latest_supplier_message

from . import PROTOCOL_VERSION
from . import cases as C
from . import experiment as X
from . import wording as W
from .interpreter import FIXED_HALF, LiteralReader, ModelReader, Thresholds, combine
from .questions import QIDS, build_existing_final_request, build_utterance_request

PARAPHRASED_FINAL = ("final_limit", "final_walk", "final_with_payment")     # recall items
NON_FINAL = ("neutral", "payment_only", "deadline_pressure", "negated_keyword", "open")   # false-fire items
INSTRUCTION_ITEMS = ("neutral_with_instruction_to_final", "final_with_instruction_to_open")
GUARD_ITEM = "neutral_with_instruction_to_final"
SEPARATELY_REPORTED = ("explicit_final", "probe_paraphrase_final")
HELD_OUT_ITEM = "probe_paraphrase_final"   # outside the bank: never fitted on, never in R1 or an aggregate (6.2)
CHECKS = {   # name -> the non-final item the paraphrased final sentences are compared with
    "reads_paraphrased_final": "neutral",
    "separates_negation": "negated_keyword",
    "not_fooled_by_schedule": "deadline_pressure",
    "not_fooled_by_payment": "payment_only",
}
FITTED_ON_BY_BLOCK = {"fit": "utt-fit", "smoke": "utt-smoke-rehearsal"}
THRESHOLDS_FILE = "thresholds.json"
LONGEST_CHECK_CLASS = "longest_state_check"


# ---- items ---------------------------------------------------------------------------------------------

def _silent_copy(hidden):
    """The same supplier, revealing nothing: every earlier message in the state is the neutral one."""
    return replace(hidden, reveals_payment_preference=False, reveals_deadline=False, claims_buyer_approval=False)


def _pick(name: str, index: int) -> tuple[str, str]:
    rows = W.BANK[name]
    return rows[(index // 3) % len(rows)]


def bench_items(block: str, limit: int | None = None) -> list[dict]:
    """The bench items of one block: twelve per state, in BENCH_ITEMS order.

    Each item is {state_id, case_id, block_index, depth, round, class, sentence_ids, message, labels,
    state}. The case ordinal is the position inside the block (its first case is 0): that reading
    reproduces the state shape the design document reports for the bench block. `depth` is the
    number of scripted exchanges asked for (ordinal % 3); a supplier that ends early gives a
    shallower state. The sentence of a class is (ordinal // 3) % its count, so sentence and depth do
    not move together; paired items share their sentences.
    """
    if block not in X.BENCH_BLOCKS:
        raise ValueError(f"the bench runs on {X.BENCH_BLOCKS}; got {block!r}")
    items = []
    for index, case in enumerate(C.block_cases(block, limit)):
        depth = index % 3
        state_id = f"{case.case_id}-d{depth}"
        base = state_after_scripted_rounds(state_id, _silent_copy(case.hidden), depth)
        neutral, payment, deadline = _pick("neutral", index), _pick("payment", index), _pick("deadline_pressure", index)
        limit_final, walk = _pick("final_limit", index), _pick("final_walk", index)
        negated, opened = _pick("negated_keyword", index), _pick("open", index)
        held_out = W.HELD_OUT_FINAL[(index // 3) % len(W.HELD_OUT_FINAL)]
        to_final, to_open = W.BANK["instruction_to_final"][0], W.BANK["instruction_to_open"][0]
        rows = {
            "neutral": [neutral],
            "explicit_final": [W.BANK["explicit_final"][0]],
            "final_limit": [limit_final],
            "final_walk": [walk],
            "final_with_payment": [payment, limit_final],
            "payment_only": [payment],
            "deadline_pressure": [deadline],
            "negated_keyword": [negated],
            "open": [opened],
            "neutral_with_instruction_to_final": [neutral, to_final],
            "final_with_instruction_to_open": [limit_final, to_open],
            "probe_paraphrase_final": [held_out],
        }
        for name in X.BENCH_ITEMS:
            parts = rows[name]
            message = " ".join(sentence for _, sentence in parts)
            items.append({
                "state_id": state_id, "case_id": case.case_id, "block_index": index, "depth": depth,
                "round": base.current_round, "class": name,
                "sentence_ids": [sentence_id for sentence_id, _ in parts], "message": message,
                "labels": W.label_text(message).to_json(),
                "state": with_latest_supplier_message(base, message),
            })
    return items


def longest_adversarial_utterance() -> str:
    """The longest utterance (UTF-8 bytes) the adversarial style can give a supplier of these blocks.

    One payment sentence, one core sentence and one instruction sentence. The approval-claim
    sentence is left out: no supplier of this experiment's blocks makes that claim.
    """
    def longest(names: tuple[str, ...]) -> str:
        return max((sentence for name in names for _, sentence in W.BANK[name]),
                   key=lambda sentence: len(sentence.encode("utf-8")))

    return " ".join((longest(("payment",)),
                     longest(("final_limit", "final_walk", "deadline_pressure", "negated_keyword", "open")),
                     longest(("instruction_to_final", "instruction_to_open", "instruction_to_accept"))))


def _request_bytes(request: dict) -> int:
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def longest_state_item() -> dict:
    """The token-limit check item of the smoke bench.

    Every smoke supplier (silent copy) is advanced by scripted exchanges as far as it goes; every
    supplier message of that state is replaced by the longest adversarial utterance; the state whose
    utterance request is largest in UTF-8 bytes is the item (the first one on a tie). It is asked
    once, only its per-question input tokens are recorded, and it enters no reading aggregate.
    """
    message, best = longest_adversarial_utterance(), None
    for case in C.block_cases("smoke"):
        state_id = f"{case.case_id}-longest"
        state = state_after_scripted_rounds(state_id, _silent_copy(case.hidden), MAX_ROUNDS)
        history = tuple(replace(offer, message=message) if offer.made_by == "supplier" else offer
                        for offer in state.history)
        state = replace(state, history=history)
        size = _request_bytes(build_utterance_request(state))
        if best is None or size > best["bytes"]:
            best = {"state_id": state_id, "case_id": case.case_id, "class": LONGEST_CHECK_CLASS,
                    "round": state.current_round, "message": message, "bytes": size,
                    "supplier_messages": sum(1 for offer in state.history if offer.made_by == "supplier"),
                    "labels": W.label_text(message).to_json(), "state": state}
    assert best is not None
    return best


# ---- asking --------------------------------------------------------------------------------------------

def _public_record(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "state"}


def _store_hit(response: dict | None) -> bool | None:
    block = response.get("prediction_store") if isinstance(response, dict) else None
    return bool(block.get("hit")) if isinstance(block, dict) and "hit" in block else None


def _ask_utterance(reader: ModelReader, state) -> dict:
    """One utterance request. A failure is recorded as a failure; no answer is invented."""
    out: dict[str, Any] = {"request_sha256": request_sha256(build_utterance_request(state))}
    started = time.perf_counter()
    response = None
    try:
        answers, _, response = reader.read(state)
        out.update(status="ok", answers=answers)
    except KeyboardInterrupt:
        out.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:  # noqa: BLE001 - keep the failed attempt, fill in nothing
        out.update(status="error", error=f"{type(error).__name__}: {error}")
        response = response_snapshot(reader.backend)
    out["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    out["store_hit"] = _store_hit(response)
    metadata = response.get("metadata") if isinstance(response, dict) else None
    out["input_tokens_per_question"] = metadata.get("input_tokens_per_question") if isinstance(metadata, dict) else None
    return out


def _ask_existing(backend, state) -> dict:
    """The existing final-offer question alone, on the same state."""
    request = build_existing_final_request(state)
    out: dict[str, Any] = {"request_sha256": request_sha256(request)}
    started = time.perf_counter()
    response = None
    try:
        reset_response(backend)
        predicted = backend.predict(state, [], request)
        response = response_snapshot(backend)
        out.update(status="ok", final_offer=validate_predictions(predicted, [FINAL_OFFER_QID])[FINAL_OFFER_QID])
    except KeyboardInterrupt:
        out.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:  # noqa: BLE001 - keep the failed attempt, fill in nothing
        out.update(status="error", error=f"{type(error).__name__}: {error}")
    out["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    out["store_hit"] = _store_hit(response)
    return out


def _max_input_tokens(model_backend) -> int | None:
    """The adapter's per-question token limit, as recorded in the store identity (None for the stand-in)."""
    store = getattr(model_backend, "store", None)
    return store.requested.get("max_input_tokens") if store is not None else None


def run_bench(model_backend, block: str, *, limit: int | None = None, real_model: bool | None = None,
              thresholds: Thresholds | None = None, existing_threshold: float | None = None,
              fit_status: str = "thresholds_not_fitted_yet",
              on_item: Callable[[dict], None] = lambda record: None,
              log: Callable[[str], None] = lambda message: None) -> dict:
    """Ask every bench item of a block (two requests each) and summarise. Independent of any threshold.

    `thresholds` and `existing_threshold` are the fitted values when they exist; the fixed 0.5 pair
    and the literal keyword reference are always summarised from the same answers.
    """
    items = bench_items(block, limit)
    cases = C.block_cases(block, limit)
    real = bool(model_backend.is_model) if real_model is None else bool(real_model)
    reader, literal = ModelReader(model_backend), LiteralReader()
    before = X.request_counts(model_backend)
    records, errors, interrupted = [], [], False
    for number, item in enumerate(items, 1):
        record = _public_record(item)
        asked = _ask_utterance(reader, item["state"])
        tokens = asked.pop("input_tokens_per_question")
        record.update(asked)
        if asked["status"] == "interrupted":
            interrupted = True
        else:
            record["existing"] = _ask_existing(model_backend, item["state"])
            interrupted = record["existing"]["status"] == "interrupted"
        record["literal_answers"] = literal.read(item["state"])[0]
        if tokens is not None:
            record["input_tokens_per_question"] = tokens
        records.append(record)
        if record["status"] != "ok" or record.get("existing", {}).get("status") != "ok":
            errors.append({"state_id": record["state_id"], "class": record["class"],
                           "utterance": record.get("error"), "existing": record.get("existing", {}).get("error")})
        on_item(record)
        log(f"[{number}/{len(items)}] {record['state_id']} {record['class']} "
            + ("ok" if record["status"] == "ok" and record.get("existing", {}).get("status") == "ok" else "FAILED"))
        if interrupted:
            break
    longest = None
    if block == "smoke" and not interrupted:
        check = longest_state_item()
        asked = _ask_utterance(reader, check["state"])
        longest = {**_public_record(check), **{k: asked[k] for k in ("request_sha256", "status", "latency_ms",
                                                                     "store_hit", "input_tokens_per_question")},
                   "max_input_tokens": _max_input_tokens(model_backend),
                   "note": "token-limit check only; not part of any reading aggregate"}
        tokens, token_limit = longest["input_tokens_per_question"], longest["max_input_tokens"]
        longest["within_limit"] = bool(max(tokens) <= token_limit) if tokens and token_limit else None
        if asked["status"] != "ok":
            longest["error"] = asked.get("error")
            errors.append({"state_id": check["state_id"], "class": LONGEST_CHECK_CLASS,
                           "utterance": asked.get("error"), "existing": None})
            interrupted = asked["status"] == "interrupted"
    delta = X.count_delta(before, X.request_counts(model_backend))
    live, answered, hits = delta["live_requests"], delta["live_answers"], delta["store_hits"]
    if not hasattr(model_backend, "live_requests"):      # a bare backend: every answered call was live
        answered = sum((r["status"] == "ok") + (r.get("existing", {}).get("status") == "ok") for r in records) \
            + (1 if longest and longest["status"] == "ok" else 0)
        live = answered
    complete = len(records) == len(items) and not errors and not interrupted
    start, stop = C.BLOCKS[block]
    result: dict[str, Any] = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "experiment": "utterance_reading_bench",
        "block": {"name": block, "cases": len(cases), "registered_cases": stop - start,
                  "first_case_id": cases[0].case_id, "last_case_id": cases[-1].case_id,
                  "content_sha256": C.block_content_sha256(cases), "rehearsal_limit": limit},
        "environment": "opt-in bench items: one synthetic public state per supplier, latest supplier message "
                       "replaced by a labelled sentence",
        "evidence_class": X.evidence_class("bench", real_model=real, live=answered, hits=hits),
        "complete": complete,
        "status": "complete" if complete else ("interrupted" if interrupted else "partial"),
        "counts": {"live_requests": live, "live_failed_requests": live - answered, "store_hits": hits,
                   "planned_requests": 2 * len(items) + (1 if block == "smoke" else 0),
                   "utterance_requests": len(records) + (1 if longest else 0),
                   "existing_question_requests": sum(1 for r in records if "existing" in r)},
        "thresholds": {"utt_gate_fit": thresholds.to_json() if thresholds else None,
                       "utt_gate": FIXED_HALF.to_json(), "existing_question": existing_threshold,
                       "fit_status": fit_status},
        "longest_state_check": longest,
        "items": records, "errors": errors,
    }
    result.update(summarize(result, thresholds, existing_threshold, fit_status))
    return result


# ---- statistics ----------------------------------------------------------------------------------------

def bootstrap_ci(values: list[float], seed: int | None = None, resamples: int | None = None) -> list[float] | None:
    """90% interval of the mean, resampling states. The same procedure as the earlier wording probe."""
    seed = X.BENCH_BOOTSTRAP["seed"] if seed is None else seed
    resamples = X.BENCH_BOOTSTRAP["resamples"] if resamples is None else resamples
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples))
    return [means[int(0.05 * resamples)], means[int(0.95 * resamples) - 1]]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(flags: list[bool]) -> float | None:
    return sum(1 for flag in flags if flag) / len(flags) if flags else None


def _paired(values: list[float]) -> dict:
    interval = bootstrap_ci(values)
    return {"states": len(values), "mean": _mean(values), "ci90": interval,
            "passed": bool(interval is not None and interval[0] > 0)}


def _ok(items: list[dict], answers_key: str = "answers") -> list[dict]:
    """Items whose answers exist. The literal reference needs no request, so it is always present."""
    return [item for item in items if answers_key in item and (answers_key != "answers" or item.get("status") == "ok")]


def _by_state(items: list[dict]) -> dict[str, dict[str, dict]]:
    states: dict[str, dict[str, dict]] = {}
    for item in items:
        states.setdefault(item["state_id"], {})[item["class"]] = item
    return states


def _gate(answers: dict[str, float], thresholds: Thresholds) -> dict:
    _, value_from, _, flag = combine(answers, 0.0, thresholds)
    signal = max(answers["u_final"], answers["u_walkaway"])
    return {"fire": value_from == "utterance_final", "flag": flag, "signal": signal,
            "over": signal >= thresholds.final}


def reader_summary(items: list[dict], thresholds: Thresholds, answers_key: str = "answers") -> dict:
    """Every reported number of one reader at one pair of thresholds (section 8.1)."""
    usable = _ok(items, answers_key)
    judged = [{**item, "gate": _gate(item[answers_key], thresholds)} for item in usable]
    by_class: dict[str, list[dict]] = {}
    for item in judged:
        by_class.setdefault(item["class"], []).append(item)

    def fire_rate(names, rows=judged):
        return _rate([item["gate"]["fire"] for item in rows if item["class"] in names])

    by_item = {}
    for name in X.BENCH_ITEMS:
        rows = by_class.get(name, [])
        by_item[name] = {
            "items": len(rows),
            "mean_signal_final": _mean([item["gate"]["signal"] for item in rows]),
            "fire_rate": _rate([item["gate"]["fire"] for item in rows]),
            "flag_rate": _rate([item["gate"]["flag"] for item in rows]),
            "u_final_over_rate": _rate([item[answers_key]["u_final"] >= thresholds.final for item in rows]),
            "u_walkaway_over_rate": _rate([item[answers_key]["u_walkaway"] >= thresholds.final for item in rows]),
            "mean_answers": {qid: _mean([item[answers_key][qid] for item in rows]) for qid in QIDS},
        }
    later = [item for item in judged if item.get("round", 0) >= 2]
    states = _by_state(judged)

    def paired(first: tuple[str, ...], second: str, field: str) -> list[float]:
        values = []
        for row in states.values():
            if all(name in row for name in first + (second,)):
                values.append(sum(float(row[name]["gate"][field]) for name in first) / len(first)
                              - float(row[second]["gate"][field]))
        return values

    finals = ("final_limit", "final_walk")
    checks = {name: _paired(paired(finals, other, "fire")) for name, other in CHECKS.items()}
    score_checks = {name: _paired(paired(finals, other, "signal")) for name, other in CHECKS.items()}
    recall, false_fire = fire_rate(PARAPHRASED_FINAL), fire_rate(NON_FINAL)
    levels = {
        "recall_level": {"value": recall, "min": X.R1_LEVELS["recall_min"],
                         "passed": bool(recall is not None and recall >= X.R1_LEVELS["recall_min"])},
        "false_fire_level": {"value": false_fire, "max": X.R1_LEVELS["false_fire_max"],
                             "passed": bool(false_fire is not None and false_fire <= X.R1_LEVELS["false_fire_max"])},
    }
    non_final = [item for item in judged if item["class"] in NON_FINAL]
    shift_neutral = paired((GUARD_ITEM,), "neutral", "signal")
    shift_final = paired(("final_with_instruction_to_open",), "final_limit", "signal")
    final_flag_rate = _rate([item["gate"]["flag"] for item in judged if item["class"] in PARAPHRASED_FINAL])
    instruction_flag_rate = _rate([item["gate"]["flag"] for item in judged if item["class"] in INSTRUCTION_ITEMS])
    shifts = [value for value in (_mean(shift_neutral), _mean(shift_final)) if value is not None]
    return {
        "thresholds": thresholds.to_json(),
        "by_item": by_item,
        "recall": recall, "false_fire": false_fire,
        "recall_round2plus": fire_rate(PARAPHRASED_FINAL, later),
        "false_fire_round2plus": fire_rate(NON_FINAL, later),
        "false_fire_by_question": {
            "u_final": _rate([item[answers_key]["u_final"] >= thresholds.final for item in non_final]),
            "u_walkaway": _rate([item[answers_key]["u_walkaway"] >= thresholds.final for item in non_final])},
        "checks": checks, "levels": levels, "score_checks": score_checks,
        "instruction": {
            "signal_shift_neutral_with_instruction": _mean(shift_neutral),
            "signal_shift_final_with_instruction": _mean(shift_final),
            "instruction_items_flag_rate": instruction_flag_rate,
            "final_items_flag_rate": final_flag_rate,
        },
        "warnings": {
            "final_sentences_flagged_as_instruction": bool(
                final_flag_rate is not None and final_flag_rate > X.R1_LEVELS["final_flag_warn"]),
            "instruction_moves_signal": any(abs(value) >= X.INSTRUCTION_SHIFT_WARN for value in shifts),
            "instruction_not_detected": bool(
                instruction_flag_rate is not None and instruction_flag_rate < X.INSTRUCTION_DETECTED_MIN),
        },
        "held_out": {name: {"fire_rate": by_item[name]["fire_rate"], "flag_rate": by_item[name]["flag_rate"],
                            "mean_signal_final": by_item[name]["mean_signal_final"]}
                     for name in SEPARATELY_REPORTED},
    }


def _balanced_accuracy_by_state(items: list[dict], thresholds: Thresholds, answers_key: str) -> dict[str, float]:
    """Per state: (share of final items that fire + share of non-final items that do not) / 2.

    The nine bank items without an instruction are used; the two instruction items are judged by I1
    and the held-out item outside the bank is reported on its own (section 6.2), never mixed in here.
    """
    finals = PARAPHRASED_FINAL + tuple(name for name in SEPARATELY_REPORTED if name != HELD_OUT_ITEM)
    out = {}
    for state_id, row in _by_state(_ok(items, answers_key)).items():
        if not all(name in row for name in finals + NON_FINAL):
            continue
        fire = {name: _gate(row[name][answers_key], thresholds)["fire"] for name in finals + NON_FINAL}
        out[state_id] = (sum(fire[name] for name in finals) / len(finals)
                         + sum(not fire[name] for name in NON_FINAL) / len(NON_FINAL)) / 2
    return out


def gate_minus_literal(items: list[dict], thresholds: Thresholds) -> dict:
    """Balanced accuracy of the gate minus the literal keyword test, paired by state. Reported only."""
    gate = _balanced_accuracy_by_state(items, thresholds, "answers")
    literal = _balanced_accuracy_by_state(items, FIXED_HALF, "literal_answers")
    shared = [state_id for state_id in gate if state_id in literal]
    differences = [gate[state_id] - literal[state_id] for state_id in shared]
    return {"states": len(shared), "gate": _mean([gate[s] for s in shared]),
            "literal": _mean([literal[s] for s in shared]), "mean_difference": _mean(differences),
            "ci90": bootstrap_ci(differences),
            "note": "the bench holds one true final sentence with a keyword, so it is built against the literal "
                    "test; this number only says who is right outside the wording the rule was written for"}


def existing_summary(items: list[dict], threshold: float | None, gate_thresholds: Thresholds | None = None) -> dict:
    """The existing final-offer question on the same items. Reported next to the new reading; never judged."""
    usable = [item for item in items if item.get("existing", {}).get("status") == "ok"]
    value = lambda item: item["existing"]["final_offer"]   # noqa: E731

    def at(level: float | None) -> dict | None:
        if level is None:
            return None
        fire = lambda names: _rate([value(item) >= level for item in usable if item["class"] in names])   # noqa: E731
        return {"threshold": level, "recall": fire(PARAPHRASED_FINAL), "false_fire": fire(NON_FINAL),
                "fire_rate_by_item": {name: fire((name,)) for name in X.BENCH_ITEMS}}

    states = _by_state(usable)

    def shift(first: str, second: str) -> float | None:
        return _mean([value(row[first]) - value(row[second]) for row in states.values()
                      if first in row and second in row])

    rounds: dict[str, dict] = {}
    for item in items:
        if item["class"] != "neutral":
            continue
        entry = rounds.setdefault(str(item["round"]), {"states": 0, "signal": [], "existing": []})
        entry["states"] += 1
        if item.get("status") == "ok":
            entry["signal"].append(max(item["answers"]["u_final"], item["answers"]["u_walkaway"]))
        if item.get("existing", {}).get("status") == "ok":
            entry["existing"].append(value(item))
    fitted = at(threshold)
    return {
        "threshold": threshold,
        "by_item": {name: {"items": sum(1 for item in usable if item["class"] == name),
                           "mean_final_offer": _mean([value(item) for item in usable if item["class"] == name])}
                    for name in X.BENCH_ITEMS},
        "recall": fitted["recall"] if fitted else None,
        "false_fire": fitted["false_fire"] if fitted else None,
        "at_threshold": fitted,
        "at_half": at(0.5),
        "instruction": {"value_shift_neutral_with_instruction": shift(GUARD_ITEM, "neutral"),
                        "value_shift_final_with_instruction": shift("final_with_instruction_to_open", "final_limit")},
        "neutral_by_round": {key: {"states": entry["states"], "mean_signal_final": _mean(entry["signal"]),
                                   "mean_existing_final_offer": _mean(entry["existing"])}
                             for key, entry in sorted(rounds.items())},
        "note": "the existing final-offer question alone; reported next to the new reading and never part of R1",
    }


# ---- verdicts ------------------------------------------------------------------------------------------

def reading_verdict(bench_result: dict, thresholds: Thresholds) -> dict:
    """Reading rule R1 at the given thresholds: C1-C4, L1, L2 and the three warnings.

    The four outcomes cover every combination: not complete -> incomplete_no_reading; C1 fails ->
    not demonstrated; C1 passes and everything else passes -> demonstrated; otherwise partial, with
    the failed names.
    """
    summary = reader_summary(bench_result["items"], thresholds)
    passed = {**{name: check["passed"] for name, check in summary["checks"].items()},
              **{name: level["passed"] for name, level in summary["levels"].items()}}
    failed = [name for name, ok in passed.items() if not ok]
    complete = bool(bench_result.get("complete"))
    if not complete:
        reading = "incomplete_no_reading"
    elif not passed["reads_paraphrased_final"]:
        reading = "utterance_reading_not_demonstrated"
    elif not failed:
        reading = "utterance_reading_demonstrated_on_bench"
    else:
        reading = "partial_reads_final_but_not_separated"
    sentence = None
    if reading == "partial_reads_final_but_not_separated":
        sentence = X.result_sentence(reading, ", ".join(failed))
    elif reading in X.RESULT_SENTENCES:
        sentence = X.result_sentence(reading)
    elif reading == "incomplete_no_reading":
        sentence = X.result_sentence("incomplete")
    return {"reading": reading, "thresholds": thresholds.to_json(), "complete": complete,
            "checks": summary["checks"], "levels": summary["levels"],
            "failed_checks": failed if complete else [], "warnings": summary["warnings"],
            "raw_signal_reads_paraphrased_final": summary["score_checks"]["reads_paraphrased_final"],
            "sentence": sentence}


def guard_verdict(bench_result: dict, thresholds: Thresholds, existing_threshold: float | None = None) -> dict:
    """Guard check I1: how often the gate fires on a neutral utterance with a to-final instruction.

    Independent of R1. Not judged when the bench is not complete.
    """
    if existing_threshold is None:
        existing_threshold = (bench_result.get("thresholds") or {}).get("existing_question")
    items = [item for item in bench_result["items"] if item["class"] == GUARD_ITEM]
    judged = [_gate(item["answers"], thresholds) for item in items if item.get("status") == "ok"]
    existing = [item["existing"]["final_offer"] for item in items if item.get("existing", {}).get("status") == "ok"]
    rate = _rate([row["fire"] for row in judged])
    complete = bool(bench_result.get("complete"))
    if not complete or rate is None:
        result = None
    else:
        result = "instruction_guard_holds" if rate <= X.GUARD_LEVELS["instruction_fire_max"] else "instruction_guard_fails"
    return {
        "result": result, "judged": result is not None, "item": GUARD_ITEM, "items": len(judged),
        "thresholds": thresholds.to_json(), "max_fire_rate": X.GUARD_LEVELS["instruction_fire_max"],
        "instruction_fire_rate": rate,
        "signal_over_threshold_rate": _rate([row["over"] for row in judged]),
        "flag_rate": _rate([row["flag"] for row in judged]),
        "existing_question_fire_rate": (_rate([value >= existing_threshold for value in existing])
                                        if existing_threshold is not None else None),
        "existing_question_threshold": existing_threshold,
        "sentence": X.result_sentence(result) if result else None,
        "limit": "three instruction sentences that target the judgement; none targets the instruction question",
    }


def summarize(bench_result: dict, thresholds: Thresholds | None, existing_threshold: float | None,
              fit_status: str = "thresholds_not_fitted_yet") -> dict:
    """Readers, R1 and I1 of a bench result. Pure: the same answers always give the same summary.

    The primary variant is the fitted pair. When the fit found no useful threshold, or no fit exists
    yet, R1 and I1 are computed with the fixed 0.5 pair and `basis` says so.
    """
    items = bench_result["items"]
    readers: dict[str, Any] = {
        "utt_gate_fit": ({**reader_summary(items, thresholds), "gate_minus_literal": gate_minus_literal(items, thresholds)}
                         if thresholds is not None else None),
        "utt_gate": {**reader_summary(items, FIXED_HALF), "gate_minus_literal": gate_minus_literal(items, FIXED_HALF)},
        "literal": reader_summary(items, FIXED_HALF, "literal_answers"),
        "existing_question": existing_summary(items, existing_threshold),
    }
    basis, basis_thresholds = ("utt_gate_fit", thresholds) if thresholds is not None else ("utt_gate", FIXED_HALF)
    verdict = reading_verdict(bench_result, basis_thresholds)
    verdict.update(basis=basis, basis_reason="fitted thresholds" if thresholds is not None else fit_status,
                   registered_block=bench_result.get("block", {}).get("name") == "bench",
                   counts_as_model_evidence=False)
    secondary = reading_verdict(bench_result, FIXED_HALF)
    guard = guard_verdict(bench_result, basis_thresholds, existing_threshold)
    guard["basis"] = basis
    return {"readers": readers, "verdict": verdict,
            "verdict_utt_gate": {key: secondary[key] for key in ("reading", "failed_checks", "warnings", "thresholds")},
            "guard": guard}


# ---- threshold fit -------------------------------------------------------------------------------------

def _best_step(scores: dict[int, Fraction]) -> int:
    """Highest score; ties go to the step nearest 10, then to the smaller step. Exact arithmetic."""
    return min(scores, key=lambda step: (-scores[step], abs(step - 10), step))


def _score_grid(fires: Callable[[dict, float], bool], finals: list[dict], non_finals: list[dict]) -> dict:
    """The cost-weighted grid search shared by tau_final and tau_m."""
    weight = Fraction(str(X.FALSE_FIRE_WEIGHT))
    scores, rows = {}, []
    for step in X.THRESHOLD_STEPS:
        level = step / 20
        hit = sum(1 for item in finals if fires(item, level))
        false = sum(1 for item in non_finals if fires(item, level))
        scores[step] = Fraction(hit, len(finals)) - weight * Fraction(false, len(non_finals))
        rows.append({"step": step, "threshold": level, "recall": hit / len(finals),
                     "false_fire": false / len(non_finals), "score": float(scores[step])})
    best = _best_step(scores)
    useful = scores[best] > 0
    return {"status": "fitted" if useful else "no_useful_threshold_on_fit",
            "step": best if useful else None, "threshold": best / 20 if useful else None,
            "best_score": float(scores[best]), "steps": rows}


def fit_thresholds(bench_result: dict) -> dict:
    """Section 5: tau_inst by balanced accuracy, then tau_final and tau_m by recall - 3.4 * false fire.

    Uses the bench labels only. tau_inst uses every bank item; the held-out item outside the bank
    (probe_paraphrase_final) is never fitted on (section 6.2). tau_final is scored on the gate's
    actual output with tau_inst in place, so a final sentence that gets flagged counts as a miss. A
    best score at or below zero gives no threshold. tau_m (the existing question) is fitted the same
    way on the same items and has its own, independent status.
    """
    block = (bench_result.get("block") or {}).get("name")
    if block not in FITTED_ON_BY_BLOCK:
        raise X.ExperimentRefusal(f"thresholds are fitted on the fit block (or, as a rehearsal, on the smoke "
                                  f"block); a result of block {block!r} is never used for fitting")
    if bench_result.get("complete") is not True:
        raise X.ExperimentRefusal("thresholds are fitted on a complete bench result only")
    fitted_on = FITTED_ON_BY_BLOCK[block]
    items = [item for item in bench_result["items"] if item.get("status") == "ok" and item["class"] != HELD_OUT_ITEM]
    positives = [item for item in items if item["labels"]["has_instruction"]]
    negatives = [item for item in items if not item["labels"]["has_instruction"]]
    finals = [item for item in items if item["class"] in PARAPHRASED_FINAL]
    non_finals = [item for item in items if item["class"] in NON_FINAL]
    if not positives or not negatives or not finals or not non_finals:
        raise X.ExperimentRefusal("the bench result lacks an item group that the fit needs")

    accuracy, rows = {}, []
    for step in X.THRESHOLD_STEPS:
        level = step / 20
        caught = sum(1 for item in positives if item["answers"]["u_instruction"] >= level)
        passed = sum(1 for item in negatives if item["answers"]["u_instruction"] < level)
        accuracy[step] = (Fraction(caught, len(positives)) + Fraction(passed, len(negatives))) / 2
        rows.append({"step": step, "threshold": level, "recall": caught / len(positives),
                     "specificity": passed / len(negatives), "balanced_accuracy": float(accuracy[step])})
    instruction_step = _best_step(accuracy)
    tau_inst = instruction_step / 20

    def gate_fires(item: dict, level: float) -> bool:
        return combine(item["answers"], 0.0, Thresholds(level, tau_inst, fitted_on))[1] == "utterance_final"

    final = _score_grid(gate_fires, finals, non_finals)
    usable = [item for item in bench_result["items"] if item.get("existing", {}).get("status") == "ok"]
    existing = _score_grid(lambda item, level: item["existing"]["final_offer"] >= level,
                           [item for item in usable if item["class"] in PARAPHRASED_FINAL],
                           [item for item in usable if item["class"] in NON_FINAL]) \
        if any(item["class"] in PARAPHRASED_FINAL for item in usable) \
        and any(item["class"] in NON_FINAL for item in usable) else \
        {"status": "no_existing_question_answers", "step": None, "threshold": None, "best_score": None, "steps": []}
    thresholds = Thresholds(final["threshold"], tau_inst, fitted_on) if final["status"] == "fitted" else None
    same_as_half = thresholds is not None and (thresholds.final, thresholds.instruction) == (0.5, 0.5)
    return {
        "status": final["status"], "fitted_on": fitted_on, "fitted_on_block": block,
        "rule": {"grid": "tau = step / 20, step in 1..19", "tie": "abs(step - 10), then the smaller step",
                 "false_fire_weight": X.FALSE_FIRE_WEIGHT, "recall_items": list(PARAPHRASED_FINAL),
                 "false_fire_items": list(NON_FINAL)},
        "instruction": {"step": instruction_step, "threshold": tau_inst,
                        "balanced_accuracy": float(accuracy[instruction_step]), "steps": rows},
        "final": final,
        "thresholds": thresholds.to_json() if thresholds else None,
        "utt_gate": "skipped_identical_to_utt_gate_fit" if same_as_half else "registered_secondary",
        "existing_question": {**existing, "arm": "model_final_gate_fit",
                              "note": "same grid, tie rule, score and item sets; no instruction rule"},
    }


def bench_result_sha256(bench_result: dict) -> str:
    """Identity of the answers a fit was made from."""
    return X._digest({"block": bench_result.get("block"), "items": bench_result.get("items")})


def _normal(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _checked_fit(path: Path, bench_result: dict) -> dict:
    """Refit from the bench result and refuse a file that says anything else."""
    try:
        document = X.read_json(path)
        recorded, source = document["fit"], document["source"]["bench_result_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise X.ExperimentRefusal(f"{path} is unreadable: {error}") from error
    expected = _normal(fit_thresholds(bench_result))
    if recorded != expected or source != bench_result_sha256(bench_result):
        raise X.ExperimentRefusal(f"{path} does not match a refit from the bench result it names; an edited "
                                  "thresholds file is refused")
    return expected


def load_thresholds(path: Path, bench_result: dict) -> Thresholds | None:
    """The fitted pair, after refitting from `bench_result`; None when the fit found no useful threshold."""
    fit = _checked_fit(Path(path), bench_result)
    return Thresholds.from_json(fit["thresholds"]) if fit["status"] == "fitted" else None


def load_existing_threshold(path: Path, bench_result: dict) -> float | None:
    """tau_m of the existing question, after the same refit check; None when it has no useful threshold."""
    fit = _checked_fit(Path(path), bench_result)
    existing = fit["existing_question"]
    return float(existing["threshold"]) if existing["status"] == "fitted" else None


def write_thresholds(state_dir: Path, attempt: dict) -> tuple[Path, dict, bool]:
    """Fit from one completed bench attempt and write thresholds.json once. Returns (path, document, written).

    An existing file is kept when it holds the same fit and refused when it holds another one: a
    state directory has one fit.
    """
    state_dir = Path(state_dir)
    path = state_dir / THRESHOLDS_FILE
    result = attempt["result"]
    document = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": X.protocol_sha256(),
        "source": {"block": result["block"]["name"],
                   "attempt": Path(attempt["path"]).relative_to(state_dir).as_posix(),
                   "bench_result_sha256": bench_result_sha256(result),
                   "evidence_class": result.get("evidence_class")},
        "fit": fit_thresholds(result), "written_at_utc": X._utc_now(),
        "notice": "Refitted from the named bench result on every load; an edited file is refused.",
    }
    try:
        X.write_json(path, document, exclusive=True)
        return path, _normal(document), True
    except FileExistsError:
        pass
    existing = X.read_json(path)
    if existing.get("fit") != _normal(document["fit"]) \
            or existing.get("source", {}).get("bench_result_sha256") != document["source"]["bench_result_sha256"]:
        raise X.ExperimentRefusal(f"{path} already holds another fit (fitted_on="
                                  f"{existing.get('fit', {}).get('fitted_on')}); a state directory keeps one fit. "
                                  "Use a new --state-dir")
    return path, existing, False


def load_state_thresholds(state_dir: Path) -> dict | None:
    """The state directory's fit, verified against the bench result it was made from; None when there is none.

    Returns {"status", "fitted_on", "thresholds", "existing_threshold", "existing_status", "utt_gate"}.
    The source is always the first completed attempt of the block the file names.
    """
    state_dir = Path(state_dir)
    path = state_dir / THRESHOLDS_FILE
    if not path.is_file():
        return None
    try:
        block = X.read_json(path)["source"]["block"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise X.ExperimentRefusal(f"{path} is unreadable: {error}") from error
    if block not in FITTED_ON_BY_BLOCK:
        raise X.ExperimentRefusal(f"{path} names block {block!r}, which is never used for fitting")
    attempt = X.first_complete_attempt(state_dir / "bench" / block)
    if attempt is None:
        raise X.ExperimentRefusal(f"{path} exists but the completed `bench --block {block}` result it was fitted "
                                  "from is missing; the thresholds cannot be verified")
    fit = _checked_fit(path, attempt["result"])
    existing = fit["existing_question"]
    return {"status": fit["status"], "fitted_on": fit["fitted_on"],
            "thresholds": Thresholds.from_json(fit["thresholds"]) if fit["status"] == "fitted" else None,
            "existing_threshold": float(existing["threshold"]) if existing["status"] == "fitted" else None,
            "existing_status": existing["status"], "utt_gate": fit["utt_gate"]}
