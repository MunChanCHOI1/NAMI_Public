"""Evaluator-side metrics for the candidate sensitivity probe. Standard library only.

A record is one (state, ladder, candidate) with the predicted probability `p`, the
simulator's ground truth `truth`, our saving if accepted `saving`, and `rank`, the
candidate's position on its ladder from least to most favourable for the supplier.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

# Proposed go/no-go thresholds, fixed before any real-model run. Coarse on purpose:
# the probe has a few hundred correlated points from two dozen synthetic states.
THRESHOLDS = {
    "auc_across_states_min": 0.70,
    "within_state_pair_accuracy_min": 0.70,
    "cross_ladder_pair_accuracy_min": 0.60,
    "median_spread_min": 0.10,
    "overall_std_min": 0.10,
    "brier_skill_min": 0.05,
    "ece_max": 0.15,
    "value_gain_ci_low_min": 0.0,
    "invariance_mean_abs_diff_warn": 0.10,
}


def _groups(records, key):
    out = defaultdict(list)
    for r in records:
        out[key(r)].append(r)
    return out


def auc(records, key: str = "p") -> float | None:
    pos = [r[key] for r in records if r["truth"]]
    neg = [r[key] for r in records if not r["truth"]]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def pair_accuracy(records, cross_ladder_only: bool = False, key: str = "p") -> tuple[float | None, int]:
    """Within one state: is a candidate the supplier accepts scored above one it rejects?

    Pairs inside a single ladder are answered correctly by any monotone output, so
    `cross_ladder_only` keeps only pairs that require trading one condition against another.
    """
    score, pairs = 0.0, 0
    for group in _groups(records, lambda r: r["state_id"]).values():
        for a in group:
            for b in group:
                if a["truth"] and not b["truth"] and not (cross_ladder_only and a["ladder"] == b["ladder"]):
                    pairs += 1
                    score += 1.0 if a[key] > b[key] else 0.5 if a[key] == b[key] else 0.0
    return (score / pairs if pairs else None), pairs


def discrimination(records, key: str = "p") -> dict:
    within, within_pairs = pair_accuracy(records, key=key)
    cross, cross_pairs = pair_accuracy(records, cross_ladder_only=True, key=key)
    return {"auc_across_states": auc(records, key), "within_state_pair_accuracy": within,
            "within_state_pairs": within_pairs, "cross_ladder_pair_accuracy": cross,
            "cross_ladder_pairs": cross_pairs}


def spread(records) -> dict:
    """How much the output moves. Sets where the supplier treats every candidate alike are
    excluded from the within-set figure: a correct predictor should be flat there."""
    groups = _groups(records, lambda r: (r["state_id"], r["ladder"])).values()
    mixed = [g for g in groups if len({r["truth"] for r in g}) > 1]
    spreads = [max(r["p"] for r in g) - min(r["p"] for r in g) for g in mixed]
    values = [r["p"] for r in records]
    return {"sets_with_both_outcomes": len(mixed),
            "median_within_set": statistics.median(spreads) if spreads else 0.0,
            "overall_std": statistics.pstdev(values) if len(values) > 1 else 0.0}


def monotonicity(records, strict_margin: float = 0.01) -> dict:
    """Adjacent ladder steps. `responsive` requires a real increase, so a flat output scores 0."""
    out = {}
    for ladder, items in _groups(records, lambda r: r["ladder"]).items():
        non_decreasing = responsive = steps = 0
        for group in _groups(items, lambda r: r["state_id"]).values():
            ordered = sorted(group, key=lambda r: r["rank"])
            for a, b in zip(ordered, ordered[1:]):
                steps += 1
                non_decreasing += b["p"] >= a["p"] - 1e-9
                responsive += b["p"] > a["p"] + strict_margin
        out[ladder] = {"steps": steps,
                       "non_decreasing_rate": non_decreasing / steps if steps else None,
                       "responsive_rate": responsive / steps if steps else None}
    return out


def brier(probabilities, truths) -> float:
    return sum((p - (1.0 if t else 0.0)) ** 2 for p, t in zip(probabilities, truths)) / len(truths)


def reliability(records, bins: int = 5) -> dict:
    table, ece = [], 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [r for r in records if lo <= r["p"] < hi or (i == bins - 1 and r["p"] == 1.0)]
        if not inside:
            table.append({"bin": [lo, hi], "n": 0})
            continue
        mean_p = sum(r["p"] for r in inside) / len(inside)
        rate = sum(1 for r in inside if r["truth"]) / len(inside)
        ece += len(inside) / len(records) * abs(mean_p - rate)
        table.append({"bin": [lo, hi], "n": len(inside), "mean_p": round(mean_p, 4),
                      "accept_rate": round(rate, 4)})
    return {"ece": ece, "bins": table}


def _logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def _fit_platt(xs, ys, iterations: int = 400, lr: float = 0.3, l2: float = 1e-3):
    """Logistic regression on one feature by gradient descent. Robust to a constant feature."""
    a, b = 1.0, 0.0
    n = len(xs)
    if not n or n != len(ys):
        raise ValueError("Platt calibration needs non-empty aligned training data")
    for _ in range(iterations):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            err = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, a * x + b)))) - y
            ga += err * x
            gb += err
        a -= lr * (ga / n + l2 * a)
        b -= lr * (gb / n)
    return a, b


def group_id(record) -> str:
    """A supplier seed and its statement twin are one independent sampling unit."""
    return record.get("group_id", record["state_id"])


def platt_cross_validated(records) -> list[float] | None:
    """Leave-one-supplier-group-out probabilities; unavailable with fewer than two groups."""
    groups = sorted({group_id(r) for r in records})
    if len(groups) < 2:
        return None
    xs = [_logit(r["p"]) for r in records]
    ys = [1.0 if r["truth"] else 0.0 for r in records]
    out = [0.0] * len(records)
    for group in groups:
        train = [i for i, r in enumerate(records) if group_id(r) != group]
        test = [i for i, r in enumerate(records) if group_id(r) == group]
        a, b = _fit_platt([xs[i] for i in train], [ys[i] for i in train])
        for i in test:
            out[i] = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, a * xs[i] + b))))
    return out


def calibration(records) -> dict:
    truths = [r["truth"] for r in records]
    base_rate = sum(truths) / len(truths)
    reference = brier([base_rate] * len(truths), truths)
    raw = brier([r["p"] for r in records], truths)
    cv_predictions = platt_cross_validated(records)
    recalibrated = brier(cv_predictions, truths) if cv_predictions is not None else None
    skill = (lambda b: 1 - b / reference) if reference > 0 else (lambda b: None)
    return {"n": len(records), "base_rate": base_rate, "brier_base_rate": reference,
            "brier_raw": raw, "brier_skill_raw": skill(raw),
            "brier_platt_leave_one_group_out": recalibrated,
            "brier_skill_platt": skill(recalibrated) if recalibrated is not None else None,
            "platt_cv": {"unit": "supplier_group", "groups": len({group_id(r) for r in records}),
                         "available": cv_predictions is not None,
                         "reason": None if cv_predictions is not None else "fewer_than_two_groups"},
            **reliability(records)}


def one_step_choice_value(records, probability_key: str) -> dict[str, float]:
    """Per state, over all its candidates: pick argmax p * saving, realise saving only if truly accepted.

    One-step value: a rejected counter-offer is worth 0 (the alternative quote).
    Only candidates we are allowed to send and that beat the alternative are eligible.
    """
    out = {}
    for key, group in _groups(records, lambda r: r["state_id"]).items():
        eligible = [r for r in group if r["authority"] != "forbidden" and r["saving"] > 0]
        if not eligible:
            out[key] = 0.0
            continue
        if probability_key == "oracle":
            out[key] = float(max((r["saving"] for r in eligible if r["truth"]), default=0.0))
            continue
        chosen = max(eligible, key=lambda r: (r[probability_key] * r["saving"], r["saving"]))
        out[key] = float(chosen["saving"]) if chosen["truth"] else 0.0
    return out


def decision_value(records, seed: int = 20261116, resamples: int = 2000) -> dict:
    """State-mean realised savings; resample whole supplier groups, keeping twins together."""
    model = one_step_choice_value(records, "p")
    rule = one_step_choice_value(records, "p_rule")
    greedy = one_step_choice_value(records, "p_flat")
    oracle = one_step_choice_value(records, "oracle")
    keys = sorted(model)
    state_groups = {r["state_id"]: group_id(r) for r in records}
    clusters = _groups(keys, lambda state: state_groups[state])
    group_keys = sorted(clusters)
    mean = lambda d: sum(d[k] for k in keys) / len(keys)  # noqa: E731
    rng = random.Random(seed)
    means = []
    if len(group_keys) >= 2:
        for _ in range(resamples):
            sampled_states = [state for _ in group_keys for state in clusters[rng.choice(group_keys)]]
            sample = [model[k] - rule[k] for k in sampled_states]
            means.append(sum(sample) / len(sample))
    means.sort()
    return {"states": len(keys), "groups": len(group_keys), "bootstrap_unit": "supplier_group",
            "bootstrap_resamples": resamples if means else 0,
            "mean_saving_model": mean(model), "mean_saving_rule_heuristic": mean(rule),
            "mean_saving_always_max_saving": mean(greedy), "mean_saving_oracle": mean(oracle),
            "gain_vs_rule_mean": mean(model) - mean(rule),
            "gain_vs_rule_ci90_by_group": ([means[int(0.05 * resamples)], means[int(0.95 * resamples) - 1]]
                                            if means else None)}


def invariance(base_records, variant_records) -> dict:
    """Mean absolute change of p under wording changes; the inverted question should give 1 - p."""
    base = {(r["state_id"], r["ladder"], r["candidate_id"]): r["p"] for r in base_records}
    out = {}
    for variant, items in _groups(variant_records, lambda r: r["variant"]).items():
        diffs = []
        for r in items:
            reference = base.get((r["state_id"], r["ladder"], r["candidate_id"]))
            if reference is None:
                continue
            diffs.append(abs(reference + r["p"] - 1.0) if variant == "inverted" else abs(reference - r["p"]))
        if diffs:
            out[variant] = {"n": len(diffs), "mean_abs_diff": sum(diffs) / len(diffs), "max_abs_diff": max(diffs)}
    return out


def message_sensitivity(records, twins: dict[str, str], fast_id: str, control_id: str) -> dict:
    """Twin states share every number and differ only in the supplier's stated payment preference.

    Reports whether the statement raises the prediction for the pay-immediately candidate
    more than for the unchanged-payment candidate.
    """
    p = {(r["state_id"], r["candidate_id"]): r["p"] for r in records if r["ladder"] == "payment"}
    effects = []
    for revealed, neutral in sorted(twins.items()):
        try:
            fast = p[(revealed, fast_id)] - p[(neutral, fast_id)]
            control = p[(revealed, control_id)] - p[(neutral, control_id)]
        except KeyError:
            continue
        effects.append(fast - control)
    if not effects:
        return {"pairs": 0}
    return {"pairs": len(effects), "mean_effect": sum(effects) / len(effects),
            "positive_rate": sum(1 for e in effects if e > 0.005) / len(effects)}


def verdict(metrics: dict, is_model: bool, complete: bool) -> dict:
    """Map the metrics to the pre-registered routes. Never upgrades a mock or partial run."""
    t = THRESHOLDS
    disc, rule = metrics["discrimination"], metrics["reference_rule_discrimination"]
    cal, val = metrics["calibration"], metrics["decision_value"]
    at_least = lambda value, minimum: value is not None and value >= minimum  # noqa: E731
    responsive = (metrics["spread"]["median_within_set"] >= t["median_spread_min"]
                  and metrics["spread"]["overall_std"] >= t["overall_std_min"])
    ranking_ok = (at_least(disc["within_state_pair_accuracy"], t["within_state_pair_accuracy_min"])
                  and at_least(disc["cross_ladder_pair_accuracy"], t["cross_ladder_pair_accuracy_min"]))
    beats_rule = (disc["within_state_pair_accuracy"] or 0) > (rule["within_state_pair_accuracy"] or 0)
    threshold_ok = at_least(disc["auc_across_states"], t["auc_across_states_min"])
    raw_ok = (cal["brier_skill_raw"] or 0) >= t["brier_skill_min"] and cal["ece"] <= t["ece_max"]
    platt_ok = (cal["brier_skill_platt"] or 0) >= t["brier_skill_min"]
    interval = val["gain_vs_rule_ci90_by_group"]
    value_ok = interval is not None and interval[0] > t["value_gain_ci_low_min"]
    ranking_useful = responsive and ranking_ok and beats_rule
    if ranking_useful and threshold_ok and (raw_ok or platt_ok) and value_ok:
        route = "A_probability_expected_value"
    elif ranking_useful:
        route = "B_ranking_plus_code_comparison"
    else:
        route = "C_or_D_candidate_scoring_weak_run_action_choice_probe"
    return {
        "route": route,
        "checks": {"responsive": responsive, "ranking_ok": ranking_ok, "ranking_beats_rule": beats_rule,
                   "threshold_across_states_ok": threshold_ok, "calibration_raw_ok": raw_ok,
                   "calibration_after_platt_ok": platt_ok, "value_vs_rule_ok": value_ok},
        "thresholds": t,
        "counts_as_model_evidence": bool(is_model and complete),
        "note": ("real-model probe on synthetic suppliers; not a final evaluation" if is_model and complete
                 else "partial run; do not use for the decision" if is_model
                 else "CPU reference backend; pipeline check only, not model evidence"),
    }
