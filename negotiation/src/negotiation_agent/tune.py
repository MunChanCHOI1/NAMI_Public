"""Resumable tuning runner: how the estimates of the pinned decision model are USED.

NOT EXECUTED ON A GPU as of 2026-11-18. This module was authored and verified on CPU only,
with the CPU reference backends and test doubles. No real-model response has gone through
it yet; nothing it has produced so far is model evidence. Minutes and request counts that
`plan` prints are estimates, never measurements.

What is tuned is the usage of the model's two estimates (acceptance per counter-offer and
"is the standing offer final"), not the model, the request wording or the policy constants.
Every candidate reads the same unchanged public request, so one stored answer serves all of
them (`prediction_store`), and every hybrid is labelled as one (`estimate_sources`).

The protocol is fixed in this file and written to `<state>/protocol.json` before any backend
is built:
- development cases only for choosing and confirming; calibration cases only to fit maps;
  the held-out split is never named, accepted or touched here, and no repository evidence
  folder is created;
- candidates, block boundaries, selection rule, confirmation rule, bootstrap parameters and
  the allowed result sentences cannot change inside one state directory;
- a failed, invalid or interrupted estimate is never stored, never filled in and never
  retried automatically; an attempt folder is never overwritten;
- real-model, replayed and CPU results carry different evidence classes.

Run through `negotiation/run_tune.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import metrics as M, policy
from .backends import (ConstantBackend, HeuristicBackend, KeywordBackend, SystemOneBackend, ZoroHttpBackend,
                       ZoroNativeBackend)
from .estimate_sources import TWIN_ACCEPTANCE_CONSTANT, ComposedBackend, EstimateSpec, shrink
from .evaluation import (BASELINES, REFERENCE_POLICY, TrueAcceptanceReference, fit_calibration, make_cases,
                         paired_difference, run_case, summarize)
from .policy import PolicyConfig
from .prediction_store import PredictionStore, StoredPredictionBackend, identity_digest
from .probe import add_backend_arguments, make_backend
from .provenance import (REAL_MODEL_BACKENDS, REPOSITORY_ROOT, backend_execution_config, file_sha256, input_hashes,
                         json_safe, provenance, reasoning_requested, request_sha256)
from .requests import FINAL_OFFER_QID, build_decision_request
from .session import TERMINAL
from .simulator import default_buyer_private, initial_state

# ---- fixed protocol ------------------------------------------------------------------------------------

PROTOCOL_VERSION = "tune-2026-11-18.v1"
GPU_NOTICE = ("NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only, with CPU reference "
              "backends and test doubles.")
# name -> (split, start, stop): the cases are evaluation.make_cases(split, stop)[start:stop]. A split draws
# ONE sequential random stream, so a block is always a slice of the prefix, never a reseeded range.
BLOCKS: dict[str, tuple[str, int, int]] = {
    "d0": ("dev", 0, 30),            # dev-1100..1129, already seen by the real model: smoke and regression only
    "select": ("dev", 30, 150),      # dev-1130..1249: choosing
    "confirm": ("dev", 150, 450),    # dev-1250..1549: one use
    "cal": ("calibration", 0, 60),   # calibration-2000..2059: map fitting only
}
ALLOWED_SPLITS = ("dev", "calibration")
BLOCK_USE = {
    "d0": "smoke and regression only; already seen by the real model; never enters selection",
    "select": "choosing among the candidates",
    "confirm": "one confirmatory use of the selected configuration",
    "cal": "fitting probability maps only; its savings are never printed, compared or ranked",
}
RESERVE = {"split": "dev", "start": 450, "stop": 900, "case_ids": "dev-1550..dev-1999",
           "rule": "reserved for a later protocol version; never touched by this one"}
BOOTSTRAP = {"resamples": 2000, "seed": 20261116, "level": 0.90}   # evaluation.paired_difference defaults
SHRINK_GRID = tuple(step / 10 for step in range(11))
PLAN_SECONDS_PER_REQUEST = 1.75          # estimate from the 2026-11-16 T4 run; not a measurement of any later run
PLAN_REQUESTS_PER_30_DEV_CASES = 77      # estimate from the same run
GOLDEN_REQUEST = {"case_id": "dev-1100", "round": 1,
                  "sha256": "17596ee5f90744034016e2a31e641147299d88ca2b681490abd23c427fa16fc7"}
REGRESSION_REFERENCE = "negotiation/artifacts/20261116_colab/dev30_summary.json"
KNOWN_EXPOSURE = ("as of protocol registration on 2026-11-18: the real model has seen dev-1100..1129 only; "
                  "model-free stand-in aggregates were computed on dev-1130..1999 during design, so no "
                  "development block is unseen; SELECT and CONFIRM had not been run with the real model. What "
                  "this state directory has run since is listed in its attempts")
STAGES = ("smoke", "regress", "calibrate", "select", "confirm")
ALL_STEPS = ("smoke", "regress", "calibrate", "fit-map", "select", "board")
STAGE_PURPOSE = {"smoke": "smoke_store_check_only", "regress": "regression_on_seen_cases_only",
                 "calibrate": "map_fitting_only", "select": "selection", "confirm": "confirmation"}
SELECTION_RULE = ("among candidates that are complete on SELECT with zero authority violations and zero "
                  "needs_review: the highest mean realized saving; ties go to the earlier candidate, so M0 wins "
                  "every tie. selection.json is written only when every candidate is terminal (complete, failed "
                  "with a reason, or skipped); CPU reference rows are never selectable. Pre-declared for a "
                  "candidate whose estimate failed on SELECT: it is retried once with --retry-failed before the "
                  "failure is accepted with an explicit `board`. All candidates must have run under the same "
                  "hashed source files; a mix blocks the selection")
CONFIRM_RULE = {
    "primary": ("improvement_confirmed iff S is not M0, the 90% interval of (S minus M0) realized saving on "
                "CONFIRM has a lower bound above 0, the mean net-saving difference is not negative and S has no "
                "authority violation; not_confirmed otherwise (S equal to M0 means: default kept); incomplete "
                "when a needed run is not complete"),
    "secondary_model_contribution": ("pre-declared secondary: model_contribution_shown iff the 90% interval of "
                                     "(S minus its model-free twin) realized saving has a lower bound above 0"),
    "secondary_rule_policy": ("S minus rule_policy is reported and labelled "
                              "not_confirmatory_reference_seen_in_design"),
    "single_use": ("CONFIRM is used once per state directory: confirm_first_use.json is written before the first "
                   "confirm call; a second confirm is refused unless it is explicitly run as not_confirmatory. "
                   "The single use is enforced inside one state directory only: an earlier state ZIP that is "
                   "restored, or a new state directory, is not detected. Every confirm use on these cases, "
                   "including earlier ones elsewhere, must be reported together with the result"),
    "continuation": ("an unfinished first use (budget stop, interruption, failed estimate) may be continued as "
                     "the first use only while protocol, block, selection, store identity and every hashed source "
                     "file are unchanged, and every answer recorded by its earlier attempts is still in the store "
                     "with the same values; otherwise only a not_confirmatory rerun is possible. Per-case savings "
                     "are not printed while CONFIRM runs"),
    "failed_estimate_on_confirm": ("pre-declared: a configuration whose estimate failed on CONFIRM is retried "
                                   "once with --retry-failed; if that fails too, the verdict stays incomplete "
                                   "and is reported as incomplete. The choice is not made after looking at rows"),
    "before_first_use": ("for the real model, one live request that is not stored checks that the runtime is the "
                         "one whose answers are stored; a first use is refused in read-only store mode"),
}
VERDICTS = ("improvement_confirmed", "not_confirmed", "incomplete")
# The only sentences a result may be reported with (user-facing, Korean).
RESULT_SENTENCES = {
    "improvement_confirmed": ("합성 개발 사례에서(선택 구간에서 고르고, 확인 구간은 이 상태 폴더 기준 첫 사용으로 확인) "
                              "선택한 사용 설정의 실현 절감액이 기본 설정보다 높았다(대응 부트스트랩 90% 구간의 하한이 "
                              "0보다 큼). 보류 평가 결과가 아니며 실제 협상에 대한 근거가 아니다. 같은 확인 구간을 다른 "
                              "상태 폴더에서 쓴 적이 있으면 그 사용도 함께 보고한다."),
    "not_confirmed": "확인 구간에서 기본 설정보다 나아졌다는 것이 확인되지 않았다. 기본 설정(M0)을 유지한다.",
    "incomplete": "확인 실행이 완료되지 않아 결론을 내리지 않는다.",
    "model_contribution_shown": ("확인 구간에서 선택한 설정이 모델 값을 쓰지 않는 쌍둥이 설정보다 실현 절감액이 "
                                 "높았다(90% 구간의 하한이 0보다 큼)."),
    "model_contribution_not_shown": "확인 구간에서 모델 값을 쓰는 것이 쓰지 않는 쌍둥이 설정보다 낫다는 것이 드러나지 않았다.",
}
STOPPING = ("partial_budget", "interrupted", "initialization_failed", "store_identity_mismatch",
            "store_write_failed", "runner_error")
STOP_HINTS = {
    "partial_budget": "the budget was reached; run the same command again to continue from the store",
    "interrupted": "interrupted; nothing was stored for the interrupted request; run again to continue",
    "store_write_failed": "a write to the prediction store failed (disk full or unavailable); no further model "
                          "call was made. Free space and run the same command again, no --retry-failed is needed",
    "initialization_failed": "the backend could not be built (see the error in the attempt's decisions.jsonl); "
                             "fix the environment and run again, no --retry-failed is needed",
    "store_identity_mismatch": "the live runtime differs from the one the stored answers came from; answers of "
                               "two runtimes are not mixed: use a new --state-dir for this runtime",
    "runner_error": "the runner itself failed (see `error` in the attempt's result.json)",
    "smoke_failed": "the smoke check failed; nothing else was run",
    "failed": "an estimate failed (the error is in the lines above and in `first_errors` of the attempt's "
              "result.json); nothing was stored or filled in. Run the same command again to continue with the "
              "other configurations and leave this one failed, or add --retry-failed to ask again",
}
DERIVED_VIEWS = ("leaderboard.json", "leaderboard.md")     # regenerated by `board`; never the record itself
MANIFEST_NAME = "tuning_state_manifest.json"
LOCK_NAME = "run.lock"                 # held by one `run` process; never packed
BACKEND_CLASSES = {"constant": ConstantBackend, "heuristic": HeuristicBackend, "keyword": KeywordBackend,
                   "zoro": ZoroNativeBackend, "zoro-http": ZoroHttpBackend, "systemone": SystemOneBackend}
ADAPTER_FILES = {"zoro": "src/sessionguard/backends/zoro.py",
                 "zoro-http": "src/sessionguard/backends/zoro_http.py"}
NATIVE_BACKEND = "zoro"            # the only real-model backend this protocol version accepts
NEGOTIATION_BACKENDS_FILE = "negotiation/src/negotiation_agent/backends.py"
REQUESTS_FILE = "negotiation/src/negotiation_agent/requests.py"


class TuneRefusal(RuntimeError):
    """A command was refused. Nothing that the refusal protects was changed."""


@dataclass(frozen=True)
class Arm:
    """One pre-registered candidate. `needs` names the calibration-split fits its template waits for."""

    name: str
    description: str
    template: EstimateSpec
    needs: tuple[str, ...] = ()
    zero_weight_status: str | None = None


_CONSTANT = {"acceptance_source": "constant", "acceptance_constant": TWIN_ACCEPTANCE_CONSTANT}
# Order matters: ties go to the earlier candidate.
CANDIDATES: tuple[Arm, ...] = (
    Arm("M0", "model acceptance + model final_offer (today's agent; wins all ties)", EstimateSpec()),
    Arm("M1", "model acceptance + policy_prior final_offer", EstimateSpec(final_source="policy_prior")),
    Arm("M2", "model acceptance + keyword final_offer", EstimateSpec(final_source="keyword")),
    Arm("M3", "model acceptance + shrunk model final_offer (weight fitted on CAL)",
        EstimateSpec(final_source="shrink"), ("final_offer_shrink",), "skipped_identical_to_M1"),
    Arm("F1", "constant 0.15 acceptance + shrunk model final_offer (weight fitted on CAL)",
        EstimateSpec(final_source="shrink", **_CONSTANT), ("final_offer_shrink",), "skipped_identical_to_twin"),
    Arm("F2", "constant 0.15 acceptance + keyword, else shrunk model final_offer (weight fitted on CAL)",
        EstimateSpec(final_source="keyword_else_shrink", **_CONSTANT), ("final_offer_shrink",),
        "skipped_identical_to_twin"),
    Arm("M4", "Platt-mapped model acceptance (map fitted on CAL) + model final_offer", EstimateSpec(), ("platt",)),
    Arm("M5", "Platt-mapped model acceptance (map fitted on CAL) + keyword final_offer",
        EstimateSpec(final_source="keyword"), ("platt",)),
)
_NAMED_REFERENCE_SPECS = (
    ("rule_policy", "rule acceptance (HeuristicBackend) + policy_prior final_offer",
     EstimateSpec(acceptance_source="rule", final_source="policy_prior")),
    ("constant_policy_prior", "constant 0.15 acceptance + policy_prior final_offer",
     EstimateSpec(final_source="policy_prior", **_CONSTANT)),
    ("constant_keyword", "constant 0.15 acceptance + keyword final_offer",
     EstimateSpec(final_source="keyword", **_CONSTANT)),
    ("rule_keyword", "rule acceptance (HeuristicBackend) + keyword final_offer",
     EstimateSpec(acceptance_source="rule", final_source="keyword")),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def policy_json() -> dict:
    """The policy constants every tuning run uses: the defaults, as they read back from JSON."""
    # The fields added on 2026-11-19 are left out at their (legacy) defaults so the frozen protocol
    # digest of tune-2026-11-18.v1 is unchanged; a missing field reads back as that default.
    from .policy import LEGACY_POLICY_FIELDS
    return json.loads(json.dumps({k: v for k, v in asdict(PolicyConfig()).items()
                                  if not (k in LEGACY_POLICY_FIELDS and v == LEGACY_POLICY_FIELDS[k])}))


@dataclass(frozen=True)
class Config:
    """One runnable configuration: a candidate or a CPU reference."""

    name: str
    role: str                     # "candidate" | "reference"
    kind: str                     # "estimates" | "baseline" | "true_acceptance"
    description: str
    spec: EstimateSpec | None = None

    def identity(self) -> dict:
        if self.kind == "estimates":
            return {"policy": policy_json(), "estimates": self.spec.to_json()}
        key = "baseline" if self.kind == "baseline" else "evaluator_reference"
        return {"policy": policy_json(), key: self.name}

    @property
    def id(self) -> str:
        return f"{self.name}-{_digest(self.identity())[:12]}"

    def uses_model(self) -> bool:
        return self.kind == "estimates" and self.spec.uses_model()

    def estimate_sources(self) -> dict:
        if self.kind == "estimates":
            return self.spec.estimate_sources()
        value = "none_baseline_policy" if self.kind == "baseline" else "evaluator_only_true_answer"
        return {"acceptance": value, "final_offer": value}

    def display_sources(self) -> dict:
        """The sources as shown in the leaderboard: a probability map and a shrink weight are visible."""
        sources = dict(self.estimate_sources())
        if self.kind == "estimates":
            if self.spec.acceptance_platt is not None:
                sources["acceptance"] += "+platt(calibration)"
            if self.spec.acceptance_source == "constant":
                sources["acceptance"] += f"({self.spec.acceptance_constant})"
            if self.spec.final_shrink_weight is not None:
                sources["final_offer"] += f"(w={self.spec.final_shrink_weight})"
        return sources

    def view(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "kind": self.kind,
                "description": self.description, "policy": policy_json(),
                "estimates": self.spec.to_json() if self.spec is not None else None}


def reference_configs() -> list[Config]:
    """CPU rows with zero model calls. Never candidates. Twins of the candidates are deduplicated in."""
    out = [Config(name, "reference", "estimates", text, spec) for name, text, spec in _NAMED_REFERENCE_SPECS]
    seen = {config.spec.digest() for config in out}
    for arm in CANDIDATES:
        twin = arm.template.twin()
        if twin.digest() not in seen:
            seen.add(twin.digest())
            out.append(Config(f"twin_of_{arm.name}", "reference", "estimates",
                              f"model-free twin of {arm.name}", twin))
    out += [Config(name, "reference", "baseline", f"baseline policy {name}; no estimate is used")
            for name in BASELINES]
    out.append(Config(REFERENCE_POLICY, "reference", "true_acceptance",
                      "evaluator-only reference using hidden supplier information; not a competitor"))
    return out


def twin_reference(spec: EstimateSpec) -> Config:
    """The reference row that is the model-free twin of `spec`."""
    digest = spec.twin().digest()
    for config in reference_configs():
        if config.spec is not None and config.spec.digest() == digest:
            return config
    return Config("twin_of_selected", "reference", "estimates", "model-free twin of the selected configuration",
                  spec.twin())


@dataclass(frozen=True)
class ArmState:
    arm: Arm
    status: str                   # "resolved" | "pending" | "skipped_identical_to_*" | "failed"
    reason: str | None
    config: Config | None

    @property
    def config_id(self) -> str:
        if self.config is not None:
            return self.config.id
        return f"{self.arm.name}-{'pending' if self.status == 'pending' else self.reason}"

    @property
    def terminal_without_run(self) -> bool:
        return self.status.startswith("skipped_") or self.status == "failed"


def resolve_arms(maps: dict | None) -> list[ArmState]:
    """Fill the calibration-split fits into the candidate templates. Before `fit-map`, such arms are pending."""
    states = []
    for arm in CANDIDATES:
        spec, status, reason = arm.template, "resolved", None
        if arm.needs and maps is None:
            states.append(ArmState(arm, "pending", "waiting_for_fit_map", None))
            continue
        if "final_offer_shrink" in arm.needs:
            weight = (maps.get("final_offer_shrink") or {}).get("weight")
            if weight is None:
                status, reason = "failed", "map_unavailable"
            else:
                spec = replace(spec, final_shrink_weight=float(weight))
                if float(weight) == 0.0:      # the model value has weight 0: no model final-offer value is used
                    status, reason = arm.zero_weight_status, "final_offer_shrink_weight_is_zero"
        if "platt" in arm.needs and status == "resolved":
            platt = maps.get("platt")
            if not isinstance(platt, dict):
                status, reason = "failed", "map_unavailable"
            else:
                spec = replace(spec, acceptance_platt=json.loads(json.dumps(platt)))
        if status == "failed":
            states.append(ArmState(arm, status, reason, None))
            continue
        spec.validate()
        states.append(ArmState(arm, status, reason, Config(arm.name, "candidate", "estimates", arm.description, spec)))
    return states


# ---- blocks and protocol document ----------------------------------------------------------------------

def parse_rehearsal_blocks(text: str | None) -> dict[str, int] | None:
    """`d0=4,select=6,confirm=8,cal=6` -> the first N cases of each named block (CPU rehearsal only)."""
    if not text:
        return None
    sizes: dict[str, int] = {}
    for part in text.split(","):
        name, _, value = part.strip().partition("=")
        if name not in BLOCKS or not value.isdigit():
            raise TuneRefusal(f"--rehearsal-blocks: expected name=N with a name in {sorted(BLOCKS)}, got {part!r}")
        sizes[name] = int(value)
    return _checked_rehearsal(sizes)


def _checked_rehearsal(sizes: Any) -> dict[str, int] | None:
    if sizes is None:
        return None
    if not isinstance(sizes, dict) or not sizes:
        raise TuneRefusal("rehearsal blocks must name at least one block")
    for name, size in sizes.items():
        if name not in BLOCKS or isinstance(size, bool) or not isinstance(size, int):
            raise TuneRefusal(f"rehearsal blocks: unknown block or size {name!r}={size!r}")
        _, start, stop = BLOCKS[name]
        if not (2 if name == "d0" else 1) <= size <= stop - start:
            raise TuneRefusal(f"rehearsal block {name} must keep between {2 if name == 'd0' else 1} and "
                              f"{stop - start} cases")
    return {name: sizes[name] for name in sorted(sizes)}


def block_cases(name: str, rehearsal: dict[str, int] | None = None) -> list:
    """The cases of one block: a slice of the split's single prefix stream."""
    split, start, stop = BLOCKS[name]
    if split not in ALLOWED_SPLITS:
        raise TuneRefusal(f"block {name} names a split that tuning may not use")
    if rehearsal and name in rehearsal:
        stop = start + rehearsal[name]
    return make_cases(split, stop)[start:stop]


def block_content_sha256(cases: list) -> str:
    """Identity of the block's content (suppliers and our private inputs), independent of formatting."""
    return _digest({"identity_schema_version": "1.0",
                    "cases": [{"case_id": c.case_id, "supplier_hidden": asdict(c.hidden)} for c in cases],
                    "buyer_private": asdict(default_buyer_private())})


def block_info(name: str, cases: list, subset: str | None = None) -> dict:
    split, start, stop = BLOCKS[name]
    info = {"name": name, "split": split, "registered_slice": [start, stop], "cases": len(cases),
            "first_case_id": cases[0].case_id, "last_case_id": cases[-1].case_id,
            "content_sha256": block_content_sha256(cases), "use": BLOCK_USE[name]}
    if subset:
        info["subset"] = subset
    return info


def golden_request_sha256() -> str:
    """Hash of the default round-1 decision request of dev-1100. Model-free; guards the request contract."""
    case = make_cases("dev", 1)[0]
    state = initial_state(case.case_id, case.hidden)
    candidates = policy.generate_candidates(state, default_buyer_private(), PolicyConfig())
    return request_sha256(build_decision_request(state, candidates))


def check_golden_request() -> str:
    actual = golden_request_sha256()
    if actual != GOLDEN_REQUEST["sha256"]:
        raise TuneRefusal(f"the default dev-1100 round-1 request hashes to {actual}, not the recorded "
                          f"{GOLDEN_REQUEST['sha256']}: the request contract changed; stored answers and earlier "
                          "runs would not be comparable")
    return actual


def build_protocol(rehearsal: dict[str, int] | None = None) -> dict:
    """The whole pre-registered protocol as plain data. Its digest binds a state directory."""
    rehearsal = _checked_rehearsal(rehearsal)
    references = reference_configs()
    by_digest = {config.spec.digest(): config.name for config in references if config.spec is not None}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "gpu_notice": GPU_NOTICE,
        "what_is_tuned": "which source supplies each estimate the policy uses; request wording, policy constants "
                         "and the model are unchanged",
        "blocks": {name: block_info(name, block_cases(name, rehearsal)) for name in BLOCKS},
        "reserve": dict(RESERVE),
        "rehearsal_blocks": rehearsal,
        "registered_protocol": rehearsal is None,
        "policy": policy_json(),
        "config_id_rule": "<name>-<first 12 hex of sha256 of canonical {policy, estimates}>",
        "candidates": [{"name": arm.name, "description": arm.description, "estimates_template": arm.template.to_json(),
                        "needs_fit_on_calibration_split": list(arm.needs),
                        "if_shrink_weight_is_zero": arm.zero_weight_status,
                        "model_free_twin": by_digest[arm.template.twin().digest()]} for arm in CANDIDATES],
        "references": [{"name": config.name, "kind": config.kind, "description": config.description,
                        "estimates": config.spec.to_json() if config.spec is not None else None,
                        "label": "cpu_reference_not_model_evidence", "selectable": False} for config in references],
        "primary_metric": "mean realized saving versus the alternative quote, per case",
        "guard_metric": "mean net saving after process costs",
        "selection_rule": SELECTION_RULE,
        "confirm_rule": dict(CONFIRM_RULE),
        "verdict_vocabulary": list(VERDICTS),
        "bootstrap": dict(BOOTSTRAP),
        "map_fitting": {"split": "calibration", "from": "the complete calibrate result of M0 only",
                        "verified_on_load": "the maps file is refitted from that result on every load and must "
                                            "be equal; a hand-written or edited file is refused",
                        "if_calibration_cannot_complete": "`fit-map --declare-unavailable --reason TEXT` is a "
                                                          "person's recorded decision, allowed only after the "
                                                          "calibrate run failed, was retried with --retry-failed "
                                                          "and failed again; the five arms that need a map then "
                                                          "become failed/map_unavailable and M0, M1, M2 remain",
                        "platt": "evaluation.fit_calibration on the acceptance records",
                        "temperature_equivalent": "one-parameter fit (b = 0), reported only; never applied",
                        "final_offer_shrink_grid": list(SHRINK_GRID),
                        "final_offer_shrink_rule": "lowest Brier score of w*p_model + (1-w)*round_prior against the "
                                                   "simulator's last-chance truth; ties go to the smaller weight"},
        "stage_purposes": dict(STAGE_PURPOSE),
        "allowed_result_sentences": dict(RESULT_SENTENCES),
        "known_exposure": KNOWN_EXPOSURE,
        "golden_request": dict(GOLDEN_REQUEST),
        "plan_estimates": {"seconds_per_request": PLAN_SECONDS_PER_REQUEST,
                           "requests_per_30_dev_cases": PLAN_REQUESTS_PER_30_DEV_CASES,
                           "note": "estimates from the 2026-11-16 T4 run; an upper bound is a full pass per "
                                   "configuration"},
        "real_model_rule": "the only real-model backend is the native adapter (--backend zoro) with an explicit "
                           "--decision-temperature 1; the HTTP backends are refused because their answers cannot "
                           "be bound to a server identity; rehearsal blocks are refused for real-model backends",
        "failed_estimate_rule": "an attempt stops at its first failed estimate and the stage stops with it; "
                                "nothing is stored or filled in for it. Running again continues with the other "
                                "configurations and leaves that one failed; --retry-failed is a person's recorded "
                                "decision to ask again",
    }


def plan_estimates(protocol: dict) -> list[dict]:
    """Upper-bound request and minute ESTIMATES per stage (a full pass per configuration)."""
    per_case = PLAN_REQUESTS_PER_30_DEV_CASES / 30
    blocks = protocol["blocks"]
    table = (("smoke", 2, 1), ("regress", blocks["d0"]["cases"], 1), ("calibrate", blocks["cal"]["cases"], 1),
             ("select", blocks["select"]["cases"], len(CANDIDATES)), ("confirm", blocks["confirm"]["cases"], 2))
    out = []
    for stage, cases, configurations in table:
        requests = math.ceil(cases * per_case * configurations)
        out.append({"stage": stage, "cases": cases, "model_configurations": configurations,
                    "estimated_requests_upper_bound": requests,
                    "estimated_minutes_upper_bound": round(requests * PLAN_SECONDS_PER_REQUEST / 60, 1)})
    return out


# ---- small file helpers --------------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, exclusive: bool = False) -> None:
    """Exclusive create, or an atomic replacement of a file this runner owns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        # Complete content first, then the final name: a kill or a full disk leaves no half-written file.
        staged = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(2)}")
        try:
            with staged.open("x", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(staged, path)            # fails with FileExistsError when the name is taken
            except FileExistsError:
                raise
            except OSError:                      # a file system without hard links: fall back to exclusive open
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
        finally:
            staged.unlink(missing_ok=True)
        return
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _protocol_digest_of_file(document: dict) -> str:
    return _digest({k: v for k, v in document.items() if k not in ("protocol_sha256", "created_at_utc")})


# ---- map fitting (calibration split only) ----------------------------------------------------------------

def _fit_scale_only(xs: list[float], ys: list[float], iterations: int = 400, lr: float = 0.3,
                    l2: float = 1e-3) -> float:
    """metrics._fit_platt with the intercept fixed at 0: the one-parameter (temperature) fit."""
    a, n = 1.0, len(xs)
    for _ in range(iterations):
        gradient = sum((1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, a * x)))) - y) * x for x, y in zip(xs, ys))
        a -= lr * (gradient / n + l2 * a)
    return a


def fit_final_offer_shrink(records: list[dict]) -> dict:
    """Weight of the model's final-offer value against the public round prior, by Brier score on a grid.

    `records` hold `p_model_raw`, `round_prior` and `truth` of decisions that were not at the
    public round limit. Ties go to the smaller weight. With fewer than two positives or two
    negatives no weight is fitted.
    """
    usable = [r for r in records if r.get("p_model_raw") is not None and r.get("round_prior") is not None]
    truths = [bool(r["truth"]) for r in usable]
    positives = sum(truths)
    out: dict[str, Any] = {"weight": None, "n": len(usable), "positives": positives,
                           "brier_model": None, "brier_prior": None, "brier_shrunk": None, "grid": []}
    if positives < 2 or len(usable) - positives < 2:
        out["unavailable_reason"] = "fewer than 2 positives or 2 negatives among the final-offer records"
        return out
    best = None
    for weight in SHRINK_GRID:
        score = M.brier([shrink(float(r["p_model_raw"]), float(r["round_prior"]), weight) for r in usable], truths)
        out["grid"].append({"weight": weight, "brier": score})
        if best is None or score < best[1]:          # strict: a tie keeps the smaller weight
            best = (weight, score)
    out.update(weight=best[0], brier_shrunk=best[1],
               brier_model=M.brier([float(r["p_model_raw"]) for r in usable], truths),
               brier_prior=M.brier([float(r["round_prior"]) for r in usable], truths))
    return out


def fit_maps(result: dict, source_sha256: str) -> dict:
    """Fit every map from the complete calibrate result of M0. Anything else is refused (ValueError)."""
    config = result.get("config") or {}
    problems = [text for ok, text in (
        (result.get("experiment") == "negotiation_usage_tuning", "not a tuning result"),
        (result.get("stage") == "calibrate" and result.get("purpose") == "map_fitting_only",
         "not a calibrate-stage result"),
        ((result.get("block") or {}).get("split") == "calibration", "not run on the calibration split"),
        (result.get("status") == "complete" and result.get("complete") is True, "the run is not complete"),
        (config.get("name") == "M0" and config.get("estimates") == EstimateSpec().to_json()
         and config.get("policy") == policy_json(), "the configuration is not the default M0"),
    ) if not ok]
    if problems:
        raise ValueError("fit-map accepts only the complete calibrate result of M0: " + "; ".join(problems))
    maps: dict[str, Any] = {"fitted_on_split": "calibration", "platt": None, "platt_unavailable_reason": None,
                            "temperature_equivalent": None}
    records = result.get("acceptance_predictions") or []
    shim = {"experiment": "full_negotiation_evaluation", "split": "calibration", "backend": result.get("backend", {}),
            "evaluator_only": {"acceptance_predictions": {"agent": records}}}
    try:
        maps["platt"] = fit_calibration(shim, source=source_sha256)     # the same schema as fit-calibration
    except ValueError as error:
        maps["platt_unavailable_reason"] = str(error)
    else:
        xs = [M._logit(r["p"]) for r in records]
        ys = [1.0 if r["truth"] else 0.0 for r in records]
        a = _fit_scale_only(xs, ys)
        fitted = [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, a * x)))) for x in xs]
        maps["temperature_equivalent"] = {
            "a": a, "temperature": (1.0 / a) if a > 0 else None, "n": len(records),
            "brier_before": M.brier([r["p"] for r in records], [r["truth"] for r in records]),
            "brier_after_in_sample": M.brier(fitted, [r["truth"] for r in records]),
            "note": "information only: the b = 0 one-parameter fit. NOT a candidate and never applied",
        }
    finals = [r for r in result.get("final_offer_records") or [] if r.get("policy_source") == "backend_estimate"]
    maps["final_offer_shrink"] = fit_final_offer_shrink(finals)
    return maps


# ---- selection and verdict (pure functions of the leaderboard rows) ----------------------------------------

def choose_candidate(candidate_rows: list[dict]) -> dict | None:
    """The pre-registered selection rule over the candidate rows, which are in candidate order."""
    eligible = [row for row in candidate_rows if row["selectable"]]
    # max() keeps the first of equal values: a tie goes to the earlier candidate, so M0 wins every tie.
    return max(eligible, key=lambda row: row["mean_realized_saving"]) if eligible else None


def primary_verdict(primary: dict, default_kept: bool, runs_complete: bool,
                    selected_authority_violations: int | None) -> tuple[str, str]:
    """improvement_confirmed | not_confirmed | incomplete, with the reason. `primary` is S minus M0."""
    if not runs_complete:
        return "incomplete", "a needed confirm run is not complete"
    if default_kept:
        return "not_confirmed", "default kept: the selected configuration is M0"
    if primary.get("realized_saving") is None or primary.get("net_saving") is None:
        return "incomplete", primary.get("reason") or "the comparison could not be computed"
    lower = primary["realized_saving"]["ci90"][0]
    net = primary["net_saving"]["mean_difference"]
    if lower <= 0:
        return "not_confirmed", "the 90% interval of S minus M0 does not lie above 0"
    if net < 0:
        return "not_confirmed", "the mean net-saving difference is negative"
    if selected_authority_violations != 0:
        return "not_confirmed", "the selected configuration has authority violations"
    return "improvement_confirmed", "lower bound above 0 and net saving not worse"


# ---- per-decision accounting ---------------------------------------------------------------------------

def classify_decision(model: dict) -> str:
    """Where one decision's estimates came from, read from the recorded response only."""
    if not model:
        return "no_request"
    if model.get("cached"):
        return "session_cache"
    response = model.get("response")
    block = response.get("prediction_store") if isinstance(response, dict) else None
    if isinstance(block, dict):
        if block.get("hit"):
            return "store_hit"
        return "live_failed" if block.get("stored") is False else "live"
    if (isinstance(response, dict) and "composition" in response) or model.get("status") == "ok":
        return "cpu_estimate"
    return "no_answer"       # nothing answered: a read-only store miss or a backend that could not be built


def final_offer_records(row: dict) -> list[dict]:
    """run_case's evaluator-only final-offer rows, each with the composed source of the same decision."""
    finals, seen, out = list(row["final_offer_estimates"]), set(), []
    for decision in row["internal_record"]:
        model = decision["model"]
        digest = model.get("request_sha256")
        if not digest or digest in seen or "error" in model:       # the same filter as evaluation.run_case
            continue
        seen.add(digest)
        if decision["facts"].get("p_final_offer_source") not in ("backend_estimate", "policy_prior"):
            continue
        if len(out) >= len(finals) or finals[len(out)]["round"] != decision["round"]:
            raise RuntimeError(f"final-offer records of {row['case_id']} do not line up with its decisions")
        base = finals[len(out)]
        response = model.get("response") if isinstance(model.get("response"), dict) else {}
        composition = response.get("composition") or {}
        raw = composition.get("raw_model_predictions") or {}
        out.append({"case_id": row["case_id"], "round": base["round"], "request_sha256": digest,
                    "policy_source": base["source"], "p_used": base["p"], "truth": base["truth"],
                    "composed_source": (composition.get("estimate_sources") or {}).get("final_offer"),
                    "value_from": composition.get("final_offer_value_from"),
                    "p_model_raw": raw.get(FINAL_OFFER_QID), "round_prior": composition.get("round_prior")})
    if len(out) != len(finals):
        raise RuntimeError(f"final-offer records of {row['case_id']} do not line up with its decisions")
    return out


def _kept_row(row: dict) -> dict:
    """A run_case row without the per-decision record (that is in decisions.jsonl) and with honest names."""
    kept = {k: v for k, v in row.items() if k not in ("internal_record", "model_requests", "latencies_ms")}
    kept["session_decisions_with_request"] = row["model_requests"]   # NOT a count of live model calls
    kept["decision_latencies_ms"] = row["latencies_ms"]              # live calls and store hits mixed
    return kept


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# ---- the runner ----------------------------------------------------------------------------------------

class Tuner:
    """One state directory. Builds at most one model backend per process, lazily, behind the store."""

    def __init__(self, state_dir: Path, args: Any, *, backend_factory: Callable[[], Any] | None = None,
                 declared: tuple[str, bool, bool] | None = None, real_model: bool | None = None,
                 store_requested: dict | None = None, clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] | None = None) -> None:
        self.state = Path(state_dir)
        self.args = args
        self.backend_choice = getattr(args, "backend", None)
        self._factory = backend_factory if backend_factory is not None else (lambda: make_backend(args))
        self._declared = declared
        self._real_model = real_model
        self._store_requested = store_requested
        self.clock, self.started = clock, clock()
        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.store_mode = getattr(args, "store_mode", None) or "read-write"
        budget = getattr(args, "budget_minutes", None)
        self.budget_seconds = None if budget is None else float(budget) * 60.0
        self.max_live = getattr(args, "max_live_requests", None)
        self.live_total = 0                    # live requests made by this process
        self.protocol: dict | None = None
        self.protocol_sha256: str | None = None
        self.identity: dict | None = None
        self._stored: StoredPredictionBackend | None = None
        self._cases: dict[str, list] = {}
        self._results: dict[tuple, dict] = {}
        self.outcomes: list[dict] = []         # one entry per configuration this process touched
        self.last_board: dict | None = None
        self._maps_cache: tuple | None = None
        self._lock: Path | None = None

    # ---- protocol ------------------------------------------------------------------------------------

    def ensure_protocol(self, create: bool, rehearsal: dict[str, int] | None = None) -> bool:
        """Verify `<state>/protocol.json` against this code, or write it exclusively. True when written now."""
        path = self.state / "protocol.json"
        created = False
        refusal = "rehearsal blocks are a CPU control-flow check and are refused for real-model backends"
        if rehearsal and self.backend_choice in REAL_MODEL_BACKENDS:
            raise TuneRefusal(refusal)
        if not path.exists():
            if not create:
                raise TuneRefusal(f"{path} does not exist; run `plan` first")
            if any(self.state.glob("*")) if self.state.is_dir() else False:
                raise TuneRefusal(f"{self.state} is not empty and has no protocol.json; use a new --state-dir")
            document = build_protocol(rehearsal)
            try:
                _write_json(path, {**document, "protocol_sha256": _digest(document), "created_at_utc": _utc_now()},
                            exclusive=True)
                created = True
            except FileExistsError:
                pass                        # written by someone else in the meantime: verify it like any other
        try:
            stored = _read_json(path)
        except (OSError, ValueError) as error:
            raise TuneRefusal(f"{path} is unreadable: {error}") from error
        recorded = stored.get("rehearsal_blocks") if isinstance(stored, dict) else None
        if rehearsal is not None and _checked_rehearsal(rehearsal) != recorded:
            raise TuneRefusal("this state directory was planned with different blocks; "
                              "protocol digest differs, use a new --state-dir")
        expected = build_protocol(recorded)
        digest = _digest(expected)
        if not isinstance(stored, dict) or stored.get("protocol_sha256") != digest \
                or _protocol_digest_of_file(stored) != digest:
            raise TuneRefusal(f"{path} does not match protocol {PROTOCOL_VERSION} of this code "
                              "(protocol digest differs); use a new --state-dir")
        self.protocol, self.protocol_sha256 = expected, digest
        if recorded and self.backend_choice in REAL_MODEL_BACKENDS:
            raise TuneRefusal(refusal)
        return created

    @property
    def rehearsal(self) -> dict[str, int] | None:
        return self.protocol["rehearsal_blocks"]

    def cases(self, name: str) -> list:
        if name not in self._cases:
            self._cases[name] = block_cases(name, self.rehearsal)
            if block_content_sha256(self._cases[name]) != self.protocol["blocks"][name]["content_sha256"]:
                raise TuneRefusal(f"the content of block {name} changed after the protocol was written")
        return self._cases[name]

    # ---- backend identity ----------------------------------------------------------------------------

    @property
    def declared(self) -> tuple[str, bool, bool]:
        if self._declared is not None:
            return self._declared
        cls = BACKEND_CLASSES[self.backend_choice]
        return cls.kind, bool(cls.is_model), bool(getattr(cls, "supports_final_offer", cls.is_model))

    @property
    def real_model(self) -> bool:
        return self.backend_choice in REAL_MODEL_BACKENDS if self._real_model is None else bool(self._real_model)

    def requested_identity(self) -> dict:
        """What must be equal for stored answers to be reused. No path and no URL."""
        if self._store_requested is not None:
            return dict(self._store_requested)
        choice, args = self.backend_choice, self.args
        requested: dict[str, Any] = {
            "backend": choice, "backend_kind": self.declared[0],
            "requests_sha256": file_sha256(REPOSITORY_ROOT / REQUESTS_FILE),
            "model_revision": None, "source_revision": None, "dtype": None, "max_input_tokens": None,
            "decision_temperature": None, "reasoning": False,
        }
        adapter = ADAPTER_FILES.get(choice, NEGOTIATION_BACKENDS_FILE)
        requested.update(adapter_file=adapter, adapter_sha256=file_sha256(REPOSITORY_ROOT / adapter))
        if choice in REAL_MODEL_BACKENDS:
            pins = _read_json(REPOSITORY_ROOT / "config" / "model_pins.json")
            requested.update(model_revision=pins["model_revision"], source_revision=pins["source_revision"],
                             decision_temperature=float(args.decision_temperature),
                             reasoning=bool(getattr(args, "reasoning", False)))
            if choice == "zoro":
                requested.update(dtype=args.dtype, max_input_tokens=args.max_input_tokens)
            if choice == "systemone":
                requested["served_model_requested"] = args.model
        else:
            requested["cpu_stand_in_not_a_model"] = True
        return requested

    def check_run_arguments(self) -> None:
        """Everything that can be refused before a file is written or a backend is built."""
        choice, args = self.backend_choice, self.args
        if choice is None:
            raise TuneRefusal("--backend is required for `run` (keyword for a CPU rehearsal, zoro for the pinned "
                              "model); a state directory is bound to the first choice")
        try:
            reasoning_requested(args)
        except SystemExit as error:
            raise TuneRefusal(str(error)) from error
        if not self.declared[2]:
            raise TuneRefusal(f"--backend {choice} does not answer the final-offer question, so it cannot stand in "
                              "for the model here; use --backend keyword for a CPU rehearsal")
        if self.store_mode not in ("read-write", "read-only"):
            raise TuneRefusal("--store-mode must be read-write or read-only")
        if choice in REAL_MODEL_BACKENDS and choice != NATIVE_BACKEND:
            raise TuneRefusal(
                f"--backend {choice} is refused by this protocol version: tuning uses the native adapter only "
                f"(--backend {NATIVE_BACKEND}). An HTTP client cannot bind the answering server to a store "
                "identity, so answers of different servers could mix, and the official server measured about "
                "6 s per request on a T4 against about 1.5 s natively")
        if choice in REAL_MODEL_BACKENDS:
            if getattr(args, "decision_temperature", None) is None or float(args.decision_temperature) != 1.0:
                raise TuneRefusal("real-model backends need an explicit --decision-temperature 1 so that stored "
                                  "answers are comparable")
            if self.store_mode == "read-write":
                if choice == "zoro" and not (args.original_checkpoint and args.exported_checkpoint):
                    raise TuneRefusal("--backend zoro needs --original-checkpoint and --exported-checkpoint")
                if choice in ("zoro-http", "systemone") and not args.url:
                    raise TuneRefusal(f"--url is required for --backend {choice}")

    def bind_identity(self) -> None:
        """A state directory holds results of ONE store identity; another one is refused."""
        requested = json.loads(_canonical(self.requested_identity()))
        identity = identity_digest(requested)
        kind, is_model, final = self.declared
        document = {"schema_version": 1, "identity": identity, "identity12": identity[:12], "requested": requested,
                    "backend_choice": self.backend_choice, "real_model_backend": self.real_model,
                    "declared": {"kind": kind, "is_model": is_model, "supports_final_offer": final},
                    "bound_at_utc": _utc_now(),
                    "notice": "Every result in this state directory used this store identity. The first live "
                              "response additionally binds the observed runtime inside the store."}
        path = self.state / "store_identity.json"
        try:
            _write_json(path, document, exclusive=True)
        except FileExistsError:
            document = _read_json(path)
            if document.get("identity") != identity:
                raise TuneRefusal(
                    f"this state directory is bound to store identity {document.get('identity12')} "
                    f"(--backend {document.get('backend_choice')}); the current flags and source files give "
                    f"{identity[:12]}. Results of different identities are not mixed: use a new --state-dir")
        self.identity = document

    def load_identity(self) -> dict | None:
        path = self.state / "store_identity.json"
        if self.identity is None and path.exists():
            self.identity = _read_json(path)
        return self.identity

    def _build_backend(self) -> Any:
        try:
            return self._factory()
        except SystemExit as error:      # a CLI-style refusal must not end the process inside an estimate
            raise RuntimeError(f"backend initialization refused: {error}") from error

    def stored_backend(self) -> StoredPredictionBackend:
        if self._stored is None:
            try:
                store = PredictionStore(self.state / "store", self.identity["requested"], mode=self.store_mode)
            except RuntimeError as error:
                raise TuneRefusal(f"prediction store refused: {type(error).__name__}: {error}") from error
            kind, is_model, final = self.declared
            self._stored = StoredPredictionBackend(store, self._build_backend, kind=kind, is_model=is_model,
                                                   supports_final_offer=final)
        return self._stored

    def close(self) -> None:
        self.release_lock()
        if self._stored is not None:
            self._stored.close()

    # ---- attempts ------------------------------------------------------------------------------------

    def _attempts(self, stage: str, config_id: str) -> list[dict]:
        folder = self.state / "runs" / stage / config_id
        out = []
        if folder.is_dir():
            for path in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("attempt-")):
                file = path / "result.json"
                try:
                    stat = file.stat()
                    key = (str(file), stat.st_mtime_ns, stat.st_size)
                    if key not in self._results:
                        self._results[key] = _read_json(file)
                    result = self._results[key]
                    if result.get("protocol_sha256") != self.protocol_sha256:
                        result = {"status": "foreign_protocol", "complete": False}
                    else:
                        bound, used = self.load_identity(), result.get("store_identity")
                        if bound and isinstance(used, dict) and used.get("identity12") not in (
                                None, bound.get("identity12")):
                            result = {"status": "foreign_store_identity", "complete": False}
                except (OSError, ValueError):
                    result = {"status": "unreadable", "complete": False}
                out.append({"name": path.name, "path": path, "result": result})
        return out

    def state_of(self, stage: str, config_id: str, use: str | None = None) -> dict:
        """complete | failed (terminal unless retried) | incomplete (rerun continues from the store) | not_run."""
        attempts = [a for a in self._attempts(stage, config_id)
                    if use is None or a["result"].get("confirm_use") == use]
        complete = [a for a in attempts
                    if a["result"].get("complete") is True and a["result"].get("status") == "complete"]
        if complete:
            return {"status": "complete", "attempt": complete[0], "attempts": len(attempts)}
        if not attempts:
            return {"status": "not_run", "attempt": None, "attempts": 0}
        last = attempts[-1]
        if last["result"].get("status") == "failed":
            return {"status": "failed", "attempt": last, "attempts": len(attempts),
                    "reason": last["result"].get("reason")}
        return {"status": "incomplete", "attempt": last, "attempts": len(attempts),
                "detail": last["result"].get("status")}

    def _new_attempt_folder(self, stage: str, config_id: str) -> Path:
        parent = self.state / "runs" / stage / config_id
        parent.mkdir(parents=True, exist_ok=True)
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            folder = parent / f"attempt-{stamp}-{secrets.token_hex(2)}"
            try:
                folder.mkdir()               # exclusive: an attempt folder is never reused
                return folder
            except FileExistsError:
                continue

    def budget_reached(self) -> bool:
        if self.budget_seconds is not None and self.clock() - self.started >= self.budget_seconds:
            return True
        return self.max_live is not None and self.live_total >= self.max_live

    def _backend_settings(self) -> dict:
        try:
            return backend_execution_config(self.args)
        except SystemExit as error:
            raise TuneRefusal(str(error)) from error

    def _policy_backend(self, config: Config) -> tuple[Any, Callable]:
        if config.kind == "baseline":
            return None, BASELINES[config.name]
        if config.kind == "true_acceptance":
            return None, policy.decide              # the per-case reference backend is built in the case loop
        table = PolicyConfig().final_prior_by_round
        inner = self.stored_backend() if config.uses_model() else None      # model-free: no store, no backend
        return ComposedBackend(inner, config.spec, table), policy.decide

    def _evidence_class(self, config: Config, counts: dict) -> str:
        if not config.uses_model():
            return "cpu_reference_not_model_evidence"
        if not self.real_model:
            return "cpu_rehearsal_not_model_evidence"
        live, hits = counts["live_requests"], counts["store_hits"]
        if live and hits:
            return "real_model_run_partly_replayed_from_store"
        if live:
            return "real_model_run"
        return "real_model_replay_from_store" if hits else "real_model_requested_no_model_answer"

    def run_config(self, stage: str, block: dict, cases: list, config: Config, *, retry_failed: bool = False,
                   labels: dict | None = None, use: str | None = None, force: bool = False,
                   after: Callable[[dict, list], None] | None = None) -> dict:
        """Run one configuration on one block unless it is already done. Returns {"status", "result"}."""
        state = self.state_of(stage, config.id, use)
        previous = state["attempt"]
        if not force:
            if state["status"] == "complete":
                self.log(f"{stage} {config.id}: already complete ({previous['name']}); skipped")
                return self._note(stage, config, "skipped_complete", previous["result"])
            if state["status"] == "failed" and not retry_failed:
                self.log(f"{stage} {config.id}: failed earlier ({state.get('reason')}); not rerun without "
                         "--retry-failed")
                return self._note(stage, config, "skipped_failed", previous["result"])
        if self.budget_reached():
            self.log(f"{stage} {config.id}: budget reached before it started")
            return self._note(stage, config, "partial_budget", None)
        link: dict[str, Any] = {}
        if previous is not None and not force:
            if state["status"] == "failed":
                link = {"retry_of": previous["name"], "retry_failed_flag": True}
            else:
                link = {"continues_after": previous["name"]}
        result = self._attempt(stage, block, cases, config, {**(labels or {}), **link}, after)
        return self._note(stage, config, result["status"], result)

    def _note(self, stage: str, config: Config, status: str, result: dict | None) -> dict:
        self.outcomes.append({"stage": stage, "config_id": config.id, "status": status})
        return {"status": status, "result": result}

    def _attempt(self, stage: str, block: dict, cases: list, config: Config, labels: dict,
                 after: Callable[[dict, list], None] | None) -> dict:
        uses_model = config.uses_model()
        settings = {**self._backend_settings(),
                    "tuning": {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
                               "stage": stage, "block": block["name"], "config": config.view(),
                               "store_mode": self.store_mode if uses_model else None}}
        snapshot = provenance(settings)              # before any backend is built for this attempt
        folder = self._new_attempt_folder(stage, config.id)
        identity = self.identity or {}
        result: dict[str, Any] = {
            "schema_version": 1, "experiment": "negotiation_usage_tuning",
            "protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
            "rehearsal_blocks": self.rehearsal, "stage": stage, "purpose": STAGE_PURPOSE[stage], "block": block,
            "config": config.view(), "attempt": folder.name, "created_at_utc": _utc_now(), **labels,
            "status": "initializing", "complete": False,
            "backend_choice": self.backend_choice, "real_model_backend": self.real_model,
            "estimate_sources": config.estimate_sources(),
            "store_identity": ({"identity": identity.get("identity"), "identity12": identity.get("identity12"),
                                "requested": identity.get("requested"), "mode": self.store_mode}
                               if uses_model else None),
            "provenance": snapshot, "decisions_file": "decisions.jsonl",
            "scope": "synthetic rule-based suppliers, development or calibration cases; not evidence about real "
                     "negotiations and not a held-out result",
        }
        _write_json(folder / "result.json", result, exclusive=True)

        counts = {key: 0 for key in ("decisions", "live_requests", "live_failed_requests", "store_hits",
                                     "session_cache_hits", "cpu_estimates", "no_request", "no_answer",
                                     "failed_estimates", "read_only_store_misses")}
        names = {"live": "live_requests", "live_failed": "live_failed_requests", "store_hit": "store_hits",
                 "session_cache": "session_cache_hits", "cpu_estimate": "cpu_estimates",
                 "no_request": "no_request", "no_answer": "no_answer"}
        errors_by_type: dict[str, int] = {}
        first_errors: list[dict] = []
        live_latencies: list[float] = []
        flags = {"identity_mismatch": False, "initialization_failed": False, "interrupted": False,
                 "store_write_failed": False}
        rows: list[dict] = []
        predictions: list[dict] = []
        status: str | None = None
        reason: str | None = None
        error: str | None = None
        quiet = stage == "calibrate"                 # calibration savings are never printed
        stored = self._stored if uses_model else None
        backend = None
        before = None
        wall_started = time.perf_counter()
        try:
            backend, decide_fn = self._policy_backend(config)
            stored = self._stored if uses_model else None
            before = (stored.live_requests, stored.stored_answers, stored.store_hits, stored.store_misses,
                      stored.live_seconds, stored.factory_seconds) if stored is not None else None
            private, config_policy = default_buyer_private(), PolicyConfig()
            with (folder / "decisions.jsonl").open("x", encoding="utf-8") as stream:
                for number, case in enumerate(cases, 1):
                    if self.budget_reached():
                        status = "partial_budget"
                        break

                    def save(record: dict, case_id: str = case.case_id) -> None:
                        line = json_safe({"case_id": case_id, "decision": record})
                        stream.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        model = record.get("model") or {}
                        kind = classify_decision(model)
                        counts["decisions"] += 1
                        counts[names[kind]] += 1
                        if kind in ("live", "live_failed"):
                            self.live_total += 1         # failed attempts count toward --max-live-requests too
                        if kind == "live":
                            # Time inside the wrapped predict only; the decision's own latency_ms would
                            # include building the backend (the checkpoint load) on the first miss.
                            response = model.get("response") if isinstance(model.get("response"), dict) else {}
                            inside = (response.get("prediction_store") or {}).get("latency_ms")
                            if isinstance(inside, (int, float)):
                                live_latencies.append(float(inside))
                        if "error" in model and str(model["error"]).startswith("KeyboardInterrupt"):
                            # An interrupt that landed outside the backend call is still an interrupt: it is
                            # neither a failed estimate nor a backend that could not be built.
                            flags["interrupted"] = True
                        elif "error" in model:
                            counts["failed_estimates"] += 1
                            text = str(model["error"])
                            kind_of_error = text.split(":", 1)[0]
                            if kind_of_error not in errors_by_type:
                                first_errors.append({"case_id": case_id, "round": record.get("round"),
                                                     "request_sha256": model.get("request_sha256"),
                                                     "error": text[:2000]})
                                self.log(f"{stage} {config.name} {case_id}: estimate failed, nothing stored or "
                                         f"filled in: {text[:400]}")
                            errors_by_type[kind_of_error] = errors_by_type.get(kind_of_error, 0) + 1
                            if text.startswith("StoreMiss"):
                                counts["read_only_store_misses"] += 1
                            elif text.startswith("StoreIdentityMismatch"):
                                flags["identity_mismatch"] = True
                            elif text.startswith(("StoreWriteFailed", "OSError")):
                                flags["store_write_failed"] = True
                            elif uses_model and kind == "no_answer" and self.store_mode == "read-write" \
                                    and model.get("status") != "interrupted":
                                flags["initialization_failed"] = True      # the backend could not be built

                    used = TrueAcceptanceReference(case.hidden) if config.kind == "true_acceptance" else backend
                    row, scored = run_case(case, private, used, decide_fn, config_policy, True, on_decision=save)
                    rows.append(row)
                    predictions.extend(scored)
                    if uses_model:               # progress of the slow rows only; CPU rows log one line
                        self.log(f"{stage} {config.name} [{number}/{len(cases)}] {case.case_id} {row['status']}"
                                 + ("" if quiet or stage == "confirm" else f" saving={row['realized_saving']}"))
                    if row["interrupted"] or flags["interrupted"]:
                        status = "interrupted"
                    elif flags["store_write_failed"]:
                        status = "store_write_failed"
                    elif flags["identity_mismatch"]:
                        status = "store_identity_mismatch"
                    elif flags["initialization_failed"]:
                        status = "initialization_failed"
                    elif row["status"] == "needs_review":
                        # A failed estimate stops for a person: no later case may ask the same request again.
                        status, reason = "failed", "estimate_failed"
                    if status is not None:
                        break
        except KeyboardInterrupt:
            status = "interrupted"
        except TuneRefusal:
            result.update(status="runner_error", error="refused before the first case",
                          finished_at_utc=_utc_now())
            _write_json(folder / "result.json", result)
            raise
        except Exception as failure:                 # noqa: BLE001 - keep the attempt record, never a value
            status, error = "runner_error", f"{type(failure).__name__}: {failure}"

        if status is None:
            if any(row["status"] == "needs_review" for row in rows):
                status, reason = "failed", "estimate_failed"
            elif any(row["status"] not in TERMINAL for row in rows) or len(rows) != len(cases):
                status, reason = "failed", "case_not_terminal"
            else:
                status = "complete"
        if config.kind == "true_acceptance":
            predictions = []                         # its "estimates" are the answers; nothing to score
        summary = summarize(rows, predictions) if rows else None
        if summary is not None:
            process = summary["process"]
            process.pop("model_requests", None)      # a session count, not a count of live model calls
            process.pop("latency_ms_per_request", None)
            process["note"] = "live requests, store hits and their latency are under `counts`"
            if quiet:
                for key in ("mean_realized_saving", "mean_net_saving_after_process_costs",
                            "agreements_not_better_than_alternative", "feasible_cases", "infeasible_cases",
                            "recovery_after_rejected_counter"):
                    summary.pop(key, None)
                summary["note"] = "calibration split: map fitting only; savings are not summarized or compared"
        counts["live_seconds"] = round(sum(live_latencies) / 1000.0, 3)
        counts["live_latency_ms"] = ({"n": len(live_latencies), "median": _median(live_latencies),
                                      "max": max(live_latencies)} if live_latencies else None)
        if stored is not None and before is not None:
            counts["store_backend"] = {
                "live_requests": stored.live_requests - before[0], "stored_answers": stored.stored_answers - before[1],
                "store_hits": stored.store_hits - before[2], "read_only_misses": stored.store_misses - before[3],
                "live_seconds_inside_backend": round(stored.live_seconds - before[4], 3),
                # Building the backend (the checkpoint load) is reported apart from request latency.
                "backend_build_seconds": round(stored.factory_seconds - before[5], 3),
                "store_entries": len(stored.store), "observed": stored.store.observed,
            }
        try:
            finals = [record for row in rows for record in final_offer_records(row)]
        except RuntimeError as failure:
            finals, status, error = [], "runner_error", f"{type(failure).__name__}: {failure}"
        result.update(
            status=status, complete=status == "complete", reason=reason,
            evidence_class=self._evidence_class(config, counts), counts=counts,
            cases_planned=len(cases), cases_run=len(rows),
            needs_review_cases=[row["case_id"] for row in rows if row["status"] == "needs_review"],
            summary=summary, rows=[_kept_row(row) for row in rows],
            acceptance_predictions=predictions, final_offer_records=finals,
            wall_seconds=round(time.perf_counter() - wall_started, 3), finished_at_utc=_utc_now(),
        )
        if error is not None:
            result["error"] = error
        if first_errors:
            result["first_errors"] = first_errors            # the first error of each kind, verbatim
            result["errors_by_type"] = errors_by_type
        if backend is not None:
            result["backend"] = backend.describe()
        else:
            result["backend"] = {"kind": config.kind, "is_model": False, "estimates": config.estimate_sources(),
                                 "uses_hidden_information": config.kind == "true_acceptance"}
        if uses_model and stored is not None:
            result["store_identity"]["observed"] = stored.store.observed
        if after is not None:
            after(result, rows)
        _write_json(folder / "result.json", result)
        text = f"{stage} {config.id}: status={status}"
        if reason:
            text += f" reason={reason}"
        if first_errors and status != "complete":
            text += f" first_error={first_errors[0]['error'][:300]!r}"
        text += (f" cases={len(rows)}/{len(cases)} live={counts['live_requests']} store_hits={counts['store_hits']}"
                 f" evidence={result['evidence_class']}")
        if status == "complete" and not quiet:
            text += f" mean_realized_saving={summary['mean_realized_saving']:.0f}"
        self.log(text)
        return result

    # ---- stages --------------------------------------------------------------------------------------

    def _m0(self) -> Config:
        return next(state.config for state in resolve_arms(None) if state.arm.name == "M0")

    def stage_smoke(self, retry_failed: bool = False) -> str:
        """M0 on the first two D0 cases, twice. The second pass must not make a live call."""
        config, cases = self._m0(), self.cases("d0")[:2]
        block = block_info("d0", cases, subset="first 2 cases")
        attempts = self._attempts("smoke", config.id)
        if any((a["result"].get("smoke") or {}).get("verdict") == "passed" for a in attempts):
            self.log(f"smoke {config.id}: already passed; skipped")
            return "skipped_complete"
        if attempts and attempts[-1]["result"].get("status") == "failed" and not retry_failed:
            self.log(f"smoke {config.id}: an estimate failed earlier ({attempts[-1]['name']}); not rerun without "
                     "--retry-failed")
            return "smoke_failed"
        retry = ({"retry_of": attempts[-1]["name"], "retry_failed_flag": True}
                 if attempts and attempts[-1]["result"].get("status") == "failed" and retry_failed else {})
        first = self.run_config("smoke", block, cases, config, force=True,
                                labels={"smoke": {"pass": 1}, **retry})
        if first["status"] != "complete":
            return first["status"] if first["status"] in STOPPING else "smoke_failed"
        first_rows = {row["case_id"]: row for row in first["result"]["rows"]}

        def verdict(result: dict, rows: list) -> None:
            reasons = []
            if result["status"] != "complete":
                reasons.append(f"second pass ended {result['status']}")
            if result["counts"]["live_requests"]:
                reasons.append(f"second pass made {result['counts']['live_requests']} live requests; "
                               "every answer should have come from the store")
            for row in rows:
                earlier = first_rows.get(row["case_id"], {})
                if any(earlier.get(key) != row[key] for key in ("status", "agreement", "realized_saving")):
                    reasons.append(f"{row['case_id']} ended differently in the second pass")
            if rows and rows[0]["case_id"] == GOLDEN_REQUEST["case_id"] and rows[0]["internal_record"]:
                digest = rows[0]["internal_record"][0]["model"].get("request_sha256")
                if digest != GOLDEN_REQUEST["sha256"]:
                    reasons.append("the first request does not hash to the recorded golden value")
            result["smoke"] = {"pass": 2, "compared_with": first["result"]["attempt"],
                               "verdict": "failed" if reasons else "passed", "reasons": reasons}

        second = self.run_config("smoke", block, cases, config, force=True, labels={"smoke": {"pass": 2}},
                                 after=verdict)
        if second["status"] in STOPPING:
            return second["status"]
        smoke = (second["result"] or {}).get("smoke", {})
        if smoke.get("verdict") != "passed":
            self.log("smoke FAILED: " + "; ".join(smoke.get("reasons", ["second pass did not run"])))
            return "smoke_failed"
        self.log("smoke passed: the second pass made zero live requests and ended identically")
        return "complete"

    def _regression(self, result: dict, rows: list) -> None:
        """Compare with the recorded real-model run on the same seen cases. A difference is drift, not error."""
        path = REPOSITORY_ROOT / REGRESSION_REFERENCE
        block: dict[str, Any] = {"reference_file": REGRESSION_REFERENCE}
        if not path.is_file():
            block.update(available=False, note="the reference file is not present in this checkout")
        else:
            document = _read_json(path)
            reference = {c["case_id"]: c["agent"] for c in document["case_outcomes"]}
            differing = []
            compared = 0
            for row in rows:
                recorded = reference.get(row["case_id"])
                if recorded is None:
                    continue
                compared += 1
                if recorded["status"] != row["status"] or recorded["realized_saving"] != row["realized_saving"]:
                    differing.append({"case_id": row["case_id"],
                                      "recorded": {k: recorded[k] for k in ("status", "realized_saving")},
                                      "this_run": {k: row[k] for k in ("status", "realized_saving")}})
            described = document.get("backend", {})
            block.update(
                available=True, reference_sha256=file_sha256(path),
                reference_backend={k: described.get(k) for k in ("kind", "dtype", "gpu_name", "decision_temperature")},
                cases_compared=compared, cases_differing=len(differing), differing=differing,
                meaningful=self.real_model and result["status"] == "complete",
                note=("a difference is reported as environment drift against the 2026-11-16 T4 FP16 native run; "
                      "D0 never enters selection" if self.real_model else
                      "CPU stand-in backend: differences from the recorded real-model run are expected and say "
                      "nothing about environment drift; D0 never enters selection"))
        result["regression"] = block

    def stage_regress(self, retry_failed: bool) -> str:
        outcome = self.run_config("regress", self.protocol["blocks"]["d0"], self.cases("d0"), self._m0(),
                                  retry_failed=retry_failed, after=self._regression)
        regression = (outcome["result"] or {}).get("regression")
        if regression and regression.get("available"):
            self.log(f"regress: {regression['cases_differing']} of {regression['cases_compared']} cases differ from "
                     f"the recorded run ({'environment drift check' if regression['meaningful'] else 'not meaningful'}"
                     f": {regression['note']})")
        return self._stage_status([outcome["status"]])

    def stage_calibrate(self, retry_failed: bool) -> str:
        outcome = self.run_config("calibrate", self.protocol["blocks"]["cal"], self.cases("cal"), self._m0(),
                                  retry_failed=retry_failed)
        return self._stage_status([outcome["status"]])

    @staticmethod
    def _stage_status(statuses: list[str]) -> str:
        for status in statuses:
            if status in STOPPING:
                return status
        if "failed" in statuses:
            return "failed"                 # an estimate failed now: stop for a person
        if "skipped_failed" in statuses:
            return "failed_earlier"         # failed in an earlier run and not retried: left as it is
        return "complete"

    def stage_select(self, retry_failed: bool) -> str:
        block, cases, statuses = self.protocol["blocks"]["select"], self.cases("select"), []
        pending = False
        for arm_state in resolve_arms(self.load_maps()):
            if arm_state.status != "resolved":
                pending = pending or arm_state.status == "pending"
                self.log(f"select {arm_state.config_id}: {arm_state.status} ({arm_state.reason}); not run")
                continue
            statuses.append(self.run_config("select", block, cases, arm_state.config,
                                            retry_failed=retry_failed)["status"])
            if statuses[-1] in STOPPING or statuses[-1] == "failed":
                return statuses[-1]         # no later candidate asks the failed request again by itself
        for config in reference_configs():
            statuses.append(self.run_config("select", block, cases, config, retry_failed=retry_failed)["status"])
            if statuses[-1] in STOPPING or statuses[-1] == "failed":
                return statuses[-1]
        status = self._stage_status(statuses)
        return "pending_maps" if status == "complete" and pending else status

    def stage_confirm(self, retry_failed: bool, rerun_not_confirmatory: bool) -> str:
        selection = self.load_selection()
        if selection is None:
            raise TuneRefusal("confirm needs selection.json: finish SELECT for every candidate and run `board`")
        block, cases = self.protocol["blocks"]["confirm"], self.cases("confirm")
        selected = Config(selection["selected"]["name"], "candidate", "estimates",
                          "the configuration selected on SELECT",
                          EstimateSpec.from_json(selection["selected"]["estimates"]))
        configs = self.confirm_configs(selected)
        current = {"protocol_sha256": self.protocol_sha256, "block_content_sha256": block["content_sha256"],
                   "selection_sha256": selection["selection_sha256"], "store_identity12": self.identity["identity12"]}
        source = _digest(input_hashes())
        marker = self.state / "confirm_first_use.json"
        if not rerun_not_confirmatory and selection.get("input_sha256_digest") not in (None, source):
            raise TuneRefusal("the hashed source files changed since the selection was made, so the procedure "
                              "that would be confirmed is not the one that was selected; a confirmatory use is "
                              "refused (--rerun-not-confirmatory after a first use, or a new --state-dir)")
        if not marker.exists():
            if rerun_not_confirmatory:
                raise TuneRefusal("CONFIRM has not been used yet; run confirm without --rerun-not-confirmatory")
            if self.store_mode != "read-write":
                raise TuneRefusal("a first confirm use needs live answers; --store-mode read-only is refused "
                                  "before CONFIRM counts as used")
            self._preflight_runtime()
            # Written before the first confirm call: from here on CONFIRM counts as used.
            _write_json(marker, {
                "schema_version": 1, **current, "block": block, "input_sha256_digest": source,
                "selected": {"name": selected.name, "id": selected.id},
                "first_use_started_at_utc": _utc_now(),
                "notice": "First use of the CONFIRM block in this state directory. A later confirm run is not "
                          "confirmatory. This does not prove the block was unseen elsewhere."}, exclusive=True)
            use = "first_use"
            self.log(f"confirm: first use of CONFIRM recorded in {marker.name}")
        else:
            first = _read_json(marker)
            if any(first.get(key) != value for key, value in current.items()):
                raise TuneRefusal("confirm_first_use.json belongs to another selection, block or store identity")
            finished = all(self.state_of("confirm", c.id, "first_use")["status"] == "complete" for c in configs)
            if rerun_not_confirmatory:
                use = "not_confirmatory"
                self.log("confirm: rerun labelled not_confirmatory; it does not change the verdict")
            elif finished:
                raise TuneRefusal(f"CONFIRM was already used (first use {first.get('first_use_started_at_utc')}); a "
                                  "second confirm is refused. --rerun-not-confirmatory runs it again, labelled "
                                  "not_confirmatory")
            elif first.get("input_sha256_digest") != source:
                raise TuneRefusal("the source files changed since the first confirm use, so the unfinished first "
                                  "use cannot be continued as confirmatory; --rerun-not-confirmatory labels a new "
                                  "run not_confirmatory")
            else:
                self._check_first_use_answers(configs)
                use = "first_use"
                self.log("confirm: continuing the unfinished first use (same selection, same source files, "
                         "recorded answers still in the store)")
        statuses = []
        for config in configs:
            outcome = self.run_config("confirm", block, cases, config, retry_failed=retry_failed, use=use,
                                      force=use == "not_confirmatory", labels={"confirm_use": use})
            statuses.append(outcome["status"])
            if statuses[-1] in STOPPING or statuses[-1] == "failed":
                return statuses[-1]
        return self._stage_status(statuses)

    def _check_first_use_answers(self, configs: list[Config]) -> None:
        """Every answer the earlier attempts of this first use recorded must still be in the store, unchanged.

        Otherwise answers could be removed after partial CONFIRM results were seen and asked again,
        and the result would still be called confirmatory.
        """
        store = None
        for config in configs:
            if not config.uses_model():
                continue
            for attempt in self._attempts("confirm", config.id):
                if attempt["result"].get("confirm_use") != "first_use":
                    continue
                path = attempt["path"] / "decisions.jsonl"
                if not path.is_file():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        model = json.loads(line)["decision"].get("model") or {}
                    except (ValueError, KeyError, AttributeError):
                        continue                 # a line cut off by a killed process records no answer
                    raw = ((model.get("response") or {}).get("composition") or {}).get("raw_model_predictions")
                    digest = model.get("request_sha256")
                    if not digest or not isinstance(raw, dict) or "error" in model:
                        continue
                    if store is None:
                        store = PredictionStore(self.state / "store", self.identity["requested"], mode="read-only")
                    kept = store.get(digest)
                    if kept is None or kept["predictions"] != raw:
                        raise TuneRefusal(
                            "an answer recorded by the unfinished first confirm use is no longer in the store "
                            f"with the same values (request {digest[:12]}); the first use cannot be continued "
                            "as confirmatory. --rerun-not-confirmatory labels a new run not_confirmatory")

    def _preflight_runtime(self) -> None:
        """Real model only: one live answer that is not stored must come from the stored runtime.

        Without it, a state restored on another GPU would record its first confirm use, load the
        model, and only then stop on the identity mismatch, with CONFIRM spent for nothing.
        """
        if not self.real_model:
            return
        case = make_cases("dev", 1)[0]
        state = initial_state(case.case_id, case.hidden)
        candidates = policy.generate_candidates(state, default_buyer_private(), PolicyConfig())
        request = build_decision_request(state, candidates)
        try:
            observed = self.stored_backend().check_runtime(state, candidates, request)
        except KeyboardInterrupt:
            raise
        except Exception as error:                   # noqa: BLE001 - CONFIRM must stay unused
            raise TuneRefusal("the runtime check before the first confirm use failed, so CONFIRM was NOT used: "
                              f"{type(error).__name__}: {error}") from error
        self.live_total += 1
        self.log(f"confirm: runtime check passed ({observed}); its answer was not stored")

    def confirm_configs(self, selected: Config) -> list[Config]:
        """M0, the selected configuration, its model-free twin, then the other references."""
        m0, twin = self._m0(), twin_reference(selected.spec)
        configs = [m0] + ([selected] if selected.id != m0.id else []) + [twin]
        return configs + [c for c in reference_configs() if c.id != twin.id]

    # ---- maps ----------------------------------------------------------------------------------------

    def maps_path(self) -> Path | None:
        identity = self.load_identity()
        return self.state / "maps" / f"calibration-{identity['identity12']}.json" if identity else None

    def load_maps(self) -> dict | None:
        path = self.maps_path()
        if path is None or not path.exists():
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        if self._maps_cache is not None and self._maps_cache[0] == key:
            return self._maps_cache[1]
        maps = _read_json(path)
        if maps.get("protocol_sha256") != self.protocol_sha256 or maps.get("fitted_on_split") != "calibration" \
                or maps.get("store_identity12") != self.identity["identity12"] \
                or (maps.get("platt") is not None and maps["platt"].get("fitted_on_split") != "calibration"):
            raise TuneRefusal(f"{path} does not belong to this protocol, store identity or the calibration split")
        self._verify_maps(path, maps)
        self._maps_cache = (key, maps)
        return maps

    def _verify_maps(self, path: Path, maps: dict) -> None:
        """The file's own labels are not trusted: its fits must follow from the calibrate result here."""
        calibrate = self.state_of("calibrate", self._m0().id)
        if maps.get("declared_unavailable") is not None:
            if maps.get("platt") is not None or (maps.get("final_offer_shrink") or {}).get("weight") is not None:
                raise TuneRefusal(f"{path} declares the maps unavailable but carries a fit")
            if calibrate["status"] == "complete":
                raise TuneRefusal(f"{path} declares the maps unavailable although a complete calibrate result exists")
            return
        attempt = calibrate["attempt"] if calibrate["status"] == "complete" else None
        source = maps.get("source") or {}
        if attempt is None:
            raise TuneRefusal(f"{path} has no complete calibrate result of M0 behind it in this state directory")
        result_file = attempt["path"] / "result.json"
        if source.get("result_sha256") != file_sha256(result_file):
            raise TuneRefusal(f"{path} was not fitted from the calibrate result in this state directory")
        try:
            refit = json.loads(json.dumps(fit_maps(attempt["result"], source_sha256=source["result_sha256"])))
        except ValueError as error:
            raise TuneRefusal(f"{path}: the calibrate result cannot be refitted: {error}") from error
        if any(maps.get(part) != refit.get(part) for part in ("platt", "final_offer_shrink")):
            raise TuneRefusal(f"{path} does not equal the fit of the calibrate result; a hand-written or edited "
                              "maps file is refused")

    def declare_maps_unavailable(self, reason: str) -> tuple[Path, dict]:
        """A person's recorded decision that no map can be fitted. Only after a retried, failed calibrate run."""
        if not reason or not reason.strip():
            raise TuneRefusal("--declare-unavailable needs --reason TEXT (what failed and why it cannot be fixed)")
        if self.load_identity() is None:
            raise TuneRefusal("nothing has been run in this state directory yet")
        path = self.maps_path()
        if path.exists():
            raise TuneRefusal(f"{path.name} already exists; the maps are decided once per state directory")
        state = self.state_of("calibrate", self._m0().id)
        retried = [a for a in self._attempts("calibrate", self._m0().id) if a["result"].get("retry_failed_flag")]
        if state["status"] != "failed" or not retried:
            raise TuneRefusal("the maps can be declared unavailable only after the calibrate run failed an "
                              "estimate, was run again with --retry-failed and failed again; its state is "
                              f"{state['status']} with {len(retried)} retried attempt(s)")
        last = state["attempt"]["result"]
        maps = {
            "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
            "created_at_utc": _utc_now(), "store_identity12": self.identity["identity12"],
            "fitted_on_split": "calibration", "source": None, "platt": None,
            "platt_unavailable_reason": "declared_unavailable", "temperature_equivalent": None,
            "final_offer_shrink": {"weight": None, "unavailable_reason": "declared_unavailable"},
            "declared_unavailable": {
                "reason": reason.strip(), "decided_by": "operator_command",
                "calibrate_attempts": [a["name"] for a in self._attempts("calibrate", self._m0().id)],
                "last_attempt_status": last.get("status"), "last_attempt_reason": last.get("reason"),
                "first_errors": last.get("first_errors")},
            "notice": "No map was fitted. The arms that need one are failed/map_unavailable; M0, M1 and M2 remain.",
        }
        maps["arms"] = {s.arm.name: {"status": s.status, "reason": s.reason, "config_id": s.config_id,
                                     "estimates": s.config.spec.to_json() if s.config else None}
                        for s in resolve_arms(maps) if s.arm.needs}
        try:
            _write_json(path, maps, exclusive=True)
        except FileExistsError as error:
            raise TuneRefusal(f"{path.name} already exists") from error
        return path, maps

    def fit_map(self) -> tuple[Path, dict, bool]:
        """Fit once from the complete calibrate result of M0. Returns (path, maps, written_now)."""
        if self.load_identity() is None:
            raise TuneRefusal("fit-map needs the complete calibrate result of M0; nothing has been run here yet")
        path = self.maps_path()
        existing = self.load_maps()
        if existing is not None:
            return path, existing, False
        state = self.state_of("calibrate", self._m0().id)
        if state["status"] != "complete":
            raise TuneRefusal("fit-map accepts only the complete calibrate result of M0; its state is "
                              f"{state['status']}" + (f" ({state.get('reason') or state.get('detail')})"
                                                      if state["status"] != "not_run" else ""))
        attempt = state["attempt"]
        result_file = attempt["path"] / "result.json"
        result = attempt["result"]
        if (result.get("store_identity") or {}).get("identity12") != self.identity["identity12"] \
                or result.get("protocol_sha256") != self.protocol_sha256:
            raise TuneRefusal("the calibrate result belongs to another protocol or store identity")
        try:
            maps = fit_maps(result, source_sha256=file_sha256(result_file))
        except ValueError as error:
            raise TuneRefusal(str(error)) from error
        maps = {
            "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
            "created_at_utc": _utc_now(), "store_identity12": self.identity["identity12"],
            "source": {"result_file": result_file.relative_to(self.state).as_posix(),
                       "result_sha256": file_sha256(result_file), "config_id": result["config"]["id"],
                       "evidence_class": result["evidence_class"], "block": result["block"],
                       "purpose": result["purpose"]},
            **maps,
            "notice": "Fitted on the calibration split only, from the default configuration M0. A map is a usage "
                      "setting; it is not evidence that the model's estimates are calibrated on other states.",
        }
        maps["arms"] = {s.arm.name: {"status": s.status, "reason": s.reason, "config_id": s.config_id,
                                     "estimates": s.config.spec.to_json() if s.config else None}
                        for s in resolve_arms(maps) if s.arm.needs}
        try:
            _write_json(path, maps, exclusive=True)          # one fit per state directory
        except FileExistsError:
            return path, self.load_maps(), False
        return path, maps, True

    # ---- leaderboard, selection, verdict -------------------------------------------------------------

    def load_selection(self) -> dict | None:
        path = self.state / "selection.json"
        if not path.exists():
            return None
        selection = _read_json(path)
        digest = _digest({k: v for k, v in selection.items() if k not in ("selection_sha256", "created_at_utc")})
        selected = selection.get("selected") or {}
        try:
            spec = EstimateSpec.from_json(selected.get("estimates"))
            spec.validate()
            identifier = Config(selected.get("name"), "candidate", "estimates", "", spec).id
        except (TypeError, ValueError) as error:
            raise TuneRefusal(f"selection.json is unreadable: {error}") from error
        if selection.get("selection_sha256") != digest or selection.get("protocol_sha256") != self.protocol_sha256 \
                or identifier != selected.get("id") or selected.get("name") not in {a.name for a in CANDIDATES}:
            raise TuneRefusal("selection.json does not match its own digest, this protocol or a registered candidate")
        registered = next((s for s in resolve_arms(self.load_maps()) if s.arm.name == selected.get("name")), None)
        if registered is None or registered.config is None or registered.status != "resolved" \
                or registered.config.id != identifier or selected.get("policy") != policy_json():
            raise TuneRefusal("selection.json names a configuration that is not the registered candidate as "
                              "resolved from the maps of this state directory; an edited selection is refused")
        return selection

    def _row(self, stage: str, config: Config, use: str | None = None) -> tuple[dict, list | None]:
        state = self.state_of(stage, config.id, use)
        row: dict[str, Any] = {
            "id": config.id, "name": config.name, "role": config.role, "description": config.description,
            "status": state["status"], "reason": state.get("reason") or state.get("detail"),
            "estimate_sources": config.estimate_sources(), "estimate_sources_display": config.display_sources(),
            "evidence_class": "cpu_reference_not_model_evidence" if not config.uses_model() else None,
            "selectable": False, "attempts": state["attempts"],
        }
        for key in ("cases", "mean_realized_saving", "mean_net_saving", "feasible_agreements", "feasible_cases",
                    "supplier_walk_aways", "approval_cards", "authority_violations", "needs_review",
                    "live_requests", "store_hits", "session_cache_hits", "live_seconds", "attempt",
                    "result_sha256", "read_only_store_misses"):
            row[key] = None
        attempt = state["attempt"]
        if attempt is None:
            return row, None
        result = attempt["result"]
        counts, summary, rows = result.get("counts") or {}, result.get("summary") or {}, result.get("rows") or []
        row.update(
            evidence_class=result.get("evidence_class", row["evidence_class"]),
            attempt=attempt["path"].relative_to(self.state).as_posix(),
            result_sha256=file_sha256(attempt["path"] / "result.json"),
            cases=result.get("cases_run"), needs_review=summary.get("needs_review"),
            authority_violations=summary.get("authority_violations"),
            live_requests=counts.get("live_requests"), store_hits=counts.get("store_hits"),
            session_cache_hits=counts.get("session_cache_hits"), live_seconds=counts.get("live_seconds"),
            read_only_store_misses=counts.get("read_only_store_misses"),
            input_sha256_digest=_digest((result.get("provenance") or {}).get("input_sha256") or {}),
        )
        if state["status"] != "complete":
            return row, None                 # no savings for a run that is not complete
        feasible = [r for r in rows if r["feasible"]]
        row.update(
            mean_realized_saving=summary["mean_realized_saving"],
            mean_net_saving=summary["mean_net_saving_after_process_costs"],
            feasible_cases=len(feasible),
            feasible_agreements=sum(1 for r in feasible if r["status"] == "agreed"),
            supplier_walk_aways=sum(1 for r in rows if r["status"] == "supplier_walked"),
            approval_cards=sum(r["approval_cards"] for r in rows),
        )
        return row, rows

    @staticmethod
    def difference(first: list | None, second: list | None, label: str) -> dict:
        """evaluation.paired_difference, only for two complete runs over the same cases in the same order."""
        if first is None or second is None:
            return {"comparison": label, "realized_saving": None, "net_saving": None,
                    "reason": "one of the two runs is not complete"}
        if [r["case_id"] for r in first] != [r["case_id"] for r in second]:
            return {"comparison": label, "realized_saving": None, "net_saving": None,
                    "reason": "the two runs do not cover the same cases"}
        assert BOOTSTRAP["level"] == 0.90        # paired_difference reports the 90% interval as `ci90`
        return {"comparison": label, "reason": None,
                "realized_saving": paired_difference(first, second, "realized_saving", seed=BOOTSTRAP["seed"],
                                                     resamples=BOOTSTRAP["resamples"]),
                "net_saving": paired_difference(first, second, "net_saving", seed=BOOTSTRAP["seed"],
                                                resamples=BOOTSTRAP["resamples"])}

    def _table(self, stage: str, entries: list[tuple[dict, list | None]]) -> dict:
        rows = [row for row, _ in entries]
        cases_of = {row["id"]: cases for row, cases in entries}
        m0 = next((row["id"] for row in rows if row["name"] == "M0"), None)
        rule = next((row["id"] for row in rows if row["name"] == "rule_policy"), None)
        differences = {}
        for row in rows:
            entry = {}
            for label, other in (("minus_M0", m0), ("minus_rule_policy", rule)):
                if other is None or other == row["id"]:
                    entry[label] = {"comparison": label, "realized_saving": None, "net_saving": None,
                                    "reason": "the same configuration" if other else "the other run does not exist"}
                else:
                    entry[label] = self.difference(cases_of[row["id"]], cases_of[other], label)
            differences[row["id"]] = entry
        return {"block": self.protocol["blocks"][stage], "rows": rows, "differences": differences}

    def board(self, write_selection: bool = True) -> dict:
        """Write leaderboard.json / leaderboard.md, and selection.json once every candidate is terminal.

        `run --stage all` passes write_selection=False: a failed estimate stops for a person, so when
        a candidate failed one, the selection is written only by an explicit `board` (accepting the
        failure) or after `--retry-failed`. Without a failed candidate the flag changes nothing.
        """
        identity = self.load_identity()
        maps = self.load_maps()
        arm_states = resolve_arms(maps)

        # SELECT: candidates in their fixed order, then the CPU references.
        entries, blocking = [], []
        for arm_state in arm_states:
            if arm_state.config is None or arm_state.status != "resolved":
                config = arm_state.config or Config(arm_state.arm.name, "candidate", "estimates",
                                                    arm_state.arm.description, arm_state.arm.template)
                row, _ = self._row("select", config)
                row.update(id=arm_state.config_id, status=arm_state.status, reason=arm_state.reason, attempts=0,
                           evidence_class=None)
                entries.append((row, None))
                if not arm_state.terminal_without_run:
                    blocking.append({"candidate": arm_state.arm.name, "state": arm_state.status,
                                     "reason": arm_state.reason})
                continue
            row, cases = self._row("select", arm_state.config)
            read_only_miss = row["status"] == "failed" and (row["read_only_store_misses"] or 0) > 0
            if row["status"] in ("not_run", "incomplete") or read_only_miss:
                blocking.append({"candidate": row["name"], "state": row["status"],
                                 "reason": "read_only_store_miss" if read_only_miss else row["reason"]})
            row["selectable"] = (row["status"] == "complete" and row["authority_violations"] == 0
                                 and row["needs_review"] == 0)
            entries.append((row, cases))
        candidate_rows = [row for row, _ in entries]
        versions = {row.get("input_sha256_digest") for row in candidate_rows if row["status"] == "complete"}
        if len(versions) > 1:
            blocking.append({"candidate": "*", "state": "mixed_source_versions",
                             "reason": "the complete candidates ran under different hashed source files; one "
                                       "selection needs one procedure. Use a new --state-dir"})
        entries += [self._row("select", config) for config in reference_configs()]
        select_table = self._table("select", entries)
        select_table["candidates_terminal"] = not blocking
        select_table["blocking"] = blocking

        selection = self.load_selection()
        selection_view: dict[str, Any]
        held_back = [{"candidate": row["name"], "state": row["status"], "reason": row["reason"]}
                     for row in candidate_rows if row["status"] == "failed" and row["reason"] != "map_unavailable"]
        waiting_for_review = selection is None and not blocking and not write_selection and bool(held_back)
        if selection is None and not blocking and not waiting_for_review:
            best = choose_candidate(candidate_rows)
            if best is not None:
                selection = self._write_selection(best, candidate_rows, arm_states, maps)
        if selection is not None:
            self.write_selected_config(selection)       # exclusive create; a no-op when it is already there
            selection_view = {"status": "selected", **selection}
        elif blocking:
            selection_view = {"status": "blocked", "reason": "not every candidate is terminal on SELECT",
                              "blocking": blocking}
        elif waiting_for_review:
            selection_view = {"status": "blocked", "blocking": held_back,
                              "reason": "a candidate failed an estimate and waits for a person: retry it with `run "
                                        "--stage select --retry-failed`, or run `board` to accept the failure and "
                                        "select among the other candidates"}
        else:
            selection_view = {"status": "blocked", "reason": "no candidate is eligible (complete, zero authority "
                                                             "violations, zero needs_review)", "blocking": []}

        board: dict[str, Any] = {
            "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
            "created_at_utc": _utc_now(), "rehearsal_blocks": self.rehearsal,
            "backend_choice": identity.get("backend_choice") if identity else None,
            "real_model_backend": identity.get("real_model_backend") if identity else None,
            "store_identity12": identity.get("identity12") if identity else None,
            # A CPU stand-in in the model's place, or shortened blocks, is a rehearsal of the control flow.
            # ... and a real backend choice that never produced a stored answer is no evidence either.
            "counts_as_model_evidence": (bool(identity and identity.get("real_model_backend"))
                                         and not self.rehearsal and self._store_has_answers()),
            "scope": "synthetic rule-based suppliers, development cases; not a held-out result and not evidence "
                     "about real negotiations",
            "known_exposure": KNOWN_EXPOSURE, "bootstrap": dict(BOOTSTRAP),
            "blocks": {"select": select_table}, "selection": selection_view, "confirm": None,
            "other_stages": self._other_stages(maps),
        }
        if selection is not None and (self.state / "confirm_first_use.json").exists():
            board["blocks"]["confirm"], board["confirm"] = self._confirm_board(selection)
            board["confirm"]["counts_as_model_evidence"] = board["counts_as_model_evidence"]
        snapshot = self.state / "boards" / (
            f"board-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{secrets.token_hex(2)}.json")
        _write_json(snapshot, board, exclusive=True)         # every board ever written is kept
        _write_json(self.state / "leaderboard.json", board)  # the latest view, regenerated from the attempts
        temporary = self.state / f"leaderboard.md.tmp-{os.getpid()}"
        temporary.write_text(render_board(board), encoding="utf-8")
        os.replace(temporary, self.state / "leaderboard.md")
        self.last_board = board
        return board

    def _store_has_answers(self) -> bool:
        identity = self.load_identity()
        if not identity:
            return False
        path = self.state / "store" / f"{identity['identity12']}.jsonl"
        return path.is_file() and path.stat().st_size > 0

    def _other_stages(self, maps: dict | None) -> dict:
        m0 = self._m0()
        out: dict[str, Any] = {}
        attempts = self._attempts("smoke", m0.id)
        passed = any((a["result"].get("smoke") or {}).get("verdict") == "passed" for a in attempts)
        out["smoke"] = {"status": "passed" if passed else ("not_passed" if attempts else "not_run"),
                        "attempts": len(attempts)}
        regress = self.state_of("regress", m0.id)
        regression = (regress["attempt"]["result"].get("regression") if regress["attempt"] else None) or {}
        out["regress"] = {"status": regress["status"], "purpose": STAGE_PURPOSE["regress"],
                          "cases_compared": regression.get("cases_compared"),
                          "cases_differing": regression.get("cases_differing"),
                          "meaningful": regression.get("meaningful"), "note": regression.get("note")}
        calibrate = self.state_of("calibrate", m0.id)
        out["calibrate"] = {"status": calibrate["status"], "purpose": STAGE_PURPOSE["calibrate"],
                            "note": "never on the leaderboard; used only by fit-map"}
        out["maps"] = None if maps is None else {
            "file": self.maps_path().relative_to(self.state).as_posix(),
            "platt_available": maps.get("platt") is not None,
            "final_offer_shrink_weight": (maps.get("final_offer_shrink") or {}).get("weight"),
            "arms": {name: arm["status"] for name, arm in (maps.get("arms") or {}).items()}}
        return out

    def _write_selection(self, best: dict, candidate_rows: list[dict], arm_states: list[ArmState],
                         maps: dict | None) -> dict:
        chosen = next(s.config for s in arm_states if s.config is not None and s.config.id == best["id"])
        maps_path = self.maps_path()
        selection = {
            "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": self.protocol_sha256,
            "rehearsal_blocks": self.rehearsal, "block": self.protocol["blocks"]["select"], "rule": SELECTION_RULE,
            "store_identity12": self.identity["identity12"], "backend_choice": self.identity["backend_choice"],
            "real_model_backend": self.identity["real_model_backend"],
            "maps_sha256": file_sha256(maps_path) if maps is not None else None,
            "input_sha256_digest": next((row.get("input_sha256_digest") for row in candidate_rows
                                         if row["status"] == "complete"), None),
            "candidates": [{"name": row["name"], "id": row["id"], "status": row["status"], "reason": row["reason"],
                            "eligible": row["selectable"], "mean_realized_saving": row["mean_realized_saving"],
                            "evidence_class": row["evidence_class"], "result_sha256": row["result_sha256"]}
                           for row in candidate_rows],
            "selected": {"name": chosen.name, "id": chosen.id, "policy": policy_json(),
                         "estimates": chosen.spec.to_json(), "is_default": chosen.name == "M0",
                         "mean_realized_saving": best["mean_realized_saving"],
                         "evidence_class": best["evidence_class"]},
            "notice": "Selected on SELECT development cases; an in-sample choice. A claim needs the CONFIRM "
                      "block. Later attempts do not change this file.",
        }
        selection = {**selection, "selection_sha256": _digest(selection), "created_at_utc": _utc_now()}
        try:
            _write_json(self.state / "selection.json", selection, exclusive=True)
        except FileExistsError:
            selection = self.load_selection()
        self.write_selected_config(selection)
        return selection

    def write_selected_config(self, selection: dict) -> None:
        """The file a person later passes to `run_evaluation.py freeze --config-file` on the local PC."""
        selected = selection["selected"]
        document = {
            "policy": selected["policy"], "estimates": selected["estimates"],
            "selected_from": {
                "protocol_version": PROTOCOL_VERSION, "protocol_sha256": selection["protocol_sha256"],
                "selection_sha256": selection["selection_sha256"], "config_id": selected["id"],
                "config_name": selected["name"], "store_identity12": selection["store_identity12"],
                "backend_choice": selection["backend_choice"], "real_model_backend": selection["real_model_backend"],
                "evidence_class": selected["evidence_class"], "rehearsal_blocks": selection["rehearsal_blocks"],
                "counts_as_model_evidence": bool(selection["real_model_backend"]) and not selection["rehearsal_blocks"],
                "selected_on": f"{selection['block']['first_case_id']}..{selection['block']['last_case_id']}",
                "notice": "A usage configuration chosen on development cases. Writing this file ran no held-out "
                          "case. A selection from a CPU stand-in or from rehearsal blocks is not model evidence.",
            },
        }
        try:
            _write_json(self.state / "selected_config.json", document, exclusive=True)
        except FileExistsError:
            pass

    def _confirm_board(self, selection: dict) -> tuple[dict, dict]:
        selected = Config(selection["selected"]["name"], "candidate", "estimates",
                          "the configuration selected on SELECT",
                          EstimateSpec.from_json(selection["selected"]["estimates"]))
        configs = self.confirm_configs(selected)
        entries = [self._row("confirm", config, "first_use") for config in configs]
        table = self._table("confirm", entries)
        table["first_use"] = _read_json(self.state / "confirm_first_use.json")
        table["not_confirmatory_reruns"] = sum(
            1 for config in configs for a in self._attempts("confirm", config.id)
            if a["result"].get("confirm_use") == "not_confirmatory")
        by_id = {row["id"]: (row, cases) for row, cases in entries}
        m0, twin = self._m0(), twin_reference(selected.spec)
        rule = next(c for c in reference_configs() if c.name == "rule_policy")
        s_row, s_cases = by_id[selected.id]
        default_kept = selected.id == m0.id
        primary = (self.difference(s_cases, by_id[m0.id][1], "S_minus_M0") if not default_kept else
                   {"comparison": "S_minus_M0", "realized_saving": None, "net_saving": None,
                    "reason": "the selected configuration is the default M0"})
        verdict, why = primary_verdict(primary, default_kept, s_cases is not None and by_id[m0.id][1] is not None,
                                       s_row["authority_violations"])
        versus_twin = self.difference(s_cases, by_id[twin.id][1], "S_minus_twin")
        if versus_twin["realized_saving"] is None:
            contribution = "incomplete"
        else:
            contribution = ("model_contribution_shown" if versus_twin["realized_saving"]["ci90"][0] > 0
                            else "model_contribution_not_shown")
        versus_rule = self.difference(s_cases, by_id[rule.id][1], "S_minus_rule_policy")
        return table, {
            "selected": {"name": selected.name, "id": selected.id}, "default": {"name": "M0", "id": m0.id},
            "twin": {"name": twin.name, "id": twin.id},
            "verdict": verdict, "reason": why, "sentence": RESULT_SENTENCES[verdict], "primary": primary,
            "secondary": {
                "label": "pre-declared secondary comparisons; they do not change the verdict",
                "model_contribution": {"verdict": contribution, **versus_twin,
                                       "sentence": RESULT_SENTENCES.get(contribution)},
                "rule_policy": {"label": "not_confirmatory_reference_seen_in_design", **versus_rule},
            },
            "evidence_class_of_selected": s_row["evidence_class"],
            "first_use_started_at_utc": table["first_use"].get("first_use_started_at_utc"),
        }

    # ---- one invocation ------------------------------------------------------------------------------

    def _take_lock(self) -> None:
        """One runner per state directory: two would ask every request twice and could corrupt the store."""
        path = self.state / LOCK_NAME
        me = {"pid": os.getpid(), "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get(
            "COMPUTERNAME", ""), "started_at_utc": _utc_now()}
        for _ in range(2):
            try:
                _write_json(path, me, exclusive=True)
                self._lock = path
                return
            except FileExistsError:
                try:
                    other = _read_json(path)
                except (OSError, ValueError):
                    other = {}
                stale = False
                if os.name == "posix" and other.get("host") == me["host"] and isinstance(other.get("pid"), int):
                    try:
                        os.kill(other["pid"], 0)         # POSIX only: signal 0 checks, it does not kill
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        stale = False
                if not stale:
                    raise TuneRefusal(
                        f"another run is using this state directory (pid {other.get('pid')} on "
                        f"{other.get('host')}, started {other.get('started_at_utc')}). If that process is gone, "
                        f"delete {path} and run again")
                path.unlink(missing_ok=True)             # left behind by a killed process on this machine
        raise TuneRefusal(f"could not take {path}")

    def release_lock(self) -> None:
        if self._lock is not None:
            self._lock.unlink(missing_ok=True)
            self._lock = None

    def run(self, stage: str, retry_failed: bool = False, rerun_not_confirmatory: bool = False) -> str:
        """Run one stage, or `all` = smoke -> regress -> calibrate -> fit-map -> select -> board."""
        self.check_run_arguments()
        self.ensure_protocol(create=True, rehearsal=getattr(self.args, "rehearsal", None))
        check_golden_request()
        self._take_lock()
        self.bind_identity()
        if rerun_not_confirmatory and stage != "confirm":
            raise TuneRefusal("--rerun-not-confirmatory applies to --stage confirm only")
        steps = ALL_STEPS if stage == "all" else (stage,)
        outcome, failed = "complete", False
        for step in steps:
            if step == "smoke":
                outcome = self.stage_smoke(retry_failed)
            elif step == "regress":
                outcome = self.stage_regress(retry_failed)
            elif step == "calibrate":
                outcome = self.stage_calibrate(retry_failed)
            elif step == "fit-map":
                if self.state_of("calibrate", self._m0().id)["status"] != "complete" and self.load_maps() is None:
                    self.log("all: calibrate is not complete, so no map can be fitted; stopping before select. "
                             "After a retried failure, `fit-map --declare-unavailable --reason TEXT` records that "
                             "no map can be fitted and lets M0, M1 and M2 continue")
                    return "failed"
                if self.load_maps() is None:
                    path, _, written = self.fit_map()
                    self.log(f"fit-map: {path.name} {'written' if written else 'already fitted; kept'}")
                else:
                    self.log(f"fit-map: {self.maps_path().name} already present; kept")
                continue
            elif step == "select":
                outcome = self.stage_select(retry_failed)
            elif step == "confirm":
                outcome = self.stage_confirm(retry_failed, rerun_not_confirmatory)
                if outcome != "interrupted":
                    self.board()                 # CPU only: the verdict (or `incomplete`) is written right away
            elif step == "board":
                board = self.board(write_selection=False)     # `all` never accepts a failed candidate itself
                if board["selection"]["status"] == "blocked":
                    self.log(f"board: no selection ({board['selection']['reason']})")
                continue
            if outcome in STOPPING or outcome == "smoke_failed":
                self.log(f"{step}: stopped ({outcome}): {STOP_HINTS[outcome]}")
                if stage == "all" and outcome == "partial_budget":
                    # CPU only: show what is done so far. A failed candidate is not accepted on the way.
                    self.board(write_selection=False)
                return outcome
            if outcome == "failed":
                self.log(f"{step}: stopped (failed): {STOP_HINTS['failed']}")
                if stage == "all" and step == "select":
                    self.board(write_selection=False)
                return "failed"
            failed = failed or outcome == "failed_earlier"
        if failed:
            return "failed"
        return outcome if outcome in ("pending_maps",) else "complete"


# ---- leaderboard text (Korean headings; user-facing) ------------------------------------------------------

def _won(value: Any) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _interval(difference: dict | None) -> str:
    if not difference or difference.get("realized_saving") is None:
        return "-"
    data = difference["realized_saving"]
    return f"{data['mean_difference']:+,.0f} [{data['ci90'][0]:+,.0f}, {data['ci90'][1]:+,.0f}]"


def _table_text(table: dict) -> list[str]:
    block = table["block"]
    lines = [f"- 사례: {block['first_case_id']} .. {block['last_case_id']} ({block['cases']}건)", "",
             "| 구성 | 역할 | 상태 | 근거 등급 | 수락 추정 | 최종안 추정 | 사례 | 평균 실현 절감(원) | 평균 순절감(원) "
             "| 가능 사례 합의 | 공급사 이탈 | 승인 카드 | 권한 위반 | 검토 필요 | 실시간 요청 | 저장 재사용 | 실시간 초 "
             "| M0 대비 차이 [90% 구간] | 규칙 정책 대비 차이 [90% 구간] |",
             "|" + " --- |" * 19]
    for row in table["rows"]:
        differences = table["differences"].get(row["id"], {})
        feasible = "-" if row["feasible_agreements"] is None else f"{row['feasible_agreements']}/{row['feasible_cases']}"
        cells = [row["id"], "후보" if row["role"] == "candidate" else "CPU 기준(선택 불가)",
                 row["status"] + (f" ({row['reason']})" if row["reason"] else ""), row["evidence_class"] or "-",
                 (row.get("estimate_sources_display") or row["estimate_sources"])["acceptance"],
                 (row.get("estimate_sources_display") or row["estimate_sources"])["final_offer"],
                 row["cases"], _won(row["mean_realized_saving"]), _won(row["mean_net_saving"]), feasible,
                 row["supplier_walk_aways"], row["approval_cards"], row["authority_violations"], row["needs_review"],
                 row["live_requests"], row["store_hits"], row["live_seconds"],
                 _interval(differences.get("minus_M0")), _interval(differences.get("minus_rule_policy"))]
        lines.append("| " + " | ".join("-" if cell is None else str(cell) for cell in cells) + " |")
    return lines


def render_board(board: dict) -> str:
    """leaderboard.md. The numbers are exactly those of leaderboard.json."""
    real = board["real_model_backend"]
    lines = ["# 협상 에이전트 사용 설정 튜닝 리더보드", "",
             f"- 프로토콜: `{board['protocol_version']}` (sha256 `{board['protocol_sha256'][:12]}`)",
             f"- 작성 시각(UTC): {board['created_at_utc']}",
             f"- 추정 backend 선택: `{board['backend_choice']}` — "
             + ("실제 모델 backend" if real else "CPU 대역(모델이 아님). 아래 숫자는 모델 근거가 아니다."),
             f"- 저장소 식별자: `{board['store_identity12']}`",
             "- 범위: 합성 규칙 기반 공급사, 개발 사례. 보류 평가 결과가 아니며 실제 협상에 대한 근거가 아니다.",
             f"- 알려진 노출: {board['known_exposure']}"]
    if board["rehearsal_blocks"]:
        lines.append(f"- **리허설 블록**: 등록된 구간의 앞부분만 사용했다({board['rehearsal_blocks']}). "
                     "등록된 프로토콜의 결과가 아니다.")
    lines += ["", "## 선택 구간(SELECT)", ""] + _table_text(board["blocks"]["select"])
    lines += ["", "`cpu_reference_not_model_evidence` 행은 모델을 호출하지 않는 CPU 기준이며 선택 대상이 아니다. "
                  "차이는 사례 단위 대응 부트스트랩(2,000회, seed 20261116)의 평균과 90% 구간이다.",
              "", "## 선택 결과", ""]
    selection = board["selection"]
    if selection["status"] == "selected":
        chosen = selection["selected"]
        lines += [f"- 선택된 구성: `{chosen['id']}`" + (" (기본 설정 유지)" if chosen["is_default"] else ""),
                  f"- 선택 구간 평균 실현 절감: {_won(chosen['mean_realized_saving'])}원 (선택에 쓴 표본 내 값)",
                  f"- 근거 등급: {chosen['evidence_class']}",
                  f"- 선택 기록 시각(UTC): {selection['created_at_utc']} — 이후 실행은 이 선택을 바꾸지 않는다."]
    else:
        lines.append(f"- 선택 보류: {selection['reason']}")
        lines += [f"  - {item['candidate']}: {item['state']} ({item['reason']})" for item in selection["blocking"]]
    if board["confirm"] is not None:
        confirm = board["confirm"]
        lines += ["", "## 확인 구간(CONFIRM)", ""] + _table_text(board["blocks"]["confirm"])
        lines += ["", "## 판정", ""]
        if not board["counts_as_model_evidence"]:
            lines.append("- **이 판정은 CPU 대역 또는 축소 블록으로 한 리허설의 것이며 모델 근거가 아니다.**")
        lines += [f"- 1차 판정(S − M0): **{confirm['verdict']}** — {confirm['sentence']}",
                  f"  - 실현 절감 차이 [90% 구간]: {_interval(confirm['primary'])}",
                  f"- 2차(사전 선언, 판정을 바꾸지 않음) S − 쌍둥이: {confirm['secondary']['model_contribution']['verdict']}"
                  f" — {_interval(confirm['secondary']['model_contribution'])}",
                  f"- 2차 S − 규칙 정책({confirm['secondary']['rule_policy']['label']}): "
                  f"{_interval(confirm['secondary']['rule_policy'])}",
                  f"- 확인 구간 첫 사용 시각(UTC): {confirm['first_use_started_at_utc']}"]
    other = board["other_stages"]
    lines += ["", "## 그 밖의 단계", "",
              f"- smoke: {other['smoke']['status']}",
              f"- regress(D0, 선택에 쓰지 않음): {other['regress']['status']}"
              + (f", 기록된 실행과 다른 사례 {other['regress']['cases_differing']}/{other['regress']['cases_compared']}"
                 if other["regress"]["cases_compared"] is not None else ""),
              f"- calibrate(보정 구간, 맵 적합 전용 — 절감액을 표시하거나 비교하지 않음): {other['calibrate']['status']}"]
    if other["maps"] is not None:
        lines.append(f"- 맵: Platt {'있음' if other['maps']['platt_available'] else '없음'}, "
                     f"최종안 축소 가중치 {other['maps']['final_offer_shrink_weight']}")
    return "\n".join(lines) + "\n"


# ---- pack / restore ------------------------------------------------------------------------------------

def _state_files(state: Path) -> list[Path]:
    files = []
    for path in sorted(state.rglob("*")):
        if path.is_symlink():
            raise TuneRefusal(f"{path} is a symbolic link; a state directory holds plain files only")
        if path.is_file() and ".tmp-" not in path.name and path.name not in (MANIFEST_NAME, LOCK_NAME):
            files.append(path)
    return files


def pack_state(state: Path, output: Path, replace_existing: bool = False) -> dict:
    """ZIP of the state directory with a manifest of (relative path, size, sha256)."""
    state, output = Path(state), Path(output)
    protocol_file = state / "protocol.json"
    if not protocol_file.is_file():
        raise TuneRefusal(f"{state} has no protocol.json; nothing to pack")
    if output.resolve().is_relative_to(state.resolve()):
        raise TuneRefusal("--output must be outside the state directory")
    if output.exists() and not replace_existing:
        raise TuneRefusal(f"{output} already exists; choose a new --output or pass --replace")
    protocol = _read_json(protocol_file)
    identity_file = state / "store_identity.json"
    entries, payload = [], []
    for path in _state_files(state):
        data = path.read_bytes()
        name = path.relative_to(state).as_posix()
        entries.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        payload.append((name, data))
    manifest = {"schema_version": 1, "kind": "negotiation_tuning_state", "created_at_utc": _utc_now(),
                "protocol_version": protocol.get("protocol_version"),
                "protocol_sha256": protocol.get("protocol_sha256"),
                "store_identity12": _read_json(identity_file).get("identity12") if identity_file.is_file() else None,
                "file_count": len(entries), "total_bytes": sum(e["size"] for e in entries), "files": entries,
                "notice": "A copy of a tuning state directory. It contains public requests, model responses and "
                          "evaluator-only records of synthetic cases; no credential and no model weight."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for name, data in payload:
                archive.writestr(name, data)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and ":" not in name and not path.is_absolute() \
        and ".." not in path.parts and name == path.as_posix() and name != MANIFEST_NAME


def restore_state(archive_path: Path, state: Path) -> dict:
    """Verify every hash, then write. A tampered archive or a conflicting target changes nothing."""
    archive_path, state = Path(archive_path), Path(state)
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise TuneRefusal(f"{archive_path} is not a readable ZIP: {error}") from error
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in members]
        if names.count(MANIFEST_NAME) != 1 or len(set(names)) != len(names):
            raise TuneRefusal("the archive has no single manifest or has duplicate members")
        try:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            listed = {entry["path"]: entry for entry in manifest["files"]}
            kind = manifest["kind"]
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
            raise TuneRefusal(f"the archive manifest is unreadable: {error}") from error
        if kind != "negotiation_tuning_state" or len(listed) != len(manifest["files"]):
            raise TuneRefusal("the archive is not a tuning state archive")
        if set(names) - {MANIFEST_NAME} != set(listed):
            raise TuneRefusal("the archive members differ from its manifest")
        contents: dict[str, bytes] = {}
        for info in members:
            name = info.filename
            if name == MANIFEST_NAME:
                continue
            entry = listed[name]
            if not _safe_member(name) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise TuneRefusal(f"unsafe archive member {name!r}")
            if info.file_size != entry.get("size"):
                raise TuneRefusal(f"tampered archive member {name}: size differs from the manifest")
            data = archive.read(name)
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                raise TuneRefusal(f"tampered archive member {name}: sha256 differs from the manifest")
            contents[name] = data
    if "protocol.json" not in contents:
        raise TuneRefusal("the archive holds no protocol.json")
    try:
        packed_protocol = json.loads(contents["protocol.json"].decode("utf-8"))
        digest = packed_protocol["protocol_sha256"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
        raise TuneRefusal(f"the archived protocol.json is unreadable: {error}") from error
    if digest != manifest.get("protocol_sha256") or _protocol_digest_of_file(packed_protocol) != digest:
        raise TuneRefusal("the archived protocol.json does not match its digest")
    kept_views, identical, to_write = [], 0, {}
    occupied = state.is_dir() and any(state.iterdir())
    if occupied:
        target_protocol = state / "protocol.json"
        try:
            target_digest = _read_json(target_protocol).get("protocol_sha256") if target_protocol.is_file() else None
        except (OSError, ValueError):
            target_digest = None
        if target_digest != digest:
            raise TuneRefusal(f"{state} is not empty and has a different protocol digest; restore into a new "
                              "directory")
    conflicts = []
    for name, data in contents.items():
        target = state / name
        if not target.exists():
            to_write[name] = data
        elif target.is_file() and target.read_bytes() == data:
            identical += 1
        elif name in DERIVED_VIEWS:
            kept_views.append(name)          # regenerated by `board`; the target's copy stays
        else:
            conflicts.append(name)
    if conflicts:
        raise TuneRefusal(f"{state} already holds different content for {len(conflicts)} file(s), for example "
                          f"{conflicts[0]}; nothing was restored")
    if not occupied:
        # A fresh target appears only when every file is written: a stopped restore leaves no half state
        # that a later command could mistake for an existing one.
        staging = state.with_name(f"{state.name}.restoring-{os.getpid()}-{secrets.token_hex(2)}")
        try:
            for name, data in to_write.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            if state.is_dir():
                state.rmdir()                    # empty by the check above
            os.replace(staging, state)
        except BaseException:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
            raise
    else:
        for name, data in to_write.items():
            target = state / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target) if not target.exists() else None
            temporary.unlink(missing_ok=True)
    return {"written": len(to_write), "identical": identical, "kept_existing_views": kept_views,
            "protocol_sha256": digest, "protocol_version": packed_protocol.get("protocol_version"),
            "store_identity12": manifest.get("store_identity12")}


# ---- command line --------------------------------------------------------------------------------------

def _print_plan(tuner: Tuner, created: bool, budget_minutes: float | None) -> None:
    protocol = tuner.protocol
    print(f"Tuning plan {PROTOCOL_VERSION}  protocol_sha256={tuner.protocol_sha256}")
    print(GPU_NOTICE)
    print(f"protocol.json: {'written now' if created else 'already present and unchanged'}; no model call was made")
    if protocol["rehearsal_blocks"]:
        print(f"REHEARSAL BLOCKS {protocol['rehearsal_blocks']}: a CPU control-flow check, not the registered "
              "protocol")
    print("Blocks:")
    for name, block in protocol["blocks"].items():
        print(f"  {name:8s} {block['first_case_id']}..{block['last_case_id']} ({block['cases']} cases): "
              f"{block['use']}")
    print(f"  reserve  {RESERVE['case_ids']}: {RESERVE['rule']}")
    print("Candidates (in order; ties go to the earlier one): " + ", ".join(arm.name for arm in CANDIDATES))
    print("CPU references (never candidates): " + ", ".join(c.name for c in reference_configs()))
    print(f"Golden request hash check ({GOLDEN_REQUEST['case_id']} round {GOLDEN_REQUEST['round']}): ok")
    print(f"ESTIMATES, not measurements: requests x {PLAN_SECONDS_PER_REQUEST} s; upper bound = a full pass per "
          f"configuration (about {PLAN_REQUESTS_PER_30_DEV_CASES} requests per 30 development cases). Stored "
          "answers are reused, so the real number can be lower.")
    cumulative = 0.0
    for row in plan_estimates(protocol):
        note = ""
        if row["stage"] != "confirm":
            cumulative += row["estimated_minutes_upper_bound"]
            if budget_minutes is not None:
                note = "  fits the budget (estimate)" if cumulative <= budget_minutes else \
                    "  may not fit the budget (estimate); a rerun continues from the store"
        print(f"  {row['stage']:9s} {row['cases']:4d} cases x {row['model_configurations']} model configuration(s): "
              f"about {row['estimated_requests_upper_bound']} requests, about "
              f"{row['estimated_minutes_upper_bound']} min (estimate, upper bound){note}")
    print("`run --stage all` runs smoke, regress, calibrate, fit-map, select and board. CONFIRM is a separate, "
          "single-use stage.")
    print(f"Known exposure: {KNOWN_EXPOSURE}")


PROTECTED_FOLDERS = ("negotiation/evidence", "negotiation/artifacts")


def _refuse_protected_location(path: Path, flag: str) -> None:
    """Tuning state never lives in the held-out history or among the curated evidence files."""
    resolved = Path(path).resolve()
    for folder in PROTECTED_FOLDERS:
        protected = (REPOSITORY_ROOT / folder).resolve()
        if resolved == protected or protected in resolved.parents:
            raise TuneRefusal(f"{flag} {path} is inside {folder}, which holds the held-out history or curated "
                              "evidence; use a folder under runs/negotiation/ instead")


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", type=Path, required=True,
                        help="state directory, for example runs/negotiation/tuning/<label>")
    add_backend_arguments(common)
    common.set_defaults(backend=None)       # `run` needs an explicit choice; no silent CPU or model default
    common.add_argument("--rehearsal-blocks", metavar="d0=N,select=N,confirm=N,cal=N",
                        help="CPU rehearsal only: use the first N cases of each block. Not the registered "
                             "protocol; refused for real-model backends")
    parser = argparse.ArgumentParser(
        description="Tuning runner for how the negotiation agent uses the decision model's estimates. "
                    + GPU_NOTICE)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", parents=[common], help="write protocol.json and print estimates; no model call")
    plan.add_argument("--budget-minutes", type=float, help="only used to say which stages fit (an estimate)")
    run = commands.add_parser("run", parents=[common], help="run one stage, or all up to the leaderboard")
    run.add_argument("--stage", choices=STAGES + ("all",), required=True)
    run.add_argument("--budget-minutes", type=float, help="stop cleanly before the next case once this is used up")
    run.add_argument("--max-live-requests", type=int, help="stop cleanly once this many live requests were made")
    run.add_argument("--retry-failed", action="store_true",
                     help="a person's decision to run a configuration again after a failed estimate")
    run.add_argument("--rerun-not-confirmatory", action="store_true",
                     help="confirm stage only: run CONFIRM again, labelled not_confirmatory")
    run.add_argument("--store-mode", choices=("read-write", "read-only"), default="read-write",
                     help="read-only replays stored answers and never builds the backend")
    fit = commands.add_parser("fit-map", parents=[common],
                              help="fit the maps from the complete calibrate result of M0")
    fit.add_argument("--declare-unavailable", action="store_true",
                     help="a person's decision, recorded: no map can be fitted. Allowed only after the calibrate "
                          "run failed, was retried with --retry-failed and failed again")
    fit.add_argument("--reason", help="required with --declare-unavailable")
    commands.add_parser("board", parents=[common], help="write the leaderboard, and the selection when possible")
    pack = commands.add_parser("pack", parents=[common], help="ZIP the state directory with a manifest")
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--replace", action="store_true", help="replace an existing --output file")
    restore = commands.add_parser("restore", parents=[common], help="verify and unpack a state ZIP")
    restore.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None, *, backend_factory: Callable[[], Any] | None = None,
         declared: tuple[str, bool, bool] | None = None, real_model: bool | None = None,
         store_requested: dict | None = None, clock: Callable[[], float] = time.monotonic) -> int:
    """Exit codes: 0 done or cleanly stopped by the budget; 1 refused or failed; 130 interrupted."""
    args = _parser().parse_args(argv)
    tuner = None
    try:
        args.rehearsal = parse_rehearsal_blocks(args.rehearsal_blocks)
        _refuse_protected_location(args.state_dir, "--state-dir")
        if args.command == "pack":
            _refuse_protected_location(args.output, "--output")
            manifest = pack_state(args.state_dir, args.output, args.replace)
            print(f"{args.output}  files={manifest['file_count']} bytes={manifest['total_bytes']} "
                  f"protocol_sha256={manifest['protocol_sha256']}")
            return 0
        if args.command == "restore":
            summary = restore_state(args.input, args.state_dir)
            print(f"{args.state_dir}  restored: written={summary['written']} identical={summary['identical']} "
                  f"kept_existing_views={summary['kept_existing_views']} protocol_sha256={summary['protocol_sha256']}")
            return 0
        tuner = Tuner(args.state_dir, args, backend_factory=backend_factory, declared=declared,
                      real_model=real_model, store_requested=store_requested, clock=clock)
        if args.command == "plan":
            check_golden_request()
            created = tuner.ensure_protocol(create=True, rehearsal=args.rehearsal)
            _print_plan(tuner, created, args.budget_minutes)
            return 0
        if args.command == "fit-map":
            tuner.ensure_protocol(create=False, rehearsal=args.rehearsal)
            if args.declare_unavailable:
                path, maps = tuner.declare_maps_unavailable(args.reason or "")
                print(f"{path}  declared unavailable by a person; no map was fitted")
                for name, arm in maps["arms"].items():
                    print(f"  {name}: {arm['status']}" + (f" ({arm['reason']})" if arm["reason"] else ""))
                return 0
            path, maps, written = tuner.fit_map()
            shrink_fit = maps["final_offer_shrink"]
            print(f"{path}  {'written' if written else 'already fitted; kept unchanged'}  "
                  f"platt={'fitted' if maps['platt'] else 'unavailable'}  "
                  f"final_offer_shrink_weight={shrink_fit['weight']}")
            for name, arm in maps["arms"].items():
                print(f"  {name}: {arm['status']}" + (f" ({arm['reason']})" if arm["reason"] else ""))
            return 0
        if args.command == "board":
            tuner.ensure_protocol(create=False, rehearsal=args.rehearsal)
            board = tuner.board()
            selection = board["selection"]
            print(f"{args.state_dir / 'leaderboard.md'}  selection={selection['status']}"
                  + (f" selected={selection['selected']['id']}" if selection["status"] == "selected"
                     else f" ({selection['reason']})")
                  + (f"  confirm_verdict={board['confirm']['verdict']}" if board["confirm"] else "")
                  + f"  counts_as_model_evidence={board['counts_as_model_evidence']}")
            return 0
        outcome = tuner.run(args.stage, args.retry_failed, args.rerun_not_confirmatory)
        print(f"tune run: stage={args.stage} status={outcome} live_requests_this_process={tuner.live_total} "
              f"state_dir={args.state_dir}")
        board = tuner.last_board
        if board is not None:
            selection = board["selection"]
            print(f"leaderboard: selection={selection['status']}"
                  + (f" selected={selection['selected']['id']}" if selection["status"] == "selected" else "")
                  + (f" confirm_verdict={board['confirm']['verdict']}" if board["confirm"] else "")
                  + f" counts_as_model_evidence={board['counts_as_model_evidence']}")
        if outcome == "interrupted":
            return 130
        return 0 if outcome in ("complete", "skipped_complete", "partial_budget", "pending_maps") else 1
    except TuneRefusal as refusal:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as failure:
        print(f"Refused: a state file could not be read or written ({type(failure).__name__}: {failure}). "
              "Check free space; a file left empty by a killed process can be deleted if it is one of "
              "run.lock or a *.tmp-* file. Nothing was filled in.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        if tuner is not None:
            tuner.close()


if __name__ == "__main__":
    raise SystemExit(main())
