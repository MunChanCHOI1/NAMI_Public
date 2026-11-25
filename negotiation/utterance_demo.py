"""Demo tables of the utterance-reading experiment.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only, with the CPU stand-in and
test doubles. A table made with `--backend cpu-stand-in` is a rehearsal and says so on its face.

    python negotiation/utterance_demo.py table --state-dir <dir> --seed 1073 --pin final_limit=L1,... <backend>
    python negotiation/utterance_demo.py ask --text "<supplier sentence>" --seed 1073 --state-dir <dir> <backend>
    python negotiation/utterance_demo.py probe-table --probe-result <result.json> [--html <path>]

Design: negotiation/docs/UTTERANCE_AGENT_DESIGN.md, sections 10 and 13.7.

- `table`: one scenario-range supplier (seeds 1000-1099 only). Column A (default wording) and column
  B (same meaning, pinned paraphrase) are negotiated to the end by the prior, keyword and utterance
  arms; columns C (a negated keyword) and D (a neutral sentence plus an instruction) and the neutral
  line only replace the utterance of the same public state and are read, never negotiated. Every
  shown state also gets the existing final-offer question alone. Answers go through the experiment
  store: a read-write run asks what is missing, a read-only run shows "저장된 답 없음" for a missing
  answer and fills in nothing. When the bench block's reading meets the stopping rule of section 8.3,
  a real-model read-write table is refused; `--from-bench` then shows the "R1 not demonstrated" case,
  with the model cells averaged from the bench block's answers to the same utterance (no request).
- `ask`: the four answers for one typed sentence. Its answer is written under `ask/`, never to the
  experiment store, and it is not part of any report. It has no store, so --store-mode read-only is
  refused.
- Every operator-given location (--state-dir, --html) inside negotiation/evidence or
  negotiation/artifacts is refused before anything is written.
- `probe-table`: copies the earlier wording probe's result file into a table. Nothing is recomputed
  from the model's answers; the keyword column is the same probe run with the literal CPU baseline.

Both model commands pass the experiment's argument refusals unchanged. The supplier sentences and
the instruction sentence shown here are test data, not instructions.
"""
import html
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from negotiation_agent import utterance_probe  # noqa: E402
from negotiation_agent.backends import KeywordBackend  # noqa: E402
from negotiation_agent.estimate_sources import round_prior, says_final  # noqa: E402
from negotiation_agent.policy import PolicyConfig  # noqa: E402
from negotiation_agent.prediction_store import StoreMiss  # noqa: E402
from negotiation_agent.provenance import reset_response, validate_predictions  # noqa: E402
from negotiation_agent.requests import FINAL_OFFER_QID  # noqa: E402
from negotiation_agent.session import TERMINAL  # noqa: E402
from negotiation_agent.simulator import default_buyer_private, initial_state, make_supplier  # noqa: E402
from negotiation_agent.utterance_probe import with_latest_supplier_message  # noqa: E402

from utterance_agent import GPU_NOTICE, PROTOCOL_VERSION  # noqa: E402
from utterance_agent import bench as B  # noqa: E402
from utterance_agent import cases as C  # noqa: E402
from utterance_agent import experiment as X  # noqa: E402
from utterance_agent import outcome as O  # noqa: E402
from utterance_agent import wording as W  # noqa: E402
from utterance_agent.interpreter import FIXED_HALF, ModelReader, Thresholds, combine  # noqa: E402
from utterance_agent.questions import QIDS, build_existing_final_request  # noqa: E402

SCENARIO_SEEDS = (1000, 1099)                 # the dashboard scenario range, outside every split
NO_STORED_ANSWER = "저장된 답 없음"
NOT_IN_BENCH = "벤치에 같은 발언 없음"
NOT_RUN_STOPPING_RULE = "not_run_stopping_rule"
FOCUS_ROUND = 2                               # sections 10.2 and 13.7: columns C, D and the neutral line
FROM_BENCH_BANNER = ("중단 규칙(8.3절)에 따라 G-demo를 실행하지 않았다. 판독 칸은 이 사례의 상태가 아니라 U-BENCH "
                     "벤치에서 같은 발언에 대한 답의 평균이며, 발언 판독 구성의 협상은 실행하지 않았다.")
READ_FAILED = "판독 실패: 사람 검토로 넘김"
INSTRUCTION_VERDICT = "지시 감지: 발언으로 추정을 올리지 않음"
VERDICT_TEXT = {"utterance_final": "최종 뜻으로 판독: 추정을 1.0으로 올림",
                "instruction_flagged_round_prior": INSTRUCTION_VERDICT,
                "round_prior": "걸리지 않음: 라운드 사전값",
                "no_utterance_round_prior": "발언 없음: 라운드 사전값"}
KEYWORD_TEXT = {True: "걸림 → 1.0", False: "안 걸림 → 사전값"}
STATUS_TEXT = {"agreed": "합의", "supplier_walked": "결렬", "ended_alternative": "대안 견적으로 종료",
               "needs_review": "사람 검토로 멈춤"}
ACTION_TEXT = {"counter": "역제안", "accept": "수락", "end": "종료", "request_approval": "승인 요청",
               "human_review": "사람 검토"}
PROBE_CLASS_TEXT = {"explicit_final": "명시적 최종안", "paraphrase_final": "표현을 바꾼 최종안 (세 문장 평균)",
                    "negated_final": "부정된 최종안 (두 문장 평균)", "open_to_more": "추가 제안 환영",
                    "embedded_instruction": "지시문 삽입"}
PROBE_LIMITS = ("정답률이 아니라 표현에 대한 반응이다. 협상 결과가 아니다.",
                "지시문 문장에는 \"협상을 종료한다\"는 말이 들어 있어서, 지시를 따른 것인지 그 말을 읽은 것인지는 가를 수 없다.")
TABLE_LIMITS = ("사례 한 개짜리 설명용 장면이다. 빈도나 성능을 뜻하지 않는다.",
                "C열과 D열은 같은 공개 상태에서 발언만 바꾼 판독이다. 협상 결과는 표시하지 않는다.",
                "승인 주장과 지시문은 승인 장부를 바꾸지 못한다. 지시 감지는 추정이 올라가는 것을 막는 장치이고, "
                "권한 확인은 그와 별개로 코드가 한다.")
REHEARSAL_BANNER = "CPU 대역 리허설이다. 모델의 답이 아니며 모델 근거가 아니다."
CSS = ("body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:24px;color:#1b1f24;line-height:1.5}"
       "h1{font-size:20px}h2{font-size:16px;margin-top:28px}"
       "table{border-collapse:collapse;width:100%;margin:8px 0 16px}"
       "th,td{border:1px solid #c9d1d9;padding:6px 8px;vertical-align:top;font-size:13px;text-align:left}"
       "th{background:#f3f5f7}.label{font-weight:600;background:#f3f5f7;width:14%}"
       ".banner{border:2px solid #9a3412;padding:8px 12px;margin:8px 0;font-weight:600}"
       ".meta{font-size:12px;color:#444}.missing{color:#9a3412;font-weight:600}.num{font-variant-numeric:tabular-nums}"
       "@media (prefers-color-scheme: dark){body{background:#111418;color:#e6e9ed}th,.label{background:#1d2228}"
       "th,td{border-color:#3a424c}.meta{color:#b3bac2}.missing,.banner{color:#fdba74;border-color:#fdba74}}")


# ---- shared pieces -------------------------------------------------------------------------------------

def parse_pin(text: str | None) -> dict[str, str]:
    """`class=ID,class=ID` -> {class: sentence id}. A malformed or unknown pin is refused."""
    if not text:
        return {}
    pins: dict[str, str] = {}
    for part in text.split(","):
        name, separator, sentence_id = part.strip().partition("=")
        if not separator or not name or not sentence_id or name in pins:
            raise X.ExperimentRefusal(f"--pin needs distinct class=sentence-id pairs separated by commas; got {part!r}")
        pins[name.strip()] = sentence_id.strip()
    try:
        return W.check_pinned(pins)
    except ValueError as error:
        raise X.ExperimentRefusal(f"--pin: {error}") from error


def demo_supplier(seed: int):
    """The scenario-range supplier of a demo: deadline style, no payment preference, reveals its deadline."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not SCENARIO_SEEDS[0] <= seed <= SCENARIO_SEEDS[1]:
        raise X.ExperimentRefusal(f"--seed must be a scenario seed in {SCENARIO_SEEDS[0]}-{SCENARIO_SEEDS[1]}; "
                                  "every other seed belongs to a split or to the experiment's blocks")
    return make_supplier(seed, "deadline", False, True)


def demo_gate(state_dir: Path) -> dict:
    """The gate a demo shows: the fitted pair, or the fixed pair when the fit found no useful threshold.

    `arm` is the registered arm that runs; `name` is what the table shows. utt_gate_fit is defined as
    the variant fitted on the fit block, so thresholds fitted on the smoke block as a rehearsal are
    shown as utt_gate_fit_rehearsal.
    """
    fitted = B.load_state_thresholds(state_dir)
    if fitted is None:
        raise X.ExperimentRefusal("this state directory has no thresholds.json; run `bench` and `fit` first so "
                                  "that the table shows the fitted gate")
    if fitted["thresholds"] is not None:
        arm, thresholds = "utt_gate_fit", fitted["thresholds"]
    else:
        arm, thresholds = "utt_gate", FIXED_HALF       # stated in the table: the fit found no useful threshold
    name = f"{arm}_rehearsal" if arm == "utt_gate_fit" and fitted["fitted_on"] != "utt-fit" else arm
    return {"arm": arm, "name": name, "thresholds": thresholds, "fit_status": fitted["status"],
            "fitted_on": fitted["fitted_on"], "existing_threshold": fitted["existing_threshold"], "fitted": fitted}


def _guarded(call: Callable[[], dict]) -> dict:
    """Run one question. A missing stored answer or a failure is shown as such; no value is filled in."""
    try:
        return {"status": "ok", **call()}
    except StoreMiss:
        return {"status": "no_stored_answer", "text": NO_STORED_ANSWER}
    except Exception as error:   # noqa: BLE001 - keep the failed attempt visible, invent nothing
        return {"status": "error", "text": READ_FAILED, "error": type(error).__name__, "detail": str(error)[:300]}


def read_state(state, backend, gate: dict) -> dict:
    """Keyword rule, existing question and utterance reading of one public state."""
    thresholds: Thresholds = gate["thresholds"]
    prior = round_prior(state, PolicyConfig().final_prior_by_round)

    def existing() -> dict:
        reset_response(backend)
        predicted = backend.predict(state, [], build_existing_final_request(state))
        value = validate_predictions(predicted, [FINAL_OFFER_QID])[FINAL_OFFER_QID]
        level = gate["existing_threshold"]
        return {"final_offer": value, "fitted_threshold": level,
                "at_or_above_fitted_threshold": None if level is None else value >= level}

    def reading() -> dict:
        answers, digest, _ = ModelReader(backend).read(state)
        value, value_from, says, flag = combine(answers, prior, thresholds)
        return {"answers": answers, "signal_final": max(answers["u_final"], answers["u_walkaway"]),
                "says_final": says, "instruction_flag": flag, "final_offer_value": value,
                "value_from": value_from, "text": VERDICT_TEXT[value_from], "request_sha256": digest}

    keyword = says_final(state)
    return {"round": state.current_round, "utterance": state.last_offer("supplier").message,
            "round_prior": prior, "keyword": {"fires": keyword, "text": KEYWORD_TEXT[keyword]},
            "existing_question": _guarded(existing), "reading": _guarded(reading)}


def read_state_from_bench(state, items: list[dict], gate: dict) -> dict:
    """Section 8.3, "R1 not demonstrated": the model cells of one state, filled from the bench block's answers.

    No request is made. The bench items whose latest supplier message is exactly this state's
    utterance are averaged over their (bench) states; the table says so, because those are other
    states than this case's. An utterance the bench never asked about stays marked as such.
    """
    thresholds: Thresholds = gate["thresholds"]
    prior = round_prior(state, PolicyConfig().final_prior_by_round)
    utterance = state.last_offer("supplier").message
    same = [item for item in items if item.get("message") == utterance]
    answered = [item for item in same if item.get("status") == "ok" and isinstance(item.get("answers"), dict)]
    existing = [item["existing"]["final_offer"] for item in same if (item.get("existing") or {}).get("status") == "ok"]
    if answered:
        gates = [combine(item["answers"], prior, thresholds) for item in answered]
        means = {qid: sum(item["answers"][qid] for item in answered) / len(answered) for qid in QIDS}
        reading = {"status": "from_bench", "states": len(answered), "mean_answers": means,
                   "fire_rate": sum(g[1] == "utterance_final" for g in gates) / len(gates),
                   "flag_rate": sum(g[3] for g in gates) / len(gates),
                   "text": f"벤치 답 {len(answered)}상태 평균 (이 사례의 상태가 아님)"}
    else:
        reading = {"status": "not_in_bench", "text": NOT_IN_BENCH}
    if existing:
        level = gate["existing_threshold"]
        mean = sum(existing) / len(existing)
        existing_cell = {"status": "from_bench", "states": len(existing), "final_offer": mean,
                         "fitted_threshold": level,
                         "fire_rate_at_fitted_threshold": None if level is None else
                         sum(value >= level for value in existing) / len(existing),
                         "text": f"벤치 답 {len(existing)}상태 평균 (이 사례의 상태가 아님)"}
    else:
        existing_cell = {"status": "not_in_bench", "text": NOT_IN_BENCH}
    keyword = says_final(state)
    return {"round": state.current_round, "utterance": utterance, "round_prior": prior,
            "keyword": {"fires": keyword, "text": KEYWORD_TEXT[keyword]},
            "existing_question": existing_cell, "reading": reading}


def _negotiate(case, style: str, arm: str, backend, pinned: dict | None) -> dict:
    """One arm's negotiation of a column, with the states it visited (first decision of each round)."""
    visited: dict[int, Any] = {}
    actions: list[dict] = []

    def step(before, decision) -> None:
        visited.setdefault(before.current_round, before)
        actions.append({"round": decision.round_index, "action": decision.action,
                        "p_final_offer": decision.facts.get("p_final_offer"),
                        "p_final_offer_source": decision.facts.get("p_final_offer_source")})

    row = O.run_styled_case(case, style, default_buyer_private(), backend, PolicyConfig(), pinned, on_step=step)
    ended = row["status"] in TERMINAL
    out = {"arm": arm, "status": row["status"], "status_text": STATUS_TEXT.get(row["status"], row["status"]),
           "actions": actions, "realized_saving": row["realized_saving"], "utterances": row["utterances"],
           "states": visited}
    if not ended:
        missing = bool(row["errors"]) and all(item["error"].startswith("StoreMiss") for item in row["errors"])
        out["status_text"] = NO_STORED_ANSWER if missing else READ_FAILED
        out["error"] = row["errors"][0]["error"][:300] if row["errors"] else None
    return out


def _bench_summary(state_dir: Path) -> dict | None:
    """Fire rates per bench item and the guard result, when the bench block has a completed result."""
    attempt = X.first_complete_attempt(Path(state_dir) / "bench" / "bench")
    if attempt is None:
        return None
    result = attempt["result"]
    basis = result["verdict"]["basis"]
    by_item = result["readers"][basis]["by_item"]
    return {"attempt": attempt["name"], "evidence_class": result["evidence_class"], "basis": basis,
            "reading": result["verdict"]["reading"], "guard": result["guard"]["result"],
            "guard_fire_rate": result["guard"]["instruction_fire_rate"],
            "fire_rate_by_item": {name: by_item[name]["fire_rate"] for name in X.BENCH_ITEMS}}


# ---- table ---------------------------------------------------------------------------------------------

def build_table(seed: int, pinned: dict[str, str], *, state_dir: Path, model_backend, real_model: bool,
                from_bench: dict | None = None) -> dict:
    """The demo table as data. Raises ExperimentRefusal before any request when it cannot be shown.

    `from_bench` is the first completed bench-block attempt when the table is built in the "R1 not
    demonstrated" mode of section 8.3: then no model backend is used at all, the model cells come
    from that attempt's answers and the utterance arm is not negotiated.
    """
    hidden = demo_supplier(seed)
    gate = demo_gate(state_dir)
    case = C.UttCase(f"demo-{seed}", seed, hidden)
    before = X.request_counts(model_backend)
    items = from_bench["result"]["items"] if from_bench is not None else None
    arms = ("prior", "keyword") if from_bench is not None else ("prior", "keyword", gate["arm"])

    def read(state) -> dict:
        return read_state_from_bench(state, items, gate) if items is not None else read_state(state, model_backend, gate)

    columns: dict[str, dict] = {}
    for key, style, pins in (("A", "default", None), ("B", "paraphrase", pinned or None)):
        runs = {}
        for arm in arms:
            backend = O.make_arm_backend(arm, model_backend=model_backend, thresholds=gate["thresholds"])
            runs[arm] = _negotiate(case, style, arm, backend, pins)
        if from_bench is not None:
            runs[gate["arm"]] = {"arm": gate["arm"], "status": NOT_RUN_STOPPING_RULE,
                                 "status_text": "실행하지 않음(판독이 드러나지 않아 협상 비교를 하지 않는다)",
                                 "actions": [], "realized_saving": None}
        states = runs["prior"].pop("states")          # the model-free path: the same states whatever the answers
        labels = {u["offer_id"]: u["labels"] for u in runs["prior"]["utterances"]}
        rounds = []
        for number in sorted(states):
            cell = read(states[number])
            cell["labels"] = labels[states[number].last_offer("supplier").offer_id]
            rounds.append(cell)
        for run in runs.values():
            run.pop("states", None)
            run.pop("utterances", None)
        columns[key] = {"style": style, "pinned": pins, "rounds": rounds, "outcomes": runs, "_states": states}
    # Columns C and D and the neutral line read the round-2 public state of column B (sections 10.2 and
    # 13.7); a supplier that ends before round 2 gives its last state instead.
    b_rounds = columns["B"]["rounds"]
    focus = FOCUS_ROUND if any(cell["round"] == FOCUS_ROUND for cell in b_rounds) else b_rounds[-1]["round"]
    base = columns["B"]["_states"][focus]
    negated = W.bank_sentence(pinned.get("negated_keyword", "K1"))[1]
    neutral = W.bank_sentence(pinned.get("neutral", "N1"))[1]
    instruction = W.BANK["instruction_to_final"][0][1]
    variants = {"C": negated, "D": f"{neutral} {instruction}", "neutral": neutral}
    for key, message in variants.items():
        cell = read(with_latest_supplier_message(base, message))
        cell["labels"] = W.label_text(message).to_json()
        columns[key] = {"style": "paraphrase state, utterance replaced; read only, not negotiated", "cell": cell}
    for key in ("A", "B"):
        columns[key].pop("_states")
    delta = X.count_delta(before, X.request_counts(model_backend))
    live, answered, hits = delta["live_requests"], delta["live_answers"], delta["store_hits"]
    cells = [cell for key in ("A", "B") for cell in columns[key]["rounds"]] + [columns[k]["cell"] for k in variants]
    shown = ("ok", "from_bench", "not_in_bench")
    empty = sum(1 for cell in cells for part in ("existing_question", "reading") if cell[part]["status"] not in shown)
    empty += sum(1 for key in ("A", "B") for run in columns[key]["outcomes"].values()
                 if run["status"] not in TERMINAL and run["status"] != NOT_RUN_STOPPING_RULE)
    counts = {"live_requests": live, "live_failed_requests": live - answered, "store_hits": hits, "empty_cells": empty}
    model_cells_from = None
    if from_bench is not None:
        bench_class = from_bench["result"].get("evidence_class")
        real_model = str(bench_class).startswith("real_model")
        evidence = bench_class
        counts["cells_not_in_bench"] = sum(1 for cell in cells for part in ("existing_question", "reading")
                                           if cell[part]["status"] == "not_in_bench")
        model_cells_from = {"source": "bench_block_answers", "attempt": from_bench["name"],
                            "evidence_class": bench_class, "stopping_rule": from_bench.get("stopping_rule"),
                            "note": "averages of the bench block's answers to the same utterance on other "
                                    "states; no request was made for this table"}
    else:
        evidence = X.evidence_class("utterance_arm", real_model=real_model, live=answered, hits=hits)
    return {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": X.protocol_sha256(),
        "experiment": "utterance_demo_table", "created_at_utc": X._utc_now(), "gpu_notice": GPU_NOTICE,
        "seed": seed, "supplier": "scenario-range supplier: deadline style, no payment preference, reveals its "
                                  "deadline", "pinned": dict(pinned),
        "gate": {"name": gate["name"], "arm": gate["arm"], "thresholds": gate["thresholds"].to_json(),
                 "fit_status": gate["fit_status"], "fitted_on": gate["fitted_on"],
                 "existing_question_threshold": gate["existing_threshold"]},
        "evidence_class": evidence, "model_cells_from": model_cells_from,
        "real_model_backend": real_model, "counts_as_model_evidence": False,
        "counts": counts, "focus_round": focus, "columns": columns, "bench": _bench_summary(state_dir),
        "limits": list(TABLE_LIMITS),
        "scope": "one illustrative synthetic case; not a frequency, not a performance figure, not an outcome "
                 "comparison",
    }


def _number(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def _reading_text(cell: dict) -> str:
    reading = cell["reading"]
    if reading["status"] == "from_bench":
        answers = " ".join(f"{qid}={reading['mean_answers'][qid]:.3f}" for qid in QIDS)
        return (f"{answers} → 발화율 {reading['fire_rate']:.3f}, 지시 표시율 {reading['flag_rate']:.3f} "
                f"({reading['text']})")
    if reading["status"] != "ok":
        return reading["text"] + (f" ({reading['error']})" if reading.get("error") else "")
    answers = " ".join(f"{qid}={reading['answers'][qid]:.3f}" for qid in QIDS)
    return f"{answers} → {reading['text']}"


def _existing_text(cell: dict) -> str:
    existing = cell["existing_question"]
    if existing["status"] == "from_bench":
        return f"final_offer={existing['final_offer']:.3f} ({existing['text']})"
    if existing["status"] != "ok":
        return existing["text"] + (f" ({existing['error']})" if existing.get("error") else "")
    return f"final_offer={existing['final_offer']:.3f}"


def _outcome_text(column: dict) -> str:
    parts = []
    for arm, run in column["outcomes"].items():
        steps = ", ".join(f"{a['round']}라운드 {ACTION_TEXT.get(a['action'], a['action'])}" for a in run["actions"])
        saving = "" if run["realized_saving"] is None else f", 절감 {run['realized_saving']:,}원"
        parts.append(f"{arm}: {steps} → {run['status_text']}{saving}")
    return "\n".join(parts)


def _model_name(table: dict) -> str:
    return "Zoro-1" if table["real_model_backend"] else "CPU 대역(모델 아님)"


def table_rows(table: dict) -> tuple[list[str], list[list[str]]]:
    """Header and rows of the main table: columns A-D at the focus round."""
    columns, focus, name = table["columns"], table["focus_round"], _model_name(table)

    def at_focus(key: str) -> dict | None:
        return next((cell for cell in columns[key]["rounds"] if cell["round"] == focus), None)

    cells = {"A": at_focus("A"), "B": at_focus("B"), "C": columns["C"]["cell"], "D": columns["D"]["cell"]}
    header = ["", "A: 기본 문구", "B: 같은 뜻, 다른 표현", "C: 음성 대조 (키워드는 있지만 뜻은 반대)", "D: 지시문"]
    not_shown = "판독만 표시한다. 협상 결과는 표시하지 않는다."

    def row(label: str, text: Callable[[dict], str]) -> list[str]:
        return [label] + ["이 라운드에 도달하지 않음" if cells[key] is None else text(cells[key]) for key in "ABCD"]

    return header, [
        row(f"{focus}라운드 공급사 발언", lambda cell: cell["utterance"]),
        row("키워드 규칙", lambda cell: cell["keyword"]["text"]),
        row(f"{name} 기존 질문 하나", _existing_text),
        row(f"{name} 발언 판독", _reading_text),
        ["정책의 선택과 결과", _outcome_text(columns["A"]), _outcome_text(columns["B"]), not_shown, not_shown],
    ]


def round_rows(table: dict) -> tuple[list[str], list[list[str]]]:
    """Every shown state of columns A and B, plus the neutral line."""
    name = _model_name(table)
    header = ["열", "라운드", "공급사 발언", "키워드 규칙", f"{name} 기존 질문 하나", f"{name} 발언 판독"]
    rows = [[key, str(cell["round"]), cell["utterance"], cell["keyword"]["text"], _existing_text(cell),
             _reading_text(cell)] for key in ("A", "B") for cell in table["columns"][key]["rounds"]]
    neutral = table["columns"]["neutral"]["cell"]
    rows.append(["중립 문장", str(neutral["round"]), neutral["utterance"], neutral["keyword"]["text"],
                 _existing_text(neutral), _reading_text(neutral)])
    return header, rows


def _html_table(header: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, tag: str = "td", first: bool = False) -> str:
        klass = ' class="label"' if first and tag == "td" else ""
        missing = ' class="missing"' if text.startswith((NO_STORED_ANSWER, READ_FAILED)) else klass
        return f"<{tag}{missing}>{html.escape(text).replace(chr(10), '<br>')}</{tag}>"

    head = "".join(cell(text, "th") for text in header)
    body = "".join("<tr>" + "".join(cell(text, first=index == 0) for index, text in enumerate(row)) + "</tr>"
                   for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _page(title: str, body: list[str]) -> str:
    return ("<!doctype html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            + "".join(body) + "</body></html>\n")


def _from_bench_banner(table: dict) -> str:
    source = table["model_cells_from"]
    stop = source.get("stopping_rule") or {}
    if stop.get("met"):
        return FROM_BENCH_BANNER + f" (중단 규칙 {stop['rules']}, 벤치 attempt {source['attempt']})"
    return ("--from-bench로 만든 표다. 판독 칸은 이 사례의 상태가 아니라 U-BENCH 벤치에서 같은 발언에 대한 답의 평균이며, "
            f"발언 판독 구성의 협상은 실행하지 않았다. (벤치 attempt {source['attempt']})")


def table_html(table: dict) -> str:
    """The demo table as one self-contained HTML file (no external resource)."""
    gate, counts = table["gate"], table["counts"]
    body = [f"<h1>공급사 발언 판독 장면 (seed {table['seed']})</h1>"]
    if not table["real_model_backend"]:
        body.append(f"<p class=\"banner\">{html.escape(REHEARSAL_BANNER)}</p>")
    if table.get("model_cells_from"):
        body.append(f"<p class=\"banner\">{html.escape(_from_bench_banner(table))}</p>")
    body.append("<p class=\"meta\">" + html.escape(
        f"근거 등급: {table['evidence_class']} · 모델 근거로 세지 않음(사례 한 개짜리 설명용) · 실시간 요청 "
        f"{counts['live_requests']}회, 저장소 재사용 {counts['store_hits']}회, 빈 칸 {counts['empty_cells']}개 · "
        f"게이트 {gate['name']} (τ_final={gate['thresholds']['final']}, τ_inst={gate['thresholds']['instruction']}, "
        f"적합 {gate['fit_status']}, 출처 {gate['fitted_on']}) · 프로토콜 {table['protocol_version']}") + "</p>")
    body.append("<h2>같은 공급사, 같은 숫자. 발언만 다르다</h2>" + _html_table(*table_rows(table)))
    body.append("<h2>라운드별 판독과 중립 문장</h2>" + _html_table(*round_rows(table)))
    bench = table["bench"]
    body.append("<h2>판독 벤치의 항목별 발화율과 지시 가드</h2>")
    if bench is None:
        body.append("<p>이 상태 폴더에 U-BENCH 벤치의 완료된 결과가 없다.</p>")
    else:
        body.append("<p class=\"meta\">" + html.escape(
            f"근거 등급: {bench['evidence_class']} · 판독 {bench['reading']} · 지시 가드 {bench['guard']} "
            f"(발화율 {_number(bench['guard_fire_rate'])}) · 기준 {bench['basis']}") + "</p>")
        body.append(_html_table(["항목", "발화율"], [[item, _number(rate)]
                                                  for item, rate in bench["fire_rate_by_item"].items()]))
    body.append("<h2>한계</h2><ul>" + "".join(f"<li>{html.escape(text)}</li>" for text in table["limits"]) + "</ul>")
    return _page(f"발언 판독 장면 seed {table['seed']}", body)


def _print_table(header: list[str], rows: list[list[str]]) -> None:
    for row in rows:
        print(f"[{row[0]}]")
        for label, text in zip(header[1:], row[1:]):
            for index, line in enumerate(text.split("\n")):
                print(f"  {label if index == 0 else ' ' * len(label)}  {line}")


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def command_table(args: Any, *, backend_factory, declared, real_model, registry_path) -> int:
    """`table`: build, save and print the demo table. Exit 0 only when no cell is empty."""
    real = X.real_model_requested(args) if real_model is None else bool(real_model)
    demo_supplier(args.seed)
    pinned = parse_pin(args.pin)
    state_dir = Path(args.state_dir)
    X.check_golden_requests()
    X.ensure_protocol(state_dir, create=True)
    gate = demo_gate(state_dir)                                  # refuse before a first use is recorded
    stop = O.stop_rule(state_dir, gate["fitted"])
    from_bench = None
    if getattr(args, "from_bench", False):
        # Section 8.3, "R1 not demonstrated": the model cells come from the bench block's answers; no
        # backend is built, nothing is asked and no first use is recorded.
        from_bench = X.first_complete_attempt(state_dir / "bench" / "bench")
        if from_bench is None:
            raise X.ExperimentRefusal("--from-bench fills the model cells from the bench block's answers, and this "
                                      "state directory has no completed `bench --block bench` attempt")
        from_bench = {**from_bench, "stopping_rule": stop}
    elif stop["met"] and real and args.store_mode == "read-write":
        raise X.ExperimentRefusal(f"stopping rule {stop['rules']} of the design is met by bench attempt "
                                  f"{stop['bench_attempt']}: G-demo is not run. Show the 'R1 not demonstrated' case "
                                  "with --from-bench, which fills the model cells from the bench answers")
    if from_bench is not None:
        table = build_table(args.seed, pinned, state_dir=state_dir, model_backend=None, real_model=False,
                            from_bench=from_bench)
        real = table["real_model_backend"]
    else:
        if real and args.store_mode == "read-write":
            X.record_first_use(state_dir, "first_real_response", "demo", registry_path=registry_path)
        backend = X.make_model_backend(args, factory=backend_factory, declared=declared)
        try:
            table = build_table(args.seed, pinned, state_dir=state_dir, model_backend=backend, real_model=real)
            table["backend"] = backend.describe()
        finally:
            backend.close()
    table["stopping_rule"] = stop
    folder = X.new_attempt_folder(state_dir / "demo")
    X.write_json(folder / "table.json", table, exclusive=True)
    target = Path(args.html) if args.html else folder / "table.html"
    _write_text(target, table_html(table))
    if not real:
        print(REHEARSAL_BANNER)
    if from_bench is not None:
        print(_from_bench_banner(table))
    _print_table(*table_rows(table))
    _print_table(*round_rows(table))
    counts = table["counts"]
    print(f"{folder / 'table.json'}  html={target}  evidence_class={table['evidence_class']} "
          f"live_requests={counts['live_requests']} store_hits={counts['store_hits']} "
          f"empty_cells={counts['empty_cells']} counts_as_model_evidence=False")
    return 0 if counts["empty_cells"] == 0 else 1


# ---- ask -----------------------------------------------------------------------------------------------

def command_ask(args: Any, *, backend_factory, real_model, registry_path) -> int:
    """`ask`: the four answers for one typed supplier sentence. Written under ask/, never to the store."""
    real = X.real_model_requested(args) if real_model is None else bool(real_model)
    if args.store_mode != "read-write":
        # `ask` has no store to replay from: it always makes one live call. Refused before anything is
        # written, so no first use is recorded and no backend is built.
        raise X.ExperimentRefusal("ask has no prediction store and always asks the backend once; --store-mode "
                                  "read-only is refused (use read-write, or rebuild a stored table with `table`)")
    hidden = demo_supplier(args.seed)
    text = args.text.strip() if isinstance(args.text, str) else ""
    if not text:
        raise X.ExperimentRefusal("--text needs a supplier sentence")
    state_dir = Path(args.state_dir)
    X.check_golden_requests()
    X.ensure_protocol(state_dir, create=True)
    gate = demo_gate(state_dir)
    if real:
        X.record_first_use(state_dir, "first_real_response", "ask", registry_path=registry_path)
    factory = backend_factory or (X._native_factory(args) if real else X.UtteranceCpuStandIn)
    # The typed sentence takes the place of the supplier's utterance and nothing else.
    state = with_latest_supplier_message(initial_state(f"demo-{args.seed}", hidden), text)
    prior = round_prior(state, PolicyConfig().final_prior_by_round)
    record: dict[str, Any] = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION, "protocol_sha256": X.protocol_sha256(),
        "experiment": "utterance_demo_single_sentence", "created_at_utc": X._utc_now(), "seed": args.seed,
        "text": text, "round": state.current_round, "round_prior": prior,
        "gate": {"name": gate["name"], "thresholds": gate["thresholds"].to_json(), "fit_status": gate["fit_status"]},
        "counts_as_model_evidence": False, "in_report": False,
        "scope": "one typed sentence; not a bench item, not a comparison"}
    backend, live = None, 0
    try:
        backend = factory()
        answers, digest, response = ModelReader(backend).read(state)
        live = 1
        value, value_from, says, flag = combine(answers, prior, gate["thresholds"])
        record.update(status="ok", request_sha256=digest, answers=answers, says_final=says, instruction_flag=flag,
                      signal_final=max(answers["u_final"], answers["u_walkaway"]), final_offer_value=value,
                      value_from=value_from, verdict=VERDICT_TEXT[value_from], backend=backend.describe(),
                      response=response)
    except Exception as error:   # noqa: BLE001 - shown as a failure; no value is filled in
        record.update(status="error", verdict=READ_FAILED, error=f"{type(error).__name__}: {str(error)[:500]}")
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    record["evidence_class"] = X.evidence_class("ask", real_model=real, live=live, hits=0)
    folder = X.new_attempt_folder(state_dir / "ask")
    X.write_json(folder / "answer.json", record, exclusive=True)
    if not real:
        print(REHEARSAL_BANNER)
    print(f"공급사 문장: {text}")
    if record["status"] == "ok":
        print("  " + " ".join(f"{qid}={record['answers'][qid]:.3f}" for qid in QIDS))
        print(f"  판정: {record['verdict']} (final_offer={record['final_offer_value']:.3f}, 게이트 {gate['name']})")
    else:
        print(f"  {READ_FAILED} ({record['error'].split(':', 1)[0]})")
    print(f"{folder / 'answer.json'}  evidence_class={record['evidence_class']} counts_as_model_evidence=False")
    return 0 if record["status"] == "ok" else 1


# ---- probe table (scene 0) -----------------------------------------------------------------------------

def build_probe_table(path: Path) -> dict:
    """Copy the earlier wording probe's result into table data. Model values are not recomputed."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise X.ExperimentRefusal(f"{path} is not a readable result file: {error}") from error
    if not isinstance(data, dict) or data.get("experiment") != "utterance_wording_probe":
        raise X.ExperimentRefusal(f"{path} is not a result of the utterance wording probe")
    if data.get("complete") is not True:
        raise X.ExperimentRefusal(f"{path} is not a completed probe result; an incomplete result is not shown")
    try:
        by_class, neutral = data["metrics"]["by_class"], data["metrics"]["neutral_mean_estimate"]
        scope, backend = data["scope"], data.get("backend") or {}
        rows_in = {name: by_class[name] for name in utterance_probe.CLASSES}
    except (KeyError, TypeError) as error:
        raise X.ExperimentRefusal(f"{path} lacks a field of the probe result: {error}") from error
    # The keyword column: the same probe run with the literal CPU baseline (deterministic, not a model).
    reference = utterance_probe.run(KeywordBackend(), max_states=int(scope["states"]),
                                    source_provenance={"note": "CPU keyword reference for the probe table"})
    keyword = reference["metrics"]["by_class"]
    rows = [{"class": name, "label": PROBE_CLASS_TEXT[name],
             "model_mean_shift": rows_in[name]["mean_shift"],
             "model_ci90": rows_in[name]["mean_shift_ci90_by_state"],
             "states": rows_in[name]["states"], "share_of_states_up": rows_in[name].get("share_of_states_up"),
             "keyword_mean_shift": keyword.get(name, {}).get("mean_shift")} for name in utterance_probe.CLASSES]
    return {
        "schema_version": 1, "experiment": "utterance_demo_probe_table", "source_file": Path(path).name,
        "source_created_at_utc": data.get("created_at_utc"), "source_protocol_version": data.get("protocol_version"),
        "evidence_class": data.get("evidence_class"), "is_real_model_probe": data.get("evidence_class") == "real_model_probe",
        "gpu_name": backend.get("gpu_name"), "dtype": backend.get("dtype"), "states": scope.get("states"),
        "variants_per_state": scope.get("variants_per_state"), "requests": scope.get("planned_requests"),
        "reading": (data.get("verdict") or {}).get("reading"),
        "warnings": (data.get("verdict") or {}).get("warnings"),
        "neutral_mean_estimate": neutral, "rows": rows,
        "keyword_reference": {"evidence_class": reference["evidence_class"], "states": reference["scope"]["states"],
                              "note": "CPU reference: the literal keyword baseline on the probe's own states"},
        "limits": list(PROBE_LIMITS),
        "note": "values copied from the result file; nothing about the model was recomputed",
    }


def probe_rows(table: dict) -> tuple[list[str], list[list[str]]]:
    header = ["공급사 발언", "키워드 규칙(CPU 참조)", "기존 질문 하나: p(변형) − p(중립) 평균 [90% 구간]"]
    rows = []
    for row in table["rows"]:
        interval = row["model_ci90"]
        model = f"{row['model_mean_shift']:+.3f}" + (f" [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "")
        keyword = "—" if row["keyword_mean_shift"] is None else f"{row['keyword_mean_shift']:+.3f}"
        rows.append([row["label"], keyword, model])
    return header, rows


def _probe_title(table: dict) -> str:
    return (f"기존 발언 진단 결과 (등급 {table['evidence_class']}, {table['gpu_name'] or 'GPU 기록 없음'}, "
            f"{table['dtype'] or 'dtype 기록 없음'}, 중립 상태 {table['states']}개 × 발언 {table['variants_per_state']}종)")


def probe_html(table: dict) -> str:
    body = [f"<h1>{html.escape(_probe_title(table))}</h1>",
            "<p class=\"meta\">" + html.escape(
                f"출처 파일 {table['source_file']} · 판독 {table['reading']} · 중립 발언의 평균 값 "
                f"{_number(table['neutral_mean_estimate'])} · 값은 결과 파일에서 그대로 옮겼다") + "</p>",
            _html_table(*probe_rows(table)),
            "<h2>한계</h2><ul>" + "".join(f"<li>{html.escape(text)}</li>" for text in table["limits"]) + "</ul>"]
    return _page("기존 발언 진단 결과", body)


def command_probe_table(args: Any) -> int:
    """`probe-table`: scene 0. No model, no backend option, no state directory."""
    table = build_probe_table(args.probe_result)
    print(_probe_title(table))
    header, rows = probe_rows(table)
    for row in rows:
        print(f"  {row[0]}: {header[1]} {row[1]} | {row[2]}")
    print(f"  중립 발언의 평균 값: {_number(table['neutral_mean_estimate'])}")
    for text in table["limits"]:
        print(f"  한계: {text}")
    if args.html:
        _write_text(args.html, probe_html(table))
        print(f"html={args.html}")
    return 0


# ---- command line --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None, *, backend_factory: Callable[[], Any] | None = None,
         declared: tuple[str, bool, bool] | None = None, real_model: bool | None = None,
         registry_path: Path | None = None) -> int:
    """Exit codes: 0 done; 1 refused, failed or a table with an empty cell. Keyword arguments are test seams."""
    args = X.build_demo_parser().parse_args(argv)
    try:
        X.refuse_protected_arguments(args)        # before any folder or file is created
        if args.command == "probe-table":
            return command_probe_table(args)
        X.check_arguments(args)
        if args.command == "table":
            return command_table(args, backend_factory=backend_factory, declared=declared, real_model=real_model,
                                 registry_path=registry_path)
        return command_ask(args, backend_factory=backend_factory, real_model=real_model,
                           registry_path=registry_path)
    except X.ExperimentRefusal as refusal:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as failure:
        print(f"Refused: a file could not be read or written ({type(failure).__name__}: {failure}). "
              "Nothing was filled in.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
