"use strict";
const $ = id => document.getElementById(id);
const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
const won = v => v == null ? "—" : `${Math.round(v).toLocaleString("ko-KR")}원`;
const signedWon = v => v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(Math.round(v)).toLocaleString("ko-KR")}원`;
const percent = v => v == null ? "—" : `${Math.round(v * 100)}%`;
const terms = t => t ? `${t.unit_price.toLocaleString("ko-KR")}원 · ${t.delivery_days}일 · ${t.payment_days === 0 ? "즉시 결제" : `${t.payment_days}일 결제`}` : "대안 견적 선택";
const actions = {accept:"현재 제안 수락",counter:"역제안",request_approval:"역제안 · 승인 대기",end:"대안 견적 선택",human_review:"사람 검토 필요"};
const kinds = {standing_offer:"현재 제안 수락",same_terms:"역제안 · 같은 조건",pay_sooner:"역제안 · 결제 앞당김",deliver_sooner:"역제안 · 납기 앞당김",pay_and_deliver_sooner:"역제안 · 납기·결제 앞당김",usual_payment:"역제안 · 기본 결제",softer_same_terms:"역제안 · 양보안",take_alternative:"협상 종료 · 대안 선택"};
const statuses = {active:"다음 판단 가능",awaiting_approval:"내부 승인 대기",awaiting_supplier:"공급사 응답 대기",awaiting_info:"공급사 정보 확인 대기",needs_review:"사람 검토 필요",agreed:"합의 완료",ended_alternative:"협상 종료 · 대안 선택",supplier_walked:"공급사 협상 종료"};
const authority = {auto:"담당자 권한",needs_approval:"승인 필요",forbidden:"실행 불가"};
const reasons = {highest_expected_saving:"후보 중 비교값이 가장 높습니다.",expected_saving_beats_accepting_now:"지금 수락하는 것보다 비교값이 높습니다.",certain_saving_beats_best_counter:"확정 절감액이 역제안 비교값보다 큽니다.",no_valid_counter_offer:"진행 가능한 역제안이 남아 있지 않습니다.",nothing_beats_the_alternative:"대안 견적을 넘는 편익을 찾지 못했습니다.",exceeds_negotiator_limit:"담당자 한도를 넘어 내부 승인이 필요합니다.",acceptance_estimate_unavailable:"추정에 실패하여 값을 지어내지 않고 멈췄습니다."};
const finalSources = {backend_estimate:"추정 backend의 답",policy_prior:"정책의 라운드별 사전값",public_round_limit:"공개된 마지막 라운드"};
const approvalNotes = {synthetic_auto:"합성 자동 승인",synthetic_deny:"합성 자동 거절",human_dashboard:"화면에서 사람이 승인·거절",unspecified:"승인 방식 미기록"};
const TERMINAL = ["agreed", "ended_alternative", "supplier_walked"];
const reviewTexts = {
  acceptance_estimate_unavailable: ["추정 요청이 실패했습니다. 값을 지어내지 않고 멈췄습니다. 실패 기록은 그대로 남습니다.", "다시 추정 요청"],
  system1_unavailable: ["System 1 판단 요청이 실패했습니다. 진행하지 않고 멈췄습니다. 실패 기록은 그대로 남습니다.", "검토 후 다시 진행"],
  off_script_demand: ["공급사가 단가·납기·결제 밖의 새 요구를 붙였습니다. 정해진 절차만으로 진행하지 않고 사람에게 넘겼습니다.", "검토했음 · 협상 계속"],
  acceptance_unverified: ["공급사 답변이 보낸 조건 그대로의 수락으로 확인되지 않았습니다. 체결하지 않았습니다.", "검토했음 · 협상 계속"],
  final_offer_claim: ["공급사 발언이 최종 조건 선언이나 철수 예고로 읽혔습니다(System 1 판독). 역제안하면 결렬될 수 있어 보내지 않고 멈췄습니다. 계속하면 평소대로 판단합니다.", "검토했음 · 협상 계속"],
};
const share = v => v == null ? "—" : Number(v).toFixed(2);
function showSystem1(s1) {
  $("system1").hidden = !s1;
  if (!s1) return;
  $("system1-reader").textContent = s1.is_model ? "Zoro 판독" : "규칙 참조 · 모델 아님";
  $("system1-reader").className = `badge ${s1.is_model ? "estimate" : "calc"}`;
  $("system1-note").textContent = `공급사의 모든 제안을 진단하고(최종안·철수·추가 조정·지시문), 범위 밖 요구면 사람에게 넘기며, 수락 답변은 보낸 조건 그대로인지 검증합니다. 기준 ${s1.threshold}는 데이터로 맞춘 값이 아닙니다. 판단은 멈출 수만 있고 권한을 주지 않습니다.`;
  const items = s1.checks.slice(-6).reverse().map(c => {
    const routing = c.check === "routing";
    const stop = c.status === "error" || (routing ? c.route === "human_review" : !c.verified);
    const li = node("li", `system1-item ${stop ? "stop" : routing ? "" : "pass"}`), head = node("div", "system1-head");
    const result = c.status === "error" ? "응답 실패 · 멈춤" : routing ? (c.route === "human_review" ? "사람 검토" : c.reasons.includes("instruction_ignored") ? "지시문 감지 · 따르지 않음" : "계속") : (c.verified ? "체결" : "체결 보류");
    head.append(node("span", "", `R${c.round}`), node("span", `badge ${routing ? "calc" : "execute"}`, routing ? "진단·라우팅" : "수락 검증"),
      node("span", "muted small", c.status === "error" ? (c.error || "") : c.latency_ms == null ? "" : `${Math.round(c.latency_ms)}ms`), node("span", "system1-result", result));
    li.append(head);
    const values = node("div", "system1-values");
    const add = (label, value) => { const span = node("span", "", `${label} `); span.append(node("b", "", value)); values.append(span); };
    if (c.status === "skipped_no_text") add("문장 없음", "화면 조건 그대로 수락 버튼");
    else if (routing && c.status === "ok") { add("최종안", share(c.final)); add("철수", share(c.walkaway)); add("추가 조정", share(c.concession)); add("지시문", share(c.instruction)); add("범위 밖 요구", share(c.off_script)); }
    else if (!routing && c.status === "ok") { add("조건 그대로 수락", share(c.accept_exact)); if (c.new_condition != null) add("새 조건", share(c.new_condition)); }
    if (!routing) add("코드 숫자 대조", c.number_mismatches.length ? `불일치 ${c.number_mismatches.join(", ")}` : "일치");
    li.append(values);
    if (!routing && c.reply) li.append(node("p", "system1-reply", `“${c.reply}”`));
    return li;
  });
  $("system1-checks").replaceChildren(...items);
}
const AUTOPLAY_PAUSE_MS = 700, POLL_MS = 2000;
let view = null, liveId = null, mode = "simulated", replayStep = 0, busy = false, config = null, evidence = null;

async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "입력 값을 확인해 주세요. 숫자와 필수 항목이 올바른지 살펴보세요.");
  return data;
}
async function task(fn) {
  if (busy) return;
  busy = true; $("workspace").setAttribute("aria-busy", "true"); $("error").hidden = true;
  document.querySelectorAll("button, select").forEach(b => b.disabled = true);
  try { await fn(); $("notice").textContent = "화면이 업데이트되었습니다."; }
  catch (e) { $("error").textContent = e.message; $("error").hidden = false; }
  finally { busy = false; $("workspace").setAttribute("aria-busy", "false"); render(); }
}
function row(label, value) { const div = node("div"); div.append(node("dt", "", label), node("dd", "", value)); return div; }

function backendText(backend) {
  if (!backend) return "추정 backend 확인 중";
  if (!backend.is_model) return `${backend.kind} · CPU 참조`;
  return `${backend.kind} · 실시간 모델 호출`;
}
function liveNote() {
  const backend = view?.backend || config?.backend;
  if (!backend) return "";
  const who = mode === "human" ? "사람 공급사" : "합성 공급사";
  if (!backend.is_model) {
    const reads = backend.reads_supplier_messages ? "발언의 정해진 문구만 읽습니다" : "공급사 발언을 읽지 않습니다";
    return `${who} · 규칙 기반 CPU 참조(${backend.kind}) · 실제 Zoro 모델 추론이 아닙니다. 이 추정은 ${reads}.`;
  }
  if (backend.server_identity_verified_by_client === false) {
    const served = (backend.served_model_ids_reported_by_server || []).join(", ");
    return `${who} · 외부 판단 서버 실시간 호출(${backend.kind}) · 서버의 가중치와 엔진은 이 화면이 검증하지 못합니다.${served ? ` 서버가 보고한 모델: ${served}.` : ""} 수락 추정은 보정된 확률이 아닙니다.`;
  }
  return `${who} · 실시간 모델 호출(${backend.kind}) · 수락 추정은 보정된 현실 수락 확률이 아닙니다.`;
}

function showHistory(rows) {
  $("history").replaceChildren();
  rows.forEach((o, i) => {
    const li = node("li", i === rows.length - 1 ? "latest" : "");
    const meta = node("div", "history-meta");
    meta.append(node("span", "", `R${o.round}`), node("span", `badge ${o.by === "buyer" ? "estimate" : "calc"}`, o.by === "buyer" ? "우리" : "공급사"), node("span", "", o.offer_id || "전달"));
    li.append(meta, node("div", "history-terms", terms(o.terms)));
    if (o.message) li.append(node("p", "", o.message));
    $("history").append(li);
  });
}
function showScale(facts, auto) {
  const entries = [["목표",facts.target_total_cost],["담당자 한도",auto],["현재 제안",facts.standing_total_cost],["대안 견적",facts.alternative_total_cost]].filter(e => e[1] != null);
  const values = entries.map(e => e[1]); const low = Math.min(...values), high = Math.max(...values), span = Math.max(1, high - low);
  $("cost-scale").replaceChildren();
  entries.forEach(([label, value]) => {
    const marker = node("div", "scale-marker"); marker.style.left = `${8 + (value - low) / span * 84}%`;
    const text = node("span", "label", label); text.append(node("strong", "", won(value))); marker.append(text); $("cost-scale").append(marker);
  });
}
function showOptions(options, chosen) {
  $("options").replaceChildren();
  if (!options.length) { $("options").append(node("p", "empty", "‘다음 판단’을 누르면 비용·추정·선택지를 비교합니다.")); return; }
  options.forEach(o => {
    const selected = chosen && o.kind === chosen.kind && o.offer_id === chosen.offer_id;
    const item = node("div", `option${selected ? " selected" : ""}`), summary = node("div");
    summary.append(node("div", "option-title", `${selected ? "✓ " : ""}${kinds[o.kind] || actions[o.action] || o.kind}`), node("div", "option-terms", terms(o.terms)));
    if (o.authority) summary.append(node("span", `badge ${o.authority === "needs_approval" ? "approval-badge" : "calc"}`, authority[o.authority] || o.authority));
    const p = node("div", "probability", o.p_accept == null ? "확정값" : `추정 ${percent(o.p_accept)}`);
    if (o.p_accept != null) { const track = node("div", "track"), fill = node("div", "fill"); fill.style.width = `${Math.max(0, Math.min(100, o.p_accept * 100))}%`; track.append(fill); p.append(track); }
    const value = node("div", "option-value"); value.append(node("small", "", "비교값"), document.createTextNode(won(o.value)));
    item.append(summary, p, value); $("options").append(item);
  });
}
function showResult(replay) {
  const outcome = view.outcome || {}, done = !replay && TERMINAL.includes(view.status);
  $("result").hidden = !done;
  if (!done) return;
  const saving = outcome.realized_saving_vs_alternative ?? outcome.realized_saving;
  $("result-title").textContent = {agreed:"합의했습니다",ended_alternative:"대안 견적으로 끝냈습니다",supplier_walked:"공급사가 협상을 끝냈습니다"}[view.status];
  $("result-badge").textContent = view.status === "agreed" ? "합의" : "합의 없음";
  $("result-badge").className = `badge ${view.status === "agreed" ? "execute" : "calc"}`;
  $("result-values").replaceChildren(
    row("합의 조건", outcome.agreement ? terms(outcome.agreement) : "없음 · 대안 견적 사용"),
    row("대안 대비 실현 절감 · 계산", won(saving)),
    row("승인 카드 / 승인 / 거절", `${outcome.approval_cards} / ${outcome.approvals_granted} / ${outcome.approvals_denied}`),
    row("추정 요청", `${outcome.model_requests}회`),
    row("권한 위반", `${(outcome.authority_violations || []).length}건`));
}

function showEvidence() {
  const dev = evidence?.development;
  if (!dev) return;
  const b = dev.backend || {};
  $("evidence-scope").textContent = `합성 공급사 개발 분할 ${dev.cases}건 · ${dev.trials}회 실행 · ${b.model_id || "모델"} ${b.gpu_name || ""} ${b.dtype || ""} · 요청 ${dev.requests}회, 질문 ${dev.questions}개 · 평가 분할 미사용 · 합성 자동 승인`;
  const top = Math.max(...dev.policies.map(p => p.mean_realized_saving), 1);
  $("evidence-bars").replaceChildren(...dev.policies.map(p => {
    const line = node("div", `bar-row${p.evaluator_only ? " reference" : ""}`); line.setAttribute("role", "row");
    line.title = `${p.label}: 평균 실현 절감 ${won(p.mean_realized_saving)}, 과정 비용 차감 후 ${won(p.mean_net_saving)}`;
    const track = node("div", "bar-track"), fill = node("div", "bar-fill"); fill.style.width = `${Math.max(0, p.mean_realized_saving) / top * 100}%`; track.append(fill);
    const detail = `합의 ${p.feasible_agreements}/${p.feasible_n} · 승인 카드 ${p.approval_cards} · 권한 위반 ${p.authority_violations}`;
    line.append(node("div", "bar-label", p.label), track, node("div", "bar-value", won(p.mean_realized_saving)), node("div", "bar-detail", detail));
    return line;
  }));
  // The midpoint baseline is far below the others; it stays in the bars above so this scale remains readable.
  const rows = dev.differences.filter(d => d.ci90 && d.against !== "midpoint");
  const low = Math.min(0, ...rows.map(d => d.ci90[0])), high = Math.max(0, ...rows.map(d => d.ci90[1])), span = Math.max(1, high - low);
  const at = v => `${(v - low) / span * 100}%`;
  $("evidence-differences").replaceChildren(...rows.map(d => {
    const line = node("div", "interval-row"), plot = node("div", "interval-plot");
    const zero = node("span", "interval-zero"); zero.style.left = at(0);
    const range = node("span", "interval-range"); range.style.left = at(d.ci90[0]); range.style.width = `${(d.ci90[1] - d.ci90[0]) / span * 100}%`;
    const dot = node("span", "interval-dot"); dot.style.left = at(d.mean_difference);
    plot.append(zero, range, dot);
    const excludes = d.ci90[0] > 0 || d.ci90[1] < 0;
    const reading = d.evaluator_only ? "참고 상한과의 거리" : excludes ? (d.mean_difference > 0 ? "0 초과 · 개발 사례에서 우위" : "0 미만") : "0 포함 · 우위 불확실";
    line.append(node("div", "interval-label", `Zoro − ${d.label}`), plot,
      node("div", "interval-value", `${signedWon(d.mean_difference)} [${signedWon(d.ci90[0])}, ${signedWon(d.ci90[1])}]`),
      node("div", "interval-reading", `${reading} · ${d.wins}승 ${d.ties}무 ${d.losses}패`));
    return line;
  }));
  const s = evidence.official_sglang;
  const tiles = [
    ["권한 위반", `${dev.policies.reduce((n, p) => n + (p.authority_violations || 0), 0)}건`, "모든 정책 · 실행된 행동 전수 재검사"],
    ["대안보다 불리한 합의", `${dev.policies.reduce((n, p) => n + (p.agreements_not_better_than_alternative || 0), 0)}건`, "모든 정책"],
    ["판단 1회 지연 중앙값", dev.median_latency_ms == null ? "—" : `${(dev.median_latency_ms / 1000).toFixed(1)}초`, `${b.gpu_name || ""} native ${b.dtype || ""} · 요청당`],
  ];
  if (s) tiles.push(["공식 SGLang 대조", `${s.same_outcomes}/${s.development_cases}건 결과 일치`, `${s.platform} · SGLang ${s.sglang_version} · 확률 최대 차이 ${Number(s.probe_max_abs_difference).toFixed(4)} · 경로 ${s.same_decision_paths}/${s.development_cases}`]);
  $("evidence-tiles").replaceChildren(...tiles.map(([label, value, note]) => { const t = node("div", "tile"); t.append(node("span", "", label), node("strong", "", value), node("small", "", note)); return t; }));
  const probe = evidence.candidate_probe;
  $("evidence-probe").textContent = probe ? `후보 점수 실험(${probe.states}상태 · ${probe.questions}질문)의 사전 고정 판정은 C/D입니다. 상태 내 순위 정확도 ${Number(probe.within_state_pair_accuracy).toFixed(2)}(규칙 ${Number(probe.rule_within_state_pair_accuracy).toFixed(2)}), 한 단계 선택의 규칙 대비 차이 ${signedWon(probe.gain_vs_rule_mean)}. 모델 점수를 확률로 쓰지 않는 이유입니다.` : "";
}

function render() {
  const replay = mode === "replay", shown = mode === "evidence", live = !replay && !shown;
  ["simulated", "human", "restart", "refresh", "autoplay", "retry", "scenario", "replay-case", "replay-step"].forEach(id => $(id).disabled = false);
  $("replay-mode").disabled = !$("replay-case").options.length;
  $("evidence-mode").disabled = !evidence?.available;
  for (const key of ["simulated", "human"]) $(key).classList.toggle("selected", mode === key);
  $("replay-mode").classList.toggle("selected", replay); $("evidence-mode").classList.toggle("selected", shown);
  $("replay-controls").hidden = !replay; $("scenario-controls").hidden = mode !== "simulated";
  $("workspace").hidden = shown; $("evidence-panel").hidden = !shown;
  $("step").hidden = !live; $("restart").hidden = !live; $("autoplay").hidden = !live; $("refresh").hidden = mode !== "human";
  const backend = config?.backend;
  $("backend-chip").textContent = `추정: ${backendText(backend)}`;
  $("backend-chip").className = `backend-chip${backend?.is_model ? " model" : ""}`;
  $("footer-backend").textContent = config?.saving_sessions ? `추정: ${backendText(backend)} · 실시간 세션을 서버에 저장 중` : `추정: ${backendText(backend)}`;
  if (shown) { $("evidence").textContent = "저장소에 보존한 측정 요약 · 읽기 전용 · 지금 모델을 호출하지 않습니다."; $("scenario-card").hidden = true; showEvidence(); return; }
  if (!view) return;
  const records = view.records || [], record = replay ? records[replayStep] : records.at(-1);
  const facts = (replay ? record?.facts : view.facts) || {}, chosen = record?.chosen;
  const older = Boolean(!replay && record && record.facts.standing_offer_id !== facts.standing_offer_id);
  $("options-heading").textContent = older ? `직전 R${record.round} 선택지` : "이번 라운드 선택지";
  $("decision-heading").textContent = older ? `직전 R${record.round} 판단` : "이번 라운드 결정";
  $("step").disabled = view.status !== "active"; $("autoplay").disabled = view.status !== "active";
  const approvalNote = approvalNotes[view.approval_mode] || "승인 방식 미기록";
  $("evidence").textContent = replay
    ? `${String(view.evidence_class).startsWith("real_model_run") ? (view.evidence_class === "real_model_run" ? "실제 모델 실행 기록" : "실제 모델 + 규칙 조합 실행 기록") : "규칙·CPU 참조 실행 기록"} · 읽기 전용 재생 · ${approvalNote} · 현재 모델을 호출하지 않습니다.${view.complete === false ? " 미완료 실행 기록입니다." : ""}`
    : liveNote();
  const scenario = live ? view.scenario : null;
  $("scenario-card").hidden = !scenario;
  if (scenario) { $("scenario-title").textContent = scenario.title; $("scenario-summary").textContent = scenario.summary; $("scenario-watch").textContent = `볼 것 · ${scenario.watch}`; if ($("scenario").value !== scenario.key) $("scenario").value = scenario.key; }
  $("round").textContent = replay ? `판단 ${replayStep + 1} / ${records.length}` : `라운드 ${view.round} / ${view.max_rounds}`;
  $("item").textContent = replay ? `${view.case_id} · 완료된 공개 이력` : `${view.item} · ${view.quantity.toLocaleString("ko-KR")}개`;
  showHistory(view.transcript || []);
  $("supplier-link").hidden = !view.supplier_url; if (view.supplier_url) $("supplier-link").href = view.supplier_url;
  $("standing-title").textContent = `${facts.standing_offer_id || "현재 제안"} ${replay ? "기록" : "현재 조건"} 평가`;
  $("cost").textContent = won(facts.standing_total_cost); $("saving").textContent = won(facts.standing_saving_vs_alternative);
  $("saving").className = facts.standing_saving_vs_alternative >= 0 ? "positive" : "negative";
  const gap = facts.standing_total_cost == null ? null : facts.standing_total_cost - facts.target_total_cost;
  $("target-gap").textContent = gap == null ? "—" : `${won(Math.abs(gap))} ${gap > 0 ? "초과" : "이내"}`; $("target-gap").className = gap > 0 ? "negative" : "positive";
  $("authority").textContent = authority[facts.standing_authority] || "—"; $("authority").className = facts.standing_authority === "auto" ? "positive" : "warning";
  showScale(facts, view.auto_bounds?.max_total_cost); showOptions(record?.options || [], chosen);
  $("option-count").textContent = record ? `${record.options.length}개 비교` : "판단 대기";
  $("status").textContent = replay ? `저장된 판단 ${replayStep + 1} · 최종 결과: ${statuses[view.outcome?.status] || view.outcome?.status || "—"}` : older ? `새 제안 ${facts.standing_offer_id} 도착 · 다음 판단 대기` : statuses[view.status] || view.status;
  $("action-title").textContent = record ? (chosen ? kinds[chosen.kind] || actions[record.action] : actions[record.action]) : "다음 판단을 시작하세요";
  $("chosen-terms").replaceChildren();
  if (chosen?.terms) {
    [["단가", won(chosen.terms.unit_price)], ["납기", `${chosen.terms.delivery_days}일`], ["결제", chosen.terms.payment_days === 0 ? "납품 즉시" : `${chosen.terms.payment_days}일`]].forEach(([label, value]) => {
      const cell = node("div", "term"); cell.append(node("span", "", label), document.createTextNode(value)); $("chosen-terms").append(cell);
    });
  }
  $("reasons").replaceChildren(...(record?.reasons || []).map(r => node("p", "", `• ${reasons[r] || r}`)));
  $("estimates").replaceChildren(); $("estimates").hidden = !record;
  if (record) {
    const source = finalSources[record.facts.p_final_offer_source];
    $("estimates").append(
      row(`상대 제안이 최종안일 가능성 · 추정${source ? ` (${source})` : ""}`, percent(record.facts.p_final_offer)),
      row("대안 대비 절감 · 계산", won(chosen?.saving)), row("선택지 비교값 · 추정", won(chosen?.value)));
  }
  const m = record?.model || {};
  const metaParts = [m.backend || "추정 없음"];
  if (m.questions) metaParts.push(`질문 ${m.questions}개`);
  if (m.cached) metaParts.push("앞선 응답 재사용"); else if (m.latency_ms != null) metaParts.push(`응답 ${m.latency_ms >= 1000 ? `${(m.latency_ms / 1000).toFixed(1)}초` : `${Math.round(m.latency_ms)}ms`}`);
  $("model-meta").textContent = record ? `${metaParts.join(" · ")} · 수락 추정은 보정된 실제 협상 확률이 아닙니다.${m.error ? ` 오류: ${m.error}` : ""}` : "";
  $("review").hidden = !(live && view.status === "needs_review");
  const review = reviewTexts[view.review_reason] || reviewTexts.acceptance_estimate_unavailable;
  $("review-text").textContent = review[0]; $("retry").textContent = review[1];
  showSystem1(replay ? null : view.system1);
  showResult(replay);
  const pending = replay ? record?.approval_request : view.pending;
  $("approval").hidden = !pending; $("approval-actions").hidden = replay; $("approval-recorded").hidden = !replay;
  $("approval-recorded").textContent = `저장된 당시 승인 카드입니다. ${approvalNote} 기록이며 재생에서는 승인할 수 없습니다.`;
  $("approve").disabled = false; $("deny").disabled = false;
  if (pending) {
    $("approval-summary").textContent = `${pending.kind === "accept_offer" ? "현재 제안 수락" : "역제안 전송"}에 필요한 승인입니다.`;
    $("approval-values").replaceChildren(row("승인할 조건", terms(pending.terms)), row("환산 총비용", won(pending.total_cost)), row("대안 대비 절감", won(pending.saving_vs_alternative)));
    pending.exceeded_auto_limits.forEach(e => $("approval-values").append(row(`한도 초과 · ${{payment_days:"결제일",delivery_days:"납기",unit_price:"단가",total_cost:"총비용"}[e.field] || e.field}`, `${e.value.toLocaleString("ko-KR")} / 한도 ${e.auto_limit.toLocaleString("ko-KR")}`)));
    $("approval-scope").textContent = `제안 ${pending.offer_id} · 조건 ${pending.terms_version}. 이 조건에만 적용됩니다. 승인 후 ‘다음 판단’에서 다시 권한을 확인합니다.`;
  }
  const sent = replay ? (view.transcript || []).filter(t => t.by === "buyer").map(t => ({text: t.message})) : view.outbox || [];
  $("sent-count").textContent = `${sent.length}건`; $("sent-message").textContent = sent.at(-1)?.text || "아직 전달한 문장이 없습니다.";
}

async function start(selected, preset = "start") {
  mode = selected;
  view = await api("/api/sessions", {mode, preset, scenario: $("scenario").value || config.default_scenario});
  liveId = view.id;
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
$("simulated").onclick = () => task(() => start("simulated"));
$("human").onclick = () => task(() => start("human"));
$("restart").onclick = () => task(() => start(mode));
$("scenario").onchange = () => task(() => start("simulated"));
$("refresh").onclick = () => task(async () => { view = await api(`/api/sessions/${liveId}`); });
$("step").onclick = () => task(async () => { view = await api(`/api/sessions/${liveId}/step`, {revision: view.revision}); });
$("retry").onclick = () => task(async () => { view = await api(`/api/sessions/${liveId}/retry`, {revision: view.revision}); });
// Runs visible steps until a person is needed: an approval card, a failed estimate, the other party, or the end.
$("autoplay").onclick = () => task(async () => {
  for (let i = 0; i < 12 && view.status === "active"; i++) {
    view = await api(`/api/sessions/${liveId}/step`, {revision: view.revision});
    render(); document.querySelectorAll("button, select").forEach(b => b.disabled = true);
    if (view.status === "active") await pause(AUTOPLAY_PAUSE_MS);
  }
});
for (const decision of ["approve", "deny"]) $(decision).onclick = () => task(async () => {
  const p = view.pending;
  view = await api(`/api/sessions/${liveId}/approval`, {revision: view.revision, request_id: p.request_id, terms_version: p.terms_version, decision});
});
async function selectReplay() {
  mode = "replay"; view = await api(`/api/replays/${$("replay-case").value}`); replayStep = 0;
  $("replay-step").replaceChildren(...view.records.map((r, i) => { const o = node("option", "", `${i + 1}. R${r.round} ${actions[r.action] || r.action}`); o.value = i; return o; }));
}
$("replay-mode").onclick = () => task(selectReplay); $("replay-case").onchange = () => task(selectReplay);
$("replay-step").onchange = () => { replayStep = Number($("replay-step").value); render(); };
$("evidence-mode").onclick = () => { if (busy) return; mode = "evidence"; render(); $("evidence-panel").focus(); };
// While the other party is typing, pick up their answer without a manual refresh.
setInterval(async () => {
  if (busy || mode !== "human" || !liveId || !view || !["awaiting_supplier", "awaiting_info"].includes(view.status)) return;
  try { const next = await api(`/api/sessions/${liveId}`); if (!busy && mode === "human" && next.id === liveId && next.revision !== view.revision) { view = next; render(); $("notice").textContent = "공급사 응답이 도착했습니다."; } } catch (e) { /* the next poll retries */ }
}, POLL_MS);
task(async () => {
  config = await api("/api/config");
  $("scenario").replaceChildren(...config.scenarios.map(s => { const o = node("option", "", s.title); o.value = s.key; return o; }));
  $("scenario").value = config.default_scenario;
  if (config.evidence_available) evidence = await api("/api/evidence");
  const result = await api("/api/replays");
  $("replay-case").replaceChildren(...result.items.map(r => { const o = node("option", "", `${r.case_id}${r.evidence_class === "real_model_run" ? " · 실제 모델" : String(r.evidence_class).startsWith("real_model_run") ? " · 실제 모델+규칙 조합" : " · CPU"}`); o.value = r.index; return o; }));
  $("replay-availability").textContent = result.items.length ? `연결된 실행 기록 ${result.items.length}건 · 읽기 전용` : "실행 기록: 연결되지 않음";
  // A model backend starts at round 1 so that every model call happens in front of the audience.
  await start("simulated", config.backend.is_model ? "start" : "approval");
});
