"use strict";
// NAMI product prototype. Same session API as the analysis dashboard.
// The browser may choose a fixed scenario, type the supplier's side, and set our own criteria;
// backend, paths and seeds stay on the operator's command line.

const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
const num = v => Math.round(v).toLocaleString("ko-KR");
const won = v => v == null ? "—" : `${num(v)}원`;
const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
const pay = d => d === 0 ? "즉시 결제" : `결제 ${d}일`;
const termsText = t => t ? `${num(t.unit_price)}원 · ${t.delivery_days}일 · ${pay(t.payment_days)}` : "대안 견적";
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const BEAT_MS = 450, AFTER_TURN_MS = 600, POLL_MS = 2000;
const TERMINAL = ["agreed", "ended_alternative", "supplier_walked"];
const LOGO = '<img src="/static/nami-logo.png" alt="">';

// Synthetic supplier names, display only.
const SIM_NAMES = { payment_tradeoff: "나래물산", false_approval_claim: "다온상사", final_offer_signal: "새봄물산", no_deal: "한결테크", within_authority: "누리공업" };
const KIND = { standing_offer: "제안 수락", same_terms: "같은 조건", pay_sooner: "결제 앞당김", deliver_sooner: "납기 앞당김",
  pay_and_deliver_sooner: "납기·결제 앞당김", usual_payment: "기본 결제", softer_same_terms: "양보안", take_alternative: "대안 선택" };
const REASON = {
  highest_expected_saving: "기대 절감이 가장 큼", expected_saving_beats_accepting_now: "지금 수락보다 기대 절감이 큼",
  certain_saving_beats_best_counter: "확정 절감이 더 큼", no_valid_counter_offer: "보낼 역제안이 없음",
  nothing_beats_the_alternative: "대안보다 나은 조건 없음", exceeds_negotiator_limit: "내 한도 초과",
  acceptance_estimate_unavailable: "추정 실패", counter_withheld_by_reviewer: "검토자가 역제안 없이 진행하도록 정함",
};
const LIMIT = {
  payment_days: (v, l) => `결제 ${v === 0 ? "즉시" : `${v}일`} (기준 ${l}일 이상)`,
  delivery_days: (v, l) => `납기 ${v}일 (기준 ${l}일 이하)`,
  unit_price: (v, l) => `단가 ${num(v)}원 (기준 ${num(l)}원 이하)`,
  total_cost: (v, l) => `환산 총비용 ${num(v)}원 (기준 ${num(l)}원 이하)`,
};
const limitText = e => (LIMIT[e.field] || ((v, l) => `${e.field} ${v}/${l}`))(e.value, e.auto_limit);
const STATUS = {
  active: ["NAMI 차례", "live"], awaiting_approval: ["승인 대기", "warn"], awaiting_supplier: ["공급사 차례", ""],
  awaiting_info: ["공급사 차례", ""], needs_review: ["검토 필요", "danger"], agreed: ["합의", "ok"],
  ended_alternative: ["대안으로 종료", ""], supplier_walked: ["공급사 종료", "danger"],
};

// Why NAMI stopped for a person. Every reason gets its own words: the panel must not claim an engine
// failure when the engine answered and the stop is about the supplier's offer.
const ENGINE_DOWN = "엔진 응답 없음 · 오른쪽에서 다시 요청";
const REVIEW = {
  acceptance_estimate_unavailable: { tag: "엔진 응답 없음", cls: "danger", title: "값을 지어내지 않고 멈췄어요", button: "다시 판단 요청", status: ENGINE_DOWN,
    text: () => "수락 추정을 받지 못했습니다. 아무것도 보내지 않았습니다." },
  system1_unavailable: { tag: "엔진 응답 없음", cls: "danger", title: "공급사 메시지를 읽지 못해 멈췄어요", button: "다시 읽기 요청", status: ENGINE_DOWN,
    text: () => "메시지 판독(System 1)에 실패했습니다. 다시 요청하면 이 메시지를 다시 읽은 뒤 판단합니다." },
  off_script_demand: { tag: "범위 밖 요구", cls: "warn", title: "단가·납기·결제 밖의 요구가 있어요", button: "확인했음 · 계속 진행", status: "범위 밖 요구 · 오른쪽에서 확인",
    text: () => "NAMI는 이 요구를 협상하지 않습니다. 계속하면 세 조건만 다루고, 그 요구는 합의에 포함하지 않는다고 공급사에 밝힙니다." },
  implausible_offer: { tag: "이상한 조건", cls: "danger", title: "정상 견적과 너무 다른 조건이에요", button: "확인했음 · 이 조건으로 계속", status: "이상한 조건 · 오른쪽에서 확인",
    text: (v, offer) => `공급사 제안 ${termsText(offer)} · 대안 견적 ${num(settingsFrom(v).alternative_unit_price)}원. 오타일 가능성이 큽니다. 계속하면 NAMI가 이 숫자 그대로 판단하고, 내 한도 안이면 바로 수락할 수 있습니다.` },
  acceptance_unverified: { tag: "수락 확인 안 됨", cls: "warn", title: "보낸 조건 그대로의 수락인지 확인되지 않았어요", button: "확인했음 · 계속 진행", status: "수락 확인 안 됨 · 오른쪽에서 확인",
    text: v => acceptanceProblem(v) },
  final_offer_claim: { tag: "최종안 신호", cls: "warn", title: "공급사가 마지막 조건이라고 했어요", button: "평소대로 계속 · 역제안 가능", status: "최종안 신호 · 오른쪽에서 선택",
    text: (v, offer) => `공급사 제안(${termsText(offer)})이 최종 조건이나 철수 예고로 읽혔습니다(System 1). 역제안하면 결렬될 수 있어 보내지 않았습니다. `
      + "역제안 없이 진행하면 이 제안의 수락과 종료만 비교하고, 한도를 넘으면 승인을 요청합니다.", withhold: true },
  supplier_reply_without_terms: { tag: "응답 확인 필요", cls: "warn", title: "공급사 응답에 조건이 없어요", button: "확인했음 · 계속 진행", status: "응답 확인 필요 · 오른쪽에서 확인",
    text: () => "역제안인데 단가·납기·결제가 없어 판단하지 않았습니다." },
};
// Why an acceptance was not recorded, in the supplier's own words. "수락하되 운송비 별도" is not a clean accept:
// the reader flags the new condition, and the person sees the sentence that carried it.
const VERIFY_REASON = { reader_says_new_condition: "수락하면서 새 조건(예: 운송비 별도, 최소 수량)을 붙인 것으로 읽혔습니다",
  reader_says_terms_changed: "보낸 조건과 다른 조건으로 수락한 것으로 읽혔습니다", code_number_mismatch: "답장 속 숫자가 보낸 조건과 다릅니다" };
const PRICE_CHANGING = /(운송비|배송비|운임|물류비|부가세|VAT|포장비|설치비|수수료)[^.,]{0,8}?(별도|불포함|미포함|추가|부담)/i;
function acceptanceProblem(v) {
  const check = [...(v.system1?.checks || [])].reverse().find(c => c.check === "verification");
  const why = (check?.reasons || []).map(r => VERIFY_REASON[r]).filter(Boolean);
  const said = check?.reply ? ` 공급사 원문: “${check.reply}”` : "";
  // A charge put outside the unit price changes the real cost even when the three numbers match.
  const extra = (check?.reply || "").match(PRICE_CHANGING);
  const flag = extra ? ` 가격을 바꾸는 문구: “${extra[0].trim()}” · 총비용이 달라질 수 있습니다.` : "";
  return `${why.length ? why.join(" · ") + "." : "보낸 조건 그대로인지 확인되지 않았습니다."}${said}${flag} 합의를 기록하지 않았습니다. 계속하면 NAMI가 다시 판단합니다.`;
}
const reviewInfo = v => REVIEW[v.review_reason] || { tag: "검토 필요", cls: "warn", title: "사람 확인이 필요해 멈췄어요", button: "확인했음 · 계속 진행",
  status: "검토 필요 · 오른쪽에서 확인", text: () => v.review_reason || "" };
// What NAMI sent that is not an offer: questions back to the supplier. They are only in the outbox, so
// without this the supplier's message seems to vanish when NAMI asks instead of answering.
const NOTE_LABEL = { ask_info: "확인 요청", clarify: "확인 요청", clarify_offer: "숫자 확인" };
const OFFER_ACTIONS = ["counter", "accept_terms_only"];
function notesByOffer(v) {   // notes keyed by how many buyer offers had been sent before them
  const notes = new Map(); let sent = 0;
  (v.outbox || []).forEach((o, i) => {
    if (OFFER_ACTIONS.includes(o.action)) sent += 1;
    else if (NOTE_LABEL[o.action]) notes.set(sent, [...(notes.get(sent) || []), { key: `note-${i}`, text: o.text, label: NOTE_LABEL[o.action] }]);
  });
  return notes;
}

let config = null, evidence = null, busy = false, thinking = false, current = null, typedCount = 0;
const threads = [];   // {sessionId, mode:"typed"|"simulated", scenarioKey, name, view, seen, expanded, prefilled}

async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) && d[0]?.msg ? String(d[0].msg).replace(/^Value error, /, "") : "요청을 처리하지 못했습니다.");
  }
  return data;
}
const engine = () => config?.backend?.is_model ? "Zoro-1" : "규칙 추정";

// ---------- timeline pieces ----------
function chips(terms, hot = []) {
  const box = el("div", "chips");
  if (!terms) return box;
  [["unit_price", `${num(terms.unit_price)}원`], ["delivery_days", `납기 ${terms.delivery_days}일`], ["payment_days", pay(terms.payment_days)]]
    .forEach(([field, text]) => box.append(el("span", `chip${hot.includes(field) ? " hot" : ""}`, text)));
  return box;
}
function supplierMsg(row, t) {
  const m = el("div", "msg"), b = el("div", "bubble"), who = el("div", "who");
  who.append(el("strong", "", t.name), document.createTextNode(row.offer_id));
  b.append(who, chips(row.terms));
  if (row.message) b.append(el("p", "", row.message));
  m.append(el("div", "avatar", t.name.slice(0, 1)), b);
  return m;
}
function ourMsg(text, label, pendingTerms, hot) {
  const m = el("div", `msg ours${pendingTerms ? " pending" : ""}`), b = el("div", "bubble"), who = el("div", "who");
  who.append(el("strong", "", "NAMI"), document.createTextNode(label));
  b.append(who);
  if (pendingTerms) b.append(chips(pendingTerms, hot)); else b.append(el("p", "", text));
  const avatar = el("div", "avatar"); avatar.innerHTML = LOGO;   // constant markup, no data
  m.append(avatar, b);
  return m;
}
function event(text, cls = "") { const e = el("div", `event ${cls}`); e.append(el("span", "", text)); return e; }

function step(label, sub, cls) {
  const li = el("li", `step ${cls || ""}`), l = el("div", "step-label", label), body = el("div");
  l.append(el("small", "", sub)); li.append(l, body);
  return { li, body };
}
function decisionCard(t, r, i, latest) {
  const m = r.model || {}, f = r.facts || {}, chosen = r.chosen, failed = r.action === "human_review";
  // No counter-offer to compare (a withheld counter, nothing left to send): the engine was not asked at all.
  const asked = failed || !!m.backend, source = !asked ? "추정 없음" : m.is_model ? "Zoro-1" : "규칙 추정";
  const key = `d-${i}`, open = latest || t.expanded.has(key);
  const card = el("section", `decision${open ? "" : " collapsed"}`), head = el("div", "decision-head");
  const summary = failed ? "판단 보류" : chosen ? (chosen.kind === "standing_offer" ? "제안 수락" : chosen.kind === "take_alternative" ? "협상 종료" : `${KIND[chosen.kind] || ""} 역제안`) : r.action;
  head.append(el("strong", "", `NAMI 판단 · R${r.round}`));
  if (!open) head.append(el("span", "muted small", `→ ${summary}`));
  const lat = m.cached ? "앞선 답 재사용" : m.latency_ms == null ? "" : m.latency_ms >= 1000 ? `${(m.latency_ms / 1000).toFixed(1)}초` : `${Math.round(m.latency_ms)}ms`;
  head.append(el("span", "meta", [source, lat].filter(Boolean).join(" · ")));
  if (!latest) {
    const b = el("button", "ghost", open ? "접기" : "보기"); b.type = "button";
    b.onclick = () => { open ? t.expanded.delete(key) : t.expanded.add(key); render(false); };
    head.append(b);
  }
  card.append(head);
  const steps = el("ol", "steps"), parts = [];

  const s1 = step("계산", "코드");
  const saving = f.standing_saving_vs_alternative;
  s1.body.append(el("p", "", f.standing_total_cost == null ? "필수 납기(30일)를 넘어 받을 수 없는 조건"
    : `환산 ${won(f.standing_total_cost)} · `));
  if (saving != null) s1.body.firstChild.append(el("span", saving >= 0 ? "positive" : "negative", saving >= 0 ? `대안보다 ${won(saving)} 저렴` : `대안보다 ${won(-saving)} 비쌈`));
  parts.push(s1.li);

  const s2 = step("추정", !asked ? "요청 안 함" : m.is_model ? "Zoro-1" : "규칙·모델 아님", failed ? "danger" : "zoro");
  if (failed) {
    s2.body.append(el("p", "negative", `${engine()} 응답 없음 → 값을 만들지 않고 멈춤`));
    if (m.error) s2.body.append(el("p", "mono small muted", m.error));
  } else if (!asked) {
    s2.body.append(el("p", "muted small", "비교할 역제안이 없어 추정을 요청하지 않았습니다. 공급사 제안의 수락과 종료만 코드로 비교했습니다."));
  } else {
    s2.body.append(el("p", "muted small", "공급사가 그대로 받을까?"));
    const list = el("div", "estimates");
    (r.options || []).filter(o => o.p_accept != null).forEach(o => {
      const isChosen = chosen && o.kind === chosen.kind && o.offer_id === chosen.offer_id;
      const line = el("div", `est${isChosen ? " chosen" : ""}`), name = el("span", "", `${isChosen ? "✓ " : ""}${KIND[o.kind] || o.kind}`);
      name.append(el("span", "t", `${num(o.terms.unit_price)} · ${o.terms.delivery_days}일 · ${o.terms.payment_days === 0 ? "즉시" : `${o.terms.payment_days}일`}`));
      const bar = el("div", "bar"), fill = el("i"); fill.style.width = `${Math.max(1, Math.min(100, o.p_accept * 100))}%`; bar.append(fill);
      line.append(name, bar, el("span", "pct", pct(o.p_accept)));
      list.append(line);
    });
    s2.body.append(list);
    const foot = el("div", "est-foot");
    if (f.p_final_offer != null) foot.append(el("span", "", `마지막 제안일 가능성 ${pct(f.p_final_offer)}${f.p_final_offer_source && f.p_final_offer_source !== "backend_estimate" ? " (정책 사전값)" : ""}`));
    if (m.asked?.length) {
      const d = el("details", "asked"); d.append(el("summary", "", `보낸 질문 ${m.asked.length}개`));
      const ol = el("ol"); m.asked.forEach(q => ol.append(el("li", "", q.text))); d.append(ol); foot.append(d);
    }
    foot.append(el("span", "", "보정된 확률 아님"));
    s2.body.append(foot);
  }
  parts.push(s2.li);

  const s3 = step("결정", "코드");
  if (failed) s3.body.append(el("p", "", "보류"));
  else {
    const what = chosen?.kind === "standing_offer" ? "공급사 제안 수락" : chosen?.kind === "take_alternative" || r.action === "end" ? "협상 종료 · 대안 견적 사용" : `${KIND[chosen?.kind] || ""} 역제안 · ${termsText(chosen?.terms)}`;
    s3.body.append(el("p", "", what));
    const why = (r.reasons || []).map(x => REASON[x] || x).filter(x => x !== "내 한도 초과")[0];
    if (why) s3.body.append(el("p", "muted small", why));
  }
  parts.push(s3.li);

  let s4;
  if (failed) { s4 = step("권한", "코드", "danger"); s4.body.append(el("p", "", "아무것도 보내지 않음")); }
  else if (r.action === "request_approval") {
    s4 = step("권한", "코드", "warn");
    s4.body.append(el("p", "", "내 한도 초과 → 승인 요청"));
    (r.approval_request?.exceeded_auto_limits || []).forEach(e => s4.body.append(el("p", "warning small", limitText(e))));
  } else if (r.action === "end") { s4 = step("권한", "코드", "ok"); s4.body.append(el("p", "", "대안보다 나은 조건 없음 → 종료")); }
  else { s4 = step("권한", "실행", "ok"); s4.body.append(el("p", "", chosen?.approved ? "승인된 조건 → 재검사 후 전송" : "내 한도 안 → 재검사 후 전송")); }
  parts.push(s4.li);

  parts.forEach(p => steps.append(p));
  card.append(steps);
  return { node: card, parts: open ? parts : [] };
}

function buildItems(t) {
  const v = t.view, records = v.records || [], items = [], used = new Set();
  const push = (key, node, parts = []) => items.push({ key, node, parts });
  const notes = notesByOffer(v), addNotes = n => (notes.get(n) || []).forEach(x => push(x.key, ourMsg(x.text, x.label)));
  let lastRound = 0, sent = 0;
  const addRecord = (r, i) => {
    used.add(i);
    const card = decisionCard(t, r, i, i === records.length - 1);
    push(`d-${i}`, card.node, card.parts);
    if (r.action === "request_approval" && r.approval_request) {
      const log = (v.approval_log || []).find(a => a.request_id === r.approval_request.request_id);
      if (log) push(`ap-${i}`, log.outcome === "approved" ? event("승인됨 · 이 조건에만 적용", "ok") : event("거절됨 · 보내지 않음", "warn"));
    }
    if (r.action === "human_review" && (v.review_log || []).some(x => x.decision_index === i)) push(`rv-${i}`, event("다시 판단 요청"));
  };
  addNotes(0);
  (v.transcript || []).forEach(row => {
    if (row.round !== lastRound) { push(`sep-${row.round}`, el("div", "round-sep", `라운드 ${row.round}`)); lastRound = row.round; }
    if (row.by === "supplier") {
      push(`s-${row.offer_id}`, supplierMsg(row, t));
      records.forEach((r, i) => { if (!used.has(i) && r.facts?.standing_offer_id === row.offer_id) addRecord(r, i); });
    } else {
      push(`b-${row.offer_id || `end-${row.round}`}`, ourMsg(row.message, row.offer_id ? row.offer_id : "확정"));
      if (row.offer_id) addNotes(++sent);
    }
  });
  records.forEach((r, i) => { if (!used.has(i)) addRecord(r, i); });
  if (v.status === "needs_review") { const info = reviewInfo(v); push(`review-${v.revision}`, event(`${info.tag} · 사람 확인 대기`, info.cls)); }
  if (v.pending) push(`pend-${v.pending.request_id}`, ourMsg("", "승인 대기 · 아직 안 보냄", v.pending.terms, v.pending.exceeded_auto_limits.map(e => e.field)));
  const lastSent = (v.outbox || []).at(-1);
  if (v.status === "agreed") push("end-agreed", event(lastSent?.action === "counter" ? "공급사 수락 · 합의 완료" : "합의 완료", "ok"));
  if (v.status === "supplier_walked") push("end-walked", event("공급사가 협상 종료", "danger"));
  if (v.status === "ended_alternative") push("end-alt", event("대안 견적으로 종료"));
  if (thinking) push(`thinking-${v.revision}`, event(`NAMI 판단 중 · ${engine()} 호출`, "thinking"));
  return items;
}

// What the supplier sees: the public messages only. Their own on the right, the buyer's on the left.
function buildSupplierItems(t) {
  const v = t.view, items = [];
  const push = (key, node, buyerSide = false) => items.push({ key, node, buyerSide });
  const bubble = (mine, text, terms, label) => {
    const m = el("div", `msg ${mine ? "mine" : "theirs"}`), b = el("div", "bubble"), who = el("div", "who");
    who.append(el("strong", "", mine ? t.name : "구매사"), document.createTextNode(label || ""));
    b.append(who);
    if (terms) b.append(chips(terms));
    if (text) b.append(el("p", "", text));
    const avatar = el("div", "avatar");
    if (mine) avatar.textContent = t.name.slice(0, 1); else avatar.innerHTML = LOGO;   // constant markup
    m.append(avatar, b); return m;
  };
  const notes = notesByOffer(v), addNotes = n => (notes.get(n) || []).forEach(x => push(x.key, bubble(false, x.text, null, ""), true));
  let lastRound = 0, sent = 0;
  addNotes(0);
  (v.transcript || []).forEach(row => {
    if (row.round !== lastRound) { push(`sep-${row.round}`, el("div", "round-sep", `라운드 ${row.round}`)); lastRound = row.round; }
    if (row.by === "supplier") push(`s-${row.offer_id}`, bubble(true, row.message, row.terms, row.offer_id));
    else {
      push(`b-${row.offer_id || `end-${row.round}`}`, bubble(false, row.message, null, ""), true);
      if (row.offer_id) addNotes(++sent);
    }
  });
  if (v.status === "agreed") push("end-agreed", event("합의 완료", "ok"), true);
  if (v.status === "supplier_walked") push("end-walked", event("협상을 끝냈습니다"));
  if (["active", "awaiting_approval", "needs_review"].includes(v.status) && (v.transcript || []).length) push(`typing-${v.revision}`, event("구매사가 검토 중…", "thinking"));
  return items;
}

// ---------- render ----------
function render(animate = true) {
  renderThreadSelect();
  if (!current) return 0;
  const t = current, v = t.view;
  const code = t.name.charCodeAt(t.name.length - 1) - 0xac00;      // 과/와 by the last syllable's final consonant
  $("thread-title").textContent = `${t.name}${code >= 0 && code < 11172 && code % 28 ? "과" : "와"} 협상`;
  $("sup-name").textContent = t.name;
  $("thread-sub").textContent = t.mode === "typed" ? "SKU-1000 · 1,000개" : `시뮬레이션 · ${v.scenario?.title || ""}`;
  $("round-dots").replaceChildren(...Array.from({ length: v.max_rounds }, (_, i) => el("li", i + 1 < v.round ? "done" : i + 1 === v.round ? "now" : "", String(i + 1))));
  const [label, cls] = STATUS[v.status] || [v.status, ""];
  $("thread-status").textContent = label; $("thread-status").className = `status-pill ${cls}`;

  const items = buildItems(t);
  let delay = 0; const reveal = [], shownAt = new Map();
  const nodes = items.map(item => {
    if (animate && !t.seen.has(item.key)) {
      shownAt.set(item.key, delay);
      item.node.classList.add("enter"); item.node.style.setProperty("--d", `${delay}ms`); reveal.push([item.node, delay]); delay += BEAT_MS;
      item.parts.forEach(p => { p.classList.add("enter"); p.style.setProperty("--d", `${delay}ms`); reveal.push([p, delay]); delay += BEAT_MS; });
    }
    if (!item.key.startsWith("thinking")) t.seen.add(item.key);
    return item.node;
  });
  if (!nodes.length) {
    const empty = el("div", "empty-state");
    empty.append(el("strong", "", t.mode === "typed" ? "공급사의 첫 제안을 기다리는 중" : "NAMI에게 맡겨 보세요"),
      document.createTextNode(t.mode === "typed" ? "왼쪽 공급사 화면에서 제안이 오면 NAMI가 판단해 답합니다." : ""));
    nodes.push(empty);
  }
  $("timeline").replaceChildren(...nodes);
  if (reveal.length) reveal.forEach(([node, ms]) => setTimeout(() => node.scrollIntoView({ block: "end", behavior: "smooth" }), ms + 60));
  else $("timeline").scrollTop = $("timeline").scrollHeight;

  // Supplier screen: each message appears when the same message appears on our side, so both screens
  // follow the negotiation order (our reply first, then the supplier's next offer). One step returns
  // both at once; without this the supplier's next offer showed before our reply had arrived.
  const supItems = buildSupplierItems(t), supNodes = [];
  supItems.forEach(item => {
    const key = `sup:${item.key}`;
    if (animate && !t.seen.has(key)) {
      const d = shownAt.has(item.key) ? shownAt.get(item.key) : item.key.startsWith("typing") ? delay : 0;
      item.node.classList.add("enter"); item.node.style.setProperty("--d", `${d}ms`);
      setTimeout(() => item.node.scrollIntoView({ block: "end", behavior: "smooth" }), d + 60);
    }
    if (!item.key.startsWith("typing")) t.seen.add(key);
    supNodes.push(item.node);
  });
  if (!supNodes.length) {
    const empty = el("div", "empty-state");
    empty.append(el("strong", "", t.mode === "typed" ? "구매사에 첫 제안을 보내세요" : "시뮬레이션 공급사"),
      document.createTextNode(t.mode === "typed" ? "단가·납기·결제와 메시지를 입력합니다." : "자동으로 제안합니다."));
    supNodes.push(empty);
  }
  $("sup-timeline").replaceChildren(...supNodes);
  if (!animate) $("sup-timeline").scrollTop = $("sup-timeline").scrollHeight;

  renderComposer(t);
  renderSide(t);
  return delay;
}

function renderThreadSelect() {
  const waiting = threads.filter(t => t.view.status === "awaiting_approval").length;
  $("approvals-pill").hidden = !waiting; $("approvals-pill").textContent = `승인 대기 ${waiting}`;
  const select = $("thread-select");
  select.replaceChildren(...threads.map((t, i) => {
    const o = el("option", "", `${t.name} · ${t.mode === "typed" ? "직접 입력" : "시뮬레이션"} · ${(STATUS[t.view.status] || [t.view.status])[0]}`);
    o.value = String(i); return o;
  }));
  if (current) select.value = String(threads.indexOf(current));
  select.disabled = busy;
}

function supplierTurn(v) { return ["awaiting_supplier", "awaiting_info"].includes(v.status); }
function renderComposer(t) {
  const v = t.view, typed = t.mode === "typed";
  const turn = typed && supplierTurn(v) && !busy;
  $("supplier-form").hidden = !typed; $("sup-auto").hidden = typed;
  const showButtons = !typed || (v.status === "active" && !busy);
  $("step").hidden = !showButtons; $("run").hidden = !showButtons;
  ["in-price", "in-delivery", "in-payment", "in-message", "send"].forEach(id => { $(id).disabled = !turn; });
  const waitingOffer = v.status === "awaiting_supplier" && (v.transcript || []).at(-1)?.by === "buyer";
  $("quick").hidden = !(turn && waitingOffer);
  $("in-message").placeholder = turn ? "메시지 입력 (예: 단가 10,800원, 납기 14일 가능합니다)"
    : TERMINAL.includes(v.status) ? "협상이 끝났습니다" : "구매사의 답을 기다리는 중…";
  if (turn && t.prefilled !== v.revision) {           // start of the supplier's turn: carry over the last offer
    const last = [...(v.transcript || [])].reverse().find(r => r.by === "supplier");
    $("in-price").value = last ? last.terms.unit_price : ""; $("in-delivery").value = last ? last.terms.delivery_days : ""; $("in-payment").value = last ? last.terms.payment_days : "";
    $("in-message").value = ""; t.prefilled = v.revision;
    document.querySelectorAll(".term").forEach(x => x.classList.remove("filled"));
  }
  const canStep = v.status === "active" && !busy;
  $("step").disabled = !canStep; $("run").disabled = !canStep;
  $("composer-status").textContent = thinking ? `NAMI 판단 중 · ${engine()} 호출` : { active: "NAMI 차례", awaiting_supplier: typed ? "공급사 제안이 오면 NAMI가 바로 판단합니다" : "공급사 응답 대기", awaiting_info: "공급사에게 빠진 항목을 물었습니다", awaiting_approval: "승인 대기 · 승인 전에는 보내지 않음", needs_review: reviewInfo(v).status, agreed: "합의 완료", ended_alternative: "대안 견적으로 종료", supplier_walked: "공급사가 종료" }[v.status] || "";
}

function settingsFrom(v) { return v.buyer_settings || config.buyer_defaults; }
function readCriteria() {
  const n = id => Number($(id).value);
  return { target_unit_price: n("c-target"), max_unit_price: n("c-limit"), alternative_unit_price: n("c-alt"), max_delivery_days: n("c-delivery"), min_payment_days: n("c-payment") };
}
function criteriaError(c) {
  if (Object.values(c).some(x => !Number.isInteger(x))) return "숫자를 모두 입력하세요.";
  if (!(c.target_unit_price <= c.max_unit_price)) return "목표 단가는 내 단가 한도 이하여야 합니다.";
  if (!(c.max_unit_price < c.alternative_unit_price)) return "내 단가 한도는 대안 견적보다 낮아야 합니다.";
  if (c.max_delivery_days < 1 || c.max_delivery_days > 30) return "납기 상한은 1~30일입니다.";
  if (c.min_payment_days < 0 || c.min_payment_days > 120) return "결제 최소는 0~120일입니다.";
  if (c.target_unit_price < 1000 || c.alternative_unit_price > 100000) return "단가는 1,000~100,000원 사이로 입력하세요.";
  return "";
}
const FIELD_IDS = { "c-target": "target_unit_price", "c-limit": "max_unit_price", "c-alt": "alternative_unit_price", "c-delivery": "max_delivery_days", "c-payment": "min_payment_days" };
function fillCriteria(t) {
  const s = settingsFrom(t.view);
  Object.entries(FIELD_IDS).forEach(([id, key]) => { $(id).value = s[key]; $(id).classList.remove("changed"); });
  t.dirty = false;
  updateCriteria(t);
}
function updateCriteria(t) {
  const s = settingsFrom(t.view), c = readCriteria(), error = criteriaError(c);
  let changed = false;
  Object.entries(FIELD_IDS).forEach(([id, key]) => { const diff = Number($(id).value) !== s[key]; $(id).classList.toggle("changed", diff); changed ||= diff; });
  t.dirty = changed;
  $("criteria-error").hidden = !error || !changed; $("criteria-error").textContent = error;
  $("apply-criteria").disabled = busy || !changed || Boolean(error);
  drawLadder(t, error ? s : c);
}
function drawLadder(t, c) {
  const v = t.view, tr = v.transcript || [];
  const lastSupplier = [...tr].reverse().find(r => r.by === "supplier" && r.terms), lastOurs = [...tr].reverse().find(r => r.by === "buyer" && r.terms);
  // A vertical price ladder, highest first: one row per price, so labels never collide however close the
  // values are. The rail colour is the zone the price falls in: our authority, needs approval, or no deal.
  const rows = [["target", "목표", c.target_unit_price], ["limit", "내 단가 한도", c.max_unit_price], ["alt", "대안 견적", c.alternative_unit_price]];
  if (lastSupplier) rows.push(["offer", "공급사 최근 제안", lastSupplier.terms.unit_price]);
  if (lastOurs) rows.push(["ours", "NAMI 최근 제안", lastOurs.terms.unit_price]);
  const order = { offer: 0, alt: 1, limit: 2, target: 3, ours: 4 };      // ties: offers outside the criteria
  rows.sort((a, b) => b[2] - a[2] || order[a[0]] - order[b[0]]);
  const zone = value => value > c.alternative_unit_price ? "no" : value > c.max_unit_price ? "ask" : "ok";
  $("ladder").replaceChildren(...rows.map(([cls, label, value]) => {
    const r = el("div", `rung ${cls} zone-${zone(value)}`);
    r.append(el("span", "rail"), el("span", "dot"), el("span", "price", num(value)), el("span", "name", label));
    return r;
  }), legend());
}
function legend() {
  const box = el("div", "zones");
  [["ok", "내 권한"], ["ask", "승인 필요"], ["no", "합의 안 함"]].forEach(([z, text]) => { const s = el("span", `zone-${z}`); s.append(el("i"), document.createTextNode(text)); box.append(s); });
  return box;
}

function renderSide(t) {
  const v = t.view, last = (v.records || []).at(-1), p = v.pending;
  $("approval").hidden = !p;
  if (p) {
    const kind = last?.chosen?.kind;
    $("approval-question").textContent = p.kind === "accept_offer" ? "공급사 제안을 수락할까요?" : `${KIND[kind] || ""} 역제안을 보낼까요?`;
    $("approval-terms").replaceChildren(chips(p.terms, p.exceeded_auto_limits.map(e => e.field)));
    $("approval-limit").textContent = `내 한도 초과 · ${p.exceeded_auto_limits.map(limitText).join(" · ")}`;
    const values = [["대안 견적보다", p.saving_vs_alternative >= 0 ? `${won(p.saving_vs_alternative)} 절감` : `${won(-p.saving_vs_alternative)} 손해`, p.saving_vs_alternative >= 0 ? "positive" : "negative"]];
    // Accepting the supplier's own offer has no acceptance estimate: there is nothing for them to accept.
    if (p.p_accept != null) values.push([config.backend.is_model ? "Zoro-1 수락 추정" : "수락 추정(규칙)", pct(p.p_accept), ""]);
    $("approval-values").replaceChildren(...values.map(([a, b, c]) => { const d = el("div"); d.append(el("dt", "", a), el("dd", c, b)); return d; }));
    $("approve").disabled = busy; $("deny").disabled = busy;
  }
  $("review").hidden = v.status !== "needs_review";
  if (v.status === "needs_review") {
    const info = reviewInfo(v), offer = [...(v.transcript || [])].reverse().find(r => r.by === "supplier")?.terms;
    $("review-tag").textContent = info.tag; $("review-tag").className = `panel-tag ${info.cls}`;
    $("review-title").textContent = info.title; $("review-text").textContent = info.text(v, offer);
    // The raw engine error belongs only to a stop that was an engine failure.
    const engineError = v.review_reason === "system1_unavailable" ? (v.system1?.checks || []).at(-1)?.error : last?.model?.error;
    $("review-error").textContent = info.status === ENGINE_DOWN ? engineError || "" : "";
    $("retry").textContent = info.button; $("retry").disabled = busy;
    $("withhold").hidden = !info.withhold; $("withhold").disabled = busy;
    $("retry").className = info.withhold ? "ghost" : "primary";
  }
  const done = TERMINAL.includes(v.status);
  $("result").hidden = !done;
  if (done) {
    const o = v.outcome || {}, s = o.realized_saving_vs_alternative || 0;
    $("result-tag").textContent = v.status === "agreed" ? "합의" : "합의 없음";
    $("result-tag").className = `panel-tag ${v.status === "agreed" ? "ok" : "neutral"}`;
    $("result-title").textContent = o.agreement ? termsText(o.agreement) : { ended_alternative: "대안 견적 사용", supplier_walked: "공급사가 종료" }[v.status];
    // Baselines fixed at the start: the alternative quote and the supplier's first quote.
    const f = o.saving_vs_first_offer || 0, secs = (o.engine_time_ms || 0) / 1000;
    const rows = [["대안보다 절감", won(s), s > 0 ? "positive" : ""],
      ["첫 견적 대비 총비용 절감", won(f), f > 0 ? "positive" : ""], ["첫 견적", o.first_offer_terms ? termsText(o.first_offer_terms) : "—", ""],
      ["사람 개입", `${o.human_touches ?? 0}회 · ${o.rounds_used}라운드`, ""], ["엔진 시간", `${secs.toFixed(secs < 10 ? 1 : 0)}초`, ""],
      ["승인 / 거절", `${o.approvals_granted} / ${o.approvals_denied}`, ""],
      [`${engine()} 호출`, `${o.model_requests}회`, ""], ["권한 위반", `${(o.authority_violations || []).length}건`, (o.authority_violations || []).length ? "negative" : "positive"]];
    $("result-values").replaceChildren(...rows.map(([a, b, c]) => { const d = el("div"); d.append(el("dt", "", a), el("dd", c, b)); return d; }));
  }
  if (!t.dirty) fillCriteria(t); else updateCriteria(t);
}

function renderEngine() {
  const b = config.backend, chip = $("engine-chip");
  if (b.is_model) { chip.textContent = "Zoro-1 실시간 판단"; chip.className = "engine-chip model"; chip.title = (b.served_model_ids_reported_by_server || []).join(", "); }
  else { chip.textContent = "규칙 추정 · 모델 아님"; chip.className = "engine-chip"; chip.title = `${b.kind} · CPU 참조, 실제 Zoro-1 추론이 아닙니다`; }
}

// ---------- actions ----------
async function guarded(fn) {
  if (busy) return;
  busy = true; $("error").hidden = true; render(false);
  try { await fn(); }
  catch (e) { $("error").textContent = e.message; $("error").hidden = false; }
  finally { busy = false; thinking = false; render(false); }
}
async function refresh(t) { t.view = await api(`/api/sessions/${t.sessionId}`); }
async function show(t) { const ms = render(true); await pause(ms + AFTER_TURN_MS); }
async function stepOnce(t) {
  thinking = true; render(false);
  const next = await api(`/api/sessions/${t.sessionId}/step`, { revision: t.view.revision });
  thinking = false; t.view = next; await show(t);
}
// Runs until a person is needed: an approval, a failed estimate, the supplier's turn, or the end.
async function runAgent(t) { for (let i = 0; i < 12 && t.view.status === "active" && t === current; i++) await stepOnce(t); }

$("step").onclick = () => guarded(() => stepOnce(current));
$("run").onclick = () => guarded(() => runAgent(current));
for (const decision of ["approve", "deny"]) $(decision).onclick = () => guarded(async () => {
  const t = current, p = t.view.pending;
  t.view = await api(`/api/sessions/${t.sessionId}/approval`, { revision: t.view.revision, request_id: p.request_id, terms_version: p.terms_version, decision });
  await show(t); await runAgent(t);      // the next step re-checks authority before anything is sent
});
for (const [id, withhold_counter] of [["retry", false], ["withhold", true]]) $(id).onclick = () => guarded(async () => {
  const t = current;
  t.view = await api(`/api/sessions/${t.sessionId}/retry`, { revision: t.view.revision, withhold_counter });
  await show(t); await runAgent(t);
});

async function supplierSend(body) {
  const t = current, token = t.view.supplier_url.split("/").pop();
  await api(`/api/supplier/${token}/response`, { revision: t.view.revision, ...body });
  await refresh(t); await show(t);
  await runAgent(t);
}
$("supplier-form").onsubmit = e => {
  e.preventDefault();
  const n = id => $(id).value === "" ? undefined : Number($(id).value);
  guarded(() => supplierSend({ kind: "counter", unit_price: n("in-price"), delivery_days: n("in-delivery"), payment_days: n("in-payment"), message: $("in-message").value.trim() }));
};
$("accept-offer").onclick = () => guarded(() => supplierSend({ kind: "accept", message: $("in-message").value.trim() }));
$("walk-away").onclick = () => guarded(() => supplierSend({ kind: "walk_away", message: $("in-message").value.trim() }));

// Numbers typed in the message fill the term fields (the fields are what is sent).
function parseTerms(text) {
  const s = text.replace(/,/g, ""), out = {};
  // Price: the amount the sentence offers ("N원으로", "N원에"), else the only amount that is not being
  // refused. The first amount is often the buyer's price ("10,450원은 어렵고 10,800원으로 ..."), and a
  // wrong guess would be sent as the offer, so an unclear sentence fills nothing.
  const amounts = [...s.matchAll(/(?<!\d)(\d{4,6})\s*원([^\d]{0,10})/g)];
  const offered = amounts.filter(a => /^(으로|에(?![가-힣])|이면|까지)/.test(a[2]));
  const kept = amounts.filter(a => !/^[은는이도]?\s*(어렵|어려|안\s*[됩되돼]|불가|힘들|무리|곤란)/.test(a[2]));
  let m = offered.length === 1 ? offered[0] : kept.length === 1 ? kept[0] : amounts.length ? null : s.match(/단가\D{0,4}(\d{3,6})/);
  if (m) out["in-price"] = m[1];
  m = s.match(/납기\D{0,4}(\d{1,3})\s*일/) || s.match(/(\d{1,3})\s*일\s*(?:이내|안에|내|만에)/);
  if (m) out["in-delivery"] = m[1];
  if (/즉시\s*결제|결제\D{0,3}즉시|선결제|현금\s*결제/.test(s)) out["in-payment"] = "0";
  else { m = s.match(/결제\D{0,8}(\d{1,3})\s*일/) || s.match(/(\d{1,3})\s*일\s*(?:후\s*)?결제/); if (m) out["in-payment"] = m[1]; }
  return out;
}
$("in-message").addEventListener("input", () => {
  Object.entries(parseTerms($("in-message").value)).forEach(([id, value]) => {
    if ($(id).value !== value) { $(id).value = value; $(id).closest(".term").classList.add("filled"); }
  });
});

// criteria
["c-target", "c-limit", "c-alt", "c-delivery", "c-payment"].forEach(id => $(id).addEventListener("input", () => updateCriteria(current)));
$("criteria-form").onsubmit = e => {
  e.preventDefault();
  const buyer = readCriteria();
  if (criteriaError(buyer)) return;
  guarded(async () => {
    const t = current;
    t.view = await api("/api/sessions", sessionBody(t, buyer));
    t.sessionId = t.view.id; t.seen = new Set(); t.expanded = new Set(); t.prefilled = null; t.dirty = false;
    render(false);
    if (t.mode === "typed") $("in-message").focus();
  });
};
function sessionBody(t, buyer) {
  const body = t.mode === "typed" ? { mode: "human", preset: "blank" } : { mode: "simulated", preset: "start", scenario: t.scenarioKey };
  if (buyer) body.buyer = buyer;
  return body;
}
async function addThread(mode, scenarioKey, name, buyer) {
  const t = { mode, scenarioKey, name, seen: new Set(), expanded: new Set(), prefilled: null, dirty: false };
  t.view = await api("/api/sessions", sessionBody(t, buyer)); t.sessionId = t.view.id;
  threads.push(t); return t;
}
$("new-thread").onclick = () => guarded(async () => {
  typedCount += 1;
  const buyer = current && current.mode === "typed" ? settingsFrom(current.view) : null;
  const clean = buyer ? Object.fromEntries(Object.entries(buyer).filter(([k]) => k !== "alternative_terms")) : null;
  current = await addThread("typed", null, `새 공급사 ${typedCount}`, clean);
  threads.splice(threads.indexOf(current), 1); threads.splice(typedCount - 1, 0, current);
  render(false); $("in-message").focus();
});
$("thread-select").onchange = () => select(threads[Number($("thread-select").value)]);

// Design versions: the choice is a per-viewer convenience kept in this browser only.
$("theme-select").value = document.documentElement.getAttribute("data-theme") || "clean";
$("theme-select").onchange = () => {
  const theme = $("theme-select").value;
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("nami-theme", theme); } catch (e) { /* private mode: still switches for this view */ }
  const url = new URL(location.href);
  if (url.searchParams.has("theme")) { url.searchParams.set("theme", theme); history.replaceState(null, "", url); }
};
async function select(t) {
  if (busy || !t || t === current) return;
  current = t; t.dirty = false;
  try { await refresh(t); } catch (e) { /* keep the last view */ }
  buildItems(t).forEach(item => t.seen.add(item.key));
  render(false);
}

// evidence report
$("open-evidence").onclick = async () => {
  try { evidence = evidence || await api("/api/evidence"); } catch (e) { evidence = { available: false }; }
  showReport(); $("evidence").hidden = false; document.querySelector(".report").focus();
};
$("close-evidence").onclick = () => { $("evidence").hidden = true; $("open-evidence").focus(); };
$("evidence").onclick = e => { if (e.target === $("evidence")) $("close-evidence").onclick(); };
document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("evidence").hidden) $("close-evidence").onclick(); });
function showReport() {
  const dev = evidence?.development;
  if (!dev) { $("report-scope").textContent = "보존된 측정 요약이 없습니다."; return; }
  const b = dev.backend || {};
  $("report-scope").textContent = `실제 Zoro-1 · 합성 공급사 ${dev.cases}건 · ${b.gpu_name || ""} · 요청 ${dev.requests}회 · 저장된 요약`;
  const fixed = dev.differences.find(d => d.against === "fixed_concession");
  const violations = dev.policies.reduce((n, p) => n + (p.authority_violations || 0), 0);
  const worse = dev.policies.reduce((n, p) => n + (p.agreements_not_better_than_alternative || 0), 0);
  const tiles = [["고정 양보율보다 더 절감", fixed ? `+${won(fixed.mean_difference)}` : "—", fixed?.ci90 ? `90% 구간 +${num(fixed.ci90[0])} ~ +${num(fixed.ci90[1])}원` : ""],
    ["권한 위반", `${violations}건`, "모든 정책"], ["대안보다 불리한 합의", `${worse}건`, "모든 정책"]];
  $("report-tiles").replaceChildren(...tiles.map(([a, v, c]) => { const d = el("div", "report-tile"); d.append(el("span", "", a), el("strong", "", v), el("small", "", c)); return d; }));
  const top = Math.max(...dev.policies.map(p => p.mean_realized_saving || 0), 1);
  $("report-bars").replaceChildren(...dev.policies.map(p => {
    const r = el("div", `rbar${p.evaluator_only ? " reference" : ""}${p.key === "agent" ? " ours" : ""}`), track = el("div", "track"), fill = el("i");
    fill.style.width = `${Math.max(0, p.mean_realized_saving || 0) / top * 100}%`; track.append(fill);
    r.append(el("span", "", p.key === "agent" ? "NAMI (Zoro-1)" : p.label), track, el("span", "v", won(p.mean_realized_saving))); return r;
  }));
  $("report-diffs").replaceChildren(...dev.differences.filter(d => d.ci90 && !d.evaluator_only && d.against !== "midpoint").map(d => {
    const line = el("div"), wins = d.ci90[0] > 0;
    line.append(el("span", "", `NAMI − ${d.label}  ${d.mean_difference >= 0 ? "+" : "−"}${won(Math.abs(d.mean_difference))}`), el("strong", wins ? "positive" : "muted", wins ? "우위 (구간 0 초과)" : "우위 불확실 (구간 0 포함)"));
    return line;
  }));
}

// A supplier may also answer from the separate supplier page; pick that up.
setInterval(async () => {
  const t = current;
  if (busy || !t || t.mode !== "typed" || !supplierTurn(t.view)) return;
  try {
    const next = await api(`/api/sessions/${t.sessionId}`);
    if (!busy && t === current && next.revision !== t.view.revision) { t.view = next; render(true); if (next.status === "active") guarded(() => runAgent(t)); }
  } catch (e) { /* next poll retries */ }
}, POLL_MS);

guarded(async () => {
  config = await api("/api/config");
  renderEngine();
  current = await addThread("typed", null, "가온산업", null);
  typedCount = 1;
  for (const s of config.scenarios) await addThread("simulated", s.key, SIM_NAMES[s.key] || s.title, null);
});
