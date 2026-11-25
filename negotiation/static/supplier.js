"use strict";
const $=id=>document.getElementById(id), token=location.pathname.split("/").at(-1);
let state=null,busy=false;
const node=(tag,cls,text)=>{const n=document.createElement(tag);n.className=cls||"";if(text!==undefined)n.textContent=text;return n;};
const labels={awaiting_buyer:"구매자가 다음 행동을 결정하고 있습니다.",awaiting_supplier:"구매자 제안에 답할 수 있습니다.",awaiting_info:"누락된 조건을 확인해 주세요.",agreed:"합의가 완료되었습니다.",supplier_walked:"공급사가 협상을 종료했습니다.",ended_alternative:"구매자가 협상을 종료했습니다."};
async function request(body){const r=await fetch(`/api/supplier/${token}${body ? "/response" : ""}`,body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {});const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==="string" ? data.detail : "숫자 입력 범위를 확인해 주세요.");return data;}
function render(){if(!state)return;$("status").textContent=labels[state.status]||state.status;$("round").textContent=`R${state.round} / ${state.max_rounds}`;$("history").replaceChildren(...state.transcript.map(o=>{const li=node("li"),meta=node("div","history-meta",`R${o.round} · ${o.by==="buyer" ? "구매자" : "공급사"} · ${o.offer_id||"전달"}`);li.append(meta);if(o.terms)li.append(node("div","history-terms",`${o.terms.unit_price.toLocaleString("ko-KR")}원 · 납기 ${o.terms.delivery_days}일 · 결제 ${o.terms.payment_days}일`));if(o.message)li.append(node("p","",o.message));return li;}));const enabled=["awaiting_supplier","awaiting_info"].includes(state.status);for(const id of ["counter","accept","walk","price","delivery","payment","message"])$(id).disabled=!enabled||busy;$("refresh").disabled=busy;$("clarification").hidden=!state.clarification;$("clarification").textContent=state.clarification||"";}
async function task(body){if(busy)return;busy=true;render();$("error").hidden=true;try{state=await request(body);}catch(e){$("error").textContent=e.message;$("error").hidden=false;}finally{busy=false;render();}}
$("refresh").onclick=()=>task();
$("response-form").onsubmit=e=>{e.preventDefault();const number=id=>$(id).value==="" ? null : Number($(id).value);task({revision:state.revision,kind:"counter",unit_price:number("price"),delivery_days:number("delivery"),payment_days:number("payment"),message:$("message").value});};
$("accept").onclick=()=>task({revision:state.revision,kind:"accept",message:$("message").value});$("walk").onclick=()=>task({revision:state.revision,kind:"walk_away",message:$("message").value});
task();
// While the buyer side is deciding, pick up its next message without a manual refresh.
setInterval(async()=>{if(busy||!state||state.status!=="awaiting_buyer")return;try{const next=await request();if(!busy&&next.revision!==state.revision){state=next;render();}}catch(e){/* the next poll retries */}},2000);
