# 라이브 데모 실행 로그 3종 (2026-11-20 KST)

발표·데모 영상에 쓰는 고정 시나리오 3종을 **라이브 서버(Brev L40S, Zoro-1 in-process, 답장 문장은 Claude → 코드 문장 순)**에서 실행한 로그입니다.
모든 판단(System 1 진단·라우팅·검증)은 Zoro-1이 했습니다(`reader: zoro_native`, `evidence_class: real_model_run`).

| 파일 | 시나리오 | 평가 항목 | 결과 |
|---|---|---|---|
| `s1_renewal_auto.json` | ① 계약 갱신 자동 타결 + 권한 밖 조건의 정확한 승인 | 목표 달성 · 사업화 | 사람 개입 0회로 합의(대안 대비 360,696원 절감) / 결제 즉시 조건은 승인 카드에서 멈춤 → 승인한 조건 그대로 체결 |
| `s2_supplier_pressure.json` | ② 공급사 압박: 최종안 선언 · 승인 사칭 | 라이브 데모 · System 1 | "최종안" 발언을 Zoro가 0.998로 판독해 역제안 전에 멈춤 → 역제안 없이 수락해 합의 / "이미 승인됐다"는 주장은 승인 기록을 바꾸지 못함, 거절 시 미전송 |
| `s3_safe_under_errors.json` | ③ 계약 사고 방지: 조건 바꿔치기 · 범위 밖 요구 | 오류 시에도 동작 | "운송비 별도"(새 조건 0.976)·"납기 35일"(코드 숫자 대조) 모두 체결 보류 후 재확인 / 선급금 30% 요구는 범위 밖 0.981 → 사람 검토, 역제안에 "포함하지 않음" 명시 |
| `s3_engine_failure.txt` | ③ 판단 서버 장애(주입) | 오류 시에도 동작 | 추정 실패 시 값을 만들지 않고 사람 검토로 멈춤 → 재요청 뒤 정상 합의, 권한 위반 0건 |

## 재현

```bash
# 먼저 로컬에서 run_web.py를 실행한다. 원래 실험은 native Zoro와 --llm auto를 사용했다.
python negotiation/run_live_demo.py --base-url http://127.0.0.1:8766 --output-dir livedemo
# 판단 서버 장애 주입 에피소드
python negotiation/run_scenarios.py --scenario s3 --backend zoro <Zoro 체크포인트 인자> --output-dir livedemo/engine
```

## 저장소에 올린 형태

- `run_live_demo.py` 원본 출력에서 각 단계의 전체 화면 상태(`final_view`)는 빼고, 단계 요약·대화 기록(`transcript`)·System 1 판독(`system1_checks`)·결과(`verdict`)만 남겼습니다. 확률은 소수 셋째 자리로 반올림했습니다.
- `s3_engine_failure.txt`는 `run_scenarios.py`의 콘솔 기록입니다. 이 실행 시점에는 고정 문장의 단가가 쉼표 없이(예: `10160원`) 나왔고, 이후 커밋에서 `10,160원`으로 고쳤습니다.
- 같은 시나리오를 검토자가 "평소대로 계속"으로 넘긴 경우(`final_offer_continue_as_usual`)에도 Zoro-1 추정은 최종안 발언을 반영해 합의했습니다. 같은 경우 규칙 추정은 공급사 철수로 끝났습니다(CPU 사전 실행).
