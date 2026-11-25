# NAMI 구매 협상 에이전트

에이전트 코드, 실행기, 테스트, 문서가 있는 폴더입니다. 개요와 설치는 [루트 README](../README.md)에 있습니다. 공개 체험 서버는 종료되었습니다. 아래 명령은 루트 README에 따라 설치한 가상환경을 활성화한 뒤 저장소 루트에서 실행합니다.

## 실행 (CPU)

```sh
python -m pytest -q negotiation/tests
python negotiation/run_negotiation.py --seed 1000 --payment-sensitive --reveal --output runs/negotiation/demo.json
python negotiation/run_negotiation.py --seed 1000 --payment-sensitive --reveal --approver ask --output runs/negotiation/demo_ask.json
python negotiation/run_evaluation.py evaluate --split dev --output runs/negotiation/eval_dev.json
python negotiation/run_evaluation.py evaluate --split dev --backend keyword --output runs/negotiation/keyword_dev.json
python negotiation/run_probe.py --backend heuristic --output runs/negotiation/probe_heuristic.json
python negotiation/run_utterance_probe.py --backend keyword --output runs/negotiation/utterance_keyword.json
python negotiation/run_tune.py plan --state-dir runs/negotiation/tuning/cpu-rehearsal
python negotiation/run_tune.py run --stage all --state-dir runs/negotiation/tuning/cpu-rehearsal --backend keyword
python negotiation/run_reader_check.py --backend keyword --output-dir runs/negotiation/final_reader/cpu-rehearsal
python negotiation/run_scenarios.py --backend keyword --output-dir runs/negotiation/scenarios-cpu
python negotiation/run_play.py
python negotiation/run_web.py --backend keyword --replay negotiation/artifacts/20261116_colab/demo_replay.json
```

- `run_negotiation.py`는 협상 한 건을 끝까지 돌리며 라운드마다 계산값, 추정, 선택, 승인 카드를 출력합니다. `--approver ask`는 승인 카드가 나오면 터미널에서 승인·거부를 받습니다.
- `heuristic`, `keyword`, `constant`는 모델 없는 CPU 참조이며 결과 파일에 `cpu_reference_not_model_evidence`로 표시됩니다. 실제 모델 명령은 [docs/PLAN.md](docs/PLAN.md) 5절에 있습니다.
- 평가 분할(`--split eval`)은 동결 기록 없이는 실행되지 않으며, 첫 시도가 이력으로 남습니다.
- `run_scenarios.py`는 업무 시나리오 3종의 실행 로그를 만듭니다. 설명과 실제 Zoro 결과는 [docs/SCENARIOS.md](docs/SCENARIOS.md)에 있습니다.
- `run_play.py`는 내가 공급사가 되어 에이전트와 협상하는 화면(`/play`)을 엽니다. `run_web.py`와 같은 포트(8766)를 쓰므로 하나씩 실행합니다. 사용법은 [docs/PLAYGROUND.md](docs/PLAYGROUND.md)에 있습니다.

## 대시보드

- 제품 시제품 화면은 `http://127.0.0.1:8766/app`입니다. 왼쪽은 **공급사 화면**(주고받은 메시지만 보임), 오른쪽은 **구매팀 화면**(NAMI 판단·승인·내 기준)입니다. 첫 협상(가온산업)은 사람이 왼쪽에서 공급사로 단가·납기·결제와 메시지를 입력하고(메시지 속 숫자는 칸에 자동으로 채워짐), NAMI가 오른쪽에서 판단 카드(계산·코드 → 추정·Zoro-1 → 결정·코드 → 권한·코드)를 거쳐 답합니다. 오른쪽 `내 협상 기준`에서 목표 단가·내 단가 한도·대안 견적·납기 상한·결제 최소를 바꾸고 새로 시작할 수 있습니다(서버에서 범위와 순서를 검증하며, 이 값은 모델 요청에 들어가지 않습니다). 나머지 다섯 협상은 고정 합성 공급사 시뮬레이션입니다.
- `http://127.0.0.1:8766`에 열립니다. 추정 backend는 운영자가 `--backend`로 고르고, 화면 상단에 CPU 참조인지 실시간 모델 호출인지 표시합니다. 2026-11-20에는 native Zoro-1을 연결한 대시보드·체험 API에서 합성 시나리오 3종을 실행했습니다. [당시 대화와 판독 기록](artifacts/20261120_live_demo/README.md)을 확인할 수 있습니다.
- 오른쪽 열의 `System 1 판단` 패널이 최근 판단(진단·라우팅, 수락 검증, 코드 숫자 대조, 결과)을 보여 줍니다.
- `시나리오`에서 고정 합성 공급사 5종을 고릅니다. `자동 진행`은 승인 카드, 추정 실패, 상대 응답, 종료에서 멈춥니다. 승인한 조건도 `다음 판단`에서 권한을 다시 확인한 뒤 전송합니다.
- `사람 공급사`는 우리 측 목표·대안·절감액이 없는 별도 공급사 화면을 엽니다. 다른 기기에서 열려면 `--host 0.0.0.0`을 붙입니다(인증 없는 합성 시연이므로 신뢰하는 네트워크에서만).
- `--save-dir`는 실시간 세션을 판단마다 저장하고, `--replay`는 저장한 기록을 읽기 전용으로 보여 줍니다. `검증 결과`는 보존한 실제 Zoro 개발 30건 요약을 보여 줍니다.

## 구조

| 위치 | 책임 | 볼 수 있는 정보 |
|---|---|---|
| `src/negotiation_agent/contracts.py` | 조건, 제안, 공개 상태, 우리 측 비공개 정보, 상대 숨은 정보, 승인 | 타입 정의 |
| `src/negotiation_agent/economics.py` | 환산 총비용, 대안 대비 절감액, 권한 판정, 기대가치 | 우리 측 비공개 |
| `src/negotiation_agent/simulator.py` | 규칙 기반 공급사와 합성 사례 생성 | 상대 숨은 정보 |
| `src/negotiation_agent/requests.py` | 모델 요청 생성 | 공개 상태와 후보만 |
| `src/negotiation_agent/backends.py` | 예측기: 상수, 규칙, 실제 Zoro 어댑터, 공식 서버 | 공개 상태와 후보만 |
| `src/negotiation_agent/estimate_sources.py` | 수락 가능성과 마지막 제안 추정의 출처 조합(모델, 상수, 사전값, 키워드) | 공개 상태와 후보만 |
| `src/negotiation_agent/system1.py` | 공급사 메시지 진단, 범위 밖 요구 라우팅, 수락 답변 검증(멈출 수만 있음) | 공개 상태, 보낸 조건, 공급사 문장만 |
| `src/negotiation_agent/policy.py` | 후보 생성, 수락·역제안·종료 비교, 승인 카드 작성 | 우리 측 비공개 + 공개 상태 |
| `src/negotiation_agent/session.py` | 승인 대기·승인·거부, 실행 직전 권한 확인, 전송, 기록 | 우리 측 비공개 + 공개 상태 |
| `src/negotiation_agent/baselines.py` | 고정 양보율, 중간값 기준선(같은 승인 관문 사용) | 우리 측 비공개 + 공개 상태 |
| `src/negotiation_agent/evaluation.py` | 분할, 정책 비교, 동결과 평가 이력 | 평가기 |
| `src/negotiation_agent/probe.py`, `utterance_probe.py`, `reader_check.py` | 후보 민감도, 발언 반응, 마지막 제안 판독 점검 | 평가기 |
| `src/negotiation_agent/tune.py`, `prediction_store.py` | 사전 고정 프로토콜의 설정 튜닝, 요청 해시별 응답 저장 | 평가기(평가 분할 제외) |
| `src/negotiation_agent/scenarios.py` | 업무 시나리오 3종, 바꿔치기·범위 밖 요구 공급사, 장애 주입 | 평가기 |
| `src/negotiation_agent/playground.py`, `static/play.*` | 협상 체험 화면 | 공급사 화면은 공개 정보만, 테스터용 내부 보기는 우리 측 비공개 포함 |
| `src/negotiation_agent/web.py`, `static/` | 대시보드 | 구매자 화면은 우리 측 비공개 포함, 공급사 화면은 공개 정보만 |
| `src/utterance_agent/` | 공급사 발언을 읽는 실험 에이전트(CPU 검증) | 공개 상태 |
| `artifacts/` | 실제 실행과 CPU 참조 결과를 구분해 보존한 작은 요약 | |

## 문서

- [docs/SUBMISSION.md](docs/SUBMISSION.md): 요약, 시연 대본, 실행 순서
- [docs/PLAN.md](docs/PLAN.md): 설계, 측정, 재현 명령, 한계
- [docs/SCENARIOS.md](docs/SCENARIOS.md): 업무 시나리오 3종과 System 1 판단
- [docs/PLAYGROUND.md](docs/PLAYGROUND.md): 협상 체험 사용법
- [docs/TUNING.md](docs/TUNING.md): 모델 사용 설정 튜닝 프로토콜
- [DESIGN.md](DESIGN.md): 대시보드 디자인
