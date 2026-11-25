# NAMI · Negotiation Agent for Maximum Impact

**권한 안에서 협상하는 구매 협상 에이전트**입니다. 모든 거래와 공급사는 합성 데이터입니다.

## 로컬에서 체험하기

공개 체험 서버는 운영을 종료했습니다. 아래 [실행 안내](#실행)에 따라 내 컴퓨터에서 협상 화면을 열거나, 저장소에 포함된 실제 Zoro-1 합성 협상 기록을 대시보드에서 재생할 수 있습니다. CPU 모드와 기록 재생에는 모델 계정이나 API 키가 필요하지 않습니다.

## 동작 방식

Decision Model Zoro-1은 공개된 제안 이력만 보고 "상대가 이 조건을 받을까", "이것이 상대의 마지막 제안일까"를 추정합니다. 공급사 메시지가 오면 단가·납기·결제 밖의 요구가 있는지, 수락 답변이 보낸 조건 그대로인지도 읽습니다(System 1). 이 판단은 에이전트를 멈출 수만 있고 권한을 주지 않습니다. 비용 계산, 역제안 후보 생성, 권한 판정, 실행은 코드가 합니다.

| 단계 | 담당 | 볼 수 있는 정보 |
|---|---|---|
| 비용 환산, 역제안 후보 생성(최대 5개), 권한 판정 | 코드 | 우리 측 목표·대안 견적·한도와 공개 이력 |
| 후보별 수락 가능성, 마지막 제안인지 추정 | Zoro-1 | 공개 이력과 후보 조건만 |
| 공급사 메시지 진단, 범위 밖 요구를 사람에게 넘김, 수락 답변 검증 | Zoro-1 | 공개 이력, 보낸 조건, 공급사 문장만 |
| 수락·역제안·종료 선택 | 코드 | 비용 계산과 Zoro-1의 수락·마지막 제안 추정 (System 1 판단은 멈춤만 가능) |
| 한도를 넘는 조건의 승인 | 사람 | 그 제안과 그 조건 버전만 |
| 실행 | 코드 | 한도 안이거나 승인된 조건만, 전송 직전에 다시 검사 |
| 답장 문장 작성 | 기본값은 고정 문구(`--llm none`). `--llm auto`는 설정된 Claude, 설정된 로컬 문장 모델, 코드가 쓰는 문장 순으로 시도 | 코드가 정한 행동과 조건, 공급사의 마지막 문장, 우리가 보낸 최근 답장만(한도·대안 견적은 받지 않음). 보내기 전에 코드가 조건 숫자가 정확한지, 다른 숫자나 입장을 드러내는 말이 없는지 검사 |

추정이 틀려도 권한 밖 조건과 대안보다 불리한 합의는 실행되지 않습니다. 추정이 실패하면 값을 지어내지 않고 사람 검토로 멈추며, 공급사가 "이미 승인받았다"고 써도 승인으로 처리하지 않습니다.

## 결과

실제 Zoro-1로 합성 개발 30건을 돌린 결과입니다(Colab T4, FP16, 1회).

| 정책 | 평균 실현 절감 | 권한 위반 |
|---|---:|---:|
| NAMI (Zoro 추정) | 306,426원 | 0 |
| 규칙 추정 | 284,264원 | 0 |
| 고정 양보율 | 260,146원 | 0 |
| 참고 상한(숨은 정답 사용) | 348,004원 | 0 |

- 이 30건에서 한도를 넘는 조건의 승인 카드는 합성 자동 승인(`approve_all`)이 처리했습니다. 권한 위반 0건은 그 조건에서 잰 값이며, 사람이 직접 승인하는 흐름은 웹 체험(`/app`, `/play`)에 있습니다.
- 고정 양보율보다 평균 +46,281원 더 절감했습니다(90% 구간 [+13,456, +83,421]). 규칙 추정과의 차이 +22,163원은 구간이 0을 포함해 우위라고 말할 수 없습니다.
- 공급사가 "더는 양보하지 않겠다"는 뜻을 정해진 문구 없이 말하는 경우를 따로 시험했습니다(합성 공급사 200건, 로컬 GPU, 1회). Zoro로 이를 읽으면 키워드 규칙보다 평균 +9,824원 더 절감했습니다. 다만 지시문이 섞인 발언에도 판독이 흔들렸습니다.
- 업무 시나리오 3종(계약 갱신 자동 타결, 권한 밖 조건의 승인, 조건 바꿔치기·범위 밖 요구·판단 서버 장애)을 실제 Zoro-1로 돌렸습니다(Kaggle T4, FP16, native 어댑터, v1·v2 각 1회). 승인과 검토는 자동 fixture가 맡았고(사람 아님), 서버 장애는 주입한 것입니다. v1에서는 승인 카드 장면이 나오지 않아, 실제 Zoro로 seed 100개를 돌려 고른 사례로 v2를 다시 실행했습니다. 두 실행 모두 권한 위반과 검증 없는 체결이 0건이었습니다. 로그와 한계는 [docs/SCENARIOS.md](negotiation/docs/SCENARIOS.md)에 있습니다.

수치의 해석과 한계는 [docs/SUBMISSION.md](negotiation/docs/SUBMISSION.md), 재현 명령은 [docs/PLAN.md](negotiation/docs/PLAN.md)에 있습니다.

## 실행

Python 3.11 이상(검증은 3.12). 저장소 루트에서 실행합니다.

```sh
python scripts/bootstrap.py
.venv/bin/python negotiation/run_play.py
.venv/bin/python negotiation/run_web.py --backend keyword --replay negotiation/artifacts/20261116_colab/demo_replay.json
```

첫 줄은 `.venv`를 만들고 테스트까지 실행합니다. 이후 명령은 반드시 그 가상환경의 Python으로 실행합니다(Windows는 `.venv\Scripts\python.exe`). 시스템 `python`으로 실행하면 `fastapi`를 찾지 못합니다.

- **직접 협상해 보기** (`run_play.py`): 내가 공급사가 되어 에이전트와 제안을 주고받는 화면이 <http://127.0.0.1:8766/play>에 열립니다. 권한을 넘는 조건의 승인 카드와 사람 검토도 내가 결정합니다. 사용법과 해 볼 만한 장면은 [docs/PLAYGROUND.md](negotiation/docs/PLAYGROUND.md)에 있습니다.
- **대시보드** (`run_web.py`): <http://127.0.0.1:8766>에 열리며, 실제 Zoro 실행 기록 3건을 재생할 수 있습니다.
- 두 명령은 같은 주소(127.0.0.1:8766)를 쓰므로 하나씩 실행합니다. `run_web.py`로 띄운 서버에서도 `/play`가 열립니다. 기본 엔진은 모델 없는 CPU 참조이고, 화면 상단에 엔진 종류가 표시됩니다.

실제 Zoro-1을 쓰려면 Hugging Face에서 `trillionlabs/Zoro-1` 접근 승인을 받고 토큰을 `HF_TOKEN`에 둔 뒤 모델을 준비합니다. 그다음 `run_play.py`나 `run_web.py`에 `--backend zoro --original-checkpoint <원본 snapshot> --exported-checkpoint <export> --dtype fp16 --decision-temperature 1`을 붙이면 이 PC의 GPU에서 Zoro-1이 답합니다. 두 경로는 아래 명령이 `.cache/gpu_paths.json`에 기록합니다. `--backend zoro-http --url <판단 서버>`는 외부 판단 서버를 부르며, 그 서버의 모델은 이 프로그램이 검증하지 못합니다. 대시보드 실행 옵션은 [docs/SUBMISSION.md 7절](negotiation/docs/SUBMISSION.md#7-현장-실행-순서)에서 확인할 수 있습니다.

```sh
python scripts/gpu_setup.py --steps env,install,sources,model,export --dtype fp16
```

## 구성

| 경로 | 내용 |
|---|---|
| `negotiation/` | 에이전트, 실행기, 테스트, 문서, 대시보드, 결과 요약 ([안내](negotiation/README.md)) |
| `src/sessionguard/` | Zoro-1 호출 어댑터(native, HTTP) |
| `config/model_pins.json` | 고정한 모델 revision과 공식 소스 해시 |
| `scripts/` | CPU 설치, GPU 준비, 공식 판단 서버 실행 |
| `notebooks/` | Colab·Kaggle 실험 노트북과 생성기 |

```sh
.venv/bin/python -m pytest -q
```

## 라이선스

[Apache License 2.0](LICENSE). Zoro-1 모델과 공식 소스는 저장소에 포함하지 않으며, 실행할 때 각자의 라이선스로 내려받습니다([NOTICE](NOTICE)).
