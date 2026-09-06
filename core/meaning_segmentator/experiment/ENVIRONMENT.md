# 실험 환경 — `en-multi/run13` 계열과 `{de,ja,zh}-en/run02`

논문 "실험 설정" 절에 그대로 옮겨 쓸 수 있게, 두 묶음이 **어떤 기계에서 어떤 소프트웨어로**
돌았는지를 실측과 로그 근거로 적는다. 하이퍼파라미터의 의미와 런별 결과는
[README.md](README.md) 에 있고, 이 문서는 그 아래층(하드웨어·버전·모델 설정)만 다룬다.

## 0. 한 줄 요약

두 묶음은 **서로 다른 기계에서 돌았다.** 설정은 같지만 하드웨어·파이썬 환경이 다르므로,
"같은 조건에서 비교했다"고 쓰면 안 된다. 점수는 API 모델(gpt-5-mini)과 GPU 채점기가
결정하고 두 기계 모두 같은 체크포인트를 쓰므로 **비교 가능성 자체는 유지되지만**,
벽시계 시간·처리량 수치를 기계 구분 없이 합치면 틀린다.

| 묶음 | 실행 기계 | GPU | 근거 |
|---|---|---|---|
| `en-multi/run13`, `run13ta-{de,ja,zh}` | 기계 B (`/home/skkai`) | NVIDIA **GB10** | `runs/en-multi/run13ta-*.log` 의 CUDA 경고 |
| `{ja,zh}-en/run02` | 기계 A (`/home/mobility`) | NVIDIA **RTX 4090** | 로그 전체가 `/home/mobility` 경로 |
| `de-en/run02` | **iter_00 만 기계 B**, iter_01~05 + 최종은 기계 A | GB10 → RTX 4090 | `de-en_run02.log:9-20` 이 GB10, `:246` 이후가 기계 A. 커밋 `35a2a40` ("iter0 산출물 — 재개 지점") 로 넘어왔다 |

`de-en` 이 갈라진 것은 기계 B 에서 iter_00 까지 돌린 뒤 산출물을 커밋해 기계 A 에서
`--resume` 으로 이어받았기 때문이다. iter_00 의 점수는 기계 B 산이고 나머지는 기계 A 산이다.

## 1. 하드웨어

### 기계 A — x2en 세 트랙 (실측)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB (24GB GDDR6X) |
| 드라이버 / CUDA | 580.173.02 / CUDA 13.0 |
| CPU | Intel Core i9-13900KF, 24코어 32스레드 (1소켓) |
| RAM | 62 GiB |
| OS / 커널 | Ubuntu 22.04.5 LTS / Linux 6.8.0-138-generic |

### 기계 B — run13 계열 (로그에서 읽히는 것만)

| 항목 | 값 | 근거 |
|---|---|---|
| GPU | NVIDIA **GB10**, CUDA compute capability **12.1** | 로그의 PyTorch 경고 |
| 파이썬 | 3.12, anaconda3 `speech_ai` 환경 | 로그의 traceback 경로 |

로그에 남은 경고가 그대로 재현 조건의 일부다:

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

즉 **그 PyTorch 빌드는 GB10 을 공식 지원하지 않는 상태로 돌았다.** 결과가 나온 이상
계산은 됐지만, 그 기계의 속도 수치를 논문에 쓸 거면 이 경고를 함께 적어야 한다.
기계 B 의 CPU·RAM·드라이버·정확한 패키지 버전은 **이 레포에서 확인할 수 없다** —
필요하면 그 기계에서 직접 받아 이 표를 채울 것.

## 2. 소프트웨어 (기계 A, 실측)

| 항목 | 버전 |
|---|---|
| 파이썬 | 3.10.12 (`.venv-autoseg`) |
| torch | 2.13.0+cu130 (CUDA 13.0, cuDNN 9.2.0) |
| transformers | 4.57.6 |
| unbabel-comet | 2.2.7 |
| pytorch-lightning | 2.6.5 |
| tokenizers / huggingface_hub | 0.22.2 / 0.36.2 |
| sentencepiece / datasets | 0.2.2 / 5.0.1 |
| numpy / scipy / pandas | 1.26.4 / 1.15.3 / 2.3.3 |
| sacrebleu / nltk | 2.6.0 / 3.10.3 |

**가상환경은 `.venv` 가 아니라 `.venv-autoseg` 다.** `.venv` 에는 `langcodes` 와
`unbabel-comet` 이 없어 `pipeline.to_lang_code` 에서 즉사한다. 또 이 기계는 ROS humble 이
`PYTHONPATH` 에 끼어 venv 의 site-packages 보다 먼저 검색되므로, 실행 스크립트가
`PYTHONPATH=.` 로 **덮어쓴다** (`tools/autoseg_x2en/run_chain.sh`).

## 3. 언어모델 (분절·판정·프롬프트 개정)

| 항목 | 값 |
|---|---|
| 모델 | `gpt-5-mini` (분절·판정자·프롬프트 엔지니어·크리틱 전부 동일) |
| 프로바이더 | OpenAI 공식 API (`https://api.openai.com/v1`, `--provider openai`) |
| temperature | **지정 못 함** — 모델이 거부해 기본값으로 돈다. 따라서 분절은 비결정론적이다 |
| reasoning effort | `--agent-reasoning-effort none` / `--seg-reasoning-effort none`. 이는 "사고 끄기"가 아니라 **파라미터를 빼서 모델 기본값으로 둔다**는 뜻이고, gpt-5-mini 기본값은 사고 켜짐이다 |
| 실측 사고 토큰 | 분절 콜당 10,000~17,000 토큰, 재시도 콜당 7,800~11,000 |
| 동시성 | `--workers 24` |
| 배치 | `--batch-size 6` (한 콜에 문장 6개) |

비결정론이 결과에 미치는 영향은 **쌍체 비교(paired)** 로 흡수한다 — 개정 전후를 같은
문장 집합에서 재고, 채택 판정은 dev Δ 의 부호와 표준오차로 한다.

## 4. GPU 채점기·번역기 (세 모델 모두 로컬)

| 역할 | 체크포인트 | 설정 |
|---|---|---|
| 번역 (목적함수용) | `google/madlad400-3b-mt` | fp16, **greedy** (`num_beams=1`), 마이크로배치 48, 입력 384토큰 절단, 출력 최대 192토큰, 클라이언트 스레드 64, 문맥 미사용 |
| adequacy (주지표) | `Unbabel/wmt22-cometkiwi-da` | **참조 없는 QE**, `gpus=1`, batch 16, DataLoader `num_workers=0` |
| contradiction / consistency | `vicgalle/xlm-roberta-large-xnli-anli` | `device=0`, batch 16, `truncation=True, max_length=512` |
| 강제정렬 (전처리) | `Qwen/Qwen3-ForcedAligner-0.6B` | `min_gap` 유도용 단위 종료시각 산출, 80ms 격자 |

세부 규약 셋을 함께 적어야 재현된다.

- **번역은 빔이 아니라 greedy 다.** 빔4 대비 29배 빠르고 CometKiwi 품질은 0.8554 → 0.8473
  으로 0.008 만 떨어진다. VRAM 8.8GB. 이터당 번역 약 24,650건이 10분에 끝난다.
- **문맥 번역을 쓰지 않는다** (`ctx=False`). seq2seq 번역기는 개행 구조를 보존하지 않아
  Google 경로의 문맥 규약이 성립하지 않고, 조각을 독립 번역해야 쌍체 비교가 결정론적이다.
- **NLI 는 truncation 을 반드시 켠다.** xlm-roberta 위치 임베딩이 514 라 긴 문장에서
  premise+hypothesis 가 한계를 넘으면 조용한 오차가 아니라 RuntimeError 로 런이 죽는다.
- 모델은 **프로세스당 1회** 로드해 타깃 5개가 공유한다. 타깃마다 새로 올리면 VRAM 이 5배다.
- `remote_mt_*` (Seed-X, `localhost:8010`) 필드가 config 에 남아 있지만 **이 런들에서는
  안 썼다** — 전부 `translate_backend: local` 이다.

## 5. 데이터

| 항목 | 값 |
|---|---|
| 출처 | FLEURS n-way 병렬 매니페스트 (`evaluation/ast/manifests/fleurs_nway_*_loop405.jsonl`) |
| 규모 | 매니페스트 전량 **405문장** = train 40 / dev 265 / test 100 |
| 분할 seed | `20260806` (전 트랙 동일) |
| 소스 특성 (en) | 백과·뉴스체 편집 산문, 문말 부호 비율 0.9934, 공백비율 0.159 (n=305 실측) |

트랙별 매니페스트:

| 트랙 | `--dataset` | 매니페스트 |
|---|---|---|
| run13 / run13ta-de / run13ta-zh | `fleurs-en-multi` | `fleurs_nway_en-de_multi_loop405.jsonl` |
| run13ta-ja | `fleurs-en-multi-ja` | `fleurs_nway_en-ja_multi_loop405.jsonl` |
| de-en | `fleurs-de-en` | `fleurs_nway_de-en_multi2en_loop405.jsonl` |
| ja-en | `fleurs-ja-en` | `fleurs_nway_ja-en_multi2en_loop405.jsonl` |
| zh-en | `fleurs-zh-en` | `fleurs_nway_zh-en_multi2en_loop405.jsonl` |

`run13ta-zh` 가 en-zh 가 아니라 en-de 매니페스트로 돈 사정은 [README.md](README.md) 의
경고 절에 있다 — 채점은 참조 없는 QE 라 점수는 오염되지 않지만 prompt_v0 의 근거가 다르다.

**정답 번역은 채점에 안 쓴다.** 목적함수는 참조 없는 QE 이고, 매니페스트의 정답 번역은
`--target-aware` 런이 타깃 프로파일을 만들 때만 읽는다.

## 6. 실행 하이퍼파라미터

전 트랙 공통 (`tools/autoseg_x2en/run_chain.sh` 가 run13 설정을 그대로 복제한다):

```
--model gpt-5-mini --provider openai
--agent-reasoning-effort none --seg-reasoning-effort none
--iterations 5 --train 40 --dev 265 --test 100
--patience 5 --workers 24
--translate-backend local            # google/madlad400-3b-mt
--adequacy-backend cometkiwi --consistency-backend nli --adopt-se-mult 0.5
```

config 에만 있고 인자로 안 주는 값: `revision_candidates 3`, `v0_candidates 1`,
`min_boundaries_per 4`, `skip_translation_below 0.95`, `judge_frac 0.1`,
`comet_batch_size 16`, `max_prompt_growth 1.6`, `coverage_required true`.

**`--min-gap` / `--t-grid` / `--t-floor` 는 일부러 안 준다.** 고정 시간상수
`MIN_GAP_MS = 1200` 에 코퍼스 발화속도(강제정렬 실측)를 곱해 언어마다 유도된다:

| 트랙 | 단위 | 발화속도 | → `min_gap` | 루프 격자 | 최종 격자 | 주 작동점 T |
|---|---|---|---|---|---|---|
| en-multi (run13 계열) | 어절 | 2.51 word/s | 3 | [4, 6, 12] | [4, 6, 8, 12] | 6 |
| de-en | 어절 | 2.40 word/s | 3 | [4, 6, 12] | [4, 6, 8, 12] | 6 |
| ja-en | 글자 | 5.70 char/s | 7 | [9, 14, 27] | [9, 14, 18, 27] | 14 |
| zh-en | 글자 | 4.73 char/s | 6 | [8, 12, 24] | [8, 12, 16, 24] | 12 |

ja/zh 의 값이 큰 것은 언어 특성이 아니라 **단위가 다르기 때문**이다 — 띄어쓰기가 없는
언어는 어절이 아니라 공백 제거 후 글자를 센다. 그래서 **언어 간 절대값 비교는 성립하지
않는다.** 발화속도는 총 길이가 아니라 `speech_ms`(첫 발화 시작 ~ 마지막 발화 끝)로 잰다 —
FLEURS 녹음은 앞뒤 무음이 1~2초씩 있어 총 길이로 재면 과소평가된다.

검증 타깃은 기본 풀(English, Korean, Japanese, Chinese, Spanish, German)에서 **소스 언어만
뺀 5개**이고, 목적함수는 타깃별 z-정규화 effective 의 평균이다. `--target-aware` 런
(`run13ta-*`) 만 타깃이 1개다.

## 7. 벽시계 시간과 비용

| 런 | 시작 → 종료 | 소요 | 기계 | 비용 |
|---|---|---|---|---|
| run13 | — (launch 로그 없음) | 최종 test 평가만 3,113초 기록 | B | $10.72 |
| run13ta-zh | 08-31 00:32 → 03:43 | 3h 11m | B | $15.28 |
| run13ta-ja | 08-31 13:36 → 18:45 | 5h 09m | B | $18.07 |
| run13ta-de | — | — | B | $11.87 |
| de-en/run02 | iter_00 (기계 B) + 09-01 00:46 → 03:18 | 2h 32m (기계 A 구간) | B→A | $17.61 |
| ja-en/run02 | 09-01 03:18 → 06:56 (예산 초과 종료) + 11:18 → 13:59 재개 | 3h 38m + 2h 41m | A | $42.40 |
| zh-en/run02 | 09-01 06:56 → 11:17 (예산 초과 종료) + 13:59 → 14:23 재개 | 4h 21m + 24m | A | $26.38 |

run13 계열 4런 합계 **$55.94**, x2en 세 트랙 합계 **$86.39**. 트랙당 상한 $25 로 출발했고
ja/zh 가 벽에 걸려 `--resume` 으로 한 번씩 더 돌렸다 (ja +$17.39, zh +$1.38).
x2en 세 트랙은 순차 실행이라 기계 A 를 **연속 14시간** 점유했다.

집계는 `iter_*/metrics.json` 의 런 누적값에서 런당 최댓값을 취하고 로그의 게이트웨이
추정값과 비교해 큰 쪽을 쓴다:

```bash
PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.infra.cost_report --run-id en-multi/run13
```

## 8. 재현

```bash
# x2en 세 트랙 (기계 A 기준). history.json 이 있으면 자동으로 --resume 을 붙인다.
bash core/meaning_segmentator/tools/autoseg_x2en/run_chain.sh

# 단일 트랙 직접 실행
PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.loop \
  --dataset fleurs-de-en --src-lang German --tgt-lang English \
  --pair-id de-en --run-id run02 \
  --model gpt-5-mini --provider openai \
  --agent-reasoning-effort none --seg-reasoning-effort none \
  --iterations 5 --train 40 --dev 265 --test 100 \
  --patience 5 --budget 25 --workers 24 \
  --translate-backend local \
  --adequacy-backend cometkiwi --consistency-backend nli --adopt-se-mult 0.5
```

사전 요건 둘: `OPENAI_API_KEY`, 그리고 **CometKiwi 는 HF 게이트 모델**이라
huggingface.co 에서 라이선스에 동의하고 `hf auth login` 을 해 둬야 한다. 안 하면 루프가
시작 즉시 실패한다.

오래 도는 작업이므로 tmux 로 띄운다 (에이전트 백그라운드 실행은 부모가 죽을 때 함께
죽는다). tmux 서버는 기존 환경을 안 물고 있으므로 스크립트 안에서 `set -a; . ./.env; set +a`
로 키를 직접 읽고, 파이썬에는 `-u` 를 붙여 로그가 버퍼링으로 0바이트가 되지 않게 한다.

## 9. 논문에 쓸 때 반드시 함께 적을 것

1. **두 묶음의 실행 기계가 다르다** (§0). 속도·처리량을 합쳐 쓰면 안 된다.
2. **기계 B 의 PyTorch 는 GB10 을 공식 지원하지 않는 빌드였다** (§1).
3. **분절이 비결정론적이다** — temperature 고정이 불가능하다 (§3). 그래서 채택 판정이
   단일 실행 점수가 아니라 쌍체 Δ 다.
4. **`--seg-reasoning-effort none` 은 사고를 끄지 않는다** (§3).
5. **언어 간 절대값 비교 불가** — de 는 어절, ja/zh 는 글자다 (§6).
6. **x2en 세 트랙의 실제 개정 평가 횟수가 다르다** (de 4 / ja 3 / zh 3). 사유는
   [README.md](README.md) 의 "재개가 슬롯을 먹는 자리".
7. **`run13ta-zh` 는 en-de 매니페스트로 돌았다** (§5).
8. 기계 B 의 CPU·RAM·패키지 버전은 아직 미확인이다 (§1).
