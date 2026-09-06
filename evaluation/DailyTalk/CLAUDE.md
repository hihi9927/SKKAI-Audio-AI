# evaluation/DailyTalk/

두 가지 일을 한다. 하나는 **한국어 대화 음성 ASR 벤치마크**(메트릭 CER), 다른 하나는
**파인튜닝용 `<SEG>` 라벨을 만드는 작업장**이다. 디렉토리 대부분은 뒤쪽에 속한다.

## ASR 평가

```bash
python evaluation/DailyTalk/test_qwen3_dailytalk.py \
  --data-dir evaluation/DailyTalk/transcribe/test\(1008\).json \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

`test(1008).json` 이 전체 테스트셋, `toy(200).json` 이 빠른 확인용이다.
한국어는 단어 분절이 없어 WER 대신 **CER** 을 쓴다.

## SEG 라벨 파이프라인

```
new_seg_all.json                 랭크 라벨 원본 <SEG:n> — 2,541그룹 / 23,773발화. 이것만이 원본이다
  ├─ rank_to_seg.py            → new_seg_all_t2.json      (랭크 컷: rank <= T 인 태그만 유지)
  └─ rank_to_seg_budget.py     → new_seg_all_T{n}[_mg{g}].json  (예산 컷: 문장당 k = round(단위수/T))
       build_splits.py         → split_assign.json / partial_input.json
       generate_split_data.py  → split_train2.json + split_audio/*.wav
       assemble_dailytalk.py   → Qwen3-ASR/finetuning/data/DailyTalk/{train,val,test}.jsonl
```

두 컷 방식은 다르다. `rank_to_seg.py` 는 상위 몇 개를 남길지만 보고 문장 길이를 안 봐서,
만들어진 라벨이 autoseg 의 T 축 위에 점을 갖지 못한다. `rank_to_seg_budget.py` 는
`pipeline.truncate` 를 그대로 써서 평가·논문 곡선과 같은 절단기를 쓴다.

## 지금 학습에 들어간 라벨은 `t2` 다

`Qwen3-ASR/finetuning/data/DailyTalk/train.jsonl`(9월 4일) 의 full 발화 11,207개를 각 라벨과
대조하면 **`new_seg_all_t2.json` 과 98.1% 일치**한다 (T4 81.5%, T4_mg2 71.1%, T5 68.1%).
즉 배포된 학습 데이터는 `rank_to_seg.py` 의 랭크 컷으로 만들어졌고, `rank_to_seg_budget.py` 의
예산 컷 라벨은 **아직 학습에 쓰인 적이 없다**(9월 5일 생성, 하루 늦다).

## 학습 데이터에는 세대가 둘 있다

| 세대 | 만든 것 | 경로 |
|---|---|---|
| 1세대 (8월 24일) | `split_train.jsonl` / `train_split.jsonl` / `val_split.jsonl` / `train_and_val.jsonl` | `generate_split_data.py`(기본 인자) → `split_train2.json` → `convert_split_to_jsonl.py` |
| **2세대 (9월 4일)** | `train.jsonl` / `val.jsonl` / `test.jsonl` | `build_splits.py` → `generate_split_data.py`(`--input partial_input.json`) → `partial_all.json` → `assemble_dailytalk.py` |

같은 스크립트를 인자만 바꿔 쓴다. `generate_split_data.py` 의 파일 상단 상수
(`INPUT_JSON` / `OUTPUT_JSON`)와 docstring 은 1세대 것이 그대로 남아 있어, 그것만 보고
돌리면 2세대가 아니라 1세대를 다시 만든다.

## 파생물은 두지 않는다 — 명령으로 되살린다

2026-09-06 에 **코드로 다시 만들 수 있는 파일은 전부 지웠다.** 지우기 전에 각각을
재생성해 원본과 바이트 단위로 같은지 확인했다(전부 일치). 필요할 때 아래를 돌린다.

```bash
# 랭크 컷 라벨 (build_splits.py 와 assemble_dailytalk.py 의 기본 입력)
python evaluation/DailyTalk/utils/rank_to_seg.py                      # → new_seg_all_t2.json

# 예산 컷 라벨 — 9월 5일에 T4 / T4+min-gap2 / T5 세 벌을 만들어 두었었다
python evaluation/DailyTalk/utils/rank_to_seg_budget.py -T 4          # → new_seg_all_T4.json
python evaluation/DailyTalk/utils/rank_to_seg_budget.py -T 4 --min-gap 2
python evaluation/DailyTalk/utils/rank_to_seg_budget.py -T 5

# 분할 (seed 42 고정, 오디오 디렉토리를 훑어 존재하는 파일만 담는다)
python evaluation/DailyTalk/utils/build_splits.py                     # → split_assign.json, partial_input.json
```

`build_splits.py` 는 오디오(`Qwen3-ASR/finetuning/data/DailyTalk/audio/*.wav`, 23,773개)가
있어야 같은 결과가 나온다. 오디오가 늘거나 줄면 분할도 달라진다.

## 일회성 스크립트 — `generate_split_test.py`

test 셋의 절반을 문장 중간에서 자른 스크립트다. 호출자가 없고 한 번 돌고 끝났다 —
증거는 `Qwen3-ASR/finetuning/data/DailyTalk/split_test_state.json` 으로, 어느 항목을
잘랐는지가 거기 저장돼 있어 다시 돌려도 같은 선택을 재사용한다.

`test.jsonl` 을 **제자리에서** 고쳐 쓴다는 점이 함정이다. 이미 잘린 `test.jsonl` 에
다시 돌리면 안 된다. 새로 만들려면 `assemble_dailytalk.py` 로 `test.jsonl` 을 다시 조립한
뒤 상태 파일을 지우고 돌린다.

```bash
# 기본값이 test.jsonl / split_audio_test / split_test_state.json 을 가리킨다.
# 처음부터 다시 뽑으려면 --no-resume 을 준다 (기본은 상태 파일을 재사용하는 resume).
python evaluation/DailyTalk/utils/generate_split_test.py --no-resume
```

## 남긴 것과 그 이유

| 파일 | 왜 남기나 |
|---|---|
| `transcribe/new_seg_all.json` | 라벨 원본. 나머지 라벨은 전부 여기서 파생된다 |
| `transcribe/partial_all.json` | `generate_split_data.py --input partial_input.json --output partial_all.json` 의 산물. 되살리려면 forced aligner(GPU)로 `split_audio/*.wav` 를 다시 써야 한다 |
| `transcribe/split_train2.json` | 같은 스크립트의 **기본 인자** 산물(1세대). `Qwen3-ASR/finetuning/utils/convert_split_to_jsonl.py` 가 읽는다 |
| `transcribe/test(1008).json`, `toy(200).json` | ASR 평가 드라이버 |
| `results/train2_seg_en.json` | `generate_split_data.py` 의 입력. LLM 라벨 + 구글 번역이라 공짜로 못 만든다 |
| `results/eval_dailytalk_*.json` (4개) | `core/research/context_scoring/`, `Qwen3-ASR/finetuning/utils/convert_to_jsonl.py` 가 읽는다 |

## 지운 것 — 2026-09-06

`results/` 는 `{model}/{scope}/{tag}/` 구조였던 적이 없다. LLM 라벨링 실험 JSON 이
평평하게 쌓인 곳이고, 그 중 참조가 없는 것을 지웠다.

- 라벨러 모델 비교 `toy200_*.json` 8개, 스모크 `smoke_*.json` 3개, `fcl/`(2발화 측정)
- `dailytalk_p2.json` — train1_val 분할에 라벨·번역을 붙인 것. 참조 없음
- `transcribe/dailytalk_all.json` — `new_seg_all.json` 에서 `seg_text` 만 뺀 같은 데이터
- `transcribe/train2(10000).json`, `train1_val(9500+500).json` — 옛 분할. 지금은 `split_assign.json` 이 그 일을 한다
- CoVoST2 자료 2개(`covost2_sample300_en.json`, `covost2_sample300_llama33_ranked.json`) — AST 트랙 소관이다
