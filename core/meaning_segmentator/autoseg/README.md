# autoseg — 의미 분절 프롬프트 자동 생성 루프

사람이 언어마다 `<SEG>` 삽입 프롬프트를 직접 쓰던 작업을 에이전트 루프로 대체한다.
입력은 평문 문장 데이터와 언어쌍뿐이고, 출력은 **프롬프트**와 그 프롬프트로 분절된 데이터다.

- 설계와 근거: [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md)
- 문헌 대조: [../SEGMENTATION_CRITERIA_RELATED_WORK.md](../SEGMENTATION_CRITERIA_RELATED_WORK.md)
- 폐기된 v1 설계·지표 명세: [../docs_v1/](../docs_v1/)

## 한 장 요약

```
프롬프트는 경계를 찍고 순위를 매긴다:  … 돌려보고 <SEG:1> 결과 나오면 <SEG:2> 그때 …
검증기가 최소 개수를 강제한다 (문장 어절수 / min T − 1 개 이상)
결정론적 절단이 상위 (k−1)개만 남긴다 (k = 문장 어절 수 / T)
   → 지연은 노브 T 가 정한다.  프롬프트는 지연을 건드릴 수 없다.
   → 목적함수가 단일축이 된다:  score = T 격자 평균 effective
```

| 축 | 뜻 | 목적함수 |
|---|---|---|
| `format_pass_rate` | 포맷 검증 통과율 | 보고만 (하드 게이트 아님) |
| `adequacy` | QE(조각 원문, 조각 번역). **참조 없음** | `effective` 를 통해 |
| `contradiction` | NLI(full 번역, 누적 방출분) 모순 확률 | `effective` 를 통해 |
| **`effective`** | **`adequacy × (1 − contradiction)`** | **최대화** |
| `laal_words` | Length-Adaptive Average Lagging (소스 어절) | 보고만 |
| `missing_boundaries` | 예산이 요구한 경계 중 못 준 개수 | 요건 (§검증기) |
| `consistency` | 조각 번역 합본 vs 전체 번역 = v1 의 `Q` | 보고만 |
| `premature_rate` | 판정자가 조기 방출로 본 경계 비율 | 부록 |

임의 상수는 T 격자 하나뿐이다. v1 의 `Q_floor`·`ratio`·`q_weight`·`z` 는 전부 사라졌다.

**두 실패를 다른 지표가 맡는다.** `adequacy` 는 조각 자체 오역(F1)을 잡지만 조기 방출(F2)은
원리적으로 못 본다 — `(조각 원문, 조각 번역)` 만의 함수라 미래의 반박을 알 수 없고, 실측에서
오히려 **조기 완성을 보상했다**(케이스 5건 중 4건 순위 위반). `contradiction` 이 그 자리를
메우고, 곱셈이라 새 상수가 안 생긴다.

**`effective` 는 같은 `k` 에서만 비교 가능하다.** 무분절은 경계가 없어 모순을 낼 기회 자체가
없다. 루프 안에서는 커버리지 요건이 `k` 를 고정하므로 유효하지만, **논문 곡선의 y축은
`adequacy`** 로 하고 `effective`·`contradiction` 은 같은 지연에서의 비교표로 낸다.

## 실행

```bash
# 의존성: pip install -r core/meaning_segmentator/requirements.txt
# .env 에 CLAUDE_API_KEY (Letsur AI Gateway) 필요
# CometKiwi 는 HF 게이트 모델 — 라이선스 동의 + `hf auth login` 선행
#   (huggingface_hub 1.x 에서 huggingface-cli -> hf 로 개명됨)

# 0) 판정자 관문 — 판정자 모델/프롬프트를 바꿨다면 여기부터
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.judge_check --repeats 3

# 1) 지표 타당도 — consistency 백엔드를 바꿨다면
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.validity_check --backends comet embed chrf

# 2) 루프
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run13 --translator google \
    --iterations 6 --train 30 --dev 60 --test 100 --min-chars 25 --budget 20

# 3) 비교군을 같은 자로 평가 (사람 프롬프트는 순위 태그가 없으므로 --no-priority)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
    --run-id ko-en/run13 --split test --label human_current --no-priority
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--dataset` | `kspon` | `data.py` 의 `LOADERS` 키 (`kokoro`=ja, `kspon`=ko) |
| `--model` | `claude-sonnet-5` | 분절·에이전트 모델 |
| `--judge-model` | `--model` | 판정자. **분절기와 다른 모델을 쓰면 순환이 준다** |
| `--translator` | `google` | `llm` 또는 `google`. 운영 서버 경로는 `google` |
| `--t-grid` | `3 6` | 루프가 쓰는 목표 조각 크기. **다른 격자로 잰 `score` 와 비교 불가** |
| `--final-t-grid` | `2 3 4 6` | 최종 test 곡선용 격자 |
| `--main-t` | 격자 중앙값 | 판정자가 도는 주 작동점 |
| `--judge-rows` | `8` | 이터레이션당 판정할 문장 수. 예산 절반은 `contradiction` 최상위, 절반은 `adequacy` 최하위 — 두 실패 유형을 각각 겨냥한다 |
| `--no-judge` | — | 판정자를 끄고 adequacy 만으로 조향 |
| `--adequacy-backend` | `cometkiwi` | `cometkiwi` / `cometkiwi-xl` |
| `--contradiction-backend` | `deberta-mnli` | 조기 방출 NLI. 타깃이 영어가 아니면 `mdeberta-xnli` |
| `--no-coverage-rule` | — | 최소 경계 수 요건 해제. **노브가 k 를 통제 못 하게 된다** |
| `--no-contradiction` | — | NLI 해제. `effective = adequacy` 가 되어 조기 방출이 안 벌받는다 |
| `--consistency-backend` | `comet` | `comet` / `xcomet` / `embed` / `chrf` |
| `--patience` | `3` | dev 무개선 연속 횟수 상한 |
| `--budget` | `5.0` | 게이트웨이 추정 비용 상한. 초과 시 중단 |
| `--fresh` | — | 런 디렉토리를 지우고 처음부터 (캐시도 삭제) |

`--fresh` 없이 같은 `--run-id` 로 다시 실행하면 언어 프로파일·prompt_v0·번역 캐시를
재사용해 이어서 돈다.

## 구성

| 파일 | 역할 | LLM |
|---|---|---|
| `gateway.py` | Letsur AI Gateway 클라이언트, 재시도, 비용 집계, 예산 가드, JSON 복구 | — |
| `data.py` | A0 Data Preparer — 정규화, 층화 분할, **측정 프로파일** | — |
| `pipeline.py` | A2 Segmenter / A3 Validator / **A9 Truncator** / A4 번역 툴 2종 + 캐시 | 분절·번역만 |
| `metrics.py` | A5 Scorer — `adequacy`(QE) / `consistency` / `laal_words` / `score` | — |
| `agents.py` | A1 Profiler / **A6′ Judge** / A6 Critic / A7 Prompt Engineer / Compressor | ● |
| `loop.py` | A8 Loop Controller — T 격자 평가, 채택·롤백·중단, 곡선·비교군·리포트 | — |
| `eval_prompt.py` | 임의 프롬프트 1개를 루프와 동일 지표로 평가 | — |
| `validity_check.py` | consistency 백엔드 타당도 게이트 — 오류 주입 후 순위 확인 | — |
| `judge_check.py` | **판정자 + NLI 타당도 게이트** (`--skip-judge` 로 NLI 만 검사 가능) | ● |
| `validity_cases.json` / `premature_cases.json` | 고정 케이스. **사람이 작성**, LLM 생성 아님 | — |
| `human_prompts/` | 사람 작성 한국어 프롬프트 3종 (비교군) | — |

LLM 판단이 들어가는 곳은 `agents.py` 네 곳뿐이다. 포맷 검증, 절단, 점수, 채택 판정, 재시도는
전부 결정론적 코드다.

### 언어 무관성

에이전트 프롬프트에는 특정 언어 지식이 없다. 언어 지식은 **데이터로만** 들어간다.

- 표기 체계와 구두점은 `measured_profile.json` — 코퍼스에서 **직접 측정**한다.
  "앞 텍스트에 붙어 나오는 비율 ≥ 90%" 라는 언어 무관 규칙으로 뽑으므로 일본어 `、` 는
  포함되고 스페인어 `¿` 는 빠진다.
- 어순·문체·함정처럼 셀 수 없는 것만 `language_profile.json` (LLM 산출) 이 채운다.
- 의존 파서 같은 **언어별 자원은 쓰지 않는다** (설계 §12.1).

언어 종속이 남아 있는 곳은 `data.py` 의 로더뿐이다 — 파일 포맷과 전처리가 데이터셋 고유다.

## 산출물

```
runs/{src}-{tgt}/{run_id}/
  config.json  data/{train,dev,test}.json
  measured_profile.json    language_profile.json
  iter_NN/{prompt.txt, train_rows.json, dev_rows.json, violations.json,
           metrics.json, judgements.json, critique.json, changelog.json}
  history.json  best_prompt.txt  test_rows.json  test_judgements.json
  curve.json  final_report.md  cache/  prompt_eval/
```

`train_rows.json` 의 한 행:

```jsonc
{
  "id": "...", "text": "...", "seg_text": "… <SEG:1> … <SEG:2> …",
  "valid": true, "full_trans": "...",
  "by_T": {
    "6": {"seg_text": "…", "k": 2, "missing_boundaries": 0,
          "pieces_src": [...], "pieces_tgt": [...],
          "pieces_contra": [0.93, 0.0],       // 경계별. 마지막은 항상 0 (미래 없음)
          "effective": 0.80, "adequacy": 0.83, "contradiction": 0.04,
          "consistency": 0.91, "laal_words": 4.1}
  }
}
```

`runs/**/cache/` 와 `runs/**/*_rows.json` 은 `.gitignore` 처리했다.

기존 런(`run01`~`run12`)은 **v1 지표로 측정된 것이라 v2 수치와 비교할 수 없다.**
`gain`·`Q_floor`·달성률은 더 이상 산출되지 않는다.

## 새 언어 추가

`data.py` 의 `LOADERS` 에 로더 하나만 추가하면 된다. 프롬프트는 손대지 않는다.

```python
LOADERS = {
    "kokoro": load_kokoro,
    "my_data": lambda: load_json_entries(Path("..."), text_field="text"),
}
```

## 관문 두 개 — 루프보다 먼저

둘 다 루프 밖, 데이터 무관, 1회성이다.

| | 대상 | 통과 조건 |
|---|---|---|
| `validity_check.py` | consistency 백엔드 | 심각한 의미 오류 점수 < `benign_minimal` |
| `judge_check.py` | 판정자 (모델 + 프롬프트) | `safe`/`not-safe` 오분류 0건 **+ 반복 실행 동일** |
| `judge_check.py --skip-judge` | **NLI contradiction 백엔드** | 케이스마다 `min(premature) > max(safe)` |

NLI 는 판정자와 달리 **목적함수에 직접 들어가므로** 관문이 더 중요하다. 기준이 라벨이 아니라
**확률 순위**인 이유는, argmax 가 `neutral` 로 나와도 순위가 유지되면 임계값 없이 연속 점수로
쓸 수 있기 때문이다.

판정자 관문의 기준선은 `premature_benign` 이다. **조기 방출 자체는 죄가 아니고 뒤가
반박할 때만 문제**인데, 이를 구별하지 못하는 판정자는 "짧은 조각은 다 나쁨"으로 퇴화해
루프를 보수화한다.

최종 지표가 아닌데도 관문이 필요한 이유: 판정자는 프롬프트 개선을 **조향**한다.
지표는 틀리면 숫자로 드러나지만 조향은 조용히 발산한다 — v1 에서 `embed` 백엔드가 부정
뒤집힘에 최고점(0.9278)을 줘 5회 런이 무효가 된 것이 그 사례다.

## 환경 주의사항

- 게이트웨이의 `claude-sonnet-5` 는 thinking 모델이고 **사고 토큰이 `max_tokens` 에 함께
  잡힌다.** `pipeline.py` 의 `SEG_MAX_TOKENS = 8192` 를 줄이면 긴 문장에서 빈 출력이 나오고
  포맷 통과율이 1.0 에 도달하지 못한다. 모델을 바꿀 때 먼저 확인할 것.
- `unbabel-comet` 을 쓰려면 **`setuptools<81` 을 핀해야 한다.** 함께 설치되는
  `torchmetrics 0.10.x` 가 `pkg_resources` 를 import 하는데 setuptools 81+ 에서 제거됐다.
- **CometKiwi 는 HF 게이트 모델**이다. `huggingface.co` 에서 라이선스에 동의하고
  `hf auth login` 을 먼저 해야 `adequacy` 백엔드가 뜬다 (huggingface_hub 1.x 에서 `huggingface-cli` -> `hf` 로 개명).
- 조각 번역 호출이 T 격자 크기에 비례한다. 루프 기본 격자를 `3 6` 두 개로 둔 이유가 이것이며,
  전체 격자는 최종 test 에서만 돈다.
- 프롬프트 캐싱이 걸리지 않는다 (`cached_tokens: 0`). 입력 토큰 전액 과금.
