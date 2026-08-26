# autoseg — 의미 분절 프롬프트 자동 생성 루프

사람이 언어마다 `<SEG>` 삽입 프롬프트를 직접 쓰던 작업을 에이전트 루프로 대체한다.
입력은 평문 문장 데이터와 언어쌍뿐이고, 출력은 **프롬프트**와 그 프롬프트로 분절된 데이터다.

- 설계와 근거: [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md)
- 문헌 대조: [../SEGMENTATION_CRITERIA_RELATED_WORK.md](../SEGMENTATION_CRITERIA_RELATED_WORK.md)
- NLI 대체 후보 탐색: [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md)
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
| `contradiction` | NLI(full 번역, 누적 방출분) 모순 확률. **문장 값 = 경계 (k−1)개 평균, 무분절은 미정의** | `effective` 를 통해 |
| **`effective`** | **`adequacy × (1 − contradiction)`**. 무분절은 미정의 (`n_effective`) | **최대화** |
| `laal_words` | Length-Adaptive Average Lagging (소스 어절) | 보고만 |
| `missing_boundaries` | 예산이 요구한 경계 중 못 준 개수 | 요건 (§검증기) |
| `consistency` | 합본 vs 전체 번역의 **양방향 NLI entailment 의 min** = v1 `Q` 의 어순 무관 후계 | 보고만 |
| `premature_rate` | 판정자가 조기 방출로 본 경계 비율. **test 는 무작위 표본** (루프 중엔 실패 조준이라 상향 편향) | 부록 |
| `reference_suspect_rate` | 판정자가 오라클(full 번역) 자체를 의심한 비율. 높으면 contradiction·consistency 오염 신호 | 부록 |
| **`rank_contra_gap`** | 순위 **하위 절반 − 상위 절반**의 경계 contra 차. 잡음 바닥 보정 후. 양수 = 절단이 위험을 덜어냄, **0 이하 = 순위 무정보** | `focus="priority"` 판정 |
| `rank_contra_spearman` | 같은 축의 방향만 보는 보조값. **raw 라 길이 교란 포함** — 음수면 `noise_floor.py --recheck-t` 로 보정 확인 | 진단 |

임의 상수는 T 격자 하나뿐이다. v1 의 `Q_floor`·`ratio`·`q_weight`·`z` 는 전부 사라졌다.

**두 실패를 다른 지표가 맡는다.** `adequacy` 는 조각 자체 오역(F1)을 잡지만 조기 방출(F2)은
원리적으로 못 본다 — `(조각 원문, 조각 번역)` 만의 함수라 미래의 반박을 알 수 없고, 실측에서
오히려 **조기 완성을 보상했다**(케이스 5건 중 4건 순위 위반). `contradiction` 이 그 자리를
메우고, 곱셈이라 새 상수가 안 생긴다.

**`min_gap` 은 지표가 아니라 사용 요건으로 정해진 값이다 — 되돌리기 전에 이걸 읽을 것.**
7언어쌍 스윕(같은 마킹을 `min_gap` 만 바꿔 재절단·재채점, `mg=0` 대비 **같은 지연에서**,
ko→{en,de,es,ja,zh} + en→de + en→ko, `xlmr-anli`):

| | mg=2 | mg=3 | mg=4 |
|---|---|---|---|
| `effective` 평균 | −0.0032 | −0.0029 | −0.0361 |
| `consistency` 평균 | +0.0162 | −0.0116 | −0.0234 |
| 1위 한 언어쌍 (14비교 중) | 4 | **0** | 1 |

**`mg=3` 은 한 번도 이기지 못했고, 언어로 일반화되는 최적값도 없다** — 소스가 같은
en-de(mg4 최고)와 en-ko(mg4 최악)가 정반대다. `mg=2` 의 consistency 이득(+0.016)은
양수 4개가 전부 한국어 소스 그룹인데 그 5쌍은 **같은 분절을 공유**하므로 독립 관측 1개다.

그런데 **두 지표 모두 이 값이 막으려는 실패를 볼 수 없다.** `adequacy` 는 (조각 원문,
조각 번역) 쌍만 보므로 `'What' → 'What'` 을 충실한 번역으로 채점하고, `consistency` 는
합본만 보므로 어디서 잘랐든 값이 같다. 실제 관측된 출력:

```
What <SEG:1> are <SEG:2> you <SEG:3> working <SEG:4> on?
  min_gap=0 →  "What / are you working on?"     한 단어를 방출한다
  min_gap=3 →  "What are you working on?"       무분절 (옳다)
```

청자에게 무의미한 한 단어 방출은 지표가 아니라 사용 요건으로 막는다. 지표 숫자만 보고
기본값을 0 으로 되돌리면 이 실패가 그대로 돌아온다.

**무분절은 절단기에서 나온다 — 지표나 예산이 아니라 `min_gap` 이 만든다.** `chunk_budget`
의 하한이 2 라 T 를 아무리 키워도 경계 1개는 요구되지만, `min_gap` 을 만족하는 자리가
0개인 짧은 문장은 절단기가 **경계를 못 놓고 통째로 내보낸다** (run05 실측: min_gap=3 에서
1/150, =4 에서 7/150. T 를 키워도 이 잔여는 상수라 **T 의 함수가 아니라 문장 고유 성질**이다).
예전에는 순위 순 보충 루프가 이걸 덮어 강제로 잘랐다 — en-de 에서 한 번도 발동을 안 해
무해해 보였는데 ko-en 에서는 T=2 에서 150/150 발동한다. 보충을 뺀 것이 그 변경이다.

**contradiction 의 문장 값은 경계 평균이다 — 조각 가중 평균이 아니다.** 마지막 조각의
구조적 0 을 평균에 넣으면 k 가 클수록 값이 기계적으로 올라, 곡선 기울기가 측정이 아니라
지표 정의에서 나온다 (run03 재집계 실측: 조각 가중 평균의 effective 기울기 0.667→0.732 가
경계 평균에서 **전부 소멸** — per-boundary 위험은 T 무관 ~0.16 평탄). 경계 평균은 노출이
정규화되어 **k≥2 인 점들 사이의 비교가 유효**하다. 무분절은 경계가 없어 0(무죄)이 아니라
**미정의(None)** 이고, 곡선의 점이 아니라 offline 기준선으로 병기한다.

**consistency 가 논문 주 곡선의 y축이다** — "지연을 얼마나 사면 offline 번역의 의미에서
얼마나 멀어지나"의 직접 측정값 (무분절 = 1.0 은 축의 기준점, 상한선으로 그린다).
run03 재집계에서 유일하게 기울기가 살아있는 축이다 (laal 2.0→3.3 에서 0.55→0.76).
`effective` 는 루프 목적함수 전용 — consistency 는 복구 마스킹을 못 벌하고 NLI 확률
포화라 0.003 규모 개선 검출이 안 되므로 목적함수로 쓰지 않는다 (설계 §11.1).

**consistency 가 양방향인 이유** — 함의는 비대칭이다. `ent(full ⇒ 합본)` 은 환각·왜곡을,
`ent(합본 ⇒ full)` 은 누락을 잡고, min 이라 어느 쪽 실패든 걸린다. COMET consistency 는
참조 기반이라 어순을 단조화한 좋은 분절을 감점했다(관문 실측 soft 위반 12건). NLI 는
명제만 봐서 그 편향이 없다(soft 위반 0건). 단 **고유명사 음역 변이를 다른 개체로 읽는
맹점**이 있다 — ja-ko 관문 케이스에서 확인. en 타깃은 깨끗이 통과, 비영어 타깃은 관문
결과를 먼저 볼 것.

**잡음 바닥은 두 성분이다 — 편향은 고칠 수 있고 잡음은 못 줄인다.** 자기-prefix 2292쌍
실측: 바닥은 hypothesis 길이와 Spearman −0.670 으로 강하게 얽혀 있지만(짧을수록 높다),
**어떤 covariate 로 10분위 보정해도 산포는 0.1036 → 0.094 로 9% 밖에 안 준다.**
커버리지 비율(hyp/premise 어절)로 바꿔도 −0.664 로 사실상 동등하니 covariate 교체는
이득이 없다. 즉 `noise_floor.py` 의 보정은 **앞쪽 경계의 구조적 불리함(편향)** 을 고치는
것이지 잡음을 줄이는 것이 아니다.

남은 산포 0.094 가 검출 한계를 정한다 — 경계 1003개에서 표준오차 ≈ 0.094/√1003 ≈
**0.003** 으로, 루프가 검출하려는 프롬프트 차이(0.003)와 **같은 자릿수**다. 그래서
`paired_delta` 의 쌍체 비교가 선택이 아니라 필수다(같은 문장을 쓰면 조각 고유의 특이성이
상쇄된다). 편향 쪽 근본 처방은 `retrans` premise 다 — 커버리지 비율 중앙값이
0.214 → 0.500 으로 오르고 짧은 구간 바닥이 35% 준다. 커버리지 1.0 은 미래를 반영하는
번역기(=LLM)가 있어야 가능하고, gtx 로는 구조적으로 0.5 가 상한이다.

**비영어 타깃에 `deberta-mnli` 를 쓰면 값이 뒤집힌다.** 최소쌍 진단(`../metric_probes/runs/minimal_pairs/`)
에서 한국어 `김 대리가 박 과장에게 보고서를 넘겼다` 기준으로 **역할 교환문의 함의가
0.9777, 진짜 재서술이 0.07** 로 나왔다 — 완전한 역전이다. 영어 전용 모델을 언어 밖에서
쓴 결과이고, `contra_alt` 실측에서 유일한 관문 위반이 ja-ko 케이스였던 것과 같은 원인이다.
**~~타깃이 영어가 아니면 `mdeberta-xnli`~~ — 이 처방은 폐기됐다.** 기본값이 `xlmr-anli`
(`vicgalle/xlm-roberta-large-xnli-anli`)로 바뀌었고 **타깃 언어와 무관하게 이것 하나면 된다**
(en-de test 100문장 + 관문 6케이스 실측, 2026-08-19):

| | `mdeberta-xnli` | `xlmr-anli` |
|---|---|---|
| 관문 최소 여유 | 0.0027 (통과선상) | **0.0994** |
| 5개 타깃 곡선 정상 | 2/5 (ko·zh·ja 역전) | **5/5** |
| 잡음 바닥 | 0.102 — 실측 신호 0.075 보다 커서 **무정보** | — |
| dev 쌍체 se | 0.0065 | 0.0144 (분산 배증) |

대가는 문장별 분산이다 — 채택 문턱이 `Δ > adopt_se_mult·se` 라 그만큼 보수화되므로
`--adopt-se-mult` 기본값을 1.0 → **0.5** 로 함께 낮췄다.

## 실행

```bash
# 의존성: pip install -r core/meaning_segmentator/requirements.txt
# .env 에 API 키 필요 — 엔드포인트는 키 접두사로 정해진다 (`sk-` → OpenAI, 그 외 → Letsur).
#   읽는 순서: LETSUR_API_KEY → OPENAI_API_KEY → CLAUDE_API_KEY. `AUTOSEG_BASE_URL` 이 최우선.
# CometKiwi 는 HF 게이트 모델 — 라이선스 동의 + `hf auth login` 선행
#   (huggingface_hub 1.x 에서 huggingface-cli -> hf 로 개명됨)

# 0) 판정자 관문 — 판정자 모델/프롬프트를 바꿨다면 여기부터
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.judge_check --repeats 3

# 1) 지표 타당도 — consistency 백엔드를 바꿨다면 (nli-* 포함)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.validity_check --backends nli-mdeberta nli-deberta comet

# 1b) adequacy 조각 게이트 — adequacy 백엔드를 바꿨다면
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.adequacy_check

# 1c) contradiction 잡음 바닥 + 순위 정렬 재검 (기존 런 재활용, 번역 호출 0)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.noise_floor \
    --run-id ko-en/run03 --split test --recheck-t 2

# 2) 루프
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run13 \
    --iterations 6 --train 30 --dev 60 --test 100 --min-chars 25 --budget 20

# 3) 비교군을 같은 자로 평가 (사람 프롬프트는 순위 태그가 없으므로 --no-priority)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
    --run-id ko-en/run13 --split test --label human_current --no-priority
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--dataset` | `kspon` | 등록된 이름(`data.DATASETS`) 또는 매니페스트 경로(`.jsonl`) |
| `--model` | `gpt-5-mini` | 분절·에이전트 모델. en-de test 100문장 실측에서 `gpt-5.4-mini` 대비 **비용 1/3.9 에 품질 차이 검출 안 됨**(쌍체 t=−1.0~0.0). `gpt-5-nano` 는 지시를 못 따라 커버리지 1/15 로 실격 |
| `--judge-model` | `--model` | 판정자. **분절기와 다른 모델을 쓰면 순환이 준다** |
| `--t-grid` | `3 6` | 루프가 쓰는 목표 조각 크기. **다른 격자로 잰 `score` 와 비교 불가** |
| `--final-t-grid` | `2 3 4 6` | 최종 test 곡선용 격자 |
| `--main-t` | 격자 중앙값 | 판정자가 도는 주 작동점 |
| `--judge-rows` | `8` | 이터레이션당 판정할 문장 수. 예산 절반은 `contradiction` 최상위, 절반은 `adequacy` 최하위 — 두 실패 유형을 각각 겨냥한다 |
| `--no-judge` | — | 판정자를 끄고 adequacy 만으로 조향 |
| `--adequacy-backend` | `cometkiwi` | `cometkiwi` / `cometkiwi-xl` |
| `--contradiction-backend` | `xlmr-anli` | 조기 방출 NLI. **다국어 — 타깃별로 바꿀 필요 없다.** `mdeberta-xnli` 는 잡음 바닥(0.102)이 실측 신호(0.075)보다 커 사실상 무정보이고 ko/zh/ja 타깃에서 곡선이 역전한다(5개 타깃 중 2개만 정상). `deberta-mnli` 는 영어 전용 |
| `--adopt-se-mult` | `0.5` | 채택 요건 `dev 쌍체 Δ > k·se`. 점 비교는 오차막대 안 잡음까지 채택했다. `0` = 이전 방식. **1.0 → 0.5 로 낮춘 이유가 `xlmr-anli` 다** — 지표 타당도가 훨씬 나은 대신 문장별 분산이 커서 dev 쌍체 se 가 0.0065 → 0.0144 로 배증한다. 배수를 그대로 두면 문턱이 두 배가 되어 채택이 더 어려워진다 (run01~03 이 이미 채택 0회) |
| `--no-coverage-rule` | — | 최소 경계 수 요건 해제. **노브가 k 를 통제 못 하게 된다** |
| `--no-contradiction` | — | NLI 해제. `effective = adequacy` 가 되어 조기 방출이 안 벌받는다 |
| `--consistency-backend` | `nli` | `nli`(양방향 entailment, 모델은 `--contradiction-backend` 를 따름) / `comet` / `xcomet` / `embed` / `chrf` |
| `--min-gap` | `3` | 절단 시 경계 간 최소 어절 간격 = **조각 길이의 절대 하한**. **지표는 이 값을 지지하지 않지만 켜 둔다** — 아래 표 참조. T 는 평균이라 1어절 조각을 못 막는다 (ko-en/run05 실측 T=2 에서 51%, T=6 에서도 14%). T 에 비례하지 않으므로 저지연 작동점에서 유일하게 듣는 제약이고, 짧은 문장의 **무분절이 나오는 유일한 경로**다. **T 는 이 값의 1.5배 이상**일 것 — `T ≤ min_gap` 인 점들은 서로 완전히 같은 분절이 된다 |
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
| `pipeline.py` | A2 Segmenter / A3 Validator / **A4 Truncator** / A5 Google 번역 + 캐시 | 분절만 |
| `metrics.py` | A6 Scorer — `adequacy`(QE) / `consistency` / `laal_words` / `score` | — |
| `agents.py` | A1 Profiler / **A7 Judge** / A8 Critic / A9 Prompt Engineer / Compressor | ● |
| `loop.py` | A11 Loop Controller — T 격자 평가, 채택·롤백·중단, 곡선·비교군·리포트 | — |
| `eval_prompt.py` | 임의 프롬프트 1개를 루프와 동일 지표로 평가 | — |
| `validity_check.py` | consistency 백엔드 타당도 게이트 — 오류 주입 후 순위 확인 | — |
| `adequacy_check.py` | **adequacy 백엔드 조각 입력 게이트** — QE 가 조각에서도 오류 순위를 지키는지 | — |
| `noise_floor.py` | **contradiction 잡음 바닥 측정** — full 번역 자기-prefix 의 NLI base rate. `--recheck-t` 로 바닥 보정 순위 정렬도 재계산 | — |
| `judge_check.py` | **판정자 + NLI 타당도 게이트** (`--skip-judge` 로 NLI 만 검사 가능) | ● |
| `validity_cases.json` / `premature_cases.json` | 고정 케이스. **사람이 작성**, LLM 생성 아님 | — |
| `adequacy_cases.json` | 조각 오류 주입 케이스 — 실제 발화 조각 기반. 문안은 사람 확정 전 (잠정) | — |
| `human_prompts/` | 사람 작성 한국어 프롬프트 3종 (비교군) | — |

**지표 대체 실험 코드는 [`../metric_probes/`](../metric_probes/) 로 옮겼다.** 루프 경로에 안 들어가는 탐침들이라 여기 두면 `autoseg/` 가 무엇을 실제로 쓰는지가 흐려진다. 결론은 [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md).

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

ko-en v2 런(`runs/ko-en/run01`~`run03`)의 `effective`·`contradiction` 은 **구 집계
(조각 가중 평균)** 이고 `consistency` 는 COMET 이라, 경계 평균·양방향 NLI 로 재집계하기
전에는 새 런과 비교할 수 없다. 재집계는 `*_rows.json` 의 `pieces_contra` 로 오프라인
가능하다 (재번역 불필요).

## 새 언어 추가

**코드를 안 고치는 것이 기본이다.** `evaluation/ast/manifests/*.jsonl` 형식이면
경로를 그대로 넘긴다 — 프롬프트도 로더도 손대지 않는다.

```bash
--dataset evaluation/ast/manifests/fleurs_en-fr_test.jsonl
```

자주 쓸 이름은 `data.py` 의 `MANIFESTS` 에 한 줄로 등록한다.

```python
MANIFESTS = {
    "fleurs-en-fr": _AST / "fleurs_en-fr_test.jsonl",
}
```

`LOADERS` 에 함수를 추가하는 것은 **파일 포맷 자체가 고유할 때만**이다
(KsponSpeech 의 JSON 두 구조 흡수 같은 전처리).

## 관문 두 개 — 루프보다 먼저

둘 다 루프 밖, 데이터 무관, 1회성이다.

| | 대상 | 통과 조건 |
|---|---|---|
| `validity_check.py` | consistency 백엔드 (`nli-mdeberta`/`nli-deberta`/`comet`/…) | 심각한 의미 오류 점수 < `benign_minimal` |
| `adequacy_check.py` | **adequacy 백엔드 (QE 조각 채점)** | 조각 케이스마다 심각한 오류 < `benign_minimal` |
| `judge_check.py` | 판정자 (모델 + 프롬프트) | `safe`/`not-safe` 오분류 0건 **+ 반복 실행 동일** |
| `judge_check.py --skip-judge` | **NLI contradiction 백엔드** | 케이스마다 `min(premature) > max(safe)` |

adequacy 관문 실측 (`runs/adequacy_validity/`): 부정 뒤집힘·의미 변경·무관 문장은 전
케이스 정상 검출. 위반 2건은 관용구 조각("밀려 썼던") 하나에 국한 — 특히 **source_echo
(원문 그대로 반환)를 정답 번역보다 높게** 주는 복사 편향. 번역 층의 `looks_untranslated`
재시도가 1차 방어이나, 관용구 밀도가 높은 데이터에서는 adequacy 를 과신하지 말 것.

양방향 NLI 관문 실측 (`runs/validity_nli/`): **en 타깃 4케이스 위반 0** (두 모델 모두),
soft 위반(재서술 편향) comet 12건 → nli 0건. 위반은 전부 ja-ko 케이스 — mdeberta 가
고유명사 음역 교체(병십→헤이주)를 다른 개체로 읽는다. 비영어 타깃에서 nli consistency 를
쓰려면 이 맹점을 감수하든지 comet 을 유지할 것 (comet 도 같은 케이스에서 role_swap 위반).

**임베딩 유사도(semantic similarity)로 NLI 를 대체할 수 없다** (`../metric_probes/runs/embed_vs_nli/`,
`../metric_probes/embed_check.py`, run04 경계 1003개 + 관문 고정 케이스). MTEB 상위 다국어 모델 4종
(`multilingual-e5-large-instruct`, `Qwen3-Embedding-0.6B/4B`, `gte-multilingual-base`)
어느 것도 contradiction 관문(0/6 필요)을 통과하지 못했고, 관문 신호
(`mean(premature) − mean(safe)`)가 대부분 **음수**였다 — 잘못 자른 방출이 안전한 방출보다
full 번역에 *더* 가깝게 나온다 (ko-en-p01: deberta 0.9971 vs 0.0221, e5 0.1910 vs 0.2132).
**원인은 부정 맹목이 아니다** — 통제된 최소쌍(`../metric_probes/runs/minimal_pairs/`)에서 코사인은 부정을
정확히 가른다(재서술 0.975 > 부정 0.871). 실제 원인은 둘이다: (a) **영어 참여자 뒤바뀜**
은 어순이 역할을 나르므로 토큰 집합이 그대로라 안 잡힌다(재서술 0.992 < 역할 교환 0.996),
(b) **미완성 조각 교란** — 내용이 희박한 안전 조각(`As for that,`)이 유창하지만 틀린
방출보다 참조에서 멀다. 길이 잡음 바닥을 구조적으로
없앤 `+align` 구성(`1 − max_i cos(hyp, full 의 i-어절 prefix)`)도 부호를 되돌리지 못했다.
consistency 자리에서도 4종 모두 T2 를 탈락했다 (`negation_flip` ≥ `benign_minimal`).

**단 기각된 것은 임베딩이 아니라 코사인이다** (`../metric_probes/runs/embed_probe/`, `../metric_probes/embed_probe.py`).
인코더를 얼린 채 pair feature 위에 헤드만 MNLI 로 학습하면 모순 정보가 실제로 나온다:
MNLI dev 3-way 정확도가 `cos` 0.440 → `[u,v,|u−v|,u∘v]`+MLP **0.719**, 모순 AUC
0.723 → 0.899. 관문 신호의 **부호도 음수에서 양수로 뒤집힌다**(`cos` −0.014 →
`|u−v|` +0.146). 그러나 같은 데이터에서 교차 인코더는 0.916 / AUC 0.991 이고,
**모순 오류율로 보면 0.101 vs 0.009 로 11배**다. bi-encoder 는 두 문장이 풀링 전에
서로를 못 봐서 토큰 정렬이 남지 않는다 — 헤드를 키워도 복구되지 않는다(+0.03).
관문은 최선 2/6 위반으로 여전히 탈락(코사인 단독은 5/6).

**오라클을 안 쓰는 소스 쪽 대안도 목적함수를 대신하지는 못한다** (`../metric_probes/runs/future_dep/`,
`../metric_probes/future_dep.py`). `fd = 1 − cos(소스 prefix 단독 표현, 문장 안에서의 같은 prefix 표현)` 은
미래 소스가 이미 방출된 구간의 읽기를 바꾸는 정도를 재는데(양방향 인코더 전용 — causal
디코더는 구조상 fd ≡ 0), 위치 교란(남은 어절 수. 실측 8배 차이)을 없앤 뒤 실측
contradiction 과의 순위 상관이 최대 0.28 이다. 임베딩 코사인(0.03~0.10)보다는 낫지만
대체재로는 낮고, 한계도 같다 — 미래가 읽기를 *얼마나* 바꾸는지만 알고 *어느 방향으로*
바꾸는지는 모른다. 좋은 절단은 대개 절 경계라 fd 가 원래 높아서(프롬프트가 고른 절단이
이웃 위치보다 높다), 정제와 반박이 섞인다. 값싼 **사전 필터**로는 쓸 수 있다.
참고: Google gtx 는 개행으로 붙여도 줄 간 문맥을 전파하지 않으므로(실측 4/4 동일),
"같은 구간의 미래 인지 렌더링"을 만들려면 LLM 번역기가 필요하다.

**표현 급변점도 분절 경계가 아니다** (`../metric_probes/runs/boundary_probe/`, `../metric_probes/boundary_probe.py`).
어절을 하나씩 붙이며 표현 변화가 큰 곳을 경계로 제안하는 구성 3종(`delta_prefix`,
TextTiling 식 `tile`, 문맥 표현 위의 `ctx_delta`)은 raw AUC 0.61~0.81 로 유망해 보이지만,
**문장 내용을 하나도 안 보는 위치 사전확률**(상대 위치 10분위의 코퍼스 평균)이 전 항목에서
더 높다 (AUC 0.685~0.864, 최대 상승폭 **−0.049**). 위치 교란을 뺀 뒤에는 AUC 0.42~0.54 로
무정보에 수렴한다. prefix 가 길수록 한 어절의 비중이 줄어 점수가 단조 감소하는데, 그
감소 곡선이 LLM 경계의 위치 분포(노브 T 가 만든 등간격 구조)와 겹쳤을 뿐이다. 새 지표를
경계 일치로 검증할 때 **무작위(0.235)가 아니라 위치 사전확률(0.541)을 기준선으로 둘 것.**

**대체 후보 실측** (`../metric_probes/runs/contra_alt/`, `../metric_probes/contra_alt.py`, [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md)).
실험을 **premise 축**(`oracle` = full 번역 / `retrans` = 다음 조각까지의 소스 재번역) ×
**scorer 축**(`nli`/`summac`/`minicheck`/표면형 `erasure`)으로 인수분해해 잰 결과:

- **SummaC 식 창 집계는 기각.** 길이 바닥을 없애려던 것이 반대로 0.030 → 0.257 로
  올렸다. 창 max 는 잡음의 최대값을 뽑고, hypothesis 가 짧을수록 창이 많아진다.
- **MiniCheck**(`lytang/MiniCheck-DeBERTa-v3-Large`)는 raw·floor·z 세 변이 모두 0/6 으로
  **바닥 보정 후에도 통과하는 유일한 백엔드**이고 바닥이 평탄하다(0.61→0.48. 현행 NLI 는
  낮지만 45배 기울어져 있다). 그러나 `benign_reordered`(0.7895)를 `benign_incomplete`
  (0.0938)보다 8배 나쁘게 준다 — ko→en 어순 단조화를 벌하는 것은 COMET consistency 를
  버린 이유와 같은 편향이라 **보류**.
- **`retrans` premise 가 최고 SNR 5.88**(현행 5.74). 바닥이 더 낮고 평탄하며(1–2어절
  0.0874 vs 0.1353, 35% 감소) **오라클이 필요 없어 `reference_suspect` 오염이 구조적으로
  사라진다.** 비용은 경계당 gtx 1회. 관문 위반 1건은 영어 전용 모델을 쓴 ja-ko 케이스뿐이고
  en 타깃은 5/5 통과. 표면형 erasure 는 gtx 가 매 호출 독립이라 접두 보존이 관측되지 않아 사망.

**`contradiction` 은 MU 정의와 핀트가 다르다 — 그 차이를 수치화했다**
(`../metric_probes/runs/fixed_point/`, `../metric_probes/fixed_point.py`). Zhang+ 2020 의 Meaningful Unit 은 *후속 텍스트에
의해 번역이 바뀌지 않는 최소 구간*이고, 반박은 그 위반의 **한 종류**일 뿐이다.
소스 어절 prefix 를 전부 gtx 로 번역해 궤적을 만들고 `1 − P(S_j ⊨ S_i)` 로 불안정도를 재면:

- **폐기된 것은 개념이 아니라 자였다.** 엄격한 고정점 비율이 표면형 `chrf` 로는 1.1%,
  어순 무관 `entail` 로는 **25.1%**. v2 가 prefix-consistency 를 폐기한 사유(어순 오탐)가
  그대로 확인되고, 판정만 함의로 바꾸면 4분의 1이 살아난다.
- LLM 은 **더 안정적인 곳에서 자른다** — 단 위치를 통제해야 보인다 (raw AUC 0.529 →
  위치 사전확률 0.766 → 잔차 0.447). **이 축의 기준선은 무작위가 아니라 위치 사전확률이다.**
- 현행 contradiction 과 순위 상관 0.592(잔차 통제 후 0.576) — **분산의 약 35% 만 공유**한다.
  방향성 있는 척도가 이긴다: `entail` 0.592 > `chrf` 0.403 > `cos` 0.375.

`contradiction` 은 "틀린 것을 보여줬는가", 고정점은 "고쳐야 하는가" 를 묻는다. 수정이
없는 우리 경로에서는 둘 다 실패지만 심각도가 다르므로 현행 선택도 방어 가능하다.
`entail`+`final` 은 지평 중 가장 좋고 가장 싸며(문장당 NLI n 회) 오라클이 필요 없어,
목적함수를 건드리기 전에 **보고 지표로 얹어 보는 것**이 위험이 가장 낮다. 다만 이 값은
절단 위치의 성질이라 현행 관문(렌더링 변이)으로는 판정할 수 없다.

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

- **결정론은 현재 어느 OpenAI 모델로도 얻을 수 없다.** `gpt-5-mini` 는 `temperature=0` 을
  400 으로 거부한다. `gpt-5.4-mini` 는 **사고를 끈 기본 상태에서만** 받아주는데, 그 상태로는
  태그를 필요량의 1/3 밖에 안 찍어 실사용이 불가하다 — `reasoning_effort` 를 켜는 순간
  똑같이 거부한다. 즉 **실사용 설정에서는 분절기가 항상 비결정론적**이고, `paired_delta`
  쌍체 비교가 선택이 아니라 필수다 (설계 §8.6-5). 결정론이 요건이면 Letsur
  `claude-sonnet-5` 뿐이다.
  `gateway._post` 는 거부를 플래그로 기억해 떼고 가므로 죽지는 않는다 — 기억하지 않으면
  **매 호출이 400 왕복을 한 번씩 낭비**한다 (run05 실측: POST 493건 = 400 247 + 200 246).
- 게이트웨이의 thinking 모델(`claude-sonnet-5`, `gpt-5*` 계열)은 **사고 토큰이
  `max_tokens` 에 함께 잡힌다.** `pipeline.py` 의 `SEG_MAX_TOKENS = 8192` 를 줄이면 긴
  문장에서 빈 출력이 나오고 포맷 통과율이 1.0 에 도달하지 못한다.
- `unbabel-comet` 을 쓰려면 **`setuptools<81` 을 핀해야 한다.** 함께 설치되는
  `torchmetrics 0.10.x` 가 `pkg_resources` 를 import 하는데 setuptools 81+ 에서 제거됐다.
- **CometKiwi 는 HF 게이트 모델**이다. `huggingface.co` 에서 라이선스에 동의하고
  `hf auth login` 을 먼저 해야 `adequacy` 백엔드가 뜬다 (huggingface_hub 1.x 에서 `huggingface-cli` -> `hf` 로 개명).
- 조각 번역 호출이 T 격자 크기에 비례한다. 루프 기본 격자를 `3 6` 두 개로 둔 이유가 이것이며,
  전체 격자는 최종 test 에서만 돈다.
- **비용은 사실상 분절 호출의 사고 토큰 하나다** (en-de run01 실측: 총액의 96%,
  출력 중 사고 97.6%, 본문은 콜당 133토큰뿐). 손잡이 우선순위는
  **모델 > `--seg-reasoning-effort` > 그 외**이고, 프롬프트 길이는 무의미하다(입력 76% 캐시).
  `--priority-depth` 로 순위 계산을 75% 줄여 봤으나 **사고가 오히려 17% 늘었다** —
  상위 N 개를 고르려면 전부 평가해야 하므로 정렬 생략이 평가 생략이 아니다.
- 프롬프트 캐싱이 걸리지 않는다 (`cached_tokens: 0`). 입력 토큰 전액 과금.
