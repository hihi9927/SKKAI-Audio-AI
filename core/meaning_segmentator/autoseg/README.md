# autoseg — 의미 분절 프롬프트 자동 생성 루프

사람이 언어마다 `<SEG>` 삽입 프롬프트를 직접 쓰던 작업을 에이전트 루프로 대체한다.
입력은 평문 문장 데이터와 언어쌍뿐이고, 출력은 **프롬프트**와 그 프롬프트로 분절된 데이터다.

이 README 는 **어떻게 돌리는가**를 적는다. 나머지 셋:

- 무엇을 하는가 — [AUTOSEG_SIMPLIFY.md](AUTOSEG_SIMPLIFY.md) (A0~A11 각 단계)
- 왜 그렇게 됐나 · 무엇을 버렸나 — [AUTOSEG_DETAILS.md](AUTOSEG_DETAILS.md)
- 문헌 대조 — [../SEGMENTATION_CRITERIA_RELATED_WORK.md](../SEGMENTATION_CRITERIA_RELATED_WORK.md)

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
| **`rank_lift`** | 경계 **위치는 그대로 두고 순위 번호만 무작위로 섞었을 때** `effective` 가 떨어지는 폭. 순위가 실제로 일하는지를 위치와 분리해 잰다. 크면 `[Priority Rules]` 를 다듬을 값어치가 있고, 0 근처면 문제는 "어디를 찍느냐"지 "어떻게 줄 세우냐"가 아니다 | 진단 (Critic 에게 감) |
| `rank_contra_gap` | 순위 하위 절반 − 상위 절반의 경계 contra 차. **`rank_lift` 로 대체됐다** — 생존 경계 4개 이상인 문장만 세므로 순위가 실제로 일하는 큰 T 에서 정의역이 사라진다. 보고만 | — |
| `rank_contra_spearman` | 같은 축의 방향만 보는 보조값. **raw 라 길이 교란 포함** — 음수면 `noise_floor.py --recheck-t` 로 보정 확인 | 진단 |

**A7 판정자는 비율을 만들지 않는다.** 종전 `premature_rate` 는 없앴다 — 판정자는
`contradiction` 상위 경계에 `cause` / `shift` / `generalized_rule` 만 붙여 Critic 에게
넘기고, 숫자는 결정론적 지표만 쓴다 ([AUTOSEG_SIMPLIFY.md](AUTOSEG_SIMPLIFY.md) A7).

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

**비영어 타깃에 `deberta-mnli` 를 쓰면 값이 뒤집힌다.** 최소쌍 진단에서 한국어 `김 대리가 박 과장에게 보고서를 넘겼다` 기준으로 **역할 교환문의 함의가
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
# 엔드포인트는 `--provider` 하나가 정한다 (기본 letsur). 키는 그 프로바이더의 환경변수
# **한 개**만 본다 — 환경변수 > 레포 루트 `.env`. 폴백 탐색은 없다.
#   letsur → LETSUR_API_KEY   openai → OPENAI_API_KEY   local → 키 불필요(ollama :11434)
#   포트가 다른 로컬 서버는 `--provider local --base-url http://호스트:포트/v1`.
#   실제로 붙은 엔드포인트는 런의 `config.json` 에 `provider`/`api_base_url` 로 남는다.
# CometKiwi 는 HF 게이트 모델 — 라이선스 동의 + `hf auth login` 선행
#   (huggingface_hub 1.x 에서 huggingface-cli -> hf 로 개명됨)

# 0) 판정자 관문 — 판정자 모델/프롬프트를 바꿨다면 여기부터
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.judge_check --repeats 3

# 1) 지표 타당도 — consistency 백엔드를 바꿨다면
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.validity_check --backends nli comet

# 1b) adequacy 조각 게이트 — adequacy 백엔드를 바꿨다면
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.adequacy_check

# 1c) contradiction 잡음 바닥 + 순위 정렬 재검 (기존 런 재활용, 번역 호출 0)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.noise_floor \
    --run-id ko-en/run03 --split test --recheck-t 2

# 2) 루프
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run13 \
    --iterations 6 --train 30 --dev 60 --test 100 --budget 20

#    중단된 런을 이어서 (분절·번역 캐시는 남아 있고 루프 상태만 복원한다)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop ... --run-id run13 --resume

# 3) 비교군을 같은 자로 평가 (사람 프롬프트는 순위 태그가 없으므로 --no-priority)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
    --run-id ko-en/run13 --split test --label human_current --no-priority

# 4) 참조 기반 평가 — BLEU/chrF2, 그 다음 COMET (번역은 재사용, 추가 비용 없음)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.bleu_eval \
    --run-id en-multi/clean500 --targets de ja zh \
    --baselines punct syntax causal_align alignatt mu_prefix
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.comet_eval \
    --run-id en-multi/clean500 --targets de ja zh

# 5) 런 하나가 실제로 쓴 비용 (기록 + 캐시 증분 역산)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.cost_report --run-id en-multi/run07
```

주요 옵션:

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--dataset` | `kspon` | 등록된 이름(`data.DATASETS`) 또는 매니페스트 경로(`.jsonl`) |
| `--pair-id` | 언어명에서 생성 | 런 디렉토리 이름 (`runs/{pair-id}/{run-id}/`) |
| `--model` | `gpt-5-mini` | 분절·에이전트 모델. en-de test 100문장 실측에서 `gpt-5.4-mini` 대비 **비용 1/3.9 에 품질 차이 검출 안 됨**(쌍체 t=−1.0~0.0). `gpt-5-nano` 는 지시를 못 따라 커버리지 1/15 로 실격 |
| `--judge-model` | `--model` | 판정자. **분절기와 다른 모델을 쓰면 순환이 준다** |
| `--min-gap` | **코퍼스에서 유도** | 절단 시 경계 간 최소 간격 = 조각 길이의 절대 하한. 미지정 시 `1200ms × 발화속도`(단위/초)로 환산한다 — 발화속도는 강제정렬 산출물에서 읽고, 없으면 `--units-per-sec` 로 준다. `0` = 끔. 아래 표 참조 |
| `--units-per-sec` | 강제정렬 산출물 | 코퍼스 발화 속도(단위/초). 산출물이 없을 때 직접 준다 |
| `--t-floor` | `min_gap` 에서 유도 | 마킹 밀도 하한 기준 T. `max(min_gap+1, ceil(1.25×min_gap))` — 포화 바닥과 마킹 한계 중 큰 쪽 |
| `--t-grid` | `t_floor × {1, 1.5, 3}` | 루프가 쓰는 목표 조각 크기. **다른 격자로 잰 `score` 와 비교 불가** (격자 평균이므로). 격자는 `config.json` 에 남는다 |
| `--final-t-grid` | `t_floor × {1, 1.5, 2, 3}` | 최종 test 곡선용 격자 |
| `--main-t` | `--t-grid` 중앙값 | 판정자가 도는 주 작동점 |
| `--batch-size` | `6` | 한 분절 호출에 넣을 문장 수. **비용의 유일한 큰 레버** (b=1 $1.05 → b=6 $0.47). **6 을 넘기지 말 것** — b=12 부터 1차 통과율이 0.75/0.27 로 무너져 비용이 U자로 되돌아오고 품질도 유의하게 나빠진다 |
| `--judge-frac` | `0.10` | 타깃별로 판정할 **경계 비율** (contradiction 상위부터). 개수가 아니라 비율인 이유: 고정 8문장이던 시절 `cause` 6범주에 이터당 표가 중앙 5개뿐이라 Critic 이 지배적 실패를 읽을 수 없었다 |
| `--no-judge` | — | 판정자를 끈다. 사례에 '왜·어디로' 설명이 안 붙는다 |
| `--revision-candidates` | `3` | 이터레이션당 개정 후보 수. 첫 개는 자유 개정, 나머지는 Critic 의 `proposed_rule` 을 하나씩만 반영. **train 으로 골라 쓴다** |
| `--v0-candidates` | `1` | `prompt_v0` 후보 수 |
| `--select-n` | `0` (train 전체) | 후보 선별에 쓸 **train** 문장 수. dev 는 채택 판정 전용이라 선별에 안 쓴다. 정확도는 문장 수만 따른다 (1위적중 20문장 36% / 40문장 58% / 60문장 76%) |
| `--adequacy-backend` | `cometkiwi` | 참조 없는 QE. y축 주지표 |
| `--consistency-backend` | `nli` | 보고용 가설 검증값. `nli`(양방향 entailment, 어순 무관) / `comet` / `xcomet`. **NLI 모델은 `metrics.NLI_MODEL` 로 고정**돼 있다 (`xlm-roberta-large-xnli-anli`) — 다국어라 타깃별로 바꿀 필요가 없고, 자리마다 모델을 하나로 못박은 것이 런 간 값이 섞이는 것을 막는다 |
| `--translate-backend` | 자동 | `v2`(공식 Cloud Translation Basic, API 키 필요) / `gtx`(무료 비공식). 미지정 시 `GOOGLE_TRANSLATE_API_KEY` 가 있으면 v2. **두 백엔드의 번역문은 같지 않다** — 기존 gtx 캐시 18건 재번역 대조에서 일치 0/18. `translator_id` 와 캐시 키에 백엔드가 들어가 섞이지는 않지만, **gtx 로 잰 기존 26개 런의 점수는 v2 런과 비교할 수 없다** |
| `--tgt-langs` | 기본 풀 | 목적함수를 다언어로. 분절은 타깃 무관이라 비용의 90% 가 그대로다. 소스 언어는 자동 제외 |
| `--target-aware` | — | **언어쌍 전용 프롬프트(비교군)**. 타깃 1개 필수. 네 에이전트가 타깃 문법을 근거로 쓸 수 있게 풀고 `check_target_agnostic` 게이트를 끈다. 아래 §언어쌍 전용 비교군 |
| `--seg-reasoning-effort` | `medium` | 분절 호출 사고량. **비용의 98% 가 여기다** |
| `--agent-reasoning-effort` | `medium` | Profiler/Judge/Critic/PE 사고량 |
| `--adopt-se-mult` | `0.5` | 채택 요건 `dev 쌍체 Δ > k·se`. 점 비교는 오차막대 안 잡음까지 채택했다. `0` = 이전 방식. **1.0 → 0.5 로 낮춘 이유가 `xlmr-anli` 다** — 지표 타당도가 훨씬 나은 대신 문장별 분산이 커서 dev 쌍체 se 가 0.0065 → 0.0144 로 배증한다 |
| `--max-prompt-growth` | `1.6` | 프롬프트 길이 천장 (v0 대비 배수). 품질 노브가 아니라 **비용 천장**이다 — 넘치면 압축기가 깎는다. **1.3 에서 올렸다** — 그 값은 섹션 관문 시절 실측(발동 2%)에서 나왔는데, 관문을 없애자 run08 에서 발동이 연속 100% 가 되어 이터레이션이 통째로 헛돌았다. 비용 영향은 전체 토큰의 5% 남짓 |
| `--skip-translation-below` | `0.95` | 원문 보존율이 이 값 미만이면 번역·채점을 통째로 생략. **분할 크기에 따라 유효 허용 건수가 달라진다** — 0.95 는 train 60 에서 3건, train 40 에서 2건까지 |
| `--no-coverage-rule` | — | 최소 경계 수 요건 해제. **노브가 k 를 통제 못 하게 된다** |
| `--no-contradiction` | — | NLI 해제. `effective = adequacy` 가 되어 조기 방출이 안 벌받는다 |
| `--patience` | `3` | dev 무개선 연속 횟수 상한 |
| `--budget` | `5.0` | 게이트웨이 추정 비용 상한. 초과 시 중단 |
| `--seed` | `data.DEFAULT_SEED` | 층화 분할 시드. 바꾸면 train/dev/test 가 통째로 달라져 런 간 비교가 깨진다 |
| `--fresh` | — | 런 디렉토리를 지우고 처음부터 (캐시도 삭제) |
| `--resume` | — | 같은 `--run-id` 의 `history.json` 을 이어받아 **중단된 이터레이션부터** 계속한다. 캐시는 남아 있으므로 잃는 것은 루프 상태뿐이고 그건 전부 디스크에 있다 |
| `--final-only` | — | 이터레이션을 건너뛰고 기존 `best_prompt.txt` 로 최종 test 평가만 |

크기 인자(`--iterations` 6 / `--train` 30 / `--dev` 60 / `--test` 100)와 나머지(`--workers`, `--train-pool`, `--tgt-code`, `--tgt-spaced`, `--no-google-context`,
`--comet-batch-size`)는 `--help` 로 볼 것.

`--fresh` 없이 같은 `--run-id` 로 다시 실행하면 언어 프로파일·prompt_v0·번역 캐시를
재사용해 이어서 돈다.

## 언어쌍 전용 비교군 (`--target-aware`)

기본 프롬프트는 **타깃 무관**이다 — 하나로 모든 타깃을 커버하는 것이 설계 전제고,
`check_target_agnostic` 과 다중 타깃 목적함수가 그걸 지킨다. 그런데 "가장 좋은 프롬프트"의
상한선은 **언어쌍에 최적화된 것**이다. 이 모드는 그 상한선을 실제로 만들어서, 무관 프롬프트가
거기 얼마나 근접하는지를 재기 위한 **비교군 전용**이다.

풀리는 것은 네 군데다 — A1 Profiler, prompt_v0 작성기, A8 Critic, A9 PE 가 타깃 언어명과 그
문법을 근거로 쓸 수 있게 되고, 결정론적 거부 게이트가 꺼진다. 대신 네 지시문 모두에 같은
단서가 붙는다: **분절기는 추론 시점에 소스 문장만 본다.** 타깃 지식은 "어디를 자를까"를 소스
표면형으로 판정하는 규칙의 *근거*로만 들어갈 수 있고, 실행 중에 참조할 수 있는 것이 아니다.

```bash
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --target-aware \
    --dataset fleurs-en-multi --src-lang English --tgt-lang German \
    --pair-id en-de-aware --run-id run01 \
    --iterations 8 --patience 5 --budget 25 --seg-reasoning-effort low \
    --train 40 --dev 265 --test 100 --workers 24 \
    --judge-model gpt-5-mini --translate-backend v2
```

**두 런의 `score` 를 직접 비교하면 안 된다.** 다중 타깃 런의 `score` 는 타깃별 z-정규화
평균이고 이 런은 단일 타깃 원값이라 애초에 다른 수다. 비교는 반드시 **같은 타깃으로 다시
재서** 한다 — 같은 런 디렉토리에 두 프롬프트를 넣으면 분할·캐시·백엔드가 전부 같아진다.

```bash
for L in agnostic:runs/en-multi/run09/best_prompt.txt aware:runs/en-de-aware/run01/best_prompt.txt; do
  PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
      --run-id en-multi/run09 --tgt-lang German --split test \
      --prompt "${L#*:}" --label "${L%%:*}_de"
done
```

비교 런은 기준 런과 **분할이 같아야** 한다: `--dataset`·`--seed`·`--train`/`--dev`/`--test`
가 하나라도 다르면 문장 집합이 달라져 쌍체 비교가 깨진다. `min_gap`·`t_grid` 는 같은
데이터셋에서 자동으로 같은 값이 유도된다.

**검출력 주의.** 단일 타깃 `effective` 의 test 100문장 se 는 ~0.017 인데 예상 격차는 0.005
수준이다 — test 100 만으로는 "차이 없음"과 "못 잼"이 구분되지 않는다. 주 근거는 500문장
홀드아웃(`runs/en-multi/clean500/`)에 두 프롬프트를 넣고 `bleu_eval` 의 쌍체 부트스트랩으로
잡는 쪽이다.

## 구성

| 파일 | 역할 | LLM |
|---|---|---|
| `gateway.py` | Letsur AI Gateway 클라이언트, 재시도, 비용 집계, 예산 가드, JSON 복구 | — |
| `data.py` | A0 Data Preparer — 정규화, 층화 분할, **측정 프로파일** | — |
| `pipeline.py` | A2 Segmenter / A3 Validator / **A4 Truncator** / A5 Google 번역 + 캐시 | 분절만 |
| `metrics.py` | A6 Scorer — `adequacy`(QE) / `contradiction`(NLI) / `effective` / `consistency` / `laal_words` / `score` + Critic 에게 넘기는 지표 용어집(`GLOSSARY`) | — |
| `agents.py` | A1 Profiler / **A7 Judge** / A8 Critic / A9 Prompt Engineer / A10 Compressor | ● |
| `loop.py` | A11 Loop Controller — T 격자 평가, 채택·롤백·중단, 곡선·비교군·리포트 | — |
| `tracing.py` | 호출마다 용도(`purpose`) 라벨. `Usage.by_purpose` 는 항상, LangSmith 는 키가 있을 때만. 키가 없으면 통째로 no-op | — |

루프 밖에서 도는 것들:

| 파일 | 역할 | LLM |
|---|---|---|
| `eval_prompt.py` | 임의 프롬프트 1개를 루프와 동일 지표로 평가. `bleu_eval` 이 읽는 `prompt_eval/` 산출도 여기서 나온다 | 분절만 |
| `validity_check.py` | consistency 백엔드 타당도 게이트 — 오류 주입 후 순위 확인 | — |
| `adequacy_check.py` | **adequacy 백엔드 조각 입력 게이트** — QE 가 조각에서도 오류 순위를 지키는지 | — |
| `noise_floor.py` | **contradiction 잡음 바닥 측정** — full 번역 자기-prefix 의 NLI base rate. `--recheck-t` 로 바닥 보정 순위 정렬도 재계산. 루프도 이 모듈을 쓴다 | — |
| `judge_check.py` | **판정자 + NLI 타당도 게이트** (`--skip-judge` 로 NLI 만 검사 가능) | ● |
| `bleu_eval.py` | 참조 기반 corpus BLEU / chrF2 + 쌍체 부트스트랩 + ms LAAL. 조건은 `unsegmented` / `auto @T` / `mechanical_8` / 비교군 | — |
| `comet_eval.py` | `bleu_eval` 이 남긴 번역을 재사용한 COMET (재번역·API 비용 0) | — |
| `comet_x2en.py` | {de,zh,ja}→en 런의 품질–지연 곡선. 세 트랙 타깃이 모두 영어라 비교가 깨끗하다 | — |
| `cost_report.py` | 런 하나가 실제로 쓴 LLM 비용. 크래시한 실행은 캐시 증분으로 역산 — 그 차이가 "기록되지 않은 지출" | — |
| `baselines/` | Table 1a 타 정책 구현 (`punct` / `syntax` / `causal_align` / `alignatt` / `mu_prefix`) + 강제정렬 타임스탬프 빌더. [baselines/README.md](baselines/README.md) | — |
| `validity_cases.json` / `premature_cases.json` | 고정 케이스. **사람이 작성**, LLM 생성 아님 | — |
| `adequacy_cases.json` | 조각 오류 주입 케이스 — 실제 발화 조각 기반. 문안은 사람 확정 전 (잠정) | — |
| `human_prompts/` | 사람 작성 한국어 프롬프트 3종 (비교군) | — |

LLM 판단이 들어가는 곳은 `agents.py` 네 곳뿐이다. 포맷 검증, 절단, 점수, 채택 판정, 재시도는
전부 결정론적 코드다.

### 언어 무관성

에이전트 프롬프트에는 특정 언어 지식이 없다. 언어 지식은 **데이터로만** 들어간다.

- 표기 체계와 구두점은 `measured_profile.json` — 코퍼스에서 **직접 측정**한다.
  "앞 텍스트에 붙어 나오는 비율 ≥ 90%" 라는 언어 무관 규칙으로 뽑으므로 일본어 `、` 는
  포함되고 스페인어 `¿` 는 빠진다.
- 어순·문체·함정처럼 셀 수 없는 것만 `language_profile.json` (LLM 산출) 이 채운다.
- 의존 파서 같은 **언어별 자원은 쓰지 않는다** ([AUTOSEG_DETAILS.md](AUTOSEG_DETAILS.md) '검토했으나 채택하지 않은 것').

언어 종속이 남아 있는 곳은 `data.py` 의 로더뿐이다 — 파일 포맷과 전처리가 데이터셋 고유다.

## 산출물

```
runs/{pair_id}/{run_id}/
  config.json  data/{train,dev,test}.json
  measured_profile.json    language_profile.json    prompt_v0_cand*.txt
  contra_floor.json        z_baseline.json
  iter_NN/{prompt.txt, train_rows.json, dev_rows.json,
           violations.json, dev_violations.json, metrics.json,
           judgements.json, priority_audit.json, critique.json,
           changelog.json, timing.json}
  history.json  best_prompt.txt  test_rows.json  test_judgements.json
  curve.json  final_report.md  cache/  prompt_eval/
```

`timing.json` 은 이터레이션 단계별 소요 시간이다 — 어디서 시간을 쓰는지 안 남기면
"판정이 이터당 8~13분"같은 병목을 못 찾는다. 참조 기반 평가를 돌린 런은
`bleu/` 와 `baselines/` 가 더 붙는다.

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

**런 사이의 비교 가능성 — 세 번 끊겼다.** 오래된 런의 숫자를 표에 같이 올리기 전에 확인할 것.

| 언제 | 무엇이 바뀌었나 | 결과 |
|---|---|---|
| v1 → v2 | 목적함수가 `Q`·`gain` 에서 `effective` 로 | v1 런은 아예 비교 불가. 산출물도 정리하면서 지웠다 |
| 구 집계 → 경계 평균 | `contradiction` 의 문장 값이 조각 가중 평균 → **경계 평균**, `consistency` 가 COMET → 양방향 NLI | `runs/ko-en/run01`~`run03` 이 해당. 재집계는 `*_rows.json` 의 `pieces_contra` 로 오프라인 가능(재번역 불필요) |
| gtx → v2 번역 | 번역 백엔드 자체가 다른 문장을 낸다 (재번역 대조 일치 0/18) | **gtx 로 잰 기존 26개 런의 점수는 v2 런과 비교할 수 없다.** 캐시는 `translator_id` 로 분리돼 섞이지는 않는다 |

격자를 바꾼 경우도 마찬가지다 — `score` 가 T 격자 평균이라 격자가 다르면 다른 수다.
어떤 격자를 썼는지는 `config.json` 에 남는다.

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
(원문 그대로 반환)를 정답 번역보다 높게** 주는 복사 편향. **현재 이 실패에 방어가 없다** —
번역 층의 에코 재시도는 LLM 번역기와 함께 사라졌다. 관용구 밀도가 높은 데이터에서는
adequacy 를 과신하지 말 것.

양방향 NLI 관문 실측 (`runs/validity_nli/`): **en 타깃 4케이스 위반 0** (두 모델 모두),
soft 위반(재서술 편향) comet 12건 → nli 0건. 위반은 전부 ja-ko 케이스 — mdeberta 가
고유명사 음역 교체(병십→헤이주)를 다른 개체로 읽는다. 비영어 타깃에서 nli consistency 를
쓰려면 이 맹점을 감수하든지 comet 을 유지할 것 (comet 도 같은 케이스에서 role_swap 위반).

**NLI 자리를 무엇으로 대신할 수 있는지는 전부 재 봤고, 전부 졌다.** 임베딩 코사인,
학습 헤드를 얹은 bi-encoder, 소스만 보는 future-dependency, 표현 급변점, SummaC 식 창
집계, MiniCheck, `retrans` premise, prefix 고정점 — 실측 결과와 각각의 탈락 사유는
[AUTOSEG_DETAILS.md](AUTOSEG_DETAILS.md) '검토했으나 채택하지 않은 것' 에 있다.
실험 코드(`metric_probes/`)는 정리하면서 지웠다 — 결론만 남기고 git 이력에 둔다.

그중 다시 볼 값어치가 있는 둘만 여기 옮겨 둔다.

- **`retrans` premise** (전체번역 대신 다음 조각까지의 소스 재번역) — SNR 5.88 로 현행
  5.74 보다 높고 바닥이 더 낮고 평탄하며, **오라클이 필요 없어 `reference_suspect` 오염이**
  **구조적으로 사라진다.** 비용은 경계당 번역 1회. 다음 대개편 때 1순위 후보다.
- **경계 일치로 새 지표를 검증할 때 기준선은 무작위(0.235)가 아니라 위치 사전확률(0.541)**
  이다. 문장 내용을 하나도 안 보는 위치 사전확률이 raw AUC 0.61~0.81 짜리 후보들을
  전부 이겼다. 이걸 빼먹으면 유망해 보이는 가짜 신호를 채택하게 된다.

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
  부분 순위(상위 N 개만 번호)로 순위 계산을 75% 줄여 봤으나 **사고가 오히려 17% 늘었다** —
  상위 N 개를 고르려면 전부 평가해야 하므로 정렬 생략이 평가 생략이 아니다. **배선은
  제거했다** — 되살리려면 `validate` / `evaluate` / `eval_prompt` 세 곳에 다시 넣어야 한다.
- 프롬프트 캐싱이 걸리지 않는다 (`cached_tokens: 0`). 입력 토큰 전액 과금.
