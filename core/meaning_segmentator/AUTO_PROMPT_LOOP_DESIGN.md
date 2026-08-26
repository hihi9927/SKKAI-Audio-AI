# 분절 프롬프트 자동 생성 루프 — 설계

> **상태: 구현됨.** 이 문서는 `autoseg/` 코드가 실제로 하는 일을 기술한다.
> 사용법·CLI 옵션은 [autoseg/README.md](autoseg/README.md), 진행 상황과 다음 작업은
> [autoseg/HANDOFF.md](autoseg/HANDOFF.md), 문헌 근거는
> [SEGMENTATION_CRITERIA_RELATED_WORK.md](SEGMENTATION_CRITERIA_RELATED_WORK.md).
>
> 미구현으로 남은 것은 §13 미해결 항목뿐이다.

---

## 1. 목표

**입력**: 어떤 언어든 평문 문장 데이터 (`text` 필드만 있는 JSON) + 소스/타깃 언어
**출력**: ① 그 언어쌍에 최적화된 `<SEG>` 삽입 시스템 프롬프트, ② 그 프롬프트로 분절된 데이터

프롬프트가 산출물이고 데이터는 부산물이다. 라벨(정답 분절)은 산출물이 아니다 —
정답 생성이 문장 전체 번역을 요구하므로 정의상 실시간에 못 쓴다. 선행연구도 동일하다
(Zhang 2020/2022 에서 MU 라벨은 분류기 학습용 supervision 이고 배포되는 것은 분류기다).

---

## 2. 설계의 축 — 지연은 노브가 고정한다

프롬프트가 조각 수를 정하게 두면 품질과 지연 두 축을 동시에 최적화해야 하고, 두 축을
합치는 가중치가 곧 임의 상수가 된다. 그래서 **조각 수를 프롬프트에서 떼어낸다.**

```
프롬프트는 경계를 찍고 확신 순위를 매긴다:  … 돌려보고 <SEG:1> 결과 나오면 <SEG:2> 그때 …
검증기가 최소 개수를 강제한다 (문장 길이 / 최소 T − 1 개 이상)
결정론적 절단이 상위 (k−1)개만 남긴다 (k = 문장 길이 / T)
   → 지연은 노브 T 가 정한다. 프롬프트는 지연을 건드릴 수 없다.
   → 목적함수가 단일축이 된다:  score = T 격자 평균 effective
```

결과: **프롬프트가 덜 잘라서 점수를 얻는 경로가 구조적으로 없다.** 임의 상수는 T 격자
하나뿐이고, 그것도 코퍼스의 어절 수 분포에서 유도된다 (§6.3).

---

## 3. 왜 이 지표 조합인가 — 세 제약

### 3.1 두 축을 스칼라로 합치는 절차는 문헌에 없다

SimulST/StreamST 논문 7편 중 품질과 지연을 가중합하는 것은 없다. 전부 노브 하나
(wait-k 의 `k`, MU 의 `δ`, AlignAtt 의 `f`, chunk 크기)를 스윕해 **곡선**을 그리고 **같은
지연에서 품질을 비교**한다. 품질 가중치를 정할 근거가 없는 이유가 이것이다 — 그런 절차
자체가 없다. 노브를 도입하면 지연이 외생 변수가 되고, 목적함수는 "이 지연에서 품질
최대화" 하나가 된다.

### 3.2 참조를 offline 번역으로 두면 어순 편향이 생긴다

`유사도(세그 번역 합본, full 번역)` 형태의 지표는 참조가 자기 시스템의 offline 출력이므로
**어순을 단조화한 좋은 분절이 감점된다.** ko→en 은 SOV→SVO 라 이 편향이 크게 걸린다.

자체 실측이 크기를 보여준다 (`runs/validity/validity_report.md`, comet 열):

| 변이 | 점수 |
|---|---|
| `benign_minimal` (동의어·음역 차이) | 0.9308 |
| **`benign_paraphrase`** (구조까지 바꾼 재서술, 의미 보존) | **0.8414** |
| `negation_flip` (부정 뒤집힘, 실제 오류) | 0.8843 |

의미를 보존한 재서술이 부정 뒤집힘보다 낮다. 편향이 실제 오류와 혼동될 크기다.

문헌의 해법은 단조 참조(Zhang 2020 의 prefix-attention 모델) 또는 타깃 재배열(SASST)인데,
**둘 다 우리가 "그 절차 없이 된다"고 주장하는 대상**이다. 도입하면 컨트리뷰션이 상쇄된다.

남는 길은 두 개다. **주지표에서 참조를 없애고**(`adequacy` = 참조 없는 QE), 참조가 필요한
축은 **표면형이 아니라 명제만 보는 판정기**로 재는 것(`consistency` = 양방향 NLI).

### 3.3 합본만 보면 복구가 실패를 가린다

가설을 정확히 쓰면 이렇다.

```
좋은 분절  ⟹  합본 ≈ 전체 번역          (성립. 필요조건)
좋은 분절  ⟸  합본 ≈ 전체 번역          (성립 안 함)
```

역이 안 되는 이유가 **복구**다. 나쁜 경계로 잘려도 뒤 조각이 결손을 메우면 합본은 통과한다.

```
원문       그건 문제가 안 될 것 같은데
나쁜 분절   그건 문제가 <SEG> 안 될 것 같은데

t1 사용자가 본 것:  "That's a problem"          ← 의미 정반대. 무수정 제약상 되돌릴 수 없음
t2:                "...probably won't be an issue"
합본:              통과. 합본 지표 무난
```

무수정 제약(스트리밍 번역이 앞 조각을 확정) 때문에 t1 의 오류는 그대로 남는다. 합본 지표의
정의에는 중간 시점이 없으므로 이 실패를 구조적으로 못 본다.

→ **방출 시점을 별도로 재야 한다.** 결정론적 축(`contradiction`, §5.3)이 목적함수에
들어가고, LLM 판정자(§9.3)는 위치와 이유를 붙여 프롬프트 개선을 조향한다.

---

## 4. 용어

"품질"이라는 한 단어가 서로 다른 세 가지를 가리키기 쉬우므로 분리한다.

| 개념 | 질문 | 이름 |
|---|---|---|
| 조각이 원문 대비 타당한가 (참조 없음) | "이 번역 맞나" | **adequacy** |
| 방출 시점에 미래가 반박하는가 | "너무 일찍 냈나" | **contradiction** / prematurity |
| 합본이 전체 번역과 같은 뜻인가 | "합쳐도 같은 뜻인가" | **consistency** |

지표 이름 목록:

| 이름 | 뜻 | 쓰임 |
|---|---|---|
| `format_pass_rate` | 포맷 검증 통과율 (`_no_retry` = 재시도 전) | 보고 |
| `adequacy` | QE(조각 원문, 조각 번역). 참조 없음 | `effective` 를 통해 |
| `contradiction` | NLI(full 번역, 누적 방출분) 모순 확률. **경계 (k−1)개 평균** | `effective` 를 통해 |
| **`effective`** | `adequacy × (1 − contradiction)` | **목적함수** |
| `score` | T 격자에서의 `effective` 평균 | 채택 판정 |
| `consistency` | 합본 vs full 번역의 양방향 NLI entailment 의 min | 보고 (논문 주곡선 y축) |
| `laal_words` | Length-Adaptive Average Lagging, 소스 어절 | 보고 (x축) |
| `missing_boundaries` | 예산이 요구한 경계 중 프롬프트가 못 준 개수 | 요건 (§6.4) |
| `chunks_per_sentence` / `split_ratio` | 평균 `k` / 분절된 문장 비율 | 보고 |
| `n_scored` / `n_effective` | 채점된 문장 수 / `effective` 가 정의된 문장 수 | 보고 |
| `premature_rate` / `reference_suspect_rate` / `unsafe_rate` | 판정자 산출 비율 | 부록·진단 |
| `rank_contra_spearman` | 모델 순위 vs 실측 경계 위험의 정렬도 | 진단 |
| `target_chunk_words` (T) | 목표 조각 길이 | 노브 |
| `priority` | `<SEG:n>` 의 n | 절단 기준 |
| `focus` | Critic 이 지목하는 개정 방향 | 조향 |
| `boundary_verdict` | `safe / premature / mistranslated / reference_suspect` | 조향 |

---

## 5. 지표

### 5.1 `format_pass_rate` — 보고 지표 (하드 게이트 아님)

번역 호출 **전에** 도는 순수 문자열 검사 (`pipeline.validate`). 규칙 **4종**:

| 위반 코드 | 조건 |
|---|---|
| `text_modified` | 태그를 제거한 결과가 원문과 다름 |
| `too_few_tags` | 경계 수 < `chunk_budget(문장, 커버리지 T) − 1` (§6.4) |
| `gap_too_small` | 조각 하나가 `min_gap` 미만 (§6.5) |
| `bad_priority_format` | 태그에 번호가 없음 (`<SEG>`). 전부 무번호면 정규화가 그대로 둔다 |

**표기 규칙 6종은 검증기에서 뺐다** — `leading_tag`·`trailing_tag`·`consecutive_tags`·
`missing_space`·`duplicate_priority`·`priority_gap`. `normalize_fn` 이 `validate_fn` 보다
먼저 돌아 전부 고쳐 놓으므로 구조적으로 걸릴 수 없다 (마지막 실측이 ko-en/run01 로,
정규화 도입 직전이다). 죽은 규칙을 위반 목록에 두면 산출물이 "검사하고 있다"고 거짓말한다.
남긴 4종은 정규화가 **원리적으로 못 고치는** 것뿐이다 — 없는 태그를 만들 수도, 경계를
옮길 수도, 모델이 고쳐 쓴 원문을 되살릴 수도 없다.

**같은 항목은 `_self_check` 가 지킨다 — 위반이 아니라 버그 신고로.** 정규화 직후 자기
출력을 검사해 약속(맨앞뒤 태그 없음 · 연속 없음 · 태그 좌우 공백 하나 · 태그 뒤 구두점
없음 · 번호가 1..N)을 어겼는지 본다. 여기서 걸리면 **프롬프트가 아니라 정규화가 잘못한
것**이라 대응이 완전히 다르다: 재시도로 고칠 수 없고, 코드를 고쳐야 한다. 채점에 섞지
않고 `sink` 기록 + 종류당 경고 1회로 남긴다 (예외를 던지면 100문장 런이 통째로 죽는다).

`punct_after_tag` 도 검증기에서 뺐는데, **그 근거는 나중에야 성립했다.** 정규화가 태그 뒤
구두점을 "바로 왼쪽" 조각에 얹었기 때문에, 그 자리가 연속 태그 사이의 빈 조각이면
정규화가 **스스로** `<SEG:n> ,` 를 만들어냈다. v2 런에서 0건이었던 건 그 경로가
`text_modified` 쪽으로 먼저 잡혀서다. 받는 조각을 "가장 가까운 내용 있는" 쪽으로 바꾼
뒤에야 실제로 닫혔고, 지금은 `_self_check` 가 계속 지켜본다.

**결정론적 수정은 위반이 아니라 기록이다.** `normalize_tags` 가 고치는 것 —
`punct_moved`(태그 뒤 구두점 재배치) · `tag_dropped`(맨 앞/뒤 태그) ·
`tags_merged`(연속 태그를 같은 자리로 합치고 **가장 확신한 번호**를 남김) ·
`renumbered`(번호 조밀화, 순위 관계 보존) — 은 전부 산출물을 조용히 바꾸므로, 남기지
않으면 규칙이 틀렸을 때 흔적이 없다. 중국어 여는 따옴표 `“` 가 `trailing_punct` 에
잘못 들어가 13건이 인용어 한복판에서 잘렸는데 위반 로그는 깨끗했다.
`iter_NN/normalizations.json` 에 남는다.

언어 지식(`trailing_punctuation`)은 데이터로 주입되고, 출처는 LLM 추정이 아니라 코퍼스
측정값이다 (§9.1). 프로파일에 없으면 유니코드 범주 폴백(`Po`/`Pe`/`Pf`, 문장 여는
`¿¡` 제외).

**복구는 두 단계다.**

```
1. normalize_tags()   결정론적. 번호 재부여(순서 보존), 빈 조각을 만드는 태그 삭제,
                      태그 직후 구두점 재배치, 좌우 공백 정리. LLM 없음, 경계 위치 불변
2. LLM 재시도 1회      정규화로 못 고치는 것 — 주로 text_modified, too_few_tags.
                      재시도 지시문이 위반 목록과 "부족하면 다음으로 안전한 위치를
                      찍고 순위를 뒤로 두라"를 함께 넘긴다
```

`format_pass_rate_no_retry` 는 **정규화 이후·재시도 이전**에 잰다 — 표기 흔들림은
프롬프트 품질이 아니고, 재시도가 프롬프트 품질을 가리는 것도 막아야 한다.

**하드 게이트는 없다.** `format_pass_rate < 1.0` 에 `score = −10 + rate` 를 주던 방식은
run01 에서 **30문장 중 1건**의 표기 위반이 프롬프트 전체를 `−9.03` 으로 폐기시켰다
(iter1·2 연속). 그 위반은 프롬프트의 성질이 아니라 분절 모델의 표본 사건이고, 크기가
실제 프롬프트 차이(0.003)를 3000배 압도해 **hill climbing 이 "이번에 위반이 났는가"로
결정됐다.**

채점에서 제외하는 위반은 하나로 좁혔다:

```python
SCORING_BLOCKERS = {"text_modified"}   # 원문이 훼손돼 채점이 무의미한 것만
```

특히 **`too_few_tags` 를 제외 대상으로 두면 안 된다** — 마킹이 부족한 문장(= 긴 문장)이
통째로 빠져 짧은 문장만으로 채점되는 우회로가 열린다. 덜 찍을수록 점수가 오르는 걸 막으려고
만든 규칙이 정반대로 작동하게 된다.

저비용 게이트(`skip_translation_below`, 기본 0.95)도 **원문 훼손 비율**로만 판단한다 —
`format_pass_rate` 가 아니라 `blocks_scoring` 을 통과한 비율이다. 커버리지 미달로 번역을
통째로 건너뛰면 개선 신호 자체가 사라진다.

### 5.2 `adequacy` — 조각 품질, 참조 없음

```
adequacy(조각) = QE(조각 원문, 조각 번역)          # 참조 없음
adequacy(문장) = 조각별 QE 의 소스 길이 가중 평균
adequacy(코퍼스) = 문장 평균
```

- 백엔드: `Unbabel/wmt22-cometkiwi-da` (기본) / `wmt23-cometkiwi-da-xl`. HF 게이트 모델
- 길이 가중을 하는 이유: 1어절 조각과 10어절 조각을 같은 무게로 세면 짧은 조각을 많이
  만드는 쪽이 유리해진다
- **참조가 없으므로 어순 편향이 없다** (§3.2)
- 무분절 문장도 제외하지 않는다 (조각이 하나인 것으로 계산)
- 품질 상한(앵커) 개념이 없다. 무분절은 캘리브레이션 기준이 아니라 그래프의 점 하나다

**`adequacy` 단독은 조기 방출을 보상한다.** `(조각 원문, 조각 번역)` 만의 함수는 미래의
반박을 원리적으로 못 보고, QE 는 유창하고 완결된 조각을 선호하므로 정직한 파편이 벌을
받는다 — 실측에서 `그건 문제가 → "That's a problem"`(반박당함)이 0.8653,
`"As for that, the problem"`(무해)이 0.7403 이었다. 케이스 5건 중 4건에서 순위가 뒤집혔다.
그래서 §5.3 이 필요하다.

### 5.3 `contradiction` — 조기 방출, 목적함수에 들어감

```
contradiction(경계 i) = NLI( premise = full 번역,
                             hypothesis = 조각 1..i 의 번역 누적 ) 의 contradiction 확률
contradiction(문장)   = mean( 경계 (k−1)개 )        # 무분절(k=1)이면 미정의(None)
```

마지막 조각 뒤에는 미래가 없으므로 판정 대상이 아니다 (`pieces_contra` 의 마지막 원소는
항상 0.0 — "안전"이 아니라 "대상 아님").

**왜 NLI 인가 — 미래를 끌어들이는 세 방법 중 유일하게 결정론적이다.**

| 방법 | 판정 | 이유 |
|---|---|---|
| QE 에 문장 전체를 src 로 | **기각** | 누락도 모순만큼 벌한다. 순위 위반 4/6 로 개선 없음 |
| LLM 판정자를 점수에 주입 | **기각** | 오판 1건이 평균을 0.02 움직여 검출 대상(0.003)을 압도 |
| **NLI 모순 검사** | **채택** | 결정론적. 불완전함(neutral)과 모순(contradiction)을 구별 |

**실측** (`premature_cases.json`, contradiction 확률 순위):

| 지표 | 순위 위반 |
|---|---|
| `adequacy` (QE) | **4/5** |
| NLI `microsoft/deberta-large-mnli` | **0/5** |
| NLI `mDeBERTa-v3-base-xnli` (다국어) | **0/6** |

걱정했던 실패 — 문장 조각이 MNLI 분포 밖이라 전부 `contradiction` 으로 밀리는 것 — 은
일어나지 않았다. 무해한 불완전은 `entailment`/`neutral` 로 가고 확률이 0.001~0.16 이다.
argmax 라벨이 `neutral` 로 어긋나도 **확률 순위는 유지**되므로 임계값 없이 연속 점수로 쓴다.

실전 검증 (run02 test): 기계 8자분절의 `contradiction` 이 0.1849 로 우리(0.032~0.048)의
4~6배다. `adequacy` 만으로는 격차가 0.063 인데 `effective` 로는 0.157 — 분리력이 2.5배.

백엔드는 `--contradiction-backend` (**`xlmr-anli` 기본** / `deberta-mnli` / `deberta-anli` /
`mdeberta-xnli` / `xlmr-xnli`). premise·hypothesis 가 둘 다 타깃 언어라 **소스 언어별 자원이
필요 없다.** 기본값이 다국어라 **타깃에 따라 바꿀 필요도 없다** — 예전 처방이던
`mdeberta-xnli` 는 잡음 바닥 0.102 가 실측 신호 0.075 를 넘어 무정보이고 ko·zh·ja 타깃에서
곡선이 역전했다(5개 중 2개만 정상). `xlmr-anli` 는 5/5, 관문 최소 여유 0.0994 (mdeberta 0.0027).
대가로 dev 쌍체 se 가 0.0065 → 0.0144 로 배증하므로 `--adopt-se-mult` 를 1.0 → 0.5 로 낮췄다.

**문장 집계는 경계 평균이다 — 조각 가중 평균이 아니다.** 마지막 조각(미래 없음, 구조적 0)을
평균에 넣으면 경계당 잡음 기대값이 ε 일 때 문장 값이 ≈ ε·(1 − w_last/W) 로 **k 에 단조
증가**한다 — 분절이 완벽해도 조각을 많이 낼수록 벌받고, 무분절은 노출이 없어 자동 0 점
(만점)을 받는다. 곡선 기울기의 부호가 측정이 아니라 지표 정의에서 나오는 구조다. run03
test 재집계가 크기를 보여줬다: 조각 가중 평균의 `effective` 는 T=2→6 에서 0.667→0.732 로
올랐는데, 경계 평균에서는 per-boundary contradiction 이 전 T 에서 ~0.15–0.16, `effective`
가 ~0.652 로 **평탄하다. 기울기 전체가 집계 artifact 였다.**

경계 평균은 iid 잡음 기대값이 k 무관이라 노출이 정규화되고, **무분절(k=1)은 0 이 아니라
미정의(None)** 가 된다 — 모순을 낼 기회가 없었던 것은 무죄가 아니라 판정 대상 아님이다.
None 은 집계에서 빠지고 `n_effective` 가 그 규모를 남긴다.

**잡음 바닥은 측정됐고 차감은 안 한다 (raw).** `noise_floor.py` 가 full 번역의 자기-prefix
로 hypothesis 길이별 바닥 c₀ 를 잰다 (run03 실측: 1-2어절 0.113, 3-4어절 0.041, 10어절+
0.003, 전체 mean 0.025). 경계 hypothesis 의 전형 길이에서 바닥이 0.01~0.04 라 관측된 경계
contra ~0.15 의 대부분은 실제 신호다. 목적함수에서 차감할지는 미결 (§13-6).

### 5.4 `effective` 와 `score` — 목적함수

```
effective(문장) = adequacy(문장) × (1 − contradiction(문장))     # 무분절이면 None
score           = mean over T of  effective(T)                  # None 인 T 는 제외
```

| 성질 | |
|---|---|
| 가중치 없음 | 곱셈이다. 두 축을 가중합하지 않는다 |
| 임계값 없음 | 품질 하한(`Q_floor` 류) 상수가 없다 |
| 보간 없음 | T 는 우리가 정한 이산값 |
| 노출 정규화 | 경계 평균이라 k 가 달라도 비교 가능 (§5.3) |

의미도 그대로다 — **사용자가 틀린 것을 본 조각은 지연 이득을 벌지 못한다.**
`laal_words` 는 T 마다 산출해 **보고만** 한다. 목적함수에 안 들어간다.
`effective_min`·`effective_p10` 도 함께 보고해 평균이 가리는 꼬리를 남긴다.

### 5.5 `consistency` — 논문 주곡선의 y축

```
consistency = min( ent(full 번역 ⇒ 합본),  ent(합본 ⇒ full 번역) )     # 양방향 NLI
```

목적함수에 안 들어간다. "지연을 얼마나 사면 offline 번역의 의미에서 얼마나 멀어지나"의
직접 측정값이고, 문헌의 판정 프레임(offline 상한 대비 격차, Zhang 2022)과 같은 축이다.
run03 재집계에서 **유일하게 기울기가 살아 있는 축**이다 (laal 2.0→3.3 에서 0.549→0.764).

**양방향인 이유**: 함의는 비대칭이다. `full ⇒ 합본` 실패 = 합본에 full 이 지지하지 않는
명제가 있음(환각·왜곡), `합본 ⇒ full` 실패 = 누락. min 이라 어느 쪽이든 걸리고, 두 방향을
따로 보면 실패 유형이 분리된다. 무분절은 합본 = full 이라 정의상 1.0 이다 (모든 백엔드가
같은 동일-문자열 규약을 쓴다).

**목적함수로 쓰지 않는 이유**: 합본만 보므로 복구 마스킹(§3.3)을 벌하지 못하고, NLI 확률이
포화라 0.003 규모 개선의 검출 해상도가 없다.

모델은 `--contradiction-backend` 를 따른다 — 둘 다 (합본, full) 타깃 언어 쌍이라 선택
기준이 같다. 관문 실측 (`runs/validity_nli/`): **en 타깃 4케이스 위반 0** (mdeberta·deberta
모두), COMET 의 soft 위반(재서술을 의미 오류보다 낮게 매김) 12건 → **0건**. 남은 위반은
ja-ko 케이스의 고유명사 음역 변이(병십→헤이주)를 다른 개체로 읽는 것 — 비영어 타깃에서
쓰려면 이 맹점을 확인하고 갈 것 (COMET 도 같은 케이스에서 role_swap 위반으로 탈락).
백엔드 옵션은 `nli`(기본) / `comet` / `xcomet` / `embed` / `chrf` 이고, chrF 는 표면형
일치도로 `consistency_chrf` 에 항상 병기된다 (의미 판정용은 아니다).

### 5.6 `laal_words` — x축

Length-Adaptive Average Lagging. 소스 어절 단위 (비띄어쓰기 언어는 문자).

```
laal = (1/τ) · Σ_{i=1..τ} ( d_i − d*_i )
d*_i = (i−1) · |X| / max(|Y|, |Y*|)
τ    = min{ i : d_i = |X| }
```

우리 설정에서의 계산:

```
조각 길이 c_1..c_k, 누적 C_j, 조각 j 의 번역 토큰 수 m_j
조각 j 의 모든 목표 토큰:  d_i = C_j        (동시 방출)
|X| = C_k
τ   = 마지막 조각의 첫 목표 토큰 index
Y*  = full 번역 (gold 참조가 없으므로)
```

- 목표측 토큰 단위는 타깃 언어의 표기 체계를 따른다 (`--tgt-spaced`, ja/zh/th 는 문자)
- 마지막 조각은 항목 하나만 기여하고 그 뒤는 잘린다
- 논문은 ms 로 보고한다. **어절 단위임을 표에 명시**해야 직접 비교 오류가 안 난다
- **순서에 민감하다.** 앞쪽을 빨리 내면 낮아진다 — `0.7/0.1/0.1/0.1` 과 `0.1/0.1/0.1/0.7`
  을 같게 보던 Average Proportion 계열의 사각지대가 없다

### 5.7 진단 지표

목적함수에 안 들어간다. `rank_contra_gap` 만 예외적으로 **조향에 쓰인다** (§9.4의
`focus="priority"` 판정) — 나머지는 최종 리포트에 부록 한 줄로 남는다.

| 이름 | 정의 | 읽는 법 |
|---|---|---|
| `premature_rate` | `premature` 판정 경계 / 판정 대상 경계 | **test 는 무작위 표본으로 재야 한다** — 루프 중 표본은 실패 조준이라 조건부 상향 추정치다 (run03 test 0.2727 이 그 값) |
| `reference_suspect_rate` | 판정자가 오라클(full 번역) 자체를 의심한 비율 | 높으면 지표가 아니라 번역기를 의심할 것. `contradiction`·`consistency` 가 같이 오염된다 |
| `unsafe_rate` | `premature` + `mistranslated` | 라벨이 두 값 사이에서 흔들려도 안정적이다 (관문 실측: 같은 경계가 3회 중 1/2 로 갈렸는데 `cause`·`conflict` 는 3회 동일) |
| **`rank_contra_gap`** | 순위 **하위 절반 − 상위 절반**의 경계 contra 평균 차, 문장 평균. 잡음 바닥 보정 후 | 양수 = 절단이 실제로 위험을 덜어냄. **0 이하 = 순위 무정보** → `focus="priority"`. 기준점 0 은 순위에 정보가 없을 때의 기대값이라 임의 상수가 아니다 |
| `rank_contra_spearman` | 문장 내 `<SEG:n>` 순위 vs 실측 경계 contra 의 Spearman 평균 | 같은 축의 **방향만** 보는 보조값. 순위 상관만 보므로 "정렬은 됐는데 격차가 없다"를 못 가른다. raw 라 아래 교란 포함 |

둘 다 **경계가 가장 많이 살아남는 최소 T 에서** 재고, **raw 값에는 길이 교란이 섞여
있다** — NLI 잡음 바닥은 hypothesis 가 짧을수록 크고(run03: 1-2어절 0.113, 10어절+ 0.003)
상위 순위 경계는 문장 앞쪽에 몰린다. 그래서 보정 없이는 **상위 순위가 구조적으로 불리**해
값이 음수 쪽으로 편향된다. run03 test 에서 Spearman raw −0.25 가 보정 후 **+0.14 로
뒤집혔고**, run04 dev 에서 gap raw −0.0012 가 보정 후 **+0.0249** 였다.

`rank_contra_gap` 은 조향에 쓰이므로 루프가 `loop.load_contra_floor` 로 바닥을 재서
**항상 보정한 값**을 쓴다 (런당 1회, 번역 호출 0, `contra_floor.json` 캐시).
`rank_contra_spearman` 은 raw 로 남으므로, 음수가 나오면 결론 내리기 전에
`noise_floor.py --recheck-t` 로 보정값을 확인할 것.

---

## 6. 노브

### 6.1 순위 태그 + 사후 절단

Segmenter 가 경계에 확신 순위를 달아 **한 번만** 출력한다.

```
그 다음에 이거 벤치마크 돌려보고 <SEG:1> 결과 나오면 <SEG:2> 그때 얘기하자고 했는데
```

`<SEG:1>` 이 가장 확실한 경계. 이후 조각 수 조절은 **결정론적 사후 절단**이다.

| 성질 | 이유 |
|---|---|
| 추론 1회로 곡선 전체 | 절단은 문자열 연산. 번역·채점만 노브값마다 반복 (캐시 적중 높음) |
| 단조성 구조적 보장 | 경계를 빼기만 하므로 `k` 가 반드시 줄고 `laal_words` 가 반드시 오름 |
| LLM 지시 준수에 의존 안 함 | 순위의 **전순서**만 필요. 확률 캘리브레이션 불필요 |

문헌 대응물은 Zhang 2020/2022 의 MU 분류기 임계 `δ` 다 — 하나의 모델, 추론 시 임계만 이동.

**노브는 라벨링·평가 전용이다.** 런타임 분절 정책으로 옮기는 것은 이번 범위 밖 (§13-4).

### 6.2 노브는 조각 수 `k` 가 아니라 목표 조각 길이 `T`

| | 뜻 | 성격 |
|---|---|---|
| `k` | 그 문장이 몇 조각인가 | 결과값. 문장마다 다름 |
| **`T`** (`target_chunk_words`) | 조각 하나의 목표 어절 수 | **노브** |

```
k_s = max(2, round(어절수_s / T))          # chunk_budget()
그 문장의 상위 (k_s − 1) 개 경계만 남긴다
```

`k` 를 직접 노브로 쓰면 안 되는 이유 셋:

**① 문장 길이를 무시한다.** `k=4` 를 전 문장에 적용하면 6어절 문장은 1.5어절 조각(무의미),
30어절 문장은 7.5어절 조각(여전히 느림)이 된다.

**② x축이 번진다.** `laal_words` 는 대략 조각 크기에 비례한다. `T` 고정이면 조각 크기가 문장
길이와 무관하게 일정해 코퍼스 평균이 좁게 뭉치고, `k` 고정이면 한 점 안에 서로 다른 지연이
섞인다.

**③ 도달률 문제.** `k` 고정은 짧은 문장이 그 `k` 를 못 만든다. 실측 (`runs/ko-en/run03-google-fix`
전체 190문장, 어절 수 `min 6 / p25 9 / 중앙 12 / p75 16 / max 30`, 최소 조각 3어절 가정):

| k | 도달률 |
|---|---|
| 2 | 1.00 |
| 3 | 0.82 |
| 4 | 0.52 |
| 5 | 0.33 |

`k=4` 에서 절반이 탈락한다. 쓸 수 있는 격자가 두 점뿐이라 곡선이 안 된다.
`T` 고정이면 모든 문장이 자기 길이에 맞는 `k` 를 받으므로 **도달률 100%** 다.

### 6.3 격자 — 루프 격자 / 최종 격자 / 커버리지 기준

같은 190문장 실측:

| T | 평균 k | k 분포 |
|---|---|---|
| 2 | ~6.4 | 공격적. 곡선 오른쪽 끝(붕괴 지점) 확인용 |
| 3 | 4.41 | k3:0.33 k4:0.24 k5:0.14 … |
| 4 | 3.29 | k2:0.39 k3:0.24 k4:0.18 … |
| 6 | 2.41 | k2:0.71 k3:0.18 k4:0.10 |
| 8 | 2.12 | k2:0.89 — `max(2,·)` 하한에 눌려 T=6 과 사실상 동일 |

**최종 격자 `T ∈ {2, 3, 4, 6}`** (`--final-t-grid`). T≥8 은 정보가 없다. 여기에 무분절을
기준선으로 병기한다.

**루프 격자는 `{3, 6}`** (`--t-grid`). 조각 번역 호출이 격자 크기에 비례하므로 루프 중에는
부분집합만 쓰고 전체 격자는 최종 test 에서만 돈다. `score` 는 격자 평균이라 **다른 격자로
잰 `score` 끼리는 비교할 수 없다** — 어떤 격자를 썼는지 `config.json` 에 남는다.

**커버리지 요건의 기준 T 는 최종 격자의 최소값이다** (`coverage_t = min(final_grid)`).
루프 격자가 아니다: 배포할 프롬프트는 최종 곡선의 **모든** 점을 지탱해야 하고, 루프가 그보다
느슨한 요건으로 학습하면 마지막 평가에서만 무너진다. run03 에서 실제로 T=3 요건으로
최적화된 프롬프트가 T=2 요건으로 심판받아 test 1차 통과율이 0.34 까지 떨어졌다
(train/dev 는 0.63~0.98). 검증기(`validate`)와 프롬프트 문면(`output_rules`)이 **같은 값**을
쓴다 — `config.json` 의 `min_boundaries_per`.

임의 상수는 이 격자 하나뿐이고, 그것도 어절 수 분포에서 유도된다.

### 6.4 커버리지 요건 — 노브가 `k` 를 통제하게 만드는 장치

절단은 **빼기만** 한다. 프롬프트가 안 찍은 경계는 만들어낼 수 없으므로 `T` 는 `k` 의
**상한**일 뿐이고 실제 `k` 는 프롬프트가 정한다. run02 가 그 결과를 그대로 보여줬다:

| T | `missing_boundaries` | k | laal |
|---|---|---|---|
| 2 | **3.18** | 3.54 | 3.02 |
| 3 | 1.08 | 3.42 | 3.09 |
| 4 | 0.33 | 3.03 | 3.23 |
| 6 | 0.03 | 2.44 | 3.52 |

`T=2` 와 `T=3` 은 **진짜 작동점이 아니다.** 둘 다 "프롬프트가 찍은 걸 전부 쓴다"에 걸려
`k` 가 3.54/3.42 로 붙어 있고, 곡선 왼쪽이 뭉갰다.

문장 길이별로 보면 원인이 분명하다 (run02 test 100문장):

| 어절 | 평균 경계 | 필요(T=3) | 충족률 |
|---|---|---|---|
| 0-9 | 1.82 | 1.7 | **106%** |
| 10-14 | 2.08 | 2.9 | 73% |
| 15-19 | 2.95 | 4.3 | 68% |
| 20+ | 4.24 | 6.9 | **62%** |

경계/어절 비율 0.194 (T=3 이면 0.33 이어야). **긴 문장일수록 굶는다.**

그리고 이건 단순한 부족이 아니라 **목적함수를 우회하는 탈출구**였다. 마킹을 줄이면 실효
`k` 가 내려가고 → `contradiction` 이 줄고 → `effective` 가 오른다. run02 에서 `focus=coverage`
가 4번 나왔고 PE 가 4번 따랐는데 목적함수가 4번 다 벌했다:

```
it0  k 3.13  contra 0.0158  eff 0.7728  채택
it1  k 3.30  contra 0.0284  eff 0.7655  거부
it2  k 3.57  contra 0.0381  eff 0.7485  거부   ← 커버리지·속도 최고, 점수 최저
it3  k 3.13  contra 0.0283  eff 0.7684  채택   ← it0 수준으로 되돌아옴
```

**해법: 커버리지를 지표가 아니라 입력 요건으로 올린다.**

```
검증기 규칙  too_few_tags:  경계 수 < chunk_budget(문장, 커버리지 T) − 1
[Output Rules]  "Mark AT LEAST one boundary per {min_t} words ... extra boundaries cost
                 nothing — but a boundary you never marked can never be used.
                 Output with too few boundaries is rejected."
복구          LLM 재시도가 필요 개수와 "부족하면 순위를 뒤로 두라"를 명시해 다시 시킨다
```

포맷과 같은 층에 두면 "덜 찍어서 점수 얻기"가 애초에 성립하지 않는다. 억지로 채운 경계는
순위 하위로 가서 큰 T 에서 버려지고 작은 T 에서만 쓰이며, 그 손해는 점수에 그대로 잡힌다 —
**정직한 측정**이다.

검증 (스모크, train 10 / dev 6): `missing_boundaries` 전 T 에서 **0.00**, `format_pass_rate`
1.00, 1차 통과율 0.83 → 재시도가 전부 복구. `k` 3.42 → 3.90, `laal` 3.09 → 2.50~2.99.
**모델은 개수를 명시받으면 맞춘다.** 해제 스위치는 `--no-coverage-rule` (§8.7).

대안이던 "`k` 를 직접 지시하고 위치만 찾게 한다"(wait-k 방식)는 보류했다. 확실하지만 추론이
T 격자 배로 늘고 순위 메커니즘을 잃는다. 커버리지 요건이 실패할 때의 후퇴선으로 남긴다.

---

## 7. 채택 판정

### 7.1 채택은 `score` 점 비교가 아니라 쌍체 Δ 로 한다

`score` 는 dev 에서 개선 후보를 고르는 값이고, **채택 게이트는 문장별 쌍체 차이**다
(`metrics.paired_delta`).

```
문장 s 마다:   δ_s = mean over T of ( effective_new(s,T) − effective_best(s,T) )
채택 조건:      mean(δ) > adopt_se_mult · se(δ)        # 기본 k = 1.0
```

- dev 가 고정 집합이라 점추정은 절대 평균 비교와 동일하다. 얻는 것은 **오차막대와 유효
  표본**이다: 프롬프트를 1~2 섹션만 고치므로 대부분 문장은 분절이 그대로이고 δ 가 정확히
  0 이라 분산에 기여하지 않는다. se 가 훨씬 작게 나오고, `n_changed` 가 실제로 판정에
  참여한 문장 수를 알려준다
- 절대 평균 비교는 문장 난이도 분산에 묻힌다 — run01 dev 실측에서 문장별 sd 0.0496,
  평균의 표준오차 0.0064 인데 검출하려는 차이는 0.0035 였다
- 점 비교(`>` 만)는 오차막대 안 잡음(예: −0.013±0.014)까지 채택 후보로 만들었다.
  `--adopt-se-mult 0` 으로 그 방식으로 되돌릴 수 있다
- 어느 한쪽이 무분절(`effective = None`)인 T 는 쌍체 비교에서 뺀다 — 0 으로 치면
  "분절 안 함"이 이득이 된다
- 첫 후보는 비교 대상이 없으므로 무조건 채택한다

dev 재평가는 **train `score` 가 개선됐을 때만** 돈다 (비용).

### 7.2 `effective` 의 비교 가능 범위

경계 평균 집계(§5.3)로 노출이 정규화되므로 **k≥2 인 점들 사이의 비교 — 즉 T 스윕 곡선의
y축 — 이 유효하다.** 예외는 무분절 하나다: 경계가 없어 `contradiction` 이 미정의이므로
곡선의 점이 될 수 없고, **offline 기준선(수평선)으로 병기한다.** 문헌 관행과 같다 —
Zhang 2020 은 full-sentence 성능을 점으로 따로 찍고 StreamAtt 는 SimulST 를 상한으로 둔다.

주의 둘. ① 경계당 잡음 바닥이 차감되기 전까지 `effective` 의 절대값에는 base rate 가
섞여 있다 (§13-6) — 곡선 내 상대 비교와 등지연 비교는 유효하다. ② 시스템 우열 주장의
본체는 여전히 **같은 지연에서의 비교**다 (우리 T=2 vs 기계 8자분절, k 비슷 → 노출 동일).

**T 별로 루프를 따로 돌지 않는다.** 순위 태그를 쓰는 순간 한 프롬프트가 전 구간을 커버한다.
T 별 루프는 T 개의 다른 프롬프트를 만들어 그 설계를 부정한다.

---

## 8. 전체 구조

### 8.1 구성 요소

LLM 판단은 다섯 종류(프로파일링·분절·판정·비평/개정·압축)뿐이고 나머지 여섯은 결정론이다.
**LLM 으로 할 수 있다고 LLM 으로 하지 않는다** — 비용과 분산이 같이 는다.

| | 이름 | 종류 | 역할 | 파일 |
|---|---|---|---|---|
| A0 | Data Preparer | 결정론 | 정규화, 층화 샘플링, train/dev/test 분할, 측정 프로파일 | `data.py` |
| A1 | Language Profiler | LLM 1회 | 언어 특성 구조화 → `prompt_v0` | `agents.py` |
| A2 | Segmenter | LLM | 프롬프트 주입, `<SEG:n>` 삽입, 복구 재시도 | `pipeline.py` |
| A3 | Format Validator | 결정론 | 태그 문법·원문 보존·커버리지 검사 + 정규화 | `pipeline.py` |
| A4 | Truncator | 결정론 | T 마다 상위 (k_s−1) 경계만 남김 | `pipeline.py` |
| A5 | Translation Tools | 결정론 래퍼 | Google 번역(v2/gtx) — full / 스트리밍 조각 + 캐시 | `pipeline.py` |
| A6 | Scorer | 결정론 | `adequacy`·`contradiction`·`consistency`·`laal_words` 집계 | `metrics.py` |
| A7 | Judge | LLM | 경계별 조기 방출 판정 + 이동 제안 | `agents.py` |
| A8 | Critic | LLM | 실패를 문장 단위로 언어화, 일반화 규칙 제안 | `agents.py` |
| A9 | Prompt Engineer | LLM | 프롬프트 개정 + changelog | `agents.py` |
| A10 | Compressor | LLM | 길이 예산을 넘긴 개정본 축소 (이번에 바꾼 섹션은 보호) | `agents.py` |
| A11 | Loop Controller | 결정론 | hill climbing, 채택·롤백, 재개, 예산 가드, 곡선·리포트 | `loop.py` |

번역기는 `GoogleTranslator`(기본, gtx 엔드포인트) 또는 LLM `Translator` 다. 조각 번역은
앞 조각 번역을 컨텍스트로 확정한 채 진행하고(무수정 제약), 출력이 원문 그대로면
(`looks_untranslated`) 한 번 재시도한다. gtx 컨텍스트 번역의 줄 수 불일치는 카운트해
런 끝에 경고한다.

### 8.2 데이터 흐름

```
문장 데이터 (text만)
      │
      ▼
[A0 Data Preparer] ─── train / dev / test  +  measured_profile.json
      │
      ▼
[A1 Language Profiler] ─── language_profile.json + prompt_v0.txt (골격 검사 3회 재시도)
      │
      ▼
┌───────────────────── 루프 (n = 0, 1, 2, ...) ──────────────────────┐
│                                                                    │
│  prompt_vN ──▶ [A2 Segmenter] ──▶ seg_text  (<SEG:1> <SEG:2> …)    │
│                      │                                             │
│                      ▼                                             │
│              [A3 Validator + normalize_tags]                       │
│                채점 가능 비율 < 0.95 ───────────────┐ (번역 생략)   │
│                      │ ≥ 0.95                      │               │
│                      ▼                             │               │
│              [A4 Truncator]   루프 격자 T ∈ {3,6}   │               │
│                      │  각 T 마다 분절 확정         │               │
│        ┌─────────────┴──────────────┐              │               │
│        ▼                            ▼              │               │
│  [A5 Full Translator]   [A5 Streaming Seg Translator]              │
│    full_trans (1회, 캐시)   조각별 번역 (앞 번역 확정)              │
│        └─────────────┬──────────────┘              │               │
│                      ▼                             │               │
│              [A6 Scorer]  adequacy / contradiction / effective     │
│                          / consistency / laal_words × T            │
│                      │                             │               │
│                      ├──▶ rank_contra_spearman (최소 T)            │
│                      ▼                             │               │
│              [A7 Judge]  경계별 verdict + shift    │               │
│                      │  (주 작동점 T, 실패 조준 N문장)              │
│                      ▼                             ▼               │
│              [A11]  train 개선 시 dev 재평가 → 쌍체 Δ 로 채택 판정    │
│                      │                                             │
│                      ▼                                             │
│              [A8 Critic] ◀── best_prompt 의 사례·지표·위반          │
│                      │  구조화 피드백 (규칙 + 예시쌍 + focus)        │
│                      ▼                                             │
│              [A9 Prompt Engineer] ◀── 프롬프트 버전 이력            │
│                      │  prompt_v(N+1) ─▶ [A10 Compressor] (예산 초과 시)
│                      ▼                                             │
└──────────────────────┴─────────────────────────────────────────────┘
                       │ 수렴 (patience / 최대 이터레이션 / 예산)
                       ▼
   best_prompt.txt + test 곡선(전체 격자) + 비교군 + 최종 표
```

### 8.3 이터레이션 1회의 실행 순서

**저비용 게이트를 먼저 통과시킨다.** 포맷(무료) → 절단(무료) → 번역(중간) → QE·NLI(비쌈) → 판정(LLM).

| # | 단계 | 비고 |
|---|---|---|
| 1 | Segmenter (train) | 순위 태그 포함, 문장당 1회. `--train-pool` 이 있으면 이터레이션마다 시드 고정 재표집 |
| 1b | Segmenter (train, best) | 현재 배치에 **best 프롬프트를 다시 잰다.** 재표집 때문에 배치가 이터레이션마다 절반만 겹쳐(실측 17~20/40) 쌍체 Δ 가 검출력을 절반 버렸다. 겹치는 문장은 캐시 적중이라 추가 호출은 새 문장 몫뿐 |
| 2 | Validator + 정규화 | 위반 시 1회 복구 재시도. `format_pass_rate`(+`_no_retry`) 산출 |
| 3 | **게이트** — `text_modified` 아닌 비율 < 0.95 면 4~7 생략 | 번역 비용 방어. `score = 0` 이 된다 |
| 4 | Truncator | 루프 격자마다 분절 확정. 문자열 연산 |
| 5 | Full 번역 | 원문에만 의존 → 캐시 영구. 사실상 1회차만 |
| 6 | 스트리밍 조각 번역 | 캐시 키 `hash(절단된 seg_text)`. T 간·이터레이션 간 재사용 |
| 7 | Scorer | `adequacy`/`contradiction`/`effective`/`consistency`/`laal_words` × T, `rank_contra_spearman` |
| 8 | Judge | 주 작동점 T × 실패 조준 N문장 (`--judge-rows`, 기본 8) |
| 9 | train `score` 개선 시 dev 재실행 (1~7 반복) | |
| 10 | 채택 판정 — dev 쌍체 Δ > `k·se` 면 `best_prompt` 교체 | §7.1 |
| 11 | Critic | best 의 사례 + 판정 결과 + 포맷 위반 |
| 12 | Prompt Engineer (→ 필요 시 Compressor) | 이력 강제 주입, 1~2 섹션만 수정, 골격 검사 후 반영 |

**비평 대상 = 개정 대상 = `best_prompt`.** 거부된 후보를 진단해 best 에 적용하는 실패
모드를 구조적으로 막는다. 새 best 가 채택되면 캐시된 비평을 버리고 다시 받는다.

중단 조건: dev 무개선이 `--patience`(기본 3) 연속 / 최대 이터레이션 / 예산 소진
(`BudgetExceeded`). 에이전트 호출이 실패하면 루프만 끊고 최종 평가로 넘어간다.

### 8.4 산출물 레이아웃

```
core/meaning_segmentator/runs/{src}-{tgt}/{run_id}/
  config.json                 # 전체 args + 격자·모델·판정자 프롬프트 해시·커버리지 기준
  data/{train,dev,test}.json
  measured_profile.json       # 공백비율, 문말 구두점, 부호 히스토그램
  language_profile.json       # LLM 산출. 내용 불변, 일부 필드는 소비 시 덮임 (§9.1)
  iter_00/
    prompt.txt
    train_rows.json           # 스키마 아래
    dev_rows.json             # dev 를 돌린 이터레이션에만
    violations.json  dev_violations.json
    metrics.json              # train/dev 지표 + 쌍체 Δ + 채택 여부 + 누적 usage
    judgements.json           # 경계별 verdict/cause/shift
    critique.json  changelog.json
  iter_01/ ...
  history.json
  best_prompt.txt
  test_rows.json  test_judgements.json     # test 판정은 무작위 표본
  curve.json                  # ours(T별) + baselines(무분절·기계8자) + 축 설명
  noise_floor_{split}.json    # noise_floor.py 산출 (선택)
  final_report.md
  cache/{segment,translate}.json
  prompt_eval/{label}_{split}.json         # eval_prompt.py 산출 (비교군)
```

`cache/` 와 `*_rows.json` 은 `.gitignore` 처리했다 — 오프라인 재집계(`noise_floor.py`)에
필요하므로 재검이 필요하면 런을 다시 돌려야 한다.

**`train_rows.json` 한 행**

```jsonc
{
  "id": "...", "text": "...",
  "seg_text": "… <SEG:1> … <SEG:2> …",     // 순위 태그 원본 (T 무관)
  "valid": true,
  "full_trans": "...",                      // T 무관
  "by_T": {
    "6": {"seg_text": "…", "k": 2, "missing_boundaries": 0,
          "pieces_src": ["…"], "pieces_tgt": ["…"], "seg_trans": "…",
          "pieces_contra": [0.93, 0.0],     // 경계별. 마지막은 항상 0 (미래 없음)
          "effective": 0.80, "adequacy": 0.83, "contradiction": 0.04,
          "consistency": 0.91, "chrf": 0.62, "laal_words": 4.1}
  }
}
```

**`config.json` 주요 필드**

```
t_grid / final_t_grid / main_t          루프 격자 / 최종 격자 / 판정자 작동점
min_boundaries_per / coverage_required  커버리지 기준 T (= min(final_t_grid)) / 요건 사용 여부
adequacy_model / consistency_model      실제 체크포인트 이름 (백엔드 키가 아니라 모델)
judge_model / judge_prompt_hash         판정자 모델 + JUDGE_SYSTEM 해시 (순환·재현성 추적)
translator_id / tgt_spaced              예: google:en:ctx=True / LAAL 목표측 단위
adopt_se_mult / patience / budget       채택 임계 / 중단 조건
```

### 8.5 비용

이터레이션당 LLM 호출 (train 30문장, 루프 격자 2개 기준):

| 단계 | 호출 | 비고 |
|---|---|---|
| Segmenter | 30 (+위반 재시도) | 순위 태그라도 1회 |
| Full 번역 | 30 | 캐시 영구 → 1회차만 |
| 조각 번역 | ~250 | 격자 2개 × 평균 조각 수. Google 이면 비용 ≈ 0 |
| Judge | 20~30 | 주 작동점 1개 × `--judge-rows` 문장의 전 경계 |
| Critic / PE / Compressor | 1~3 | |

조각 번역이 격자 크기에 비례한다. 완화 셋:

1. **루프 중에는 격자 부분집합만** (`--t-grid 3 6`). 전체 격자는 dev 채택 판정과 최종
   평가에서만. 어떤 격자를 썼는지 `config.json` 에 남는다
2. 캐시 키가 절단 후 분절이므로 **T 간 상위 경계가 겹치면 재사용**된다
3. Google 번역기는 비용이 사실상 0 이고 결정론적이다. LLM 번역기를 쓸 때만 문제

게이트웨이 사용량은 `Usage` 가 집계한다 — 호출 수, 토큰, 추정 비용, `truncated`
(`finish_reason == "length"` 응답 수). 판정자를 다른 모델로 쓰면 별도 Gateway 가 생기므로
사용량을 합산해 예산 가드에 넣는다.

### 8.6 설계 원칙

1. **LLM 은 다섯 곳에만** — 프로파일링, 분절, 판정, 비평·개정, 압축. 나머지는 결정론
2. **저비용 게이트 먼저** — §8.3
3. **프롬프트가 곧 산출물** — §1
4. **평가 데이터 분리** — train(루프가 봄) / dev(채택 판정) / test(최종 1회)
5. **번역기·모델은 런 전체에서 고정** — 흔들리면 점수 변화의 귀속이 불가능
6. **언어 지식은 데이터로만** — 코드에 언어 분기 없음, 형태소 분석기·의존 파서 없음 (§12.1)

### 8.7 절제 스위치

각 장치가 실제로 필요한지 확인하는 경로를 남긴다. 전부 기본 꺼짐(= 장치 켜짐)이다.

| 스위치 | 끄면 무엇이 되나 |
|---|---|
| `--no-contradiction` | `effective = adequacy`. 조기 방출이 벌받지 않는다 |
| `--no-coverage-rule` | `too_few_tags` 해제. 노브가 `k` 를 통제하지 못한다 (§6.4) |
| `--no-judge` | 판정자 없이 `adequacy`·`contradiction` 만으로 조향 |
| `--no-priority` (eval_prompt) | 순위 없는 `<SEG>` 허용 — 사람·기계 비교군을 같은 검증기로 통과시킨다 |
| `--adopt-se-mult 0` | 채택을 쌍체 점추정 비교로 되돌린다 (§7.1) |
| `--no-google-context` | 조각 번역에서 앞 문맥 제거 (운영 서버 기본 동작과 정렬, §12.5) |

---

## 9. 구성 요소별 상세

### 9.1 A0 Data Preparer — 측정 프로파일

`language_profile.json` 은 LLM 이 쓴다. 그중 **결정론적 코드를 움직이는 필드는 측정
가능한데 추측으로 채워지고 있었다.**

| 필드 | 소비처 | 측정 방법 |
|---|---|---|
| `uses_spaces_between_words` | `spaced` → validator 공백 규칙, 길이·T 단위 | 공백 문자 비율 > 0.02 |
| `trailing_punctuation` | `normalize_tags` | 괄호·따옴표(`Ps`/`Pi`/`Pf`)는 분류로 제외, 나머지는 "앞 텍스트에 붙어 나오는 비율 ≥ 0.9" |

`measure_profile()` 산출: `n`, `space_ratio`, `uses_spaces_between_words`,
`trailing_punctuation`, `punctuation_counts`, `final_punctuation_counts`,
`punctuation_final_rate`.

부착 비율 규칙은 **언어 무관**이다 — 일본어 `、` 는 포함되고 스페인어 `¿` 는 빠진다.

**실제로 발생한 실패**: `runs/ja-ko/ja-ko-test04/language_profile.json` 의
`trailing_punctuation` 이 `null` 이다 (LLM 이 필드를 통째로 빠뜨렸다). 같은 데이터의 `test06`
은 `["。","、","」","』","！","？"]` 를 냈다. ko 도 런마다 갈렸다 — 측정값은 `['.', '?']`
인데 run02·run05 는 `,` 를 넣었고, 그 코퍼스에 `,` 가 2회 존재해 **동작이 실제로 달라졌다.**

**고치는 방법 — JSON 은 건드리지 않는다.**

```
data.py    measure_profile(texts) → measured_profile.json
loop.py    reconcile_profile() 로 spaced / trailing_punct 를 측정값 우선으로 결정
agents.py  LLM 에 넘기는 JSON 은 language_profile.json 원본 그대로
```

JSON 을 고치면 `prompt_v0` 가 달라져 기존 런과 비교 불가가 된다. **소비 지점에서만
덮어쓴다.** 불일치는 경고 로그로 남긴다.

### 9.2 A4 Truncator — 결정론

```python
def truncate(seg_text, target_chunk_words, spaced):
    body = strip_tags(seg_text, spaced)
    want = max(0, chunk_budget(body, target_chunk_words, spaced) - 1)
    keep = 순위 상위 want 개 태그
    return rebuild(seg_text, keep), max(0, want - 전체_태그_수)
```

- 문자열 연산만. LLM 없음
- 가용 경계가 `want` 보다 적으면 있는 만큼만 쓴다
- 부족분이 `missing_boundaries` 로 집계된다 — 프롬프트가 예산만큼 찍지 않았다는 신호이며
  Critic 의 `focus = "coverage"` 판정에 쓰인다
- 순위가 없으면(비교군) 절단하지 않고 그대로 둔다

### 9.3 A7 Judge — LLM

full 번역을 오라클로 삼아 **경계별로** 조기 방출을 판정한다.

**입력** (경계 *i* 마다)

```
원문 전체 / full 번역(오라클) / 조각 1..i 의 (원문 → 번역) 목록
hypothesis = 조각 1..i 번역의 누적       ← 그 시점까지 사용자가 본 것 전체
방금 추가된 조각 (원문 → 번역)           ← 귀속용
아직 보이지 않은 소스                    ← 반박 여부 판정용
```

판정을 **조각 단독이 아니라 누적**으로 하는 이유: 조각 2 자체는 중립인데 조각 1+2 가 합쳐져
모순이 되는 형태가 있다. 사용자가 보는 것이 누적이므로 판정도 누적이어야 한다.

**출력** (구조화 JSON)

```jsonc
{
  "verdict": "safe | premature | mistranslated | reference_suspect",
  "conflict": "오라클과 상충하는 명제 한 절. safe 면 null",
  "cause": "polarity not yet settled | wrong participant | modifier scope |
            head not yet arrived | referent lost | other",
  "shift": {"units": 2, "to_after": "경계가 뒤따라야 할 소스 어절"},
  "generalized_rule": "이 부류의 경계를 막을 규칙 한 줄 (미지의 문장에도 적용되게)"
}
```

`shift` 는 항상 **오른쪽(늦게)** 이동이다 — 조기 방출의 교정은 늦추는 것이므로 방향 필드가
필요 없다. **진단이 곧 편집 지시가 된다.** `cause` 는 파이썬이 집계해 Critic·PE 에 올린다.

**결정적 장점 — 어순 편향을 지시로 배제할 수 있다.** NLI·QE·접두사 검사는 "무엇을
무시하라"를 지정할 방법이 없다. 프롬프트는 된다:

```
문제가 아닌 것: 오라클과 다른 어순, 다른 표현·문체, 그리고 불완전함
                ("the meeting materials for tomorrow" 처럼 그냥 끊긴 것은 정상)
문제인 것 둘:   1. 방출된 명제를 남은 문장이 반박한다 (premature)
                2. 자기 조각의 번역 자체가 틀렸다 (mistranslated)
```

§3.2 의 어순 편향이 지시 몇 줄로 빠진다. 단조 참조도 타깃 재배열도 필요 없다.

**목적함수에 안 들어간다.** Critic 의 입력을 고르고 이유를 붙이는 역할만 한다 — 오판 1건이
평균을 0.02 움직여 검출 대상(0.003)을 압도하기 때문이다.

**표본**: 루프 중에는 `sample="failure"` (실패 조준). `rank_by_failure` 가 예산을 반으로
갈라 절반은 `contradiction` 최상위, 절반은 `adequacy` 최하위에서 뽑는다 — 두 순위는 실측에서
무상관이고 최상위에서는 오히려 반대였으므로, 한쪽만 보면 실패 유형 하나가 통째로 빠진다.
**보고용 test 판정은 `sample="random"`** (시드 고정) — 실패 조준 표본으로 잰
`premature_rate` 는 조건부 상향 추정치다.

**위험과 완화**

| 위험 | 완화 |
|---|---|
| 비결정성 | `verdict` 를 4분류로 좁힌다. 루프 중에는 경계당 1회 호출이고, 반복 안정성은 관문에서 `--repeats 3` 으로 실측한다. 조향에는 세부 라벨보다 `unsafe_rate` 가 안정적이다 |
| 순환 (분절기와 같은 모델이 채점) | `--judge-model` 로 다른 모델 지정. 최소한 `config.json` 에 기록 |
| 오라클이 틀림 | `reference_suspect` 를 `verdict` 값에 포함. `reference_suspect_rate` 가 높으면 번역기 상향 신호 |
| 비용 | 주 작동점 T 하나 × `--judge-rows` 문장. 판정 실패는 `verdict="error"` 로 흘려보내고 루프를 죽이지 않는다 |

### 9.4 A8 Critic — 입력과 `focus`

**입력**: 판정자가 `premature`/`mistranslated` 로 표시한 경계 + 포맷 위반 + `rank_by_failure`
하위 문장 + `missing_boundaries` 가 있는 문장. 각 사례에 `adequacy_by_T`,
`contradiction_after_each_piece`, `selected_because` 를 붙여 넘긴다. 과소분절 쿼터는 없다 —
조각 수를 노브가 정하므로 그 실패 유형이 존재하지 않는다.

`focus` 는 **측정된 지표에서 결정론적으로** 도출한다 (`summarize_critique`). 사례 카운트로
정하면 안 된다: Critic 에게는 망가진 사례만 보내므로 특정 유형이 항상 다수가 되고 방향이
영구히 고정된다 (실측: 방향 5회 고착, 분절률 0.72 → 0.38).

```python
MISSING_BOUNDARIES_LIMIT = 0.5      # 문장당 평균 부족 경계
PREMATURE_LIMIT          = 0.15     # 판정 대상 경계 중 조기 방출 비율
RANK_GAP_MIN             = 0.0      # 순위 무정보의 기준점 (임의 상수 아님)
PRIORITY_MARGIN          = 0.03     # 폴백 전용 — 작은 T 와 큰 T 의 adequacy 격차

format_pass_rate < 1.0                                  → focus = "format"
max missing_boundaries > 0.5                            → focus = "coverage"
rank_contra_gap ≤ 0                                     → focus = "priority"
  (gap 미측정 시 폴백) adequacy(작은 T) − adequacy(큰 T) > 0.03  → focus = "priority"
max premature_rate > 0.15                               → focus = "placement"
그 외                                                    → focus = "placement"
```

세 번째가 **순위 태그로 새로 생긴 진단**이다. 경계 위치는 맞는데 **어느 게 가장 확실한지를
틀리게 매긴 것**이면 고칠 곳이 `[When to Segment]` 가 아니라 `[Priority Rules]` 다.

**이 축은 경계 단위로 잰다** (`metrics.rank_contra_gap`): 순위 하위 절반과 상위 절반의
경계 contradiction 평균 차, 문장 평균. 양수면 확신 낮다고 매긴 경계가 실제로 더 반박당한다는
뜻이고, 상위만 남기는 절단이 위험을 덜어낸다. **0 이하면 순위가 정보를 주지 않는다.**

종전에는 T 대비(`adequacy(작은 T) − adequacy(큰 T) > 0.03`)로 잤는데 두 결함이 있었다.

1. **중첩 집합 비교.** `keep(큰 T) ⊆ keep(작은 T)` 이므로 이 차이는 "하위가 상위보다
   나은가"가 아니라 "상위에 하위를 얹으면 나아지는가"다. 하위 경계의 기여가 상위와
   섞인 채로 나온다.
2. **QE 길이 편향 유입.** 조각 수가 함께 바뀐다. run04 실측에서 작은 T 의 adequacy 가
   순위 품질과 무관하게 일관되게 +0.003~0.005 높았고(T3 0.7565 vs T6 0.7532 등 4/4),
   그 부호가 진단과 같아 신호와 편향이 분리되지 않았다. `PRIORITY_MARGIN = 0.03` 은
   그 편향을 덮으려는 잠정 상수였지 측정된 문턱이 아니었다.

`rank_contra_gap` 은 상위/하위가 **배타적이고 동수**라 두 교란이 상쇄되고, 기준점이
임의 상수가 아니라 **0**(순위 무정보 시의 기대값)이 된다. 홀수 개일 때 가운데 순위는 버린다.

**잡음 바닥 보정이 필수다.** NLI 는 무해한 미완성에도 0 이 아닌 모순 확률을 주고 그 값이
hypothesis 가 짧을수록 크다(run03: 1-2어절 0.113, 10어절+ 0.003). 상위 순위 경계는 문장
앞쪽 = 짧은 hypothesis 에 몰리므로 보정 없이는 **상위가 구조적으로 불리**해 gap 이 음수로
편향된다. `loop.load_contra_floor` 가 full 번역의 어절 prefix(정의상 무해한 미완성)로
바닥을 재서 뺀다 — 런당 1회, 번역 호출 0, `contra_floor.json` 에 캐시. 바닥은 (코퍼스,
번역기, NLI 백엔드)의 성질이라 프롬프트가 바뀌어도 불변이다.

run04 dev 재계산이 그 크기를 보여준다: **raw −0.0012 → 보정 후 +0.0249** (T=3, n=140).
보정 없이 조향했으면 순위에 문제가 없는데 `focus="priority"` 로 갔을 값이다.

`rank_contra_spearman` (§5.7)은 같은 축의 방향만 보는 보조값으로 남는다 — 순위 상관만
보므로 "정렬은 됐는데 격차가 없다"를 못 가른다.

**고착 방지**: dev 무개선이 2회 이상이면 이전 `focus` 를 피하도록 반전시킨다. 캐시된 비평이
이 장치를 우회하지 않도록 집계만 다시 계산한다 (LLM 호출 없음).

### 9.5 A9 Prompt Engineer + A10 Compressor

골격은 고정이고 PE 는 섹션 **내용**만 바꾼다:

```
[Role] / [Core Principles] / [When to Segment] / [Never Segment] /
[Priority Rules] / [Decision Procedure] / [Output Rules] /
[Examples — Segment] / [Examples — Do NOT Segment]
```

제약: 이터레이션당 1~2 섹션, 예시 개수 상한, 이력 강제 참조, `focus` 준수.
개정본이 골격 검사(`check_skeleton`)를 통과하지 못하거나 500자 미만이면 **거부하고 이전
프롬프트를 유지한다.**

**길이 예산**: `prompt_v0` 길이 × `--max-prompt-growth`(기본 1.3). 넘으면 Compressor 가
줄인다 — PE 에게 "짧게 다시 써라"를 맡기지 않는 이유는 개선과 축소를 한 호출에 섞으면
**방금 추가한 규칙을 스스로 지우기** 때문이다. 압축기는 이번에 바꾼 섹션을 보호하고 누적된
옛 규칙에서만 깎으므로 과적합 기전(규칙 누적)을 정확히 겨냥한다. 압축이 실패하면 개정 자체를
거부한다.

### 9.6 A11 Loop Controller

hill climbing + dev 게이트 + 재개 + 예산 가드. 채택 판정은 §7.1.

`--fresh` 없이 같은 `--run-id` 로 재실행하면 언어 프로파일·`prompt_v0`·번역 캐시·분절
캐시를 재사용해 이어서 돈다. 재개한 `prompt_v0` 가 골격 검사에 걸리면 예산을 태우기 전에
중단한다 — 결함 `prompt_v0` 로 돌면 PE 개정본이 계속 거부되어 루프가 아무것도 학습하지 못한다.

---

## 10. 검증 관문 넷

전부 **루프 밖, 데이터 무관, 1회성**이다. 지표는 틀리면 숫자로 드러나지만 **조향은 조용히
발산한다** — `embed` 백엔드가 부정 뒤집힘에 최고점(0.9278)을 줘서 5회 런이 무효가 된 적이
있고, 알아내는 데 예산 대부분이 들었다.

| 스크립트 | 대상 | 통과 조건 | 언제 |
|---|---|---|---|
| `validity_check.py` | consistency 백엔드 | 심각한 의미 오류 < `benign_minimal` | consistency 백엔드 교체 시 |
| `adequacy_check.py` | adequacy 백엔드 (**조각 입력**) | 조각 케이스마다 심각한 오류 < `benign_minimal` | adequacy 백엔드 교체 시 |
| `judge_check.py --skip-judge` | contradiction NLI 백엔드 | 케이스마다 `min(premature) > max(safe)` | NLI 모델 교체 시 |
| `judge_check.py` | 판정자 (모델 + `JUDGE_SYSTEM`) | `safe`/`not-safe` 오분류 0건 **+ 반복 3회 동일** | 판정자 모델·프롬프트 교체 시 |

보조: `noise_floor.py` 는 관문이 아니라 **측정 도구**다 — full 번역의 자기-prefix 로 NLI
잡음 바닥을 재고 `--recheck-t` 로 순위 정렬도를 바닥 보정 후 재계산한다. 기존 런의
`*_rows.json` 을 재활용하므로 번역 호출이 0 이다.

### 10.1 케이스 파일

```jsonc
// premature_cases.json — 5케이스 × 5변이
{
  "id": "ko-en-p01",
  "src": "그건 문제가 안 될 것 같은데",
  "full_translation": "I don't think that will be a problem.",
  "boundary_after": "그건 문제가",
  "variants": {
    "premature_negation": "That's a problem",     // 부정 미확정
    "premature_role":     "...",                   // 주체 미확정
    "premature_scope":    "...",                   // 수식 범위 미확정
    "premature_benign":   "...",                   // 조기지만 반박 안 됨  ← 기준선
    "safe":               "..."                    // 정상 경계
  }
}
```

`premature_benign` 이 기준선이다. **조기 방출 자체는 죄가 아니다** — 뒤가 반박할 때만
문제다. 구별 못 하는 판정자는 "짧은 조각은 다 나쁨"으로 퇴화해 루프를 보수화한다.

**케이스는 사람이 쓴다.** LLM 이 만들면 판정자와 편향을 공유해 관문이 무력해진다. 실용적으로는
기존 런의 `train_rows.json` 에서 후보를 채굴한 뒤 손으로 확정하는 것이 싸고, 지어낸 문장보다
관문으로 강하다.

### 10.2 실측 결과 요약

| 관문 | 결과 |
|---|---|
| 양방향 NLI consistency (`runs/validity_nli/`) | **en 타깃 4케이스 위반 0** (mdeberta·deberta 모두). COMET 의 soft 위반 12건 → 0건. 위반은 전부 ja-ko — 고유명사 음역 변이를 다른 개체로 읽는다 (§13-8) |
| NLI contradiction (`judge_check --skip-judge`) | 순위 위반 **0/6** |
| adequacy 조각 (`runs/adequacy_validity/`) | 부정 뒤집힘·의미 변경·무관 문장은 전 케이스 검출. 위반 2건이 관용구 조각 1케이스에 국한 — 특히 **source_echo(원문 그대로 반환)를 정답보다 높게** 주는 복사 편향 (§13-7) |

NLI 는 판정자와 달리 **목적함수에 직접 들어가므로** 관문이 더 중요하다. 기준이 라벨이 아니라
확률 순위인 이유는, argmax 가 `neutral` 로 나와도 순위가 유지되면 임계값 없이 연속 점수로
쓸 수 있기 때문이다.

---

## 11. 평가 프로토콜

### 11.1 그래프

```
y = consistency (양방향 NLI, §5.5)           x = laal_words (어절)
```

**주 곡선의 y축은 `consistency` 다** — 판정 질문("지연을 얼마나 사면 offline 번역의
의미에서 얼마나 멀어지나")의 직접 측정값이고, 문헌의 판정 프레임(offline 상한 대비 격차,
Zhang 2022)과 같은 축이다. 무분절 = 1.0 은 artifact 가 아니라 **축의 정의**다 — 기준점
자신이므로 상한선으로 그린다.

- 우리 프롬프트: T ∈ {2,3,4,6} 4점. 무분절 = 1.0 상한선
- 비교군: **기계 8자분절 / 사람 프롬프트** — 노브가 없으므로 각 1점
  (`compare_baselines`, `eval_prompt.py --no-priority`)
- **x 좌표는 목표 T 가 아니라 실측 `laal_words`**
- 보조 그림(투명성): `adequacy`·`contradiction` vs laal 2단 패널 — "조각 품질은 평탄,
  per-boundary 위험도 평탄, 격차는 조립에서 생긴다"를 합성 점수 없이 보여준다
- `effective` 는 루프 목적함수 전용이다 (§5.5 의 두 이유)
- 주의: `consistency` 의 기준은 gold 가 아니라 자기 offline 출력이다 — §13-2 의 gold
  교차검증이 이 축을 주 그림으로 쓸수록 중요해진다

### 11.2 최종 표

`final_report.md` 가 그대로 낸다.

```
T (목표 조각 어절) | laal_words ↓ | effective ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계
2 / 3 / 4 / 6      | …            | …           | …        | …               | …           | … | …
unsegmented (노브 없음) | …       | —           | …        | —               | 1.00        | 1.00 | —
mechanical_8 (노브 없음) | …      | …           | …        | …               | …           | …  | —
```

부록 한 줄로: `format_pass_rate`(+재시도 전), `premature_rate`(주 작동점, **무작위 표본**),
`reference_suspect_rate`, `rank_contra_spearman`(최소 T).

무분절의 `effective`·`contradiction` 칸은 0 이 아니라 `—` 다 (§5.3). `laal_words` 는 소스
어절 단위이므로 논문의 ms 와 직접 비교하지 않는다.

---

## 12. 검토했으나 채택하지 않은 것

### 12.1 구문 경계 정렬률 (SASST 방식)

`<SEG>` 위치가 의존 파서 경계에서 1토큰 이내인 비율. SASST 는 syntax-aware 82% vs
fixed-length 23% 를 보였고, **번역기·참조와 무관한 유일한 축**이라는 값어치가 있다.

**제외 이유: 언어 독립성.** ko 는 kiwipiepy 같은 **언어별** 자원이 필요하다. 현행 설계는
"코드에 언어 분기 없음, 언어 지식은 데이터로만"이고 파서 레지스트리는 그 선을 넘는다.

대체: **다중 타깃 교차검증**(ko→en 과 ko→ja 를 동시에 평가). 소스 언어별 자원 0. 한 번역기가
복구해준 경계는 어순이 다른 다른 타깃에서 살아남기 어렵다. 필요해지면 도입.

### 12.2 MU / prefix-consistency 라벨

Zhang 2020/2022 의 "prefix 번역이 full 번역의 접두사인가" 이진 판정. **임계값이 없다**는 것이
큰 장점이었다.

**제외 이유: 단조성을 전제한다.** 판정 기준이 offline full 번역의 **어순**이라 두 경우를 못 가른다.

| | 상황 | 접두사 | 실제 |
|---|---|---|---|
| A | 구를 가로질러 잘라 미래를 찍었다 | 실패 | 나쁜 경계 |
| B | 경계는 멀쩡한데 offline 이 어순을 바꿨다 | 실패 | **좋은 경계** |

ko→en 은 B 가 흔하다. Zhang 의 해법(단조 번역 모델)은 우리가 "그 절차 없이 된다"고 주장하는
대상이라 도입하면 컨트리뷰션이 상쇄된다. **임계값 없음은 단조성을 판 대가였다.**

남는 용도: 접두사 검사는 **충분조건**이므로(통과 → 확실히 안전, 실패 → 판정 불가),
통과한 경계를 프롬프트 few-shot 양성 예시로 뽑는 데는 쓸 수 있다.

### 12.3 참조 없는 QE 를 조기 방출 검출에 쓰기

`adequacy` 는 `(조각 원문, 조각 번역)` 만 본다. `그건 문제가 → That's a problem` 은 그 조각의
번역으로 완벽하다. **미래에 의한 반박을 원리적으로 못 잡는다** — 못 잡는 데 그치지 않고
보상한다 (§5.2 실측). 그래서 NLI(§5.3)와 판정자(§9.3)가 따로 있다.

### 12.4 LLM 판정자를 목적함수에 넣기

판정자는 이유와 이동 방향까지 주므로 점수로도 쓰고 싶어진다. **기각.** 오판 1건이 문장 평균을
0.02 움직이는데 검출 대상은 0.003 규모다 — 포맷 하드 게이트가 신호를 파괴한 것과 같은 구조다.
목적함수에는 결정론적인 NLI 확률만 넣고, 판정자는 조향에만 쓴다.

### 12.5 `use_context` 정렬

평가는 `GoogleTranslator(use_context=True)`, 운영 서버 기본은 독립 번역이다. 복구 능력이
평가에서 더 크다는 불일치가 있다. `--no-google-context` 로 재채점하는 경로는 있으나 아직
돌리지 않았다 (§13-9).

---

## 13. 미해결

1. **긴 문장에서 분절 출력이 비는 원인 미확정.** run04(`kspon-train`, 꼬리가 길다)에서 60행 중
   6행이 빈 문자열로 돌아와 `text_modified` 로 잡혔고, 채점 가능 비율 0.90 이 게이트(0.95)에
   막혀 `score = 0` 이 됐다. 실패 행은 전부 긴 문장(87자 초과 3/3 실패)이다. thinking 토큰이
   `SEG_MAX_TOKENS`(8192)를 먹는 절단이라는 증거는 아직 없다 — 과거 실측(9~14 tok/자)을
   외삽하면 193자도 ~2700 토큰이다. `Usage.truncated` 와 `finish_reason == "length"` 경고를
   심어 뒀으니 다음 런이 답을 준다. 절단이면 예산을 올리고, 아니면 데이터 상한/게이트 임계를
   사람이 결정해야 한다.
2. **gold 참조 부재.** `adequacy`(참조 없음)와 `consistency`(자기 offline 출력 기준)로
   우회했으나, gold 참조가 있는 데이터셋으로 1회 교차 검증하면 주장이 강해진다.
   KsponSpeech 에는 번역 참조가 없다 (AIHub 한-영 신청 필요).
3. **`laal_words` 단위.** 논문은 ms. 어절→ms 환산에는 발화 속도 상수가 필요하고 텍스트
   데이터셋에는 타이밍이 없다. 어절 단위 + 각주로 갈지, 강제정렬로 속도를 재서 환산할지 미정.
4. **런타임 반영 경로.** 노브를 라벨링·평가 전용으로 쓰기로 했으므로 이 결정이 미뤄져 있다.
   산출물이 (a) 연구용 데이터, (b) 파인튜닝 SEG 라벨, (c) 런타임 LLM 분절 중 어디로 가는지에
   따라 요구 지연 예산이 달라진다.
5. **타깃 언어 처리.** 쌍별 프롬프트 유지 (`runs/{src}-{tgt}/`). 한 프롬프트가 여러 타깃을
   덮게 할지는 미정.
6. **`contradiction` 잡음 바닥 차감 여부.** 바닥은 측정됐다 (§5.3). 경계 hypothesis 의 전형
   길이에서 0.01~0.04 라 관측값 ~0.15 의 대부분은 실제 신호다. 부수 성과로 순위 정렬 Spearman
   raw −0.25 가 보정 후 **+0.14 로 뒤집혔다** (여전히 약하다). 남은 결정: 루프 목적함수에서
   c₀ 를 차감할지 — 지금은 raw.
7. **adequacy 조각 관문의 지위가 잠정.** `adequacy_cases.json` 문안이 사람 확정 전이다.
   국소 탈락 2건(관용구 조각의 복사 편향)은 번역 층의 `looks_untranslated` 재시도가 1차
   방어이나, 관용구 밀도가 높은 데이터에서 `adequacy` 를 과신하지 말 것.
8. **비영어 타깃의 `consistency` 백엔드.** 양방향 NLI 는 고유명사 음역 변이를 다른 개체로
   읽는다 (ja-ko 관문에서 mdeberta 위반 2건). COMET 도 같은 케이스에서 role_swap 위반이라
   **현재 비영어 타깃을 깨끗이 통과하는 백엔드가 없다.**
9. **`use_context` 불일치 재채점.** §12.5. 스위치는 있고 런은 없다.
10. **프롬프트 캐싱이 걸리지 않는다** (`cached_tokens: 0`). 입력 토큰 전액 과금이라 반복
    호출이 많은 Segmenter 단계의 비용이 설계 추정보다 크다.
