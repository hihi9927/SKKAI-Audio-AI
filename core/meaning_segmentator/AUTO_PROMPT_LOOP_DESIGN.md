# 분절 프롬프트 자동 생성 루프 — 설계 (v2)

> **상태: 구현됨.** 코드가 이 문서를 따른다. 사용법은 [autoseg/README.md](autoseg/README.md).
>
> **개정 (2026-08-10).** ① `contradiction` 문장 집계를 조각 가중 평균 → **경계 평균**으로
> (§5.5), ② `consistency` 를 COMET → **양방향 NLI** 로 (§5.3), ③ 그에 따라 §7.1 의
> 비교 가능성 결론이 바뀜 — 곡선 y축으로 `effective` 를 쓸 수 있게 됐고 무분절은
> 기준선이 된다. **`runs/ko-en/run01`~`run03` 의 effective·contradiction·consistency
> 수치는 구 집계라 새 수치와 비교 불가** (`pieces_contra` 로 오프라인 재집계 가능).
>
> 폐기된 v1 설계·지표 명세는 [docs_v1/](docs_v1/) — `runs/ko-en/run01`~`run12` 등 기존 런의
> 수치를 읽을 때만 필요하다. **v1 수치와 v2 수치는 축이 달라 같은 표에 넣을 수 없다.**
>
> 문헌 근거는 [SEGMENTATION_CRITERIA_RELATED_WORK.md](SEGMENTATION_CRITERIA_RELATED_WORK.md).
>
> 구현되지 않은 것은 §13 미해결 항목뿐이다.

---

## 1. 목표 (v1 과 동일)

**입력**: 어떤 언어든 평문 문장 데이터 (`text` 필드만 있는 JSON) + 소스/타깃 언어
**출력**: ① 그 언어쌍에 최적화된 `<SEG>` 삽입 시스템 프롬프트, ② 그 프롬프트로 분절된 데이터

프롬프트가 산출물이고 데이터는 부산물이다. 라벨(정답 분절)은 산출물이 아니다 —
정답 생성이 문장 전체 번역을 요구하므로 정의상 실시간에 못 쓴다. 선행연구도 동일하다
(Zhang 2020/2022 에서 MU 라벨은 분류기 학습용 supervision 이고 배포되는 것은 분류기다).

---

## 2. 무엇이 바뀌는가

| | v1 | v2 |
|---|---|---|
| 지연 | 프롬프트가 결정 → 목적함수의 한 축 | **노브가 고정.** 목적함수에서 빠짐 |
| 지연 지표 | `gain = 1 − L` (자체 프록시 = Average Proportion) | **`laal_words`** (LAAL, 문헌 표준) |
| 품질 참조 | 자기 시스템의 full 번역 | **참조 없음** (CometKiwi) |
| 목적함수 | `gain + q_weight·(LCB − Q_floor)` | **`score` = T 격자 평균 `effective`** = `adequacy × (1 − contradiction)` |
| 임의 상수 | `Q_floor`, `ratio`, `q_weight`, `z` | **없음** |
| 참조 분절 | `reference.py` 의 `gain*`, 달성률 | **폐기** |
| 개선 근거 | COMET 하위 N개 문장 | **경계별 조기 방출 판정 + 위치 이동 제안** |
| 검증 관문 | 지표 타당도 1개 | **지표 타당도 + 판정자 타당도 2개** |
| 신규 구성요소 | — | Truncator(결정론), NLI(결정론), Judge(LLM) |
| 커버리지 | — | **검증기 요건** `too_few_tags` (§6.4) |

핵심은 하나다. **지연을 노브로 고정하면 목적함수가 단일축이 되고, v1 의 임의 상수 네 개가
전부 소멸한다.**

---

## 3. 왜 바꾸나 — 근본 원인 셋

### 3.1 두 축을 스칼라로 합치는 절차는 문헌에 없다

SimulST/StreamST 논문 7편 중 품질과 지연을 가중합하는 것은 없다. 전부 노브 하나
(wait-k 의 `k`, MU 의 `δ`, AlignAtt 의 `f`, chunk 크기)를 스윕해 **곡선**을 그리고 **같은
지연에서 품질을 비교**한다. `q_weight` 를 정하는 근거가 없는 이유가 이것이다 — 그런 절차
자체가 없다.

노브를 도입하면 지연이 외생 변수가 되고, 목적함수는 "이 지연에서 품질 최대화" 하나가 된다.

부수 효과: v1 §4 가 경고한 **"무분절로 수렴" 위험이 구조적으로 사라진다.** 조각 수를 노브가
정하므로 프롬프트가 덜 잘라서 점수를 얻을 수 없다.

### 3.2 참조가 offline 번역이라 어순 편향이 있다

v1 의 `Q` 는 `유사도(세그 번역 합본, full 번역)` 다. 참조가 자기 시스템의 offline 출력이므로
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

남는 길은 **참조를 없애는 것**이다. 참조 없는 QE 는 어순 편향이 원천적으로 없다.

### 3.3 `Q` 는 최종 합본만 본다

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
합본:              통과. COMET 무난
```

무수정 제약(`streaming_segments` 가 앞 번역을 확정) 때문에 t1 의 오류는 그대로 남는다.
`Q` 의 정의에 중간 시점이 없으므로 이 실패를 구조적으로 못 본다.

→ **방출 시점 판정이 별도로 필요하다.** 단, 목적함수가 아니라 프롬프트 개선 신호로 쓴다.

---

## 4. 용어

v1 에서 "품질"이라는 한 단어가 서로 다른 세 가지를 가리키고 있었다. 분리한다.

| 개념 | 질문 | 이름 |
|---|---|---|
| 조각이 원문 대비 타당한가 (참조 없음) | "이 번역 맞나" | **adequacy** |
| 합본이 전체 번역과 같은 뜻인가 | "합쳐도 같은 뜻인가" | **consistency** |
| 방출 시점에 미래가 반박하는가 | "너무 일찍 냈나" | **prematurity** |

### 매핑

| v1 | v2 | 비고 |
|---|---|---|
| `V`, `valid_rate` | `format_pass_rate` | 포맷만 잰다는 게 이름에 보임 |
| `first_pass_valid_rate` | `format_pass_rate_no_retry` | |
| `Q`, `quality` | `consistency` | 가설 검증값. 보고 지표로 유지 |
| `Q_seg`, `quality_segmented` | `consistency_split` | |
| — | **`adequacy`** | 신규. y축 주지표 |
| — | **`premature_rate`** | 신규. 조기 방출 경계 비율 |
| — | **`laal_words`** | 신규. 단위를 이름에 박는다 |
| `L`, `latency_proxy` | 폐기 | |
| `gain`, `latency_gain` | 폐기 | |
| `k_eff`, `mean_k_eff` | 폐기 | |
| `Q_floor`, `Q_bad`, `Q_ceiling`, `ratio`, `q_weight`, `LCB` | 폐기 | 노브가 지연을 고정하므로 불필요 |
| `objective` | `score` | |
| `mean_segments` | `chunks_per_sentence` | |
| `segmented_rate` | `split_ratio` | |
| `gain*`, 달성률, `reference.py` | 폐기 | |
| Critic `direction` | `focus` | 값도 바뀜 (§8.4) |
| `error_type` | `boundary_verdict` | `safe / premature / mistranslated / reference_suspect` |
| — | `target_chunk_words` (T) | 노브 |
| — | `priority` | `<SEG:n>` 의 n |
| — | **`contradiction`** | NLI 모순 확률. 목적함수에 들어감 (§5.6) |
| — | **`effective`** | `adequacy × (1 − contradiction)`. 목적함수가 보는 값 |
| — | **`missing_boundaries`** | 예산이 요구한 경계 중 못 준 개수 (§6.4) |
| — | `n_scored` | 포맷 위반을 뺀 실제 채점 문장 수 |

---

## 5. 지표

### 5.1 `format_pass_rate` — 보고 지표 (하드 게이트 아님)

포맷 검증 통과율. 번역 호출 **전에** 돈다. 태그 문법이 `<SEG>` → `<SEG:n>` 으로 바뀌므로
규칙 네 개 추가:

| 위반 코드 | 조건 | 복구 |
|---|---|---|
| `bad_priority_format` | 태그가 `<SEG:정수>` 형태가 아님 | 정규화 |
| `duplicate_priority` | 한 문장 안에 같은 번호가 둘 이상 | 정규화 |
| `priority_gap` | 번호가 1부터 연속이 아님 | 정규화 |
| **`too_few_tags`** | 경계 수 < `chunk_budget(문장, min T) − 1` | LLM 재시도 (§6.4) |

기존 6종(`text_modified`, `leading_tag`, `trailing_tag`, `consecutive_tags`, `punct_after_tag`,
`missing_space`)은 그대로. 언어 지식(`trailing_punctuation`)은 v1 처럼 데이터로 주입하되
출처가 LLM 이 아니라 측정값이 된다 (§9.1).

**복구는 두 단계다.**

```
1. normalize_tags()   결정론적. 번호 재부여(순서 보존), 빈 조각을 만드는 태그 삭제,
                      태그 직후 구두점 재배치. LLM 없음, 경계 위치 불변
2. LLM 재시도 1회      정규화로 못 고치는 것 — text_modified, too_few_tags
```

**하드 게이트를 제거했다.** v1·초기 v2 는 `format_pass_rate < 1.0` 이면 `score = −10 + rate`
를 줬는데, run01 에서 **30문장 중 1건**의 표기 위반이 프롬프트 전체를 `−9.03` 으로 폐기시켰다
(iter1·2 연속). 그 위반은 프롬프트의 성질이 아니라 분절 모델의 표본 사건이고, 크기가
실제 프롬프트 차이(0.003)를 3000배 압도해 **hill climbing 이 "이번에 위반이 났는가"로
결정됐다.**

지금은 채점에서 제외하는 위반을 하나로 좁혔다:

```python
SCORING_BLOCKERS = {"text_modified"}   # 원문이 훼손돼 채점이 무의미한 것만
```

특히 **`too_few_tags` 를 제외 대상으로 두면 안 된다** — 마킹이 부족한 문장(= 긴 문장)이
통째로 빠져 짧은 문장만으로 채점되는 우회로가 열린다. 덜 찍을수록 점수가 오르는 걸 막으려고
만든 규칙이 정반대로 작동하게 된다.

저비용 게이트(`skip_translation_below`)도 **원문 훼손 비율**로만 판단한다. 커버리지 미달로
번역을 통째로 건너뛰면 개선 신호 자체가 사라지기 때문이다.

### 5.2 `adequacy` — y축 주지표

```
adequacy = QE(조각 원문, 조각 번역)          # 참조 없음
```

- 백엔드: `Unbabel/wmt22-cometkiwi-da` (HF 게이트 모델 — `XCOMET-XL` 과 같은 라이선스 동의 절차)
- 문장 점수 = 조각별 QE 의 길이 가중 평균. 코퍼스 점수 = 문장 평균
- **참조가 없으므로 어순 편향이 없다** (§3.2)
- 무분절 문장도 제외하지 않는다 (조각이 하나인 것으로 계산)

무분절은 캘리브레이션 앵커가 아니라 **그래프의 점 하나**다. v1 의 `Q_ceiling` 은 `Q_floor` 를
만들기 위한 것이었고 `Q_floor` 가 사라지면서 같이 소멸한다. **품질 상한 개념 자체가 없다.**

### 5.3 `consistency` — 가설 검증값

```
consistency = min( ent(full 번역 ⇒ 합본),  ent(합본 ⇒ full 번역) )     # 양방향 NLI
```

목적함수에 안 들어간다. 최종 표에 한 줄로 병기한다. 버리면 가설 (i)이 측정 없이 남는다.

**COMET(v1 의 `Q` 그대로)에서 양방향 NLI 로 교체했다.** COMET 은 참조 기반이라 §3.2 의
어순 편향을 그대로 안고 있었다 — 가설 검증값 자체가 "어순을 단조화한 좋은 분절"을
감점하면 가설 검증이 안 된다. NLI 는 명제만 본다. 함의가 비대칭이므로 양방향이 필요하다:
`full ⇒ 합본` 실패 = 합본에 full 이 지지 않는 명제가 있음(환각·왜곡), `합본 ⇒ full` 실패
= 누락. min 이라 어느 쪽이든 걸리고, 두 방향을 따로 보면 실패 유형이 분리된다.

모델은 `--contradiction-backend` 를 따른다 — 둘 다 (합본, full) 타깃 언어 쌍이라 선택
기준이 같다. 관문 실측 (`runs/validity_nli/`): **en 타깃 4케이스 위반 0** (mdeberta·deberta
모두), COMET 의 soft 위반(재서술을 의미 오류보다 낮게 매김) 12건 → **0건**. 남은 위반은
ja-ko 케이스의 고유명사 음역 변이(병십→헤이주)를 다른 개체로 읽는 것 — 비영어 타깃에서
쓰려면 이 맹점을 확인하고 갈 것 (COMET 도 같은 케이스에서 role_swap 위반으로 탈락).

### 5.4 `laal_words` — x축

Length-Adaptive Average Lagging. 소스 어절 단위.

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

- 마지막 조각은 항목 하나만 기여하고 그 뒤는 잘린다
- 논문은 ms 로 보고한다. **어절 단위임을 표에 명시**해야 직접 비교 오류가 안 난다
- `gain` 과 달리 **순서에 민감하다**. 앞쪽을 빨리 내면 낮아진다. `gain` 이 `0.7/0.1/0.1/0.1` 과
  `0.1/0.1/0.1/0.7` 을 같게 본 사각지대가 사라진다

### 5.5 `contradiction` — 조기 방출, 목적함수에 들어감

```
contradiction = NLI( premise = full 번역,  hypothesis = 그 시점까지 방출된 누적 번역 )
                의 contradiction 확률
```

경계마다 잰다. 마지막 조각 뒤에는 미래가 없으므로 항상 0.

**왜 NLI 인가 — 미래를 끌어들이는 세 방법 중 유일하게 결정론적이다.**

| 방법 | 판정 | 이유 |
|---|---|---|
| QE 에 문장 전체를 src 로 | **기각** | 누락도 모순만큼 벌한다. 순위 위반 4/6 로 개선 없음 |
| LLM 판정자를 점수에 주입 | **기각** | 오판 1건이 평균을 0.02 움직여 검출 대상(0.003)을 압도. 포맷 하드게이트와 같은 병 |
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

백엔드는 `--contradiction-backend` 로 고른다. premise·hypothesis 가 둘 다 타깃 언어라
**소스 언어별 자원이 필요 없다.** 타깃이 영어가 아니면 `mdeberta-xnli`.

**문장 집계는 경계 (k−1)개의 평균이다 — 조각 가중 평균이 아니다.** 처음 구현은 조각
길이 가중 평균이었고 마지막 조각(미래 없음, 구조적 0)이 평균에 들어갔다. 그러면 경계당
잡음 기대값이 ε 일 때 문장 값이 ≈ ε·(1 − w_last/W) 로 **k 에 단조 증가**한다 — 분절이
완벽해도 조각을 많이 낼수록 벌받고, 무분절은 노출이 없어 자동 0 점(만점)을 받는다. 곡선
기울기의 부호가 측정이 아니라 지표 정의에서 나오는 구조다.

run03 test 재집계가 크기를 보여줬다: 조각 가중 평균의 effective 는 T=2→6 에서
0.667→0.732 로 올랐는데, 경계 평균으로 바꾸면 per-boundary contradiction 이 전 T 에서
~0.15–0.16, effective 가 ~0.652 로 **평탄하다. 기울기 전체가 집계 artifact 였다.**

경계 평균은 iid 잡음 기대값이 k 무관이라 노출이 정규화되고, **무분절(k=1)은 0 이 아니라
미정의(None)** 가 된다 — 모순을 낼 기회가 없었던 것은 무죄가 아니라 판정 대상 아님이다.
None 은 집계에서 빠지고 `n_effective` 가 그 규모를 남긴다. 남는 한계: 경계당 잡음
바닥(무해한 불완전 조각의 NLI base rate, 실측 0.001~0.16)이 아직 보정되지 않았다 — §13.

### 5.6 `premature_rate` — 개선 신호 (최종 지표 아님)

```
premature_rate = premature 판정 경계 수 / 판정 대상 경계 수
판정 대상 = 문장당 (k − 1) 개          # 마지막 조각은 미래가 없어 반박 불가
```

§8.3 의 판정자가 산출한다. 목적함수에 안 들어가고, 최종 리포트에는 부록 한 줄로만 남긴다.

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

**노브는 라벨링·평가 전용이다.** 런타임 분절 정책으로 옮기는 것은 이번 범위 밖 (§12-4).

### 6.2 노브는 조각 수 `k` 가 아니라 목표 조각 길이 `T`

| | 뜻 | 성격 |
|---|---|---|
| `k` | 그 문장이 몇 조각인가 | 결과값. 문장마다 다름 |
| **`T`** (`target_chunk_words`) | 조각 하나의 목표 어절 수 | **노브** |

```
k_s = max(2, round(어절수_s / T))
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

### 6.3 격자

같은 190문장 실측:

| T | 평균 k | k 분포 |
|---|---|---|
| 2 | ~6.4 | 공격적. 곡선 오른쪽 끝(붕괴 지점) 확인용 |
| 3 | 4.41 | k3:0.33 k4:0.24 k5:0.14 … |
| 4 | 3.29 | k2:0.39 k3:0.24 k4:0.18 … |
| 6 | 2.41 | k2:0.71 k3:0.18 k4:0.10 |
| 8 | 2.12 | k2:0.89 — `max(2,·)` 하한에 눌려 T=6 과 사실상 동일 |

**격자 `T ∈ {2, 3, 4, 6}`.** T≥8 은 정보가 없다. 여기에 무분절(k=1)을 별도 점으로 찍어
x축에 5점.

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
`k` 가 3.54/3.42 로 붙어 있다. 노브가 실제로 통제한 것은 `T=4`·`T=6` 뿐이고, 곡선 왼쪽이
뭉갰다.

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
검증기 규칙  too_few_tags:  경계 수 < chunk_budget(문장, min T) − 1
[Output Rules]  "Mark AT LEAST one boundary per {min T} words ...
                 extra boundaries cost nothing — but a boundary you never
                 marked can never be used."
복구          LLM 재시도가 필요 개수를 명시해 다시 시킨다
```

포맷과 같은 층에 두면 "덜 찍어서 점수 얻기"가 애초에 성립하지 않는다. 억지로 채운 경계는
순위 하위로 가서 큰 T 에서 버려지고 작은 T 에서만 쓰이며, 그 손해는 점수에 그대로 잡힌다 —
**정직한 측정**이다.

검증 (스모크, train 10 / dev 6): `missing_boundaries` 전 T 에서 **0.00**, `format_pass_rate`
1.00, 1차 통과율 0.83 → 재시도가 전부 복구. `k` 3.42 → 3.90, `laal` 3.09 → 2.50~2.99.
**모델은 개수를 명시받으면 맞춘다.**

대안이던 "`k` 를 직접 지시하고 위치만 찾게 한다"(wait-k 방식)는 보류했다. 확실하지만 추론이
T 격자 배로 늘고 순위 메커니즘을 잃는다. 커버리지 요건이 실패할 때의 후퇴선으로 남긴다.

---

## 7. 목적함수

```
contradiction(문장) = mean( 경계 (k−1)개의 모순 확률 )      # 무분절이면 미정의
effective(문장)     = adequacy(문장) × (1 − contradiction(문장))
score               = mean over T of  effective(T)
```

| 성질 | |
|---|---|
| 가중치 없음 | 곱셈이다. 두 축을 가중합하지 않는다 |
| 임계값 없음 | `Q_floor` 불필요 |
| 보간 없음 | T 는 우리가 정한 이산값 |
| 노출 정규화 | 경계 평균이라 k 가 달라도 비교 가능 (§5.5) |

`laal_words` 는 T 마다 산출해 **보고만** 한다. 목적함수에 안 들어간다.

**`adequacy` 단독이 목적함수였을 때 조기 완성을 보상했다.** `(조각 원문, 조각 번역)` 만의
함수는 미래의 반박을 원리적으로 못 보고, QE 는 유창하고 완결된 조각을 선호하므로 정직한
파편이 벌을 받는다 — 실측에서 `그건 문제가 → "That's a problem"`(반박당함)이 0.8653,
`"As for that, the problem"`(무해)이 0.7403 이었다. 케이스 5건 중 4건에서 순위가 뒤집혔고,
run02 에서 루프가 실제로 그 방향으로 밀렸다 (iter3: `premature` 2배인데 `adequacy` −0.0035).

`contradiction`(§5.5)이 그 사각지대를 메운다. 곱셈이라 상수가 안 생기고, 의미도 그대로다 —
**사용자가 틀린 것을 본 조각은 지연 이득을 벌지 못한다.**

### 7.1 `effective` 의 비교 가능 범위 — 집계 개정 후

구 집계(조각 가중 평균)에서는 무분절이 "노출이 없어서" 항상 유리했고(run02 test 무분절
0.7719 > 우리 최고 0.7503), `effective` 를 같은 `k` 끼리만 비교할 수 있었다.

**경계 평균 집계(§5.5)가 이 제약을 대부분 푼다.** 노출이 정규화되므로 **k≥2 인 점들
사이의 비교 — 즉 T 스윕 곡선의 y축 — 이 유효하다.** 남는 예외는 무분절 하나다: 경계가
없어 `contradiction` 이 미정의이므로 곡선의 점이 될 수 없고, **offline 기준선(수평선)으로
병기한다.** 문헌 관행과 같다 — Zhang 2020 은 full-sentence 성능을 점으로 따로 찍고
StreamAtt 는 SimulST 를 상한으로 둔다.

주의 둘. ① 경계당 잡음 바닥이 보정되기 전까지 `effective` 의 절대값에는 base rate 가
섞여 있다 (§13) — 곡선 내 상대 비교와 등지연 비교는 유효하다. ② 시스템 우열 주장의
본체는 여전히 **같은 지연에서의 비교**다 (우리 T=2 vs 기계 8자분절, k 비슷 → 노출 동일).

**T 별로 루프를 따로 돌지 않는다.** 순위 태그를 쓰는 순간 한 프롬프트가 전 구간을 커버한다.
T 별 루프는 T 개의 다른 프롬프트를 만들어 그 설계를 부정한다.

채택 게이트는 `score` 다. 같은 숫자를 Prompt Engineer 에게도 보여준다 — 단일축이라 LLM 이
해석하기 쉽다.

---

## 8. 전체 구조

### 8.1 구성 요소

LLM 4곳, 결정론 6곳. **LLM 으로 할 수 있다고 LLM 으로 하지 않는다** — 비용과 분산이 같이 는다.

| | 이름 | 종류 | 역할 | v2 변경 |
|---|---|---|---|---|
| A0 | Data Preparer | 결정론 | 정규화, 층화 샘플링, train/dev/test 분할 | **측정 프로파일 추가** |
| A1 | Language Profiler | LLM 1회 | 언어 특성 구조화 → `prompt_v0` | 소비 필드 일부를 측정값이 덮음 |
| A2 | Segmenter | LLM | 프롬프트 주입, `<SEG:n>` 삽입 | **순위 태그** |
| A3 | Format Validator | 결정론 | 태그 문법·원문 보존 검사. 하드 게이트 | **규칙 3종 추가** |
| **A9** | **Truncator** | **결정론** | **T 마다 상위 (k_s−1) 경계만 남김** | **신규** |
| A4 | Translation Tools | 결정론 래퍼 | full 번역 / 스트리밍 조각 번역 | 변경 없음 |
| A5 | Scorer | 결정론 | `adequacy`·`laal_words`·`consistency` 집계 | **지표 교체** |
| **A6′** | **Judge** | **LLM** | **경계별 조기 방출 판정 + 이동 제안** | **신규** |
| A6 | Critic | LLM | 실패를 문장 단위로 언어화, 일반화 규칙 제안 | **입력이 판정 결과로** |
| A7 | Prompt Engineer | LLM | 프롬프트 개정 + changelog | **`[Priority Rules]` 섹션** |
| A8 | Loop Controller | 결정론 | hill climbing, 채택·롤백, 재개, 예산 가드 | **목적함수 교체** |

폐기: `reference.py`(참조 분절 탐색), `Q_floor` 캘리브레이션 단계.

### 8.2 데이터 흐름

```
문장 데이터 (text만)
      │
      ▼
[A0 Data Preparer] ─── train / dev / test  +  measured_profile.json
      │
      ▼
[A1 Language Profiler] ─── language_profile.json + prompt_v0.txt
      │
      ▼
┌───────────────────── 루프 (n = 0, 1, 2, ...) ──────────────────────┐
│                                                                    │
│  prompt_vN ──▶ [A2 Segmenter] ──▶ seg_text  (<SEG:1> <SEG:2> …)    │
│                      │                                             │
│                      ▼                                             │
│              [A3 Format Validator]                                 │
│                format_pass_rate < 1.0 ─────────────┐ (번역 생략)   │
│                      │ = 1.0                       │               │
│                      ▼                             │               │
│              [A9 Truncator]   T ∈ {2,3,4,6}        │               │
│                      │  각 T 마다 분절 확정         │               │
│        ┌─────────────┴──────────────┐              │               │
│        ▼                            ▼              │               │
│  [A4 Full Translator]   [A4 Streaming Seg Translator]              │
│    full_trans (1회, 캐시)   조각별 번역 (앞 번역 확정)              │
│        └─────────────┬──────────────┘              │               │
│                      ▼                             │               │
│              [A5 Scorer]  adequacy(T) / laal_words(T)              │
│                          / consistency(T) / format_pass_rate       │
│                      │                             │               │
│                      ▼                             │               │
│              [A6′ Judge]  경계별 verdict + shift    │               │
│                      │  (주 작동점 T, 하위 N문장)   │               │
│                      ▼                             ▼               │
│              [A6 Critic] ◀───────────────────────────              │
│                      │  구조화 피드백 (규칙 + 예시쌍)               │
│                      ▼                                             │
│              [A7 Prompt Engineer] ◀── 프롬프트 버전 이력            │
│                      │  prompt_v(N+1)                              │
│                      ▼                                             │
│              [A8 Loop Controller]  score → dev 검증 → 채택·롤백     │
│                      │                                             │
└──────────────────────┴─────────────────────────────────────────────┘
                       │ 수렴
                       ▼
        best_prompt.txt + test 곡선 + 최종 표
```

### 8.3 이터레이션 1회의 실행 순서

**저비용 게이트를 먼저 통과시킨다.** 포맷(무료) → 절단(무료) → 번역(중간) → QE(비쌈) → 판정(LLM).

| # | 단계 | 비고 |
|---|---|---|
| 1 | Segmenter (train) | 순위 태그 포함, 문장당 1회 |
| 2 | Format Validator | 위반 시 1회 복구 재시도. `format_pass_rate` 산출 |
| 3 | **게이트** — `format_pass_rate < 0.95` 면 4~7 생략하고 9 로 | 번역 비용 방어 |
| 4 | Truncator | T 격자마다 분절 확정. 문자열 연산 |
| 5 | Full 번역 | 원문에만 의존 → 캐시 영구. 사실상 1회차만 |
| 6 | 스트리밍 조각 번역 | 캐시 키 `hash(절단된 seg_text)`. T 간·이터레이션 간 재사용 |
| 7 | Scorer | `adequacy`/`laal_words`/`consistency` × T |
| 8 | `score` 계산 → train 개선 시 dev 재실행 (1~7 반복) | |
| 9 | Judge | 주 작동점 T × 하위 N문장 |
| 10 | Critic | 판정 결과 + 포맷 위반 전량 |
| 11 | Prompt Engineer | 이력 강제 주입, 1~2 섹션만 수정 |
| 12 | Loop Controller | dev `score` 개선 시에만 `best_prompt` 교체 |

**비평 대상 = 개정 대상 = `best_prompt`.** v1 의 실패 모드(거부된 후보를 진단해 best 에 적용)를
그대로 방지한다.

중단 조건 (v1 동일): dev `score` 가 `patience` 이터레이션 연속 미개선 / 최대 이터레이션 /
예산 소진.

### 8.4 산출물 레이아웃

```
core/meaning_segmentator/runs/{src}-{tgt}/{run_id}/
  config.json                 # 모델, 예산, T 격자, 판정자 해시, 시드
  data/{train,dev,test}.json
  measured_profile.json       # 신규 — 공백비율, 문말 구두점, 부호 비율
  language_profile.json       # LLM 산출. 내용 불변, 일부 필드는 소비 시 덮임
  iter_00/
    prompt.txt
    train_rows.json           # 스키마 변경 (아래)
    dev_rows.json
    violations.json
    dev_violations.json
    metrics.json              # 스키마 변경
    judgements.json           # 신규 — 경계별 verdict/shift
    critique.json
    changelog.json
  iter_01/ ...
  history.json
  best_prompt.txt
  test_rows.json
  curve.json                  # 신규 — T별 (laal_words, adequacy, consistency)
  final_report.md
  cache/                      # 번역 캐시
  prompt_eval/                # eval_prompt.py 산출 (비교군)

삭제:  baseline.json  (앵커 없음)
       reference/     (참조 분절 폐기)
```

**`train_rows.json` 스키마**

```jsonc
{
  "id": "...", "text": "...",
  "seg_text": "… <SEG:1> … <SEG:2> …",     // 순위 태그 원본
  "valid": true,
  "full_trans": "...",                      // T 무관
  "by_T": {
    "4": {"k": 3, "chunks": ["…"], "seg_pieces": ["…"],
          "adequacy": 0.83, "laal_words": 4.1, "consistency": 0.91}
  }
}
```

삭제 필드: `quality`(→`consistency`), `q_thresh`, `missed_gain`, `n_segments`(→`by_T.k`).

**`config.json` 변경**

```
삭제:  q_floor, q_floor_ratio, q_weight, objective,
       ref_min_units, ref_ratio, no_reference, no_llm_reference
개명:  quality_backend → adequacy_backend,  comet_model → qe_model
추가:  target_chunk_words (격자), judge_model, judge_prompt_hash
```

### 8.5 비용

이터레이션당 LLM 호출 (train 30문장 기준):

| 단계 | v1 | v2 | 비고 |
|---|---|---|---|
| Segmenter | 30 | 30 | 순위 태그라도 1회. 시스템 프롬프트 캐싱 |
| Full 번역 | 30 | 30 | 캐시 영구 → 1회차만 |
| 조각 번역 | ~90 | **~500** | T 4개 합산 평균 조각 수 ≈ 16.5/문장 |
| Judge | — | 20~30 | 주 작동점 1개 × 하위 N문장 |
| Critic | 1~2 | 1~2 | |
| Prompt Engineer | 1 | 1 | |

**조각 번역이 4배로 는다.** 완화 셋:

1. **루프 중에는 T 격자 부분집합만** 쓴다 (예: `{3, 6}`). 전체 격자는 dev 채택 판정과 최종
   평가에서만. `score` 정의에 어떤 격자를 썼는지 `config.json` 에 남긴다
2. 캐시 키가 절단 후 분절이므로 **T 간 상위 경계가 겹치면 재사용**된다. 실측 필요
3. Google 번역기는 비용이 사실상 0 이고 결정론적이다. LLM 번역기를 쓸 때만 문제

### 8.6 설계 원칙 (v1 계승)

1. **LLM 은 네 곳에만** — 프로파일링, 분절, 판정, 비평·개정. 나머지는 결정론
2. **저비용 게이트 먼저** — §8.3
3. **프롬프트가 곧 산출물** — §1
4. **평가 데이터 분리** — train(루프가 봄) / dev(채택 판정) / test(최종 1회)
5. **번역기·모델은 런 전체에서 고정** — 흔들리면 점수 변화의 귀속이 불가능

---

## 9. 구성 요소별 변경

### 9.1 A0 Data Preparer — 측정 필드 추가

v1 은 `language_profile.json` 전체를 LLM 이 쓴다. 그중 **결정론적 코드를 움직이는 두 필드가
측정 가능한데 추측으로 채워지고 있다.**

| 필드 | 소비처 | 측정 방법 |
|---|---|---|
| `uses_spaces_between_words` | `spaced` → validator 공백 규칙, 길이 계산 | 공백 문자 비율 |
| `trailing_punctuation` | validator `punct_after_tag` | 문장 끝 문자 히스토그램 + 유니코드 범주 `P*` |
| `punctuation_present` | 보고용 | 문말 부호로 끝나는 문장 비율 |

**실제로 발생한 실패**: `runs/ja-ko/ja-ko-test04/language_profile.json` 의
`trailing_punctuation` 이 `null` 이다. LLM 이 필드를 통째로 빠뜨렸다. 같은 데이터의 `test06`
은 `["。","、","」","』","！","？"]` 를 냈다. 유니코드 폴백이 받아줘서 결과적으로 무사했으나
우연이다.

ko 도 런마다 갈렸다. 측정값은 `['.', '?']` 인데 (190문장에서 `.`×124, `?`×30, `,`×2,
문말은 `.`×70 `?`×12, 공백 비율 0.272):

| 런 | LLM 값 | 동작 차이 |
|---|---|---|
| run03~run12 | `['.', '?']` | 없음 |
| run01 | `['.','?','!','…']` | 없음 (코퍼스에 없는 문자라 무해) |
| run02, run05 | `['.','?',',', …]` | **있음** — `,` 가 2회 존재 |

**고치는 방법 — JSON 은 건드리지 않는다.**

```
data.py    measure_profile(texts) → measured_profile.json
loop.py    spaced / trailing_punct 를 measured 우선으로 결정
agents.py  LLM 에 넘기는 JSON 은 language_profile.json 원본 그대로
```

JSON 을 고치면 `prompt_v0` 가 달라져 기존 런과 비교 불가가 된다. **소비 지점에서만 덮어쓴다.**

| 상황 | 결정론 경로 | LLM 경로 |
|---|---|---|
| 측정 = LLM | 바이트 단위 동일 | 동일 |
| 측정 ≠ LLM | 바뀜 (의도) | 동일 |

불일치는 경고 로그 + `config.json` 기록.

### 9.2 A9 Truncator — 신규, 결정론

```python
def truncate(seg_text, target_chunk_words):
    words = strip_tags(seg_text).split()
    k = max(2, round(len(words) / target_chunk_words))
    keep = {p for p in priorities(seg_text) if p <= k - 1}
    return drop_tags_not_in(seg_text, keep)
```

- 문자열 연산만. LLM 없음
- 가용 경계가 `k−1` 개보다 적으면 있는 만큼만 (문장이 짧거나 프롬프트가 덜 잘랐을 때)
- 이 경우를 `missing_boundaries` 로 집계해 리포트. 프롬프트가 공격적으로 자르지 않는다는 신호

### 9.3 A6′ Judge — 신규, LLM

full 번역을 오라클로 삼아 **경계별로** 조기 방출을 판정한다.

**입력** (경계 *i* 마다)

```
원문 전체 / full 번역(오라클) / 앞 조각들 / 방금 추가된 조각 원문·번역
hypothesis = m_1 + … + m_i          ← 그 시점까지 사용자가 본 것 전체
```

판정을 **조각 단독이 아니라 누적**으로 하는 이유: 조각 2 자체는 중립인데 조각 1+2 가 합쳐져
모순이 되는 형태가 있다. 사용자가 보는 것이 누적이므로 판정도 누적이어야 한다. 귀속(어느
조각이 원인인가)은 방금 추가된 조각을 함께 넘겨 판정자가 지목하게 한다.

**출력** (구조화 JSON)

```jsonc
{
  "verdict": "safe | premature | mistranslated | reference_suspect",
  "conflict": "오라클과 상충하는 명제. safe 면 null",
  "cause": "부정 미확정 | 주체 미확정 | 수식 범위 미확정 | 핵어 미도달 | 지시대상 소실",
  "shift": {"direction": "right", "units": 2, "to_after": "안 될"},
  "generalized_rule": "부정이 걸릴 수 있는 서술부 앞에서는 자르지 않는다"
}
```

**`shift` 가 핵심이다. 진단이 곧 편집 지시가 된다.** 같은 `cause` 가 반복되면 파이썬이 집계해
`generalized_rule` 을 Prompt Engineer 에 올린다.

**결정적 장점 — 어순 편향을 지시로 배제할 수 있다.** COMET·접두사 검사·NLI 는 "무엇을
무시하라"를 지정할 방법이 없다. 프롬프트는 된다:

```
어순이 오라클과 다른 것은 문제가 아니다. 다음만 본다:
  1. 방출된 번역이 전달하는 명제가 오라클과 상충하는가
  2. 원문 조각에 없는 정보를 지어냈는가
```

§3.2 의 어순 편향이 지시 한 줄로 빠진다. 단조 참조도 타깃 재배열도 필요 없다.

**설계규칙과의 관계.** v1 은 "Critic 은 점수를 새로 매기지 않는다"를 규칙으로 둔다.
판정자의 `verdict` 는 **목적함수에 들어가지 않으므로** 규칙 위반이 아니다. Critic 의 입력을
고르고 이유를 붙이는 역할만 한다.

**위험과 완화**

| 위험 | 완화 |
|---|---|
| 비결정성 | `verdict` 를 4분류로 좁히고 n=3 다수결. 안정성을 관문에서 실측 |
| 순환 (분절기와 같은 모델이 채점) | 판정자를 다른 모델로. 최소한 `config.json` 에 기록 |
| 오라클이 틀림 | `reference_suspect` 를 `verdict` 값에 포함. 비율이 높으면 번역 모델 상향 신호 |
| 비용 | 루프 중에는 주 작동점 T 하나 × 하위 N문장 |

### 9.4 A6 Critic — 입력과 `focus` 변경

**입력**: COMET 하위 N개 문장 → **판정자가 `premature` 로 표시한 경계 + 포맷 위반 전량**.
과소분절 쿼터는 불필요해진다 — 조각 수를 노브가 정하므로 과소분절이라는 실패 자체가 없다.

`focus` (v1 의 `direction`) 재정의. `gain` 이 사라지면서 `mean_k_eff < 2.0` 규칙의 근거가
없어지고, 더 중요하게는 **"더 잘라라 / 덜 잘라라" 방향 자체가 무의미해진다** — 개수는 `T` 가
정하기 때문이다. 남는 실패는 위치와 순위 둘뿐이다.

```
format_pass_rate < 1.0                              → focus = "format"
missing_boundaries 높음                            → focus = "coverage"   (경계를 충분히 안 냄)
premature_rate 높음                                  → focus = "placement"
adequacy(T=3) 은 무너지는데 adequacy(T=6) 은 괜찮음    → focus = "placement"
adequacy(T=6) 이 특히 나쁨                            → focus = "priority"
그 외                                                → focus = "placement"
```

마지막에서 두 번째가 **순위 태그로 새로 생긴 진단**이다. `T=6` 은 최상위 경계 한둘만 남긴다.
거기서만 품질이 무너지면 경계 위치는 맞는데 **어느 게 가장 확실한지를 틀리게 매긴 것**이다.
고칠 곳이 `[When to Segment]` 가 아니라 순위 규칙 섹션이라 구분이 필요하다.

임계값은 실측 후에 정한다. 지금 정하면 또 임의 상수가 된다.

### 9.5 A7 Prompt Engineer — 섹션 추가

골격에 `[Priority Rules]` 추가:

```
[Role] / [Core Principles] / [When to Segment] / [Never Segment] /
[Priority Rules] / [Decision Procedure] / [Output Rules] /
[Examples — Segment] / [Examples — Do NOT Segment]
```

`[Priority Rules]` 는 **어떤 경계가 더 확실한가**의 기준을 서술한다. 나머지 제약(이터레이션당
1~2섹션만 수정, 예시 개수 상한, 이력 강제 참조)은 v1 그대로.

### 9.6 A8 Loop Controller — 목적함수 교체

`objective()` → `score()`. 나머지(hill climbing, dev 게이트, 재개, 예산 가드) 변경 없음.
`Q_floor` 캘리브레이션 단계(`calibrate_q_floor`)가 통째로 사라지므로 런 시작이 짧아진다.

### 9.7 폐기

- **`reference.py`** 전체 — `gain*`, 달성률, 후보 생성 프롬프트, `MIN_UNITS`/`SHIFT_RANGE`/
  `MAX_VARIANTS_PER_K`, `ratio`/`ceiling`
- **`calibrate_q_floor()`** 와 `baseline.json`
- **pseudo labeling 도입 안 함** — MU(prefix-consistency) 라벨링은 검토 후 폐기 (§11.2)

---

## 10. 검증 관문 두 개

둘 다 **루프 밖, 데이터 무관, 1회성**이다.

| | 지표 타당도 | 판정자 타당도 |
|---|---|---|
| 케이스 | `validity_cases.json` (5×8) | **`premature_cases.json`** (5×5, 신규) |
| 스크립트 | `validity_check.py` | **`judge_check.py`** (신규) |
| 산출 | `runs/validity/validity_report.md` | `runs/judge_validity/report.md` |
| 대상 | 품질 백엔드 | 판정자 (모델 + 프롬프트) |
| 실행 시점 | 백엔드 교체 시 | 판정자 모델/프롬프트 교체 시 |

### 10.1 `premature_cases.json`

```jsonc
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

**통과 조건 두 개** (COMET 검사보다 하나 많다):

| | 조건 |
|---|---|
| 정확도 | `premature_*` 3종 → `premature`. `premature_benign`·`safe` → `safe`. **오분류 0건** |
| 안정성 | 같은 입력 3회 반복에 라벨 동일. LLM 이라 필요 (COMET 은 결정론이라 불필요했음) |

`premature_benign` 이 기준선인 이유는 `benign_minimal` 과 같다. **조기 방출 자체는 죄가
아니다** — 뒤가 반박할 때만 문제다. 구별 못 하면 판정자가 "짧은 조각은 다 나쁨"으로 퇴화하고
루프가 보수적으로만 흐른다.

**케이스는 사람이 쓴다.** LLM 이 만들면 판정자와 편향을 공유해 관문이 무력해진다. 실용적으로는
기존 런의 `train_rows.json` 에서 후보를 채굴한 뒤 손으로 확정하는 것이 싸고, 지어낸 문장보다
관문으로 강하다.

규모: 5케이스 × 5변이 × 3회 = 75호출.

### 10.2 왜 최종 지표가 아닌데도 관문이 필요한가

판정자는 **프롬프트 개선을 조향한다.** 틀리면 Critic 이 틀린 위치를 지목하고 Prompt Engineer 가
틀린 규칙을 넣는다. 지표는 틀리면 숫자로 드러나지만 **조향은 조용히 발산한다.**

v1 이 정확히 그 실패를 겪었다 — `embed` 백엔드가 부정 뒤집힘에 최고점(0.9278)을 줘서 5회 런이
무효가 됐고, 알아내는 데 예산 대부분이 들었다.

최종 지표가 아니므로 인간 라벨 단계는 생략하고 오류 주입 케이스 통과만 요구한다.

---

## 11. 평가 프로토콜

### 11.1 그래프

```
y = consistency (양방향 NLI, §5.3)           x = laal_words (어절)
```

**주 곡선의 y축은 `consistency` 다** — 판정 질문("지연을 얼마나 사면 offline 번역의
의미에서 얼마나 멀어지나")의 직접 측정값이고, 문헌의 판정 프레임(offline 상한 대비
격차, Zhang 2022)과 같은 축이다. 무분절 = 1.0 은 artifact 가 아니라 **축의 정의**다 —
기준점 자신이므로 상한선으로 그린다. run03 재집계 실측: laal 2.0→3.3 에서
0.549→0.764 로, 유일하게 기울기가 살아있는 축이다 (adequacy·경계평균 contradiction 은
평탄).

- 우리 프롬프트: T ∈ {2,3,4,6} 4점. 무분절 = 1.0 상한선
- 비교군: **기계 8자분절 / 사람 `current`** — 노브가 없으므로 각 1점
- **x 좌표는 목표 T 가 아니라 실측 `laal_words`**
- 보조 그림(투명성): `adequacy`·`contradiction` vs laal 2단 패널 — "조각 품질은 평탄,
  per-boundary 위험도 평탄, 격차는 조립에서 생긴다"를 합성 점수 없이 보여준다
- `effective` 는 루프 목적함수로만 쓴다. consistency 를 목적함수로 쓰면 복구
  마스킹(§3.3)이 안 벌받고, NLI 확률 포화라 0.003 규모의 개선 검출 해상도가 없다.
  주의: consistency 의 기준은 gold 가 아니라 자기 offline 출력 — §13-2 의 gold
  교차검증이 이 축을 주 그림으로 쓸수록 중요해진다

### 11.2 최종 표

```
프롬프트        T=2   T=3   T=4   T=6   | consistency | premature_rate (T=4)
ours           …                        | …           | …
current        —     —     …     —      | …           | …
기계 8자분절     …                        | …           | …
무분절          —                         | 1.00        | —
```

`premature_rate` 는 부록. 루프에서 어차피 계산되므로 최종 평가에서 주 작동점 하나만 더
돌리면 되고, "COMET 이 높은 건 번역기가 복구해서 아닌가"라는 질문에 답할 카드로 갖고 있는다.

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
대상이라 도입하면 컨트리뷰션이 상쇄된다.

**임계값 없음은 단조성을 판 대가였다.** 단조 가정 없이 임계값도 없는 판정은 문헌에도 없다.
우리는 단조 가정을 거부하므로 임계값 쪽을 낸다.

남는 용도: 접두사 검사는 **충분조건**이므로(통과 → 확실히 안전, 실패 → 판정 불가),
통과한 경계를 프롬프트 few-shot 양성 예시로 뽑는 데는 쓸 수 있다.

### 12.3 참조 없는 QE 를 조기 방출 검출에 쓰기

`adequacy` 는 `(조각 원문, 조각 번역)` 만 본다. `그건 문제가 → That's a problem` 은 그 조각의
번역으로 완벽하다. **미래에 의한 반박을 원리적으로 못 잡는다.** 그래서 판정자가 따로 필요하다.

### 12.4 NLI 모순 검사

`premise = full 번역`, `hypothesis = 누적 방출분` 으로 3분류. 이진이고 어순 무관이라 매력적이나,
**이유와 이동 방향이 안 나온다.** 프롬프트 개선이 목적이므로 판정자가 우월하다. 판정자가
관문에서 떨어지면 대안으로 검토.

### 12.5 `use_context` 정렬

평가는 `GoogleTranslator(use_context=True)`, 운영 서버 기본은 독립 번역이다. 복구 능력이
평가에서 더 크다는 불일치가 있으나 이번 범위에서 다루지 않는다.

---

## 13. 미해결

1. **조기 방출 압력이 목적함수에 없다.** `adequacy` 는 조기 방출을 벌하지 않으므로 루프는
   복구된 경계를 벌하지 않는다. 판정자는 Critic 에게 말해줄 뿐이다. 목적함수에 넣으면 설계규칙
   ("점수 두 개") 위반이라, 현재 결론은 **부록 지표로 노출**하는 것.
2. **gold 참조 부재.** `adequacy`(참조 없음)로 우회했으나, gold 참조가 있는 데이터셋으로
   1회 교차 검증하면 주장이 강해진다. KsponSpeech 에는 번역 참조가 없다.
3. **`laal_words` 단위.** 논문은 ms. 어절→ms 환산에는 발화 속도 상수가 필요하고 텍스트
   데이터셋에는 타이밍이 없다. 어절 단위 + 각주로 갈지, 코퍼스에서 속도를 재서 환산할지 미정.
4. **런타임 반영 경로.** v1 §10-3 그대로. 노브를 라벨링 전용으로 쓰기로 했으므로 이 결정이
   더 미뤄진다. 산출물이 (a) 연구용 데이터, (b) 파인튜닝 SEG 라벨, (c) 런타임 LLM 분절 중
   어디로 가는지에 따라 요구 지연 예산이 달라진다.
5. **타깃 언어 처리.** v1 §10-1 그대로. 쌍별 프롬프트 유지 (`runs/{src}-{tgt}/`).
6. **`contradiction` 잡음 바닥 — 측정됨, 목적함수 차감은 미결.** `noise_floor.py` 가
   full 번역의 자기-prefix 로 길이별 바닥 c₀ 를 잰다 (run03 실측: 1-2어절 0.113,
   3-4어절 0.041, 10어절+ 0.003 — 전체 mean 0.025). 경계 hypothesis 의 전형 길이에서
   바닥은 0.01~0.04 라 관측된 경계 contra ~0.15 의 대부분은 실제 신호다. **부수 성과:
   순위 정렬 Spearman raw −0.25 가 바닥 보정 후 +0.14 로 뒤집힘** — 역전은 대부분 위치
   (짧은 hypothesis) 교란이었고, 순위는 약하게나마 정방향이다 (+0.14 는 여전히 약함 —
   개선 여지). 남은 결정: 루프 목적함수에서 c₀ 를 차감할지 (지금은 raw).
7. **adequacy 조각 관문 — 만들어 실측함. 국소 탈락 2건.** `adequacy_check.py` +
   `adequacy_cases.json` (실제 KsponSpeech 발화의 조각, 6케이스 12검사). 부정 뒤집힘·
   의미 변경·무관 문장은 전 케이스에서 정상 검출. 위반은 관용구 조각("밀려 썼던") 1케이스
   — 의미 변경(+0.03)과 **source_echo(원문 그대로 반환)를 정답보다 높게(+0.09)** 줌.
   복사 편향은 알려진 QE 결함이고 번역 층의 `looks_untranslated` 재시도가 1차 방어.
   케이스 문안은 사람 확정 전이라 관문 지위는 잠정.
8. **비영어 타깃의 `consistency` 백엔드.** 양방향 NLI 는 고유명사 음역 변이를 다른
   개체로 읽는다 (ja-ko 관문 케이스에서 mdeberta 위반 2건). COMET 도 같은 케이스에서
   role_swap 위반이라 현재 비영어 타깃을 깨끗이 통과하는 백엔드가 없다.

---

## 14. 실행 순서

앞 단계를 건너뛰면 뒤 단계가 무효가 되는 순서다.

| # | 항목 | 선행 | 비고 |
|---|---|---|---|
| 0 | `premature_cases.json` 작성 + 판정자 프롬프트 + `judge_check.py` 통과 | — | **여기서 걸리면 §9.3 전체 폐기.** 반나절 |
| 1 | A0 측정 필드(`measure_profile`) + 소비 지점 병합 | — | 0번과 독립. 현행 코드의 결함 |
| 2 | `adequacy` 백엔드(CometKiwi) + `laal_words` 구현 | — | 기존 런에 소급 적용해 먼저 확인 |
| 3 | 순위 태그 — 프롬프트 규칙 + validator 규칙 3종 | 2 | 단조성 실측으로 검증 |
| 4 | Truncator + T 격자 + `score` | 3 | `objective()` 교체 |
| 5 | 판정자를 Critic 입력에 연결, `focus` 재정의 | 0, 4 | |
| 6 | 곡선 + 비교군 3종 + 최종 표 | 4, 5 | |
| 7 | `reference.py`·폐기 지표 제거, v1 문서 대체 | 6 | |

0·1·2 는 서로 독립이라 병렬 가능하다.
