# autoseg 지표 명세

`metrics.py` 가 계산하는 세 축(`V`, `Q`, `L`)과 목적함수의 정확한 정의.
사용법은 [README.md](README.md), 설계 배경은 [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md).

이 문서의 모든 수식은 구현과 대조해 확인했고, 표의 수치는 저장소에 있는 산출물
(`runs/**/baseline.json`, `runs/validity/validity_report.md`)에서 그대로 옮긴 것이다.

---

## 1. 세 축

| 축 | 이름 | 범위 | 방향 | 역할 |
|---|---|---|---|---|
| `V` | 포맷 유효율 | 0 ~ 1 | 높을수록 좋음 | 하드 게이트. 1.0 미만이면 탈락 |
| `Q` | 분절 품질 | 백엔드 의존 | 높을수록 좋음 | 하한 제약 (`LCB ≥ Q_floor`) + floor 초과분은 가중 가산 |
| `L` | 지연 프록시 | 0.5 ~ 1.0 | 낮을수록 좋음 | 최대화 대상 (`gain = 1 − L`) |

품질 단일 축으로 최적화하면 최적해가 "`<SEG>` 를 하나도 넣지 않는 프롬프트"로 수렴한다.
분절이 없으면 세그 번역 = 전체 번역이라 `Q` 가 자동으로 1.0 이 되기 때문이다. 지연축이
반드시 필요한 이유가 이것이다.

---

## 2. V — 포맷 유효율

### 정의

```
V = 포맷 검증을 통과한 문장 수 / 전체 문장 수
```

`Metrics.valid_rate` ([metrics.py:250](metrics.py#L250)). 품질과 무관하다 —
**"자를 자리를 잘 골랐나"가 아니라 "시킨 대로 태그만 넣었나"** 를 잰다.

### 검증 규칙

[pipeline.py:168](pipeline.py#L168) `validate()`. LLM 없이 순수 문자열 검사이며
번역 호출 **전에** 돈다.

| 위반 코드 | 조건 |
|---|---|
| `text_modified` | `<SEG>` 를 제거한 결과가 원문과 다름 (모델이 텍스트를 고쳐 씀) |
| `leading_tag` | 맨 앞에 태그 |
| `trailing_tag` | 맨 뒤에 태그 |
| `consecutive_tags` | 태그가 연속 |
| `punct_after_tag` | 태그 직후에 문말 구두점 |
| `missing_space` | 태그 앞 또는 뒤에 공백 없음 |

`punct_after_tag` 가 쓰는 구두점 목록은 코드에 하드코딩돼 있지 않고
`language_profile.json` 의 `trailing_punctuation` 에서 온다. 프로파일에 없으면
유니코드 범주로 추정한다. 검증 규칙 자체는 언어 무관이고 언어 지식은 데이터로만 들어간다.

### 두 종류

| 필드 | 뜻 |
|---|---|
| `valid_rate` (`V`) | 복구 재시도 1회 **이후**. 목적함수가 보는 값 |
| `first_pass_valid_rate` | 재시도 없이 한 번에 성공한 비율. 프롬프트 품질 진단용 |

### 비용 게이트

`V < 0.95` 면 번역·점수 계산을 아예 건너뛴다 ([loop.py:60](loop.py#L60)).
포맷이 무너진 프롬프트에 번역 비용을 쓰지 않기 위한 것이며, 이 경우 `Q` 관련 필드는 0 이다.

### 인프라 의존성

`SEG_MAX_TOKENS = 8192` ([pipeline.py:33](pipeline.py#L33)).
게이트웨이의 `claude-sonnet-5` 는 thinking 모델이고 **사고 토큰이 `max_tokens` 에 함께
잡힌다.** 한도가 작으면 긴 문장에서 사고만으로 예산을 소진해 `content` 가 빈 문자열로
돌아오고, 검증기는 이를 `text_modified` 로 잡는다. 복구 재시도도 같은 한도라 똑같이 실패한다.

실측 (103자 문장): `max_tokens` 1024 → 빈 출력 / 4096 → 정상(1071 토큰) / 8192 → 정상(918 토큰).

즉 **`V` 가 낮을 때 원인이 프롬프트가 아닐 수 있다.** 모델을 바꾸면 이 한도를 먼저 확인한다.

---

## 3. L — 지연 프록시

### 정의

[metrics.py:211](metrics.py#L211) `latency_proxy()`.

조각 길이 `c_1 … c_k`, `N = Σc_i`, `누적_i = c_1 + … + c_i`, `p_i = c_i / N` 이라 할 때

```
L = Σ_i (c_i / N) · (누적_i / N) = Σ_i p_i · P_i        (P_i = Σ_{j≤i} p_j)
gain = 1 − L
```

**"정보 1단위가 평균적으로 기다린 문장 비율"** 이다. 시간 단위가 아니라 프록시다.
길이는 `spaced` 언어면 공백 포함, 아니면 공백을 제거하고 센다.

각 조각의 대기시간을 **그 조각이 담은 정보량 비중으로 가중**하는 것이 핵심이다.
조각 수로 평균하는 이전 정의(`Σ 누적_i / (k·N)`)는 1글자 조각과 30어절 조각을 같은 무게로
세기 때문에, 무의미한 조각을 앞에 만들수록 점수가 좋아지는 허점이 있었다. 실측:

| 분절 | L | gain |
|---|---|---|
| `그 다음에 이거 벤치마크 돌려보고 결과 나오면 그때 얘기하자고 했는데` (무분절) | 1.0000 | 0.0000 |
| `… 돌려보고 <SEG> 결과 나오면 …` (의미 분절) | 0.7502 | 0.2498 |
| `그 <SEG> 다음에 이거 …` (앞에 1글자만) | 0.9737 | 0.0263 |

정보 가중에서는 마지막 사례가 무분절과 거의 동등하게 평가된다.

### 닫힌 형태

`누적_i` 를 전개하면 `p_i p_j` 곱을 `j ≤ i` 인 모든 쌍에 대해 더한 것이 된다.

```
L = Σ_i Σ_{j≤i} p_i p_j
```

행렬 `M_ij = p_i p_j` 로 보면 전체 합이 `(Σp_i)² = 1`, 대각선이 `Σp_i²`, 대칭이므로
대각선을 뺀 나머지가 상·하 삼각에 정확히 반씩 들어간다. 따라서

```
L    = Σp_i² + (1 − Σp_i²)/2 = (1 + Σp_i²) / 2
gain = (1 − Σp_i²) / 2
```

항 하나를 쪼개면 의미가 드러난다.

```
p_i · P_i = p_i · (p_1 + … + p_{i−1})  +  p_i · p_i
             앞 조각들의 발화를 기다림      자기 자신의 발화를 기다림
```

- **자기 대기 총합 = `Σp_i²`** — 조각은 자기가 다 말해지기 전에는 나갈 수 없다.
- **앞선 대기 총합 = `(1 − Σp_i²)/2`** — 대칭성으로 고정된다.

정리하면 `L = 1/2 + Σp_i²/2`. **바닥 0.5 는 어떻게 잘라도 없어지지 않고, 그 위의 초과분은
전부 자기 대기 비용이다.** 최적화 여지는 `Σp_i²` 하나뿐이다.

### 성질

**① 순서에 무관하다.** `L` 은 `Σp_i²` 만의 함수이므로 조각을 재배열해도 값이 변하지 않는다.

**② 등분할이 최적이다.** `Σp_i²` 는 `p_i = 1/k` 에서 최소 `1/k`. 그때 `gain = 0.5 − 1/(2k)`.

**③ 상한이 0.5 다.** `gain = 1 − L` 이지만 `L ≥ 0.5` 이므로 실질 상한은 0.5.
마지막 조각은 항상 문장 끝까지 기다리고, 그 조각의 무게가 0으로 수렴해야 0.5 에 닿는다.

**④ 조각 수가 아니라 균등성이 지배한다.** `k_eff = 1/Σp_i²` (허핀달 역수)로 쓰면
`gain = (1 − 1/k_eff)/2`. 실제 `k` 가 커도 한 조각이 무거우면 `k_eff` 가 오르지 않는다.
`effective_segments()` 로 계산하며 항등식 `k_eff = 1/(2L − 1)` 로 `latency_proxy` 에서
직접 유도하므로 두 값이 어긋날 수 없다.

**⑤ `gain` 은 교차항 합과 같다.** `gain = Σ_{j<i} p_i p_j`.
`1 − Σp_i²` 는 "무작위로 뽑은 정보 단위 둘이 서로 다른 조각에 속할 확률"이므로,
`gain` 은 분절이 정보를 갈라놓은 정도의 절반이다.

### 검증

`latency_proxy()` 실측값과 닫힌 형태가 일치한다.

| 분절 | L (코드) | L (닫힌 형태) | gain | Σp² | k_eff |
|---|---|---|---|---|---|
| k=1 (무분절) | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.00 |
| k=2 등분할 | 0.7500 | 0.7500 | 0.2500 | 0.5000 | 2.00 |
| k=3 등분할 | 0.6667 | 0.6667 | 0.3333 | 0.3333 | 3.00 |
| k=4 등분할 | 0.6250 | 0.6250 | 0.3750 | 0.2500 | 4.00 |
| k=8 등분할 | 0.5625 | 0.5625 | 0.4375 | 0.1250 | 8.00 |
| 0.9 / 0.1 | 0.9100 | 0.9100 | 0.0900 | 0.8200 | 1.22 |
| 0.1 / 0.9 | 0.9100 | 0.9100 | 0.0900 | 0.8200 | 1.22 |
| 0.5 / 0.3 / 0.2 | 0.6900 | 0.6900 | 0.3100 | 0.3800 | 2.63 |
| 0.2 / 0.3 / 0.5 | 0.6900 | 0.6900 | 0.3100 | 0.3800 | 2.63 |
| 0.7 / 0.1 / 0.1 / 0.1 | 0.7600 | 0.7600 | 0.2400 | 0.5200 | 1.92 |
| 0.1 / 0.1 / 0.1 / 0.7 | 0.7600 | 0.7600 | 0.2400 | 0.5200 | 1.92 |

`k=4` 에 한 조각이 70% 를 차지하면 `gain` 0.2400 으로 `k=2` 등분할(0.2500)에 못 미친다.

재현:

```bash
PYTHONPATH=. python -c "
from core.meaning_segmentator.autoseg.metrics import latency_proxy
print(latency_proxy(' <SEG> '.join(['x'*250]*4), spaced=False))"
```

### 리포트되는 파생 지표

| 필드 | 정의 |
|---|---|
| `latency` | `L`. 문장별 값의 평균 |
| `latency_gain` | `1 − L`. 목적함수의 최대화 대상 |
| `mean_segments` | 문장별 조각 수 `k` 의 평균 |
| `mean_k_eff` | 문장별 `k_eff` 의 평균. `gain` 을 실제로 지배하는 값 |
| `segmented_rate` | `k > 1` 인 문장의 비율 |

`mean_segments` 와 `mean_k_eff` 는 둘 다 문장별 평균이라 같은 축에서 비교할 수 있다.
실측(ko→en test 100문장)에서 둘의 차이가 실제로 벌어진다.

| 프롬프트 | `mean_segments` | `mean_k_eff` | gain |
|---|---|---|---|
| 자동 run03 | 1.81 | 1.62 | 0.1383 |
| 사람 current | 2.12 | 1.92 | 0.1935 |
| 사람 v1 | 1.49 | 1.39 | 0.0927 |
| 사람 v2 | 1.41 | 1.32 | 0.0770 |

---

## 4. Q — 분절 품질

### 정의

```
Q = 유사도( 세그먼트별 번역을 이어붙인 것,  문장 전체를 한 번에 번역한 것 )
```

- 세그 번역: 조각을 **순서대로** 번역하되 앞선 번역은 확정으로 두고 수정하지 않는다
  ([pipeline.py:324](pipeline.py#L324) `streaming_segments`). 스트리밍 조건을 모사한 것이다.
- 참조: 같은 문장 전체를 한 번에 번역한 결과. 미래 문맥을 다 본 상한이다.
- 분절이 없는 문장은 세그 번역 = 전체 번역이므로 `Q = 1.0`. **제외하지 않는다.**

COMET 계열은 원문(src)도 입력으로 받으므로 `scorer.score(texts, seg_joined, full)` 로
세 가지를 모두 넘긴다 ([loop.py:74](loop.py#L74)).

### Q 와 Q_seg

| 필드 | 대상 | 용도 |
|---|---|---|
| `quality` | 전 문장 (무분절의 1.0 포함) | 보고용 |
| `quality_segmented` (`Q_seg`) | `k > 1` 인 문장만 | **제약 대상** |

제약을 분절된 문장에만 거는 것이 핵심이다. 전 문장 평균을 쓰면 무분절 문장의 1.0 이
평균을 끌어올려 "적게 분절하면 나쁘게 잘라도 통과"하는 역인센티브가 생긴다.
ja 실측에서 분절률 0.24 일 때 전체 `Q` 0.9668 이었지만 분절 문장만 보면 0.8618 로
`Q_floor` 미달이었다. `Q_floor` 를 전 문장을 자르는 기계분절로 캘리브레이션하므로
축을 맞추는 의미도 있다.

### LCB — 표본 수 방어

[metrics.py:300](metrics.py#L300) `quality_lcb()`.

```
LCB = Q_seg − z · (σ_seg / √n_segmented)        (z = 1.0)
```

목적함수는 `Q_seg` 가 아니라 `LCB` 로 판정한다. 표본이 작을수록 불리해지므로 루프가
**품질을 충분한 수의 분절 문장에서 입증**하도록 강제한다. 분절을 적게 하는 프롬프트는
그만큼 입증 부담을 진다.

`LCB` 는 표본 수에 대한 방어이지 개별 문장 붕괴에 대한 방어가 아니다. 꼬리는
`quality_min` / `quality_p10` 으로 리포트되지만 목적함수에는 들어가지 않는다.

### 백엔드

`--quality-backend` 로 고른다. 이름을 나눠 두는 이유는 `baseline.json` 에 어느
체크포인트로 캘리브레이션했는지가 남아야 하기 때문이다.

| 값 | 구현 | 체크포인트 | 비고 |
|---|---|---|---|
| `comet` (기본) | `CometBackend` | `Unbabel/wmt22-comet-da` | 운영 기준. `unbabel-comet` + GPU 필요 |
| `xcomet` | `CometBackend` | `Unbabel/XCOMET-XL` | HF 게이트 모델. 라이선스 동의 필요 |
| `embed` | `EmbedBackend` | 게이트웨이 `text-embedding-3-large` | 로컬 ML 의존성 없음 |
| `chrf` | `ChrfBackend` | 내장 | 문자 n-gram F-score. 보조 지표로 항상 함께 리포트 |

---

## 5. Q_floor 캘리브레이션

### 방법

[loop.py:211](loop.py#L211) `calibrate_q_floor()`. 앵커 두 개를 train 분할에서 **실측**한다.

```
Q_bad     = 의미를 무시하고 8자마다 자른 기계분절의 Q      (하한 앵커)
Q_ceiling = 무분절 + 동일 문장 2회 번역의 Q               (상한 앵커, 지표 잡음 바닥)
Q_floor   = Q_bad + ratio · (Q_ceiling − Q_bad)          (ratio 기본 0.7)
```

`Q_floor` 를 언어쌍과 무관한 고정 상수로 두면 위험하므로 '나쁜 분절'의 `Q` 를 실측해
상대 기준으로 삼는다. 두 앵커 사이가 좁으면(예: 0.05 미만) 그 백엔드는 분절 품질을
분해하지 못한다는 뜻이다.

**앵커는 백엔드와 번역기 양쪽에 의존한다.** `baseline.json` 에 둘 다 기록되고, 다른
조합으로 같은 런을 이어 돌리면 루프가 경고한다.

### 측정값 (ko→en, KsponSpeech, train 30문장, 8자 기계분절, ratio 0.7)

LLM 번역기 (`claude-sonnet-5`) — `runs/ko-en/run01/baseline.json`

| 백엔드 | 하한 | 상한 | 간격 | `Q_floor` |
|---|---|---|---|---|
| comet | 0.6457 | 0.8591 | 0.2134 | 0.7951 |
| embed | 0.7356 | 0.9150 | 0.1794 | 0.8612 |
| chrf | 0.4706 | 0.7123 | 0.2417 | 0.6398 |

Google Translate (`--translator google`, `ctx=True`) — `runs/ko-en/run03-google-fix/baseline.json`

| 백엔드 | 하한 | 상한 | 간격 | `Q_floor` |
|---|---|---|---|---|
| comet | 0.6070 | 1.0000 | 0.3930 | 0.8821 |
| embed | 0.6517 | 1.0000 | 0.3483 | 0.8955 |
| chrf | 0.4322 | 1.0000 | 0.5678 | 0.8297 |

상한이 정확히 1.0 인 것은 gtx 가 결정론적이기 때문이다 — 같은 문장을 두 번 번역하면
완전히 일치하므로 `Q` 에서 번역기 잡음이 사라진다. LLM 번역기는 `temperature=0` 에서도
같은 입력에 다른 출력을 내므로 상한이 1.0 에 못 미치고, 그 차이가 곧 비결정성의 크기다.

**간격만으로 백엔드를 고르면 안 된다.** chrF 간격이 가장 넓지만 §6 타당도 검사에서
위반이 가장 많다. 간격은 분해능이고 타당도는 별개이며 둘 다 필요하다.

---

## 6. 백엔드 타당도 검사

캘리브레이션은 **스케일만** 맞춘다. 순위가 맞는지는 보증하지 않는다. 그래서 백엔드를
바꿀 때마다 오류 주입 검사를 먼저 통과시킨다.

```bash
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.validity_check --backends comet embed chrf
```

케이스 5건(ja→ko 1, ko→en 4) × 변이 8종. 오류는 `validity_cases.json` 에 고정되어 있어
LLM 생성이 아니고 매 실행 결과가 같다.

**통과 조건: 모든 케이스에서 심각한 의미 오류의 점수가 `benign_minimal` 보다 낮을 것.**

무해한 변이를 두 단계로 나눈 것이 이 검사의 설계 핵심이다. 참조 기반 지표는 표현이
크게 바뀌면(재서술) 의미가 같아도 점수를 깎는 반면, 심각한 의미 오류는 오히려 표면형이
참조와 가깝다. `benign_paraphrase` 를 기준으로 삼으면 **어떤 백엔드든** 탈락한다.

- `benign_minimal` — 동의어·음역 차이. **판정 기준.**
- `benign_paraphrase` — 문장 구조까지 바꾼 재서술. 참고용.

### 결과 (`runs/validity/validity_report.md`)

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal (기준) | 0.9308 | 0.9509 | 0.8620 |
| benign_paraphrase | 0.8414 | 0.8533 | 0.3830 |
| **negation_flip** | 0.8843 | 0.8976 | 0.8081 |
| **role_swap** | 0.9365 | 0.9122 | 0.8678 |
| **clause_omission** | 0.7693 | 0.8593 | 0.5497 |
| **referent_loss** | 0.8660 | 0.9016 | 0.7326 |
| unrelated | 0.3948 | 0.0633 | 0.1216 |
| **순위 위반 / 17** | **1** | 3 | 5 |

- **comet** — 위반 1건. `ja-ko-01` 의 `role_swap` 이 0.9353 으로 무해한 변이 0.9224 보다
  0.0129 높다. 부정 뒤집힘은 5개 케이스 전부에서 무해한 변이보다 낮고, 절 누락도
  확실히 잡는다(0.7693, 전 변이 중 `unrelated` 다음으로 낮음).
- **embed** — 위반 3건. 절 누락(0.8593)과 지시대상 소실(0.9635)을 무해한 변이보다
  높게 매긴다.
- **chrf** — 위반 5건. 주체 뒤바뀜(0.8678)을 무해한 변이(0.8620)보다 높게 준다.
  단독 사용 불가.

리포트의 판정 열은 위반이 1건이라도 있으면 **탈락**으로 찍는다. 세 백엔드 모두 현재
탈락 표기이며, 실제 운영 기준은 위반 수가 가장 적은 `comet` 이다.

`comet` 과 `chrf` 열은 재실행해도 값이 정확히 같다. `embed` 열만 게이트웨이 임베딩 API 의
비결정성으로 소수 4자리에서 흔들린다 (판정은 바뀌지 않는다).

---

## 7. 목적함수

[metrics.py:315](metrics.py#L315) `objective()`.

```
maximize   gain + q_weight · (LCB − Q_floor)
s.t.       V = 1.0
           LCB(Q_seg) ≥ Q_floor
```

```python
def objective(m, q_floor, q_weight=1.0):
    if m.valid_rate < 1.0:  return -10.0 + m.valid_rate
    if m.n_segmented == 0:  return 0.0
    lcb = quality_lcb(m)
    if lcb < q_floor:       return -1.0 + (lcb - q_floor)
    return m.latency_gain + q_weight * (lcb - q_floor)
```

판정 순서와 각 반환값의 의미:

| 조건 | 반환 | 이유 |
|---|---|---|
| `V < 1.0` | `−10 + V` | 하드 탈락. `+V` 는 V 0.90 이 0.97 보다 나쁘다는 방향을 남기기 위함 |
| 분절 문장 0건 | `0.0` | 아무것도 자르지 않으면 지연 이득도 0 — 최악과 동치 |
| `LCB < Q_floor` | `−1 + (LCB − Q_floor)` | 위반 정도에 비례한 음수. 방향을 잃지 않게 함 |
| 통과 | `gain + q_weight·여유` | 지연 이득 + 품질 여유 |

### 왜 가중항이 있나

순수 제약형(`return m.latency_gain`)에서는 floor 를 넘은 뒤 `Q` 가 점수에서 완전히
사라진다. `Q_floor` 만 넘으면 `Q` 0.99 든 0.89 든 동점이고 `gain` 으로만 갈리므로
**최적해가 floor 경계에 수렴한다.** 실측에서 채택된 프롬프트의 여유가 0.0012 였다.
경계에 붙은 해는 데이터가 조금만 달라져도 제약을 위반한다.

가중항은 여유를 점수로 사서 해를 경계 안쪽으로 밀어낸다. floor 자체는 하드로 남기므로
가중항이 위반을 상쇄하지 못한다.

### q_weight

`--q-weight` (기본 `1.0`). "품질 여유 1점 = 지연 이득 몇 점"의 교환비다.
`config.json` 에 기록되며 `eval_prompt.py` 는 기준 런에서 상속한다.

**다른 `q_weight` 로 잰 objective 는 비교할 수 없다.** `0` 을 주면 순수 제약형으로 돌아간다.

교환비 감각: `gain` 의 실측 범위는 0.08 ~ 0.20, 품질 여유(`LCB − Q_floor`)의 범위는
0 ~ 0.05 정도다. `q_weight = 1.0` 이면 품질 여유가 `gain` 차이를 뒤집을 수 있지만
지배하지는 않는다.

### 저장된 프롬프트 재채점 (ko→en test 100문장, `Q_floor` 0.8821)

`runs/ko-en/run03-google-fix/` 의 `test_rows.json` 과 `prompt_eval/*.json` 을 현재
코드로 다시 채점한 값이다.

| 프롬프트 | V | LCB | 여유 | gain | `q_weight=0` | `q_weight=1` |
|---|---|---|---|---|---|---|
| 자동 run03 | 1.00 | 0.8833 | +0.0012 | 0.1383 | **0.1383** | **0.1395** |
| 사람 v1 | 1.00 | 0.9215 | +0.0394 | 0.0927 | 0.0927 | 0.1320 |
| 사람 v2 | 1.00 | 0.9277 | +0.0456 | 0.0770 | 0.0770 | 0.1226 |
| 사람 current | 1.00 | 0.8481 | −0.0340 | 0.1935 | −1.0340 | −1.0340 |

순위는 바뀌지 않았지만 자동 run03 과 사람 v1 의 격차가 0.0456 → 0.0075 로 줄었다.
floor 에 0.0012 로 붙어 있던 해가 더 이상 안전 마진 없이 이기지 못한다.
`current` 는 floor 미달이라 가중치와 무관하게 탈락한다.

재현:

```bash
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_v1.txt \
    --run-id ko-en/run03-google-fix --split test --label human_v1 --q-weight 1.0
```

### Critic 의 direction 과의 정합

[agents.py:222](agents.py#L222) `summarize_critique()` 의 `direction` 은 목적함수와 같은
기준(`LCB`)을 쓴다. 평균 `Q_seg` 를 보면 판정이 어긋난다 — 실측에서 `Q_seg` 0.8846 >
floor 0.8821 이라 "품질 통과"로 보고 `segment more aggressively` 를 냈지만, 목적함수는
같은 이터레이션을 `LCB` 0.8647 < floor 로 거부했다. 병목이 품질인데 더 자르라고 지시하는
상태였다.

분절량 판정도 같은 이유로 `mean_k_eff` 를 쓴다. `gain` 은 `k_eff` 하나에만 의존하므로
(§3 성질 ④) 조각 수로 재면 사각지대가 생긴다 — `0.7/0.1/0.1/0.1` 은 `mean_segments` 4.0
으로 "분절 충분" 판정을 받지만 `k_eff` 1.92 라 `k=2` 등분할보다 느리다.

현재 `direction` 결정 순서:

```
V 미달                                       → fix output format
LCB < Q_floor                                → fix boundary placement
segmented_rate < 0.6  또는  mean_k_eff < 2.0 → segment more aggressively
over_seg + wrong_boundary > under_seg        → fix boundary placement
그 외                                         → segment more aggressively
```

임계값 `2.0` 의 뜻은 "평균적으로 등분할 2조각". Critic LLM 에 넘기는 설명문에도
`mean_k_eff` 의 해석과 "지연을 사는 것은 조각 수가 아니라 균등성"이라는 사실이 들어간다.
Prompt Engineer 프롬프트의 목표 서술도 가중합 형태와 균등성 조건을 함께 명시한다.

---

## 8. 재현

```bash
# 1) 타당도 게이트 — 백엔드를 바꿨다면 여기부터
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.validity_check --backends comet embed chrf

# 2) 캘리브레이션 + 루프 (앵커는 자동 산출되어 baseline.json 에 기록된다)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id runNN --translator google \
    --iterations 6 --train 30 --dev 60 --test 100 --min-chars 25 --budget 20

# 3) 임의 프롬프트를 같은 분할·앵커·백엔드로 평가 (번역기·백엔드는 기준 런에서 상속)
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.eval_prompt \
    --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
    --run-id ko-en/runNN --split test --label human_current
```
