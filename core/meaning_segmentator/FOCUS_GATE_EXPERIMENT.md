# focus 관문 실험 (브랜치 `autoseg/focus-gate`)

기준 커밋 `2349834` (other_policy). 이 브랜치를 만들 때의 미커밋 diff는
`.git/autoseg-focus-gate-baseline.patch` 에 스냅샷해 뒀다 — 실험 변경분만 보려면
`git diff` 에서 이 패치를 빼면 된다.

## 왜 팠나

de/ja/zh→en run01 세 런 전부, **모든 이터레이션에서 `focus = format`** 이었다.
`placement` 도 `priority` 도 한 번도 선택되지 않았다.

| 런 | iter0 | iter1 | iter2 | focus_reason |
|---|---|---|---|---|
| de-en/run01 | format | format | format | `format_pass_rate 0.9667 < 1.0` |
| zh-en/run01 | format | format | format | `0.7333 < 1.0` |
| ja-en/run01 | format | format | format | `0.8667 / 0.8000 < 1.0` |
| en-de/run04 | placement | placement | — | 기본값 (지표 무발화) |

그 결과 `priority_audit` 가 4회 내리 지목한 **독일어 쉼표 과신**(`뒤 구두점 ','`
백분위 0.575 / contra 0.2100 vs 그 외 0.10, over_trust +0.0606, 표의 1위)이
한 번도 PE 에 전달되지 않았다. PE 시스템 프롬프트가 `priority_audit` 를
**`focus == "priority"` 일 때만** 쓰라고 지시하기 때문이다 (`agents.py` PE 지시문 8번).

## 결함 두 개

### D1 — `rank_lift*` 가 합치는 단계에서 사라진다 (버그)

`loop.evaluate` 는 순위축을 직접 측정한다(태그 번호만 섞은 대조군, **LLM 호출 0건**).
그런데 `loop.evaluate_multi` 가 타깃별 결과를 합칠 때 `Metrics` 를 새로 만들면서
`rank_lift / rank_lift_se / rank_lift_t / rank_lift_n / rank_lift_T` 5개 필드를 **버린다**:

```python
merged_m = metrics.Metrics(n=m0.n, format_pass_rate=m0.format_pass_rate,
                           format_pass_rate_no_retry=m0.format_pass_rate_no_retry,
                           by_T=split_m)          # ← rank_lift* 누락
```

루프의 모든 평가는 타깃이 1개여도 `run_eval → evaluate_multi` 를 지나므로
`rank_lift_t` 는 **항상 `None`** 이고, 순위 분기 조건
`lift_t is not None and lift_t < RANK_LIFT_T_MIN` 이 **영원히 거짓**이다.
→ `focus = "priority"` 는 구조적으로 도달 불가능.

실측 확인: de/ja/zh run01 + en-de run04 의 모든 `critique.json` 에서
`rank_lift = None, rank_lift_t = None`.

### D2 — `format` 관문이 절대 비교라 못 고치는 1문장이 런을 볼모로 잡는다

`focus` 는 우선순위 목록이 아니라 캐스케이드다. 첫 관문이
`format_pass_rate < 1.0` 인데, de 는 train 30문장 중 **재시도 후에도 실패하는 1문장**
때문에 0.9667 에 고정됐고 4 이터레이션 방향이 전부 `format` 으로 묶였다.
뒤 관문(coverage/priority/placement)은 **평가조차 되지 않는다.**

## 가설

- **H1** — D1 을 고치면 de 에서 `focus = priority` 가 실제로 선택된다.
- **H2** — 쉼표 규칙을 고치면 순위축이 개선된다.
  근거: en-de 에서 같은 개입이 `rank_contra_gap` −0.005 → +0.032 (순열 p=0.005) 를
  냈고, `metrics.priority_audit` 주석이 이를 "순위를 유의하게 개선한 **유일한** 개입"
  으로 기록하고 있다.
- **H3** — D2 를 고치면 format 에 갇힌 런이 뒤 관문으로 넘어간다.

## 절차

### Step 0 — 오프라인 재생 (**$0, API 호출 없음**)

고치기 전에 먼저 잰다. `rank_lift` 계산은 저장된 태그의 번호만 섞으므로
`train_rows.json` 만으로 재현된다. `summarize_critique` 도 순수 함수다.

1. de/ja/zh run01 의 저장된 rows 로 `rank_lift` 를 계산한다.
2. 그 값을 넣어 `summarize_critique` 를 다시 돌려 **focus 가 무엇이 되었을지** 본다.
3. `format` 관문을 완화했을 때의 focus 도 함께 낸다.

여기서 `priority` 가 안 나오면 D1 을 고쳐도 소용없다 — 그때는 `RANK_LIFT_T_MIN`
자체를 다시 봐야 한다. **이 단계 결과 없이는 코드를 고치지 않는다.**

### Step 1 — 수정

- D1: `evaluate_multi` 의 `Metrics` 재구성에 `rank_lift*` 5개 필드 전달.
- D2: format 관문을 상대화 (후보: `< min(1.0, 직전 baseline)`, 또는
  "개선 없이 2회 연속 format 이면 다음 관문으로"). Step 0 결과를 보고 고른다.

### Step 2 — 재현 런

`runs/de-en/exp01` 로 de 를 다시 돌린다 (기존 `run01` 은 건드리지 않는다).
예상 비용은 run01 실적 기준 **$4~5**.

### Step 3 — 비교

`run01` vs `exp01`: iter 별 focus, `rank_lift`, 채택 여부, dev 쌍체 Δ/se,
test 곡선. H1~H3 각각에 대해 참/거짓을 적는다.

## 판정 기준

- H1: exp01 에서 `focus = priority` 가 1회 이상 선택되면 참.
- H2: 쉼표 규칙 개정이 dev 쌍체 Δ > 0.5·se 를 넘으면 참.
  (넘지 못해도 `rank_lift` 가 오르면 부분 성공으로 기록한다 — 채택 문턱과 순위축
  개선은 별개 질문이다.)
- H3: exp01 의 focus 가 전 이터레이션 `format` 이 아니면 참.

## 기록

결과는 여기에 이어 적는다. `MULTI2EN_DATASET.md` 에는 결론이 나온 뒤에만 옮긴다.

---

## 환경

- 브랜치 `autoseg/focus-gate` (기준 `2349834`). 미커밋 WIP 스냅샷:
  `.git/autoseg-focus-gate-baseline.patch`
- venv `.venv-autoseg` (py3.10, torch 2.13.0+cu130, unbabel-comet, sacrebleu 2.6). **`PYTHONPATH=` 를 반드시 비울 것** — ROS 의 PYTHONPATH 가 venv 를 가린다.
- 재생 도구: `autoseg/replay_focus.py`

```bash
PYTHONPATH= .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.replay_focus \
    --run core/meaning_segmentator/runs/de-en/run01 --offline --device cpu
```

`--offline` 은 번역 캐시 미스에서 즉시 죽는다. de/ja/zh run01 은 **미스 0건**으로 돌았다
— 셔플 대조군의 번역이 이미 캐시에 있다는 뜻이고(버그는 계산이 아니라 **필드 전달**만
망가뜨렸다), 따라서 Step 0 은 API·네트워크 비용이 **정확히 0** 이다.

### GPU 가 죽어 있다 (실험과 무관한 기계 문제)

`nvidia-smi` 는 정상인데 `cuInit → 100 (NO_DEVICE)` 로 CUDA 초기화가 실패한다.
커널 모듈과 유저스페이스 libcuda 는 둘 다 580.173.02 로 일치하고, 서스펜드 이력도 없다
(uptime 2주). 어제 런들은 `Device set to use cuda:0` 로 정상이었으므로 그 사이에 깨졌다.

복구는 sudo 가 필요하다 (사람이 직접):

```bash
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
```

그 전까지는 `--device cpu` 로 돈다. 재생 규모(30문장 × 3셔플)에서는 CPU 로 충분하다.
**본 런(Step 2)은 GPU 없이는 비현실적이다** — CometKiwi + NLI 를 매 이터 수백 번 부른다.

---

## Step 0 결과 (2026-08-23, 비용 $0, 캐시 미스 0건, CPU)

`runs/replay/{de,ja,zh}-en_run01_focus.json`, 로그 `runs/replay/step0.log`.
셔플 3벌, seed 0, lift_T = max(t_grid).

| 런 | iter | rank_lift | se | t | focus (D1만) | focus (D1+D2) |
|---|---|---|---|---|---|---|
| de-en | 0 | +0.0379 | 0.0203 | +1.87 | format | placement |
| de-en | 1 | +0.0530 | 0.0188 | +2.82 | format | placement |
| de-en | 2 | +0.0530 | 0.0188 | +2.82 | format | placement |
| ja-en | 0 | +0.0184 | 0.0550 | +0.33 | format | **priority** |
| ja-en | 1 | −0.0275 | 0.0509 | −0.54 | format | **priority** |
| ja-en | 2 | +0.1069 | 0.0440 | +2.43 | format | placement |
| zh-en | 0 | +0.0535 | 0.0314 | +1.70 | format | placement |
| zh-en | 1 | +0.0416 | 0.0313 | +1.33 | format | placement |
| zh-en | 2 | +0.0261 | 0.0344 | +0.76 | format | **priority** |

### 판정

- **H1 = 거짓 (de), 참 (ja/zh).** D1 을 고쳐도 de 는 `priority` 에 도달하지 않는다.
  de 의 순위축은 **측정상 건강하다** — 순위를 섞으면 effective 가 0.038~0.053 떨어지고
  t 가 1.87~2.82 로 문턱(`RANK_LIFT_T_MIN = 1.0`)을 여유 있게 넘는다.
  ja/zh 는 9회 중 3회 `priority` 가 걸린다.
- **H3 = 참, 그리고 D2 가 유일한 구속조건이다.** `focus(D1만)` 열이 9/9 전부 `format`
  이다. D1 을 고쳐도 **혼자서는 아무것도 안 바뀐다** — format 관문이 첫 번째라
  뒤 관문이 평가되지 않기 때문이다. D2 가 실제 병목이고 D1 은 그 뒤에서 드러난다.
- **de 의 쉼표 문제는 두 결함을 다 고쳐도 여전히 PE 에 안 간다.**

### 이게 뜻하는 것 — 축은 건강한데 규칙 하나가 틀렸다

`rank_lift` 는 "순위가 무작위보다 나은가"를 묻고, `priority_audit` 는 "어느 규칙이
틀렸는가"를 묻는다. **두 질문은 다르고, de 에서 답이 갈린다**: 축은 통과(t=2.82)인데
쉼표 규칙은 틀려 있다(백분위 0.575 로 평균보다 확신하는데 contra 0.2100 vs 그 외 0.10).

그런데 지금 구조는 **축이 통과하면 감사표를 아예 안 읽는다** — PE 지시문이
`priority_audit` 를 `focus == "priority"` 일 때만 쓰라고 못박고 있다.
즉 "3순위라 밀렸다"가 아니라 **순위 매김 대상에 애초에 안 올라간다.** 입도(granularity)
불일치이지 우선순위 오배정이 아니다.

### 그래서 D3 이 필요하다

축 수준 검사를 통과해도 **특정 특징이 강하게 과신되면** 감사표가 PE 에 닿아야 한다.
후보:

- **D3a** — `priority_audit[0]["over_trust"]` 가 문턱을 넘고 `n` 이 충분하면
  `focus = "priority"` 를 허용 (축 검사와 OR).
- **D3b** — focus 는 그대로 두되 감사표를 **항상** PE 에 붙이고, 지시문을
  "focus 섹션을 고치되, 감사표가 특정 규칙을 지목하면 그 줄도 고쳐도 된다"로 완화.
  `check_revision` 의 섹션 허용 목록도 함께 풀어야 한다.
- **D3c** — 진단만 하고 프롬프트에는 안 넣는다 (현상 유지 + 기록).

`over_trust` 문턱은 **임의로 정하지 말 것.** 세 언어 9 이터의 감사표 분포를 먼저 뽑아
발화율을 보고 잡는다 (이것도 비용 0 — 저장된 `priority_audit.json` 만 읽으면 된다).

### 부수 관찰

- ja 의 `se` 가 de/zh 의 2~3배(0.044~0.055 vs 0.019~0.034)다. ja 순위축은 측정
  자체가 불안정하다 — ja iter1 은 lift 가 **음수**(−0.0275)지만 t 는 −0.54 로 잡음 범위다.
- de iter1 과 iter2 의 값이 완전히 동일하다. 채택이 없어 best 프롬프트가 안 바뀌었고
  분절이 캐시 히트라 같은 입력이 재생된 것으로, 재생이 결정론적임을 보여준다.
- zh 재생이 종료 직전 `corrupted double-linked list` 로 코어를 떨궜다. **결과 저장
  이후**라 데이터는 온전하다. CPU 백엔드 teardown 문제로 보이며 실험과 무관하다.

---

## Step 0b — `over_trust` 귀무 분포 (비용 $0, GPU 불필요)

### 먼저: 생값 문턱은 못 쓴다

저장된 `priority_audit.json` 16 이터(5개 런)의 **1위 `over_trust`** 분포를 보면
어떤 문턱을 잡아도 거의 항상 발화한다:

| 문턱 | 1위가 넘는 이터 |
|---|---|
| > 0.02 | 15/16 (94%) |
| > 0.05 | 13/16 (81%) |
| > 0.10 | 10/16 (62%) |

당연하다. `over_trust = (기준 백분위 − 이 특징 백분위) × (contra 비)` 는 **상대량**이라
평균보다 확신하는 특징이 항상 하나는 있다. **순위표이지 검출기가 아니다.**
생값 문턱을 쓰면 `format` 고착을 `priority` 고착으로 바꾸는 것뿐이다.

### 귀무는 `rank_lift` 와 같은 수법으로 만든다

경계 위치와 `pieces_contra` 는 그대로 두고 **순위 번호만 치환**해
(`pipeline.shuffle_priorities`) `priority_audit` 을 다시 돌린다. 순열 200회, seed 0.
번역도 모델도 안 부르므로 **비용 0**이다.

**측정 T 에 주의.** `priority_audit` 은 `low_t = min(t_grid)` 에서 잰다
(`loop.py:1313`). `rank_lift` 는 반대로 `max(t_grid)` 다 — 폐기율이 가장 높은 곳.
두 순위 진단이 **서로 다른 T 를 본다**는 사실 자체가 둘이 어긋날 수 있는 이유 중 하나다.

### 결과 (`runs/replay/over_trust_null.json`)

| 런 | it | 관측 1위 특징 | 관측 | 귀무평균 | 귀무95% | z | p |
|---|---|---|---|---|---|---|---|
| de-en | 0 | 뒤 구두점 ',' | 0.0606 | 0.0618 | 0.1274 | −0.03 | 0.465 |
| de-en | 1 | 뒤 구두점 ',' | 0.2227 | 0.0861 | 0.1921 | **+2.44** | **0.020** |
| de-en | 2 | 뒤 구두점 ',' | 0.2227 | 0.0861 | 0.1921 | **+2.44** | **0.020** |
| de-en | 3 | 뒤 구두점 ',' | 0.1583 | 0.0840 | 0.1789 | +1.37 | 0.090 |
| ja-en | 0 | 뒤 구두점 '、' | 0.0890 | 0.0942 | 0.2644 | −0.06 | 0.390 |
| ja-en | 1 | 상대위치 0/3 | 0.2176 | 0.0686 | 0.1485 | **+3.70** | **0.005** |
| ja-en | 2 | 상대위치 0/3 | 0.1497 | 0.0736 | 0.1577 | +1.73 | 0.070 |
| zh-en | 0 | 상대위치 1/3 | 0.1352 | 0.0724 | 0.1363 | +1.60 | 0.055 |
| zh-en | 1 | 뒤 구두점 '，' | 0.1308 | 0.0695 | 0.1392 | +1.68 | 0.075 |
| zh-en | 2 | 상대위치 1/3 | 0.1344 | 0.0682 | 0.1329 | +1.86 | 0.050 |
| en-de | 0 | 뒤 구두점 ',' | 0.0000 | 0.0857 | 0.3140 | −0.79 | 1.000 |
| en-de | 1 | 뒤 구두점 ',' | 0.0939 | 0.0515 | 0.1008 | +1.69 | 0.075 |
| en-de | 2 | 상대위치 0/3 | 0.1690 | 0.0582 | 0.1310 | **+3.04** | **0.010** |
| en-multi | 0 | 상대위치 0/3 | 0.0489 | 0.0580 | 0.1387 | −0.23 | 0.500 |
| en-multi | 1 | 뒤 구두점 ',' | 0.1770 | 0.0692 | 0.1423 | **+2.47** | **0.025** |
| en-multi | 2 | 상대위치 2/3 | 0.0426 | 0.0592 | 0.1412 | −0.39 | 0.515 |

**발화율 `p < 0.05` = 5/16 (31%).** 고착도 침묵도 아니다 — 쓸 수 있는 관문이다.

앞서 후보로 적었던 생값 문턱 0.02~0.10 은 **전부 귀무 95 분위(0.10~0.31) 아래**다.
그대로 썼으면 잡음에 발화했을 것이다.

### de 쉼표는 실재한다

de iter1/2 에서 쉼표 과신이 **z=+2.44, p=0.020** 으로 귀무를 넘는다.
"순위축이 건강하니(rank_lift t=2.82) 볼 것 없다"가 **틀렸다** — 축은 건강한데
그 안의 쉼표 규칙은 유의하게 틀려 있다. 두 진단이 각자 맞는 말을 하고 있었고,
지금 구조가 축 쪽 답만 듣고 있었을 뿐이다.

de iter0 은 p=0.465 로 유의하지 않다. **매 이터 걸리는 것이 아니라** 프롬프트가
바뀌면서 드러난다는 뜻이고, 그래서 "감사표를 항상 반영"이 아니라 관문이 필요하다.

### D3 확정안

**D3a′** — `priority_audit` 1위의 `over_trust` 가 **순열 귀무의 95 분위를 넘으면**
`focus = "priority"` 를 허용한다 (축 검사 `rank_lift` 와 **OR**).
문턱은 상수가 아니라 **매 이터 계산되는 귀무**라, 새로 생기는 임의 상수가 없다.

비용: 순열 200회 × `priority_audit` (순수 산술). 번역·모델 호출 0.
