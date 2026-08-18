# NLI 대체 후보 실측 — SummaC 집계 / MiniCheck / 재번역 premise

런: `ko-en/run04` · 경계 1003개 · 바닥 측정 문장 150개 · LLM 호출 0

설계와 후보 선정 근거는 [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md). 실험을 **premise 축 × scorer 축**으로 인수분해해 어느 쪽이 기여했는지 분리한다.

재번역 premise: gtx 고유 호출 712건. 오라클과 실제로 다른 경계 **593개**, 나머지 410개는 다음 조각이 문장 끝이라 오라클과 같아진다.

> **관문의 한계.** `premature_cases.json` 은 전 케이스가 2조각이라 '다음 조각' 이 곧 문장 전체다 — 관문에서는 `retrans` 가 `gtx(소스 전체)` 이고 `oracle`(손으로 쓴 full 번역)과 같은 것을 가리킨다. **관문은 scorer 축만 가른다.** premise 축의 증거는 실데이터 쪽에 있다.

## T1 — contradiction 관문 (`premature_cases.json`)

통과 조건은 `judge_check.check_nli` 와 동일: 케이스마다 `min(premature) > max(safe)`.

| premise | scorer | 변이 | 위반/케이스 | mean(prem) | mean(safe) | 격차 | 판정 |
|---|---|---|---|---|---|---|---|
| oracle | nli | raw | 0/6 | 0.6264 | 0.0318 | 0.5946 | 통과 |
| oracle | nli | floor | 1/6 | 0.5955 | 0.0194 | 0.5761 | **탈락** |
| oracle | nli | z | 0/6 | 10.0223 | -0.0379 | 10.0602 | 통과 |
| oracle | summac | raw | 0/6 | 0.7529 | 0.326 | 0.4269 | 통과 |
| oracle | summac | floor | 1/6 | 0.4167 | 0.1049 | 0.3118 | **탈락** |
| oracle | summac | z | 0/6 | 0.9622 | -0.3511 | 1.3133 | 통과 |
| oracle | minicheck | raw | 0/6 | 0.92 | 0.5134 | 0.4066 | 통과 |
| oracle | minicheck | floor | 0/6 | 0.5188 | 0.1485 | 0.3703 | 통과 |
| oracle | minicheck | z | 0/6 | 1.9517 | 0.2131 | 1.7385 | 통과 |
| oracle | erasure | raw | 4/6 | 0.5754 | 0.6333 | -0.0579 | **탈락** |
| oracle | erasure_p | raw | 5/6 | 0.8651 | 0.8571 | 0.0079 | **탈락** |
| retrans | nli | raw | 1/6 | 0.5547 | 0.0335 | 0.5212 | **탈락** |
| retrans | nli | floor | 1/6 | 0.5399 | 0.0126 | 0.5273 | **탈락** |
| retrans | nli | z | 0/6 | 12.441 | -0.0286 | 12.4696 | 통과 |
| retrans | summac | raw | 2/6 | 0.6942 | 0.2437 | 0.4505 | **탈락** |
| retrans | summac | floor | 3/6 | 0.4571 | 0.1002 | 0.357 | **탈락** |
| retrans | summac | z | 1/6 | 1.1998 | -0.1893 | 1.3891 | **탈락** |
| retrans | minicheck | raw | 2/6 | 0.8737 | 0.6613 | 0.2123 | **탈락** |
| retrans | minicheck | floor | 1/6 | 0.4153 | 0.2153 | 0.2 | **탈락** |
| retrans | minicheck | z | 1/6 | 1.7078 | 0.5773 | 1.1305 | **탈락** |
| retrans | erasure | raw | 5/6 | 0.627 | 0.7762 | -0.1492 | **탈락** |
| retrans | erasure_p | raw | 6/6 | 1.0 | 1.0 | 0.0 | **탈락** |

## T2 — 잡음 바닥 (정의상 무해한 미완성)

premise 의 어절 prefix 를 hypothesis 로 넣은 점수. 모순일 수 없는 입력이므로 여기 나오는 값이 바닥이다. **바닥이 낮고 평탄할수록 좋다** — 길이에 따라 출렁이면 앞쪽 경계가 구조적으로 불리해진다.

| premise | scorer | 전체 mean | sd | 1-2 | 3-4 | 5-6 | 7-9 | 10-14 | 15+ |
|---|---|---|---|---|---|---|---|---|---|
| oracle | nli | 0.0302 | 0.1036 | 0.1353 | 0.0479 | 0.0204 | 0.0078 | 0.0063 | 0.003 |
| oracle | summac | 0.2574 | 0.3472 | 0.5065 | 0.4625 | 0.3755 | 0.2169 | 0.1121 | 0.0714 |
| oracle | minicheck | 0.4586 | 0.2594 | 0.6139 | 0.4007 | 0.3839 | 0.4373 | 0.4467 | 0.4845 |
| oracle | erasure | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| oracle | erasure_p | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| retrans | nli | 0.0263 | 0.0887 | 0.0874 | 0.0266 | 0.0104 | 0.0063 | 0.0063 | 0.0037 |
| retrans | summac | 0.2124 | 0.321 | 0.3867 | 0.312 | 0.249 | 0.1417 | 0.0765 | 0.04 |
| retrans | minicheck | 0.4893 | 0.2564 | 0.6623 | 0.4808 | 0.446 | 0.4157 | 0.4201 | 0.4922 |
| retrans | erasure | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| retrans | erasure_p | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### 신호 대 바닥 (SNR)

신호 = 관문의 `mean(premature) − mean(safe)`(raw). **SNR 이 클수록 잘못 자른 경계가 무해한 미완성의 산포 위로 솟는다.**

| premise | scorer | 신호 | 바닥 sd | SNR |
|---|---|---|---|---|
| retrans | nli | 0.5212 | 0.0887 | 5.88 |
| oracle | nli | 0.5946 | 0.1036 | 5.74 |
| oracle | minicheck | 0.4066 | 0.2594 | 1.57 |
| retrans | summac | 0.4505 | 0.321 | 1.4 |
| oracle | summac | 0.4269 | 0.3472 | 1.23 |
| retrans | minicheck | 0.2123 | 0.2564 | 0.83 |
| oracle | erasure | -0.0579 | 0.0 | None |
| oracle | erasure_p | 0.0079 | 0.0 | None |
| retrans | erasure | -0.1492 | 0.0 | None |
| retrans | erasure_p | 0.0 | 0.0 | None |

## T3 — 실데이터 거동 (정답 없음)

경계 1003개에는 라벨이 없다. 아래 상관은 **현행 NLI(oracle)와 얼마나 같게 움직이나** 이지 정확도가 아니다. `topk 겹침` 은 루프가 판정자에게 보내는 최상위 경계 집합의 겹침 — 조향이 바뀌는지를 본다.

| premise | scorer | 변이 | Spearman(전역) | Spearman(문장 내) | topk 겹침 |
|---|---|---|---|---|---|
| oracle | summac | raw | 0.4202 | 0.3153 (n=148) | 0.1 |
| oracle | summac | floor | 0.2752 | 0.1658 (n=117) | 0.25 |
| oracle | summac | z | 0.2844 | 0.182 (n=148) | 0.25 |
| oracle | minicheck | raw | 0.4448 | 0.424 (n=148) | 0.15 |
| oracle | minicheck | floor | 0.3693 | 0.3171 (n=141) | 0.1 |
| oracle | minicheck | z | 0.3276 | 0.2772 (n=148) | 0.0 |
| oracle | erasure | raw | 0.3043 | 0.2111 (n=139) | 0.05 |
| oracle | erasure_p | raw | 0.2022 | 0.205 (n=25) | 0.0 |
| retrans | nli | raw | 0.7651 | 0.5109 (n=148) | 0.7 |
| retrans | nli | floor | 0.6018 | 0.3565 (n=120) | 0.6 |
| retrans | nli | z | 0.4896 | 0.1543 (n=148) | 0.3 |
| retrans | summac | raw | 0.4722 | 0.3097 (n=148) | 0.15 |
| retrans | summac | floor | 0.3201 | 0.1499 (n=108) | 0.3 |
| retrans | summac | z | 0.2419 | 0.0782 (n=148) | 0.2 |
| retrans | minicheck | raw | 0.3862 | 0.2208 (n=148) | 0.05 |
| retrans | minicheck | floor | 0.2565 | 0.1505 (n=146) | 0.2 |
| retrans | minicheck | z | 0.2365 | 0.1125 (n=148) | 0.05 |
| retrans | erasure | raw | 0.2731 | 0.2087 (n=145) | 0.05 |
| retrans | erasure_p | raw | 0.1301 | 0.0259 (n=58) | 0.0 |

## 판정

관문을 통과한 조합 (변이별):

- `oracle` × `nli` (raw) — SNR 5.74
- `oracle` × `nli` (z) — SNR 5.74
- `oracle` × `summac` (raw) — SNR 1.23
- `oracle` × `summac` (z) — SNR 1.23
- `oracle` × `minicheck` (raw) — SNR 1.57
- `oracle` × `minicheck` (floor) — SNR 1.57
- `oracle` × `minicheck` (z) — SNR 1.57
- `retrans` × `nli` (z) — SNR 5.88

### 실험 1 — SummaC 집계: **기각**

목적을 정반대로 달성했다. 바닥이 0.0302 → 0.2574 로 오르고 길이 기울기도 그대로다 (0.5065 → 0.0714). SNR 도 5.74 → 1.23 으로 떨어졌다.

원인은 집계 방식에 있다. **창을 여러 개 만들어 max 를 취하면 잡음의 최대값을 뽑는다** — hypothesis 가 짧을수록 창이 많아지므로, 하필 고치려던 축(짧은 방출분의 높은 바닥)에서 더 나빠진다. SummaC 의 원래 문제(문서 vs 문장)와 우리 문제 (문장 vs 조각)는 방향이 반대였다: 저쪽은 premise 가 너무 길어 희석되는 것이고, 이쪽은 hypothesis 가 너무 짧아 판단 근거가 없는 것이다.

### 실험 2 — MiniCheck: **통과하나 채택 보류**

raw·floor·z 세 변이 모두 0/6 으로, **바닥 보정 후에도 통과하는 유일한 백엔드**다 (현행 NLI 는 floor 변이에서 1/6 탈락). 그러나 세 가지가 걸린다.

1. **바닥이 높다** — 0.4586. 무해한 미완성에 절반 가까운 '미지원' 을 준다. `NLI_ALTERNATIVES.md` §2.2 에서 예고한 R3 우려가 그대로 나왔다: 사실검증기는 *미지원* 과 *반박* 을 안 나눈다.
2. 다만 **평탄하다** (0.6139 → 0.4845). 현행 NLI 는 낮지만 45배 기울어져 있다 (0.1353 → 0.003). **상수 오프셋은 경계 간 비교에서 상쇄되고 기울기는 앞쪽 경계를 구조적으로 벌한다** — 이 축만 보면 MiniCheck 가 낫다.
3. **결정적 결함은 어순 편향이다.** ko-en-p04 에서 `benign_reordered`(0.7895)를 같은 케이스의 `benign_incomplete`(0.0938)보다 8배 나쁘게 준다. ko→en 어순 단조화는 우리가 **장려하는** 분절 결과다. COMET consistency 를 버리고 NLI 로 간 이유가 정확히 이 편향이었다 (설계 §11.1).

격차도 얇다 — ja-ko-p05 에서 0.7985 vs 0.7196, 여유 0.079. 통과는 통과지만 SNR 1.57 이 그 얇음을 그대로 보여준다.

### 실험 3 — 재번역: **erasure 는 사망, premise 교체는 성공**

**표면형 erasure 는 못 쓴다.** 관문 4~6/6 위반이고, 엄격형(`erasure_p`)은 retrans 에서 전 케이스가 1.0 이다 — gtx 는 매번 처음부터 번역하므로 **어절 접두 보존이 사실상 관측되지 않는다.** 재번역 문헌의 Normalized Erasure 는 같은 디코더가 점진적으로 출력을 갱신하는 상황을 전제하는데, 우리는 매번 독립 호출이라 그 전제가 성립하지 않는다. 이식 실패.

**반면 premise 를 재번역으로 바꾼 것은 성공했다.**

- SNR **5.88** 로 현행(5.74)을 넘어 전 조합 중 1위
- 바닥이 더 낮고 더 평탄하다: 전체 0.0263 vs 0.0302, 특히 1–2어절에서 0.0874 vs 0.1353 (**35% 감소**). premise 와 hypothesis 의 길이 차가 줄어 granularity 불일치가 실제로 완화됐다 — SummaC 가 하려다 실패한 것을 premise 축에서 달성한 셈이다.
- **오라클이 필요 없다.** full 번역을 안 쓰므로 `reference_suspect` 오염이 구조적으로 사라진다. 비용은 경계당 gtx 1회 (실측 고유 712건/1003경계).

관문 raw 위반 1건은 **ja-ko-p05 하나뿐이고, 그건 영어 전용 `deberta-large-mnli` 를 한국어 타깃에 쓴 케이스다** — premise 축의 실패가 아니라 모델 언어 밖의 값이다. en 타깃 5케이스는 5/5 통과한다.

### 다음 할 일

1. `retrans × nli` 를 후보로 승격. 단 **채택 전 두 가지**: (a) ja-ko 케이스를 `mdeberta-xnli` 로 재검해 언어 밖 값이 맞는지 확인, (b) 실데이터에서 이 백엔드로 바꿨을 때 `paired_delta` 채택 판정이 뒤집히는지 확인 — 순위 상관 0.765/topk 겹침 0.7 은 '비슷하지만 같지 않다' 이고, 루프 결정이 바뀌는지는 따로 봐야 한다.
2. MiniCheck 는 **어순 편향 관문을 추가로 통과하기 전까지 보류**. `validity_cases.json` 의 `benign_paraphrase` 를 조각 단위로 옮긴 케이스가 필요하다.
3. SummaC 는 종결. 대신 현행 NLI 의 길이 바닥은 `noise_floor.py` 사후 보정을 계속 쓰거나, premise 축 교체(1번)로 줄인다.