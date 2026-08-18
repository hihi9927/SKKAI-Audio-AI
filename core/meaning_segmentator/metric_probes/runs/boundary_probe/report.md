# 어절 점진 추가 — 표현 급변점이 의미 분절 지점인가

런: `ko-en/run04` · 문장×T 410개 · LLM 경계 1003개 · LLM 호출 0 · 번역 호출 0

여기 나오는 값은 **벌점이 아니라 경계 제안**이다. 그래서 경쟁 상대가 `contradiction` 이 아니라 **A2 Segmenter(LLM 프롬프트)** 이고, 판정 기준도 관문이 아니라 동의율이다. LLM 경계는 정답이 아니라 비교 대상이다 — 우리가 개선하려는 대상이 바로 그 프롬프트이기 때문이다.

점수마다 문장의 LLM 경계와 **같은 개수**를 뽑으므로 정밀도=재현율=F1 이다.

## 일치율 (정확 일치 / ±1 어절)

| 인코더 | 점수 | 정확 | ±1 |
|---|---|---|---|
| multilingual-e5-large@L-1 | delta_prefix | 0.4925 | 0.8046 |
| multilingual-e5-large@L-1 | delta_prefix_prior | 0.5414 | 0.8285 |
| multilingual-e5-large@L-1 | delta_prefix_resid | 0.2822 | 0.5284 |
| multilingual-e5-large@L-1 | tile | 0.337 | 0.5244 |
| multilingual-e5-large@L-1 | tile_prior | 0.4397 | 0.668 |
| multilingual-e5-large@L-1 | tile_resid | 0.2692 | 0.4526 |
| multilingual-e5-large@L-1 | ctx_delta | 0.5025 | 0.8415 |
| multilingual-e5-large@L-1 | ctx_delta_prior | 0.5414 | 0.8285 |
| multilingual-e5-large@L-1 | ctx_delta_resid | 0.2692 | 0.4875 |
| xlm-roberta-large@L-1 | delta_prefix | 0.4756 | 0.8136 |
| xlm-roberta-large@L-1 | delta_prefix_prior | 0.5414 | 0.8285 |
| xlm-roberta-large@L-1 | delta_prefix_resid | 0.2263 | 0.4536 |
| xlm-roberta-large@L-1 | tile | 0.3141 | 0.5145 |
| xlm-roberta-large@L-1 | tile_prior | 0.348 | 0.5055 |
| xlm-roberta-large@L-1 | tile_resid | 0.2453 | 0.4367 |
| xlm-roberta-large@L-1 | ctx_delta | 0.4756 | 0.8126 |
| xlm-roberta-large@L-1 | ctx_delta_prior | 0.5414 | 0.8285 |
| xlm-roberta-large@L-1 | ctx_delta_resid | 0.2393 | 0.4606 |

| 비교군 | 정확 | ±1 |
|---|---|---|
| 무작위 위치 | 0.2347 | 0.4428 |
| 기계적 등분 | 0.1974 | 0.4337 |

## 분포 전체 — 경계에서 실제로 급변하는가

상위 k 개만 보는 일치율은 '거의 맞았다' 를 못 본다. `AUC` 는 무작위로 고른 (경계, 비경계) 한 쌍에서 경계 쪽 점수가 높을 확률이다. **0.5 = 무정보.**

| 인코더 | 점수 | AUC | 경계−비경계 평균차 |
|---|---|---|---|
| multilingual-e5-large@L-1 | delta_prefix | 0.7934 | 0.0392 |
| multilingual-e5-large@L-1 | delta_prefix_prior | 0.8641 | 0.0337 |
| multilingual-e5-large@L-1 | delta_prefix_resid | 0.4943 | 0.0055 |
| multilingual-e5-large@L-1 | tile | 0.6465 | 0.0149 |
| multilingual-e5-large@L-1 | tile_prior | 0.7341 | 0.0104 |
| multilingual-e5-large@L-1 | tile_resid | 0.5422 | 0.0045 |
| multilingual-e5-large@L-1 | ctx_delta | 0.8148 | 0.0049 |
| multilingual-e5-large@L-1 | ctx_delta_prior | 0.8641 | 0.0035 |
| multilingual-e5-large@L-1 | ctx_delta_resid | 0.451 | 0.0015 |
| xlm-roberta-large@L-1 | delta_prefix | 0.7974 | 0.004 |
| xlm-roberta-large@L-1 | delta_prefix_prior | 0.8641 | 0.0031 |
| xlm-roberta-large@L-1 | delta_prefix_resid | 0.4333 | 0.0008 |
| xlm-roberta-large@L-1 | tile | 0.6102 | 0.0024 |
| xlm-roberta-large@L-1 | tile_prior | 0.6852 | 0.0018 |
| xlm-roberta-large@L-1 | tile_resid | 0.5271 | 0.0006 |
| xlm-roberta-large@L-1 | ctx_delta | 0.7926 | 0.0021 |
| xlm-roberta-large@L-1 | ctx_delta_prior | 0.8641 | 0.0014 |
| xlm-roberta-large@L-1 | ctx_delta_resid | 0.4189 | 0.0007 |

## 판정 — 의미인가 위치인가

`prior` 는 **문장 내용을 하나도 안 보고** 상대 위치의 코퍼스 평균만으로 매긴 점수다. raw 가 prior 를 못 넘으면 그 점수가 잡은 것은 의미가 아니라 위치다.

| 인코더 | 점수 | AUC(raw) | AUC(prior) | AUC(resid) | 일치(raw) | 일치(prior) |
|---|---|---|---|---|---|---|
| multilingual-e5-large@L-1 | delta_prefix | 0.7934 | 0.8641 | 0.4943 | 0.4925 | 0.5414 |
| multilingual-e5-large@L-1 | tile | 0.6465 | 0.7341 | 0.5422 | 0.337 | 0.4397 |
| multilingual-e5-large@L-1 | ctx_delta | 0.8148 | 0.8641 | 0.451 | 0.5025 | 0.5414 |
| xlm-roberta-large@L-1 | delta_prefix | 0.7974 | 0.8641 | 0.4333 | 0.4756 | 0.5414 |
| xlm-roberta-large@L-1 | tile | 0.6102 | 0.6852 | 0.5271 | 0.3141 | 0.348 |
| xlm-roberta-large@L-1 | ctx_delta | 0.7926 | 0.8641 | 0.4189 | 0.4756 | 0.5414 |

위치 사전확률 대비 최대 AUC 상승폭 **-0.049**, 위치 보정 후 최고 AUC **0.542**.

**급변점이 잡은 것은 의미가 아니라 위치다.** 문장 내용을 전혀 안 쓰는 위치 사전확률이 같은 성적을 내고, 위치 교란을 빼면 AUC 가 0.5(무정보)로 내려앉는다. `delta_prefix`·`ctx_delta` 는 prefix 가 길수록 한 어절의 비중이 줄어 단조 감소하는데, 그 감소 곡선이 LLM 경계의 위치 분포와 겹쳤을 뿐이다.

즉 '표현이 급변하는 곳'과 '의미 단위가 끝나는 곳'은 이 데이터에서 같은 지점이 아니다. 이 접근은 여기서 접는 것이 맞다.