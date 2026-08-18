# NLI → semantic similarity 대체 가능성 측정

런: `ko-en/run04` · 경계 1003개 · 바닥 측정 문장 150개 · LLM 호출 0

임베딩 후보는 MTEB/MMTEB 의 **STS·pair-classification** 축에서 골랐다 — 여기서 재는 것은 검색 적합도가 아니라 두 문장의 의미 동치이고, 타깃 언어가 영어가 아닐 수 있어 다국어가 요건이다.

| 키 | 모델 |
|---|---|
| `e5-inst` | `intfloat/multilingual-e5-large-instruct` |
| `qwen3-06b` | `Qwen/Qwen3-Embedding-0.6B` |
| `qwen3-4b` | `Qwen/Qwen3-Embedding-4B` |
| `gte-base` | `Alibaba-NLP/gte-multilingual-base` |

## T1 — contradiction 관문 (`premature_cases.json`)

통과 조건은 `judge_check.check_nli` 와 같다: 케이스마다 `min(premature) > max(safe)`. `floor` 는 잡음 바닥 보정, `z` 는 바닥 대비 표준화. 바닥은 아래 T3 에서 잰 값을 쓴다.

| 백엔드 | 변이 | 위반/케이스 | mean(premature) | mean(safe) | 격차 | 판정 |
|---|---|---|---|---|---|---|
| deberta-mnli | raw | 0/6 | 0.6264 | 0.0318 | 0.5946 | 통과 |
| deberta-mnli | floor | 1/6 | 0.5955 | 0.0194 | 0.5761 | **탈락** |
| deberta-mnli | z | 0/6 | 10.0223 | -0.0379 | 10.0602 | 통과 |
| embed:e5-inst | raw | 5/6 | 0.1222 | 0.1303 | -0.0081 | **탈락** |
| embed:e5-inst | floor | 5/6 | 0.0009 | 0.0045 | -0.0036 | **탈락** |
| embed:e5-inst | z | 3/6 | -0.8457 | -1.3794 | 0.5338 | **탈락** |
| embed:e5-inst+align | raw | 3/6 | 0.1146 | 0.1084 | 0.0062 | **탈락** |
| embed:qwen3-06b | raw | 5/6 | 0.236 | 0.3309 | -0.0949 | **탈락** |
| embed:qwen3-06b | floor | 6/6 | 0.0 | 0.0163 | -0.0163 | **탈락** |
| embed:qwen3-06b | z | 3/6 | -1.3903 | -1.398 | 0.0077 | **탈락** |
| embed:qwen3-06b+align | raw | 3/6 | 0.229 | 0.2697 | -0.0407 | **탈락** |
| embed:qwen3-4b | raw | 4/6 | 0.2516 | 0.335 | -0.0834 | **탈락** |
| embed:qwen3-4b | floor | 6/6 | 0.0 | 0.0255 | -0.0255 | **탈락** |
| embed:qwen3-4b | z | 2/6 | -1.363 | -1.4919 | 0.1289 | **탈락** |
| embed:qwen3-4b+align | raw | 2/6 | 0.2501 | 0.2874 | -0.0372 | **탈락** |
| embed:gte-base | raw | 5/6 | 0.2355 | 0.2587 | -0.0232 | **탈락** |
| embed:gte-base | floor | 5/6 | 0.0085 | 0.0181 | -0.0096 | **탈락** |
| embed:gte-base | z | 3/6 | -0.7748 | -1.1889 | 0.414 | **탈락** |
| embed:gte-base+align | raw | 3/6 | 0.2061 | 0.2022 | 0.0039 | **탈락** |

### 케이스별 원점수 (raw, 높을수록 '모순')

| 케이스 | 변이 | 기대 | deberta-mnli | embed:e5-inst | embed:e5-inst+align | embed:qwen3-06b | embed:qwen3-06b+align | embed:qwen3-4b | embed:qwen3-4b+align | embed:gte-base | embed:gte-base+align |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ko-en-p01 | **premature_negation** | premature | 0.9971 | 0.1910 | 0.1839 | 0.4067 | 0.4067 | 0.4422 | 0.4422 | 0.4033 | 0.3776 |
| ko-en-p01 | safe_boundary | safe | 0.0221 | 0.2132 | 0.1918 | 0.6526 | 0.5584 | 0.7235 | 0.6715 | 0.4984 | 0.4265 |
| ko-en-p02 | **premature_role** | premature | 0.9958 | 0.0921 | 0.0921 | 0.1826 | 0.1826 | 0.2015 | 0.2015 | 0.1269 | 0.1269 |
| ko-en-p02 | benign_incomplete | safe | 0.1565 | 0.1095 | 0.1095 | 0.2172 | 0.2172 | 0.1953 | 0.1953 | 0.1440 | 0.1440 |
| ko-en-p03 | **premature_scope** | premature | 0.0410 | 0.1626 | 0.1626 | 0.3830 | 0.3830 | 0.3750 | 0.3750 | 0.2694 | 0.2694 |
| ko-en-p03 | benign_incomplete | safe | 0.0049 | 0.1998 | 0.1280 | 0.5058 | 0.3792 | 0.4808 | 0.3310 | 0.3653 | 0.2170 |
| ko-en-p04 | **premature_head** | premature | 0.5227 | 0.1085 | 0.1085 | 0.2060 | 0.2060 | 0.2369 | 0.2369 | 0.2141 | 0.2141 |
| ko-en-p04 | benign_incomplete | safe | 0.0011 | 0.0831 | 0.0831 | 0.1754 | 0.1754 | 0.2022 | 0.2022 | 0.1528 | 0.1528 |
| ko-en-p04 | benign_reordered | safe | 0.0008 | 0.0895 | 0.0895 | 0.1632 | 0.1632 | 0.1927 | 0.1927 | 0.1597 | 0.1597 |
| ko-en-p06 | benign_incomplete | safe | 0.0283 | 0.1569 | 0.1569 | 0.3942 | 0.3942 | 0.4188 | 0.4188 | 0.3157 | 0.3157 |
| ko-en-p06 | **premature_modal** | premature | 0.9921 | 0.1277 | 0.1277 | 0.1665 | 0.1665 | 0.2292 | 0.2292 | 0.2311 | 0.2311 |
| ja-ko-p05 | **premature_negation** | premature | 0.2098 | 0.0512 | 0.0130 | 0.0711 | 0.0292 | 0.0246 | 0.0160 | 0.1681 | 0.0177 |
| ja-ko-p05 | benign_incomplete | safe | 0.0089 | 0.0600 | 0.0000 | 0.2076 | 0.0000 | 0.1315 | 0.0000 | 0.1751 | 0.0000 |

## T2 — consistency 관문 (`validity_cases.json`)

통과 조건은 `validity_check` 와 같다: 심각한 의미 오류 < `benign_minimal`.

| 백엔드 | 순위 검사 | 위반 (minimal 기준) | 참고 (paraphrase 기준) | 판정 |
|---|---|---|---|---|
| embed:e5-inst | 17 | 4 | 8 | **탈락** |
| embed:qwen3-06b | 17 | 6 | 11 | **탈락** |
| embed:qwen3-4b | 17 | 9 | 12 | **탈락** |
| embed:gte-base | 17 | 3 | 11 | **탈락** |

변이 유형별 평균:

| 변이 | embed:e5-inst | embed:qwen3-06b | embed:qwen3-4b | embed:gte-base |
|---|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9869 | 0.9255 | 0.9218 | 0.9763 |
| benign_paraphrase | 0.9691 | 0.9103 | 0.8883 | 0.9246 |
| **negation_flip** | 0.9499 | 0.9656 | 0.9680 | 0.8827 |
| **role_swap** | 0.9970 | 0.9612 | 0.9568 | 0.9950 |
| **clause_omission** | 0.9411 | 0.8575 | 0.8715 | 0.9077 |
| **referent_loss** | 0.9680 | 0.9119 | 0.8957 | 0.9473 |
| unrelated | 0.6783 | 0.1909 | 0.2220 | 0.3487 |

## T3 — 잡음 바닥 (full 번역의 자기-prefix. 정의상 무해한 미완성)

prefix 는 같은 번역의 앞부분이라 **모순일 수 없다**. 여기서 나오는 점수가 바닥이고, 실제 경계 점수에서 이걸 빼야 '잘못 잘라서 받은 벌점'만 남는다.

| 백엔드 | 전체 mean | sd | 1-2 | 3-4 | 5-6 | 7-9 | 10-14 | 15+ |
|---|---|---|---|---|---|---|---|---|
| deberta-mnli | 0.0302 | 0.1036 | 0.1353 | 0.0479 | 0.0204 | 0.0078 | 0.0063 | 0.003 |
| embed:e5-inst | 0.1171 | 0.0743 | 0.2294 | 0.1907 | 0.145 | 0.1031 | 0.0641 | 0.044 |
| embed:e5-inst+align | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| embed:qwen3-06b | 0.326 | 0.2336 | 0.6855 | 0.5383 | 0.3989 | 0.2752 | 0.1662 | 0.1167 |
| embed:qwen3-06b+align | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| embed:qwen3-4b | 0.3331 | 0.234 | 0.698 | 0.545 | 0.4017 | 0.2805 | 0.1741 | 0.1238 |
| embed:qwen3-4b+align | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| embed:gte-base | 0.2269 | 0.1519 | 0.4666 | 0.3717 | 0.2757 | 0.1948 | 0.1211 | 0.081 |
| embed:gte-base+align | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

(길이 버킷 = hypothesis 어절 수)

### 신호 대 바닥 (SNR)

신호 = 관문의 `mean(premature) − mean(safe)` (raw 기준). 바닥 산포 = 자기-prefix 점수의 전체 sd. **SNR 이 1 미만이면 실제 잘못 자른 경계가 무해한 미완성의 산포에 묻힌다.**

| 백엔드 | 신호 | 바닥 sd | SNR |
|---|---|---|---|
| deberta-mnli | 0.5946 | 0.1036 | 5.74 |
| embed:e5-inst | -0.0081 | 0.0743 | -0.11 |
| embed:e5-inst+align | 0.0062 | 0.0 | None |
| embed:qwen3-06b | -0.0949 | 0.2336 | -0.41 |
| embed:qwen3-06b+align | -0.0407 | 0.0 | None |
| embed:qwen3-4b | -0.0834 | 0.234 | -0.36 |
| embed:qwen3-4b+align | -0.0372 | 0.0001 | -372.0 |
| embed:gte-base | -0.0232 | 0.1519 | -0.15 |
| embed:gte-base+align | 0.0039 | 0.0 | None |

### 실데이터 경계에서 NLI 와의 일치도

경계 1003개 (`pieces_contra` 재활용 — NLI 재계산 없음). `topk_overlap` = 루프가 판정자에게 보내는 contradiction 최상위 경계 집합의 겹침 비율. **조향이 바뀌는가**의 직접 측정값이다.

| 백엔드 | 변이 | Spearman(전역) | Spearman(문장 내) | topk 겹침 | 문장 effective Spearman |
|---|---|---|---|---|---|
| embed:e5-inst | raw | 0.392 | 0.2946 (n=148) | 0.0 | 0.3218 |
| embed:e5-inst | floor | 0.1006 | 0.0674 (n=139) | 0.1 | 0.3307 |
| embed:e5-inst | z | 0.0901 | 0.0204 (n=148) | 0.0 | 0.2689 |
| embed:e5-inst+align | raw | 0.4082 | 0.3201 (n=147) | 0.0 | 0.4084 |
| embed:qwen3-06b | raw | 0.3252 | 0.3149 (n=148) | 0.0 | 0.2777 |
| embed:qwen3-06b | floor | 0.0317 | 0.072 (n=138) | 0.05 | 0.2462 |
| embed:qwen3-06b | z | 0.0317 | 0.0394 (n=148) | 0.0 | 0.228 |
| embed:qwen3-06b+align | raw | 0.3211 | 0.3293 (n=148) | 0.0 | 0.3037 |
| embed:qwen3-4b | raw | 0.3255 | 0.2625 (n=148) | 0.0 | 0.2802 |
| embed:qwen3-4b | floor | 0.0307 | 0.0759 (n=139) | 0.1 | 0.2834 |
| embed:qwen3-4b | z | 0.0428 | 0.0634 (n=148) | 0.0 | 0.2745 |
| embed:qwen3-4b+align | raw | 0.2858 | 0.272 (n=148) | 0.05 | 0.2831 |
| embed:gte-base | raw | 0.3866 | 0.3087 (n=148) | 0.0 | 0.2857 |
| embed:gte-base | floor | 0.0908 | 0.1075 (n=144) | 0.1 | 0.2864 |
| embed:gte-base | z | 0.0696 | 0.1317 (n=148) | 0.0 | 0.3027 |
| embed:gte-base+align | raw | 0.4136 | 0.3181 (n=148) | 0.05 | 0.3936 |

## 판정

- `deberta-mnli` contradiction 관문 최소 위반 0건 (raw=0, floor=1, z=0)
- `embed:e5-inst` contradiction 관문 최소 위반 3건 (raw=5, floor=5, z=3)
- `embed:e5-inst+align` contradiction 관문 최소 위반 3건 (raw=3, floor=3, z=3)
- `embed:qwen3-06b` contradiction 관문 최소 위반 3건 (raw=5, floor=6, z=3)
- `embed:qwen3-06b+align` contradiction 관문 최소 위반 3건 (raw=3, floor=3, z=3)
- `embed:qwen3-4b` contradiction 관문 최소 위반 2건 (raw=4, floor=6, z=2)
- `embed:qwen3-4b+align` contradiction 관문 최소 위반 2건 (raw=2, floor=2, z=2)
- `embed:gte-base` contradiction 관문 최소 위반 3건 (raw=5, floor=5, z=3)
- `embed:gte-base+align` contradiction 관문 최소 위반 3건 (raw=3, floor=3, z=3)
- `embed:e5-inst` consistency 관문 위반 4건 (soft 8건)
- `embed:qwen3-06b` consistency 관문 위반 6건 (soft 11건)
- `embed:qwen3-4b` consistency 관문 위반 9건 (soft 12건)
- `embed:gte-base` consistency 관문 위반 3건 (soft 11건)

### 결론

**임베딩 유사도로 contradiction 을 대체할 수 없다.** 통과한 임베딩 구성이 하나도 없고, 관문 신호(`mean(premature) − mean(safe)`)가 여러 구성에서 **음수**다 — 잘못 자른 방출이 안전한 방출보다 full 번역에 *더* 가깝게 나온다.

원인은 캘리브레이션이 아니라 도구의 성질이다. 코사인 유사도는 **대칭**이고 표면 의미의 근접도만 재는데, 우리가 잡아야 하는 오류(부정 뒤집힘·주체 뒤바뀜)는 **어휘를 그대로 둔 채 명제만 뒤집는다**. 그래서 오히려 정직한 파편보다 참조와 가까워진다 (케이스별 원점수 표에서 직접 보인다). 길이 바닥을 구조적으로 없앤 `+align` 구성도 부호를 되돌리지 못했다 — 문제는 길이 교란이 아니라 명제 관계를 못 본다는 것이다.

NLI 는 같은 케이스에서 정확히 이 축을 본다: `neutral`(미완성)과 `contradiction`(반박)을 나누므로, 무해한 미완성은 바닥에 두고 뒤집힌 방출만 1.0 쪽으로 보낸다.

**대안이 필요하다면** 대칭 유사도가 아니라 방향성 있는 것을 찾아야 한다 — 다른 NLI 체크포인트(`metrics.NLI_MODELS`), cross-encoder 계열, 또는 참조 기반 QE. 임베딩은 `consistency` 보조 지표로도 T2 를 통과하지 못했다.