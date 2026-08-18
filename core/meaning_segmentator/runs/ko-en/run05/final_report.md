# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon-train` (train 60 / dev 150 / test 150)
- 분절 모델 `gpt-5-mini` / 판정자 `gpt-5-mini` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [2, 3, 4, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV | verbs/auxiliaries and predicates consistently appear at the end of clauses in the sample (e.g. '못 먹어.', '싫대.', '가기로 했는데'). / 측정: 공백비율 0.2745, 문말 부호 [',', '.', '?']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **deberta-mnli** (`microsoft/deberta-large-mnli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 0.98 | 0.6995 | 0.6751 | — | — | O |
| 1 | 0.95 | 0.6861 | — | — | — | X |
| 2 | 1.00 | 0.6894 | — | — | — | X |
| 3 | 0.98 | 0.6765 | — | — | — | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 2 | 2.03 | **0.6186** | 0.7448 | 0.1696 | 0.4948 | 7.13 | 0.00 |
| 3 | 2.69 | **0.6482** | 0.7453 | 0.1309 | 0.5738 | 4.80 | 0.00 |
| 4 | 3.41 | **0.6656** | 0.7511 | 0.1144 | 0.6427 | 3.55 | 0.00 |
| 6 | 4.17 | **0.6934** | 0.7520 | 0.0790 | 0.7163 | 2.59 | 0.00 |
| unsegmented (노브 없음) | 14.41 | **—** | 0.7497 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.99 | **0.5565** | 0.7032 | 0.2078 | 0.3034 | 6.61 | — |

- 포맷 통과율 0.9800 (재시도 없이 0.9400), 위반 3건
- premature_rate (T=6, 부록 지표, **무작위 표본**): **0.0870**
- reference_suspect_rate (T=6): 0.0000 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- **순위 격차 `rank_contra_gap` (T=2, 바닥 보정)**: **+0.0166** — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** (기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)
- 순위정렬 Spearman (T=2, raw): +0.1426 — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 (run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 182회, 입력 토큰 429,785 (캐시 384,640), 출력 토큰 211,417
- 게이트웨이 추정 비용 0.4437