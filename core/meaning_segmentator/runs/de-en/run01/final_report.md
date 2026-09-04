# 자동 분절 프롬프트 루프 결과 (v2) — German → English

- 데이터셋: `fleurs-de-en` (train 30 / dev 60 / test 100)
- 분절 모델 `gpt-5-mini` / 판정자 `gpt-5-mini` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [6, 12], 최종 격자 [4, 6, 8, 12], 주 작동점 T=12
- 언어 프로파일: German, 어순 SOV | many subordinate clauses in the sample show verb-final order (e.g. "der ... gegangen war", "auf dem die Entscheidung ... beruhe"), while main clauses show verb-second patterns. / 측정: 공백비율 0.1343, 문말 부호 ["'", ')', ',', '-', '.', ':', ';', '“']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **xlmr-anli** (`vicgalle/xlm-roberta-large-xnli-anli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 0.97 | 0.6960 | 0.7411 | — | — | O |
| 1 | 0.90 | 0.7238 | 0.7406 | -0.00061 ±0.01196 | 50 | X |
| 2 | 0.90 | 0.7238 | 0.7406 | -0.00061 ±0.01196 | 50 | X |
| 3 | 1.00 | 0.6834 | — | — | — | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 4 | 2.79 | **0.6513** | 0.7217 | 0.0960 | 0.7616 | 5.15 | 0.00 |
| 6 | 4.05 | **0.6983** | 0.7747 | 0.0999 | 0.8234 | 3.47 | 0.00 |
| 8 | 5.18 | **0.7251** | 0.7994 | 0.0893 | 0.8379 | 2.67 | 0.00 |
| 12 | 6.14 | **0.7797** | 0.8151 | 0.0438 | 0.8892 | 2.10 | 0.00 |
| unsegmented (노브 없음) | 20.69 | **—** | 0.8493 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.47 | **0.3863** | 0.5536 | 0.3011 | 0.2839 | 19.01 | — |

- 포맷 통과율 0.9600 (재시도 없이 0.6400), 위반 4건
- premature_rate (T=12, 부록 지표, **무작위 표본**): **0.0000**
- reference_suspect_rate (T=12): 0.0000 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- **순위 격차 `rank_contra_gap` (T=4, 바닥 보정)**: **-0.0349** — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** (기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)
- 순위정렬 Spearman (T=4, raw): -0.0033 — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 (run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 288회, 입력 토큰 772,643 (캐시 523,008), 출력 토큰 2,033,359
- 게이트웨이 추정 비용 4.1422

| 용도 | 호출 | 비용 | 비중 | 사고 토큰/콜 |
|---|---|---|---|---|
| `segment_retry` | 158 | 2.1116 | 51.0% | 6,487 |
| `segment` | 64 | 1.7934 | 43.3% | 13,531 |
| `prompt_engineer` | 10 | 0.1179 | 2.8% | 2,553 |
| `judge` | 53 | 0.0722 | 1.7% | 530 |
| `prompt_v0` | 1 | 0.0264 | 0.6% | 10,432 |
| `critic` | 1 | 0.0128 | 0.3% | 4,544 |
| `profiler` | 1 | 0.0079 | 0.2% | 3,264 |