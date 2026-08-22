# 자동 분절 프롬프트 루프 결과 (v2) — German → English

- 데이터셋: `fleurs-de-en` (train 10 / dev 20 / test 20)
- 분절 모델 `gpt-5-mini` / 판정자 `gpt-5-mini` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [6, 12], 최종 격자 [4, 6, 8, 12], 주 작동점 T=12
- 언어 프로파일: German, 어순 SVO — main declarative clauses show subject before the finite verb (V2) as in 'Die Architektur befasst sich...', while subordinate clauses in the samples show verb-final order / 측정: 공백비율 0.1294, 문말 부호 [')', ',', '-', '.', ':', ';', '“']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **xlmr-anli** (`vicgalle/xlm-roberta-large-xnli-anli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7692 | 0.6835 | — | — | O |
| 1 | 1.00 | 0.7692 | — | — | — | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 4 | 2.29 | **0.6374** | 0.7340 | 0.0882 | 0.7512 | 4.89 | 0.00 |
| 6 | 3.70 | **0.7368** | 0.7633 | 0.0402 | 0.8527 | 3.22 | 0.00 |
| 8 | 4.39 | **0.7816** | 0.7908 | 0.0039 | 0.8258 | 2.56 | 0.00 |
| 12 | 5.54 | **0.7909** | 0.8002 | 0.0077 | 0.8391 | 2.17 | 0.00 |
| unsegmented (노브 없음) | 19.85 | **—** | 0.8472 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.47 | **0.3384** | 0.5469 | 0.3751 | 0.2823 | 19.15 | — |

- 포맷 통과율 0.9000 (재시도 없이 0.5000), 위반 2건
- premature_rate (T=12, 부록 지표, **무작위 표본**): **0.0000**
- reference_suspect_rate (T=12): 0.0000 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- **순위 격차 `rank_contra_gap` (T=4, 바닥 보정)**: **+0.0395** — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** (기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)
- 순위정렬 Spearman (T=4, raw): +0.2637 — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 (run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 71회, 입력 토큰 138,072 (캐시 75,776), 출력 토큰 240,749
- 게이트웨이 추정 비용 0.4990

| 용도 | 호출 | 비용 | 비중 | 사고 토큰/콜 |
|---|---|---|---|---|
| `segment_retry` | 20 | 0.2171 | 43.5% | 5,254 |
| `segment` | 10 | 0.1962 | 39.3% | 9,401 |
| `judge` | 36 | 0.0248 | 5.0% | 200 |
| `prompt_engineer` | 2 | 0.0235 | 4.7% | 2,016 |
| `prompt_v0` | 1 | 0.0192 | 3.8% | 6,720 |
| `critic` | 1 | 0.0107 | 2.1% | 3,200 |
| `profiler` | 1 | 0.0075 | 1.5% | 3,200 |