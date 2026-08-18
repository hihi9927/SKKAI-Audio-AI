# 자동 분절 프롬프트 루프 결과 (v2) — English → German

- 데이터셋: `fleurs-en-de` (train 40 / dev 60 / test 100)
- 분절 모델 `gpt-5-mini` / 판정자 `gpt-5-mini` / 번역기 `google:de:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [6, 12], 최종 격자 [6, 8, 12], 주 작동점 T=6
- 언어 프로파일: English, 어순 SVO | Subjects precede verbs and objects follow verbs in the samples (e.g. "Travellers are strongly advised...", "He has been unable to take the drugs...") / 측정: 공백비율 0.1596, 문말 부호 ["'", ')', ',', '-', '.', '/', ':', ';', '’', '”']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **mdeberta-xnli** (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7925 | 0.7785 | — | — | O |
| 1 | 1.00 | 0.7844 | — | — | — | X |
| 2 | 1.00 | 0.7949 | 0.7714 | -0.00710 ±0.00893 | 37 | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 6 | 4.58 | **0.7706** | 0.8166 | 0.0564 | 0.8324 | 3.69 | 0.00 |
| 8 | 5.53 | **0.7822** | 0.8293 | 0.0569 | 0.8529 | 2.81 | 0.00 |
| 12 | 6.63 | **0.7972** | 0.8374 | 0.0481 | 0.8920 | 2.14 | 0.00 |
| unsegmented (노브 없음) | 21.75 | **—** | 0.8580 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.52 | **0.4392** | 0.5653 | 0.2222 | 0.4423 | 16.64 | — |

- 포맷 통과율 0.9800 (재시도 없이 0.9700), 위반 2건
- premature_rate (T=6, 부록 지표, **무작위 표본**): **0.0000**
- reference_suspect_rate (T=6): 0.0000 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- **순위 격차 `rank_contra_gap` (T=6, 바닥 보정)**: **-0.0067** — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** (기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)
- 순위정렬 Spearman (T=6, raw): +0.0437 — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 (run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 477회, 입력 토큰 1,195,282 (캐시 990,720), 출력 토큰 1,938,402
- 게이트웨이 추정 비용 3.9527

| 용도 | 호출 | 비용 | 비중 | 사고 토큰/콜 |
|---|---|---|---|---|
| `segment` | 340 | 3.5331 | 89.4% | 5,013 |
| `judge` | 123 | 0.2532 | 6.4% | 865 |
| `segment_retry` | 9 | 0.1027 | 2.6% | 5,482 |
| `prompt_v0` | 1 | 0.0234 | 0.6% | 8,448 |
| `prompt_engineer` | 2 | 0.0218 | 0.6% | 1,088 |
| `critic` | 1 | 0.0122 | 0.3% | 3,968 |
| `profiler` | 1 | 0.0064 | 0.2% | 2,432 |