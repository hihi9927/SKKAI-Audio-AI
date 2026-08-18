# 자동 분절 프롬프트 루프 결과 (v2) — English → German

- 데이터셋: `fleurs-en-de` (train 40 / dev 60 / test 100)
- 분절 모델 `gpt-5.4-mini` / 판정자 `gpt-5.4-mini` / 번역기 `google:de:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [3, 4, 6], 주 작동점 T=6
- 언어 프로파일: English, 어순 SVO: the sample is dominated by subject-first clauses like "Travellers are advised" and "driver behavior cannot be predicted", with fronted subordinates like "Since..., Paraguay has..." and "After..., Gibson was...". / 측정: 공백비율 0.1596, 문말 부호 ["'", ')', ',', '-', '.', '/', ':', ';', '’', '”']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **mdeberta-xnli** (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7420 | 0.7487 | — | — | O |
| 1 | 0.97 | 0.7495 | 0.7259 | -0.02281 ±0.00719 | 57 | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 3 | 2.69 | **0.7116** | 0.7670 | 0.0726 | 0.8131 | 7.24 | 0.00 |
| 4 | 3.43 | **0.7442** | 0.7918 | 0.0601 | 0.8474 | 5.37 | 0.00 |
| 6 | 4.46 | **0.7626** | 0.8156 | 0.0646 | 0.8446 | 3.69 | 0.00 |
| unsegmented (노브 없음) | 21.75 | **—** | 0.8578 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.53 | **0.4401** | 0.5652 | 0.2206 | 0.4404 | 16.64 | — |

- 포맷 통과율 0.9900 (재시도 없이 0.8200), 위반 1건
- premature_rate (T=6, 부록 지표, **무작위 표본**): **0.0244**
- reference_suspect_rate (T=6): 0.0000 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- **순위 격차 `rank_contra_gap` (T=3, 바닥 보정)**: **-0.0018** — 순위 하위 절반 − 상위 절반의 경계 contradiction 차. 양수 = 절단이 실제로 위험을 덜어냄. **0 이하면 순위가 정보를 주지 않는다** (기준점이 0 인 것은 순위 무정보 시 기대값이 정확히 0 이기 때문 — 임의 상수 아님)
- 순위정렬 Spearman (T=3, raw): -0.0375 — 같은 축의 방향만 보는 보조값. **바닥 보정이 없어 음수 쪽으로 편향**된다 (run03: raw −0.25 → 보정 후 +0.14). 판정은 위의 gap 으로 한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 447회, 입력 토큰 636,119 (캐시 440,320), 출력 토큰 2,106,867
- 게이트웨이 추정 비용 9.6608

| 용도 | 호출 | 비용 | 비중 | 사고 토큰/콜 |
|---|---|---|---|---|
| `segment` | 300 | 7.8854 | 81.6% | 5,639 |
| `segment_retry` | 50 | 1.4309 | 14.8% | 6,114 |
| `judge` | 93 | 0.1966 | 2.0% | 292 |
| `critic` | 1 | 0.0537 | 0.6% | 10,044 |
| `prompt_v0` | 1 | 0.0497 | 0.5% | 9,198 |
| `prompt_engineer` | 1 | 0.0312 | 0.3% | 4,224 |
| `profiler` | 1 | 0.0131 | 0.1% | 2,291 |