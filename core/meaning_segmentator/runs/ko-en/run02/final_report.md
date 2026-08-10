# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon` (train 30 / dev 60 / test 100)
- 분절 모델 `claude-sonnet-5` / 판정자 `claude-sonnet-5` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: comet (보고용)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [2, 3, 4, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV — verbs/predicates consistently appear clause-finally, e.g. '식당 밥 먹다 보면은', '적성에 맞는지 안 맞는지부터 판단을 한 후에', object/complement precedes the verb throughout. / 측정: 공백비율 0.2723, 문말 부호 [',', '.', '?']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **deberta-mnli** (`microsoft/deberta-large-mnli`)
- 채택된 프롬프트: iter_03

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7697 | 0.7449 | — | — | O |
| 1 | 1.00 | 0.7673 | — | — | — | X |
| 2 | 1.00 | 0.7537 | — | — | — | X |
| 3 | 1.00 | 0.7726 | 0.7499 | +0.00499 ±0.00664 | 19 | O |
| 4 | 1.00 | 0.7692 | — | — | — | X |
| 5 | 1.00 | 0.7705 | — | — | — | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | shortfall |
|---|---|---|---|---|---|---|---|
| 2 | 3.02 | **0.7416** | 0.7788 | 0.0482 | 0.7839 | 3.54 | 3.18 |
| 3 | 3.09 | **0.7437** | 0.7784 | 0.0450 | 0.7938 | 3.42 | 1.08 |
| 4 | 3.23 | **0.7484** | 0.7772 | 0.0377 | 0.8172 | 3.03 | 0.33 |
| 6 | 3.52 | **0.7503** | 0.7747 | 0.0323 | 0.8389 | 2.44 | 0.03 |
| unsegmented (노브 없음) | 13.44 | **0.7719** | 0.7719 | 0.0000 | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 2.10 | **0.5850** | 0.7159 | 0.1849 | 0.5968 | 6.20 | — |

- 포맷 통과율 1.0000 (재시도 없이 1.0000), 위반 0건
- premature_rate (T=6, 부록 지표): **0.1667**

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다.

## 비용

- 호출 506회, 입력 토큰 1,802,078 (캐시 0), 출력 토큰 739,507
- 게이트웨이 추정 비용 10.9992