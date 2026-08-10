# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon` (train 30 / dev 60 / test 100)
- 분절 모델 `claude-sonnet-5` / 판정자 `claude-sonnet-5` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: comet (보고용)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [2, 3, 4, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV — verbs/predicates consistently appear clause-finally, e.g. '식당 밥 먹다 보면은', '적성에 맞는지 안 맞는지부터 판단을 한 후에', with objects/complements preceding the verb throughout. / 측정: 공백비율 0.2723, 문말 부호 [',', '.', '?']
- score = T 격자 평균 adequacy (가중치·임계값 없음). 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | 채택 |
|---|---|---|---|---|
| 0 | 1.00 | 0.7846 | 0.7814 | O |
| 1 | 0.97 | -9.0333 | — | X |
| 2 | 0.97 | -9.0333 | — | X |
| 3 | 1.00 | 0.7811 | — | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | adequacy ↑ | consistency | k | shortfall |
|---|---|---|---|---|---|
| 2 | 3.86 | **0.7882** | 0.8380 | 3.02 | 3.70 |
| 3 | 3.92 | **0.7872** | 0.8424 | 2.95 | 1.55 |
| 4 | 4.04 | **0.7851** | 0.8522 | 2.78 | 0.58 |
| 6 | 4.57 | **0.7807** | 0.8753 | 2.33 | 0.14 |
| unsegmented (노브 없음) | 13.44 | **0.7719** | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 2.10 | **0.7159** | 0.5968 | 6.20 | — |

- 포맷 통과율 1.0000 (재시도 없이 1.0000), 위반 0건
- premature_rate (T=6, 부록 지표): **0.0556**

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다.

## 비용

- 호출 356회, 입력 토큰 1,321,470 (캐시 0), 출력 토큰 418,557
- 게이트웨이 추정 비용 6.8285