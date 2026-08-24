# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon-train` (train 60 / dev 150 / test 150)
- 분절 모델 `claude-sonnet-5` / 판정자 `claude-sonnet-5` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: nli (보고용, 양방향 entailment 의 min — 어순 무관)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [2, 3, 4, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV — verbs/predicates consistently appear clause-finally, e.g. '수연이가 뭘 좋아하는지 모르는데' (object before verb '모르는데'). / 측정: 공백비율 0.2745, 문말 부호 [',', '.', '?']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **deberta-mnli** (`microsoft/deberta-large-mnli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 0.93 | 0.0000 | 0.6406 | — | — | O |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| unsegmented (노브 없음) | 14.41 | **—** | 0.7497 | — | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 1.99 | **0.5565** | 0.7032 | 0.2078 | 0.3034 | 6.61 | — |

- 포맷 통과율 0.9467 (재시도 없이 0.6400), 위반 15건
- premature_rate (T=6, 부록 지표, **무작위 표본**): 미측정
- reference_suspect_rate (T=6): 미측정 — 높으면 오라클(full 번역)을 의심할 것. contradiction·consistency 가 오염된다
- 순위정렬 Spearman (T=2): 미측정 — 양수 = 모델 순위가 실측 위험과 정렬. 음수면 절단이 위험을 줄이지 못한다

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다. `contradiction` 은 경계 (k−1)개의 평균이라 **무분절에는 정의되지 않는다**(—) — 무분절은 곡선의 점이 아니라 offline 기준선으로 읽을 것.

## 비용

- 호출 1회, 입력 토큰 5,383 (캐시 0), 출력 토큰 1,748
- 게이트웨이 추정 비용 0.0282