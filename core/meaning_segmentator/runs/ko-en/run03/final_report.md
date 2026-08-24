# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon` (train 30 / dev 60 / test 100)
- 분절 모델 `claude-sonnet-5` / 판정자 `claude-sonnet-5` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: comet (보고용)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [2, 3, 4, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV — verbs/predicates consistently appear clause-finally (e.g. '맞는 거 같애', '판단을 한 후에', '돌아온거야'), with objects/complements preceding them. / 측정: 공백비율 0.2723, 문말 부호 [',', '.', '?']
- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **deberta-mnli** (`microsoft/deberta-large-mnli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7431 | 0.7346 | — | — | O |
| 1 | 0.97 | 0.7458 | 0.7322 | -0.01307 ±0.01377 | 44 | X |
| 2 | 0.97 | 0.7285 | — | — | — | X |
| 3 | 1.00 | 0.7531 | 0.7288 | -0.00427 ±0.00593 | 42 | X |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 2 | 2.03 | **0.6733** | 0.7708 | 0.1277 | 0.6642 | 6.55 | 0.13 |
| 3 | 2.31 | **0.7085** | 0.7767 | 0.0892 | 0.7345 | 4.47 | 0.00 |
| 4 | 2.73 | **0.7288** | 0.7755 | 0.0616 | 0.7893 | 3.33 | 0.00 |
| 6 | 3.35 | **0.7398** | 0.7696 | 0.0397 | 0.8254 | 2.45 | 0.00 |
| unsegmented (노브 없음) | 13.44 | **0.7721** | 0.7721 | 0.0000 | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 2.10 | **0.5843** | 0.7158 | 0.1859 | 0.5970 | 6.20 | — |

- 포맷 통과율 0.9100 (재시도 없이 0.3400), 위반 10건
- premature_rate (T=6, 부록 지표): **0.2727**

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다.

## 비용

- 호출 584회, 입력 토큰 2,278,509 (캐시 0), 출력 토큰 1,801,212
- 게이트웨이 추정 비용 22.5691