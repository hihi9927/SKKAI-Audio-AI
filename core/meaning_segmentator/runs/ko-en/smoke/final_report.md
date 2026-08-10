# 자동 분절 프롬프트 루프 결과 (v2) — Korean → English

- 데이터셋: `kspon` (train 8 / dev 6 / test 8)
- 분절 모델 `claude-sonnet-5` / 판정자 `claude-sonnet-5` / 번역기 `google:en:ctx=True`
- adequacy 백엔드: **cometkiwi** (`Unbabel/wmt22-cometkiwi-da`, 참조 없음)
- consistency 백엔드: comet (보고용)
- 노브: 목표 조각 크기 T. 루프 격자 [3, 6], 최종 격자 [3, 6], 주 작동점 T=6
- 언어 프로파일: Korean, 어순 SOV / 측정: 공백비율 0.2708, 문말 부호 ['.', '?']
- score = T 격자 평균 **final_score** = `adequacy × (1 − contradiction)` (가중치·임계값 없음). 채택 판정은 **쌍체 비교**
- contradiction 백엔드: **deberta-mnli** (`microsoft/deberta-large-mnli`)
- 채택된 프롬프트: iter_00

## 이터레이션 이력

| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.7234 | 0.7724 | — | — | O |

## 최종 test 곡선

| T (목표 조각 어절) | laal_words ↓ | **final_score** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |
|---|---|---|---|---|---|---|---|
| 3 | 2.12 | **0.6538** | 0.7854 | 0.1615 | 0.7208 | 4.38 | 0.00 |
| 6 | 3.58 | **0.6996** | 0.7477 | 0.0593 | 0.8274 | 2.25 | 0.00 |
| unsegmented (노브 없음) | 13.00 | **0.7562** | 0.7562 | 0.0000 | 1.0000 | 1.00 | — |
| mechanical_8 (노브 없음) | 2.26 | **0.4534** | 0.7025 | 0.3628 | 0.6025 | 6.00 | — |

- 포맷 통과율 1.0000 (재시도 없이 1.0000), 위반 0건
- premature_rate (T=6, 부록 지표): **0.5000**

`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). `adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다.

## 비용

- 호출 34회, 입력 토큰 114,935 (캐시 0), 출력 토큰 122,439
- 게이트웨이 추정 비용 1.4543