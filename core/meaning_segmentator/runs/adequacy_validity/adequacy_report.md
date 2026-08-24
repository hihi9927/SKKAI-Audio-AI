# adequacy 백엔드 타당도 검사 (조각 입력)

백엔드: `cometkiwi` — 케이스 6건, 순위 검사 12건, 위반 2건 → **탈락**

통과 조건: 심각한 오류 변이(의미 변경 / 부정 뒤집힘 / 원문 반환 / 무관) 점수가 `benign_minimal`(동의어 수준)보다 낮을 것.

## 위반

| 케이스 | 오류 유형 | 오류 점수 | benign | 차이 |
|---|---|---|---|---|
| ko-en-a04 | meaning_error | 0.5049 | 0.4748 | +0.0301 |
| ko-en-a04 | source_echo | 0.5653 | 0.4748 | +0.0905 |

## 케이스별 원점수

### ko-en-a01 — `내가 딱 들어갔을 때`

| 변이 | 점수 |
|---|---|
| correct | 0.7307 |
| benign_minimal | 0.7966 |
| **meaning_error** | 0.4662 |
| **unrelated** | 0.4742 |

### ko-en-a02 — `학생회 형들이 진짜 엄청 착하고`

| 변이 | 점수 |
|---|---|
| correct | 0.8247 |
| benign_minimal | 0.8516 |
| **negation_flip** | 0.6329 |
| **meaning_error** | 0.6337 |

### ko-en-a03 — `그 해부학 할 때가 제일 재밌긴 했는데`

| 변이 | 점수 |
|---|---|
| correct | 0.7280 |
| benign_minimal | 0.7804 |
| **negation_flip** | 0.6531 |
| **meaning_error** | 0.7498 |

### ko-en-a04 — `그때 밀려 썼던 그 기억이`

| 변이 | 점수 |
|---|---|
| correct | 0.4693 |
| benign_minimal | 0.4748 |
| **meaning_error** | 0.5049 |
| **source_echo** | 0.5653 |

### ko-en-a05 — `같이 사는 거 생각 없어?`

| 변이 | 점수 |
|---|---|
| correct | 0.8612 |
| benign_minimal | 0.8509 |
| **meaning_error** | 0.7343 |
| **unrelated** | 0.5294 |

### ko-en-a06 — `우린 또 그런 거 안 하잖아`

| 변이 | 점수 |
|---|---|
| correct | 0.8265 |
| benign_minimal | 0.8230 |
| **negation_flip** | 0.6408 |
| **unrelated** | 0.5100 |
