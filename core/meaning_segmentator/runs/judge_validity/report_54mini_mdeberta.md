# 판정자 타당도 검사

- 판정자 모델: `gpt-5.4-mini`
- `JUDGE_SYSTEM` 해시: `2681c69704bafe3c66d86a1a7afb29f4`
- 케이스 6건 × 변이 13종 × 3회 반복

## 통과 조건

1. **정확도** — 모든 변이의 다수결이 `expect` 와 `safe`/`not-safe` 축에서 일치
2. **안정성** — 반복 실행에서 `safe` / `not-safe` 판정이 동일

두 조건 모두 세부 라벨이 아니라 **safe / not-safe 이진**으로 본다. 이유: `premature` 와
`mistranslated` 는 루프에서 같은 행동을 부른다 — 경계를 표시하고 `cause`·`shift` 를
Critic 에 넘긴다. 채택 게이트도 `unsafe_rate` 를 쓴다. 세부 라벨을 소비하는 곳이
없으므로 그 축을 관문 조건으로 두면 과잉 명세다. `라벨정확`·`라벨고정` 열에 진단으로만 남긴다.

판정 기준선은 `benign_*` 이다. 조기 방출 자체가 아니라 **뒤가 반박하는지**를
구별하는지 보는 것이다. 이 행이 없으면 "무조건 premature" 라고 답하는 판정자가
관문을 통과하고, 루프는 모든 경계를 문제로 보며 보수화한다.

## 결과

| 케이스 | 변이 | expect | 판정 | 반복 | 정확 | 안정 | 라벨정확 | 라벨고정 |
|---|---|---|---|---|---|---|---|---|
| ko-en-p01 | premature_negation | premature | premature | {'premature': 3} | O | O | O | O |
| ko-en-p01 | safe_boundary | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p02 | premature_role | premature | mistranslated | {'mistranslated': 3} | O | O | - | O |
| ko-en-p02 | benign_incomplete | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p03 | premature_scope | premature | mistranslated | {'mistranslated': 3} | O | O | - | O |
| ko-en-p03 | benign_incomplete | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p04 | premature_head | premature | mistranslated | {'mistranslated': 3} | O | O | - | O |
| ko-en-p04 | benign_incomplete | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p04 | benign_reordered | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p06 | benign_incomplete | safe | safe | {'safe': 3} | O | O | O | O |
| ko-en-p06 | premature_modal | premature | mistranslated | {'mistranslated': 2, 'premature': 1} | O | O | - | - |
| ja-ko-p05 | premature_negation | premature | premature | {'premature': 3} | O | O | O | O |
| ja-ko-p05 | benign_incomplete | safe | safe | {'safe': 3} | O | O | O | O |

오분류 0건 / 불안정 0건 → **통과**  (세부 라벨: 불일치 4건 / 흔들림 1건 — 판정에 반영 안 함)

---

# NLI contradiction 백엔드 검사

- 백엔드: `mdeberta-xnli`

판정자와 달리 이 값은 **목적함수에 직접 들어간다** (`effective = adequacy × (1 − contradiction)`). 그래서 기준이 라벨이 아니라
**순위**다 — argmax 가 `neutral` 이어도 확률이 `premature > safe` 이면
임계값 없이 연속 점수로 쓸 수 있다.

**통과 조건: 모든 케이스에서 `min(premature) > max(safe)`.**

| 케이스 | 변이 | expect | contradiction |
|---|---|---|---|
| ko-en-p01 | premature_negation | premature | 0.0283 |
| ko-en-p01 | safe_boundary | safe | 0.0108 |
| ko-en-p02 | premature_role | premature | 0.9589 |
| ko-en-p02 | benign_incomplete | safe | 0.0593 |
| ko-en-p03 | premature_scope | premature | 0.0133 |
| ko-en-p03 | benign_incomplete | safe | 0.0106 |
| ko-en-p04 | premature_head | premature | 0.3370 |
| ko-en-p04 | benign_incomplete | safe | 0.0420 |
| ko-en-p04 | benign_reordered | safe | 0.0536 |
| ko-en-p06 | benign_incomplete | safe | 0.0275 |
| ko-en-p06 | premature_modal | premature | 0.7970 |
| ja-ko-p05 | premature_negation | premature | 0.9746 |
| ja-ko-p05 | benign_incomplete | safe | 0.3413 |

순위 위반 0/6 → **통과**
