# 품질 백엔드 타당도 검사

케이스 5건 x 변이 8종. 오류는 JSON 에 고정되어 있고 매 실행 동일하다.

**통과 조건: 심각한 의미 오류(부정 뒤집힘 / 주체 뒤바뀜 / 절 누락 / 지시대상 소실)의 점수가 `benign_minimal`(동의어·음역 차이)보다 낮을 것.**

`benign_paraphrase`(구조까지 바꾼 재서술)는 판정에 쓰지 않는다 — 참조 기반 지표는 재서술을 의미 손상과 구분하지 못하는 것이 알려진 성질이고, 그걸 기준으로 삼으면 어떤 백엔드든 탈락한다.

## 변이 유형별 평균

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9308 | 0.9509 | 0.8620 |
| benign_paraphrase | 0.8414 | 0.8533 | 0.3830 |
| **negation_flip** | 0.8843 | 0.8976 | 0.8081 |
| **role_swap** | 0.9365 | 0.9122 | 0.8678 |
| **clause_omission** | 0.7693 | 0.8593 | 0.5497 |
| **referent_loss** | 0.8660 | 0.9016 | 0.7326 |
| unrelated | 0.3948 | 0.0633 | 0.1216 |

## 판정

| 백엔드 | 순위 검사 | 위반 (기준: minimal) | 참고 (기준: paraphrase) | 판정 |
|---|---|---|---|---|
| comet | 17 | 1 | 12 | **탈락** |
| embed | 17 | 3 | 12 | **탈락** |
| chrf | 17 | 5 | 16 | **탈락** |

### comet 위반 상세

| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |
|---|---|---|---|---|
| ja-ko-01 | role_swap | 0.9353 | 0.9224 | +0.0129 |

### embed 위반 상세

| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |
|---|---|---|---|---|
| ja-ko-01 | negation_flip | 0.929 | 0.849 | +0.08 |
| ja-ko-01 | clause_omission | 0.8878 | 0.849 | +0.0388 |
| ja-ko-01 | referent_loss | 0.9635 | 0.849 | +0.1145 |

### chrf 위반 상세

| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |
|---|---|---|---|---|
| ja-ko-01 | negation_flip | 0.784 | 0.7187 | +0.0653 |
| ja-ko-01 | role_swap | 0.8244 | 0.7187 | +0.1057 |
| ja-ko-01 | referent_loss | 0.8199 | 0.7187 | +0.1012 |
| ko-en-01 | role_swap | 0.9111 | 0.8673 | +0.0438 |
| ko-en-04 | negation_flip | 0.8839 | 0.8522 | +0.0317 |

## 케이스별 원점수

### ja-ko-01 (Japanese → Korean)

- src: `ごんはうなぎを盗みませんでした。兵十がそれを川へ返しました。`
- ref: `곤은 뱀장어를 훔치지 않았습니다. 병십이 그것을 강에 돌려보냈습니다.`

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9224 | 0.8490 | 0.7187 |
| benign_paraphrase | 0.8326 | 0.8162 | 0.2688 |
| negation_flip | 0.8973 | 0.9290 | 0.7840 |
| role_swap | 0.9353 | 0.8467 | 0.8244 |
| clause_omission | 0.7401 | 0.8878 | 0.4920 |
| referent_loss | 0.8647 | 0.9635 | 0.8199 |
| unrelated | 0.4384 | 0.1697 | 0.0658 |

### ko-en-01 (Korean → English)

- src: `내가 딱 들어갔을 때 학생회 형들이 진짜 엄청 착하고 다 좋으셨어 그래가지구 그 형들이 우리한테 엄청 잘해주고`
- ref: `When I first joined, the student council seniors were really nice and all great, so those seniors treated us really well.`

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9431 | 0.9867 | 0.8673 |
| benign_paraphrase | 0.9114 | 0.9119 | 0.4851 |
| negation_flip | 0.8498 | 0.8332 | 0.6731 |
| role_swap | 0.9377 | 0.9776 | 0.9111 |
| clause_omission | 0.9067 | 0.9510 | 0.6941 |
| referent_loss | 0.9214 | 0.9799 | 0.8403 |
| unrelated | 0.3839 | -0.0152 | 0.1157 |

### ko-en-02 (Korean → English)

- src: `그 해부학 할 때가 제일 재밌긴 했는데 그때 밀려 썼던 그 기억이 아직도 잊혀지지가 않네.`
- ref: `Anatomy was the most fun, but I still can't forget the memory of filling in the answers off by one back then.`

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9043 | 0.9706 | 0.9242 |
| benign_paraphrase | 0.7798 | 0.8739 | 0.4660 |
| negation_flip | 0.8657 | 0.9557 | 0.7691 |
| clause_omission | 0.6084 | 0.7617 | 0.2469 |
| referent_loss | 0.8067 | 0.8334 | 0.5675 |
| unrelated | 0.4075 | 0.1027 | 0.1699 |

### ko-en-03 (Korean → English)

- src: `아니 그니까 같이 사는 거 생각 없어? 딴 딴 데 유성이나 이쪽에.`
- ref: `No, I mean, do you not want to live together? Somewhere else, like Yuseong or around here.`

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9173 | 0.9585 | 0.9478 |
| benign_paraphrase | 0.7885 | 0.8411 | 0.4417 |
| negation_flip | 0.9119 | 0.9409 | 0.9303 |
| clause_omission | 0.6680 | 0.7648 | 0.5240 |
| referent_loss | 0.8298 | 0.8370 | 0.7855 |
| unrelated | 0.3616 | 0.0421 | 0.1205 |

### ko-en-04 (Korean → English)

- src: `아. 우린 또 그런 거 안 하잖아. 어. 그치.`
- ref: `Ah. We don't do that kind of thing, you know. Yeah. Right.`

| 변이 | comet | embed | chrf |
|---|---|---|---|
| identical | 1.0000 | 1.0000 | 1.0000 |
| benign_minimal | 0.9671 | 0.9896 | 0.8522 |
| benign_paraphrase | 0.8948 | 0.8236 | 0.2535 |
| negation_flip | 0.8970 | 0.8291 | 0.8839 |
| clause_omission | 0.9235 | 0.9312 | 0.7916 |
| referent_loss | 0.9073 | 0.8942 | 0.6497 |
| unrelated | 0.3825 | 0.0172 | 0.1359 |
