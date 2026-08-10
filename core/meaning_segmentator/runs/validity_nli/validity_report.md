# 품질 백엔드 타당도 검사

케이스 5건 x 변이 8종. 오류는 JSON 에 고정되어 있고 매 실행 동일하다.

**통과 조건: 심각한 의미 오류(부정 뒤집힘 / 주체 뒤바뀜 / 절 누락 / 지시대상 소실)의 점수가 `benign_minimal`(동의어·음역 차이)보다 낮을 것.**

`benign_paraphrase`(구조까지 바꾼 재서술)는 판정에 쓰지 않는다 — 참조 기반 지표는 재서술을 의미 손상과 구분하지 못하는 것이 알려진 성질이고, 그걸 기준으로 삼으면 어떤 백엔드든 탈락한다.

## 변이 유형별 평균

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.7381 | 0.8445 |
| benign_paraphrase | 0.9690 | 0.7441 |
| **negation_flip** | 0.0141 | 0.0877 |
| **role_swap** | 0.0190 | 0.8334 |
| **clause_omission** | 0.2818 | 0.1315 |
| **referent_loss** | 0.3042 | 0.2567 |
| unrelated | 0.0084 | 0.0364 |

## 판정

| 백엔드 | 순위 검사 | 위반 (기준: minimal) | 참고 (기준: paraphrase) | 판정 |
|---|---|---|---|---|
| nli-mdeberta | 17 | 2 | 0 | **탈락** |
| nli-deberta | 17 | 2 | 2 | **탈락** |

### nli-mdeberta 위반 상세

| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |
|---|---|---|---|---|
| ja-ko-01 | clause_omission | 0.3955 | 0.0477 | +0.3478 |
| ja-ko-01 | referent_loss | 0.6699 | 0.0477 | +0.6222 |

### nli-deberta 위반 상세

| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |
|---|---|---|---|---|
| ja-ko-01 | negation_flip | 0.3595 | 0.2473 | +0.1122 |
| ja-ko-01 | role_swap | 0.9951 | 0.2473 | +0.7478 |

## 케이스별 원점수

### ja-ko-01 (Japanese → Korean)

- src: `ごんはうなぎを盗みませんでした。兵十がそれを川へ返しました。`
- ref: `곤은 뱀장어를 훔치지 않았습니다. 병십이 그것을 강에 돌려보냈습니다.`

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.0477 | 0.2473 |
| benign_paraphrase | 0.9838 | 0.2004 |
| negation_flip | 0.0016 | 0.3595 |
| role_swap | 0.0048 | 0.9951 |
| clause_omission | 0.3955 | 0.0624 |
| referent_loss | 0.6699 | 0.1371 |
| unrelated | 0.0120 | 0.0797 |

### ko-en-01 (Korean → English)

- src: `내가 딱 들어갔을 때 학생회 형들이 진짜 엄청 착하고 다 좋으셨어 그래가지구 그 형들이 우리한테 엄청 잘해주고`
- ref: `When I first joined, the student council seniors were really nice and all great, so those seniors treated us really well.`

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.8908 | 0.9969 |
| benign_paraphrase | 0.9424 | 0.9964 |
| negation_flip | 0.0001 | 0.0001 |
| role_swap | 0.0332 | 0.6717 |
| clause_omission | 0.0064 | 0.0095 |
| referent_loss | 0.0338 | 0.8931 |
| unrelated | 0.0004 | 0.0388 |

### ko-en-02 (Korean → English)

- src: `그 해부학 할 때가 제일 재밌긴 했는데 그때 밀려 썼던 그 기억이 아직도 잊혀지지가 않네.`
- ref: `Anatomy was the most fun, but I still can't forget the memory of filling in the answers off by one back then.`

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.9544 | 0.9923 |
| benign_paraphrase | 0.9806 | 0.5397 |
| negation_flip | 0.0027 | 0.0021 |
| clause_omission | 0.0023 | 0.0002 |
| referent_loss | 0.0026 | 0.0002 |
| unrelated | 0.0015 | 0.0182 |

### ko-en-03 (Korean → English)

- src: `아니 그니까 같이 사는 거 생각 없어? 딴 딴 데 유성이나 이쪽에.`
- ref: `No, I mean, do you not want to live together? Somewhere else, like Yuseong or around here.`

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.8385 | 0.9924 |
| benign_paraphrase | 0.9898 | 0.9943 |
| negation_flip | 0.0652 | 0.0767 |
| clause_omission | 0.0793 | 0.0003 |
| referent_loss | 0.0080 | 0.0004 |
| unrelated | 0.0277 | 0.0453 |

### ko-en-04 (Korean → English)

- src: `아. 우린 또 그런 거 안 하잖아. 어. 그치.`
- ref: `Ah. We don't do that kind of thing, you know. Yeah. Right.`

| 변이 | nli-mdeberta | nli-deberta |
|---|---|---|
| identical | 1.0000 | 1.0000 |
| benign_minimal | 0.9589 | 0.9936 |
| benign_paraphrase | 0.9483 | 0.9899 |
| negation_flip | 0.0010 | 0.0003 |
| clause_omission | 0.9254 | 0.5853 |
| referent_loss | 0.8069 | 0.2527 |
| unrelated | 0.0002 | 0.0002 |
