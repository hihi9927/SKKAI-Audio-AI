# 통제된 최소쌍 진단 — 임베딩은 무엇을 잡고 무엇을 못 잡는가

관문이 아니라 **진단**이다. 통과·탈락을 매기지 않고 어떤 변이가 어떤 값을 받는지만 본다. 각 변이는 기준문에서 **한 곳만** 바꾼다.

읽는 법: `paraphrase`(무해한 재서술)가 기준선이다. **어떤 의미 변이가 `paraphrase` 보다 높으면 그 유형은 그 백엔드의 맹점이다.**

## `polarity-en`

기준문: `That's a problem`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `That's a problem` | 1.0 | 1.0 | 0.9788 |
| paraphrase | `That's an issue` | 0.9747 | 0.9578 | 0.9889 |
| negation_flip | `That's not a problem` | 0.8706 | 0.7295 | 0.0003 |
| negation_flip2 | `That isn't a problem` | 0.8643 | 0.7561 | 0.0004 |
| antonym | `That's a blessing` | 0.8169 | 0.617 | 0.0002 |
| unrelated | `The weather is nice today` | 0.6855 | 0.4646 | 0.0069 |

## `participant-en`

기준문: `Kim handed the report to Manager Park`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `Kim handed the report to Manager Park` | 1.0 | 1.0 | 0.9961 |
| paraphrase | `Kim gave the report to Manager Park` | 0.9921 | 0.994 | 0.9969 |
| negation_flip | `Kim didn't hand the report to Manager Park` | 0.9089 | 0.7581 | 0.0004 |
| role_swap | `Manager Park handed the report to Kim` | **0.9959** ⚠ | **0.9948** ⚠ | 0.0012 |
| unrelated | `The weather is nice today` | 0.6605 | 0.4493 | 0.0563 |

## `quantifier-en`

기준문: `All the students passed the exam`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `All the students passed the exam` | 1.0 | 1.0 | 0.9965 |
| paraphrase | `Every student passed the exam` | 0.9901 | 0.9683 | 0.9963 |
| negation_flip | `None of the students passed the exam` | 0.8611 | 0.6873 | 0.0002 |
| scope_change | `Some of the students passed the exam` | 0.9254 | 0.8532 | 0.0267 |
| unrelated | `The weather is nice today` | 0.7006 | 0.4714 | 0.0348 |

## `modality-en`

기준문: `The meeting will be held tomorrow`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `The meeting will be held tomorrow` | 1.0 | 1.0 | 0.9961 |
| paraphrase | `The meeting is going to take place tomorrow` | 0.9822 | 0.9795 | 0.9977 |
| negation_flip | `The meeting will not be held tomorrow` | 0.9096 | 0.6916 | 0.0003 |
| modal_change | `The meeting might be held tomorrow` | 0.9595 | 0.9318 | 0.0963 |
| unrelated | `The weather is nice today` | 0.7217 | 0.4611 | 0.009 |

## `number-en`

기준문: `We hired three engineers last quarter`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `We hired three engineers last quarter` | 1.0 | 1.0 | 0.9971 |
| paraphrase | `We took on three engineers last quarter` | 0.9711 | 0.944 | 0.9983 |
| negation_flip | `We didn't hire three engineers last quarter` | 0.9079 | 0.788 | 0.0005 |
| number_change | `We hired thirty engineers last quarter` | 0.9451 | 0.8576 | 0.0002 |
| unrelated | `The weather is nice today` | 0.6886 | 0.4468 | 0.1316 |

## `participant-ko`

기준문: `김 대리가 박 과장에게 보고서를 넘겼다`

| 변이 | 문장 | cos:e5-inst | cos:gte-base | entail:deberta-mnli |
|---|---|---|---|---|
| identical | `김 대리가 박 과장에게 보고서를 넘겼다` | 1.0 | 1.0 | 0.9821 |
| paraphrase | `김 대리가 박 과장에게 보고서를 전달했다` | 0.9936 | 0.9916 | 0.07 |
| negation_flip | `김 대리가 박 과장에게 보고서를 넘기지 않았다` | 0.9592 | 0.9103 | **0.2517** ⚠ |
| role_swap | `박 과장이 김 대리에게 보고서를 넘겼다` | 0.9796 | 0.9073 | **0.9777** ⚠ |
| unrelated | `오늘 날씨가 좋아서 산책을 했다` | 0.7595 | 0.3736 | 0.0426 |

⚠ = 그 변이가 `paraphrase` 이상으로 유사하게 나온 칸 (= 맹점).

## 요약 — 유형별 맹점

| 변이 유형 | 맹점인 백엔드 |
|---|---|
| `negation_flip` | entail:deberta-mnli(participant-ko) |
| `negation_flip2` | — |
| `antonym` | — |
| `role_swap` | cos:e5-inst(participant-en), cos:gte-base(participant-en), entail:deberta-mnli(participant-ko) |
| `scope_change` | — |
| `modal_change` | — |
| `number_change` | — |
| `unrelated` | — |

## 조각 vs 전체 문장 — 어절 수를 맞춘 대조

위 세트는 완결 문장끼리 비교한다. 그런데 실제 관문·목적함수는 **조각 vs 전체 문장**이라 길이가 어긋난다. 여기서는 전체 문장 하나에 대해 **어절 수가 같은** 조각 두 개(극성만 다름)를 두어, 길이 교란을 통제한 채 극성만 본다.

| 백엔드 | 정답 | 평균 격차 |
|---|---|---|
| `cos:e5-inst` | 6/6 | +0.1016 |
| `cos:gte-base` | 6/6 | +0.2109 |
| `contra:deberta-mnli` | 6/6 | +0.9684 |

| 전체 문장 | aligned | flipped |
|---|---|---|
| `I don't think that will be a problem.` | `I don't think that` | `I do think that` |
| `She didn't finish the report on time.` | `She didn't finish the report` | `She did finish the report` |
| `The meeting will not be held tomorrow.` | `The meeting won't be held` | `The meeting will be held` |
| `He never agreed to the terms.` | `He never agreed` | `He fully agreed` |
| `We can't accept this proposal.` | `We can't accept this` | `We can accept this` |
| `They have not left the building.` | `They have not left` | `They have now left` |

**길이만 맞추면 코사인도 극성을 읽는다.** 그러면 실제 관문에서 극성이 묻히는 이유는 폴라리티 맹목이 아니라 **후보들의 길이가 서로 다르다는 것**이다. 자기-prefix 바닥 실측에서 길이가 만드는 코사인 거리 변동은 1–2어절 0.229 → 15+ 0.044 로 약 0.185 인데, 위 극성 신호는 0.10 규모다 — **길이 교란이 극성 신호보다 크다.**


## 판정

- `cos:e5-inst` — 부정 맹점 0세트, 참여자 맹점 1세트 ['participant-en']
- `cos:gte-base` — 부정 맹점 0세트, 참여자 맹점 1세트 ['participant-en']
- `entail:deberta-mnli` — 부정 맹점 1세트 ['participant-ko'], 참여자 맹점 1세트 ['participant-ko']

**구조적 긴장.** 길이를 맞추면 극성이 읽히지만, 조기 방출을 검출하려면 **미래를 아는 무언가**와 비교해야 하고 그것은 정의상 방출분보다 길다. 즉 *미래 지식*과 *길이 정합*은 직접 충돌한다. 그래서 필요한 것은 길이 불일치에 **본래 관대한** 자다 — 함의는 비대칭이라 '긴 전제가 짧은 가설을 함의한다' 가 자연스러운 관계지만, 코사인은 대칭이라 길이 차를 그냥 거리로 읽는다. NLI 가 이 자리를 지키는 진짜 이유가 부정 감지력이 아니라 **비대칭성**이다.