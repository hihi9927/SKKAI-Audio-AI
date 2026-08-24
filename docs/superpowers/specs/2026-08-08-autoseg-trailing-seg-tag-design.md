# autoseg — 발화 끝 `<SEG>` 허용 설계

- 날짜: 2026-08-08
- 대상: `core/meaning_segmentator/autoseg/`
- 관련 문서: [METRICS.md](../../../core/meaning_segmentator/autoseg/METRICS.md), [AUTO_PROMPT_LOOP_DESIGN.md](../../../core/meaning_segmentator/AUTO_PROMPT_LOOP_DESIGN.md)

---

## 1. 문제

autoseg 의 포맷 검증기는 발화 맨 뒤의 `<SEG>` 를 `trailing_tag` 위반으로 잡는다
([pipeline.py:183](../../../core/meaning_segmentator/autoseg/pipeline.py#L183)). 이 규칙은
"발화 끝은 어차피 경계이므로 태그가 무의미하다"는 가정에서 나왔다.

그 가정은 틀렸다. autoseg 가 라벨링하는 코퍼스는 VAD 로 잘린 자발 발화이고, VAD 절단은
휴지에서 일어나므로 **절 경계와 자주 겹친다.** 발화의 마지막 위치는 다른 위치와 똑같이
"의미 경계인가"를 판단해야 하는 자리이며, 참인 경우 태그가 라벨에 남아야 한다.

기준은 **문장이 끝났는가가 아니라 의미 경계인가**이다. 둘은 다르다.

```
이거 분위기가 막 지브리 같다고 <SEG> 막 뭐 한창 그거 했던 거 같은데 <SEG>
```

`-같은데` 는 문말 구두점도 아니고 문장도 완결되지 않았지만 절 경계다 — ko 프로파일의
`clause_boundary_signals` 에 `-는데` 가 들어 있다. 마지막 태그는 정당하다. 반면

```
엄청 어렸을 때는 이제 대전 어디 지역에 살았는지는
```

은 명사구 중간에서 잘렸으므로 끝에 태그가 오면 안 된다.

현재 검증기는 두 경우를 구분하지 않고 전부 막는다.

## 2. 확인된 사실

### 2.1 데이터에 미완결 발화가 이미 들어 있다

`runs/ko-en/run06-qw1.0/data/` 실측. 문말 구두점으로 끝나는 항목이 절반 이하다.

| 분할 | n | 문말 구두점으로 끝남 | 없음 |
|---|---|---|---|
| train | 30 | 11 (37%) | 19 (63%) |
| dev | 60 | 23 (38%) | 37 (62%) |
| test | 100 | 48 (48%) | 52 (52%) |

같은 데이터를 **절 경계 신호 기준**으로 다시 나누면 분포가 달라진다. 이쪽이 §1 의
판정 기준과 맞는 축이다 (분류 방법은 §3.3).

| 분할 | n | 경계 신호로 끝남 | 조각으로 끝남 |
|---|---|---|---|
| train | 30 | 18 (60%) | 12 (40%) |
| dev | 60 | 41 (68%) | 19 (32%) |
| test | 100 | 74 (74%) | 26 (26%) |

즉 **구두점 없이 끝난 발화의 다수가 실은 절 경계에서 끝났다.** 끝 태그를 금지하면
그만큼의 경계가 라벨에서 통째로 빠진다.

실제 항목 예:

```
엄청 어렸을 때는 이제 대전 어디 지역에 살았는지는          조각 종료 — 태그 없어야
이거 분위기가 막 지브리 같다고 막 뭐 한창 그거 했던 거 같은데   경계 종료 — 태그 있어야
```

`load_kspon` 주석이 이미 이 성질을 명시한다 —
"구두점이 없는 항목이 많은 것이 정상이며, 그것이 바로 실시간 분절이 풀어야 하는 조건이다"
([data.py:130](../../../core/meaning_segmentator/autoseg/data.py#L130)).

**따라서 데이터 추가·교체는 필요 없다.** ko 경로에 필요한 입력은 이미 있다.
ja/kokoro 로더는 `_is_wellformed` 로 미완결 발화를 버리지만 이번 범위 밖이다 (§4).

### 2.2 지표는 이 변경에 무해하다

`latency_proxy`, `effective_segments`, `aggregate` 가 모두 `split_segments` 를 거치고,
그 함수가 빈 조각을 버린다
([pipeline.py:203](../../../core/meaning_segmentator/autoseg/pipeline.py#L203)).

```python
return [p.strip() for p in seg_text.split(SEG) if p.strip()]
```

끝 태그는 마지막에 빈 문자열 조각을 만들고 그것이 필터된다. 따라서 `k`, `L`, `Q` 는
끝 태그의 유무에 대해 **정확히 불변**이다.

검증기의 나머지 규칙도 끝 태그를 오탐하지 않는다.

- `text_modified` — `_canon` 이 태그를 공백으로 치환한 뒤 strip 하므로 원문과 일치한다.
- `missing_space` — 태그 뒤 문자열이 빈 문자열이면 검사를 건너뛴다.
- `punct_after_tag` — 태그 뒤에 문자가 없으므로 매칭되지 않는다.

**즉 금지 규칙 하나만 제거하면 나머지 배관은 그대로 돌아간다.**

### 2.3 지표가 무해하다는 것은 곧 지표가 못 본다는 뜻이다

`Q`·`L` 이 끝 태그에 불변이므로 목적함수 입장에서 끝 태그는 **공짜**다. 프롬프트가
모든 발화에 끝 태그를 붙여도 점수가 깎이지 않는다. 그 상태의 라벨은
"발화 끝은 항상 커밋"을 뜻하는데, ko 데이터의 52~63%가 어중 종료이므로 정확히 틀린
방향이며 ASR 은 그 오염된 신호를 학습한다.

이 설계는 이를 목적함수 축이 아니라 **감시 지표**로 다룬다 (§3.3).

## 3. 설계

### 3.1 검증기 — 금지 해제

[pipeline.py:183-184](../../../core/meaning_segmentator/autoseg/pipeline.py#L183-L184) 의
`trailing_tag` 위반을 제거한다.

`leading_tag`, `consecutive_tags`, `text_modified`, `punct_after_tag`, `missing_space` 는
그대로 둔다. `V` 는 계속 하드 게이트로 동작한다.

### 3.2 프롬프트 규칙 — 조건부 허용으로 교체

검증기만 풀고 프롬프트를 두면 모델은 계속 태그를 찍지 않는다. 규칙 문자열을 함께 고친다.

`OUTPUT_RULES_SPACED` / `OUTPUT_RULES_UNSPACED`
([agents.py:29-48](../../../core/meaning_segmentator/autoseg/agents.py#L29-L48)) 에서

```
- No tag at the very start or the very end of the text.
```

를 다음으로 교체한다.

```
- Never place a tag at the very start of the text.
- The END of the text is a candidate boundary like any other. Place a tag there if and only
  if the text ends at a meaning boundary, judged by exactly the same criteria you apply to
  boundaries inside the text.
- The text does NOT have to be a finished sentence for that. These utterances are cut by
  silence detection, so they often stop right after a completed clause even though the
  speaker was still talking — that is a boundary and gets a tag. Only when the text breaks
  off mid-clause, mid-phrase, or on a filler is there no boundary at the end.
```

핵심은 **끝을 특별 취급하지 않는 것**이다. 조건절을 따로 다는 것이 아니라 "마지막 위치도
내부와 같은 기준으로 본다"를 명시한다. 조건을 복잡하게 쓰면 모델이 끝을 예외로 인식해
보수화한다.

`OUTPUT_RULES_UNSPACED` 의 공백 규칙도 함께 고친다. 현재 문장은
"Always place exactly one space on both sides of every `<SEG>` tag" 인데 끝 태그에는
한쪽 면밖에 없어 모순된다. 다음을 덧붙인다.

```
  (a tag at the very end has only a preceding space)
```

같은 문구를 `OUTPUT_RULES_SPACED` 에도 적용한다.

### 3.3 감시 지표

새 LLM 호출 없이 `seg_texts` 문자열만으로 계산한다.

단일 비율은 드리프트만 잡고 정확도는 못 본다. **발화 종료 유형으로 쪼개면 정확도 프록시가
된다.** 단 종료 유형을 문말 구두점으로 가르면 안 된다 — §1 이 보인 대로 구두점 없이
절 경계에서 끝나는 발화가 다수다.

**분류 기준: 절 경계 신호.** `language_profile.json` 의 `clause_boundary_signals` 와
`trailing_punctuation` 을 합집합으로 쓴다. 발화 말미가 그중 하나와 일치하면 `경계 종료`,
아니면 `조각 종료`.

- `clause_boundary_signals` 항목 중 `-` 로 시작하는 것(어미)만 접미사 매칭에 쓴다.
  `"sentence-final '.', '?'"` 처럼 서술문인 항목은 버린다.
- `"어서/아서"` 처럼 한 항목에 변이형이 `/` 로 묶여 있으므로 **`/` 로 분리한 뒤** 매칭한다.
  분리하지 않으면 `-해서` 로 끝난 발화를 조각으로 오분류한다 (실측에서 확인).
- `trailing_punctuation` 은 마지막 문자 일치로 본다.

`Metrics` ([metrics.py:262](../../../core/meaning_segmentator/autoseg/metrics.py#L262)) 에
다섯 필드를 추가한다.

| 필드 | 정의 |
|---|---|
| `trailing_tag_rate` | 끝 태그가 붙은 발화 비율 (전체 기준) |
| `n_boundary_ended` | 경계 종료로 분류된 발화 수. 아래 두 비율의 표본 크기 |
| `trailing_on_boundary_end` | 경계 종료 발화 중 끝 태그가 붙은 비율 — 높아야 함 |
| `trailing_on_fragment_end` | 조각 종료 발화 중 끝 태그가 붙은 비율 — 낮아야 함 |
| `trailing_gap` | `trailing_on_boundary_end − trailing_on_fragment_end` |

퇴화가 `trailing_gap` 하나로 드러난다.

| 프롬프트 행동 | on_boundary | on_fragment | gap |
|---|---|---|---|
| 전부 찍음 | ~1.0 | ~1.0 | ~0 |
| 아무것도 안 찍음 (현재 상태) | ~0.0 | ~0.0 | ~0 |
| 제대로 판단 | 높음 | 낮음 | 큼 |

두 목록 모두 언어 프로파일에서 오므로 언어 종속이 새로 생기지 않는다. 검증기의
`punct_after_tag` 가 `trailing_punctuation` 을 쓰는 것과 같은 설계 원칙이다.

**클래스 균형** (§2.1). test 74/26, dev 68/32, train 60/40 으로 양쪽 표본이 확보된다.

**배선.** 프로파일 파싱은 한 곳에만 둔다. `metrics.py` 에 헬퍼를 추가한다.

```python
def profile_boundary_endings(profile: dict) -> list[str]:
    """경계 종료 판정용 접미사 목록. 언어 지식은 프로파일에서만 온다."""
```

`clause_boundary_signals` 중 `-` 로 시작하는 항목만 취해 접두 `-` 를 떼고 `/` 로 분리한 뒤,
`trailing_punctuation` 을 합친다.

호출 사슬:

| 위치 | 변경 |
|---|---|
| [metrics.py:284](../../../core/meaning_segmentator/autoseg/metrics.py#L284) `aggregate()` | `boundary_endings: list[str] \| None = None` 파라미터 추가 |
| [loop.py:30](../../../core/meaning_segmentator/autoseg/loop.py#L30) `evaluate()` | 같은 파라미터 추가, [loop.py:61](../../../core/meaning_segmentator/autoseg/loop.py#L61)·[loop.py:85](../../../core/meaning_segmentator/autoseg/loop.py#L85) 의 `aggregate()` 두 호출에 전달 |
| [loop.py:429](../../../core/meaning_segmentator/autoseg/loop.py#L429), [445](../../../core/meaning_segmentator/autoseg/loop.py#L445), [541](../../../core/meaning_segmentator/autoseg/loop.py#L541) | `evaluate()` 호출 3곳에 인자 추가. 값은 [loop.py:362](../../../core/meaning_segmentator/autoseg/loop.py#L362) 에서 프로파일을 읽는 자리 옆에서 한 번 계산 |
| [eval_prompt.py:98](../../../core/meaning_segmentator/autoseg/eval_prompt.py#L98) | `evaluate()` 호출에 인자 추가. 프로파일은 [eval_prompt.py:56](../../../core/meaning_segmentator/autoseg/eval_prompt.py#L56) 에서 이미 읽고 있다 |

기본값 `None` 은 "프로파일 정보 없음"을 뜻하고 이때 감시 지표는 §3.3 의 표본 부족 규약을
따른다. 호출부를 빠뜨려도 점수 축(`V`·`Q`·`L`·objective)은 영향받지 않는다.

**표본이 없을 때의 규약.** 프로파일에 두 목록이 모두 비어 있거나 한쪽 클래스 표본이 0이면
해당 비율과 `trailing_gap` 을 `0.0` 으로 둔다 (JSON 이 `NaN` 을 못 담는다).
`n_boundary_ended` 를 함께 기록하므로 "표본이 없어서 0" 과 "모델이 안 찍어서 0" 을
사람이 구분할 수 있다.

### 3.4 리포트 노출

- **이터레이션 이력 표** ([loop.py:589](../../../core/meaning_segmentator/autoseg/loop.py#L589))
  에 `trail gap` 열을 추가한다. 드리프트는 이터레이션 간 변화로 나타나므로 여기가
  1차 감시 지점이다 (iter00 0.62 → iter03 0.04 같은 변화가 한 줄에 보인다).
- **최종 test 결과 표** 에 `trailing_gap`, `trailing_on_boundary_end`,
  `trailing_on_fragment_end` 세 행을 추가한다. 의미 열에 §6 의 한계를 한 문장으로 적는다.

### 3.5 종료 시 경고

`build_report` 이후 stderr 에 경고를 낸다. `validity_check.py` 의 `[경고] 타당도 탈락`
([validity_check.py:213](../../../core/meaning_segmentator/autoseg/validity_check.py#L213))
과 같은 패턴이다.

```python
TRAILING_GAP_WARN = 0.2      # 판단값. 이 아래면 끝 태그가 종료 유형과 무관하게 찍힌 것
MIN_TRAILING_SAMPLES = 5     # 양쪽 표본이 이보다 적으면 gap 을 신뢰하지 않는다
```

조건: `n_boundary_ended >= 5` **그리고** `n − n_boundary_ended >= 5` **그리고**
`trailing_gap < 0.2`.

```
[경고] 끝 태그가 발화 종료 유형과 무관하게 찍힘 (gap 0.03) — ASR 라벨 오염 가능
```

CLI 플래그로 빼지 않는다. 두 상수는 코드에 두고 근거를 주석으로 남긴다.

### 3.6 Critic 노출

`Critic.review` 는 `metrics` dict 를 JSON 으로 통째로 넘기므로
([agents.py:206](../../../core/meaning_segmentator/autoseg/agents.py#L206)) 새 필드는
자동으로 컨텍스트에 들어간다. 두 가지만 손댄다.

1. **지표 설명 괄호** ([agents.py:207-215](../../../core/meaning_segmentator/autoseg/agents.py#L207-L215))
   에 새 필드의 뜻과 "gap 이 클수록 좋다"를 추가한다. 설명이 없으면 LLM 이 필드를 무시하거나
   오독한다.
2. **`CRITIC_SYSTEM`** ([agents.py:140](../../../core/meaning_segmentator/autoseg/agents.py#L140))
   에 발화 끝 판단을 설명하는 문단을 추가하고, `error_type` 열거에
   `trailing_decision` 을 넣는다.

**`summarize_critique` 의 `direction` 사다리는 바꾸지 않는다.** 사다리는 목적함수의 제약
순서(포맷 → 품질 → 지연)를 그대로 반영하는 것이 설계 의도이고
([agents.py:227](../../../core/meaning_segmentator/autoseg/agents.py#L227) 주석), 끝 태그는
목적함수에 들어가지 않으므로 사다리에 넣으면 두 기준이 어긋난다. `trailing_decision` 은
`counts` 에 집계되지만 어느 분기 조건에도 쓰이지 않는다 — `counts` 가 동적 dict 이므로
추가 변경이 필요 없다.

Prompt Engineer 에게는 Critic 의 `summary` 와 사례별 `proposed_rule` 을 통해 간접적으로
전달된다. 목적함수 점수는 움직이지 않는다.

## 4. 범위 밖

- **ja/kokoro 로더의 `_is_wellformed` 필터**
  ([data.py:95](../../../core/meaning_segmentator/autoseg/data.py#L95)). 미완결 발화를
  버리므로 ja 에는 §2.1 의 분포가 없다. 별건이며 ko 실험에 영향이 없다.
- **목적함수 변경.** `objective()` 는 그대로 둔다. `q_weight`, `Q_floor`, 앵커 모두 불변.
- **인과 판정 축.** 라벨 경계가 좌문맥만으로 결정 가능한지 재는 별도 지표는 채택하지
  않는다. 검토했으나 이번 목표(발화 끝 판단을 라벨에 담기)보다 범위가 크다.

## 5. 하위 호환

저장된 모든 런에서 끝 태그 사용 건수가 0이다. `runs/ko-en/run06-qw1.0` 의 train·dev
4개 파일 180행 실측 결과 `trailing=0, leading=0, invalid=0`.

따라서 검증기를 풀어도 **기존 점수는 한 자리도 바뀌지 않는다.**

- `Q_floor` 재캘리브레이션 불필요.
- 앵커(`baseline.json`) 재측정 불필요.
- 백엔드 타당도 검사(`validity_check.py`) 재실행 불필요 — 품질 백엔드도 번역기도
  건드리지 않는다.
- `eval_prompt.py` 로 저장된 프롬프트를 재채점해도 같은 값이 나온다.

`Metrics` 에 필드가 늘어나므로 새 `history.json` 은 기존보다 키가 많다. 읽는 쪽은
`build_report` 뿐이고 새 키를 명시적으로 참조하므로 옛 런 디렉토리를 새 코드로 다시
리포트하면 `KeyError` 가 난다. `history` 표 렌더링에서 `h["train"].get("trailing_gap")`
형태로 읽고 없으면 `—` 를 찍는다.

## 6. 한계

**접미사 매칭은 프록시다.** 어미가 같아도 실제 경계 여부는 문맥에 달렸다. 프로파일에
`non_boundary_traps` 필드가 있는 이유가 이것이다 (ko 는 `이제`, `막`, `그` 등). 분류는
발화 끝만 보므로 함정 어절이 종결에 오면 조각으로 분류되어 이번 목적에는 유리하게
작동하지만, 반대 방향 오분류는 남는다.

test 26건의 조각 종료 분류를 눈으로 검사한 결과 8건 표본에서 7건이 정확했고, 오분류 1건은
프로파일의 `"어서/아서"` 표기를 분리하지 않은 데서 왔다 — §3.3 의 `/` 분리로 해소된다.

따라서 `trailing_gap` 의 **방향**은 신뢰할 수 있지만 **절댓값**은 그렇지 않다.
리포트 의미 열과 `METRICS.md` 에 이 문장을 남긴다.

**최적화 압력이 간접적이다.** 끝 태그는 목적함수에 들어가지 않으므로, 프롬프트가 이
판단을 잘하게 만드는 힘은 Critic 의 자연어 지적뿐이다. 루프가 자동으로 개선하지 못하면
사람이 프롬프트를 손봐야 한다. 이는 §2.3 의 대안(목적함수 축 추가)을 의도적으로
포기한 결과다.

## 7. 검증

**단위 테스트**

1. `validate()` — 끝 태그가 더 이상 위반이 아니다. 같은 입력에서 `leading_tag`,
   `consecutive_tags`, `text_modified`, `punct_after_tag`, `missing_space` 는 여전히
   위반으로 잡힌다.
2. `latency_proxy()` / `effective_segments()` — 끝 태그를 붙인 문자열과 붙이지 않은
   문자열의 값이 같다 (§2.2 의 불변성 회귀 테스트).
3. `profile_boundary_endings()` — ko 프로파일 입력에서 `"-어서/아서"` 가 `어서`, `아서`
   두 항목으로 분리되고, `"sentence-final '.', '?'"` 처럼 `-` 로 시작하지 않는 서술문
   항목은 제외되며, `trailing_punctuation` 이 합쳐진다.
4. `aggregate()` — 합성 `seg_texts` 로 `trailing_on_boundary_end`,
   `trailing_on_fragment_end`, `trailing_gap` 을 검증한다. 세 경우를 모두 덮는다:
   전부 찍음 / 아무것도 안 찍음 / 경계 종료에만 찍음.
5. `aggregate()` 표본 부족 — `boundary_endings=None` 이거나 한쪽 클래스 표본이 0일 때
   `0.0` 을 반환하고 예외를 내지 않는다.

**회귀 확인**

6. `runs/ko-en/run06-qw1.0` 의 저장된 rows 를 새 코드로 재채점해 `V`, `Q`, `L`,
   `objective` 가 변경 전과 동일한지 확인한다.

**행동 확인**

7. run06 의 채택 프롬프트에 §3.2 의 새 규칙만 적용해 test 100문장을 다시 분절하고,
   `trailing_gap` 이 0보다 유의하게 큰지 본다. 0 근처면 프롬프트 규칙 문구를 고쳐야
   한다는 뜻이며, 그 사실 자체가 감시 지표가 동작한다는 증거다.
8. 같은 산출물에서 `trailing_on_boundary_end` 가 `segmented_rate` 와 크게 어긋나지 않는지
   본다. 발화 끝을 내부와 같은 기준으로 본다면 두 값이 같은 수준이어야 하고, 끝만 현저히
   낮으면 모델이 여전히 끝을 예외 취급하는 것이다.

## 8. 변경 파일 요약

| 파일 | 변경 |
|---|---|
| `autoseg/pipeline.py` | `trailing_tag` 위반 제거 |
| `autoseg/agents.py` | 출력 규칙 2종 교체, Critic 지표 설명 추가, `CRITIC_SYSTEM` 문단 + `trailing_decision` 열거값 |
| `autoseg/metrics.py` | `Metrics` 필드 5개, `profile_boundary_endings()` 헬퍼, `aggregate()` 에 `boundary_endings` 파라미터 |
| `autoseg/loop.py` | `evaluate()` 파라미터 추가 + 호출 3곳, `aggregate()` 호출 2곳, 이력 표 열, test 표 행, 종료 경고 |
| `autoseg/eval_prompt.py` | `evaluate()` 호출에 `boundary_endings` 전달 |
| `autoseg/METRICS.md` | §2 에 끝 태그 규약과 감시 지표 절 추가 |
