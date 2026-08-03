## 📅 날짜
2026-08-03

## 📌 이 문서

`dot_commit_confirm`(확정 게이트) 현재 구현의 전체 로직 명세. v6 시점 코드 기준.
구현 위치: `Qwen3-ASR/examples/streaming_websocket_server.py`, `Qwen3-ASR/qwen_asr/inference/sentence_boundary.py`.
설계 배경은 [08_01 확정게이트 설계검증 요약](08_01_dot_commit_확정게이트_설계검증_요약.md), 이번 변경 경위는 [08_03 finish 제거 요약](08_03_dot_commit_finish_제거_요약.md) 참조.

---

## 1. 왜 게이트가 필요한가

Qwen3-ASR은 청크 버퍼가 끝나면 문장이 미완성이어도 마침표를 붙여 마무리짓는다. 학습분포(완결 발화)와 추론입력(중간에 잘린 버퍼)의 불일치 때문이다. 따라서 **가설 맨 끝(프론티어)의 마침표는 `P(마침표 | 버퍼 끝) ≈ 1`이라 정보량이 0**이다. 마침표를 감지하는 즉시 커밋하면 매 청크마다 문장 조각이 커밋된다.

게이트는 **감지기는 그대로 두고 커밋 시점만** 바꾼다. 마침표가 "확정"된 뒤에만 커밋한다.

핵심 판별 신호: **오디오가 그 지점을 지나간 뒤 재디코딩했을 때도 그 경계가 살아남는가.** 실측으로 청크 끝 마침표 329개 중 141개(42.9%)만 다음 청크에서 생존했다 — 판별 가능하다는 직접 증거.

---

## 2. 경계 감지기 (`sentence_boundary.py`)

```python
_ABBREV_LOOKBEHIND = r"(?<!Mr)(?<!Mrs)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<!No)"
_CLOSERS = r"[\"'”’»›\)\]\}]*"

DOT_COMMIT_BOUNDARY_RE = re.compile(
    r"(?:"
    rf"{_ABBREV_LOOKBEHIND}\.{_CLOSERS}(?:\s+|$)"   # 마침표 (+ 닫는 따옴표/괄호)
    rf"|[?!]{_CLOSERS}(?:\s+|$)"                     # 물음표/느낌표
    rf"|[。？！]{_CLOSERS}"                           # CJK 종결부호
    r"|<SEG>"                                        # 파인튜닝 모델의 SEG 토큰
    r")"
)
```

| 요소 | 목적 |
|---|---|
| `_ABBREV_LOOKBEHIND` | `Mr.` `Dr.` 등 약어 뒤 마침표 제외 |
| `_CLOSERS` | 인용문 종료 흡수. 대화체는 마침표가 인용부호 **안쪽**에 찍혀(`... a shop boy."`) 이게 없으면 경계로 인정되지 않는다 |
| `(?:\s+\|$)` | 소수점(`3.14`) 자동 제외 — 마침표 뒤가 숫자라 매치 안 됨 |
| `$` 허용 | 문자열 끝 마침표도 경계로 인정. 예전엔 `(?=\S)` 룩어헤드 때문에 발화의 **마지막 문장이 항상 finish로만** 커밋됐다 |

---

## 3. 판정이 일어나는 지점

`_process_slot_updates(slot_key, force_reason=None, chunk_end=False)`가 유일한 판정 함수다. 호출 경로는 3개:

| 호출자 | `chunk_end` | 비고 |
|---|---|---|
| `_on_seg` 콜백 (generate 루프 중) | `False` | SEG 토큰 디코딩 순간 |
| `_on_dot` 콜백 (generate 루프 중) | `False` | 마침표 디코딩 순간 |
| `_asr_streaming_transcribe` 청크 종료 | **`True`** | 한 청크 디코딩 완료 후 |

**규칙 2·3은 청크 간 가설 비교이므로 `chunk_end=True`에서만 판정한다.** 규칙 1은 청크 중간에서도 커밋 가능해 지연 손해가 없다.

### 조기 리턴 가드

```python
_recheck_pending = chunk_end and self.dot_commit_confirm
if not current_text or (current_text == slot["last_text"] and not _recheck_pending):
    return None
```

`dot_commit_confirm`이 켜져 있으면 **텍스트가 그대로여도 chunk_end 호출은 항상 통과**시킨다. 세 가지 이유가 겹친다:
- 무음 청크처럼 가설이 안 바뀌는 상황이 바로 규칙 2가 성립하는 경우다.
- 규칙 3 정체 카운터도 가설이 안 자란 청크에서 올라가야 한다.
- `_on_dot` 콜백이 함수 진입부에서 이미 `last_text`를 갱신해버린다. "pending이 있을 때만 통과"로 두면 pending 등록 기회 자체가 사라져 **조기 리턴 ↔ 미등록 교착**이 생긴다(실측: `DOT-PENDING` 0건, 파일 전체가 finish 1건).

---

## 4. 확정 규칙 4개

매칭된 경계마다 아래 순서로 판정한다. `after` = 경계 뒤에 남은 텍스트, `sentence` = 경계까지의 문장.

### 규칙 1 — 문맥 확정 (`rule=context`)

```python
if self._count_tokens(after) > state.unfixed_token_num:   # 기본 5
    commit
```

마침표 **뒤에 남은 토큰 수가 롤백 창(`unfixed_token_num`, 기본 5)보다 크면** 그 마침표는 이미 롤백 창 밖이다. 다음 청크에서 모델이 수정할 수 없으므로 즉시 커밋한다. 청크 중간(`chunk_end=False`)에서도 발동하므로 지연이 가장 짧다.

### 규칙 2 — 합의 확정 (`rule=stable`)

```python
elif (chunk_end
      and self._boundary_key(sentence) in _prev_sentences
      and _accum_now > _prev_accum):
    commit
```

**직전 chunk_end 가설에도 있던 경계가 재디코딩 후에도 살아남았으면 확정.** 이게 VAD를 대체한다 — 발화가 끝나고 침묵이 이어지면 새 토큰이 안 나와 가설이 유지되므로 다음 청크에서 자동 확정된다. 오디오 에너지·임계값·의존성 없이 텍스트 불변 감지로 같은 일을 한다.

세부 설계 3가지:

**(a) 단일 후보가 아니라 경계 "목록"으로 비교한다.**
`prev_boundary_sentences`에 직전 chunk_end의 경계 전체를 담는다. 후보 하나만 들고 있으면 한 청크에 경계가 둘 이상일 때 뒤쪽 경계는 앞 경계가 커밋될 때까지 평가조차 안 된다 — 발화 마지막 문장이 딱 그 경우라 오디오가 떨어질 때까지 밀린다(실측: `"He's a good man. He is."`가 2청크 연속 동일했는데 `He is.`는 후보 슬롯을 못 잡아 finish로 빠짐).

**(b) 비교는 정규화 키로 한다.**
```python
_boundary_key(s) = re.sub(r'[.,!?;:。？！、，\'"“”‘’\s]+', '', s.replace("<SEG>","")).lower()
```
모델은 롤백 창 안에서 문구는 그대로 두고 구두점만 바꾸는 일이 잦다(실측: `'house, mother.'` ↔ `'house mother.'`가 청크마다 번갈아 나와 완전 일치 비교로는 영원히 확정 불가). 경계가 같은 자리에 살아남았는지만 보면 되므로 표기 변형은 무시한다. **커밋은 최신 텍스트로 하고 비교에만 키를 쓴다.**

**(c) `_accum_now > _prev_accum` — 오디오가 실제로 늘어난 뒤에만 확정.**
같은 청크 안에서 등록→확정이 연달아 일어나면 재디코딩 검증 없이 통과된다.

### 규칙 3 — 정체 확정 (`rule=stall`)

```python
elif _stall_hit and not after.strip():
    commit
```

**오디오는 누적되는데 미커밋 가설의 토큰 수가 N청크 연속 그대로면 발화 종료로 판정.** 규칙 2가 놓치는 "롤백 창 안 문구 수정"(`Anon.` → `And on.`)을 흡수한다. 오디오 에너지를 안 보므로 VAD 의존성은 없다.

카운터:
```python
_same_base = len(slot["committed_display"]) == slot.get("stall_committed_len")
if _same_base and _unc_tokens == slot.get("stall_tokens") and _accum_c > slot.get("stall_accum", -1):
    slot["stall_count"] += 1
else:
    slot["stall_count"] = 0
    slot["stall_tokens"] = _unc_tokens
```

발동 조건이 3중으로 좁혀져 있다:

| 가드 | 이유 |
|---|---|
| `_same_base` (committed_display 길이 동일) | 직전 청크 이후 커밋이 있었으면 `uncommitted`가 가리키는 **구간 자체가 달라진다**. 그 상태로 토큰 수만 비교하면 서로 다른 텍스트가 우연히 같은 길이일 때 오판한다 (실측: `'My dear," said Miss.'`와 `'"Pray don't."'`가 **둘 다 6토큰** → 미완성 문장을 커밋해 WER 7.14% → 50%) |
| `not after.strip()` (frontier 한정) | "발화가 끝났다"는 판정이므로 마지막 경계에만 적용. 중간 경계는 규칙 1·2 담당 |
| `_accum_c > stall_accum` | 오디오가 실제로 늘어난 청크에서만 카운트 |

설정: `--dot-commit-stall-chunks` (기본 **1**, 0이면 비활성화).

### 규칙 4 — finish (안전망)

스트림 종료 시 `flush_uncommitted(force=True, reason="finish")`가 남은 구간을 통째로 커밋한다. **제거하지 않았다** — 마침표가 아예 안 찍힌 가설은 원리상 dot으로 커밋할 수 없다. LibriSpeech 100파일 v6 기준 발동 0건.

### 미확정 시

```python
else:
    if chunk_end and pending != sentence:
        slot["pending_dot_text"] = sentence
        slot["pending_dot_accum"] = _accum_now
    break
```

**등록은 `chunk_end`에서만.** generate 루프 중의 `_on_dot`(`chunk_end=False`)이 중간 가설로 덮어쓰면, 청크 종료 호출이 "직전 청크"가 아니라 "같은 청크 중간값"과 비교하게 되어 규칙 2가 영원히 성립하지 않는다(실측: 매 청크 pending이 두 문장 사이를 왕복). `break`으로 이번 호출의 남은 경계 처리를 중단한다 — 앞 경계가 미확정인데 뒤 경계를 커밋하면 순서가 깨지기 때문.

---

## 5. 슬롯 리셋과 게이트 상태 carry

커밋이 발생하면 서버는 슬롯을 리셋해 `audio_accum`을 버린다(누적 30초 넘으면 모델 출력이 무너지므로 필수). `_reset_stream_slot`은 슬롯 dict를 통째로 갈아끼우므로 **게이트 상태도 같이 날아간다.**

그러면 발화 마지막 문장이 리셋 직후 후보로 다시 등록되고, 확정에 필요한 다음 청크가 오기 전에 오디오가 떨어져 finish로 빠진다. `carry_audio`는 같은 단어로 재디코딩되므로 직전 청크의 경계 목록은 **여전히 유효**하다:

```python
carry_boundaries = _s.get("prev_boundary_sentences", ())
self._reset_stream_slot(slot_key)
new_slot["state"].audio_accum = carry_audio
if carry_boundaries:
    new_slot["prev_boundary_sentences"] = carry_boundaries
    new_slot["prev_boundary_accum"] = -1   # accum이 줄어드므로 재기준화
```

`prev_boundary_accum = -1`은 "이미 한 번 재디코딩을 견딘 경계라 다음 청크에서 바로 확정돼도 안전"이라는 의미다.

한편 `flush_uncommitted`(VAD/finish)는 남은 구간을 통째로 커밋하므로 보류 중인 후보를 무효화한다:
```python
slot.pop("pending_dot_text", None)
slot.pop("pending_dot_accum", None)
```

---

## 6. 커밋 단계의 중복 방어 (게이트 밖, 모든 커밋 경로 공통)

확정된 문장이라도 아래를 통과해야 실제로 방출된다.

**(a) 커서 추적 — `_committed_cursor`**
`committed_display`를 현재 텍스트의 prefix로 매칭해 미커밋 구간 시작점을 찾는다. 정규화 수준을 단계적으로 높이며 시도한다: 원문 → 구두점 통일 → trailing punct 제거 → 따옴표/콜론 제거 → **소문자**. 마지막 소문자 후보가 없으면 커밋 후 모델이 `'Oh, Papa!'` → `'Oh, papa!'`로 고쳐 쓸 때 커서가 `-1`로 떨어지고, dot 커밋은 `committed_seg_count`를 올리지 않아 SEG fallback도 막혀 있어서 **텍스트 전체가 미커밋으로 취급**된다 → 매 콜백 재검출 + finish flush가 이미 커밋한 구간까지 재방출(실측: WER 0.00% → 18.18%). `_walk`가 원본 길이 기준이라 **길이가 보존될 때만** 소문자 후보를 쓴다.

**(b) 교차 중복 제거 — `committed_asr_set`**
```python
_asr_key = lambda s: ' '.join(s.split()).rstrip('.,!?;:。？！').strip().lower()
```
이미 커밋한 문장이 재디코딩으로 다시 나타나면 skip(`COMMIT-SKIP reason=cross-dedup`). 대소문자·trailing punct 변형을 같은 문장으로 취급한다.

**(c) 그 외 dedup 레이어**
- `rep-dedup`: 한 호출 안에서 연속 중복
- `seg-boundary-dedup`: SEG 리셋 경계에서 단어 중복
- `dot-suffix-dedup`: 슬롯 스위치 직후 이전 커밋의 suffix 반복

---

## 7. 설정

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--enable-dot-commit` | baseline 계열 자동 True | dot 기반 커밋 활성화 |
| `--dot-commit-confirm` | **False** | 확정 게이트. 켜야 규칙 1·2·3 동작 |
| `--dot-commit-stall-chunks` | 1 | 규칙 3 발동에 필요한 정체 청크 수. 0이면 비활성화 |
| `chunk_size_sec` | 2.0 | 확정 기회는 누적 오디오가 이 배수에 도달할 때만 생긴다 |
| `unfixed_token_num` | 5 | 롤백 창. 규칙 1 임계값 |
| `MAX_AUDIO_ACCUM_SEC` | 90.0 | 강제 리셋 임계값 |

> **`dot_commit_confirm` 기본값은 아직 False다.** 프로덕션 서버는 명시적으로 켜야 게이트가 걸린다. `streaming_websocket_server_dualbase.py`에는 미반영.

---

## 8. 확정 기회의 양자화 — 실사용과 평가의 차이

확정 판정은 chunk_end에서만 일어나고, chunk_end는 누적 오디오가 `chunk_size_sec`(2.0s) 배수에 도달할 때만 발생한다. 따라서

```
확정 기회 수 = floor((발화길이 + 뒤따르는 오디오) / chunk_size_sec)
```

**실사용(라이브 마이크)에서는 화자가 멈춰도 스트림이 계속 흐르므로 기회가 무제한**이다. 반면 평가 하네스는 `--trailing-silence-ms`만큼만 무음을 보내고 끊는다. 실측으로 이 차이가 그대로 드러났다:

| 파일 | 전송량 | 다음 청크 필요 | 5500ms | 8000ms |
|---|---|---|---|---|
| 0055 | 7.89s | 8.0s (**0.11초 부족**) | finish | **dot** |
| 0085 | 9.48s | 10.0s (0.52초 부족) | finish | dot (정규화 키 병행 필요) |

→ LibriSpeech 클라이언트의 `--trailing-silence-ms` 기본값을 5500 → **8000**으로 변경했다.

---

## 9. LibriSpeech 100파일 실측 (v6)

| | run01 (게이트 도입 시점) | v6 |
|---|---|---|
| finish | 9 | **0** |
| dot | 114 | 123 |
| WER | 2.79% | **2.55%** |
| 커밋 수 | 123 | 123 |
| 커밋당 토큰 | 12.75 | 12.75 |

파일별 동일 97 / 개선 3 / 악화 0. 규칙 분포: **context 14 / stable 107 / stall 2**.

커밋 수와 커밋당 토큰이 사실상 동일한 채 finish만 사라졌다 — 커밋 단위를 흔들지 않고 귀속만 바꿨다는 직접 지표.

---

## 10. 알려진 한계

- **검증 범위가 좁다.** 영어 낭독체 LibriSpeech test-other 100파일, 화자 2명. 한국어·자유발화 미검증. 2.55%는 test-other 전체 수치가 아니다.
- **지연이 구조적으로 늘어난다.** 확정을 최소 한 청크 기다리므로 chunk 2.0s 기준 최대 +2초. 오프라인 실험에서 chunk 1.0s로 줄이면 지연 페널티가 +2.44초 → +0.71초로 감소하고 WER도 개선됐으나 RTF가 0.049 → 0.161로 올라 동시 수용량이 약 1/3. 프로덕션 경로로는 미검증.
- **규칙 3 조건 적정성 미확정.** v6에서 2회 발동해 실효는 생겼으나, 오발동으로 WER을 크게 망친 전력이 있어 가드 3개가 충분히 좁은지 추가 검토 필요.
- **전역 변경이 다른 모드 기준선에 영향.** `_committed_cursor` 소문자 후보와 `_asr_key` `.lower()`는 게이트 밖이라 모든 커밋 경로에 적용된다. mode2(always-commit)·mode4(파인튜닝 SEG) 재측정 미실시.
