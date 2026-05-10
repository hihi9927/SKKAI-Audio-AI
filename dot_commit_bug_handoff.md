# Dot Commit WER 증가 버그 — 핸드오프 문서

## 배경

**STiTy** 프로젝트. 실시간 스트리밍 ASR(Qwen3-ASR-1.7B) 서버.  
평가 데이터셋: AMI Meeting Corpus (ES2004a~d, Mix-Headset, 영어 회의 음성).  
평가 서버: `evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py`  
핵심 서버: `Qwen3-ASR/examples/streaming_websocket_server.py`

---

## 관찰된 문제

커밋 **cc01be9** ("dot 커밋 복구")이후 AMI 평가 WER이 크게 증가함.

| 런 | commit 기준 | WER | 총 commits | dot commits |
|---|---|---|---|---|
| **run_04** | cc01be9 이전 (dot 없음) | **15.01%** | 2,158 (vad 100%) | 0 |
| **run_05** | cc01be9 이후 (dot 있음) | **20.84%** | 919 (vad 59% + dot 41%) | 374 |

WER 계산 방식: **순수 텍스트 비교** (timestamp 무관).  
`jiwer.wer(reference, hypothesis)` — reference는 전 화자 단어 시간순 합산 join, hypothesis는 모든 commit된 텍스트 join.  
관련 코드: `evaluation/AMI/test_qwen3_ami.py:425` (`compute_wer_for_rows`)

---

## cc01be9에서 바뀐 것

1. **`streaming_websocket_server.py`** — 각 청크 처리 후 `_process_slot_updates` 호출 복구 (line 599):
   ```python
   await self.asr.streaming_transcribe(chunk, slot["state"], ...)
   await self._process_slot_updates(slot_key)   # ← 이 줄이 추가됨
   ```
   → 스트리밍 중 dot 패턴 감지 시 즉시 dot commit 가능

2. **`qwen3_asr.py`** — `_raw_decoded` 누적 로직 수정 (prefix 창 초과 시 텍스트 소실 방지)

---

## 근본 원인: `_uncommitted_from` 커서 추적 실패

### 버그 발생 흐름

```
① 스트리밍 중 청크 N에서 모델이 부분 transcription 생성:
   state.text = "Okay."

② _process_slot_updates 호출 → "Okay." 끝에 마침표 감지
   → dot commit 발동
   → committed_display = "Okay."
   → committed_seg_count = 1

③ 청크 N+1에서 더 많은 오디오 컨텍스트가 주어지자 모델이 텍스트 수정(revision):
   state.text = "Okay, this is the working design presented by me..."
                      ↑ 마침표가 쉼표로 바뀜

④ 다음 _uncommitted_from 호출:
   _uncommitted_from(
       current_text    = "Okay, this is the working design...",
       committed_display = "Okay.",       ← 점(.)
       committed_seg_count = 1
   )

   1차 prefix 매칭 (line 544):
   "Okay, this is...".startswith("Okay.")  →  False
   ("Okay," vs "Okay." — 쉼표 ≠ 마침표)

   2차 fallback — SEG 카운트 (line 556-559):
   committed_seg_count = 1이므로 <SEG> 태그 1개를 찾아야 함.
   그런데 current_text에 <SEG> 없음 → idx = -1 → return ""

⑤ uncommitted = ""
   → VAD flush에서 uncommitted_display가 비어있어 아무것도 commit 안 됨 (line 639-640)
   → "this is the working design..." 전체 소실
```

### 실제 데이터로 확인

ES2004b hypothesis 비교:

```
run_04: "Okay, this is the working design presented by me, the industrial designer extraordinaire."
run_05: "Okay."   ← 이후 344단어 소실

run_04: "Findings: Most people prefer user-friendly rather than complex remote controls, because..."
run_05: "Findings."   ← 이후 118단어 소실

run_04: "don't well I don't know because if you get you get combined TV and videos..."
run_05: "don't."   ← 이후 346단어 소실
```

총 단어 수 차이:
- ES2004b: run_04 hyp=5,949 words → run_05 hyp=5,404 words (**−545 words**)
- ES2004c: run_04 hyp=6,165 words → run_05 hyp=5,537 words (**−628 words**)

### 버그의 본질

Dot commit은 **스트리밍 중 모델의 부분적(partial) transcription**에 반응한다.  
그런데 Qwen3-ASR는 더 많은 오디오 컨텍스트가 주어지면 이전 transcription을 **revision(수정)** 한다.  
"Okay." → "Okay, this is..."처럼 **마침표가 쉼표로 바뀌면**:
- `committed_display` prefix 매칭 실패 (1차)
- `<SEG>` 태그 기반 fallback도 SEG 태그 없어서 `""` 반환 (2차)
- 수정된 텍스트 전체 소실

---

## 관련 코드 위치

| 파일 | 라인 | 역할 |
|---|---|---|
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 530–562 | `_uncommitted_from` — 버그 위치 |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 597–599 | `_asr_streaming_transcribe` — dot commit 트리거 지점 |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 697–824 | `_process_slot_updates` — dot 감지 및 commit 로직 |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 620–695 | `flush_uncommitted` — VAD flush 로직 |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 800–808 | dot commit 시 `committed_*` 필드 업데이트 |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | 818 | `audio_end_sec=self.current_time` — 별도 timestamp 버그 |

### `_uncommitted_from` 전체 코드

```python
@staticmethod
def _uncommitted_from(current_text: str, committed_display: str,
                      committed_seg_count: int = 0) -> str:
    seg_tag = "<SEG>"
    seg_len = len(seg_tag)

    # 1차: display prefix 매칭
    if committed_display:
        current_no_seg = current_text.replace(seg_tag, "")
        if current_no_seg.startswith(committed_display):
            pos, disp_pos, target = 0, 0, len(committed_display)
            while pos < len(current_text) and disp_pos < target:
                if current_text[pos:pos + seg_len] == seg_tag:
                    pos += seg_len
                else:
                    disp_pos += 1
                    pos += 1
            return current_text[pos:]

    # 2차 fallback: SEG 카운트 기준
    pos, found = 0, 0
    while found < committed_seg_count:
        idx = current_text.find(seg_tag, pos)
        if idx == -1:
            return ""  # ← SEG 없으면 전부 소실
        pos = idx + seg_len
        found += 1
    return current_text[pos:]
```

### dot commit 시 `committed_seg_count` 증가 방식

```python
# streaming_websocket_server.py:807
slot["committed_seg_count"] += len(ready_to_emit)
```

dot commit은 `<SEG>` 토큰을 생성하지 않는다. 그러므로 `committed_seg_count`가 증가해도 `current_text`에 `<SEG>`가 없어 fallback이 항상 `""` 반환.

---

## 부수적 버그 (WER 영향 없음, 하지만 수정 필요)

**`audio_end_sec = self.current_time` (line 818)**  
dot commit 시 실제 발화 위치가 아닌 현재 스트리밍 위치를 audio_end로 사용.  
standby 슬롯이 50초 쌓인 후 decode될 때 최대 50초 timestamp 오차 발생.  
WER 계산에는 영향 없지만 클라이언트 표시 타이밍과 평가 지표(`fsl`, `commit_delay`)가 왜곡됨.

---

## 해결 방향 (미구현)

핵심 문제: **partial transcription에 dot commit을 걸면 모델 revision 시 커서 추적이 깨진다.**

옵션 A — **dot commit을 VAD 경계에서만 허용**  
스트리밍 중 dot을 감지해 큐에 쌓되, 실제 commit은 VAD/SEG 시점에만 수행.  
revision 문제를 원천 차단하지만 latency 개선 효과가 줄어듦.

옵션 B — **`_uncommitted_from` fallback 강화**  
`committed_seg_count` 기반 SEG 탐색이 실패할 때 `committed_display`를 substring 검색으로 fallback.  
`"Okay."` 대신 `"Okay"` (구두점 제거)로 탐색해 revision된 텍스트 "Okay, this is..."에서도 커서를 찾음.

옵션 C — **dot commit 후 `committed_display`에서 구두점 제거**  
`committed_display = "Okay"` (마침표 제거)로 저장하면 "Okay, this is...".startswith("Okay") → True.  
단, "Okay"가 여러 번 등장하면 오매칭 가능성 있음.

옵션 D — **committed text를 `_raw_decoded`에서 실제로 제거**  
dot commit 후 `state._raw_decoded`에서 committed 부분을 제거하고 새 inference 시작.  
가장 확실하지만 streaming inference 재시작 비용 발생.
