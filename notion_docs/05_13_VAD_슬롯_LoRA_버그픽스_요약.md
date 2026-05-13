# 05_13_VAD_슬롯_LoRA_버그픽스_요약

## 📅 날짜
2026-05-13

## 🔧 작업 내용

### 1. `pending_reset` 타이밍 버그 수정 → `seg_count_before` 방식으로 교체
- `streaming_websocket_server.py` — `_asr_streaming_transcribe`
- 기존: `on_seg` 콜백 내부에서 trailing 없으면 즉시 리셋 → 이후 디코딩된 텍스트 유실
- 변경: `streaming_transcribe` 완료 후 `committed_seg_count` 증가분 확인 + uncommitted 없을 때만 리셋

### 2. VAD-gate 완전 제거
- `vad_speech_detected` 플래그 및 `_process_slot_updates` 내 gate 로직 제거
- 슬롯 전환 시 플래그 리셋 → 상대방 발화 commit 차단 버그 → 근본 해결로 gate 자체 삭제

### 3. VAD end 후 speech start까지 오디오 차단 (`vad_waiting_for_start`) — 추가 후 제거
- VAD end → 슬롯 전환 후 침묵 구간 ASR 전달 방지 목적으로 추가
- `_run_vad_sync`에 start 이벤트 수집 추가, `process_audio_chunk` 이벤트 기반 처리로 재작성
- 실제 테스트에서 발화 시작 대비 ~200ms 이상 지연 발생 확인 → 제거 후 원래 방식 복귀

### 4. `allowed_languages` 두 언어로 확장 (핵심 수정)
- `_new_stream_slot`: `client_lang`만 허용 → `client_lang + client_target_lang` 모두 허용
- 예: ko↔en → `["Korean", "English"]`
- 기존 문제: 영어 발화도 Korean logit bias 적용 → `state.language = "Korean"` → Korean LoRA 선택 → 영어 SEG 미생성
- 수정 후: 영어 발화 → English 감지 → English LoRA → SEG 정상 생성

### 5. 영어 언어 감지 실패 번역 버그 수정 (en→en 번역)
- `lang_to_code`: "Australian English" 등 variant를 `[:2]` = "au"로 잘라내던 문제 → keyword 기반 매핑으로 교체
- `_correct_and_translate`: 언어 감지 완전 실패(`effective=""`) 시 Latin 문자 비율 80% 초과이면 `client_lang`으로 flip 번역하는 fallback 추가

## 🐛 발견된 문제 및 해결

| 버그 | 원인 | 해결 |
|---|---|---|
| 임진왜란 때 텍스트 유실 | `pending_reset`이 `on_seg` 시점에 발동, 이후 디코딩분 삭제 | `streaming_transcribe` 완료 후 판단으로 이동 |
| 영어 발화에 SEG 미생성 | `allowed_languages=["Korean"]` → Korean LoRA → 영어 SEG 패턴 불일치 | 두 언어 모두 허용으로 English LoRA 정상 선택 |
| en→en 번역 (flip 미발동) | `state.language="Australian English"` → `lang_to_code` → "au" ≠ "en" | keyword 기반 variant 정규화 |
| `vad_waiting_for_start` 지연 | VAD start 감지까지 ~200ms 대기 | 로직 제거, 원래 방식 복귀 |
| `AttributeError: _get_partial_text` | 메서드명 오류 | `_slot_uncommitted_display`로 수정 |

## ⏭ 해결되지 않은 작업

- **침묵 구간 ASR 입력 문제**: VAD end 후 새 슬롯에 침묵이 그대로 들어감. 기존(82e623e)에는 `vad_speech_detected` gate로 commit을 차단했는데 현재는 그 보호도 없는 상태. 핸들러 레벨 `vad_speech_detected` 플래그로 가볍게 재도입하는 방안 논의 중.
- **컨텍스트 버퍼 미활용**: `init_streaming_state` 호출 시 `context` 파라미터 미전달. `last_committed_asr_text` 또는 `committed_display`를 context로 넘겨 ASR 연속성 개선 가능.
