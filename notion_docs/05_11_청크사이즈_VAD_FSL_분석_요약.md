## 📅 날짜
2026-05-11

## 🔧 작업 내용

- `Qwen3-ASR/examples/streaming_websocket_server.py` — VAD flush 시점 디버그 로그 추가
  - `_asr_finish_streaming` 직후 `state.text`, `committed_display`, `committed_seg_count`, `uncommitted` 텍스트를 `[vad-dbg]` 태그로 출력
  - VAD 발동 시 uncommitted가 비어있는 원인 추적 목적

## 📊 결과 / 수치

### chunk_size별 WER (finetuned 1.0.1, LibriSpeech test-other)

| chunk_size | WER | tokens/commit | SEG 비율 | VAD 비율 | model_runtime |
|---|---|---|---|---|---|
| 0.25s | 51.0% | 6.9 | 97.3% | 2.7% | -1.06s |
| 0.5s | 22.0% | 7.3 | 96.4% | 3.6% | -0.16s |
| 1.0s | 13.4% | 7.7 | 87.4% | 12.6% | +0.45s |
| 2.0s | 10.7% | 8.2 | 64.0% | 36.0% | +0.81s |

## 🐛 발견된 문제 및 해결

### 1. VAD flush 시 uncommitted 텍스트 누락 버그

**샘플:** LibriSpeech 1688-142285-0000, chunk 2s

**증상:** "He makes me harshly feel" 문장이 출력되지 않음

**원인 (로그로 확인):**
- 스트리밍 추론 중 partial output에서 `"But his. <SEG> He makes me harsh..."` 형태로 SEG 감지
- `on_seg` 콜백이 "But his."를 커밋 → `committed_seg_count = 1`
- 동시에 VAD 발동 → `finish_streaming_transcribe`가 전체 오디오로 재추론
- 재추론 결과: `"But his, he makes me harshly feel. <SEG>"` (두 문장을 하나로 합침, SEG 1개)
- `_uncommitted_from`이 `committed_seg_count=1` 기준으로 1번째 SEG 이후를 찾으나 → 이미 끝 → `uncommitted = ''`
- "He makes me harshly feel"이 사라짐

**근본 원인:** partial `on_seg` 커밋으로 세운 `committed_seg_count` 커서가, `finish_streaming`의 텍스트 구조 재작성으로 무효화됨

**해결 여부:** 미해결 (보류). 방향 검토만 완료:
- 방향 A: VAD 발동 시 committed 롤백 + 재emit → 중복 emit 위험
- 방향 B: VAD 구간에서 partial on_seg 커밋 억제 → latency 증가 (최소 chunk_size + 800ms)
- 절충안: `committed_seg_count >= final SEG count` 감지 시 텍스트 길이 기준 fallback

**발생 빈도:** chunk가 클수록 VAD 비율 증가(2s: 36%) → 구조적으로 흔하게 발생 가능

### 2. 소형 청크(0.25s)의 WER 폭발 원인

**원인:**
- 0.25s는 모델이 문장 경계 판단에 필요한 컨텍스트보다 너무 짧음
- 단어 3~4개 단위로 SEG 남발 (avg_tokens/commit = 6.9) → 과분절
- prefix rollback이 잘못 끊긴 텍스트를 계속 이어받음
- `model_runtime = -1.06s`: 과분절로 인해 오디오 전송 완료 전에 마지막 커밋 발생 → 후반부 오디오 미처리

## ⏭ 해결되지 않은 작업

- VAD + partial on_seg 커밋 충돌 버그 픽스 (방향 미결정)
- 최적 chunk_size 실험: 0.75~0.8s 구간 벤치마크 미실시 (이론적 최적 = VAD_MIN_SILENCE_MS 800ms 이하)
- FSL 측정 방식 불일치 개선
  - SEG path: snapshot 기반 추정 시각 사용 (SEG 등장 오디오 위치보다 이른 시점)
  - VAD path: `final_decode + trans` 합산으로 audio 기준 latency 없음
  - SEG/VAD 비율이 chunk_size에 따라 달라지므로 `avg_fsl_sec` 비교 시 단위 불일치 발생
