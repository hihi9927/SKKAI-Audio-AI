# evaluation/smartturn/

SmartTurn-v3 VAD 통합 실험 트랙. 메인 ASR 벤치마크와 분리된 독립 실험이라 WER/CER 메트릭이 없다.
RMS 기반 침묵 감지를 대체할 턴 감지 모델을 여기서 검증한 뒤 프로덕션 병합 여부를 정한다.

| 파일 | 역할 |
|---|---|
| `run_qwen3_streaming_server_smartturn.py` | SmartTurn 심을 끼운 스트리밍 서버 (parity 테스트용 주 경로) |
| `run_qwen_pipeline_smartturn.py` | SmartTurn VAD 전체 파이프라인 |
| `verify_architecture_smartturn.py` | 오프라인 트레이스 검증 — `score_backend`로 `smartturn` / `rms_fallback` 확인 |
| `librispeech_qwen_smartturn.py` | SmartTurn VAD로 LibriSpeech 실행 |
| `turn_detector.py` | SmartTurn 우선, 실패 시 RMS 폴백 |
| `silero_vad.py`, `pipecat_hook.py` | 서버용 SmartTurn 심 / 스코어 훅 |

결과는 `results_json/`에 쌓인다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `STITY_SMARTTURN_SCORE_HOOK` | — | 커스텀 스코어러 (`module:function`) |
| `STITY_SMARTTURN_PROB_MODE` | `endpoint` | `endpoint` 또는 `speech` |
| `STITY_SMARTTURN_CONTEXT_SEC` | `1.0` | 컨텍스트 윈도우 (초) |
| `STITY_SMARTTURN_INFER_HOP_MS` | `240` | 추론 hop (ms) |

## 튜닝 인자

`--st-prob-mode`, `--st-threshold-on/-off`, `--st-min-silence-ms`, `--st-min-speech-ms`, `--st-min-utterance-ms`, `--st-end-cooldown-ms`, `--st-ema-alpha`

설치·실행 예시: @README.md
