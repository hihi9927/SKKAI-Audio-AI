# evaluation/smartturn/

SmartTurn-v3 VAD 통합 실험 트랙. 메인 ASR 벤치마크 파이프라인과 분리된 독립 실험. WER 메트릭 없음.

## SmartTurn이란

RMS 기반 단순 침묵 감지를 대체하는 턴 감지/VAD 모델. 여기서 통합 검증 후 프로덕션 병합 여부 결정.

## 핵심 진입점

| 파일 | 역할 |
|---|---|
| `run_qwen_pipeline_smartturn.py` | SmartTurn VAD를 사용한 전체 파이프라인 |
| `run_qwen3_streaming_server_smartturn.py` | SmartTurn 심을 가진 스트리밍 서버 |
| `verify_architecture_smartturn.py` | 오프라인 트레이스 검증 |
| `librispeech_qwen_smartturn.py` | SmartTurn VAD로 LibriSpeech 벤치마크 실행 |
| `turn_detector.py` | SmartTurn 우선, 실패 시 RMS 폴백 |

## 설치

```bash
pip install -r evaluation/smartturn/requirements.txt
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `STITY_SMARTTURN_SCORE_HOOK` | — | 커스텀 스코어러 (`module:function`) |
| `STITY_SMARTTURN_PROB_MODE` | `endpoint` | `endpoint` 또는 `speech` |
| `STITY_SMARTTURN_CONTEXT_SEC` | `1.0` | 컨텍스트 윈도우 (초) |
| `STITY_SMARTTURN_INFER_HOP_MS` | `240` | 추론 hop (ms) |

## SmartTurn 튜닝 파라미터

`--st-prob-mode`, `--st-threshold-on`, `--st-threshold-off`, `--st-min-silence-ms`, `--st-min-utterance-ms`, `--st-end-cooldown-ms`, `--st-ema-alpha`

자세한 내용: `evaluation/smartturn/README.md`
