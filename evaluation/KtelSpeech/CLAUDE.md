# evaluation/KtelSpeech/

한국어 전화 대화 벤치마크. KtelSpeech 데이터셋 사용. 메트릭: CER.

## 데이터 레이아웃

`--data-dir evaluation/KtelSpeech` 지정 시 스크립트가 내부적으로 `KtelSpeech/`(오디오)와 `label/`(전사)로 분기.

## 오디오 특성 주의

전화 대역폭 음성 (협대역, 8kHz). WER/CER 수치를 LibriSpeech 등과 직접 비교 금지.

## 빠른 시작

```bash
# 터미널 1
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py --no-idle-shutdown

# 터미널 2
python evaluation/KtelSpeech/test_qwen3_ktelspeech.py \
  --data-dir evaluation/KtelSpeech \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 결과 위치

`evaluation/KtelSpeech/results/{model}/{scope}/{tag}/`

## 디렉토리 구성 참고

현재 `test_qwen3_ktelspeech.py`와 `results/`만 존재. 유틸리티 추가 시 다른 데이터셋 패턴(`utils/`, `transcribe/`) 따를 것.
