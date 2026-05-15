# evaluation/KsponSpeech/

한국어 자유발화 벤치마크. KsponSpeech 데이터셋 사용. 메트릭: CER.

## 사용 가능한 분할

| 파일 | 용도 |
|---|---|
| `transcribe/eval_clean.json` | 전체 clean 평가셋 |
| `transcribe/eval_clean_1000.json` | 1000개 샘플 (빠른 실행용) |
| `transcribe/train.json` 등 | 파인튜닝용 (평가 아님) |

`utils/extract_trn_to_json.py`: 원본 `.trn` → JSON 변환 도구.

## 빠른 시작

```bash
python evaluation/KsponSpeech/test_qwen3_kspon.py \
  --data-json evaluation/KsponSpeech/transcribe/eval_clean_1000.json \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 서버

공유 평가 서버: `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`

## 진단 도구

`send_one.py`: 단일 오디오 파일을 서버에 전송해 응답 확인 (디버깅용).

## 결과 위치

`evaluation/KsponSpeech/results/{model}/{scope}/{tag}/`
