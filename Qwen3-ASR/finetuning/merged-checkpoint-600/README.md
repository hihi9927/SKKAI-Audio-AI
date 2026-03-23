# merged-checkpoint-600

이 디렉터리는 **평가/추론용으로 생성된 실행 모델 폴더**이며, 원본 파인튜닝 체크포인트 자체는 아닙니다.

## 이 폴더가 무엇인가요?

- 원본 파인튜닝 체크포인트:
  - `Qwen3-ASR/finetuning/qwen3-asr-finetuning-out/checkpoint-600`
- 베이스 모델:
  - `Qwen/Qwen3-ASR-1.7B`
- 용도:
  - LibriSpeech FCL 평가 서버에서 바로 읽을 수 있는 단일 모델 디렉터리 제공

즉, `checkpoint-600`은 LoRA 어댑터 체크포인트이고, 기존 평가 서버는 "바로 실행 가능한 모델 폴더"를 기대하기 때문에 그 중간 결과물로 이 폴더가 생성됩니다.

## 누가 만들었나요?

이 폴더는 아래 스크립트가 생성합니다.

- `evaluation/LibriSpeech/servers/run_finetuned_fcl_server.py`

이 스크립트는 다음 순서로 동작합니다.

1. 베이스 Qwen3-ASR 모델 로드
2. `checkpoint-600`의 LoRA 어댑터 로드
3. 어댑터를 베이스 모델에 merge
4. merge된 결과를 이 폴더에 저장
5. 이 폴더를 `--model`로 사용하여 FCL 스트리밍 평가 서버 실행

## 왜 필요한가요?

기존 평가 코드와 서버 코드를 수정하지 않고 아래 평가 스크립트를 그대로 사용하기 위해 필요합니다.

- `evaluation/LibriSpeech/test_qwen3_librispeech.py`

특히 아래 평가 흐름을 파인튜닝 모델 기준으로 그대로 태우기 위한 용도입니다.

- FCL 계산
- VAD 기반 commit 동작
- segmentation (`seg` / `vad`)
- first-token latency 측정

## 중요한 점

이 폴더는 **원본 학습 산출물(canonical finetuning output)** 이 아닙니다.

원본 파인튜닝 산출물은 아래 경로입니다.

- `Qwen3-ASR/finetuning/qwen3-asr-finetuning-out/checkpoint-600`

즉:

- `checkpoint-600` = 원본 LoRA 파인튜닝 체크포인트
- `merged-checkpoint-600` = 평가/서빙을 위해 생성한 실행용 merge 결과

## 삭제해도 되나요?

현재 누군가 이 폴더를 평가/서빙에 사용 중이 아니라면 삭제는 가능합니다.

다만 이 폴더는 실수로 생긴 불명 파일이 아니라, 의도적으로 생성되는 평가용 산출물입니다.  
삭제하더라도 아래 명령으로 다시 만들 수 있습니다.

```bash
python evaluation/LibriSpeech/servers/run_finetuned_fcl_server.py
```

따라서 "누가 만든지 모르는 이상한 폴더"로 보고 바로 삭제하지 않도록 주의 부탁드립니다.
