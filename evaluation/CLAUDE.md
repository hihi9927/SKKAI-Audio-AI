# evaluation/

STiTy ASR+번역 파이프라인의 벤치마킹 하네스. 모든 평가는 별도로 실행 중인 WebSocket 서버에 연결해 진행한다.

## 공통 실행 패턴

```bash
# 터미널 1 — 평가 서버 (모든 데이터셋에서 공유)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py --no-idle-shutdown

# 터미널 2 — 벤치마크 클라이언트
python evaluation/{Dataset}/test_qwen3_{dataset}.py \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 데이터셋별 디렉토리

| 디렉토리 | 언어 | 주요 메트릭 | 특이사항 |
|---|---|---|---|
| `LibriSpeech/` | 영어 | WER + FSL 타이밍 | 주 벤치마크, 평가 서버가 여기 있음 |
| `AMI/` | 영어 | WER | 다화자 회의 코퍼스 |
| `DailyTalk/` | 한국어 | CER | 대화 음성 |
| `KsponSpeech/` | 한국어 | CER | 자유발화 |
| `KtelSpeech/` | 한국어 | CER | 전화 품질 음성 |
| `smartturn/` | — | VAD F1/지연 | SmartTurn VAD 실험 (독립 트랙) |

## 결과 디렉토리 구조

```
evaluation/{Dataset}/results/{model}/{scope}/{tag}/
├── metric.json
├── meta.json
├── description.txt
├── plots/
└── logs/
```

## 주요 동작

- `--tag run_01` 로 부분 완료된 실행 재개 가능
- `--fresh-start` 로 초기화 후 재시작
- `--auto-server` 로 서버 자동 시작 (단, `dot commit` 동작이 baseline/finetuned 간 다름)

자세한 CLI 레퍼런스: @TESTING_MANUAL.md
