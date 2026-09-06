# evaluation/

STiTy ASR+번역 파이프라인의 벤치마킹 하네스. 모든 평가는 별도로 실행 중인 WebSocket 서버에 연결해 진행한다.

트랙이 둘이다. **ASR 트랙**(WER/CER + FSL)과 **AST 트랙**(LAAL + BLEU). **서버는 하나를 공유하고**
(`evaluation/streaming_websocket_server_ast.py`) 클라이언트만 트랙·데이터셋마다 다르다 —
AST 쪽은 @ast/README.md 참조.

상속은 `Qwen3-ASR/examples/streaming_websocket_server.py`(프로덕션) →
`LibriSpeech/servers/streaming_websocket_server_fsl.py`(FSL 타이밍) →
`streaming_websocket_server_ast.py`(LAAL 필드 + 번역 계측) 순이다. 맨 아래만 띄우면 위 둘의
기능이 다 따라온다. AST 전용 동작은 기본이 꺼져 있어(`--ast-hide-seg`) 인자를 안 주면 FSL 서버와
같게 돈다.

## 공통 실행 패턴 (ASR 트랙)

```bash
# 터미널 1 — 평가 서버 (모든 데이터셋에서 공유)
python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown

# 터미널 2 — 벤치마크 클라이언트
python evaluation/{Dataset}/test_qwen3_{dataset}.py \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

LibriSpeech만 클라이언트가 `servers/` 아래에 있다 (`servers/test_qwen3_librispeech.py`). 나머지는 데이터셋 디렉토리 바로 아래.

## 데이터셋별 디렉토리

| 디렉토리 | 언어 | 주요 메트릭 | 특이사항 |
|---|---|---|---|
| `LibriSpeech/` | 영어 | WER + FSL 타이밍 | 주 벤치마크, 평가 서버가 여기 있음 |
| `DailyTalk/` | 한국어 | CER | 대화 음성 |
| `KsponSpeech/` | 한국어 | CER | 자유발화 |
| `ast/` | en→de/ko/ja/zh/es | **LAAL + BLEU** | AST 트랙. 데이터는 FLEURS(리포 밖 `~/datasets/fleurs`), 서버는 `streaming_websocket_server_ast.py`, 데이터셋은 manifest로 교체 (독립 트랙) |

## 데이터 추적 정책

오디오는 git에 없다. 결과 디렉토리(`results/`)도 마찬가지다 — 로컬 디스크에만 남는다.

지금 있는 데이터셋은 LibriSpeech, DailyTalk, KsponSpeech, ast 넷이다. 예전에 돌렸던 다른 언어
벤치마크들의 측정치는 @ARCHIVED_DATASETS_METRICS.md 에 표로 남아 있다.

## 결과 디렉토리 구조

```
evaluation/{Dataset}/results/{model}/{scope}/{tag}/
├── metric.json
├── meta.json
├── description.txt
├── plots/
└── logs/
```

AST 트랙은 데이터셋 이름이 한 단계 더 들어간다: `evaluation/ast/results/{dataset}/{model}/{scope}/{tag}/`.

## 주요 동작

- `--tag run_01` 로 부분 완료된 실행 재개 가능
- `--fresh-start` 로 초기화 후 재시작
- `--auto-server` 로 서버 자동 시작 (단, `dot commit` 기본값이 baseline/finetuned 간 다름)

## 논문용 모드 비교

`LibriSpeech/paper_result/ASR/scripts/`에 커밋 정책별 서버/실행 스크립트가 있다 —
**mode2** always-commit(포트 8765), **mode3** dot-commit + 확정 게이트(8766), **mode4** en 파인튜닝 가중치 + SEG 커밋(8767).
서버 종료는 반드시 `bash stop_server.sh <port>` (kill/pkill은 vLLM EngineCore를 남긴다).

자세한 CLI 레퍼런스: @TESTING_MANUAL.md
