# evaluation/

STiTy ASR+번역 파이프라인의 벤치마킹 하네스. 모든 평가는 별도로 실행 중인 WebSocket 서버에 연결해 진행한다.

트랙이 둘이다. **ASR 트랙**(WER/CER + FSL)은 아래 공통 패턴을 쓰고, **AST 트랙**(LAAL + BLEU)은
서버·클라이언트가 따로다 — @ast/README.md 참조.

## 공통 실행 패턴 (ASR 트랙)

```bash
# 터미널 1 — 평가 서버 (모든 데이터셋에서 공유)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py --no-idle-shutdown

# 터미널 2 — 벤치마크 클라이언트
python evaluation/{Dataset}/test_qwen3_{dataset}.py \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

LibriSpeech만 클라이언트가 `servers/` 아래에 있다 (`servers/test_qwen3_librispeech.py`). 나머지는 데이터셋 디렉토리 바로 아래.

## 데이터셋별 디렉토리

| 디렉토리 | 언어 | 주요 메트릭 | 특이사항 |
|---|---|---|---|
| `LibriSpeech/` | 영어 | WER + FSL 타이밍 | 주 벤치마크, 평가 서버가 여기 있음 |
| `AMI/` | 영어 | WER | 다화자 회의 코퍼스 |
| `DailyTalk/` | 한국어 | CER | 대화 음성 |
| `KsponSpeech/` | 한국어 | CER | 자유발화 |
| `KtelSpeech/` | 한국어 | CER | 전화 품질 음성 |
| `KokoroSpeech/` | 일본어 | CER | 단문 낭독 클립(tiny=308개), 파이프 구분 메타데이터 |
| `ReazonSpeech/` | 일본어 | CER | 단문 독립 클립, CSV 레이블 |
| `AliMeeting/` | 중국어 | CER | 다화자 회의, 화자별 TextGrid 레이블 |
| `(zh)RAMC/` | 중국어 | CER | 단문 발화 11,793개 / 화자 20명, TSV 레이블 |
| `(es)CIEMPIESS/` | 스페인어 | WER | 1,000개 단문 클립, 4개 서브셋(train/read/fm/description) |
| `smartturn/` | — | VAD F1/지연 | SmartTurn VAD 실험 (독립 트랙) |
| `ast/` | en→de/ko/ja/zh/es | **LAAL + BLEU** | AST 트랙. 데이터는 FLEURS(리포 밖 `~/datasets/fleurs`), 서버는 `streaming_websocket_server_ast.py`, 데이터셋은 manifest로 교체 (독립 트랙) |

## 데이터 추적 정책

오디오는 git에 없다. `AMI/`, `AliMeeting/`, `KokoroSpeech/`, `KtelSpeech/`, `ReazonSpeech/`, `(es)CIEMPIESS/`, `(zh)RAMC/`는 `.gitignore`로 통째 제외되어 있고, 테스트 스크립트와 레이블만 force-add로 추적한다. 새 파일을 커밋하려면 `git add -f`가 필요하다.

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
