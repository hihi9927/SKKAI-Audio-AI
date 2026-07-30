#!/bin/bash
# 모드4(finetuning ko 모델 / seg-commit) 평가 서버 실행
# baseline 대신 ko finetuning 가중치(Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged)를 --model로 직접 로드.
# always-commit/dot-commit 모두 끄고 SEG 토큰 기반 커밋만 사용.
# 사용법: bash serve_mode4.sh <scope> <tag>
#   예:   bash serve_mode4.sh sample run01
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$PAPER_RESULT_DIR/../../.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode4/$SCOPE/$TAG"

# HF 허브의 Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged가 401(Repository Not Found)로
# 막혀서 로컬 사본을 직접 로드하도록 고정함. 원격 리포 접근 복구되어도 로컬 사본이
# 동일 가중치라 실험 재현성엔 영향 없음.
python "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "$MODEL_PATH" \
  --disable-dot-commit \
  --no-vad \
  --port 8767 \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
# --disable-dot-commit: 이 모델명은 baseline 판정(_infer_dot_commit_default)에 걸리지 않아
# 기본값도 이미 False지만, seg-only 커밋임을 명시적으로 고정
