#!/bin/bash
# 모드2(always-commit) 평가 서버 실행 (baseline(1.0.0) 고정)
# 사용법: bash serve_mode2.sh <scope> <tag>
#   예:   bash serve_mode2.sh sample run01
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode2/$SCOPE/$TAG"

python "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "Qwen/Qwen3-ASR-1.7B" \
  --always-commit \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
