#!/bin/bash
# 모드3(rule-based/dot-commit) 평가 서버 실행 (baseline(1.0.0) 고정)
# mode2에서 always-commit(2초 고정 청킹/커밋)만 빼고 dot 기반 rule-based commit을 켠 버전
# 사용법: bash serve_mode3.sh <scope> <tag>
#   예:   bash serve_mode3.sh sample run01
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode3/$SCOPE/$TAG"

python "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "Qwen/Qwen3-ASR-1.7B" \
  --enable-dot-commit \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
