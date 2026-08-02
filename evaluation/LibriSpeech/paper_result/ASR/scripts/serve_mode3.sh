#!/bin/bash
# 모드3(rule-based/dot-commit) 평가 서버 실행 (baseline(1.0.0) 고정)
# mode2에서 always-commit(2초 고정 청킹/커밋)만 빼고 dot 기반 rule-based commit을 켠 버전
# 사용법: bash serve_mode3.sh <scope> <tag>
#   예:   bash serve_mode3.sh sample run01
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# --- env 가드: conda env 'stity'가 아니면 중단 (우회: STITY_SKIP_ENV_CHECK=1) ---
STITY_ENV_NAME="${STITY_ENV_NAME:-stity}"
if [[ "${STITY_SKIP_ENV_CHECK:-0}" != "1" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$STITY_ENV_NAME" ]]; then
    echo "[env-guard] 중단: conda env '$STITY_ENV_NAME'에서 실행해야 합니다 (현재: ${CONDA_DEFAULT_ENV:-<없음>})" >&2
    echo "  해결: source ~/miniforge3/etc/profile.d/conda.sh && conda activate $STITY_ENV_NAME" >&2
    exit 1
  fi
  if [[ "$(command -v python || true)" != "${CONDA_PREFIX:-}/bin/python" ]]; then
    echo "[env-guard] 중단: python이 env '$STITY_ENV_NAME' 것이 아닙니다" >&2
    echo "  실제: $(command -v python || echo '<없음>') / 기대: ${CONDA_PREFIX:-<없음>}/bin/python" >&2
    exit 1
  fi
  echo "[env-guard] OK — conda env '$CONDA_DEFAULT_ENV' ($(python -V 2>&1))" >&2
fi
# --- env 가드 끝 ---

PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode3/$SCOPE/$TAG"

python "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "Qwen/Qwen3-ASR-1.7B" \
  --enable-dot-commit \
  --no-vad \
  --port 8766 \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
