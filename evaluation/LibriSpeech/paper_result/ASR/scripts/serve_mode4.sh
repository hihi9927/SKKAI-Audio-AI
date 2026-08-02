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
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode4/$SCOPE/$TAG"

python "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged" \
  --disable-dot-commit \
  --no-vad \
  --port 8767 \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
# --disable-dot-commit: 이 모델명은 baseline 판정(_infer_dot_commit_default)에 걸리지 않아
# 기본값도 이미 False지만, seg-only 커밋임을 명시적으로 고정
