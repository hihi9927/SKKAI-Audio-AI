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
STITY_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode3/$SCOPE/$TAG"

# --- env 가드 ---------------------------------------------------------------
# 평가 환경은 머신마다 다르다: 로컬은 venv($STITY_ROOT/.venv), 원격 GPU 서버는 conda env
# 'stity'. 그래서 특정 도구를 요구하지 않고 "vllm이 import되는 파이썬"만 확인한다.
# 또한 ROS humble이 PYTHONPATH를 오염시키는데 PYTHONPATH가 site-packages보다 먼저
# 검색되므로, 파이썬을 호출할 때는 PYTHONPATH를 비워야 한다.
# 우회: STITY_SKIP_ENV_CHECK=1, 인터프리터 직접 지정: STITY_PYTHON=/path/to/python
if [[ -z "${STITY_PYTHON:-}" ]]; then
  if [[ -x "$STITY_ROOT/.venv/bin/python" ]]; then
    STITY_PYTHON="$STITY_ROOT/.venv/bin/python"       # 로컬 venv
  else
    STITY_PYTHON="$(command -v python || command -v python3 || true)"  # 활성 conda env 등
  fi
fi
if [[ "${STITY_SKIP_ENV_CHECK:-0}" != "1" ]]; then
  if [[ ! -x "$STITY_PYTHON" ]]; then
    echo "[env-guard] 중단: 파이썬을 찾을 수 없습니다: $STITY_PYTHON" >&2
    echo "  STITY_PYTHON 환경변수로 경로를 지정하거나 STITY_SKIP_ENV_CHECK=1로 우회하세요." >&2
    exit 1
  fi
  if ! PYTHONPATH= "$STITY_PYTHON" -c "import vllm, websockets" 2>/dev/null; then
    echo "[env-guard] 중단: $STITY_PYTHON 에서 vllm/websockets import 실패" >&2
    exit 1
  fi
  echo "[env-guard] OK — $STITY_PYTHON ($(PYTHONPATH= "$STITY_PYTHON" -V 2>&1))" >&2
fi
# --- env 가드 끝 -------------------------------------------------------------

PYTHONPATH= "$STITY_PYTHON" "$SCRIPT_DIR/../../../servers/streaming_websocket_server_fsl.py" \
  --model "Qwen/Qwen3-ASR-1.7B" \
  --enable-dot-commit \
  --no-vad \
  --port 8766 \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
