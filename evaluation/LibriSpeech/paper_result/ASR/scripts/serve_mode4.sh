#!/bin/bash
# 모드4(finetuning en 모델 / seg-commit) 평가 서버 실행
# baseline 대신 en finetuning 가중치(Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged)를 --model로 직접 로드.
# LibriSpeech는 영어 데이터셋이므로 en-silence 가중치가 짝이다. ko-silence로 돌리면
# 반복 루프가 터져 finish 커밋에 수백 토큰이 덤프된다(08/04 mode4 full 실행 참고).
# always-commit/dot-commit 모두 끄고 SEG 토큰 기반 커밋만 사용.
# 사용법: bash serve_mode4.sh <split> <scope> <tag>
#   예:   bash serve_mode4.sh test-other full c16_run01   → mode4/full/testother_c16_run01
# run_mode4.sh에 같은 3인자를 넘겨야 서버 로그와 결과가 같은 폴더에 모인다.
# 종료는 `bash stop_server.sh 8767`로 할 것 — kill/pkill은 vLLM EngineCore를 남긴다.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_split.sh" "$@"   # SPLIT / SCOPE / TAG / FULL_TAG / TEST_DIR

PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STITY_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
RUN_DIR="$PAPER_RESULT_DIR/ASR/mode4/$SCOPE/$FULL_TAG"

# mode4 가중치. 기본은 HF 허브 리포지만, 접근이 401(Repository Not Found)로 막히는
# 환경이 있어 로컬 사본으로 갈아끼울 수 있게 열어둔다. 동일 가중치라 재현성엔 영향 없음.
#   예) MODE4_MODEL=/path/to/models/Qwen3-ASR-1.7B-en-silence-c80-merged
MODE4_MODEL="${MODE4_MODEL:-Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged}"

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
  --model "$MODE4_MODEL" \
  --disable-dot-commit \
  --no-vad \
  --port 8767 \
  --no-idle-shutdown \
  --log-file "$RUN_DIR/logs/server.log"
# --disable-dot-commit: 이 모델명은 baseline 판정(_infer_dot_commit_default)에 걸리지 않아
# 기본값도 이미 False지만, seg-only 커밋임을 명시적으로 고정
