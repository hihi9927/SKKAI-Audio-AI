#!/bin/bash
# 모드3(rule-based/dot-commit) 벤치마크 클라이언트 실행 (baseline(1.0.0) 고정)
# serve_mode3.sh로 서버를 먼저 띄운 뒤, 같은 scope/tag로 실행할 것
# 사용법: bash run_mode3.sh <scope> <tag> [--clients N] [추가 옵션...]
#   순차: bash run_mode3.sh sample run01
#   순차: bash run_mode3.sh full run01 --limit 50
#   동시: bash run_mode3.sh full concurrent16_run01 --clients 16
#
# --clients 유무로 러너가 갈리므로 쓸 수 있는 추가 옵션도 달라진다.
#   순차 전용: --limit --random-sample --auto-server --correction --gpt-translation ...
#   동시 전용: --max-chapters --dry-run
#   공통:      --fresh-start --host --policy --chunk-size-ms --trailing-silence-ms ...
#
# trailing-silence는 8000ms를 쓴다(mode2/mode4는 5500 유지). 확정 게이트는 누적 오디오가
# chunk_size 배수에 도달할 때만 판정하므로, 무음이 짧으면 마지막 문장이 확정되기 전에
# 스트림이 끊겨 finish 커밋으로 빠진다(실측: 0.11초 모자라 finish 발생).
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01, concurrent16_run01)}"
shift 2

# --- 실행 모드 선택 -----------------------------------------------------------
# --clients N 을 주면 동시 접속 부하 벤치마크(챕터 큐 + 워커 N개)로 돌고,
# 없으면 기존과 동일하게 파일을 순차 처리한다.
NUM_CLIENTS=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clients)
      NUM_CLIENTS="${2:?--clients 뒤에 클라이언트 수가 필요합니다}"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STITY_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# --- env 가드 ---------------------------------------------------------------
# 평가 환경은 머신마다 다르다: 로컬은 venv($STITY_ROOT/.venv), 원격 GPU 서버는 conda env
# 'stity'. 그래서 특정 도구를 요구하지 않고 "필요 패키지가 import되는 파이썬"만 확인한다.
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
  if ! PYTHONPATH= "$STITY_PYTHON" -c "import websockets, soundfile, jiwer" 2>/dev/null; then
    echo "[env-guard] 중단: $STITY_PYTHON 에서 websockets/soundfile/jiwer import 실패" >&2
    exit 1
  fi
  echo "[env-guard] OK — $STITY_PYTHON ($(PYTHONPATH= "$STITY_PYTHON" -V 2>&1))" >&2
fi
# --- env 가드 끝 -------------------------------------------------------------

# 벤치마크 시작 전, 지금 떠 있는 서버가 진짜 모드3 설정인지 확인.
# 실패하면 set -e 에 의해 여기서 바로 중단됨 (잘못된 설정으로 실험 도는 것 방지).
PYTHONPATH= "$STITY_PYTHON" "$PAPER_RESULT_DIR/check_server_config.py" \
  --port 8766 \
  --expect model=Qwen/Qwen3-ASR-1.7B \
  --expect always_commit=false \
  --expect enable_dot_commit=true \
  --expect no_vad=true

if [[ -n "$NUM_CLIENTS" ]]; then
  RUNNER="$SCRIPT_DIR/../../../run_concurrent_chapters.py"
  MODE_ARGS=(--num-clients "$NUM_CLIENTS" --send-interval-ms 200)
  DESC_SUFFIX=" / 동시 클라이언트 ${NUM_CLIENTS}명, 챕터 큐"
else
  RUNNER="$SCRIPT_DIR/../../../servers/test_qwen3_librispeech.py"
  MODE_ARGS=()
  DESC_SUFFIX=""
fi

PYTHONPATH= "$STITY_PYTHON" "$RUNNER" \
  --test-dir "$STITY_ROOT/evaluation/LibriSpeech/LibriSpeech/test-other" \
  --model mode3 \
  --scope "$SCOPE" \
  --tag "$TAG" \
  --results-root "$PAPER_RESULT_DIR/ASR" \
  --port 8766 \
  --target-lang "" \
  --trailing-silence-ms 8000 \
  --chunk-size-ms 200 \
  --description "mode3 (rule-based/dot-commit) / baseline(1.0.0) / ASR only, translation off${DESC_SUFFIX}" \
  "${MODE_ARGS[@]}" \
  "${ARGS[@]}"
# 번역 비활성화: targetLang을 빈 문자열로 보내면 서버가 Google Translate 호출을 건너뜀
# (Qwen3-ASR/examples/streaming_websocket_server.py의 google_translate_async: `if not target_lang: return`)
# --gpt-translation / --correction 은 기본값이 이미 off라 별도 지정 불필요
