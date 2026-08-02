#!/bin/bash
# 모드2(always-commit) 벤치마크 클라이언트 실행 (baseline(1.0.0) 고정)
# serve_mode2.sh로 서버를 먼저 띄운 뒤, 같은 scope/tag로 실행할 것
# 사용법: bash run_mode2.sh <scope> <tag> [추가 옵션...]
#   예:   bash run_mode2.sh sample run01
#   예:   bash run_mode2.sh full run01 --limit 50
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"
shift 2

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

# 벤치마크 시작 전, 지금 떠 있는 서버가 진짜 모드2 설정인지 확인.
# 실패하면 set -e 에 의해 여기서 바로 중단됨 (잘못된 설정으로 실험 도는 것 방지).
python "$PAPER_RESULT_DIR/check_server_config.py" \
  --port 8765 \
  --expect model=Qwen/Qwen3-ASR-1.7B \
  --expect always_commit=true \
  --expect enable_dot_commit=false \
  --expect no_vad=true

python "$SCRIPT_DIR/../../../servers/test_qwen3_librispeech.py" \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model mode2 \
  --scope "$SCOPE" \
  --tag "$TAG" \
  --results-root "$PAPER_RESULT_DIR/ASR" \
  --port 8765 \
  --target-lang "" \
  --trailing-silence-ms 5500 \
  --chunk-size-ms 200 \
  --description "mode2 (always-commit) / baseline(1.0.0) / ASR only, translation off" \
  "$@"
# 번역 비활성화: targetLang을 빈 문자열로 보내면 서버가 Google Translate 호출을 건너뜀
# (Qwen3-ASR/examples/streaming_websocket_server.py의 google_translate_async: `if not target_lang: return`)
# --gpt-translation / --correction 은 기본값이 이미 off라 별도 지정 불필요
