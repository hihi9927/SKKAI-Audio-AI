#!/bin/bash
# 모드4(finetuning ko 모델 / seg-commit) 벤치마크 클라이언트 실행
# serve_mode4.sh로 서버를 먼저 띄운 뒤, 같은 scope/tag로 실행할 것
# 사용법: bash run_mode4.sh <scope> <tag> [추가 옵션...]
#   예:   bash run_mode4.sh sample run01
#   예:   bash run_mode4.sh full run01 --limit 50
set -e

SCOPE="${1:?scope 필요 (예: sample, full)}"
TAG="${2:?tag 필요 (예: run01)}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_RESULT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 벤치마크 시작 전, 지금 떠 있는 서버가 진짜 모드4 설정인지 확인.
# 실패하면 set -e 에 의해 여기서 바로 중단됨 (잘못된 설정으로 실험 도는 것 방지).
python "$PAPER_RESULT_DIR/check_server_config.py" \
  --port 8767 \
  --expect model=Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged \
  --expect always_commit=false \
  --expect enable_dot_commit=false \
  --expect no_vad=true

python "$SCRIPT_DIR/../../../servers/test_qwen3_librispeech.py" \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model mode4 \
  --scope "$SCOPE" \
  --tag "$TAG" \
  --results-root "$PAPER_RESULT_DIR/ASR" \
  --port 8767 \
  --target-lang "" \
  --trailing-silence-ms 5500 \
  --chunk-size-ms 200 \
  --description "mode4 (seg-commit) / finetuning ko(Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged) / ASR only, translation off" \
  "$@"
# 번역 비활성화: targetLang을 빈 문자열로 보내면 서버가 Google Translate 호출을 건너뜀
# (Qwen3-ASR/examples/streaming_websocket_server.py의 google_translate_async: `if not target_lang: return`)
# --gpt-translation / --correction 은 기본값이 이미 off라 별도 지정 불필요
