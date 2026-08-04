# shellcheck shell=bash
# <split> <scope> <tag> 3인자 해석 공통 로직. serve_mode*.sh / run_mode*.sh가 source 한다.
# SPLIT / SCOPE / TAG / FULL_TAG / TEST_DIR 를 채운다. 인자 shift는 호출자가 한다.
#
# 왜 공통 파일인가 — 서버는 로그를 {model}/{scope}/{tag}/logs/server.log에, 클라이언트는
# 결과를 {model}/{scope}/{tag}/에 쓴다. 두 스크립트가 tag를 다르게 만들면 로그와
# metric.json이 다른 폴더로 흩어진다. 그래서 규칙을 한 곳에만 둔다.
#
# 왜 tag에 split을 박는가 — 결과 경로에 split 차원이 없다. 같은 tag로 다른 split을 돌리면
# 덮어쓰기가 아니라 resume으로 합쳐진다 (run_concurrent_chapters.py의 load_resume_ids가
# 기존 file_id를 건너뛰고 prior_results를 집계에 그대로 더한다). 4개 split이 섞인
# metric.json이 오류 없이 조용히 만들어지므로, 사람이 tag 규칙을 지키는 데 맡기지 않는다.
#   test-clean + c16_run01 → testclean_c16_run01

_SPLIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SPLIT_DATASET_ROOT="$(cd "$_SPLIT_SCRIPT_DIR/../../../LibriSpeech" && pwd)"

SPLIT="${1:?split 필요 (test-clean|test-other|dev-clean|dev-other)}"
SCOPE="${2:?scope 필요 (예: sample, full)}"
TAG="${3:?tag 필요 (예: run01, c16_run01)}"

TEST_DIR="$_SPLIT_DATASET_ROOT/$SPLIT"
if [[ ! -d "$TEST_DIR" ]]; then
  echo "[split] 중단: split 디렉토리 없음: $TEST_DIR" >&2
  echo "  사용 가능: $(ls "$_SPLIT_DATASET_ROOT" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

# 이미 접두사가 붙은 tag를 그대로 다시 넘길 수 있어야 한다 (resume 시 로그에 찍힌
# 폴더명을 그대로 복사해 쓰는 경우).
_SPLIT_PREFIX="${SPLIT//-/}"
if [[ "$TAG" == "${_SPLIT_PREFIX}_"* ]]; then
  FULL_TAG="$TAG"
else
  FULL_TAG="${_SPLIT_PREFIX}_${TAG}"
fi
