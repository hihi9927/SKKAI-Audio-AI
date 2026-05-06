#!/usr/bin/env bash
# chunk size sweep: 2.0 → 1.5 → 1.0 → 0.5 초
# 사용법:
#   bash run_chunk_sweep.sh
#   bash run_chunk_sweep.sh --model finetuned --scope full --limit 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_qwen3_librispeech.py"
SERVER_SCRIPT="$SCRIPT_DIR/streaming_websocket_server_fcl.py"

# ── 기본값 ────────────────────────────────────────────────────────────────────
SERVER_MODEL="/home/ubuntu/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"
MODEL_TAG="finetuned(1.0.1)"
SCOPE="chunk_size_test"
LIMIT=""
EXTRA_TEST_ARGS=""
CHUNK_SIZES=(2.0 1.5 1.0 0.5)

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-model) SERVER_MODEL="$2"; shift 2 ;;
        --model)        MODEL_TAG="$2";    shift 2 ;;
        --scope)        SCOPE="$2";        shift 2 ;;
        --limit)        LIMIT="$2";        shift 2 ;;
        --chunks)       IFS=',' read -ra CHUNK_SIZES <<< "$2"; shift 2 ;;
        *) EXTRA_TEST_ARGS="$EXTRA_TEST_ARGS $1"; shift ;;
    esac
done

LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"

# ── sweep ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Chunk size sweep: ${CHUNK_SIZES[*]}"
echo "  model  : $SERVER_MODEL"
echo "  tag    : $MODEL_TAG / $SCOPE"
echo "============================================================"

for CS in "${CHUNK_SIZES[@]}"; do
    TAG="chunk${CS}"
    echo ""
    echo "------------------------------------------------------------"
    echo "  chunk_size=${CS}s  →  tag=${TAG}"
    echo "------------------------------------------------------------"

    python "$TEST_SCRIPT" \
        --auto-server \
        --server-script "$SERVER_SCRIPT" \
        --server-model  "$SERVER_MODEL" \
        --server-args   "--chunk-size ${CS} --enforce-eager --no-idle-shutdown" \
        --model  "$MODEL_TAG" \
        --scope  "$SCOPE" \
        --tag    "$TAG" \
        --description "chunk_size=${CS}s sweep" \
        --fresh-start \
        $LIMIT_ARG \
        $EXTRA_TEST_ARGS

    echo "  ✓ chunk_size=${CS}s 완료"
done

echo ""
echo "============================================================"
echo "  sweep 완료. 결과: results/${MODEL_TAG}/${SCOPE}/"
echo "============================================================"

if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
    echo "  EC2 인스턴스 종료 중..."
    sudo shutdown -h now
fi
