#!/usr/bin/env bash
# partial 스트리밍 데모용 ASR 서버 기동.
#
# 이 워크트리의 코드로 돌린다. qwen_asr 는 원본 저장소에 editable 로 설치돼 있어서
# PYTHONPATH 로 이 워크트리를 앞에 세워야 여기서 고친 코드가 쓰인다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

: "${PYTHON:=/home/mobility/STiTy/.venv/bin/python}"
: "${PORT:=8766}"
# 번역기와 ASR 이 GPU 를 나눠 쓴다. MADLAD-3B 는 fp16 으로 약 6GB 를 잡으므로
# ASR 쪽 점유를 0.42 로 낮춰 자리를 비운다(ASR 12GB, KV 캐시 4.88GB = 45,696 토큰).
# NLLB 로 되돌리려면 TRANSLATION_MODEL 만 바꾸면 되고, 그때는 GPU_UTIL 을 올려도 된다.
: "${TRANSLATION_MODEL:=google/madlad400-3b-mt}"
: "${GPU_UTIL:=0.42}"

export PYTHONPATH="$REPO/Qwen3-ASR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

exec "$PYTHON" -u Qwen3-ASR/examples/streaming_websocket_server.py \
  --port "$PORT" --no-idle-shutdown --enforce-eager \
  --gpu-memory-utilization "$GPU_UTIL" \
  --local-translation --local-translation-model "$TRANSLATION_MODEL" \
  > "$HERE/server.log" 2>&1
