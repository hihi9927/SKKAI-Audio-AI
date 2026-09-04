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

export PYTHONPATH="$REPO/Qwen3-ASR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

exec "$PYTHON" -u Qwen3-ASR/examples/streaming_websocket_server.py \
  --port "$PORT" --no-idle-shutdown --enforce-eager \
  --gpu-memory-utilization 0.75 \
  --local-translation \
  > "$HERE/server.log" 2>&1
