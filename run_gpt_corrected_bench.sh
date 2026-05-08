#!/bin/bash
set -e
cd /home/ubuntu/STiTy

PYTHON=/home/ubuntu/miniconda3/envs/qwen3-asr/bin/python
MODEL=Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged
TEST_DIR=evaluation/LibriSpeech/LibriSpeech/test-other
SERVER_SCRIPT=evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py

mkdir -p logs

ts() { date "+%H:%M:%S"; }

echo "[$(ts)] eval 서버 시작 (GPT correction 활성화)..."
: "${OPENAI_API_KEY:?OPENAI_API_KEY 환경변수를 설정하세요}"
$PYTHON $SERVER_SCRIPT   --model $MODEL   --correction   --enforce-eager   --no-idle-shutdown   > logs/gpt_corrected_server_stdout.log 2>&1 &

SERVER_PID=$!
echo "[$(ts)] 서버 PID=$SERVER_PID, 포트 대기 중..."

for i in $(seq 1 60); do
  $PYTHON -c "import socket,sys; socket.create_connection(('localhost',8765),2); sys.exit(0)" 2>/dev/null && break
  echo "[$(ts)] 대기 중 ${i}/60..."
  sleep 5
done
echo "[$(ts)] 서버 준비 완료"

echo "[$(ts)] 테스트 시작 (548파일, GPT correction + eval 서버)..."
$PYTHON evaluation/LibriSpeech/servers/test_qwen3_librispeech.py   --test-dir $TEST_DIR   --host localhost   --port 8765   --model "finetuned(1.0.1)"   --scope gpt_corrected_test   --tag run_01   --limit 548   --fresh-start   --skip-protocol-smoke   --trailing-silence-ms 5500   > logs/gpt_corrected_test_stdout.log 2>&1

echo "[$(ts)] 테스트 완료"
kill $SERVER_PID 2>/dev/null || true
echo "[$(ts)] 완료"
