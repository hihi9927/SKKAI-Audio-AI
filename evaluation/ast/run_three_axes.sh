#!/bin/bash
# AST 세 축(dot / seg / always)을 **순차** 실행한다.
#
#   bash evaluation/ast/run_three_axes.sh [manifest] [limit]
#
# GPU 에 vLLM 인스턴스가 하나만 올라가므로 축마다 서버를 띄우고 → 클라이언트를 돌리고 →
# 서버를 내린다. 서버 종료는 반드시 stop_server.sh 로 한다(pkill 은 EngineCore 를 남긴다).
#
# 축 정의는 논문용 mode2/3/4 (paper_result/ASR/scripts/serve_mode*.sh)를 따른다:
#   dot    = baseline 모델 + dot 커밋(확정 게이트)      ← mode3
#   seg    = en 파인튜닝 모델 + SEG 커밋                 ← mode4
#   always = baseline 모델 + 매 청크 커밋                ← mode2
# **dot/always 는 baseline, seg 만 파인튜닝 가중치다.** 정책과 모델이 함께 바뀌므로
# 결과를 "정책 차이"로만 읽으면 안 된다. 모델을 고정해 비교하려면 MODEL_* 를 맞출 것.
#
# 공통 조건: --no-vad, 2초 청크, 클라이언트 16병렬, 뒤 침묵 500ms.
# 침묵을 짧게 두는 근거는 ast/README.md 의 "VAD off 로 돌릴 때" 절 참고
# (AST 서버가 종료 시 최종 디코딩 + 번역 드레인을 보완하므로 침묵을 늘릴 필요가 없다).

set -u

REPO="/home/mobility/STiTy"
PY="$REPO/.venv/bin/python"
STOP="$REPO/evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh"
PORT=8765
# **GPU 점유 상한.** vLLM 기본 0.8 은 24GB 카드에서 19.2GB 를 선점한다 — 1.7B 모델이
# 실제로 쓰는 양이 아니라 "남는 걸 다 잡아두는" 설계다. 2026-08-28 00:21 에 이것 때문에
# 같은 카드에서 돌던 autoseg 루프(CometKiwi+NLI, 4.1GB)가 CUDA OOM 으로 죽었다
# (EngineCore pid 149462 가 19.28GiB 점유, 여유 9.5MiB). 다른 트랙들은 이미 0.6~0.65 를
# 쓴다. 0.5 면 12GB 로 이 모델엔 충분하고 나머지 12GB 를 비워 둔다.
# 카드를 혼자 쓸 때는 `GPU_UTIL=0.8` 로 올리면 된다.
GPU_UTIL="${GPU_UTIL:-0.5}"

# 기본 manifest 는 리포에 없다 — build_manifest_covost2.py 로 만든다(그 파일 상단 주석에 명령).
MANIFEST="${1:-$REPO/evaluation/ast/manifests/covost2_en-de_spk.jsonl}"
LIMIT="${2:-}"

MODEL_BASELINE="Qwen/Qwen3-ASR-1.7B"
MODEL_FINETUNED="$REPO/models/Qwen3-ASR-1.7B-en-silence-c80-merged"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/evaluation/ast/results/_runlogs/$STAMP"
mkdir -p "$LOGDIR"

CLIENT_ARGS=(--dataset CoVoST2-spk --src-lang en --target-lang de
             --port "$PORT" --clients 16 --trailing-silence-ms 500
             --scope full --tag "$STAMP")
[ -n "$LIMIT" ] && CLIENT_ARGS+=(--limit "$LIMIT")

echo "manifest : $MANIFEST ($(wc -l < "$MANIFEST") 발화)"
echo "로그      : $LOGDIR"
echo "시작      : $(date '+%F %T')"
echo

run_axis() {
  local name="$1" model="$2"; shift 2
  local server_args=("$@")
  local slog="$LOGDIR/${name}_server.log"
  local clog="$LOGDIR/${name}_client.log"

  echo "═══ [$name] 서버 기동 ═══ $(date '+%T')"
  "$PY" "$REPO/evaluation/streaming_websocket_server_ast.py" \
      --model "$model" --no-vad --chunk-size 2.0 \
      --port "$PORT" --no-idle-shutdown \
      --gpu-memory-utilization "$GPU_UTIL" "${server_args[@]}" > "$slog" 2>&1 &
  local spid=$!

  # 기동 대기 (최대 10분). **로그 줄이 아니라 포트 접속으로 판정한다.**
  # 서버는 "Starting WebSocket server" 를 찍은 뒤에 워밍업을 하고, 그게 끝나야
  # websockets.serve() 로 포트를 연다. 로그 줄만 보고 클라이언트를 띄우면 워밍업
  # 도중에 붙어 Connect call failed 로 16 워커가 전부 즉사한다(실측).
  local waited=0
  until "$PY" -c "import socket,sys; socket.create_connection(('127.0.0.1',$PORT),2).close()" 2>/dev/null; do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "[$name] 서버 기동 실패 — $slog 확인"; tail -5 "$slog"; return 1
    fi
    sleep 5; waited=$((waited+5))
    if [ "$waited" -ge 600 ]; then echo "[$name] 기동 타임아웃"; return 1; fi
  done
  echo "═══ [$name] 서버 준비 (${waited}초) — 클라이언트 시작 ═══"

  "$PY" "$REPO/evaluation/ast/test_ast.py" \
      --manifest "$MANIFEST" --model "$name" "${CLIENT_ARGS[@]}" 2>&1 | tee "$clog"

  # 0발화로 끝났으면 축이 통째로 실패한 것이다 — 조용히 다음 축으로 넘어가지 않는다.
  if grep -q "발화 0개" "$clog"; then
    echo "!!! [$name] 발화 0개로 종료 — 서버 연결/기동 문제. $clog 확인"
  fi

  echo "═══ [$name] 서버 종료 ═══ $(date '+%T')"
  bash "$STOP" "$PORT" 2>&1 | tail -2
  sleep 5
  echo
}

# 요청 순서: dot → seg → always
run_axis dot    "$MODEL_BASELINE"  --enable-dot-commit --no-rep-dedup
run_axis seg    "$MODEL_FINETUNED" --disable-dot-commit
run_axis always "$MODEL_BASELINE"  --always-commit --disable-dot-commit

echo "═══════════════════ 요약 ═══════════════════ $(date '+%F %T')"
for axis in dot seg always; do
  echo "── $axis"
  grep -E "발화 [0-9]+개|LAAL      :|LAAL_CA   :|BLEU      :|커밋 사유|실시간 대비" \
       "$LOGDIR/${axis}_client.log" 2>/dev/null | sed 's/^.*INFO - /   /'
done
echo
echo "결과: $REPO/evaluation/ast/results/CoVoST2-spk/{dot,seg,always}/full/$STAMP/metric.json"
