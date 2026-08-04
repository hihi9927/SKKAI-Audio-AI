#!/bin/bash
# mode2/3/4 × LibriSpeech 4개 split(test-clean/other, dev-clean/other) 전체를 순차 실행한다.
# 사용법: bash run_all_splits.sh [tag]        (기본 tag: c16_run01)
#   tmux 예: tmux new -d -s stity_eval "bash run_all_splits.sh"
#
# 왜 순차인가 — GPU가 1장(RTX 4090 24GB)이고 서버 하나가 20GB를 쓴다. 모드를 동시에
# 띄울 수 없다. 실행 12회 × 약 37분 ≈ 7.5시간.
#
# 왜 실행마다 서버를 새로 띄우는가 — 서버 설정은 split과 무관하므로 한 모드에서
# 4개 split을 한 서버로 돌 수도 있다. 그러면 서버 로그(DOT-PENDING / SEG-IN-TEXT /
# COMMIT-SKIP)가 한 파일에 섞여 사후 분석에서 split을 분리할 수 없다. 08/04에 로그가
# 섞여 유실된 적이 있어 재기동 비용(회당 약 1분)을 내고 분리한다.
#
# 실패해도 멈추지 않는다 — 한 실행이 죽어도 나머지가 밤새 돌아야 하므로 기록만 남기고
# 다음으로 넘어간다. 마지막에 성공/실패 요약을 출력한다.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STITY_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
STITY_PYTHON="${STITY_PYTHON:-$STITY_ROOT/.venv/bin/python}"

TAG="${1:-c16_run01}"
SCOPE="full"
CLIENTS=16
SPLITS=(test-clean test-other dev-clean dev-other)
MODES=(2 3 4)
port_of() { case "$1" in 2) echo 8765 ;; 3) echo 8766 ;; 4) echo 8767 ;; esac; }

SERVER_BOOT_TIMEOUT=600   # vLLM 로딩 + 워밍업. 실측 45초지만 콜드 캐시를 감안해 넉넉히.
GPU_FREE_TIMEOUT=120      # 다음 서버를 띄우기 전 VRAM이 실제로 빠질 때까지 기다린다.
GPU_FREE_MIB=2000         # 이 아래면 "비었다"로 본다 (디스플레이 등 잔여분 존재).

LOG_ROOT="$ASR_DIR/logs"
mkdir -p "$LOG_ROOT"
MAIN_LOG="$LOG_ROOT/run_all_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

# 오케스트레이터가 죽어도 서버가 GPU를 물고 남지 않게 한다. setsid로 띄우므로
# 프로세스 그룹이 분리돼 있어 자동으로 따라 죽지 않는다.
CURRENT_PORT=""
cleanup() {
  if [[ -n "$CURRENT_PORT" ]]; then
    log "인터럽트 — 포트 $CURRENT_PORT 서버 정리"
    bash "$SCRIPT_DIR/stop_server.sh" "$CURRENT_PORT" >>"$MAIN_LOG" 2>&1
  fi
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1" i
  for ((i = 0; i < SERVER_BOOT_TIMEOUT; i += 5)); do
    sleep 5
    ss -ltn 2>/dev/null | grep -q ":${port}\b" && return 0
  done
  return 1
}

wait_gpu_free() {
  local i used
  for ((i = 0; i < GPU_FREE_TIMEOUT; i += 5)); do
    used="$(gpu_used_mib)"
    [[ -n "$used" && "$used" -lt "$GPU_FREE_MIB" ]] && return 0
    sleep 5
  done
  log "  경고: VRAM이 ${GPU_FREE_TIMEOUT}초 안에 회수되지 않음 (현재 $(gpu_used_mib) MiB)"
  return 1
}

report_metric() {
  # 실행 직후 핵심 수치만 뽑아 메인 로그에 남긴다. 밤새 돌린 뒤 로그만 보고
  # 이상 유무를 판단할 수 있어야 한다.
  local run_dir="$1"
  PYTHONPATH= "$STITY_PYTHON" - "$run_dir" <<'PY' 2>/dev/null || echo "  (metric.json 읽기 실패)"
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "metric.json"
o = json.load(open(p))["overall"]
c = o.get("commit_stats", {})
ratios = c.get("ratios", {})
hot = " ".join(f"{k}={v:.1%}" for k, v in ratios.items() if v)
print(f"  파일 {o['num_files']} · WER {o['wer']:.2%} · FSL {o['avg_fsl_sec']:.4f}s"
      f" · FTL {o['first_token_latency']:.3f}s · 커밋 {c.get('total', 0)} [{hot}]")
PY
}

log "=========================================================="
log "mode ${MODES[*]} × split ${SPLITS[*]} — scope=$SCOPE tag=$TAG clients=$CLIENTS"
log "총 12회 · 예상 7~8시간 · 로그: $MAIN_LOG"
log "=========================================================="

declare -a SUMMARY=()
RUN_NO=0

for MODE in "${MODES[@]}"; do
  PORT="$(port_of "$MODE")"
  for SPLIT in "${SPLITS[@]}"; do
    RUN_NO=$((RUN_NO + 1))
    PREFIX="${SPLIT//-/}"
    RUN_DIR="$ASR_DIR/mode$MODE/$SCOPE/${PREFIX}_$TAG"
    START_TS=$(date +%s)

    log ""
    log "▶ [$RUN_NO/12] mode$MODE / $SPLIT → ${PREFIX}_$TAG (포트 $PORT)"
    mkdir -p "$RUN_DIR/logs"

    # setsid: 서버에 독립 프로세스 그룹을 준다. stop_server.sh가 그룹 전체를 죽여
    # vLLM EngineCore까지 회수하는데, 같은 그룹이면 오케스트레이터가 함께 죽는다.
    setsid bash "$SCRIPT_DIR/serve_mode$MODE.sh" "$SPLIT" "$SCOPE" "$TAG" \
      >"$RUN_DIR/logs/serve_stdout.log" 2>&1 &
    CURRENT_PORT="$PORT"

    if ! wait_for_port "$PORT"; then
      log "  ✗ 서버 기동 실패 (${SERVER_BOOT_TIMEOUT}초 초과) — serve_stdout.log 확인"
      SUMMARY+=("✗ mode$MODE/$SPLIT  서버 기동 실패")
      bash "$SCRIPT_DIR/stop_server.sh" "$PORT" >>"$MAIN_LOG" 2>&1
      CURRENT_PORT=""
      wait_gpu_free
      continue
    fi
    log "  서버 기동 완료 ($(( $(date +%s) - START_TS ))초)"

    bash "$SCRIPT_DIR/run_mode$MODE.sh" "$SPLIT" "$SCOPE" "$TAG" --clients "$CLIENTS" \
      >>"$RUN_DIR/logs/client.log" 2>&1
    RC=$?
    ELAPSED=$(( $(date +%s) - START_TS ))

    if [[ $RC -eq 0 ]]; then
      log "  ✓ 완료 ($((ELAPSED / 60))분 $((ELAPSED % 60))초)"
      report_metric "$RUN_DIR" | tee -a "$MAIN_LOG"
      SUMMARY+=("✓ mode$MODE/$SPLIT  $((ELAPSED / 60))분")
    else
      log "  ✗ 클라이언트 실패 (rc=$RC, $((ELAPSED / 60))분) — client.log 확인"
      tail -5 "$RUN_DIR/logs/client.log" | sed 's/^/    /' | tee -a "$MAIN_LOG"
      SUMMARY+=("✗ mode$MODE/$SPLIT  rc=$RC")
    fi

    bash "$SCRIPT_DIR/stop_server.sh" "$PORT" >>"$MAIN_LOG" 2>&1
    CURRENT_PORT=""
    wait_gpu_free
  done
done

log ""
log "=========================================================="
log "전체 종료"
for line in "${SUMMARY[@]}"; do log "  $line"; done
log "결과: $ASR_DIR/mode{2,3,4}/$SCOPE/{testclean,testother,devclean,devother}_$TAG/"
log "=========================================================="
