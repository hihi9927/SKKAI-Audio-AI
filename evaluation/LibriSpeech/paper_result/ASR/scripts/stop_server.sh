#!/bin/bash
# 평가 서버를 GPU 메모리까지 확실히 회수하며 종료한다.
# 사용법: bash stop_server.sh [port ...]     (기본: 8765 8766 8767 = mode2/3/4)
#   예:   bash stop_server.sh 8766
#         bash stop_server.sh            # 세 포트 모두 정리
#
# 왜 이 스크립트가 필요한가
# ------------------------
# vLLM은 VRAM을 부모 파이썬이 아니라 자식 프로세스 `VLLM::EngineCore`가 쥔다.
# 그래서 부모만 죽이면 EngineCore가 PPID=1로 재부모화되어 20GB를 계속 점유한다.
# 실측(2026-08-05, RTX 4090): 부모에 SIGTERM → 부모는 1초 안에 죽었지만 EngineCore는
# 40초 넘게 살아남았고 스스로 종료하지 않았다. 서버 코드에 시그널 핸들러가 없어
# 엔진을 내려주는 사람이 아무도 없다.
#
# 특히 `pkill -f streaming_websocket`은 절대 통하지 않는다 — EngineCore의 cmdline은
# 'VLLM::EngineCore'라 그 패턴에 매칭되지 않는다. 이게 과거 실험이 지연된 경로다.
#
# 해법: EngineCore는 serve 스크립트와 같은 프로세스 그룹에 있으므로 그룹 전체에
# 시그널을 보낸다(kill -TERM -PGID). 그래도 남으면 미리 기록해 둔 자식 PID만
# 골라 SIGKILL한다.
#
# 참고: 터미널에서 포그라운드로 띄운 서버를 Ctrl+C로 끄면 SIGINT가 포그라운드
# 프로세스 그룹 전체에 가므로 정상 회수된다. 문제가 되는 건 `kill <pid>` /
# `pkill -f` / nohup·백그라운드로 띄운 뒤 부모만 죽이는 경우다.
set -u

PORTS=("$@")
if [[ ${#PORTS[@]} -eq 0 ]]; then
  PORTS=(8765 8766 8767)
fi

MY_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"
STOPPED=0

gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "n/a"
}

find_listener() {
  # 포트를 듣고 있는 PID. ss가 없거나 못 찾으면 cmdline으로 폴백한다.
  local port="$1" pid
  pid="$(ss -ltnp 2>/dev/null | grep -oP ":${port}\s.*pid=\K[0-9]+" | head -1)"
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -f "streaming_websocket_server.*--port $port" | head -1)"
  fi
  echo "$pid"
}

alive() { kill -0 "$1" 2>/dev/null; }

for PORT in "${PORTS[@]}"; do
  PID="$(find_listener "$PORT")"
  if [[ -z "$PID" ]]; then
    echo "[$PORT] 실행 중인 서버 없음"
    continue
  fi

  PGID="$(ps -o pgid= -p "$PID" | tr -d ' ')"
  # EngineCore는 여기서 미리 잡아둔다. 죽은 뒤에는 부모-자식 관계를 되짚을 수 없어
  # "고아 EngineCore를 전부 쓸기"가 되는데, 그러면 다른 실험의 엔진까지 건드린다.
  CHILDREN="$(pgrep -P "$PID" | tr '\n' ' ')"
  echo "[$PORT] 서버 pid=$PID pgid=$PGID 자식=[${CHILDREN:-없음}] GPU=$(gpu_used)"

  if [[ -n "$PGID" && "$PGID" != "$MY_PGID" ]]; then
    kill -TERM -"$PGID" 2>/dev/null || true
  else
    # 이 스크립트가 서버와 같은 그룹에서 실행됐다 — 그룹을 죽이면 자신도 죽는다.
    echo "[$PORT] 경고: 서버와 같은 프로세스 그룹. 개별 종료로 전환한다."
    kill -TERM "$PID" $CHILDREN 2>/dev/null || true
  fi

  for _ in $(seq 1 15); do
    sleep 1
    REMAIN=""
    for p in $PID $CHILDREN; do alive "$p" && REMAIN+="$p "; done
    [[ -z "$REMAIN" ]] && break
  done

  if [[ -n "$REMAIN" ]]; then
    echo "[$PORT] SIGTERM 후에도 남음: $REMAIN → SIGKILL"
    # shellcheck disable=SC2086
    kill -KILL $REMAIN 2>/dev/null || true
    sleep 2
  fi

  STILL=""
  for p in $PID $CHILDREN; do alive "$p" && STILL+="$p "; done
  if [[ -n "$STILL" ]]; then
    echo "[$PORT] 실패: 아직 살아있음 → $STILL"
  else
    echo "[$PORT] 종료 완료"
    STOPPED=$((STOPPED + 1))
  fi
done

# GPU가 실제로 비었는지 확인한다. 스크립트가 "완료"라고 말했는데 VRAM이 안 빠지면
# 이 스크립트가 모르는 경로로 샌 것이므로 그 사실이 보여야 한다.
sleep 1
echo "--- 잔여 GPU 프로세스 ---"
LEFT="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
echo "${LEFT:-(없음)}"
echo "--- GPU 사용량: $(gpu_used) (서버 $STOPPED개 종료) ---"
