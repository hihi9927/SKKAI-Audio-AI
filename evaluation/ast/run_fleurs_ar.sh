#!/usr/bin/env bash
# FLEURS ar_eg → en — 아랍어 AST 지연(StreamLAAL) + BLEU.
#
#   bash evaluation/ast/run_fleurs_ar.sh [LIMIT]
#
# 왜 이 조합인가 (2026-08-31)
#   모델   : **베이스 `Qwen/Qwen3-ASR-1.7B`**. `en-dailytalk-seg` 는 영어 파인튜닝이라
#            아랍어에 못 쓴다. 대신 베이스는 `<SEG>` 를 뱉지 않으므로 **seg 축이 성립하지
#            않는다** — 축은 static/punct 둘뿐이다. seg 를 넣으려면 아랍어 SEG 파인튜닝이
#            먼저다.
#   데이터 : FLEURS 는 n-way 병렬이라 ar_eg 음성 + en_us 참조로 **진짜 LAAL 이 나온다.**
#            Casablanca 는 `tgt_text` 가 비어 있어 metrics_ast 의 분모가 |Y_hyp| 로 떨어져
#            (metrics_ast.py:117,141) LAAL 이 조용히 AL 로 바뀐다 — 그래서 안 쓴다.
#   번역   : MADLAD-400-3B 로컬. Cloud Translation v2 키는 403(User Rate Limit Exceeded)
#            로 죽어 있다(2026-08-31 확인). RESULTS.md 현행 표도 MADLAD 라 비교도 이쪽이 맞다.
#   LAAL   : 타깃이 영어라 `--laal-unit word`.
#
# 환경변수: AXES(기본 "static punct"), CHUNK(기본 2.0), GPU_UTIL, TRANS_BACKEND, TRANS_BATCH
# 종료는 반드시 stop_server.sh 로 한다(pkill 은 vLLM EngineCore 를 남긴다).

set -u

REPO="/home/mobility/STiTy"
PY="$REPO/.venv/bin/python"
STOP="$REPO/evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh"
PORT=8765
# vLLM 12GB + MADLAD 6.75GB = 18.75GB. 24GB 카드에 5GB 남는다.
GPU_UTIL="${GPU_UTIL:-0.5}"
MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
MANIFEST="$REPO/evaluation/ast/manifests/fleurs_ar-en_test.jsonl"

LIMIT="${1:-}"
AXES="${AXES:-static punct}"
CHUNK="${CHUNK:-2.0}"
TRANS_BACKEND="${TRANS_BACKEND:-local}"
TRANS_BATCH="${TRANS_BATCH:-8}"

axis_label() {
  if [ "$CHUNK" = "2.0" ]; then echo "ar-$1"; else echo "ar-$1-c${CHUNK%.0}"; fi
}
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/evaluation/ast/results/_runlogs/fleurs_ar_$STAMP"
mkdir -p "$LOGDIR"

[ -f "$MANIFEST" ] || { echo "!! 매니페스트 없음: $MANIFEST"; exit 1; }

echo "══════════════════════════════════════════════"
echo " FLEURS ar_eg→en  아랍어 AST  tag=$STAMP"
echo " 모델   : $MODEL (베이스 — seg 축 없음)"
echo " 청크   : ${CHUNK}s     축: $AXES"
echo " 번역   : $TRANS_BACKEND (배치 $TRANS_BATCH)   GPU: $GPU_UTIL"
echo " 발화   : ${LIMIT:-전량(283)}"
echo " 로그   : $LOGDIR"
echo " 시작   : $(date '+%F %T')"
echo "══════════════════════════════════════════════"
echo

run_axis() {
  local axis="$1"; shift
  local server_args=("$@")
  local label; label="$(axis_label "$axis")"
  local slog="$LOGDIR/${label}_server.log"

  echo "═══ [$label] 서버 기동 ═══ $(date '+%T')"
  "$PY" "$REPO/evaluation/streaming_websocket_server_ast.py" \
      --model "$MODEL" --no-vad --chunk-size "$CHUNK" \
      --port "$PORT" --no-idle-shutdown \
      --gpu-memory-utilization "$GPU_UTIL" \
      --ast-hide-seg \
      --trans-backend "$TRANS_BACKEND" \
      --trans-local-batch "$TRANS_BATCH" \
      --trans-stats-out "$LOGDIR/${label}_trans_stats.json" \
      "${server_args[@]}" > "$slog" 2>&1 &
  local spid=$!

  # 기동 판정은 로그가 아니라 포트 접속으로 한다(워밍업 뒤에야 포트가 열린다).
  local waited=0
  until "$PY" -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2).close()" 2>/dev/null; do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "!! [$label] 서버 기동 실패 — $slog"; tail -30 "$slog"; return 1
    fi
    sleep 5; waited=$((waited+5))
    if [ "$waited" -ge 900 ]; then echo "!! [$label] 기동 타임아웃"; return 1; fi
  done
  echo "═══ [$label] 준비 완료 (${waited}초) ═══"

  local clog="$LOGDIR/${label}_client.log"
  local cargs=(--manifest "$MANIFEST"
               --dataset FLEURS --model "$label" --scope "ar-en" --tag "$STAMP"
               --src-lang ar --target-lang en --laal-unit word
               --port "$PORT" --clients 16 --trailing-silence-ms 500)
  [ -n "$LIMIT" ] && cargs+=(--limit "$LIMIT")
  "$PY" "$REPO/evaluation/ast/test_ast.py" "${cargs[@]}" 2>&1 | tee "$clog" | \
      grep -E "발화 [0-9]+개|LAAL|BLEU|FTL|번역 호출|번역 실패|커밋 사유|실시간 대비"

  grep -q "이번 실행 0개" "$clog" && echo "!!! [$label] 이번 실행 0개 — 결과 경로 충돌 의심"
  grep -q "발화 0개"      "$clog" && echo "!!! [$label] 발화 0개 — 서버 연결 문제"
  grep -qE "번역 실패 : [1-9]" "$clog" && echo "!!! [$label] 번역 실패 — 이 런의 점수는 신뢰 불가"

  echo "═══ [$label] 서버 종료 ═══ $(date '+%T')"
  bash "$STOP" "$PORT" 2>&1 | tail -2
  sleep 5
  echo
}

for axis in $AXES; do
  case "$axis" in
    static) run_axis static "--always-commit" "--disable-dot-commit" ;;
    punct)  run_axis punct  "--enable-dot-commit" "--no-rep-dedup" ;;
    seg)    echo "!! seg 축은 베이스 모델에서 성립하지 않는다(<SEG> 미출력)"; exit 1 ;;
    *) echo "!! 알 수 없는 축: $axis (static|punct)"; exit 1 ;;
  esac
done

echo "═══════════════ 요약 ═══════════════ $(date '+%F %T')"
for axis in $AXES; do
  label="$(axis_label "$axis")"
  echo "── $label"
  grep -E "발화 [0-9]+개|LAAL|BLEU|FTL|번역 호출|번역 실패|커밋 사유" \
       "$LOGDIR/${label}_client.log" 2>/dev/null | sed 's/^.*INFO - /   /;s/^.*ERROR - /   !! /'
  echo
done
echo "결과: $REPO/evaluation/ast/results/FLEURS/ar-{static,punct}/ar-en/$STAMP/metric.json"
