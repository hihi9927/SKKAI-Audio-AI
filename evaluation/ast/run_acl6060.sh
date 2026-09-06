#!/usr/bin/env bash
# ACL 60/60 장문 — 발표 통짜 스트리밍, 분절 정책 3축 × 타깃 3언어.
#
#   bash evaluation/ast/run_acl6060.sh [SPLIT]      # SPLIT: dev(기본) | eval
#
# CoVoST2 와 결정적으로 다른 점
#   발화 하나 = **12분짜리 발표 하나**. 끊지 않고 통째로 흘려보낸다. 그래서 시스템이
#   내는 조각과 참조 문장의 경계가 전혀 맞지 않고, 채점은 사후에 mwerSegmenter 재분절을
#   거쳐 StreamLAAL 로 낸다(`score_acl6060.py`). 이 스크립트는 **런만** 담당한다.
#
#   발표가 5개뿐이라 병렬도를 5로 둔다(그 이상은 놀기만 한다). 실시간 페이싱이므로
#   런당 대략 "가장 긴 발표 길이" ≈ 12분이다.
#
# 환경변수
#   AXES           기본 "static punct seg"
#   CHUNK          기본 2.0
#   TRANS_BACKEND  기본 v2. `local` 이면 MADLAD-400-3B(greedy)를 같은 GPU 에 올린다 —
#                  **번역 품질이 달라 v2 로 낸 결과와 같은 표에 올리면 안 된다**
#   GPU_MEM        vLLM gpu_memory_utilization. local 번역기(6.75GB)와 나눠 쓰려면 0.65
#   AST_CAP_FREEZE / AST_AUDIO_END_AT_COMMIT   서버 수정 스위치(환경 그대로 상속된다)
# 종료는 반드시 stop_server.sh 로 한다(pkill 은 vLLM EngineCore 를 남긴다).

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
MODEL="$REPO/models/Qwen3-ASR-1.7B-en-dailytalk-seg"

SPLIT="${1:-dev}"
AXES="${AXES:-static punct seg}"
CHUNK="${CHUNK:-2.0}"
TRANS_BACKEND="${TRANS_BACKEND:-v2}"
# 로컬 번역기 배치. punct/seg 는 커밋이 문장 단위라 길어서, 16 이면 활성값이
# 커져 OOM 이 난다(실측 2026-08-30: punct/de 에서 OutOfMemoryError 61건).
TRANS_BATCH="${TRANS_BATCH:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/evaluation/ast/results/_runlogs/acl_${SPLIT}_$STAMP"
mkdir -p "$LOGDIR"

set -a && . "$REPO/.env" && set +a
if [ -z "${GOOGLE_TRANSLATE_API_KEY:-}" ]; then
  echo "!! GOOGLE_TRANSLATE_API_KEY 가 없습니다 (.env 확인)"; exit 1
fi

# LAAL 단위는 CoVoST2 와 같은 규칙(de=word, ja/zh=char). StreamLAAL 채점에서도
# 같은 단위를 써야 하므로 채점기가 이 값을 meta.json 에서 읽는다.
declare -A LAAL_UNIT=( [de]=word [ja]=char [zh]=char )

axis_label() {
  if [ "$CHUNK" = "2.0" ]; then echo "$1"; else echo "$1-c${CHUNK%.0}"; fi
}

echo "═══════════════════════════════════════════════════════"
echo " ACL 60/60 장문 실험  split=$SPLIT  tag=$STAMP"
echo " 모델   : $(basename "$MODEL")"
echo " 청크   : ${CHUNK}s     축: $AXES"
echo " 번역   : $TRANS_BACKEND (배치 $TRANS_BATCH)   GPU: $GPU_UTIL"
echo " 스위치 : CAP_FREEZE=${AST_CAP_FREEZE:-off}  AUDIO_END=${AST_AUDIO_END_AT_COMMIT:-off}"
echo " 단위   : 발표 통짜 (발표 5개 × 3언어)"
echo " 로그   : $LOGDIR"
echo " 시작   : $(date '+%F %T')"
echo "═══════════════════════════════════════════════════════"
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
      --trans-local-batch "$TRANS_BATCH" \
      --ast-hide-seg \
      --trans-backend "$TRANS_BACKEND" \
      --trans-stats-out "$LOGDIR/${label}_trans_stats.json" \
      "${server_args[@]}" > "$slog" 2>&1 &
  local spid=$!

  local waited=0
  until "$PY" -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2).close()" 2>/dev/null; do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "!! [$label] 서버 기동 실패 — $slog"; tail -20 "$slog"; return 1
    fi
    sleep 5; waited=$((waited+5))
    if [ "$waited" -ge 900 ]; then echo "!! [$label] 기동 타임아웃"; return 1; fi
  done
  echo "═══ [$label] 준비 완료 (${waited}초) ═══"

  for lang in de ja zh; do
    local clog="$LOGDIR/${label}_${lang}_client.log"
    echo "─── [$label/$lang] $(date '+%T')"
    "$PY" "$REPO/evaluation/ast/test_ast.py" \
        --manifest "$REPO/evaluation/ast/manifests/acl6060_${SPLIT}_en-${lang}.jsonl" \
        --dataset ACL6060 --model "$label" --scope "${SPLIT}-${lang}" --tag "$STAMP" \
        --src-lang en --target-lang "$lang" \
        --laal-unit "${LAAL_UNIT[$lang]}" \
        --port "$PORT" --clients 5 --trailing-silence-ms 500 \
        --ws-ping-interval 30 --ws-ping-timeout 600 2>&1 | tee "$clog" | \
        grep -E "발화 [0-9]+개|LAAL|BLEU|FTL|번역 호출|번역 실패|커밋 사유|실시간 대비"

    if grep -q "이번 실행 0개" "$clog"; then
      echo "!!! [$label/$lang] 이번 실행 0개 — 결과 경로가 겹쳤을 수 있다"
    fi
    if grep -q "발화 0개" "$clog"; then
      echo "!!! [$label/$lang] 발화 0개 — 서버 연결 문제. $clog 확인"
    fi
    if grep -q "번역 실패 : [1-9]" "$clog"; then
      echo "!!! [$label/$lang] 번역 실패 발생 — 이 런의 점수는 신뢰할 수 없다"
    fi
    # 워커 사망으로 발화가 빠지면 축 간 비교가 깨진다. 조용히 넘어가면 안 된다.
    if grep -q "발화 유실" "$clog"; then
      echo "!!! [$label/$lang] 발화 유실 — 재실행 필요"
      grep -m1 "발화 유실" "$clog" | sed 's/^/    /'
    fi
    if grep -q "워커 w[0-9]* 종료" "$clog"; then
      echo "!!! [$label/$lang] 워커가 죽었다:"
      grep -m3 "워커 w[0-9]* 종료" "$clog" | sed 's/^/    /'
    fi
    # 사후 채점(StreamLAAL)이 첫 완료 런부터 붙을 수 있게 신호를 남긴다.
    echo "$REPO/evaluation/ast/results/ACL6060/${label}/${SPLIT}-${lang}/${STAMP}" \
        >> "$LOGDIR/completed_runs.txt"
    echo
  done

  echo "═══ [$label] 서버 종료 ═══ $(date '+%T')"
  bash "$STOP" "$PORT" 2>&1 | tail -2
  sleep 5
  echo
}

for axis in $AXES; do
  case "$axis" in
    static) run_axis static "--always-commit" "--disable-dot-commit" ;;
    punct)  run_axis punct  "--enable-dot-commit" "--no-rep-dedup" ;;
    seg)    run_axis seg    "--disable-dot-commit" ;;
    *) echo "!! 알 수 없는 축: $axis (static|punct|seg)"; exit 1 ;;
  esac
done

echo "═══════════════ 요약 ═══════════════ $(date '+%F %T')"
for axis in $AXES; do
  label="$(axis_label "$axis")"
  for lang in de ja zh; do
    echo "── $label / $lang"
    grep -E "발화 [0-9]+개|LAAL      :|BLEU      :|FTL       :|번역 호출|번역 실패" \
         "$LOGDIR/${label}_${lang}_client.log" 2>/dev/null | sed 's/^.*INFO - /   /;s/^.*ERROR - /   !! /'
  done
  echo
done
echo "결과: $REPO/evaluation/ast/results/ACL6060/*/${SPLIT}-*/$STAMP/metric.json"
echo "채점: .venv-streamlaal/bin/python evaluation/ast/score_acl6060.py --tag $STAMP --split $SPLIT"
