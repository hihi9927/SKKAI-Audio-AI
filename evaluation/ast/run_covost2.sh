#!/usr/bin/env bash
# CoVoST2 단클립 3,000발화 — 분절 정책 3축 × 타깃 3언어.
#
#   bash evaluation/ast/run_covost2.sh [LIMIT]
#     LIMIT 를 주면 발화 수를 제한한다(스모크용). 없으면 전량.
#
# 설계 결정 (2026-08-25)
#   ASR 모델   : en-dailytalk-seg **한 종류로 고정**. 기존 run_three_axes.sh 는 축마다
#                모델이 달라(dot/always=baseline, seg=파인튜닝) 정책 차이와 모델 차이가
#                섞였다. 세 축이 서로 간섭하지 않는 것은 코드로 확인했다:
#                  --always-commit    → SEG/dot 트리거를 아예 보지 않는다
#                  --enable-dot-commit → 구두점 정규식만 매칭, <SEG> 는 트리거가 아니다
#                  (기본)             → <SEG> 만 매칭
#   번역       : Cloud Translation Basic(v2), 문맥 없음 → 커밋 1건 = 호출 1건.
#                무료 gtx 는 2026-08 에 IP 차단됐다(30만 건 호출). evaluation/ast/trans_guard.py 참고.
#   서버 기동  : **축당 1회**. 타깃 언어는 클라이언트가 start 메시지로 보내므로 서버는
#                언어와 무관하다. 축마다 3언어를 이어 돌려 기동 6회를 아낀다.
#
# 종료는 반드시 stop_server.sh 로 한다(pkill 은 vLLM EngineCore 를 남긴다).

set -u

REPO="/home/mobility/STiTy"
PY="$REPO/.venv/bin/python"
STOP="$REPO/evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh"
PORT=8765
MODEL="$REPO/models/Qwen3-ASR-1.7B-en-dailytalk-seg"

LIMIT="${1:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$REPO/evaluation/ast/results/_runlogs/$STAMP"
mkdir -p "$LOGDIR"

set -a && . "$REPO/.env" && set +a
if [ -z "${GOOGLE_TRANSLATE_API_KEY:-}" ]; then
  echo "!! GOOGLE_TRANSLATE_API_KEY 가 없습니다 (.env 확인)"; exit 1
fi

# LAAL 의 |Y| 단위: de 는 단어, zh/ja 는 글자. BLEU 토크나이저는 target-lang 으로
# 자동 결정된다(de→13a, ja→ja-mecab, zh→zh). ast/README.md "점수를 바꾸는 설정" 참고.
declare -A LAAL_UNIT=( [de]=word [ja]=char [zh]=char )

echo "═══════════════════════════════════════════════════════"
echo " CoVoST2 단클립 실험  tag=$STAMP"
echo " 모델   : $(basename "$MODEL")"
echo " 번역   : Cloud Translation v2"
echo " 발화   : ${LIMIT:-전량(3,000)} × 3언어 × 3축"
echo " 로그   : $LOGDIR"
echo " 시작   : $(date '+%F %T')"
echo "═══════════════════════════════════════════════════════"
echo

run_axis() {
  local axis="$1"; shift
  local server_args=("$@")
  local slog="$LOGDIR/${axis}_server.log"

  echo "═══ [$axis] 서버 기동 ═══ $(date '+%T')"
  "$PY" "$REPO/evaluation/streaming_websocket_server_ast.py" \
      --model "$MODEL" --no-vad --chunk-size "${CHUNK:-2.0}" \
      --port "$PORT" --no-idle-shutdown \
      --trans-backend v2 \
      --trans-stats-out "$LOGDIR/${axis}_trans_stats.json" \
      "${server_args[@]}" > "$slog" 2>&1 &
  local spid=$!

  # 기동 판정은 로그 줄이 아니라 **포트 접속**으로 한다. 서버는 "Starting WebSocket
  # server" 를 찍은 뒤 워밍업을 하고, 그게 끝나야 실제로 포트를 연다. 로그만 보고
  # 붙으면 16 워커가 Connect call failed 로 전부 즉사한다(실측).
  local waited=0
  until "$PY" -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2).close()" 2>/dev/null; do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "!! [$axis] 서버 기동 실패 — $slog"; tail -20 "$slog"; return 1
    fi
    sleep 5; waited=$((waited+5))
    if [ "$waited" -ge 900 ]; then echo "!! [$axis] 기동 타임아웃"; return 1; fi
  done
  echo "═══ [$axis] 준비 완료 (${waited}초) ═══"

  for lang in de ja zh; do
    local clog="$LOGDIR/${axis}_${lang}_client.log"
    echo "─── [$axis/$lang] $(date '+%T')"
    # scope 에 언어를 넣어야 한다. 결과 경로가 {dataset}/{model}/{scope}/{tag} 인데
    # 언어가 빠지면 세 언어가 같은 폴더를 쓴다. 그러면 --tag 재사용의 "중단 지점부터
    # 재개" 로직이 걸려 ja/zh 가 "이번 실행 0개"로 통째로 건너뛰고, 마지막 언어의
    # 채점 설정으로 앞 언어의 번역을 다시 채점한 metric.json 이 남는다
    # (실측: 독일어 가설이 tok=zh / laal_unit=char 로 채점됨).
    local cargs=(--manifest "$REPO/evaluation/ast/manifests/covost2_en-${lang}_n3000.jsonl"
                 --dataset CoVoST2 --model "$axis" --scope "n3000-${lang}" --tag "$STAMP"
                 --src-lang en --target-lang "$lang"
                 --laal-unit "${LAAL_UNIT[$lang]}"
                 --port "$PORT" --clients 16 --trailing-silence-ms 500)
    [ -n "$LIMIT" ] && cargs+=(--limit "$LIMIT")
    "$PY" "$REPO/evaluation/ast/test_ast.py" "${cargs[@]}" 2>&1 | tee "$clog" | \
        grep -E "발화 [0-9]+개|LAAL|BLEU|FTL|번역 호출|번역 실패|커밋 사유|실시간 대비"

    # 재개 로직에 걸려 통째로 건너뛰면 "이번 실행 0개"가 찍힌다. 조용히 넘어가면 안 된다.
    if grep -q "이번 실행 0개" "$clog"; then
      echo "!!! [$axis/$lang] 이번 실행 0개 — 결과 경로가 겹쳤을 수 있다"
    fi
    if grep -q "발화 0개" "$clog"; then
      echo "!!! [$axis/$lang] 발화 0개 — 서버 연결 문제. $clog 확인"
    fi
    if grep -q "번역 실패 : [1-9]" "$clog"; then
      echo "!!! [$axis/$lang] 번역 실패 발생 — 이 런의 점수는 신뢰할 수 없다"
    fi
    echo
  done

  echo "═══ [$axis] 서버 종료 ═══ $(date '+%T')"
  bash "$STOP" "$PORT" 2>&1 | tail -2
  sleep 5
  echo
}

# 축 정의 — 서버 인자만 다르고 모델은 동일하다.
run_axis static "--always-commit" "--disable-dot-commit"
run_axis punct  "--enable-dot-commit" "--no-rep-dedup"
run_axis seg    "--disable-dot-commit"

echo "═══════════════ 요약 ═══════════════ $(date '+%F %T')"
for axis in static punct seg; do
  for lang in de ja zh; do
    echo "── $axis / $lang"
    grep -E "발화 [0-9]+개|LAAL      :|BLEU      :|FTL       :|번역 호출|번역 실패" \
         "$LOGDIR/${axis}_${lang}_client.log" 2>/dev/null | sed 's/^.*INFO - /   /;s/^.*ERROR - /   !! /'
  done
  [ -f "$LOGDIR/${axis}_trans_stats.json" ] && \
    echo "   번역 통계: $(tr -d '\n ' < "$LOGDIR/${axis}_trans_stats.json" | head -c 300)"
  echo
done
echo "결과: $REPO/evaluation/ast/results/CoVoST2/{static,punct,seg}/n3000-{de,ja,zh}/$STAMP/metric.json"
