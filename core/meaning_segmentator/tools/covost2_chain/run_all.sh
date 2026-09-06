#!/bin/bash
# claude 자식으로 띄우면 세션 종료와 함께 죽는다. tmux 로 격리해서 띄운다.
cd /home/mobility/STiTy || exit 1
S=covost2
TMUX_BIN=$(command -v tmux)
$TMUX_BIN has-session -t $S 2>/dev/null && { echo "!! '$S' 세션이 이미 있다. tmux kill-session -t $S 후 다시."; exit 1; }
rm -f core/meaning_segmentator/experiment/artifacts/covost2/_status/*

$TMUX_BIN new-session  -d -s $S -n zh   -c /home/mobility/STiTy "bash core/meaning_segmentator/tools/covost2_chain/01_zh_relabel.sh 2>&1 | tee -a core/meaning_segmentator/experiment/artifacts/covost2/_status/01_zh.out; exec bash"
$TMUX_BIN new-window   -t $S   -n enx   -c /home/mobility/STiTy "bash core/meaning_segmentator/tools/covost2_chain/02_enx_eval.sh  2>&1 | tee -a core/meaning_segmentator/experiment/artifacts/covost2/_status/02_enx.out; exec bash"
$TMUX_BIN new-window   -t $S   -n x2en  -c /home/mobility/STiTy "bash core/meaning_segmentator/tools/covost2_chain/03_x2en.sh      2>&1 | tee -a core/meaning_segmentator/experiment/artifacts/covost2/_status/03_x2en.out; exec bash"
echo "띄웠다:"; $TMUX_BIN list-windows -t $S
