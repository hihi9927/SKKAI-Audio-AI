# 모든 체인이 공유하는 준비. tmux 서버는 예전 환경을 물고 있어서 키가 없다 —
# 여기서 .env 를 직접 읽는다. (2026-09-02 세션 동반 사망 사고 이후 규칙)
cd /home/mobility/STiTy || exit 1
set -a; . ./.env 2>/dev/null; set +a
export PYTHONPATH=/home/mobility/STiTy
PY=/home/mobility/STiTy/.venv-autoseg/bin/python
R=core/meaning_segmentator/runs/covost2
ST=$R/_status
mkdir -p "$ST"

ts () { date '+%H:%M:%S'; }
mark () { echo "$(date '+%F %T') $2" > "$ST/$1"; }        # mark <이름> <내용>
# wait_for <마커...> — PID 가 아니라 파일로 기다린다. 세션이 죽어도 의미가 유지된다.
wait_for () {
  for m in "$@"; do
    while [ ! -f "$ST/$m" ]; do sleep 30; done
    echo "  [$(ts)] 선행 완료: $m -> $(cat "$ST/$m")"
  done
}
