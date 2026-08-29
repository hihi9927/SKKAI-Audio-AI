#!/usr/bin/env bash
# CUDA cuInit=100 (NO_DEVICE) 복구. nvidia_uvm 만 재적재한다 — 디스플레이(nvidia_drm,
# nvidia_modeset)는 건드리지 않으므로 화면은 안 꺼진다.
set -u
echo "== 전:"; lsmod | grep -i "^nvidia" || true
if lsmod | awk '$1=="nvidia_uvm" && $3!=0 {exit 1}'; then :; else
  echo "!! nvidia_uvm refcount != 0 — 쓰는 프로세스가 있다. 중단."; exit 1
fi
sudo modprobe -r nvidia_uvm || { echo "!! 언로드 실패"; exit 1; }
sudo modprobe nvidia_uvm    || { echo "!! 재적재 실패"; exit 1; }
echo "== 후:"; lsmod | grep -i "^nvidia" || true
echo "== 검증:"
PYTHONPATH= /home/mobility/STiTy/.venv-autoseg/bin/python -c "
import torch
ok = torch.cuda.is_available()
print('torch.cuda.is_available() =', ok)
if ok:
    print('device:', torch.cuda.get_device_name(0))
    torch.zeros(1).cuda(); print('할당 테스트 통과 — 복구 완료')
else:
    print('여전히 실패. 다음 단계는 재부팅.')
" 2>&1 | grep -viE "userwarning|pkg_resources|warnings.warn"
