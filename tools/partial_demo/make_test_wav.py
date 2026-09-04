"""LibriSpeech test-other 발화 몇 개를 침묵으로 이어 붙여 테스트용 wav 를 만든다.

침묵 구간을 넣는 이유는 VAD 커밋과 슬롯 리셋을 실제로 태우기 위해서다.
그 지점에서 빈 partial 이 나가는지가 이번 확인의 핵심이다.
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

SR = 16000
GAP_SEC = 1.2
LEAD_SEC = 0.3

# LibriSpeech 오디오는 저장소에 없다(gitignore). 워크트리에서 돌릴 때는 데이터가 있는
# 원본 저장소 경로를 세 번째 인자로 넘긴다.
DEFAULT_ROOT = "../../evaluation/LibriSpeech/LibriSpeech/test-other"

out = sys.argv[1]
n_want = int(sys.argv[2]) if len(sys.argv) > 2 else 3
root = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), DEFAULT_ROOT)
root = os.path.normpath(root)
if not os.path.isdir(root):
    sys.exit(f"LibriSpeech test-other 를 찾을 수 없다: {root}\n"
             f"사용법: python make_test_wav.py <out.wav> [발화수] [test-other 경로]")

flacs = sorted(glob.glob(os.path.join(root, "*", "*", "*.flac")))
picked, chunks, meta = [], [], []
for f in flacs:
    x, sr = sf.read(f, dtype="float32")
    assert sr == SR, (f, sr)
    dur = len(x) / SR
    if not (2.5 <= dur <= 7.0):        # 너무 길거나 짧은 건 건너뛴다
        continue
    picked.append((f, dur, x))
    if len(picked) == n_want:
        break

# 전사 정답도 같이 뽑아 둔다 (트랜스크립트 파일은 디렉터리당 하나)
def transcript_of(path):
    d = os.path.dirname(path)
    uid = os.path.basename(path)[:-5]
    tpath = glob.glob(os.path.join(d, "*.trans.txt"))[0]
    for line in open(tpath, encoding="utf-8"):
        k, _, text = line.partition(" ")
        if k == uid:
            return text.strip()
    return ""

sil = np.zeros(int(GAP_SEC * SR), dtype="float32")
chunks.append(np.zeros(int(LEAD_SEC * SR), dtype="float32"))
for f, dur, x in picked:
    chunks.append(x)
    chunks.append(sil)
    meta.append((os.path.basename(f)[:-5], round(dur, 2), transcript_of(f)))

y = np.concatenate(chunks)
sf.write(out, y, SR, subtype="PCM_16")
print(f"wrote {out}  total={len(y)/SR:.2f}s  utts={len(meta)}")
for uid, dur, text in meta:
    print(f"  {uid}  {dur:>5.2f}s  {text}")
