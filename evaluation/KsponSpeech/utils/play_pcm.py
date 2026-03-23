# play_pcm.py
import numpy as np
import sounddevice as sd
import sys
from pathlib import Path

PCM_DIR = Path(r"/home/skkai/Documents/00_skkai_session/01_2026/02_speech/STiTy/evaluation/KsponSpeech/data/KsponSpeech_0001")

arg = sys.argv[1]
if arg.isdigit():
    path = PCM_DIR / f"KsponSpeech_{int(arg):06d}.pcm"
else:
    path = Path(arg)

print(f"재생: {path.name}")
audio = np.fromfile(path, dtype="<i2").astype("float32") / 32768.0
print(f"길이: {len(audio)/16000:.2f}초")
sd.play(audio, samplerate=16000)
sd.wait()