# play_pcm.py
import numpy as np
import sounddevice as sd
import sys

path = sys.argv[1]  # 실행 시 파일 경로 인자로 받음
audio = np.fromfile(f"{path}/eval_clean/KsponSpeech_E00443.pcm", dtype="<i2").astype("float32") / 32768.0
print(f"길이: {len(audio)/16000:.2f}초")
sd.play(audio, samplerate=16000)
sd.wait()