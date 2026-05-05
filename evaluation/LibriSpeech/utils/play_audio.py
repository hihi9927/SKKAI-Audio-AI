"""
Play FLAC (or any audio) files from the command line.

Usage:
    python utils/play_audio.py <file.flac>
    python utils/play_audio.py evaluation/LibriSpeech/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac

Dependencies (try in order):
    System: ffplay (apt install ffmpeg) or aplay (apt install alsa-utils)
    Python: pip install sounddevice soundfile   (requires: apt install libportaudio2)
            pip install playsound
"""

import sys
import os
import subprocess


def play_with_ffplay(path: str) -> None:
    result = subprocess.run(["which", "ffplay"], capture_output=True)
    if result.returncode != 0:
        raise FileNotFoundError("ffplay not found")
    print(f"[info] playing via ffplay: {os.path.basename(path)}")
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)


def play_with_aplay(path: str) -> None:
    result = subprocess.run(["which", "aplay"], capture_output=True)
    if result.returncode != 0:
        raise FileNotFoundError("aplay not found")
    print(f"[info] playing via aplay: {os.path.basename(path)}")
    subprocess.run(["aplay", path], check=True)


def play_with_sounddevice(path: str) -> None:
    import soundfile as sf
    import sounddevice as sd

    data, samplerate = sf.read(path)
    print(f"[info] {os.path.basename(path)} | sr={samplerate}Hz | shape={data.shape}")
    sd.play(data, samplerate)
    sd.wait()


def play_with_playsound(path: str) -> None:
    from playsound import playsound
    print(f"[info] playing via playsound: {os.path.basename(path)}")
    playsound(path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python utils/play_audio.py <audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"[error] file not found: {path}")
        sys.exit(1)

    for player in [play_with_ffplay, play_with_aplay, play_with_sounddevice, play_with_playsound]:
        try:
            player(path)
            return
        except (ImportError, FileNotFoundError, OSError):
            continue

    print("[error] no audio player found. install one of:")
    print("  sudo apt install ffmpeg          # ffplay")
    print("  sudo apt install alsa-utils      # aplay")
    print("  sudo apt install libportaudio2 && pip install sounddevice soundfile")
    sys.exit(1)


if __name__ == "__main__":
    main()
