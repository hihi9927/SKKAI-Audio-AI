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
import threading
import time


def _get_duration(path: str) -> float | None:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


def _progress_printer(start: float, total: float | None, stop: threading.Event) -> None:
    while not stop.is_set():
        elapsed = time.perf_counter() - start
        if total:
            print(f"\r  {elapsed:.1f}s / {total:.1f}s", end="", flush=True)
        else:
            print(f"\r  {elapsed:.1f}s", end="", flush=True)
        time.sleep(0.1)
    print()


def play_with_ffplay(path: str) -> None:
    result = subprocess.run(["which", "ffplay"], capture_output=True)
    if result.returncode != 0:
        raise FileNotFoundError("ffplay not found")
    total = _get_duration(path)
    print(f"[info] playing via ffplay: {os.path.basename(path)}"
          + (f" | {total:.1f}s" if total else ""))
    stop = threading.Event()
    t = threading.Thread(target=_progress_printer, args=(time.perf_counter(), total, stop), daemon=True)
    t.start()
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
    stop.set()
    t.join()


def play_with_aplay(path: str) -> None:
    result = subprocess.run(["which", "aplay"], capture_output=True)
    if result.returncode != 0:
        raise FileNotFoundError("aplay not found")
    total = _get_duration(path)
    print(f"[info] playing via aplay: {os.path.basename(path)}"
          + (f" | {total:.1f}s" if total else ""))
    stop = threading.Event()
    t = threading.Thread(target=_progress_printer, args=(time.perf_counter(), total, stop), daemon=True)
    t.start()
    subprocess.run(["aplay", path], check=True)
    stop.set()
    t.join()


def play_with_sounddevice(path: str) -> None:
    import soundfile as sf
    import sounddevice as sd

    data, samplerate = sf.read(path)
    total = len(data) / samplerate
    print(f"[info] {os.path.basename(path)} | sr={samplerate}Hz | {total:.1f}s")
    sd.play(data, samplerate)
    start = time.perf_counter()
    try:
        while sd.get_stream().active:
            elapsed = time.perf_counter() - start
            print(f"\r  {elapsed:.1f}s / {total:.1f}s", end="", flush=True)
            time.sleep(0.1)
    except Exception:
        pass
    print()
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
