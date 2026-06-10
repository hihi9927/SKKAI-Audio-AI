"""
KsponSpeech 100h SFT 데이터셋 최종 조립.

split(forced-align) 결과 + seg 소스 + WAV를 모아 학습용 jsonl 생성.
- 원본 = 전용 원본 + split 스킵분(교차 흡수)
- split = forced-align 성공분
- 교차 스킵 소스의 WAV가 없으면 PCM에서 변환

출력 (Qwen3-ASR/finetuning/data/KSponSpeech/):
  train.jsonl       (원본 ~60k + split ~50k)
  val.jsonl         (원본 3k + split 3k)
  eval_clean.jsonl  (원본 1.5k + split 1.5k)
  eval_other.jsonl  (원본 1.5k + split 1.5k)
"""
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

BASE = Path(__file__).resolve().parents[1]          # finetuning/
STITY = BASE.parents[1]                              # STiTy/
R = STITY / "evaluation/KsponSpeech/results"
T = STITY / "evaluation/KsponSpeech/transcribe"
DATA = STITY / "evaluation/KsponSpeech/data"
KEVAL = STITY.parent / "KsponSpeech data/KsponSpeech_eval.zip"
AUDIO = BASE / "data/KSponSpeech/audio"
EVALAUDIO = BASE / "data/KSponSpeech/eval_audio"
SPLITAUDIO = BASE / "data/KSponSpeech/split_audio"
EVALSPLITAUDIO = BASE / "data/KSponSpeech/eval_split_audio"
PREFIX = "language Korean<asr_text>"
SR = 16000


def load(p):
    p = Path(p)
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["data"] if isinstance(d, dict) else d


def pcm_folder(file_id):
    num = int(file_id.split("_")[-1])
    folder = (num - 1) // 1000 + 1
    return DATA / f"KsponSpeech_{folder:04d}" / f"{file_id}.pcm"


def ensure_wav(file_id, out_dir, flat_pcm_dir=None):
    """file_id의 WAV가 없으면 PCM에서 변환. WAV 경로 반환(없으면 None)."""
    dst = out_dir / f"{file_id}.wav"
    if dst.exists():
        return dst
    src = (flat_pcm_dir / f"{file_id}.pcm") if flat_pcm_dir else pcm_folder(file_id)
    if not src.exists():
        return None
    buf = src.read_bytes()
    if len(buf) % 2:
        buf = buf[:-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst, SR, np.frombuffer(buf, dtype=np.int16))
    return dst


def orig_line(entry, audio_dir, flat_pcm_dir=None):
    """원본(full) 한 줄. WAV 보장."""
    wav = ensure_wav(entry["file"], audio_dir, flat_pcm_dir)
    if wav is None:
        return None
    return {"audio": str(wav), "text": PREFIX + entry["seg_text"].strip()}


def split_line(entry, audio_dir):
    """split 클립 한 줄. file=split_{orig}, seg_text=partial_seg."""
    wav = audio_dir / f"{entry['file']}.wav"   # generate_split_data가 이미 저장
    if not wav.exists():
        return None
    seg = (entry.get("seg_text") or entry.get("text") or "").strip()
    if not seg:
        return None
    return {"audio": str(wav), "text": PREFIX + seg}


def build(name, orig_entries, split_entries, orig_audio, split_audio, flat_pcm_dir=None):
    lines, miss_o, miss_s = [], 0, 0
    for e in orig_entries:
        l = orig_line(e, orig_audio, flat_pcm_dir)
        if l: lines.append(l)
        else: miss_o += 1
    n_orig = len(lines)
    for e in split_entries:
        l = split_line(e, split_audio)
        if l: lines.append(l)
        else: miss_s += 1
    out = BASE / "data/KSponSpeech" / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"[{name}] 원본 {n_orig}(누락{miss_o}) + split {len(lines)-n_orig}(누락{miss_s}) = {len(lines)}줄 → {out}")
    return len(lines)


def main():
    # ── TRAIN ──────────────────────────────────────────────────────────
    train_orig = load(T / "orig_dedicated.json") + load(R / "split_skipped.json")
    train_split = load(R / "split_data.json")
    build("train", train_orig, train_split, AUDIO, SPLITAUDIO)

    # ── VAL ────────────────────────────────────────────────────────────
    val_orig = load(T / "val_orig_dedicated.json") + load(R / "val_split_data_skipped.json")
    val_split = load(R / "val_split_data.json")
    build("val", val_orig, val_split, AUDIO, SPLITAUDIO)

    # ── EVAL (clean / other 각각 원본1.5k + split1.5k) ──────────────────
    for sub, pcmdir in [("eval_clean", KEVAL / "eval_clean"), ("eval_other", KEVAL / "eval_other")]:
        ori = load(T / f"{sub}_orig.json") + load(R / f"{sub}_split_data_skipped.json")
        spl = load(R / f"{sub}_split_data.json")
        build(sub, ori, spl, EVALAUDIO, EVALSPLITAUDIO, flat_pcm_dir=pcmdir)

    print("\n조립 완료.")


if __name__ == "__main__":
    main()
