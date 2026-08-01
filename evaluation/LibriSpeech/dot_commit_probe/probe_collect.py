"""청크별 가설(hypothesis) 수집기 — 커밋/리셋 없이 순수 누적 디코딩만 관찰.

목적: "청크 끝 온점"이 다음 청크에서 살아남는지(진짜 문장 끝) 사라지는지(기계적)
      판별 가능한지 실측하기 위한 원자료(JSONL) 생성.

서버(streaming_websocket_server.py)는 커밋 시 슬롯을 리셋해 audio_accum을 버리므로
재디코딩 기회가 사라진다. 이 스크립트는 그 리셋을 하지 않는다.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import warmup_streaming

SR = 16000


def load_flac(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == SR, f"expected 16k, got {sr}"
    return wav


def collect_files(test_dir: str, limit: int, spread: bool = False):
    """(flac_path, reference_text) 목록 수집.

    spread=True면 챕터(화자)별로 라운드로빈해 한 화자에 쏠리지 않게 한다.
    """
    per_chapter = []
    for trans in sorted(Path(test_dir).rglob("*.trans.txt")):
        refs = {}
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            refs[uid] = text.strip()
        group = []
        for uid in sorted(refs):
            flac = trans.parent / f"{uid}.flac"
            if flac.exists():
                group.append((str(flac), refs[uid]))
        if group:
            per_chapter.append(group)
        if not spread and sum(len(g) for g in per_chapter) >= limit:
            break

    items = []
    if spread:
        i = 0
        while len(items) < limit and any(len(g) > i for g in per_chapter):
            for g in per_chapter:
                if i < len(g):
                    items.append(g[i])
                    if len(items) >= limit:
                        break
            i += 1
    else:
        for g in per_chapter:
            items.extend(g)
    return items[:limit]


def build_streams(files, concat: int, gap_sec: float):
    """utterance를 concat개씩 이어붙여 연속 발화 스트림 구성.

    반환: [(name, wav, reference, [utterance별 (시작초, 텍스트)])]
    """
    streams = []
    for i in range(0, len(files) - concat + 1, concat):
        group = files[i: i + concat]
        parts, refs, marks = [], [], []
        cursor = 0.0
        for flac, ref in group:
            wav = load_flac(flac)
            marks.append((round(cursor, 3), ref))
            parts.append(wav)
            cursor += len(wav) / SR
            if gap_sec > 0:
                parts.append(np.zeros(int(gap_sec * SR), dtype=np.float32))
                cursor += gap_sec
            refs.append(ref)
        name = f"{Path(group[0][0]).stem}+{concat-1}"
        streams.append((name, np.concatenate(parts), " ".join(refs), marks))
    return streams


async def run(args):
    asr = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    await warmup_streaming(asr)

    files = collect_files(args.test_dir, args.limit, spread=args.spread)
    if args.concat > 1:
        streams = build_streams(files, args.concat, args.gap_sec)
    else:
        streams = [
            (os.path.relpath(f, args.test_dir), load_flac(f), ref, [(0.0, ref)])
            for f, ref in files
        ]
    print(f"[probe] {len(streams)} streams (from {len(files)} utterances), "
          f"chunk={args.chunk_sec}s trailing_silence={args.trailing_silence_sec}s concat={args.concat}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")

    chunk_samples = int(round(args.chunk_sec * SR))
    tok = asr.processor.tokenizer

    for idx, (name, wav, ref, marks) in enumerate(streams):
        dur = len(wav) / SR
        if args.trailing_silence_sec > 0:
            wav = np.concatenate([wav, np.zeros(int(args.trailing_silence_sec * SR), dtype=np.float32)])

        state = asr.init_streaming_state(
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_sec,
            allowed_languages=["English"],
        )

        chunks = []
        pos = 0
        t0 = time.perf_counter()
        while pos < len(wav):
            piece = wav[pos: pos + chunk_samples]
            pos += chunk_samples
            await asr.streaming_transcribe(piece, state)
            text = (state.text or "").strip()
            chunks.append({
                "chunk_idx": len(chunks),
                "audio_sec": round(min(pos, len(wav)) / SR, 3),
                "text": text,
                "raw": state._raw_decoded,
                "n_tokens": len(tok.encode(state._raw_decoded or "")),
                "decode_sec": round(time.perf_counter() - t0, 3),
            })

        await asr.finish_streaming_transcribe(state)
        final_text = (state.text or "").strip()

        rec = {
            "file": name,
            "reference": ref,
            "utterance_marks": marks,
            "duration_sec": round(dur, 3),
            "chunk_sec": args.chunk_sec,
            "unfixed_token_num": args.unfixed_token_num,
            "chunks": chunks,
            "final_text": final_text,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{idx+1}/{len(streams)}] {rec['file']} dur={dur:.1f}s chunks={len(chunks)} final={final_text!r}")

    fout.close()
    print(f"[probe] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--test-dir", default="evaluation/LibriSpeech/LibriSpeech/test-other")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--chunk-sec", type=float, default=2.0)
    p.add_argument("--trailing-silence-sec", type=float, default=4.0)
    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=3072)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--spread", action="store_true", help="화자/챕터 라운드로빈 샘플링")
    p.add_argument("--concat", type=int, default=1, help="N개 발화를 이어붙여 연속 스트림 구성")
    p.add_argument("--gap-sec", type=float, default=1.0, help="concat 시 발화 사이 무음 길이")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
