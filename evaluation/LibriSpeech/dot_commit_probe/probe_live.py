"""라이브 하네스 — 커밋 + 슬롯 리셋(캐리오버)까지 서버와 동일하게 재현하고
naive(현재 브랜치) / gate(안 A) 두 정책을 실제 오디오로 비교한다.

서버 대응:
  - 커밋 후 remaining 없으면 오디오 통째 폐기 (SEG-SLOT-RESET)
  - remaining 있으면 마지막 청크 오디오만 캐리 (DOT-SLOT-SWITCH)
  - 커밋 텍스트 중복은 단어 단위 겹침 제거로 정리 (dot-suffix-dedup 대응)
  - audio_accum > MAX_AUDIO_ACCUM_SEC 면 강제 리셋
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_collect import SR, build_streams, collect_files, load_flac
from gate import DOT_COMMIT_BOUNDARY_RE

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import warmup_streaming

MAX_AUDIO_ACCUM_SEC = 90.0
_WORD = re.compile(r"[^\w']+")


def words(s):
    return [w for w in _WORD.split(s.upper()) if w]


def dedup(new_text: str, prev_committed: str, max_overlap: int = 8) -> str:
    """이전 커밋 꼬리와 겹치는 앞부분을 잘라낸다 (리셋 후 재디코딩 중복 방지)."""
    if not prev_committed or not new_text:
        return new_text
    pw, nw = words(prev_committed), words(new_text)
    if not pw or not nw:
        return new_text
    k = min(max_overlap, len(pw), len(nw))
    while k > 0:
        if pw[-k:] == nw[:k]:
            toks = new_text.split()
            return " ".join(toks[k:]).strip()
        k -= 1
    if len(nw) <= max_overlap and " ".join(pw).endswith(" ".join(nw)):
        return ""
    return new_text


class Session:
    """한 스트림의 커밋 상태. 슬롯 리셋을 넘어 유지된다."""

    def __init__(self, policy, unfixed_token_num, count_tokens):
        self.policy = policy
        self.unfixed_token_num = unfixed_token_num
        self.count_tokens = count_tokens
        self.transcript = []          # (audio_sec, text, reason)
        self.slot_committed = ""      # 현재 슬롯 안에서 커밋된 prefix
        self.pending = None           # 확정 대기 온점 후보 (슬롯 내 prefix)
        self.resets = 0
        self.forced_resets = 0

    @property
    def full_text(self):
        return " ".join(t for _, t, _ in self.transcript)

    def _emit(self, slot_text_upto: str, audio_sec: float, reason: str):
        raw = slot_text_upto[len(self.slot_committed):].strip()
        self.slot_committed = slot_text_upto
        text = dedup(raw, self.full_text)
        if text:
            self.transcript.append((round(audio_sec, 3), text, reason))
            return True
        return False

    def step(self, hyp: str, audio_sec: float) -> bool:
        """청크 디코딩 직후 호출. 커밋이 일어났으면 True."""
        hyp = hyp.strip()
        if self.slot_committed and not hyp.startswith(self.slot_committed):
            self.slot_committed = ""  # 재디코딩으로 prefix 깨짐 — 슬롯 기준 초기화
        committed_any = False

        # gate1 = 규칙 1(문맥 확정)만 사용 — 합의 확정 없음, 프론티어 온점은 finish까지 대기
        if self.policy == "gate" and self.pending is not None:
            if hyp.startswith(self.pending):
                committed_any |= self._emit(self.pending, audio_sec, "dot-stable")
            self.pending = None

        pos = len(self.slot_committed)
        while True:
            m = DOT_COMMIT_BOUNDARY_RE.search(hyp, pos)
            if not m:
                break
            end = m.end()
            if self.policy == "naive":
                committed_any |= self._emit(hyp[:end], audio_sec, "dot")
                pos = len(self.slot_committed)
                continue
            if self.count_tokens(hyp[end:]) > self.unfixed_token_num:
                committed_any |= self._emit(hyp[:end], audio_sec, "dot-context")
                pos = len(self.slot_committed)
            else:
                if self.policy == "gate":
                    self.pending = hyp[:end]
                break
        return committed_any

    def flush(self, hyp: str, audio_sec: float):
        hyp = hyp.strip()
        if self.slot_committed and not hyp.startswith(self.slot_committed):
            self.slot_committed = ""
        if hyp[len(self.slot_committed):].strip():
            self._emit(hyp, audio_sec, "finish")
        self.pending = None

    def slot_reset(self):
        self.slot_committed = ""
        self.pending = None
        self.resets += 1


async def run_stream(asr, wav, args, policy, count_tokens):
    chunk_samples = int(round(args.chunk_sec * SR))
    sess = Session(policy, args.unfixed_token_num, count_tokens)

    def new_state():
        return asr.init_streaming_state(
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_sec,
            allowed_languages=["English"],
        )

    state = new_state()
    pos = 0
    while pos < len(wav):
        piece = wav[pos: pos + chunk_samples]
        pos += chunk_samples
        await asr.streaming_transcribe(piece, state)
        audio_sec = min(pos, len(wav)) / SR
        hyp = (state.text or "").strip()
        if not hyp:
            continue
        committed = sess.step(hyp, audio_sec)

        remaining = hyp[len(sess.slot_committed):].strip()
        accum_sec = state.audio_accum.shape[0] / SR
        if committed:
            carry = None
            if remaining:
                carry = state.audio_accum[-chunk_samples:].copy()
            state = new_state()
            if carry is not None:
                state.audio_accum = carry
            sess.slot_reset()
        elif accum_sec > MAX_AUDIO_ACCUM_SEC:
            carry = state.audio_accum[-chunk_samples:].copy()
            state = new_state()
            state.audio_accum = carry
            sess.slot_reset()
            sess.forced_resets += 1

    await asr.finish_streaming_transcribe(state)
    sess.flush((state.text or "").strip(), len(wav) / SR)
    return sess


async def run(args):
    asr = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    await warmup_streaming(asr)
    tok = asr.processor.tokenizer
    count_tokens = lambda s: len(tok.encode(s)) if s.strip() else 0

    files = collect_files(args.test_dir, args.limit, spread=args.spread)
    if args.concat > 1:
        streams = build_streams(files, args.concat, args.gap_sec)
    else:
        streams = [(os.path.relpath(f, args.test_dir), load_flac(f), ref, [(0.0, ref)])
                   for f, ref in files]
    print(f"[live] {len(streams)} streams, policies={args.policies}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("w")

    for i, (name, wav, ref, marks) in enumerate(streams):
        dur = len(wav) / SR
        if args.trailing_silence_sec > 0:
            wav_fed = np.concatenate(
                [wav, np.zeros(int(args.trailing_silence_sec * SR), dtype=np.float32)])
        else:
            wav_fed = wav
        rec = {"file": name, "reference": ref, "duration_sec": round(dur, 3),
               "utterance_marks": marks, "policies": {}}
        for policy in args.policies:
            t0 = time.perf_counter()
            sess = await run_stream(asr, wav_fed, args, policy, count_tokens)
            rec["policies"][policy] = {
                "commits": [[s, t, r] for s, t, r in sess.transcript],
                "resets": sess.resets,
                "forced_resets": sess.forced_resets,
                "wall_sec": round(time.perf_counter() - t0, 2),
            }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        summary = " | ".join(
            f'{p}:{len(rec["policies"][p]["commits"])}c' for p in args.policies)
        print(f"[{i+1}/{len(streams)}] {name} dur={dur:.1f}s {summary}")

    fout.close()
    print(f"[live] wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--test-dir", default="evaluation/LibriSpeech/LibriSpeech/test-other")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--chunk-sec", type=float, default=2.0)
    p.add_argument("--trailing-silence-sec", type=float, default=4.0)
    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=3072)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--spread", action="store_true")
    p.add_argument("--concat", type=int, default=1)
    p.add_argument("--gap-sec", type=float, default=1.0)
    p.add_argument("--policies", nargs="+", default=["naive", "gate"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
