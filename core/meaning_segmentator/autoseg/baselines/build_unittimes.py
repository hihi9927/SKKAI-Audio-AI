"""소스 언어별 **단위 종료 시각**을 Qwen 강제정렬로 뽑는다 (de/ja/zh 등 비영어 소스).

`build_wordtimes_qwen.py` 는 영어 전용이었다 — 정렬 언어가 `"English"` 로 박혀 있고
`to_word_ends` 가 공백으로 어절을 나눈다. 일본어·중국어에는 공백이 없어 그대로는 못 쓴다.

**단위는 `pipeline.unit_count` 와 같은 규칙을 따른다**: 띄어쓰기 언어면 어절, 아니면
공백 제거 후 글자. 그래야 `bleu_eval.laal_ms` 가 누적 단위 수로 인덱싱하는 규약이 맞는다.
저장 필드명은 `word_end_ms` 그대로 둔다 — 소비자(`laal_ms`)가 그 이름을 쓰고, 값의
의미는 "단위 i 까지 발화가 끝난 시각"으로 동일하다.

산출: evaluation/ast/manifests/fleurs_nway_{lang}_multi2en_loop240_unittimes.json
      {utt_id: {"wav":…, "dur_ms":…, "speech_ms":…, "word_end_ms":[…]}}

`speech_ms` 는 첫 span 시작 ~ 마지막 span 끝이다. FLEURS 녹음은 앞뒤 무음이 1~2초씩
있어서 `dur_ms` 로 발화 속도를 재면 과소평가된다 (ja 실측: 총 11.1초 중 발화 7.4초).

    .venv/bin/python -m core.meaning_segmentator.autoseg.baselines.build_unittimes --lang ja
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO))

ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
_WS = re.compile(r"\s+")

# **쉼의 기준.** 문헌 표준은 200ms(지각 임계 — 이보다 짧으면 사람이 쉼으로 못 느낀다)
# 와 250ms(Goldman-Eisler, IPU 정의에 가장 널리 쓰임)다. 정렬기가 80ms 격자로만 시각을
# 내므로(실측: 타임스탬프 98% 가 80 의 배수) 고를 수 있는 값은 80·160·240·320… 뿐이고,
# 240ms 가 250ms 표준에 가장 가깝다. 종전 160ms 는 지각 임계 아래였다.
#
# 이 값으로 나온 IPU 분포는 **진단용**이다 — `min_gap` 산출에는 쓰지 않는다. 실측에서
# IPU 백분위는 두 독립 앵커(de 3, ko 3)를 동시에 재현하지 못했다: p25 는 de 만(ko 2),
# p30 은 ko 만(de 4) 맞는다. 시간 상수 1200ms 는 둘 다 맞힌다. 즉 청자의 최소 단위는
# 화자가 어떻게 끊느냐가 아니라 시간에 붙어 있다. IPU 는 레지스터 지표로만 쓴다
# (de p25 1360ms vs ko 960ms — 낭독 vs 자발발화 차이가 드러난다).
PAUSE_MS = 240

# (fleurs 디렉토리, 정렬기 언어명, 띄어쓰기 여부)
LANGS = {
    "de": ("de_de", "German", True),
    "ja": ("ja_jp", "Japanese", False),
    "zh": ("cmn_hans_cn", "Chinese", False),
    "ko": ("ko_kr", "Korean", True),
    "en": ("en_us", "English", True),
}


def load_tsv(base: Path, split: str) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    f = base / f"{split}.tsv"
    if not f.exists():
        return out
    with f.open(encoding="utf-8") as fh:
        for c in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(c) >= 6 and c[5].isdigit():
                out.setdefault(c[0], []).append((c[1], int(c[5])))
    return out


def to_ipu_ms(items) -> list[float]:
    """쉼(PAUSE_MS 이상)으로 끊은 덩어리들의 길이(ms).

    화자가 스스로 말을 어떻게 끊는지다. `min_gap` 을 이 분포의 백분위로 잡으면
    코퍼스의 레지스터(낭독 vs 자발발화)에 자동으로 맞춰진다.
    """
    out, cur = [], 0.0
    seq = [it for it in items if (it.text or "").strip()]
    for a, b in zip(seq, seq[1:]):
        gap = (b.start_time - a.end_time) * 1000
        cur += (a.end_time - a.start_time) * 1000
        if gap >= PAUSE_MS:
            out.append(cur + gap)
            cur = 0.0
    if seq:
        cur += (seq[-1].end_time - seq[-1].start_time) * 1000
    if cur:
        out.append(cur)
    return [round(x, 1) for x in out]


def to_unit_ends(items, text: str, dur_s: float, spaced: bool) -> tuple[list[float], float, float]:
    """span 들을 단위 종료 시각으로 접는다. 반환 `(ends, 첫 발화 s, 마지막 발화 s)`.

    span.text 를 원문에서 **순차 탐색**해 문자 위치를 잡는다. 정렬기가 구두점을 빼고
    내놓기도 해서 못 찾은 span 은 건너뛴다.
    """
    if spaced:
        units = _WS.split(text.strip())
        bounds, pos = [], 0
        for w in units:
            j = text.index(w, pos)
            pos = j + len(w)
            bounds.append(pos)            # 이 단위가 끝나는 원문 문자 위치
    else:
        # 공백을 뺀 글자 하나가 단위 하나. 원문 위치 -> 단위 인덱스 표를 만든다.
        bounds = [i + 1 for i, ch in enumerate(text) if not ch.isspace()]

    ends = [0.0] * len(bounds)
    cur = 0
    first = last = None
    for it in items:
        t = (it.text or "").strip()
        if not t:
            continue
        j = text.find(t, cur)
        if j < 0:
            j = text.lower().find(t.lower(), cur)
        if j < 0:
            continue
        cur = j + len(t)
        if first is None:
            first = float(it.start_time)
        last = float(it.end_time)
        for i, e in enumerate(bounds):
            if e >= cur:
                ends[i] = max(ends[i], float(it.end_time))
                break

    run = 0.0                              # 빈 칸은 앞 값으로 채운다 (단조)
    for i, e in enumerate(ends):
        run = max(run, e)
        ends[i] = run
    ends[-1] = max(ends[-1], dur_s * 0.999)
    return ends, (first or 0.0), (last or dur_s)


def main() -> int:
    p = argparse.ArgumentParser(description="비영어 소스 강제정렬 단위 타임스탬프")
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("--manifest", default=None,
                   help="기본: fleurs_nway_{lang}-en_multi2en_loop240.jsonl")
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    langdir, langname, spaced = LANGS[args.lang]
    base = Path.home() / "datasets" / "fleurs" / "data" / langdir
    man = Path(args.manifest or (_REPO / "evaluation" / "ast" / "manifests"
               / f"fleurs_nway_{args.lang}-en_multi2en_loop240.jsonl"))
    dest = Path(args.out or (man.with_name(man.stem + "_unittimes.json")))

    tsv = {s: load_tsv(base, s) for s in ("train", "dev", "test")}
    jobs, missing = [], 0
    for line in man.open(encoding="utf-8"):
        e = json.loads(line)
        # 매니페스트가 wav 경로를 직접 갖고 있으면 그걸 쓴다 (en-de 트랙 형식).
        # 없으면 TSV 에서 talk_id -> 파일명을 찾는다 (n-way 형식).
        if e.get("wav"):
            wav = Path(e["wav"])
            dur = float(e.get("duration") or 0.0) * 1000
            if not wav.exists() or dur <= 0:
                missing += 1
                continue
            jobs.append((e["utt_id"], e["src_text"], wav, dur))
            continue
        split = e.get("fleurs_split") or "train"
        cands = tsv.get(split, {}).get(str(e["talk_id"]))
        if not cands:
            missing += 1
            continue
        fn, ns = sorted(cands, key=lambda x: x[1])[len(cands) // 2]   # 길이 중앙값 녹음
        wav = base / "audio" / split / fn
        if not wav.exists():
            missing += 1
            continue
        jobs.append((e["utt_id"], e["src_text"], wav, ns / 16000 * 1000))

    print(f"[{args.lang}] 정렬 대상 {len(jobs)}건, 오디오 없음 {missing}건", flush=True)
    if not jobs:
        return 2

    al = Qwen3ForcedAligner.from_pretrained(ALIGNER, device_map="cuda:0",
                                            dtype=torch.bfloat16)
    out: dict[str, dict] = {}
    fails: list[str] = []
    t0 = time.time()
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i: i + args.batch]
        try:
            res = al.align(audio=[str(c[2]) for c in chunk],
                           text=[c[1] for c in chunk],
                           language=[langname] * len(chunk))
        except Exception as exc:                       # noqa: BLE001
            fails += [f"{c[0]}:{type(exc).__name__}" for c in chunk]
            continue
        for c, r in zip(chunk, res):
            key, text, wav, dur_ms = c
            try:
                ends, f0, f1 = to_unit_ends(list(r), text, dur_ms / 1000, spaced)
            except Exception as exc:                   # noqa: BLE001
                fails.append(f"{key}:{type(exc).__name__}")
                continue
            out[key] = {"wav": wav.name, "dur_ms": round(dur_ms, 1),
                        "speech_ms": round((f1 - f0) * 1000, 1),
                        "ipu_ms": to_ipu_ms(list(r)),
                        "word_end_ms": [round(e * 1000, 1) for e in ends]}
        done = i + len(chunk)
        if done % 40 == 0 or done >= len(jobs):
            el = time.time() - t0
            print(f"  {done}/{len(jobs)}  {el:.0f}s  "
                  f"ETA {el / done * (len(jobs) - done) / 60:.1f}m", flush=True)

    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.lang}] 성공 {len(out)}/{len(jobs)}, 실패 {len(fails)} -> {dest}")
    if fails:
        print("  실패 예:", fails[:8])
    return 0


if __name__ == "__main__":
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    raise SystemExit(main())
