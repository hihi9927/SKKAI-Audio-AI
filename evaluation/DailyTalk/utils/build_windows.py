#!/usr/bin/env python3
"""대화 안의 연속 발화를 30초 미만 창으로 묶어 학습 샘플을 만든다.

발화 하나가 샘플 하나이던 것을 7~8개짜리 창으로 바꾼다. 노리는 것은 하나다 —
**발화 끝 `<SEG>` 를 "오디오가 끝났다" 는 단서에서 떼어놓는 것.** 지금 학습 데이터는
full 행 11,207개 전부가 "오디오 끝 = `<SEG>`" 라, 모델이 문장 완결이 아니라 오디오가
끊기는 것을 신호로 배울 수 있다. 이어붙이면 창 안쪽의 SEG 는 뒤에 다음 발화가 바로
이어지므로 그 단서가 없다.

창 길이 상한이 28초인 이유는 특징 추출기다. `preprocessor_config.json` 이
`chunk_length: 30` / `n_samples: 480000` 인 Whisper 계열이라 **30초에서 오디오가 잘린다.**
넘으면 들리지 않는 뒷부분까지 텍스트 라벨로 주게 되어 할루시네이션을 직접 가르친다.
2초는 리샘플링 오차와 무음 누적분의 여유다.

절단 창(`--partial-ratio`)은 반대 신호를 준다. 창의 마지막 발화를 문장 중간에서 자르고
끝-SEG 를 붙이지 않아, "오디오가 끝나도 커밋하지 마라" 를 **긴 문맥에서** 가르친다.
기존 partial 은 3초짜리 짧은 문맥에서 같은 것을 가르쳤는데, 실제로 커밋이 무너지는 곳은
문맥이 길어진 뒤다. 절단 시각은 `partial_all.json` 에 이미 계산돼 있어 forced aligner 를
다시 돌리지 않는다 (발화 10,966개 = 46.1% 가 절단 시각 보유).

분할은 **대화 단위**다. `build_splits.py` 는 발화를 무작위로 섞지만, 창은 한 대화의 발화를
묶으므로 그 방식을 쓰면 같은 창의 발화가 train 과 test 로 갈린다.

    python evaluation/DailyTalk/utils/build_windows.py \
        --seg evaluation/DailyTalk/transcribe/new_seg_all_T2_mg2.json

출력:
    Qwen3-ASR/finetuning/data/DailyTalk/window_audio/w_{대화}_{창}.wav   16kHz mono
    Qwen3-ASR/finetuning/data/DailyTalk/{train,val,test}.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PREFIX = "language English<asr_text>"
SR = 16000


def load_cuts(path: Path) -> dict:
    """partial_all.json → {원본파일: (절단시각, 절단까지의 seg_text)}"""
    if not path.exists():
        return {}
    src = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for v in src.values():
        for e in v["data"]:
            try:
                t = float(e["end_time"])
            except (TypeError, ValueError, KeyError):
                continue
            if t > 0:
                out[e["original_file"]] = (t, e["seg_text"].strip())
    return out


def pack_equal(utts, durs, max_sec, gap):
    """한 대화의 발화를 **고르게** 나눈다. 탐욕적으로 채우지 않는다.

    탐욕은 앞 창을 상한까지 채우고 남은 것을 뒤로 미뤄, 발화 1~2개짜리 쪼가리 창을
    만든다. 실측(전 대화): 5초 미만 창이 276개 생긴다. 그런 창은 짧은 오디오 끝에
    `<SEG>` 가 붙은 모양이라 **지금 학습 데이터와 똑같고**, 오디오 끝과 SEG 를
    떼어놓겠다는 이 작업의 목적에 정면으로 역행한다.

    필요한 창 개수를 먼저 정하고(`ceil(전체 / 상한)`) 그 수로 발화를 균등 분배한다.
    창 평균 길이는 바뀌지 않지만(전체 오디오 ÷ 창 개수는 그대로다) 5초 미만 창이
    0이 된다. 발화 길이가 고르지 않아 상한을 넘는 묶음이 나오면 창을 하나 늘려 다시 나눈다.
    """
    n = len(utts)
    total = sum(durs[e["file"]] for e in utts) + gap * (n - 1)
    k = max(1, math.ceil(total / max_sec))
    while k <= n:
        groups = [utts[round(i * n / k):round((i + 1) * n / k)] for i in range(k)]
        groups = [g for g in groups if g]
        if all(sum(durs[e["file"]] for e in g) + gap * (len(g) - 1) <= max_sec
               for g in groups):
            return groups
        k += 1
    return [[e] for e in utts]  # 발화 하나가 상한을 넘는 경우 (DailyTalk 최대 10초라 미발생)


def render(win, cut, audio_dir, gap_lo, gap_hi, rng):
    """창 → (16kHz 파형, 텍스트). 마지막 발화는 cut 이 있으면 그 시각까지만 쓴다."""
    parts, texts = [], []
    for k, e in enumerate(win):
        if k:
            parts.append(np.zeros(int(rng.uniform(gap_lo, gap_hi) * SR), dtype=np.float32))
        wav, _ = librosa.load(str(audio_dir / f"{e['file']}.wav"), sr=SR, mono=True)
        if k == len(win) - 1 and cut is not None:
            wav = wav[: int(cut[0] * SR)]
            texts.append(cut[1])          # 절단본은 끝-SEG 가 없다
        else:
            t = (e.get("seg_text") or e["text"]).strip()
            if not t.endswith("<SEG>"):
                t += " <SEG>"
            texts.append(t)
        parts.append(wav)
    return np.concatenate(parts), " ".join(texts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="evaluation/DailyTalk/transcribe/new_seg_all_t2.json",
                    help="T 컷이 끝난 라벨 (rank_to_seg*.py 산물)")
    ap.add_argument("--partial", default="evaluation/DailyTalk/transcribe/partial_all.json",
                    help="절단 시각 출처. 없으면 절단 창을 만들지 않는다")
    ap.add_argument("--audio-dir", default="Qwen3-ASR/finetuning/data/DailyTalk/audio")
    ap.add_argument("--out-audio", default="Qwen3-ASR/finetuning/data/DailyTalk/window_audio")
    ap.add_argument("--outdir", default="Qwen3-ASR/finetuning/data/DailyTalk")
    ap.add_argument("--window-sec", type=float, default=28.0, help="창 상한 (30초 벽 − 여유)")
    ap.add_argument("--gap-min", type=float, default=0.1, help="발화 간 무음 하한")
    ap.add_argument("--gap-max", type=float, default=0.3, help="발화 간 무음 상한")
    ap.add_argument("--partial-ratio", type=float, default=0.5, help="절단으로 끝내는 창의 비율")
    ap.add_argument("--val", type=int, default=1500, help="val 목표 발화 수")
    ap.add_argument("--test", type=int, default=1500, help="test 목표 발화 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-dialogs", type=int, default=0, help="앞 N개 대화만 (스모크용)")
    ap.add_argument("--dry-run", action="store_true", help="오디오를 쓰지 않고 통계만")
    a = ap.parse_args()

    audio_dir = Path(a.audio_dir)
    src = json.loads(Path(a.seg).read_text(encoding="utf-8"))
    cuts = load_cuts(Path(a.partial))
    have = {p.stem for p in audio_dir.glob("*.wav")}

    # 대화별 발화 (idx 오름차순). 오디오 없는 발화는 뺀다 — 순서가 어긋나면 정렬이 깨진다.
    dialogs = {}
    dropped = 0
    for gk, v in src.items():
        rows = [e for e in v["data"] if e["file"] in have]
        dropped += len(v["data"]) - len(rows)
        rows.sort(key=lambda e: int(e["file"].split("_")[0]))
        if rows:
            dialogs[gk] = rows
    if dropped:
        print(f"경고: 오디오 없는 {dropped}건 제외")

    durs = {}
    for rows in dialogs.values():
        for e in rows:
            durs[e["file"]] = librosa.get_duration(path=str(audio_dir / f"{e['file']}.wav"))

    keys = sorted(dialogs, key=lambda k: int(k) if k.isdigit() else k)
    if a.limit_dialogs:
        keys = keys[: a.limit_dialogs]

    # 대화 단위 분할 — 목표 발화 수를 채울 때까지 대화를 담는다
    rng = random.Random(a.seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    split_of, n = {}, 0
    for k in shuffled:
        name = "test" if n < a.test else ("val" if n < a.test + a.val else "train")
        split_of[k] = name
        n += len(dialogs[k])

    out_audio = Path(a.out_audio)
    if not a.dry_run:
        out_audio.mkdir(parents=True, exist_ok=True)

    # 1) 먼저 창을 다 만든다 — 절단 대상은 그 다음에 고른다.
    gap_avg = (a.gap_min + a.gap_max) / 2
    plan = []
    for k in keys:
        for wi, win in enumerate(pack_equal(dialogs[k], durs, a.window_sec, gap_avg)):
            plan.append([k, wi, win, None])

    # 2) 절단 창 고르기. 마지막 발화에 절단 시각이 있는 창만 대상이라, 요청 비율보다
    #    가능 비율이 낮으면 가능한 것을 전부 쓴다. 창 경계를 절단 가능한 발화에 맞춰
    #    옮기지는 않는다 — 균등 분배가 깨지고 발화가 버려진다.
    eligible = [i for i, (_k, _w, win, _c) in enumerate(plan) if win[-1]["file"] in cuts]
    want = int(len(plan) * a.partial_ratio)
    chosen = eligible if want >= len(eligible) else rng.sample(eligible, want)
    for i in chosen:
        plan[i][3] = cuts[plan[i][2][-1]["file"]]
    print(f"절단 대상 창 {len(eligible)}/{len(plan)} 가능 ({len(eligible)/max(1,len(plan))*100:.1f}%) "
          f"→ {len(chosen)}개 선택 (요청 {a.partial_ratio:.0%})")

    buckets = {"train": [], "val": [], "test": []}
    stat = {"win": 0, "cut": 0, "utt": 0, "sec": 0.0}
    for k, wi, win, cut in plan:
        wav, text = render(win, cut, audio_dir, a.gap_min, a.gap_max, rng)
        p = out_audio / f"w_{k}_{wi:02d}.wav"
        if not a.dry_run:
            sf.write(str(p), wav, SR)
        buckets[split_of[k]].append({"audio": str(p.resolve()), "text": PREFIX + text})
        stat["win"] += 1
        stat["cut"] += cut is not None
        stat["utt"] += len(win)
        stat["sec"] += len(wav) / SR

    for name, rows in buckets.items():
        p = Path(a.outdir) / f"{name}.jsonl"
        if not a.dry_run:
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5} {len(rows):6d}창  → {p}")

    w = stat["win"] or 1
    print(f"\n창 {stat['win']}개 | 발화 {stat['utt']}개 | 창당 발화 {stat['utt']/w:.2f}")
    print(f"절단 창 {stat['cut']}개 ({stat['cut']/w*100:.1f}%) | "
          f"오디오 {stat['sec']/3600:.2f}시간 | 창당 {stat['sec']/w:.1f}초")
    if a.dry_run:
        print("(dry-run — 오디오·jsonl 을 쓰지 않았다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
