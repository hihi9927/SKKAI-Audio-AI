#!/usr/bin/env python3
"""CoVoST2 en test 에서 층화 무작위 서브샘플 발화 id 를 고른다.

**언어별로 따로 뽑으면 안 된다.** en→de / en→ja / en→zh-CN 은 같은 영어 클립
15,531개를 공유하므로, id 를 한 번만 골라 세 언어에 똑같이 쓰면 언어 간 비교가
자동으로 매칭된다. 이 스크립트가 그 "한 번"을 담당하고, manifest 빌더는
여기서 나온 id 목록만 받는다.

층화 기준은 **오디오 길이**다. 그냥 무작위로 뽑으면 짧은 발화가 몰릴 수 있고,
짧은 발화는 커밋이 발화당 1회로 수렴해 분절 정책 간 차이를 지운다(= seg 축에
유리한 방향으로 편향). 전체 test 의 길이 분포를 십분위로 나눠 각 구간에서
비례 추출한다.

화자 편중도 막는다 — 한 화자당 최대 `--max-per-speaker` 개까지만 뽑는다.
en_de test 는 화자 9,472명 / 클립 15,531개라, 3,000개를 화자당 1개로 뽑을 수 있다.

    python evaluation/ast/select_covost2_subset.py --n 3000 \
        --out evaluation/ast/subsets/covost2_en_test_n3000.json
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# 세 언어가 같은 영어 오디오를 공유한다는 전제를 여기서 검증한다.
CONFIGS = ["en_de", "en_ja", "en_zh-CN"]
AUDIO_CONFIG = "en_de"   # 길이·화자 정보를 읽어올 기준 (오디오는 세 config 이 동일)

# CoVoST2 test 에는 정제 과정에서 지우려다 남은 행이 15,531 중 14개(0.09%) 있다.
# 번역 필드가 통째로 `[TO REMOVE]` / `TO REMOVE` 이거나, 원문이 영어가 아니다
# (예: common_voice_en_19210802 은 원문이 페르시아어다).
# 참조가 플레이스홀더면 그 발화의 BLEU/COMET 은 의미가 없다. 대부분 **한 언어에서만**
# 오염돼 있지만, 세 언어가 같은 id 집합을 써야 언어 간 비교가 성립하므로 통째로 뺀다.
_PLACEHOLDER_RE = re.compile(r"^\W*to\s*remove\W*$", re.I)


def is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(text or ""))


def nonlatin_ratio(text: str) -> float:
    """알파벳 문자 중 라틴 범위(U+02FF 이하) 밖의 비율. 영어가 아닌 원문 탐지용."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x02FF) / len(letters)


def read_meta(root: Path, config: str, split: str, with_audio: bool) -> dict[str, dict]:
    """id → {client_id, sentence, translation, duration?}. 오디오는 필요할 때만 훑는다."""
    import pyarrow.parquet as pq

    files = sorted((root / config).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"parquet 이 없습니다: {root / config}/{split}-*.parquet\n"
            f'  hf download fixie-ai/covost2 --repo-type dataset '
            f'--include "{config}/{split}-*.parquet" --local-dir {root}'
        )
    cols = ["id", "client_id", "sentence", "translation"]
    if with_audio:
        cols.append("audio")

    out: dict[str, dict] = {}
    for f in files:
        t = pq.read_table(f, columns=cols)
        d = t.to_pydict()
        for i, uid in enumerate(d["id"]):
            rec = {
                "client_id": d["client_id"][i],
                "sentence": (d["sentence"][i] or "").strip(),
                "translation": (d["translation"][i] or "").strip(),
            }
            if with_audio:
                # 전체 디코딩 없이 헤더만 읽는다 — 15,531개라 디코딩하면 수십 분이다.
                try:
                    info = sf.info(io.BytesIO(d["audio"][i]["bytes"]))
                    rec["duration"] = info.frames / info.samplerate
                except Exception:
                    rec["duration"] = None
            out[uid] = rec
        del t, d
    return out


def select(args) -> int:
    root = Path(args.covost_root).expanduser().resolve()

    print(f"parquet 로드: {root}")
    meta = {}
    for cfg in CONFIGS:
        meta[cfg] = read_meta(root, cfg, args.split, with_audio=(cfg == AUDIO_CONFIG))
        print(f"  {cfg:10s} {len(meta[cfg]):,}행")

    # ── 세 언어 공통 & 번역이 실제로 있는 발화만 후보로 ────────────────────────
    common = set(meta[CONFIGS[0]])
    for cfg in CONFIGS[1:]:
        common &= set(meta[cfg])
    print(f"\n세 언어 공통 id: {len(common):,}")

    base = meta[AUDIO_CONFIG]
    pool = []
    drop = Counter()
    for uid in sorted(common):
        if not base[uid]["sentence"]:
            drop["원문 없음"] += 1; continue
        if any(not meta[c][uid]["translation"] for c in CONFIGS):
            drop["번역 없음(한 언어 이상)"] += 1; continue
        if is_placeholder(base[uid]["sentence"]) or any(
                is_placeholder(meta[c][uid]["translation"]) for c in CONFIGS):
            drop["[TO REMOVE] 플레이스홀더"] += 1; continue
        if nonlatin_ratio(base[uid]["sentence"]) > args.max_nonlatin:
            drop["원문이 영어가 아님"] += 1; continue
        dur = base[uid]["duration"]
        if dur is None:
            drop["오디오 읽기 실패"] += 1; continue
        if not (args.min_duration <= dur <= args.max_duration):
            drop["길이 범위 밖"] += 1; continue
        pool.append((uid, dur, base[uid]["client_id"]))

    print(f"후보 발화: {len(pool):,}")
    for k, v in drop.most_common():
        print(f"  제외 — {k}: {v:,}")
    if len(pool) < args.n:
        print(f"\n후보({len(pool)})가 요청 수({args.n})보다 적습니다.", file=sys.stderr)
        return 2

    durs = np.array([d for _, d, _ in pool])
    print(f"\n전체 후보 길이: 평균 {durs.mean():.2f}s 중앙 {np.median(durs):.2f}s "
          f"p10 {np.percentile(durs,10):.2f} p90 {np.percentile(durs,90):.2f}")

    # ── 길이 십분위로 층화, 층마다 비례 추출 ─────────────────────────────────
    edges = np.percentile(durs, np.linspace(0, 100, args.strata + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    layer = defaultdict(list)
    for uid, d, cid in pool:
        k = int(np.searchsorted(edges, d, side="right") - 1)
        layer[min(max(k, 0), args.strata - 1)].append((uid, d, cid))

    rng = random.Random(args.seed)
    chosen, per_speaker = [], Counter()
    # 층 크기에 비례해 배분하되, 반올림 오차는 마지막 층이 흡수한다.
    quota = [round(args.n * len(layer[k]) / len(pool)) for k in range(args.strata)]
    quota[-1] += args.n - sum(quota)

    for k in range(args.strata):
        cand = layer[k][:]
        rng.shuffle(cand)
        got = 0
        for uid, d, cid in cand:
            if got >= quota[k]:
                break
            if per_speaker[cid] >= args.max_per_speaker:
                continue
            chosen.append((uid, d, cid)); per_speaker[cid] += 1; got += 1
        if got < quota[k]:
            # 화자 상한 때문에 못 채운 층 — 상한을 풀지 않고 부족분을 보고한다.
            print(f"  [주의] 층{k} 할당 {quota[k]}개 중 {got}개만 채움 "
                  f"(화자 상한 {args.max_per_speaker})")

    # 부족분을 층 무시하고 채우면 층화가 깨지므로, 남는 자리는 층 크기 순으로 재배분
    if len(chosen) < args.n:
        taken = {u for u, _, _ in chosen}
        rest = [x for x in pool if x[0] not in taken
                and per_speaker[x[2]] < args.max_per_speaker]
        rng.shuffle(rest)
        need = args.n - len(chosen)
        for uid, d, cid in rest[:need]:
            chosen.append((uid, d, cid)); per_speaker[cid] += 1
        print(f"  층화 후 부족분 {need}개를 후보 전체에서 무작위 보충")

    chosen.sort(key=lambda x: x[0])
    sel_dur = np.array([d for _, d, _ in chosen])

    print(f"\n선택 {len(chosen):,}발화 / 오디오 {sel_dur.sum()/3600:.2f}시간")
    print(f"  길이: 평균 {sel_dur.mean():.2f}s 중앙 {np.median(sel_dur):.2f}s "
          f"p10 {np.percentile(sel_dur,10):.2f} p90 {np.percentile(sel_dur,90):.2f}")
    print(f"  화자 {len(per_speaker):,}명 (화자당 최대 {max(per_speaker.values())}개)")
    print(f"  분포 검증 — 전체 대비 평균 차 {sel_dur.mean()-durs.mean():+.3f}s, "
          f"중앙 차 {np.median(sel_dur)-np.median(durs):+.3f}s")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "utt_ids": [u for u, _, _ in chosen],
        "provenance": {
            "dataset": "fixie-ai/covost2",
            "split": args.split,
            "configs": CONFIGS,
            "audio_config": AUDIO_CONFIG,
            "n_requested": args.n,
            "n_selected": len(chosen),
            "seed": args.seed,
            "strata": args.strata,
            "strata_edges_sec": [round(float(e), 4) for e in edges],
            "max_per_speaker": args.max_per_speaker,
            "max_nonlatin": args.max_nonlatin,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "pool_size": len(pool),
            "pool_total_rows": len(meta[AUDIO_CONFIG]),
            "dropped": dict(drop),
            "selected_audio_hours": round(float(sel_dur.sum()) / 3600, 4),
            "selected_duration_stats": {
                "mean": round(float(sel_dur.mean()), 4),
                "median": round(float(np.median(sel_dur)), 4),
                "p10": round(float(np.percentile(sel_dur, 10)), 4),
                "p90": round(float(np.percentile(sel_dur, 90)), 4),
            },
            "pool_duration_stats": {
                "mean": round(float(durs.mean()), 4),
                "median": round(float(np.median(durs)), 4),
                "p10": round(float(np.percentile(durs, 10)), 4),
                "p90": round(float(np.percentile(durs, 90)), 4),
            },
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="CoVoST2 en test 층화 서브샘플 id 선택")
    p.add_argument("--covost-root", default="~/datasets/covost2")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--strata", type=int, default=10, help="길이 층 개수(십분위=10)")
    p.add_argument("--max-per-speaker", type=int, default=1,
                   help="한 화자에서 뽑을 최대 클립 수. 화자 편중 방지")
    p.add_argument("--max-nonlatin", type=float, default=0.5,
                   help="원문의 비라틴 문자 비율이 이 값을 넘으면 영어가 아닌 것으로 보고 제외")
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--out", required=True)
    sys.exit(select(p.parse_args()))


if __name__ == "__main__":
    main()
