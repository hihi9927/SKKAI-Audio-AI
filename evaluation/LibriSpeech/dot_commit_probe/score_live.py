"""probe_live.py 결과 채점 — 정책별 WER / 커밋 수 / 지연."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import edit_distance, norm, sentence_latencies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--gap", type=float, default=1.0, help="concat 발화 사이 무음 길이")
    args = p.parse_args()

    recs = [json.loads(l) for l in open(args.jsonl)]
    policies = list(recs[0]["policies"].keys())
    agg = {p_: dict(err=0, words=0, commits=0, resets=0, forced=0, ftl=[], tail=[],
                    slat=[], reasons={}) for p_ in policies}

    for ri, r in enumerate(recs):
        ref = norm(r["reference"])
        for p_ in policies:
            d = r["policies"][p_]
            commits = [(c[0], c[1], c[2]) for c in d["commits"]]
            hyp = norm(" ".join(t for _, t, _ in commits))
            a = agg[p_]
            a["err"] += edit_distance(ref, hyp)
            a["words"] += len(ref)
            a["commits"] += len(commits)
            a["resets"] += d["resets"]
            a["forced"] += d["forced_resets"]
            for _, _, why in commits:
                a["reasons"][why] = a["reasons"].get(why, 0) + 1
            if commits:
                a["ftl"].append(commits[0][0])
                a["tail"].append(commits[-1][0] - r["duration_sec"])
            a["slat"] += sentence_latencies(r, commits, gap=args.gap)

            if ri < args.show:
                if p_ == policies[0]:
                    print("=" * 78)
                    print(f'{r["file"]} dur={r["duration_sec"]}s')
                    print("  REF :", r["reference"][:300])
                print(f'  {p_} ({len(commits)} commits, resets={d["resets"]}):')
                for sec, t, why in commits:
                    print(f"    @{sec:>6.2f}s [{why:11}] {t!r}")

    n = len(recs)
    print("\n" + "=" * 78)
    print(f'{"policy":6} {"WER%":>7} {"commits":>8} {"c/strm":>7} {"FTL(s)":>8} {"tail(s)":>8} '
          f'{"sentLat":>8} {"resets":>7} {"forced":>7}')
    for p_ in policies:
        a = agg[p_]
        f = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f'{p_:6} {100*a["err"]/max(1,a["words"]):>7.2f} {a["commits"]:>8} '
              f'{a["commits"]/n:>7.2f} {f(a["ftl"]):>8.2f} {f(a["tail"]):>8.2f} '
              f'{f(a["slat"]):>8.2f} {a["resets"]:>7} {a["forced"]:>7}')
    for p_ in policies:
        print(f'  {p_}: {agg[p_]["reasons"]}')


if __name__ == "__main__":
    main()
