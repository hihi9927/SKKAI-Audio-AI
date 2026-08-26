"""contradiction 잡음 바닥 측정 — 결정론적, 번역 호출 0.

NLI 는 **무해한 미완성**에도 0 이 아닌 모순 확률을 준다 (실측 0.001~0.16). 그래서
경계 contradiction 0.15 가 "잘못 자른 벌점"인지 "NLI 가 원래 주는 기본 점수"인지
구분되지 않고, 순위 정렬도(Spearman)의 음수가 진짜 순위 오류인지 위치(=hypothesis
길이) 교란인지도 판정할 수 없다.

측정: full 번역을 어절 단위 prefix 로 자른다. prefix 는 **정의상 무해한 미완성**이다 —
같은 번역의 앞부분이므로 모순일 수 없다. 그 prefix 들을 hypothesis 로 넣어 나오는
contradiction 확률이 곧 잡음 바닥 c0 이고, 길이별로 집계하면 길이 의존성이 드러난다.

  PYTHONPATH=. python -m core.meaning_segmentator.autoseg.noise_floor \
      --run-id ko-en/run03 --split test --recheck-t 2

`--recheck-t` 를 주면 그 T 의 경계별 contradiction 에서 c0(hypothesis 길이)를 빼고
순위 정렬도를 다시 계산한다 — 보정 후에도 음수면 순위 오류가 실제라는 뜻이다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import metrics

_HERE = Path(__file__).resolve().parent

# 길이 버킷 상한. 마지막 버킷은 무한대.
#
# **단위는 타깃 언어의 표기 체계를 따른다** — 띄어쓰기 타깃은 어절, 아니면 문자다.
# 어절로 고정하면 ja/zh 에서 `full.split()` 이 문장 전체를 토큰 1개로 세어
# `len < 2` 로 전량 탈락한다 (실측: 150문장 중 ja 12개·zh 25개 prefix 만 살아남았고
# 버킷 대부분이 n=0~2 였다). 바닥이 사실상 미측정인 채로 보정이 돌면 순위 진단이
# 통째로 무의미해진다.
#
# 바닥은 **타깃 안에서만** 쓰이므로 타깃마다 단위가 달라도 문제가 없다. 무공백 타깃의
# 상한은 한자·가나 기준 어절당 대략 3자로 잡아 어절 버킷을 스케일한다.
BUCKETS = [2, 4, 6, 9, 14]
_UNSPACED_SCALE = 3


def buckets_for(tgt_spaced: bool = True) -> list[int]:
    return BUCKETS if tgt_spaced else [b * _UNSPACED_SCALE for b in BUCKETS]


def bucket_of(n: int, tgt_spaced: bool = True) -> str:
    bs = buckets_for(tgt_spaced)
    lo = 1
    for hi in bs:
        if n <= hi:
            return f"{lo}-{hi}"
        lo = hi + 1
    return f"{bs[-1] + 1}+"


def bucket_labels(tgt_spaced: bool = True) -> list[str]:
    bs = buckets_for(tgt_spaced)
    labels, lo = [], 1
    for hi in bs:
        labels.append(f"{lo}-{hi}")
        lo = hi + 1
    return labels + [f"{bs[-1] + 1}+"]


def split_units(text: str, tgt_spaced: bool = True) -> list[str]:
    """타깃 표기 체계에 맞는 토큰. 띄어쓰기 언어는 어절, 아니면 문자."""
    if tgt_spaced:
        return (text or "").split()
    return list(re.sub(r"\s+", "", text or ""))


def measure_floor(fulls: list[str], backend, max_prefixes_per_sentence: int = 20,
                  tgt_spaced: bool = True) -> dict:
    """full 번역의 prefix 를 hypothesis 로 넣은 contradiction 확률 분포.

    prefix 는 1..(n−1) 토큰 전부다 (마지막 = 문장 전체는 identity 라 제외). 토큰 단위는
    **타깃 표기 체계**를 따른다 (`tgt_spaced`) — 어절로 고정하면 ja/zh 가 통째로 탈락한다.

    반환: 버킷별 {n, mean, p50, p90} + 전체 통계 + 쓰인 단위(`tgt_spaced`).
    `floor_lookup` 이 그 단위를 읽어 같은 버킷 체계로 조회한다.
    """
    joiner = " " if tgt_spaced else ""
    prems, hyps, lens = [], [], []
    for full in fulls:
        units = split_units(full, tgt_spaced)
        if len(units) < 2:
            continue
        cut = min(len(units) - 1, max_prefixes_per_sentence)
        for i in range(1, cut + 1):
            prems.append(full)
            hyps.append(joiner.join(units[:i]))
            lens.append(i)
    scores = backend.score(prems, hyps)

    by_bucket: dict[str, list[float]] = {b: [] for b in bucket_labels(tgt_spaced)}
    for n, s in zip(lens, scores):
        by_bucket[bucket_of(n, tgt_spaced)].append(s)

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "p50": None, "p90": None}
        v = sorted(vals)
        return {"n": len(v),
                "mean": round(sum(v) / len(v), 4),
                "p50": round(v[len(v) // 2], 4),
                "p90": round(v[min(len(v) - 1, int(0.9 * (len(v) - 1)))], 4)}

    return {
        "backend": backend.name,
        "tgt_spaced": tgt_spaced,
        "unit": "words" if tgt_spaced else "characters",
        "n_sentences": len(fulls),
        "n_prefixes": len(scores),
        "overall": stats(scores),
        "by_length_bucket": {b: stats(v) for b, v in by_bucket.items()},
    }


def floor_lookup(floor: dict, hyp_units: int) -> float:
    """길이별 바닥값 c0(len). 버킷 mean, 비면 전체 mean.

    `hyp_units` 는 **바닥을 잰 것과 같은 단위**여야 한다 — `floor["tgt_spaced"]` 가
    그 단위를 알려주므로 호출부는 그것을 보고 세면 된다. 옛 파일에는 이 키가 없으므로
    어절로 간주한다 (그때는 어절 전용이었다)."""
    b = floor["by_length_bucket"].get(
        bucket_of(hyp_units, floor.get("tgt_spaced", True))) or {}
    if b.get("mean") is not None:
        return b["mean"]
    return floor["overall"]["mean"] or 0.0


def recheck_ranks(rows: list[dict], T: int, floor: dict) -> dict:
    """바닥 보정 후 순위 정렬도 재계산.

    경계 j 의 hypothesis 길이 = 조각 번역 1..j+1 을 이어붙인 토큰 수 — 단위는 바닥을
    잰 것과 같아야 하므로 `floor["tgt_spaced"]` 를 따른다.
    보정: c' = max(0, c − c0(길이)). 보정 전/후 Spearman 과 순위별 평균을 함께 낸다.
    """
    tag_re = re.compile(r"<SEG:(\d+)>")
    tgt_spaced = floor.get("tgt_spaced", True)
    key = str(T)
    raw_pairs, cor_pairs = [], []
    by_rank_raw: dict[int, list[float]] = {}
    by_rank_cor: dict[int, list[float]] = {}

    for r in rows:
        d = (r.get("by_T") or {}).get(key)
        if not d:
            continue
        ranks = [int(m) for m in tag_re.findall(d.get("seg_text") or "")]
        contras = (d.get("pieces_contra") or [])[:-1]
        pieces_tgt = d.get("pieces_tgt") or []
        if len(ranks) != len(contras) or not ranks:
            continue
        sent_raw, sent_cor = [], []
        for j, (rk, c) in enumerate(zip(ranks, contras)):
            hyp = " ".join(p for p in pieces_tgt[: j + 1] if p)
            hyp_len = len(split_units(hyp, tgt_spaced))
            cc = max(0.0, c - floor_lookup(floor, max(1, hyp_len)))
            sent_raw.append((rk, c))
            sent_cor.append((rk, cc))
            by_rank_raw.setdefault(min(rk, 6), []).append(c)
            by_rank_cor.setdefault(min(rk, 6), []).append(cc)
        if len(sent_raw) >= 3:
            raw_pairs.append(sent_raw)
            cor_pairs.append(sent_cor)

    def mean_spearman(sent_pairs):
        cors = []
        for s in sent_pairs:
            c = metrics._spearman([float(a) for a, _ in s], [float(b) for _, b in s])
            if c is not None:
                cors.append(c)
        return (round(sum(cors) / len(cors), 4) if cors else None), len(cors)

    sp_raw, n_raw = mean_spearman(raw_pairs)
    sp_cor, n_cor = mean_spearman(cor_pairs)
    return {
        "T": T,
        "spearman_raw": sp_raw, "spearman_corrected": sp_cor, "n_sentences": n_cor,
        "by_rank_raw": {k: round(sum(v) / len(v), 4)
                        for k, v in sorted(by_rank_raw.items())},
        "by_rank_corrected": {k: round(sum(v) / len(v), 4)
                              for k, v in sorted(by_rank_cor.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="contradiction 잡음 바닥 측정")
    p.add_argument("--run-id", required=True, help="runs/ 이하 경로 (예: ko-en/run03)")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--recheck-t", type=int, default=None,
                   help="이 T 의 경계 contradiction 을 바닥 보정해 순위 정렬도 재계산")
    p.add_argument("--max-prefixes", type=int, default=20)
    args = p.parse_args()

    run_dir = _HERE.parent / "runs" / args.run_id
    rows_path = run_dir / (f"{args.split}_rows.json" if args.split == "test"
                           else None)
    if args.split != "test":
        # train/dev 는 이터레이션별 파일 — 마지막 이터레이션 것을 쓴다
        iters = sorted(run_dir.glob(f"iter_*/{args.split}_rows.json"))
        if not iters:
            print(f"{args.split}_rows.json 없음: {run_dir}", file=sys.stderr)
            return 1
        rows_path = iters[-1]
    if not rows_path.exists():
        print(f"rows 파일 없음: {rows_path}", file=sys.stderr)
        return 1

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    backend = metrics.make_contradiction_backend()

    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    fulls = [r.get("full_trans") or "" for r in rows if r.get("full_trans")]
    print(f"[floor] {len(fulls)}문장, 백엔드 {backend.name}", flush=True)

    floor = measure_floor(fulls, backend, args.max_prefixes,
                          tgt_spaced=bool(cfg.get("tgt_spaced", True)))
    out_path = run_dir / f"noise_floor_{args.split}.json"

    print(f"\n전체: n={floor['n_prefixes']} mean={floor['overall']['mean']} "
          f"p50={floor['overall']['p50']} p90={floor['overall']['p90']}")
    print(f"\n길이({floor['unit']})  n     mean    p50     p90")
    for b, s in floor["by_length_bucket"].items():
        if s["n"]:
            print(f"{b:>9} {s['n']:>5} {s['mean']:>7} {s['p50']:>7} {s['p90']:>7}")

    result = {"floor": floor}
    if args.recheck_t is not None:
        rc = recheck_ranks(rows, args.recheck_t, floor)
        result["rank_recheck"] = rc
        print(f"\n[순위 재검 T={rc['T']}] Spearman 보정 전 {rc['spearman_raw']} "
              f"→ 보정 후 {rc['spearman_corrected']} (n={rc['n_sentences']})")
        print("rank  raw    corrected")
        for k in rc["by_rank_raw"]:
            print(f"{k:>4} {rc['by_rank_raw'][k]:>6} {rc['by_rank_corrected'].get(k):>9}")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[done] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
