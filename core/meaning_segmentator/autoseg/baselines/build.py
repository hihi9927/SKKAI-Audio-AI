"""비교군 라벨 생성기 — 무겁고 캐시 가능한 쪽만 여기서 돈다.

산출: `runs/<run-id>/baselines/<policy>_<tgt>_<split>.json`
      {"policy":…, "tgt":…, "rows":[{"id","text","pieces"}]}

평가(gtx 번역 + BLEU)는 `bleu_eval --baselines <policy>…` 가 이 파일을 조건으로 읽는다.
분리한 이유는 라벨 생성이 GPU 를 쓰고 느린 반면 평가는 번역 캐시로 반복이 싸기 때문이다.

    python -m core.meaning_segmentator.autoseg.baselines.build \
        --run-id en-multi/clean500 --policy mu_prefix --targets de ja
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from core.meaning_segmentator.autoseg.baselines import (  # noqa: E402
    alignatt, causal_align, mu_prefix, punct, syntax_sasst)

SPACED = {"de": True, "es": True, "ja": False, "zh": False, "ko": True}


def load_rows(run_dir: Path, label: str, split: str) -> list[dict]:
    ev = json.loads((run_dir / "prompt_eval" / f"{label}_{split}.json"
                     ).read_text(encoding="utf-8"))
    return [{"id": r["id"], "text": r["text"]} for r in ev["rows"]]


def load_refs(tag: str, tgt: str) -> dict[str, str]:
    path = (_REPO_ROOT / "evaluation" / "ast" / "manifests"
            / f"fleurs_nway_en-{tgt}_{tag}.jsonl")
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            out[e["utt_id"]] = e["tgt_text"]
    return out


def run_policy(policy: str, rows: list[dict], tgt: str, args) -> list[dict]:
    out: list[dict] = []
    t0 = time.time()

    if policy == "punct":
        for r in rows:
            out.append({**r, "pieces": punct.segment(r["text"])})

    elif policy == "causal_align":
        refs = load_refs(args.manifest_tag, tgt)
        aligner = causal_align.CausalAligner(device=args.device)
        skipped = 0
        for i, r in enumerate(rows):
            ref = refs.get(r["id"])
            if not ref:
                skipped += 1
                out.append({**r, "pieces": [r["text"]], "no_ref": True})
                continue
            out.append({**r, "pieces": aligner.segment(r["text"], ref, tgt)})
            if (i + 1) % 100 == 0:
                print(f"  [{tgt}] {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        if skipped:
            print(f"  [{tgt}] 참조 없어 무분절 처리 {skipped}건")

    elif policy == "mu_prefix":
        from core.meaning_segmentator.autoseg.baselines.nmt import Nmt

        nmt = Nmt(src="en", tgt=tgt, device=args.device)
        for i, r in enumerate(rows):
            pieces = mu_prefix.segment(nmt, r["text"], SPACED[tgt], args.n_cands)
            out.append({**r, "pieces": pieces})
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(rows) - i - 1)
                print(f"  [{tgt}] {i + 1}/{len(rows)}  {el:.0f}s  ETA {eta / 60:.1f}m",
                      flush=True)
    elif policy == "syntax":
        seg = syntax_sasst.SyntaxSegmenter(max_chunk=args.max_chunk)
        for r in rows:
            out.append({**r, "pieces": seg.segment(r["text"])})

    elif policy == "alignatt":
        from core.meaning_segmentator.autoseg.baselines.nmt import Nmt

        nmt = Nmt(src="en", tgt=tgt, device=args.device,
                  attentions=True, attn_layer=args.attn_layer)
        for i, r in enumerate(rows):
            out.append({**r, "pieces": alignatt.segment(nmt, r["text"], args.f)})
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(rows) - i - 1)
                print(f"  [{tgt}] {i + 1}/{len(rows)}  {el:.0f}s  ETA {eta / 60:.1f}m",
                      flush=True)
    else:
        raise SystemExit(f"unknown policy: {policy}")

    ks = [len(r["pieces"]) for r in out]
    print(f"  [{tgt}] {policy}: 평균 조각수 {sum(ks) / len(ks):.2f}, "
          f"무분절 문장 {sum(1 for k in ks if k == 1)}/{len(ks)}, "
          f"{time.time() - t0:.0f}s")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Table 1a 비교군 라벨 생성")
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy", required=True,
                   choices=["punct", "causal_align", "mu_prefix",
                            "syntax", "alignatt"])
    p.add_argument("--targets", nargs="+", default=["de", "ja"])
    p.add_argument("--label", default="auto_best")
    p.add_argument("--split", default="test")
    p.add_argument("--manifest-tag", default="clean500")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-cands", type=int, default=10, help="Zhang 2020 beam 후보 수")
    p.add_argument("--f", type=int, default=2, help="AlignAtt 노브 — 최근 f 어절")
    p.add_argument("--attn-layer", type=int, default=5, help="AlignAtt 정렬 층")
    p.add_argument("--max-chunk", type=int, default=7, help="SASST 최대 청크 어절")
    p.add_argument("--limit", type=int, default=0, help="스모크용 앞 N 문장")
    args = p.parse_args()

    run_dir = _HERE.parents[1] / "runs" / args.run_id
    rows = load_rows(run_dir, args.label, args.split)
    if args.limit:
        rows = rows[: args.limit]
    out_dir = run_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # punct 는 타깃 독립이라 한 번만 낸다.
    targets = ["all"] if args.policy in ("punct", "syntax") else args.targets
    for tgt in targets:
        print(f"[{args.policy}] en→{tgt}, {len(rows)}문장")
        res = run_policy(args.policy, rows, tgt if tgt != "all" else "de", args)
        path = out_dir / f"{args.policy}_{tgt}_{args.split}.json"
        path.write_text(json.dumps(
            {"policy": args.policy, "tgt": tgt, "split": args.split,
             "n_cands": args.n_cands if args.policy == "mu_prefix" else None,
             "rows": res}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
