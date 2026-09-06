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

from ..paths import REPO_ROOT, RUNS_DIR
sys.path.insert(0, str(REPO_ROOT))

from core.meaning_segmentator.autoseg.baselines import (  # noqa: E402
    alignatt, causal_align, mu_prefix, punct, syntax_sasst)

SPACED = {"de": True, "es": True, "ja": False, "zh": False, "ko": True}


def load_rows(run_dir: Path, label: str, split: str,
              dataset: str = "fleurs", tag: str = "", tgt: str = "de") -> list[dict]:
    """평가 문장 목록.

    `prompt_eval/<label>_<split>.json`(제안 정책 산출)이 있으면 **그 순서를 그대로** 따른다 —
    조건들이 같은 인덱스를 공유해야 `bleu_eval` 이 쌍체 비교를 할 수 있다.

    없으면 매니페스트에서 직접 읽는다. 비교군은 제안 라벨과 무관하므로, 라벨링을 다른
    환경에서 하는 동안에도 비교군을 먼저 만들어 둘 수 있다.
    """
    ev_path = run_dir / "prompt_eval" / f"{label}_{split}.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        return [{"id": r["id"], "text": r["text"]} for r in ev["rows"]]

    from core.meaning_segmentator.autoseg.baselines import datasets as _ds

    print(f"  ({ev_path.name} 없음 — 매니페스트에서 문장을 읽는다)")
    return [{"id": k, "text": e.src}
            for k, e in _ds.get(dataset).entries(tag, tgt).items()]


def load_refs(tag: str, tgt: str, dataset: str = "fleurs") -> dict[str, str]:
    from core.meaning_segmentator.autoseg.baselines import datasets as _ds

    return {k: e.ref for k, e in _ds.get(dataset).entries(tag, tgt).items()}


class Checkpoint:
    """진행분을 줄 단위로 흘려 쓴다.

    `alignatt` 가 CUDA `unspecified launch failure` 로 두 번 죽었다 (2026-09-03,
    8,125/15,430 와 12,175/15,430 지점). 종전에는 **끝에 한 번만** 파일을 써서 그때까지의
    한 시간이 통째로 날아갔다. 여기서는 100건마다 `<out>.partial.jsonl` 에 append 하고,
    `--resume` 이 그걸 읽어 남은 것만 돌린다.
    """

    def __init__(self, path: Path, every: int = 100):
        self.f = path.open("a", encoding="utf-8")
        self.every, self.n = every, 0

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1
        if self.n % self.every == 0:
            self.f.flush()

    def close(self) -> None:
        self.f.flush()
        self.f.close()


def load_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # 죽는 순간 잘린 마지막 줄
                pass
    return out


def run_policy(policy: str, rows: list[dict], tgt: str, args,
               ckpt: "Checkpoint | None" = None) -> list[dict]:
    out: list[dict] = []
    t0 = time.time()

    def add(row: dict) -> None:
        out.append(row)
        if ckpt is not None:
            ckpt.write(row)

    if policy == "punct":
        for r in rows:
            add({**r, "pieces": punct.segment(r["text"])})

    elif policy == "causal_align":
        refs = load_refs(args.manifest_tag, tgt, args.dataset)
        aligner = causal_align.CausalAligner(device=args.device)
        skipped = 0
        for i, r in enumerate(rows):
            ref = refs.get(r["id"])
            if not ref:
                skipped += 1
                add({**r, "pieces": [r["text"]], "no_ref": True})
                continue
            add({**r, "pieces": aligner.segment(r["text"], ref, tgt)})
            if (i + 1) % 100 == 0:
                print(f"  [{tgt}] {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        if skipped:
            print(f"  [{tgt}] 참조 없어 무분절 처리 {skipped}건")

    elif policy == "mu_prefix":
        from core.meaning_segmentator.autoseg.baselines.nmt import Nmt

        nmt = Nmt(src="en", tgt=tgt, device=args.device)
        for i, r in enumerate(rows):
            pieces = mu_prefix.segment(nmt, r["text"], SPACED[tgt], args.n_cands)
            add({**r, "pieces": pieces})
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(rows) - i - 1)
                print(f"  [{tgt}] {i + 1}/{len(rows)}  {el:.0f}s  ETA {eta / 60:.1f}m",
                      flush=True)
    elif policy == "syntax":
        seg = syntax_sasst.SyntaxSegmenter(max_chunk=args.max_chunk)
        for r in rows:
            add({**r, "pieces": seg.segment(r["text"])})

    elif policy == "alignatt":
        from core.meaning_segmentator.autoseg.baselines.nmt import Nmt

        nmt = Nmt(src="en", tgt=tgt, device=args.device,
                  attentions=True, attn_layer=args.attn_layer)
        for i, r in enumerate(rows):
            add({**r, "pieces": alignatt.segment(nmt, r["text"], args.f)})
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
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "covost2"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-cands", type=int, default=10, help="Zhang 2020 beam 후보 수")
    p.add_argument("--f", type=int, default=2, help="AlignAtt 노브 — 최근 f 어절")
    p.add_argument("--attn-layer", type=int, default=5, help="AlignAtt 정렬 층")
    p.add_argument("--max-chunk", type=int, default=7, help="SASST 최대 청크 어절")
    p.add_argument("--limit", type=int, default=0, help="스모크용 앞 N 문장")
    p.add_argument("--resume", action="store_true",
                   help="`<out>.partial.jsonl` 을 읽어 **이미 끝난 문장은 건너뛴다.** "
                        "CUDA 오류로 죽은 뒤 이어 돌릴 때 쓴다")
    p.add_argument("--out-name", default=None,
                   help="산출 파일 stem (기본: 정책 이름). 같은 정책을 노브만 바꿔 "
                        "여러 벌 만들 때 쓴다 — 예: --policy alignatt --f 4 "
                        "--out-name alignatt_f4. `bleu_eval --baselines` 가 이 이름을 읽는다")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    rows = load_rows(run_dir, args.label, args.split,
                     args.dataset, args.manifest_tag, args.targets[0])
    if args.limit:
        rows = rows[: args.limit]
    out_dir = run_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # punct 는 타깃 독립이라 한 번만 낸다.
    targets = ["all"] if args.policy in ("punct", "syntax") else args.targets
    for tgt in targets:
        stem = args.out_name or args.policy
        path = out_dir / f"{stem}_{tgt}_{args.split}.json"
        part = path.with_suffix(".partial.jsonl")

        done = load_partial(part) if args.resume else []
        if not args.resume and part.exists():
            part.unlink()                      # 이어 돌리는 게 아니면 이전 잔해를 지운다
        done_ids = {r["id"] for r in done}
        todo = [r for r in rows if r["id"] not in done_ids]
        print(f"[{args.policy}] en→{tgt}, {len(rows)}문장"
              + (f" (이어서: 완료 {len(done)}, 남은 {len(todo)})" if done else ""))

        ckpt = Checkpoint(part)
        try:
            res = run_policy(args.policy, todo, tgt if tgt != "all" else "de", args, ckpt)
        finally:
            ckpt.close()

        # 원래 순서로 되돌린다 — 소비자(`bleu_eval`)는 id 로 찾지만 산출물은 읽는 사람도 본다.
        order = {r["id"]: i for i, r in enumerate(rows)}
        res = sorted(done + res, key=lambda r: order.get(r["id"], 1 << 30))
        path.write_text(json.dumps(
            {"policy": args.policy, "tgt": tgt, "split": args.split,
             "n_cands": args.n_cands if args.policy == "mu_prefix" else None,
             "f": args.f if args.policy == "alignatt" else None,
             "attn_layer": args.attn_layer if args.policy == "alignatt" else None,
             "rows": res}, ensure_ascii=False, indent=2), encoding="utf-8")
        part.unlink(missing_ok=True)       # 최종본이 나왔으니 진행분은 버린다
        print(f"  → {path} ({len(res)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
