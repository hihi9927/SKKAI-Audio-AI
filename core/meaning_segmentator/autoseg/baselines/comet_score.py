"""조건별 COMET 점수 — 저장된 번역문을 재사용해 **재번역 없이** 채점한다.

`bleu_eval` 이 `bleu/<tgt>.json` 의 각 조건에 `hyps`(조각 번역 합본)를 이미 남긴다.
소스·참조와 함께 COMET 에 넣으면 같은 조건 집합에 대해 두 번째 품질축이 생긴다.

BLEU 는 토크나이저가 타깃마다 달라 언어 간 비교가 원천적으로 불가능한데, COMET 은
다국어 인코더 하나로 재므로 그 제약이 훨씬 약하다 — 그것이 이 축을 붙이는 이유다.
다만 **참조 기반**이라 어순을 단조화한 좋은 분절을 감점하는 편향은 BLEU 와 공유한다
(SEGMENTATION_CRITERIA_RELATED_WORK.md §3).

    python -m core.meaning_segmentator.autoseg.baselines.comet_score \\
        --run-id en-multi/clean500 --targets de ja
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ..paths import REPO_ROOT, RUNS_DIR
sys.path.insert(0, str(REPO_ROOT))

from core.meaning_segmentator.autoseg.runtime import metrics  # noqa: E402
from core.meaning_segmentator.autoseg.baselines import datasets as _ds  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="조건별 COMET (번역 재사용)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--targets", nargs="+", default=["de", "ja"])
    p.add_argument("--label", default="auto_best")
    p.add_argument("--split", default="test")
    p.add_argument("--manifest-tag", default="clean500")
    p.add_argument("--dataset", default="fleurs", choices=["fleurs", "covost2"])
    p.add_argument("--src", default="en",
                   help="소스 언어. CoVoST2 는 X→en 매니페스트도 있어서 이걸 안 주면 "
                        "`covost2_en-<tgt>_…` 만 찾는다 (`bleu_eval --src` 와 같은 값)")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--only-missing", action="store_true",
                   help="`comet` 값이 이미 있는 조건은 건너뛴다. 조건 몇 개만 추가로 "
                        "넣었을 때 전체 재채점(타깃당 ~6분)을 피한다")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    ev_path = run_dir / "prompt_eval" / f"{args.label}_{args.split}.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        ids = [r["id"] for r in ev["rows"]]
        srcs_by_id = {r["id"]: r["text"] for r in ev["rows"]}
    else:
        # 제안 라벨이 아직 없으면 매니페스트 순서를 따른다 — `bleu_eval` 과 같은 규칙이라
        # 조건별 `hyps` 의 인덱스가 어긋나지 않는다.
        ents = _ds.get(args.dataset, **({'src': args.src} if args.dataset == 'covost2' else {})).entries(args.manifest_tag, args.targets[0])
        ids = list(ents)
        srcs_by_id = {k: e.src for k, e in ents.items()}

    backend = metrics.CometBackend(model_name=args.model, batch_size=args.batch_size)
    out_dir = run_dir / "bleu"

    for tgt in args.targets:
        path = out_dir / f"{tgt}.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        refs_map = {k: e.ref for k, e
                    in _ds.get(args.dataset, **({'src': args.src} if args.dataset == 'covost2' else {})).entries(args.manifest_tag, tgt).items()}
        keep = [i for i in ids if i in refs_map]
        srcs = [srcs_by_id[i] for i in keep]
        refs = [refs_map[i] for i in keep]

        scored = 0
        t0 = time.time()
        for name, cell in blob["conditions"].items():
            if args.only_missing and cell.get("comet") is not None:
                continue
            hyps = cell.get("hyps")
            if not hyps or len(hyps) != len(refs):
                print(f"  [{tgt}] {name}: hyps {len(hyps or [])} != refs {len(refs)} — 건너뜀")
                continue
            seg = backend.score(srcs, hyps, refs)
            # **문장별 점수를 남긴다** — 평균만 두면 쌍체 신뢰구간을 못 낸다.
            cell["comet_seg"] = [round(float(x), 5) for x in seg]
            cell["comet"] = round(sum(seg) / len(seg), 4)
            scored += 1
            print(f"  [{tgt}] {name:20s} COMET {cell['comet']:.4f}  "
                  f"BLEU {cell['bleu']:6.2f}  {cell['laal_ms']:.0f}ms", flush=True)
        blob["comet_model"] = args.model
        path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{tgt}] {scored}조건 채점, {time.time() - t0:.0f}s → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
