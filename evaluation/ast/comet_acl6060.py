#!/usr/bin/env python3
"""ACL 60/60 장문 COMET 채점 — **재분절된 문장 단위**로 낸다.

    .venv-autoseg/bin/python evaluation/ast/comet_acl6060.py \\
        --split dev --tags 20260828_080447 20260828_144627 20260828_152354

왜 `metric.json` 을 그냥 못 쓰나
--------------------------------
단문(CoVoST2)은 발화 하나 = 문장 하나라 `comet_ast.py` 가 행을 그대로 먹였다. 장문은
행 하나가 **12분짜리 발표 전체**다. COMET 인코더 입력 길이를 한참 넘고, 넘지 않더라도
"발표 한 편"에 점수 하나는 해석이 안 된다.

그래서 `score_acl6060.py` 가 남긴 `reseg_{split}_{tag}.json` 을 쓴다. mwerSegmenter 가
시스템 출력을 참조 문장 경계로 다시 자른 결과라, **참조 문장 468개와 1:1로 맞는** 가설이
들어 있다. 축이 달라도 참조 문장 집합은 같으므로 `(utt_id, seg_id)` 로 짝지으면 축 간
짝지은 부트스트랩이 성립한다.

src 는 **정답 영어 전사**를 쓴다(`comet_ast.py` 와 같은 규칙). 축마다 ASR 출력이 달라
ASR 텍스트를 src 로 쓰면 축 간 비교가 깨진다.

빈 가설(재분절이 아무것도 배정하지 못한 문장)도 **버리지 않는다.** 어려운 문장을
빠뜨려 점수를 올리는 길을 막는다. 빈 문자열로 채점되어 낮은 점수를 받는다.

여러 태그를 함께 받는다 — 청크 스윕(static-c4/c6)이 별도 런이라 태그가 다르기 때문이다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comet_ast import paired_bootstrap  # noqa: E402

LANGS = ["de", "ja", "zh"]
ORDER = ["static", "static-c4", "seg", "static-c6", "punct"]   # 지연 오름차순


def main() -> int:
    p = argparse.ArgumentParser(description="ACL 60/60 장문 COMET 채점")
    p.add_argument("--results-root", default=str(HERE / "results" / "ACL6060"))
    p.add_argument("--split", default="dev")
    p.add_argument("--tags", nargs="+", required=True,
                   help="reseg_{split}_{tag}.json 을 만든 태그들. 청크 스윕은 태그가 다르다")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()

    # ── 수집: {(축, 언어): {문장키: (src, mt, ref)}} ──────────────────────
    data: dict[tuple[str, str], dict] = {}
    for tag in a.tags:
        f = root / f"reseg_{a.split}_{tag}.json"
        if not f.exists():
            print(f"[건너뜀] 없음: {f}")
            continue
        blob = json.loads(f.read_text(encoding="utf-8"))
        for key, pairs in blob.items():
            ax, lg = key.split("/")
            if (ax, lg) in data:
                print(f"  [주의] {key} 가 여러 태그에 있다 — 먼저 온 것을 쓴다")
                continue
            data[(ax, lg)] = {f"{r['utt_id']}#{r['seg_id']}": r for r in pairs}
            print(f"  {ax:11s} {lg}  문장 {len(pairs)}개  (tag {tag})")
    if not data:
        print("채점할 결과가 없습니다."); return 2

    from comet import download_model, load_from_checkpoint
    print(f"\nCOMET 모델 로드: {a.model}")
    model = load_from_checkpoint(download_model(a.model))

    seg_scores: dict[tuple[str, str], dict[str, float]] = {}
    system: dict[str, dict[str, float]] = {}
    n_empty: dict[str, dict[str, int]] = {}
    for (ax, lg), d in sorted(data.items()):
        keys = sorted(d)
        trip = [{"src": d[k]["src"], "mt": d[k]["hyp"], "ref": d[k]["ref"]} for k in keys]
        out = model.predict(trip, batch_size=a.batch_size, gpus=a.gpus, progress_bar=True)
        seg_scores[(ax, lg)] = dict(zip(keys, (float(x) for x in out.scores)))
        system.setdefault(ax, {})[lg] = round(float(out.system_score), 4)
        n_empty.setdefault(ax, {})[lg] = sum(1 for k in keys if not d[k]["hyp"].strip())
        print(f"[{ax}/{lg}] COMET {out.system_score:.4f}  (빈 가설 {n_empty[ax][lg]})")

    # ── 축 간 짝지은 부트스트랩 — 문장 키로 맞춘다 ───────────────────────
    axes = [x for x in ORDER if any(k[0] == x for k in data)]
    axes += [x for x in sorted({k[0] for k in data}) if x not in axes]
    pairs_out: dict[str, dict] = {}
    for lg in LANGS:
        for x, y in itertools.combinations(axes, 2):
            if (x, lg) not in seg_scores or (y, lg) not in seg_scores:
                continue
            common = sorted(set(seg_scores[(x, lg)]) & set(seg_scores[(y, lg)]))
            if not common:
                continue
            r = paired_bootstrap([seg_scores[(x, lg)][k] for k in common],
                                 [seg_scores[(y, lg)][k] for k in common])
            r["n"] = len(common)
            pairs_out.setdefault(lg, {})[f"{x}-{y}"] = r

    out_path = Path(a.out) if a.out else root / f"comet_{a.split}_{'_'.join(a.tags)}.json"
    out_path.write_text(json.dumps({
        "comet_model": a.model, "split": a.split, "tags": a.tags,
        "system": system, "n_empty_hyp": n_empty, "paired_bootstrap": pairs_out,
        "segment_scores": {f"{ax}/{lg}": s for (ax, lg), s in seg_scores.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== COMET (system) ===")
    print(f"{'축':11s} " + " ".join(f"{lg:>9s}" for lg in LANGS))
    for ax in axes:
        print(f"{ax:11s} " + " ".join(f"{system.get(ax, {}).get(lg, float('nan')):>9.4f}"
                                      for lg in LANGS))
    print("\n=== 짝지은 부트스트랩 (COMET 차이, 95% CI, * = 유의) ===")
    for lg in LANGS:
        for k, v in pairs_out.get(lg, {}).items():
            sig = "" if v["ci95"][0] <= 0 <= v["ci95"][1] else "  *"
            print(f"  {lg}  {k:22s} {v['delta']:+.4f}  "
                  f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  n={v['n']}{sig}")
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
