#!/usr/bin/env python3
"""커밋된 ASR 전사의 WER — 축·언어별.

왜 따로 재나
------------
`score_acl6060.py` 의 `diagnostics.as_wer` 는 **재분절(mwerSegmenter)이 달성한 mWER**
이지 ASR 품질이 아니다. 절대값으로 읽으면 안 된다고 그 스크립트도 적어놨다.
여기서 재는 것은 `metric.json` 의 `asr_text`(커밋된 원문을 이어붙인 것) 대 `src_text`
(참조 전사)의 WER 로, **커밋 정책이 전사 품질에 얼마나 손해를 끼치는가**를 본다.

정규화는 `plot_covost2.py` 와 동일하다 — `<SEG>` 제거, 구두점을 공백으로, 소문자화.
축 간 비교가 목적이라 정규화가 같기만 하면 되고, 절대 WER 을 다른 논문과 비교하지 않는다.

GPU 를 쓰지 않는다(jiwer 는 편집거리만 계산). 다른 런이 GPU 를 쓰는 중에도 안전하다.

사용:
    python evaluation/ast/asr_wer.py \
        --results-root evaluation/ast/results/ACL6060 --split dev \
        --tags 20260830_180201 20260830_172429 20260831_102723
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def clean(t: str) -> str:
    """plot_covost2.py 와 같은 정규화."""
    return re.sub(r"\s+", " ",
                  re.sub(r"[^\w\s']", " ", (t or "").replace("<SEG>", " "))).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="커밋된 ASR 전사의 WER")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--split", default="dev", help="ACL 은 dev/eval, CoVoST2 는 n3000")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="축마다 태그가 다르다. 여러 개 주면 전부 훑는다")
    ap.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import jiwer

    root = Path(a.results_root).expanduser().resolve()
    rows: list[dict] = []
    for axis_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for lang in a.langs:
            run = axis_dir / f"{a.split}-{lang}"
            if not run.is_dir():
                continue
            for tag in a.tags:
                mj = run / tag / "metric.json"
                if not mj.exists():
                    continue
                data = json.loads(mj.read_text())["rows"]
                # 참조가 빈 행은 WER 이 정의되지 않는다 — 버리고 개수를 남긴다
                pairs = [(clean(r.get("asr_text")), clean(r.get("src_text"))) for r in data]
                dropped = sum(1 for _, ref in pairs if not ref)
                pairs = [(h, f) for h, f in pairs if f]
                if not pairs:
                    continue
                wer = jiwer.wer([f for _, f in pairs], [h for h, _ in pairs]) * 100
                rows.append({
                    "axis": axis_dir.name, "lang": lang, "tag": tag,
                    "wer": round(wer, 3), "n_utts": len(pairs), "n_dropped": dropped,
                })

    if not rows:
        raise SystemExit("해당하는 metric.json 을 못 찾았다 — --split / --tags 를 확인할 것")

    width = max(len(r["axis"]) for r in rows) + 2
    print(f"{'축':<{width}}{'lang':<6}{'WER%':>8}{'발화':>7}  태그")
    for r in sorted(rows, key=lambda r: (r["axis"], r["lang"])):
        drop = f"  (참조없음 {r['n_dropped']})" if r["n_dropped"] else ""
        print(f"{r['axis']:<{width}}{r['lang']:<6}{r['wer']:>8.2f}{r['n_utts']:>7}  {r['tag']}{drop}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
