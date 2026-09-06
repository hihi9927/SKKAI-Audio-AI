"""런 하나가 실제로 쓴 LLM 비용을 모아 본다 — 예산 감시용.

게이트웨이 usage 스냅샷은 산출물마다 흩어져 있다(`prompt_eval/*.json`, `history.json`,
`iter_*/metrics.json`). 크래시한 실행은 스냅샷을 못 남기므로 **분절 캐시 증분으로 역산**한
추정치도 함께 낸다 — 그 차이가 곧 "기록되지 않은 지출"이다.

    python -m core.meaning_segmentator.autoseg.infra.cost_report --run-id en-multi/clean500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import RUNS_DIR


def usage_snapshots(run_dir: Path) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in ("config.json", "language_profile.json", "measured_profile.json"):
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(blob, dict) and isinstance(blob.get("usage"), dict):
            out.append((str(path.relative_to(run_dir)), blob["usage"]))
        elif isinstance(blob, list):
            for i, item in enumerate(blob):
                if isinstance(item, dict) and isinstance(item.get("usage"), dict):
                    out.append((f"{path.relative_to(run_dir)}[{i}]", item["usage"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="런의 LLM 비용 집계")
    p.add_argument("--run-id", required=True)
    p.add_argument("--budget", type=float, default=None, help="넘으면 비영 종료코드")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run_id
    snaps = usage_snapshots(run_dir)
    total = 0.0
    print(f"== {args.run_id}")
    for name, u in snaps:
        total += float(u.get("cost", 0.0))
        print(f"  {name:44s} 호출 {u.get('calls', 0):5d}  ${float(u.get('cost', 0.0)):8.4f}")
        for k, v in sorted(u.get("by_purpose", {}).items(), key=lambda x: -x[1]["cost"]):
            print(f"      {k:18s} {v['calls']:5d}콜  ${v['cost']:7.4f}")
    print(f"  {'기록된 합계':44s} {'':10s} ${total:8.4f}")

    seg = run_dir / "cache" / "segment.json"
    if seg.exists():
        n = len(json.loads(seg.read_text(encoding="utf-8")))
        rate = (total / n) if n else 0.0
        print(f"\n  분절 캐시 {n}건 → 기록 기준 문장당 ${rate:.5f}")
        print("  **캐시 건수가 기록된 분절 수보다 많으면 그 차이는 스냅샷을 못 남긴 "
              "(크래시한) 실행이 쓴 돈이다.**")
    if args.budget is not None and total > args.budget:
        print(f"\n[초과] ${total:.4f} > 예산 ${args.budget:.4f}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
