#!/usr/bin/env python3
"""CoVoST2 단클립 manifest 3종의 정합성 검증.

세 언어가 **같은 영어 오디오**를 쓴다는 전제 위에 실험을 세웠으므로, 그 전제가
실제로 성립하는지 매번 확인한다. 어긋난 채로 돌리면 언어 간 비교가 조용히 깨진다.

    python evaluation/ast/check_manifest_covost2.py \
        --manifests evaluation/ast/manifests/covost2_en-{de,ja,zh}_n3000.jsonl \
        --subset evaluation/ast/subsets/covost2_en_test_n3000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                out[e["utt_id"]] = e
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="CoVoST2 단클립 manifest 검증")
    p.add_argument("--manifests", nargs="+", required=True)
    p.add_argument("--subset", default=None, help="select_covost2_subset.py 산출물")
    a = p.parse_args()

    mans = {Path(m).name: load(Path(m)) for m in a.manifests}
    fail = []

    print("=== 발화 수 ===")
    for name, m in mans.items():
        print(f"  {name:34s} {len(m):,}")
    sizes = {len(m) for m in mans.values()}
    if len(sizes) != 1:
        fail.append("manifest 간 발화 수가 다릅니다")

    # ── id 집합이 동일한가 ───────────────────────────────────────────────
    key_sets = [set(m) for m in mans.values()]
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    print(f"\n=== id 집합 ===\n  공통 {len(common):,} / 합집합 {len(union):,}")
    if common != union:
        fail.append(f"id 집합 불일치 — 일부에만 있는 id {len(union - common)}개")

    if a.subset:
        want = set(json.loads(Path(a.subset).read_text(encoding="utf-8"))["utt_ids"])
        miss = want - common
        extra = common - want
        print(f"  subset 대비 누락 {len(miss)} / 초과 {len(extra)}")
        if miss:
            fail.append(f"subset 의 id {len(miss)}개가 manifest 에 없습니다 "
                        f"(예: {sorted(miss)[:3]})")
        if extra:
            fail.append(f"subset 에 없는 id {len(extra)}개가 manifest 에 있습니다")

    # ── 같은 wav / 같은 길이 / 같은 원문을 쓰는가 ─────────────────────────
    names = list(mans)
    base = mans[names[0]]
    bad_wav = bad_dur = bad_src = 0
    for uid in sorted(common):
        for n in names[1:]:
            o = mans[n][uid]
            if o["wav"] != base[uid]["wav"]:
                bad_wav += 1
            if abs(o["duration"] - base[uid]["duration"]) > 1e-6:
                bad_dur += 1
            if o["src_text"] != base[uid]["src_text"]:
                bad_src += 1
    print(f"\n=== 언어 간 일치 (기준: {names[0]}) ===")
    print(f"  wav 경로 불일치 {bad_wav} / 길이 불일치 {bad_dur} / 원문 불일치 {bad_src}")
    for label, cnt in (("wav 경로", bad_wav), ("길이", bad_dur), ("원문", bad_src)):
        if cnt:
            fail.append(f"언어 간 {label}이 {cnt}건 다릅니다")

    # ── 타깃 번역은 서로 달라야 정상 ─────────────────────────────────────
    same_tgt = sum(
        1 for uid in sorted(common)
        if len({mans[n][uid]["tgt_text"] for n in names}) < len(names)
    )
    print(f"  타깃 번역이 겹치는 발화 {same_tgt} (0에 가까워야 정상)")

    # ── 파일 실재 & 빈 텍스트 ────────────────────────────────────────────
    missing_wav = sum(1 for uid in sorted(common) if not Path(base[uid]["wav"]).exists())
    empty = {n: sum(1 for uid in common if not mans[n][uid]["tgt_text"].strip())
             for n in names}
    print(f"\n=== 파일/텍스트 ===\n  wav 없음 {missing_wav}")
    print(f"  빈 번역 {empty}")
    if missing_wav:
        fail.append(f"wav 파일 {missing_wav}개가 실제로 없습니다")
    if any(empty.values()):
        fail.append(f"빈 번역이 있습니다: {empty}")

    # ── 길이 분포 ────────────────────────────────────────────────────────
    d = np.array([base[uid]["duration"] for uid in sorted(common)])
    print(f"\n=== 길이 분포 ===")
    print(f"  총 {d.sum()/3600:.2f}시간 / 평균 {d.mean():.2f}s / 중앙 {np.median(d):.2f}s "
          f"/ p10 {np.percentile(d,10):.2f} / p90 {np.percentile(d,90):.2f} / 최대 {d.max():.2f}")
    for n_cli in (8, 16):
        print(f"  실시간 페이싱 예상({n_cli}병렬): {(d.sum()+len(d))/60/n_cli:.0f}분/런")

    print()
    if fail:
        print("실패:")
        for f in fail:
            print(f"  ✗ {f}")
        return 1
    print("✓ 모든 검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
