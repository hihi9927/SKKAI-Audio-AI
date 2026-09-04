#!/usr/bin/env python3
"""AST 런의 COMET 채점 — 저장된 결과를 재사용한다(재번역·API 비용 0).

BLEU 는 표면 n-gram 이라 **언어 간 절대 비교가 불가능**하다(de 는 13a, ja 는 mecab,
zh 는 문자 단위로 토크나이저가 다르다). COMET 은 다국어 인코더 하나로 점수를 내므로
언어를 가로지르는 판독이 상대적으로 가능하다. 쌍별 편향은 남으므로 BLEU 도 같이 본다.

src 는 **정답 영어 전사**(`src_text`)를 쓴다. 축마다 ASR 출력이 다르므로(커밋이 잦을수록
전사가 나빠진다) ASR 텍스트를 src 로 쓰면 축마다 src 가 달라져 COMET 을 축 간에 비교할
수 없다. gold src 로 고정하면 "ASR 오류까지 포함한 최종 번역 품질"을 재게 된다.

GPU 를 vLLM 과 나눠 쓸 수 없으므로 **평가 서버를 내린 뒤** 실행할 것.

    .venv-autoseg/bin/python evaluation/ast/comet_ast.py --tag 20260825_134119
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
AXES = ["static", "punct", "seg"]
LANGS = ["de", "ja", "zh"]


def paired_bootstrap(a: list[float], b: list[float], iters: int = 1000,
                     seed: int = 12345) -> dict:
    """조건 a − b 의 평균 차이와 95% CI. 발화별 점수만 재표집하므로 비용이 없다."""
    n = len(a)
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    delta = statistics.mean(diffs)
    boots = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    return {
        "delta": round(delta, 4),
        "ci95": [round(boots[int(0.025 * iters)], 4),
                 round(boots[int(0.975 * iters) - 1], 4)],
        "n_boot": iters,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="AST 런 COMET 채점")
    p.add_argument("--results-root", default=str(HERE / "results"))
    p.add_argument("--dataset", default="CoVoST2")
    p.add_argument("--scope-prefix", default="n3000")
    p.add_argument("--tag", required=True)
    p.add_argument("--axes", nargs="+", default=None,
                   help="채점할 축 (기본 static punct seg). 청크 스윕은 static-c4 등")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", default=None, help="기본: results/<dataset>/comet_<tag>.json")
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve() / a.dataset
    global AXES
    if a.axes:
        AXES = list(a.axes)

    # ── 입력 수집 ────────────────────────────────────────────────────────
    data: dict[tuple[str, str], dict] = {}
    for ax in AXES:
        for lg in LANGS:
            f = root / ax / f"{a.scope_prefix}-{lg}" / a.tag / "metric.json"
            if not f.exists():
                print(f"[건너뜀] 없음: {f}")
                continue
            rows = json.load(f.open(encoding="utf-8"))["rows"]
            data[(ax, lg)] = {
                "utt_ids": [r["utt_id"] for r in rows],
                "src": [r["src_text"] for r in rows],
                "mt": [r["hyp_text"] for r in rows],
                "ref": [r["ref_text"] for r in rows],
            }
            print(f"  {ax:7s} {lg}  {len(rows)}발화")
    if not data:
        print("채점할 결과가 없습니다."); return 2

    from comet import download_model, load_from_checkpoint
    print(f"\nCOMET 모델 로드: {a.model}")
    model = load_from_checkpoint(download_model(a.model))

    # ── 채점 ────────────────────────────────────────────────────────────
    seg_scores: dict[tuple[str, str], list[float]] = {}
    system: dict[str, dict[str, float]] = {}
    for (ax, lg), d in data.items():
        triplets = [{"src": s, "mt": m, "ref": r}
                    for s, m, r in zip(d["src"], d["mt"], d["ref"])]
        out = model.predict(triplets, batch_size=a.batch_size, gpus=a.gpus,
                            progress_bar=True)
        seg_scores[(ax, lg)] = list(out.scores)
        system.setdefault(ax, {})[lg] = round(float(out.system_score), 4)
        print(f"[{ax}/{lg}] COMET {out.system_score:.4f}")

    # ── 축 간 짝지은 부트스트랩 ──────────────────────────────────────────
    # **순서로 짝지으면 안 된다.** 클라이언트가 16 병렬이라 발화 완료 순서가 축마다
    # 다르고, rows 는 완료 순으로 쌓인다. utt_id 로 맞춰야 같은 발화끼리 비교된다.
    by_id = {k: dict(zip(d["utt_ids"], seg_scores[k])) for k, d in data.items()}
    pairs = {}
    for lg in LANGS:
        import itertools
        for x, y in itertools.combinations(AXES, 2):
            if (x, lg) not in by_id or (y, lg) not in by_id:
                continue
            common = sorted(set(by_id[(x, lg)]) & set(by_id[(y, lg)]))
            if len(common) < len(by_id[(x, lg)]):
                print(f"  [주의] {lg} {x}/{y} 공통 발화 {len(common)} "
                      f"(< {len(by_id[(x, lg)])}) — 한쪽에만 있는 발화는 제외")
            pairs.setdefault(lg, {})[f"{x}-{y}"] = paired_bootstrap(
                [by_id[(x, lg)][i] for i in common],
                [by_id[(y, lg)][i] for i in common])
            pairs[lg][f"{x}-{y}"]["n"] = len(common)

    out_path = Path(a.out) if a.out else root / f"comet_{a.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "comet_model": a.model,
        "tag": a.tag,
        "system": system,
        "paired_bootstrap": pairs,
        # utt_id 와 함께 저장한다 — 순서만 저장하면 나중에 다시 짝지을 수 없다.
        "segment_scores": {f"{ax}/{lg}": by_id[(ax, lg)] for (ax, lg) in seg_scores},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== COMET (system) ===")
    print(f"{'축':8s} " + " ".join(f"{lg:>9s}" for lg in LANGS))
    for ax in AXES:
        if ax in system:
            print(f"{ax:8s} " + " ".join(f"{system[ax].get(lg, float('nan')):>9.4f}"
                                         for lg in LANGS))
    print(f"\n=== 짝지은 부트스트랩 (COMET 차이, 95% CI) ===")
    for lg in LANGS:
        for k, v in pairs.get(lg, {}).items():
            sig = "" if v["ci95"][0] <= 0 <= v["ci95"][1] else "  *"
            print(f"  {lg}  {k:16s} {v['delta']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{sig}")
    print(f"\n저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
