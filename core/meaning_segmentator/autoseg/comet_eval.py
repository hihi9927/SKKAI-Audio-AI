"""분절 조건별 COMET — `bleu_eval` 이 남긴 번역을 재사용한다 (재번역·API 비용 없음).

BLEU 는 표면 n-gram 이라 **언어 간 절대 비교가 불가능**하고(토크나이저가 다르다), 그래서
`report.md` 는 retention 으로 우회한다. COMET 은 다국어 인코더 하나로 0~1 점수를 내므로
언어 쌍을 가로지르는 판독이 상대적으로 가능하다 — "프롬프트 하나가 세 언어에서 유지되는가"
라는 질문에 더 직접적으로 답한다. 쌍별 편향은 남으므로 retention 도 같이 낸다.

입력은 `runs/<run-id>/bleu/<tgt>.json` 의 조건별 `hyps` (문장 순서는 prompt_eval 의 rows 순서,
참조 없는 문장은 제외된 상태) 와 FLEURS n-way 매니페스트의 gold 참조·소스다.

    python -m core.meaning_segmentator.autoseg.comet_eval \
        --run-id en-multi/clean500 --targets de ja zh
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]

TGT_ORDER = ["de", "ja", "zh"]


def load_manifest(tag: str, tgt: str) -> dict[str, tuple[str, str]]:
    """utt_id → (소스 원문, 참조 번역). bleu_eval.load_manifest 와 같은 파일을 읽는다."""
    path = (_REPO_ROOT / "evaluation" / "ast" / "manifests"
            / f"fleurs_nway_en-{tgt}_{tag}.jsonl")
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            out[e["utt_id"]] = (e["src_text"], e["tgt_text"])
    return out


def paired_bootstrap(a: list[float], b: list[float], iters: int,
                     seed: int = 12345) -> dict:
    """조건 a − 조건 b 의 평균 차이와 95% CI. 세그먼트 점수만 재표집하므로 비용이 없다."""
    n = len(a)
    rng = random.Random(seed)
    delta = statistics.mean(a) - statistics.mean(b)
    diffs = [x - y for x, y in zip(a, b)]
    boots = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * iters)]
    hi = boots[int(0.975 * iters) - 1]
    return {"delta": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_boot": iters}


def main() -> int:
    p = argparse.ArgumentParser(description="분절 조건별 COMET (번역 재사용)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--label", default="auto_best")
    p.add_argument("--split", default="test")
    p.add_argument("--targets", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--manifest-tag", default="clean500")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--pair-tol-ms", type=float, default=80.0,
                   help="실측 지연이 이만큼 이내로 겹치는 쌍만 맞대결로 인정한다")
    p.add_argument("--report-only", action="store_true",
                   help="이미 채점된 json 으로 표만 다시 만든다 (GPU 불필요)")
    args = p.parse_args()

    run_dir = _HERE.parent / "runs" / args.run_id
    bleu_dir = run_dir / "bleu"
    ev = json.loads((run_dir / "prompt_eval" / f"{args.label}_{args.split}.json"
                     ).read_text(encoding="utf-8"))
    ids_all = [r["id"] for r in ev["rows"]]

    model = None
    if not args.report_only:
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(args.model))
        print(f"모델 {args.model}")

    report: dict[str, dict] = {}
    seg_all: dict[str, dict[str, list[float]]] = {}
    for tgt in args.targets:
        man = load_manifest(args.manifest_tag, tgt)
        # bleu_eval 과 **같은 필터·같은 순서**여야 hyps 와 정렬이 맞는다.
        ids = [i for i in ids_all if i in man]
        srcs = [man[i][0] for i in ids]
        refs = [man[i][1] for i in ids]

        data = json.loads((bleu_dir / f"{tgt}.json").read_text(encoding="utf-8"))
        conds = data["conditions"]
        assert data["n"] == len(ids), f"{tgt}: n {data['n']} != {len(ids)}"

        if args.report_only:
            report[tgt] = data
            seg_all[tgt] = json.loads(
                (bleu_dir / f"comet_seg_{tgt}.json").read_text(encoding="utf-8"))
            continue

        # 같은 (문장, 번역) 쌍은 조건이 달라도 한 번만 채점한다.
        uniq: dict[tuple[int, str], int] = {}
        rows: list[dict] = []
        for name, cell in conds.items():
            hyps = cell["hyps"]
            assert len(hyps) == len(ids), f"{tgt}/{name}: hyps {len(hyps)}"
            for j, h in enumerate(hyps):
                key = (j, h)
                if key not in uniq:
                    uniq[key] = len(rows)
                    rows.append({"src": srcs[j], "mt": h, "ref": refs[j]})
        total = sum(len(c["hyps"]) for c in conds.values())
        print(f"[{tgt}] 조건 {len(conds)}개 × {len(ids)}문장 = {total}세그먼트 "
              f"→ 중복 제거 후 {len(rows)}건 채점")

        out = model.predict(rows, batch_size=args.batch_size, gpus=args.gpus,
                            progress_bar=True)
        scores = list(out.scores)

        seg: dict[str, list[float]] = {}
        for name, cell in conds.items():
            seg[name] = [scores[uniq[(j, h)]] for j, h in enumerate(cell["hyps"])]
            cell["comet"] = round(statistics.mean(seg[name]), 4)

        base = conds.get("unsegmented")
        for name, cell in conds.items():
            if base and base.get("comet"):
                cell["retention_comet"] = round(cell["comet"] / base["comet"], 4)
            if name != "unsegmented" and base:
                cell["paired_comet_vs_unseg"] = paired_bootstrap(
                    seg[name], seg["unsegmented"], args.bootstrap)
            print(f"[{tgt}] {name:18s} COMET {cell['comet']:.4f}  "
                  f"ret {cell.get('retention_comet', float('nan')):.4f}")

        data["comet_model"] = args.model
        (bleu_dir / f"{tgt}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        (bleu_dir / f"comet_seg_{tgt}.json").write_text(
            json.dumps({k: [round(x, 5) for x in v] for k, v in seg.items()},
                       ensure_ascii=False), encoding="utf-8")
        report[tgt] = data
        seg_all[tgt] = seg

    write_report(bleu_dir / "report_comet.md", report, args, seg_all)
    print(f"\n{bleu_dir / 'report_comet.md'}")
    return 0


def matched_pairs(conds: dict, tol_ms: float) -> list[tuple[str, str, float]]:
    """실측 laal_ms 가 tol 이내로 겹치는 (제안, 상대) 쌍. 같은 T 는 같은 지연이 아니므로
    T 를 맞추는 대신 **지연을 맞춘다** — BASELINE_COMPARISON.md §2 의 규칙과 같다."""
    ours = [n for n in conds if n.startswith(("auto_T", "auto_greedy_T"))]
    out = []
    for a in ours:
        ma = conds[a].get("laal_ms")
        if ma is None:
            continue
        for b, cb in conds.items():
            if b == a or b == "unsegmented":
                continue
            mb = cb.get("laal_ms")
            if mb is None or abs(ma - mb) > tol_ms:
                continue
            if (b, a, -0.0) in [(x[0], x[1], 0.0) for x in out]:
                continue
            if any(x[0] == b and x[1] == a for x in out):
                continue
            out.append((a, b, ma - mb))
    return out


def write_report(path: Path, report: dict[str, dict], args,
                 seg_all: dict[str, dict[str, list[float]]]) -> None:
    L = [f"# 분절 조건별 COMET — {args.run_id}", "",
         f"- 지표: `{args.model}` (참조 기반, gold FLEURS n-way 참조)",
         f"- 번역은 `bleu_eval` 산출을 **그대로 재사용** — 재번역·API 비용 없음",
         "- BLEU 는 언어 간 절대 비교가 불가하지만 COMET 은 다국어 인코더 하나를 쓰므로",
         "  절대값 비교가 상대적으로 가능하다. 다만 언어쌍별 편향이 남으므로 retention 도 같이 본다.",
         "- 조각 번역을 이어붙인 문자열을 gold 문장과 맞대므로, COMET 이 학습한 단일 문장과는",
         "  성격이 다르다 (문장 정렬 1:1 은 보장됨). 논문에 실으면 각주 필요.", ""]
    for tgt, data in report.items():
        L += [f"## en→{tgt} (n={data['n']})", "",
              "| 조건 | k | laal_ms ↓ | BLEU ↑ | chrF2 | COMET ↑ | ret(BLEU) | ret(COMET) | ΔCOMET vs unseg [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, c in data["conditions"].items():
            ms = "—" if c.get("laal_ms") is None else f"{c['laal_ms']:.0f}"
            pb = c.get("paired_comet_vs_unseg")
            d = ("—" if pb is None else
                 f"{pb['delta']:+.4f} [{pb['ci95'][0]:+.4f}, {pb['ci95'][1]:+.4f}]")
            L.append(f"| {name} | {c['k']:.2f} | {ms} | {c['bleu']:.2f} | "
                     f"{c['chrf2']:.2f} | {c['comet']:.4f} | "
                     f"{c.get('retention_bleu', 0):.4f} | "
                     f"{c.get('retention_comet', 0):.4f} | {d} |")
        L.append("")

    names = [n for n in report[list(report)[0]]["conditions"]]
    tgts = list(report)
    L += ["## 언어 간 안정성", "",
          "| 조건 | " + " | ".join(f"{t} COMET" for t in tgts) + " | "
          + " | ".join(f"{t} ret" for t in tgts) + " | COMET 폭 | ret 폭 |",
          "|---" * (1 + 2 * len(tgts) + 2) + "|"]
    for name in names:
        cells, rets = [], []
        for t in tgts:
            c = report[t]["conditions"].get(name)
            cells.append(None if c is None else c["comet"])
            rets.append(None if c is None else c.get("retention_comet"))
        got = [x for x in cells if x is not None]
        gotr = [x for x in rets if x is not None]
        if not got:
            continue
        f = lambda v: "—" if v is None else f"{v:.4f}"
        # 언어가 하나뿐인 조건에 "폭 0" 을 찍으면 가장 안정적인 것처럼 보인다 — 비운다.
        w = f"{max(got) - min(got):.4f}" if len(got) > 1 else "—"
        wr = f"{max(gotr) - min(gotr):.4f}" if len(gotr) > 1 else "—"
        L.append(f"| {name} | " + " | ".join(f(x) for x in cells) + " | "
                 + " | ".join(f(x) for x in rets) + " | " + w + " | " + wr + " |")
    auto = [n for n in names if n.startswith("auto_T")]
    if auto and len(tgts) > 1:
        L += ["", "## 지표별 retention 폭 (max−min) — 낮을수록 언어 간 안정",
              "", "| 지표 | " + " | ".join(auto) + " |", "|---" * (1 + len(auto)) + "|"]
        for key, lab in (("retention_bleu", "BLEU"), ("retention_chrf2", "chrF2"),
                         ("retention_comet", "COMET")):
            vals = []
            for n in auto:
                xs = [report[t]["conditions"][n].get(key) for t in tgts
                      if n in report[t]["conditions"]]
                xs = [x for x in xs if x is not None]
                vals.append(f"{max(xs) - min(xs):.4f}" if len(xs) > 1 else "—")
            L.append(f"| {lab} | " + " | ".join(vals) + " |")
    L += ["", f"## 지연이 {args.pair_tol_ms:.0f}ms 이내로 겹치는 맞대결 (COMET)", "",
          "같은 `T` 는 같은 지연이 **아니다**. 등간격 선택은 정의상 지연을 덜 쓰므로 T 를 맞춘",
          "비교는 지연 차이를 품질 차이로 오독한다. 실측 `laal_ms` 가 겹치는 쌍만 싣는다.", "",
          "| 타깃 | 대조 | 지연차(ms) | ΔCOMET [95% CI] | |",
          "|---|---|---|---|---|"]
    for t in tgts:
        segs = seg_all.get(t)
        if not segs:
            continue
        for a, b, dms in matched_pairs(report[t]["conditions"], args.pair_tol_ms):
            r = paired_bootstrap(segs[a], segs[b], args.bootstrap)
            lo, hi = r["ci95"]
            mark = "n.s." if lo < 0 < hi else "**"
            L.append(f"| {t} | {a} vs {b} | {dms:+.0f} | "
                     f"{r['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {mark} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
