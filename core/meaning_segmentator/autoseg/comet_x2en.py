"""{de,zh,ja}→en 런의 품질–지연 곡선 — y축 참조 기반 COMET, x축 실측 LAAL(ms).

`comet_eval.py` 는 `bleu_eval` 산출(clean500, en→X)을 읽지만 이쪽은 **루프 런의
`test_rows.json`** 을 읽는다. 재번역은 없다 — 조각 번역(`pieces_tgt`)이 이미 있다.

en→X 판(`comet_tradeoff.png`)보다 비교가 깨끗하다: **세 트랙의 타깃이 모두 영어**라
COMET 이 같은 방향의 번역을 재고, 문장도 같다(§1 의 en 기준 정렬 재사용).

    python -m core.meaning_segmentator.autoseg.comet_x2en \
        --runs zh=zh-en/smoke04 ja=ja-en/smoke01 de=de-en/smoke01
"""
from __future__ import annotations

import argparse, json, statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

FLEURS_DIR = {"de": "de_de", "ja": "ja_jp", "zh": "cmn_hans_cn"}
SPACED_SRC = {"de": True, "ja": False, "zh": False}   # measured_profile 와 일치


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="lang=경로 (runs/ 이하)")
    p.add_argument("--tag", default="multi2en_loop240")
    p.add_argument("--model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="core/meaning_segmentator/runs/x2en_comet.json")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(_REPO))
    from core.meaning_segmentator.autoseg.bleu_eval import load_durations, laal_ms

    out_path = Path(args.out)
    if args.report_only and out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        plot(report, out_path)
        return 0

    specs = [s.split("=", 1) for s in args.runs]
    rows_all, index, report = [], [], {}

    for lang, rel in specs:
        run_dir = _HERE.parent / "runs" / rel
        rows = json.loads((run_dir / "test_rows.json").read_text(encoding="utf-8"))
        man = {}
        mp = (_REPO / "evaluation" / "ast" / "manifests"
              / f"fleurs_nway_{lang}-en_{args.tag}.jsonl")
        for line in mp.open(encoding="utf-8"):
            e = json.loads(line)
            man[e["utt_id"]] = e["tgt_text"]
        durs = load_durations(FLEURS_DIR[lang])
        spaced = SPACED_SRC[lang]
        conds: dict[str, dict] = {}

        for r in rows:
            ref = man.get(r["id"])
            if not ref:
                continue
            talk = str(r["id"]).split("_")[-1]
            d = durs.get(talk)
            # 무분절 기준선 — 지연은 발화 전체를 기다린 값이다.
            if r.get("full_trans") and d:
                c = conds.setdefault("unsegmented", {"laal": [], "pairs": []})
                c["pairs"].append((r["text"], r["full_trans"], ref)); c["laal"].append(d)
            for T, cell in (r.get("by_T") or {}).items():
                pt = cell.get("pieces_tgt") or []
                ps = cell.get("pieces_src") or []
                if not pt:
                    continue
                hyp = " ".join(x for x in pt if x)
                c = conds.setdefault(f"T{T}", {"laal": [], "pairs": []})
                c["pairs"].append((r["text"], hyp, ref))
                v = laal_ms(ps, pt, d or 0.0, ref, spaced, "word") if d else None
                if v is not None:
                    c["laal"].append(v)

        report[lang] = {}
        for name, c in conds.items():
            report[lang][name] = {
                "n": len(c["pairs"]),
                "laal_ms": round(statistics.mean(c["laal"]), 1) if c["laal"] else None,
                "n_laal": len(c["laal"]),
            }
            for src, mt, ref in c["pairs"]:
                index.append((lang, name))
                rows_all.append({"src": src, "mt": mt, "ref": ref})
        print(f"[{lang}] {rel}: 조건 {len(conds)}개, 채점 {sum(len(c['pairs']) for c in conds.values())}건")

    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(args.model))
    scores = list(model.predict(rows_all, batch_size=args.batch_size, gpus=1,
                                progress_bar=True).scores)
    agg: dict[tuple, list[float]] = {}
    for (lang, name), s in zip(index, scores):
        agg.setdefault((lang, name), []).append(s)
    for (lang, name), v in agg.items():
        report[lang][name]["comet"] = round(statistics.mean(v), 4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "by_lang": report},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out_path}")
    plot({"model": args.model, "by_lang": report}, out_path)
    return 0


def plot(blob: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e1"
    COLOR = {"de": "#2a78d6", "ja": "#eb6834", "zh": "#1baf7a"}
    MARK = {"de": "o", "ja": "s", "zh": "^"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for lang, conds in blob["by_lang"].items():
        pts = sorted((c["laal_ms"], c["comet"], name)
                     for name, c in conds.items()
                     if name != "unsegmented" and c.get("laal_ms") and c.get("comet"))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                    marker=MARK[lang], color=COLOR[lang], lw=2, ms=8,
                    mec=SURFACE, mew=2, zorder=5, label=f"{lang}→en")
            for x, y, name in pts:
                ax.annotate(name, xy=(x, y), xytext=(0, -13),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=INK2, zorder=6)
        u = conds.get("unsegmented")
        if u and u.get("comet"):
            ax.axhline(u["comet"], color=COLOR[lang], lw=1, ls=(0, (4, 3)),
                       alpha=0.55, zorder=1)
            ax.text(0.995, u["comet"], f" {lang} full-sentence {u['comet']:.3f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    color=COLOR[lang], fontsize=7.5)

    ax.set_xlabel("LAAL (ms of source audio)  ←  lower latency is better")
    ax.set_ylabel(f"COMET  ({blob['model'].split('/')[-1]}, gold en reference)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.text(0.01, 0.975, "Quality–latency on {de, zh, ja} → en",
             ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.928,
             "Same sentences, same target language (English) — so COMET is directly "
             "comparable across the three curves.\nDashed line = that language's "
             "full-sentence ceiling. Point labels are the T knob.",
             ha="left", va="top", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print("saved", out_path.with_suffix(".png"))


if __name__ == "__main__":
    raise SystemExit(main())
