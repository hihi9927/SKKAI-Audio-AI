"""저장된 런을 **재생**해 `rank_lift` 를 되살리고 `focus` 가 무엇이 되었을지 다시 뽑는다.

왜 필요한가: `loop.evaluate_multi` 가 `Metrics` 를 재구성하면서 `rank_lift*` 5개 필드를
버려서, 지금까지의 모든 런에서 `focus = "priority"` 가 **구조적으로 도달 불가능**했다.
고치기 전에 "고치면 실제로 priority 가 뽑히는가"를 먼저 재는 것이 이 스크립트다.

**API 비용 0.** 분절은 다시 하지 않고 저장된 태그의 번호만 섞는다. 셔플 대조군의 번역은
버그가 있던 런에서도 **계산은 됐다가 버려진 것**이라 런 캐시에 이미 들어 있다.
`--offline` 이면 캐시 미스에서 즉시 죽으므로 네트워크를 안 탔음을 증명할 수 있다.

    python -m core.meaning_segmentator.autoseg.replay_focus \
        --run core/meaning_segmentator/runs/de-en/run01 --offline
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import agents, metrics
from .loop import _NO_CONSISTENCY, score_split
from .pipeline import GoogleTranslator, JsonCache, shuffle_priorities, truncate


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class OfflineTranslator(GoogleTranslator):
    """캐시 미스면 즉시 실패한다 — 재생이 네트워크를 안 탔음을 증명하는 장치."""

    misses: int = 0

    def _raw(self, text: str) -> str:                       # noqa: D102
        type(self).misses += 1
        raise RuntimeError(f"캐시 미스 (offline): {text[:60]!r}")


def replay_iter(it_dir: Path, cfg: dict, translator, adequacy, contradiction,
                shuffles: int, seed: int) -> dict:
    rows = json.loads((it_dir / "train_rows.json").read_text(encoding="utf-8"))
    lift_T = max(cfg["t_grid"])
    key = str(lift_T)
    spaced = bool(cfg.get("spaced", True))
    tgt_spaced = bool(cfg.get("tgt_spaced", True))
    min_gap = int(cfg["min_gap"])

    idx, seg_texts, texts, full, real = [], [], [], [], []
    for i, r in enumerate(rows):
        d = (r.get("by_T") or {}).get(key)
        if not d or not r.get("seg_text") or r.get("full_trans") is None:
            continue
        eff = d.get("effective_single", d.get("effective"))
        if eff is None:
            continue
        idx.append(i)
        seg_texts.append(r["seg_text"])
        texts.append(r["text"])
        full.append(r["full_trans"])
        real.append(float(eff))

    rng = random.Random(seed)
    per_shuf: list[list[float | None]] = []
    for _ in range(shuffles):
        shuf = [shuffle_priorities(s, rng) for s in seg_texts]
        cut = [truncate(s, lift_T, spaced, min_gap)[0] for s in shuf]
        per_shuf.append(score_split(cut, texts, full, translator, adequacy,
                                    _NO_CONSISTENCY, spaced, tgt_spaced,
                                    contradiction).effective)
    mean_shuf: list[float | None] = []
    for i in range(len(seg_texts)):
        vals = [s[i] for s in per_shuf if s[i] is not None]
        mean_shuf.append(sum(vals) / len(vals) if vals else None)

    keep = [i for i, v in enumerate(mean_shuf) if v is not None]
    rl = metrics.rank_lift([real[i] for i in keep], [mean_shuf[i] for i in keep])
    rl["T"] = lift_T
    rl["n_rows"] = len(rows)
    return rl


def refocus(it_dir: Path, rl: dict, fmt_gate: float | None) -> dict:
    """저장된 지표 + 되살린 rank_lift 로 `summarize_critique` 를 다시 돌린다.

    `fmt_gate` 가 주어지면 format 관문을 그 값으로 낮춰 **뒤 관문까지 갔을 때** 무엇이
    뽑히는지 본다 (D2 검증). `None` 이면 원래대로 1.0.
    """
    crit = json.loads((it_dir / "critique.json").read_text(encoding="utf-8"))
    agg = crit.get("aggregate", {})
    reason = agg.get("focus_reason") or ""
    fmt = 1.0
    if "format_pass_rate" in reason:
        try:
            fmt = float(reason.split("format_pass_rate")[1].split("<")[0].strip())
        except (IndexError, ValueError):
            pass
    if fmt_gate is not None and fmt < 1.0:
        fmt = 1.0                     # 관문을 통과시킨 셈치고 뒤 관문을 본다
    m = {
        "format_pass_rate": fmt,
        "by_T": {"x": {"missing_boundaries": agg.get("max_missing_boundaries") or 0.0,
                       "premature_rate": agg.get("max_premature_rate") or 0.0}},
        "rank_lift": rl["lift"], "rank_lift_t": rl["t"],
    }
    return agents.summarize_critique(crit.get("cases") or [], m, agg.get("summary"),
                                     None, agg.get("priority_audit"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="런 디렉터리 (runs/de-en/run01)")
    p.add_argument("--iters", nargs="*", default=None, help="기본: train_rows 가 있는 전부")
    p.add_argument("--shuffles", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offline", action="store_true", help="캐시 미스면 즉시 실패")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="auto 면 torch.cuda.is_available() 로 고른다. 이 기계는 "
                        "드라이버는 멀쩡한데 cuInit 이 100(NO_DEVICE)으로 죽는 상태가 "
                        "있어서, 재생은 CPU 로도 끝까지 돌 수 있어야 한다")
    p.add_argument("--out", default=None, help="결과 JSON 경로")
    a = p.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    prof_p = run / "measured_profile.json"
    if prof_p.exists():
        prof = json.loads(prof_p.read_text(encoding="utf-8"))
        cfg.setdefault("spaced", prof.get("uses_spaces_between_words", True))

    code = cfg.get("tgt_code") or "en"
    tr_cache = JsonCache(run / "cache" / f"translate_{code}.json")
    cls = OfflineTranslator if a.offline else GoogleTranslator
    translator = cls(tgt_code=code, cache=tr_cache, workers=4,
                     use_context=not cfg.get("no_google_context", False))
    use_cuda = (a.device == "cuda" or
                (a.device == "auto" and _cuda_ok()))
    print(f"[device] {'cuda' if use_cuda else 'cpu'}", flush=True)
    adequacy = metrics.make_adequacy_backend(
        cfg.get("adequacy_backend", "cometkiwi"),
        batch_size=cfg.get("comet_batch_size", 16),
        gpus=1 if use_cuda else 0)
    contradiction = (None if cfg.get("no_contradiction") else
                     metrics.make_contradiction_backend(
                         cfg.get("contradiction_backend", "xlmr-anli"),
                         device=0 if use_cuda else -1))

    it_dirs = ([run / x for x in a.iters] if a.iters else
               sorted(d for d in run.glob("iter_*") if (d / "train_rows.json").exists()))
    out = {"run": str(run), "t_lift": max(cfg["t_grid"]), "iters": {}}
    for d in it_dirs:
        rl = replay_iter(d, cfg, translator, adequacy, contradiction, a.shuffles, a.seed)
        orig = refocus(d, rl, None)
        relaxed = refocus(d, rl, 1.0)
        out["iters"][d.name] = {"rank_lift": rl,
                                "focus_now": orig["focus"],
                                "focus_now_reason": orig["focus_reason"],
                                "focus_relaxed_fmt": relaxed["focus"],
                                "focus_relaxed_reason": relaxed["focus_reason"]}
        print(f"{d.name}: lift {rl['lift']:+.4f} se {rl['se']:.4f} t {rl['t']:+.2f} "
              f"n {rl['n']} | focus(D1만 고침) {orig['focus']} "
              f"| focus(D1+D2) {relaxed['focus']}", flush=True)
        print(f"    D1: {orig['focus_reason']}")
        print(f"    D1+D2: {relaxed['focus_reason']}")

    if a.offline:
        print(f"[offline] 캐시 미스 {OfflineTranslator.misses} 건")
    dst = Path(a.out) if a.out else run / "replay_focus.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
