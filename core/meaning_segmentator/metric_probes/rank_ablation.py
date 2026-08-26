"""순위 축 어블레이션 — 절단기가 쓰는 순위만 대조군으로 갈아끼우고 `effective` 를 다시 잰다.

## 왜 이게 필요한가

`rank_contra_gap` 은 **절단 후 살아남은 경계들끼리만** 잰다 (`metrics.rank_contra_gaps` —
`by_T[T].seg_text` 의 태그와 `pieces_contra` 를 쓴다). 그런데 en-de run04 test 실측에서
후보는 문장당 15.36개인데 T=6 생존은 2.69개다. **순위 결정의 82%(폐기분)를 지표가
원리적으로 못 본다** — 폐기된 경계는 렌더링이 안 되니 contra 값 자체가 없다.

즉 지금까지의 진단(`docs/RANK_METRIC_DIAGNOSIS.md` §6·§8)은 "순위를 고쳐도 품질이 안
오른다"까지 왔지만, **순위가 애초에 값을 하는가**(keep vs discard)는 한 번도 직접 재지
않았다. 이 스크립트가 그걸 잰다.

## 대조군 설계

절단기에 넘기는 **순위 라벨만** 바꾸고 나머지는 전부 고정한다 — 후보 집합, `want`
(= `chunk_budget−1`), `min_gap`, 번역기, 세 백엔드, 문장 집합이 모두 같다. 바뀌는 것은
`truncate()` 가 고르는 **keep 집합** 하나뿐이다.

  `real`     저장된 `<SEG:n>` 그대로 (모델이 매긴 확신 순위)
  `shufNN`   같은 후보에서 순위 라벨만 무작위 치환 → **순열 귀무분포**
  `even`     이 예산에서 가장 등간격에 가까운 자리부터 순위 부여 (위치 사전확률 기준선)

`even` 이 있는 이유는 `boundary_probe` 의 교훈이다 — **경계 관련 지표의 기준선은
무작위가 아니라 위치 사전확률이다** (`../NLI_ALTERNATIVES.md` §2 표). 무작위만 이겨서는
"순위가 내용을 읽는다"가 아니라 "순위가 위치를 읽는다"일 수 있다.

## 왜 `--no-priority` 가 아니라 순열인가

`eval_prompt.py --no-priority` 로 순위 없는 프롬프트를 재는 것은 이 질문의 답이 안 된다.
`truncate()` 는 **순위가 없으면 절단을 하지 않는다**(`pipeline.py`, "순위 없음 — 절단 불가").
그러면 T 가 무력해져 후보 15개가 전부 남고, 조각 수도 `laal` 도 NLI 잡음 바닥도 달라진다.
현행 설계에서 "순위 없음"은 "절단 없음"과 동의어라 등지연 비교가 성립하지 않는다.

## 비용

**분절 LLM 호출 0건.** 기존 런의 `*_rows.json` 에 순위 태그가 그대로 남아 있어 재분절이
필요 없다. 드는 것은 조각 번역(gtx, 무료)과 로컬 GPU 채점뿐이다. `consistency` 는
`effective` 에 안 들어가는 보고 지표라 기본으로 끈다 (`--consistency` 로 켤 수 있다).

  PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.metric_probes.rank_ablation \
      --run-id en-de/run04 --split test --t 6 --shuffles 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from ..autoseg import metrics, noise_floor
from ..autoseg.gateway import Gateway
from ..autoseg.loop import score_split, target_is_spaced
from ..autoseg.pipeline import (TAG_RE, GoogleTranslator, parse_translator_id, JsonCache, chunk_budget, strip_tags, tag, truncate, unit_count)
from .paths import OUT_RUNS, SEG_RUNS


# ── 순위 라벨 조작 ────────────────────────────────────────────────────────

def retag(seg_text: str, prios: list[int]) -> str:
    """등장 순서대로 새 순위를 붙인다. **경계 위치는 건드리지 않는다.**"""
    it = iter(prios)
    return TAG_RE.sub(lambda _m: tag(next(it)), seg_text)


def tag_positions(seg_text: str, spaced: bool) -> tuple[list[int], int]:
    """태그별 어절 위치(그 앞까지의 누적 어절 수)와 문장 전체 어절 수.

    `truncate()` 의 `min_gap` 계산과 같은 방식으로 센다 — 두 곳이 어긋나면 `even` 이
    실제로는 등간격이 아닌 자리를 고르게 된다.
    """
    parts = TAG_RE.split(seg_text.strip())
    pieces = [p.strip() for p in parts[::2]]
    wc = [unit_count(p, spaced) for p in pieces]
    pos, acc = [], 0
    for w in wc[:-1]:
        acc += w
        pos.append(acc)
    return pos, acc + wc[-1]


def even_prios(seg_text: str, T: int, spaced: bool) -> list[int]:
    """이 예산에서 **등간격에 가장 가까운 자리부터** 1..N 순위를 매긴다.

    순위가 T 에 의존하는 것은 의도된 것이다 — "몇 개를 남길지"가 정해져야 이상적 자리가
    정해진다. `real`/`shuf` 는 T 무관한 순위를 쓰지만, 셋 다 같은 `want` 만큼만 살아남고
    같은 `min_gap` 을 통과하므로 비교는 등가다.
    """
    pos, total = tag_positions(seg_text, spaced)
    n = len(pos)
    want = max(0, chunk_budget(strip_tags(seg_text, spaced), T, spaced) - 1)
    want = min(want, n)
    ideals = [total * (j + 1) / (want + 1) for j in range(want)]

    prios = [0] * n
    used: set[int] = set()
    rank = 1
    for target in ideals:                      # 이상적 자리마다 가장 가까운 후보를 뽑는다
        cand = min((i for i in range(n) if i not in used),
                   key=lambda i: abs(pos[i] - target), default=None)
        if cand is None:
            break
        used.add(cand)
        prios[cand] = rank
        rank += 1
    # 남은 후보는 이상적 자리들과의 최소 거리 순으로 뒤에 붙인다 (min_gap 보충용).
    rest = sorted((i for i in range(n) if i not in used),
                  key=lambda i: min((abs(pos[i] - t) for t in ideals), default=pos[i]))
    for i in rest:
        prios[i] = rank
        rank += 1
    return prios


# ── 결정론적 재순위 ──────────────────────────────────────────────────────
#
# **왜 어블레이션으로만 검증하는가.** 표면 특징 → contra 대응을 문장 안 Spearman 으로
# 미리 재 보면 부정적이다 (교차런에서 구두점 표의 부호가 뒤집힌다: ko-en 자기런 +0.226,
# en-de 표로 코딩 −0.226). 그러나 그 자는 **생존 경계끼리만** 보는 것이라
# `docs/RANK_METRIC_DIAGNOSIS.md` §8.2 에서 이미 신뢰 못 한다고 결론 난 계기이고,
# 유효 표본도 문장×T 78개 중 19개뿐이다. keep-vs-discard 를 직접 재는 것은 이 파일의
# 어블레이션뿐이므로, 판정은 여기서 한다.

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "while", "whereas",
                  "instead", "rather", "despite", "nevertheless", "nonetheless",
                  "unless", "except", "still"}

# 위험 배수 — 전역 평균 contra 대비. 기본값은 en-multi/run06 test 실측 (바닥 보정 후).
# **다른 런에 쓰면 교차 검증, 같은 런에 쓰면 자기적합**이다. `--risk-table` 로 갈아낀다.
RISK_DEFAULT = {
    "punct": {"문말": 0.56, "쉼표": 0.88, "쉼표+역접": 0.33, "세미콜론/콜론": 0.90,
              "구두점 없음": 1.04, "기타구두점": 0.90},
    "pos": {"0/3": 0.77, "1/3": 1.30, "2/3": 1.13},
    "len": {"1-3": 1.08, "4-7": 0.75, "8+": 1.29},
}


def boundary_feats(pieces: list[str], j: int, n: int, spaced: bool) -> dict:
    """경계 j 의 표면 특징. **번역을 안 본다** — 절단 시점에 알 수 있는 것만."""
    prev = (pieces[j] or "").rstrip()
    nxt = (pieces[j + 1] if j + 1 < len(pieces) else "") or ""
    w = nxt.strip().split()[:1]
    w = w[0].strip(",.;:!?\"'()").lower() if w else ""
    ch = prev[-1:] if prev else ""
    if ch in ".?!\u2026\u3002\uff1f\uff01":
        punct = "문말"
    elif ch in ",\uff0c\u3001":
        punct = "쉼표+역접" if w in CONTRAST_WORDS else "쉼표"
    elif ch in ";:\uff1b\uff1a":
        punct = "세미콜론/콜론"
    elif ch and not ch.isalnum():
        punct = "기타구두점"
    else:
        punct = "구두점 없음"
    L = unit_count(prev, spaced)
    return {"punct": punct,
            "pos": f"{int(j / max(1, n) * 3)}/3",
            "len": "1-3" if L <= 3 else ("4-7" if L <= 7 else "8+")}


def risk_scores(seg_text: str, spaced: bool, table: dict) -> list[float]:
    """경계별 위험 점수 (작을수록 안전 = 높은 순위). 배수의 곱."""
    parts = TAG_RE.split(seg_text.strip())
    pieces = [p.strip() for p in parts[::2]]
    n = len(pieces) - 1
    out = []
    for j in range(n):
        f = boundary_feats(pieces, j, n, spaced)
        r = 1.0
        for ax, key in (("punct", f["punct"]), ("pos", f["pos"]), ("len", f["len"])):
            r *= float(table.get(ax, {}).get(key, 1.0))
        out.append(r)
    return out


def _ranks_from_scores(scores: list[float]) -> list[int]:
    """점수 오름차순으로 1..N. 동점은 등장 순서."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    prios = [0] * len(scores)
    for k, i in enumerate(order):
        prios[i] = k + 1
    return prios


def rerank_prios(seg_text: str, spaced: bool, table: dict) -> list[int]:
    return _ranks_from_scores(risk_scores(seg_text, spaced, table))


def blend_prios(seg_text: str, spaced: bool, table: dict, lam: float) -> list[int]:
    """LLM 순위 백분위와 위험 백분위를 섞는다. lam=0 이면 real, 1 이면 rerank.

    **백분위로 섞는 이유**: 위험 점수는 배수라 스케일이 문장마다 다르고, 순위는 정수다.
    둘 다 문장 안 백분위로 바꿔야 λ 가 의미를 갖는다.
    """
    llm = [int(m.group(1)) for m in TAG_RE.finditer(seg_text) if m.group(1)]
    risk = risk_scores(seg_text, spaced, table)
    n = len(risk)
    if len(llm) != n or n < 2:
        return llm or list(range(1, n + 1))

    def pct(vals: list) -> list[float]:
        order = sorted(range(n), key=lambda i: (vals[i], i))
        out = [0.0] * n
        for k, i in enumerate(order):
            out[i] = k / (n - 1)
        return out

    lp, rp = pct(llm), pct(risk)
    return _ranks_from_scores([(1 - lam) * lp[i] + lam * rp[i] for i in range(n)])


def arm_seg_texts(rows: list[dict], T: int, spaced: bool, min_gap: int,
                  mode: str, rng: random.Random, table: dict | None = None,
                  lam: float = 0.5) -> tuple[list[str], list[int]]:
    """한 대조군의 절단 결과. 반환 `(절단된 seg_text 목록, missing_boundaries 목록)`."""
    table = table or RISK_DEFAULT
    cut_texts, missings = [], []
    for r in rows:
        seg = r["seg_text"]
        n = len(TAG_RE.findall(seg))
        if mode == "real":
            src = seg
        elif mode == "even":
            src = retag(seg, even_prios(seg, T, spaced)) if n else seg
        elif mode == "rerank":
            src = retag(seg, rerank_prios(seg, spaced, table)) if n else seg
        elif mode == "blend":
            src = retag(seg, blend_prios(seg, spaced, table, lam)) if n else seg
        else:                                   # shuffle
            p = list(range(1, n + 1))
            rng.shuffle(p)
            src = retag(seg, p) if n else seg
        cut, miss = truncate(src, T, spaced, min_gap)
        cut_texts.append(cut)
        missings.append(miss)
    return cut_texts, missings


# ── 통계 ─────────────────────────────────────────────────────────────────

def paired(a: list[float | None], b: list[float | None]) -> dict:
    """쌍체 Δ = a − b. 같은 문장끼리만 센다 (둘 다 값이 있는 자리)."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 2:
        return {"n": len(d), "delta": None, "se": None, "t": None}
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / (len(d) ** 0.5)
    return {"n": len(d), "delta": round(mean, 5), "se": round(se, 5),
            "t": round(mean / se, 2) if se else None}


def mean_of(xs: list[float | None]) -> float | None:
    ok = [x for x in xs if x is not None]
    return statistics.fmean(ok) if ok else None


def r5(x: float | None, nd: int = 5) -> float | None:
    return None if x is None else round(x, nd)


def cell(x: float | None, fmt: str = "+.5f", width: int = 11) -> str:
    return ("—" if x is None else format(x, fmt)).rjust(width)


# ── 본체 ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="순위 축 어블레이션 (오프라인, 분절 호출 0건)")
    p.add_argument("--run-id", required=True, help="기준 런 경로 (runs/ 이하)")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--rows", default=None, help="직접 지정할 *_rows.json (기본: {split}_rows.json)")
    p.add_argument("--t", type=int, nargs="+", default=[6], help="평가할 T. 기본 6")
    p.add_argument("--shuffles", type=int, default=20, help="순열 대조군 개수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="문장 수 제한 (스모크용)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--comet-batch-size", type=int, default=16)
    p.add_argument("--consistency", action="store_true",
                   help="보고 지표 consistency 도 잰다 (기본은 끔 — effective 에 안 들어간다)")
    p.add_argument("--rerank", action="store_true",
                   help="결정론적 재순위 대조군을 추가한다 (표면 특징 기반)")
    p.add_argument("--risk-table", default=None,
                   help="위험 배수표 JSON. 없으면 내장 기본값(en-multi/run06 실측). "
                        "**평가 대상과 다른 런의 표를 줘야 교차 검증**이다")
    p.add_argument("--blend-lambda", type=float, nargs="*", default=[0.3, 0.5],
                   help="blend 대조군의 λ. 0=real, 1=rerank")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    run_dir = SEG_RUNS / args.run_id
    if not run_dir.exists():
        print(f"런 디렉토리 없음: {run_dir}", file=sys.stderr)
        return 1
    rows_fp = Path(args.rows) if args.rows else run_dir / f"{args.split}_rows.json"
    if not rows_fp.exists():
        print(f"행 파일 없음: {rows_fp}", file=sys.stderr)
        return 1

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = json.loads(rows_fp.read_text(encoding="utf-8"))
    # 순위 없는 행(비교군 프롬프트)·전체 번역이 없는 행은 절단 자체가 성립하지 않는다.
    rows = [r for r in rows
            if r.get("full_trans") and all(m for m in TAG_RE.findall(r.get("seg_text") or ""))]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("쓸 수 있는 행이 없다 (순위 태그 또는 full_trans 누락)", file=sys.stderr)
        return 1

    profile = json.loads((run_dir / "language_profile.json").read_text(encoding="utf-8"))
    mp = run_dir / "measured_profile.json"
    if mp.exists():
        from ..autoseg import data as _data
        spaced, _tp = _data.profile_settings(
            json.loads(mp.read_text(encoding="utf-8")))
    else:
        spaced = bool(profile.get("uses_spaces_between_words", True))
    tgt_spaced = bool(cfg.get("tgt_spaced", target_is_spaced(cfg.get("tgt_lang", "English"))))
    min_gap = int(cfg.get("min_gap") or 0)

    texts = [r["text"] for r in rows]
    full = [r["full_trans"] for r in rows]

    # 번역기·백엔드는 기준 런에서 그대로 상속한다. 다른 자로 재면 같은 축이 아니다.
    gw = Gateway(model=cfg.get("model", "gpt-5-mini"), budget=0.5)
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")
    # 번역기는 Google 하나다 (LLM 분기 제거). `translator_id` 가 없는 옛 런은 타깃
    # 언어명에서 코드를 유도한다.
    backend, code, ctx = parse_translator_id(cfg.get("translator_id") or "",
                                             to_lang_code(cfg["tgt_lang"]))
    translator = GoogleTranslator(tgt_code=code, cache=tr_cache,
                                  workers=min(args.workers, 4),
                                  use_context=ctx, backend=backend)

    adequacy = metrics.make_adequacy_backend(cfg.get("adequacy_backend", "cometkiwi"),
                                             batch_size=args.comet_batch_size)
    nli_key = cfg.get("contradiction_backend", "deberta-mnli")
    contradiction = metrics.make_contradiction_backend(nli_key)

    class _NoConsistency:
        """`effective = adequacy × (1 − contradiction)` 에 안 들어가는 보고 지표.
        어블레이션은 `effective` 만 보므로 기본으로 끄고 호출 수를 아낀다."""
        name = "none"

        def score(self, srcs, hyps, refs):
            return [0.0] * len(hyps)

    if args.consistency:
        cons_name = cfg.get("consistency_backend", "nli")
        consistency = (metrics.make_backend("nli", model_name=metrics.NLI_MODELS[nli_key],
                                            batch_size=args.comet_batch_size)
                       if cons_name == "nli" else
                       metrics.make_backend(cons_name, gw=gw,
                                            **({"batch_size": args.comet_batch_size}
                                               if cons_name in metrics.COMET_CHECKPOINTS else {})))
    else:
        consistency = _NoConsistency()

    # 순위 정렬도(`rank_contra_gap`)를 함께 찍기 위한 잡음 바닥. 이미 잰 파일만 쓴다
    # (없으면 보정 없이 raw — 그 경우 앞쪽 경계가 구조적으로 불리하다).
    floor_fp = run_dir / "contra_floor.json"
    floor_fn = None
    if floor_fp.exists():
        floor = json.loads(floor_fp.read_text(encoding="utf-8"))
        floor_fn = lambda n: noise_floor.floor_lookup(floor, n)

    risk_table = (json.loads(Path(args.risk_table).read_text(encoding="utf-8"))
                  if args.risk_table else RISK_DEFAULT)
    rr_arms = []
    if args.rerank:
        rr_arms = ["rerank"] + [f"blend{lam:g}" for lam in args.blend_lambda]
    arms = ["real", "even"] + rr_arms + [f"shuf{i:02d}" for i in range(args.shuffles)]
    out: dict = {
        "run_id": args.run_id, "split": args.split, "rows": len(rows),
        "t_grid": args.t, "shuffles": args.shuffles, "seed": args.seed,
        "min_gap": min_gap, "translator": tr_id,
        "adequacy_backend": adequacy.name, "contradiction_backend": contradiction.name,
        "consistency_backend": (cfg.get("consistency_backend") if args.consistency else None),
        "rerank": bool(args.rerank),
        "risk_table": (risk_table if args.rerank else None),
        "risk_table_src": (args.risk_table or "built-in(en-multi/run06)") if args.rerank else None,
        "by_T": {},
    }

    try:
        for T in args.t:
            per_arm: dict[str, dict] = {}
            eff_by_arm: dict[str, list] = {}
            rng = random.Random(args.seed + T)
            t0 = time.time()
            for a_i, arm in enumerate(arms):
                lam = 0.0
                if arm in ("real", "even", "rerank"):
                    mode = arm
                elif arm.startswith("blend"):
                    mode, lam = "blend", float(arm[5:])
                else:
                    mode = "shuffle"
                cut_texts, missings = arm_seg_texts(rows, T, spaced, min_gap, mode, rng,
                                                    table=risk_table, lam=lam)
                sp = score_split(cut_texts, texts, full, translator, adequacy,
                                 consistency, spaced, tgt_spaced, contradiction)

                # `rank_contra_gap` 도 같은 방식으로 다시 잰다 — 순위를 셔플했는데도
                # 지표가 안 움직이면, 그 지표가 순위를 재고 있지 않다는 직접 증거다.
                gap_rows = [{"by_T": {str(T): {"seg_text": ct,
                                               "pieces_contra": pc,
                                               "pieces_tgt": pt}}}
                            for ct, pc, pt in zip(cut_texts, sp.pieces_contra, sp.pieces_tgt)]
                gaps = metrics.rank_contra_gaps(gap_rows, T, floor_fn=floor_fn,
                                                tgt_spaced=tgt_spaced)

                eff_by_arm[arm] = list(sp.effective)
                per_arm[arm] = {
                    "effective": r5(mean_of(sp.effective)),
                    "adequacy": r5(mean_of(sp.adequacy)),
                    "contradiction": r5(mean_of(sp.contradiction)),
                    "laal_words": r5(mean_of(sp.laal_words), 4),
                    "k": r5(mean_of([float(x) for x in sp.k]), 4),
                    "missing_boundaries": r5(mean_of([float(x) for x in missings]), 4),
                    "n_effective": sum(1 for x in sp.effective if x is not None),
                    "rank_contra_gap": r5(statistics.fmean(gaps) if gaps else None),
                    "rank_contra_gap_n": len(gaps),
                }
                print(f"  [T={T}] {arm:8s} ({a_i + 1}/{len(arms)})  "
                      f"effective {per_arm[arm]['effective']}  "
                      f"laal {per_arm[arm]['laal_words']}  "
                      f"gap {per_arm[arm]['rank_contra_gap']}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

            # 쌍체 비교 — 같은 문장끼리. real 대비 각 대조군.
            shuf = [a for a in arms if a.startswith("shuf")]
            deltas = {a: paired(eff_by_arm["real"], eff_by_arm[a])
                      for a in arms if a != "real"}
            # 셔플 대조군을 문장별로 평균낸 "평균적 무작위 순위" 와의 쌍체 비교
            mean_shuf_row: list[float | None] = []
            for i in range(len(rows)):
                vals = [eff_by_arm[a][i] for a in shuf if eff_by_arm[a][i] is not None]
                mean_shuf_row.append(statistics.fmean(vals) if vals else None)
            vs_shuffle = paired(eff_by_arm["real"], mean_shuf_row)

            # 순열 귀무분포 — 셔플 평균들의 분포에서 real 이 어디 서는가 (단측).
            shuf_means = [per_arm[a]["effective"] for a in shuf
                          if per_arm[a]["effective"] is not None]
            real_mean = per_arm["real"]["effective"]
            ge = sum(1 for m in shuf_means if m >= real_mean)
            perm_p = (ge + 1) / (len(shuf_means) + 1) if shuf_means else None
            if not shuf:                        # --shuffles 0 (재순위만 볼 때)
                vs_shuffle = {"n": 0, "delta": None, "se": None, "t": None}

            # 셔플 대조군의 `rank_contra_gap` 분포 — 지표가 순위를 재고 있는지의 직접 검사
            shuf_gaps = [per_arm[a]["rank_contra_gap"] for a in shuf
                         if per_arm[a]["rank_contra_gap"] is not None]

            out["by_T"][str(T)] = {
                # **문장별 값을 남긴다.** 집계만 남기면 나중에 다른 통계를 물을 때마다
                # 채점을 다시 해야 한다 — 특히 셔플끼리의 쌍체 비교(참값 0 인 귀무 상태)는
                # 문장별 값 없이는 재계산이 불가능하다.
                "per_sentence": {"ids": [r.get("id") for r in rows], **eff_by_arm},
                "arms": per_arm,
                "paired_vs_real": deltas,
                "real_vs_mean_shuffle": vs_shuffle,
                "shuffle_mean": r5(statistics.fmean(shuf_means) if shuf_means else None),
                "shuffle_sd": r5(statistics.stdev(shuf_means) if len(shuf_means) > 1 else None),
                "shuffle_gap_mean": r5(statistics.fmean(shuf_gaps) if shuf_gaps else None),
                "shuffle_gap_sd": r5(statistics.stdev(shuf_gaps) if len(shuf_gaps) > 1 else None),
                "perm_p_one_sided": r5(perm_p, 4),
                "seconds": round(time.time() - t0, 1),
            }

        label = args.label or f"{args.run_id.replace('/', '_')}_{args.split}"
        out_dir = OUT_RUNS / "rank_ablation"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"{label}.json"
        fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n== 순위 어블레이션  {args.run_id}/{args.split}  n={len(rows)}  "
              f"min_gap={min_gap}  번역기={tr_id}")
        for T in args.t:
            d = out["by_T"][str(T)]
            a = d["arms"]
            print(f"\n[T={T}]  Δ = real − 대조군 (양수면 모델 순위가 이긴 것)")
            print("  " + "대조군".ljust(12) + "effective".rjust(10) + "Δ vs real".rjust(12)
                  + "t".rjust(7) + "laal".rjust(8) + "k".rjust(7) + "gap".rjust(10))

            def line(name: str, arm_d: dict, delta: dict | None) -> None:
                print("  " + name.ljust(12)
                      + cell(arm_d["effective"], ".5f", 10)
                      + cell((delta or {}).get("delta"), "+.5f", 12)
                      + cell((delta or {}).get("t"), "+.2f", 7)
                      + cell(arm_d["laal_words"], ".3f", 8)
                      + cell(arm_d["k"], ".2f", 7)
                      + cell(arm_d["rank_contra_gap"], "+.4f", 10))

            line("real", a["real"], None)
            line("even", a["even"], d["paired_vs_real"].get("even"))
            for nm in rr_arms:
                line(nm, a[nm], d["paired_vs_real"].get(nm))
            line("shuffle(평균)", {"effective": d["shuffle_mean"],
                                  "laal_words": mean_of([a[s]["laal_words"] for s in
                                                         a if s.startswith("shuf")]),
                                  "k": mean_of([a[s]["k"] for s in a if s.startswith("shuf")]),
                                  "rank_contra_gap": d["shuffle_gap_mean"]},
                 d["real_vs_mean_shuffle"])
            print(f"    shuffle 평균의 sd {d['shuffle_sd']}  ·  "
                  f"순열 p(단측, real 이 더 좋은가) {d['perm_p_one_sided']}  ·  "
                  f"gap 셔플 분포 {d['shuffle_gap_mean']} ± {d['shuffle_gap_sd']}")
        print(f"\n저장: {fp}")
        return 0
    finally:
        tr_cache.flush()
        if isinstance(translator, GoogleTranslator):
            translator.close()
        gw.close()


if __name__ == "__main__":
    sys.exit(main())
