"""어절을 점진적으로 붙이며 표현이 **급변하는 지점**을 찾는다 — 분절기 후보 측정.

`embed_check.py` 와 `future_dep.py` 는 NLI 가 앉은 **지표** 자리를 겨눴다. 이 스크립트가
겨누는 자리는 다르다: 여기 나오는 값은 벌점이 아니라 **경계 제안**이므로, 경쟁 상대는
`contradiction` 이 아니라 **A2 Segmenter(LLM 프롬프트)** 다. 따라서 판정 기준도 관문이
아니라 "LLM 이 찍은 경계와 우연 이상으로 일치하는가" 다.

세 가지 점수를 잰다. 후보 위치 i 는 어절 i 와 i+1 사이다.

  delta_prefix(i)  1 − cos(E(w1..wi), E(w1..w_{i+1}))
                   prefix 를 **각각 독립 인코딩**해 한 어절 추가의 충격을 본다.
                   prefix 가 길어질수록 한 어절의 비중이 줄어 값이 구조적으로 작아진다.
  tile(i, k)       1 − cos(E(w_{i−k+1..i}), E(w_{i+1..i+k}))
                   인접 **창** 두 개의 유사도. TextTiling 의 구성이고, 창 길이가 고정이라
                   위치 교란이 delta_prefix 보다 작다.
  ctx_delta(i)     문장 **한 번만** 인코딩하고 `[0,i)` 와 `[0,i+1)` 토큰 평균을 비교한다.
                   문맥 표현 위에서의 급변이라 가장 싸고, 미래 정보가 이미 섞여 있다.

**정답은 LLM 경계가 아니다.** 우리가 개선하려는 대상이 그 프롬프트이므로 일치도는
정답률이 아니라 *동의율*이다. 그래서 세 비교군을 같이 낸다 — 무작위 위치, 기계적 등분
(`metrics.mechanical_split` 과 같은 발상), 그리고 문장별로 경계 수를 맞춘 상한.
우연 수준이면 이 접근은 여기서 끝내는 것이 맞다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.boundary_probe --run-id ko-en/run04
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ..autoseg import metrics
from .future_dep import ENCODERS, ContextEncoder, _cos

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS


def word_spans(text: str) -> list[tuple[int, int]]:
    """어절별 (시작, 끝) 문자 오프셋."""
    out, pos = [], 0
    for w in text.split():
        s = text.index(w, pos)
        out.append((s, s + len(w)))
        pos = s + len(w)
    return out


def load_sentences(run_dir: Path) -> list[dict]:
    """`{text, words, by_T: {T: 경계 어절 인덱스 목록}}`.

    경계 인덱스 i 는 "어절 i 와 i+1 사이" 를 뜻한다 (1-based 어절, i ∈ [1, n−1])."""
    paths = sorted(run_dir.glob("iter_*/train_rows.json"))
    paths += sorted(run_dir.glob("iter_*/dev_rows.json"))
    if (run_dir / "test_rows.json").exists():
        paths.append(run_dir / "test_rows.json")

    seen: dict[tuple, dict] = {}
    for p in paths:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = f"{p.parent.name}/{p.stem}"
        for r in rows:
            for T, d in (r.get("by_T") or {}).items():
                pieces = d.get("pieces_src") or []
                if len(pieces) < 2:
                    continue
                text = " ".join(pieces)
                words = text.split()
                cuts, acc = [], 0
                ok = True
                for piece in pieces[:-1]:
                    acc += len(piece.split())
                    if not (1 <= acc <= len(words) - 1):
                        ok = False
                        break
                    cuts.append(acc)
                if not ok or not cuts:
                    continue
                key = (tag, r["id"], int(T))
                seen[key] = {"key": ["/".join(map(str, key))], "text": text,
                             "n_words": len(words), "T": int(T), "cuts": cuts}
    return list(seen.values())


# ── 점수 ─────────────────────────────────────────────────────────────────

def score_delta_prefix(enc: ContextEncoder, sents: list[dict]) -> None:
    texts, spans, index = [], [], []
    for si, s in enumerate(sents):
        ws = word_spans(s["text"])
        for i in range(1, s["n_words"]):
            texts.append(s["text"][: ws[i - 1][1]])          # w1..wi
            spans.append((0, None))
            index.append((si, i, "a"))
            texts.append(s["text"][: ws[i][1]])              # w1..w_{i+1}
            spans.append((0, None))
            index.append((si, i, "b"))
    vecs = enc.span_range_vectors(texts, spans)
    buf: dict[tuple, dict] = {}
    for (si, i, which), v in zip(index, vecs):
        buf.setdefault((si, i), {})[which] = v
    for (si, i), d in buf.items():
        sents[si].setdefault("delta_prefix", {})[i] = 1.0 - _cos(d["a"], d["b"])


def score_tile(enc: ContextEncoder, sents: list[dict], k: int = 3) -> None:
    texts, spans, index = [], [], []
    for si, s in enumerate(sents):
        ws = word_spans(s["text"])
        n = s["n_words"]
        for i in range(1, n):
            l0 = ws[max(0, i - k)][0]
            l1 = ws[i - 1][1]
            r0 = ws[i][0]
            r1 = ws[min(n - 1, i + k - 1)][1]
            texts.append(s["text"][l0:l1])
            spans.append((0, None))
            index.append((si, i, "a"))
            texts.append(s["text"][r0:r1])
            spans.append((0, None))
            index.append((si, i, "b"))
    vecs = enc.span_range_vectors(texts, spans)
    buf: dict[tuple, dict] = {}
    for (si, i, which), v in zip(index, vecs):
        buf.setdefault((si, i), {})[which] = v
    for (si, i), d in buf.items():
        sents[si].setdefault("tile", {})[i] = 1.0 - _cos(d["a"], d["b"])


def score_ctx_delta(enc: ContextEncoder, sents: list[dict]) -> None:
    texts, spans, index = [], [], []
    for si, s in enumerate(sents):
        ws = word_spans(s["text"])
        for i in range(1, s["n_words"]):
            texts.append(s["text"])
            spans.append((0, ws[i - 1][1]))
            index.append((si, i, "a"))
            texts.append(s["text"])
            spans.append((0, ws[i][1]))
            index.append((si, i, "b"))
    vecs = enc.span_range_vectors(texts, spans)
    buf: dict[tuple, dict] = {}
    for (si, i, which), v in zip(index, vecs):
        buf.setdefault((si, i), {})[which] = v
    for (si, i), d in buf.items():
        sents[si].setdefault("ctx_delta", {})[i] = 1.0 - _cos(d["a"], d["b"])


def positional_prior(sents: list[dict], key: str) -> None:
    """**문장 내용을 전혀 안 보는** 위치 사전확률 기준선.

    각 후보 위치에 그 상대 위치(i/n) 10분위의 **코퍼스 평균 점수**를 그대로 준다 —
    이 문장이 무슨 말인지는 하나도 안 쓴다. 그런데도 일치율·AUC 가 실제 점수와
    비슷하게 나오면, 그 점수가 잡은 것은 의미가 아니라 **위치**다.

    이 대조가 필요한 이유: `delta_prefix`·`ctx_delta` 는 prefix 가 길수록 한 어절의
    비중이 줄어 단조 감소하고, LLM 경계도 문장 안에서 균등하게 흩어져 있지 않다.
    두 분포가 겹치기만 해도 일치율이 우연보다 높게 나온다."""
    buckets: dict[int, list[float]] = {}
    for s in sents:
        for i, v in (s.get(key) or {}).items():
            buckets.setdefault(int(10 * i / s["n_words"]), []).append(v)
    mu = {b: sum(v) / len(v) for b, v in buckets.items() if v}
    for s in sents:
        s[key + "_prior"] = {i: mu.get(int(10 * i / s["n_words"]), 0.0)
                             for i in (s.get(key) or {})}


def positional_residual(sents: list[dict], key: str) -> None:
    """상대 위치(i/n) 10분위의 코퍼스 평균을 빼 위치 교란을 없앤다.

    `delta_prefix` 는 prefix 가 길수록 한 어절의 비중이 줄어 값이 구조적으로 작아지고,
    그러면 봉우리 탐색이 **문장 앞쪽만** 고르게 된다. `future_dep` 의 남은-어절 버킷
    보정과 같은 구조다."""
    buckets: dict[int, list[float]] = {}
    for s in sents:
        for i, v in (s.get(key) or {}).items():
            buckets.setdefault(int(10 * i / s["n_words"]), []).append(v)
    mu = {b: sum(v) / len(v) for b, v in buckets.items() if v}
    for s in sents:
        out = {}
        for i, v in (s.get(key) or {}).items():
            out[i] = v - mu.get(int(10 * i / s["n_words"]), 0.0)
        s[key + "_resid"] = out


# ── 평가 ─────────────────────────────────────────────────────────────────

def topk_positions(scores: dict[int, float], k: int) -> list[int]:
    return sorted(sorted(scores, key=lambda i: -scores[i])[:k])


def agreement(sents: list[dict], key: str, tolerance: int = 0) -> dict:
    """LLM 경계와 같은 개수를 뽑아 일치율을 낸다. 개수가 같으니 P=R=F1 이다."""
    hits = tot = 0
    per_T: dict[int, list[int]] = {}
    for s in sents:
        sc = s.get(key) or {}
        if not sc:
            continue
        gold = set(s["cuts"])
        pred = topk_positions(sc, len(gold))
        h = sum(1 for p in pred if any(abs(p - g) <= tolerance for g in gold))
        hits += h
        tot += len(gold)
        per_T.setdefault(s["T"], [0, 0])
        per_T[s["T"]][0] += h
        per_T[s["T"]][1] += len(gold)
    return {"hit_rate": round(hits / tot, 4) if tot else None, "n_boundaries": tot,
            "by_T": {T: round(a / b, 4) for T, (a, b) in sorted(per_T.items()) if b}}


def chance_rate(sents: list[dict], tolerance: int = 0, trials: int = 20,
                seed: int = 0) -> dict:
    """무작위 위치 기준선. 해석 가능한 하한은 이것이지 0 이 아니다."""
    rng = random.Random(seed)
    hits = tot = 0
    for s in sents:
        gold = set(s["cuts"])
        cand = list(range(1, s["n_words"]))
        if len(cand) < len(gold):
            continue
        for _ in range(trials):
            pred = rng.sample(cand, len(gold))
            hits += sum(1 for p in pred if any(abs(p - g) <= tolerance for g in gold))
            tot += len(gold)
    return {"hit_rate": round(hits / tot, 4) if tot else None, "n_boundaries": tot}


def mechanical_rate(sents: list[dict], tolerance: int = 0) -> dict:
    """등간격 절단 기준선. 곡선의 하한 비교군과 같은 발상이다."""
    hits = tot = 0
    for s in sents:
        gold = set(s["cuts"])
        k = len(gold)
        n = s["n_words"]
        pred = [max(1, min(n - 1, round((j + 1) * n / (k + 1)))) for j in range(k)]
        hits += sum(1 for p in pred if any(abs(p - g) <= tolerance for g in gold))
        tot += k
    return {"hit_rate": round(hits / tot, 4) if tot else None, "n_boundaries": tot}


def rank_alignment(sents: list[dict], key: str) -> dict:
    """LLM 경계에서의 점수 vs 비경계 위치에서의 점수. 봉우리가 실제로 거기 있는가.

    일치율은 상위 k 개만 보므로 "거의 맞았다" 를 못 본다. 이 값은 분포 전체를 본다:
    **경계 위치의 평균 점수 − 비경계 위치의 평균 점수**. 양수 = 경계에서 급변한다."""
    gaps, aucs = [], []
    for s in sents:
        sc = s.get(key) or {}
        if not sc:
            continue
        gold = set(s["cuts"])
        pos = [v for i, v in sc.items() if i in gold]
        neg = [v for i, v in sc.items() if i not in gold]
        if not pos or not neg:
            continue
        gaps.append(sum(pos) / len(pos) - sum(neg) / len(neg))
        wins = sum(1 for a in pos for b in neg if a > b)
        ties = sum(1 for a in pos for b in neg if a == b)
        aucs.append((wins + 0.5 * ties) / (len(pos) * len(neg)))
    return {"mean_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
            "auc": round(sum(aucs) / len(aucs), 4) if aucs else None,
            "n_sentences": len(gaps)}


def render(result: dict) -> str:
    L = ["# 어절 점진 추가 — 표현 급변점이 의미 분절 지점인가", "",
         f"런: `{result['run_id']}` · 문장×T {result['n_units']}개 · "
         f"LLM 경계 {result['n_boundaries']}개 · LLM 호출 0 · 번역 호출 0", "",
         "여기 나오는 값은 **벌점이 아니라 경계 제안**이다. 그래서 경쟁 상대가 "
         "`contradiction` 이 아니라 **A2 Segmenter(LLM 프롬프트)** 이고, 판정 기준도 "
         "관문이 아니라 동의율이다. LLM 경계는 정답이 아니라 비교 대상이다 — 우리가 "
         "개선하려는 대상이 바로 그 프롬프트이기 때문이다.", "",
         "점수마다 문장의 LLM 경계와 **같은 개수**를 뽑으므로 정밀도=재현율=F1 이다.", ""]

    L += ["## 일치율 (정확 일치 / ±1 어절)", "",
          "| 인코더 | 점수 | 정확 | ±1 |", "|---|---|---|---|"]
    for r in result["encoders"]:
        for k in result["score_keys"]:
            a0 = r["agreement"][k]["hit_rate"]
            a1 = r["agreement_tol1"][k]["hit_rate"]
            L.append(f"| {r['encoder']} | {k} | {a0} | {a1} |")
    L += ["", "| 비교군 | 정확 | ±1 |", "|---|---|---|",
          f"| 무작위 위치 | {result['chance']['hit_rate']} | {result['chance_tol1']['hit_rate']} |",
          f"| 기계적 등분 | {result['mechanical']['hit_rate']} | {result['mechanical_tol1']['hit_rate']} |"]

    L += ["", "## 분포 전체 — 경계에서 실제로 급변하는가", "",
          "상위 k 개만 보는 일치율은 '거의 맞았다' 를 못 본다. `AUC` 는 무작위로 고른 "
          "(경계, 비경계) 한 쌍에서 경계 쪽 점수가 높을 확률이다. **0.5 = 무정보.**", "",
          "| 인코더 | 점수 | AUC | 경계−비경계 평균차 |", "|---|---|---|---|"]
    for r in result["encoders"]:
        for k in result["score_keys"]:
            ra = r["rank_alignment"][k]
            L.append(f"| {r['encoder']} | {k} | {ra['auc']} | {ra['mean_gap']} |")

    # 판정의 기준선은 무작위가 아니라 **같은 점수의 위치 사전확률**이다.
    bases = ["delta_prefix", "tile", "ctx_delta"]
    rows = []
    for r in result["encoders"]:
        for b in bases:
            raw = r["rank_alignment"][b]["auc"] or 0.5
            pri = r["rank_alignment"][b + "_prior"]["auc"] or 0.5
            res = r["rank_alignment"][b + "_resid"]["auc"] or 0.5
            rows.append((r["encoder"], b, raw, pri, res,
                         r["agreement"][b]["hit_rate"] or 0,
                         r["agreement"][b + "_prior"]["hit_rate"] or 0))
    L += ["", "## 판정 — 의미인가 위치인가", "",
          "`prior` 는 **문장 내용을 하나도 안 보고** 상대 위치의 코퍼스 평균만으로 매긴 "
          "점수다. raw 가 prior 를 못 넘으면 그 점수가 잡은 것은 의미가 아니라 위치다.", "",
          "| 인코더 | 점수 | AUC(raw) | AUC(prior) | AUC(resid) | 일치(raw) | 일치(prior) |",
          "|---|---|---|---|---|---|---|"]
    for enc, b, raw, pri, res, hr, hp in rows:
        L.append(f"| {enc} | {b} | {raw} | {pri} | {res} | {hr} | {hp} |")

    lifts = [(raw - pri) for _, _, raw, pri, _, _, _ in rows]
    best_lift = max(lifts) if lifts else 0.0
    best_res = max((res for _, _, _, _, res, _, _ in rows), default=0.5)
    L += ["", f"위치 사전확률 대비 최대 AUC 상승폭 **{best_lift:+.3f}**, "
              f"위치 보정 후 최고 AUC **{best_res:.3f}**.", ""]
    if best_lift < 0.05 and best_res < 0.55:
        L += ["**급변점이 잡은 것은 의미가 아니라 위치다.** 문장 내용을 전혀 안 쓰는 "
              "위치 사전확률이 같은 성적을 내고, 위치 교란을 빼면 AUC 가 0.5(무정보)로 "
              "내려앉는다. `delta_prefix`·`ctx_delta` 는 prefix 가 길수록 한 어절의 비중이 "
              "줄어 단조 감소하는데, 그 감소 곡선이 LLM 경계의 위치 분포와 겹쳤을 뿐이다.",
              "",
              "즉 '표현이 급변하는 곳'과 '의미 단위가 끝나는 곳'은 이 데이터에서 같은 "
              "지점이 아니다. 이 접근은 여기서 접는 것이 맞다."]
    else:
        L += ["위치만으로 설명되지 않는 상승이 있다. 다음 단계는 **끝까지 돌려 보는 것**이다 — "
              "급변점 분절을 Google 번역(운영 경로와 동일, LLM 0)으로 조각 번역해 "
              "기존 지표(`adequacy`·`contradiction`)로 LLM 프롬프트와 직접 비교한다. "
              "동의율은 대리 지표이고, 실제로 물어야 할 것은 '어느 쪽 분절이 덜 반박당하는가' 다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="표현 급변점의 분절 경계 일치도")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--encoders", nargs="+", default=["e5-large", "xlmr-large"],
                   choices=sorted(ENCODERS))
    p.add_argument("--tile-k", type=int, default=3, help="TextTiling 창 길이 (어절)")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-sentences", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = SEG_RUNS / args.run_id
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "boundary_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    sents = load_sentences(run_dir)
    if args.max_sentences:
        sents = sents[: args.max_sentences]
    if not sents:
        print(f"문장을 못 찾음: {run_dir}", file=sys.stderr)
        return 1
    n_b = sum(len(s["cuts"]) for s in sents)
    print(f"[data] 문장×T {len(sents)}개, LLM 경계 {n_b}개", flush=True)

    keys = ["delta_prefix", "delta_prefix_prior", "delta_prefix_resid",
            "tile", "tile_prior", "tile_resid",
            "ctx_delta", "ctx_delta_prior", "ctx_delta_resid"]
    result: dict = {"run_id": args.run_id, "n_units": len(sents), "n_boundaries": n_b,
                    "score_keys": keys, "encoders": [],
                    "chance": chance_rate(sents, 0), "chance_tol1": chance_rate(sents, 1),
                    "mechanical": mechanical_rate(sents, 0),
                    "mechanical_tol1": mechanical_rate(sents, 1)}

    for name in args.encoders:
        enc = ContextEncoder(ENCODERS[name], layer=args.layer, batch_size=args.batch_size)
        print(f"\n[{enc.name}] delta_prefix...", flush=True)
        score_delta_prefix(enc, sents)
        print(f"[{enc.name}] tile(k={args.tile_k})...", flush=True)
        score_tile(enc, sents, args.tile_k)
        print(f"[{enc.name}] ctx_delta...", flush=True)
        score_ctx_delta(enc, sents)
        for base in ("delta_prefix", "tile", "ctx_delta"):
            positional_prior(sents, base)
            positional_residual(sents, base)
        result["encoders"].append({
            "encoder": enc.name,
            "agreement": {k: agreement(sents, k, 0) for k in keys},
            "agreement_tol1": {k: agreement(sents, k, 1) for k in keys},
            "rank_alignment": {k: rank_alignment(sents, k) for k in keys},
        })
        enc.unload()

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
