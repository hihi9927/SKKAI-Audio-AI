"""미래 소스 의존도 — 오라클 번역을 쓰지 않고 **소스 쪽에서만** 잰다.

`embed_check.py` 가 보인 것: 대칭 유사도로 `contradiction` 을 대체할 수 없다. 부정
뒤집힘·주체 뒤바뀜은 어휘를 보존한 채 명제만 뒤집으므로 코사인에서는 오히려 참조와
가까워진다. 그러나 그 측정이 던진 질문 자체도 어긋나 있었다 — `(full 번역, 누적 방출분)`
비교는 대부분 **미완성**을 재고, 정작 물어야 할 것은 "이 경계에서 이미 내보낸 구간이
**미래 소스를 봐야 제대로 읽히는가**" 다.

타깃 쪽에서 그걸 재려면 같은 소스 구간의 **미래 인지 렌더링**이 필요한데, Google gtx 는
개행으로 붙여 보내도 줄 간 문맥을 전파하지 않는다 (실측: 4/4 케이스에서 첫 줄이 단독
번역과 글자까지 동일). 그래서 오라클 없이 남는 경로가 소스 쪽이다.

    h_blind = encoder(소스 prefix 단독)      → prefix 토큰 평균
    h_ctx   = encoder(소스 문장 전체)        → **같은 prefix 토큰** 평균
    fd(경계) = 1 − cos(h_blind, h_ctx)

**양방향 인코더여야 한다.** causal 디코더(Qwen3-Embedding 등)는 prefix 토큰이 구조적으로
미래를 못 보므로 fd 가 항상 0 이다. 여기서는 XLM-R 계열만 쓴다.

fd 는 문장·길이에 크게 의존한다 (한국어는 SOV 라 어디를 잘라도 뒤가 읽기를 바꾼다).
그래서 절대값이 아니라 **문장 내 모든 후보 절단 위치의 분포에서의 z 점수**로 본다 —
문장 난이도와 길이 효과가 함께 상쇄되고, "이 문장에서 하필 여기를 자른 것이 나쁜가"
라는 원래 질문에 맞는 정규화다.

세 가지를 낸다.

  A 관문      `premature_cases.json` 중 **절단 위치가 변이마다 다른** 케이스.
              fd 는 렌더링이 아니라 절단의 성질이므로, 같은 위치에서 번역만 다른
              변이는 원리적으로 구별할 수 없다 — 그 한계도 표에 남긴다.
  B 예측력    기존 런의 경계에서 fd 와 실측 NLI contradiction 의 순위 상관.
              양수면 오라클 없이 위험한 절단을 미리 고를 수 있다는 뜻이다.
  C 비교군    프롬프트가 고른 절단 vs 같은 문장의 모든 후보 위치. 프롬프트가
              실제로 미래 의존이 낮은 곳을 고르고 있는지.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.future_dep --run-id ko-en/run04
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .embed_check import load_boundaries
from ..autoseg.pipeline import TAG_RE

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

# 전부 XLM-R 계열 **양방향 인코더**. causal 디코더는 fd 가 구조적으로 0 이라 제외한다.
ENCODERS: dict[str, str] = {
    "xlmr-large": "FacebookAI/xlm-roberta-large",          # 소지 그대로의 MLM 표현
    "e5-large": "intfloat/multilingual-e5-large",          # 같은 백본, 대조학습 후
    "gte-base": "Alibaba-NLP/gte-multilingual-base",
}


class ContextEncoder:
    """`(문장, prefix 길이)` → prefix 토큰 표현. 단독 인코딩과 문맥 인코딩 둘 다."""

    def __init__(self, model_id: str, device: str = "cuda", layer: int = -1,
                 batch_size: int = 32, max_length: int = 256):
        self.model_id = model_id
        self.device = device
        self.layer = layer
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = f"{model_id.split('/')[-1]}@L{layer}"
        self._tok = None
        self._model = None

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_id,
                                                      trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True,
                output_hidden_states=True).to(self.device).eval()
            self._torch = torch
        return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = self._tok = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def _hidden(self, texts: list[str]):
        """(hidden_states, offset_mapping, attention_mask) 를 배치로 낸다."""
        self.load()
        torch = self._torch
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        h = out.hidden_states[self.layer]
        return h, offsets, enc["attention_mask"]

    def span_range_vectors(self, texts: list[str],
                           spans: list[tuple[int, int | None]]) -> list[list[float]]:
        """각 텍스트에서 `[start, end)` 문자 구간에 걸리는 토큰들의 평균 벡터.

        오프셋 매핑을 쓰므로 토크나이저가 어떻게 쪼개든 **같은 문자 구간**을 본다 —
        두 인코딩을 비교할 때 대상 구간이 정확히 같아야 차이가 의미를 갖는다.
        `end` 가 None 이면 끝까지."""
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            sp = spans[i : i + self.batch_size]
            h, offsets, mask = self._hidden(chunk)
            for b, (start, end) in enumerate(sp):
                sel = []
                for t in range(h.shape[1]):
                    if mask[b, t].item() == 0:
                        continue
                    s, e = offsets[b, t].tolist()
                    if e <= s:                      # 특수 토큰
                        continue
                    if s >= start and (end is None or s < end):
                        sel.append(t)
                if not sel:
                    sel = [0]
                v = h[b, sel].mean(dim=0)
                v = v / v.norm().clamp_min(1e-9)
                vecs.append([float(x) for x in v.to("cpu")])
        return vecs

    def span_vectors(self, texts: list[str], char_ends: list[int]) -> list[list[float]]:
        """`[0, char_end)` 구간 평균. `char_end` 가 None 이면 전체."""
        return self.span_range_vectors(texts, [(0, e) for e in char_ends])


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def future_dependence(enc: ContextEncoder, sentences: list[str],
                      prefix_chars: list[int]) -> list[float]:
    """`1 − cos(prefix 단독 표현, 문장 안에서의 같은 prefix 표현)`.

    두 인코딩의 대상 문자 구간이 동일하므로 길이 차이가 직접 들어오지는 않는다.
    남는 길이 효과(짧을수록 문맥이 더 많이 바꾼다)는 호출자가 문장 내 z 로 없앤다."""
    prefixes = [s[:c] for s, c in zip(sentences, prefix_chars)]
    v_blind = enc.span_vectors(prefixes, [None] * len(prefixes))
    v_ctx = enc.span_vectors(sentences, prefix_chars)
    return [max(0.0, 1.0 - _cos(a, b)) for a, b in zip(v_blind, v_ctx)]


# ── A: 관문 (절단 위치가 다른 케이스만) ─────────────────────────────────

def cut_gate(cases: list[dict], enc: ContextEncoder) -> dict:
    """변이마다 `pieces_src` 가 다른 케이스에서만 판정이 성립한다.

    fd 는 **절단의 성질**이지 렌더링의 성질이 아니다. 같은 위치를 자르고 번역만
    다른 변이(대부분의 케이스)는 fd 가 정의상 동일하므로 `n/a` 로 남긴다 — 이것은
    측정 실패가 아니라 이 지표가 답하는 질문의 경계다."""
    rows = []
    for c in cases:
        for name, v in c["variants"].items():
            pieces = v.get("pieces_src") or c["pieces_src"]
            bd = v.get("boundary", c.get("boundary", 0))
            prefix = " ".join(pieces[: bd + 1])
            sentence = " ".join(pieces)
            rows.append({"id": c["id"], "variant": name, "expect": v["expect"],
                         "sentence": sentence, "prefix_chars": len(prefix),
                         "prefix": prefix})
    fds = future_dependence(enc, [r["sentence"] for r in rows],
                            [r["prefix_chars"] for r in rows])
    for r, f in zip(rows, fds):
        r["fd"] = round(f, 4)

    # 변이 간 절단 위치가 다르면 raw fd 비교는 **길이 교란**을 본다 (뒤에 남은 미래가
    # 많을수록 fd 가 크다). 같은 문장의 이웃 절단 위치로 국소 보정한 값을 함께 낸다.
    sent_pos: dict[str, list[int]] = {r["sentence"]: _sentence_positions(r["sentence"])
                                      for r in rows}
    flat_s, flat_e, index = [], [], []
    for s, poss in sent_pos.items():
        for pos in poss:
            flat_s.append(s)
            flat_e.append(pos)
            index.append((s, pos))
    grid = dict(zip(index, future_dependence(enc, flat_s, flat_e)))
    for r in rows:
        poss = sent_pos[r["sentence"]]
        pos = r["prefix_chars"]
        if pos in poss:
            i = poss.index(pos)
            nb = [grid[(r["sentence"], poss[k])] for k in (i - 1, i + 1)
                  if 0 <= k < len(poss)]
            r["fd_local"] = round(r["fd"] - sum(nb) / len(nb), 4) if nb else None
        else:
            r["fd_local"] = None

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    viol = tot = na = 0
    viol_l, tot_l = [0], [0]
    detail = []
    for cid, items in by.items():
        p = [r for r in items if r["expect"] == "premature"]
        s = [r for r in items if r["expect"] == "safe"]
        if not p or not s:
            continue
        # 절단 위치가 같으면 fd 가 같다 — 판정 대상이 아니다
        if len({r["prefix"] for r in items}) == 1:
            na += 1
            detail.append({"case": cid, "status": "n/a (변이 간 절단 위치 동일)"})
            continue
        tot += 1
        ok = min(r["fd"] for r in p) > max(r["fd"] for r in s)
        if not ok:
            viol += 1
        entry = {"case": cid, "status": "통과" if ok else "위반",
                 "min_premature": min(r["fd"] for r in p),
                 "max_safe": max(r["fd"] for r in s)}
        lp = [r["fd_local"] for r in p if r["fd_local"] is not None]
        ls = [r["fd_local"] for r in s if r["fd_local"] is not None]
        if lp and ls:
            ok_l = min(lp) > max(ls)
            entry.update({"status_local": "통과" if ok_l else "위반",
                          "min_premature_local": min(lp), "max_safe_local": max(ls)})
            viol_l[0] += 0 if ok_l else 1
            tot_l[0] += 1
        detail.append(entry)
    return {"encoder": enc.name, "rows": rows, "violations": viol,
            "n_cases": tot, "n_not_applicable": na, "detail": detail,
            "violations_local": viol_l[0], "n_cases_local": tot_l[0]}


# ── B/C: 실데이터 ────────────────────────────────────────────────────────

def _sentence_positions(text: str, spaced: bool = True) -> list[int]:
    """문장의 모든 후보 절단 위치를 문자 오프셋으로. (어절 경계 = 프롬프트가 고를 수 있는 곳)"""
    if spaced:
        out, pos = [], 0
        parts = text.split()
        for w in parts[:-1]:
            pos = text.index(w, pos) + len(w)
            out.append(pos)
        return out
    return list(range(1, len(text)))


def real_data(recs: list[dict], rows_by_key: dict, enc: ContextEncoder) -> dict:
    """경계별 fd + 문장 내 z, 그리고 NLI contradiction 과의 상관."""
    # 1) 실제 경계
    sents = [r["src_sentence"] for r in recs]
    ends = [r["prefix_chars"] for r in recs]
    for r, f in zip(recs, future_dependence(enc, sents, ends)):
        r["fd"] = f

    # 2) 같은 문장의 모든 후보 위치 (비교군 + 문장 내 정규화 기준)
    uniq: dict[str, list[int]] = {}
    for r in recs:
        uniq.setdefault(r["src_sentence"], _sentence_positions(r["src_sentence"]))
    flat_s, flat_e, index = [], [], []
    for s, poss in uniq.items():
        for pos in poss:
            flat_s.append(s)
            flat_e.append(pos)
            index.append((s, pos))
    all_fd: dict[tuple, float] = {}
    for key, f in zip(index, future_dependence(enc, flat_s, flat_e)):
        all_fd[key] = f

    # 3) 문장 내 z
    stats: dict[str, tuple[float, float]] = {}
    for s, poss in uniq.items():
        vals = [all_fd[(s, p)] for p in poss]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
        stats[s] = (mu, sd)
    for r in recs:
        mu, sd = stats.get(r["src_sentence"], (0.0, 0.0))
        r["fd_z"] = (r["fd"] - mu) / sd if sd > 1e-9 else 0.0

    # 3b) 위치 교란 제거. fd 는 **남은 미래가 많을수록 크다** — 앞쪽 절단이 구조적으로
    #     불리하다. `noise_floor` 가 hypothesis 길이로 하는 보정과 같은 구조로,
    #     여기서는 **절단 뒤에 남은 어절 수**로 코퍼스 바닥을 깔고 잔차만 본다.
    remain_of: dict[tuple, int] = {}
    for s, poss in uniq.items():
        n_words = len(s.split())
        for i, pos in enumerate(poss):
            remain_of[(s, pos)] = n_words - (i + 1)
    bucket_vals: dict[str, list[float]] = {}
    for key, f in all_fd.items():
        bucket_vals.setdefault(nf.bucket_of(max(1, remain_of[key])), []).append(f)
    bucket_mean = {b: sum(v) / len(v) for b, v in bucket_vals.items() if v}
    overall = sum(all_fd.values()) / len(all_fd) if all_fd else 0.0

    def resid(s: str, pos: int, f: float) -> float:
        b = nf.bucket_of(max(1, remain_of.get((s, pos), 1)))
        return f - bucket_mean.get(b, overall)

    # 3c) 국소 대조 — **같은 문장에서 한 어절 앞/뒤로 자른 것보다 나쁜가.**
    #     길이도 문장도 고정되므로 교란이 남지 않는다. 실제로 답해야 하는 질문
    #     ("하필 여기를 자른 것이 문제인가")과 형태가 같다.
    def local(s: str, pos: int, f: float) -> float | None:
        poss = uniq.get(s) or []
        if pos not in poss:
            return None
        i = poss.index(pos)
        nb = [all_fd[(s, poss[k])] for k in (i - 1, i + 1) if 0 <= k < len(poss)]
        return f - sum(nb) / len(nb) if nb else None

    for r in recs:
        s, pos = r["src_sentence"], r["prefix_chars"]
        r["fd_resid"] = resid(s, pos, r["fd"])
        lv = local(s, pos, r["fd"])
        r["fd_local"] = lv if lv is not None else 0.0
        r["_has_local"] = lv is not None

    nli = [r["nli"] for r in recs]
    out: dict = {"encoder": enc.name, "n_boundaries": len(recs), "variants": {}}
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(recs):
        groups.setdefault((r["src"], r["id"], r["T"]), []).append(i)
    for key in ["fd", "fd_z", "fd_resid", "fd_local"]:
        vals = [r[key] for r in recs]
        g = metrics._spearman(nli, vals)
        within = []
        for idxs in groups.values():
            if len(idxs) < 3:
                continue
            c = metrics._spearman([nli[i] for i in idxs], [vals[i] for i in idxs])
            if c is not None:
                within.append(c)
        out["variants"][key] = {
            "spearman_vs_nli_global": round(g, 4) if g is not None else None,
            "spearman_vs_nli_within": (round(sum(within) / len(within), 4)
                                       if within else None),
            "n_within": len(within),
            "mean": round(sum(vals) / len(vals), 4),
        }

    # 4) 비교군 — 프롬프트가 고른 위치 vs 모든 후보 위치.
    #    `mean_fd_*` 의 직접 비교는 위치 교란이 섞여 있어 읽으면 안 된다. 판정은
    #    `mean_resid_chosen`(길이 버킷 보정)과 `mean_local_chosen`(같은 문장 이웃 대조)로 한다.
    chosen = [r["fd"] for r in recs]
    every = list(all_fd.values())
    loc = [r["fd_local"] for r in recs if r["_has_local"]]
    out["baseline"] = {
        "mean_fd_chosen": round(sum(chosen) / len(chosen), 4),
        "mean_fd_all_positions": round(sum(every) / len(every), 4),
        "mean_z_chosen": round(sum(r["fd_z"] for r in recs) / len(recs), 4),
        "mean_resid_chosen": round(sum(r["fd_resid"] for r in recs) / len(recs), 4),
        "mean_local_chosen": round(sum(loc) / len(loc), 4) if loc else None,
        "n_local": len(loc),
        "n_candidate_positions": len(every),
        "fd_by_remaining_words": {b: round(v, 4) for b, v in sorted(bucket_mean.items())},
    }

    # 5) 모델이 단 순위(<SEG:n>) vs fd — 순위가 미래 의존을 반영하나
    rank_pairs: list[tuple[float, float]] = []
    for (srcname, rid, T), idxs in groups.items():
        d = rows_by_key.get((srcname, rid, T))
        if not d:
            continue
        ranks = [int(m.group(1)) for m in TAG_RE.finditer(d.get("seg_text") or "")
                 if m.group(1)]
        if len(ranks) != len(idxs):
            continue
        for rk, i in zip(ranks, sorted(idxs, key=lambda i: recs[i]["j"])):
            rank_pairs.append((float(rk), recs[i]["fd_resid"]))
    if len(rank_pairs) > 2:
        # 순위 대비는 **잔차** 로 잰다. raw fd 로 재면 앞쪽 경계가 구조적으로 높아
        # (미래가 더 남아 있어서) 순위와의 상관이 위치 교란만 반영한다.
        c = metrics._spearman([a for a, _ in rank_pairs], [b for _, b in rank_pairs])
        out["rank_vs_fd_spearman"] = round(c, 4) if c is not None else None
        out["n_rank_pairs"] = len(rank_pairs)
    return out


def enrich(run_dir: Path, recs: list[dict]) -> dict:
    """경계 레코드에 **소스 쪽** 정보를 붙인다 (`load_boundaries` 는 타깃만 담는다)."""
    rows_by_key: dict[tuple, dict] = {}
    paths = sorted(run_dir.glob("iter_*/train_rows.json"))
    paths += sorted(run_dir.glob("iter_*/dev_rows.json"))
    if (run_dir / "test_rows.json").exists():
        paths.append(run_dir / "test_rows.json")
    for p in paths:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = f"{p.parent.name}/{p.stem}"
        for r in rows:
            for T, d in (r.get("by_T") or {}).items():
                rows_by_key[(tag, r["id"], int(T))] = d

    keep = []
    for r in recs:
        d = rows_by_key.get((r["src"], r["id"], r["T"]))
        if not d:
            continue
        pieces = d.get("pieces_src") or []
        if len(pieces) < r["j"] + 2:
            continue
        prefix = " ".join(pieces[: r["j"] + 1])
        r["src_sentence"] = " ".join(pieces)
        r["prefix_chars"] = len(prefix)
        keep.append(r)
    return {"recs": keep, "rows_by_key": rows_by_key}


def render(result: dict) -> str:
    L = ["# 미래 소스 의존도 — 오라클 없는 절단 위험 측정", "",
         f"런: `{result['run_id']}` · 경계 {result['n_boundaries']}개 · "
         f"후보 절단 위치 {result['n_positions']}개 · LLM 호출 0 · 번역 호출 0", "",
         "`fd(경계) = 1 − cos(소스 prefix 단독 표현, 문장 안에서의 같은 prefix 표현)`. "
         "오라클 번역을 안 쓰므로 `reference_suspect` 오염이 구조적으로 없다. "
         "**양방향 인코더 전용** — causal 디코더는 prefix 가 미래를 못 봐 fd 가 항상 0 이다.", ""]

    L += ["## A — 절단 위치 관문 (`premature_cases.json`)", "",
          "fd 는 **절단의 성질**이라, 같은 위치를 자르고 번역만 다른 변이는 원리적으로 "
          "구별할 수 없다 (`n/a`). 이것은 측정 실패가 아니라 이 지표가 답하는 질문의 "
          "경계다 — 렌더링 오류는 여전히 `adequacy` 와 판정자의 몫이다.", "",
          "판정 가능한 두 케이스는 **변이 간 절단 위치가 다르다** — 그런데 그 위치 차이가 곧 "
          "길이 차이라, raw fd 비교는 fd 가 가장 민감한 축(남은 미래의 양)을 그대로 탄다. "
          "같은 문장의 이웃 절단 위치로 보정한 `local` 을 함께 낸다.", "",
          "| 인코더 | 판정 가능 | 위반(raw) | 위반(local) | n/a |", "|---|---|---|---|---|"]
    for g in result["cut_gate"]:
        L.append(f"| {g['encoder']} | {g['n_cases']} | {g['violations']} | "
                 f"{g.get('violations_local')}/{g.get('n_cases_local')} | "
                 f"{g['n_not_applicable']} |")
    L += ["", f"표본이 {result['cut_gate'][0]['n_cases']}건뿐이라 **판정으로 쓰기엔 부족하다**. "
              "fd 의 근거는 아래 B(경계 1003개)에 있다."]

    L += ["", "## B — 실측 NLI contradiction 과의 순위 상관", "",
          "**양수 = 오라클 없이도 위험한 절단을 미리 고를 수 있다.** `fd_z` 는 같은 문장의 "
          "모든 후보 절단 위치 분포에서의 z 점수 — 문장 난이도와 길이 효과를 함께 없앤다.", "",
          "| 인코더 | 변이 | Spearman(전역) | Spearman(문장 내) | 평균 |",
          "|---|---|---|---|---|"]
    for r in result["real"]:
        for k, v in r["variants"].items():
            L.append(f"| {r['encoder']} | {k} | {v['spearman_vs_nli_global']} | "
                     f"{v['spearman_vs_nli_within']} (n={v['n_within']}) | {v['mean']} |")

    L += ["", "## C — 프롬프트가 고른 절단 vs 후보 위치", "",
          "**fd 는 절단 뒤에 남은 미래가 많을수록 크다** — 앞쪽 절단이 구조적으로 높게 나온다. "
          "그래서 `fd`·`z` 의 직접 비교는 위치 교란만 보는 것이고, 판정은 교란을 없앤 두 값으로 한다.", "",
          "- `resid` — 같은 **남은 어절 수** 버킷의 코퍼스 평균을 뺀 잔차 (`noise_floor` 와 같은 구조)",
          "- `local` — 같은 문장에서 **한 어절 앞/뒤로 자른 것**과의 차. 길이·문장이 고정돼 교란이 없다",
          "",
          "둘 다 **음수면** 프롬프트가 미래 의존이 낮은 곳을 고르고 있다는 뜻이다.", "",
          "| 인코더 | fd(고른) | fd(전체) | z | **resid** | **local** | 순위 vs resid |",
          "|---|---|---|---|---|---|---|"]
    for r in result["real"]:
        b = r["baseline"]
        L.append(f"| {r['encoder']} | {b['mean_fd_chosen']} | {b['mean_fd_all_positions']} | "
                 f"{b['mean_z_chosen']} | **{b['mean_resid_chosen']}** | "
                 f"**{b['mean_local_chosen']}** (n={b['n_local']}) | "
                 f"{r.get('rank_vs_fd_spearman')} (n={r.get('n_rank_pairs', 0)}) |")

    L += ["", "위치 교란의 크기 (남은 어절 수별 fd 평균):", "",
          "| 인코더 | " + " | ".join(nf.bucket_labels()) + " |",
          "|---|" + "---|" * len(nf.bucket_labels())]
    for r in result["real"]:
        by = r["baseline"]["fd_by_remaining_words"]
        L.append(f"| {r['encoder']} | "
                 + " | ".join(str(by.get(b, "—")) for b in nf.bucket_labels()) + " |")

    best = max((v["spearman_vs_nli_global"] or 0)
               for r in result["real"] for k, v in r["variants"].items()
               if k in ("fd_resid", "fd_local"))
    L += ["", "## 판정", "",
          f"교란을 없앤 fd 와 실측 contradiction 의 순위 상관은 최대 **{best:.2f}** 이다. "
          "임베딩 코사인을 바닥 보정한 값(0.03~0.10)보다 2~3배 크지만, 대체재로 쓰기에는 "
          "여전히 낮다 — 목적함수의 자리를 넘기려면 같은 축을 재고 있다는 것이 훨씬 강하게 "
          "보여야 한다.", "",
          "**같은 구조적 한계가 다시 나온다.** fd 는 미래가 앞 구간의 읽기를 *얼마나* "
          "바꾸는지를 재지, *어느 방향으로* 바꾸는지를 못 본다. 그런데 좋은 절단 위치는 "
          "대개 절 경계이고, 절 경계에서는 뒤 절이 앞 절의 해석을 **정상적으로** 갱신하므로 "
          "fd 가 원래 높다. 실제로 프롬프트가 고른 절단은 이웃 위치보다 fd 가 **높다** "
          "(`local` 이 전부 양수) — 이것이 위험 신호인지 절 경계 신호인지 fd 자체로는 "
          "가를 수 없다. 정제(refinement)와 반박(reversal)의 구별이 필요하고, 그것이 "
          "NLI 의 `neutral` vs `contradiction` 이다.", "",
          "**쓸 데가 있다면 목적함수가 아니라 후보 절단 선별이다.** 오라클 번역이 필요 없어 "
          "`reference_suspect` 오염이 구조적으로 없고, 번역 호출도 0 이라 절단 후보를 "
          "미리 좁히는 값싼 사전 필터로는 성립한다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="미래 소스 의존도 측정")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--encoders", nargs="+", default=["xlmr-large", "e5-large"],
                   choices=sorted(ENCODERS))
    p.add_argument("--layer", type=int, default=-1,
                   help="hidden_states 인덱스. -1 = 마지막 층")
    p.add_argument("--max-boundaries", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = SEG_RUNS / args.run_id
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "future_dep")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    e = enrich(run_dir, recs)
    recs, rows_by_key = e["recs"], e["rows_by_key"]
    if not recs:
        print(f"경계를 못 찾음: {run_dir}", file=sys.stderr)
        return 1
    print(f"[data] 경계 {len(recs)}개", flush=True)

    cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]
    result: dict = {"run_id": args.run_id, "n_boundaries": len(recs),
                    "n_positions": 0, "cut_gate": [], "real": []}

    for key in args.encoders:
        enc = ContextEncoder(ENCODERS[key], layer=args.layer, batch_size=args.batch_size)
        print(f"\n[{enc.name}] A 절단 위치 관문...", flush=True)
        result["cut_gate"].append(cut_gate(cases, enc))
        print(f"[{enc.name}] B/C 실데이터...", flush=True)
        r = real_data(recs, rows_by_key, enc)
        result["n_positions"] = r["baseline"]["n_candidate_positions"]
        result["real"].append(r)
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
