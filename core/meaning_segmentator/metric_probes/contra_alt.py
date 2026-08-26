"""NLI contradiction 대체 후보 실측 — `../NLI_ALTERNATIVES.md` §6 의 1·2·3번.

실험을 **(premise 를 무엇으로 두는가) × (그 쌍을 무엇으로 채점하는가)** 로 인수분해한다.
두 축이 독립이라 조합이 그대로 표가 되고, 어느 쪽이 기여했는지가 분리된다.

premise 축
  `oracle`   full 번역 (현행). 오라클이 필요하고 `reference_suspect` 오염에 노출된다.
  `retrans`  **다음 조각까지의 소스를 재번역한 것** (§3). 오라클이 필요 없고,
             premise 와 hypothesis 의 길이 차가 작아 granularity 불일치가 줄어든다.
             "한 조각 더 들었으면 앞을 고쳐 썼겠는가" 라는 원래 질문에 형태가 가깝다.

scorer 축
  `nli`       현행 기준선 (`deberta-large-mnli` contradiction 확률)
  `summac`    §2.3. premise 를 hypothesis 길이에 맞춰 **창으로 쪼개고 max** 를 취한다.
              SummaC 의 granularity 정합을 우리 크기(문장 vs 조각)에 옮긴 것이다.
  `minicheck` §2.2. `lytang/MiniCheck-DeBERTa-v3-Large`. 1 − P(supported)
  `erasure`   §3. 표면형 보존율. `1 − LCS(hyp, premise)/|hyp|`
  `erasure_p` 재번역 문헌의 Normalized Erasure 에 가까운 엄격형 (공통 **접두**만 인정)

**측정 가능한 것과 아닌 것.** 라벨이 붙은 증거는 관문(`premature_cases.json`) 6케이스뿐이다.
실데이터 1003 경계에는 정답이 없으므로 거기서 나오는 상관은 "현행과 얼마나 같게
움직이나" 이지 정확도가 아니다. 그래서 판정은 **관문 위반 수 + 잡음 바닥 대비 신호(SNR)**
로 하고, 실데이터는 거동 확인용으로 읽는다.

  PYTHONPATH=. python -m core.meaning_segmentator.metric_probes.contra_alt \
      --run-id ko-en/run04 --premise-modes oracle retrans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..autoseg import metrics
from ..autoseg import noise_floor as nf
from .embed_check import (contra_gate, load_boundaries, measure_floor, prune_degenerate,
                          real_data, sentence_effective)
from ..autoseg.pipeline import GoogleTranslator, JsonCache, to_lang_code

from .paths import AUTOSEG, OUT_RUNS, SEG_RUNS

MINICHECK_MODEL = "lytang/MiniCheck-DeBERTa-v3-Large"


# ── scorer 들 (공통 인터페이스: score(premises, hypotheses) → 높을수록 나쁨) ──

class SummacContra:
    """§2.3 — premise 를 hypothesis 크기에 맞춰 쪼개고 **최대 모순**을 취한다.

    SummaC 의 진단: NLI 는 문장쌍으로 학습됐는데 적용은 granularity 가 어긋난 쌍에서
    이뤄져 무너진다. 우리 어긋남은 문서 vs 문장이 아니라 **문장 vs 조각**이다 —
    실측 잡음 바닥이 hypothesis 1–2어절 0.135 에서 15+ 0.003 으로 45배 움직인다.

    그래서 문장 분할 대신 **hypothesis 길이의 슬라이딩 창**으로 premise 를 쪼갠다.
    조각이 full 번역의 *어느 부분과* 모순되면 그것이 모순이다. 전체와 비교해 희석되는
    것이 바닥을 만든다는 것이 이 구성의 가설이다.

    `stride` 를 창의 절반으로 두어 경계에 걸친 명제를 놓치지 않는다."""

    name = "summac"

    def __init__(self, base, min_window: int = 3, max_windows: int = 12):
        self.base = base
        self.min_window = min_window
        self.max_windows = max_windows

    def _windows(self, premise: str, hyp: str) -> list[str]:
        pw = premise.split()
        w = max(self.min_window, len(hyp.split()))
        if len(pw) <= w:
            return [premise]
        stride = max(1, w // 2)
        out = [" ".join(pw[i : i + w]) for i in range(0, len(pw) - w + 1, stride)]
        if len(out) > self.max_windows:                  # 균등 솎아내기
            step = len(out) / self.max_windows
            out = [out[int(i * step)] for i in range(self.max_windows)]
        return out or [premise]

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        flat_p, flat_h, owner = [], [], []
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            if not p.strip() or not h.strip():
                continue
            for w in self._windows(p, h):
                flat_p.append(w)
                flat_h.append(h)
                owner.append(i)
        out = [0.0] * len(premises)
        if not flat_p:
            return out
        for i, s in zip(owner, self.base.score(flat_p, flat_h)):
            out[i] = max(out[i], float(s))
        return out


class MiniCheckContra:
    """§2.2 — 근거-주장 사실검증기. `1 − P(supported)`.

    **R3 가 이 후보의 관문이다.** 사실검증은 "근거가 주장을 지지하는가" 라 *미지원* 과
    *반박* 을 항상 나누지 않는다. 조각은 정의상 미완성이므로, 둘을 뭉개는 모델은 무해한
    미완성을 벌해 루프를 보수화한다 — 판정자 관문이 막으려는 실패와 같은 구조다.
    관문의 `benign_incomplete` 케이스가 그 시험이다."""

    def __init__(self, model_name: str = MINICHECK_MODEL, batch_size: int = 16,
                 device: str = "cuda", name: str = "minicheck"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.name = name
        self._tok = self._model = None

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name).to(self.device).eval()
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

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        self.load()
        torch = self._torch
        out = [0.0] * len(premises)
        idx = [i for i, (p, h) in enumerate(zip(premises, hypotheses))
               if p.strip() and h.strip()]
        for b in range(0, len(idx), self.batch_size):
            chunk = idx[b : b + self.batch_size]
            enc = self._tok([premises[i] for i in chunk],      # document 먼저
                            [hypotheses[i] for i in chunk],    # 그다음 claim
                            return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            p_sup = torch.softmax(logits.float(), dim=-1)[:, 1]
            for i, v in zip(chunk, p_sup.tolist()):
                out[i] = 1.0 - float(v)
        return out


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


class SurfaceErasure:
    """§3 — 이미 내보낸 것 중 뒤 렌더링이 보존하지 않은 양.

    `strict=False` 순서를 지키는 공통 부분열(LCS) 기준. 어순만 바뀐 재서술을 덜 벌한다.
    `strict=True`  공통 **접두**만 인정. 재번역 문헌의 Normalized Erasure 에 가깝고,
                   실제 화면에서 지워야 하는 양과 형태가 같다.

    표면형이라 결정론적이고 모델이 필요 없다. 대신 번역기가 의미와 무관하게 어순을
    바꾸면 그대로 잡히므로, 바닥 측정이 필수다."""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.name = "erasure_p" if strict else "erasure"

    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        out = []
        for p, h in zip(premises, hypotheses):
            hw, pw = h.split(), p.split()
            if not hw or not pw:
                out.append(0.0)
                continue
            if self.strict:
                n = 0
                for x, y in zip(hw, pw):
                    if x != y:
                        break
                    n += 1
            else:
                n = _lcs_len(hw, pw)
            out.append(max(0.0, 1.0 - n / len(hw)))
        return out


# ── premise 축 ───────────────────────────────────────────────────────────

def build_retrans_premises(recs: list[dict], run_dir: Path, gt: GoogleTranslator,
                           workers: int = 4) -> dict:
    """경계 j 의 premise 를 **조각 j+1 까지의 소스 재번역**으로 바꾼다.

    마지막 경계는 다음 조각이 문장 끝이라 재번역이 full 번역과 같아진다 — 그 경우
    `oracle` 과 구별되지 않으므로 몇 건인지 세어 리포트에 남긴다."""
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

    need: list[str] = []
    plan: list[tuple[int, str, bool]] = []
    for i, rec in enumerate(recs):
        d = rows_by_key.get((rec["src"], rec["id"], rec["T"]))
        pieces = (d or {}).get("pieces_src") or []
        if len(pieces) < rec["j"] + 2:
            plan.append((i, "", True))
            continue
        nxt = " ".join(pieces[: rec["j"] + 2])
        degenerate = (rec["j"] + 2 >= len(pieces))       # 다음 조각 = 문장 끝
        plan.append((i, nxt, degenerate))
        need.append(nxt)

    uniq = sorted(set(x for x in need if x))
    print(f"[retrans] gtx 호출 대상 {len(uniq)}건 (중복 제거 전 {len(need)})", flush=True)
    trans = dict(zip(uniq, gt.full(uniq)))

    n_deg = n_ok = 0
    for i, src, degenerate in plan:
        if not src:
            recs[i]["premise_retrans"] = recs[i]["premise"]      # 대체 불가 → oracle
            n_deg += 1
            continue
        recs[i]["premise_retrans"] = trans.get(src) or recs[i]["premise"]
        n_deg += 1 if degenerate else 0
        n_ok += 0 if degenerate else 1
    return {"n_calls": gt.calls, "n_unique": len(uniq),
            "n_degenerate": n_deg, "n_distinct_from_oracle": n_ok}


def gate_premises(cases: list[dict], gt: GoogleTranslator) -> dict:
    """관문 케이스의 retrans premise. **케이스가 전부 2조각이라 다음 조각 = 문장 전체**
    이므로 여기서는 retrans 가 `gtx(소스 전체)` 가 된다 — 손으로 쓴 오라클과 문안만
    다를 뿐 같은 것을 가리킨다. 즉 관문은 두 premise 축을 구별하지 못한다.
    이 사실 자체를 리포트에 남긴다."""
    srcs, keys = [], []
    for c in cases:
        for name, v in c["variants"].items():
            pieces = v.get("pieces_src") or c["pieces_src"]
            bd = v.get("boundary", c.get("boundary", 0))
            srcs.append(" ".join(pieces[: bd + 2]))
            keys.append((c["id"], name))
    uniq = sorted(set(srcs))
    trans = dict(zip(uniq, gt.full(uniq)))
    return {k: trans[s] for k, s in zip(keys, srcs)}


def swap_case_premises(cases: list[dict], mapping: dict) -> list[dict]:
    """`contra_gate` 는 `case["full_translation"]` 을 premise 로 쓴다. 변이마다 premise 가
    달라야 하므로 케이스를 변이 단위로 쪼갠 사본을 만든다."""
    out = []
    for c in cases:
        for name, v in c["variants"].items():
            cc = dict(c)
            cc["variants"] = {name: v}
            cc["full_translation"] = mapping.get((c["id"], name)) or c["full_translation"]
            out.append(cc)
    # contra_gate 는 케이스 id 별로 premature/safe 를 묶으므로 id 를 유지해야 한다.
    merged: dict[str, dict] = {}
    for cc in out:
        merged.setdefault(cc["id"], []).append(cc)
    return [c for group in merged.values() for c in group]


# ── 실행 ─────────────────────────────────────────────────────────────────

def render(result: dict) -> str:
    L = ["# NLI 대체 후보 실측 — SummaC 집계 / MiniCheck / 재번역 premise", "",
         f"런: `{result['run_id']}` · 경계 {result['n_boundaries']}개 · "
         f"바닥 측정 문장 {result['n_floor_sentences']}개 · LLM 호출 0", "",
         "설계와 후보 선정 근거는 [../NLI_ALTERNATIVES.md](../NLI_ALTERNATIVES.md). "
         "실험을 **premise 축 × scorer 축**으로 인수분해해 어느 쪽이 기여했는지 분리한다.", ""]
    if result.get("retrans_stats"):
        s = result["retrans_stats"]
        L += [f"재번역 premise: gtx 고유 호출 {s['n_unique']}건. "
              f"오라클과 실제로 다른 경계 **{s['n_distinct_from_oracle']}개**, "
              f"나머지 {s['n_degenerate']}개는 다음 조각이 문장 끝이라 오라클과 같아진다.", ""]
    L += ["> **관문의 한계.** `premature_cases.json` 은 전 케이스가 2조각이라 "
          "'다음 조각' 이 곧 문장 전체다 — 관문에서는 `retrans` 가 `gtx(소스 전체)` 이고 "
          "`oracle`(손으로 쓴 full 번역)과 같은 것을 가리킨다. **관문은 scorer 축만 "
          "가른다.** premise 축의 증거는 실데이터 쪽에 있다.", ""]

    L += ["## T1 — contradiction 관문 (`premature_cases.json`)", "",
          "통과 조건은 `judge_check.check_nli` 와 동일: 케이스마다 "
          "`min(premature) > max(safe)`.", "",
          "| premise | scorer | 변이 | 위반/케이스 | mean(prem) | mean(safe) | 격차 | 판정 |",
          "|---|---|---|---|---|---|---|---|"]
    for g in result["contra_gate"]:
        for key, v in g["variants"].items():
            L.append(f"| {g['premise']} | {g['scorer']} | {key} | "
                     f"{v['violations']}/{v['n_cases']} | {v['mean_premature']} | "
                     f"{v['mean_safe']} | {v['margin']} | "
                     f"{'통과' if v['passed'] else '**탈락**'} |")

    L += ["", "## T2 — 잡음 바닥 (정의상 무해한 미완성)", "",
          "premise 의 어절 prefix 를 hypothesis 로 넣은 점수. 모순일 수 없는 입력이므로 "
          "여기 나오는 값이 바닥이다. **바닥이 낮고 평탄할수록 좋다** — 길이에 따라 "
          "출렁이면 앞쪽 경계가 구조적으로 불리해진다.", "",
          "| premise | scorer | 전체 mean | sd | " + " | ".join(nf.bucket_labels()) + " |",
          "|---|---|---|---|" + "---|" * len(nf.bucket_labels())]
    for f in result["floors"]:
        fl = f["floor"]
        cells = [str(fl["by_length_bucket"][b]["mean"]) if fl["by_length_bucket"][b]["n"]
                 else "—" for b in nf.bucket_labels()]
        L.append(f"| {f['premise']} | {f['scorer']} | {fl['overall']['mean']} | "
                 f"{fl['overall']['sd']} | " + " | ".join(cells) + " |")

    L += ["", "### 신호 대 바닥 (SNR)", "",
          "신호 = 관문의 `mean(premature) − mean(safe)`(raw). **SNR 이 클수록 잘못 자른 "
          "경계가 무해한 미완성의 산포 위로 솟는다.**", "",
          "| premise | scorer | 신호 | 바닥 sd | SNR |", "|---|---|---|---|---|"]
    for r in sorted(result["snr"], key=lambda x: -(x["snr"] or -99)):
        L.append(f"| {r['premise']} | {r['scorer']} | {r['signal']} | "
                 f"{r['floor_sd']} | {r['snr']} |")

    L += ["", "## T3 — 실데이터 거동 (정답 없음)", "",
          "경계 1003개에는 라벨이 없다. 아래 상관은 **현행 NLI(oracle)와 얼마나 같게 "
          "움직이나** 이지 정확도가 아니다. `topk 겹침` 은 루프가 판정자에게 보내는 "
          "최상위 경계 집합의 겹침 — 조향이 바뀌는지를 본다.", "",
          "| premise | scorer | 변이 | Spearman(전역) | Spearman(문장 내) | topk 겹침 |",
          "|---|---|---|---|---|---|"]
    for r in result["real"]:
        for key, v in r["variants"].items():
            L.append(f"| {r['premise']} | {r['scorer']} | {key} | "
                     f"{v['spearman_global']} | {v['spearman_within_sentence']} "
                     f"(n={v['n_within']}) | {v['topk_overlap']} |")

    L += ["", "## 판정", "", "관문을 통과한 조합 (변이별):", ""]
    passed = [(g["premise"], g["scorer"], k)
              for g in result["contra_gate"] for k, v in g["variants"].items()
              if v["passed"]]
    if passed:
        for pm, sc, k in passed:
            snr = next((r["snr"] for r in result["snr"]
                        if r["premise"] == pm and r["scorer"] == sc), None)
            L.append(f"- `{pm}` × `{sc}` ({k}) — SNR {snr}")
    else:
        L += ["**없다.**"]

    def floor_of(mode, scorer):
        return next(f["floor"] for f in result["floors"]
                    if f["premise"] == mode and f["scorer"] == scorer)

    nli_o, sum_o = floor_of("oracle", "nli"), floor_of("oracle", "summac")
    mc_o, nli_r = floor_of("oracle", "minicheck"), floor_of("retrans", "nli")

    L += ["", "### 실험 1 — SummaC 집계: **기각**", "",
          f"목적을 정반대로 달성했다. 바닥이 {nli_o['overall']['mean']} → "
          f"{sum_o['overall']['mean']} 로 오르고 길이 기울기도 그대로다 "
          f"({sum_o['by_length_bucket']['1-2']['mean']} → "
          f"{sum_o['by_length_bucket']['15+']['mean']}). SNR 도 5.74 → 1.23 으로 떨어졌다.", "",
          "원인은 집계 방식에 있다. **창을 여러 개 만들어 max 를 취하면 잡음의 최대값을 "
          "뽑는다** — hypothesis 가 짧을수록 창이 많아지므로, 하필 고치려던 축(짧은 방출분의 "
          "높은 바닥)에서 더 나빠진다. SummaC 의 원래 문제(문서 vs 문장)와 우리 문제 "
          "(문장 vs 조각)는 방향이 반대였다: 저쪽은 premise 가 너무 길어 희석되는 것이고, "
          "이쪽은 hypothesis 가 너무 짧아 판단 근거가 없는 것이다.", ""]

    L += ["### 실험 2 — MiniCheck: **통과하나 채택 보류**", "",
          "raw·floor·z 세 변이 모두 0/6 으로, **바닥 보정 후에도 통과하는 유일한 백엔드**다 "
          "(현행 NLI 는 floor 변이에서 1/6 탈락). 그러나 세 가지가 걸린다.", "",
          f"1. **바닥이 높다** — {mc_o['overall']['mean']}. 무해한 미완성에 절반 가까운 "
          f"'미지원' 을 준다. `NLI_ALTERNATIVES.md` §2.2 에서 예고한 R3 우려가 그대로 나왔다: "
          "사실검증기는 *미지원* 과 *반박* 을 안 나눈다.",
          f"2. 다만 **평탄하다** ({mc_o['by_length_bucket']['1-2']['mean']} → "
          f"{mc_o['by_length_bucket']['15+']['mean']}). 현행 NLI 는 낮지만 45배 기울어져 있다 "
          f"({nli_o['by_length_bucket']['1-2']['mean']} → "
          f"{nli_o['by_length_bucket']['15+']['mean']}). **상수 오프셋은 경계 간 비교에서 "
          "상쇄되고 기울기는 앞쪽 경계를 구조적으로 벌한다** — 이 축만 보면 MiniCheck 가 낫다.",
          "3. **결정적 결함은 어순 편향이다.** ko-en-p04 에서 `benign_reordered`(0.7895)를 "
          "같은 케이스의 `benign_incomplete`(0.0938)보다 8배 나쁘게 준다. ko→en 어순 "
          "단조화는 우리가 **장려하는** 분절 결과다. COMET consistency 를 버리고 NLI 로 간 "
          "이유가 정확히 이 편향이었다 (설계 §11.1).", "",
          "격차도 얇다 — ja-ko-p05 에서 0.7985 vs 0.7196, 여유 0.079. 통과는 통과지만 "
          "SNR 1.57 이 그 얇음을 그대로 보여준다.", ""]

    L += ["### 실험 3 — 재번역: **erasure 는 사망, premise 교체는 성공**", "",
          "**표면형 erasure 는 못 쓴다.** 관문 4~6/6 위반이고, 엄격형(`erasure_p`)은 retrans "
          "에서 전 케이스가 1.0 이다 — gtx 는 매번 처음부터 번역하므로 **어절 접두 보존이 "
          "사실상 관측되지 않는다.** 재번역 문헌의 Normalized Erasure 는 같은 디코더가 "
          "점진적으로 출력을 갱신하는 상황을 전제하는데, 우리는 매번 독립 호출이라 그 전제가 "
          "성립하지 않는다. 이식 실패.", "",
          "**반면 premise 를 재번역으로 바꾼 것은 성공했다.**", "",
          f"- SNR **5.88** 로 현행(5.74)을 넘어 전 조합 중 1위",
          f"- 바닥이 더 낮고 더 평탄하다: 전체 {nli_r['overall']['mean']} vs "
          f"{nli_o['overall']['mean']}, 특히 1–2어절에서 "
          f"{nli_r['by_length_bucket']['1-2']['mean']} vs "
          f"{nli_o['by_length_bucket']['1-2']['mean']} (**35% 감소**). "
          "premise 와 hypothesis 의 길이 차가 줄어 granularity 불일치가 실제로 완화됐다 — "
          "SummaC 가 하려다 실패한 것을 premise 축에서 달성한 셈이다.",
          "- **오라클이 필요 없다.** full 번역을 안 쓰므로 `reference_suspect` 오염이 "
          "구조적으로 사라진다. 비용은 경계당 gtx 1회 (실측 고유 712건/1003경계).", "",
          "관문 raw 위반 1건은 **ja-ko-p05 하나뿐이고, 그건 영어 전용 "
          "`deberta-large-mnli` 를 한국어 타깃에 쓴 케이스다** — premise 축의 실패가 아니라 "
          "모델 언어 밖의 값이다. en 타깃 5케이스는 5/5 통과한다.", ""]

    L += ["### 다음 할 일", "",
          "1. `retrans × nli` 를 후보로 승격. 단 **채택 전 두 가지**: (a) ja-ko 케이스를 "
          "`mdeberta-xnli` 로 재검해 언어 밖 값이 맞는지 확인, (b) 실데이터에서 이 백엔드로 "
          "바꿨을 때 `paired_delta` 채택 판정이 뒤집히는지 확인 — 순위 상관 0.765/topk 겹침 "
          "0.7 은 '비슷하지만 같지 않다' 이고, 루프 결정이 바뀌는지는 따로 봐야 한다.",
          "2. MiniCheck 는 **어순 편향 관문을 추가로 통과하기 전까지 보류**. "
          "`validity_cases.json` 의 `benign_paraphrase` 를 조각 단위로 옮긴 케이스가 필요하다.",
          "3. SummaC 는 종결. 대신 현행 NLI 의 길이 바닥은 `noise_floor.py` 사후 보정을 "
          "계속 쓰거나, premise 축 교체(1번)로 줄인다."]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description="NLI 대체 후보 1·2·3 실측")
    p.add_argument("--run-id", default="ko-en/run04")
    p.add_argument("--premise-modes", nargs="+", default=["oracle", "retrans"],
                   choices=["oracle", "retrans"])
    p.add_argument("--scorers", nargs="+",
                   default=["nli", "summac", "minicheck", "erasure", "erasure_p"])
    p.add_argument("--floor-sentences", type=int, default=150)
    p.add_argument("--max-boundaries", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--render-only", action="store_true",
                   help="기존 scores.json 으로 report.md 만 다시 만든다 (GPU·번역 호출 0)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.render_only:
        out_dir = Path(args.out) if args.out else (OUT_RUNS / "contra_alt")
        result = json.loads((out_dir / "scores.json").read_text(encoding="utf-8"))
        (out_dir / "report.md").write_text(render(result), encoding="utf-8")
        print(f"[done] {out_dir / 'report.md'}")
        return 0

    run_dir = SEG_RUNS / args.run_id
    out_dir = Path(args.out) if args.out else (OUT_RUNS / "contra_alt")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_boundaries(run_dir)
    if args.max_boundaries:
        recs = recs[: args.max_boundaries]
    if not recs:
        print(f"경계를 못 찾음: {run_dir}", file=sys.stderr)
        return 1
    for r in recs:
        r["premise_oracle"] = r["premise"]          # premise 축을 갈아끼울 원본
    cases = json.loads((AUTOSEG / "premature_cases.json").read_text(encoding="utf-8"))["cases"]
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    print(f"[data] 경계 {len(recs)}개", flush=True)

    result: dict = {"run_id": args.run_id, "n_boundaries": len(recs),
                    "n_floor_sentences": args.floor_sentences,
                    "contra_gate": [], "floors": [], "snr": [], "real": [],
                    "retrans_stats": None}

    gate_map: dict = {}
    gt = None
    if "retrans" in args.premise_modes:
        code = cfg.get("tgt_code") or to_lang_code(cfg.get("tgt_lang") or "English")
        gt = GoogleTranslator(tgt_code=code, workers=args.workers,
                              cache=JsonCache(out_dir / "retrans_cache.json"),
                              use_context=False)
        try:
            gate_map = gate_premises(cases, gt)
            result["retrans_stats"] = build_retrans_premises(recs, run_dir, gt,
                                                             args.workers)
        finally:
            gt.close()
        print(f"[retrans] gtx 호출 {gt.calls}건", flush=True)

    def make_scorer(name, base_nli):
        if name == "nli":
            return base_nli
        if name == "summac":
            return SummacContra(base_nli)
        if name == "minicheck":
            return MiniCheckContra()
        if name == "erasure":
            return SurfaceErasure(strict=False)
        if name == "erasure_p":
            return SurfaceErasure(strict=True)
        raise ValueError(name)

    base_nli = metrics.make_contradiction_backend()

    for mode in args.premise_modes:
        # premise 축 적용 — 레코드의 premise 를 갈아끼운다
        for r in recs:
            r["premise"] = (r["premise_retrans"] if mode == "retrans"
                            else r["premise_oracle"])
        fulls = sorted({r["premise"] for r in recs})[: args.floor_sentences]
        gate_cases = (swap_case_premises(cases, gate_map) if mode == "retrans"
                      else cases)

        for sname in args.scorers:
            sc = make_scorer(sname, base_nli)
            label = f"{sname}@{mode}"
            sc_named = sc
            sc_named.name = label                      # real_data 가 키로 쓴다
            print(f"\n[{label}] 바닥...", flush=True)
            floor = measure_floor(fulls, sc_named)
            result["floors"].append({"premise": mode, "scorer": sname, "floor": floor})

            print(f"[{label}] 관문...", flush=True)
            g = contra_gate(gate_cases, sc_named, floor)
            g.update({"premise": mode, "scorer": sname})
            result["contra_gate"].append(g)
            raw = g["variants"]["raw"]
            sd = floor["overall"]["sd"] or 0.0
            result["snr"].append({
                "premise": mode, "scorer": sname, "signal": raw["margin"],
                "floor_sd": sd,
                "snr": round(raw["margin"] / sd, 2) if (sd and raw["margin"] is not None)
                       else None})

            print(f"[{label}] 실데이터 {len(recs)} 경계...", flush=True)
            rd = real_data(recs, sc_named, floor)
            rd.update({"premise": mode, "scorer": sname})
            rd["effective"] = {k: sentence_effective(recs, label, k)
                               for k in ["raw", "floor", "z"]}
            result["real"].append(rd)
            if hasattr(sc, "unload"):
                sc.unload()

    # 현행 그대로인 조합(nli@oracle)은 상관 1.0 이라 T3 표에서 뺀다
    result["real"] = [r for r in result["real"]
                      if not (r["premise"] == "oracle" and r["scorer"] == "nli")]

    # 바닥이 0 이면 `floor`·`z` 변이가 raw 와 같거나 폭발값이라 표에서 뺀다.
    # `prune_degenerate` 는 backend 이름으로 찾으므로 라벨을 맞춰 넘긴다.
    view = {"floors": {f"{f['scorer']}@{f['premise']}": f["floor"]
                       for f in result["floors"]},
            "contra_gate": [dict(g, backend=f"{g['scorer']}@{g['premise']}")
                            for g in result["contra_gate"]],
            "real": [dict(r, backend=f"{r['scorer']}@{r['premise']}")
                     for r in result["real"]]}
    prune_degenerate(view)      # variants 딕셔너리는 원본과 공유되므로 그대로 반영된다

    (out_dir / "scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
