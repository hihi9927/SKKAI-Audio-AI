"""CoVoST2 에 autoseg 프롬프트로 <SEG:n> 라벨을 단다. **라벨링 전용** — 번역·채점 안 한다.

루프의 분절 경로(`pipeline.segment_batch`)를 그대로 쓴다. 정규화·검증·재시도가 런과
동일해야 라벨이 같은 자로 나온다.

**산출물은 레포 안에 쓴다.** 종전에 `/tmp` 스크래치패드에 두었다가 머신 재부팅으로
3모델 비교 결과를 통째로 잃었다 (2026-08-31). 중간 결과를 볼 수 있도록 캐시 저장
임계치도 낮춘다.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.meaning_segmentator.autoseg import gateway
from core.meaning_segmentator.autoseg.gateway import Gateway
from core.meaning_segmentator.autoseg import pipeline as P

SEG = re.compile(r"<SEG(?::\d+)?>")
strip_seg = lambda s: re.sub(r"\s+", " ", SEG.sub(" ", s)).strip()


def unwrap(t: str) -> str:
    """문장 전체를 감싼 큰따옴표를 벗긴다.

    CoVoST2 원문 300건 중 **79건(26%)이 양끝 따옴표로 감싸여** 있는데, 분절기가 그것을
    떼고 출력해 검증기가 전부 `text_modified` 로 잡았다 (실측: 원문보존 26~31/60 이
    따옴표를 무시하면 44~46/60 으로 회복). 모델 품질이 아니라 표기 아티팩트이고, 실제
    ASR 출력에는 이 따옴표가 없으므로 도메인상으로도 벗기는 쪽이 맞다.

    **감싼 경우만 벗긴다** — 실측 79건 전부 양끝이 따옴표였고 내부에만 있는 문장은 0건.
    """
    t = t.strip()
    if len(t) > 1 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
        return t[1:-1].strip()
    return t


class LiveCache(P.JsonCache):
    """`JsonCache` 는 20건마다 저장한다. 긴 런의 중간 결과를 못 봐서 임계치를 낮춘다.

    사고 모델은 콜당 9분이 걸려 60문장에 1시간이 넘는데, 그동안 산출물이 하나도 안
    보이면 진행 상황을 판단할 수 없다 (실측: 46분 경과 시점에 캐시 파일이 아직 없었다).
    """

    def __init__(self, path: Path, every: int = 1):
        super().__init__(path)
        self._every = max(1, every)

    def put(self, k: str, v) -> None:
        with self._lock:
            self._data[k] = v
            self._dirty += 1
            if self._dirty >= self._every:
                self._flush_locked()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--manifest",
                    default="evaluation/ast/manifests/covost2_en-de_sample300.jsonl")
    ap.add_argument("--out", required=True, help="라벨 jsonl (레포 안 경로 권장)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--t-floor", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=100000.0)
    ap.add_argument("--seg-reasoning-effort", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-every", type=int, default=1,
                    help="캐시 저장 주기(건). 1이면 매 건 저장 — 중간 결과를 즉시 본다")
    # **분절 출력 상한.** 기본 SEG_MAX_TOKENS 는 32768 인데 그 값은 thinking 모델의
    # 사고 토큰까지 감당하려고 잡힌 것이다. 비사고 모델에는 과하고(폭주한 콜 하나가
    # 3.1 tok/s 로 2.9시간을 먹는다), 사고 모델에는 이 값이 필요하다.
    ap.add_argument("--max-tokens", type=int, default=2048)
    # **사고 모델은 타임아웃이 예산과 함께 움직여야 한다.** Gateway 기본 900초인데
    # 16,000 토큰을 10.8 tok/s 로 쓰면 콜 하나가 최대 24분이다. 실측에서 15분에 잘리고
    # 재시도 5회가 전부 같은 이유로 잘려 런이 통째로 죽었다 (2026-08-31, 30문장 런).
    ap.add_argument("--timeout", type=float, default=900.0)
    gateway.add_provider_args(ap)
    a = ap.parse_args()

    SPACED = True
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    for r in rows:
        r["src_text_raw"] = r["src_text"]
        r["src_text"] = unwrap(r["src_text"])
    texts = [r["src_text"] for r in rows]
    prompt = Path(a.prompt).read_text(encoding="utf-8")

    need_fn = lambda t: P.coverage_need(t, a.t_floor, SPACED, a.min_gap)
    validate_fn = lambda t, out: P.validate("", t, out, SPACED, None, True, need_fn(t))
    normalize_fn = lambda t, o: P.normalize_tags(o, SPACED, None, min_gap=a.min_gap)

    P.SEG_MAX_TOKENS = a.max_tokens
    gw = Gateway.from_args(a, model=a.model, budget=a.budget, timeout=a.timeout)
    cache = LiveCache(Path(a.cache), every=a.cache_every)
    first: list = []
    t0 = time.time()
    outs, ok = P.segment_batch(
        gw, prompt, texts, cache=cache, workers=a.workers,
        validate_fn=validate_fn, normalize_fn=normalize_fn,
        reasoning_effort=a.seg_reasoning_effort, batch_size=a.batch_size,
        first_pass_sink=first, need_fn=need_fn)
    dt = time.time() - t0

    n_ok = n_pres = 0
    out_rows = []
    for r, o, first_ok in zip(rows, outs, ok):
        viol = validate_fn(r["src_text"], o)
        pres = strip_seg(o) == strip_seg(r["src_text"])
        n_pres += pres
        n_ok += not viol
        out_rows.append({**r, "seg_text": o,
                         "n_boundaries": len(SEG.findall(o)),
                         "required": need_fn(r["src_text"]),
                         "first_pass_ok": bool(first_ok),
                         "text_preserved": pres,
                         "violations": [getattr(v, "rule", str(v)) for v in viol]})
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    u = gw.usage.snapshot()
    nb = [r["n_boundaries"] for r in out_rows]
    rq = [r["required"] for r in out_rows]
    n = len(out_rows)
    print(json.dumps({
        "model": a.model, "prompt": a.prompt, "n": n, "wall_sec": round(dt, 1),
        "format_pass": round(n_ok / n, 3),
        "first_pass": round(sum(ok) / len(ok), 3),
        "text_preserved": round(n_pres / n, 3),
        "mean_boundaries": round(sum(nb) / n, 2),
        "mean_required": round(sum(rq) / n, 2),
        "coverage_met": round(sum(1 for r in out_rows
                                  if r["n_boundaries"] >= r["required"]) / n, 3),
        "calls": u["calls"], "completion_tokens": u["completion_tokens"],
        "cost": round(u["cost"], 4),
    }, ensure_ascii=False, indent=1))
    print("LABELDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
