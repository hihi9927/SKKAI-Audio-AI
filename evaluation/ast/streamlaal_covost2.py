#!/usr/bin/env python3
"""CoVoST2(단문)를 **ACL 60/60 과 같은 지표**로 재채점 — StreamLAAL v2.

    .venv-streamlaal/bin/python evaluation/ast/streamlaal_covost2.py \\
        --axes static:20260825_134119 static-c4:20260826_030555 seg:20260825_134119 \\
               static-c6:20260826_035304 punct:20260825_134119

왜 다시 재나
------------
CoVoST2 는 우리 `metrics_ast.compute_laal`(`laal_ms`)로, ACL 60/60 은 공식 구현의
StreamLAAL 로 냈다. 두 값은 정의의 뿌리가 같지만 **한 표에 섞으면 안 된다**:

  · `d_i` 의 기준점 — CoVoST2 는 발화 시작(0초), StreamLAAL 은 **문장 시작**
  · 소스 길이 상한 — CoVoST2 는 `--laal-cap-source` 로 지연을 발화 길이에서 잘랐고
    (실측 seg/de: 2,645ms capped vs 2,882ms uncapped), StreamLAAL 은 자르지 않고
    τ 절단으로 처리한다
  · 집계 단위 — 발화 평균 vs (재분절된) 문장 평균

여기서 두 데이터셋을 같은 자로 다시 재서, 표를 가로질러 읽을 수 있게 만든다.
**재실행이 아니다** — 저장된 `metric.json` 만 읽으므로 번역 비용도 GPU 도 안 든다.

재분절을 왜 건너뛰나
--------------------
CoVoST2 는 발화 하나 = 참조 문장 하나다. mwerSegmenter 에 참조를 1개만 주면 가설을
쪼갤 자리가 없어 **통째로 돌려준다** — 실측으로 확인했다(seg/de 40건, 40/40 항등).
그래서 3,000발화 × 15조건 = 45,000번의 mweralign 호출을 항등 연산에 쓰지 않는다.
대신 어댑터가 만든 `OutputWithDelays` 를 그대로 한 조각짜리 재분절 결과로 넘긴다.

`|X_s|` 는 gold 발화 길이(`src_duration_ms`)를 쓴다. 하네스가 뒤에 붙인 침묵은
포함하지 않는다 — 그건 우리가 만든 것이지 소스가 아니다.

가설이 빈 발화는 공식 구현이 평균에서 제외한다(`skipped_sentences`). 그 비율을
`null_rate` 로 반드시 함께 보고할 것 — 어려운 발화를 못 내면 지연이 좋아 보인다.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from simulstream.metrics.readers import (  # noqa: E402
    OutputWithDelays, ReferenceSentenceDefinition)
from simulstream.metrics.scorers.latency.mwersegmenter import (  # noqa: E402
    ResegmentedLatencyScoringSample)
from simulstream.metrics.scorers.latency.stream_laal import StreamLaal  # noqa: E402

from streamlaal_adapter import (  # noqa: E402
    Diagnostics, build_output_with_delays)
from score_acl6060 import commits_from_row  # noqa: E402

LANG_UNIT = {"de": "word", "ja": "char", "zh": "char"}


def score_one(run_dir: Path, lang: str) -> dict:
    unit = LANG_UNIT[lang]
    scorer = StreamLaal(argparse.Namespace(latency_unit=unit))
    metric = json.loads((run_dir / "metric.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    scfg = meta.get("server_config") or {}
    axis_server, axis_client = scfg.get("axis"), meta["args"]["model"]
    if axis_server and axis_server != axis_client.split("-c")[0]:
        raise SystemExit(f"!! 축 라벨 불일치: 서버 {axis_server!r} vs 클라 {axis_client!r}\n"
                         f"   {run_dir}")

    diag = Diagnostics()
    samples, n_null, n_neg = [], 0, 0
    for row in metric["rows"]:
        dur = float(row.get("src_duration_ms") or 0.0) / 1000.0
        ref = ReferenceSentenceDefinition(
            content=row.get("ref_text") or "", start_time=0.0, duration=dur)
        d = Diagnostics()
        owd = build_output_with_delays(commits_from_row(row), unit, d)
        for f in ("n_commits", "n_empty_commits_dropped", "n_units",
                  "n_violation_intra", "n_violation_boundary"):
            setattr(diag, f, getattr(diag, f) + getattr(d, f))
        if not owd.ideal_delays or dur <= 0:
            n_null += 1
            # 빈 가설도 표본에 남긴다 — 공식 구현이 평균에서 빼되 개수는 센다.
            samples.append(ResegmentedLatencyScoringSample(
                row["utt_id"], [OutputWithDelays("", [], [])], [ref]))
            continue
        n_neg += sum(1 for x in owd.ideal_delays if x < 0)
        # 참조 1문장 → 재분절은 항등(모듈 docstring 참고). 조각 하나로 그대로 넘긴다.
        samples.append(ResegmentedLatencyScoringSample(row["utt_id"], [owd], [ref]))

    warn = io.StringIO()
    with contextlib.redirect_stderr(warn):
        scores = scorer._do_score(samples)

    n = len(metric["rows"])
    return {
        "run_dir": str(run_dir), "axis": axis_client, "lang": lang, "latency_unit": unit,
        "stream_laal_sec": round(scores.ideal_latency, 4),
        "stream_laal_ca_sec": round(scores.computational_aware_latency, 4),
        "n_utts": n, "n_null": n_null,
        "null_rate": round(n_null / n, 4) if n else None,
        "legacy_laal_ms": metric["summary"].get("laal_ms"),
        "legacy_laal_ca_ms": metric["summary"].get("laal_ca_ms"),
        "diagnostics": {
            "n_negative_delays": n_neg,
            "n_violation_intra": diag.n_violation_intra,
            "n_violation_boundary": diag.n_violation_boundary,
            "n_commits": diag.n_commits, "n_units": diag.n_units,
            "n_empty_commits_dropped": diag.n_empty_commits_dropped,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="CoVoST2 StreamLAAL 재채점")
    p.add_argument("--results-root", default=str(HERE / "results" / "CoVoST2"))
    p.add_argument("--scope-prefix", default="n3000")
    p.add_argument("--axes", nargs="+", required=True, help="축:태그 (예: seg:20260825_134119)")
    p.add_argument("--langs", nargs="+", default=["de", "ja", "zh"])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.results_root).expanduser().resolve()
    results = []
    for spec in a.axes:
        ax, _, tag = spec.partition(":")
        if not tag:
            print(f"!! '축:태그' 형식이 아닙니다: {spec}"); return 2
        for lg in a.langs:
            run = root / ax / f"{a.scope_prefix}-{lg}" / tag
            if not (run / "metric.json").exists():
                print(f"[건너뜀] 없음: {run}"); continue
            r = score_one(run, lg)
            results.append(r)
            d = r["diagnostics"]
            print(f"── {ax}/{lg}  StreamLAAL {r['stream_laal_sec']:7.3f}s  "
                  f"CA {r['stream_laal_ca_sec']:7.3f}s  "
                  f"(기존 LAAL {r['legacy_laal_ms']:.0f}ms)")
            print(f"   null {r['n_null']}/{r['n_utts']} ({r['null_rate']*100:.1f}%) | "
                  f"음수 {d['n_negative_delays']} | 조각내부위반 {d['n_violation_intra']} | "
                  f"단위 {d['n_units']}")
            if d["n_violation_intra"]:
                print("   !! 조각 내부 단조 위반 — 펼치기가 새고 있다")
    if not results:
        print("채점할 결과가 없습니다."); return 2

    out = Path(a.out) if a.out else root / "streamlaal_covost2.json"
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n=== StreamLAAL (초) — 기존 LAAL(ms) 병기 ===")
    print(f"{'축':11s} " + " ".join(f"{lg:>18s}" for lg in a.langs))
    for spec in a.axes:
        ax = spec.split(":")[0]
        cells = []
        for lg in a.langs:
            m = [r for r in results if r["axis"] == ax and r["lang"] == lg]
            cells.append(f"{m[0]['stream_laal_sec']:7.3f} ({m[0]['legacy_laal_ms']:5.0f})"
                         if m else f"{'-':>18s}")
        print(f"{ax:11s} " + " ".join(f"{c:>18s}" for c in cells))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
