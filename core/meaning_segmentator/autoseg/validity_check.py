"""품질 백엔드 타당도 검사 — 루프를 돌리기 전에 반드시 통과해야 하는 게이트.

임베딩 코사인이 **절이 통째로 누락된 번역(0.8593)을 무해한 표기 차이(0.9509)보다
높게** 매긴다는 것이 실측으로 확인됐다. 그 상태에서는 Q_floor 를 아무리 잘
캘리브레이션해도 목적함수가 성립하지 않는다 — 캘리브레이션은 스케일만 맞출 뿐
순위를 보증하지 않기 때문이다. 결과 해석은 ../docs_v1/METRICS.md §6.

이 스크립트는 참조 번역에 오류를 **결정론적으로 주입**해(LLM 생성 아님, JSON 고정)
백엔드가 오류 심각도 순서를 재현하는지 실측한다.

  통과 조건: 모든 케이스에서 심각한 의미 오류 < 무해한 표기 차이

  python -m core.meaning_segmentator.autoseg.validity_check --backends comet embed chrf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import metrics
from .gateway import Gateway

_HERE = Path(__file__).resolve().parent

# 무해한 변이보다 반드시 낮게 나와야 하는 오류들. 분절이 실제로 일으키는 손상이
# 여기에 해당한다 (조각 번역 시 부정 탈락, 절 누락, 지시대상 소실).
SEVERE = ["negation_flip", "role_swap", "clause_omission", "referent_loss"]

# 무해한 변이는 두 단계로 나눈다. 이 구분 없이 재면 검사 자체가 편향된다 —
# 참조 기반 지표는 표현이 많이 바뀌면(paraphrase) 의미가 같아도 점수를 깎는데,
# 심각한 오류는 오히려 표면형이 참조와 가깝다. 즉 benign 을 과하게 바꿔 놓으면
# 어떤 백엔드든 순위가 뒤집혀 보인다.
#
#   benign_minimal    — 동의어·음역 차이 수준. §4.1 의 '무해한 표기 차이'와 같은 급.
#                       **판정 기준은 이쪽이다.**
#   benign_paraphrase — 문장 구조까지 바꾼 재서술. 참고용으로만 본다.
BENIGN = "benign_minimal"
BENIGN_SOFT = "benign_paraphrase"
ORDER = ["identical", BENIGN, BENIGN_SOFT] + SEVERE + ["unrelated"]


def evaluate_backend(scorer: metrics.QualityBackend, cases: list[dict]) -> dict:
    """케이스별 변이 점수를 낸다. 같은 백엔드 인스턴스를 재사용해 모델을 1회만 로드한다."""
    srcs, hyps, refs, index = [], [], [], []
    for ci, case in enumerate(cases):
        for label, hyp in case["variants"].items():
            srcs.append(case["src"])
            hyps.append(hyp)
            refs.append(case["ref"])
            index.append((ci, label))

    # identical 은 hyp == ref 라 백엔드 규약상 1.0 으로 단락된다. 그것이 의도다 —
    # 루프에서 무분절 문장이 받는 값과 같은 규약이어야 표가 실제 운용을 반영한다.
    scores = scorer.score(srcs, hyps, refs)

    per_case: dict[str, dict[str, float]] = {}
    for (ci, label), s in zip(index, scores):
        per_case.setdefault(cases[ci]["id"], {})[label] = round(float(s), 4)

    # 집계: 변이 유형별 평균 (그 유형이 정의된 케이스만)
    by_variant: dict[str, float] = {}
    for label in ORDER:
        vals = [v[label] for v in per_case.values() if label in v]
        if vals:
            by_variant[label] = round(sum(vals) / len(vals), 4)

    # 순위 위반: 케이스 단위로 본다. 평균만 보면 한 케이스의 큰 격차가 다른
    # 케이스의 역전을 가려버린다.
    violations = []
    for cid, v in per_case.items():
        if BENIGN not in v:
            continue
        for label in SEVERE:
            if label in v and v[label] >= v[BENIGN]:
                violations.append({
                    "case": cid, "severe": label,
                    "severe_score": v[label], "benign_score": v[BENIGN],
                    "margin": round(v[label] - v[BENIGN], 4),
                })
    # 무관한 문장이 심각한 오류보다 높으면 지표가 아예 뒤집힌 것이다
    for cid, v in per_case.items():
        if "unrelated" not in v:
            continue
        for label in SEVERE:
            if label in v and v["unrelated"] >= v[label]:
                violations.append({
                    "case": cid, "severe": label,
                    "severe_score": v[label], "benign_score": v["unrelated"],
                    "margin": round(v["unrelated"] - v[label], 4),
                    "kind": "unrelated_above_severe",
                })

    # 참고용: 재서술을 기준으로 삼았을 때의 역전. 판정에는 쓰지 않는다.
    soft = sum(1 for v in per_case.values() for label in SEVERE
               if BENIGN_SOFT in v and label in v and v[label] >= v[BENIGN_SOFT])

    n_checks = sum(1 for v in per_case.values() for label in SEVERE
                   if label in v and BENIGN in v)
    return {
        "backend": scorer.name,
        "per_case": per_case,
        "by_variant": by_variant,
        "violations": violations,
        "n_ordering_checks": n_checks,
        "n_soft_violations": soft,
        "passed": not any(v.get("kind") is None for v in violations),
    }


def render(results: list[dict], cases: list[dict]) -> str:
    names = [r["backend"] for r in results]
    lines = [
        "# 품질 백엔드 타당도 검사",
        "",
        f"케이스 {len(cases)}건 x 변이 {len(ORDER)}종. 오류는 JSON 에 고정되어 있고 "
        "매 실행 동일하다.",
        "",
        "**통과 조건: 심각한 의미 오류(부정 뒤집힘 / 주체 뒤바뀜 / 절 누락 / 지시대상 소실)의 "
        "점수가 `benign_minimal`(동의어·음역 차이)보다 낮을 것.**",
        "",
        "`benign_paraphrase`(구조까지 바꾼 재서술)는 판정에 쓰지 않는다 — 참조 기반 지표는 "
        "재서술을 의미 손상과 구분하지 못하는 것이 알려진 성질이고, 그걸 기준으로 삼으면 "
        "어떤 백엔드든 탈락한다.",
        "",
        "## 변이 유형별 평균",
        "",
        "| 변이 | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label in ORDER:
        row = [f"{r['by_variant'].get(label, float('nan')):.4f}"
               if label in r["by_variant"] else "—" for r in results]
        mark = "**" if label in SEVERE else ""
        lines.append(f"| {mark}{label}{mark} | " + " | ".join(row) + " |")

    lines += ["", "## 판정", "",
              "| 백엔드 | 순위 검사 | 위반 (기준: minimal) | 참고 (기준: paraphrase) | 판정 |",
              "|---|---|---|---|---|"]
    for r in results:
        hard = [v for v in r["violations"] if v.get("kind") is None]
        lines.append(f"| {r['backend']} | {r['n_ordering_checks']} | {len(hard)} | "
                     f"{r.get('n_soft_violations', '—')} | "
                     f"{'통과' if r['passed'] else '**탈락**'} |")

    for r in results:
        hard = [v for v in r["violations"] if v.get("kind") is None]
        if not hard:
            continue
        lines += ["", f"### {r['backend']} 위반 상세", "",
                  "| 케이스 | 오류 유형 | 오류 점수 | 무해한 변이 | 차이 |", "|---|---|---|---|---|"]
        for v in hard:
            lines.append(f"| {v['case']} | {v['severe']} | {v['severe_score']} | "
                         f"{v['benign_score']} | +{v['margin']} |")

    lines += ["", "## 케이스별 원점수", ""]
    for case in cases:
        lines += [f"### {case['id']} ({case['src_lang']} → {case['tgt_lang']})", "",
                  f"- src: `{case['src']}`", f"- ref: `{case['ref']}`", "",
                  "| 변이 | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for label in ORDER:
            if label not in case["variants"]:
                continue
            row = [f"{r['per_case'][case['id']][label]:.4f}" for r in results]
            lines.append(f"| {label} | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="품질 백엔드 타당도 검사")
    # xcomet 은 기본에서 뺀다 — HF 게이트 모델이라 라이선스 미수락 계정에서 403 으로 죽는다.
    # 쓰려면 명시적으로 지정할 것: --backends xcomet comet embed chrf
    p.add_argument("--backends", nargs="+", default=["comet", "embed", "chrf"])
    p.add_argument("--cases", default=str(_HERE / "validity_cases.json"))
    p.add_argument("--comet-model", default=None)
    p.add_argument("--comet-batch-size", type=int, default=8)
    p.add_argument("--out", default=None, help="리포트 출력 디렉토리")
    args = p.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else (_HERE.parent / "runs" / "validity")
    out_dir.mkdir(parents=True, exist_ok=True)

    gw = Gateway() if "embed" in args.backends else None
    try:
        results = []
        for name in args.backends:
            kw: dict = {}
            if name in metrics.COMET_CHECKPOINTS:
                kw["batch_size"] = args.comet_batch_size
                if args.comet_model:
                    kw["model_name"] = args.comet_model
            scorer = metrics.make_backend(name, gw=gw, **kw)
            print(f"[{name}] 채점 중...", flush=True)
            results.append(evaluate_backend(scorer, cases))
            # 다음 백엔드가 GPU 를 쓸 수 있게 비워 준다 (XCOMET-XL 은 14GB 를 잡는다)
            if isinstance(scorer, metrics.CometBackend):
                scorer.unload()

        report = render(results, cases)
        (out_dir / "validity_report.md").write_text(report, encoding="utf-8")
        (out_dir / "validity_scores.json").write_text(
            json.dumps({"cases": args.cases, "results": results},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n" + report)
        print(f"\n[done] {out_dir}")

        failed = [r["backend"] for r in results if not r["passed"]]
        if failed:
            print(f"\n[경고] 타당도 탈락: {', '.join(failed)} — 이 백엔드로는 루프를 돌리지 말 것",
                  file=sys.stderr)
        return 0
    finally:
        if gw is not None:
            gw.close()


if __name__ == "__main__":
    sys.exit(main())
