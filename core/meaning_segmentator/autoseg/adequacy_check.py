"""adequacy 백엔드(참조 없는 QE) 타당도 검사 — 조각 입력 관문.

CometKiwi 는 완전한 문장으로 학습된 모델인데 우리는 1~10어절 **조각**을 넣는다.
contradiction(NLI)·판정자는 오류 주입 관문을 통과시켰지만 adequacy 는 무시험이었다 —
루프 목적함수(`effective`)에 들어가는 값이므로, 조각에서 채점이 뒤집혀 있으면
루프가 조용히 엉뚱한 방향으로 간다 (v1 에서 embed 백엔드가 부정 뒤집힘에 최고점을 줘
5회 런이 무효가 된 것과 같은 구조).

  통과 조건: 케이스마다 모든 심각한 오류 변이 < `benign_minimal`(동의어 수준 변이)

케이스는 validity_cases 의 실제 KsponSpeech 발화에서 딴 조각이다. CometKiwi 는
결정론적이라 반복 실행 검사는 불필요하다 (COMET 관문과 같은 이유).

  PYTHONPATH=. python -m core.meaning_segmentator.autoseg.adequacy_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import metrics

_HERE = Path(__file__).resolve().parent

BENIGN = "benign_minimal"
NON_SEVERE = {"correct", BENIGN}


def evaluate_backend(backend, cases: list[dict]) -> dict:
    srcs, hyps, index = [], [], []
    for ci, case in enumerate(cases):
        for label, hyp in case["variants"].items():
            srcs.append(case["src"])
            hyps.append(hyp)
            index.append((ci, label))

    scores = backend.score(srcs, hyps)

    per_case: dict[str, dict[str, float]] = {}
    for (ci, label), s in zip(index, scores):
        per_case.setdefault(cases[ci]["id"], {})[label] = round(float(s), 4)

    violations = []
    for cid, v in per_case.items():
        if BENIGN not in v:
            continue
        for label, s in v.items():
            if label in NON_SEVERE:
                continue
            if s >= v[BENIGN]:
                violations.append({"case": cid, "severe": label,
                                   "severe_score": s, "benign_score": v[BENIGN],
                                   "margin": round(s - v[BENIGN], 4)})

    n_checks = sum(len([l for l in v if l not in NON_SEVERE])
                   for v in per_case.values() if BENIGN in v)
    return {"backend": backend.name, "per_case": per_case,
            "violations": violations, "n_ordering_checks": n_checks,
            "passed": not violations}


def render(result: dict, cases: list[dict]) -> str:
    lines = [
        "# adequacy 백엔드 타당도 검사 (조각 입력)",
        "",
        f"백엔드: `{result['backend']}` — 케이스 {len(cases)}건, "
        f"순위 검사 {result['n_ordering_checks']}건, 위반 {len(result['violations'])}건 → "
        + ("**통과**" if result["passed"] else "**탈락**"),
        "",
        "통과 조건: 심각한 오류 변이(의미 변경 / 부정 뒤집힘 / 원문 반환 / 무관) 점수가 "
        "`benign_minimal`(동의어 수준)보다 낮을 것.",
        "",
    ]
    if result["violations"]:
        lines += ["## 위반", "",
                  "| 케이스 | 오류 유형 | 오류 점수 | benign | 차이 |", "|---|---|---|---|---|"]
        for v in result["violations"]:
            lines.append(f"| {v['case']} | {v['severe']} | {v['severe_score']} | "
                         f"{v['benign_score']} | +{v['margin']} |")
        lines.append("")
    lines += ["## 케이스별 원점수", ""]
    for case in cases:
        v = result["per_case"][case["id"]]
        lines += [f"### {case['id']} — `{case['src']}`", "",
                  "| 변이 | 점수 |", "|---|---|"]
        for label, s in v.items():
            mark = "**" if label not in NON_SEVERE else ""
            lines.append(f"| {mark}{label}{mark} | {s:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="adequacy 백엔드 조각 입력 타당도 검사")
    p.add_argument("--backend", default="cometkiwi",
                   choices=sorted(metrics.QE_CHECKPOINTS))
    p.add_argument("--cases", default=str(_HERE / "adequacy_cases.json"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else (_HERE.parent / "runs" / "adequacy_validity")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = metrics.make_adequacy_backend(args.backend, batch_size=args.batch_size)
    print(f"[{backend.name}] 채점 중...", flush=True)
    result = evaluate_backend(backend, cases)

    report = render(result, cases)
    (out_dir / "adequacy_report.md").write_text(report, encoding="utf-8")
    (out_dir / "adequacy_scores.json").write_text(
        json.dumps({"cases": args.cases, "result": result},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"[done] {out_dir}")
    if not result["passed"]:
        print("\n[경고] adequacy 백엔드가 조각 관문에서 탈락 — 이 백엔드로 루프를 돌리지 말 것",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
