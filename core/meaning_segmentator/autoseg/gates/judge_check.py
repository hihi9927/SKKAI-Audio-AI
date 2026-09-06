"""판정자 타당도 관문 — 루프 밖, 데이터 무관, 1회성.

`validity_check.py` 와 같은 자리다. 그쪽이 품질 백엔드를 검증하듯 이쪽은 **판정자
(모델 + 프롬프트)** 를 검증한다. 판정자 모델이나 `JUDGE_SYSTEM` 을 바꿀 때마다 돈다.

**왜 최종 지표가 아닌데도 관문이 필요한가.** 판정자는 프롬프트 개선을 조향한다.
틀리면 Critic 이 틀린 위치를 지목하고 Prompt Engineer 가 틀린 규칙을 넣는다.
지표는 틀리면 숫자로 드러나지만 **조향은 조용히 발산한다** — v1 에서 `embed` 백엔드가
부정 뒤집힘에 최고점(0.9278)을 줘서 5회 런이 무효가 된 것이 그 사례다.

통과 조건 두 개. COMET 검사보다 하나 많다 (LLM 이라 비결정적이므로).

  정확도 — 모든 변이의 다수결이 `expect` 와 일치. 오분류 0건.
  안정성 — 같은 입력 반복 실행에 판정이 동일.

둘 다 **`safe` / `not-safe` 이진**으로 본다. 판정자는 이제 점수를 안 낸다 — 모순이 가장
큰 경계에 `cause`·`shift` 를 붙여 Critic 에 넘기는 것이 전부이고, `premature` 와
`mistranslated` 는 거기서 같은 행동을 부른다. 세부 라벨은 진단으로만 남긴다.

`benign_*` 이 기준선이다. 조기 방출 자체는 죄가 아니고 뒤가 반박할 때만 문제인데,
이를 구별하지 못하는 판정자는 "짧은 조각은 다 나쁨"으로 퇴화해 루프를 보수화한다.

  PYTHONPATH=. python -m core.meaning_segmentator.autoseg.gates.judge_check --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..runtime import agents
from ..infra import gateway
from ..infra.gateway import Gateway
from ..runtime.pipeline import JsonCache
from ..paths import RUNS_DIR

_HERE = Path(__file__).resolve().parent
CASES_PATH = _HERE / "premature_cases.json"


def run_variant(judge: agents.Judge, case: dict, variant: dict, repeats: int) -> list[dict]:
    pieces_src = variant.get("pieces_src") or case["pieces_src"]
    boundary = variant.get("boundary", case.get("boundary", 0))
    out = []
    for _ in range(repeats):
        try:
            out.append(judge.judge(case["src"], case["full_translation"],
                                   pieces_src, variant["pieces_tgt"], boundary))
        except Exception as e:
            out.append({"verdict": "error", "conflict": str(e)[:200]})
    return out


def is_safe(v: str) -> bool:
    return v == "safe"


def check_nli(cases: list[dict]) -> tuple[list[dict], int, int]:
    """같은 fixture 로 **NLI contradiction 백엔드**를 검사한다.

    NLI 는 판정자와 달리 목적함수에 직접 들어간다 (`effective = adequacy × (1 − contradiction)`).
    그래서 관문이 더 중요하고, 기준은 라벨이 아니라 **순위**다 — argmax 가 `neutral` 로
    나와도 contradiction 확률이 premature > safe 이면 연속 점수로 쓸 수 있다.

    통과 조건: 모든 케이스에서 `min(premature 확률) > max(safe 확률)`.
    """
    from ..runtime import metrics
    b = metrics.make_contradiction_backend()
    rows = []
    for c in cases:
        for name, v in c["variants"].items():
            bd = v.get("boundary", c.get("boundary", 0))
            rows.append({"id": c["id"], "variant": name, "expect": v["expect"],
                         "premise": c["full_translation"],
                         "hypothesis": " ".join(v["pieces_tgt"][: bd + 1])})
    scores = b.score([r["premise"] for r in rows], [r["hypothesis"] for r in rows])
    for r, s in zip(rows, scores):
        r["contradiction"] = round(float(s), 4)

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    viol = tot = 0
    for items in by.values():
        p = [r["contradiction"] for r in items if r["expect"] == "premature"]
        s = [r["contradiction"] for r in items if r["expect"] == "safe"]
        if not p or not s:
            continue
        tot += 1
        viol += 0 if min(p) > max(s) else 1
    return rows, viol, tot


def build_report(results: list[dict], model: str, repeats: int,
                 nli: tuple | None = None) -> str:
    n_mis = sum(1 for r in results if not r["accurate"])
    n_unstable = sum(1 for r in results if not r["stable"])
    n_label_flicker = sum(1 for r in results if not r["label_stable"])
    n_label_mis = sum(1 for r in results if not r["label_accurate"])
    lines = [
        "# 판정자 타당도 검사",
        "",
        f"- 판정자 모델: `{model}`",
        f"- `JUDGE_SYSTEM` 해시: `{JsonCache.key(agents.JUDGE_SYSTEM)}`",
        f"- 케이스 {len({r['id'] for r in results})}건 × 변이 {len(results)}종 × {repeats}회 반복",
        "",
        "## 통과 조건",
        "",
        "1. **정확도** — 모든 변이의 다수결이 `expect` 와 `safe`/`not-safe` 축에서 일치",
        "2. **안정성** — 반복 실행에서 `safe` / `not-safe` 판정이 동일",
        "",
        "두 조건 모두 세부 라벨이 아니라 **safe / not-safe 이진**으로 본다. 이유: `premature` 와",
        "`mistranslated` 는 루프에서 같은 행동을 부른다 — 경계를 표시하고 `cause`·`shift` 를",
        "Critic 에 넘긴다. 판정자는 점수를 내지 않는다. 세부 라벨을 소비하는 곳이",
        "없으므로 그 축을 관문 조건으로 두면 과잉 명세다. `라벨정확`·`라벨고정` 열에 진단으로만 남긴다.",
        "",
        "판정 기준선은 `benign_*` 이다. 조기 방출 자체가 아니라 **뒤가 반박하는지**를",
        "구별하는지 보는 것이다. 이 행이 없으면 \"무조건 premature\" 라고 답하는 판정자가",
        "관문을 통과하고, 루프는 모든 경계를 문제로 보며 보수화한다.",
        "",
        "## 결과",
        "",
        "| 케이스 | 변이 | expect | 판정 | 반복 | 정확 | 안정 | 라벨정확 | 라벨고정 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['variant']} | {r['expect']} | {r['verdict']} | "
            f"{r['counts']} | {'O' if r['accurate'] else '**X**'} | "
            f"{'O' if r['stable'] else '**X**'} | "
            f"{'O' if r['label_accurate'] else '-'} | {'O' if r['label_stable'] else '-'} |")
    verdict = "**통과**" if (n_mis == 0 and n_unstable == 0) else "**탈락**"
    lines += [
        "",
        f"오분류 {n_mis}건 / 불안정 {n_unstable}건 → {verdict}"
        + (f"  (세부 라벨: 불일치 {n_label_mis}건 / 흔들림 {n_label_flicker}건 — 판정에 반영 안 함)"
           if (n_label_flicker or n_label_mis) else ""),
        "",
    ]
    if n_mis:
        lines += ["탈락이면 판정자 프롬프트를 고치고 다시 돌린다.", ""]

    if nli:
        rows, viol, tot, backend = nli
        lines += [
            "---", "",
            "# NLI contradiction 백엔드 검사",
            "",
            f"- 백엔드: `{backend}`",
            "",
            "판정자와 달리 이 값은 **목적함수에 직접 들어간다** "
            "(`effective = adequacy × (1 − contradiction)`). 그래서 기준이 라벨이 아니라",
            "**순위**다 — argmax 가 `neutral` 이어도 확률이 `premature > safe` 이면",
            "임계값 없이 연속 점수로 쓸 수 있다.",
            "",
            "**통과 조건: 모든 케이스에서 `min(premature) > max(safe)`.**",
            "",
            "| 케이스 | 변이 | expect | contradiction |",
            "|---|---|---|---|",
        ]
        for r in rows:
            lines.append(f"| {r['id']} | {r['variant']} | {r['expect']} | "
                         f"{r['contradiction']:.4f} |")
        lines += ["", f"순위 위반 {viol}/{tot} → "
                  + ("**통과**" if viol == 0 else "**탈락**"), ""]
        if viol:
            lines += ["탈락이면 NLI 체크포인트 교체를 검토한다 (`metrics.NLI_MODEL`). "
                      "전부 떨어지면 조기 방출을 목적함수에서 검출할 수단이 없으므로, "
                      "판정자 결과를 채택 게이트의 비악화 조건으로 쓰는 방어책으로 후퇴한다.", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="판정자 타당도 관문")
    p.add_argument("--model", default="gpt-5-mini")
    # 관문은 루프가 실제로 쓰는 설정으로 돌아야 한다. 루프의 판정자는
    # --agent-reasoning-effort(기본 medium)로 도므로 여기 기본값도 같게 둔다.
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["minimal", "low", "medium", "high", "none"])
    p.add_argument("--repeats", type=int, default=3, help="안정성 확인용 반복 횟수")
    p.add_argument("--budget", type=float, default=1.0)
    gateway.add_provider_args(p)
    p.add_argument("--cases", default=str(CASES_PATH))
    p.add_argument("--out", default=None, help="리포트 경로 (기본 runs/judge_validity/report.md)")
    p.add_argument("--skip-judge", action="store_true",
                   help="LLM 판정자 검사를 건너뛰고 NLI 만 검사 (API 호출 0)")
    p.add_argument("--skip-nli", action="store_true", help="NLI 검사를 건너뛴다")
    args = p.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    gw = Gateway.from_args(args, model=args.model, budget=args.budget,
                           reasoning_effort=(None if args.reasoning_effort == "none"
                                             else args.reasoning_effort))
    judge = agents.Judge(gw)

    results: list[dict] = []
    try:
        for case in ([] if args.skip_judge else cases):
            for name, variant in case["variants"].items():
                runs = run_variant(judge, case, variant, args.repeats)
                verdicts = [r.get("verdict", "error") for r in runs]
                counts = Counter(verdicts)
                majority = counts.most_common(1)[0][0]
                results.append({
                    "id": case["id"], "variant": name, "expect": variant["expect"],
                    "verdict": majority, "counts": dict(counts),
                    # 정확도·안정성 모두 safe / not-safe 축에서 본다.
                    # premature 와 mistranslated 를 소비하는 곳이 없다 — 둘 다 "이 경계를
                    # 표시하고 cause·shift 를 Critic 에 넘긴다"로 같은 행동을 부르고,
                    # 판정자는 점수를 내지 않는다. 쓰이지 않는 구별을 관문 조건으로
                    # 두면 과잉 명세다. 세부 라벨은 진단으로만 보고한다.
                    "accurate": is_safe(majority) == is_safe(variant["expect"]),
                    "label_accurate": majority == variant["expect"],
                    "stable": len({is_safe(v) for v in verdicts}) == 1,
                    "label_stable": len(counts) == 1,
                    "runs": runs,
                })
                print(f"[{case['id']}/{name}] expect={variant['expect']} "
                      f"got={majority} {dict(counts)}", flush=True)
    finally:
        gw.close()

    nli = None
    if not args.skip_nli:
        # `check_nli` 와 같은 이유로 지역 import — metrics 는 torch/COMET 을 끌어오므로
        # --skip-nli 로 도는 경로에서는 불러오지 않는다.
        from ..runtime import metrics

        rows, viol, tot = check_nli(cases)
        nli = (rows, viol, tot, metrics.NLI_MODEL)
        print(f"\n[NLI xlmr-anli] 순위 위반 {viol}/{tot}", flush=True)

    out_dir = Path(args.out).parent if args.out else (RUNS_DIR / "judge_validity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / "report.md")
    out_path.write_text(build_report(results, args.model, args.repeats, nli),
                        encoding="utf-8")
    (out_dir / "raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [r for r in results if not r["accurate"] or not r["stable"]]
    nli_failed = bool(nli and nli[1])
    print(f"\n리포트: {out_path}")
    print("탈락" if (failed or nli_failed) else "통과")
    return 1 if (failed or nli_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
