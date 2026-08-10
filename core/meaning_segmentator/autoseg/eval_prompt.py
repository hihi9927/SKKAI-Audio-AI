"""임의의 프롬프트 하나를 루프와 동일한 지표로 평가한다.

용도는 두 가지다.

1. **비교군을 같은 자로 재기.** 사람이 쓴 프롬프트(`human_prompts/`)를 자동 생성
   프롬프트와 직접 맞붙인다. 사람 프롬프트는 순위 태그를 달지 않으므로 절단이
   불가능하고, 곡선 위의 **점 하나**로 평가된다 (설계 v2 §11.1).
2. **지표 검증.** 사람이 품질 순서를 아는 프롬프트들을 넣었을 때 지표가 그 순서를
   재현하지 못하면, 에이전트가 아니라 지표를 고쳐야 한다는 신호다.

  python -m core.meaning_segmentator.autoseg.eval_prompt \
      --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
      --run-id ko-en/run13 --split test --label human_current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data, metrics
from .gateway import Gateway
from .loop import _cell, evaluate, target_is_spaced
from .pipeline import GoogleTranslator, JsonCache, Translator

_HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="프롬프트 1개를 루프와 동일 지표로 평가")
    p.add_argument("--prompt", required=True, help="평가할 프롬프트 파일")
    p.add_argument("--run-id", required=True,
                   help="기준 런 경로 (runs/ 이하). 데이터 분할·프로파일·캐시·백엔드를 재사용한다")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--label", default=None, help="결과 파일 이름에 쓸 라벨")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--t-grid", type=int, nargs="+", default=None,
                   help="미지정 시 기준 런의 final_t_grid 를 그대로 쓴다")
    p.add_argument("--no-priority", action="store_true",
                   help="순위 태그를 요구하지 않는다. 사람이 쓴 비교군 프롬프트용")
    p.add_argument("--adequacy-backend", default=None, help="미지정 시 기준 런에서 상속")
    p.add_argument("--consistency-backend", default=None, help="미지정 시 기준 런에서 상속")
    p.add_argument("--no-contradiction", action="store_true",
                   help="NLI 조기 방출 검출을 끈다. 기본은 기준 런 설정 상속 — "
                        "루프와 다른 자로 재면 비교가 무의미하다")
    p.add_argument("--comet-batch-size", type=int, default=16)
    args = p.parse_args()

    run_dir = _HERE.parent / "runs" / args.run_id
    if not run_dir.exists():
        print(f"런 디렉토리 없음: {run_dir}", file=sys.stderr)
        return 1

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "language_profile.json").read_text(encoding="utf-8"))

    # 측정 프로파일이 있으면 그것이 이긴다 (루프와 같은 규칙).
    mp = run_dir / "measured_profile.json"
    if mp.exists():
        measured = json.loads(mp.read_text(encoding="utf-8"))
        spaced, trailing_punct, _ = data.reconcile_profile(measured, profile)
    else:
        spaced = bool(profile.get("uses_spaces_between_words", True))
        trailing_punct = "".join(profile.get("trailing_punctuation") or []) or None

    tgt_spaced = bool(cfg.get("tgt_spaced", target_is_spaced(cfg.get("tgt_lang", "English"))))
    t_grid = sorted(set(args.t_grid or cfg.get("final_t_grid") or cfg.get("t_grid") or [3, 6]))

    sentences = data.read_split(run_dir / "data" / f"{args.split}.json")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    label = args.label or Path(args.prompt).stem

    gw = Gateway(model=args.model, budget=args.budget)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")

    # **번역기와 백엔드는 기준 런에서 상속한다.** 다른 조합으로 재면 같은 축이 아니다.
    tr_id = cfg.get("translator_id") or f"llm:{cfg.get('translator_model')}"
    if tr_id.startswith("google:"):
        _, code, ctx = tr_id.split(":", 2)
        translator = GoogleTranslator(tgt_code=code, cache=tr_cache,
                                      workers=min(args.workers, 4),
                                      use_context=ctx.endswith("True"))
    else:
        translator = Translator(gw=gw, src_name=cfg["src_lang"], tgt_name=cfg["tgt_lang"],
                                model=tr_id.split(":", 1)[-1], cache=tr_cache,
                                workers=args.workers)

    adequacy = metrics.make_adequacy_backend(
        args.adequacy_backend or cfg.get("adequacy_backend", "cometkiwi"),
        batch_size=args.comet_batch_size)
    cons_name = args.consistency_backend or cfg.get("consistency_backend", "comet")
    nli_key = cfg.get("contradiction_backend", "deberta-mnli")
    if cons_name == "nli":
        consistency = metrics.make_backend(
            "nli", model_name=metrics.NLI_MODELS[nli_key],
            batch_size=args.comet_batch_size)
    else:
        consistency = metrics.make_backend(
            cons_name, gw=gw,
            **({"batch_size": args.comet_batch_size}
               if cons_name in metrics.COMET_CHECKPOINTS else {}))

    # 조기 방출 NLI 도 기준 런에서 상속한다 — 루프의 effective 와 같은 자로 재야
    # 비교군 표에 나란히 놓을 수 있다.
    contradiction = (None if (args.no_contradiction or cfg.get("no_contradiction"))
                     else metrics.make_contradiction_backend(nli_key))

    try:
        rows, m, viol = evaluate(gw, translator, prompt, sentences, spaced, seg_cache,
                                 args.workers, adequacy, consistency, t_grid,
                                 trailing_punct, tgt_spaced=tgt_spaced,
                                 require_priority=not args.no_priority,
                                 contradiction=contradiction)
        out_dir = run_dir / "prompt_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}_{args.split}.json").write_text(json.dumps({
            "prompt_file": str(args.prompt),
            "split": args.split,
            "t_grid": t_grid,
            "require_priority": not args.no_priority,
            "adequacy_backend": adequacy.name,
            "consistency_backend": cons_name,
            "contradiction_backend": (None if contradiction is None else contradiction.name),
            "translator": tr_id,
            "metrics": m.to_dict(),
            "violations": viol,
            "rows": rows,
            "usage": gw.usage.snapshot(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{label} / {args.split}] n={m.n} adequacy={adequacy.name} 번역기={tr_id}")
        print(f"  포맷 통과율 {m.format_pass_rate:.4f} (재시도 없이 "
              f"{m.format_pass_rate_no_retry:.4f}), 위반 {len(viol)}건")
        for k in sorted(m.by_T, key=int):
            s = m.by_T[k]
            print(f"  T={k:<3} laal {s.laal_words:6.2f}  effective {_cell(s.effective, '.4f')}  "
                  f"adequacy {s.adequacy:.4f}  contradiction {_cell(s.contradiction, '.4f')}  "
                  f"consistency {s.consistency:.4f}  k {s.chunks_per_sentence:.2f}  "
                  f"부족경계 {s.missing_boundaries:.2f}")
        print(f"  score {metrics.score(m):.4f}  비용 {gw.usage.snapshot()['cost']:.4f}")
        return 0
    finally:
        seg_cache.flush()
        tr_cache.flush()
        if isinstance(translator, GoogleTranslator):
            translator.close()
        gw.close()


if __name__ == "__main__":
    sys.exit(main())
