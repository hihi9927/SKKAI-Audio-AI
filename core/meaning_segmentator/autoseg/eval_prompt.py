"""임의의 프롬프트 하나를 루프와 동일한 지표로 평가한다.

용도는 두 가지다.

1. **사람이 쓴 프롬프트와 자동 생성 프롬프트를 같은 자로 비교.**
   `../utils/prompt.txt` 에 사람이 쓴 한국어 분절 프롬프트가 있으므로,
   자동 루프의 산출물과 직접 맞붙일 수 있다.
2. **지표 검증.** 사람이 품질 순서를 아는 프롬프트들을 넣었을 때 지표가 그 순서를
   재현하지 못하면, 에이전트를 고칠 게 아니라 지표를 고쳐야 한다는 신호다.

  python -m core.meaning_segmentator.autoseg.eval_prompt \
      --prompt core/meaning_segmentator/runs/ko-en/run01/best_prompt.txt \
      --run-id ko-en/run01 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data, metrics
from .gateway import Gateway
from .loop import evaluate
from .pipeline import GoogleTranslator, JsonCache, Translator

_HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="프롬프트 1개를 루프와 동일 지표로 평가")
    p.add_argument("--prompt", required=True, help="평가할 프롬프트 파일")
    p.add_argument("--run-id", required=True,
                   help="기준 런 경로 (runs/ 이하). 데이터 분할·프로파일·캐시를 재사용한다")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--label", default=None, help="결과 파일 이름에 쓸 라벨")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--translator-model", default="claude-sonnet-5")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quality-backend", default=None,
                   help="미지정 시 기준 런의 baseline.json 이 쓴 백엔드를 그대로 따른다")
    p.add_argument("--comet-model", default=None)
    p.add_argument("--comet-batch-size", type=int, default=16)
    args = p.parse_args()

    run_dir = _HERE.parent / "runs" / args.run_id
    if not run_dir.exists():
        print(f"런 디렉토리 없음: {run_dir}", file=sys.stderr)
        return 1

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "language_profile.json").read_text(encoding="utf-8"))
    cal = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    spaced = bool(profile.get("uses_spaces_between_words", True))
    trailing_punct = "".join(profile.get("trailing_punctuation") or []) or None
    q_floor = cal["q_floor"]

    sentences = data.read_split(run_dir / "data" / f"{args.split}.json")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    label = args.label or Path(args.prompt).stem

    gw = Gateway(model=args.model, budget=args.budget)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")

    # 번역기도 백엔드와 같이 기준 런에서 상속한다. 앵커가 번역기에 의존하므로
    # (LLM 은 비결정론적이라 상한 < 1.0, gtx 는 결정론적이라 상한 = 1.0)
    # 다른 번역기로 재면서 기준 런의 Q_floor 를 쓰면 비교가 무의미해진다.
    tr_id = cal.get("translator") or f"llm:{args.translator_model}"
    if tr_id.startswith("google:"):
        _, code, ctx = tr_id.split(":", 2)
        translator = GoogleTranslator(tgt_code=code, cache=tr_cache,
                                      workers=min(args.workers, 4),
                                      use_context=ctx.endswith("True"))
    else:
        translator = Translator(gw=gw, src_name=cfg["src_lang"], tgt_name=cfg["tgt_lang"],
                                model=tr_id.split(":", 1)[-1], cache=tr_cache,
                                workers=args.workers)

    # 비교 대상 프롬프트들은 반드시 같은 자로 재야 한다 — 기준 런의 Q_floor 를
    # 쓰면서 백엔드만 다르면 그 비교는 무의미하다.
    backend_name = args.quality_backend or cal.get("backend") or "embed"
    comet_kw = {"batch_size": args.comet_batch_size}
    if args.comet_model:
        comet_kw["model_name"] = args.comet_model
    scorer = metrics.make_backend(
        backend_name, gw=gw,
        **(comet_kw if backend_name in metrics.COMET_CHECKPOINTS else {}))

    try:
        rows, m, viol = evaluate(gw, translator, prompt, sentences, spaced,
                                 seg_cache, args.workers, scorer, trailing_punct)
        obj = metrics.objective(m, q_floor)
        out_dir = run_dir / "prompt_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}_{args.split}.json").write_text(json.dumps({
            "prompt_file": str(args.prompt),
            "split": args.split,
            "quality_backend": backend_name,
            "translator": tr_id,
            "q_floor": q_floor,
            "metrics": m.to_dict(),
            "objective": round(obj, 4),
            "violations": viol,
            "rows": rows,
            "usage": gw.usage.snapshot(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{label} / {args.split}] n={m.n} backend={backend_name} 번역기={tr_id}")
        print(f"  V  {m.valid_rate:.4f} (1차 {m.first_pass_valid_rate:.4f})")
        print(f"  Qs {m.quality_segmented:.4f} (n={m.n_segmented}, floor {q_floor}, "
              f"LCB {metrics.quality_lcb(m):.4f})")
        print(f"  Q  {m.quality:.4f} (chrF {m.quality_chrf:.4f}, p10 {m.quality_p10:.4f})")
        print(f"  L  {m.latency:.4f}  gain {m.latency_gain:.4f}")
        print(f"  k  {m.mean_segments:.2f}  분절률 {m.segmented_rate:.4f}")
        print(f"  objective {obj:.4f}")
        print(f"  비용 {gw.usage.snapshot()['cost']:.4f}")
        return 0
    finally:
        seg_cache.flush()
        tr_cache.flush()
        if isinstance(translator, GoogleTranslator):
            translator.close()
        gw.close()


if __name__ == "__main__":
    sys.exit(main())
