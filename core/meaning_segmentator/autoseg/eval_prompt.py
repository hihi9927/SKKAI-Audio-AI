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
from .pipeline import GoogleTranslator, JsonCache, Translator, to_lang_code

_HERE = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="프롬프트 1개를 루프와 동일 지표로 평가")
    p.add_argument("--prompt", required=True, help="평가할 프롬프트 파일")
    p.add_argument("--run-id", required=True,
                   help="기준 런 경로 (runs/ 이하). 데이터 분할·프로파일·캐시·백엔드를 재사용한다")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--label", default=None, help="결과 파일 이름에 쓸 라벨")
    p.add_argument("--tgt-lang", default=None,
                   help="타깃 언어를 기준 런과 다르게 잡는다 (다언어 런의 타깃별 표를 "
                        "뽑을 때). 캐시는 loop 과 같은 translate_{code}.json 을 쓰므로 "
                        "그 런이 이미 번역했다면 gtx 재호출이 없다")
    p.add_argument("--min-gap", type=int, default=None, help="미지정 시 기준 런에서 상속")
    p.add_argument("--model", default="gpt-5-mini")
    # 비교군도 루프와 같은 사고량으로 재야 표에 나란히 놓을 수 있다 (기준 런 config 의
    # seg_reasoning_effort 를 상속하고, 없으면 루프 기본값 low).
    # 후보 풀 하한을 곡선 격자와 **분리**한다. 검증기는 `round(어절/t_floor)−1` 개를
    # 요구하므로 이 값이 작을수록 모델이 더 많이 찍어야 한다. 순위 절단이 실제로 고를 수
    # 있으려면 후보가 채택 수보다 충분히 많아야 하는데, 기본값(=min(final_t_grid))에서는
    # 후보 7.2개 / 채택 6.2개로 폐기가 1개뿐이라 순위가 무력했다
    # (docs/RANK_METRIC_DIAGNOSIS.md §6). **프롬프트 문면의 숫자와 반드시 일치시킬 것** —
    # 어긋나면 전 문장이 too_few_tags 로 재시도돼 비용이 두 배가 된다.
    # 절단이 실제로 소비하는 순위 깊이만 요구한다. 나머지는 무번호 <SEG> 로 받아
    # 사고 토큰을 아낀다 (docs/RANK_METRIC_DIAGNOSIS.md 부록).
    # 한 콜에 넣는 문장 수. 분절 사고 토큰이 비용의 90% 라 가장 큰 레버다
    # (실측 24문장: b=1 5,237 → b=3 2,994 → b=6 1,680 사고/문장).
    p.add_argument("--batch-size", type=int, default=1,
                   help="한 분절 호출에 넣을 문장 수. 1 = 종전 동작")
    p.add_argument("--t-floor", type=int, default=None,
                   help="후보 마킹 하한 기준 T. 미지정 시 min(t_grid)")
    p.add_argument("--seg-reasoning-effort", default=None,
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="미지정 시 기준 런에서 상속")
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

    # **폴백도 LLM 이 아니라 실측이다.** 종전에는 `measured_profile.json` 이 없으면
    # `language_profile.json`(LLM)의 같은 이름 필드를 읽었는데, 그 둘은 26개 런 중
    # 25개에서 `trailing_punctuation` 이 달랐다. 이제 Profiler 는 그 필드를 내지도
    # 않는다 (agents.measured_facts). 분할 파일은 항상 있으므로 거기서 다시 잰다.
    mp = run_dir / "measured_profile.json"
    if mp.exists():
        measured = json.loads(mp.read_text(encoding="utf-8"))
    else:
        fit = (data.read_split(run_dir / "data" / "train.json")
               + data.read_split(run_dir / "data" / "dev.json"))
        measured = data.measure_profile([x.text for x in fit])
    spaced, trailing_punct = data.profile_settings(measured)

    tgt_name = args.tgt_lang or cfg.get("tgt_lang", "English")
    tgt_spaced = (target_is_spaced(tgt_name) if args.tgt_lang
                  else bool(cfg.get("tgt_spaced", target_is_spaced(tgt_name))))
    t_grid = sorted(set(args.t_grid or cfg.get("final_t_grid") or cfg.get("t_grid") or [3, 6]))

    sentences = data.read_split(run_dir / "data" / f"{args.split}.json")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    label = args.label or Path(args.prompt).stem

    gw = Gateway(model=args.model, budget=args.budget)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")

    # **번역기와 백엔드는 기준 런에서 상속한다.** 다른 조합으로 재면 같은 축이 아니다.
    tr_id = cfg.get("translator_id") or f"llm:{cfg.get('translator_model')}"
    if tr_id.startswith("google:"):
        _, code, ctx = tr_id.split(":", 2)
        if args.tgt_lang:
            code = to_lang_code(tgt_name)
            tr_id = f"google:{code}:{ctx}"
        # 다언어 런은 타깃별 캐시 파일을 쓴다. 옛 런의 단일 translate.json 도 계속 읽는다.
        cf = run_dir / "cache" / f"translate_{code}.json"
        if not cf.exists() and (run_dir / "cache" / "translate.json").exists():
            cf = run_dir / "cache" / "translate.json"
        tr_cache = JsonCache(cf)
        translator = GoogleTranslator(tgt_code=code, cache=tr_cache,
                                      workers=min(args.workers, 4),
                                      use_context=ctx.endswith("True"))
    else:
        tr_cache = JsonCache(run_dir / "cache" / "translate.json")
        translator = Translator(gw=gw, src_name=cfg["src_lang"], tgt_name=tgt_name,
                                model=tr_id.split(":", 1)[-1], cache=tr_cache,
                                workers=args.workers)

    adequacy = metrics.make_adequacy_backend(
        args.adequacy_backend or cfg.get("adequacy_backend", "cometkiwi"),
        batch_size=args.comet_batch_size)
    cons_name = args.consistency_backend or cfg.get("consistency_backend", "comet")
    nli_key = cfg.get("contradiction_backend", "xlmr-anli")
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

    seg_effort = args.seg_reasoning_effort or cfg.get("seg_reasoning_effort", "low")
    if seg_effort == "none":
        seg_effort = None

    try:
        rows, m, viol = evaluate(gw, translator, prompt, sentences, spaced, seg_cache,
                                 args.workers, adequacy, consistency, t_grid,
                                 trailing_punct, tgt_spaced=tgt_spaced,
                                 require_priority=not args.no_priority,
                                 contradiction=contradiction,
                                 reasoning_effort=seg_effort,
                                 coverage_t=(args.t_floor or cfg.get("t_floor") or cfg.get("density")
                                             or cfg.get("candidate_t")
                                             or min(t_grid)),
                                 batch_size=args.batch_size,
                                 min_gap=(args.min_gap if args.min_gap is not None
                                          else int(cfg.get("min_gap", 0))),
                                 skip_translation_below=float(
                                     cfg.get("skip_translation_below", 0.95)))
        out_dir = run_dir / "prompt_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label}_{args.split}.json").write_text(json.dumps({
            "prompt_file": str(args.prompt),
            "split": args.split,
            "t_grid": t_grid,
            "tgt_lang": tgt_name,
            "min_gap": (args.min_gap if args.min_gap is not None
                        else int(cfg.get("min_gap", 0))),
            "t_floor": (args.t_floor or cfg.get("t_floor") or cfg.get("density")
                        or cfg.get("candidate_t") or min(t_grid)),
            "batch_size": args.batch_size,
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

        print(f"[{label} / {args.split}] n={m.n} 타깃={tgt_name} "
          f"adequacy={adequacy.name} 번역기={tr_id}")
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
