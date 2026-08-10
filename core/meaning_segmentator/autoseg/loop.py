"""A8 Loop Controller — 결정론적 오케스트레이터.

LLM 판단은 없다. 실행 순서, 채택/롤백, 예산, 중단 조건만 관리한다.
설계는 `../AUTO_PROMPT_LOOP_DESIGN.md`.

  python -m core.meaning_segmentator.autoseg.loop \
      --dataset kspon --src-lang Korean --tgt-lang English \
      --translator google --iterations 6 --train 30 --dev 60 --test 100
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import agents, data, metrics
from .gateway import BudgetExceeded, Gateway
from .pipeline import (GoogleTranslator, JsonCache, Translator, blocks_scoring,
                       chunk_budget, normalize_tags, segment_batch, split_segments,
                       to_lang_code, truncate, unit_count, validate)

_HERE = Path(__file__).resolve().parent

# 타깃 언어의 표기 체계 — LAAL 의 목표측 토큰 수를 세는 단위를 정한다.
_UNSPACED_TARGETS = {"japanese", "chinese", "thai", "ja", "zh", "th"}


def target_is_spaced(name: str) -> bool:
    return name.strip().lower() not in _UNSPACED_TARGETS


# ── 채점 코어 ────────────────────────────────────────────────────────────

@dataclass
class ScoredSplit:
    """분절된 텍스트 한 벌을 채점한 결과. 전부 문장 단위 리스트다 (`pieces_*` 만 중첩)."""
    joined: list[str]                    # 조각 번역 합본
    pieces_src: list[list[str]]          # 조각 원문
    pieces_tgt: list[list[str]]          # 조각 번역
    pieces_contra: list[list[float]]     # 경계별 모순 확률. 마지막 원소는 항상 0.0
    effective: list[float]
    adequacy: list[float]
    contradiction: list[float]
    consistency: list[float]
    chrf: list[float]
    laal_words: list[float]
    k: list[int]


def score_split(seg_texts: list[str], texts: list[str], full: list[str],
                translator, adequacy: metrics.AdequacyBackend,
                consistency: metrics.QualityBackend, spaced: bool, tgt_spaced: bool,
                contradiction: "metrics.ContradictionBackend | None" = None) -> ScoredSplit:
    """분절 텍스트 한 벌 -> 문장별 지표.

    루프(노브 T 로 절단한 것)와 비교군(무분절·기계분절)이 **같은 자로** 재져야 하므로
    한 함수로 둔다. 예전에는 두 벌로 복제돼 있었고, 한쪽만 고친 탓에
    `NameError: name 'eff' is not defined` 로 최종 리포트 직전에 죽은 적이 있다.

    `seg_texts` 는 `<SEG>` 가 남아 있는 채점 대상, `texts` 는 원문(consistency 의 소스),
    `full` 은 오라클 전체 번역이다.
    """
    joined, pieces = translator.seg_batch(seg_texts, full)

    # adequacy 는 **조각 단위**다. 문장 점수는 조각 길이로 가중 평균한다 —
    # 1어절 조각과 10어절 조각을 같은 무게로 세면 짧은 조각을 많이 만드는 쪽이
    # 유리해진다 (v1 의 지연 프록시가 같은 허점으로 무너졌다).
    pair_src: list[str] = []
    pair_tgt: list[str] = []
    owner: list[int] = []
    chunk_lists: list[list[str]] = []
    for i, (st, ps) in enumerate(zip(seg_texts, pieces)):
        chunks = split_segments(st)
        ps = list(ps)[: len(chunks)] + [""] * max(0, len(chunks) - len(ps))
        chunk_lists.append(chunks)
        for c, p in zip(chunks, ps):
            pair_src.append(c)
            pair_tgt.append(p)
            owner.append(i)
    qe = adequacy.score(pair_src, pair_tgt) if pair_src else []

    # 조기 방출 — 조각 j 뒤의 경계에서 "그 시점까지 방출된 누적 번역"이 오라클(full
    # 번역)과 모순되는가. 마지막 조각 뒤에는 미래가 없으므로 대상이 아니다.
    contra = [0.0] * len(pair_src)
    if contradiction is not None and pair_src:
        prem, hyp, slot = [], [], []
        pos = 0
        for i, ps in enumerate(pieces):
            k_i = len(chunk_lists[i])
            for j in range(k_i):
                if j < k_i - 1:
                    prem.append(full[i])
                    hyp.append(" ".join(x for x in list(ps)[: j + 1] if x).strip())
                    slot.append(pos)
                pos += 1
        for s, v in zip(slot, contradiction.score(prem, hyp)):
            contra[s] = v

    # 경계별 값을 문장에 되돌린다. 문장 평균만 남기면 **어느 경계가** 반박당했는지가
    # 사라지는데, 판정자·비평가 조준에 필요한 것이 바로 그 위치다 (이미 계산된 값).
    # 각 문장 마지막 원소는 뒤에 미래가 없어 항상 0.0 — "안전"이 아니라 "대상 아님".
    contra_rows: list[list[float]] = [[] for _ in texts]
    for i, v in zip(owner, contra):
        contra_rows[i].append(round(v, 4))

    effective_pair = [metrics.effective_of(q, c) for q, c in zip(qe, contra)]

    def weighted(vals):
        num = [0.0] * len(texts)
        den = [0.0] * len(texts)
        for i, c, v in zip(owner, pair_src, vals):
            w = float(max(1, unit_count(c, spaced)))
            num[i] += v * w
            den[i] += w
        return [n / d if d else 0.0 for n, d in zip(num, den)]

    return ScoredSplit(
        joined=list(joined),
        pieces_src=chunk_lists,
        pieces_tgt=[list(p) for p in pieces],
        pieces_contra=contra_rows,
        effective=weighted(effective_pair),
        adequacy=weighted(qe),
        contradiction=weighted(contra),
        consistency=consistency.score(texts, joined, full),
        chrf=[metrics.chrf(h, r) for h, r in zip(joined, full)],
        laal_words=[metrics.laal_words(st, ps, f, spaced, tgt_spaced)
                    for st, ps, f in zip(seg_texts, pieces, full)],
        k=[max(1, len(c)) for c in chunk_lists],
    )


# ── 평가 1회 ─────────────────────────────────────────────────────────────

def evaluate(
    gw: Gateway,
    translator,
    prompt: str,
    sentences: list[data.Sentence],
    spaced: bool,
    seg_cache: JsonCache,
    workers: int,
    adequacy: metrics.AdequacyBackend,
    consistency: metrics.QualityBackend,
    t_grid: list[int],
    trailing_punct: str | None = None,
    skip_translation_below: float = 0.95,
    tgt_spaced: bool = True,
    require_priority: bool = True,
    require_coverage: bool = True,
    contradiction: "metrics.ContradictionBackend | None" = None,
    coverage_t: int | None = None,
) -> tuple[list[dict], metrics.Metrics, list[dict]]:
    """분절 1회 + 노브 값마다 번역·채점.

    분절 호출은 **T 와 무관하게 한 번**이다. 순위 태그가 붙어 나오므로 이후 조각 수
    조절은 결정론적 절단이고, 곡선 전체가 추론 1회로 나온다.
    """
    texts = [s.text for s in sentences]
    # 커버리지 요건은 **가장 조인 예산**에서 온다. 그보다 적게 찍으면 그 T 에서 노브가
    # 무력해지므로, 요건이 곡선에 그릴 최소 T 를 기준으로 잡혀야 격자 전체가 의미를 갖는다.
    #
    # `coverage_t` 는 그래서 `t_grid` 와 분리돼 있다. 둘을 묶어 두면 루프(격자 3 6)와
    # 최종 평가(격자 2 3 4 6)가 **서로 다른 요건**을 쓰게 된다 — run03 에서 실제로
    # T=3 요건으로 최적화된 프롬프트가 T=2 요건으로 심판받아 test 1차 통과율이
    # 0.34 까지 떨어졌다 (train/dev 는 0.63~0.98).
    min_t = coverage_t or (min(t_grid) if t_grid else 3)
    need = lambda txt: (max(0, chunk_budget(txt, min_t, spaced) - 1)
                        if require_coverage else None)

    seg_texts, first_pass = segment_batch(
        gw, prompt, texts, cache=seg_cache, workers=workers,
        validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct,
                                            require_priority, need(t)),
        normalize_fn=lambda out: normalize_tags(out, spaced, trailing_punct),
    )

    violations: list[dict] = []
    valid_flags: list[bool] = []
    scored_flags: list[bool] = []
    for s, seg in zip(sentences, seg_texts):
        vs = validate(s.id, s.text, seg, spaced, trailing_punct, require_priority,
                      need(s.text))
        valid_flags.append(not vs)
        scored_flags.append(not blocks_scoring(vs))
        violations.extend({"id": v.id, "rule": v.rule, "detail": v.detail,
                           "text": s.text, "seg_text": seg} for v in vs)

    # 저비용 게이트는 **원문 훼손** 비율로 판단한다. 커버리지 미달로 번역을
    # 통째로 건너뛰면 개선 신호가 사라진다 — 커버리지는 채점에서 다뤄야 한다.
    rate = sum(scored_flags) / len(scored_flags) if scored_flags else 0.0
    rows = [{"id": s.id, "text": s.text, "seg_text": seg, "valid": v,
             "full_trans": None, "by_T": {}}
            for s, seg, v in zip(sentences, seg_texts, valid_flags)]

    # 포맷이 무너진 프롬프트에는 번역·채점을 쓰지 않는다 (저비용 게이트 먼저)
    if rate < skip_translation_below:
        return rows, metrics.aggregate(len(rows), valid_flags, first_pass, {}), violations

    full = translator.full(texts)
    for r, f in zip(rows, full):
        r["full_trans"] = f

    by_T: dict[str, metrics.SplitMetrics] = {}
    for T in t_grid:
        cut = [truncate(seg, T, spaced) for seg in seg_texts]
        cut_texts = [c[0] for c in cut]
        missings = [c[1] for c in cut]

        sp = score_split(cut_texts, texts, full, translator, adequacy, consistency,
                         spaced, tgt_spaced, contradiction)

        for i, r in enumerate(rows):
            r["by_T"][str(T)] = {
                "seg_text": cut_texts[i], "k": sp.k[i], "missing_boundaries": missings[i],
                "pieces_src": sp.pieces_src[i], "pieces_tgt": sp.pieces_tgt[i],
                "pieces_contra": sp.pieces_contra[i], "seg_trans": sp.joined[i],
                "effective": round(sp.effective[i], 4), "adequacy": round(sp.adequacy[i], 4),
                "contradiction": round(sp.contradiction[i], 4),
                "consistency": round(sp.consistency[i], 4),
                "chrf": round(sp.chrf[i], 4), "laal_words": round(sp.laal_words[i], 4),
            }
        # **포맷 위반 문장은 채점에서 뺀다.** 규칙을 어긴 분절의 점수는 의미가 없고,
        # 반대로 위반 1건으로 프롬프트 전체를 폐기하면 표본 잡음이 신호를 압도한다.
        keep = [i for i, v in enumerate(scored_flags) if v]
        pick = lambda xs: [xs[i] for i in keep]
        by_T[str(T)] = metrics.aggregate_split(
            T, pick(sp.effective), pick(sp.adequacy), pick(sp.contradiction),
            pick(sp.consistency), pick(sp.chrf), pick(sp.laal_words), pick(sp.k),
            pick(missings), n_total=len(rows))

    return rows, metrics.aggregate(len(rows), valid_flags, first_pass, by_T), violations


def fmt_metrics(m: metrics.Metrics, tag: str) -> str:
    parts = [f"{tag} fmt={m.format_pass_rate:.2f}(1st {m.format_pass_rate_no_retry:.2f})"]
    for k in sorted(m.by_T, key=int):
        s = m.by_T[k]
        parts.append(f"T{k}: eff={s.effective:.4f} adq={s.adequacy:.4f} "
                     f"contra={s.contradiction:.3f} laal={s.laal_words:.2f} "
                     f"k={s.chunks_per_sentence:.2f} miss={s.missing_boundaries:.2f}")
    parts.append(f"score={metrics.score(m):.4f}")
    return "  ".join(parts)


# ── 메인 ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="의미 분절 프롬프트 자동 생성 루프 (v2)")
    p.add_argument("--dataset", default="kspon", choices=sorted(data.LOADERS))
    p.add_argument("--src-lang", default="Korean")
    p.add_argument("--tgt-lang", default="English")
    p.add_argument("--run-id", default=None)
    p.add_argument("--pair-id", default=None,
                   help="런 디렉토리 이름. 미지정 시 언어명에서 생성")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--judge-model", default=None,
                   help="판정자 모델. 미지정 시 --model. 분절기와 다른 모델을 쓰면 순환이 준다")
    p.add_argument("--translator-model", default="claude-sonnet-5")
    p.add_argument("--translator", default="google", choices=["llm", "google"])
    p.add_argument("--tgt-code", default=None)
    p.add_argument("--no-google-context", action="store_true")
    p.add_argument("--tgt-spaced", default=None, choices=["yes", "no"],
                   help="타깃 언어가 띄어쓰기를 쓰는가. 미지정 시 --tgt-lang 에서 추론 (LAAL 단위)")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--train", type=int, default=30)
    p.add_argument("--train-pool", type=int, default=None)
    p.add_argument("--max-prompt-growth", type=float, default=1.3)
    p.add_argument("--dev", type=int, default=60)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--min-chars", type=int, default=25)
    # 노브. 루프에서는 부분집합만 쓴다 — 조각 번역이 격자 크기에 비례해 늘기 때문이다.
    p.add_argument("--t-grid", type=int, nargs="+", default=[3, 6],
                   help="루프가 쓰는 목표 조각 크기. score 는 이 격자에서의 adequacy 평균이라 "
                        "다른 격자로 잰 score 와 비교할 수 없다")
    p.add_argument("--final-t-grid", type=int, nargs="+", default=[2, 3, 4, 6],
                   help="최종 test 곡선용 격자")
    p.add_argument("--main-t", type=int, default=None,
                   help="판정자가 도는 주 작동점. 미지정 시 --t-grid 의 중앙값")
    p.add_argument("--judge-rows", type=int, default=8,
                   help="이터레이션당 판정할 문장 수 (adequacy 하위부터)")
    p.add_argument("--no-judge", action="store_true", help="판정자를 끄고 adequacy 만으로 조향")
    p.add_argument("--adequacy-backend", default="cometkiwi",
                   choices=sorted(metrics.QE_CHECKPOINTS),
                   help="참조 없는 QE. y축 주지표")
    p.add_argument("--consistency-backend", default="comet",
                   choices=["comet", "xcomet", "embed", "chrf"],
                   help="참조 기반. 보고용")
    p.add_argument("--contradiction-backend", default="deberta-mnli",
                   choices=sorted(metrics.NLI_MODELS),
                   help="조기 방출 검출 NLI. 타깃이 영어가 아니면 mdeberta-xnli")
    p.add_argument("--no-coverage-rule", action="store_true",
                   help="최소 경계 수 요건을 끈다. 노브가 k 를 통제하지 못하게 된다")
    p.add_argument("--no-contradiction", action="store_true",
                   help="NLI 를 끈다. effective = adequacy 가 되어 조기 방출이 벌받지 않는다")
    p.add_argument("--comet-batch-size", type=int, default=16)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    t_grid = sorted(set(args.t_grid))
    final_grid = sorted(set(args.final_t_grid) | set(t_grid))
    # 커버리지 요건은 **곡선에 그릴 가장 조인 점**에서 온다. 루프 격자가 아니다 —
    # 배포할 프롬프트는 최종 곡선의 모든 점을 지탱해야 하고, 루프가 그보다 느슨한
    # 요건으로 학습하면 마지막 평가에서만 무너진다 (run03: test 1차 통과율 0.34).
    # 검증기(`evaluate`)와 프롬프트 문면(`output_rules`)이 **같은 값**을 써야 한다.
    coverage_t = min(final_grid)
    main_t = args.main_t or t_grid[len(t_grid) // 2]
    if main_t not in t_grid:
        print(f"--main-t {main_t} 가 --t-grid {t_grid} 에 없습니다", file=sys.stderr)
        return 2

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    pair_id = args.pair_id or f"{args.src_lang}-{args.tgt_lang}".lower().replace(" ", "_")
    run_dir = _HERE.parent / "runs" / pair_id / run_id
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gw = Gateway(model=args.model, budget=args.budget)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")
    translator = None

    adequacy = metrics.make_adequacy_backend(
        args.adequacy_backend, batch_size=args.comet_batch_size)
    consistency = metrics.make_backend(
        args.consistency_backend, gw=gw,
        **({"batch_size": args.comet_batch_size}
           if args.consistency_backend in metrics.COMET_CHECKPOINTS else {}))
    contradiction = (None if args.no_contradiction else
                     metrics.make_contradiction_backend(args.contradiction_backend))

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        # ── A0 데이터 ────────────────────────────────────────────────────
        sentences = data.LOADERS[args.dataset]()
        pool_n = max(args.train, args.train_pool or args.train)
        splits = data.split_data(sentences, pool_n, args.dev, args.test,
                                 min_chars=args.min_chars)
        data.write_splits(splits, run_dir / "data")
        log(f"[data] {args.dataset}: 전체 {len(sentences)}, "
            f"train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])}")

        # 측정 프로파일 — 결정론적 코드를 움직이는 필드는 코퍼스에서 직접 잰다
        measured = data.measure_profile([s.text for s in
                                         splits["train"] + splits["dev"] + splits["test"]])
        (run_dir / "measured_profile.json").write_text(
            json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── A1 Language Profiler ────────────────────────────────────────
        profiler = agents.Profiler(gw)
        profile_path = run_dir / "language_profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            samples = [s.text for s in splits["train"][:20]]
            profile = profiler.profile(samples, args.tgt_lang)
            profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

        # **JSON 은 고치지 않는다.** 고치면 prompt_v0 가 달라져 기존 런과 비교가 깨진다.
        # 덮어쓰기는 소비 지점인 여기서만 한다.
        spaced, trailing_punct, warns = data.reconcile_profile(measured, profile)
        for w in warns:
            log(f"[profile] 경고: {w}")
        tgt_spaced = (target_is_spaced(args.tgt_lang) if args.tgt_spaced is None
                      else args.tgt_spaced == "yes")
        log(f"[profile] {profile.get('source_language')} / 어순 {profile.get('word_order')} / "
            f"띄어쓰기 {spaced}(측정) / 타깃 띄어쓰기 {tgt_spaced}")

        if args.translator == "google":
            translator = GoogleTranslator(
                tgt_code=args.tgt_code or to_lang_code(args.tgt_lang),
                cache=tr_cache, workers=min(args.workers, 4),
                use_context=not args.no_google_context)
            translator_id = f"google:{translator.tgt_code}:ctx={translator.use_context}"
        else:
            translator = Translator(gw=gw, src_name=args.src_lang, tgt_name=args.tgt_lang,
                                    model=args.translator_model, cache=tr_cache,
                                    workers=args.workers)
            translator_id = f"llm:{args.translator_model}"
        log(f"[translator] {translator_id}")

        (run_dir / "config.json").write_text(json.dumps({
            **vars(args), "t_grid": t_grid, "final_t_grid": final_grid, "main_t": main_t,
            "translator_id": translator_id, "tgt_spaced": tgt_spaced,
            "adequacy_model": metrics.QE_CHECKPOINTS[args.adequacy_backend],
            "judge_prompt_hash": JsonCache.key(agents.JUDGE_SYSTEM),
            "judge_model": args.judge_model or args.model,
            "min_boundaries_per": coverage_t,    # [Output Rules] 에 박히는 값 = 검증기 요건
            "coverage_required": not args.no_coverage_rule,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        prompt_path = run_dir / "iter_00" / "prompt.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt = profiler.initial_prompt(profile, args.tgt_lang, spaced, coverage_t)
            missing = agents.check_skeleton(prompt)
            if missing:
                log(f"[profiler] 골격 누락 {missing} — 재시도")
                prompt = profiler.initial_prompt(profile, args.tgt_lang, spaced, coverage_t)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
        prompt_v0_len = len(prompt)
        log(f"[profiler] prompt_v0 작성 완료 ({prompt_v0_len}자)")

        # ── 루프 ────────────────────────────────────────────────────────
        critic = agents.Critic(gw)
        engineer = agents.PromptEngineer(gw)
        compressor = agents.Compressor(gw)
        judge = None if args.no_judge else agents.Judge(
            Gateway(model=args.judge_model or args.model, budget=args.budget)
            if args.judge_model else gw)

        history: list[dict] = []
        best = {"version": 0, "prompt": prompt, "train_score": None, "dev_score": None}
        best_ctx: dict = {}
        best_critique: dict | None = None
        last_focus: str | None = None
        stale = 0

        def run_eval(prompt_: str, split_sentences, grid):
            return evaluate(gw, translator, prompt_, split_sentences, spaced, seg_cache,
                            args.workers, adequacy, consistency, grid, trailing_punct,
                            tgt_spaced=tgt_spaced, contradiction=contradiction,
                            require_coverage=not args.no_coverage_rule,
                            coverage_t=coverage_t)

        for it in range(args.iterations):
            it_dir = run_dir / f"iter_{it:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)
            (it_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            batch = splits["train"]
            if len(batch) > args.train:
                batch = random.Random(20260808 + it).sample(batch, args.train)
            rows, m, viol = run_eval(prompt, batch, t_grid)
            sc = metrics.score(m)

            # ── A6′ Judge — 주 작동점에서만 (비용) ────────────────────────
            judgements: list[dict] = []
            if judge is not None and m.by_T:
                judgements = agents.judge_rows(judge, rows, main_t, args.judge_rows)
                pr = agents.premature_rate(judgements)
                if pr is not None and str(main_t) in m.by_T:
                    m.by_T[str(main_t)].premature_rate = round(pr, 4)
                (it_dir / "judgements.json").write_text(
                    json.dumps(judgements, ensure_ascii=False, indent=2), encoding="utf-8")

            (it_dir / "train_rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            (it_dir / "violations.json").write_text(
                json.dumps(viol, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"[iter {it}] {fmt_metrics(m, 'train')}"
                + (f"  premature={m.by_T[str(main_t)].premature_rate}"
                   if m.by_T.get(str(main_t)) and m.by_T[str(main_t)].premature_rate is not None
                   else ""))

            # 쌍체 차이는 **진단으로만** 기록한다. dev 가 고정 집합이라
            # `mean(new) − mean(old) = mean(new − old)` 가 항등이고, 따라서 점추정도
            # 채택 결정도 절대값 비교와 동일하다. 얻는 것은 **오차막대와 유효 표본**이다:
            # 분절이 안 바뀐 문장은 차이가 정확히 0 이라 분산에 기여하지 않으므로
            # se 가 훨씬 작게 나오고, `n_changed` 가 실제로 판정에 참여한 문장 수를 알려준다.
            # 잡음 채택을 막는 임계값(`Δ > k·se`)은 이 분포를 실측한 뒤에 정한다.
            train_delta = (metrics.paired_delta(rows, best_ctx["rows"], t_grid)
                           if best_ctx else None)
            if train_delta and train_delta["mean_delta"] is not None:
                log(f"[iter {it}] train Δ={train_delta['mean_delta']:+.5f} "
                    f"±{train_delta['se_delta']:.5f} (분절 변경 {train_delta['n_changed']}"
                    f"/{train_delta['n_pairs']}문장)")

            # dev 검증은 train 이 개선됐을 때만 (비용 절약)
            dev_m = None
            dev_score = None
            dev_delta = None
            dev_rows = None
            if best["train_score"] is None or sc > best["train_score"]:
                dev_rows, dev_m, dev_viol = run_eval(prompt, splits["dev"], t_grid)
                dev_score = metrics.score(dev_m)
                dev_delta = (metrics.paired_delta(dev_rows, best_ctx["dev_rows"], t_grid)
                             if best_ctx.get("dev_rows") else None)
                (it_dir / "dev_rows.json").write_text(
                    json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
                (it_dir / "dev_violations.json").write_text(
                    json.dumps(dev_viol, ensure_ascii=False, indent=2), encoding="utf-8")
                viol = viol + dev_viol
                log(f"[iter {it}] {fmt_metrics(dev_m, 'dev  ')}"
                    + (f"  Δ={dev_delta['mean_delta']:+.5f} ±{dev_delta['se_delta']:.5f} "
                       f"(변경 {dev_delta['n_changed']}/{dev_delta['n_pairs']})"
                       if dev_delta and dev_delta["mean_delta"] is not None else ""))

            adopted = False
            if dev_score is not None and (best["dev_score"] is None
                                          or dev_score > best["dev_score"]):
                best = {"version": it, "prompt": prompt,
                        "train_score": sc, "dev_score": dev_score}
                (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
                best_ctx = {"rows": rows, "dev_rows": dev_rows, "metrics": m.to_dict(),
                            "violations": viol, "judgements": judgements}
                best_critique = None      # 새 best — 비평을 다시 받아야 한다
                adopted = True
                stale = 0
            else:
                stale += 1

            (it_dir / "metrics.json").write_text(json.dumps({
                "train": m.to_dict(),
                "dev": dev_m.to_dict() if dev_m else None,
                "score_train": round(sc, 4),
                "score_dev": round(dev_score, 4) if dev_score is not None else None,
                "paired_train": train_delta, "paired_dev": dev_delta,
                "adopted": adopted,
                "usage": gw.usage.snapshot(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            history.append({
                "version": it, "adopted": adopted,
                "train": m.to_dict(), "dev": dev_m.to_dict() if dev_m else None,
                "score_train": round(sc, 4),
                "score_dev": round(dev_score, 4) if dev_score is not None else None,
                "paired_dev": dev_delta,
                "changelog": history[-1].get("next_changelog") if history else None,
            })
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

            u = gw.usage.snapshot()
            log(f"[iter {it}] adopted={adopted} stale={stale} "
                f"calls={u['calls']} cost={u['cost']:.4f}")

            if it == args.iterations - 1:
                break
            if stale >= args.patience:
                log(f"[stop] dev 개선 없음 {stale}회 연속 — 조기 종료")
                break

            # 에이전트 호출이 실패해도 런 전체를 버리지 않는다.
            ctx = best_ctx or {"rows": rows, "metrics": m.to_dict(),
                               "violations": viol, "judgements": judgements}
            try:
                # ── A6 Critic ───────────────────────────────────────────
                # 비평 대상과 개정 대상은 반드시 같은 프롬프트여야 한다.
                if best_critique is None:
                    cases = agents.select_cases(ctx["rows"], main_t, ctx.get("judgements"))
                    best_critique = critic.review(cases, ctx["metrics"], ctx["violations"],
                                                  avoid=last_focus if stale >= 2 else None)
                critique = best_critique
                # 캐시가 고착 방지를 우회하지 않도록 aggregate 만 다시 계산한다 (LLM 없음).
                if stale >= 2 and last_focus:
                    critique = {**critique, "aggregate": agents.summarize_critique(
                        critique.get("cases") or [], ctx["metrics"],
                        critique.get("summary"), avoid=last_focus)}
                (it_dir / "critique.json").write_text(
                    json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
                agg = critique.get("aggregate", {})
                last_focus = agg.get("focus")
                log(f"[iter {it}] critic dominant={agg.get('dominant_error')} "
                    f"focus={last_focus}")

                # ── A7 Prompt Engineer ──────────────────────────────────
                revised = engineer.revise(best["prompt"], critique, history, profile, t_grid)
                new_prompt = revised.get("prompt", "")
                budget = int(prompt_v0_len * args.max_prompt_growth)
                if new_prompt and len(new_prompt) > budget:
                    log(f"[iter {it}] 개정본 {len(new_prompt)}자 > 예산 {budget}자 — 압축")
                    packed = compressor.compress(
                        new_prompt, budget, revised.get("sections_changed") or [])
                    if (packed and not agents.check_skeleton(packed)
                            and len(packed) <= budget):
                        log(f"[iter {it}] 압축 성공 {len(new_prompt)} -> {len(packed)}자")
                        new_prompt = packed
                    else:
                        log(f"[iter {it}] 압축 실패({len(packed or '')}자) — 개정 거부")
                        new_prompt = ""
                missing = agents.check_skeleton(new_prompt) if new_prompt else []
                if not new_prompt:
                    log(f"[iter {it}] 개정 없음 — 이전 프롬프트 유지")
                elif missing or len(new_prompt) < 500:
                    log(f"[iter {it}] 개정 프롬프트 골격 누락 {missing} — 이전 프롬프트 유지")
                else:
                    prompt = new_prompt
                (it_dir / "changelog.json").write_text(json.dumps({
                    "sections_changed": revised.get("sections_changed"),
                    "changelog": revised.get("changelog"),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                history[-1]["next_changelog"] = revised.get("changelog")
                log(f"[iter {it}] 개정: {revised.get('sections_changed')}")
            except BudgetExceeded:
                raise
            except Exception as e:
                log(f"[iter {it}] 에이전트 실패 — 루프 중단하고 최종 평가로 넘어간다: {e}")
                break

        # ── 최종 test 평가 (전체 격자) ───────────────────────────────────
        log(f"[final] test 평가 — 격자 {final_grid} (루프가 한 번도 보지 않은 데이터)")
        test_rows, test_m, test_viol = run_eval(best["prompt"], splits["test"], final_grid)
        if judge is not None and test_m.by_T:
            tj = agents.judge_rows(judge, test_rows, main_t, args.judge_rows * 2)
            pr = agents.premature_rate(tj)
            if pr is not None and str(main_t) in test_m.by_T:
                test_m.by_T[str(main_t)].premature_rate = round(pr, 4)
            (run_dir / "test_judgements.json").write_text(
                json.dumps(tj, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "test_rows.json").write_text(
            json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        # ── 비교군 (노브 없음 — 각 1점) ──────────────────────────────────
        baselines = compare_baselines(translator, adequacy, consistency, splits["test"],
                                      spaced, tgt_spaced, args.workers, contradiction=contradiction)
        curve = {
            "ours": {k: v.to_dict() for k, v in test_m.by_T.items()},
            "baselines": baselines,
            "axes": {"x": "laal_words (source words, lower = faster)",
                     "y": "adequacy (reference-free QE)"},
        }
        (run_dir / "curve.json").write_text(
            json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")

        report = build_report(args, run_dir, profile, measured, history, best, test_m,
                              test_viol, gw.usage.snapshot(), baselines, t_grid,
                              final_grid, main_t, translator_id)
        (run_dir / "final_report.md").write_text(report, encoding="utf-8")
        log(f"\n{report}")
        log(f"[done] {run_dir}")
        return 0

    except BudgetExceeded as e:
        log(f"[stop] {e}")
        return 2
    finally:
        seg_cache.flush()
        tr_cache.flush()
        if isinstance(translator, GoogleTranslator):
            translator.close()
        gw.close()


# ── 비교군 ───────────────────────────────────────────────────────────────

def compare_baselines(translator, adequacy, consistency, sentences, spaced,
                      tgt_spaced, workers, mech_every: int = 8,
                      contradiction=None) -> dict:
    """무분절 / 기계 8자분절. 순위가 없어 절단이 불가능하므로 곡선 위의 점 하나씩이다."""
    texts = [s.text for s in sentences]
    full = translator.full(texts)
    out: dict[str, dict] = {}
    for name, segs in (("unsegmented", list(texts)),
                       ("mechanical_8", [metrics.mechanical_split(t, mech_every, spaced)
                                         for t in texts])):
        sp = score_split(segs, texts, full, translator, adequacy, consistency,
                         spaced, tgt_spaced, contradiction)
        out[name] = metrics.aggregate_split(
            0, sp.effective, sp.adequacy, sp.contradiction, sp.consistency,
            sp.chrf, sp.laal_words, sp.k, [0] * len(texts)).to_dict()
    return out


def build_report(args, run_dir, profile, measured, history, best, test_m, test_viol,
                 usage, baselines, t_grid, final_grid, main_t, translator_id) -> str:
    lines = [
        f"# 자동 분절 프롬프트 루프 결과 (v2) — {args.src_lang} → {args.tgt_lang}",
        "",
        f"- 데이터셋: `{args.dataset}` (train {args.train} / dev {args.dev} / test {args.test})",
        f"- 분절 모델 `{args.model}` / 판정자 `{args.judge_model or args.model}` / "
        f"번역기 `{translator_id}`",
        f"- adequacy 백엔드: **{args.adequacy_backend}** "
        f"(`{metrics.QE_CHECKPOINTS[args.adequacy_backend]}`, 참조 없음)",
        f"- consistency 백엔드: {args.consistency_backend} (보고용)",
        f"- 노브: 목표 조각 크기 T. 루프 격자 {t_grid}, 최종 격자 {final_grid}, 주 작동점 T={main_t}",
        f"- 언어 프로파일: {profile.get('source_language')}, 어순 {profile.get('word_order')} / "
        f"측정: 공백비율 {measured['space_ratio']}, 문말 부호 {measured['trailing_punctuation']}",
        f"- score = T 격자 평균 **effective** = `adequacy × (1 − contradiction)` "
        f"(가중치·임계값 없음). 채택 판정은 **쌍체 비교**",
        f"- contradiction 백엔드: "
        + ("없음 (조기 방출이 벌받지 않음)" if args.no_contradiction
           else f"**{args.contradiction_backend}** (`{metrics.NLI_MODELS[args.contradiction_backend]}`)"),
        f"- 채택된 프롬프트: iter_{best['version']:02d}",
        "",
        "## 이터레이션 이력",
        "",
        "| iter | fmt | train score | dev score | dev Δ (쌍체) | 변경 문장 | 채택 |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in history:
        t = h["train"]
        pd = h.get("paired_dev") or {}
        d = (f"{pd['mean_delta']:+.5f} ±{pd['se_delta']:.5f}"
             if pd.get("mean_delta") is not None else "—")
        lines.append(
            f"| {h['version']} | {t['format_pass_rate']:.2f} | {h['score_train']:.4f} | "
            + (f"{h['score_dev']:.4f}" if h.get("score_dev") is not None else "—")
            + f" | {d} | {pd.get('n_changed', '—')} | {'O' if h['adopted'] else 'X'} |")

    lines += [
        "",
        "## 최종 test 곡선",
        "",
        "| T (목표 조각 어절) | laal_words ↓ | **effective** ↑ | adequacy | contradiction ↓ | consistency | k | 부족 경계 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(test_m.by_T, key=int):
        s = test_m.by_T[k]
        lines.append(f"| {k} | {s.laal_words:.2f} | **{s.effective:.4f}** | {s.adequacy:.4f} | "
                     f"{s.contradiction:.4f} | {s.consistency:.4f} | "
                     f"{s.chunks_per_sentence:.2f} | {s.missing_boundaries:.2f} |")
    for name, b in baselines.items():
        lines.append(f"| {name} (노브 없음) | {b['laal_words']:.2f} | **{b['effective']:.4f}** | "
                     f"{b['adequacy']:.4f} | {b['contradiction']:.4f} | "
                     f"{b['consistency']:.4f} | {b['chunks_per_sentence']:.2f} | — |")

    pr = (test_m.by_T.get(str(main_t)).premature_rate
          if test_m.by_T.get(str(main_t)) else None)
    lines += [
        "",
        f"- 포맷 통과율 {test_m.format_pass_rate:.4f} (재시도 없이 "
        f"{test_m.format_pass_rate_no_retry:.4f}), 위반 {len(test_viol)}건",
        f"- premature_rate (T={main_t}, 부록 지표): "
        + (f"**{pr:.4f}**" if pr is not None else "미측정"),
        "",
        "`laal_words` 는 **소스 어절** 단위다 (논문의 ms 와 직접 비교 불가). "
        "`adequacy` 는 참조가 없으므로 offline 번역과 어순이 달라도 감점되지 않는다.",
        "",
        "## 비용",
        "",
        f"- 호출 {usage['calls']}회, 입력 토큰 {usage['prompt_tokens']:,} "
        f"(캐시 {usage['cached_tokens']:,}), 출력 토큰 {usage['completion_tokens']:,}",
        f"- 게이트웨이 추정 비용 {usage['cost']:.4f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
