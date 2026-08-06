"""A8 Loop Controller — 결정론적 오케스트레이터.

LLM 판단은 없다. 실행 순서, 채택/롤백, 예산, 중단 조건만 관리한다.

  python -m core.meaning_segmentator.autoseg.loop \
      --dataset kokoro --src-lang Japanese --tgt-lang Korean \
      --iterations 4 --train 20 --dev 30 --test 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import agents, data, metrics
from .gateway import BudgetExceeded, Gateway
from .pipeline import JsonCache, Translator, segment_batch, split_segments, validate

_HERE = Path(__file__).resolve().parent


# ── 평가 1회 ─────────────────────────────────────────────────────────────

def evaluate(
    gw: Gateway,
    translator: Translator,
    prompt: str,
    sentences: list[data.Sentence],
    spaced: bool,
    seg_cache: JsonCache,
    workers: int,
    trailing_punct: str | None = None,
    skip_translation_below: float = 0.95,
) -> tuple[list[dict], metrics.Metrics, list[dict]]:
    texts = [s.text for s in sentences]
    seg_texts, first_pass = segment_batch(
        gw, prompt, texts, cache=seg_cache, workers=workers,
        validate_fn=lambda t, out: validate("", t, out, spaced, trailing_punct),
    )

    violations: list[dict] = []
    valid_flags: list[bool] = []
    for s, seg in zip(sentences, seg_texts):
        vs = validate(s.id, s.text, seg, spaced, trailing_punct)
        valid_flags.append(not vs)
        violations.extend({"id": v.id, "rule": v.rule, "detail": v.detail,
                           "text": s.text, "seg_text": seg} for v in vs)

    valid_rate = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    n_seg = [max(1, len(split_segments(t))) for t in seg_texts]

    # 포맷이 무너진 프롬프트에는 번역·점수를 쓰지 않는다 (저비용 게이트 먼저)
    if valid_rate < skip_translation_below:
        m = metrics.aggregate([], [], seg_texts, valid_flags, spaced, first_pass)
        rows = [
            {"id": s.id, "text": s.text, "seg_text": seg, "valid": v,
             "n_segments": k, "quality": 0.0, "chrf": 0.0,
             "full_trans": None, "seg_trans": None, "seg_pieces": None}
            for s, seg, v, k in zip(sentences, seg_texts, valid_flags, n_seg)
        ]
        return rows, m, violations

    full = translator.full(texts)
    seg_joined, seg_pieces = translator.seg_batch(seg_texts, full)

    q = metrics.embed_similarity(gw, seg_joined, full)
    c = [metrics.chrf(h, r) for h, r in zip(seg_joined, full)]

    rows = [
        {"id": s.id, "text": s.text, "seg_text": seg, "valid": v, "n_segments": k,
         "quality": round(qi, 4), "chrf": round(ci, 4),
         "full_trans": f, "seg_trans": sj, "seg_pieces": sp}
        for s, seg, v, k, qi, ci, f, sj, sp in zip(
            sentences, seg_texts, valid_flags, n_seg, q, c, full, seg_joined, seg_pieces
        )
    ]
    m = metrics.aggregate(q, c, seg_texts, valid_flags, spaced, first_pass)
    return rows, m, violations


# ── Q_floor 캘리브레이션 ─────────────────────────────────────────────────

def calibrate_q_floor(
    gw: Gateway,
    translator: Translator,
    sentences: list[data.Sentence],
    spaced: bool,
    ratio: float,
    every: int = 8,
) -> dict:
    """Q_floor 를 하한·상한 앵커 두 개 사이에서 잡는다.

    **상한을 1.0 으로 두면 안 된다.** 번역기가 결정론적이지 않기 때문이다.
    분절을 전혀 하지 않고 같은 문장을 두 번 번역해도 Q 가 1.0 이 나오지 않는다
    (ja 실측 0.9273, 최악 문장 0.5866). 이 값이 이 지표로 도달 가능한 **실제
    상한**이고, 그보다 위를 요구하면 루프는 번역기 잡음을 쫓게 된다.

      Q_bad     = 의미를 무시한 기계적 분절            (하한 앵커)
      Q_ceiling = 무분절 + 동일 문장 2회 번역          (상한 앵커, 지표 잡음 바닥)
      Q_floor   = Q_bad + ratio * (Q_ceiling - Q_bad)

    두 앵커 사이 거리가 좁으면(예: 0.05 미만) 지표가 분절 품질을 분해하지 못한다는
    뜻이므로 그대로 리포트해 판단 근거로 남긴다.
    """
    texts = [s.text for s in sentences]
    full = translator.full(texts)

    bad = [metrics.mechanical_split(t, every, spaced) for t in texts]
    joined, _ = translator.seg_batch(bad, full)

    # 상한: 분절 없이 같은 문장을 한 번 더 번역해 서로 비교
    full2 = translator.full_uncached(texts)

    n = len(texts)
    q_bad = sum(metrics.embed_similarity(gw, joined, full)) / n
    q_ceiling = sum(metrics.embed_similarity(gw, full2, full)) / n

    # 백엔드 선택은 논증이 아니라 측정으로 한다. 두 앵커 사이가 넓은 백엔드일수록
    # 분절 품질을 잘 분해한다. 번역은 이미 끝났으므로 추가 API 호출은 없다.
    chrf_bad = sum(metrics.chrf(h, r) for h, r in zip(joined, full)) / n
    chrf_ceiling = sum(metrics.chrf(h, r) for h, r in zip(full2, full)) / n

    floor = q_bad + ratio * (q_ceiling - q_bad)
    return {
        "backend": "embed",
        "q_mechanical_split": round(q_bad, 4),
        "q_ceiling_retranslate": round(q_ceiling, 4),
        "anchor_gap": round(q_ceiling - q_bad, 4),
        "mechanical_every_chars": every,
        "ratio": ratio,
        "q_floor": round(floor, 4),
        # 참고용 — 백엔드를 바꿀 때 어느 쪽이 분해능이 좋은지 판단하는 근거
        "alt_backends": {
            "chrf": {
                "q_mechanical_split": round(chrf_bad, 4),
                "q_ceiling_retranslate": round(chrf_ceiling, 4),
                "anchor_gap": round(chrf_ceiling - chrf_bad, 4),
                "q_floor": round(chrf_bad + ratio * (chrf_ceiling - chrf_bad), 4),
            }
        },
    }


# ── 메인 ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="의미 분절 프롬프트 자동 생성 루프")
    p.add_argument("--dataset", default="kokoro", choices=sorted(data.LOADERS))
    p.add_argument("--src-lang", default="Japanese")
    p.add_argument("--tgt-lang", default="Korean")
    p.add_argument("--run-id", default=None)
    p.add_argument("--pair-id", default=None,
                   help="런 디렉토리 이름. 미지정 시 언어명에서 생성 (표시용일 뿐, 로직에 쓰지 않음)")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--translator-model", default="claude-sonnet-5")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--train", type=int, default=20)
    p.add_argument("--dev", type=int, default=30)
    p.add_argument("--test", type=int, default=50)
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--q-floor", type=float, default=None, help="미지정 시 캘리브레이션으로 산출")
    p.add_argument("--q-floor-ratio", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fresh", action="store_true", help="기존 런 디렉토리를 지우고 처음부터")
    args = p.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    pair_id = args.pair_id or f"{args.src_lang}-{args.tgt_lang}".lower().replace(" ", "_")
    run_dir = _HERE.parent / "runs" / pair_id / run_id
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    gw = Gateway(model=args.model, budget=args.budget)
    seg_cache = JsonCache(run_dir / "cache" / "segment.json")
    tr_cache = JsonCache(run_dir / "cache" / "translate.json")

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        # ── A0 데이터 ────────────────────────────────────────────────────
        sentences = data.LOADERS[args.dataset]()
        splits = data.split_data(sentences, args.train, args.dev, args.test,
                                 min_chars=args.min_chars)
        data.write_splits(splits, run_dir / "data")
        log(f"[data] {args.dataset}: 전체 {len(sentences)}, "
            f"train {len(splits['train'])} / dev {len(splits['dev'])} / test {len(splits['test'])}")

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
        spaced = bool(profile.get("uses_spaces_between_words", True))
        # 언어 지식은 프로파일에서 데이터로 들어온다 — 검증기 코드는 언어 무관이다
        trailing_punct = "".join(profile.get("trailing_punctuation") or []) or None
        log(f"[profiler] {profile.get('source_language')} / 어순 {profile.get('word_order')} / "
            f"띄어쓰기 {spaced} / 문체 {profile.get('register')}")

        translator = Translator(gw=gw, src_name=args.src_lang, tgt_name=args.tgt_lang,
                                model=args.translator_model, cache=tr_cache, workers=args.workers)

        prompt_path = run_dir / "iter_00" / "prompt.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt = profiler.initial_prompt(profile, args.tgt_lang, spaced)
            missing = agents.check_skeleton(prompt)
            if missing:
                log(f"[profiler] 골격 누락 {missing} — 재시도")
                prompt = profiler.initial_prompt(profile, args.tgt_lang, spaced)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
        log(f"[profiler] prompt_v0 작성 완료 ({len(prompt)}자)")

        # ── Q_floor 캘리브레이션 ────────────────────────────────────────
        cal_path = run_dir / "baseline.json"
        if args.q_floor is not None:
            cal = {"q_floor": args.q_floor, "source": "manual"}
        elif cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
        else:
            cal = calibrate_q_floor(gw, translator, splits["train"], spaced, args.q_floor_ratio)
            cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
        q_floor = cal["q_floor"]
        log(f"[baseline] 기계분절 Q={cal.get('q_mechanical_split')} -> Q_floor={q_floor}")

        # ── 루프 ────────────────────────────────────────────────────────
        critic = agents.Critic(gw)
        engineer = agents.PromptEngineer(gw)

        history: list[dict] = []
        best = {"version": 0, "prompt": prompt, "train_obj": None, "dev_obj": None}
        best_ctx: dict = {}          # 채택된 프롬프트의 rows/metrics/violations
        best_critique: dict | None = None
        stale = 0

        for it in range(args.iterations):
            it_dir = run_dir / f"iter_{it:02d}"
            it_dir.mkdir(parents=True, exist_ok=True)
            (it_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            rows, m, viol = evaluate(gw, translator, prompt, splits["train"], spaced,
                                     seg_cache, args.workers, trailing_punct)
            obj = metrics.objective(m, q_floor)
            (it_dir / "train_rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            (it_dir / "violations.json").write_text(
                json.dumps(viol, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"[iter {it}] train  V={m.valid_rate:.2f}(1st {m.first_pass_valid_rate:.2f}) "
                f"Qs={m.quality_segmented:.4f}(n={m.n_segmented}) Q={m.quality:.4f} "
                f"gain={m.latency_gain:.4f} k={m.mean_segments:.2f} "
                f"seg%={m.segmented_rate:.2f} obj={obj:.4f}")

            # dev 검증은 train이 개선됐을 때만 (비용 절약)
            dev_m = None
            dev_obj = None
            if best["train_obj"] is None or obj > best["train_obj"]:
                dev_rows, dev_m, dev_viol = evaluate(gw, translator, prompt, splits["dev"],
                                                     spaced, seg_cache, args.workers, trailing_punct)
                dev_obj = metrics.objective(dev_m, q_floor)
                (it_dir / "dev_rows.json").write_text(
                    json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
                (it_dir / "dev_violations.json").write_text(
                    json.dumps(dev_viol, ensure_ascii=False, indent=2), encoding="utf-8")
                # dev 위반은 train 에 안 나타난 프롬프트 구멍이다 — 비평 대상에 합류시킨다
                viol = viol + dev_viol
                log(f"[iter {it}] dev    V={dev_m.valid_rate:.2f}(1st {dev_m.first_pass_valid_rate:.2f}) "
                    f"Qs={dev_m.quality_segmented:.4f}(n={dev_m.n_segmented}) "
                    f"gain={dev_m.latency_gain:.4f} k={dev_m.mean_segments:.2f} obj={dev_obj:.4f}")

            adopted = False
            if dev_obj is not None and (best["dev_obj"] is None or dev_obj > best["dev_obj"]):
                best = {"version": it, "prompt": prompt, "train_obj": obj, "dev_obj": dev_obj}
                (run_dir / "best_prompt.txt").write_text(prompt, encoding="utf-8")
                best_ctx = {"rows": rows, "metrics": m.to_dict(), "violations": viol}
                best_critique = None      # 새 best — 비평을 다시 받아야 한다
                adopted = True
                stale = 0
            else:
                stale += 1

            (it_dir / "metrics.json").write_text(json.dumps({
                "train": m.to_dict(),
                "dev": dev_m.to_dict() if dev_m else None,
                "objective_train": round(obj, 4),
                "objective_dev": round(dev_obj, 4) if dev_obj is not None else None,
                "adopted": adopted,
                "usage": gw.usage.snapshot(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            history.append({
                "version": it, "adopted": adopted,
                "train": m.to_dict(), "dev": dev_m.to_dict() if dev_m else None,
                "objective_train": round(obj, 4),
                "objective_dev": round(dev_obj, 4) if dev_obj is not None else None,
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

            # ── A6 Critic ────────────────────────────────────────────────
            # 비평 대상과 개정 대상은 반드시 같은 프롬프트여야 한다. 거부된 후보를
            # 진단해 놓고 best 를 고치면 진단이 엉뚱한 곳에 적용된다.
            # best 가 안 바뀌었으면 비평도 그대로이므로 캐시를 재사용한다 (비용 절감).
            if best_critique is None:
                ctx = best_ctx or {"rows": rows, "metrics": m.to_dict(), "violations": viol}
                cases = agents.select_cases(ctx["rows"])
                best_critique = critic.review(cases, ctx["metrics"], ctx["violations"],
                                              q_floor)
            critique = best_critique
            (it_dir / "critique.json").write_text(
                json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")
            agg = critique.get("aggregate", {})
            log(f"[iter {it}] critic dominant={agg.get('dominant_error')} "
                f"direction={agg.get('direction')}")

            # ── A7 Prompt Engineer ──────────────────────────────────────
            # 채택 실패 시 후보를 버리고 현재 best 에서 다시 출발한다 (롤백)
            revised = engineer.revise(best["prompt"], critique, history, profile, q_floor)
            new_prompt = revised.get("prompt", "")
            missing = agents.check_skeleton(new_prompt)
            if missing or len(new_prompt) < 500:
                log(f"[iter {it}] 개정 프롬프트 골격 누락 {missing} — 이전 프롬프트 유지")
            else:
                prompt = new_prompt
            (it_dir / "changelog.json").write_text(json.dumps({
                "sections_changed": revised.get("sections_changed"),
                "changelog": revised.get("changelog"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            history[-1]["next_changelog"] = revised.get("changelog")
            log(f"[iter {it}] 개정: {revised.get('sections_changed')}")

        # ── 최종 test 평가 ───────────────────────────────────────────────
        log("[final] test 평가 (루프가 한 번도 보지 않은 데이터)")
        test_rows, test_m, test_viol = evaluate(gw, translator, best["prompt"], splits["test"],
                                                spaced, seg_cache, args.workers, trailing_punct)
        test_obj = metrics.objective(test_m, q_floor)
        (run_dir / "test_rows.json").write_text(
            json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        report = build_report(args, run_dir, profile, cal, history, best, test_m, test_obj,
                              test_viol, gw.usage.snapshot(), test_rows)
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
        gw.close()


def build_report(args, run_dir, profile, cal, history, best, test_m, test_obj,
                 test_viol, usage, test_rows) -> str:
    lines = [
        f"# 자동 분절 프롬프트 루프 결과 — {args.src_lang} → {args.tgt_lang}",
        "",
        f"- 데이터셋: `{args.dataset}` (train {args.train} / dev {args.dev} / test {args.test})",
        f"- 분절 모델: `{args.model}` / 번역 모델: `{args.translator_model}`",
        f"- 언어 프로파일: {profile.get('source_language')}, 어순 {profile.get('word_order')}, "
        f"띄어쓰기 {profile.get('uses_spaces_between_words')}, 문체 {profile.get('register')}",
        f"- Q_floor: **{cal.get('q_floor')}** = 하한 {cal.get('q_mechanical_split')} + "
        f"{cal.get('ratio')} x (상한 {cal.get('q_ceiling_retranslate')} - 하한), "
        f"앵커 간격 {cal.get('anchor_gap')}",
        f"- 채택된 프롬프트: iter_{best['version']:02d}",
        "",
        "## 이터레이션 이력",
        "",
        "| iter | train V | train Qs (n) | train gain | train k | dev Qs | dev gain | dev obj | 채택 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in history:
        t, d = h["train"], h["dev"]
        lines.append(
            f"| {h['version']} | {t['valid_rate']:.2f} | {t['quality_segmented']:.4f} ({t['n_segmented']}) | "
            f"{t['latency_gain']:.4f} | {t['mean_segments']:.2f} | "
            f"{d['quality_segmented']:.4f} | {d['latency_gain']:.4f} | {h['objective_dev']:.4f} |"
            if d else
            f"| {h['version']} | {t['valid_rate']:.2f} | {t['quality_segmented']:.4f} ({t['n_segmented']}) | "
            f"{t['latency_gain']:.4f} | {t['mean_segments']:.2f} | — | — | — |"
        )
        lines[-1] += f" {'O' if h['adopted'] else 'X'} |"

    lines += [
        "",
        "## 최종 test 결과",
        "",
        "| 지표 | 값 | 의미 |",
        "|---|---|---|",
        f"| V (포맷 유효율) | {test_m.valid_rate:.4f} | 복구 재시도 이후. 1.0 이어야 통과 |",
        f"| 1차 통과율 | {test_m.first_pass_valid_rate:.4f} | 복구 없이 한 번에 지킨 비율 |",
        f"| **Q (분절된 문장만, n={test_m.n_segmented})** | **{test_m.quality_segmented:.4f}** | 제약 대상. Q_floor {cal.get('q_floor')} |",
        f"| Q (전체 평균) | {test_m.quality:.4f} | 무분절 1.0 포함 — 보고용 |",
        f"| Q p10 | {test_m.quality_p10:.4f} | 하위 10% |",
        f"| Q min | {test_m.quality_min:.4f} | 최악 문장 |",
        f"| chrF (보조) | {test_m.quality_chrf:.4f} | 표면형 일치도 |",
        f"| L (지연 프록시) | {test_m.latency:.4f} | 1.0=무분절, 낮을수록 빠름 |",
        f"| gain (1-L) | {test_m.latency_gain:.4f} | 무분절 대비 지연 이득 |",
        f"| 평균 세그먼트 수 | {test_m.mean_segments:.2f} | |",
        f"| 분절된 문장 비율 | {test_m.segmented_rate:.4f} | |",
        f"| objective | {test_obj:.4f} | 제약 만족 시 = gain |",
        f"| 포맷 위반 건수 | {len(test_viol)} | |",
        "",
        "## 베이스라인 대비",
        "",
        "| 기준 | Q | L | gain | 비고 |",
        "|---|---|---|---|---|",
        f"| 상한: 무분절 + 동일 문장 재번역 | {cal.get('q_ceiling_retranslate')} | 1.0000 | 0.0000 | "
        f"지표 잡음 바닥. 1.0 이 아닌 이유는 번역기가 결정론적이지 않기 때문 |",
        f"| 하한: 기계분절 ({cal.get('mechanical_every_chars')}자마다) | "
        f"{cal.get('q_mechanical_split')} | — | — | 의미 무시 |",
        f"| 자동 생성 프롬프트 | {test_m.quality_segmented:.4f} | {test_m.latency:.4f} | "
        f"{test_m.latency_gain:.4f} | 분절된 문장 {test_m.n_segmented}건 기준 |",
        "",
        "## 비용",
        "",
        f"- 호출 {usage['calls']}회, 입력 토큰 {usage['prompt_tokens']:,} "
        f"(캐시 {usage['cached_tokens']:,}), 출력 토큰 {usage['completion_tokens']:,}",
        f"- 게이트웨이 추정 비용 {usage['cost']:.4f}",
        "",
        "## 샘플 (test 상위 5)",
        "",
    ]
    for r in test_rows[:5]:
        lines += [
            f"- 원문: {r['text']}",
            f"  - 분절: {r['seg_text']}",
            f"  - Q={r['quality']} / k={r['n_segments']}",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
