"""
분절 전략별 COMET 비교 실험.

세 가지 분절 전략의 번역 품질을 동일한 ref 기준으로 비교한다:
  - mt_pipeline : 파이프라인 실제 분절 번역 (metric.json의 seg_translations)
  - mt_random   : 공백 기준 랜덤 분절 → context-aware GPT 번역
  - mt_minword  : 단어 최소 N개 제약 랜덤 분절 → context-aware GPT 번역

ref: hypothesis 전체를 통째로 GPT에 입력한 번역
     (문장 단위 번역투 유사도 편향을 방지하기 위해 분절 없이 한 번에 번역)

Usage:
    python evaluation/LibriSpeech/utils/comet_seg_ablation.py \\
        --metric-json evaluation/LibriSpeech/results/finetuned_silence_gpt(1.0.4)/full/run_09/metric.json \\
        --api-key sk-...
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Qwen3-ASR" / "examples"))
sys.path.insert(0, str(PROJECT_ROOT))
from streaming_websocket_server import google_translate_async  # noqa: E402

_LANG_NAMES = {"ko": "Korean", "ja": "Japanese", "zh": "Chinese", "en": "English"}


# ---------------------------------------------------------------------------
# 분절 전략
# ---------------------------------------------------------------------------

def segment_random(text: str, seed: int = 42, split_prob: float = 0.35) -> list[str]:
    """공백 위치 기준 랜덤 분절. 각 단어 경계에서 split_prob 확률로 분할."""
    words = text.split()
    if len(words) <= 1:
        return [text]
    rng = random.Random(seed)
    segments, buf = [], []
    for word in words:
        buf.append(word)
        if len(buf) > 1 and rng.random() < split_prob:
            segments.append(" ".join(buf))
            buf = []
    if buf:
        if segments:
            segments[-1] = segments[-1] + " " + " ".join(buf)
        else:
            segments.append(" ".join(buf))
    return [s for s in segments if s.strip()]


def segment_minword(text: str, min_words: int = 3, seed: int = 42, split_prob: float = 0.45) -> list[str]:
    """단어 최소 min_words개 제약 랜덤 분절."""
    words = text.split()
    if len(words) <= min_words:
        return [text]
    rng = random.Random(seed)
    segments, buf = [], []
    for word in words:
        buf.append(word)
        if len(buf) >= min_words and rng.random() < split_prob:
            segments.append(" ".join(buf))
            buf = []
    if buf:
        if segments:
            segments[-1] = segments[-1] + " " + " ".join(buf)
        else:
            segments.append(" ".join(buf))
    return [s for s in segments if s.strip()]


# ---------------------------------------------------------------------------
# 번역 함수 — Google
# ---------------------------------------------------------------------------

_SEP = "\n[|||]\n"  # Google Translate가 건드리지 않는 구분자
_MAX_CHARS = 4500   # Google Translate 무료 API 요청당 안전 한도


async def translate_whole_google(texts: list[str], target_lang: str, pbar: Optional[tqdm] = None) -> list[str]:
    """모든 hypothesis를 배치로 이어붙여 Google Translate → 구분자로 재분리.

    파일 단위가 아닌 전체 corpus를 통째로 번역함으로써
    개별 문장 번역투 편향(short-segment translation style bias)을 방지한다.
    API 요청당 _MAX_CHARS 이하로 파일 경계에서 배치를 나눈다.
    """
    # 파일 경계에서 배치 분할
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_len = 0
    for i, text in enumerate(texts):
        needed = len(text) + len(_SEP) if current_batch else len(text)
        if current_batch and current_len + needed > _MAX_CHARS:
            batches.append(current_batch)
            current_batch = [i]
            current_len = len(text)
        else:
            current_batch.append(i)
            current_len += needed
    if current_batch:
        batches.append(current_batch)

    async with aiohttp.ClientSession() as session:
        async def translate_batch(indices: list[int]) -> list[str]:
            joined = _SEP.join(texts[i] for i in indices)
            translated, _ = await google_translate_async(session, joined, target_lang)
            parts = translated.split(_SEP)
            if len(parts) != len(indices):
                fallback = await asyncio.gather(
                    *[google_translate_async(session, texts[i], target_lang) for i in indices]
                )
                result = [r[0] for r in fallback]
            else:
                result = parts
            if pbar is not None:
                pbar.update(len(indices))
            return result

        results_nested = await asyncio.gather(*[translate_batch(b) for b in batches])

    # 배치 순서대로 flatten
    flat = [t for batch_result in results_nested for t in batch_result]
    return flat


async def translate_segmented_files_google(
    file_segments: list[list[str]], target_lang: str, pbar: Optional[tqdm] = None
) -> list[str]:
    """파일별 segments를 Google Translate로 각각 번역 후 concat.

    전체 segments를 한 번에 병렬 번역하고 파일별로 재조합한다.
    """
    sizes = [len(segs) for segs in file_segments]
    flat = [seg for segs in file_segments for seg in segs]

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[google_translate_async(session, t, target_lang) for t in flat]
        )
    translations = [r[0] for r in results]

    # 파일별로 재조합
    out, idx = [], 0
    for size in sizes:
        out.append(" ".join(translations[idx:idx + size]))
        idx += size
        if pbar is not None:
            pbar.update(1)
    return out


# ---------------------------------------------------------------------------
# 번역 함수 — GPT
# ---------------------------------------------------------------------------

_GPT_BATCH_CHARS = 12000  # GPT는 컨텍스트 윈도우가 크므로 넉넉하게


async def translate_whole(
    texts: list[str],
    target_lang: str,
    src_lang: str,
    model: str,
    api_key: str | None,
    pbar: Optional[tqdm] = None,
) -> list[str]:
    """모든 hypothesis를 배치로 이어붙여 GPT 번역 → 구분자로 재분리 (ref용).

    Google 방식과 동일하게 전체를 통째로 번역함으로써
    개별 문장 번역투 편향을 방지한다.
    파일 경계에서 _GPT_BATCH_CHARS 이하로 배치를 나눈다.
    """
    from openai import AsyncOpenAI

    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    system_prompt = (
        f"Translate the following text to {lang_name}. "
        f"The text contains multiple passages separated by '{_SEP.strip()}'. "
        f"Translate each passage and output them separated by '{_SEP.strip()}' in the same order. "
        "Output only the translations, no explanations."
    )
    sem = asyncio.Semaphore(5)

    # 파일 경계에서 배치 분할
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_len = 0
    for i, text in enumerate(texts):
        needed = len(text) + len(_SEP) if current_batch else len(text)
        if current_batch and current_len + needed > _GPT_BATCH_CHARS:
            batches.append(current_batch)
            current_batch = [i]
            current_len = len(text)
        else:
            current_batch.append(i)
            current_len += needed
    if current_batch:
        batches.append(current_batch)

    async with AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY")) as client:
        async def translate_batch(indices: list[int]) -> list[str]:
            joined = _SEP.join(texts[i] for i in indices)
            async with sem:
                for attempt in range(5):
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": joined},
                            ],
                            temperature=0,
                        )
                        translated = resp.choices[0].message.content.strip()
                        parts = translated.split(_SEP.strip())
                        if len(parts) == len(indices):
                            result = [p.strip() for p in parts]
                        else:
                            result = []
                            for i in indices:
                                fb_resp = await client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": f"Translate the following text to {lang_name}. Output only the translation."},
                                        {"role": "user", "content": texts[i]},
                                    ],
                                    temperature=0,
                                )
                                result.append(fb_resp.choices[0].message.content.strip())
                        if pbar is not None:
                            pbar.update(len(indices))
                        return result
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg or "rate_limit" in msg.lower():
                            m = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", msg)
                            wait = (
                                float(m.group(1)) / 1000 if m and m.group(2) == "ms"
                                else float(m.group(1)) if m
                                else float(2 ** attempt)
                            )
                            await asyncio.sleep(max(wait + 0.2, 1.0))
                        else:
                            raise
            raise RuntimeError(f"Max retries exceeded for batch of {len(indices)} files")

        results_nested = await asyncio.gather(*[translate_batch(b) for b in batches])

    return [t for batch_result in results_nested for t in batch_result]


async def translate_segmented_files(
    file_segments: list[list[str]],
    target_lang: str,
    src_lang: str,
    model: str,
    api_key: str | None,
    max_context: int,
    pbar: Optional[tqdm] = None,
) -> list[str]:
    """각 파일의 segments를 context 유지하며 순차 번역 후 concat."""
    from core.translator import GPTTranslator

    translator = GPTTranslator(api_key=api_key, model=model, max_context=max_context)
    src_lang_name = _LANG_NAMES.get(src_lang, src_lang)
    sem = asyncio.Semaphore(10)

    async def translate_file(segments: list[str]) -> str:
        async with sem:
            ctx: deque[tuple[str, str]] = deque(maxlen=max_context)
            results = []
            for seg in segments:
                if not seg.strip():
                    continue
                _, translation, _ = await translator.correct_and_translate(
                    text=seg,
                    source_lang_name=src_lang_name,
                    target_lang_code=target_lang,
                    context=list(ctx) if ctx else None,
                )
                results.append(translation)
                ctx.append((seg, translation))
            if pbar is not None:
                pbar.update(1)
            return " ".join(results)

    try:
        return await asyncio.gather(*[translate_file(segs) for segs in file_segments])
    finally:
        await translator._client.close()


# ---------------------------------------------------------------------------
# COMET
# ---------------------------------------------------------------------------

def run_comet(model, srcs, mts, refs=None):
    if refs is not None:
        data = [{"src": s, "ref": r, "mt": m} for s, r, m in zip(srcs, refs, mts)]
    else:
        data = [{"src": s, "mt": m} for s, m in zip(srcs, mts)]
    out = model.predict(data, batch_size=8, gpus=1, progress_bar=True)
    return out.system_score, out.scores


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-json", required=True)
    parser.add_argument("--target-lang", default="ko")
    parser.add_argument("--src-lang", default="en")
    parser.add_argument(
        "--translator", choices=["google", "gpt"], default="gpt",
        help="ref 및 분절 번역에 사용할 번역기 (default: gpt)",
    )
    parser.add_argument("--gpt-model", default="gpt-5.4-mini")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수)")
    parser.add_argument("--qe", action="store_true", help="Reference-free QE 모드: ref 번역 생략, src+mt만 평가")
    parser.add_argument("--reuse", nargs="?", const=True, default=None,
                        help="기존 번역 결과 재사용 (API 호출 생략). 경로 지정 시 해당 파일 사용, 경로 생략 시 metric-json 디렉토리의 comet_seg_ablation.json 사용")
    parser.add_argument("--comet-model", default=None, help="COMET 모델 (기본값: QE 시 wmt22-cometkiwi-da, 일반 시 wmt22-comet-da)")
    parser.add_argument("--max-context", type=int, default=5)
    parser.add_argument("--min-words", type=int, default=3, help="minword 분절 시 세그먼트 최소 단어 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.comet_model is None:
        args.comet_model = "Unbabel/wmt22-cometkiwi-da" if args.qe else "Unbabel/wmt22-comet-da"

    metric_path = Path(args.metric_json)
    with open(metric_path) as f:
        metric = json.load(f)

    raw = metric.get("raw_results", [])
    if not raw:
        sys.exit("No raw_results found in metric.json")

    file_ids, hypotheses, mt_pipeline = [], [], []
    for entry in raw:
        segs = entry.get("segment_metrics", [])
        seg_trans = " ".join(
            s["translation"] for s in segs if s.get("translation", "").strip()
        )
        file_ids.append(entry["file_id"])
        hypotheses.append(entry["hypothesis"])
        mt_pipeline.append(seg_trans)

    n = len(file_ids)
    print(f"Loaded {n} files from {metric_path}")

    avg_pipeline = sum(len(entry.get("segment_metrics", [])) for entry in raw) / n
    use_gpt = args.translator == "gpt"

    default_name = "comet_seg_ablation_qe.json" if args.qe else "comet_seg_ablation.json"
    out_path = Path(args.output) if args.output else metric_path.parent / default_name
    checkpoint_path = metric_path.parent / "comet_seg_ablation_checkpoint.json"

    if args.reuse:
        if isinstance(args.reuse, str):
            src_ablation = Path(args.reuse)
        else:
            src_ablation = metric_path.parent / "comet_seg_ablation.json"
        if not src_ablation.exists():
            sys.exit(f"--reuse 지정했지만 {src_ablation} 가 없습니다.")
        with open(src_ablation) as f:
            prev = json.load(f)
        prev_map = {e["file_id"]: e for e in prev["per_file"]}
        mt_pipeline = [prev_map[fid]["mt_pipeline"] for fid in file_ids]
        mt_random   = [prev_map[fid]["mt_random"]   for fid in file_ids]
        mt_minword  = [prev_map[fid]["mt_minword"]  for fid in file_ids]
        refs = [prev_map[fid]["ref"] for fid in file_ids] if not args.qe and "ref" in prev["per_file"][0] else None
        avg_random  = prev["segmentation"]["avg_seg_count"]["random"]
        avg_minword = prev["segmentation"]["avg_seg_count"]["minword"]
        print(f"기존 번역 재사용: {src_ablation}")
        print(f"평균 분절 수: pipeline={avg_pipeline:.1f}  random={avg_random:.1f}  minword={avg_minword:.1f}")
    else:
        # 분절 생성 (결정적, 체크포인트 불필요)
        segs_random  = [segment_random(h, seed=args.seed) for h in hypotheses]
        segs_minword = [segment_minword(h, min_words=args.min_words, seed=args.seed) for h in hypotheses]
        avg_random  = sum(len(s) for s in segs_random)  / n
        avg_minword = sum(len(s) for s in segs_minword) / n
        print(f"\n평균 분절 수: pipeline={avg_pipeline:.1f}  random={avg_random:.1f}  minword={avg_minword:.1f}")

        # 체크포인트 로드 (설정이 동일한 경우만)
        refs, mt_random, mt_minword = None, None, None
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    ckpt = json.load(f)
                if (ckpt.get("translator") == args.translator
                        and ckpt.get("target_lang") == args.target_lang
                        and ckpt.get("qe") == args.qe):
                    refs       = ckpt.get("refs")
                    mt_random  = ckpt.get("mt_random")
                    mt_minword = ckpt.get("mt_minword")
                    loaded = [k for k, v in [("refs", refs), ("mt_random", mt_random), ("mt_minword", mt_minword)] if v is not None]
                    if loaded:
                        print(f"체크포인트에서 재사용: {', '.join(loaded)}")
            except Exception:
                pass

        def save_checkpoint():
            data = {
                "translator": args.translator,
                "target_lang": args.target_lang,
                "qe": args.qe,
                "refs":       refs,
                "mt_random":  mt_random,
                "mt_minword": mt_minword,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(data, f, ensure_ascii=False)

        def make_pbar(desc: str) -> tqdm:
            return tqdm(total=n, desc=desc, unit="file", ncols=80)

        if args.qe:
            refs = None
            print("QE 모드: ref 번역 생략")
        elif refs is not None:
            print(f"  ref: 체크포인트 재사용 ({len(refs)}개)")
        else:
            with make_pbar(f"ref ({args.translator})") as pbar:
                if use_gpt:
                    refs = asyncio.run(translate_whole(hypotheses, args.target_lang, args.src_lang, args.gpt_model, args.api_key, pbar=pbar))
                else:
                    refs = asyncio.run(translate_whole_google(hypotheses, args.target_lang, pbar=pbar))
            save_checkpoint()

        if mt_random is not None:
            print(f"  mt_random: 체크포인트 재사용 ({len(mt_random)}개)")
        else:
            with make_pbar(f"mt_random ({args.translator})") as pbar:
                if use_gpt:
                    mt_random = asyncio.run(translate_segmented_files(
                        segs_random,
                        target_lang=args.target_lang, src_lang=args.src_lang,
                        model=args.gpt_model, api_key=args.api_key, max_context=args.max_context,
                        pbar=pbar,
                    ))
                else:
                    mt_random = asyncio.run(translate_segmented_files_google(segs_random, args.target_lang, pbar=pbar))
            save_checkpoint()

        if mt_minword is not None:
            print(f"  mt_minword: 체크포인트 재사용 ({len(mt_minword)}개)")
        else:
            with make_pbar(f"mt_minword ({args.translator})") as pbar:
                if use_gpt:
                    mt_minword = asyncio.run(translate_segmented_files(
                        segs_minword,
                        target_lang=args.target_lang, src_lang=args.src_lang,
                        model=args.gpt_model, api_key=args.api_key, max_context=args.max_context,
                        pbar=pbar,
                    ))
                else:
                    mt_minword = asyncio.run(translate_segmented_files_google(segs_minword, args.target_lang, pbar=pbar))
            save_checkpoint()

    print(f"\nLoading COMET model: {args.comet_model}")
    from comet import download_model, load_from_checkpoint
    comet_model = load_from_checkpoint(download_model(args.comet_model))

    print("\nRunning COMET (pipeline) ...")
    score_pipeline, scores_pipeline = run_comet(comet_model, hypotheses, mt_pipeline, refs)
    print(f"  pipeline  : {score_pipeline:.4f}")

    print("Running COMET (random) ...")
    score_random, scores_random = run_comet(comet_model, hypotheses, mt_random, refs)
    print(f"  random    : {score_random:.4f}")

    print("Running COMET (minword) ...")
    score_minword, scores_minword = run_comet(comet_model, hypotheses, mt_minword, refs)
    print(f"  minword   : {score_minword:.4f}")

    print(f"\n=== 결과 요약 ===")
    print(f"  pipeline  : {score_pipeline:.4f}")
    print(f"  minword   : {score_minword:.4f}  (Δ {score_minword - score_pipeline:+.4f} vs pipeline)")
    print(f"  random    : {score_random:.4f}  (Δ {score_random  - score_pipeline:+.4f} vs pipeline)")

    per_file_refs = refs if refs is not None else [None] * n
    output = {
        "comet_model": args.comet_model,
        "mode": "qe" if args.qe else "ref",
        "translator": args.translator,
        **({"gpt_model": args.gpt_model} if use_gpt else {}),
        **({"ref_method": f"whole-hypothesis {args.translator.upper()} translation (no segmentation)"} if not args.qe else {}),
        "target_lang": args.target_lang,
        "num_files": n,
        "segmentation": {
            "min_words": args.min_words,
            "seed": args.seed,
            "avg_seg_count": {
                "pipeline": round(avg_pipeline, 2),
                "random":   round(avg_random,   2),
                "minword":  round(avg_minword,  2),
            },
        },
        "system_scores": {
            "pipeline": score_pipeline,
            "minword":  score_minword,
            "random":   score_random,
        },
        "per_file": [
            {
                "file_id": fid,
                "src": hyp,
                **({} if ref is None else {"ref": ref}),
                "mt_pipeline": mtp,
                "mt_random":   mtr,
                "mt_minword":  mtm,
                "comet_pipeline": sp,
                "comet_random":   sr,
                "comet_minword":  sm,
            }
            for fid, hyp, ref, mtp, mtr, mtm, sp, sr, sm in zip(
                file_ids, hypotheses, per_file_refs,
                mt_pipeline, mt_random, mt_minword,
                scores_pipeline, scores_random, scores_minword,
            )
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")

    # 정상 완료 시 체크포인트 삭제
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
