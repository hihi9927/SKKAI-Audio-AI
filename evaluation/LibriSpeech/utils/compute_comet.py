"""
COMET evaluation for LibriSpeech translation results.

COMET inputs (per file_id):
  src  – ASR hypothesis (raw English)
  ref  – translation of hypothesis segments with rolling context (gold reference)
  mt   – pipeline real-time translations concatenated per file_id (seg_translation)

Usage:
    # Google Translate ref (default)
    python evaluation/LibriSpeech/utils/compute_comet.py \
        --metric-json evaluation/LibriSpeech/results/finetuned_silence(1.0.3)/full/run_05/metric.json

    # GPT ref with context (matches pipeline translation environment)
    python evaluation/LibriSpeech/utils/compute_comet.py \
        --metric-json evaluation/LibriSpeech/results/finetuned_silence(1.0.3)/full/run_05/metric.json \
        --translator gpt --api-key sk-...

    # Reference-free QE (src + mt only, no ref needed)
    python evaluation/LibriSpeech/utils/compute_comet.py \
        --metric-json evaluation/LibriSpeech/results/finetuned_silence_gpt(1.0.4)/full/run_09/metric.json \
        --qe
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Qwen3-ASR" / "examples"))
sys.path.insert(0, str(PROJECT_ROOT))
from streaming_websocket_server import google_translate_async  # noqa: E402

_LANG_NAMES = {"ko": "Korean", "ja": "Japanese", "zh": "Chinese", "en": "English"}


async def translate_texts(texts: list[str], target_lang: str = "ko") -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [google_translate_async(session, t, target_lang) for t in texts]
        results = await asyncio.gather(*tasks)
    return [r[0] for r in results]


async def translate_files_with_context(
    file_segments: list[list[str]],
    target_lang: str = "ko",
    src_lang: str = "en",
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    max_context: int = 5,
) -> list[str]:
    """각 파일의 segment 텍스트를 컨텍스트를 유지하며 순차 번역 후 concat.

    파이프라인의 GPTTranslator + _segment_history 방식과 동일한 환경으로 번역.
    파일 간 병렬, 파일 내 세그먼트는 순차 처리.
    """
    from core.correct_and_trans import GPTTranslator

    translator = GPTTranslator(api_key=api_key, model=model, max_context=max_context)
    src_lang_name = _LANG_NAMES.get(src_lang, src_lang)
    sem = asyncio.Semaphore(10)

    async def translate_file(segments: list[str]) -> str:
        async with sem:
            context: deque[tuple[str, str]] = deque(maxlen=max_context)
            translations = []
            for seg in segments:
                if not seg.strip():
                    continue
                _, translation, _ = await translator.correct_and_translate(
                    text=seg,
                    source_lang_name=src_lang_name,
                    target_lang_code=target_lang,
                    context=list(context) if context else None,
                )
                translations.append(translation)
                context.append((seg, translation))
            return " ".join(translations)

    return await asyncio.gather(*[translate_file(segs) for segs in file_segments])


def run_comet(model, srcs, mts, refs=None):
    if refs is not None:
        data = [{"src": s, "ref": r, "mt": m} for s, r, m in zip(srcs, refs, mts)]
    else:
        data = [{"src": s, "mt": m} for s, m in zip(srcs, mts)]
    output = model.predict(data, batch_size=8, gpus=1, progress_bar=True)
    return {"system_score": output.system_score, "segment_scores": output.scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-json", required=True)
    parser.add_argument("--target-lang", default="ko")
    parser.add_argument("--src-lang", default="en", help="원문 언어 코드 (GPT context 번역 시 사용)")
    parser.add_argument("--comet-model", default=None, help="COMET 모델 (기본값: QE 모드 시 wmt22-cometkiwi-da, 일반 모드 시 wmt22-comet-da)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--qe", action="store_true", help="Reference-free QE 모드: src + mt만 사용, ref 번역 생략")
    parser.add_argument(
        "--translator", choices=["google", "gpt"], default="google",
        help="ref 번역에 사용할 번역기 (default: google). gpt는 파이프라인과 동일한 컨텍스트 반영 번역 사용",
    )
    parser.add_argument("--gpt-model", default="gpt-5.4-mini", help="--translator gpt 시 사용할 모델")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--max-context", type=int, default=5, help="GPT context 번역 시 유지할 직전 세그먼트 수")
    args = parser.parse_args()

    if args.comet_model is None:
        args.comet_model = "Unbabel/wmt22-cometkiwi-da" if args.qe else "Unbabel/wmt22-comet-da"

    metric_path = Path(args.metric_json)
    with open(metric_path) as f:
        metric = json.load(f)

    raw = metric.get("raw_results", [])
    if not raw:
        sys.exit("No raw_results found in metric.json")

    file_ids, hypotheses, seg_translations, file_segments = [], [], [], []
    for entry in raw:
        segs = entry.get("segment_metrics", [])
        seg_trans = " ".join(
            s["translation"] for s in segs if s.get("translation", "").strip()
        )
        file_ids.append(entry["file_id"])
        hypotheses.append(entry["hypothesis"])
        seg_translations.append(seg_trans)
        file_segments.append([s["text"] for s in segs if s.get("text", "").strip()])

    n = len(file_ids)
    print(f"Loaded {n} files from {metric_path}")

    if args.qe:
        hyp_translations = None
        print("QE 모드: ref 번역 생략")
    elif args.translator == "gpt":
        print(f"\nTranslating {n} hypothesis texts → {args.target_lang} (COMET ref, translator=gpt) ...")
        hyp_translations = asyncio.run(
            translate_files_with_context(
                file_segments,
                target_lang=args.target_lang,
                src_lang=args.src_lang,
                model=args.gpt_model,
                api_key=args.api_key,
                max_context=args.max_context,
            )
        )
    else:
        print(f"\nTranslating {n} hypothesis texts → {args.target_lang} (COMET ref, translator=google) ...")
        hyp_translations = asyncio.run(translate_texts(hypotheses, args.target_lang))

    print(f"\nLoading COMET model: {args.comet_model}")
    from comet import download_model, load_from_checkpoint
    model_path = download_model(args.comet_model)
    model = load_from_checkpoint(model_path)

    print("\nRunning COMET ...")
    result = run_comet(model, hypotheses, seg_translations, refs=hyp_translations)

    if args.qe:
        comet_inputs = {
            "src": "hypothesis (ASR English)",
            "mt": "pipeline seg translations concatenated per file_id",
        }
        per_file = [
            {"file_id": fid, "src": hyp, "mt": seg_tr, "comet_score": score}
            for fid, hyp, seg_tr, score in zip(
                file_ids, hypotheses, seg_translations, result["segment_scores"]
            )
        ]
    else:
        ref_desc = (
            "Google Translate of full hypothesis"
            if args.translator == "google"
            else f"GPT ({args.gpt_model}) context-aware translation of hypothesis segments (max_context={args.max_context})"
        )
        comet_inputs = {
            "src": "hypothesis (ASR English)",
            "ref": ref_desc,
            "mt": "pipeline seg translations concatenated per file_id",
        }
        per_file = [
            {"file_id": fid, "src": hyp, "ref": ref_tr, "mt": seg_tr, "comet_score": score}
            for fid, hyp, ref_tr, seg_tr, score in zip(
                file_ids, hypotheses, hyp_translations, seg_translations, result["segment_scores"]
            )
        ]

    output = {
        "comet_model": args.comet_model,
        "mode": "qe" if args.qe else "ref",
        **({"ref_translator": args.translator if args.translator == "google" else f"gpt/{args.gpt_model}"} if not args.qe else {}),
        "target_lang": args.target_lang,
        "num_files": n,
        "comet_inputs": comet_inputs,
        "system_score": result["system_score"],
        "per_file": per_file,
    }

    out_path = Path(args.output) if args.output else metric_path.parent / "comet_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nCOMET system score: {result['system_score']:.4f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
