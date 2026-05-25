"""
COMET evaluation for LibriSpeech translation results.

COMET inputs (per file_id):
  src  – ASR hypothesis (raw English)
  ref  – Google Translate of the full hypothesis (used as gold reference)
  mt   – pipeline real-time translations concatenated per file_id (seg_translation)

Usage:
    python evaluation/LibriSpeech/utils/compute_comet.py \
        --metric-json evaluation/LibriSpeech/results/finetuned_silence(1.0.3)/full/run_05/metric.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Qwen3-ASR" / "examples"))
from streaming_websocket_server import google_translate_async  # noqa: E402


async def translate_texts(texts: list[str], target_lang: str = "ko") -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [google_translate_async(session, t, target_lang) for t in texts]
        results = await asyncio.gather(*tasks)
    return [r[0] for r in results]


def run_comet(model, srcs, refs, mts):
    data = [{"src": s, "ref": r, "mt": m} for s, r, m in zip(srcs, refs, mts)]
    output = model.predict(data, batch_size=8, gpus=1, progress_bar=True)
    return {"system_score": output.system_score, "segment_scores": output.scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-json", required=True)
    parser.add_argument("--target-lang", default="ko")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metric_path = Path(args.metric_json)
    with open(metric_path) as f:
        metric = json.load(f)

    raw = metric.get("raw_results", [])
    if not raw:
        sys.exit("No raw_results found in metric.json")

    file_ids, hypotheses, seg_translations = [], [], []
    for entry in raw:
        segs = entry.get("segment_metrics", [])
        seg_trans = " ".join(
            s["translation"] for s in segs if s.get("translation", "").strip()
        )
        file_ids.append(entry["file_id"])
        hypotheses.append(entry["hypothesis"])
        seg_translations.append(seg_trans)

    n = len(file_ids)
    print(f"Loaded {n} files from {metric_path}")

    print(f"\nTranslating {n} hypothesis texts → {args.target_lang} (COMET ref) ...")
    hyp_translations = asyncio.run(translate_texts(hypotheses, args.target_lang))

    print(f"\nLoading COMET model: {args.comet_model}")
    from comet import download_model, load_from_checkpoint
    model_path = download_model(args.comet_model)
    model = load_from_checkpoint(model_path)

    print("\nRunning COMET ...")
    result = run_comet(model, hypotheses, hyp_translations, seg_translations)

    output = {
        "comet_model": args.comet_model,
        "target_lang": args.target_lang,
        "num_files": n,
        "comet_inputs": {
            "src": "hypothesis (ASR English)",
            "ref": "Google Translate of full hypothesis",
            "mt": "pipeline seg translations concatenated per file_id",
        },
        "system_score": result["system_score"],
        "per_file": [
            {
                "file_id": fid,
                "src": hyp,
                "ref": ref_tr,
                "mt": seg_tr,
                "comet_score": score,
            }
            for fid, hyp, ref_tr, seg_tr, score in zip(
                file_ids, hypotheses, hyp_translations, seg_translations,
                result["segment_scores"],
            )
        ],
    }

    out_path = Path(args.output) if args.output else metric_path.parent / "comet_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nCOMET system score: {result['system_score']:.4f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
