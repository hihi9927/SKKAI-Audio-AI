"""
파인튜닝된 모델로 eval_clean.json 오디오 전사 스크립트.

LoRA 어댑터 방식:
    python transcribe_finetuned.py \
        --checkpoint ../../Qwen3-ASR/finetuning/qwen3-asr-finetuning-out/checkpoint-600 \
        --base_model Qwen/Qwen3-ASR-1.7B \
        --output results/eval_clean_finetuned.json

Merge 모델 방식 (--checkpoint 불필요):
    python transcribe_finetuned.py \
        --base_model ../Qwen3-ASR-1.7B-en-merged \
        --output results/eval_clean_merged.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# qwen_asr 모듈 경로 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "Qwen3-ASR"))

from qwen_asr import Qwen3ASRModel


def load_pcm(path: str, sr: int = 16000) -> np.ndarray:
    """16-bit PCM 파일을 float32 numpy 배열로 로드."""
    raw = np.fromfile(path, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def load_model(base_model: str, checkpoint: str | None):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if checkpoint:
        from peft import PeftModel
        print(f"[1/3] base model 로드: {base_model}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map="auto")
        model = asr_wrapper.model

        print(f"[2/3] LoRA adapter 로드: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        model.eval()
        asr_wrapper.model = model
    else:
        print(f"[1/1] merge 모델 직접 로드: {base_model}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map="auto")
        asr_wrapper.model.eval()

    return asr_wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="LoRA 체크포인트 경로. 생략하면 --base_model을 merge 모델로 직접 로드")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--language", type=str, default="English",
                   help="전사 언어 (예: English, Korean)")
    EVAL_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "evaluation/KsponSpeech"))
    p.add_argument("--eval_json", type=str,
                   default=os.path.join(EVAL_ROOT, "transcribe/eval_clean.json"))
    p.add_argument("--audio_dir", type=str,
                   default=os.path.join(EVAL_ROOT, "data/eval_clean"))
    p.add_argument("--output", type=str,
                   default=os.path.join(EVAL_ROOT, "finetuned_results/eval_clean.json"))
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n", type=int, default=0, help="처리할 샘플 수 (0=전체)")
    args = p.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        data = json.load(f)["data"]

    if args.n > 0:
        data = data[:args.n]

    asr_wrapper = load_model(args.base_model, args.checkpoint)
    model_label = args.checkpoint or args.base_model

    print(f"전사 시작: {len(data)}개 샘플  언어={args.language}\n" + "=" * 60)

    records = []
    for i, entry in enumerate(data):
        file_id = entry["file"]
        gt = entry["text"]
        audio_path = os.path.join(args.audio_dir, file_id + ".pcm")

        if not os.path.exists(audio_path):
            print(f"[{i+1}/{len(data)}] 파일 없음: {audio_path}")
            records.append({**entry, "finetuned_text": None, "error": "file_not_found"})
            continue

        wav = load_pcm(audio_path, args.sr)
        results = asr_wrapper.transcribe([(wav, args.sr)], language=args.language)
        pred = results[0].text.strip()

        records.append({**entry, "finetuned_text": pred})
        print(f"[{i+1}/{len(data)}] {file_id}")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.eval_json, encoding="utf-8") as f:
        original = json.load(f)
    output = {
        **original,
        "stats": {**original.get("stats", {}), "checkpoint": model_label, "base_model": args.base_model},
        "data": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
