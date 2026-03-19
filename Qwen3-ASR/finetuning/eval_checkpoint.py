"""
checkpoint-N 으로 validation 샘플 추론 스크립트.

단일 문장 모드 (기본):
    python eval_checkpoint.py \
        --checkpoint ./qwen3-asr-finetuning-out/checkpoint-600 \
        --val_file ./data/val_split.jsonl \
        --n 100

연속 오디오 모드 (--concat_n):
    python eval_checkpoint.py \
        --checkpoint ./qwen3-asr-finetuning-out/checkpoint-600 \
        --val_file ./data/val_split.jsonl \
        --n 100 \
        --concat_n 5 \
        --silence_ms 500
    → 5개 문장을 500ms 무음으로 이어붙여 한 번에 추론
"""
import argparse
import json
import re

import librosa
import numpy as np
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


def parse_gt(text: str) -> str:
    """'language Korean<asr_text>...' 에서 실제 전사 텍스트만 추출."""
    m = re.search(r"<asr_text>(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def concat_wavs_with_silence(wavs: list, sr: int, silence_ms: int) -> np.ndarray:
    """여러 wav를 silence_ms 간격 무음으로 이어붙임."""
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        parts.append(w)
        if i < len(wavs) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def load_model(args):
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
    )
    model = asr_wrapper.model

    print(f"[2/3] LoRA adapter 로드: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.merge_and_unload()
    model.eval()
    asr_wrapper.model = model
    return asr_wrapper


def run_single(asr_wrapper, samples, args):
    """문장 단위 개별 추론."""
    correct = 0
    records = []
    for i, sample in enumerate(samples):
        wav, _ = librosa.load(sample["audio"], sr=args.sr, mono=True)
        gt = parse_gt(sample["text"])

        results = asr_wrapper.transcribe([(wav, args.sr)], language="Korean")
        pred = results[0].text.strip()

        match = pred == gt
        if match:
            correct += 1

        records.append({"audio": sample["audio"], "gt": gt, "pred": pred, "exact_match": match})
        print(f"[{i+1}/{len(samples)}]")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")
        print(f"  {'✓' if match else '✗'}")
        print()

    print(f"Exact Match: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
    return records, correct


def run_concat(asr_wrapper, samples, args):
    """concat_n개씩 이어붙여 연속 오디오로 추론."""
    n = args.concat_n
    groups = [samples[i:i+n] for i in range(0, len(samples), n)]
    records = []

    for gi, group in enumerate(groups):
        wavs = [librosa.load(s["audio"], sr=args.sr, mono=True)[0] for s in group]
        gts = [parse_gt(s["text"]) for s in group]
        gt_combined = " ".join(gts)

        combined_wav = concat_wavs_with_silence(wavs, args.sr, args.silence_ms)
        results = asr_wrapper.transcribe([(combined_wav, args.sr)], language="Korean")
        pred = results[0].text.strip()

        records.append({
            "group": gi,
            "audios": [s["audio"] for s in group],
            "gt_sentences": gts,
            "gt_combined": gt_combined,
            "pred": pred,
            "silence_ms": args.silence_ms,
        })

        print(f"[그룹 {gi+1}/{len(groups)}] ({len(group)}개 문장, {args.silence_ms}ms 무음)")
        for j, gt in enumerate(gts):
            print(f"  GT[{j+1}]: {gt}")
        print(f"  PRED: {pred}")
        print()

    return records, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base_model", type=str, default="./Qwen3-ASR-1.7B")
    p.add_argument("--val_file", type=str, default="./data/val_split.jsonl")
    p.add_argument("--n", type=int, default=10, help="평가할 샘플 수")
    p.add_argument("--sr", type=int, default=16000)
    # 연속 오디오 옵션
    p.add_argument("--concat_n", type=int, default=0,
                   help="N개 문장을 이어붙여 한 번에 추론 (0=개별 처리)")
    p.add_argument("--silence_ms", type=int, default=500,
                   help="이어붙일 때 문장 사이 무음 길이 (ms)")
    p.add_argument("--output", type=str, default="",
                   help="결과 JSON 저장 경로 (미지정 시 checkpoint 경로 기반 자동 생성)")
    args = p.parse_args()

    print(f"[1/3] base model 로드: {args.base_model}")
    asr_wrapper = load_model(args)

    with open(args.val_file) as f:
        samples = [json.loads(l) for l in f][:args.n]

    mode = "concat" if args.concat_n > 0 else "single"
    print(f"[3/3] {len(samples)}개 샘플 추론 시작 (모드: {mode})\n" + "=" * 60)

    if mode == "single":
        records, correct = run_single(asr_wrapper, samples, args)
        summary = {"mode": "single", "exact_match": f"{correct}/{len(samples)}", "results": records}
    else:
        records, _ = run_concat(asr_wrapper, samples, args)
        summary = {"mode": "concat", "concat_n": args.concat_n, "silence_ms": args.silence_ms, "results": records}

    print("=" * 60)
    out_path = args.output if args.output else args.checkpoint.rstrip("/") + f"_eval_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
