#!/usr/bin/env python3
"""
ko LoRA adapter를 base model에 병합해 저장하는 스크립트.

사용법:
    python merge_ko.py \
        --base-model /home/ubuntu/models/Qwen3-ASR-1.7B \
        --checkpoint finetuning-out-ko/checkpoint-ko \
        --output merged-ko
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel

HERE = Path(__file__).resolve().parent
QWEN3_ROOT = HERE.parent
sys.path.insert(0, str(QWEN3_ROOT))

from qwen_asr import Qwen3ASRModel


def sanitize_generation_config(model) -> None:
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    if getattr(gen_cfg, "do_sample", False):
        return
    for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k"):
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, None)


def merge_and_save(base_model: str, checkpoint: str, output: str) -> None:
    output_path = Path(output)
    done_flag = output_path / ".merge_complete"

    if done_flag.exists():
        print(f"[merge] 이미 병합된 모델 존재: {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"dtype: {dtype}")

    print(f"[1/4] base model 로드: {base_model}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map="auto")
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    print(f"[2/4] LoRA adapter 로드: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)

    print("[3/4] adapter 병합 (merge_and_unload)")
    model = model.merge_and_unload()
    model.eval()
    sanitize_generation_config(model)

    print(f"[4/4] 병합 모델 저장: {output_path}")
    model.save_pretrained(str(output_path), safe_serialization=True)
    processor.save_pretrained(str(output_path))
    done_flag.write_text("ok", encoding="utf-8")

    del model, processor, asr_wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n완료: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="/home/ubuntu/models/Qwen3-ASR-1.7B")
    p.add_argument("--checkpoint", default=str(HERE / "finetuning-out-ko/checkpoint-ko"))
    p.add_argument("--output", default=str(HERE / "merged-ko"))
    args = p.parse_args()

    merge_and_save(args.base_model, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
