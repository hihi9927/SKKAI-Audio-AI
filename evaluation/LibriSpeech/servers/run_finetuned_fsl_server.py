#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
QWEN3_ROOT = PROJECT_ROOT / "Qwen3-ASR"

sys.path.insert(0, str(QWEN3_ROOT))

from qwen_asr import Qwen3ASRModel


def sanitize_generation_config(model) -> None:
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return

    # Some released Qwen3-ASR configs keep sampling-only fields while
    # do_sample=False, which blocks save_pretrained() validation.
    if getattr(gen_cfg, "do_sample", False):
        return

    for attr in ("temperature", "top_p", "min_p", "typical_p"):
        if hasattr(gen_cfg, attr):
            setattr(gen_cfg, attr, None)

    if hasattr(gen_cfg, "top_k"):
        setattr(gen_cfg, "top_k", None)


def merge_lora(base_model: str, checkpoint: str, merged_dir: str) -> str:
    merged_path = Path(merged_dir)
    done_flag = merged_path / ".merge_complete"

    if done_flag.exists():
        print(f"[merge] Reusing existing merged model: {merged_path}")
        return str(merged_path)

    merged_path.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[1/4] Loading base model: {base_model}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto",
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    print(f"[2/4] Loading LoRA checkpoint: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)

    print("[3/4] Merging adapter into base model")
    model = model.merge_and_unload()
    model.eval()
    sanitize_generation_config(model)

    print(f"[4/4] Saving merged model to: {merged_path}")
    model.save_pretrained(str(merged_path), safe_serialization=True)
    processor.save_pretrained(str(merged_path))
    done_flag.write_text("ok", encoding="utf-8")

    del model
    del processor
    del asr_wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return str(merged_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch LibriSpeech FSL server with a finetuned Qwen3-ASR LoRA checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT
            / "Qwen3-ASR"
            / "finetuning"
            / "qwen3-asr-finetuning-out"
            / "checkpoint-600"
        ),
        help="LoRA checkpoint directory",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "Qwen3-ASR"
            / "finetuning"
            / "merged-checkpoint-600"
        ),
        help="Directory to save merged model",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    merged_dir = merge_lora(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        merged_dir=args.merged_dir,
    )

    server_script = HERE / "streaming_websocket_server_fsl.py"
    cmd = [
        sys.executable,
        str(server_script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        merged_dir,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--no-idle-shutdown",
    ]

    print("[server] Launching FSL server with merged finetuned model")
    print("[server] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
