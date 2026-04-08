"""
PEFT LoRA 어댑터를 vLLM 호환 형식으로 변환.

PEFT/HuggingFace 학습 시 키 이름:
    base_model.model.thinker.model.layers.X.self_attn.q_proj.lora_A.weight

vLLM 모델의 실제 모듈 경로 (hf_to_vllm_mapper 적용 결과):
    language_model.model.layers.X.self_attn.q_proj  (→ qkv_proj 로 packed)

변환 규칙:
    base_model.model.thinker.model.  →  base_model.model.language_model.model.
    base_model.model.thinker.audio_tower.*  →  제거 (vLLM은 audio_tower LoRA 미지원)

Usage:
    python convert_lora_for_vllm.py \\
        --adapter ../finetuning-out-en-retry/checkpoint-200_final \\
        --output  ../finetuning-out-en-retry/checkpoint-200_final_vllm

    python convert_lora_for_vllm.py \\
        --adapter ../finetuning-out-ko-retry/checkpoint-170_final \\
        --output  ../finetuning-out-ko-retry/checkpoint-170_final_vllm
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def convert_adapter(src_dir: str, dst_dir: str) -> None:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    # ── safetensors 변환 ──────────────────────────────────────────────────
    src_st = os.path.join(src_dir, "adapter_model.safetensors")
    tensors = load_file(src_st)

    converted: dict[str, torch.Tensor] = {}
    skipped_audio = 0
    converted_lm = 0

    for key, val in tensors.items():
        if "thinker.audio_tower." in key:
            skipped_audio += 1
            continue
        new_key = key.replace(
            "base_model.model.thinker.model.",
            "base_model.model.language_model.model.",
        )
        converted[new_key] = val
        converted_lm += 1

    print(f"  LM LoRA 키 변환: {converted_lm}개")
    print(f"  audio_tower LoRA 제거: {skipped_audio}개 (vLLM 미지원)")

    dst_st = os.path.join(dst_dir, "adapter_model.safetensors")
    save_file(converted, dst_st)
    print(f"  저장: {dst_st}")

    # ── adapter_config.json 복사 ──────────────────────────────────────────
    cfg_src = os.path.join(src_dir, "adapter_config.json")
    cfg_dst = os.path.join(dst_dir, "adapter_config.json")
    with open(cfg_src) as f:
        cfg = json.load(f)
    # target_modules에서 audio_tower 관련 항목 제거 (이미 없지만 명시)
    # base_model_name_or_path는 원래 값 유지
    with open(cfg_dst, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── 토크나이저 파일 복사 ──────────────────────────────────────────────
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "added_tokens.json",
        "preprocessor_config.json", "generation_config.json",
        "chat_template.json", "config.json",
    ]
    for fname in tokenizer_files:
        src_f = os.path.join(src_dir, fname)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst_dir, fname))

    # ── seg_embedding.pt 복사 ────────────────────────────────────────────
    for extra in ["seg_embedding.pt", "seg_lm_head.pt"]:
        src_f = os.path.join(src_dir, extra)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst_dir, extra))
            print(f"  복사: {extra}")

    print(f"  완료: {dst_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="원본 PEFT 어댑터 디렉토리")
    p.add_argument("--output", required=True, help="변환된 어댑터 저장 디렉토리")
    args = p.parse_args()

    print(f"변환 시작: {args.adapter}")
    convert_adapter(args.adapter, args.output)
    print("변환 완료")


if __name__ == "__main__":
    main()
