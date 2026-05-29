"""
TowerInstruct-7B-v0.1 로드 및 번역 테스트 스크립트

Usage:
    python test_tower_instruct.py                        # fp16 로드 (기본)
    python test_tower_instruct.py --load-in-4bit         # 4-bit 양자화
    python test_tower_instruct.py --load-in-8bit         # 8-bit 양자화
    python test_tower_instruct.py --with-qwen            # Qwen3-ASR도 함께 로드 (동시 탑재 테스트)
"""

import argparse
import time
import torch

MODEL_ID = "Unbabel/TowerInstruct-7B-v0.1"

LANG_MAP = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

TEST_CASES = [
    {"src": "en", "tgt": "ko", "text": "Hello, how are you doing today?"},
    {"src": "en", "tgt": "ja", "text": "The conference starts at nine in the morning."},
    {"src": "ko", "tgt": "en", "text": "안녕하세요, 오늘 회의는 몇 시에 시작하나요?"},
]


def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return allocated, reserved, total
    return 0, 0, 0


def load_tower(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"\n{'='*60}")
    print(f"Loading TowerInstruct-7B-v0.1")
    print(f"  Mode: {'4-bit' if args.load_in_4bit else '8-bit' if args.load_in_8bit else 'fp16'}")
    print(f"{'='*60}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()

    elapsed = time.time() - t0
    alloc, reserved, total = get_vram_usage()
    print(f"  Loaded in {elapsed:.1f}s | VRAM: {alloc:.1f}GB allocated / {total:.1f}GB total")
    return model, tokenizer


def load_qwen(args):
    from qwen_asr import Qwen3ASRModel

    qwen_model_id = getattr(args, "qwen_model", "/home/ubuntu/models/Qwen3-ASR-1.7B")
    print(f"\n{'='*60}")
    print(f"Loading Qwen3-ASR-1.7B from {qwen_model_id}")
    print(f"{'='*60}")

    t0 = time.time()
    alloc_before, _, total = get_vram_usage()
    model = Qwen3ASRModel.from_pretrained(qwen_model_id)
    elapsed = time.time() - t0
    alloc_after, reserved, _ = get_vram_usage()
    print(f"  Loaded in {elapsed:.1f}s | VRAM delta: +{alloc_after - alloc_before:.1f}GB (total {alloc_after:.1f}GB / {total:.1f}GB)")
    return model


def translate(model, tokenizer, src_lang, tgt_lang, text, max_new_tokens=256):
    src_name = LANG_MAP.get(src_lang, src_lang)
    tgt_name = LANG_MAP.get(tgt_lang, tgt_lang)

    prompt = (
        f"Translate the following text from {src_name} to {tgt_name}.\n"
        f"{src_name}: {text}\n"
        f"{tgt_name}:"
    )
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated = output_ids[0][input_ids.shape[-1]:]
    translation = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return translation, elapsed


def run_translation_tests(model, tokenizer):
    print(f"\n{'='*60}")
    print("Translation Tests")
    print(f"{'='*60}")
    for tc in TEST_CASES:
        result, elapsed = translate(model, tokenizer, tc["src"], tc["tgt"], tc["text"])
        print(f"\n  [{tc['src']} → {tc['tgt']}] ({elapsed:.2f}s)")
        print(f"  Input : {tc['text']}")
        print(f"  Output: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--with-qwen", action="store_true", help="Qwen3-ASR도 함께 로드")
    parser.add_argument("--qwen-model", default="/home/ubuntu/models/Qwen3-ASR-1.7B")
    parser.add_argument("--skip-translation", action="store_true", help="번역 테스트 건너뜀")
    args = parser.parse_args()

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        _, _, total = get_vram_usage()
        print(f"VRAM total: {total:.1f}GB")

    # Qwen3-ASR 먼저 로드 (with-qwen 옵션)
    qwen_model = None
    if args.with_qwen:
        qwen_model = load_qwen(args)

    # TowerInstruct 로드
    tower_model, tokenizer = load_tower(args)

    # 최종 VRAM 상태
    alloc, reserved, total = get_vram_usage()
    print(f"\n{'='*60}")
    print(f"Final VRAM: {alloc:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total")
    if args.with_qwen and qwen_model is not None:
        print("Both models loaded successfully!")
    print(f"{'='*60}")

    # 번역 테스트
    if not args.skip_translation:
        run_translation_tests(tower_model, tokenizer)

    print("\nDone.")


if __name__ == "__main__":
    main()
