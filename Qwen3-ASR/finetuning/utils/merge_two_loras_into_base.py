"""
두 LoRA 어댑터를 베이스 모델에 동시에 merge하여 standalone 모델 생성.

PEFT / vLLM 변환 어댑터 모두 지원 (자동 감지).
vLLM 어댑터는 language_model.model. → thinker.model. 로 자동 변환 후 merge.
GPU 불필요 (CPU only, vRAM 0).

Usage:
    python merge_two_loras_into_base.py \\
        --base_model  ../Qwen3-ASR-1.7B-lora-ready \\
        --adapter_a   ../checkpoint_en_vllm \\
        --adapter_b   ../checkpoint_ko_vllm \\
        --output      ../Qwen3-ASR-1.7B-en-ko-merged

    # KO 어댑터 가중치를 1.2배로 보정 (norm 불균형 보정 시)
    python merge_two_loras_into_base.py \\
        --base_model  ../Qwen3-ASR-1.7B-lora-ready \\
        --adapter_a   ../checkpoint_en_vllm \\
        --adapter_b   ../checkpoint_ko_vllm \\
        --output      ../Qwen3-ASR-1.7B-en-ko-merged \\
        --scale_a 1.0 --scale_b 1.2
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

STRIP_PREFIX = "base_model.model."

# vLLM 변환 시 thinker.model. → language_model.model. 로 바뀜
_VLLM_PREFIX   = "language_model.model."
_PEFT_PREFIX   = "thinker.model."


def load_all_shards(model_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        tensors: dict[str, torch.Tensor] = {}
        for sf in shard_files:
            tensors.update(load_file(os.path.join(model_dir, sf)))
        return tensors, weight_map
    else:
        return load_file(os.path.join(model_dir, "model.safetensors")), None


def save_shards(tensors: dict[str, torch.Tensor], weight_map: dict[str, str] | None, out_dir: str) -> None:
    if weight_map is None:
        save_file(tensors, os.path.join(out_dir, "model.safetensors"))
        return
    file_to_keys: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        file_to_keys.setdefault(fname, []).append(key)
    for fname, keys in file_to_keys.items():
        shard = {k: tensors[k] for k in keys if k in tensors}
        save_file(shard, os.path.join(out_dir, fname))
        print(f"  저장: {fname} ({len(shard)} tensors)")
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"format": "pt"}, "weight_map": weight_map}, f, indent=2)


def detect_format(tensors: dict[str, torch.Tensor]) -> str:
    """'peft' or 'vllm' 반환."""
    for k in tensors:
        stripped = k[len(STRIP_PREFIX):] if k.startswith(STRIP_PREFIX) else k
        if stripped.startswith(_VLLM_PREFIX):
            return "vllm"
        if stripped.startswith(_PEFT_PREFIX):
            return "peft"
    return "unknown"


def load_and_normalize_adapter(
    adapter_dir: str,
    extra_scale: float,
) -> tuple[dict[str, torch.Tensor], float, str]:
    """어댑터 로드 → vLLM이면 PEFT 키로 변환 → audio_tower 제거.

    반환: (정규화된 텐서 dict, lora_scale * extra_scale, 포맷)
    """
    st_path  = os.path.join(adapter_dir, "adapter_model.safetensors")
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    tensors  = load_file(st_path)
    with open(cfg_path) as f:
        cfg = json.load(f)

    lora_scale = (cfg["lora_alpha"] / cfg["r"]) * extra_scale
    fmt = detect_format(tensors)

    result: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        # base_model.model. 제거
        new_k = k[len(STRIP_PREFIX):] if k.startswith(STRIP_PREFIX) else k
        # vLLM → PEFT 키 변환
        if fmt == "vllm":
            new_k = new_k.replace(_VLLM_PREFIX, _PEFT_PREFIX)
        # audio_tower 제외
        if "audio_tower." in new_k:
            continue
        result[new_k] = v

    return result, lora_scale, fmt


def detect_base_prefix(adapter_keys: list[str], base_keys: set[str]) -> str:
    """베이스 모델에 붙어 있는 추가 prefix 탐지 (예: 'model.')."""
    for ak in adapter_keys:
        if ".lora_A." not in ak and ".lora_B." not in ak:
            continue
        candidate = ak.rsplit(".lora_", 1)[0] + ".weight"
        if candidate in base_keys:
            return ""
        for prefix in ("model.", "thinker.", "model.thinker."):
            if (prefix + candidate) in base_keys:
                return prefix
    return ""


def apply_adapter(
    base_tensors: dict[str, torch.Tensor],
    adapter_tensors: dict[str, torch.Tensor],
    lora_scale: float,
    extra_prefix: str,
    label: str,
) -> tuple[int, int]:
    """base_tensors에 어댑터 delta를 인플레이스로 더함. (merged, skipped) 반환."""
    layer_paths = {
        k.replace(".lora_A.weight", "")
        for k in adapter_tensors if k.endswith(".lora_A.weight")
    }
    merged = skipped = 0
    for path in sorted(layer_paths):
        key_A = path + ".lora_A.weight"
        key_B = path + ".lora_B.weight"
        if key_A not in adapter_tensors or key_B not in adapter_tensors:
            skipped += 1
            continue
        base_key = extra_prefix + path + ".weight"
        if base_key not in base_tensors:
            print(f"  [{label}] 베이스 키 없음: {base_key}")
            skipped += 1
            continue
        A = adapter_tensors[key_A].float()
        B = adapter_tensors[key_B].float()
        delta = lora_scale * (B @ A)
        orig_dtype = base_tensors[base_key].dtype
        base_tensors[base_key] = (base_tensors[base_key].float() + delta).to(orig_dtype)
        merged += 1
    return merged, skipped


# ── 메인 ─────────────────────────────────────────────────────────────────────

def merge(
    base_model: str,
    adapter_a: str,
    adapter_b: str,
    output: str,
    scale_a: float,
    scale_b: float,
    update_seg_embedding: bool,
    seg_adapter: str,
) -> None:
    base_model = os.path.abspath(base_model)
    adapter_a  = os.path.abspath(adapter_a)
    adapter_b  = os.path.abspath(adapter_b)
    output     = os.path.abspath(output)

    if os.path.exists(output):
        print(f"[!] 출력 경로 이미 존재: {output}")
        ans = input("덮어쓸까요? [y/N] ").strip().lower()
        if ans != "y":
            print("중단")
            return
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    # ── [1/6] 베이스 모델 설정 파일 복사 ────────────────────────────────────
    print(f"[1/6] 베이스 모델 설정 파일 복사: {base_model} → {output}")
    for item in os.listdir(base_model):
        src = os.path.join(base_model, item)
        dst = os.path.join(output, item)
        if item.endswith(".safetensors") or item == "model.safetensors.index.json":
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print("  완료 (가중치 제외)")

    # ── [2/6] 어댑터 로드 & 정규화 ──────────────────────────────────────────
    print(f"[2/6] 어댑터 A 로드: {adapter_a}")
    tensors_a, lora_scale_a, fmt_a = load_and_normalize_adapter(adapter_a, scale_a)
    print(f"  포맷={fmt_a}, 유효 scale={lora_scale_a:.4f}, tensors={len(tensors_a)}")

    print(f"      어댑터 B 로드: {adapter_b}")
    tensors_b, lora_scale_b, fmt_b = load_and_normalize_adapter(adapter_b, scale_b)
    print(f"  포맷={fmt_b}, 유효 scale={lora_scale_b:.4f}, tensors={len(tensors_b)}")

    # ── [3/6] 베이스 모델 가중치 로드 ───────────────────────────────────────
    print("[3/6] 베이스 모델 가중치 로드 (CPU)")
    base_tensors, weight_map = load_all_shards(base_model)
    print(f"  베이스: {len(base_tensors)} tensors")

    # ── [4/6] 베이스 prefix 감지 ─────────────────────────────────────────────
    base_key_set = set(base_tensors.keys())
    all_adapter_keys = list(tensors_a.keys()) + list(tensors_b.keys())
    lora_keys = [k for k in all_adapter_keys if ".lora_A." in k or ".lora_B." in k]
    extra_prefix = detect_base_prefix(lora_keys, base_key_set)
    if extra_prefix:
        print(f"  베이스 prefix 감지: '{extra_prefix}' (자동 보정)")

    # ── [5/6] LoRA merge ──────────────────────────────────────────────────────
    print("[5/6] LoRA merge")
    merged_a, skipped_a = apply_adapter(base_tensors, tensors_a, lora_scale_a, extra_prefix, "A")
    print(f"  어댑터 A: {merged_a}개 merge, {skipped_a}개 건너뜀")
    merged_b, skipped_b = apply_adapter(base_tensors, tensors_b, lora_scale_b, extra_prefix, "B")
    print(f"  어댑터 B: {merged_b}개 merge, {skipped_b}개 건너뜀")

    # ── seg_embedding 업데이트 (옵션) ─────────────────────────────────────────
    if update_seg_embedding:
        src = seg_adapter if seg_adapter else adapter_a
        seg_pt    = os.path.join(src, "seg_embedding.pt")
        seg_lm_pt = os.path.join(src, "seg_lm_head.pt")
        if os.path.exists(seg_pt):
            from transformers import AutoTokenizer
            tok    = AutoTokenizer.from_pretrained(output)
            seg_id = tok.convert_tokens_to_ids("<SEG>")
            seg_vec = torch.load(seg_pt, map_location="cpu").float()
            print(f"  seg_embedding 업데이트: token_id={seg_id}, 소스={os.path.basename(src)}")
            embed_key = next(
                (k for k in base_tensors if "embed_tokens.weight" in k and "thinker" in k),
                next((k for k in base_tensors if "embed_tokens.weight" in k), None),
            )
            if embed_key:
                base_tensors[embed_key][seg_id] = seg_vec.to(base_tensors[embed_key].dtype)
                print(f"  embed_tokens 업데이트: {embed_key}")
            if os.path.exists(seg_lm_pt):
                seg_lm_vec = torch.load(seg_lm_pt, map_location="cpu").float()
                lm_key = next((k for k in base_tensors if "lm_head.weight" in k), None)
                if lm_key:
                    base_tensors[lm_key][seg_id] = seg_lm_vec.to(base_tensors[lm_key].dtype)
                    print(f"  lm_head 업데이트: {lm_key}")
        else:
            print(f"  [경고] seg_embedding.pt 없음 ({src}), 건너뜀")

    # ── [6/6] 저장 ────────────────────────────────────────────────────────────
    print("[6/6] merge된 가중치 저장")
    save_shards(base_tensors, weight_map, output)
    print(f"\n완료: {output}")
    print(f"  어댑터 A ({fmt_a}, scale={scale_a}) + 어댑터 B ({fmt_b}, scale={scale_b}) → 베이스에 적용됨")
    print("  사용법: --model 인자에 이 경로를 지정하면 어댑터 없이 바로 추론 가능")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="두 LoRA 어댑터를 베이스 모델에 동시 merge")
    p.add_argument("--base_model",  required=True, help="lora-ready 베이스 모델 디렉토리")
    p.add_argument("--adapter_a",   required=True, help="첫 번째 어댑터 디렉토리")
    p.add_argument("--adapter_b",   required=True, help="두 번째 어댑터 디렉토리")
    p.add_argument("--output",      required=True, help="merge된 모델 저장 경로")
    p.add_argument("--scale_a",     type=float, default=1.0,
                   help="어댑터 A 추가 스케일 (기본: 1.0)")
    p.add_argument("--scale_b",     type=float, default=1.0,
                   help="어댑터 B 추가 스케일 (기본: 1.0)")
    p.add_argument("--update_seg_embedding", action="store_true",
                   help="seg_embedding.pt로 <SEG> 토큰 embedding 업데이트")
    p.add_argument("--seg_adapter", default=None,
                   help="seg_embedding 소스 어댑터 (생략 시 adapter_a 사용)")
    args = p.parse_args()

    merge(
        base_model=args.base_model,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        output=args.output,
        scale_a=args.scale_a,
        scale_b=args.scale_b,
        update_seg_embedding=args.update_seg_embedding,
        seg_adapter=args.seg_adapter,
    )


if __name__ == "__main__":
    main()
