"""
LoRA 어댑터를 lora-ready 베이스 모델에 merge하여 standalone 모델 생성.

safetensors를 직접 조작하므로 GPU 불필요 (CPU only, vRAM 0).

Usage:
    python merge_lora_into_base.py \\
        --base_model  ../Qwen3-ASR-1.7B-lora-ready \\
        --adapter     ../finetuning-out-en-retry/checkpoint-200_final \\
        --output      ../Qwen3-ASR-1.7B-en-merged

    # seg_embedding을 EN 체크포인트 것으로 덮어쓰고 싶으면 (기본값: lora-ready에 있는 것 유지)
    python merge_lora_into_base.py \\
        --base_model  ../Qwen3-ASR-1.7B-lora-ready \\
        --adapter     ../finetuning-out-en-retry/checkpoint-200_final \\
        --output      ../Qwen3-ASR-1.7B-en-merged \\
        --update_seg_embedding
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def load_all_shards(model_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    """모든 safetensors 샤드를 로드. (tensors, weight_map_or_None) 반환."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]

        # 파일별로 로드 후 합치기
        shard_files = sorted(set(weight_map.values()))
        tensors: dict[str, torch.Tensor] = {}
        for sf in shard_files:
            tensors.update(load_file(os.path.join(model_dir, sf)))
        return tensors, weight_map
    else:
        single = os.path.join(model_dir, "model.safetensors")
        return load_file(single), None


def save_shards(
    tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str] | None,
    out_dir: str,
) -> None:
    """원본 shard 구조에 맞춰 safetensors 저장."""
    if weight_map is None:
        save_file(tensors, os.path.join(out_dir, "model.safetensors"))
        return

    # 파일별로 분리
    file_to_keys: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        file_to_keys.setdefault(fname, []).append(key)

    for fname, keys in file_to_keys.items():
        shard = {k: tensors[k] for k in keys if k in tensors}
        save_file(shard, os.path.join(out_dir, fname))
        print(f"  저장: {fname} ({len(shard)} tensors)")

    # index.json 재생성 (메타데이터 포함)
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata": {"format": "pt"}, "weight_map": weight_map}, f, indent=2)


def detect_base_prefix(adapter_keys: list[str], base_keys: set[str]) -> str:
    """
    어댑터 키(base_model.model. 제거 후)와 베이스 모델 키를 비교해
    베이스 모델에 붙어있는 추가 prefix를 탐지.
    예: adapter → 'thinker.model.layers.0.q_proj'
        base    → 'model.thinker.model.layers.0.q_proj'
        → prefix = 'model.'
    """
    for ak in adapter_keys:
        # lora_A / lora_B suffix 제거
        if ".lora_A." in ak or ".lora_B." in ak:
            base_key_candidate = ak.rsplit(".lora_", 1)[0] + ".weight"
        else:
            continue

        # 베이스 키에서 직접 매칭 시도
        if base_key_candidate in base_keys:
            return ""

        # prefix 후보 탐색
        for prefix in ("model.", "thinker.", "model.thinker."):
            if (prefix + base_key_candidate) in base_keys:
                return prefix

    return ""


# ── 메인 ─────────────────────────────────────────────────────────────────────

def merge(base_model: str, adapter: str, output: str, update_seg_embedding: bool) -> None:
    base_model = os.path.abspath(base_model)
    adapter    = os.path.abspath(adapter)
    output     = os.path.abspath(output)

    if os.path.exists(output):
        print(f"[!] 출력 경로 이미 존재: {output}")
        ans = input("덮어쓸까요? [y/N] ").strip().lower()
        if ans != "y":
            print("중단")
            return
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    # ── [1/5] 베이스 모델 파일 복사 ─────────────────────────────────────────
    print(f"[1/5] 베이스 모델 파일 복사: {base_model} → {output}")
    for item in os.listdir(base_model):
        src = os.path.join(base_model, item)
        dst = os.path.join(output, item)
        if item.endswith(".safetensors") or item == "model.safetensors.index.json":
            continue  # 가중치는 나중에 merge 후 저장
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print("  완료 (가중치 제외)")

    # ── [2/5] 어댑터 config 로드 ────────────────────────────────────────────
    print("[2/5] 어댑터 config 로드")
    adapter_cfg_path = os.path.join(adapter, "adapter_config.json")
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    lora_r     = adapter_cfg["r"]
    lora_alpha = adapter_cfg["lora_alpha"]
    scale      = lora_alpha / lora_r
    print(f"  r={lora_r}, alpha={lora_alpha}, scale={scale:.4f}")

    # ── [3/5] 가중치 로드 ───────────────────────────────────────────────────
    print("[3/5] 베이스 모델 가중치 로드 (CPU)")
    base_tensors, weight_map = load_all_shards(base_model)
    print(f"  베이스: {len(base_tensors)} tensors")

    print("  어댑터 로드")
    adapter_st = os.path.join(adapter, "adapter_model.safetensors")
    adapter_tensors = load_file(adapter_st)
    print(f"  어댑터: {len(adapter_tensors)} tensors")

    # ── [4/5] LoRA merge ────────────────────────────────────────────────────
    print("[4/5] LoRA merge")

    # 어댑터 키에서 'base_model.model.' prefix 제거
    # 예: base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.weight
    #  → thinker.model.layers.0.self_attn.q_proj.lora_A.weight
    STRIP_PREFIX = "base_model.model."

    stripped_adapter: dict[str, torch.Tensor] = {}
    for k, v in adapter_tensors.items():
        new_k = k[len(STRIP_PREFIX):] if k.startswith(STRIP_PREFIX) else k
        stripped_adapter[new_k] = v

    # audio_tower LoRA 제외 (베이스 모델에 해당 키 없음)
    stripped_adapter = {
        k: v for k, v in stripped_adapter.items()
        if "audio_tower." not in k
    }

    # 베이스 모델과의 prefix 차이 자동 감지
    base_key_set = set(base_tensors.keys())
    adapter_lm_keys = [k for k in stripped_adapter if ".lora_A." in k or ".lora_B." in k]
    extra_prefix = detect_base_prefix(adapter_lm_keys, base_key_set)
    if extra_prefix:
        print(f"  베이스 prefix 감지: '{extra_prefix}' (자동 보정)")

    # 레이어별로 쌍을 찾아 merge
    # lora_A: (r, in_features), lora_B: (out_features, r)
    # delta = scale * B @ A
    layer_paths = set()
    for k in stripped_adapter:
        if ".lora_A." in k:
            layer_paths.add(k.replace(".lora_A.weight", ""))

    merged_count = 0
    skipped_count = 0
    for path in sorted(layer_paths):
        key_A = path + ".lora_A.weight"
        key_B = path + ".lora_B.weight"

        if key_A not in stripped_adapter or key_B not in stripped_adapter:
            skipped_count += 1
            continue

        # 베이스 모델의 대응 weight 키
        base_key = extra_prefix + path + ".weight"
        if base_key not in base_tensors:
            print(f"  [경고] 베이스 키 없음: {base_key}")
            skipped_count += 1
            continue

        A = stripped_adapter[key_A].float()  # (r, in)
        B = stripped_adapter[key_B].float()  # (out, r)
        delta = scale * (B @ A)              # (out, in)

        orig_dtype = base_tensors[base_key].dtype
        base_tensors[base_key] = (base_tensors[base_key].float() + delta).to(orig_dtype)
        merged_count += 1

    print(f"  merge 완료: {merged_count}개 레이어, 건너뜀: {skipped_count}개")

    # ── seg_embedding 업데이트 (옵션) ────────────────────────────────────────
    if update_seg_embedding:
        seg_pt = os.path.join(adapter, "seg_embedding.pt")
        seg_lm_pt = os.path.join(adapter, "seg_lm_head.pt")

        if os.path.exists(seg_pt):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(output)
            seg_id = tok.convert_tokens_to_ids("<SEG>")

            seg_vec = torch.load(seg_pt, map_location="cpu").float()
            print(f"  seg_embedding 업데이트: token_id={seg_id}")

            # embed_tokens 찾기
            embed_key = None
            for k in base_tensors:
                if "embed_tokens.weight" in k and "thinker" in k:
                    embed_key = k
                    break
                if "embed_tokens.weight" in k:
                    embed_key = k
            if embed_key:
                base_tensors[embed_key][seg_id] = seg_vec.to(base_tensors[embed_key].dtype)
                print(f"  embed_tokens 업데이트: {embed_key}")

            # lm_head
            if os.path.exists(seg_lm_pt):
                seg_lm_vec = torch.load(seg_lm_pt, map_location="cpu").float()
                lm_key = None
                for k in base_tensors:
                    if "lm_head.weight" in k:
                        lm_key = k
                        break
                if lm_key:
                    base_tensors[lm_key][seg_id] = seg_lm_vec.to(base_tensors[lm_key].dtype)
                    print(f"  lm_head 업데이트: {lm_key}")
        else:
            print("  [경고] seg_embedding.pt 없음, 업데이트 건너뜀")

    # ── [5/5] 저장 ───────────────────────────────────────────────────────────
    print("[5/5] merge된 가중치 저장")
    save_shards(base_tensors, weight_map, output)

    print(f"\n완료: {output}")
    print("사용법: --model 인자에 이 경로를 지정하면 어댑터 없이 바로 추론 가능")


def main():
    p = argparse.ArgumentParser(description="LoRA 어댑터를 lora-ready 베이스 모델에 merge")
    p.add_argument("--base_model", required=True,
                   help="lora-ready 베이스 모델 디렉토리")
    p.add_argument("--adapter", required=True,
                   help="파인튜닝 체크포인트 디렉토리 (adapter_model.safetensors 포함)")
    p.add_argument("--output", required=True,
                   help="merge된 모델 저장 경로")
    p.add_argument("--update_seg_embedding", action="store_true",
                   help="어댑터의 seg_embedding.pt로 베이스의 <SEG> embedding을 덮어씀 "
                        "(기본: lora-ready에 있는 평균 embedding 유지)")
    args = p.parse_args()

    merge(args.base_model, args.adapter, args.output, args.update_seg_embedding)


if __name__ == "__main__":
    main()
