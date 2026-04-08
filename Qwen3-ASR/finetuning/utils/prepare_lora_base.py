"""
vLLM LoRA용 베이스 모델 준비.

베이스 모델에 <SEG> 토큰을 추가하고, EN/KO 체크포인트의
seg_embedding.pt를 평균내어 embedding table에 주입한 뒤 저장.

이렇게 만든 베이스 모델을 streaming_websocket_server.py의 --model로 사용.

Usage:
    python prepare_lora_base.py \\
        --base_model  Qwen/Qwen3-ASR-1.7B \\
        --seg_en      ../finetuning-out-en-retry/checkpoint-200_final/seg_embedding.pt \\
        --seg_ko      ../finetuning-out-ko-retry/checkpoint-170_final/seg_embedding.pt \\
        --output      ../Qwen3-ASR-1.7B-lora-ready
"""
import argparse
import os
import shutil

import torch
from transformers import AutoProcessor, AutoTokenizer


def prepare(base_model: str, seg_en: str, seg_ko: str, output: str) -> None:
    output = os.path.abspath(output)

    if os.path.exists(output):
        print(f"[!] 출력 경로 이미 존재: {output}")
        ans = input("덮어쓸까요? [y/N] ").strip().lower()
        if ans != "y":
            print("중단")
            return
        shutil.rmtree(output)

    # 로컬 경로이면 복사, 아니면 HuggingFace에서 다운로드
    if os.path.isdir(base_model):
        base_model = os.path.abspath(base_model)
        print(f"[1/4] 베이스 모델 복사: {base_model} → {output}")
        shutil.copytree(base_model, output)
    else:
        from huggingface_hub import snapshot_download
        print(f"[1/4] HuggingFace에서 모델 다운로드: {base_model} → {output}")
        snapshot_download(repo_id=base_model, local_dir=output)
        print(f"  다운로드 완료")

    # ── 토크나이저에 <SEG> 추가 ───────────────────────────────────────────
    print("[2/4] 토크나이저에 <SEG> 추가")
    try:
        tok = AutoTokenizer.from_pretrained(output)
    except Exception:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(output, fix_mistral_regex=True)
        tok = proc.tokenizer

    if "<SEG>" not in tok.get_vocab():
        tok.add_tokens(["<SEG>"], special_tokens=False)
        tok.save_pretrained(output)
        print(f"  <SEG> 추가됨 (vocab size: {len(tok)})")
    else:
        print(f"  <SEG> 이미 존재 (token_id={tok.convert_tokens_to_ids('<SEG>')})")

    seg_id = tok.convert_tokens_to_ids("<SEG>")

    # ── SEG embedding 평균 계산 ──────────────────────────────────────────
    print("[3/4] SEG embedding 평균 계산")
    vec_en = torch.load(os.path.abspath(seg_en), map_location="cpu").float()
    vec_ko = torch.load(os.path.abspath(seg_ko), map_location="cpu").float()
    cosine = torch.nn.functional.cosine_similarity(
        vec_en.unsqueeze(0), vec_ko.unsqueeze(0)
    ).item()
    print(f"  EN/KO cosine similarity: {cosine:.4f}")
    seg_avg = ((vec_en + vec_ko) / 2.0)
    print(f"  평균 벡터 shape: {seg_avg.shape}")

    # ── safetensors 로드 후 embedding 주입 ───────────────────────────────
    print("[4/4] 모델 가중치에 SEG embedding 주입")
    from safetensors.torch import load_file, save_file
    import glob, json

    index_path = os.path.join(output, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # embedding 파일 찾기
        embed_key = None
        for k in weight_map:
            if "embed_tokens.weight" in k and "thinker" in k:
                embed_key = k
                break
            if "embed_tokens.weight" in k:
                embed_key = k

        if embed_key is None:
            raise RuntimeError("embed_tokens.weight를 찾을 수 없습니다")

        shard_file = os.path.join(output, weight_map[embed_key])
        print(f"  embedding 파일: {weight_map[embed_key]}")

        tensors = load_file(shard_file)
        emb = tensors[embed_key]

        orig_size = emb.shape[0]
        if emb.shape[0] <= seg_id:
            # vocab 크기 확장
            new_emb = torch.zeros(seg_id + 1, emb.shape[1], dtype=emb.dtype)
            new_emb[:orig_size] = emb
            emb = new_emb
            print(f"  embedding table 확장: {orig_size} → {emb.shape[0]}")

        emb[seg_id] = seg_avg.to(emb.dtype)
        tensors[embed_key] = emb
        save_file(tensors, shard_file)
        print(f"  SEG embedding 주입 완료 (token_id={seg_id})")

        # lm_head도 같은 파일에 있으면 함께 처리
        lm_head_key = None
        for k in weight_map:
            if "lm_head.weight" in k:
                lm_head_key = k
                break
        if lm_head_key and weight_map.get(lm_head_key) == weight_map[embed_key]:
            lm_emb = tensors.get(lm_head_key)
            if lm_emb is not None and lm_emb.shape[0] <= seg_id:
                new_lm = torch.zeros(seg_id + 1, lm_emb.shape[1], dtype=lm_emb.dtype)
                new_lm[:lm_emb.shape[0]] = lm_emb
                new_lm[seg_id] = seg_avg.to(lm_emb.dtype)
                tensors[lm_head_key] = new_lm
                save_file(tensors, shard_file)
                print(f"  lm_head SEG 행 주입 완료")

    else:
        # 단일 model.safetensors
        single = os.path.join(output, "model.safetensors")
        tensors = load_file(single)
        embed_key = next(
            (k for k in tensors if "embed_tokens.weight" in k), None
        )
        if embed_key is None:
            raise RuntimeError("embed_tokens.weight를 찾을 수 없습니다")

        emb = tensors[embed_key]
        orig_size = emb.shape[0]
        if emb.shape[0] <= seg_id:
            new_emb = torch.zeros(seg_id + 1, emb.shape[1], dtype=emb.dtype)
            new_emb[:orig_size] = emb
            emb = new_emb
            print(f"  embedding table 확장: {orig_size} → {emb.shape[0]}")

        emb[seg_id] = seg_avg.to(emb.dtype)
        tensors[embed_key] = emb
        save_file(tensors, single)
        print(f"  SEG embedding 주입 완료 (token_id={seg_id})")

    print(f"\n완료: {output}")
    print("이 경로를 streaming_websocket_server.py의 --model 인자로 사용하세요.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="베이스 모델 (로컬 디렉토리 또는 HuggingFace 모델 ID, 예: Qwen/Qwen3-ASR-1.7B)")
    p.add_argument("--seg_en", required=True, help="EN 체크포인트의 seg_embedding.pt 경로")
    p.add_argument("--seg_ko", required=True, help="KO 체크포인트의 seg_embedding.pt 경로")
    p.add_argument("--output", required=True, help="출력 디렉토리")
    args = p.parse_args()

    prepare(args.base_model, args.seg_en, args.seg_ko, args.output)


if __name__ == "__main__":
    main()
