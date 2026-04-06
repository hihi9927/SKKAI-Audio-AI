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
import asyncio
import json
import os
import re

import jiwer
import librosa
import numpy as np
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel
from transformers import AutoTokenizer


def parse_gt(text: str) -> str:
    """'language Korean<asr_text>...' 에서 실제 전사 텍스트만 추출."""
    m = re.search(r"<asr_text>(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


SEG_TOKEN = "<SEG>"

LANG_CODE = {
    "korean": "ko",
    "english": "en"
}


def compute_metrics(gt: str, pred: str) -> dict:
    wer = jiwer.wer(gt, pred)
    cer = jiwer.cer(gt, pred)
    gt_seg = gt.count(SEG_TOKEN)
    pred_seg = pred.count(SEG_TOKEN)
    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "gt_seg": gt_seg,
        "pred_seg": pred_seg,
        "seg_match": gt_seg == pred_seg,
    }


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

    # 체크포인트 토크나이저 로드 (<SEG> 포함)
    ckpt_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if len(ckpt_tokenizer) > len(asr_wrapper.processor.tokenizer):
        asr_wrapper.processor.tokenizer = ckpt_tokenizer

    # vocab 크기 불일치 시 resize
    tokenizer = asr_wrapper.processor.tokenizer
    if model.thinker.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.thinker.resize_token_embeddings(len(tokenizer))

    print(f"[2/3] LoRA adapter 로드: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    seg_emb_path = os.path.join(args.checkpoint, "seg_embedding.pt")
    if os.path.exists(seg_emb_path):
        seg_id = tokenizer.convert_tokens_to_ids("<SEG>")
        device = model.get_input_embeddings().weight.device

        seg_vec = torch.load(seg_emb_path, map_location="cpu")
        with torch.no_grad():
            model.get_input_embeddings().weight[seg_id] = seg_vec.to(device)

        lm_head_path = os.path.join(args.checkpoint, "seg_lm_head.pt")
        if os.path.exists(lm_head_path):
            seg_lm = torch.load(lm_head_path, map_location="cpu")
            with torch.no_grad():
                model.get_output_embeddings().weight[seg_id] = seg_lm.to(device)

        print(f"  SEG 임베딩 주입 완료 (token_id={seg_id})")
    else:
        print("  [경고] seg_embedding.pt 없음 — SEG 임베딩이 랜덤 초기화 상태")

    if not args.no_merge:
        model = model.merge_and_unload()
    model.eval()
    asr_wrapper.model = model
    return asr_wrapper


def run_single(asr_wrapper, samples, args):
    """문장 단위 개별 추론."""
    records = []
    for i, sample in enumerate(samples):
        wav, _ = librosa.load(sample["audio"], sr=args.sr, mono=True)
        gt = parse_gt(sample["text"])

        results = asyncio.run(asr_wrapper.transcribe([(wav, args.sr)], language=args.language))
        pred = results[0].text.strip()

        m = compute_metrics(gt, pred)
        records.append({"audio": sample["audio"], "gt": gt, "pred": pred, **m})
        print(f"[{i+1}/{len(samples)}]")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")
        print(f"  WER={m['wer']:.4f}  CER={m['cer']:.4f}  SEG gt={m['gt_seg']} pred={m['pred_seg']} {'✓' if m['seg_match'] else '✗'}")
        print()

    avg_wer = round(sum(r["wer"] for r in records) / len(records), 4)
    avg_cer = round(sum(r["cer"] for r in records) / len(records), 4)
    seg_match_count = sum(1 for r in records if r["seg_match"])
    print(f"Avg WER: {avg_wer}  Avg CER: {avg_cer}  SEG match: {seg_match_count}/{len(records)}")
    return records, {"avg_wer": avg_wer, "avg_cer": avg_cer, "seg_match": f"{seg_match_count}/{len(records)}"}


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
        results = asyncio.run(asr_wrapper.transcribe([(combined_wav, args.sr)], language=args.language))
        pred = results[0].text.strip()

        m = compute_metrics(gt_combined, pred)
        records.append({
            "group": gi,
            "audios": [s["audio"] for s in group],
            "gt_sentences": gts,
            "gt_combined": gt_combined,
            "pred": pred,
            "silence_ms": args.silence_ms,
            **m,
        })

        print(f"[그룹 {gi+1}/{len(groups)}] ({len(group)}개 문장, {args.silence_ms}ms 무음)")
        for j, gt in enumerate(gts):
            print(f"  GT[{j+1}]: {gt}")
        print(f"  PRED: {pred}")
        print(f"  WER={m['wer']:.4f}  CER={m['cer']:.4f}  SEG gt={m['gt_seg']} pred={m['pred_seg']} {'✓' if m['seg_match'] else '✗'}")
        print()

    avg_wer = round(sum(r["wer"] for r in records) / len(records), 4)
    avg_cer = round(sum(r["cer"] for r in records) / len(records), 4)
    seg_match_count = sum(1 for r in records if r["seg_match"])
    print(f"Avg WER: {avg_wer}  Avg CER: {avg_cer}  SEG match: {seg_match_count}/{len(records)}")
    return records, {"avg_wer": avg_wer, "avg_cer": avg_cer, "seg_match": f"{seg_match_count}/{len(records)}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    _default_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Qwen3-ASR-1.7B")
    p.add_argument("--base_model", type=str, default=_default_base)
    p.add_argument("--val_file", type=str, default="./data/val_split.jsonl")
    p.add_argument("--n", type=int, default=10, help="평가할 샘플 수 (0=전체)")
    p.add_argument("--sr", type=int, default=16000)
    # 연속 오디오 옵션
    p.add_argument("--concat_n", type=int, default=0,
                   help="N개 문장을 이어붙여 한 번에 추론 (0=개별 처리)")
    p.add_argument("--language", type=str, default="Korean",
                   help="추론 언어 (예: Korean, English)")
    p.add_argument("--silence_ms", type=int, default=500,
                   help="이어붙일 때 문장 사이 무음 길이 (ms)")
    p.add_argument("--output", type=str, default="",
                   help="결과 JSON 저장 경로 (미지정 시 checkpoint 경로 기반 자동 생성)")
    p.add_argument("--no_merge", action="store_true",
                   help="adapter를 merge하지 않고 PeftModel 상태로 추론")
    args = p.parse_args()

    args.base_model = os.path.abspath(args.base_model)

    lang_code = LANG_CODE.get(args.language.lower(), args.language.lower()[:2])
    merged_path = args.checkpoint.rstrip("/") + f"_{lang_code}"

    if args.no_merge:
        print(f"[1/3] base model 로드 (no_merge): {args.base_model}")
        asr_wrapper = load_model(args)
    elif os.path.isdir(merged_path):
        print(f"[1/3] 병합 모델 캐시 감지, 직접 로드: {merged_path}")
        asr_wrapper = Qwen3ASRModel.from_pretrained(merged_path)
    else:
        print(f"[1/3] base model 로드: {args.base_model}")
        asr_wrapper = load_model(args)
        print(f"[2.5/3] 병합 모델 저장: {merged_path}")
        asr_wrapper.model.generation_config.temperature = None
        asr_wrapper.model.save_pretrained(merged_path)
        asr_wrapper.processor.save_pretrained(merged_path)

    with open(args.val_file) as f:
        samples = [json.loads(l) for l in f]
    if args.n > 0:
        samples = samples[:args.n]

    mode = "concat" if args.concat_n > 0 else "single"
    print(f"[3/3] {len(samples)}개 샘플 추론 시작 (모드: {mode})\n" + "=" * 60)

    if mode == "single":
        records, agg = run_single(asr_wrapper, samples, args)
        output = {"summary": {"mode": "single", **agg}, "results": records}
    else:
        records, agg = run_concat(asr_wrapper, samples, args)
        output = {"summary": {"mode": "concat", "concat_n": args.concat_n, "silence_ms": args.silence_ms, **agg}, "results": records}

    print("=" * 60)
    out_path = args.output if args.output else args.checkpoint.rstrip("/") + f"_eval_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
