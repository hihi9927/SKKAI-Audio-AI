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


def transcribe_one(asr_wrapper, wav: np.ndarray, sr: int, language: str) -> str:
    import asyncio
    results = asyncio.run(asr_wrapper.transcribe([(wav, sr)], language=language))
    return results[0].text.strip()


def normalize_text(text: str, language: str = "") -> str:
    """WER/CER 계산용 정규화: <SEG> 제거, 숫자→단어, 구두점 제거, 공백 정규화, 영어 소문자."""
    import num2words
    lang_is_en = language.lower() in ('english', 'en')
    lang_code = 'en' if lang_is_en else 'ko'

    text = re.sub(r'<SEG>', ' ', text)

    # 숫자 → 단어 (소수점 포함)
    def replace_num(m):
        s = m.group()
        try:
            if '.' in s:
                return num2words.num2words(float(s), lang=lang_code)
            return num2words.num2words(int(s), lang=lang_code)
        except Exception:
            return s
    text = re.sub(r'\d+\.?\d*', replace_num, text)

    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    if lang_is_en:
        text = text.lower()
    return ' '.join(text.split())


def compute_metrics(gt: str, pred: str, language: str = "") -> dict:
    gt_seg = gt.count(SEG_TOKEN)
    pred_seg = pred.count(SEG_TOKEN)
    gt_norm = normalize_text(gt, language)
    pred_norm = normalize_text(pred, language)
    wer = jiwer.wer(gt_norm, pred_norm) if gt_norm else 1.0
    cer = jiwer.cer(gt_norm, pred_norm) if gt_norm else 1.0
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
        thinker = model.thinker if hasattr(model, "thinker") else model.base_model.model.thinker
        device = thinker.get_input_embeddings().weight.device

        seg_vec = torch.load(seg_emb_path, map_location="cpu")
        with torch.no_grad():
            thinker.get_input_embeddings().weight[seg_id] = seg_vec.to(device)

        lm_head_path = os.path.join(args.checkpoint, "seg_lm_head.pt")
        if os.path.exists(lm_head_path):
            seg_lm = torch.load(lm_head_path, map_location="cpu")
            with torch.no_grad():
                thinker.get_output_embeddings().weight[seg_id] = seg_lm.to(device)

        print(f"  SEG 임베딩 주입 완료 (token_id={seg_id})")
    else:
        print("  [경고] seg_embedding.pt 없음 — SEG 임베딩이 랜덤 초기화 상태")

    if not args.no_merge:
        model = model.merge_and_unload()
    model.eval()
    asr_wrapper.model = model
    return asr_wrapper


def load_existing_records(out_path: str) -> list:
    """기존 결과 파일에서 완료된 record 목록 반환. 파일이 불완전해도 파싱 가능한 만큼 읽음."""
    if not os.path.exists(out_path):
        return []
    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content).get("results", [])
    except json.JSONDecodeError:
        pass
    # 불완전한 파일: 완성된 { } 블록만 추출
    records = []
    depth, start = 0, None
    in_results = False
    for i, c in enumerate(content):
        if not in_results:
            if content[i:i+11] == '"results":':
                in_results = True
            continue
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    records.append(json.loads(content[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return records


def run_single(asr_wrapper, samples, args, out_f, existing_records=None):
    """문장 단위 배치 추론."""
    existing_records = existing_records or []
    done_audios = {os.path.abspath(r["audio"]) for r in existing_records}
    pending = [s for s in samples if os.path.abspath(s["audio"]) not in done_audios]
    records = list(existing_records)

    # 기존 record 먼저 기록
    out_f.write('{"results": [\n')
    for i, record in enumerate(records):
        if i > 0:
            out_f.write(",\n")
        out_f.write(json.dumps(record, ensure_ascii=False, indent=2))
    out_f.flush()

    if existing_records:
        print(f"  [resume] {len(existing_records)}개 기존 결과 로드, {len(pending)}개 남음")

    for i, sample in enumerate(pending):
        wav, _ = librosa.load(sample["audio"], sr=args.sr, mono=True)
        gt = parse_gt(sample["text"])
        pred = transcribe_one(asr_wrapper, wav, args.sr, args.language)
        global_i = len(records)
        m = compute_metrics(gt, pred, args.language)
        record = {"audio": sample["audio"], "gt": gt, "pred": pred, **m}
        records.append(record)
        if global_i > 0:
            out_f.write(",\n")
        out_f.write(json.dumps(record, ensure_ascii=False, indent=2))
        out_f.flush()
        display_i = len(existing_records) + i
        print(f"[{display_i + 1}/{len(samples)}]")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")
        print(f"  WER={m['wer']:.4f}  CER={m['cer']:.4f}  SEG gt={m['gt_seg']} pred={m['pred_seg']} {'✓' if m['seg_match'] else '✗'}")
        print()

    avg_wer = round(sum(r["wer"] for r in records) / len(records), 4)
    avg_cer = round(sum(r["cer"] for r in records) / len(records), 4)
    seg_match_count = sum(1 for r in records if r["seg_match"])
    print(f"Avg WER: {avg_wer}  Avg CER: {avg_cer}  SEG match: {seg_match_count}/{len(records)}")
    return records, {"avg_wer": avg_wer, "avg_cer": avg_cer, "seg_match": f"{seg_match_count}/{len(records)}"}


def run_concat(asr_wrapper, samples, args, out_f, existing_records=None):
    """concat_n개씩 이어붙여 연속 오디오로 추론."""
    existing_records = existing_records or []
    done_groups = len(existing_records)
    n = args.concat_n
    groups = [samples[i:i+n] for i in range(0, len(samples), n)]
    records = list(existing_records)

    out_f.write('{"results": [\n')
    for i, record in enumerate(records):
        if i > 0:
            out_f.write(",\n")
        out_f.write(json.dumps(record, ensure_ascii=False, indent=2))
    out_f.flush()

    if existing_records:
        print(f"  [resume] {done_groups}개 기존 그룹 로드, {len(groups) - done_groups}개 남음")

    for gi, group in enumerate(groups[done_groups:], start=done_groups):
        wavs = [librosa.load(s["audio"], sr=args.sr, mono=True)[0] for s in group]
        gts = [parse_gt(s["text"]) for s in group]
        gt_combined = " ".join(gts)

        combined_wav = concat_wavs_with_silence(wavs, args.sr, args.silence_ms)
        pred = transcribe_one(asr_wrapper, combined_wav, args.sr, args.language)

        m = compute_metrics(gt_combined, pred, args.language)
        record = {
            "group": gi,
            "audios": [s["audio"] for s in group],
            "gt_sentences": gts,
            "gt_combined": gt_combined,
            "pred": pred,
            "silence_ms": args.silence_ms,
            **m,
        }
        records.append(record)
        if len(records) > 1:
            out_f.write(",\n")
        out_f.write(json.dumps(record, ensure_ascii=False, indent=2))
        out_f.flush()

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

    p.add_argument("--resume", action="store_true",
                   help="기존 결과 파일이 있으면 이어서 진행")
    args = p.parse_args()

    args.base_model = os.path.abspath(args.base_model)

    lang_code = LANG_CODE.get(args.language.lower(), args.language.lower()[:2])
    merged_path = args.checkpoint.rstrip("/") + f"_{lang_code}_merged"

    if args.no_merge:
        print(f"[1/3] base model 로드 (no_merge): {args.base_model}")
        asr_wrapper = load_model(args)
    elif os.path.isdir(merged_path):
        print(f"[1/3] 병합 모델 캐시 감지, 직접 로드: {merged_path}")
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        asr_wrapper = Qwen3ASRModel.from_pretrained(merged_path, dtype=dtype, device_map="auto")
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
    if args.output:
        out_path = args.output
    else:
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
        results_dir = os.path.join(os.path.dirname(args.checkpoint.rstrip("/")), "results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{ckpt_name}_eval_{mode}.json")
    existing_records = []
    if args.resume and os.path.exists(out_path):
        existing_records = load_existing_records(out_path)
        print(f"  [resume] 기존 파일 감지: {len(existing_records)}개 record 로드")

    print(f"[3/3] {len(samples)}개 샘플 추론 시작 (모드: {mode})\n결과 저장: {out_path}\n" + "=" * 60)

    with open(out_path, "w", encoding="utf-8") as out_f:
        if mode == "single":
            all_records, agg = run_single(asr_wrapper, samples, args, out_f, existing_records)
            summary = {"mode": "single", **agg}
        else:
            all_records, agg = run_concat(asr_wrapper, samples, args, out_f, existing_records)
            summary = {"mode": "concat", "concat_n": args.concat_n, "silence_ms": args.silence_ms, **agg}
        out_f.write("\n]\n}\n")  # 임시 마무리 (재작성 전)

    # 완료 후 summary를 맨 위에 두고 파일 재작성
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{\n"summary": ')
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write(',\n"results": [\n')
        for i, record in enumerate(all_records):
            if i > 0:
                f.write(",\n")
            f.write(json.dumps(record, ensure_ascii=False, indent=2))
        f.write("\n]\n}\n")

    print("=" * 60)
    print(f"결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
