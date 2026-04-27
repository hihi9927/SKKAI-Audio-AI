"""
파인튜닝된 모델로 오디오 전사 스크립트.

LoRA 어댑터 방식:
    python transcribe_finetuned.py \
        --checkpoint ../../Qwen3-ASR/finetuning/qwen3-asr-finetuning-out/checkpoint-600 \
        --base_model Qwen/Qwen3-ASR-1.7B \
        --output results/eval_clean_finetuned.json

Merge 모델 방식 (--checkpoint 불필요):
    python transcribe_finetuned.py \
        --base_model ../Qwen3-ASR-1.7B-en-merged \
        --output results/eval_clean_merged.json

DailyTalk test.jsonl (영어):
    python transcribe_finetuned.py \
        --base_model ../Qwen3-ASR-1.7B-en-merged \
        --eval_json ../finetuning/data/DailyTalk/test.jsonl \
        --output results/dailytalk_test_merged.json
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
import torch

# qwen_asr 모듈 경로 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "Qwen3-ASR"))

from qwen_asr import Qwen3ASRModel


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """PCM 또는 WAV 파일을 float32 numpy 배열로 로드."""
    if path.endswith(".wav"):
        import wave
        with wave.open(path, "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    else:
        raw = np.fromfile(path, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def load_jsonl(path: str) -> list[dict]:
    """JSONL 파일을 list[dict]로 로드. audio/text 키를 file/text로 정규화."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # "language English<asr_text>실제텍스트" 형식 파싱
            text = item.get("text", "")
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1].replace("<SEG>", "").strip()
            records.append({"file": item["audio"], "text": text, "_abs_path": True})
    return records


def load_model(base_model: str, checkpoint: str | None):
    # 로컬 경로면 절대경로로 변환 (HuggingFace repo ID 오인 방지)
    if os.sep in base_model or base_model.startswith("."):
        base_model = os.path.abspath(base_model)
    if checkpoint and (os.sep in checkpoint or checkpoint.startswith(".")):
        checkpoint = os.path.abspath(checkpoint)

    # 로컬 디렉토리면 TRANSFORMERS_OFFLINE으로 HF hub repo ID 검증 완전 우회
    _prev_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if os.path.isdir(base_model):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    try:
        if checkpoint:
            from peft import PeftModel
            print(f"[1/3] base model 로드: {base_model}")
            asr_wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map="auto")
            model = asr_wrapper.model

            print(f"[2/3] LoRA adapter 로드: {checkpoint}")
            model = PeftModel.from_pretrained(model, checkpoint)
            model = model.merge_and_unload()
            model.eval()
            asr_wrapper.model = model
        else:
            print(f"[1/1] merge 모델 직접 로드: {base_model}")
            asr_wrapper = Qwen3ASRModel.from_pretrained(base_model, dtype=dtype, device_map="auto")
            asr_wrapper.model.eval()
    finally:
        if _prev_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_offline

    return asr_wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="LoRA 체크포인트 경로. 생략하면 --base_model을 merge 모델로 직접 로드")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--language", type=str, default="English",
                   help="전사 언어 (예: English, Korean)")
    EVAL_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "evaluation/KsponSpeech"))
    p.add_argument("--eval_json", type=str,
                   default=os.path.join(EVAL_ROOT, "transcribe/eval_clean.json"))
    p.add_argument("--audio_dir", type=str,
                   default=os.path.join(EVAL_ROOT, "data/eval_clean"))
    p.add_argument("--output", type=str,
                   default=os.path.join(EVAL_ROOT, "finetuned_results/eval_clean.json"))
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n", type=int, default=0, help="처리할 샘플 수 (0=전체)")
    p.add_argument("--pad_silence", type=float, default=0.0,
                   help="오디오 뒤에 붙일 무음 길이 (초). 예: 0.5")
    args = p.parse_args()

    is_jsonl = args.eval_json.endswith(".jsonl")
    if is_jsonl:
        data = load_jsonl(args.eval_json)
    else:
        with open(args.eval_json, encoding="utf-8") as f:
            data = json.load(f)["data"]

    if args.n > 0:
        data = data[:args.n]

    asr_wrapper = load_model(args.base_model, args.checkpoint)
    model_label = args.checkpoint or args.base_model

    print(f"전사 시작: {len(data)}개 샘플  언어={args.language}\n" + "=" * 60)

    records = []
    for i, entry in enumerate(data):
        file_id = entry["file"]
        gt = entry["text"]

        if entry.get("_abs_path"):
            audio_path = file_id
        elif file_id.endswith(".wav"):
            audio_path = os.path.join(args.audio_dir, file_id)
        else:
            audio_path = os.path.join(args.audio_dir, file_id + ".pcm")

        if not os.path.exists(audio_path):
            print(f"[{i+1}/{len(data)}] 파일 없음: {audio_path}")
            records.append({"file": file_id, "text": gt, "finetuned_text": None, "error": "file_not_found"})
            continue

        wav = load_audio(audio_path, args.sr)
        if args.pad_silence > 0:
            silence = np.zeros(int(args.pad_silence * args.sr), dtype=np.float32)
            wav = np.concatenate([wav, silence])
        results = asyncio.run(asr_wrapper.transcribe([(wav, args.sr)], language=args.language))
        pred = results[0].text.strip()

        records.append({"file": file_id, "text": gt, "finetuned_text": pred})
        print(f"[{i+1}/{len(data)}] {os.path.basename(file_id)}")
        print(f"  GT  : {gt}")
        print(f"  PRED: {pred}")
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "stats": {"checkpoint": model_label, "base_model": args.base_model, "language": args.language},
        "data": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
