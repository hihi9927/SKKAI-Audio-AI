# coding=utf-8
"""
Qwen3-ASR Fine-tuning with <SEG> Special Token
- 공식 finetuning 코드 기반으로 수정
- <SEG> 스페셜 토큰 추가
- LoRA 적용 (GPU 메모리 절약)

입력 데이터 형식 (jsonl, 한 줄에 하나씩):
    {"audio": "path/to/KsponSpeech_E00001.wav", "text": "어 일단은 억지로<SEG>진실된 마음으로"}
    {"audio": "path/to/KsponSpeech_E00004.wav", "text": "응 근데 오늘 일단<SEG>뭐 가는 거고"}

data.json을 jsonl로 변환하는 스크립트는 하단 convert_to_jsonl() 참고
"""

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
SEG_TOKEN = "<SEG>"


# ─────────────────────────────────────────────
# 1. 데이터 변환 유틸 (data.json → train.jsonl)
# ─────────────────────────────────────────────

def convert_to_jsonl(
    data_json_path: str,
    audio_dir: str,
    output_jsonl_path: str,
    audio_ext: str = ".wav",
    eval_ratio: float = 0.05,
):
    """
    data.json을 Trainer가 읽을 수 있는 jsonl 형식으로 변환합니다.

    사용법:
        convert_to_jsonl("data.json", "./audios", "train.jsonl")
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    missing = 0
    for item in raw["data"]:
        audio_path = os.path.join(audio_dir, item["file"] + audio_ext)
        if not os.path.exists(audio_path):
            missing += 1
            continue

        # <seg> → <SEG> 정규화
        seg_text = item["seg_text"].replace("<seg>", SEG_TOKEN)
        samples.append({"audio": audio_path, "text": seg_text})

    if missing:
        print(f"[WARN] {missing} audio file(s) not found, skipped.")

    # train / eval 분리
    n_eval = max(1, int(len(samples) * eval_ratio))
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    train_path = output_jsonl_path
    eval_path = output_jsonl_path.replace(".jsonl", "_eval.jsonl")

    for path, data in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] Train: {len(train_samples)} → {train_path}")
    print(f"[INFO] Eval : {len(eval_samples)} → {eval_path}")
    return train_path, eval_path


# ─────────────────────────────────────────────
# 2. <SEG> 스페셜 토큰 추가
# ─────────────────────────────────────────────

def add_seg_token(processor, model):
    """
    <SEG> 토큰을 추가하고 임베딩 테이블을 resize합니다.
    새 토큰 임베딩은 기존 평균값으로 초기화합니다.
    """
    tokenizer = processor.tokenizer

    if SEG_TOKEN in tokenizer.get_vocab():
        print(f"[INFO] {SEG_TOKEN} already exists (id={tokenizer.convert_tokens_to_ids(SEG_TOKEN)})")
        return

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [SEG_TOKEN]}
    )
    print(f"[INFO] Added {num_added} token(s). New vocab size: {len(tokenizer)}")

    # 임베딩 테이블 resize
    model.resize_token_embeddings(len(tokenizer))

    # 새 토큰을 기존 임베딩 평균값으로 초기화 (랜덤보다 학습 안정적)
    with torch.no_grad():
        input_emb = model.get_input_embeddings()
        avg = input_emb.weight[:-num_added].mean(dim=0)
        input_emb.weight[-num_added:] = avg

        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            avg_out = output_emb.weight[:-num_added].mean(dim=0)
            output_emb.weight[-num_added:] = avg_out

    seg_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    print(f"[INFO] {SEG_TOKEN} token ID: {seg_id}")


# ─────────────────────────────────────────────
# 3. LoRA 적용
# ─────────────────────────────────────────────

def apply_lora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    LLM 부분에만 LoRA를 적용합니다.
    - AuT 인코더: freeze 유지
    - embed_tokens / lm_head: modules_to_save로 full 학습
      (새로 추가된 <SEG> 임베딩이 제대로 학습되도록)
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # <SEG> 임베딩 학습을 위해 전체 저장
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# 4. 공식 코드에서 가져온 유틸 함수들
# ─────────────────────────────────────────────

def patch_outer_forward(model):
    """공식 코드와 동일: thinker.forward로 라우팅"""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`."
        )

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step, best_path = None, None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step, best_path = step, path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
    return _preprocess


# ─────────────────────────────────────────────
# 5. 데이터 콜레이터 (공식 코드와 동일)
# ─────────────────────────────────────────────

@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        # prefix 부분은 loss 계산에서 제외 (-100)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ─────────────────────────────────────────────
# 6. 커스텀 Trainer & Callback (공식 코드와 동일)
# ─────────────────────────────────────────────

class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json", "generation_config.json",
        "preprocessor_config.json", "processor_config.json",
        "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json",
        "merges.txt", "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)
        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


# ─────────────────────────────────────────────
# 7. 인자 파싱
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SEG Finetuning")

    # 데이터 변환 모드
    p.add_argument("--convert_data", action="store_true",
                   help="data.json → train.jsonl 변환 후 종료")
    p.add_argument("--data_json", type=str, default="data.json")
    p.add_argument("--audio_dir", type=str, default="./audios")
    p.add_argument("--audio_ext", type=str, default=".wav")

    # 모델 & 데이터
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="train_eval.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen3_asr_seg_out")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # 학습
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    # 재시작
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


# ─────────────────────────────────────────────
# 8. 메인
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 데이터 변환 모드 ──
    if args.convert_data:
        convert_to_jsonl(
            data_json_path=args.data_json,
            audio_dir=args.audio_dir,
            output_jsonl_path=args.train_file,
            audio_ext=args.audio_ext,
        )
        print("[INFO] Conversion done. Run without --convert_data to start training.")
        return

    # ── 모델 로드 ──
    print(f"\n[STEP 1] Loading model: {args.model_path}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # ── <SEG> 토큰 추가 ──
    print(f"\n[STEP 2] Adding {SEG_TOKEN} special token...")
    add_seg_token(processor, model)

    # ── LoRA 적용 ──
    print(f"\n[STEP 3] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # ── 데이터 로드 ──
    print(f"\n[STEP 4] Loading dataset from {args.train_file}")
    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            **({"validation": args.eval_file} if args.eval_file and os.path.exists(args.eval_file) else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    # ── 콜레이터 & TrainingArguments ──
    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args.sr)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        do_eval=("validation" in ds),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
    )

    # ── 학습 시작 ──
    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    print(f"\n[STEP 5] Start training...")
    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)

    print(f"\n[STEP 6] Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.tokenizer.save_pretrained(args.output_dir)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()