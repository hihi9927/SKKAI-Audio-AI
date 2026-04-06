# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# trainable params: 70,254,592 || all params: 2,108,307,072 || trainable%: 3.3323
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jiwer
import librosa
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
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
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_with_features(processor, sampling_rate=16000):
    """map() 단계에서 오디오 로딩 + mel spectrogram 추출까지 완료해 캐싱."""
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        target = ex["text"]
        eos = processor.tokenizer.eos_token or ""

        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        full_text = prefix_text + target + eos

        wav = load_audio(ex["audio"], sr=sampling_rate)
        inputs = processor(
            text=[full_text],
            audio=[wav],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prefix_inputs = processor(
            text=[prefix_text],
            audio=[wav],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prefix_len = int(prefix_inputs["attention_mask"][0].sum().item())

        result = {
            "input_ids": inputs["input_ids"][0].tolist(),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "input_features": inputs["input_features"][0].tolist(),
            "prefix_len": prefix_len,
        }
        if "feature_attention_mask" in inputs:
            result["feature_attention_mask"] = inputs["feature_attention_mask"][0].tolist()
        return result

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id or 0

        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn_mask_list = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        feat_list = [torch.tensor(f["input_features"]) for f in features]
        prefix_lens = [f["prefix_len"] for f in features]

        max_len = max(t.shape[0] for t in input_ids_list)
        input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id)
            for t in input_ids_list
        ])
        attention_mask = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0)
            for t in attn_mask_list
        ])
        # input_features: [mel_bins, time] - time이 오디오 길이마다 다르므로 패딩
        max_feat_time = max(t.shape[-1] for t in feat_list)
        input_features = torch.stack([
            torch.nn.functional.pad(t, (0, max_feat_time - t.shape[-1]), value=0.0)
            for t in feat_list
        ])

        labels = input_ids.clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "labels": labels,
        }
        if "feature_attention_mask" in features[0]:
            fam_list = [torch.tensor(f["feature_attention_mask"], dtype=torch.long) for f in features]
            max_fam = max(t.shape[0] for t in fam_list)
            result["feature_attention_mask"] = torch.stack([
                torch.nn.functional.pad(t, (0, max_fam - t.shape[0]), value=0)
                for t in fam_list
            ])
        return result


def preprocess_logits_for_metrics(logits, labels):
    """메모리 절약: 전체 logits 대신 argmax token id만 보관."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(processor):
    seg_token_id = processor.tokenizer.convert_tokens_to_ids("<SEG>")
    print(f"[val] seg_token_id = {seg_token_id}")

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred  # (batch, seq_len)

        decoded_preds = []
        decoded_labels = []
        total_tokens = 0
        correct_tokens = 0
        total_seg = 0
        correct_seg = 0

        for pred, label in zip(pred_ids, label_ids):
            # CausalLM shift: logit[i]는 position i+1을 예측
            shifted_pred  = pred[:-1]
            shifted_label = label[1:]

            valid = shifted_label != -100
            if valid.sum() == 0:
                continue

            p_valid = shifted_pred[valid]
            l_valid = shifted_label[valid]

            seg_mask = l_valid == seg_token_id
            non_seg_mask = ~seg_mask

            total_tokens += int(non_seg_mask.sum())
            correct_tokens += int((p_valid[non_seg_mask] == l_valid[non_seg_mask]).sum())
            total_seg += int(seg_mask.sum())
            correct_seg += int((p_valid[seg_mask] == seg_token_id).sum())

            p_str = processor.tokenizer.decode(p_valid, skip_special_tokens=True)
            l_str = processor.tokenizer.decode(l_valid, skip_special_tokens=True)
            decoded_preds.append(p_str)
            decoded_labels.append(l_str)

        if not decoded_labels:
            return {"wer": 1.0, "token_accuracy": 0.0, "seg_accuracy": 0.0}

        wer = jiwer.wer(decoded_labels, decoded_preds)
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        seg_accuracy = correct_seg / total_seg if total_seg > 0 else 0.0
        # label에 실제로 seg_token_id가 있는지, 모델이 뭘 예측하는지 샘플 확인
        for pred, label in zip(pred_ids[:1], label_ids[:1]):
            shifted_pred = pred[:-1]
            shifted_label = label[1:]
            valid = shifted_label != -100
            l_valid = shifted_label[valid]
            p_valid = shifted_pred[valid]
            seg_pos = (l_valid == seg_token_id).nonzero()[0]
            print(f"[val] seg positions in label: {seg_pos[:5].tolist()}")
            if len(seg_pos) > 0:
                print(f"[val] pred at seg pos: {p_valid[seg_pos[:5]].tolist()} (expected {seg_token_id})")
        print(f"[val] total_seg={total_seg}, correct_seg={correct_seg}, seg_accuracy={seg_accuracy}")
        return {
            "wer": round(wer, 4),
            "token_accuracy": round(token_accuracy, 4),
            "seg_accuracy": round(seg_accuracy, 4),
        }

    return compute_metrics


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
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str, tokenizer):
        self.base_model_path = base_model_path
        self._tokenizer = tokenizer

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        # 수정된 tokenizer(<SEG> 포함)로 덮어쓰기
        self._tokenizer.save_pretrained(ckpt_dir)

        # SEG 토큰 임베딩 저장 (LoRA adapter에 포함되지 않으므로 별도 저장)
        model = kwargs.get("model")
        if model is not None:
            seg_id = self._tokenizer.convert_tokens_to_ids("<SEG>")

            emb = model.get_input_embeddings()
            torch.save(emb.weight[seg_id].detach().cpu(), os.path.join(ckpt_dir, "seg_embedding.pt"))

            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight is not emb.weight:
                torch.save(lm_head.weight[seg_id].detach().cpu(), os.path.join(ckpt_dir, "seg_lm_head.pt"))

            # vocab_size 반영된 config.json 저장
            base = model.base_model.model if hasattr(model, "base_model") else model
            if hasattr(base, "thinker"):
                base.thinker.config.save_pretrained(ckpt_dir)
            elif hasattr(base, "config"):
                base.config.save_pretrained(ckpt_dir)

        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="./Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="./data/KSponSpeech/train_split.jsonl")
    p.add_argument("--eval_file", type=str, default="./data/KSponSpeech/val_split.jsonl")
    p.add_argument("--output_dir", type=str, default="./finetuning-out-ko-retry")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=10)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # LoRA
    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # <SEG>를 special token으로 등록 (resume 시에도 처리)
    if "<SEG>" not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
    # vocab 크기 불일치 시 resize (최초 학습 및 resume 모두)
    if model.thinker.get_input_embeddings().weight.shape[0] != len(processor.tokenizer):
        model.thinker.resize_token_embeddings(len(processor.tokenizer))
    seg_id = processor.tokenizer.convert_tokens_to_ids("<SEG>")

    if args_cli.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args_cli.lora_r,
            lora_alpha=args_cli.lora_alpha,
            lora_dropout=args_cli.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)

    # PEFT가 freeze한 이후 SEG 임베딩 unfreeze + gradient hook (resume 포함 항상 등록)
    def _make_seg_only_hook(sid):
        def _hook(grad):
            mask = torch.zeros_like(grad)
            mask[sid] = 1.0
            return grad * mask
        return _hook

    emb = model.thinker.get_input_embeddings()
    emb.weight.requires_grad_(True)
    emb.weight.register_hook(_make_seg_only_hook(seg_id))

    lm_head = model.thinker.get_output_embeddings()
    lm_head.weight.requires_grad_(True)
    lm_head.weight.register_hook(_make_seg_only_hook(seg_id))

    model.print_trainable_parameters()

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(
        make_preprocess_fn_with_features(processor, sampling_rate=args_cli.sr),
        remove_columns=raw_ds["train"].column_names,
    )

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args_cli.eval_file else "no",
        eval_steps=args_cli.save_steps if args_cli.eval_file else None,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="tensorboard",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path, tokenizer=processor.tokenizer)],
        compute_metrics=make_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
