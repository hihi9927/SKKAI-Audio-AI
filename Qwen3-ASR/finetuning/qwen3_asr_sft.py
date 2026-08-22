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

import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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


def resolve_audio(path: str, base_dir: str) -> str:
    """상대 audio 경로를 매니페스트(jsonl) 위치 기준으로 절대경로화한다.

    매니페스트에 절대경로를 박으면 머신이 바뀔 때마다 전부 재작성해야 한다. 상대경로로
    두되 **CWD 가 아니라 jsonl 자신의 위치** 를 기준으로 푸는 이유는, 같은 매니페스트를
    학습(`finetuning/` 에서 실행)과 평가(리포 루트에서 실행)가 함께 읽기 때문이다.
    이미 절대경로인 값은 손대지 않아 기존 매니페스트와 호환된다.
    """
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def generate_silence_dataset(n_samples: int, output_dir: str, sr: int = 16000, prompt: str = "") -> str:
    """무음(pure silence + white noise) WAV 파일을 생성하고 JSONL 경로를 반환.

    인덱스(silence_00000 ~ silence_{N-1})별로 wav가 이미 있으면 재활용하고 없는 것만 생성:
      - 기존 wav가 요청 개수 이상 → 앞 N개만 잘라서 재활용 (생성 없음)
      - 기존 wav가 요청 개수 미만 → 부족분만 추가 생성
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "silence_data.jsonl")

    # 길이 분포: 0.5s 20% / 1.0s 30% / 2.0s 30% / 3.0s 20%
    durations = [0.5, 1.0, 2.0, 3.0]
    weights   = [0.2, 0.3, 0.3, 0.2]

    records = []
    reused = generated = 0
    for i in range(n_samples):
        fpath = os.path.join(output_dir, f"silence_{i:05d}.wav")
        if os.path.exists(fpath):
            reused += 1
        else:
            dur     = random.choices(durations, weights=weights)[0]
            n_frame = int(sr * dur)
            if i % 2 == 0:
                wav = np.zeros(n_frame, dtype=np.float32)                      # pure silence
            else:
                wav = (np.random.randn(n_frame) * 0.002).astype(np.float32)   # white noise
            sf.write(fpath, wav, sr)
            generated += 1
        records.append({"audio": fpath, "text": "", "prompt": prompt})

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[silence] {n_samples}개 (재활용 {reused} + 생성 {generated}) → {jsonl_path}")
    return jsonl_path


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
            return {"wer": 1.0, "cer": 1.0, "token_accuracy": 0.0, "seg_accuracy": 0.0}

        wer = jiwer.wer(decoded_labels, decoded_preds)
        cer = jiwer.cer(decoded_labels, decoded_preds)
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
            "cer": round(cer, 4),
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

            base = model.base_model.model if hasattr(model, "base_model") else model
            thinker = base.thinker if hasattr(base, "thinker") else base
            emb = thinker.get_input_embeddings()
            torch.save(emb.weight[seg_id].detach().cpu(), os.path.join(ckpt_dir, "seg_embedding.pt"))

            lm_head = thinker.get_output_embeddings()
            if lm_head is not None and lm_head.weight is not emb.weight:
                torch.save(lm_head.weight[seg_id].detach().cpu(), os.path.join(ckpt_dir, "seg_lm_head.pt"))

            # vocab_size를 config.json에 패치 (thinker config로 전체를 덮어쓰지 않도록 json 직접 수정)
            config_path = os.path.join(ckpt_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                new_vocab_size = emb.weight.shape[0]
                if "thinker_config" in cfg:
                    cfg["thinker_config"]["vocab_size"] = new_vocab_size
                else:
                    cfg["vocab_size"] = new_vocab_size
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)

        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="./Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="./data/KSponSpeech/train_split.jsonl")
    p.add_argument("--eval_file", type=str, default="./data/KSponSpeech/val_split.jsonl")
    p.add_argument("--output_dir", type=str, default="./finetuning-out-ko-plus")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_checkpointing", type=int, default=1,
                   help="활성값 메모리 절감(50~70%↓, 속도 20~30%↓). 1=on, 0=off")

    # DataLoader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=20)
    p.add_argument("--save_total_limit", type=int, default=10)

    # Speed / best-model
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="attention 구현 (sdpa 권장, eager는 느림)")
    p.add_argument("--group_by_length", type=int, default=1,
                   help="유사 길이끼리 배치(패딩 낭비 감소). 1=on")
    p.add_argument("--load_best", type=int, default=1,
                   help="학습 종료 시 best(eval_loss) 체크포인트 로드/보존. 1=on")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    # LoRA
    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--adapter_path", type=str, default="",
                   help="기존 LoRA 어댑터 경로. 지정 시 새 LoRA 대신 해당 어댑터를 로드해서 이어서 학습.")

    # Silence augmentation
    p.add_argument("--silence_samples", type=int, default=0,
                   help="무음 학습 데이터 생성 개수 (0 = 사용 안 함)")
    p.add_argument("--silence_dir", type=str, default="./data/silence_generated",
                   help="생성된 무음 WAV 및 JSONL 저장 경로")
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
        adapter_path = (args_cli.adapter_path or "").strip()
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            print(f"[lora] 기존 어댑터 로드: {adapter_path}")
        else:
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
    # audio 가 상대경로면 그 jsonl 이 놓인 디렉터리 기준으로 푼다. 절대경로는 그대로 둬
    # 기존 매니페스트(KSponSpeech 등)와 무음 데이터셋이 영향받지 않는다.
    for _split, _src in (("train", args_cli.train_file), ("validation", args_cli.eval_file)):
        if _split in raw_ds and _src:
            _base = os.path.dirname(os.path.abspath(_src))
            raw_ds[_split] = raw_ds[_split].map(
                lambda ex, b=_base: {"audio": resolve_audio(ex["audio"], b)}
            )

    ds = raw_ds.map(
        make_preprocess_fn_with_features(processor, sampling_rate=args_cli.sr),
        remove_columns=raw_ds["train"].column_names,
    )

    if args_cli.silence_samples > 0:
        default_prompt = raw_ds["train"][0].get("prompt", "") if len(raw_ds["train"]) > 0 else ""
        preprocess_fn = make_preprocess_fn_with_features(processor, sampling_rate=args_cli.sr)

        silence_jsonl = generate_silence_dataset(
            args_cli.silence_samples, args_cli.silence_dir, args_cli.sr, prompt=default_prompt
        )
        silence_raw = load_dataset("json", data_files={"train": silence_jsonl})
        silence_ds = silence_raw.map(preprocess_fn, remove_columns=silence_raw["train"].column_names)
        ds["train"] = concatenate_datasets([ds["train"], silence_ds["train"]]).shuffle(seed=42)
        print(f"[silence] train: {len(ds['train'])}개 "
              f"(원본 {len(ds['train']) - args_cli.silence_samples} + 무음 {args_cli.silence_samples})")

        if "validation" in ds:
            # val 무음 비율을 train과 동일하게 유지 (기존 하드코딩 15% → train 비율)
            _sil_ratio = args_cli.silence_samples / max(1, len(raw_ds["train"]))
            val_silence_n = max(1, round(len(ds["validation"]) * _sil_ratio))
            val_silence_jsonl = generate_silence_dataset(
                val_silence_n,
                os.path.join(args_cli.silence_dir, "val"),
                args_cli.sr,
                prompt=default_prompt,
            )
            val_silence_raw = load_dataset("json", data_files={"train": val_silence_jsonl})
            val_silence_ds = val_silence_raw.map(preprocess_fn, remove_columns=val_silence_raw["train"].column_names)
            ds["validation"] = concatenate_datasets([ds["validation"], val_silence_ds["train"]]).shuffle(seed=42)
            print(f"[silence] val:   {len(ds['validation'])}개 "
                  f"(원본 {len(ds['validation']) - val_silence_n} + 무음 {val_silence_n})")

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        weight_decay=args_cli.weight_decay,
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
        load_best_model_at_end=bool(args_cli.eval_file and args_cli.load_best),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=bool(args_cli.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False} if args_cli.gradient_checkpointing else None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        ignore_data_skip=True,
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
