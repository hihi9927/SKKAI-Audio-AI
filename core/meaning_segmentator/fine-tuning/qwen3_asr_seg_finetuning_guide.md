# Qwen3-ASR `<SEG>` 토큰 파인튜닝 가이드

## 개요

Qwen3-ASR을 `<SEG>` 토큰 기반 음성 분절 예측 태스크에 맞게 파인튜닝하는 가이드입니다.

- **베이스 모델**: Qwen3-ASR-0.6B / 1.7B
- **파인튜닝 방식**: LoRA (Low-Rank Adaptation)
- **추가 토큰**: `<SEG>` (스페셜 토큰으로 등록)
- **프레임워크**: Hugging Face Trainer + PEFT

---

## 1. 파인튜닝 방법론

### 1-1. `<SEG>` 스페셜 토큰 추가

`<SEG>`를 일반 텍스트가 아닌 **스페셜 토큰**으로 등록합니다.

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
model.resize_token_embeddings(len(tokenizer))
```

토크나이저에 새 토큰이 추가되면 vocab 크기가 1 늘어납니다.  
모델의 임베딩 테이블도 그에 맞게 resize해야 인덱스 에러가 발생하지 않습니다.

```
resize 전: 임베딩 테이블 [vocab_size × hidden_dim]
resize 후: 임베딩 테이블 [(vocab_size + 1) × hidden_dim]
                                            ↑ <SEG> 행 추가
```

새로 추가된 `<SEG>` 임베딩은 **기존 토큰 임베딩의 평균값**으로 초기화합니다 (랜덤 초기화보다 학습이 안정적).

### 1-2. LoRA 적용

GPU 메모리 제약으로 인해 전체 파인튜닝 대신 LoRA를 사용합니다.

| 구성 요소 | 처리 방식 | 이유 |
|---|---|---|
| AuT 인코더 | **Freeze** | 4천만 시간으로 학습된 음성 표현, 건드리지 않음 |
| Qwen3 LLM (attention) | **LoRA** | 메모리 효율적 파인튜닝 |
| `embed_tokens` / `lm_head` | **Full 학습** | `<SEG>` 임베딩이 제대로 학습되도록 |

LoRA 기본 설정:

```
r           = 16
lora_alpha  = 32
lora_dropout = 0.05
target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### 1-3. 출력 포맷

모델은 다음 형식으로 출력하도록 학습됩니다.

```
language Korean<asr_text>어 일단은 억지로<SEG>진실된 마음으로<SEG>공감을 시킬 수 있을까
```

---

## 2. 데이터 형식

### 원본 데이터 (data.json)

```json
{
  "data": [
    {
      "file": "KsponSpeech_E00001",
      "text": "어 일단은 억지로 과장해서 ...",
      "seg_text": "어 일단은 억지로 과장해서 이렇게 하는 것보다<seg>진실된 마음으로 이걸 어떻게 전달할 수 있을까<seg>공감을 시킬 수 있을까 해서 좀",
      "full_trans": "...",
      "seg_trans": "...",
      "comet_score": 0.7696
    }
  ]
}
```

- `file`: 오디오 파일명 (확장자 제외)
- `seg_text`: `<seg>` 가 삽입된 정답 텍스트 → 이 필드만 사용
- 나머지 필드(`text`, `full_trans`, `seg_trans`, `comet_score`)는 학습에 사용되지 않음

### 변환 후 데이터 (train.jsonl)

학습 전 `--convert_data` 플래그로 자동 변환됩니다.

```jsonl
{"audio": "./audios/KsponSpeech_E00001.wav", "text": "어 일단은 억지로<SEG>진실된 마음으로<SEG>공감을 시킬 수 있을까"}
{"audio": "./audios/KsponSpeech_E00004.wav", "text": "응 근데 오늘 일단 밥 먹고<SEG>뭐 가는 거고 안 되면"}
```

변환 시 자동으로 처리되는 것들:
- `file` + `audio_dir` + `audio_ext` → 전체 오디오 경로 조합
- `<seg>` → `<SEG>` 대소문자 정규화
- 오디오 파일이 없는 항목 자동 스킵
- 전체 데이터를 train (95%) / eval (5%) 로 자동 분리

---

## 3. 사용 순서

### 환경 설치

```bash
pip install qwen-asr peft transformers datasets librosa
```

### Step 1. 데이터 변환

```bash
python finetune_qwen3_asr_seg_v2.py \
    --convert_data \
    --data_json data.json \
    --audio_dir ./audios \
    --audio_ext .wav \
    --train_file train.jsonl
```

실행 결과:
```
train.jsonl       ← 학습 데이터 (95%)
train_eval.jsonl  ← 검증 데이터 (5%)
```

### Step 2. 학습

```bash
python finetune_qwen3_asr_seg_v2.py \
    --model_path Qwen/Qwen3-ASR-0.6B \
    --train_file train.jsonl \
    --eval_file train_eval.jsonl \
    --output_dir ./output \
    --epochs 3 \
    --batch_size 4 \
    --grad_acc 8
```

주요 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-ASR-0.6B` | 베이스 모델 경로 |
| `--train_file` | `train.jsonl` | 학습 데이터 경로 |
| `--eval_file` | `train_eval.jsonl` | 검증 데이터 경로 |
| `--output_dir` | `./qwen3_asr_seg_out` | 체크포인트 저장 경로 |
| `--epochs` | `3` | 학습 에폭 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--grad_acc` | `8` | Gradient accumulation |
| `--lr` | `2e-5` | 학습률 |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--resume` | `0` | `1`로 설정 시 마지막 체크포인트에서 재시작 |

### Step 3. 학습 재시작 (중단된 경우)

```bash
python finetune_qwen3_asr_seg_v2.py \
    --train_file train.jsonl \
    --output_dir ./output \
    --resume 1
```

---

## 4. 디렉토리 구조 예시

```
project/
├── data.json                    ← 원본 데이터
├── audios/
│   ├── KsponSpeech_E00001.wav
│   ├── KsponSpeech_E00004.wav
│   └── ...
├── train.jsonl                  ← 변환 후 자동 생성
├── train_eval.jsonl             ← 변환 후 자동 생성
├── finetune_qwen3_asr_seg_v2.py
└── output/                      ← 체크포인트 저장
    ├── checkpoint-200/
    ├── checkpoint-400/
    └── ...
```
