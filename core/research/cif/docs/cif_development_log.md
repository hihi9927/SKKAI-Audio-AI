# CIF Weight Predictor 개발 로그

CIF(Continuous Integrate-and-Fire) 기반 SEG 경계 검출 모듈의 설계·학습 과정 정리.

---

## 목표

Qwen3-ASR 스트리밍 파이프라인에서 SEG 토큰 위치를 사전에 예측해 오디오 청크를 자동 분할.  
실시간 처리를 위해 LM 디코더를 거치지 않고 encoder feature만으로 경계를 검출한다.

---

## 아키텍처 개요

```
mel → Qwen3-ASR audio_tower (frozen / LoRA) → (T, 2048)
    → CIFWeightPredictor
        ├── proj: Linear(2048, hidden)
        ├── conv: Conv1d(hidden, hidden, kernel=7)   ← 로컬 경계 패턴
        ├── out:  Linear(hidden, 1) + Sigmoid        → weights (T,)
        └── qty_head: Linear(hidden,64)→Linear(64,1)+Softplus  → qty_pred (scalar)
```

**경계 검출 방식 (CIF 원래 방식)**

```
acc[t] = Σ w[0..t]
fire at t  ←  floor(acc[t]) > floor(acc[t-1])
```

---

## 레이블 설계

### 시도 1: Gaussian soft label (실패)

- 경계 위치에 Gaussian(σ=2, peak=1) 분포
- **문제**: MSE + Gaussian(peak≈0.08) → 경계 grad ≈ -0.08, 비경계 grad ≈ +0.08 → 기울기 상쇄 → 모델이 상수 출력, 학습 불가

### 시도 2: BCE + peak=1.0 (실패)

- 경계 위치만 1, 나머지 0인 hard label + BCE loss
- **문제**: BCE 최적 Σw ≈ n_segs × T_segment ≈ 8, quantity loss 목표 Σw = n_segs ≈ 1.6 → 충돌로 weight 상한 ~0.35에서 고착

### 최종: Uniform label + MSE (CIF 논문 방식)

```python
label[prev:curr] = 1.0 / n_segment_frames  # 각 세그먼트 내 균등 분배
# 마지막 SEG 이후 무음 구간: label = 0
# sum(label) = n_segs  ← quantity loss 목표와 일치
```

기울기 상쇄 없음. Σw → n_segs 수렴이 자연스럽게 유도됨.

---

## Loss 설계

```
total = λ_qty * (Σw - n_segs)²          ← 로컬 weight 합 제어
      + λ_bnd * MSE(w, label)            ← 경계 위치 분포
      + λ_cnt * (qty_pred - n_segs)²     ← global head 직접 지도
```

λ_qty = λ_bnd = λ_cnt = 1.0 (기본값)

---

## 학습 실험 결과

### v1: frozen encoder, no head (baseline)

| 지표 | 값 |
|---|---|
| best epoch | 33 |
| val_loss | 0.0930 |
| count_acc (Σw) | 0.589 (n_segs=1 기준) |

- n_segs=1: acc=0.589, n_segs=2: acc=0.396, n_segs≥3: acc≈0.2 이하
- **문제**: 모델이 항상 Σw ≈ 1 출력. 오디오 길이와 n_segs는 무상관(짧은 "No."도 n_segs=1)

### v2: frozen encoder + LoRA

| 지표 | 값 |
|---|---|
| best epoch | 29 |
| val_loss | 0.0919 |
| lora_r | 32, lora_alpha=64 |

- val_loss 개선 미미 (0.0930 → 0.0919)
- train/val gap 10배 → 심각한 overfitting
- count_acc는 v1과 유사 수준

### v3: frozen encoder + qty_head

| 지표 | 값 |
|---|---|
| best epoch | 30 |
| val_qty | 0.0988 |
| val_cnt | 0.1036 |

**평가 결과:**

| 지표 | 값 |
|---|---|
| count_acc (Σw 기반) | 0.502 |
| count_mae (Σw 기반) | 0.516 |
| qty_head_acc | **0.830** |
| qty_head_mae | **0.182** |

- qty_head는 n_segs=1 acc 0.810으로 크게 개선
- Σw 기반 경계 검출(count_acc=0.502)은 여전히 부진 → local weights가 qty_head를 따라가지 못함
- n_segs=2 이상은 여전히 낮음 (acc: 0.172 / 0.135 / 0.000)

---

## v3 분석: Σw와 qty_head 괴리 해소 — k-peak 추론

### 발견

v3에서 qty_head는 잘 맞추는데(acc=0.830) CIF 누적 기반 count_acc는 0.502에 머물렀던 원인 분석:

- 모델은 n_segs와 무관하게 **항상 Σw ≈ 1** 출력 (n_segs=1 샘플 52%로 biased)
- weight 곡선 자체에는 경계 근처 **상대적 peak는 이미 존재** → 꺼내는 방법이 문제

### k-peak 추론 전략

qty_head가 예측한 k를 그대로 사용해 weight 곡선 상위 k개 극대점을 경계로 선택.

```python
k = round(qty_pred)
peaks, _ = find_peaks(weights, distance=3)
top_k = peaks[argsort(weights[peaks])[-k:]]
fire_frames = sort(top_k)
```

### eval_cif.py 평가 결과 (test 500샘플)

```
count_acc [CIF]    : 0.502   ← Σw 누적 방식 (기존)
count_acc [head]   : 0.830   ← qty_head 단독
count_acc [kpeak]  : 0.830   ← k-peak (= head와 동일, by construction)
pos_mae   [kpeak]  : 2.00 enc frames ≈ 154ms
```

| n_segs | CIF acc | kpeak acc |
|---|---|---|
| 1 | 0.810 | 0.871 |
| 2 | 0.172 | **0.859** |
| 3 | 0.135 | **0.568** |
| 4+ | 0.000 | 0.000 |

**결론**: weight 곡선에 경계 peak는 이미 존재했다. 문제는 Σw≈1 bias로 1개만 뽑던 것.  
pos_mae=2.0 frames(≈154ms)는 실용 수준.

---

## 태스크 재설계: one-seg 단일화

### 동기

스트리밍 파이프라인에서 모델이 실제로 보는 입력은 항상 **마지막 commit 이후 버퍼**.  
즉 commit → 버퍼 리셋이므로, 모델이 풀어야 할 태스크는 항상 **버퍼 안에서 SEG 1개 찾기**.

멀티-SEG 샘플을 억지로 단일 forward로 처리하려던 게 문제의 근원이었음.

### 데이터 재구성: `one_seg_data/`

멀티-SEG 레코드를 SEG 경계마다 청크로 분할. 각 청크는 항상 n_segs=1.

```
{"audio": "...", "seg_timestamps": [1.44, 2.8]}
  →  {"audio": "...", "start": 0.0,  "end": 1.44, "seg_t": 1.44}
     {"audio": "...", "start": 1.44, "end": 2.8,  "seg_t": 1.36}
```

- 오디오는 실제로 잘라 저장하지 않음. Dataset 로더가 `start`/`end`로 메모리 트림.
- `seg_t`: SEG 위치 (start 기준 상대 좌표)

| 파일 | 원본 records | 변환 후 chunks |
|---|---|---|
| train.jsonl | 9,000 | 14,524 |
| val.jsonl | 500 | 790 |
| test.jsonl | 500 | 793 |

### 단순화된 모델: `train_one_seg.py`

- qty_head 제거 (항상 n_segs=1이므로 불필요)
- Loss: `(Σw - 1)² + MSE(w, label)` (cnt_loss 없음)
- Label: `[0, seg_frame)` 구간에 `1/seg_frame` 균등 분배

```
mel → encoder (frozen) → (T, 2048)
    → CIFWeightPredictor (proj → Conv1d → sigmoid)
    → weights (T,)   [qty_head 없음]
```

추론:
```python
acc[t] = Σw[0..t]
fire ← acc[t] 가 처음으로 1.0 넘는 시점  → commit, 버퍼 리셋
```

### 원본 CIF 논문과의 비교

| | 원본 CIF (Dong & Xu 2020) | 우리 버전 |
|---|---|---|
| 목적 | ASR 토큰 경계 | SEG 경계 검출 |
| boundary_loss | 없음 (decoder 역전파로 implicit 학습) | MSE(w, label) 명시적 지도 |
| 학습 신호 | downstream ASR loss | forced aligner 레이블 |
| decoder | 있음 | 없음 |

decoder 없이 boundary만 뽑아야 하므로 boundary_loss로 직접 지도.

---

## mel mode 실험 (인코더 없는 경량 버전)

### 동기

인코더(2048-dim Transformer)를 거치지 않고 mel spectrogram에서 직접 weight를 예측하는 경량 버전 탐색.

### 구현: `CIFMelPredictor`

```python
class CIFMelPredictor(nn.Module):
    # mel (128, T) → Conv1d 스택 → weights (T',)
    # 인코더 불필요, 추론 빠름
```

`train_one_seg.py`에 `--mode mel` / `--mel-hidden` 인자 추가.

### 결과 및 실패 원인 분석

| 지표 | 값 |
|---|---|
| fire_acc | 0.999 |
| pos_mae | 22.33 frames (≈ 오디오 시작) |

fire_acc는 거의 1인데 pos_mae가 최악 → **항상 오디오 시작점에서 즉시 fire**.

**원인**: uniform label은 클립 전체 길이(global context)를 알아야 per-frame weight 크기를 보정할 수 있다.
- 정답 label[t] = 1 / seg_frame: "seg_frame이 클수록 각 프레임의 weight는 작아야 한다"
- 로컬 Conv1d는 전체 길이를 모름 → 학습 결과로 출력이 상수(≈ 1/mean_seg_frame)로 수렴
- 상수 weight → cumsum이 바로 1.0 돌파 → frame 0에서 fire

---

## Spike Label 추가

uniform label 실패 해결책. 로컬 conv가 학습 가능한 형태로 레이블 재설계.

```python
def make_spike_label(n_frames, seg_frame, sigma=2.0):
    # seg_frame 중심 Gaussian, sum=1
    frames = torch.arange(n_frames, dtype=torch.float32)
    label = torch.exp(-0.5 * ((frames - seg_frame) / sigma) ** 2)
    return label / label.sum()
```

**핵심 차이**: uniform label은 "각 프레임에 적절한 weight를 배분하라"는 global normalization 과제. spike label은 "이 프레임이 경계처럼 들리는지"를 묻는 local discrimination 과제 → 로컬 conv로 학습 가능.

`train_one_seg.py`에 `--label-type spike` / `--label-sigma` 인자 추가.

---

## Encoder Linear Probe (`probe_encoder.py`, `probe_linear.py`)

### 가설

"Qwen3-ASR 인코더는 SEG 위치를 이미 암묵적으로 알고 있다."

디코더가 <SEG>를 뱉는다면, 그 정보는 인코더 feature에 이미 들어 있어야 한다.

### probe_encoder.py: 판별력 분석

SEG 인접 프레임(pos)과 비인접 프레임(neg)의 인코더 특징을 비교.

```
discriminability[d] = |mean_pos[d] - mean_neg[d]| / pooled_std[d]
```

**결과**

| 지표 | 값 |
|---|---|
| max_disc | 1.282 |
| mean_acc (top-20 dims) | 0.737 |

- SEG 정보는 수백 차원에 분산 (단일 차원으로 분리 불가)
- sharp elbow 없음 → PCA/선형 결합으로 접근 필요

### probe_linear.py: Logistic Regression Probe

모든 2048차원을 입력으로 로지스틱 회귀를 학습해 per-frame P(SEG|frame) 추정.

**데이터**: 원본 멀티-SEG 오디오 (`data/train.jsonl`, `data/test.jsonl`)

**추론 방식**:
```python
proba = clf.predict_proba(features)[:, 1]   # per-frame P(SEG)
peaks, _ = find_peaks(proba, height=peak_thr, distance=3)
# peaks → 예측 SEG 경계
```

**초기 결과 (oracle n_gt 방식)**: count_acc=1.000  
→ oracle이 정답 개수를 아는 상황이라 의미 없음

**threshold 방식으로 전환**: `sweep_peak_thr()`로 0.1→0.95 스윕, 최적 threshold 자동 선택.

**주요 구현 사항**

| 기능 | 내용 |
|---|---|
| activation cache | X.npy / y.npy / clips.pkl (18분 인코더 재실행 방지) |
| solver | saga (verbose=1, 에폭별 출력) |
| probe 저장 | `linear_probe.pkl` (clf + peak_thr) |
| `--no-cache`, `--peak-thr` | 캐시 우회 / threshold 수동 지정 |

---

## 파이프라인 통합 검토 (브랜치: `feat/cif-probe-gate`)

### 목표

linear probe를 실제 STiTy 파이프라인에 붙여 인코더 실행 후 SEG 여부를 판단, **fire일 때만 디코더를 실행**한다.

### vLLM v1 아키텍처 분석

vLLM v1 내부에서 인코더-디코더 분리 지점 확인:

```
gpu_model_runner._preprocess():
  _execute_mm_encoder()   → encoder_cache에 저장
  _gather_mm_embeddings() → 캐시에서 꺼냄
  embed_input_ids()       → 텍스트 임베딩과 합침
  language_model.forward() → 디코더
```

인코더와 디코더 사이에 hook 포인트 존재 (`gpu_model_runner.py` line 2636–2637).

### 제약: 별도 ZMQ 프로세스

`world_size == 1`이면 `UniProcExecutor`를 쓰지만, `AsyncLLMEngine`은 항상 `AsyncMPClient`(ZMQ 소켓) 를 통해 **백그라운드 프로세스**에서 모델을 실행한다.

| 접근법 | 가능 여부 |
|---|---|
| `audio_tower` forward hook | **불가** (별도 프로세스) |
| `encoder_cache` 직접 접근 | **불가** (별도 프로세스 메모리) |
| `gpu_model_runner.py` 수정 | **가능** (서브프로세스 내 probe 탑재) |
| 메인 프로세스 별도 인코더 | **가능** (vLLM 미접촉, RAM +~3GB) |

### 다음 단계

두 가지 구현 옵션 검토 중:

- **A**: 메인 프로세스에 transformers 인코더 별도 로드 → probe 실행 → fire 시에만 `streaming_transcribe()` 호출
- **B**: `gpu_model_runner.py` 수정 → 서브프로세스 내에서 probe 실행 → decoder 완전 차단

---

## 현재 상태 (linear probe 단계)

- `train_one_seg.py`: encoder / mel 두 모드, uniform / spike 두 레이블 타입 지원
- `eval_one_seg.py`: `one_seg` / `streaming` 두 가지 평가 모드
- `eval_cif.py`: CIF / qty_head / k-peak 세 방식 동시 비교
- `plot_cif.py`: CIF(빨강 ▼) + k-peak(주황 ▲) 동시 시각화
- `probe_encoder.py`: 인코더 판별력 분석 (discriminability curve, heatmap)
- `probe_linear.py`: logistic regression probe (activation cache, threshold sweep, probe 저장)

---

## 방향 전환: Causal GRU 스트리밍 검출기 (`train_causal.py`)

### 동기

CIF 누적 방식과 linear probe의 한계:
- CIF: 전체 dialog를 하나의 forward로 처리 → 스트리밍 특성과 불일치
- linear probe: 시간 의존성 없음, per-frame 독립 → 문맥 포착 불가

**새 접근**: dialog 단위 연속 스트리밍 시퀀스를 causal 모델(GRU/Transformer)로 처리.  
각 프레임에서 과거 정보만 참조(causal)하여 P(SEG|frame) 추정, run detection으로 trigger.

### 아키텍처

```
dialog turns concat → Qwen3-ASR encoder → (T, 2048)
    → CausalGRUProbe(hidden=128, n_layers=2)
    → logits (T,) → sigmoid → proba (T,)
run detection → peaks → SEG triggers
```

- **CausalGRUProbe**: GRU + Linear head
- **CausalTransformerProbe**: Proj + SinusoidalPE + Causal MHA + Head (sliding window 옵션)
- **CausalCNN1DProbe**: Causal Conv1d 스택

### 데이터 구성

DailyTalk 10,000 records → dialog 단위 그룹 → 1,058 dialogs  
train/val/test = 958/50/50 split (dialog 단위)

각 dialog: turn들을 이어붙인 연속 오디오 → encoder feature (T, 2048)  
label: SEG 프레임 ±pos_window 내 1, 나머지 0 (hard binary)

---

## Causal 검출기 실험 결과

### 평가 지표

| 지표 | 설명 |
|---|---|
| PR-AUC | SEG ±1프레임 기준 정밀도-재현율 곡선 면적 |
| ROC-AUC | SEG ±1프레임 기준 ROC 곡선 면적 |
| count_acc | dialog당 SEG 개수가 GT와 일치하는 비율 |
| over_rate | 과검출 dialog 비율 |
| under_rate | 미검출 dialog 비율 |
| pos_mae | count 일치한 dialog에서의 peak 위치 오차 (ms) |

**공정 비교 원칙**: label type에 관계없이 평가는 항상 SEG ±pos_window binary GT 기준(`make_gt_binary`).

### 전체 실험 비교표

| 실험 | PR-AUC | ROC-AUC | best F1 | count_acc | over_rate | under_rate | pos_mae |
|---|---|---|---|---|---|---|---|
| pos1 GRU (baseline) | 0.880 | 0.980 | 0.808 | 0.340 | 0.520 | 0.140 | 305ms |
| h256/l2/focal | 0.876 | 0.980 | 0.806 | 0.340 | 0.460 | 0.200 | 524ms |
| Gaussian label | 0.876 | 0.954 | 0.819 | 0.280 | 0.440 | 0.280 | 318ms |
| Delta feature | 0.881 | 0.981 | 0.808 | 0.260 | 0.460 | 0.280 | 453ms |
| Transformer w256 | 0.851 | 0.975 | 0.786 | 0.280 | 0.460 | 0.260 | 740ms |
| Distance d30 | 0.387* | 0.709* | 0.408* | 0.360 | 0.360 | 0.280 | 557ms |
| Combined α=0.5 | 0.867 | 0.970 | 0.803 | 0.240 | 0.560 | 0.200 | **154ms** |
| Combined α=2.0 | 0.880 | 0.981 | 0.810 | 0.360 | 0.500 | 0.140 | 337ms |
| Curriculum (15+10) | 0.883 | 0.981 | 0.815 | 0.300 | 0.440 | 0.260 | 185ms |
| Curriculum + freeze | 0.849 | 0.973 | 0.795 | 0.300 | 0.360 | 0.340 | 350ms |
| **Curriculum (100+100)** | **0.884** | **0.981** | **0.812** | **0.340** | **0.420** | 0.240 | **196ms** |

*Distance 단독은 학습 label 기준으로 평가돼 부풀려진 값. 공정 재평가 수치.

### 주요 실험별 인사이트

**GRU > Transformer**: 데이터 958 dialogs로 Transformer의 귀납적 편향 부재가 단점으로 작용.

**Distance regression**: proximity ramp(`1 - dist/D`) 타깃으로 MSE 학습 시 pos_mae가 대폭 개선되나 PR-AUC 하락. 단독으로는 열위.

**Combined loss** (`MSE(proximity) + α×BCE(hard_binary)`):
- gradient 충돌: SEG 직전 구간에서 MSE는 "높게", BCE는 "낮게" 요구
- α=0.5: pos_mae **154ms** (최고) but count_acc 하락
- α=2.0: BCE 우세 → PR-AUC pos1과 동일 회복, count_acc도 0.360으로 개선

**Curriculum learning** (Phase1 distance → Phase2 BCE):
- gradient 충돌 없이 두 단계 분리
- Phase1에서 "SEG 접근" anticipatory signal 학습
- Phase2에서 정확한 SEG 위치로 sharpen
- freeze backbone 실험: head 2개 파라미터만으로 proximity → binary 변환 불가 → 실패
- **최종 curriculum (100+100, cosine scheduler)**: PR-AUC **0.884** (전체 최고), pos_mae **196ms** (pos1 305ms 대비 36% 개선), count_acc pos1과 동일 복원

---

## 구현 사항 (`train_causal.py`)

### 주요 인자

| 인자 | 설명 |
|---|---|
| `--arch` | `gru` / `cnn1d` / `transformer` |
| `--hidden` | GRU hidden dim 또는 Transformer d_model |
| `--label-type` | `hard` / `gaussian` / `distance` |
| `--decay-frames` | distance 타깃 ramp-up 프레임 수 |
| `--loss` | `bce` / `focal` / `distance` / `combined` |
| `--loss-alpha` | combined loss BCE 비중 |
| `--delta` | encoder output에 delta feature concat |
| `--curriculum` | 2단계 학습 활성화 |
| `--phase2-epochs` | Phase2 epoch 수 |
| `--phase2-lr` | Phase2 LR (미지정 시 lr/5) |
| `--phase1-scheduler` | Phase1 LR scheduler (plateau/cosine/none) |
| `--phase2-scheduler` | Phase2 LR scheduler (cosine 권장) |
| `--freeze-backbone` | Phase2에서 backbone freeze |
| `--attn-window` | Transformer sliding window 크기 |

### 캐시 구조

encoder feature 추출은 오래 걸리므로 dialog별 feature를 pkl로 캐시.  
캐시 key: `{data_stem}_pw{pos_window}_causal[_{label_type}][_d{decay}][_delta].pkl`  
label type이 다를 때는 hard 캐시를 재활용해 label만 재계산.

### 최적 실행 명령

```bash
python research/cif/train_causal.py \
  --curriculum \
  --label-type   distance \
  --decay-frames 30 \
  --epochs       100 \
  --phase2-epochs 100 \
  --phase1-scheduler cosine \
  --phase2-scheduler cosine \
  --cache-dir    research/cif/causal_results/gru/pos1 \
  --out-dir      research/cif/causal_results/curriculum/d30_pw1_epoch
```

---

## 현재 최종 상태

**최선 모델**: curriculum (100+100) — `research/cif/causal_results/curriculum/d30_pw1_epoch/causal_gru.pkl`

pos1 GRU 대비 개선:
- PR-AUC: 0.880 → **0.884**
- pos_mae: 305ms → **196ms** (36% 개선)
- over_rate: 0.520 → **0.420**
- count_acc: 동일 유지 (0.340)

## 다음 방향

- 최소 간격 제약 post-processing 추가 → over_rate 추가 개선 가능성
- DailyTalk 외 추가 데이터(KSponSpeech 영어 데이터 등) 확보 시 재학습
- 파이프라인 통합: STiTy 서버에서 causal GRU probe를 SEG 선제 trigger로 활용
