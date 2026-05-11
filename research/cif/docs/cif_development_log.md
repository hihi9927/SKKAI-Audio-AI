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

## 현재 상태

- `train_one_seg.py`: one-seg 학습 중 (학습 안정적, bnd 빠르게 수렴)
- `eval_one_seg.py`: 두 가지 평가 모드
  - `one_seg`: 트림 청크 기준 pos_mae / fire_acc
  - `streaming`: 원본 멀티-SEG 오디오에서 반복 commit 시뮬레이션
- `eval_cif.py`: CIF / qty_head / k-peak 세 방식 동시 비교 가능
- `plot_cif.py`: CIF(빨강 ▼) + k-peak(주황 ▲) 동시 시각화

## 다음 방향

- one-seg 학습 완료 후 `eval_one_seg.py --mode streaming` 으로 멀티-SEG 성능 확인
- streaming 시뮬레이션 pos_mae가 기존 k-peak(154ms)보다 개선되는지 검증
