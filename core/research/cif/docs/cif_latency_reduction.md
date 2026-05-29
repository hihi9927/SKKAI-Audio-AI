# CIF 기반 Dynamic Chunking으로 레이턴시 줄이기

## 배경 및 문제 정의

현재 STiTy 파이프라인은 ASR 과정에서 `<SEG>` 토큰을 삽입하도록 파인튜닝되어 있다.
`<SEG>`는 의미 있는 커밋 지점(문장/구 경계)을 표시하며, 이를 통해 디코더 출력 기준으로 세그먼트를 나눈다.

**핵심 병목**: `<SEG>`가 박히는 지점을 알려면 이미 ASR 디코딩이 완료되어야 한다.
그 전에 오디오는 **고정 청크** 단위로 인코딩되며, 이 고정 청크가 레이턴시의 실질적인 하한이다.

**목표**: `<SEG>` 위치를 텍스트가 아닌 **오디오 레벨**에서 미리 예측해 청크를 동적으로 자르면,
디코딩 전에 경계를 확정할 수 있어 레이턴시가 줄어든다.

---

## 접근법: CIF (Continuous Integrate-and-Fire)

### 왜 CIF인가

- mel 스펙토그램만으로는 `<SEG>` 경계의 **의미/통사적 정보**를 잡기 어렵다 (prosodic 경계와 다름)
- 인코더 출력은 이미 충분한 고수준 표현을 담고 있으므로, **인코더 출력 위에 CIF를 얹는 것**이 자연스럽다

### CIF 동작 원리

```
인코더 출력 frame들에 가중치 w_t 예측 (0~1)
→ Σw_t 누적
→ 누적합 ≥ 1.0 → fire (경계 신호)
→ 누적값 초기화, acoustic embedding 방출
→ 디코더에 커밋
```

---

## Qwen3-ASR 인코더 구조 분석

### 핵심 발견: 이미 청크 로컬 attention

`Qwen3ASRAudioAttention`은 `is_causal=False` (양방향)이지만,
`_prepare_attention_mask`에서 `cu_seqlens` 기반으로 **청크 간 attention을 완전 차단**한다.

```python
# 청크 내에서만 attention 허용, 청크 간은 -inf
for i in range(1, len(cu_seqlens)):
    attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i],
                        cu_seqlens[i-1]:cu_seqlens[i]] = 0
```

**주요 파라미터** (`configuration_qwen3_asr.py`):

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `n_window` | 100 | CNN 입력 청크 (mel frames, ≈1초) |
| `n_window_infer` | 400 | 인코더 내부 최대 attention 창 (≈4초) |
| `conv_chunksize` | 500 | Conv 처리 청크 수 |

Conv2D 3단계 (각 stride 2) → **8x 다운샘플링** → 100 mel frames → 13 encoder frames

### 실제 레이턴시 하한

`n_window_infer`는 인코더 **내부** attention 창이고, 인코더가 호출되는 주기는 별도다.
현재 서버(`streaming_websocket_server.py`)는 `chunk_size_sec = 2.0`초마다 인코더를 호출한다.

```
2초 오디오 (200 mel frames → 26 encoder frames) → 인코더 호출
→ 26 frames < n_window_infer(52) → 전체가 하나의 attention window
→ CIF가 26 frames 위에서 fire 위치 탐색
```

**실제 레이턴시 하한 = `chunk_size_sec` (현재 2초)**

`chunk_size_sec`를 줄이면 레이턴시↓, 각 청크가 짧아져 컨텍스트↓ — 이 값이 레이턴시-품질 트레이드오프의 손잡이.

---

## CIF의 실제 Contribution 및 실험 방향

### CIF가 실제로 하는 것

엄밀히 말하면 CIF는 **dynamic chunking이 아니다**.

```
원래 목표 (true dynamic chunking):
오디오 스트림 → [경계 감지] → 가변 길이 청크 → 인코더 → 디코더

실제 설계 (semantic decode gating):
오디오 스트림 → 고정 청크 → 인코더 → [CIF: 디코드할지 결정] → 디코더
```

CIF는 인코더 입력을 동적으로 자르는 것이 아니라, **디코더를 언제 실행할지**를 결정하는 게이트다.

**효과**: `chunk_size_sec`를 줄이면 디코더가 불필요하게 자주 실행되는 문제가 있는데,
CIF가 semantic boundary에서만 디코더를 트리거하므로 **디코더 실행 횟수 = SEG 경계 횟수**로 최소화된다.

### True Dynamic Chunking의 딜레마

mel 레벨에서 경계를 잡으려면 semantic 정보가 없고,
semantic 정보를 얻으려면 인코더를 돌려야 하고,
인코더를 돌리려면 고정 청크가 필요하다. → 순환 문제.

### 실험 방향 (두 가지)

| | **방향 A: Semantic Decode Gating** | **방향 B: Acoustic Dynamic Chunking** |
|---|---|---|
| **구조** | 고정 청크 → 인코더 → CIF → 디코더 게이트 | mel 경계 detector → 가변 청크 → 인코더 → 디코더 |
| **경계 정보** | 인코더 출력 (semantic) | mel 스펙토그램 (acoustic/prosodic) |
| **SEG 정렬도** | 높음 | acoustic boundary와 SEG의 상관관계에 의존 |
| **dynamic 여부** | 디코더만 dynamic | 인코더 입력까지 dynamic |
| **구현 복잡도** | 낮음 (CIF 모듈만 추가) | 높음 (인코더 앞 별도 모델 + 가변 길이 처리) |
| **핵심 가정** | 인코더 표현에 경계 정보 존재 | acoustic boundary ≈ SEG boundary |

**방향 B의 핵심 검증 사항**: 기존 파이프라인에서 `<SEG>` 위치와 acoustic boundary(에너지 dip, 피치 하강, pause)가 얼마나 상관관계가 있는지 먼저 확인 필요.

---

## 전체 파이프라인 구조 (제안)

```
오디오 스트림
  → 200 mel frames씩 CNN (≈2초)
  → n_window_infer(≈4초) 단위 attention (증분)
  → 인코더 출력 frame들에 CIF weight 예측
  → Σweight ≥ 1.0 → fire → acoustic embedding 방출
  → 디코더: 커밋된 embedding으로 세그먼트 텍스트 생성
  → 번역 → 클라이언트
```

### 디코더 truncation 문제는 없는가

CIF는 **디코딩 전에** 커밋 경계를 결정한다.
디코더가 생성 도중 잘리는 게 아니라, 애초에 CIF-committed frames만 입력으로 받는다.

- Fire 지점의 frames 0~N이 N+1 이후를 이미 봤더라도 (bidirectional context leakage),
  이는 품질 저하가 아니라 향상 요인이므로 무시해도 된다.
- Qwen3 LLM 디코더는 KV cache 지원 → incremental generation 가능.
- 짧은 완결 발화 처리 능력은 실험적으로 검증 필요 (별도 파인튜닝 없이도 될 가능성 높음).

---

## 현재 파인튜닝 상태

`finetuning/qwen3_asr_sft.py` 분석 결과:

- **인코더**: 완전 frozen (SEG 학습에 참여하지 않음)
- **디코더**: LoRA 적용 (학습됨)
- **SEG 토큰 임베딩**: 별도 unfreeze 후 학습 → `SEG_embedding.pt`로 저장

```python
# PEFT가 freeze한 이후 SEG 임베딩만 unfreeze
emb.weight.requires_grad_(True)
emb.weight.register_hook(_make_seg_only_hook(seg_id))
```

**의미**: 인코더 표현에는 SEG 경계 정보가 암묵적으로만 존재한다.
CIF weight predictor가 이를 추출해야 하므로, 인코더 LoRA가 필요할 수 있다.

---

## CIF 학습 전략

### CIF 모듈 구조

```python
class CIFWeightPredictor(nn.Module):
    def __init__(self, d_model):
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output):
        return self.sigmoid(self.linear(encoder_output)).squeeze(-1)  # (T,)
```

### 학습 목표 (Loss)

```
quantity loss:  (Σw_t - N_seg)²
                → 전체 weight 합이 SEG 개수와 일치하도록

boundary loss:  accumulated weight profile이 실제 SEG 프레임 위치와 정렬되도록
                (soft label 사용 권장: Gaussian smoothing, σ ≈ 5 frames)
```

### 단계별 학습 전략

| 단계 | 학습 대상 | 조건 |
|---|---|---|
| 1 | CIF weight predictor만 | 항상 시작점 |
| 2 | CIF + 인코더 LoRA | 1단계 정확도 부족 시 |
| 3 | 디코더 검증 | 짧은 세그먼트 입력 처리 확인 |

인코더 LoRA 사용 시 ASR 품질 저하 방지를 위해:
- Learning rate 매우 낮게 설정
- ASR loss를 auxiliary로 병행

---

## 학습 데이터 생성

### 핵심: 오디오를 자를 필요 없음

```
기존 오디오 (전체, 그대로 사용)
기존 전사문: "Hello world <SEG> how are you <SEG> ..."
       ↓ forced alignment (SEG 제거 후 정렬)
SEG 위치 타임스탬프: [2.4s, 5.1s, ...]
       ↓ ÷ 8 (CNN 3x stride-2 downsampling)
인코더 프레임 레이블: frame 30=1, frame 64=1, 나머지=0
```

### Forced alignment 주의점

`<SEG>`는 실제 발화 단어가 아니므로 직접 정렬 불가.
→ `<SEG>` 제거한 텍스트로 정렬 후, **SEG 앞 단어의 끝 타임스탬프**를 경계로 사용.

기존 `qwen_asr/inference/qwen3_forced_aligner.py` 활용 가능.

### 학습 레이블 형태

딱 한 프레임만 1로 주면 너무 sparse → Gaussian soft label 권장:

```python
boundary_frame = int(seg_timestamp_sec / (hop_length * 8))
label = gaussian(frame_indices, mu=boundary_frame, sigma=5)
```

---

## 요약

1. **인코더는 이미 청크 로컬 attention** → 증분 실행 가능
2. **CIF를 인코더 출력에 얹어** fire 시 디코더 트리거
3. **학습 데이터**: 기존 오디오 + forced alignment로 SEG 프레임 위치 추출 (오디오 편집 불필요)
4. **학습 순서**: CIF만 → (필요시) 인코더 LoRA 추가
5. **레이턴시 하한**: `chunk_size_sec` (현재 2초) → 이 값을 줄이면 레이턴시↓ 컨텍스트↓
