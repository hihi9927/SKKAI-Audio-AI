# Feature Scoring — Meaning Segmenter

의미 분절 경계 점수는 **Prosody / Text / Semantic** 3개 모듈의 가중 평균으로 결정된다.

```
total = (w_p * prosody + w_t * text + w_s * semantic) / (w_p + w_t + w_s)
```

**기본 가중치**

| 모듈 | 가중치 |
|------|--------|
| Prosody | 0.4 |
| Text | 0.4 |
| Semantic | 0.2 |

> semantic이 비활성화되면 해당 가중치는 0으로 처리되고 나머지로 정규화됨.

---

## 1. Prosody

**함수:** `prosody_score(audio, sr, w_before, w_after, all_tokens, gap_mean, gap_std)`

음성의 물리적 신호에서 경계 단서를 추출한다. 5개 서브피처를 합산 (max 1.0).

| 서브피처 | 내부 가중치 | 의미 | 추출 방법 |
|----------|-------------|------|-----------|
| **Gap** | 0.30 | 단어 사이 무음 구간. 발화가 끊기는 자리일수록 길다. | `w_after.start - w_before.end` → z-score 정규화<br>`g_score = clip((z + 2) / 4, 0, 1)` |
| **Final Lengthening** | 0.20 | 문장 끝 단어는 자연스럽게 발음이 늘어지는 경향이 있다. | `w_before.duration / mean_dur` (±3 토큰 로컬 평균 기준)<br>`fl_score = clip((ratio - 0.8) / 1.2, 0, 1)` |
| **F0 Drop** | 0.20 | 억양이 문장 끝으로 갈수록 내려간다 (하강 억양). | `librosa.pyin`으로 w_before F0 추출 → 전반/후반 평균 차<br>`score = clip(f0_drop * 3, 0, 1)` |
| **Pitch Reset** | 0.15 | 새 문장 시작 시 억양이 다시 높아진다 (억양 재설정). | w_before 끝 F0 vs w_after 시작 F0 상승폭<br>`score = clip(상승폭 / 80Hz, 0, 1)` |
| **Energy Drop** | 0.15 | 문장 끝에서 발성 에너지가 줄어드는 경향이 있다. | 경계 전후 100ms 윈도우 RMS 비교<br>`score = clip(e_drop * 2, 0, 1)` |

**파라미터**

- `gap_mean`, `gap_std`: 발화 전체 gap 통계. 화자/속도마다 gap 길이가 다르므로 z-score로 자동 정규화.
- `librosa.pyin`: `fmin=70`, `fmax=500`, `frame_length=512`, `hop_length=128`
- Pitch reset 정규화 기준: 80Hz 상승 = 1.0

> **실시간 모드 (cross-utterance):** 청크 경계에서는 오디오가 없으므로 gap z-score만 계산 (`p_score = 0.30 * g_score`).

---

## 2. Text

**함수:** `text_score(w_before, w_after, okt=None, nlp=None)`

형태소/구문 분석으로 경계 직전/직후 단어의 문법적 역할을 평가한다.
경계 직전 단어가 문장을 닫는 역할이고, 직후 단어가 새 문장을 여는 역할일수록 높은 점수를 준다.

### 한국어 — KoNLPy Okt

| 대상 | feature | 의미 | 점수 |
|------|---------|------|------|
| w_before 끝 | `Eomi` | 어미 → 절/문장 종결 신호 | 0.7 |
| w_before 끝 | `Punctuation` | 구두점 → 명시적 문장 끝 | 1.0 |
| w_before 패턴 | `Verb/Adj + Eomi` | 용언+어미 조합 → 절 완전 종결 보너스 | +0.4 |
| w_after 첫 | `Exclamation` | 감탄사 → 새 발화 시작 신호 | 0.7 |
| w_after 첫 | `Adverb` | 접속 부사 ("그런데", "그래서") → 화제 전환 | 0.7 |

### 비한국어 — spaCy UD POS

**w_before — 문장을 닫는 신호**

| feature | 의미 | 기본 점수 |
|---------|------|-----------|
| PUNCT | 구두점 | 1.0 |
| VERB | 본동사 → 절 종결 | 0.4 |
| AUX | 조동사 | 0.2 |
| NOUN | 명사구 종결 | 0.2 |
| ADV | 문말 부사 ("just", "so") | 0.2 |
| ADJ | 서술형 형용사 ("it's great") | 0.15 |

**추가 보너스/패널티 (w_before)**

| feature | 의미 | 점수 |
|---------|------|------|
| `AUX + VERB` 패턴 | 조동사+본동사 조합 → 절 완결 | +0.2 |
| `VerbForm=Fin` | 정형동사(finite verb) → 독립절 종결 | +0.25 |
| `Tense=Past` | 과거형 → 완결된 사건 | +0.3 |
| `Mood=Imp` | 명령법 ("Stop!") → 강한 종결 | +0.3 |
| `DEP=ROOT` | 문장 핵심 동사 위치 | +0.3 |
| `VerbForm=Inf` | 부정사 → 뒤에 더 이어짐 (연속 신호) | -0.2 |
| `DEP=relcl` | 관계절 내부 → 아직 문장 미완 | -0.2 |
| `DEP=advcl` | 부사절 내부 → 아직 문장 미완 | -0.15 |

**w_after — 새 문장을 여는 신호**

| feature | 의미 | 점수 |
|---------|------|------|
| INTJ | 감탄사 → 새 발화 시작 | 0.6 |
| SCONJ | 종속 접속사 ("because") → 새 절 | 0.5 |
| CCONJ | 등위 접속사 ("but", "and") → 새 절 | 0.4 |
| PRON | 대명사 ("I", "he") → 새 주어 등장 | 0.3 |
| PROPN | 고유명사 → 새 화제 등장 | 0.25 |

**Continuity Penalty — 연속성 패널티**

경계처럼 보이지만 실제로는 이어지는 구조를 감지해 점수를 깎는다.

| 패턴 (before → after) | 연속 의미 | 패널티 |
|-----------------------|-----------|--------|
| VERB → PART | 부정사구 시작 ("want **to** go") | -0.4 |
| AUX → VERB | 조동사+본동사 ("will **go**") | -0.3 |
| NOUN → REL | 관계절 시작 ("the man **who**...") | -0.3 |
| ADJ → SCONJ | 비교급 ("better **than**") | -0.2 |

---

## 3. Semantic

**함수:** `semantic_score(tokens, position, embed_fn, window_sizes=[1, 3, 6])`

경계 전후 텍스트의 **의미적 거리**를 측정한다.
화제가 전환되거나 다른 내용으로 넘어가면 임베딩 벡터 간 거리가 멀어진다.

| 단계 | 방법 |
|------|------|
| 텍스트 구성 | 경계 기준 좌우 `w`개 토큰을 각각 이어붙여 문자열 생성 |
| 임베딩 | `embed_fn(text)` → numpy 벡터 |
| 코사인 유사도 | `cos = dot(e_before, e_after) / (norm_b * norm_a)` |
| 비유사도 | `dissim = max(0, 1 - cos)` → 클수록 의미 거리가 멈 |
| 최종 점수 | 3개 window 크기의 dissim 평균 |

**파라미터**

- `window_sizes = [1, 3, 6]`: 창 크기를 달리해 단어 단위/구 단위/절 단위 의미 변화를 함께 포착
- 기본 모델: `paraphrase-multilingual-MiniLM-L12-v2` (다국어 sentence-transformers)
- 비활성화 시 (`--no-semantic` 또는 `embed_fn=None`): score = 0.0, 가중치 0으로 정규화

---

## 4. Peak Detection (경계 확정)

**함수:** `find_boundary_peaks(boundaries, window, min_prominence, min_floor, min_distance)`

고정 임계값 대신 **local prominence** 기반으로 경계를 확정한다.
점수가 절대적으로 높은 게 아니라 주변 대비 상대적으로 튀어야 경계로 인정한다.

```
1. total_score < min_floor  →  noise, 후보 제외
2. score - mean(±window 이웃) >= min_prominence  →  peak 후보
3. NMS: min_distance 이내 중복 후보 중 score 높은 것만 유지
```

**파라미터**

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| `window` | 3 | 좌우 몇 개 이웃과 비교할지 |
| `min_prominence` | 0.20 | 이웃 평균 대비 최소 돌출량 |
| `min_floor` | 0.15 | 후보 최소 점수 (noise floor) |
| `min_distance` | 2 | peak 간 최소 토큰 거리 (NMS) |

**실시간 모드 차이점**
- **Causal**: trailing `window`개 경계가 쌓여야 확정 가능 (구조적 지연 = `min_silence_duration_ms`)
- **`flush()`**: trailing 제약 없이 남은 전체 경계 확정
