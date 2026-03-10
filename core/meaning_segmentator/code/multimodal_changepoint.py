"""
멀티모달 단어 단위 변화점 탐지
================================================================
음성 임베딩 (wav2vec2 forced alignment) +
텍스트 임베딩 (klue/bert-base) 를 단어 단위로 정렬하여
의미/주제 전환점 탐지

파이프라인:
    PCM + 전사 텍스트
        ↓
    [음성] wav2vec2 CTC forced alignment
        → 단어별 타임스탬프 + hidden state 평균 → 음성 임베딩 [1024d]
        ↓
    [텍스트] klue/bert-base subword → 단어별 평균 → 텍스트 임베딩 [768d]
        ↓
    단어별 concat → [1792d] → PCA → 변화점 탐지
    (Cosine Drop / BOCPD / NOUGAT 앙상블)

실행:
    python multimodal_changepoint.py \\
        --transcript_index /path/to/eval_clean.trn \\
        --pcm_dir          /path/to/eval_clean \\
        --out_dir          /path/to/results \\
        --n_files          10

설치:
    pip install transformers torchaudio torch numpy matplotlib scikit-learn scipy
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
import torch

# ══════════════════════════════════════════════════════════
# 0. 설정
# ══════════════════════════════════════════════════════════

SAMPLE_RATE = 16000

# ── 모델 ──
WAV2VEC_MODEL = "kresnik/wav2vec2-large-xlsr-korean"
BERT_MODEL    = "klue/bert-base"

# ── 임베딩 결합 방식 ──
# "concat" : [음성 | 텍스트] 이어붙이기
# "mean"   : (음성 + 텍스트) / 2  (같은 차원일 때)
# "audio"  : 음성만
# "text"   : 텍스트만
FUSION_MODE = "concat"

# ── 앙상블 길이 기준 (단어 수) ──
SHORT_THRESH = 5   # 단어 수 < 5  → Cosine Drop
LONG_THRESH  = 15   # 단어 수 ≥ 15 → NOUGAT

# ── NOUGAT ──
N_REF   = 2
N_TEST  = 2
MU      = 0.01
NU      = 0.5
L_MAX   = 20
LAMBDA  = 1e-3
FALSE_ALARM_RATE = 0.01

# ── BOCPD ──
BOCPD_HAZARD = 1 / 8
BOCPD_ALPHA  = 1.0
BOCPD_BETA   = 1.0
BOCPD_KAPPA  = 1.0
BOCPD_MU0    = 0.0
BOCPD_THRESH = 0.35

# ── Cosine Drop ──
COSINE_THRESH = 0.3

# ── 전처리 ──
PCA_DIM = 64


# ══════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ══════════════════════════════════════════════════════════

def parse_transcript_index(
    index_path: str,
    pcm_dir: Optional[str],
    n_files: Optional[int],
) -> List[Dict]:
    records = []
    try:
        with open(index_path, "r", encoding="euc-kr") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(index_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or " :: " not in line:
            continue
        raw_path, transcript = line.split(" :: ", 1)
        raw_path   = raw_path.strip()
        transcript = transcript.strip()

        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            pcm_path = raw_path
        elif pcm_dir:
            pcm_path = os.path.join(pcm_dir, os.path.basename(raw_path))
        else:
            pcm_path = raw_path

        file_id = os.path.splitext(os.path.basename(pcm_path))[0]
        records.append({
            "pcm_path":   pcm_path,
            "transcript": transcript,
            "file_id":    file_id,
        })

    return records[:n_files] if n_files else records


def load_pcm(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def clean_transcript(text: str) -> str:
    """
    KsponSpeech 전사 노이즈 제거.
    o/, b/, l/, n/, +, /, *, (), {} 등 제거 후 단어 분리.
    """
    # 발화 태그 제거: o/, b/, l/, n/ 등
    text = re.sub(r"[a-zA-Z]/", "", text)
    # 괄호 안 대안 발음: (30만 원)/(삼십만 원) → 삼십만 원 (두 번째 선택)
    text = re.sub(r"\([^)]+\)/\(([^)]+)\)", r"\1", text)
    # 단일 괄호
    text = re.sub(r"[(){}]", "", text)
    # 특수기호
    text = re.sub(r"[*/+]", "", text)
    # 다중 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    """공백 기준 단어 분리 (한국어 어절 단위)"""
    cleaned = clean_transcript(text)
    words   = [w for w in cleaned.split() if w]
    return words


# ══════════════════════════════════════════════════════════
# 2. 음성 임베딩: wav2vec2 Forced Alignment
# ══════════════════════════════════════════════════════════

def load_wav2vec(model_name: str, device: str) -> Dict:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    print(f"  wav2vec2 로딩: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  wav2vec2 로딩 완료  "
          f"(hidden={model.config.hidden_size}d)")
    return {"processor": processor, "model": model,
            "device": device, "hidden": model.config.hidden_size}


@torch.no_grad()
def get_audio_word_embeddings(
    audio: np.ndarray,
    words: List[str],
    wav2vec: Dict,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    wav2vec2 hidden state를 단어 수에 맞게 균등 분할하여
    단어별 음성 임베딩 추출.

    Forced alignment (CTC) 대신 균등 분할을 사용.
    이유: CTC alignment는 모델 vocab이 맞아야 하고
          한국어 특화 모델은 자소 단위라 어절 정렬이 어려움.
    대신 hidden state를 단어 수로 균등 분할 → 근사 정렬.

    반환:
        embeddings : 단어별 음성 임베딩 리스트 [hidden_size]
        timestamps : 단어별 (start_sec, end_sec) 리스트
    """
    processor = wav2vec["processor"]
    model     = wav2vec["model"]
    device    = wav2vec["device"]
    n_words   = len(words)

    if n_words == 0:
        return [], []

    # wav2vec2 인코딩 → hidden states
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)
    out          = model(input_values, output_hidden_states=True)

    # 마지막 hidden state: [1, T_frames, hidden]
    hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # [T, H]
    T      = hidden.shape[0]

    # 단어 수로 균등 분할
    boundaries = np.linspace(0, T, n_words + 1, dtype=int)
    duration   = len(audio) / SAMPLE_RATE

    embeddings = []
    timestamps = []
    for i in range(n_words):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            e = s + 1
        emb = hidden[s:e].mean(axis=0)   # 구간 평균
        embeddings.append(emb)

        t_start = float(s) / T * duration
        t_end   = float(e) / T * duration
        timestamps.append((t_start, t_end))

    return embeddings, timestamps


# ══════════════════════════════════════════════════════════
# 3. 텍스트 임베딩: klue/bert-base
# ══════════════════════════════════════════════════════════

def load_bert(model_name: str, device: str) -> Dict:
    from transformers import AutoTokenizer, AutoModel
    print(f"  BERT 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  BERT 로딩 완료  "
          f"(hidden={model.config.hidden_size}d)")
    return {"tokenizer": tokenizer, "model": model,
            "device": device, "hidden": model.config.hidden_size}


@torch.no_grad()
def get_text_word_embeddings(
    words: List[str],
    bert: Dict,
) -> List[np.ndarray]:
    """
    klue/bert-base로 단어별 텍스트 임베딩 추출.

    전체 문장을 한 번에 인코딩한 후
    각 단어에 해당하는 subword token들의 hidden state를 평균.
    → 문맥 정보 포함 (단어 단독 인코딩보다 훨씬 풍부)
    """
    tokenizer = bert["tokenizer"]
    model     = bert["model"]
    device    = bert["device"]

    if not words:
        return []

    sentence = " ".join(words)

    # 토크나이즈: 단어별 오프셋 추적
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]

    # BERT 인코딩
    out    = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # [T_tok, 768]

    # 단어별 subword 토큰 → 평균
    # 각 단어의 문자 오프셋 계산
    word_embeddings = []
    char_pos = 0
    for word in words:
        w_start = char_pos
        w_end   = char_pos + len(word)
        char_pos = w_end + 1  # 공백 포함

        # 이 단어에 속하는 토큰 인덱스 수집
        tok_indices = []
        for tok_idx, (t_start, t_end) in enumerate(offset_mapping):
            if t_start == 0 and t_end == 0:
                continue   # [CLS], [SEP], padding
            # 단어 범위와 겹치면 포함
            if t_end > w_start and t_start < w_end:
                tok_indices.append(tok_idx)

        if tok_indices:
            emb = hidden[tok_indices].mean(axis=0)
        else:
            # fallback: 전체 문장 평균
            emb = hidden[1:-1].mean(axis=0)

        word_embeddings.append(emb)

    return word_embeddings


# ══════════════════════════════════════════════════════════
# 4. 멀티모달 융합
# ══════════════════════════════════════════════════════════

def fuse_embeddings(
    audio_embs: List[np.ndarray],
    text_embs:  List[np.ndarray],
    mode: str = FUSION_MODE,
) -> List[np.ndarray]:
    """
    단어별 음성 + 텍스트 임베딩 융합.

    concat: [audio | text] → [1024 + 768 = 1792d]
    mean  : (audio_pca + text_pca) / 2  (PCA로 같은 차원 맞춤)
    audio : 음성만
    text  : 텍스트만
    """
    assert len(audio_embs) == len(text_embs), \
        f"임베딩 수 불일치: audio={len(audio_embs)}, text={len(text_embs)}"

    if mode == "concat":
        return [np.concatenate([a, t]) for a, t in zip(audio_embs, text_embs)]
    elif mode == "mean":
        a_arr   = np.array(audio_embs)
        t_arr   = np.array(text_embs)
        min_dim = min(a_arr.shape[1], t_arr.shape[1],
                    a_arr.shape[0] - 1, t_arr.shape[0] - 1, PCA_DIM)
        min_dim = max(min_dim, 1)
        a_red = PCA(n_components=min_dim, random_state=42).fit_transform(
                    StandardScaler().fit_transform(a_arr))
        t_red = PCA(n_components=min_dim, random_state=42).fit_transform(
                    StandardScaler().fit_transform(t_arr))
        return [(a / (np.linalg.norm(a) + 1e-9) +
                t / (np.linalg.norm(t) + 1e-9)) / 2
                for a, t in zip(a_red, t_red)]
    elif mode == "audio":
        return audio_embs
    elif mode == "text":
        return text_embs
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


# ══════════════════════════════════════════════════════════
# 5. 전처리: PCA + Median Heuristic
# ══════════════════════════════════════════════════════════

def median_heuristic(embs: List[np.ndarray]) -> float:
    arr   = np.array(embs)
    n     = len(arr)
    dists = [float(np.linalg.norm(arr[i] - arr[j]))
             for i in range(n) for j in range(i+1, n)]
    return float(max(np.median(dists), 1e-3)) if dists else 1.0


def reduce_dim(embs: List[np.ndarray], n_dim: int) -> Tuple[List[np.ndarray], float]:
    arr    = np.array(embs)
    n_comp = min(n_dim, arr.shape[0] - 1, arr.shape[1])
    if n_comp < 1:
        return embs, 1.0
    scaled = StandardScaler().fit_transform(arr)
    pca    = PCA(n_components=n_comp, random_state=42)
    red    = pca.fit_transform(scaled)
    return [red[i] for i in range(len(embs))], \
           float(pca.explained_variance_ratio_.sum())


# ══════════════════════════════════════════════════════════
# 6. 변화점 탐지 방법론
# ══════════════════════════════════════════════════════════

# ── 6-1. Cosine Drop ──
def detect_cosine_drop(
    embs: List[np.ndarray],
    threshold: float = COSINE_THRESH,
) -> Tuple[List[int], List[float]]:
    if len(embs) < 2:
        return [], []
    arr    = np.array(embs)
    norms  = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / (norms + 1e-9)
    scores, detected = [], []
    for i in range(len(embs) - 1):
        score = 1.0 - float(normed[i] @ normed[i+1])
        scores.append(score)
        if score > threshold:
            detected.append(i + 1)
    return detected, scores


# ── 6-2. BOCPD ──
def bocpd_gaussian(
    embs: List[np.ndarray],
    hazard: float = BOCPD_HAZARD,
    thresh: float = BOCPD_THRESH,
) -> Tuple[List[int], List[float]]:
    from scipy.stats import t as student_t

    if len(embs) < 2:
        return [], []

    # 스칼라화: PCA 첫 번째 주성분 점수 사용 (L2 norm보다 정보량 많음)
    arr = np.array(embs)
    if arr.shape[1] > 1:
        pca1 = PCA(n_components=1, random_state=42)
        xs   = pca1.fit_transform(StandardScaler().fit_transform(arr))[:, 0]
    else:
        xs = arr[:, 0]

    T     = len(xs)
    max_r = T + 1
    R     = np.zeros((T + 1, max_r))
    R[0, 0] = 1.0

    mu    = np.full(max_r, BOCPD_MU0)
    kappa = np.full(max_r, BOCPD_KAPPA)
    alpha = np.full(max_r, BOCPD_ALPHA)
    beta  = np.full(max_r, BOCPD_BETA)

    cp_probs = []
    detected = []

    for t in range(T):
        x    = xs[t]
        pred = np.zeros(t + 1)
        for r in range(t + 1):
            if R[t, r] < 1e-12:
                continue
            df  = max(2 * alpha[r], 1e-6)
            loc = mu[r]
            sc  = max(np.sqrt(beta[r] * (kappa[r] + 1) /
                              (kappa[r] * alpha[r])), 1e-9)
            pred[r] = student_t.pdf(x, df=df, loc=loc, scale=sc)

        growth = R[t, :t+1] * pred[:t+1] * (1 - hazard)
        cp     = np.sum(R[t, :t+1] * pred[:t+1]) * hazard

        R[t+1, 1:t+2] = growth
        R[t+1, 0]     = cp

        norm = R[t+1, :t+2].sum()
        if norm > 1e-12:
            R[t+1, :t+2] /= norm

        cp_prob = float(R[t+1, 0])
        cp_probs.append(cp_prob)

        if cp_prob > thresh and (not detected or t - detected[-1] >= 2):
            detected.append(t)

        # 하이퍼파라미터 업데이트
        new_mu    = np.zeros(max_r)
        new_kappa = np.zeros(max_r)
        new_alpha = np.zeros(max_r)
        new_beta  = np.zeros(max_r)

        new_mu[0]    = BOCPD_MU0
        new_kappa[0] = BOCPD_KAPPA
        new_alpha[0] = BOCPD_ALPHA
        new_beta[0]  = BOCPD_BETA

        for r in range(1, t + 2):
            prev         = r - 1
            kp           = kappa[prev]
            new_kappa[r] = kp + 1
            new_mu[r]    = (kp * mu[prev] + x) / (kp + 1)
            new_alpha[r] = alpha[prev] + 0.5
            new_beta[r]  = beta[prev] + \
                           (kp * (x - mu[prev]) ** 2) / (2 * (kp + 1))

        mu    = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta  = new_beta

    return detected, cp_probs


# ── 6-3. NOUGAT ──
def rbf_kernel(x, y, sigma):
    d = x - y
    return float(np.exp(-np.dot(d, d) / (2 * sigma ** 2)))

def kernel_vector(x, dictionary, sigma):
    return np.array([rbf_kernel(x, d, sigma) for d in dictionary])

def coherence(x, dictionary, sigma):
    return float(max(rbf_kernel(x, d, sigma) for d in dictionary)) \
           if dictionary else 0.0


class NOUGAT:
    def __init__(self, sigma):
        self.sigma = sigma
        self.dictionary: List[np.ndarray] = []
        self.theta   = np.array([])
        self.buffer: List[np.ndarray] = []
        self.g_values: List[float] = []
        self._g_sum = self._g2_sum = 0.0
        self._n_obs = 0

    def _update_dict(self, x):
        if coherence(x, self.dictionary, self.sigma) < NU:
            if len(self.dictionary) < L_MAX:
                self.dictionary.append(x.copy())
                self.theta = np.append(self.theta, 0.0)

    def update(self, x) -> Optional[float]:
        self.buffer.append(x.copy())
        if len(self.buffer) > N_REF + N_TEST:
            self.buffer.pop(0)
        self._update_dict(x)
        if len(self.buffer) < N_REF + N_TEST or not self.dictionary:
            return None
        L      = len(self.dictionary)
        ref_b  = self.buffer[:N_REF]
        test_b = self.buffer[N_REF:]
        h_ref  = np.mean([kernel_vector(v, self.dictionary, self.sigma)
                          for v in ref_b],  axis=0)
        h_test = np.mean([kernel_vector(v, self.dictionary, self.sigma)
                          for v in test_b], axis=0)
        kv     = np.array([kernel_vector(v, self.dictionary, self.sigma)
                           for v in ref_b])
        H_ref  = kv.T @ kv / N_REF
        grad   = (H_ref + LAMBDA * np.eye(L)) @ self.theta - h_test + h_ref
        self.theta -= MU * grad
        g_t = float(self.theta @ h_test)
        self.g_values.append(g_t)
        self._n_obs  += 1
        self._g_sum  += g_t
        self._g2_sum += g_t ** 2
        return g_t

    def detect(self) -> Tuple[List[int], float]:
        from scipy import stats
        if not self.g_values:
            return [], 0.0
        if self._n_obs < 3:
            xi = float(np.mean(np.abs(self.g_values)) + np.std(self.g_values))
        else:
            mean_g = self._g_sum / self._n_obs
            var_g  = max(self._g2_sum / self._n_obs - mean_g ** 2, 1e-9)
            z      = stats.norm.ppf(1 - FALSE_ALARM_RATE / 2)
            xi     = float(abs(mean_g) + z * np.sqrt(var_g))
        detected, prev = [], -999
        for i, g in enumerate(self.g_values):
            if abs(g + 1) > xi and i - prev >= 2:
                detected.append(i)
                prev = i
        return detected, xi


def run_nougat(embs: List[np.ndarray], sigma: float) -> Tuple[List[int], float, List[float]]:
    det = NOUGAT(sigma)
    for e in embs:
        det.update(e)
    warmup       = N_REF + N_TEST - 1
    detected_raw, xi = det.detect()
    detected = [d + warmup for d in detected_raw if d + warmup < len(embs)]
    return detected, xi, det.g_values


# ── 6-4. 앙상블 디스패처 ──
def ensemble_detect(
    embs: List[np.ndarray],
    n_words: int,
    sigma: float,
    short_thresh: int = SHORT_THRESH,
    long_thresh:  int = LONG_THRESH,
) -> Dict:
    if n_words < short_thresh:
        detected, scores = detect_cosine_drop(embs, COSINE_THRESH)
        return {"method": "cosine_drop", "detected": detected,
                "scores": scores, "extra": {"threshold": COSINE_THRESH}}
    elif n_words < long_thresh:
        detected, scores = bocpd_gaussian(embs)
        return {"method": "bocpd", "detected": detected,
                "scores": scores, "extra": {"threshold": BOCPD_THRESH}}
    else:
        detected, xi, scores = run_nougat(embs, sigma)
        return {"method": "nougat", "detected": detected,
                "scores": scores, "extra": {"xi": xi}}


# ══════════════════════════════════════════════════════════
# 7. 단일 파일 처리
# ══════════════════════════════════════════════════════════

def process_one(
    record:  Dict,
    wav2vec: Dict,
    bert:    Dict,
    short_thresh: int = SHORT_THRESH,
    long_thresh:  int = LONG_THRESH,
    fusion_mode:  str = FUSION_MODE,
) -> Optional[Dict]:
    pcm_path   = record["pcm_path"]
    transcript = record["transcript"]
    file_id    = record["file_id"]

    if not os.path.exists(pcm_path):
        print(f"  [스킵] 파일 없음: {pcm_path}")
        return None

    # ── 오디오 로딩 ──
    audio    = load_pcm(pcm_path)
    duration = len(audio) / SAMPLE_RATE

    # ── 단어 분리 ──
    words = tokenize_words(transcript)
    if len(words) < 2:
        print(f"  [스킵] 단어 부족 ({len(words)}개): {file_id}")
        return None

    print(f"  단어 {len(words)}개: {' | '.join(words[:8])}"
          f"{'...' if len(words) > 8 else ''}")

    device = wav2vec["device"]

    # ── 음성 임베딩 (단어별) ──
    audio_embs, timestamps = get_audio_word_embeddings(audio, words, wav2vec)

    # ── 텍스트 임베딩 (단어별) ──
    text_embs = get_text_word_embeddings(words, bert)

    if len(audio_embs) != len(text_embs):
        print(f"  [경고] 임베딩 수 불일치: "
              f"audio={len(audio_embs)}, text={len(text_embs)}")
        n = min(len(audio_embs), len(text_embs))
        audio_embs = audio_embs[:n]
        text_embs  = text_embs[:n]
        timestamps = timestamps[:n]
        words      = words[:n]

    # ── 멀티모달 융합 ──
    fused = fuse_embeddings(audio_embs, text_embs, fusion_mode)

    # ── 전처리: PCA ──
    if PCA_DIM and len(fused) > 2:
        embs_input, expl_var = reduce_dim(fused, PCA_DIM)
    else:
        embs_input, expl_var = fused, 1.0

    sigma_used = median_heuristic(embs_input)

    # ── 앙상블 탐지 ──
    result_det = ensemble_detect(
        embs_input, len(words), sigma_used,
        short_thresh, long_thresh,
    )

    detected       = result_det["detected"]
    method         = result_det["method"]
    detected_times = [timestamps[d][0] for d in detected if d < len(timestamps)]
    detected_words = [words[d] for d in detected if d < len(words)]

    return {
        "file_id":        file_id,
        "pcm_path":       pcm_path,
        "transcript":     transcript,
        "words":          words,
        "audio":          audio,
        "duration":       duration,
        "timestamps":     timestamps,
        "audio_embs":     audio_embs,
        "text_embs":      text_embs,
        "fused_embs":     fused,
        "method":         method,
        "detected":       detected,
        "detected_times": detected_times,
        "detected_words": detected_words,
        "scores":         result_det["scores"],
        "extra":          result_det["extra"],
        "sigma_used":     sigma_used,
        "expl_var":       expl_var,
        "fusion_mode":    fusion_mode,
    }


# ══════════════════════════════════════════════════════════
# 8. 시각화
# ══════════════════════════════════════════════════════════

METHOD_COLOR = {
    "cosine_drop": "#FF6B35",
    "bocpd":       "#7B2D8B",
    "nougat":      "#3A86FF",
}
METHOD_LABEL = {
    "cosine_drop": "Cosine Drop",
    "bocpd":       "BOCPD",
    "nougat":      "NOUGAT",
}


def setup_korean_font():
    import platform, subprocess
    system = platform.system()
    if system == "Darwin":    font = "AppleGothic"
    elif system == "Windows": font = "Malgun Gothic"
    else:
        res   = subprocess.run(["fc-list", ":lang=ko", "--format=%{family}\n"],
                               capture_output=True, text=True)
        fonts = res.stdout.strip().split("\n")
        pref  = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR"]
        font  = next((f for f in pref if any(f in l for l in fonts)),
                     "DejaVu Sans")
    matplotlib.rc("font", family=font)
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_figure_one(result: Dict, out_path: str):
    setup_korean_font()
    RED = "#EF233C"; GRAY = "#8D99AE"; BLUE = "#3A86FF"
    ORANGE = "#FF6B35"; PURPLE = "#7B2D8B"; GREEN = "#06D6A0"

    audio          = result["audio"]
    timestamps     = result["timestamps"]
    words          = result["words"]
    audio_embs     = result["audio_embs"]
    text_embs      = result["text_embs"]
    fused_embs     = result["fused_embs"]
    duration       = result["duration"]
    file_id        = result["file_id"]
    transcript     = result["transcript"]
    method         = result["method"]
    detected       = result["detected"]
    detected_times = result["detected_times"]
    detected_words = result["detected_words"]
    scores         = result["scores"]
    extra          = result["extra"]
    expl_var       = result["expl_var"]
    sigma_used     = result["sigma_used"]
    n_words        = len(words)
    m_color        = METHOD_COLOR[method]
    m_label        = METHOD_LABEL[method]

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor("#F8F9FA")

    fusion_str = result.get("fusion_mode", FUSION_MODE)
    fig.suptitle(
        f"멀티모달 변화점 탐지 | {file_id}\n"
        f"방법: {m_label}  |  융합: {fusion_str}  "
        f"wav2vec2+BERT  |  단어 {n_words}개  "
        f"PCA {PCA_DIM}차원({expl_var*100:.0f}%)  "
        f"전환점: {len(detected_times)}개  "
        f"{detected_words}",
        fontsize=11, fontweight="bold", y=0.995, color="#2B2D42",
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.955, bottom=0.04)

    # ── 패널 1: 파형 + 단어 경계 + 전환점 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    t_ax = np.linspace(0, duration, len(audio))
    ax1.plot(t_ax, audio, color=BLUE, lw=0.4, alpha=0.7)

    # 단어 경계 (연한 회색)
    for i, (ts, te) in enumerate(timestamps):
        ax1.axvline(ts, color=GRAY, lw=0.6, alpha=0.35)
        ax1.text(ts + 0.01, 0.28, words[i],
                 fontsize=6, color="#555555", rotation=45,
                 va="bottom", ha="left")

    # 전환점 (빨간 점선)
    for k, t in enumerate(detected_times):
        ax1.axvline(t, color=RED, lw=2.2, linestyle="-.", alpha=0.9,
                    label=f"전환점 ({len(detected_times)}개)" if k == 0 else None)
        ax1.text(t + 0.03, ax1.get_ylim()[1] * 0.75,
                 f"{t:.1f}s\n{detected_words[k] if k < len(detected_words) else ''}",
                 color=RED, fontsize=7.5, fontweight="bold", va="top")

    ax1.set_title(
        f"파형 + 단어 경계 + 전환점  |  {file_id}  ({duration:.1f}s)",
        fontsize=10, fontweight="bold"
    )
    ax1.set_xlabel("시간 (초)", fontsize=9)
    ax1.set_ylabel("진폭", fontsize=9)
    if detected_times:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 패널 2: 방법론 점수 ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    word_x = [timestamps[i][0] for i in range(min(len(scores) + 1, n_words))]

    if scores and method == "cosine_drop":
        ax2.bar(word_x[:len(scores)], scores, width=duration / n_words * 0.8,
                color=m_color, alpha=0.75, label="코사인 거리")
        thresh = extra.get("threshold", COSINE_THRESH)
        ax2.axhline(thresh, color=RED, lw=1.5, linestyle="--",
                    label=f"임계값 ({thresh:.2f})")
        ax2.set_ylabel("1 - 코사인 유사도", fontsize=9)
        ax2.set_title("Cosine Drop 점수 (단어 단위)", fontsize=11, fontweight="bold")

    elif scores and method == "bocpd":
        ax2.plot(word_x[:len(scores)], scores, "o-",
                 color=m_color, lw=2, ms=6, label="P(변화점)")
        thresh = extra.get("threshold", BOCPD_THRESH)
        ax2.axhline(thresh, color=RED, lw=1.5, linestyle="--",
                    label=f"임계값 ({thresh:.2f})")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("P(r=0)", fontsize=9)
        ax2.set_title("BOCPD 변화점 확률 (단어 단위)", fontsize=11, fontweight="bold")

    elif scores and method == "nougat":
        warmup  = N_REF + N_TEST - 1
        g_times = [timestamps[i + warmup][0]
                   for i in range(len(scores))
                   if i + warmup < n_words]
        ax2.plot(g_times, scores[:len(g_times)], "o-",
                 color=m_color, lw=2, ms=6, label="$g_t$")
        xi = extra.get("xi", 0)
        ax2.axhline(xi - 1,  color=RED, lw=1.5, linestyle="--",
                    label=f"+ξ={xi:.3f}")
        ax2.axhline(-xi - 1, color=RED, lw=1.5, linestyle=":")
        ax2.set_ylabel("$g_t$", fontsize=9)
        ax2.set_title("NOUGAT 탐지 통계량 (단어 단위)", fontsize=11, fontweight="bold")

    for t in detected_times:
        ax2.axvline(t, color=RED, lw=1.8, linestyle="-.", alpha=0.7)
    ax2.set_xlabel("시간 (초)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── 패널 3: 음성 임베딩 PCA 2D ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    _plot_emb_trajectory(ax3, audio_embs, timestamps, detected,
                         words, "음성 임베딩 궤적 (wav2vec2 PCA 2D)", BLUE)

    # ── 패널 4: 텍스트 임베딩 PCA 2D ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    _plot_emb_trajectory(ax4, text_embs, timestamps, detected,
                         words, "텍스트 임베딩 궤적 (BERT PCA 2D)", PURPLE)

    # ── 패널 5: 융합 임베딩 유사도 히트맵 ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF")
    emb_arr = np.array(fused_embs)
    norms   = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    sim     = (emb_arr / (norms + 1e-9)) @ (emb_arr / (norms + 1e-9)).T
    cmap_h  = LinearSegmentedColormap.from_list(
        "bw", ["#EEF2FF", "#3A86FF", "#023E8A"])
    im = ax5.imshow(sim, cmap=cmap_h, aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label="코사인 유사도")
    tick_step = max(1, n_words // 10)
    ticks     = list(range(0, n_words, tick_step))
    ax5.set_xticks(ticks); ax5.set_yticks(ticks)
    ax5.set_xticklabels([words[i] for i in ticks],
                         rotation=45, ha="right", fontsize=7)
    ax5.set_yticklabels([words[i] for i in ticks], fontsize=7)
    for didx in detected:
        if didx < n_words:
            ax5.axvline(didx, color=RED, lw=1.5, linestyle="--", alpha=0.8)
            ax5.axhline(didx, color=RED, lw=1.5, linestyle="--", alpha=0.8)
    ax5.set_title("융합 임베딩 유사도 히트맵 (단어×단어)",
                  fontsize=11, fontweight="bold")
    ax5.spines[["top", "right"]].set_visible(False)

    # ── 패널 6: 전사 구간 분할 ──
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor("#FFFFFF"); ax6.axis("off")
    boundaries   = [0] + sorted(detected) + [n_words]
    colors_seg   = [m_color, BLUE, PURPLE, GREEN, ORANGE, "#FF006E"]
    x_pos        = 0.01
    y_top        = 0.92

    ax6.text(0.01, 0.98, "전사 텍스트 구간 분할 (단어 단위):",
             transform=ax6.transAxes, fontsize=9,
             fontweight="bold", color="#2B2D42", va="top")

    for k in range(len(boundaries) - 1):
        w0, w1    = boundaries[k], boundaries[k + 1]
        seg_words = words[w0:w1]
        if not seg_words:
            continue
        color = colors_seg[k % len(colors_seg)]
        t0    = timestamps[w0][0] if w0 < len(timestamps) else 0
        t1    = timestamps[min(w1 - 1, len(timestamps) - 1)][1]

        label = f"▶ 구간 {k+1} ({t0:.1f}~{t1:.1f}s): "
        seg   = " ".join(seg_words)

        full  = label + seg
        ax6.text(x_pos, y_top - k * 0.16,
                 full[:120] + ("…" if len(full) > 120 else ""),
                 transform=ax6.transAxes,
                 fontsize=8.5, color=color, va="top",
                 fontweight="bold" if k == 0 else "normal")

    for sp in ["top", "right", "bottom", "left"]:
        ax6.spines[sp].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_emb_trajectory(ax, embs, timestamps, detected, words, title, color):
    """PCA 2D 궤적 플롯 공통 함수"""
    RED = "#EF233C"
    n   = len(embs)
    if n < 3:
        ax.text(0.5, 0.5, "데이터 부족", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#888")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    emb2d = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(np.array(embs)))
    cmap_p = matplotlib.cm.get_cmap("plasma")

    for i in range(n - 1):
        ax.annotate("", xy=emb2d[i+1], xytext=emb2d[i],
                    arrowprops=dict(arrowstyle="->",
                                    color=cmap_p(i / n), lw=1.5))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=range(n),
                    cmap="plasma", s=60, zorder=4,
                    edgecolors="white", lw=0.8)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="단어 순서")

    # 단어 레이블
    for i, word in enumerate(words):
        ax.annotate(word, emb2d[i], fontsize=6, color="#444",
                    xytext=(3, 3), textcoords="offset points")

    # 전환점 마커
    for k, didx in enumerate(detected):
        if didx < n:
            ax.scatter(*emb2d[didx], color=RED, s=180, zorder=5,
                       edgecolors="black", lw=1.5, marker="D",
                       label=f"{words[didx]}" if k < 5 else None)

    if detected:
        ax.legend(fontsize=7, title="전환점 단어", title_fontsize=7)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def make_summary_figure(results: List[Dict], out_path: str):
    setup_korean_font()
    RED = "#EF233C"; GRAY = "#8D99AE"

    n   = len(results)
    fig, ax = plt.subplots(figsize=(20, max(4, n * 0.7 + 2)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    max_dur = max(r["duration"] for r in results)

    for i, r in enumerate(results):
        y     = n - 1 - i
        color = METHOD_COLOR[r["method"]]
        ax.barh(y, r["duration"], color=color, alpha=0.2, height=0.65)

        # 단어 눈금
        for ts, te in r["timestamps"]:
            ax.plot([ts, ts], [y - 0.3, y + 0.3],
                    color=GRAY, lw=0.5, alpha=0.4)

        # 전환점
        for t in r["detected_times"]:
            ax.plot(t, y, marker="D", color=RED, ms=9, zorder=4)

        label = (f"{r['file_id']}  [{r['method']}  {len(r['words'])}단어]"
                 f"  |  {' '.join(r['words'][:6])}…")
        ax.text(-0.3, y, label, va="center", ha="right",
                fontsize=7, color="#2B2D42")
        ax.text(r["duration"] + 0.1, y,
                f"{len(r['detected_times'])}개  "
                f"{r['detected_words']}",
                va="center", fontsize=7, color=RED)

    for method, color in METHOD_COLOR.items():
        ax.barh(-1, 0, color=color, alpha=0.4,
                label=f"{METHOD_LABEL[method]}")
    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlim(-max_dur * 0.5, max_dur * 1.15)
    ax.set_ylim(-1.5, n - 0.2)
    ax.set_xlabel("시간 (초)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        f"멀티모달 변화점 탐지 요약  ({n}개 파일) | "
        f"wav2vec2 + BERT + {FUSION_MODE}\n"
        f"◆ = 전환점  |  | = 단어 경계",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[요약 저장] {out_path}")


# ══════════════════════════════════════════════════════════
# 9. 메인
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="멀티모달 단어 단위 변화점 탐지 (wav2vec2 + BERT)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--transcript_index", required=True,
                        help="전사 인덱스 파일 (.trn, EUC-KR)")
    parser.add_argument("--pcm_dir", default=None,
                        help="PCM 파일 폴더")
    parser.add_argument("--out_dir", default="./multimodal_results",
                        help="결과 저장 폴더")
    parser.add_argument("--n_files", type=int, default=None,
                        help="처리할 파일 수 (기본: 전체)")
    parser.add_argument("--wav2vec_model", default=WAV2VEC_MODEL,
                        help="wav2vec2 모델명")
    parser.add_argument("--bert_model", default=BERT_MODEL,
                        help="BERT 모델명 (기본: klue/bert-base)")
    parser.add_argument("--fusion", default=FUSION_MODE,
                        choices=["concat", "mean", "audio", "text"],
                        help="융합 방식 (기본: concat)")
    parser.add_argument("--short_thresh", type=int, default=SHORT_THRESH,
                        help=f"Cosine Drop 상한 단어 수 (기본: {SHORT_THRESH})")
    parser.add_argument("--long_thresh", type=int, default=LONG_THRESH,
                        help=f"NOUGAT 하한 단어 수 (기본: {LONG_THRESH})")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  멀티모달 변화점 탐지")
    print(f"  디바이스   : {device}")
    print(f"  wav2vec2   : {args.wav2vec_model}")
    print(f"  BERT       : {args.bert_model}")
    print(f"  융합 방식  : {args.fusion}")
    print(f"  방법론 기준: <{args.short_thresh}단어=Cosine  "
          f"{args.short_thresh}~{args.long_thresh}단어=BOCPD  "
          f"≥{args.long_thresh}단어=NOUGAT")
    print(f"{'='*60}")

    records = parse_transcript_index(
        args.transcript_index, args.pcm_dir, args.n_files)
    print(f"\n  처리 대상: {len(records)}개 파일")

    # 모델 로딩 (1회)
    wav2vec = load_wav2vec(args.wav2vec_model, device)
    bert    = load_bert(args.bert_model, device)

    results       = []
    method_counts = {"cosine_drop": 0, "bocpd": 0, "nougat": 0}

    for i, record in enumerate(records):
        print(f"\n{'─'*60}")
        print(f"  [{i+1:03d}/{len(records):03d}] {record['file_id']}")
        print(f"  전사: {record['transcript'][:60]}...")

        result = process_one(record, wav2vec, bert,
                             args.short_thresh, args.long_thresh,
                             args.fusion)
        if result is None:
            continue

        method_counts[result["method"]] += 1
        print(f"  방법: {result['method']}  "
              f"전환점: {len(result['detected_times'])}개  "
              f"단어: {result['detected_words']}")

        out_path = os.path.join(args.out_dir,
                                f"{result['file_id']}_multimodal.png")
        make_figure_one(result, out_path)
        print(f"  [저장] {out_path}")
        results.append(result)

    if results:
        summary_path = os.path.join(args.out_dir, "summary_multimodal.png")
        make_summary_figure(results, summary_path)

        print(f"\n{'='*60}")
        print(f"  처리 완료: {len(results)}/{len(records)}개")
        print(f"  방법론 현황:")
        for m, cnt in method_counts.items():
            print(f"    {METHOD_LABEL[m]}: {cnt}개")
        avg_cp = np.mean([len(r["detected_times"]) for r in results])
        print(f"  평균 전환점: {avg_cp:.2f}개/발화")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()