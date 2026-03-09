"""
Ensemble Audio Changepoint Detection + Whisper
================================================================
발화 길이에 따라 방법론을 자동 선택하는 앙상블 변화점 탐지

  duration < 5초  → Cosine Similarity Drop
  5초 ≤ duration < 15초 → BOCPD
  duration ≥ 15초 → NOUGAT

공통 파이프라인:
    PCM (16kHz/16bit/mono/headerless)
        ↓ 슬라이딩 청크 분할
        ↓ Whisper encoder → 청크 임베딩
        ↓ PCA + Median Heuristic
        ↓ 길이 기반 방법론 자동 선택
        ↓ 전환점 탐지 + 시각화

실행:
    python ensemble_audio_changepoint.py \\
        --transcript_index /path/to/eval_clean.trn \\
        --pcm_dir          /path/to/eval_clean \\
        --out_dir          /path/to/results \\
        --n_files          10

설치:
    pip install openai-whisper torch numpy matplotlib scikit-learn scipy
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
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

# ── 청크 설정 ──
CHUNK_SEC = 2.0
HOP_SEC   = 1.0

# ── wav2vec2 모델 ──
# 가변 길이 입력 지원, 패딩 문제 없음, 한국어 발화에 적합
WAV2VEC_MODEL = "kresnik/wav2vec2-large-xlsr-korean"  # 한국어 특화
# 대안: "facebook/wav2vec2-large-xlsr-53" (다국어)
#       "facebook/wav2vec2-base-960h"      (영어, 빠름)

# ── 앙상블 길이 기준 (초) ──
SHORT_THRESH  = 5.0    # < 5초  → Cosine Drop
LONG_THRESH   = 15.0   # ≥ 15초 → NOUGAT

# ── NOUGAT ──
N_REF   = 2
N_TEST  = 2
MU      = 0.01
NU      = 0.5
L_MAX   = 20
LAMBDA  = 1e-3
FALSE_ALARM_RATE = 0.01

# ── BOCPD ──
BOCPD_HAZARD   = 1 / 10   # 변화점 사전 확률 (평균 10청크마다 1번)
BOCPD_ALPHA    = 1.0       # 학생 t 분포 파라미터
BOCPD_BETA     = 1.0
BOCPD_KAPPA    = 1.0
BOCPD_MU0      = 0.0
BOCPD_THRESH   = 0.3       # 변화점 사후확률 임계값

# ── Cosine Drop ──
COSINE_THRESH  = 0.15      # 유사도 하락 임계값 (0~1, 클수록 엄격)

# ── 전처리 ──
PCA_DIM    = 32
AUTO_SIGMA = True
SIGMA      = 1.0


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
        records.append({"pcm_path": pcm_path, "transcript": transcript,
                        "file_id": file_id})

    return records[:n_files] if n_files else records


def load_pcm(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def split_chunks(audio: np.ndarray) -> List[Tuple[float, float, np.ndarray]]:
    chunk_n = int(CHUNK_SEC * SAMPLE_RATE)
    hop_n   = int(HOP_SEC   * SAMPLE_RATE)
    chunks, start = [], 0
    while start + chunk_n <= len(audio):
        chunks.append((start / SAMPLE_RATE,
                       (start + chunk_n) / SAMPLE_RATE,
                       audio[start:start + chunk_n]))
        start += hop_n
    if start < len(audio):
        chunk = np.pad(audio[start:], (0, chunk_n - len(audio[start:])))
        chunks.append((start / SAMPLE_RATE, len(audio) / SAMPLE_RATE, chunk))
    return chunks


# ══════════════════════════════════════════════════════════
# 2. wav2vec2 임베딩
# ══════════════════════════════════════════════════════════

def load_wav2vec(model_name: str, device: str):
    """
    wav2vec2 모델 로딩.
    Whisper와 달리 가변 길이 입력을 그대로 처리.
    패딩 없이 실제 음성 길이 그대로 인코딩.
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    print(f"  wav2vec2 로딩: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  wav2vec2 로딩 완료")
    return {"processor": processor, "model": model, "device": device}


@torch.no_grad()
def get_chunk_embedding(chunk: np.ndarray, wmodel: dict, device: str) -> np.ndarray:
    """
    wav2vec2로 청크 임베딩 추출.

    입력: 가변 길이 float32 오디오 [-1, 1]
    출력: hidden state 시간 평균 → [hidden_size] (large: 1024, base: 768)

    Whisper와 달리 패딩 없이 실제 길이 그대로 처리.
    """
    processor = wmodel["processor"]
    model     = wmodel["model"]

    # processor: 정규화 + attention mask 생성
    inputs = processor(
        chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)

    # wav2vec2 인코딩
    out = model(input_values)
    # last_hidden_state: [1, T, hidden_size] → 시간 평균 → [hidden_size]
    emb = out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()
    return emb


def extract_embeddings(chunks, wmodel, device):
    """
    모든 청크에서 wav2vec2 임베딩 추출.
    청크가 너무 짧으면 (<0.1초) 스킵.
    """
    timestamps, embeddings = [], []
    min_samples = int(0.1 * SAMPLE_RATE)
    for start, end, chunk in chunks:
        if len(chunk) < min_samples:
            continue
        emb = get_chunk_embedding(chunk, wmodel, device)
        timestamps.append((start, end))
        embeddings.append(emb)
    return timestamps, embeddings


# ══════════════════════════════════════════════════════════
# 3. 전처리
# ══════════════════════════════════════════════════════════

def median_heuristic(embs: List[np.ndarray]) -> float:
    arr = np.array(embs)
    n   = len(arr)
    dists = [float(np.linalg.norm(arr[i] - arr[j]))
             for i in range(n) for j in range(i+1, n)]
    return float(max(np.median(dists), 1e-3)) if dists else 1.0


def reduce_dim(embs: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
    arr    = np.array(embs)
    n_comp = min(PCA_DIM, arr.shape[0] - 1, arr.shape[1])
    if n_comp < 1:
        return embs, 1.0
    scaled = StandardScaler().fit_transform(arr)
    pca    = PCA(n_components=n_comp, random_state=42)
    red    = pca.fit_transform(scaled)
    return [red[i] for i in range(len(embs))], float(pca.explained_variance_ratio_.sum())


# ══════════════════════════════════════════════════════════
# 4. 방법론 1: Cosine Similarity Drop  (짧은 발화)
# ══════════════════════════════════════════════════════════

def detect_cosine_drop(
    embs: List[np.ndarray],
    threshold: float = COSINE_THRESH,
) -> Tuple[List[int], List[float]]:
    """
    인접 청크 간 코사인 유사도를 계산하고
    유사도가 급격히 떨어지는 지점을 전환점으로 탐지.

    score[i] = 1 - cosine_sim(embs[i], embs[i+1])
    score가 threshold를 넘으면 전환점.
    """
    if len(embs) < 2:
        return [], []

    arr   = np.array(embs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / (norms + 1e-9)

    scores   = []
    detected = []
    for i in range(len(embs) - 1):
        sim   = float(normed[i] @ normed[i+1])
        score = 1.0 - sim          # 거리 (클수록 전환)
        scores.append(score)
        if score > threshold:
            detected.append(i + 1)   # i+1번 청크가 전환점

    return detected, scores


# ══════════════════════════════════════════════════════════
# 5. 방법론 2: BOCPD  (중간 발화)
# ══════════════════════════════════════════════════════════

def bocpd_gaussian(
    embs: List[np.ndarray],
    hazard: float  = BOCPD_HAZARD,
    alpha0: float  = BOCPD_ALPHA,
    beta0: float   = BOCPD_BETA,
    kappa0: float  = BOCPD_KAPPA,
    mu0: float     = BOCPD_MU0,
    thresh: float  = BOCPD_THRESH,
) -> Tuple[List[int], List[float]]:
    """
    BOCPD (Bayesian Online Changepoint Detection)
    Adams & MacKay, 2007

    각 차원 독립 가우시안 모델 (Student-t predictive):
        p(x_t | r_t) = StudentT(x_t; 2α, μ, β(κ+1)/κα)

    run_length r_t: 마지막 변화점 이후 경과 시간
    변화점 사후확률 P(r_t=0 | x_{1:t}) 가 thresh 초과 시 탐지.
    """
    if len(embs) < 2:
        return [], []

    D = len(embs[0])
    T = len(embs)

    # 충분통계량 (차원별 독립 → 평균 처리)
    # 스칼라화: 각 임베딩을 L2 norm으로 축약
    xs = np.array([float(np.linalg.norm(e)) for e in embs])

    # run-length 분포 R[t, r] = P(r_t = r | x_{1:t})
    # 초기: R[0, 0] = 1
    max_r = T + 1
    R = np.zeros((T + 1, max_r))
    R[0, 0] = 1.0

    # 하이퍼파라미터 (run-length별로 유지)
    mu    = np.full(max_r, mu0)
    kappa = np.full(max_r, kappa0)
    alpha = np.full(max_r, alpha0)
    beta  = np.full(max_r, beta0)

    cp_probs = []   # P(r_t = 0) 시계열
    detected = []

    for t in range(T):
        x = xs[t]

        # Predictive: Student-t
        # p(x | r) = StudentT(x; 2α_r, μ_r, β_r(κ_r+1)/(κ_r * α_r))
        from scipy.stats import t as student_t
        pred = np.zeros(t + 1)
        for r in range(t + 1):
            if R[t, r] < 1e-12:
                continue
            df  = 2 * alpha[r]
            loc = mu[r]
            sc  = np.sqrt(beta[r] * (kappa[r] + 1) / (kappa[r] * alpha[r]))
            pred[r] = student_t.pdf(x, df=df, loc=loc, scale=sc)

        # 성장 확률: r_{t+1} = r_t + 1
        growth = R[t, :t+1] * pred[:t+1] * (1 - hazard)
        # 변화점 확률: r_{t+1} = 0
        cp     = np.sum(R[t, :t+1] * pred[:t+1]) * hazard

        R[t+1, 1:t+2] = growth
        R[t+1, 0]     = cp

        # 정규화
        norm = R[t+1, :t+2].sum()
        if norm > 1e-12:
            R[t+1, :t+2] /= norm

        cp_prob = float(R[t+1, 0])
        cp_probs.append(cp_prob)

        if cp_prob > thresh and (not detected or t - detected[-1] >= 2):
            detected.append(t)

        # 하이퍼파라미터 업데이트 (conjugate Normal-Gamma)
        new_mu    = np.zeros(max_r)
        new_kappa = np.zeros(max_r)
        new_alpha = np.zeros(max_r)
        new_beta  = np.zeros(max_r)

        # r=0: 리셋
        new_mu[0]    = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0]  = beta0

        # r>0: 업데이트
        for r in range(1, t + 2):
            prev = r - 1
            kp   = kappa[prev]
            new_kappa[r] = kp + 1
            new_mu[r]    = (kp * mu[prev] + x) / (kp + 1)
            new_alpha[r] = alpha[prev] + 0.5
            new_beta[r]  = beta[prev] + (kp * (x - mu[prev]) ** 2) / (2 * (kp + 1))

        mu    = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta  = new_beta

    return detected, cp_probs


# ══════════════════════════════════════════════════════════
# 6. 방법론 3: NOUGAT  (긴 발화)
# ══════════════════════════════════════════════════════════

def rbf_kernel(x, y, sigma):
    d = x - y
    return float(np.exp(-np.dot(d, d) / (2 * sigma ** 2)))

def kernel_vector(x, dictionary, sigma):
    return np.array([rbf_kernel(x, d, sigma) for d in dictionary])

def coherence(x, dictionary, sigma):
    return float(max(rbf_kernel(x, d, sigma) for d in dictionary)) if dictionary else 0.0


class NOUGAT:
    def __init__(self, sigma):
        self.sigma = sigma
        self.dictionary: List[np.ndarray] = []
        self.theta = np.array([])
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
        h_ref  = np.mean([kernel_vector(v, self.dictionary, self.sigma) for v in ref_b],  axis=0)
        h_test = np.mean([kernel_vector(v, self.dictionary, self.sigma) for v in test_b], axis=0)
        kv     = np.array([kernel_vector(v, self.dictionary, self.sigma) for v in ref_b])
        H_ref  = kv.T @ kv / N_REF
        grad   = (H_ref + LAMBDA * np.eye(L)) @ self.theta - h_test + h_ref
        self.theta -= MU * grad
        g_t = float(self.theta @ h_test)
        self.g_values.append(g_t)
        self._n_obs += 1
        self._g_sum  += g_t
        self._g2_sum += g_t ** 2
        return g_t

    def detect(self) -> Tuple[List[int], float, List[float]]:
        from scipy import stats
        if not self.g_values:
            return [], 0.0, []
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
        return detected, xi, self.g_values


def run_nougat(embs: List[np.ndarray], sigma: float) -> Tuple[List[int], float, List[float]]:
    det = NOUGAT(sigma)
    for e in embs:
        det.update(e)
    warmup = N_REF + N_TEST - 1
    detected_raw, xi, g_vals = det.detect()
    # warmup 오프셋 적용
    detected = [d + warmup for d in detected_raw]
    return detected, xi, g_vals


# ══════════════════════════════════════════════════════════
# 7. 앙상블 디스패처
# ══════════════════════════════════════════════════════════

def ensemble_detect(
    embs: List[np.ndarray],
    duration: float,
    sigma: float,
    short_thresh: float = SHORT_THRESH,
    long_thresh: float  = LONG_THRESH,
) -> Dict:
    """
    발화 길이에 따라 방법론 자동 선택.

    반환:
        method        : 사용된 방법론 이름
        detected      : 탐지된 청크 인덱스 목록
        scores        : 방법론별 점수 시계열
        extra         : 방법론별 추가 정보 (xi, threshold 등)
    """
    if duration < short_thresh:
        # ── Cosine Drop ──
        detected, scores = detect_cosine_drop(embs, COSINE_THRESH)
        return {
            "method":   "cosine_drop",
            "detected": detected,
            "scores":   scores,
            "extra":    {"threshold": COSINE_THRESH},
        }
    elif duration < long_thresh:
        # ── BOCPD ──
        detected, scores = bocpd_gaussian(embs)
        return {
            "method":   "bocpd",
            "detected": detected,
            "scores":   scores,
            "extra":    {"threshold": BOCPD_THRESH},
        }
    else:
        # ── NOUGAT ──
        detected, xi, scores = run_nougat(embs, sigma)
        return {
            "method":   "nougat",
            "detected": detected,
            "scores":   scores,
            "extra":    {"xi": xi},
        }


# ══════════════════════════════════════════════════════════
# 8. 단일 파일 처리
# ══════════════════════════════════════════════════════════

def process_one(record: Dict, wmodel, device: str) -> Optional[Dict]:
    pcm_path   = record["pcm_path"]
    transcript = record["transcript"]
    file_id    = record["file_id"]

    if not os.path.exists(pcm_path):
        print(f"  [스킵] 파일 없음: {pcm_path}")
        return None

    audio    = load_pcm(pcm_path)
    duration = len(audio) / SAMPLE_RATE
    chunks   = split_chunks(audio)

    if len(chunks) < 2:
        print(f"  [스킵] 청크 부족 ({len(chunks)}개): {file_id}")
        return None

    timestamps, embeddings = extract_embeddings(chunks, wmodel, device)

    # 전처리
    if PCA_DIM and len(embeddings) > 2:
        embs_input, expl_var = reduce_dim(embeddings)
    else:
        embs_input, expl_var = embeddings, 1.0

    sigma_used = median_heuristic(embs_input) if AUTO_SIGMA else SIGMA

    # 앙상블 탐지
    result_det = ensemble_detect(embs_input, duration, sigma_used,
                                   SHORT_THRESH, LONG_THRESH)

    detected       = result_det["detected"]
    method         = result_det["method"]
    detected_times = [timestamps[d][0] for d in detected if d < len(timestamps)]

    return {
        "file_id":        file_id,
        "pcm_path":       pcm_path,
        "transcript":     transcript,
        "audio":          audio,
        "duration":       duration,
        "timestamps":     timestamps,
        "embeddings":     embeddings,
        "method":         method,
        "detected":       detected,
        "detected_times": detected_times,
        "scores":         result_det["scores"],
        "extra":          result_det["extra"],
        "sigma_used":     sigma_used,
        "expl_var":       expl_var,
    }


# ══════════════════════════════════════════════════════════
# 9. 시각화
# ══════════════════════════════════════════════════════════

METHOD_COLOR = {
    "cosine_drop": "#FF6B35",
    "bocpd":       "#7B2D8B",
    "nougat":      "#3A86FF",
}
METHOD_LABEL = {
    "cosine_drop": "Cosine Drop (짧은 발화 <5s)",
    "bocpd":       "BOCPD (중간 발화 5~15s)",
    "nougat":      "NOUGAT (긴 발화 ≥15s)",
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
        font  = next((f for f in pref if any(f in l for l in fonts)), "DejaVu Sans")
    matplotlib.rc("font", family=font)
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_figure_one(result: Dict, out_path: str):
    setup_korean_font()
    RED = "#EF233C"; GRAY = "#8D99AE"; BLUE = "#3A86FF"

    audio          = result["audio"]
    timestamps     = result["timestamps"]
    embeddings     = result["embeddings"]
    duration       = result["duration"]
    file_id        = result["file_id"]
    transcript     = result["transcript"]
    method         = result["method"]
    detected       = result["detected"]
    detected_times = result["detected_times"]
    scores         = result["scores"]
    extra          = result["extra"]
    expl_var       = result["expl_var"]
    sigma_used     = result["sigma_used"]
    n_chunks       = len(timestamps)
    m_color        = METHOD_COLOR[method]
    m_label        = METHOD_LABEL[method]

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Ensemble Changepoint Detection | {file_id}\n"
        f"방법: {m_label}  |  wav2vec2={WAV2VEC_MODEL.split('/')[-1]}  "
        f"PCA={PCA_DIM}차원({expl_var*100:.0f}%)  "
        f"σ={sigma_used:.2f}  전환점: {len(detected_times)}개",
        fontsize=11, fontweight="bold", y=0.99, color="#2B2D42",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.5, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ── 패널 1: 파형 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    t_ax = np.linspace(0, duration, len(audio))
    ax1.plot(t_ax, audio, color=BLUE, lw=0.4, alpha=0.7)
    for k, t in enumerate(detected_times):
        ax1.axvline(t, color=RED, lw=2, linestyle="-.", alpha=0.85,
                    label=f"전환점 ({len(detected_times)}개)" if k == 0 else None)
        ax1.text(t + 0.05, ax1.get_ylim()[1] * 0.82,
                 f"{t:.1f}s", color=RED, fontsize=8, fontweight="bold")
    for s, e in timestamps:
        ax1.axvline(s, color=GRAY, lw=0.4, alpha=0.2)
    ax1.set_title(
        f"파형  |  {file_id}  ({duration:.1f}s)  →  {m_label}",
        fontsize=10, fontweight="bold"
    )
    ax1.set_xlabel("시간 (초)", fontsize=9)
    ax1.set_ylabel("진폭", fontsize=9)
    if detected_times:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 패널 2: 방법론별 점수 ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")

    if scores:
        # 점수 x축: 청크 시작 시간
        if method == "cosine_drop":
            # scores[i] = embs[i]와 embs[i+1] 사이 거리
            s_times = [timestamps[i][0] for i in range(len(scores))]
            ax2.bar(s_times, scores, width=HOP_SEC * 0.8,
                    color=m_color, alpha=0.7, label="코사인 거리")
            thresh = extra.get("threshold", COSINE_THRESH)
            ax2.axhline(thresh, color=RED, lw=1.5, linestyle="--",
                        label=f"임계값 ({thresh:.2f})")
            ax2.set_ylabel("1 - 코사인 유사도", fontsize=9)
            ax2.set_title("Cosine Drop 점수", fontsize=11, fontweight="bold")

        elif method == "bocpd":
            s_times = [timestamps[i][0] for i in range(len(scores))]
            ax2.plot(s_times, scores, "o-", color=m_color, lw=2, ms=6,
                     label="P(변화점)")
            thresh = extra.get("threshold", BOCPD_THRESH)
            ax2.axhline(thresh, color=RED, lw=1.5, linestyle="--",
                        label=f"임계값 ({thresh:.2f})")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("변화점 사후확률 P(r=0)", fontsize=9)
            ax2.set_title("BOCPD 변화점 확률", fontsize=11, fontweight="bold")

        elif method == "nougat":
            warmup  = N_REF + N_TEST - 1
            s_times = [timestamps[i + warmup][0]
                       for i in range(len(scores))
                       if i + warmup < n_chunks]
            g_plot  = scores[:len(s_times)]
            ax2.plot(s_times, g_plot, "o-", color=m_color, lw=2, ms=6,
                     label="$g_t$")
            xi = extra.get("xi", 0)
            ax2.axhline( xi - 1, color=RED, lw=1.5, linestyle="--",
                        label=f"+ξ={xi:.3f}")
            ax2.axhline(-xi - 1, color=RED, lw=1.5, linestyle=":")
            ax2.axhline(0, color=GRAY, lw=1, alpha=0.4)
            ax2.set_ylabel("$g_t$", fontsize=9)
            ax2.set_title("NOUGAT 탐지 통계량", fontsize=11, fontweight="bold")

        for t in detected_times:
            ax2.axvline(t, color=RED, lw=1.8, linestyle="-.", alpha=0.75)

    ax2.set_xlabel("시간 (초)", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── 패널 3: PCA 궤적 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#FFFFFF")
    if len(embeddings) >= 3:
        emb2d = PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(np.array(embeddings)))
        cmap_p = matplotlib.cm.get_cmap("plasma")
        for i in range(n_chunks - 1):
            ax3.annotate("", xy=emb2d[i+1], xytext=emb2d[i],
                         arrowprops=dict(arrowstyle="->",
                                         color=cmap_p(i / n_chunks), lw=1.5))
        sc = ax3.scatter(emb2d[:, 0], emb2d[:, 1], c=range(n_chunks),
                         cmap="plasma", s=80, zorder=4,
                         edgecolors="white", lw=1)
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="청크 순서")
        for k, didx in enumerate(detected):
            if didx < n_chunks:
                ax3.scatter(*emb2d[didx], color=RED, s=200, zorder=5,
                            edgecolors="black", lw=1.5, marker="D",
                            label=f"{timestamps[didx][0]:.1f}s" if k < 4 else None)
        ax3.annotate("시작", emb2d[0], fontsize=8, color="#6A0572",
                     fontweight="bold", xytext=(5, 5), textcoords="offset points")
        ax3.annotate("끝",  emb2d[-1], fontsize=8, color="#B5179E",
                     fontweight="bold", xytext=(5, 5), textcoords="offset points")
        ax3.legend(fontsize=7, title="전환점(초)")
    ax3.set_title("청크 임베딩 궤적 (PCA 2D)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("PC1", fontsize=9); ax3.set_ylabel("PC2", fontsize=9)
    ax3.grid(alpha=0.25, linestyle="--")
    ax3.spines[["top", "right"]].set_visible(False)

    # ── 패널 4: 유사도 히트맵 ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    emb_arr  = np.array(embeddings)
    norms    = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    sim      = (emb_arr / (norms + 1e-9)) @ (emb_arr / (norms + 1e-9)).T
    cmap_h   = LinearSegmentedColormap.from_list(
        "bw", ["#EEF2FF", "#3A86FF", "#023E8A"])
    im = ax4.imshow(sim, cmap=cmap_h, aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="코사인 유사도")
    tick_step = max(1, n_chunks // 10)
    ticks     = list(range(0, n_chunks, tick_step))
    ax4.set_xticks(ticks); ax4.set_yticks(ticks)
    labels = [f"{timestamps[i][0]:.1f}s" for i in ticks]
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax4.set_yticklabels(labels, fontsize=7)
    for didx in detected:
        if didx < n_chunks:
            ax4.axvline(didx, color=RED, lw=1.5, linestyle="--", alpha=0.75)
            ax4.axhline(didx, color=RED, lw=1.5, linestyle="--", alpha=0.75)
    ax4.set_title("청크 유사도 히트맵", fontsize=11, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)

    # ── 패널 5: 전사 텍스트 구간 분할 ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF"); ax5.axis("off")
    total_chars  = len(transcript)
    colors       = [m_color, "#3A86FF", "#06D6A0", "#FF006E", "#7B2D8B"]
    boundaries   = [0.0] + sorted(detected_times) + [duration]
    y_pos        = 0.97
    for k in range(len(boundaries) - 1):
        t0, t1 = boundaries[k], boundaries[k + 1]
        c0 = int(t0 / duration * total_chars)
        c1 = int(t1 / duration * total_chars)
        seg = transcript[c0:c1].strip()
        if not seg:
            continue
        color = colors[k % len(colors)]
        ax5.text(0.02, y_pos,
                 f"▶ 구간 {k+1}  ({t0:.1f}s ~ {t1:.1f}s)",
                 transform=ax5.transAxes, fontsize=8.5,
                 fontweight="bold", color=color, va="top")
        y_pos -= 0.055
        disp = seg[:90] + ("…" if len(seg) > 90 else "")
        ax5.text(0.02, y_pos, disp, transform=ax5.transAxes,
                 fontsize=7.5, color="#2B2D42", va="top")
        y_pos -= 0.10
        if y_pos < 0.03:
            ax5.text(0.02, y_pos, "… (이하 생략)",
                     transform=ax5.transAxes, fontsize=7,
                     color=GRAY, va="top")
            break
    ax5.set_title("전사 텍스트 구간 분할", fontsize=11, fontweight="bold")
    for sp in ["top", "right", "bottom", "left"]:
        ax5.spines[sp].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def make_summary_figure(results: List[Dict], out_path: str):
    setup_korean_font()
    RED  = "#EF233C"; GRAY = "#8D99AE"

    n   = len(results)
    fig, ax = plt.subplots(figsize=(18, max(4, n * 0.6 + 2)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    max_dur = max(r["duration"] for r in results)

    for i, r in enumerate(results):
        y     = n - 1 - i
        color = METHOD_COLOR[r["method"]]
        ax.barh(y, r["duration"], color=color, alpha=0.2, height=0.6)
        for t in r["detected_times"]:
            ax.plot(t, y, marker="D", color=RED, ms=8, zorder=4)
        label = f"{r['file_id']}  [{r['method']}]  |  {r['transcript'][:35]}…"
        ax.text(-0.3, y, label, va="center", ha="right",
                fontsize=7, color="#2B2D42")
        ax.text(r["duration"] + 0.1, y,
                f"{len(r['detected_times'])}개",
                va="center", fontsize=7, color=RED)

    # 범례
    for method, color in METHOD_COLOR.items():
        ax.barh(-1, 0, color=color, alpha=0.4, label=METHOD_LABEL[method])
    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlim(-max_dur * 0.45, max_dur * 1.1)
    ax.set_ylim(-1.5, n - 0.2)
    ax.set_xlabel("시간 (초)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        f"Ensemble Changepoint Detection 요약  ({n}개 파일) | wav2vec2\n"
        f"◆ = 탐지된 전환점  |  색상 = 사용된 방법론",
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
# 10. 메인
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Audio Changepoint Detection (Cosine/BOCPD/NOUGAT)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--transcript_index", required=True,
                        help="전사 인덱스 파일 (.trn, EUC-KR)")
    parser.add_argument("--pcm_dir", default=None,
                        help="PCM 파일 폴더 (인덱스 경로가 상대경로일 때)")
    parser.add_argument("--out_dir", default="./ensemble_results",
                        help="결과 저장 폴더")
    parser.add_argument("--n_files", type=int, default=None,
                        help="처리할 파일 수 (기본: 전체)")
    parser.add_argument("--wav2vec_model", default=WAV2VEC_MODEL,
                        help="wav2vec2 모델명 (HuggingFace hub)")
    parser.add_argument("--short_thresh", type=float, default=SHORT_THRESH,
                        help=f"Cosine Drop 적용 상한 (기본: {SHORT_THRESH}초)")
    parser.add_argument("--long_thresh", type=float, default=LONG_THRESH,
                        help=f"NOUGAT 적용 하한 (기본: {LONG_THRESH}초)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Ensemble Changepoint Detection")
    print(f"  디바이스   : {device}")
    print(f"  wav2vec2   : {args.wav2vec_model}")
    print(f"  방법론 기준: <{args.short_thresh}s=Cosine  "
          f"{args.short_thresh}~{args.long_thresh}s=BOCPD  "
          f"≥{args.long_thresh}s=NOUGAT")
    print(f"  출력 폴더  : {args.out_dir}")
    print(f"{'='*60}")

    records = parse_transcript_index(
        args.transcript_index, args.pcm_dir, args.n_files)
    print(f"\n  처리 대상: {len(records)}개 파일")

    print(f"\n  Whisper 로딩 중...")
    wmodel = load_wav2vec(args.wav2vec_model, device)

    results = []
    method_counts = {"cosine_drop": 0, "bocpd": 0, "nougat": 0}

    for i, record in enumerate(records):
        print(f"\n{'─'*60}")
        print(f"  [{i+1:03d}/{len(records):03d}] {record['file_id']}")
        print(f"  전사: {record['transcript'][:60]}...")

        result = process_one(record, wmodel, device)
        if result is None:
            continue

        method_counts[result["method"]] += 1
        print(f"  길이: {result['duration']:.2f}s  "
              f"방법: {result['method']}  "
              f"전환점: {len(result['detected_times'])}개  "
              f"{[f'{t:.2f}s' for t in result['detected_times']]}")

        out_path = os.path.join(args.out_dir,
                                f"{result['file_id']}_ensemble.png")
        make_figure_one(result, out_path)
        print(f"  [저장] {out_path}")
        results.append(result)

    if results:
        summary_path = os.path.join(args.out_dir, "summary_ensemble.png")
        make_summary_figure(results, summary_path)

        print(f"\n{'='*60}")
        print(f"  처리 완료: {len(results)}/{len(records)}개")
        print(f"  방법론 사용 현황:")
        for m, cnt in method_counts.items():
            print(f"    {METHOD_LABEL[m]}: {cnt}개")
        print(f"  평균 전환점: "
              f"{np.mean([len(r['detected_times']) for r in results]):.2f}개/발화")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()