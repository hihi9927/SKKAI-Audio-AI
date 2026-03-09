"""
NOUGAT + Whisper: KsponSpeech PCM 배치 의미/주제 전환 탐지
================================================================
논문: "Online Change-Point Detection with Kernels"
      Ferrari et al., Pattern Recognition, 2022

전사 파일 형식 (EUC-KR):
    /path/to/KsponSpeech_E00001.pcm :: 전사 텍스트
    /path/to/KsponSpeech_E00002.pcm :: 전사 텍스트
    ...

실행:
    # 전사 파일의 처음 10개 파일 처리
    python nougat_audio_changepoint.py \
        --transcript_index /path/to/transcripts.txt \
        --pcm_dir          /path/to/pcm_folder \
        --out_dir          /path/to/output_dir \
        --n_files          10

    # 전체 파일 처리
    python nougat_audio_changepoint.py \
        --transcript_index /path/to/transcripts.txt \
        --pcm_dir          /path/to/pcm_folder \
        --out_dir          /path/to/output_dir

설치:
    pip install openai-whisper torch numpy matplotlib scikit-learn scipy
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
# 0. 실험 설정
# ══════════════════════════════════════════════════════════

# ── PCM 포맷 ──
SAMPLE_RATE = 16000   # Hz
CHANNELS    = 1       # mono

# ── 청크 설정 ──
CHUNK_SEC = 3.0       # 청크 길이 (초)
HOP_SEC   = 1.5       # 슬라이딩 hop (초), 50% overlap

# ── Whisper 모델 ──
WHISPER_MODEL = "small"   # tiny / base / small / medium / large

# ── NOUGAT 하이퍼파라미터 ──
N_REF  = 1
N_TEST = 1
MU     = 0.01
NU     = 0.5
L_MAX  = 20
LAMBDA = 1e-3

# ── 전처리 ──
PCA_DIM    = 32     # None이면 원본 차원
AUTO_SIGMA = True   # True: median heuristic
SIGMA      = 1.0    # AUTO_SIGMA=False일 때 고정값

FALSE_ALARM_RATE = 0.01


# ══════════════════════════════════════════════════════════
# 1. 전사 인덱스 파일 파싱
# ══════════════════════════════════════════════════════════

def parse_transcript_index(
    index_path: str,
    pcm_dir: Optional[str],
    n_files: Optional[int],
) -> List[Dict]:
    """
    전사 인덱스 파일 파싱.

    형식:
        /full/path/to/KsponSpeech_E00001.pcm :: 전사 텍스트
    또는:
        KsponSpeech_eval/eval_clean/KsponSpeech_E00001.pcm :: 전사 텍스트

    반환:
        [{"pcm_path": ..., "transcript": ..., "file_id": ...}, ...]
    """
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

        # PCM 경로 결정
        # 1) 절대경로면 그대로
        # 2) pcm_dir 지정 시 파일명만 추출해서 조합
        if os.path.isabs(raw_path) and os.path.exists(raw_path):
            pcm_path = raw_path
        elif pcm_dir:
            filename = os.path.basename(raw_path)
            pcm_path = os.path.join(pcm_dir, filename)
        else:
            pcm_path = raw_path

        # file_id: 확장자 제거한 파일명
        file_id = os.path.splitext(os.path.basename(pcm_path))[0]

        records.append({
            "pcm_path":   pcm_path,
            "transcript": transcript,
            "file_id":    file_id,
        })

    # n_files 옵션 적용
    if n_files is not None:
        records = records[:n_files]

    return records


# ══════════════════════════════════════════════════════════
# 2. PCM 로딩
# ══════════════════════════════════════════════════════════

def load_pcm(path: str) -> np.ndarray:
    """Headerless 16-bit little endian PCM → float32 [-1, 1]"""
    raw   = np.fromfile(path, dtype="<i2")
    audio = raw.astype(np.float32) / 32768.0
    return audio


# ══════════════════════════════════════════════════════════
# 3. 슬라이딩 청크 분할
# ══════════════════════════════════════════════════════════

def split_chunks(
    audio: np.ndarray,
    chunk_sec: float,
    hop_sec: float,
) -> List[Tuple[float, float, np.ndarray]]:
    chunk_n = int(chunk_sec * SAMPLE_RATE)
    hop_n   = int(hop_sec   * SAMPLE_RATE)
    chunks  = []
    start   = 0
    while start + chunk_n <= len(audio):
        chunks.append((start / SAMPLE_RATE,
                       (start + chunk_n) / SAMPLE_RATE,
                       audio[start:start + chunk_n]))
        start += hop_n
    # 잔여 청크 패딩
    if start < len(audio):
        chunk = audio[start:]
        chunk = np.pad(chunk, (0, chunk_n - len(chunk)), mode="constant")
        chunks.append((start / SAMPLE_RATE,
                       len(audio) / SAMPLE_RATE,
                       chunk))
    return chunks


# ══════════════════════════════════════════════════════════
# 4. Whisper encoder 임베딩
# ══════════════════════════════════════════════════════════

def load_whisper(model_name: str, device: str):
    import whisper
    model = whisper.load_model(model_name, device=device)
    return model


@torch.no_grad()
def get_chunk_embedding(chunk: np.ndarray, wmodel, device: str) -> np.ndarray:
    import whisper
    # Whisper encoder는 정확히 30초(480000 샘플) 입력을 요구함
    # 짧은 청크는 0으로 패딩, 긴 청크는 잘라냄
    N_SAMPLES_30S = 16000 * 30
    if len(chunk) < N_SAMPLES_30S:
        chunk = np.pad(chunk, (0, N_SAMPLES_30S - len(chunk)), mode="constant")
    else:
        chunk = chunk[:N_SAMPLES_30S]
    tensor = torch.tensor(chunk, dtype=torch.float32).to(device)
    mel    = whisper.log_mel_spectrogram(tensor, n_mels=wmodel.dims.n_mels)
    mel    = mel.unsqueeze(0)
    out    = wmodel.encoder(mel)          # [1, T, D]
    return out.squeeze(0).mean(dim=0).cpu().numpy()   # [D]


def extract_embeddings(
    chunks: List[Tuple[float, float, np.ndarray]],
    wmodel,
    device: str,
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
    timestamps, embeddings = [], []
    for start, end, chunk in chunks:
        emb = get_chunk_embedding(chunk, wmodel, device)
        timestamps.append((start, end))
        embeddings.append(emb)
    return timestamps, embeddings


# ══════════════════════════════════════════════════════════
# 5. 전처리: PCA + Median Heuristic
# ══════════════════════════════════════════════════════════

def median_heuristic(embs: List[np.ndarray]) -> float:
    arr   = np.array(embs)
    n     = len(arr)
    dists = [float(np.linalg.norm(arr[i] - arr[j]))
             for i in range(n) for j in range(i + 1, n)]
    return float(max(np.median(dists), 1e-3)) if dists else 1.0


def reduce_dim(
    embs: List[np.ndarray],
    n_components: int,
) -> Tuple[List[np.ndarray], float]:
    arr    = np.array(embs)
    n_comp = min(n_components, arr.shape[0] - 1, arr.shape[1])
    scaled = StandardScaler().fit_transform(arr)
    pca    = PCA(n_components=n_comp, random_state=42)
    red    = pca.fit_transform(scaled)
    return [red[i] for i in range(len(embs))], float(pca.explained_variance_ratio_.sum())


# ══════════════════════════════════════════════════════════
# 6. NOUGAT
# ══════════════════════════════════════════════════════════

def rbf_kernel(x, y, sigma):
    d = x - y
    return float(np.exp(-np.dot(d, d) / (2 * sigma ** 2)))

def kernel_vector(x, dictionary, sigma):
    return np.array([rbf_kernel(x, d, sigma) for d in dictionary])

def coherence(x, dictionary, sigma):
    return float(max(rbf_kernel(x, d, sigma) for d in dictionary)) if dictionary else 0.0


class NOUGAT:
    def __init__(self, n_ref, n_test, mu, sigma, nu, l_max, lam):
        self.n_ref = n_ref; self.n_test = n_test
        self.mu = mu; self.sigma = sigma; self.nu = nu
        self.l_max = l_max; self.lam = lam
        self.dictionary: List[np.ndarray] = []
        self.theta = np.array([])
        self.buffer: List[np.ndarray] = []
        self.g_values: List[float] = []
        self._g_sum = self._g2_sum = 0.0
        self._n_obs = 0

    def _update_dict(self, x):
        if coherence(x, self.dictionary, self.sigma) < self.nu:
            if len(self.dictionary) < self.l_max:
                self.dictionary.append(x.copy())
                self.theta = np.append(self.theta, 0.0)

    def update(self, x) -> Optional[float]:
        self.buffer.append(x.copy())
        if len(self.buffer) > self.n_ref + self.n_test:
            self.buffer.pop(0)
        self._update_dict(x)
        if len(self.buffer) < self.n_ref + self.n_test or not self.dictionary:
            return None
        L      = len(self.dictionary)
        ref_b  = self.buffer[:self.n_ref]
        test_b = self.buffer[self.n_ref:]
        h_ref  = np.mean([kernel_vector(v, self.dictionary, self.sigma) for v in ref_b],  axis=0)
        h_test = np.mean([kernel_vector(v, self.dictionary, self.sigma) for v in test_b], axis=0)
        kv     = np.array([kernel_vector(v, self.dictionary, self.sigma) for v in ref_b])
        H_ref  = kv.T @ kv / self.n_ref
        grad   = (H_ref + self.lam * np.eye(L)) @ self.theta - h_test + h_ref
        self.theta -= self.mu * grad
        g_t = float(self.theta @ h_test)
        self.g_values.append(g_t)
        self._n_obs += 1; self._g_sum += g_t; self._g2_sum += g_t ** 2
        return g_t

    def get_threshold(self, far=0.05) -> float:
        from scipy import stats
        if self._n_obs < 3:
            return float(np.mean(np.abs(self.g_values)) + np.std(self.g_values)) if self.g_values else 0.1
        mean_g = self._g_sum / self._n_obs
        var_g  = max(self._g2_sum / self._n_obs - mean_g ** 2, 1e-9)
        z      = stats.norm.ppf(1 - far / 2)
        return float(abs(mean_g) + z * np.sqrt(var_g))

    def detect(self, far=0.05) -> Tuple[List[int], float]:
        if not self.g_values:
            return [], 0.0
        xi = self.get_threshold(far)
        detected, prev = [], -999
        for i, g in enumerate(self.g_values):
            if abs(g + 1) > xi and i - prev >= 2:
                detected.append(i)
                prev = i
        return detected, xi


# ══════════════════════════════════════════════════════════
# 7. 단일 파일 처리 파이프라인
# ══════════════════════════════════════════════════════════

def process_one(
    record: Dict,
    wmodel,
    device: str,
) -> Optional[Dict]:
    """
    PCM 1개 파일에 대해 전체 파이프라인 실행.
    파일 없으면 None 반환.
    """
    pcm_path   = record["pcm_path"]
    transcript = record["transcript"]
    file_id    = record["file_id"]

    if not os.path.exists(pcm_path):
        print(f"  [스킵] 파일 없음: {pcm_path}")
        return None

    # PCM 로딩
    audio    = load_pcm(pcm_path)
    duration = len(audio) / SAMPLE_RATE

    # 청크 분할
    chunks = split_chunks(audio, CHUNK_SEC, HOP_SEC)
    if len(chunks) < N_REF + N_TEST + 1:
        print(f"  [스킵] 청크 부족 ({len(chunks)}개): {file_id}")
        return None

    # Whisper 임베딩
    timestamps, embeddings = extract_embeddings(chunks, wmodel, device)

    # PCA
    if PCA_DIM is not None and len(embeddings) > 2:
        embs_input, expl_var = reduce_dim(embeddings, PCA_DIM)
    else:
        embs_input, expl_var = embeddings, 1.0

    # σ
    sigma_used = median_heuristic(embs_input) if AUTO_SIGMA else SIGMA

    # NOUGAT
    detector = NOUGAT(N_REF, N_TEST, MU, sigma_used, NU, L_MAX, LAMBDA)
    for emb in embs_input:
        detector.update(emb)

    detected, xi = detector.detect(FALSE_ALARM_RATE)
    warmup = N_REF + N_TEST - 1
    detected_times = [timestamps[d + warmup][0]
                      for d in detected if d + warmup < len(timestamps)]

    return {
        "file_id":        file_id,
        "pcm_path":       pcm_path,
        "transcript":     transcript,
        "audio":          audio,
        "duration":       duration,
        "timestamps":     timestamps,
        "embeddings":     embeddings,
        "g_values":       detector.g_values,
        "xi":             xi,
        "detected":       detected,
        "detected_times": detected_times,
        "sigma_used":     sigma_used,
        "expl_var":       expl_var,
    }


# ══════════════════════════════════════════════════════════
# 8. 시각화
# ══════════════════════════════════════════════════════════

def setup_korean_font():
    import platform, subprocess
    system = platform.system()
    if system == "Darwin":     font = "AppleGothic"
    elif system == "Windows":  font = "Malgun Gothic"
    else:
        result = subprocess.run(["fc-list", ":lang=ko", "--format=%{family}\n"],
                                capture_output=True, text=True)
        fonts     = result.stdout.strip().split("\n")
        preferred = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR"]
        font = next((f for f in preferred if any(f in l for l in fonts)), "DejaVu Sans")
    matplotlib.rc("font", family=font)
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_figure_one(result: Dict, out_path: str):
    """단일 파일 결과 시각화 (5패널)"""
    setup_korean_font()

    BLUE = "#3A86FF"; ORANGE = "#FF6B35"; RED = "#EF233C"
    GREEN = "#06D6A0"; GRAY = "#8D99AE"; PURPLE = "#7B2D8B"

    audio          = result["audio"]
    timestamps     = result["timestamps"]
    embeddings     = result["embeddings"]
    g_values       = result["g_values"]
    xi             = result["xi"]
    detected       = result["detected"]
    detected_times = result["detected_times"]
    transcript     = result["transcript"]
    duration       = result["duration"]
    file_id        = result["file_id"]
    sigma_used     = result["sigma_used"]
    expl_var       = result["expl_var"]
    warmup         = N_REF + N_TEST - 1
    n_chunks       = len(timestamps)

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"NOUGAT + Whisper | {file_id}\n"
        f"모델: {WHISPER_MODEL}  PCA={PCA_DIM}차원 ({expl_var*100:.1f}%)  "
        f"σ={sigma_used:.3f}  N_ref={N_REF}  N_test={N_TEST}  μ={MU}  "
        f"전환점: {len(detected_times)}개",
        fontsize=12, fontweight="bold", y=0.99, color="#2B2D42",
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.5, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # ── 패널 1: 파형 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#FFFFFF")
    t_axis = np.linspace(0, duration, len(audio))
    ax1.plot(t_axis, audio, color=BLUE, lw=0.4, alpha=0.7, label="파형")
    for k, t in enumerate(detected_times):
        ax1.axvline(t, color=RED, lw=2, linestyle="-.", alpha=0.85,
                    label=f"전환점 ({len(detected_times)}개)" if k == 0 else None)
        ax1.text(t + 0.05, ax1.get_ylim()[1] * 0.82,
                 f"{t:.1f}s", color=RED, fontsize=8, fontweight="bold")
    for s, e in timestamps:
        ax1.axvline(s, color=GRAY, lw=0.4, alpha=0.25)
    ax1.set_title(f"파형 + 탐지된 의미 전환점  |  {file_id}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("시간 (초)", fontsize=9)
    ax1.set_ylabel("진폭", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)

    # ── 패널 2: g_t 통계량 ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#FFFFFF")
    g_times = [timestamps[i + warmup][0] for i in range(len(g_values))
               if i + warmup < n_chunks]
    ax2.plot(g_times, g_values[:len(g_times)], "o-", color=BLUE,
             lw=2, ms=6, label="$g_t$", zorder=3)
    ax2.axhline( xi - 1, color=RED, lw=1.5, linestyle="--", label=f"+ξ={xi:.3f}", alpha=0.8)
    ax2.axhline(-xi - 1, color=RED, lw=1.5, linestyle=":", label="-ξ", alpha=0.8)
    ax2.axhline(0, color=GRAY, lw=1, alpha=0.4)
    for t in detected_times:
        ax2.axvline(t, color=RED, lw=1.8, linestyle="-.", alpha=0.8)
    ax2.set_title("NOUGAT 탐지 통계량 $g_t$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("시간 (초)", fontsize=9)
    ax2.set_ylabel("$g_t = \\theta_t^T h_{test,t}$", fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25, linestyle="--")
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
                         arrowprops=dict(arrowstyle="->", color=cmap_p(i / n_chunks), lw=1.5))
        sc = ax3.scatter(emb2d[:, 0], emb2d[:, 1], c=range(n_chunks),
                         cmap="plasma", s=80, zorder=4, edgecolors="white", lw=1)
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="청크 순서")
        for k, didx in enumerate(detected):
            cidx = didx + warmup
            if cidx < n_chunks:
                ax3.scatter(*emb2d[cidx], color=RED, s=200, zorder=5,
                            edgecolors="black", lw=1.5, marker="D",
                            label=f"{timestamps[cidx][0]:.1f}s" if k < 4 else None)
        ax3.annotate("시작", emb2d[0], fontsize=8, color="#6A0572",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        ax3.annotate("끝", emb2d[-1], fontsize=8, color="#B5179E",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        ax3.legend(fontsize=7, title="전환점(초)")
    ax3.set_title("청크 임베딩 궤적 (PCA 2D)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("PC1", fontsize=9); ax3.set_ylabel("PC2", fontsize=9)
    ax3.grid(alpha=0.25, linestyle="--")
    ax3.spines[["top", "right"]].set_visible(False)

    # ── 패널 4: 유사도 히트맵 ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#FFFFFF")
    emb_arr = np.array(embeddings)
    norms   = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    sim     = (emb_arr / (norms + 1e-9)) @ (emb_arr / (norms + 1e-9)).T
    cmap_h  = LinearSegmentedColormap.from_list("bw", ["#EEF2FF", "#3A86FF", "#023E8A"])
    im = ax4.imshow(sim, cmap=cmap_h, aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="코사인 유사도")
    tick_step = max(1, n_chunks // 10)
    ticks = list(range(0, n_chunks, tick_step))
    ax4.set_xticks(ticks); ax4.set_yticks(ticks)
    labels = [f"{timestamps[i][0]:.1f}s" for i in ticks]
    ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax4.set_yticklabels(labels, fontsize=7)
    for didx in detected:
        cidx = didx + warmup
        if cidx < n_chunks:
            ax4.axvline(cidx, color=RED, lw=1.5, linestyle="--", alpha=0.75)
            ax4.axhline(cidx, color=RED, lw=1.5, linestyle="--", alpha=0.75)
    ax4.set_title("청크 유사도 히트맵", fontsize=11, fontweight="bold")
    ax4.spines[["top", "right"]].set_visible(False)

    # ── 패널 5: 전사 텍스트 구간 분할 ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#FFFFFF"); ax5.axis("off")
    total_chars = len(transcript)
    colors = [BLUE, PURPLE, ORANGE, GREEN, "#FF6B6B"]
    boundaries = [0.0] + sorted(detected_times) + [duration]
    y_pos = 0.97
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
            ax5.text(0.02, y_pos, "… (이하 생략)", transform=ax5.transAxes,
                     fontsize=7, color=GRAY, va="top")
            break
    ax5.set_title("전사 텍스트 구간 분할", fontsize=11, fontweight="bold")
    for sp in ["top", "right", "bottom", "left"]:
        ax5.spines[sp].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def make_summary_figure(results: List[Dict], out_path: str):
    """
    여러 파일의 탐지 결과 요약: 파일별 전환점 타임라인
    """
    setup_korean_font()
    RED  = "#EF233C"; BLUE = "#3A86FF"; GRAY = "#8D99AE"

    n = len(results)
    fig, ax = plt.subplots(figsize=(18, max(4, n * 0.5 + 2)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    max_dur = max(r["duration"] for r in results)

    for i, r in enumerate(results):
        y = n - 1 - i
        # 발화 길이 막대
        ax.barh(y, r["duration"], color=BLUE, alpha=0.15, height=0.6)
        # 전환점 마커
        for t in r["detected_times"]:
            ax.plot(t, y, marker="D", color=RED, ms=8, zorder=4)
        # 파일명 + 전사 텍스트 (앞 30자)
        label = f"{r['file_id']}  |  {r['transcript'][:40]}…"
        ax.text(-0.5, y, label, va="center", ha="right", fontsize=7,
                color="#2B2D42")
        ax.text(r["duration"] + 0.2, y,
                f"{len(r['detected_times'])}개",
                va="center", fontsize=7, color=RED)

    ax.set_xlim(-max_dur * 0.35, max_dur * 1.1)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("시간 (초)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        f"NOUGAT 의미 전환점 탐지 요약  ({n}개 파일)\n"
        f"◆ = 탐지된 전환점",
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
        description="NOUGAT + Whisper: KsponSpeech PCM 배치 의미 전환 탐지",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--transcript_index", required=True,
        help="전사 인덱스 파일 경로 (EUC-KR)\n"
             "형식: /path/to/file.pcm :: 전사텍스트",
    )
    parser.add_argument(
        "--pcm_dir", default=None,
        help="PCM 파일이 있는 폴더 (인덱스 경로가 상대경로일 때 사용)",
    )
    parser.add_argument(
        "--out_dir", default="./nougat_results",
        help="결과 이미지 저장 폴더 (기본: ./nougat_results)",
    )
    parser.add_argument(
        "--n_files", type=int, default=None,
        help="처리할 파일 수 (기본: 전체)\n예: --n_files 10",
    )
    parser.add_argument(
        "--whisper_model", default=WHISPER_MODEL,
        help=f"Whisper 모델 크기 (기본: {WHISPER_MODEL})\n"
             "선택: tiny / base / small / medium / large",
    )
    args = parser.parse_args()

    # 전역 Whisper 모델 이름 반영
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  NOUGAT + Whisper 배치 처리")
    print(f"  디바이스 : {device}")
    print(f"  Whisper  : {args.whisper_model}")
    print(f"  출력 폴더: {args.out_dir}")
    print(f"{'='*60}")

    # ── 1. 전사 인덱스 파싱 ──
    records = parse_transcript_index(
        args.transcript_index, args.pcm_dir, args.n_files
    )
    print(f"\n  처리 대상: {len(records)}개 파일")

    # ── 2. Whisper 로딩 (1회만) ──
    print(f"\n  Whisper 로딩 중...")
    wmodel = load_whisper(args.whisper_model, device)

    # ── 3. 배치 처리 ──
    results = []
    for i, record in enumerate(records):
        print(f"\n{'─'*60}")
        print(f"  [{i+1:03d}/{len(records):03d}] {record['file_id']}")
        print(f"  전사: {record['transcript'][:60]}...")
        print(f"{'─'*60}")

        result = process_one(record, wmodel, device)
        if result is None:
            continue

        print(f"  길이: {result['duration']:.2f}s  |  "
              f"청크: {len(result['timestamps'])}개  |  "
              f"전환점: {len(result['detected_times'])}개  "
              f"{[f'{t:.2f}s' for t in result['detected_times']]}")
        print(f"  σ={result['sigma_used']:.4f}  ξ={result['xi']:.5f}  "
              f"dict={len(result['detected'])}  "
              f"PCA설명분산={result['expl_var']*100:.1f}%")

        # 개별 결과 저장
        out_path = os.path.join(args.out_dir, f"{result['file_id']}_nougat.png")
        make_figure_one(result, out_path)
        print(f"  [저장] {out_path}")

        results.append(result)

    # ── 4. 요약 시각화 ──
    if results:
        summary_path = os.path.join(args.out_dir, "summary_nougat.png")
        make_summary_figure(results, summary_path)

        print(f"\n{'='*60}")
        print(f"  처리 완료: {len(results)}/{len(records)}개")
        print(f"  평균 전환점 수: {np.mean([len(r['detected_times']) for r in results]):.2f}개/발화")
        print(f"  요약 이미지: {summary_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()