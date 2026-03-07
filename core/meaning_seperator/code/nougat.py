"""
NOUGAT + KoBERT: 문장 내 의미 분절 탐지
================================================================
논문: "Online Change-Point Detection with Kernels"
      Ferrari et al., Pattern Recognition, 2022

논문 알고리즘을 그대로 구현:
  - RBF 커널 기반 RKHS에서 밀도비 r(y)-1 추정
  - KLMS 온라인 경사하강법으로 θ 업데이트
  - Coherence rule로 딕셔너리 크기 L 유지
  - 탐지 통계량: g_t = θ_t^T h_test_t
  - 임계값: MIA 기반 점근적 가우시안 분포 (평균/분산 추적)

입력: KoBERT 공백 단위 단어 임베딩 스트림
      (단어가 하나씩 들어올 때마다 온라인으로 처리)

설치:
    pip install torch transformers sentencepiece matplotlib scikit-learn numpy

실행:
    python nougat_kobert_changepoint.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple, Optional

# ══════════════════════════════════════════════════════════
# 0. 실험 설정
# ══════════════════════════════════════════════════════════

MODEL_NAME = "klue/bert-base"

SENTENCES = [
    {
        "text": "날씨가 맑지만 내일은 비가 올 것이다.",
        "label": "역접 (맑지만)",
        "hint": "맑지만",
    },
    {
        "text": "그는 열심히 공부했지만 시험에 떨어졌다.",
        "label": "역접 (공부했지만)",
        "hint": "공부했지만",
    },
    {
        "text": "봄이 지나고 여름이 되면 무더위가 시작된다.",
        "label": "시간 전환 (되면)",
        "hint": "되면",
    },
    {
        "text": "인공지능은 편리하지만 개인정보 침해 우려도 있다.",
        "label": "역접 (편리하지만)",
        "hint": "편리하지만",
    },
]

# ── NOUGAT 하이퍼파라미터 (논문 Table 1 기준) ──
N_REF  = 1      # 참조 윈도우 크기 (짧은 문장이라 2~3 권장)
N_TEST = 1      # 테스트 윈도우 크기
MU     = 0.01   # KLMS 학습률 (learning rate)
SIGMA  = 1.0    # RBF 커널 대역폭 (bandwidth)
NU     = 0.5    # Coherence rule 임계값 (0~1, 낮을수록 딕셔너리 크기↑)
L_MAX  = 20     # 딕셔너리 최대 크기
LAMBDA = 1e-3   # 정규화 계수

# PCA 차원 축소 (None이면 원본 768차원 사용)
# 고차원에서 RBF 커널이 모두 0에 가까워지는 문제 해결
PCA_DIM = 32

# σ 자동 설정 여부 (True: median heuristic, False: SIGMA 값 사용)
AUTO_SIGMA = True

# 탐지 임계값: None이면 MIA 기반 자동 설정 (False alarm rate)
FALSE_ALARM_RATE = 0.05   # 5% 유의수준

OUTPUT_PATH = "/home/ubuntu/STiTy/core/meaning_seperator/results/seg_emb/nougat_kobert_result.png"


# ══════════════════════════════════════════════════════════
# 1. 모델 로딩
# ══════════════════════════════════════════════════════════

def load_model(model_name: str):
    print(f"\n{'='*60}")
    print(f"  모델 로딩: {model_name}")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  디바이스: {device} | 로딩 완료\n")
    return tokenizer, model, device


# ══════════════════════════════════════════════════════════
# 2. KoBERT 공백 단위 임베딩 추출
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def get_word_embeddings(
    sentence: str,
    tokenizer,
    model,
    device: str,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    공백 단위로 단어를 증분하면서 각 단어의 서브워드 평균 임베딩 반환.
    NOUGAT의 입력 스트림 y_1, y_2, ..., y_T 에 해당.
    """
    words = sentence.split()
    word_embs = []

    for i in range(1, len(words) + 1):
        partial_text = " ".join(words[:i])
        encoding = tokenizer(
            partial_text, return_tensors="pt",
            truncation=True, max_length=128,
        )
        encoding_pt = {k: v.to(device) for k, v in encoding.items()}
        out = model(**encoding_pt)
        hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # [seq_len, H]

        # 마지막 추가 단어의 서브워드 위치 → 평균
        enc_obj = tokenizer(partial_text, truncation=True, max_length=128)
        word_ids = enc_obj.word_ids()
        target_idx = i - 1
        positions = [p for p, wid in enumerate(word_ids) if wid == target_idx]
        word_emb = hidden[positions].mean(axis=0) if positions else hidden[-2]
        word_embs.append(word_emb)

    return words, word_embs



# ══════════════════════════════════════════════════════════
# 3. 전처리: PCA 차원 축소 + Median Heuristic σ 설정
# ══════════════════════════════════════════════════════════

def median_heuristic(embs: List[np.ndarray]) -> float:
    """
    RBF 커널 대역폭 σ 자동 설정: pairwise 거리의 중앙값.
    고차원 임베딩에서 σ=1.0은 k(x,y)≈0 문제를 일으킴.
    median heuristic이 커널 기계학습의 표준 휴리스틱.
    """
    arr = np.array(embs)
    n = len(arr)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(arr[i] - arr[j])))
    if not dists:
        return 1.0
    sigma = float(np.median(dists))
    return max(sigma, 1e-3)   # 0 방지


def reduce_dim(
    embs: List[np.ndarray],
    n_components: int,
) -> Tuple[List[np.ndarray], object]:
    """
    PCA로 임베딩 차원 축소.
    768차원 → n_components차원으로 줄여 커널 계산을 안정화.
    fit은 전체 임베딩으로, transform도 동일 PCA 적용.
    """
    arr = np.array(embs)
    # 최대 n_components는 샘플 수 - 1
    n_comp = min(n_components, arr.shape[0] - 1, arr.shape[1])
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)
    pca = PCA(n_components=n_comp, random_state=42)
    arr_reduced = pca.fit_transform(arr_scaled)
    reduced = [arr_reduced[i] for i in range(len(embs))]
    explained = float(pca.explained_variance_ratio_.sum())
    return reduced, (scaler, pca, explained)


# ══════════════════════════════════════════════════════════
# 4. NOUGAT 알고리즘 (논문 Algorithm 1)
# ══════════════════════════════════════════════════════════

def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """
    RBF (Gaussian) 커널:
        k(x, y) = exp(-||x - y||² / (2σ²))
    """
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma ** 2)))


def kernel_vector(x: np.ndarray, dictionary: List[np.ndarray], sigma: float) -> np.ndarray:
    """
    k(x, d_i) for all d_i in dictionary → shape [L]
    """
    return np.array([rbf_kernel(x, d, sigma) for d in dictionary])


def coherence(x: np.ndarray, dictionary: List[np.ndarray], sigma: float) -> float:
    """
    Coherence: max_i k(x, d_i) / k(x, x)
    = max_i k(x, d_i)  (RBF라서 k(x,x)=1)
    """
    if not dictionary:
        return 0.0
    return float(max(rbf_kernel(x, d, sigma) for d in dictionary))


class NOUGAT:
    """
    논문 Algorithm 1: NOUGAT online change-point detector

    핵심 업데이트 (수식 7):
        θ_{t+1} = θ_t - μ * ∇J_{t+1}(θ_t)

    손실 기울기 (수식 6):
        ∇J_t(θ) = (θ^T H_ref + λI) θ - h_test_t + h_ref_t
                  ≈ H_ref θ - h_test + h_ref   (단순화)

    탐지 통계량 (수식 8):
        g_t = θ_t^T h_test_t

    귀무가설 하에서 g_t → N(0, σ²_g) (MIA 근사)
    """

    def __init__(
        self,
        n_ref: int,
        n_test: int,
        mu: float,
        sigma: float,
        nu: float,
        l_max: int,
        lam: float,
    ):
        self.n_ref  = n_ref
        self.n_test = n_test
        self.mu     = mu
        self.sigma  = sigma
        self.nu     = nu
        self.l_max  = l_max
        self.lam    = lam

        # 상태 변수
        self.dictionary: List[np.ndarray] = []   # 딕셔너리 {d_1, ..., d_L}
        self.theta: np.ndarray = np.array([])     # 가중치 벡터 θ ∈ R^L

        # 슬라이딩 버퍼
        self.buffer: List[np.ndarray] = []        # 최근 n_ref + n_test 샘플

        # 통계량 기록
        self.g_values: List[float] = []
        self.h_test_norms: List[float] = []

        # MIA 기반 분산 추적 (귀무가설 하)
        self._g_sum  = 0.0
        self._g2_sum = 0.0
        self._n_obs  = 0

    def _update_dictionary(self, x: np.ndarray):
        """Coherence rule: 기존 딕셔너리와 충분히 다를 때만 추가"""
        if coherence(x, self.dictionary, self.sigma) < self.nu:
            if len(self.dictionary) < self.l_max:
                self.dictionary.append(x.copy())
                # θ 크기 확장 (새 원소는 0으로 초기화)
                self.theta = np.append(self.theta, 0.0)

    def _compute_kernel_means(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        현재 버퍼에서 h_test, h_ref, H_ref 계산 (수식 3, 4, 5)

        h_test = (1/N_test) Σ_{i=N_ref}^{N_ref+N_test-1} k(buffer[i], ·)
        h_ref  = (1/N_ref)  Σ_{i=0}^{N_ref-1}            k(buffer[i], ·)
        H_ref  = (1/N_ref)  Σ_{i=0}^{N_ref-1}            k(buffer[i], ·) k(buffer[i], ·)^T
        """
        L = len(self.dictionary)
        if L == 0:
            return np.zeros(0), np.zeros(0), np.zeros((0, 0))

        ref_buf  = self.buffer[:self.n_ref]
        test_buf = self.buffer[self.n_ref:]

        h_ref  = np.mean([kernel_vector(x, self.dictionary, self.sigma)
                          for x in ref_buf], axis=0)           # [L]
        h_test = np.mean([kernel_vector(x, self.dictionary, self.sigma)
                          for x in test_buf], axis=0)          # [L]
        kv_ref = np.array([kernel_vector(x, self.dictionary, self.sigma)
                           for x in ref_buf])                  # [N_ref, L]
        H_ref  = kv_ref.T @ kv_ref / self.n_ref               # [L, L]

        return h_test, h_ref, H_ref

    def update(self, x: np.ndarray) -> Optional[float]:
        """
        새 샘플 x_t 하나를 받아 θ를 한 스텝 업데이트하고 통계량 g_t 반환.

        버퍼가 n_ref + n_test만큼 안 차면 None 반환 (워밍업 구간).
        """
        # 버퍼 업데이트 (슬라이딩 윈도우)
        self.buffer.append(x.copy())
        if len(self.buffer) > self.n_ref + self.n_test:
            self.buffer.pop(0)

        # 딕셔너리 업데이트
        self._update_dictionary(x)

        # 워밍업: 버퍼 미달 또는 딕셔너리 비어있음
        if len(self.buffer) < self.n_ref + self.n_test or len(self.dictionary) == 0:
            return None

        L = len(self.dictionary)
        h_test, h_ref, H_ref = self._compute_kernel_means()

        # ── KLMS 경사하강 업데이트 (논문 수식 7) ──
        # ∇J = (H_ref + λI) θ - h_test + h_ref
        grad = (H_ref + self.lam * np.eye(L)) @ self.theta - h_test + h_ref
        self.theta = self.theta - self.mu * grad

        # ── 탐지 통계량 (논문 수식 8) ──
        g_t = float(self.theta @ h_test)
        self.g_values.append(g_t)

        # MIA 분산 추적 (귀무가설 하 샘플 누적)
        self._n_obs  += 1
        self._g_sum  += g_t
        self._g2_sum += g_t ** 2

        self.h_test_norms.append(float(np.linalg.norm(h_test)))
        return g_t

    def get_threshold(self, false_alarm_rate: float = 0.05) -> float:
        """
        MIA 기반 점근적 가우시안 임계값 (논문 Section 3.3).
        귀무가설 하: g_t ~ N(0, σ²_g)
        단측 검정: |g_t| > ξ  → 변화점
        """
        from scipy import stats
        if self._n_obs < 3:
            # 샘플 부족 → 경험적 fallback
            return float(np.mean(np.abs(self.g_values)) +
                         np.std(self.g_values)) if self.g_values else 0.1

        mean_g = self._g_sum / self._n_obs
        var_g  = max(self._g2_sum / self._n_obs - mean_g ** 2, 1e-9)
        std_g  = np.sqrt(var_g)

        # 양측 검정 임계값: P(|Z| > z_α/2) = α
        z = stats.norm.ppf(1 - false_alarm_rate / 2)
        return float(abs(mean_g) + z * std_g)

    def detect(self, false_alarm_rate: float = 0.05) -> Tuple[List[int], float]:
        """
        기록된 g_t 값에서 변화점 탐지.
        |g_t + 1| > ξ  (논문 수식 9: r(y)-1을 추정하므로 +1 보정)
        """
        if not self.g_values:
            return [], 0.0

        xi = self.get_threshold(false_alarm_rate)
        detected = []
        prev = -999
        for i, g in enumerate(self.g_values):
            # 논문 수식 9: 변화점 존재 시 |g_t + 1| > ξ
            if abs(g + 1) > xi:
                if i - prev >= 2:   # 최소 거리 2 (NMS)
                    detected.append(i)
                    prev = i
        return detected, xi


# ══════════════════════════════════════════════════════════
# 4. 시각화
# ══════════════════════════════════════════════════════════

def setup_korean_font():
    import platform, subprocess
    system = platform.system()
    if system == "Darwin":
        font = "AppleGothic"
    elif system == "Windows":
        font = "Malgun Gothic"
    else:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "--format=%{family}\n"],
            capture_output=True, text=True,
        )
        fonts = result.stdout.strip().split("\n")
        preferred = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR"]
        font = next((f for f in preferred if any(f in l for l in fonts)), "DejaVu Sans")
    matplotlib.rc("font", family=font)
    matplotlib.rcParams["axes.unicode_minus"] = False


def find_hint_idx(words: List[str], hint: str) -> Optional[int]:
    hint_clean = hint.replace(" ", "")
    for i, w in enumerate(words):
        if hint_clean in w.replace(" ", "") or w.replace(" ", "") in hint_clean:
            return i
    return None


def draw_panels(
    fig, gs_row,
    words: List[str],
    word_embs: List[np.ndarray],
    g_values: List[float],
    xi: float,
    detected: List[int],
    info: dict,
    warmup: int,
    BLUE, ORANGE, RED, GREEN, GRAY, PURPLE,
):
    hint_idx = find_hint_idx(words, info["hint"])
    n_words  = len(words)
    # g_values는 워밍업 이후부터 존재 → 인덱스 오프셋
    g_offset = warmup  # g_values[0] = words[warmup]에 해당

    # ── 패널 A: NOUGAT 탐지 통계량 g_t ──
    axA = fig.add_subplot(gs_row[0])
    axA.set_facecolor("#FFFFFF")

    g_indices = list(range(g_offset, g_offset + len(g_values)))

    axA.plot(g_indices, g_values, "o-", color=BLUE, lw=2, ms=7,
             label="탐지 통계량 $g_t$", zorder=3)
    axA.axhline(0, color=GRAY, lw=1, linestyle="-", alpha=0.5)

    # 임계값 ±ξ (논문: |g_t + 1| > ξ → g_t > ξ-1 or g_t < -ξ-1)
    axA.axhline( xi - 1, color=RED, lw=1.5, linestyle="--",
                label=f"+임계값 (ξ={xi:.3f})", alpha=0.8)
    axA.axhline(-xi - 1, color=RED, lw=1.5, linestyle=":",
                label=f"-임계값", alpha=0.8)

    # 감지된 변화점
    for k, didx in enumerate(detected):
        word_idx = didx + g_offset
        label = f"감지 ({len(detected)}개)" if k == 0 else None
        axA.axvline(word_idx, color=RED, lw=2, linestyle="-.", alpha=0.85,
                    label=label)
        if word_idx < n_words:
            axA.text(word_idx + 0.05,
                     axA.get_ylim()[1] * 0.9 if g_values else 0,
                     words[word_idx], color=RED, fontsize=8, fontweight="bold")

    # 힌트 수직선
    if hint_idx is not None:
        axA.axvline(hint_idx, color=ORANGE, lw=1.8, linestyle=":",
                    label=f"힌트: [{info['hint']}]")

    axA.set_xticks(range(n_words))
    axA.set_xticklabels(
        [f"{w}\n({i})" for i, w in enumerate(words)],
        fontsize=7, rotation=40, ha="right",
    )
    axA.set_title(
        f"NOUGAT 탐지 통계량 $g_t$ | 감지: {len(detected)}개",
        fontsize=10, fontweight="bold",
    )
    axA.set_ylabel("$g_t = \\theta_t^T h_{test,t}$", fontsize=8)
    axA.legend(fontsize=7, loc="upper left")
    axA.grid(alpha=0.25, linestyle="--")
    axA.spines[["top", "right"]].set_visible(False)

    # ── 패널 B: CLS 임베딩 PCA 궤적 ──
    axB = fig.add_subplot(gs_row[1])
    axB.set_facecolor("#FFFFFF")

    if len(word_embs) >= 3:
        emb_2d = PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(np.array(word_embs))
        )
        cmap_p = matplotlib.cm.get_cmap("plasma")
        for i in range(n_words - 1):
            c = cmap_p(i / n_words)
            axB.annotate("", xy=emb_2d[i+1], xytext=emb_2d[i],
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
        sc = axB.scatter(emb_2d[:, 0], emb_2d[:, 1],
                         c=range(n_words), cmap="plasma", s=100,
                         zorder=4, edgecolors="white", lw=1)
        plt.colorbar(sc, ax=axB, fraction=0.046, pad=0.04, label="단어 순서")

        if hint_idx is not None:
            axB.scatter(*emb_2d[hint_idx], color=ORANGE, s=250, zorder=5,
                        edgecolors="black", lw=1.5, marker="*",
                        label=f"힌트: {info['hint']}")
        for k, didx in enumerate(detected):
            widx = didx + g_offset
            if widx < n_words:
                axB.scatter(*emb_2d[widx], color=RED, s=200, zorder=5,
                            edgecolors="black", lw=1.5, marker="D",
                            label=f"감지{k+1}: {words[widx]}" if k < 3 else None)

        axB.annotate("시작", emb_2d[0],  fontsize=8, color="#6A0572",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        axB.annotate("끝",   emb_2d[-1], fontsize=8, color="#B5179E",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        axB.legend(fontsize=7)

    axB.set_title("단어 임베딩 궤적 (PCA 2D)", fontsize=10, fontweight="bold")
    axB.set_xlabel("PC1", fontsize=8); axB.set_ylabel("PC2", fontsize=8)
    axB.grid(alpha=0.25, linestyle="--")
    axB.spines[["top","right"]].set_visible(False)

    # ── 패널 C: 단어 임베딩 유사도 히트맵 ──
    axC = fig.add_subplot(gs_row[2])
    axC.set_facecolor("#FFFFFF")

    from matplotlib.colors import LinearSegmentedColormap
    emb_arr = np.array(word_embs)
    n = len(emb_arr)
    sim = np.array([
        [float(1 - np.linalg.norm(emb_arr[i]-emb_arr[j]) /
               (np.linalg.norm(emb_arr[i]) * np.linalg.norm(emb_arr[j]) + 1e-9))
         for j in range(n)] for i in range(n)
    ])
    # 코사인 유사도로 재계산
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    emb_norm = emb_arr / (norms + 1e-9)
    sim = emb_norm @ emb_norm.T

    cmap_h = LinearSegmentedColormap.from_list(
        "bwhite", ["#EEF2FF", "#3A86FF", "#023E8A"])
    im = axC.imshow(sim, cmap=cmap_h, aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="코사인 유사도")
    axC.set_xticks(range(n))
    axC.set_yticks(range(n))
    axC.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    axC.set_yticklabels(words, fontsize=8)

    for didx in detected:
        widx = didx + g_offset
        if widx < n:
            axC.axvline(widx, color=RED, lw=1.5, linestyle="--", alpha=0.75)
            axC.axhline(widx, color=RED, lw=1.5, linestyle="--", alpha=0.75)

    axC.set_title("단어 임베딩 코사인 유사도 히트맵",
                  fontsize=10, fontweight="bold")
    axC.spines[["top","right"]].set_visible(False)


def make_figure(all_results: list, output_path: str, sigma_label: str = ""):
    setup_korean_font()

    BLUE   = "#3A86FF"
    ORANGE = "#FF6B35"
    RED    = "#EF233C"
    GREEN  = "#06D6A0"
    GRAY   = "#8D99AE"
    PURPLE = "#7B2D8B"

    n_sent = len(all_results)
    fig = plt.figure(figsize=(20, 6 * n_sent))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "NOUGAT + KoBERT: 문장 내 의미 분절 탐지\n"
        f"모델: {MODEL_NAME}  |  PCA={PCA_DIM}차원  σ={sigma_label}  "
        f"N_ref={N_REF}  N_test={N_TEST}  μ={MU}  ν={NU}  L_max={L_MAX}",
        fontsize=13, fontweight="bold", y=0.99, color="#2B2D42",
    )

    outer_gs = gridspec.GridSpec(
        n_sent, 1, figure=fig,
        hspace=0.55, left=0.05, right=0.97, top=0.95, bottom=0.04,
    )

    for row_idx, res in enumerate(all_results):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[row_idx], wspace=0.35,
        )
        fig.text(
            0.5,
            outer_gs[row_idx].get_position(fig).y1 + 0.005,
            f"[{row_idx+1}] \"{res['info']['text']}\"   ({res['info']['label']})",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#2B2D42",
        )
        draw_panels(
            fig=fig, gs_row=inner_gs,
            words=res["words"],
            word_embs=res["word_embs"],
            g_values=res["g_values"],
            xi=res["xi"],
            detected=res["detected"],
            info=res["info"],
            warmup=N_REF + N_TEST - 1,
            BLUE=BLUE, ORANGE=ORANGE, RED=RED,
            GREEN=GREEN, GRAY=GRAY, PURPLE=PURPLE,
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[저장 완료] {output_path}")
    plt.show()


# ══════════════════════════════════════════════════════════
# 5. 메인
# ══════════════════════════════════════════════════════════

def main():
    tokenizer, model, device = load_model(MODEL_NAME)
    all_results = []

    for i, info in enumerate(SENTENCES):
        sentence = info["text"]
        print(f"\n{'─'*60}")
        print(f"  문장 {i+1}: \"{sentence}\"")
        print(f"  예상 분절: {info['label']}")
        print(f"{'─'*60}")

        # 단어 임베딩 추출
        words, word_embs = get_word_embeddings(sentence, tokenizer, model, device)
        print(f"  단어 목록: {words}")

        # ── PCA 차원 축소 ──
        if PCA_DIM is not None:
            embs_input, (scaler, pca_obj, expl) = reduce_dim(word_embs, PCA_DIM)
            print(f"  PCA 축소: 768 → {len(embs_input[0])}차원  "
                  f"(설명 분산: {expl*100:.1f}%)")
        else:
            embs_input = word_embs
            print(f"  PCA 축소: 사용 안 함 (원본 {word_embs[0].shape[0]}차원)")

        # ── Median Heuristic σ 설정 ──
        if AUTO_SIGMA:
            sigma_used = median_heuristic(embs_input)
            print(f"  Median Heuristic σ: {sigma_used:.4f}  (기본값={SIGMA})")
        else:
            sigma_used = SIGMA
            print(f"  σ 고정값: {sigma_used}")

        # NOUGAT 온라인 처리
        detector = NOUGAT(
            n_ref=N_REF, n_test=N_TEST,
            mu=MU, sigma=sigma_used, nu=NU,
            l_max=L_MAX, lam=LAMBDA,
        )

        print(f"\n  NOUGAT 온라인 처리:")
        for t, (word, emb) in enumerate(zip(words, embs_input)):
            g_t = detector.update(emb)
            if g_t is not None:
                print(f"    t={t:02d} [{word:>12s}]  g_t={g_t:+.5f}  "
                      f"dict_size={len(detector.dictionary)}")
            else:
                print(f"    t={t:02d} [{word:>12s}]  (워밍업)")

        # 변화점 탐지
        detected, xi = detector.detect(FALSE_ALARM_RATE)
        warmup = N_REF + N_TEST - 1
        detected_words = [words[d + warmup] for d in detected if d + warmup < len(words)]

        print(f"\n  임계값 ξ            : {xi:.5f}")
        print(f"  감지된 분절 단어     : {detected_words}")
        print(f"  예상 분절 힌트       : [{info['hint']}]")
        print(f"  딕셔너리 최종 크기   : {len(detector.dictionary)}")

        all_results.append({
            "info":      info,
            "words":     words,
            "word_embs": word_embs,
            "g_values":  detector.g_values,
            "xi":        xi,
            "detected":  detected,
            "sigma_used": sigma_used,
        })

    print(f"\n{'─'*60}")
    print(f"  시각화 생성 → {OUTPUT_PATH}")
    print(f"{'─'*60}")
    sigma_str = f"median({all_results[0]['sigma_used']:.2f})" if AUTO_SIGMA else str(SIGMA)
    make_figure(all_results, OUTPUT_PATH, sigma_label=sigma_str)
    print("\n실험 완료!")


if __name__ == "__main__":
    main()