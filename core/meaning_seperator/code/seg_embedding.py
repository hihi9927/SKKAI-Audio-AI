"""
문장 내 의미 분절 탐지: KoBERT 토큰 증분 임베딩 + 변화점 시각화
================================================================
한 문장에서 토큰이 하나씩 증분될 때마다 [CLS] 임베딩을 뽑고,
슬라이딩 윈도우로 "이 토큰을 기점으로 임베딩이 얼마나 바뀌었나"를
측정하여 문장 내 의미 분절 위치를 추정합니다.

예시 문장:
  "날씨가 맑지만 내일은 비가 온다"
   → "맑지만" 근처에서 역접 분절이 감지되어야 함

설치:
    pip install transformers torch sentencepiece matplotlib scikit-learn numpy

실행:
    python kobert_intrasentence_changepoint.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple, Optional

# ══════════════════════════════════════════════════════════
# 0. 실험 설정
# ══════════════════════════════════════════════════════════

MODEL_NAME = "klue/bert-base"

# 실험할 문장들 — 각각 의도된 분절 위치(토큰 기준)가 다름
# (label: 사람이 예상하는 분절 토큰 키워드)
SENTENCES = [
    {
        "text": "날씨가 맑지만 내일은 비가 올 것이다.",
        "label": "역접 (맑지만)",
        "hint": "맑지만",   # 분절 힌트 (시각화용 표시)
    },
    {
        "text": "그는 열심히 공부했지만 시험에 떨어졌다.",
        "label": "역접 (공부했지만)",
        "hint": "공부했지만",
    },
    {
        "text": "봄이 지나고 여름이 되면 무더위가 시작된다.",
        "label": "시간 전환 (여름이)",
        "hint": "여름이",
    },
    {
        "text": "인공지능은 편리하지만 개인정보 침해 우려도 있다.",
        "label": "역접 (편리하지만)",
        "hint": "편리하지만",
    },
]

# 슬라이딩 윈도우 크기 (토큰 단위, 작을수록 민감)
WINDOW_SIZE = 2

OUTPUT_PATH = "kobert_intrasentence_result.png"


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
# 2. 토큰 증분 임베딩 추출
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def get_incremental_embeddings(
    sentence: str,
    tokenizer,
    model,
    device: str,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """
    토큰을 1개씩 증분하면서 매 시점의 임베딩을 반환.

    반환:
        tokens      : 서브워드 토큰 목록 (특수토큰 제외)
        cls_embs    : 각 시점의 [CLS] 임베딩  [n_steps, H]
        token_embs  : 각 시점에 마지막으로 추가된 토큰의 임베딩  [n_steps, H]
                      (= 해당 시점 last_hidden_state의 마지막 실제 토큰)
    """
    tokens = tokenizer.tokenize(sentence)
    cls_embs, token_embs = [], []

    for i in range(1, len(tokens) + 1):
        partial = tokens[:i]
        input_ids = torch.tensor([
            tokenizer.build_inputs_with_special_tokens(
                tokenizer.convert_tokens_to_ids(partial)
            )
        ]).to(device)
        attn_mask = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # [seq_len, H]

        cls_embs.append(hidden[0])       # [CLS]
        token_embs.append(hidden[-1])    # 마지막 실제 토큰 (= 이번에 추가된 토큰)

    return tokens, cls_embs, token_embs


# ══════════════════════════════════════════════════════════
# 3. 변화점 통계량
# ══════════════════════════════════════════════════════════

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def sliding_window_change_score(
    embs: List[np.ndarray],
    window: int,
) -> Tuple[List[int], List[float]]:
    """
    슬라이딩 윈도우로 참조 구간 vs 테스트 구간 평균 코사인 거리 계산.

    t번째 점수 = mean(embs[t-window:t]) vs mean(embs[t:t+window]) 의 코사인 거리
    → 이 값이 클수록 t 근처에서 의미 변화가 큰 것

    반환:
        indices : 점수가 계산된 토큰 인덱스 (0-based)
        scores  : 변화점 점수
    """
    n = len(embs)
    indices, scores = [], []
    for t in range(window, n - window):
        ref  = np.mean(embs[t - window : t], axis=0)
        test = np.mean(embs[t : t + window], axis=0)
        scores.append(cosine_distance(ref, test))
        indices.append(t)
    return indices, scores


def consecutive_change_score(
    embs: List[np.ndarray],
) -> Tuple[List[int], List[float]]:
    """
    연속된 두 스텝 간 코사인 거리 (토큰 하나 추가될 때마다의 변화량).
    슬라이딩 윈도우 점수와 비교용.
    """
    indices = list(range(1, len(embs)))
    scores  = [cosine_distance(embs[i-1], embs[i]) for i in indices]
    return indices, scores


def find_hint_token_idx(tokens: List[str], hint: str, tokenizer) -> Optional[int]:
    """힌트 키워드가 몇 번째 토큰 근처에 있는지 찾기 (시각화 표시용)"""
    hint_tokens = tokenizer.tokenize(hint)
    if not hint_tokens:
        return None
    first = hint_tokens[0]
    for i, t in enumerate(tokens):
        if first in t or t in first:
            return i
    return None


# ══════════════════════════════════════════════════════════
# 4. 시각화 (문장 1개 → 2행 × 3열 패널)
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


def draw_sentence_panels(
    fig, gs_row,           # GridSpec 행 슬라이스
    tokens: List[str],
    cls_embs: List[np.ndarray],
    token_embs: List[np.ndarray],
    sw_indices: List[int],
    sw_scores: List[float],
    consec_indices: List[int],
    consec_scores: List[float],
    sentence_info: dict,
    tokenizer,
    BLUE, ORANGE, RED, GREEN, GRAY, PURPLE,
):
    hint_idx = find_hint_token_idx(tokens, sentence_info["hint"], tokenizer)
    detected_idx = sw_indices[int(np.argmax(sw_scores))] if sw_scores else None
    n_steps = len(cls_embs)

    # ── 패널 A: 슬라이딩 윈도우 변화점 점수 ──
    axA = fig.add_subplot(gs_row[0])
    axA.set_facecolor("#FFFFFF")

    axA.plot(sw_indices, sw_scores, "o-", color=BLUE, lw=2, ms=7,
             label="슬라이딩 윈도우 점수", zorder=3)
    axA.plot(consec_indices, consec_scores, "s--", color=GREEN, lw=1.3, ms=5,
             alpha=0.8, label="연속 스텝 변화량", zorder=2)

    if detected_idx is not None:
        axA.axvline(detected_idx, color=RED, lw=2, linestyle="-.",
                    label=f"최대 변화 감지: [{tokens[detected_idx]}]")
    if hint_idx is not None:
        axA.axvline(hint_idx, color=ORANGE, lw=1.8, linestyle=":",
                    label=f"예상 분절: [{sentence_info['hint']}]")

    axA.set_xticks(range(n_steps))
    axA.set_xticklabels(
        [f"{t}\n({i})" for i, t in enumerate(tokens)],
        fontsize=6.5, rotation=40, ha="right",
    )
    axA.set_title("변화점 통계량 (토큰 단위)", fontsize=10, fontweight="bold")
    axA.set_ylabel("코사인 거리", fontsize=8)
    axA.legend(fontsize=7, loc="upper left")
    axA.grid(alpha=0.25, linestyle="--")
    axA.spines[["top", "right"]].set_visible(False)

    # ── 패널 B: CLS 임베딩 PCA 궤적 ──
    axB = fig.add_subplot(gs_row[1])
    axB.set_facecolor("#FFFFFF")

    if n_steps >= 3:
        cls_arr = np.array(cls_embs)
        cls_2d = PCA(n_components=2, random_state=42).fit_transform(
            StandardScaler().fit_transform(cls_arr)
        )
        cmap_p = matplotlib.cm.get_cmap("plasma")
        for i in range(n_steps - 1):
            c = cmap_p(i / n_steps)
            axB.annotate("", xy=cls_2d[i+1], xytext=cls_2d[i],
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.8))

        sc = axB.scatter(cls_2d[:, 0], cls_2d[:, 1],
                         c=range(n_steps), cmap="plasma", s=90,
                         zorder=4, edgecolors="white", lw=1)
        plt.colorbar(sc, ax=axB, fraction=0.046, pad=0.04, label="토큰 순서")

        # 힌트 위치 강조
        if hint_idx is not None and hint_idx < n_steps:
            axB.scatter(*cls_2d[hint_idx], color=ORANGE, s=200, zorder=5,
                        edgecolors="black", lw=1.5, marker="*",
                        label=f"힌트: {sentence_info['hint']}")
        if detected_idx is not None and detected_idx < n_steps:
            axB.scatter(*cls_2d[detected_idx], color=RED, s=200, zorder=5,
                        edgecolors="black", lw=1.5, marker="D",
                        label=f"감지: {tokens[detected_idx]}")

        axB.annotate("시작", cls_2d[0],  fontsize=8, color="#6A0572",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        axB.annotate("끝",   cls_2d[-1], fontsize=8, color="#B5179E",
                     fontweight="bold", xytext=(5,5), textcoords="offset points")
        axB.legend(fontsize=7)

    axB.set_title("CLS 임베딩 궤적 (PCA 2D)", fontsize=10, fontweight="bold")
    axB.set_xlabel("PC1", fontsize=8); axB.set_ylabel("PC2", fontsize=8)
    axB.grid(alpha=0.25, linestyle="--")
    axB.spines[["top","right"]].set_visible(False)

    # ── 패널 C: 토큰 임베딩 유사도 히트맵 ──
    axC = fig.add_subplot(gs_row[2])
    axC.set_facecolor("#FFFFFF")

    n = n_steps
    tok_arr = np.array(token_embs)
    sim = np.array([
        [1.0 - cosine_distance(tok_arr[i], tok_arr[j]) for j in range(n)]
        for i in range(n)
    ])
    cmap_h = LinearSegmentedColormap.from_list(
        "bwhite", ["#EEF2FF", "#3A86FF", "#023E8A"])
    im = axC.imshow(sim, cmap=cmap_h, aspect="auto", vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label="코사인 유사도")

    tick_step = max(1, n // 10)
    tick_pos = list(range(0, n, tick_step))
    axC.set_xticks(tick_pos)
    axC.set_yticks(tick_pos)
    axC.set_xticklabels([tokens[i] for i in tick_pos], rotation=45, ha="right", fontsize=7)
    axC.set_yticklabels([tokens[i] for i in tick_pos], fontsize=7)

    # 감지된 분절점 수직/수평선
    if detected_idx is not None:
        axC.axvline(detected_idx, color=RED, lw=1.5, linestyle="--", alpha=0.8)
        axC.axhline(detected_idx, color=RED, lw=1.5, linestyle="--", alpha=0.8)

    axC.set_title("토큰 임베딩 유사도 히트맵", fontsize=10, fontweight="bold")
    axC.spines[["top","right"]].set_visible(False)


def make_figure(
    all_results: list,
    tokenizer,
    output_path: str,
):
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
        f"문장 내 의미 분절 탐지 — KoBERT 토큰 증분 임베딩\n"
        f"모델: {MODEL_NAME}  |  윈도우 크기: {WINDOW_SIZE}",
        fontsize=14, fontweight="bold", y=0.99, color="#2B2D42",
    )

    outer_gs = gridspec.GridSpec(
        n_sent, 1, figure=fig,
        hspace=0.55, left=0.05, right=0.97, top=0.95, bottom=0.04,
    )

    for row_idx, result in enumerate(all_results):
        info    = result["info"]
        tokens  = result["tokens"]
        cls_embs   = result["cls_embs"]
        token_embs = result["token_embs"]
        sw_idx, sw_scores     = result["sw"]
        consec_idx, consec_scores = result["consec"]

        # 행 헤더
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[row_idx], wspace=0.35,
        )

        # 행 제목 (문장 텍스트 + 라벨)
        fig.text(
            0.5,
            outer_gs[row_idx].get_position(fig).y1 + 0.005,
            f"[{row_idx+1}] \"{info['text']}\"   ({info['label']})",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#2B2D42",
        )

        draw_sentence_panels(
            fig=fig,
            gs_row=inner_gs,
            tokens=tokens,
            cls_embs=cls_embs,
            token_embs=token_embs,
            sw_indices=sw_idx,
            sw_scores=sw_scores,
            consec_indices=consec_idx,
            consec_scores=consec_scores,
            sentence_info=info,
            tokenizer=tokenizer,
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

        # 토큰 증분 임베딩
        tokens, cls_embs, token_embs = get_incremental_embeddings(
            sentence, tokenizer, model, device
        )
        print(f"  토큰 목록: {tokens}")

        # 변화점 점수 계산
        sw_idx, sw_scores         = sliding_window_change_score(cls_embs, WINDOW_SIZE)
        consec_idx, consec_scores = consecutive_change_score(cls_embs)

        # 결과 출력
        if sw_scores:
            detected_pos = int(np.argmax(sw_scores))
            detected_tok = tokens[sw_idx[detected_pos]]
            print(f"\n  슬라이딩 윈도우 변화점 점수:")
            for idx, score in zip(sw_idx, sw_scores):
                marker = " ◀ 최대" if score == max(sw_scores) else ""
                print(f"    [{tokens[idx]:>12s}] (tok {idx:02d}): {score:.5f}{marker}")
            print(f"\n  감지된 분절 토큰 : [{detected_tok}]")
            print(f"  예상 분절 힌트   : [{info['hint']}]")

        all_results.append({
            "info": info,
            "tokens": tokens,
            "cls_embs": cls_embs,
            "token_embs": token_embs,
            "sw": (sw_idx, sw_scores),
            "consec": (consec_idx, consec_scores),
        })

    # 시각화
    print(f"\n{'─'*60}")
    print(f"  시각화 생성 → {OUTPUT_PATH}")
    print(f"{'─'*60}")
    make_figure(all_results, tokenizer, OUTPUT_PATH)
    print("\n실험 완료!")


if __name__ == "__main__":
    main()