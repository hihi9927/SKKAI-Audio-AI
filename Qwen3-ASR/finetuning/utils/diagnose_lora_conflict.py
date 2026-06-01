"""
두 LoRA 어댑터 간 충돌 진단 스크립트.

합병(merge) 전에 실행하여 레이어별 코사인 유사도와 부호 충돌 비율을 리포트.
GPU 불필요 (CPU only).

Usage:
    python diagnose_lora_conflict.py \\
        --adapter_a  ../finetuning-out-en/checkpoint-200_final \\
        --adapter_b  ../finetuning-out-ko/checkpoint-300_final

    # JSON으로 결과 저장
    python diagnose_lora_conflict.py \\
        --adapter_a  ../finetuning-out-en/checkpoint-200_final \\
        --adapter_b  ../finetuning-out-ko/checkpoint-300_final \\
        --output     conflict_report.json

    # 충돌 레이어만 출력 (cos_sim 임계값 지정)
    python diagnose_lora_conflict.py \\
        --adapter_a  ../finetuning-out-en/checkpoint-200_final \\
        --adapter_b  ../finetuning-out-ko/checkpoint-300_final \\
        --threshold  0.0
"""
import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class LayerDiagnostic:
    layer_path: str
    cos_sim: float
    sign_conflict_ratio: float  # 부호가 다른 원소 비율 (0~1)
    norm_a: float
    norm_b: float
    norm_ratio: float           # norm_b / norm_a (1에 가까울수록 균형)


@dataclass
class AdapterDiagnosticReport:
    adapter_a: str
    adapter_b: str
    lora_r_a: int
    lora_r_b: int
    lora_alpha_a: float
    lora_alpha_b: float
    total_layers: int
    conflict_layers: int        # cos_sim < threshold인 레이어 수
    mean_cos_sim: float
    min_cos_sim: float
    max_cos_sim: float
    mean_sign_conflict_ratio: float
    recommendation: str
    layers: list[LayerDiagnostic]


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

STRIP_PREFIX = "base_model.model."

# vLLM 변환 시 thinker.model. → language_model.model. 으로 바뀜
# 진단 시 두 형식을 동일한 정규화 키로 매핑
_CANONICAL_PREFIXES = [
    "thinker.model.",
    "language_model.model.",
]


def _detect_format(keys: list[str]) -> str:
    """어댑터 키에서 포맷 감지. 'peft' or 'vllm' or 'unknown' 반환."""
    for k in keys:
        stripped = k[len(STRIP_PREFIX):] if k.startswith(STRIP_PREFIX) else k
        if stripped.startswith("thinker.model."):
            return "peft"
        if stripped.startswith("language_model.model."):
            return "vllm"
    return "unknown"


def _to_canonical(key: str) -> str:
    """thinker.model. / language_model.model. 을 모두 제거해 정규화."""
    for prefix in _CANONICAL_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_adapter(adapter_dir: str) -> tuple[dict[str, torch.Tensor], dict]:
    """adapter_model.safetensors와 adapter_config.json 로드."""
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.exists(st_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter_dir}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_dir}")

    tensors = load_file(st_path)
    with open(cfg_path) as f:
        cfg = json.load(f)
    return tensors, cfg


def strip_and_filter(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    """base_model.model. prefix 제거, audio_tower 제외, 포맷 감지 후 canonical 키로 정규화.

    반환: (정규화된 텐서 dict, 포맷 문자열 'peft'|'vllm'|'unknown')
    """
    # base_model.model. 제거
    stripped: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        new_k = k[len(STRIP_PREFIX):] if k.startswith(STRIP_PREFIX) else k
        if "audio_tower." not in new_k:
            stripped[new_k] = v

    fmt = _detect_format(list(stripped.keys()))

    # canonical 정규화 (thinker.model. / language_model.model. 제거)
    result = {_to_canonical(k): v for k, v in stripped.items()}
    return result, fmt


def compute_delta(tensors: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    """각 레이어의 ΔW = scale * B @ A 계산."""
    # lora_A 키에서 레이어 경로 수집
    layer_paths = {
        k.replace(".lora_A.weight", "")
        for k in tensors if k.endswith(".lora_A.weight")
    }

    deltas = {}
    for path in layer_paths:
        key_a = path + ".lora_A.weight"
        key_b = path + ".lora_B.weight"
        if key_a not in tensors or key_b not in tensors:
            continue
        A = tensors[key_a].float()  # (r, in)
        B = tensors[key_b].float()  # (out, r)
        deltas[path] = scale * (B @ A)  # (out, in)
    return deltas


def cosine_similarity_flat(t1: torch.Tensor, t2: torch.Tensor) -> float:
    v1 = t1.reshape(-1)
    v2 = t2.reshape(-1)
    cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return float(cos)


def sign_conflict_ratio(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """부호가 다른 원소 비율. 0에 가까울수록 충돌 없음."""
    signs_differ = (t1.sign() != t2.sign()) & (t1 != 0) & (t2 != 0)
    return float(signs_differ.float().mean().item())


def make_recommendation(mean_cos: float, conflict_ratio: float, conflict_layers: int, total: int) -> str:
    conflict_pct = conflict_layers / total * 100 if total > 0 else 0
    if mean_cos > 0.7:
        return (
            f"높은 정렬 (mean cos_sim={mean_cos:.3f}). "
            "두 어댑터가 유사한 방향 학습. 가중 평균 합병 권장."
        )
    elif mean_cos > 0.3:
        return (
            f"중간 정렬 (mean cos_sim={mean_cos:.3f}). "
            "Task Arithmetic 합병 가능, 스케일 조정 후 성능 검증 권장."
        )
    elif mean_cos > 0.1:
        return (
            f"약한 정렬 (mean cos_sim={mean_cos:.3f}). "
            "어느 정도 독립적 학습. Task Arithmetic (scale=1.0 직접 합산) 권장. "
            "충돌 레이어: {conflict_pct:.0f}%."
        )
    elif mean_cos >= 0.0:
        return (
            f"직교 구조 (mean cos_sim={mean_cos:.3f}). "
            "두 어댑터가 완전히 독립적 방향 학습 — 합병에 유리한 조건. "
            "Task Arithmetic (직접 합산, scale=1.0) 사용. "
            "norm 불균형 레이어는 per-layer 스케일 보정 고려."
        )
    else:
        return (
            f"충돌 (mean cos_sim={mean_cos:.3f}, 충돌 레이어 {conflict_pct:.0f}%). "
            "단순 합산 시 두 어댑터 모두 성능 저하 예상. "
            "TIES-Merging 또는 MoE 방식(라우터 기반) 고려."
        )


# ── 메인 진단 ─────────────────────────────────────────────────────────────────

def diagnose(adapter_a: str, adapter_b: str, threshold: float) -> AdapterDiagnosticReport:
    adapter_a = os.path.abspath(adapter_a)
    adapter_b = os.path.abspath(adapter_b)

    print(f"[1/4] 어댑터 A 로드: {adapter_a}")
    tensors_a, cfg_a = load_adapter(adapter_a)
    tensors_a, fmt_a = strip_and_filter(tensors_a)
    scale_a = cfg_a["lora_alpha"] / cfg_a["r"]
    print(f"  포맷={fmt_a}, r={cfg_a['r']}, alpha={cfg_a['lora_alpha']}, scale={scale_a:.4f}, tensors={len(tensors_a)}")

    print(f"[2/4] 어댑터 B 로드: {adapter_b}")
    tensors_b, cfg_b = load_adapter(adapter_b)
    tensors_b, fmt_b = strip_and_filter(tensors_b)
    scale_b = cfg_b["lora_alpha"] / cfg_b["r"]
    print(f"  포맷={fmt_b}, r={cfg_b['r']}, alpha={cfg_b['lora_alpha']}, scale={scale_b:.4f}, tensors={len(tensors_b)}")

    if fmt_a != fmt_b:
        print(f"  [!] 포맷 불일치 ({fmt_a} vs {fmt_b}) — canonical 키로 정규화하여 비교합니다")

    print("[3/4] ΔW 계산 중...")
    deltas_a = compute_delta(tensors_a, scale_a)
    deltas_b = compute_delta(tensors_b, scale_b)

    common_paths = sorted(set(deltas_a.keys()) & set(deltas_b.keys()))
    print(f"  공통 레이어: {len(common_paths)} (A 전용: {len(deltas_a)-len(common_paths)}, B 전용: {len(deltas_b)-len(common_paths)})")

    print("[4/4] 레이어별 진단 중...")
    layer_results: list[LayerDiagnostic] = []
    for path in common_paths:
        da = deltas_a[path]
        db = deltas_b[path]

        cos = cosine_similarity_flat(da, db)
        scr = sign_conflict_ratio(da, db)
        norm_a = float(da.norm().item())
        norm_b = float(db.norm().item())
        norm_ratio = norm_b / norm_a if norm_a > 1e-8 else float("inf")

        layer_results.append(LayerDiagnostic(
            layer_path=path,
            cos_sim=round(cos, 6),
            sign_conflict_ratio=round(scr, 6),
            norm_a=round(norm_a, 6),
            norm_b=round(norm_b, 6),
            norm_ratio=round(norm_ratio, 6),
        ))

    cos_sims = [r.cos_sim for r in layer_results]
    conflict_layers = sum(1 for c in cos_sims if c < threshold)
    mean_cos = float(sum(cos_sims) / len(cos_sims)) if cos_sims else 0.0
    mean_scr = float(sum(r.sign_conflict_ratio for r in layer_results) / len(layer_results)) if layer_results else 0.0

    recommendation = make_recommendation(mean_cos, mean_scr, conflict_layers, len(layer_results))

    return AdapterDiagnosticReport(
        adapter_a=adapter_a,
        adapter_b=adapter_b,
        lora_r_a=cfg_a["r"],
        lora_r_b=cfg_b["r"],
        lora_alpha_a=cfg_a["lora_alpha"],
        lora_alpha_b=cfg_b["lora_alpha"],
        total_layers=len(layer_results),
        conflict_layers=conflict_layers,
        mean_cos_sim=round(mean_cos, 6),
        min_cos_sim=round(min(cos_sims), 6) if cos_sims else 0.0,
        max_cos_sim=round(max(cos_sims), 6) if cos_sims else 0.0,
        mean_sign_conflict_ratio=round(mean_scr, 6),
        recommendation=recommendation,
        layers=layer_results,
    )


def print_report(report: AdapterDiagnosticReport, threshold: float, top_n: int = 10) -> None:
    print()
    print("=" * 60)
    print("LoRA 어댑터 충돌 진단 리포트")
    print("=" * 60)
    print(f"  어댑터 A : {os.path.basename(report.adapter_a)}  (r={report.lora_r_a}, α={report.lora_alpha_a})")
    print(f"  어댑터 B : {os.path.basename(report.adapter_b)}  (r={report.lora_r_b}, α={report.lora_alpha_b})")
    print()
    print(f"  분석 레이어  : {report.total_layers}")
    print(f"  충돌 레이어  : {report.conflict_layers}  (cos_sim < {threshold})")
    print(f"  평균 cos_sim : {report.mean_cos_sim:+.4f}  (min={report.min_cos_sim:+.4f}, max={report.max_cos_sim:+.4f})")
    print(f"  부호 충돌률  : {report.mean_sign_conflict_ratio:.1%}")
    print()
    print(f"  [권고] {report.recommendation}")
    print()

    # cos_sim 낮은 레이어 top N
    worst = sorted(report.layers, key=lambda x: x.cos_sim)[:top_n]
    if worst:
        print(f"  ── 충돌 심한 레이어 Top {top_n} ──")
        print(f"  {'cos_sim':>8}  {'sign_cflt':>9}  {'norm_ratio':>10}  레이어")
        for r in worst:
            flag = " ★" if r.cos_sim < 0 else ""
            print(f"  {r.cos_sim:+8.4f}  {r.sign_conflict_ratio:9.1%}  {r.norm_ratio:10.3f}  {r.layer_path}{flag}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LoRA 어댑터 충돌 진단")
    p.add_argument("--adapter_a", required=True, help="첫 번째 어댑터 디렉토리")
    p.add_argument("--adapter_b", required=True, help="두 번째 어댑터 디렉토리")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="충돌로 분류할 cos_sim 임계값 (기본: 0.0)")
    p.add_argument("--top_n", type=int, default=10,
                   help="출력할 최악 레이어 수 (기본: 10)")
    p.add_argument("--output", default=None,
                   help="JSON 리포트 저장 경로 (생략 시 저장 안 함)")
    args = p.parse_args()

    report = diagnose(args.adapter_a, args.adapter_b, args.threshold)
    print_report(report, args.threshold, args.top_n)

    if args.output:
        out_path = os.path.abspath(args.output)
        data = asdict(report)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 저장: {out_path}")


if __name__ == "__main__":
    main()
