#!/usr/bin/env python3
"""
기존 run_01~run_10의 구 포맷(qwen3_test_other_fcl*.json)을
새 포맷(metric.json / meta.json / clients/ / logs/)으로 변환하고
WER을 계산해서 채워 넣는 마이그레이션 스크립트.

실행 위치: /home/ubuntu/STiTy
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/migrate_runs.py"
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean

SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent           # .../multi_speaker_test/
MODEL_TAG   = "finetuned(1.0.1)"
NUM_FILES   = 548


# ── WER 계산 ──────────────────────────────────────────────────────────────────

def compute_wer(rows: list[dict]):
    try:
        import jiwer
    except ImportError:
        return None

    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    pairs = [
        (normalize(r["reference"]), normalize(r["hypothesis"]))
        for r in rows
        if r.get("reference") and r.get("hypothesis")
    ]
    pairs = [(ref, hyp) for ref, hyp in pairs if ref.strip() and hyp.strip()]
    if not pairs:
        return None
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps))


# ── latency 통계 계산 ─────────────────────────────────────────────────────────

def build_metric(all_raw: list[dict], client_raws: list[list[dict]]) -> dict:
    fcl_vals, asr_vals, tl_vals, tok_vals, lat_vals = [], [], [], [], []
    commit_counts: dict[str, int] = {"vad": 0, "seg": 0, "dot": 0, "finish": 0}

    for r in all_raw:
        if r.get("first_token_latency") is not None:
            lat_vals.append(r["first_token_latency"])
        for seg in r.get("segment_metrics", []):
            fcl_key = "fsl_sec" if "fsl_sec" in seg else "server_fcl_sec"
            asr_key = "decode_sec" if "decode_sec" in seg else "asr_inference_sec"
            tl_key  = "trans_sec" if "trans_sec" in seg else "translation_latency_sec"
            if seg.get(fcl_key) is not None: fcl_vals.append(seg[fcl_key])
            if seg.get(asr_key) is not None: asr_vals.append(seg[asr_key])
            if seg.get(tl_key)  is not None: tl_vals.append(seg[tl_key])
            if seg.get("output_token_count") is not None:
                tok_vals.append(seg["output_token_count"])
            reason = str(seg.get("commit_reason", "")).lower()
            for key in commit_counts:
                if reason.startswith(key):
                    commit_counts[key] += 1
                    break

    def avg(vals): return mean(vals) if vals else None

    total = sum(commit_counts.values())
    per_client = []
    for i, raw in enumerate(client_raws):
        c_lat = [r["first_token_latency"] for r in raw if r.get("first_token_latency") is not None]
        per_client.append({
            "client": i,
            "num_files": len(raw),
            "wer": compute_wer(raw),
            "avg_first_token_latency": avg(c_lat),
        })

    return {
        "overall": {
            "num_files": len(all_raw),
            "wer": compute_wer(all_raw),
            "avg_first_token_latency": avg(lat_vals),
            "avg_fsl_sec": avg(fcl_vals),
            "avg_asr_inference_sec": avg(asr_vals),
            "avg_translation_latency_sec": avg(tl_vals),
            "avg_output_tokens_per_commit": avg(tok_vals),
            "commit_stats": {
                "counts": commit_counts,
                "total": total,
                "ratios": {k: v / total if total else 0 for k, v in commit_counts.items()},
            },
        },
        "per_client": per_client,
        "raw_results": all_raw,
    }


# ── 단일 run 마이그레이션 ──────────────────────────────────────────────────────

def migrate_run(run_dir: Path) -> None:
    n = int(run_dir.name.split("_")[1])  # run_02 → 2
    print(f"\n{'=' * 60}")
    print(f"  {run_dir.name}: {n}명 마이그레이션 중...")
    print(f"{'=' * 60}")

    # 구 포맷 파일 수집
    client_json_paths = sorted(run_dir.glob("qwen3_test_other_fcl_c*.json"))
    if not client_json_paths:
        print(f"  [SKIP] qwen3_test_other_fcl_c*.json 없음")
        return

    # clients/ 디렉토리 생성
    clients_dir = run_dir / "clients"
    clients_dir.mkdir(exist_ok=True)

    # logs/ 디렉토리 생성 및 기존 로그 이동
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    for log_src_name, log_dst_name in [
        ("server.log",        "server.log"),
        ("server_stdout.log", "server_stdout.log"),
    ]:
        src = run_dir / log_src_name
        dst = logs_dir / log_dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    for i in range(n):
        src = run_dir / f"test_stdout_c{i}.log"
        dst = logs_dir / f"client_c{i}.log"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # 각 클라이언트 raw_results 읽기
    all_raw: list[dict] = []
    client_raws: list[list[dict]] = []

    for i, path in enumerate(client_json_paths):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("policy_3", {}).get("raw_results", [])
        client_raws.append(raw)
        all_raw.extend(raw)
        print(f"  c{i}: {len(raw)}개 파일 읽기 완료")

        # clients/c{i}.json 저장
        c_wer = compute_wer(raw)
        c_lat = [r["first_token_latency"] for r in raw if r.get("first_token_latency") is not None]
        client_out = clients_dir / f"c{i}.json"
        with open(client_out, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "client": i,
                    "num_files": len(raw),
                    "wer": c_wer,
                    "avg_first_token_latency": mean(c_lat) if c_lat else None,
                },
                "raw_results": raw,
            }, f, indent=2, ensure_ascii=False)
        wer_str = f"{c_wer * 100:.2f}%" if c_wer is not None else "N/A"
        print(f"  c{i}: WER={wer_str}  → {client_out.relative_to(RESULTS_DIR)}")

    # metric.json 저장
    metric = build_metric(all_raw, client_raws)
    with open(run_dir / "metric.json", "w", encoding="utf-8") as f:
        json.dump(metric, f, indent=2, ensure_ascii=False)
    ov = metric["overall"]
    wer_str = f"{ov['wer'] * 100:.2f}%" if ov["wer"] is not None else "N/A"
    fsl_str = f"{ov['avg_fsl_sec']:.3f}s" if ov["avg_fsl_sec"] else "N/A"
    print(f"  metric.json: WER={wer_str}  FSL={fsl_str}  항목={ov['num_files']}")

    # meta.json 저장 (mtime으로 타임스탬프 추정)
    mtime = os.path.getmtime(client_json_paths[0])
    ts = datetime.fromtimestamp(mtime).isoformat()
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":    ts,
            "n_clients":    n,
            "mode":         "full",
            "model_tag":    MODEL_TAG,
            "server_model": "unknown (migrated from old format)",
            "num_files":    NUM_FILES,
            "migrated":     True,
        }, f, indent=2, ensure_ascii=False)
    print(f"  meta.json: 저장 완료 (ts={ts})")


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def main() -> None:
    run_dirs = sorted(d for d in RESULTS_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("run_"))
    print(f"마이그레이션 대상: {[d.name for d in run_dirs]}")

    for run_dir in run_dirs:
        # 이미 metric.json이 있으면 스킵
        if (run_dir / "metric.json").exists():
            print(f"\n[SKIP] {run_dir.name}: metric.json 이미 존재")
            continue
        migrate_run(run_dir)

    print("\n\n── 최종 요약 ──")
    for run_dir in run_dirs:
        metric_path = run_dir / "metric.json"
        if not metric_path.exists():
            print(f"  {run_dir.name}: metric.json 없음")
            continue
        with open(metric_path, encoding="utf-8") as f:
            data = json.load(f)
        ov = data.get("overall", {})
        wer = ov.get("wer")
        fsl = ov.get("avg_fsl_sec")
        n_files = ov.get("num_files", 0)
        n = int(run_dir.name.split("_")[1])
        wer_str = f"{wer * 100:.2f}%" if wer is not None else "N/A"
        fsl_str = f"{fsl:.3f}s" if fsl else "N/A"
        print(f"  {n:2d}명 ({run_dir.name}): WER={wer_str}  FSL={fsl_str}  항목={n_files}")


if __name__ == "__main__":
    main()
