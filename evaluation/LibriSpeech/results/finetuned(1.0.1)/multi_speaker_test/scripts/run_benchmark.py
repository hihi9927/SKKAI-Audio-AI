#!/usr/bin/env python3
"""
multi_speaker_test 벤치마크 러너

N명이 동시 접속했을 때 WER / FSL이 어떻게 변하는지 측정합니다.

모드:
  --mode full   각 클라이언트가 전체 파일을 처리 (기본)
                → 동시 접속에 따른 latency 저하 측정
  --mode split  파일을 N개 클라이언트에 균등 분배
                → throughput / 부하 분산 테스트

실행 위치: /home/ubuntu/STiTy
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py"
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" --mode split --start-n 3 --end-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

# ── 경로 설정 ──────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent            # .../multi_speaker_test/scripts/
LIBRISPEECH   = SCRIPT_DIR.parent.parent.parent.parent    # .../evaluation/LibriSpeech/
PROJECT_ROOT  = LIBRISPEECH.parent.parent                 # /home/ubuntu/STiTy/
SERVER_SCRIPT = LIBRISPEECH / "servers" / "streaming_websocket_server_fcl.py"
TEST_SCRIPT   = LIBRISPEECH / "servers" / "test_qwen3_librispeech.py"
TEST_DIR      = LIBRISPEECH / "LibriSpeech" / "test-other"
RESULTS_DIR   = SCRIPT_DIR.parent                         # .../multi_speaker_test/

PYTHON    = "/home/ubuntu/miniconda3/envs/qwen3-asr/bin/python"
CONDA_BIN = "/home/ubuntu/miniconda3/envs/qwen3-asr/bin"
PORT      = 8765
NUM_FILES = 548

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_server(port: int = PORT, timeout: int = 300) -> bool:
    log(f"서버 포트 {port} 대기 중 (최대 {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                log(f"서버 준비 완료 ({time.time() - start:.0f}s 경과)")
                return True
        except OSError:
            time.sleep(3)
    log("오류: 서버 시작 타임아웃")
    return False


def kill_existing_server(port: int = PORT) -> None:
    """포트 점유 프로세스 및 잔존 VLLM EngineCore 종료."""
    killed = False
    try:
        result = subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        if result.returncode == 0:
            log(f"기존 서버 프로세스 종료 (포트 {port})")
            killed = True
    except FileNotFoundError:
        try:
            r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            for pid in r.stdout.strip().split():
                subprocess.run(["kill", "-9", pid], capture_output=True)
            if r.stdout.strip():
                log(f"기존 서버 프로세스 종료 (포트 {port})")
                killed = True
        except Exception:
            pass

    for pattern in ["streaming_websocket_server_fcl", "VLLM::EngineCore"]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)

    if killed:
        time.sleep(3)

    for _ in range(10):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        free_mb = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0
        if free_mb > 10000:
            log(f"VRAM 해제 확인: {free_mb} MiB 사용 가능")
            break
        log(f"VRAM 해제 대기 중... ({free_mb} MiB 사용 가능)")
        time.sleep(3)


def get_sorted_file_ids(num_files: int) -> list[str]:
    if not TEST_DIR.exists():
        sys.exit(f"오류: test-other 디렉토리 없음: {TEST_DIR}")
    all_ids: list[str] = []
    for trans_file in sorted(TEST_DIR.rglob("*.trans.txt")):
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if parts and parts[0]:
                    all_ids.append(parts[0])
    all_ids.sort()
    result = all_ids[:num_files]
    log(f"파일 ID {len(result)}개 로드 (전체 {len(all_ids)}개 중)")
    return result


def split_chunks(items: list, n: int) -> list[list]:
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def save_file_list_json(file_ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"raw_results": [{"file_id": fid} for fid in file_ids]}, f)


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


# ── 결과 병합 & metric 생성 ───────────────────────────────────────────────────

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


# ── 개별 테스트 실행 ───────────────────────────────────────────────────────────

def run_test(n_clients: int, file_ids: list[str], args: argparse.Namespace) -> None:
    run_name   = f"run_{n_clients:02d}"
    result_dir = RESULTS_DIR / args.model / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "clients").mkdir(exist_ok=True)
    (result_dir / "logs").mkdir(exist_ok=True)

    log("=" * 60)
    log(f" {run_name}: 동시 {n_clients}명 / mode={args.mode} / model={args.model}")
    log("=" * 60)

    kill_existing_server()

    server_env = os.environ.copy()
    server_env["PATH"] = CONDA_BIN + ":" + server_env.get("PATH", "")

    server_proc = subprocess.Popen(
        [PYTHON, str(SERVER_SCRIPT),
         "--model", args.server_model,
         "--enforce-eager",
         "--no-idle-shutdown",
         "--log-file", str(result_dir / "logs" / "server.log")],
        stdout=open(result_dir / "logs" / "server_stdout.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=server_env,
    )
    log(f"서버 시작 (PID={server_proc.pid})")

    if not wait_for_server():
        server_proc.kill()
        return

    # 각 클라이언트에 줄 파일 목록 결정
    if args.mode == "full":
        client_file_lists = [file_ids] * n_clients
    else:
        client_file_lists = split_chunks(file_ids, n_clients)

    # test_qwen3_librispeech.py는 results_root/{model}/{scope}/{tag}/metric.json에 저장
    # 임시 scope "_ms_tmp"를 사용해 오염 없이 저장 후 정리
    tmp_scope   = "_ms_tmp"
    tmp_run_dir = LIBRISPEECH / "results" / args.model / tmp_scope / run_name

    client_procs:        list[subprocess.Popen] = []
    client_metric_paths: list[Path]             = []

    for i, ids in enumerate(client_file_lists):
        chunk_path  = result_dir / "clients" / f"c{i}_files.json"
        save_file_list_json(ids, chunk_path)

        metric_path = LIBRISPEECH / "results" / args.model / tmp_scope / run_name / f"c{i}" / "metric.json"
        client_metric_paths.append(metric_path)

        proc = subprocess.Popen(
            [PYTHON, str(TEST_SCRIPT),
             "--test-dir",              str(TEST_DIR),
             "--host",                  "localhost",
             "--port",                  str(PORT),
             "--model",                 args.model,
             "--scope",                 tmp_scope,
             "--tag",                   f"{run_name}/c{i}",
             "--common-files",          str(chunk_path),
             "--fresh-start",
             "--no-calculate-wer",
             "--skip-protocol-smoke",
             "--trailing-silence-ms",   "5500"],
            stdout=open(result_dir / "logs" / f"client_c{i}.log", "w"),
            stderr=subprocess.STDOUT,
            cwd=str(LIBRISPEECH),
        )
        client_procs.append(proc)
        log(f"  클라이언트 {i + 1}/{n_clients} 시작 (PID={proc.pid}, {len(ids)}개 파일)")

    t0 = time.time()
    for i, proc in enumerate(client_procs):
        proc.wait()
        log(f"  클라이언트 {i + 1} 완료 (rc={proc.returncode}, {time.time() - t0:.0f}s 경과)")
    log(f"전체 소요: {time.time() - t0:.0f}s")

    # ── 결과 읽기 & 클라이언트별 JSON 저장 ─────────────────────────────────────
    all_raw: list[dict]       = []
    client_raws: list[list]   = []

    for i, path in enumerate(client_metric_paths):
        if not path.exists():
            log(f"  경고: {path} 없음, 스킵")
            client_raws.append([])
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("raw_results", [])
        client_raws.append(raw)
        all_raw.extend(raw)
        log(f"  c{i}: {len(raw)}개 파일")

        c_wer = compute_wer(raw)
        c_lat = [r["first_token_latency"] for r in raw if r.get("first_token_latency") is not None]
        client_out = result_dir / "clients" / f"c{i}.json"
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

    # ── metric.json 저장 ────────────────────────────────────────────────────────
    if all_raw:
        metric = build_metric(all_raw, client_raws)
        with open(result_dir / "metric.json", "w", encoding="utf-8") as f:
            json.dump(metric, f, indent=2, ensure_ascii=False)
        ov      = metric["overall"]
        wer_str = f"{ov['wer'] * 100:.2f}%" if ov["wer"] is not None else "N/A"
        fsl_str = f"{ov['avg_fsl_sec']:.3f}s" if ov["avg_fsl_sec"] else "N/A"
        log(f"  병합 완료: {len(all_raw)}개 항목  WER={wer_str}  FSL={fsl_str}")
    else:
        log("  병합할 결과 없음")

    # ── meta.json 저장 ──────────────────────────────────────────────────────────
    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "n_clients":    n_clients,
            "mode":         args.mode,
            "model_tag":    args.model,
            "server_model": args.server_model,
            "num_files":    len(file_ids),
            "test_dir":     str(TEST_DIR),
        }, f, indent=2, ensure_ascii=False)

    # ── 임시 디렉토리 정리 ──────────────────────────────────────────────────────
    if tmp_run_dir.exists():
        shutil.rmtree(tmp_run_dir)
    tmp_scope_dir = LIBRISPEECH / "results" / args.model / tmp_scope
    if tmp_scope_dir.exists() and not any(tmp_scope_dir.iterdir()):
        tmp_scope_dir.rmdir()

    server_proc.terminate()
    try:
        server_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    log("서버 종료 완료")
    time.sleep(5)


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="multi_speaker_test 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 (실행 위치: /home/ubuntu/STiTy):
  # 기본 (full 모드, 1~10명)
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py"

  # split 모드, 3~5명만
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" --mode split --start-n 3 --end-n 5

  # baseline 모델
  python "evaluation/LibriSpeech/results/finetuned(1.0.1)/multi_speaker_test/scripts/run_benchmark.py" --model "baseline(1.0.0)" --server-model Qwen/Qwen3-ASR-1.7B
        """,
    )
    parser.add_argument("--mode", choices=["full", "split"], default="full",
                        help="full: 각 클라이언트가 전체 파일 처리 / split: 균등 분배")
    parser.add_argument("--start-n", type=int, default=1, metavar="N",
                        help="시작 동시 접속자 수 (기본: 1)")
    parser.add_argument("--end-n", type=int, default=10, metavar="N",
                        help="종료 동시 접속자 수 (기본: 10)")
    parser.add_argument("--model", type=str, default="finetuned(1.0.1)",
                        help="결과 폴더 대분류 (기본: finetuned(1.0.1))")
    parser.add_argument("--server-model", type=str,
                        default="/home/ubuntu/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged",
                        help="ASR 서버에 로드할 모델 경로")
    parser.add_argument("--num-files", type=int, default=NUM_FILES, metavar="N",
                        help=f"사용할 파일 수 (기본: {NUM_FILES})")
    args = parser.parse_args()

    log(f"파일 ID {args.num_files}개 로드 중...")
    file_ids = get_sorted_file_ids(args.num_files)

    log(f"벤치마크 시작: {args.start_n}~{args.end_n}명 / mode={args.mode} / model={args.model}")
    total_start = time.time()

    for n in range(args.start_n, args.end_n + 1):
        run_test(n, file_ids, args)

    log("=" * 60)
    log(f"전체 완료 ({(time.time() - total_start) / 3600:.1f}h)  결과: {RESULTS_DIR / args.model}")

    log("\n── 요약 ──")
    for n in range(args.start_n, args.end_n + 1):
        metric_path = RESULTS_DIR / args.model / f"run_{n:02d}" / "metric.json"
        if not metric_path.exists():
            continue
        with open(metric_path, encoding="utf-8") as f:
            data = json.load(f)
        ov      = data.get("overall", {})
        wer     = ov.get("wer")
        fsl     = ov.get("avg_fsl_sec")
        n_files = ov.get("num_files", 0)
        wer_str = f"{wer * 100:.2f}%" if wer is not None else "N/A"
        fsl_str = f"{fsl:.3f}s" if fsl else "N/A"
        log(f"  {n:2d}명 (run_{n:02d}): WER={wer_str}  FSL={fsl_str}  항목={n_files}")


if __name__ == "__main__":
    main()
