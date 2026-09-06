#!/usr/bin/env python3
"""
multi_speaker_test 전체 재실행 (올바른 설계)

각 클라이언트가 동일한 548개 파일을 전부 처리.
동시 접속 수에 따른 WER / 딜레이 비교를 위한 공정한 벤치마크.

  run_01: 1명 × 548파일
  run_02: 2명 각각 × 548파일 (동시)
  ...
  run_10: 10명 각각 × 548파일 (동시)

실행 위치: 리포지토리 루트
  python evaluation/LibriSpeech/run_multi_speaker_full.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[2]   # <repo>/evaluation/LibriSpeech/<this file>
LIBRISPEECH   = PROJECT_ROOT / "evaluation/LibriSpeech"
SERVER_SCRIPT = LIBRISPEECH / "servers/streaming_websocket_server_fsl.py"
TEST_SCRIPT   = LIBRISPEECH / "servers/test_qwen3_librispeech.py"
TEST_DIR      = LIBRISPEECH / "test-other"
RESULTS_DIR   = LIBRISPEECH / "results/finetuned(1.0.1)/multi_speaker_test"
TEST_RESULTS_ROOT = LIBRISPEECH / "results/finetuned(1.0.1)/full"
MODEL         = PROJECT_ROOT / "models/Qwen3-ASR-1.7B-en-dailytalk-seg"
PYTHON        = os.environ.get("EVAL_PYTHON", sys.executable)
CONDA_BIN     = os.environ.get("EVAL_BIN_DIR", str(Path(PYTHON).parent))

PORT       = 8765
NUM_FILES  = 548
# 실행할 동시접속 수 목록 (None이면 START_N~END_N 순차)
RUN_LIST   = [20, 30, 40, 50, 60]
START_N    = 1
END_N      = 10

# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    return False


def kill_existing_server(port: int = PORT) -> None:
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

    for pattern in ["streaming_websocket_server_fsl", "VLLM::EngineCore"]:
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


def get_sorted_file_ids() -> list[str]:
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
    result = all_ids[:NUM_FILES]
    log(f"파일 ID {len(result)}개 로드 (전체 {len(all_ids)}개 중)")
    return result


def save_file_list_json(file_ids: list[str], path: Path) -> None:
    data = {"raw_results": [{"file_id": fid} for fid in file_ids]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def merge_client_results(client_metric_jsons: list[Path], result_dir: Path) -> None:
    """각 클라이언트 metric.json → per-client cN.json 저장 + 전체 병합."""
    all_raw: list[dict] = []

    for i, path in enumerate(client_metric_jsons):
        if not path.exists():
            log(f"  경고: {path} 없음, 스킵")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("raw_results", [])
        all_raw.extend(raw)
        log(f"  c{i}: {len(raw)}개 파일")

        # per-client JSON 저장 (run_01/02 포맷과 동일하게)
        client_out = result_dir / f"qwen3_test_other_fsl_c{i}.json"
        client_data = {"policy_3": {"raw_results": raw}}
        with open(client_out, "w", encoding="utf-8") as f:
            json.dump(client_data, f, indent=2, ensure_ascii=False)

    if not all_raw:
        log("  병합할 결과 없음")
        return

    # 전체 통계 계산
    fsl_vals, asr_vals, tl_vals, tok_vals = [], [], [], []
    commit_counts: dict[str, int] = {"vad": 0, "seg": 0, "dot": 0, "finish": 0, "always": 0}

    for r in all_raw:
        for seg in r.get("segment_metrics", []):
            fsl_key = "fsl_sec" if "fsl_sec" in seg else "server_fsl_sec"
            asr_key = "decode_sec" if "decode_sec" in seg else "asr_inference_sec"
            tl_key  = "trans_sec" if "trans_sec" in seg else "translation_latency_sec"
            if seg.get(fsl_key) is not None: fsl_vals.append(seg[fsl_key])
            if seg.get(asr_key) is not None: asr_vals.append(seg[asr_key])
            if seg.get(tl_key)  is not None: tl_vals.append(seg[tl_key])
            if seg.get("output_token_count") is not None:
                tok_vals.append(seg["output_token_count"])
            reason = str(seg.get("commit_reason", "")).lower()
            for key in commit_counts:
                if reason.startswith(key):
                    commit_counts[key] += 1
                    break

    def avg(vals: list): return sum(vals) / len(vals) if vals else None

    total = sum(commit_counts.values())
    overall = {
        "num_files": len(all_raw),
        "avg_server_fsl_sec": avg(fsl_vals),
        "avg_asr_inference_sec": avg(asr_vals),
        "avg_translation_latency_sec": avg(tl_vals),
        "avg_output_tokens_per_commit": avg(tok_vals),
        "commit_stats": {
            "counts": commit_counts,
            "total": total,
            "ratios": {k: v / total if total else 0 for k, v in commit_counts.items()},
        },
    }

    merged = {"policy_3": {"overall": overall, "raw_results": all_raw}}
    merged_path = result_dir / "qwen3_test_other_fsl.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    fsl_str = f"{overall['avg_server_fsl_sec']:.3f}s" if overall['avg_server_fsl_sec'] else "N/A"
    asr_str = f"{overall['avg_asr_inference_sec']:.3f}s" if overall['avg_asr_inference_sec'] else "N/A"
    tl_str  = f"{overall['avg_translation_latency_sec']:.3f}s" if overall['avg_translation_latency_sec'] else "N/A"
    log(f"  병합 완료: {len(all_raw)}개 항목 (클라이언트 {len(client_metric_jsons)}명 × {NUM_FILES}파일)")
    log(f"  FSL={fsl_str}  ASR={asr_str}  번역={tl_str}")


# ── 개별 테스트 실행 ────────────────────────────────────────────────────────────

def run_test(n_clients: int, file_ids: list[str]) -> None:
    run_name   = f"run_{n_clients:02d}"
    result_dir = RESULTS_DIR / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    log(sep)
    log(f" 테스트 {run_name}: 동시 {n_clients}명, 각각 {len(file_ids)}개 파일")
    log(sep)

    kill_existing_server()

    server_log    = result_dir / "server.log"
    server_stdout = result_dir / "server_stdout.log"
    server_env    = os.environ.copy()
    server_env["PATH"] = CONDA_BIN + ":" + server_env.get("PATH", "")

    server_proc = subprocess.Popen(
        [PYTHON, str(SERVER_SCRIPT),
         "--model", str(MODEL),
         "--enforce-eager",
         "--no-idle-shutdown",
         "--log-file", str(server_log)],
        stdout=open(server_stdout, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=server_env,
    )
    log(f"서버 시작 (PID={server_proc.pid})")

    if not wait_for_server():
        log("오류: 서버 시작 타임아웃, 이 테스트 건너뜀")
        server_proc.kill()
        return

    # 전체 파일 목록 (모든 클라이언트가 공유)
    all_files_path = result_dir / "_all_files.json"
    save_file_list_json(file_ids, all_files_path)

    # chunk 파일도 동일하게 저장 (기존 포맷 호환)
    for i in range(n_clients):
        chunk_path = result_dir / f"_chunk_{i}.json"
        save_file_list_json(file_ids, chunk_path)

    client_procs:        list[subprocess.Popen] = []
    client_metric_jsons: list[Path]             = []
    tmp_tags:            list[str]              = []

    for i in range(n_clients):
        tmp_tag        = f"multi_speaker_full_tmp/{run_name}_c{i}"
        client_run_dir = TEST_RESULTS_ROOT / tmp_tag
        client_metric  = client_run_dir / "metric.json"
        client_metric_jsons.append(client_metric)
        tmp_tags.append(tmp_tag)
        client_log = result_dir / f"test_stdout_c{i}.log"

        proc = subprocess.Popen(
            [PYTHON, str(TEST_SCRIPT),
             "--test-dir",       str(TEST_DIR),
             "--host",           "localhost",
             "--port",           str(PORT),
             "--model",          "finetuned(1.0.1)",
             "--scope",          "full",
             "--tag",            tmp_tag,
             "--common-files",   str(all_files_path),
             "--fresh-start",
             "--no-calculate-wer",
             "--skip-protocol-smoke",
             "--trailing-silence-ms", "5500"],
            stdout=open(client_log, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(LIBRISPEECH),
        )
        client_procs.append(proc)
        log(f"  클라이언트 {i + 1}/{n_clients} 시작 (PID={proc.pid})")

    t0 = time.time()
    for i, proc in enumerate(client_procs):
        proc.wait()
        elapsed = time.time() - t0
        log(f"  클라이언트 {i + 1} 완료 (rc={proc.returncode}, 경과={elapsed:.0f}s)")

    log(f"전체 소요: {time.time() - t0:.0f}s")

    log("결과 저장 중...")
    merge_client_results(client_metric_jsons, result_dir)

    # 임시 디렉토리 정리
    for tag in tmp_tags:
        tmp_dir = TEST_RESULTS_ROOT / tag
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    tmp_parent = TEST_RESULTS_ROOT / "multi_speaker_full_tmp"
    if tmp_parent.exists() and not any(tmp_parent.iterdir()):
        tmp_parent.rmdir()

    server_proc.terminate()
    try:
        server_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    log("서버 종료 완료")
    time.sleep(5)


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def main() -> None:
    log("파일 ID 목록 로딩...")
    file_ids = get_sorted_file_ids()

    targets = RUN_LIST if RUN_LIST else list(range(START_N, END_N + 1))
    log(f"벤치마크 시작: {targets}, 각 클라이언트 {NUM_FILES}파일 전체 처리")
    total_start = time.time()

    for n_clients in targets:
        run_test(n_clients, file_ids)

    log("=" * 60)
    log(f"모든 테스트 완료 (총 {(time.time() - total_start) / 3600:.1f}h)")
    log(f"결과 위치: {RESULTS_DIR}")

    log("\n── 요약 ──")
    for n_clients in targets:
        run_name = f"run_{n_clients:02d}"
        merged = RESULTS_DIR / run_name / "qwen3_test_other_fsl.json"
        if merged.exists():
            with open(merged, encoding="utf-8") as f:
                data = json.load(f)
            ov = data.get("policy_3", {}).get("overall", {})
            fsl = ov.get("avg_server_fsl_sec")
            asr = ov.get("avg_asr_inference_sec")
            n_f = ov.get("num_files", 0)
            log(f"  {n_clients:2d}명 ({run_name}): "
                f"FSL={fsl:.3f}s  ASR={asr:.3f}s  항목={n_f}" if fsl and asr
                else f"  {n_clients:2d}명 ({run_name}): 항목={n_f}")


if __name__ == "__main__":
    main()
