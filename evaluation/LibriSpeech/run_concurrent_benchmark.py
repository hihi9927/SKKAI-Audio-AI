#!/usr/bin/env python3
"""
동시 클라이언트 FSL 벤치마크 러너

fsl_1.0.1_results_11 (동시 1명) ~ fsl_1.0.1_results_20 (동시 10명) 순차 실행.
548개 파일을 N개 클라이언트에 균등 분배, 결과를 하나의 JSON으로 병합.

실행 위치: /home/ubuntu/STiTy
  python evaluation/LibriSpeech/run_concurrent_benchmark.py
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

PROJECT_ROOT  = Path("/home/ubuntu/STiTy")
LIBRISPEECH   = PROJECT_ROOT / "evaluation/LibriSpeech"
SERVER_SCRIPT = LIBRISPEECH / "servers/streaming_websocket_server_fsl.py"
TEST_SCRIPT   = LIBRISPEECH / "servers/test_qwen3_librispeech.py"
TEST_DIR      = LIBRISPEECH / "LibriSpeech/test-other"
RESULTS_DIR   = LIBRISPEECH / "results/fsl"
MODEL         = PROJECT_ROOT / "Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged"
PYTHON        = "/home/ubuntu/miniconda3/envs/qwen3-asr/bin/python"
CONDA_BIN     = "/home/ubuntu/miniconda3/envs/qwen3-asr/bin"

PORT          = 8765
NUM_FILES     = 548
RESULT_START  = 11   # results_11 ~ results_20

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
    """포트 점유 프로세스 및 잔존 VLLM EngineCore 종료."""
    killed = False

    # 1. 포트 점유 프로세스 종료
    try:
        result = subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        if result.returncode == 0:
            log(f"기존 서버 프로세스 종료 완료 (포트 {port})")
            killed = True
    except FileNotFoundError:
        try:
            r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            for pid in r.stdout.strip().split():
                subprocess.run(["kill", "-9", pid], capture_output=True)
            if r.stdout.strip():
                log(f"기존 서버 프로세스 종료 완료 (포트 {port})")
                killed = True
        except Exception:
            pass

    # 2. 잔존 서버/EngineCore 프로세스 강제 종료 (VRAM 해제 보장)
    for pattern in ["streaming_websocket_server_fsl", "VLLM::EngineCore"]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)

    if killed:
        time.sleep(3)

    # 3. VRAM 해제 대기 (최대 30초)
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
    """test-other에서 파일 ID 목록 추출 후 정렬, 상위 NUM_FILES개 반환."""
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


def split_chunks(items: list, n: int) -> list[list]:
    """items를 n개의 균등 청크로 분할."""
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def save_chunk_json(file_ids: list[str], path: Path) -> None:
    """load_common_file_ids가 인식하는 raw_results 포맷으로 저장."""
    data = {"raw_results": [{"file_id": fid} for fid in file_ids]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def merge_results(client_jsons: list[Path], merged_path: Path) -> None:
    """여러 클라이언트 결과 JSON을 하나로 병합, overall 통계 재계산."""
    all_raw: list[dict] = []

    for path in client_jsons:
        if not path.exists():
            log(f"  경고: {path.name} 없음, 스킵")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("policy_3", {}).get("raw_results", [])
        all_raw.extend(raw)
        log(f"  {path.name}: {len(raw)}개 파일 병합")

    if not all_raw:
        log("  병합할 결과 없음")
        return

    # segment_metrics에서 지표 추출
    fsl_vals, asr_vals, tl_vals, tok_vals = [], [], [], []
    commit_counts: dict[str, int] = {"vad": 0, "seg": 0, "dot": 0, "finish": 0}

    for r in all_raw:
        for seg in r.get("segment_metrics", []):
            if "server_fsl_sec" in seg:
                fsl_vals.append(seg["server_fsl_sec"])
            if "asr_inference_sec" in seg:
                asr_vals.append(seg["asr_inference_sec"])
            if "translation_latency_sec" in seg:
                tl_vals.append(seg["translation_latency_sec"])
            if "output_token_count" in seg:
                tok_vals.append(seg["output_token_count"])
            reason = str(seg.get("commit_reason", "")).lower()
            for key in commit_counts:
                if reason.startswith(key):
                    commit_counts[key] += 1
                    break

    def avg(vals: list) -> float | None:
        return sum(vals) / len(vals) if vals else None

    total_commits = sum(commit_counts.values())
    overall = {
        "num_files": len(all_raw),
        "avg_server_fsl_sec": avg(fsl_vals),
        "avg_asr_inference_sec": avg(asr_vals),
        "avg_translation_latency_sec": avg(tl_vals),
        "avg_output_tokens_per_commit": avg(tok_vals),
        "commit_stats": {
            "counts": commit_counts,
            "total": total_commits,
            "ratios": {
                k: (v / total_commits if total_commits else 0)
                for k, v in commit_counts.items()
            },
        },
    }

    merged = {"policy_3": {"overall": overall, "raw_results": all_raw}}
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    log(f"  병합 완료: {len(all_raw)}개 파일 → {merged_path.name}")
    log(f"  FSL 평균: {overall['avg_server_fsl_sec']:.3f}s  "
        f"ASR: {overall['avg_asr_inference_sec']:.3f}s  "
        f"번역: {overall['avg_translation_latency_sec']:.3f}s")


# ── 개별 테스트 실행 ────────────────────────────────────────────────────────────

def run_test(n_clients: int, result_idx: int, file_ids: list[str]) -> None:
    result_dir = RESULTS_DIR / f"fsl_1.0.1_results_{result_idx}"
    result_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 60
    log(sep)
    log(f" 테스트 {result_idx}: 동시 {n_clients}명 / 파일 {len(file_ids)}개")
    log(sep)

    # 1. 기존 서버 정리 후 새 서버 시작
    kill_existing_server()

    server_log    = result_dir / "server.log"
    server_stdout = result_dir / "server_stdout.log"

    import os as _os
    server_env = _os.environ.copy()
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

    # 2. 파일 분할
    chunks = split_chunks(file_ids, n_clients)

    # 3. 클라이언트 프로세스 동시 시작
    client_procs:   list[subprocess.Popen] = []
    client_outputs: list[Path]             = []

    for i, chunk in enumerate(chunks):
        chunk_path = result_dir / f"_chunk_{i}.json"
        client_out = result_dir / f"qwen3_test_other_fsl_c{i}.json"
        client_log = result_dir / f"test_stdout_c{i}.log"
        client_outputs.append(client_out)

        save_chunk_json(chunk, chunk_path)

        proc = subprocess.Popen(
            [PYTHON, str(TEST_SCRIPT),
             "--test-dir",       str(TEST_DIR),
             "--host",           "localhost",
             "--port",           str(PORT),
             "--output",         str(client_out),
             "--common-files",   str(chunk_path),
             "--fresh-start",
             "--no-calculate-wer",
             "--skip-protocol-smoke",
             "--trailing-silence-ms", "5500"],
            stdout=open(client_log, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(LIBRISPEECH),
        )
        client_procs.append(proc)
        log(f"  클라이언트 {i + 1}/{n_clients} 시작 (PID={proc.pid}, {len(chunk)}개 파일)")

    # 4. 전체 완료 대기
    t0 = time.time()
    for i, proc in enumerate(client_procs):
        proc.wait()
        elapsed = time.time() - t0
        rc = proc.returncode
        log(f"  클라이언트 {i + 1} 완료 (rc={rc}, 경과={elapsed:.0f}s)")

    log(f"전체 소요: {time.time() - t0:.0f}s")

    # 5. 결과 병합
    merged_path = result_dir / "qwen3_test_other_fsl.json"
    if n_clients == 1 and client_outputs[0].exists():
        shutil.copy(client_outputs[0], merged_path)
        log(f"결과 복사 완료: {merged_path.name}")
    else:
        log("결과 병합 중...")
        merge_results(client_outputs, merged_path)

    # 6. 서버 종료
    server_proc.terminate()
    try:
        server_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    log("서버 종료 완료")
    time.sleep(5)  # 포트 해제 대기


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def main() -> None:
    log("파일 ID 목록 로딩...")
    file_ids = get_sorted_file_ids()

    log(f"벤치마크 시작: 동시 1~10명, results_{RESULT_START}~results_{RESULT_START + 9}")
    total_start = time.time()

    for n_clients in range(1, 11):
        result_idx = RESULT_START + n_clients - 1  # 11 ~ 20
        run_test(n_clients, result_idx, file_ids)

    log("=" * 60)
    log(f"모든 테스트 완료 (총 {(time.time() - total_start) / 3600:.1f}h)")
    log(f"결과 위치: {RESULTS_DIR}/fsl_1.0.1_results_11 ~ fsl_1.0.1_results_20")

    # 요약 출력
    log("\n── FSL 요약 ──")
    for n_clients in range(1, 11):
        idx = RESULT_START + n_clients - 1
        merged = RESULTS_DIR / f"fsl_1.0.1_results_{idx}" / "qwen3_test_other_fsl.json"
        if merged.exists():
            with open(merged, encoding="utf-8") as f:
                data = json.load(f)
            overall = data.get("policy_3", {}).get("overall", {})
            fsl = overall.get("avg_server_fsl_sec")
            asr = overall.get("avg_asr_inference_sec")
            n_files = overall.get("num_files", 0)
            log(f"  {n_clients:2d}명 (results_{idx}): FSL={fsl:.3f}s  ASR={asr:.3f}s  파일={n_files}")


if __name__ == "__main__":
    main()
