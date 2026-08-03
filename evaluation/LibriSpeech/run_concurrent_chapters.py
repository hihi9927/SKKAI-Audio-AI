#!/usr/bin/env python3
"""동시 클라이언트 부하 벤치마크 — 챕터 단위 큐 + 단일 프로세스 asyncio 워커.

N개의 워커가 하나의 이벤트 루프를 공유하며 챕터 큐에서 작업을 꺼내 간다.
각 워커는 챕터 하나를 맡아 WebSocket 연결 1회로 그 챕터의 파일을 순차 스트리밍한다.

설계 근거
---------
* **챕터 단위 큐 (파일 단위 아님)**
  기존 클라이언트(`servers/test_qwen3_librispeech.py`)는 챕터 경계에서만 재연결하고
  챕터 안에서는 연결을 유지해 서버가 이전 발화를 문맥으로 쓴다.
  파일 단위로 큐를 돌리면 매 파일마다 다른 챕터가 나와 문맥이 항상 비고, WER이
  기존 결과와 비교 불가능해진다.

* **정적 분배 아닌 동적 큐**
  챕터 길이 편차가 커서(40s~603s) 정적 분배는 종료 시점이 크게 어긋나고,
  꼬리 구간이 사실상 저부하 조건을 측정하게 된다.
  긴 챕터부터 배분(LPT)하면 N=32에서 큐 효율 약 95%.

* **단일 프로세스**
  프로세스를 N개 띄우면 OS 스케줄러가 서버 프로세스와 함께 돌려막아 200ms 전송
  시점이 밀린다(= 지터). 지터는 FSL에 그대로 더해져 서버 지연과 구분되지 않는다.
  대신 이벤트 루프가 하나뿐이므로 블로킹 호출은 반드시 `asyncio.to_thread`로 빼야 한다.

사용 예
-------
    python evaluation/LibriSpeech/run_concurrent_chapters.py \
        --num-clients 32 --scope full --tag concurrent32_run01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import websockets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "servers"))

import test_qwen3_librispeech as base  # noqa: E402

logging.basicConfig(format="%(levelname)s\t%(message)s", level=logging.INFO)
logger = logging.getLogger("concurrent")

DEFAULT_TEST_DIR = HERE / "LibriSpeech" / "test-other"


# ── 통계 유틸 ────────────────────────────────────────────────────────────────

def percentile(values, p):
    """선형 보간 분위수. numpy 의존 없이 결과 JSON에 넣기 좋은 float를 돌려준다."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (k - low))


# ── 챕터 구성 ────────────────────────────────────────────────────────────────

def chapter_id_of(file_id: str) -> str:
    """'3570-5694-0007' → '3570-5694' (speaker-chapter)."""
    parts = file_id.split("-")
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else parts[0]


def build_chapters(audio_files, skip_ids=None):
    """파일 목록을 챕터로 묶고, 총 재생시간이 긴 순으로 정렬(LPT)해 돌려준다.

    duration은 sf.info()로 헤더만 읽어 구한다(디코딩 없음).
    """
    skip_ids = skip_ids or set()
    grouped: dict[str, list] = {}
    for info in audio_files:
        grouped.setdefault(chapter_id_of(info["file_id"]), []).append(info)

    chapters = []
    for chapter, files in grouped.items():
        remaining = [f for f in files if f["file_id"] not in skip_ids]
        if not remaining:
            continue  # 이미 전부 처리된 챕터 — resume 시 통째로 건너뜀
        remaining.sort(key=lambda x: x["file_id"])
        duration = 0.0
        for f in remaining:
            try:
                duration += sf.info(f["path"]).duration
            except Exception as e:
                logger.warning("duration 조회 실패 %s: %s", f["path"], e)
        chapters.append({"chapter_id": chapter, "files": remaining, "duration_sec": duration})

    chapters.sort(key=lambda c: c["duration_sec"], reverse=True)
    return chapters


def simulate_queue_balance(chapters, num_clients, per_file_overhead_sec):
    """LPT 그리디로 makespan을 예측한다. 실행 전 큐 효율을 눈으로 확인하기 위한 것."""
    loads = [0.0] * num_clients
    for chapter in chapters:
        work = chapter["duration_sec"] + len(chapter["files"]) * per_file_overhead_sec
        idx = min(range(num_clients), key=lambda i: loads[i])
        loads[idx] += work
    total = sum(loads)
    makespan = max(loads) if loads else 0.0
    ideal = total / num_clients if num_clients else 0.0
    return {
        "total_work_sec": total,
        "ideal_makespan_sec": ideal,
        "predicted_makespan_sec": makespan,
        "predicted_efficiency": (ideal / makespan) if makespan > 0 else None,
    }


# ── 워커 ─────────────────────────────────────────────────────────────────────

class ConcurrentRunner:
    def __init__(self, args, chapters, run_dir):
        self.args = args
        self.chapters = chapters
        self.run_dir = run_dir
        self.ws_url = f"ws://{args.host}:{args.port}"
        self.queue: asyncio.Queue = asyncio.Queue()
        for chapter in chapters:
            self.queue.put_nowait(chapter)

        self.results: list[dict] = []
        self.active_clients = 0          # 현재 스트리밍 중인 워커 수
        self.worker_busy_sec = [0.0] * args.num_clients
        self.results_lock = asyncio.Lock()
        self.jsonl_path = run_dir / "results.jsonl"
        self.done_files = 0
        self.total_files = sum(len(c["files"]) for c in chapters)
        self.start_perf = None

    async def _record(self, row):
        """결과를 메모리와 JSONL에 남긴다.

        metric.json은 매 파일마다 쓰지 않는다 — WER 재계산이 이벤트 루프를 붙잡아
        나머지 워커의 전송 페이싱을 망가뜨리기 때문. 대신 append-only JSONL로
        크래시 복구만 보장하고, 집계는 종료 시 한 번 수행한다.
        """
        async with self.results_lock:
            self.results.append(row)
            self.done_files += 1
            done, total = self.done_files, self.total_files
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if done % 50 == 0 or done == total:
            elapsed = time.perf_counter() - self.start_perf
            logger.info("진행 %s/%s 파일 · 활성 %s명 · 경과 %.0fs",
                        done, total, self.active_clients, elapsed)

    async def _open_ws(self):
        ws = await websockets.connect(
            self.ws_url, ping_interval=None, ping_timeout=None,
            open_timeout=30, max_size=10 * 1024 * 1024,
        )
        await base.recv_type(ws, "hello", timeout=15)
        return ws

    async def worker(self, wid: int):
        args = self.args
        while True:
            try:
                chapter = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            chapter_start = time.perf_counter()
            try:
                ws = await self._open_ws()
            except Exception as e:
                logger.error("[w%02d] 챕터 %s 연결 실패: %s", wid, chapter["chapter_id"], e)
                continue

            # 이 워커가 실제로 스트리밍하는 동안만 활성으로 센다.
            self.active_clients += 1
            try:
                for audio_info in chapter["files"]:
                    file_id = audio_info["file_id"]
                    # 디코딩은 블로킹 — 스레드로 빼지 않으면 나머지 워커가 전부 멈춘다.
                    audio = await asyncio.to_thread(base.load_audio_file, audio_info["path"])
                    if audio is None:
                        continue
                    duration = len(audio) / base.SAMPLING_RATE

                    try:
                        # finish는 process_single_file의 _send가 이미 보낸다
                        # (recv 루프가 살아있는 동안 finish-트리거 final을 받기 위해).
                        # 여기서 한 번 더 보내면 서버가 finish를 두 번 처리하고 finish_done
                        # ack도 두 번 나간다. 서버가 밀리면 그 여분 ack가 다음 파일의 recv
                        # 루프에 도착해 아직 final이 오기 전에 종료시킬 수 있다.
                        # 중복 자체가 불필요하므로 제거한다.
                        out = await base.process_single_file(
                            ws, audio,
                            chunk_size_ms=args.chunk_size_ms,
                            send_interval_ms=args.send_interval_ms,
                            target_lang=args.target_lang,
                            trailing_silence_ms=args.trailing_silence_ms,
                        )
                    except Exception as e:
                        # 예외 타입까지 남긴다 — asyncio.TimeoutError는 str()이 빈 문자열이라
                        # 메시지만 찍으면 "처리 실패: "로 끝나 원인을 알 수 없다.
                        logger.error("[w%02d] %s 처리 실패: %s: %s",
                                     wid, file_id, type(e).__name__, e)
                        break  # 연결이 깨졌을 수 있음 — 이 챕터는 중단

                    if not out["transcript"]:
                        logger.warning("[w%02d] 빈 전사: %s", wid, file_id)
                        continue

                    await self._record({
                        "file_id": file_id,
                        "speaker_id": file_id.split("-")[0],
                        "chapter_id": chapter["chapter_id"],
                        "worker_id": wid,
                        "audio_path": audio_info["path"],
                        "reference": audio_info["reference"],
                        "hypothesis": out["transcript"],
                        "hyp_commit": base.format_commit_markers(out.get("segment_events") or []),
                        "duration": duration,
                        "total_time": out["total_time"],
                        "first_token_latency": out["first_token_latency"],
                        "model_runtime": out["total_time"] - duration,
                        "target_lang": args.target_lang,
                        "segment_metrics": out.get("segment_metrics") or [],
                        "segment_metrics_summary": out.get("segment_metrics_summary") or {},
                    })
            finally:
                self.active_clients -= 1
                try:
                    await ws.send(json.dumps({"type": "stop"}))
                except Exception:
                    pass
                try:
                    await ws.close()
                except Exception:
                    pass

            self.worker_busy_sec[wid] += time.perf_counter() - chapter_start
            logger.info("[w%02d] 챕터 %s 완료 (%.0fs, 파일 %s개, 남은 챕터 %s)",
                        wid, chapter["chapter_id"], time.perf_counter() - chapter_start,
                        len(chapter["files"]), self.queue.qsize())

    async def run(self):
        self.start_perf = time.perf_counter()
        await asyncio.gather(*[self.worker(i) for i in range(self.args.num_clients)])
        return time.perf_counter() - self.start_perf


# ── 집계 ─────────────────────────────────────────────────────────────────────

def build_concurrency_stats(runner, makespan_sec, balance_prediction):
    fsl_values = [
        seg["fsl_sec"]
        for row in runner.results
        for seg in (row.get("segment_metrics") or [])
        if seg.get("fsl_sec") is not None
    ]
    idle = [max(0.0, makespan_sec - busy) for busy in runner.worker_busy_sec]

    return {
        "num_clients": runner.args.num_clients,
        "num_chapters": len(runner.chapters),
        "makespan_sec": round(makespan_sec, 2),
        "queue_balance_prediction": balance_prediction,
        "worker_idle_sec": {
            "per_worker": [round(x, 2) for x in idle],
            "total": round(sum(idle), 2),
            "max": round(max(idle), 2) if idle else None,
            "mean": round(sum(idle) / len(idle), 2) if idle else None,
        },
        "fsl_sec_percentiles": {
            "count": len(fsl_values),
            "p50": percentile(fsl_values, 50),
            "p95": percentile(fsl_values, 95),
            "p99": percentile(fsl_values, 99),
        },
    }


def save_metrics(runner, makespan_sec, balance_prediction, policy):
    (runner.run_dir / "plots").mkdir(parents=True, exist_ok=True)
    summary = base.build_summary_payload(runner.results, policy)
    concurrency = build_concurrency_stats(runner, makespan_sec, balance_prediction)

    payload = {
        "overall": summary.get("overall"),
        "concurrency": concurrency,
        "folders": summary.get("folders"),
        "raw_results": runner.results,
    }
    with open(runner.run_dir / "metric.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_resume_ids(run_dir: Path) -> set:
    """results.jsonl(우선) 또는 metric.json에서 이미 처리된 file_id를 모은다."""
    jsonl = run_dir / "results.jsonl"
    if jsonl.exists():
        ids = set()
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["file_id"])
                except Exception:
                    continue
        return ids
    return base.load_processed_files(run_dir)


def load_prior_results(run_dir: Path) -> list:
    jsonl = run_dir / "results.jsonl"
    rows = []
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-clients", type=int, default=32,
                        help="동시 워커 수 (기본 32)")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=str, default="mode2")
    parser.add_argument("--scope", type=str, default="full")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--policy", type=int, default=base.DEFAULT_POLICY)
    parser.add_argument("--max-chapters", type=int, default=None,
                        help="처리할 챕터 수 상한 (파일럿/부하곡선용 부분집합). 긴 챕터부터 선택")
    parser.add_argument("--chunk-size-ms", type=int, default=200)
    parser.add_argument("--send-interval-ms", type=int, default=200)
    parser.add_argument("--trailing-silence-ms", type=int, default=5500)
    parser.add_argument("--target-lang", type=str, default="",
                        help="번역 대상 언어. 빈 문자열이면 서버가 번역을 건너뜀 (기본: 번역 off)")
    parser.add_argument("--fresh-start", action="store_true",
                        help="기존 results.jsonl 무시하고 처음부터")
    parser.add_argument("--dry-run", action="store_true",
                        help="서버에 접속하지 않고 챕터 구성/큐 균형 예측만 출력")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        logger.error("test-dir 없음: %s", test_dir)
        sys.exit(1)

    files = base.find_audio_files(str(test_dir))
    if not files:
        logger.error("오디오 파일을 찾지 못함: %s", test_dir)
        sys.exit(1)

    run_dir = base.resolve_run_dir(args)
    logger.info("결과 → %s", run_dir)

    skip_ids = set()
    prior_results = []
    if args.fresh_start:
        jsonl = run_dir / "results.jsonl"
        if jsonl.exists():
            jsonl.unlink()
    else:
        skip_ids = load_resume_ids(run_dir)
        prior_results = load_prior_results(run_dir)
        if skip_ids:
            logger.info("resume: 이미 처리된 파일 %s개 건너뜀", len(skip_ids))

    chapters = build_chapters(files, skip_ids=skip_ids)
    if args.max_chapters is not None:
        chapters = chapters[:args.max_chapters]
    if not chapters:
        logger.info("처리할 챕터가 없습니다.")
        return

    total_audio = sum(c["duration_sec"] for c in chapters)
    total_files = sum(len(c["files"]) for c in chapters)
    overhead = args.trailing_silence_ms / 1000.0
    balance = simulate_queue_balance(chapters, args.num_clients, overhead)

    logger.info("=" * 68)
    logger.info("챕터 %s개 · 파일 %s개 · 오디오 %.0fs (%.2fh)",
                len(chapters), total_files, total_audio, total_audio / 3600)
    logger.info("클라이언트 %s명 · trailing silence %.1fs/파일",
                args.num_clients, overhead)
    logger.info("예상 makespan %.0fs (이상 %.0fs, 효율 %.0f%%)",
                balance["predicted_makespan_sec"], balance["ideal_makespan_sec"],
                (balance["predicted_efficiency"] or 0) * 100)
    logger.info("=" * 68)

    if args.dry_run:
        logger.info("dry-run — 실행하지 않고 종료합니다.")
        for c in chapters[:10]:
            logger.info("  %s  %6.0fs  파일 %s개", c["chapter_id"], c["duration_sec"], len(c["files"]))
        return

    if args.description:
        (run_dir / "description.txt").write_text(args.description, encoding="utf-8")

    ws_url = f"ws://{args.host}:{args.port}"
    server_config = asyncio.run(base.fetch_server_config(ws_url))
    if server_config is None:
        logger.error("서버 config를 받지 못했습니다. 서버가 %s에 떠 있는지 확인하세요.", ws_url)
        sys.exit(1)
    logger.info("서버 config: %s", json.dumps(server_config, ensure_ascii=False))

    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "cli_args": vars(args),
            "server_config": server_config,
            "runner": "run_concurrent_chapters.py",
        }, f, indent=2, ensure_ascii=False)

    runner = ConcurrentRunner(args, chapters, run_dir)
    runner.results.extend(prior_results)   # resume분은 집계에만 포함
    runner.done_files = 0
    runner.total_files = total_files

    try:
        makespan = asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.warning("중단됨 — 지금까지의 결과로 집계합니다.")
        makespan = time.perf_counter() - (runner.start_perf or time.perf_counter())

    payload = save_metrics(runner, makespan, balance, args.policy)

    overall = payload["overall"]
    conc = payload["concurrency"]
    logger.info("=" * 68)
    logger.info("완료 · 파일 %s개 · makespan %.0fs", overall.get("num_files"), makespan)
    wer = overall.get("wer")
    logger.info("WER: %s", f"{wer * 100:.2f}%" if wer is not None else "N/A")
    logger.info("FSL avg %.4fs · p50 %.4f · p95 %.4f · p99 %.4f",
                overall.get("avg_fsl_sec") or 0,
                conc["fsl_sec_percentiles"]["p50"] or 0,
                conc["fsl_sec_percentiles"]["p95"] or 0,
                conc["fsl_sec_percentiles"]["p99"] or 0)
    logger.info("워커 유휴 시간 합 %.0fs (최대 %.0fs) — 클수록 꼬리 구간에서 부하가 빠진 것",
                conc["worker_idle_sec"]["total"] or 0, conc["worker_idle_sec"]["max"] or 0)
    logger.info("결과 → %s", run_dir)
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
