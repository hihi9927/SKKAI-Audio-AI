#!/usr/bin/env python3
"""공용 AST(음성번역) 평가 클라이언트 — LAAL(ms) + BLEU.

manifest(JSONL) 한 줄이 발화 하나다. 데이터셋마다 클라이언트를 복제하지 않고
manifest 생성기만 새로 쓰면 붙는다 (`build_manifest_mustc.py` 참고).

    # 터미널 1
    python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown

    # 터미널 2
    python evaluation/ast/test_ast.py \
        --manifest evaluation/ast/manifests/mustc_en-de_tst-COMMON.jsonl \
        --model "baseline(1.0.0)" --scope sample --tag run_01

지표
----
LAAL     비계산인지(non-computation-aware). d = `decisionAudioSec`(커밋 결정 시점까지
         읽은 소스 오디오). 정책만 평가하므로 GPU 가 달라도 재현된다.
LAAL_CA  계산인지. d = 클라이언트가 `final` 을 받은 실시간 경과. 실제 체감 지연.
BLEU     sacrebleu corpus BLEU. 발화별로 세그먼트 번역을 이어붙인 것 vs 참조 번역.

검산: LAAL_CA − LAAL ≈ mean(fsl). 요약에 같이 찍으므로 크게 어긋나면 배선을 의심할 것.
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
from statistics import mean

import numpy as np
import soundfile as sf
import websockets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics_ast as M  # noqa: E402

SAMPLING_RATE = 16000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── 오디오 ───────────────────────────────────────────────────────────────────

def load_segment_audio(wav_path: str, offset: float, duration: float) -> np.ndarray:
    """talk 통짜 wav 에서 [offset, offset+duration) 구간만 읽어 16kHz mono float32 로."""
    info = sf.info(wav_path)
    start = int(round(offset * info.samplerate))
    frames = int(round(duration * info.samplerate))
    data, sr = sf.read(wav_path, start=start, frames=frames, dtype="float32",
                       always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLING_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLING_RATE)
    return data


# ── 프로토콜 ─────────────────────────────────────────────────────────────────

async def recv_type(ws, expected, timeout=25.0, ignore=frozenset()):
    deadline = time.perf_counter() + timeout
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise asyncio.TimeoutError(f"{expected} 대기 시간 초과")
        msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        if data.get("type") == expected:
            return data
        if data.get("type") not in ignore:
            logger.debug("무시한 메시지: %s", data.get("type"))


async def stream_one(ws, audio, *, chunk_size_ms, send_interval_ms, target_lang,
                     trailing_silence_ms, src_lang, utt_id):
    """발화 하나를 스트리밍하고 세그먼트별 계측을 돌려준다.

    시간 원점은 **첫 오디오 청크를 보낸 시각**이다. `start`/`ready` 핸드셰이크를
    원점에 포함하면 서버 기동 상태에 따라 지연이 흔들린다.
    """
    # uttId 를 실어 보내면 서버가 커밋 시점의 발화 id 를 final 에 되돌려준다.
    # 발화 경계를 넘어 늦게 도착한 final 을 이번 발화에 잘못 붙이지 않기 위한 못이다.
    await ws.send(json.dumps({"type": "start", "lang": src_lang,
                              "targetLang": target_lang, "uttId": utt_id}))
    await recv_type(ws, "ready", timeout=25,
                    ignore={"partial", "final", "finish_done", "vad_done"})

    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_size = int((chunk_size_ms / 1000.0) * SAMPLING_RATE)
    send_interval_sec = max(0.0, send_interval_ms / 1000.0)

    origin: dict[str, float | None] = {"t": None}
    segments: list[dict] = []
    foreign: list[dict] = []   # 다른 발화에 속하는데 늦게 도착한 final
    send_done = asyncio.Event()
    real_audio_done = asyncio.Event()
    vad_fired = asyncio.Event()

    async def _pace_to(target_at):
        while True:
            remaining = target_at - time.perf_counter()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.02))

    async def _send():
        stream_origin = time.perf_counter()
        origin["t"] = stream_origin
        for i in range(0, len(audio_i16), chunk_size):
            chunk = audio_i16[i:i + chunk_size]
            if send_interval_sec > 0:
                await _pace_to(stream_origin + (i + len(chunk)) / SAMPLING_RATE)
            await ws.send(chunk.tobytes())
        real_audio_done.set()

        # VAD 가 자연스럽게 발동하도록 뒤에 묵음을 덧붙인다. VAD 발동 즉시 중단 —
        # 그 뒤 묵음은 다음 슬롯의 할루시네이션만 유발한다.
        if trailing_silence_ms > 0:
            silence = np.zeros(int(SAMPLING_RATE * trailing_silence_ms / 1000),
                               dtype=np.int16)
            silence_origin = time.perf_counter()
            for i in range(0, len(silence), chunk_size):
                if vad_fired.is_set():
                    break
                chunk = silence[i:i + chunk_size]
                if send_interval_sec > 0:
                    await _pace_to(silence_origin + (i + len(chunk)) / SAMPLING_RATE)
                if vad_fired.is_set():
                    break
                await ws.send(chunk.tobytes())

        # finish 를 여기서 보내야 _recv 가 아직 듣는 동안 finish-트리거 final 을 받는다.
        await ws.send(json.dumps({"type": "finish"}))
        send_done.set()

    async def _recv():
        # 정상 종료 신호는 서버의 finish_done ack 이다. 유휴 타임아웃은 ack 를 보내지
        # 않는 서버용 fallback 이며, 짧게 잡으면 서버가 밀렸을 때 final 을 통째로
        # 놓친다(LibriSpeech 클라이언트에서 실측된 실패 모드).
        POLL_SEC = 1.0
        POST_FINISH_GRACE_SEC = 60.0
        idle = 0.0

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=POLL_SEC)
            except asyncio.TimeoutError:
                if send_done.is_set():
                    idle += POLL_SEC
                    if idle >= POST_FINISH_GRACE_SEC:
                        break
                continue
            except Exception:
                break

            idle = 0.0
            if not isinstance(msg, str):
                continue
            data = json.loads(msg)
            msg_type = data.get("type", "")

            if msg_type == "finish_done":
                if send_done.is_set():
                    break
                continue

            if msg_type == "vad_done":
                if real_audio_done.is_set():
                    if not data.get("has_remaining", True):
                        vad_fired.set()
                        break
                continue

            if msg_type != "final":
                continue

            recv_perf = time.perf_counter()
            recv_elapsed = (recv_perf - origin["t"]) if origin["t"] else None

            # 이전 발화에서 늦게 나온 final 이면 이번 발화에 붙이지 않는다.
            # (서버가 uttId 를 안 보내는 구버전이면 검사를 건너뛴다)
            seg_utt = data.get("uttId")
            if seg_utt is not None and seg_utt != utt_id:
                foreign.append({"utt_id": seg_utt,
                                "translation": (data.get("translation") or "").strip()})
                continue

            segments.append({
                "segment_id": data.get("segmentId"),
                "original": (data.get("original") or "").strip(),
                "translation": (data.get("translation") or "").strip(),
                "commit_reason": (data.get("commitReason") or "").lower(),
                "decision_audio_sec": data.get("decisionAudioSec"),
                # 서버가 audioEndSec 로 보정한 경우에만 채워진다 (보정 전 원값)
                "decision_audio_sec_raw": data.get("decisionAudioSecRaw"),
                "audio_start_sec": data.get("audioStartSec"),
                "audio_end_sec": data.get("audioEndSec"),
                "fsl_sec": data.get("fsl_sec"),
                "trans_sec": data.get("trans_sec"),
                "emit_elapsed_sec": data.get("emitElapsedSec"),
                "recv_elapsed_sec": recv_elapsed,
            })

    await asyncio.gather(_send(), _recv())
    return segments, foreign


# ── 채점 ─────────────────────────────────────────────────────────────────────

def score_utterance(item, segments, args, warned):
    """발화 하나의 행(row)을 만든다. 시간 단위는 전부 ms."""
    src_ms = float(item["duration"]) * 1000.0
    ordered = sorted(
        segments,
        key=lambda s: (s["segment_id"] if s["segment_id"] is not None else 0),
    )

    hyp_parts, nca_pairs, nocap_pairs, ca_pairs = [], [], [], []
    for seg in ordered:
        text = seg["translation"]
        if not text:
            continue
        hyp_parts.append(text)

        decision = seg.get("decision_audio_sec")
        if decision is None:
            decision = seg.get("audio_end_sec")
            if not warned["decision"]:
                logger.warning(
                    "decisionAudioSec 가 없어 audioEndSec 로 대체합니다 — "
                    "AST 서버(streaming_websocket_server_ast.py)가 아닐 수 있습니다."
                )
                warned["decision"] = True
        if decision is not None:
            raw_ms = float(decision) * 1000.0
            # AL/LAAL 정의상 읽을 수 있는 소스는 T 를 넘지 못한다. 뒤에 붙인 묵음은
            # 평가 하네스가 만든 것이라 시스템의 지연으로 셈하면 안 된다.
            d_ms = min(raw_ms, src_ms) if args.laal_cap_source else raw_ms
            nca_pairs.append((text, d_ms))
            # 상한을 걸지 않은 값 — 검산 전용이다. CA − NCA = fsl 항등식은 상한이
            # 없을 때만 성립한다 (VAD 커밋은 speech_end + 침묵대기 시점에 결정되므로
            # 그 지점이 T 를 넘으면 상한이 차이를 fsl 보다 크게 만든다).
            nocap_pairs.append((text, raw_ms))

        if seg.get("recv_elapsed_sec") is not None:
            # CA 는 계산 비용을 포함하므로 T 로 자르지 않는다.
            ca_pairs.append((text, float(seg["recv_elapsed_sec"]) * 1000.0))

    hyp = " ".join(hyp_parts).strip()
    ref = item["tgt_text"]
    if args.strip_nonspeech:
        hyp = M.strip_nonspeech(hyp)
        ref = M.strip_nonspeech(ref)

    fsl_vals = [s["fsl_sec"] * 1000.0 for s in ordered if s.get("fsl_sec") is not None]

    return {
        "utt_id": item["utt_id"],
        "talk_id": item.get("talk_id", ""),
        "src_duration_ms": round(src_ms, 1),
        "src_text": item["src_text"],
        "ref_text": ref,
        "hyp_text": hyp,
        "asr_text": " ".join(s["original"] for s in ordered if s["original"]).strip(),
        "laal_ms": M.laal_for_utterance(nca_pairs, src_ms, ref, args.laal_unit),
        "laal_ca_ms": M.laal_for_utterance(ca_pairs, src_ms, ref, args.laal_unit),
        "laal_uncapped_ms": M.laal_for_utterance(nocap_pairs, src_ms, ref, args.laal_unit),
        "sentence_bleu": M.sentence_bleu_score(hyp, ref, args.bleu_tokenize),
        "mean_fsl_ms": mean(fsl_vals) if fsl_vals else None,
        "n_segments": len(ordered),
        "commit_reasons": [s["commit_reason"] for s in ordered],
        "segments": ordered,
    }


def summarize(rows, args):
    scored = [r for r in rows if r["hyp_text"]]
    bleu, sig = M.corpus_bleu_score(
        # 빈 가설도 반드시 포함한다. 어려운 발화를 버려서 BLEU 를 올리는 길을 막는다.
        [r["hyp_text"] for r in rows],
        [r["ref_text"] for r in rows],
        args.bleu_tokenize,
    )
    laal = M.mean_or_none(r["laal_ms"] for r in rows)
    laal_ca = M.mean_or_none(r["laal_ca_ms"] for r in rows)
    laal_nocap = M.mean_or_none(r.get("laal_uncapped_ms") for r in rows)
    mean_fsl_ms = M.mean_or_none(r["mean_fsl_ms"] for r in rows)

    reasons: dict[str, int] = {}
    # 타이밍 배선 검산은 **세그먼트 단위**로 한다. 집계 LAAL 끼리 빼면 안 된다 —
    # LAAL 은 τ 에서 잘리고 타깃 단어 수로 가중되는데 평균 FSL 은 세그먼트 균등
    # 가중이라, 배선이 멀쩡해도 두 값은 일치하지 않는다.
    # paired_* 는 fsl_sec 이 함께 온 세그먼트만 모은다. dot 커밋은 fsl_sec 이 비어 오는
    # 경우가 있어(FSL 서버 특성), 전체 평균끼리 비교하면 서로 다른 집합을 비교하게 된다.
    # LAAL 자체는 fsl 을 쓰지 않으므로 지표에는 영향이 없다.
    seg_diffs, seg_residuals, seg_fsls, paired_diffs = [], [], [], []
    for r in rows:
        for reason in r["commit_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
        for s in r["segments"]:
            dec, recv, seg_fsl = (s.get("decision_audio_sec"), s.get("recv_elapsed_sec"),
                                  s.get("fsl_sec"))
            if dec is None or recv is None:
                continue
            diff_ms = (float(recv) - float(dec)) * 1000.0
            seg_diffs.append(diff_ms)
            if seg_fsl is not None:
                seg_residuals.append(diff_ms - float(seg_fsl) * 1000.0)
                seg_fsls.append(float(seg_fsl) * 1000.0)
                paired_diffs.append(diff_ms)

    return {
        "laal_ms": laal,
        "laal_ca_ms": laal_ca,
        "bleu": bleu,
        "bleu_signature": sig,
        "bleu_tokenize": args.bleu_tokenize,
        "laal_unit": args.laal_unit,
        "laal_cap_source": args.laal_cap_source,
        "strip_nonspeech": args.strip_nonspeech,
        "mean_sentence_bleu": M.mean_or_none(r["sentence_bleu"] for r in rows),
        # mean_fsl_ms 는 발화 평균의 평균(발화별 보고용), mean_seg_fsl_ms 는 세그먼트
        # 평평 평균이다. 아래 검산은 세그먼트 단위끼리 비교해야 하므로 후자를 쓴다.
        "mean_fsl_ms": mean_fsl_ms,
        "mean_seg_fsl_ms": mean(seg_fsls) if seg_fsls else None,
        "laal_uncapped_ms": laal_nocap,
        # 세그먼트 단위 검산: (수신시각 − 결정시점) 은 계산 비용, 즉 FSL 과 같아야 한다.
        # 잔차가 크면 타이밍 배선을 의심할 것. 단 SEG 커밋의 fsl 은 토큰비율로 역추정한
        # audio_end 기준이라 결정시점 기준인 이 차이와 수백 ms 어긋나는 게 정상이다.
        "mean_seg_ca_minus_nca_ms": mean(seg_diffs) if seg_diffs else None,
        "mean_paired_ca_minus_nca_ms": mean(paired_diffs) if paired_diffs else None,
        "mean_seg_fsl_residual_ms": mean(seg_residuals) if seg_residuals else None,
        "max_abs_seg_fsl_residual_ms": max((abs(x) for x in seg_residuals), default=None),
        "n_seg_with_fsl": len(seg_residuals),
        "n_seg_total": len(seg_diffs),
        # 참고값 — 위 검산과 달리 이 둘은 일치할 이유가 없다(τ 절단 + 단어 가중).
        "laal_ca_minus_uncapped_ms": (
            (laal_ca - laal_nocap) if (laal_nocap is not None and laal_ca is not None) else None
        ),
        "n_utterances": len(rows),
        "n_empty_hypotheses": len(rows) - len(scored),
        # 발화 경계를 넘어 늦게 도착해 이번 발화에서 제외한 final 수.
        # 0 이 아니면 서버가 종료 시 커밋을 다 배출하지 못하고 있다는 뜻이다.
        "n_foreign_finals": sum(r.get("n_foreign_finals", 0) for r in rows),
        "mean_segments_per_utt": mean([r["n_segments"] for r in rows]) if rows else None,
        "commit_reason_counts": reasons,
    }


# ── 실행 ─────────────────────────────────────────────────────────────────────

def resolve_run_dir(args) -> Path:
    root = Path(args.results_root).expanduser().resolve() / args.dataset / args.model / args.scope
    if args.tag:
        return root / args.tag
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (root / f"run_{n:02d}").exists():
        n += 1
    return root / f"run_{n:02d}"


def load_manifest(path: Path, limit=None) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def load_done_rows(run_dir: Path) -> list[dict]:
    metric_file = run_dir / "metric.json"
    if not metric_file.exists():
        return []
    try:
        with open(metric_file, "r", encoding="utf-8") as f:
            return json.load(f).get("rows", [])
    except Exception:
        return []


def save_results(run_dir: Path, rows, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args)
    with open(run_dir / "metric.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {"args": vars(args), "finished_at": datetime.now().isoformat()},
            f, ensure_ascii=False, indent=2, default=str,
        )
    if args.description:
        (run_dir / "description.txt").write_text(args.description, encoding="utf-8")
    return summary


def print_summary(summary):
    def fmt(v, unit="", nd=2):
        return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) else "n/a"

    logger.info("=" * 62)
    logger.info("발화 %s개 (빈 가설 %s / 경계 넘은 final %s)", summary["n_utterances"],
                summary["n_empty_hypotheses"], summary.get("n_foreign_finals", 0))
    logger.info("LAAL      : %s", fmt(summary["laal_ms"], " ms", 1))
    logger.info("LAAL_CA   : %s", fmt(summary["laal_ca_ms"], " ms", 1))
    logger.info("검산(fsl 있는 %s/%s 세그먼트): 수신−결정 %s vs FSL %s  잔차 %s (최대 %s)",
                summary["n_seg_with_fsl"], summary["n_seg_total"],
                fmt(summary["mean_paired_ca_minus_nca_ms"], " ms", 1),
                fmt(summary["mean_seg_fsl_ms"], " ms", 1),
                fmt(summary["mean_seg_fsl_residual_ms"], " ms", 1),
                fmt(summary["max_abs_seg_fsl_residual_ms"], " ms", 1))
    logger.info("BLEU      : %s", fmt(summary["bleu"]))
    logger.info("  signature: %s", summary["bleu_signature"])
    logger.info("커밋 사유 : %s", summary["commit_reason_counts"])
    logger.info("=" * 62)


async def _worker(wid, url, queue, rows, lock, args, warned, run_dir, state, n_todo):
    """연결 하나를 잡고 큐가 빌 때까지 발화를 처리한다.

    오디오 로드와 채점은 `asyncio.to_thread` 로 뺀다 — 이벤트 루프를 막으면 다른
    워커의 실시간 페이싱이 밀려서 LAAL_CA 가 측정 대상이 아닌 이유로 늘어난다.
    """
    try:
        async with websockets.connect(url, max_size=None, ping_interval=20,
                                      ping_timeout=60, close_timeout=10) as ws:
            await recv_type(ws, "hello", timeout=60)
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    audio = await asyncio.to_thread(
                        load_segment_audio, item["wav"], item["offset"], item["duration"]
                    )
                except Exception as exc:
                    logger.error("[%s] 오디오 로드 실패: %s", item["utt_id"], exc)
                    continue

                segments, foreign = await stream_one(
                    ws, audio,
                    chunk_size_ms=args.chunk_size_ms,
                    send_interval_ms=args.send_interval_ms,
                    target_lang=args.target_lang,
                    trailing_silence_ms=args.trailing_silence_ms,
                    src_lang=args.src_lang,
                    utt_id=item["utt_id"],
                )
                row = await asyncio.to_thread(score_utterance, item, segments, args, warned)
                row["n_foreign_finals"] = len(foreign)
                if foreign:
                    # 이전 발화가 흘린 결과 — 이번 발화 가설에서는 제외했다.
                    state["foreign"] += len(foreign)

                async with lock:
                    rows.append(row)
                    state["done"] += 1
                    state["audio_sec"] += item["duration"]
                    n = state["done"]
                    if n <= 3 or n % args.log_every == 0:
                        logger.info(
                            "[%d/%d] w%02d %s | seg=%d LAAL=%s BLEU=%s",
                            n, n_todo, wid, item["utt_id"], row["n_segments"],
                            f"{row['laal_ms']:.0f}ms" if row["laal_ms"] is not None else "n/a",
                            f"{row['sentence_bleu']:.1f}" if row["sentence_bleu"] is not None else "n/a",
                        )
                    if args.save_every and n % args.save_every == 0:
                        await asyncio.to_thread(save_results, run_dir, list(rows), args)
    except Exception as exc:
        logger.error("워커 w%02d 종료: %s", wid, exc)


async def run(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    items = load_manifest(manifest_path, args.limit)
    if not items:
        logger.error("manifest 가 비었습니다: %s", manifest_path)
        return 2

    run_dir = resolve_run_dir(args)
    rows = [] if args.fresh_start else load_done_rows(run_dir)
    done = {r["utt_id"] for r in rows}
    todo = [it for it in items if it["utt_id"] not in done]
    logger.info("manifest %s개 / 완료 %s개 / 이번 실행 %s개 / 클라이언트 %s개 → %s",
                len(items), len(done), len(todo), args.clients, run_dir)
    if not todo:
        print_summary(save_results(run_dir, rows, args))
        return 0

    warned = {"decision": False}
    url = f"ws://{args.host}:{args.port}"
    queue: asyncio.Queue = asyncio.Queue()
    for it in todo:
        queue.put_nowait(it)

    lock = asyncio.Lock()
    state = {"done": 0, "audio_sec": 0.0, "foreign": 0}
    t_start = time.perf_counter()

    n_clients = max(1, min(args.clients, len(todo)))
    await asyncio.gather(*[
        _worker(i + 1, url, queue, rows, lock, args, warned, run_dir, state, len(todo))
        for i in range(n_clients)
    ])

    wall = time.perf_counter() - t_start
    summary = save_results(run_dir, rows, args)
    # 처리량: 실시간 페이싱이므로 1클라이언트의 상한이 1.0배속이다. 16병렬이 16배속에
    # 가까우면 서버가 병목이 아니라는 뜻이고, 크게 못 미치면 GPU/번역에서 밀린 것이다.
    logger.info("소요 %.1f분 | 오디오 %.2f시간 | 실시간 대비 %.1f배속 (클라이언트 %d개)",
                wall / 60.0, state["audio_sec"] / 3600.0,
                state["audio_sec"] / wall if wall > 0 else 0.0, n_clients)
    print_summary(summary)
    logger.info("결과: %s", run_dir / "metric.json")
    return 0


def main():
    p = argparse.ArgumentParser(description="AST 평가 (LAAL + BLEU)")
    p.add_argument("--manifest", required=True, help="build_manifest_*.py 로 만든 JSONL")
    p.add_argument("--dataset", default="MuST-C", help="결과 디렉토리의 데이터셋 이름")
    p.add_argument("--model", default="finetuned", help="대분류: 모델 종류")
    p.add_argument("--scope", default="sample", help="소분류: full / sample")
    p.add_argument("--tag", default=None, help="결과 폴더명. 지정 시 이어서 저장")
    p.add_argument("--description", default=None)
    p.add_argument("--results-root", default=str(HERE / "results"))
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh-start", action="store_true")
    p.add_argument("--save-every", type=int, default=20,
                   help="N개마다 중간 저장 (0이면 끝에만)")
    p.add_argument("--clients", type=int, default=1,
                   help="동시 WebSocket 연결 수. 각 연결이 큐에서 발화를 하나씩 가져간다")
    p.add_argument("--log-every", type=int, default=10,
                   help="N개마다 진행 로그 (병렬 실행 시 로그 폭주 방지)")

    p.add_argument("--src-lang", default="en",
                   help="ASR 소스 언어. 'auto' 로 두면 서버의 언어 제한이 꺼진다")
    p.add_argument("--target-lang", default="de")
    p.add_argument("--chunk-size-ms", type=int, default=200)
    p.add_argument("--send-interval-ms", type=int, default=200)
    p.add_argument("--trailing-silence-ms", type=int, default=8000)

    p.add_argument("--laal-unit", default="word", choices=["word", "char"],
                   help="LAAL 의 |Y| 단위. de/en/es 는 word, zh/ja 는 char")
    p.add_argument("--laal-cap-source", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="비계산인지 지연을 소스 길이로 상한. 뒤에 붙인 묵음이 지연으로 "
                        "새는 것을 막는다 (AL/LAAL 정의와 일치)")
    p.add_argument("--bleu-tokenize", default=None,
                   help="sacrebleu 토크나이저. 미지정 시 --target-lang 으로 결정")
    p.add_argument("--strip-nonspeech", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="참조/가설에서 (Laughter) 같은 이벤트 표기 제거")

    args = p.parse_args()
    if args.bleu_tokenize is None:
        args.bleu_tokenize = M.resolve_tokenize(args.target_lang)
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
