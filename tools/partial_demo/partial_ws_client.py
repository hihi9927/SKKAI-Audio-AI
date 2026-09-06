"""partial 스트리밍 확인용 최소 WebSocket 클라이언트.

wav 를 실시간 속도로 밀어 넣고 서버가 보내는 메시지를 시간순으로 찍는다.
마지막에 이번 변경에서 확인하려는 것들을 자동 판정한다.
"""
import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets

SR = 16000


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8766")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--target", default="ko")
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--tail-sec", type=float, default=20.0, help="finish 후 대기 시간")
    ap.add_argument("--expect-silence", action="store_true",
                    help="무음 입력. final/비어있지 않은 partial 이 하나도 없어야 통과")
    args = ap.parse_args()

    x, sr = sf.read(args.wav, dtype="float32")
    assert sr == SR, sr
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    frame_bytes = int(SR * args.frame_ms / 1000) * 2

    events = []          # (t, type, payload)
    t0 = time.perf_counter()

    def log(kind, msg):
        t = time.perf_counter() - t0
        events.append((t, kind, msg))
        if kind == "partial":
            print(f"[{t:6.2f}s] partial seq={msg.get('seq')} {msg.get('text')!r}")
        elif kind == "final":
            print(f"[{t:6.2f}s] FINAL  reason={msg.get('commitReason')} "
                  f"orig={msg.get('original')!r}")
            print(f"{'':>10} trans={msg.get('translation')!r}")
        else:
            print(f"[{t:6.2f}s] {kind}: {json.dumps(msg, ensure_ascii=False)[:160]}")

    async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        log(hello.get("type", "?"), hello)

        await ws.send(json.dumps({"type": "start", "lang": args.lang,
                                  "targetLang": args.target, "displayMode": "mode-1"}))
        ready = json.loads(await ws.recv())
        log(ready.get("type", "?"), ready)

        done = asyncio.Event()

        async def reader():
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    m = json.loads(raw)
                    log(m.get("type", "?"), m)
            except Exception as e:
                print(f"reader stopped: {e!r}")
            finally:
                done.set()

        rt = asyncio.create_task(reader())

        # 실시간 속도로 전송
        send_start = time.perf_counter()
        for i in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[i:i + frame_bytes])
            target = send_start + (i + frame_bytes) / 2 / SR
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        print(f"[{time.perf_counter()-t0:6.2f}s] -- audio sent, sending finish --")
        await ws.send(json.dumps({"type": "finish"}))

        try:
            await asyncio.wait_for(asyncio.shield(done.wait()), timeout=args.tail_sec)
        except asyncio.TimeoutError:
            pass
        rt.cancel()

    # ── 판정 ──────────────────────────────────────────────────────────────────
    partials = [(t, m) for t, _k, m in events if m.get("type") == "partial"]
    finals = [(t, m) for t, _k, m in events if m.get("type") == "final"]
    print("\n===== summary =====")
    print(f"partial 메시지: {len(partials)}건 (빈 문자열 {sum(1 for _t,m in partials if not m.get('text'))}건)")
    print(f"final 메시지  : {len(finals)}건")

    ok = True
    if args.expect_silence:
        noisy = [(t, m["text"]) for t, m in partials if m.get("text")]
        if noisy:
            print(f"FAIL: 무음인데 partial 에 텍스트가 있다: {noisy}")
            ok = False
        if finals:
            print(f"FAIL: 무음인데 final 이 나왔다: {[m.get('original') for _t, m in finals]}")
            ok = False
        print("RESULT:", "OK" if ok else "FAIL")
        return
    if not partials:
        print("FAIL: partial 이 한 건도 오지 않았다")
        ok = False
    if not finals:
        print("FAIL: final 이 한 건도 오지 않았다")
        ok = False

    # final 직후에 빈 partial 이 오는가 (화면 비우기)
    seq_events = [(t, m) for t, _k, m in events if m.get("type") in ("partial", "final")]
    for i, (t, m) in enumerate(seq_events):
        if m.get("type") != "final":
            continue
        after = [mm for _tt, mm in seq_events[i + 1:] if mm.get("type") == "partial"]
        if not after:
            print(f"WARN: [{t:.2f}s] final 이후 partial 이 없다 (마지막 final 이면 정상)")
            continue
        if after[0].get("text"):
            print(f"FAIL: [{t:.2f}s] final 직후 partial 이 비어있지 않다: {after[0]['text']!r}")
            ok = False

    # 마지막 메시지가 빈 partial 인가 (스트림 종료 후 유령 텍스트 없음)
    if partials and partials[-1][1].get("text"):
        print(f"FAIL: 마지막 partial 이 비어있지 않다: {partials[-1][1]['text']!r}")
        ok = False

    # 메타 헤더 누출: <asr_text> 태그 전 가설은 "language English" 같은 헤더를
    # 그대로 전사로 되돌려준다. 화면에 뜨면 안 된다.
    leaks = [(t, m["text"]) for t, m in partials
             if m.get("text", "").strip().lower().startswith("language")]
    if leaks:
        for t, txt in leaks:
            print(f"FAIL: [{t:.2f}s] partial 에 메타 헤더가 샜다: {txt!r}")
        ok = False

    # seq 단조 증가
    seqs = [m.get("seq") for _t, m in partials if m.get("seq") is not None]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        print(f"FAIL: partial seq 가 단조 증가가 아니다: {seqs}")
        ok = False

    # partial 전송 간격
    if len(partials) > 1:
        gaps = [partials[i + 1][0] - partials[i][0] for i in range(len(partials) - 1)]
        print(f"partial 간격: min={min(gaps)*1000:.0f}ms  중앙값={sorted(gaps)[len(gaps)//2]*1000:.0f}ms")
        # 120ms 미만 간격은 청크 끝 force 재동기화다(스로틀을 일부러 건너뛴다).
        # 클라이언트에서는 force 여부를 구분할 수 없으므로 건수만 센다.
        tight = sum(1 for g in gaps if g < 0.10)
        print(f"120ms 미만 간격 {tight}건 — 청크 끝 force 재동기화로 예상되는 수치")

    # partial 이 final 보다 얼마나 앞서 나오는가 (이 기능의 목적)
    leads = []
    prev_final_t = 0.0
    for ft, _fm in finals:
        before = [t for t, m in partials if prev_final_t < t < ft and m.get("text")]
        if before:
            leads.append(ft - before[0])
        prev_final_t = ft
    if leads:
        print("final 대비 첫 partial 선행 시간: "
              + ", ".join(f"{v:.2f}s" for v in leads))

    print("RESULT:", "OK" if ok else "FAIL")


asyncio.run(main())
