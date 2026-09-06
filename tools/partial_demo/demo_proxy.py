"""정적 페이지와 WebSocket 을 한 포트로 묶고, 언어별 ASR 서버로 갈라 보낸다.

VS Code 원격 개발에서는 포트를 하나씩 포워딩해야 해서, 페이지(8080)와
ASR 서버(8766)를 따로 열면 두 번 포워딩해야 한다. 이 프록시는 8080 하나만
열고 /ws 요청을 ASR 서버로 중계한다.

**언어별 라우팅.** ASR 서버 한 대는 모델 한 개다. 한국어 파인튜닝 모델과 영어
파인튜닝 모델을 같이 쓰려면 서버를 두 대 띄우고 어디로 보낼지 골라야 하는데,
클라이언트를 고치지 않고 여기서 고른다. 클라이언트의 첫 `start` 메시지를 읽어
`lang` 으로 업스트림을 정한 뒤에야 위쪽에 붙는다.

    python demo_proxy.py 8080 --route ko=8766,en=8767 --default 8766

업스트림 접속이 늦어지므로 `hello` 는 이 프록시가 대신 내보낸다. 클라이언트가
보기에 핸드셰이크 순서(hello -> start -> ready)는 그대로다.

**`--lid` 를 켜면 음성으로 판정한다.** `start.lang` 을 믿지 않고, VAD 가 발화
시작을 잡은 뒤 whisper-base 로 언어를 보고 그 결과로 고른다. 한 마이크에 두
언어가 섞여 들어오는 화면에서도 맞는 모델로 간다. 근거와 실측값은
[lid_router.py](lid_router.py) 문서를 보라.

    python demo_proxy.py 8080 --route ko=8766,en=8767 --lid --lid-window 1.0

한계: 라우팅은 스트림이 열릴 때 한 번만 정해진다. 발화마다 언어가 바뀌어도
모델은 안 바뀐다. 흐르는 중에 오는 `config` 는 번역 방향만 바꾸고 모델은
그대로 둔다.
"""
import argparse
import asyncio
import http
import json
import logging
import mimetypes
import pathlib
import sys

import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = pathlib.Path(__file__).resolve().parent / "web"
ROUTES: dict[str, str] = {}
DEFAULT_UPSTREAM = "ws://127.0.0.1:8766"
PORT = 8080
LID = None            # LidRouter. --lid 를 켰을 때만 채워진다.

logger = logging.getLogger("demo-proxy")


def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None  # WebSocket 업그레이드는 핸들러로 넘긴다
    path = request.path.split("?")[0]
    if path in ("", "/"):
        path = "/index.html"
    target = (ROOT / path.lstrip("/")).resolve()
    if not str(target).startswith(str(ROOT)) or not target.is_file():
        return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")
    body = target.read_bytes()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    headers = Headers({
        "Content-Type": ctype,
        "Content-Length": str(len(body)),
        "Cache-Control": "no-store",
    })
    return Response(200, "OK", headers, body)


async def relay(src, dst):
    async for msg in src:
        await dst.send(msg)


def pick_upstream(start_msg):
    """`start` 메시지에서 쓸 ASR 서버를 고른다. (주소, 고른 이유) 를 돌려준다.

    1. `lang` 이 라우팅 표에 있으면 그것. 클라이언트가 말할 언어를 밝힌 경우다.
    2. `lang` 이 auto 라 판단이 안 서면 `langMap` 의 소스 언어를 본다. 딱 하나가
       표에 있으면 그것 — 한 언어만 켜 둔 화면이 여기 걸린다.
    3. 그 밖에는 기본 서버. 여러 언어를 한 스트림에서 받겠다는 뜻이라 어느 모델도
       특별히 맞지 않는다.
    """
    lang = str(start_msg.get("lang") or "").lower()
    if lang in ROUTES:
        return ROUTES[lang], f"lang={lang}"

    lang_map = start_msg.get("langMap")
    if isinstance(lang_map, dict):
        known = [k for k in lang_map if k in ROUTES]
        if len(known) == 1:
            return ROUTES[known[0]], f"langMap={known[0]}"

    return DEFAULT_UPSTREAM, f"default (lang={lang or 'none'})"


async def route_by_voice(client, start_msg, pending_binary):
    """오디오를 모으며 LID 판정이 설 때까지 기다린다.

    (업스트림, 사유, 지금까지 모은 오디오) 를 돌려준다. 모은 오디오는 하나도
    버리지 않고 위쪽에 그대로 흘려 보내므로, 판정을 기다린 만큼 ASR 이 늦게
    시작할 뿐 인식 대상이 줄지는 않는다. 청크마다 누적 오디오 전체를 다시 넣는
    구조라 앞부분도 정상적으로 전사된다.

    판정이 안 서면(무음만 들어옴, 너무 짧음) start.lang 규칙으로 되돌아간다.
    """
    from lid_router import Session

    sess = Session(LID)
    for chunk in pending_binary:
        sess.add(chunk)

    # 대기 중에 온 제어 메시지는 버리지 않는다. finish/stop 이 오면 더 기다려도
    # 오디오가 안 오므로 그 자리에서 판정하고, 메시지는 모아 뒀다가 위쪽에 넘긴다.
    control: list[str] = []
    flush = False
    lang, reason = await sess.decide()
    while lang is None and not flush and not sess.timed_out():
        try:
            msg = await client.recv()
        except Exception:
            return None, "", [], []
        if isinstance(msg, (bytes, bytearray)):
            sess.add(bytes(msg))
        else:
            control.append(msg)
            try:
                flush = json.loads(msg).get("type") in ("finish", "stop")
            except Exception:
                flush = False
        lang, reason = await sess.decide(flush=flush)

    collected = sess.raw
    if lang is None:
        upstream, why = pick_upstream(start_msg)
        return upstream, f"lid 실패({reason}) -> {why}", collected, control
    if lang not in ROUTES:
        return DEFAULT_UPSTREAM, f"lid={lang} (표에 없음, {sess.seconds:.1f}s)", collected, control
    return ROUTES[lang], f"lid={lang}/{reason} ({sess.seconds:.1f}s)", collected, control


async def handler(client):
    # 업스트림을 고르려면 start 를 먼저 봐야 하는데, 클라이언트는 연결 직후 hello
    # 를 기다린다. 그래서 hello 만 여기서 먼저 내보내고 위쪽 hello 는 버린다.
    await client.send(json.dumps({"type": "hello", "message": "proxy ready"}))

    pending_binary = []      # start 보다 먼저 온 오디오. 순서를 지켜 뒤에 흘린다.
    start_raw = None
    try:
        async for msg in client:
            if isinstance(msg, (bytes, bytearray)):
                pending_binary.append(bytes(msg))
                continue
            start_raw = msg
            break
    except Exception as e:
        print(f"client gone before start: {e!r}", flush=True)
        return
    if start_raw is None:
        return

    try:
        start_msg = json.loads(start_raw)
    except Exception:
        start_msg = {}

    if LID is not None:
        upstream, why, pending_binary, pending_control = await route_by_voice(
            client, start_msg, pending_binary)
        if upstream is None:
            return                                # 클라이언트가 먼저 끊었다
    else:
        upstream, why = pick_upstream(start_msg)
        pending_control = []
    print(f"route -> {upstream} ({why})", flush=True)

    try:
        async with websockets.connect(upstream, ping_interval=None, max_size=None) as up:
            await up.recv()                       # 위쪽 hello 는 이미 대신 보냈다
            await up.send(start_raw)
            for chunk in pending_binary:
                await up.send(chunk)
            for msg in pending_control:           # 판정을 기다리는 사이 온 finish/stop 등
                await up.send(msg)
            tasks = [asyncio.create_task(relay(client, up)),
                     asyncio.create_task(relay(up, client))]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    except Exception as e:
        print(f"relay ended: {e!r}", flush=True)


def parse_routes(raw):
    """"ko=8766,en=8767" 또는 "ko=ws://host:8766" 을 표로 만든다."""
    out = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        lang, _, target = pair.partition("=")
        lang, target = lang.strip().lower(), target.strip()
        if not lang or not target:
            continue
        out[lang] = target if "://" in target else f"ws://127.0.0.1:{target}"
    return out


async def main():
    print(f"serving {ROOT} on http://0.0.0.0:{PORT}", flush=True)
    for lang, target in ROUTES.items():
        print(f"  {lang} -> {target}", flush=True)
    print(f"  그 밖 -> {DEFAULT_UPSTREAM}", flush=True)
    if LID is not None:
        print(f"  음성 판정: {LID.model_name}, 발화 시작 후 {LID.window_sec}s", flush=True)
    async with serve(handler, "0.0.0.0", PORT, process_request=process_request,
                     ping_interval=None, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port", nargs="?", type=int, default=8080)
    ap.add_argument("--route", default="ko=8766,en=8767",
                    help="언어별 ASR 서버. 예: ko=8766,en=8767")
    ap.add_argument("--default", dest="default_upstream", default="8766",
                    help="라우팅 표에 없을 때 쓸 서버 (포트 또는 ws:// 주소)")
    ap.add_argument("--lid", action="store_true",
                    help="start.lang 대신 음성으로 언어를 판정해 라우팅한다")
    ap.add_argument("--lid-model", default="openai/whisper-base")
    ap.add_argument("--lid-window", type=float, default=1.0,
                    help="발화 시작 후 몇 초를 듣고 판정할지. 실측 정확도 "
                         "1.0s=92.8%%, 2.0s=100%% (whisper-base)")
    ap.add_argument("--lid-max-wait", type=float, default=5.0,
                    help="이만큼 들어도 판정이 안 서면 start.lang 규칙으로 되돌아간다")
    ap.add_argument("--lid-device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    PORT = args.port
    ROUTES = parse_routes(args.route)
    DEFAULT_UPSTREAM = (args.default_upstream if "://" in args.default_upstream
                        else f"ws://127.0.0.1:{args.default_upstream}")

    if args.lid:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from lid_router import LidRouter

        LID = LidRouter(model_name=args.lid_model, window_sec=args.lid_window,
                        max_wait_sec=args.lid_max_wait, device=args.lid_device)

    asyncio.run(main())
