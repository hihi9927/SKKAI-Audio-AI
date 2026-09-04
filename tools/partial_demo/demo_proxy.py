"""정적 페이지와 WebSocket 을 한 포트로 묶는다.

VS Code 원격 개발에서는 포트를 하나씩 포워딩해야 해서, 페이지(8080)와
ASR 서버(8766)를 따로 열면 두 번 포워딩해야 한다. 이 프록시는 8080 하나만
열고 /ws 요청을 127.0.0.1:8766 으로 중계한다.
"""
import asyncio
import http
import mimetypes
import pathlib
import sys

import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = pathlib.Path(__file__).resolve().parent / "web"
UPSTREAM = "ws://127.0.0.1:8766"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


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


async def handler(client):
    try:
        async with websockets.connect(UPSTREAM, ping_interval=None, max_size=None) as up:
            tasks = [asyncio.create_task(relay(client, up)),
                     asyncio.create_task(relay(up, client))]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
    except Exception as e:
        print(f"relay ended: {e!r}", flush=True)


async def main():
    print(f"serving {ROOT} and ws->{UPSTREAM} on http://0.0.0.0:{PORT}", flush=True)
    async with serve(handler, "0.0.0.0", PORT, process_request=process_request,
                     ping_interval=None, max_size=None):
        await asyncio.Future()


asyncio.run(main())
