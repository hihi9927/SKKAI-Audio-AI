"""시연 페이지(show.html) 리허설용 목 서버. ASR·GPU 없이 화면만 확인한다.

    python STiTy-Mobile/demo-web/partial_demo/mock_show_server.py [포트] [--loop]

정적 파일은 web/ 에서 그대로 내주고, /ws 로 붙으면 web/mock_script.json 의 대본을
실제 발화 속도에 가깝게 흘려 보낸다. 페이지에서 Ctrl+Enter 를 누르면 시작된다(무음 wav 를
자동으로 내주므로 partial_test.wav 가 없어도 된다).

show.html 은 같은 대본을 직접 읽어 서버 없이도 되감는다. 이 서버가 필요한 경우는
WebSocket 을 타는 경로까지 같이 볼 때다.

--loop 를 주면 대본을 무한 반복한다. 시연장에 띄워 두고 배치를 바꿔 가며 볼 때 쓴다.
"""
import asyncio
import http
import json
import mimetypes
import pathlib
import sys

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = pathlib.Path(__file__).resolve().parent / "web"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PORT = int(args[0]) if args else 8090
LOOP = "--loop" in sys.argv

# 대본이 다 흐를 때까지 페이지가 연결을 붙들고 있어야 한다. 페이지는 wav 를 다 보내면
# finish 를 보내고 8초 뒤 연결을 닫으므로, 무음 wav 가 대본보다 짧으면 뒷부분이 잘린다.
# 그래서 길이를 대본에서 계산해 만든다(무음이라 내용은 필요 없다 — 자막은 아래 대본이
# WebSocket 으로 직접 밀어 넣는다).
def silent_wav(seconds):
    n = int(16000 * seconds) * 2
    return (
        b"RIFF" + (36 + n).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
        + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        + b"data" + n.to_bytes(4, "little") + b"\x00" * n
    )


# 대본은 web/mock_script.json 에 있다. show.html 도 같은 파일을 읽어 서버 없이 되감으므로
# (Ctrl+Enter 리허설), 대본을 고칠 때 한 군데만 고치면 양쪽이 같이 바뀐다.
SCRIPT = [(float(e["wait"]), e["msg"]) for e in json.loads((ROOT / "mock_script.json").read_text("utf-8"))]

# 대본 한 바퀴에 조금 여유를 둔 길이. --loop 로 두 바퀴 이상 보려면 페이지를 다시 연다.
SILENT_WAV = silent_wav(sum(w for w, _ in SCRIPT) + 10)


def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    path = request.path.split("?")[0]
    if path in ("", "/"):
        path = "/show.html"
    name = path.lstrip("/")
    if name.startswith("partial_test") and name.endswith(".wav") and not (ROOT / name).is_file():
        return Response(200, "OK", Headers({"Content-Type": "audio/wav",
                                            "Content-Length": str(len(SILENT_WAV))}), SILENT_WAV)
    target = (ROOT / path.lstrip("/")).resolve()
    if not str(target).startswith(str(ROOT)) or not target.is_file():
        return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")
    body = target.read_bytes()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return Response(200, "OK", Headers({"Content-Type": ctype,
                                        "Content-Length": str(len(body)),
                                        "Cache-Control": "no-store"}), body)


async def handler(ws):
    await ws.send(json.dumps({"type": "hello", "message": "mock"}))
    await ws.send(json.dumps({"type": "ready", "message": "mock"}))
    print("client connected", flush=True)
    try:
        while True:
            for wait, msg in SCRIPT:
                await asyncio.sleep(wait)
                await ws.send(json.dumps(msg))
            if not LOOP:
                break
            await asyncio.sleep(2.5)
        async for _ in ws:
            pass
    except Exception as e:
        print(f"client gone: {e!r}", flush=True)


async def main():
    print(f"mock show server: http://0.0.0.0:{PORT}/show.html  (loop={LOOP})", flush=True)
    async with serve(handler, "0.0.0.0", PORT, process_request=process_request,
                     ping_interval=None, max_size=None):
        await asyncio.Future()


asyncio.run(main())
