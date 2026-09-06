"""시연 페이지(show.html) 리허설용 목 서버. ASR·GPU 없이 화면만 확인한다.

    python tools/partial_demo/mock_show_server.py [포트] [--loop]

정적 파일은 web/ 에서 그대로 내주고, /ws 로 붙으면 아래 대본을 실제 발화 속도에
가깝게 흘려 보낸다. 페이지에서 "테스트 음성" 을 누르면 시작된다(무음 wav 를
자동으로 내주므로 partial_test.wav 가 없어도 된다).

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

# 44바이트 헤더만 있는 무음 wav. 페이지의 "테스트 음성" 경로를 열어 주기 위한 것이라
# 내용은 필요 없다 — 자막은 아래 대본이 WebSocket 으로 직접 밀어 넣는다.
SILENT_WAV = (
    b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
    + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    + b"data" + (0).to_bytes(4, "little")
)

# (보낼 때까지 기다릴 초, 메시지). partial 이 조금씩 자라다가 final 로 확정되는
# 실제 스트림 모양을 흉내낸다.
SCRIPT = [
    (0.8, {"type": "partial", "text": "안녕하세요", "language": "Korean"}),
    (0.7, {"type": "partial", "text": "안녕하세요 오늘 발표를", "language": "Korean"}),
    (0.7, {"type": "partial", "text": "안녕하세요 오늘 발표를 맡은", "language": "Korean"}),
    (0.9, {"type": "final", "original": "안녕하세요, 오늘 발표를 맡은 김하나입니다.",
           "translation": "Hello, I'm Hana Kim, presenting today.",
           "language": "Korean", "commitReason": "seg"}),
    (1.1, {"type": "partial", "text": "thank you for", "language": "English"}),
    (0.8, {"type": "partial", "text": "thank you for having me here", "language": "English"}),
    (0.9, {"type": "final", "original": "Thank you for having me here today.",
           "translation": "오늘 이 자리에 초대해 주셔서 감사합니다.",
           "language": "English", "commitReason": "seg"}),
    (1.2, {"type": "partial", "text": "この", "language": "Japanese"}),
    (0.8, {"type": "partial", "text": "このシステムは", "language": "Japanese"}),
    (0.9, {"type": "final", "original": "このシステムはリアルタイムで翻訳します。",
           "translation": "이 시스템은 실시간으로 번역합니다.",
           "language": "Japanese", "commitReason": "vad"}),
    # 여기부터는 빠르게 몰아친다 — 유지 시간과 대기열이 도는지 보기 위한 구간이다.
    (0.3, {"type": "final", "original": "질문 있으시면 언제든지 말씀해 주세요.",
           "translation": "Please ask any questions at any time.",
           "language": "Korean", "commitReason": "dot"}),
    (0.2, {"type": "final", "original": "We support ten languages.",
           "translation": "저희는 열 개 언어를 지원합니다.",
           "language": "English", "commitReason": "dot"}),
    (0.2, {"type": "final", "original": "ありがとうございました。",
           "translation": "감사합니다.",
           "language": "Japanese", "commitReason": "seg"}),
]


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
