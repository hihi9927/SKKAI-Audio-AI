"""시연 페이지(show.html) 리허설용 목 서버. ASR·GPU 없이 화면만 확인한다.

    python STiTy-Mobile/demo-web/partial_demo/mock_show_server.py [포트] [--loop]

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


# 목표 언어가 여럿인 화면(3·4·5 배치)을 보려면 final 이 목표별 번역을 다 실어야 한다.
# 서버는 아직 목표 하나만 보내므로(streaming_websocket_server.py 의 _correct_and_translate)
# 여기서 미리 그 모양을 흉내낸다 — 화면을 먼저 정하고 서버를 맞추는 순서다.
#
# translations 는 {목표 코드: 번역}, translation 은 그중 대표 하나다. 옛 화면과 앱은
# translation 만 읽으므로 둘 다 넣는다.
def final(original, language, translations, reason="seg"):
    first = next(iter(translations.values()), "")
    return {"type": "final", "original": original, "translations": translations,
            "translation": first, "language": language, "commitReason": reason}


def partial(text, language):
    return {"type": "partial", "text": text, "language": language}


# (보낼 때까지 기다릴 초, 메시지). partial 이 조금씩 자라다가 final 로 확정되는
# 실제 스트림 모양을 흉내낸다. 말하는 언어는 ko·en·ja 셋이고 청중도 그 셋이라,
# 발화마다 나머지 두 언어로 번역이 나간다.
SCRIPT = [
    (0.8, partial("안녕하세요", "Korean")),
    (0.7, partial("안녕하세요 오늘 발표를", "Korean")),
    (0.7, partial("안녕하세요 오늘 발표를 맡은", "Korean")),
    (0.9, final("안녕하세요, 오늘 발표를 맡은 김하나입니다.", "Korean", {
        "en": "Hello, I'm Hana Kim, presenting today.",
        "ja": "こんにちは、本日の発表を担当するキム・ハナです。",
    })),
    (1.0, partial("저희가 만든 것은", "Korean")),
    (0.9, partial("저희가 만든 것은 실시간 음성", "Korean")),
    (0.9, final("저희가 만든 것은 실시간 음성 번역 시스템입니다.", "Korean", {
        "en": "What we built is a real-time speech translation system.",
        "ja": "私たちが作ったのはリアルタイム音声翻訳システムです。",
    })),
    (1.1, partial("thank you for", "English")),
    (0.8, partial("thank you for having me here", "English")),
    (0.9, final("Thank you for having me here today.", "English", {
        "ko": "오늘 이 자리에 초대해 주셔서 감사합니다.",
        "ja": "本日はお招きいただきありがとうございます。",
    })),
    (1.0, partial("could you explain", "English")),
    (0.9, partial("could you explain how the delay", "English")),
    (0.9, final("Could you explain how the delay is measured?", "English", {
        "ko": "지연 시간을 어떻게 재는지 설명해 주시겠어요?",
        "ja": "遅延はどのように測定しているのか説明していただけますか。",
    }, reason="dot")),
    (1.0, partial("말이 끝난 순간부터", "Korean")),
    (0.9, final("말이 끝난 순간부터 자막이 뜰 때까지를 잽니다.", "Korean", {
        "en": "We measure from the moment the speech ends until the subtitle appears.",
        "ja": "話し終わった瞬間から字幕が出るまでを測ります。",
    })),
    (1.2, partial("この", "Japanese")),
    (0.8, partial("このシステムは", "Japanese")),
    (0.9, final("このシステムはリアルタイムで翻訳します。", "Japanese", {
        "ko": "이 시스템은 실시간으로 번역합니다.",
        "en": "This system translates in real time.",
    }, reason="vad")),
    (1.0, partial("会場の", "Japanese")),
    (0.9, final("会場の全員が同じ画面を見ています。", "Japanese", {
        "ko": "행사장에 있는 모두가 같은 화면을 보고 있습니다.",
        "en": "Everyone in the room is looking at the same screen.",
    })),
    (1.0, partial("화면은 언어마다", "Korean")),
    (0.9, final("화면은 언어마다 칸을 나눠서 보여 줍니다.", "Korean", {
        "en": "The screen splits into one lane per language.",
        "ja": "画面は言語ごとに欄を分けて表示します。",
    })),
    (1.0, partial("so each guest reads", "English")),
    (0.9, final("So each guest only reads their own lane.", "English", {
        "ko": "그래서 참석자는 자기 칸만 읽으면 됩니다.",
        "ja": "ですから参加者は自分の欄だけを読めば済みます。",
    })),
    (1.1, partial("翻訳はどのくらい", "Japanese")),
    (0.9, final("翻訳はどのくらい遅れますか。", "Japanese", {
        "ko": "번역은 얼마나 늦게 나오나요?",
        "en": "How much does the translation lag?",
    }, reason="dot")),
    (1.0, partial("문장이 끝나면", "Korean")),
    (0.9, final("문장이 끝나면 대개 1초 안에 올라옵니다.", "Korean", {
        "en": "Once a sentence ends, it usually shows up within a second.",
        "ja": "文が終われば、たいてい一秒以内に出ます。",
    })),
    (1.0, partial("한 사람이 여러 언어를", "Korean")),
    (0.9, final("한 사람이 여러 언어를 섞어 말해도 그대로 따라갑니다.", "Korean", {
        "en": "It keeps up even when one speaker mixes several languages.",
        "ja": "一人が複数の言語を混ぜて話しても、そのままついていきます。",
    })),
    (1.1, partial("that is useful for", "English")),
    (0.9, final("That is useful for a panel discussion.", "English", {
        "ko": "그건 패널 토론에 쓸모가 있습니다.",
        "ja": "それはパネルディスカッションで役に立ちます。",
    })),
    # 여기부터는 빠르게 몰아친다 — 유지 시간과 대기열이 도는지 보기 위한 구간이다.
    (0.3, final("질문 있으시면 언제든지 말씀해 주세요.", "Korean", {
        "en": "Please ask any questions at any time.",
        "ja": "ご質問があればいつでもおっしゃってください。",
    }, reason="dot")),
    (0.2, final("We support ten languages.", "English", {
        "ko": "저희는 열 개 언어를 지원합니다.",
        "ja": "私たちは十の言語に対応しています。",
    }, reason="dot")),
    (0.2, final("最後までご覧いただきありがとうございました。", "Japanese", {
        "ko": "끝까지 봐 주셔서 감사합니다.",
        "en": "Thank you for watching to the end.",
    })),
]

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
