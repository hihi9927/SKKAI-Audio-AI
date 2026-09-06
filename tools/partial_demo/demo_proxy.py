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

**`--dual` 은 두 서버에 동시에 보내고 발화마다 고른다.** `--lid` 는 스트림이
열릴 때 한 번만 판정해서, 화자가 도중에 언어를 바꾸면 모델이 안 따라간다.
`--dual` 은 ko/en 서버에 같은 오디오를 다 보내 놓고, 발화 구간마다 판정을
갱신해 그 구간의 결과를 낸 서버만 통과시킨다.

    python demo_proxy.py 8080 --route ko=8766,en=8767 --dual

**선택은 전사가 아니라 오디오로 한다.** 전사 글자(한글/라틴)로 고르면 깨진다 —
한국어 모델이 영어 발화를 `헬로 나이스 미트 유.` 처럼 한글로 받아쓰면 그 글자는
한국어로 보이므로 두 서버 결과가 모두 통과한다. VAD 로 찾은 발화 구간의 음성을
whisper-base 에 직접 넣어 판정하면 전사 내용과 무관하게 갈린다.

비용은 ASR 연산 2배다. 두 모델은 어차피 GPU 에 함께 올라가 있고, 실측에서 지연의
대부분은 번역이 차지하므로(전체 중앙 0.20초 중 번역만 0.19초) 감당할 만하다.
"""
import argparse
import asyncio
import http
import json
import logging
import mimetypes
import pathlib
import sys
import time

import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = pathlib.Path(__file__).resolve().parent / "web"
ROUTES: dict[str, str] = {}
DUAL = False          # --dual. 모든 서버에 보내고 발화마다 고른다.
REST_UPSTREAM = None  # --rest. 라우팅 표에 없는 언어를 맡는 서버(보통 베이스라인).
REST_KEY = "*"        # 그 서버를 가리키는 이름
TRANSLATE_URL = None  # --translate-url. 번역 방향을 바로잡을 때 쓴다.
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


def _secs(v):
    """서버가 주는 "H:MM:SS.ss" 를 초로 바꾼다. 숫자로 오면 그대로 쓴다."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        total = 0.0
        for part in str(v).split(":"):
            total = total * 60 + float(part)
        return total
    except ValueError:
        return None


async def wait_for_verdict(tracker, end, timeout: float = 1.0, step: float = 0.1):
    """그 구간 판정이 생길 때까지 잠깐 기다린다.

    **기다리지 않으면 전환 지점에서 중첩이 난다.** 판정이 아직 없으면 lang_for_end
    는 직전 판정으로 되돌아가는데, 언어가 막 바뀐 자리에서는 그 추측이 구조적으로
    틀린다. 실제로 영어를 말하다 '안녕하세요' 로 넘어갈 때, 먼저 도착한 영어 서버의
    'Hello,' 가 직전 영어 판정을 타고 통과했고, 잠시 뒤 한국어 판정이 생기면서
    한국어 서버의 '안녕하세요.' 도 통과해 같은 발화가 두 번 나왔다.

    판정은 발화 구간이 닫힌 뒤에 붙으므로 보통 수백 ms 안에 선다. 그때까지만
    붙들었다가, 안 서면 종전처럼 추측으로 넘어간다.
    """
    if end is None or tracker.settled(end):
        return 0.0
    # 진입할 때 한 번 강제로 돌린다. 이미 닫힌 구간이면 여기서 바로 판정이 붙는다.
    # 그 뒤로는 폴링만 한다 — _update_sync 는 silero 를 버퍼 전체에 돌리고 GPU 락을
    # 잡으므로, 50ms 마다 강제로 부르면 그 자체가 지연이 된다.
    t0 = time.perf_counter()
    await tracker.update(force=True)
    while not tracker.settled(end):
        waited = time.perf_counter() - t0
        if waited >= timeout:
            break
        await asyncio.sleep(step)
        if waited > 0.3:                  # 오래 걸리면 한 번 더 밀어 준다
            await tracker.update(force=True)
    return time.perf_counter() - t0


async def fix_direction(data, verdict_lang, lang_map, target_lang):
    """ASR 서버가 언어를 잘못 신고해 뒤집힌 번역을 판정 언어 기준으로 다시 한다.

    **모델은 프록시가 고르는데 번역 방향은 서버가 정한다.** 서버는 자기 전사에
    붙은 `language` 로 소스를 판단하는데, 그 신고는 2초 창 실측에서 영어 파인튜닝
    53% 로 못 믿을 값이다. 게다가 VAD 컷 없이 한 slot 안에서 언어가 바뀌면 slot
    전체가 앞 언어로 디코딩돼, 한국어 문장이 `language English` 를 달고 나온다.
    그러면 소스를 영어로 보고 목표를 한국어로 잡아, 이미 한국어인 문장을 한국어로
    "번역" 해 원문이 그대로 나온다. 실제로 `Oh, 반갑습니다.` 가 `오, 안녕하세요.` 로
    나왔다.

    판정은 오디오를 보고 낸 값이므로 이쪽이 옳다. 신고가 판정과 다를 때만 다시 번역한다.
    """
    declared = (data.get("language") or "").lower()
    if not TRANSLATE_URL or not verdict_lang or declared == verdict_lang:
        return data, False
    text = (data.get("original") or "").strip()
    if not text:
        return data, False
    target = (lang_map or {}).get(verdict_lang) or target_lang
    if not target or target == verdict_lang:
        return data, False
    try:
        import aiohttp

        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)) as sess:
            async with sess.post(f"{TRANSLATE_URL.rstrip('/')}/translate",
                                 json={"text": text, "target": target,
                                       "source": verdict_lang}) as resp:
                resp.raise_for_status()
                body = await resp.json()
    except Exception as e:
        print(f"retranslate failed: {e!r}", flush=True)
        return data, False
    fixed = body.get("translation") or ""
    if not fixed:
        return data, False
    data = dict(data)
    data["language"] = verdict_lang
    data["translation"] = fixed
    return data, True


async def run_dual(client, start_raw, pending_binary):
    """라우팅 표의 모든 서버에 같은 오디오를 보내고, 발화 구간마다 한쪽만 통과시킨다.

    통과 기준은 전사가 아니라 **오디오 판정**이다. VerdictTracker 가 VAD 로 찾은
    발화 구간의 음성을 whisper-base 에 넣어 언어를 정하고, 그 구간의 결과를 낸
    서버가 그 언어를 맡은 서버일 때만 클라이언트로 넘긴다.

    핸드셰이크 응답(ready 등)은 한 서버 것만 넘긴다. 두 벌이 가면 클라이언트가
    같은 신호를 두 번 본다.
    """
    from lid_router import VerdictTracker

    # 붙을 서버들. 언어 코드로 된 것들에 더해, --rest 가 있으면 "*" 로 하나 더.
    servers = dict(ROUTES)
    if REST_UPSTREAM:
        servers[REST_KEY] = REST_UPSTREAM
    langs = list(servers)
    primary = langs[0]
    # 판정이 아무 데도 안 맞을 때. rest 가 있으면 그쪽, 없으면 기본 서버의 언어.
    fallback = REST_KEY if REST_UPSTREAM else next(
        (l for l in ROUTES if ROUTES[l] == DEFAULT_UPSTREAM), primary)

    def route_of(verdict):
        """판정 언어를 붙을 서버 이름으로 바꾼다."""
        if verdict is None:
            return None
        if verdict in ROUTES:
            return verdict
        return fallback
    try:
        _start = json.loads(start_raw)
    except Exception:
        _start = {}
    lang_map = _start.get("langMap") if isinstance(_start.get("langMap"), dict) else {}
    target_lang = _start.get("targetLang") or ""
    # 웹이 고른 소스 언어로 LID 후보를 좁힌다. langMap 이 없으면 start.lang 을 쓰고,
    # 그것도 auto 면 라우팅 표 전체를 후보로 둔다.
    allowed = set(lang_map) or {c for c in [_start.get("lang")] if c and c != "auto"}
    allowed |= set(ROUTES)          # 서버가 있는 언어는 언제나 후보에 남긴다
    tracker = VerdictTracker(LID, allowed=allowed)
    for chunk in pending_binary:
        tracker.feed(chunk)

    ups = {}
    try:
        for lang in langs:
            ups[lang] = await websockets.connect(servers[lang], ping_interval=None,
                                                 max_size=None)
            await ups[lang].recv()             # 위쪽 hello 는 프록시가 이미 보냈다
            await ups[lang].send(start_raw)
            for chunk in pending_binary:
                await ups[lang].send(chunk)
        print(f"dual -> {', '.join(f'{k}:{servers[k]}' for k in langs)} "
              f"| LID 후보 {sorted(allowed)}", flush=True)

        async def pump_client():
            async for msg in client:
                if isinstance(msg, (bytes, bytearray)):
                    tracker.feed(bytes(msg))
                    await tracker.update()
                for up in ups.values():
                    await up.send(msg)

        async def pump_upstream(lang):
            # 이 서버가 마지막으로 확정한 오디오 지점. 다음 final 의 시작점으로 쓴다 —
            # 서버가 final.start 를 안 채우기 때문이다.
            group_start = 0.0
            async for msg in ups[lang].__aiter__():
                if not isinstance(msg, str):
                    continue                   # 위쪽이 이진을 보낼 일은 없다
                try:
                    data = json.loads(msg)
                except Exception:
                    if lang == primary:
                        await client.send(msg)
                    continue
                kind = data.get("type")
                if kind not in ("final", "partial"):
                    if lang == primary:        # ready 등은 한 벌만
                        await client.send(msg)
                    continue
                # final 은 자기 오디오 구간으로, partial 은 지금 시각으로 본다.
                # final 은 커밋 경계로 그 발화의 판정을 찾는다. start 는 이 서버가
                # 안 채워서(항상 0.0) 쓸 수 없다.
                b = _secs(data.get("end")) if kind == "final" else None
                if kind == "final":
                    waited = await wait_for_verdict(tracker, b)
                    span = (group_start, b)
                    # 판정 언어를 서버 이름으로 바꾼다. ko/en 밖의 언어는 --rest
                    # (베이스라인) 로 간다. 그대로 두면 어느 서버도 안 맞아 발화가
                    # 통째로 사라진다.
                    pick = route_of(tracker.lang_for_range(*span))
                    if waited > 0.05:
                        print(f"wait {waited:.2f}s for end={b} ({lang})", flush=True)
                    if b is not None and b > group_start:
                        # 같은 경계에서 여러 final 이 나오면 같은 구간을 공유한다.
                        group_start = b
                else:
                    pick = route_of(tracker.lang_at(None))
                if pick == lang:
                    if kind == "final":
                        verdict_lang = tracker.lang_for_range(*span)
                        data, fixed = await fix_direction(
                            data, verdict_lang, lang_map, target_lang)
                        if fixed:
                            print(f"fix dir end={b} {lang}: "
                                  f"{(data.get('original') or '')[:30]!r} -> "
                                  f"{(data.get('translation') or '')[:30]!r}", flush=True)
                            msg = json.dumps(data, ensure_ascii=False)
                    await client.send(msg)
                elif kind == "final":
                    print(f"drop {lang} [{span[0]:.1f}~{span[1]}] -> {pick}: "
                          f"{(data.get('original') or '')[:40]!r}", flush=True)

        tasks = [asyncio.create_task(pump_client())]
        tasks += [asyncio.create_task(pump_upstream(l)) for l in langs]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            if t.exception() and not isinstance(t.exception(), asyncio.CancelledError):
                print(f"dual pump ended: {t.exception()!r}", flush=True)
    except Exception as e:
        print(f"dual relay ended: {e!r}", flush=True)
    finally:
        if tracker.verdicts:
            spans = ", ".join(f"{v[0]:.1f}-{'' if v[1] is None else f'{v[1]:.1f}'}"
                              f":{v[2]}{'*' if len(v) > 3 and v[3] else ''}"
                              f"{'!' if len(v) > 4 and v[4] else ''}"
                              for v in tracker.verdicts)
            print(f"dual verdicts: {spans}", flush=True)
        for up in ups.values():
            try:
                await up.close()
            except Exception:
                pass


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

    if DUAL:
        await run_dual(client, start_raw, pending_binary)
        return

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
    if REST_UPSTREAM:
        print(f"  그 밖 언어 -> {REST_UPSTREAM} (베이스라인)", flush=True)
    print(f"  판정 실패 -> {DEFAULT_UPSTREAM}", flush=True)
    if LID is not None:
        mode = "양쪽 동시 전송 후 발화별 선택" if DUAL else "스트림당 1회 라우팅"
        print(f"  음성 판정: {LID.model_name}, 창 {LID.window_sec}s — {mode}", flush=True)
    if TRANSLATE_URL:
        print(f"  번역 방향 교정: {TRANSLATE_URL}", flush=True)
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
    ap.add_argument("--rest", default=None,
                    help="라우팅 표에 없는 언어를 맡을 서버 (포트 또는 ws:// 주소). "
                         "파인튜닝 모델은 자기 언어 밖에서 언어 태그부터 틀리므로, "
                         "스페인어 같은 제3언어는 베이스라인으로 보낸다")
    ap.add_argument("--lid", action="store_true",
                    help="start.lang 대신 음성으로 언어를 판정해 라우팅한다")
    ap.add_argument("--dual", action="store_true",
                    help="라우팅 표의 모든 서버에 동시에 보내고 발화마다 고른다. "
                         "--lid 를 함께 켠 것으로 친다")
    ap.add_argument("--lid-model", default="openai/whisper-base")
    ap.add_argument("--lid-window", type=float, default=1.0,
                    help="발화 시작 후 몇 초를 듣고 판정할지. 실측 정확도 "
                         "1.0s=92.8%%, 2.0s=100%% (whisper-base)")
    ap.add_argument("--lid-max-wait", type=float, default=5.0,
                    help="이만큼 들어도 판정이 안 서면 start.lang 규칙으로 되돌아간다")
    ap.add_argument("--lid-device", default="cuda")
    ap.add_argument("--translate-url", default=None,
                    help="단독 번역 서버 주소(예: http://127.0.0.1:8770). 주면 ASR "
                         "서버가 언어를 잘못 신고해 번역 방향이 뒤집힌 final 을 "
                         "판정 언어 기준으로 다시 번역한다")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    PORT = args.port
    ROUTES = parse_routes(args.route)
    DEFAULT_UPSTREAM = (args.default_upstream if "://" in args.default_upstream
                        else f"ws://127.0.0.1:{args.default_upstream}")
    if args.rest:
        REST_UPSTREAM = (args.rest if "://" in args.rest
                         else f"ws://127.0.0.1:{args.rest}")

    DUAL = args.dual
    TRANSLATE_URL = args.translate_url
    if args.lid or args.dual:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from lid_router import LidRouter

        LID = LidRouter(model_name=args.lid_model, window_sec=args.lid_window,
                        max_wait_sec=args.lid_max_wait, device=args.lid_device,
                        known_langs=ROUTES.keys())

    asyncio.run(main())
