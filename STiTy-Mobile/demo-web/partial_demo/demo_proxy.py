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
시작을 잡은 뒤 whisper-small 로 언어를 보고 그 결과로 고른다. 한 마이크에 두
언어가 섞여 들어오는 화면에서도 맞는 모델로 간다. 근거와 실측값은
[lid_router.py](lid_router.py) 문서를 보라.

    python demo_proxy.py 8080 --route ko=8766,en=8767 --lid --lid-window 1.0

**`--dual` 은 모든 서버에 붙어 두고, 발화 구간마다 그 언어의 서버에만 오디오를
보낸다.** `--lid` 는 스트림이 열릴 때 한 번만 판정해서, 화자가 도중에 언어를
바꾸면 모델이 안 따라간다. `--dual` 은 발화 구간마다 판정을 갱신하고, 그 구간의
오디오를 담당 서버에만 흘린다. 나머지 서버는 같은 길이의 무음을 받아 시간축만
맞춘다. 각 서버는 자기 언어만 들으므로 결과를 고를 일이 없다 — 왜 출력을 고르지
않고 입력을 가르는지는 `Dispatcher` 문서에 있다.

    python demo_proxy.py 8080 --route ko=8766,en=8767 --rest 8768 --dual

**판정은 전사가 아니라 오디오로 한다.** 전사 글자(한글/라틴)로 고르면 깨진다 —
한국어 모델이 영어 발화를 `헬로 나이스 미트 유.` 처럼 한글로 받아쓰면 그 글자는
한국어로 보인다. VAD 로 찾은 발화 구간의 음성을 whisper-small 에 직접 넣어
판정하면 전사 내용과 무관하게 갈린다.
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
from typing import Optional

import numpy as np
import websockets
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

SR = 16000
ROOT = pathlib.Path(__file__).resolve().parent / "web"
ROUTES: dict[str, str] = {}
DUAL = False          # --dual. 모든 서버에 보내고 발화마다 고른다.
REST_UPSTREAM = None  # --rest. 라우팅 표에 없는 언어를 맡는 서버(보통 베이스라인).
REST_KEY = "*"        # 그 서버를 가리키는 이름
TRANSLATE_URL = None  # --translate-url. 번역 방향을 바로잡을 때 쓴다.
AUDIENCE: list[str] = []   # --targets. 발화 하나를 이 언어들로 모두 번역한다.
LANG_HINT = False     # --lang-hint. 판정 언어를 서버에 알린다.
LANG_HINT_FORCE = False  # --lang-hint-force. 바이어스 대신 force_language 로.
# 슬롯을 자르는 지점을 판정 시작보다 이만큼 앞으로 당긴다. 전환 지점 추정은
# 실측 오차 중앙 0.29초·p90 0.62초라 정확히 그 자리를 자르면 새 언어의 첫 음절이
# 날아간다 — '재입국' 이 '입국' 으로, 'Esto' 가 'esto' 로 나왔다. 앞 언어가 조금
# 새는 쪽이 낫다. 바이어스 힌트는 그 앞부분을 대체로 흡수한다.
HINT_BACKOFF_SEC = 0.3
LID_SCAN_WIN = 0.0    # --lid-scan. 0 이면 구간 앞머리에서 한 번만 판정한다.
LID_SCAN_HOP = 0.25
LID_SCAN_CONFIRM = 2
LID_EARLY = False     # --lid-early. 창을 다 듣기 전에 확신도로 조기 확정한다.
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


async def _translate_to(text: str, target: str, source: str) -> str:
    """단독 번역 서버에 한 줄 번역을 요청한다. 실패는 빈 문자열이다."""
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
        async with sess.post(f"{TRANSLATE_URL.rstrip('/')}/translate",
                             json={"text": text, "target": target, "source": source}) as resp:
            resp.raise_for_status()
            return (await resp.json()).get("translation") or ""


async def add_audience_translations(data, src_lang: str, primary_target: str):
    """청중 언어마다 번역을 붙여 ``translations`` 로 실어 보낸다.

    **서버는 발화 하나당 목표 하나만 낸다.** `langMap` 이 소스 → 목표 1:1 이라
    한국어를 말하면 영어만 나오고, 같이 켜 둔 스페인어 자막은 만들어지지 않는다.
    화면(`web/show.html`)은 이미 목표별 딕셔너리 `translations` 를 읽으므로
    (배치 3·4 가 그걸 기다린다), 모자란 목표를 여기서 채운다.

    서버가 이미 낸 번역은 다시 부르지 않고 그 목표 자리에 그대로 넣는다. 나머지는
    한꺼번에 병렬로 부르므로, 언어가 몇 개든 지연은 한 번 분량이다. 옛 화면과 앱을
    위해 ``translation`` 도 그대로 남긴다.
    """
    if not TRANSLATE_URL or not AUDIENCE:
        return data
    text = (data.get("original") or "").strip()
    if not text:
        return data
    out = {}
    if primary_target and data.get("translation"):
        out[primary_target] = data["translation"]
    todo = [t for t in AUDIENCE if t != src_lang and t not in out]
    if todo:
        got = await asyncio.gather(*(_translate_to(text, t, src_lang) for t in todo),
                                   return_exceptions=True)
        for tgt, res in zip(todo, got):
            if isinstance(res, Exception):
                print(f"fanout {src_lang}->{tgt} failed: {res!r}", flush=True)
            elif res:
                out[tgt] = res
    if not out:
        return data
    data = dict(data)
    data["translations"] = out
    if not data.get("translation"):
        data["translation"] = next(iter(out.values()), "")
    return data


# 오디오를 담당 서버에만 보내기 위한 상수. 값의 근거는 Dispatcher 문서를 보라.
DISPATCH_DELAY = 0.6   # 무음 구간에서 이만큼 뒤에서 보낸다. 발화 시작·판정을 먼저 알기 위한 여유
LIVE_DELAY = 0.1       # 담당이 정해진 발화 안에서는 이만큼만 뒤에서 보낸다
PRE_ROLL = 0.25        # 발화 구간 앞에 붙여 보내는 실제 오디오
POST_ROLL = 0.25       # 발화 구간 뒤에 붙여 보내는 실제 오디오
SILENCE_LOOKBACK = 8.0 # final 의 end 앞 이만큼 안에 이 서버로 보낸 음성이 없으면 환각으로 본다
MAX_HOLD_AFTER_WINDOW = 1.0   # 판정 창을 채우고도 이만큼 지나면 붙들기를 그만둔다
PIECE = 0.1            # 보내는 조각 길이
SPLIT_BACKOFF = 0.3    # 발화 한복판 전환을 새 담당에게 되돌려 줄 때 앞으로 당기는 폭


class Dispatcher:
    """판정에 따라 오디오를 **담당 서버에만** 보낸다. 나머지 서버에는 같은 길이의 무음.

    **왜 출력을 고르지 않고 입력을 가르나.** 종전에는 세 서버에 같은 오디오를 다 주고
    final 이 오면 그 구간의 판정으로 한쪽만 통과시켰다. 그러려면 final 이 어느 오디오
    구간에서 나왔는지 알아야 하는데 서버는 커밋 시각(end)만 준다. 커밋이 늦거나 여러
    문장이 한꺼번에 나오면 엉뚱한 판정에 걸려, 맞는 문장이 버려지고(143초 세션에서
    발화 40개 중 11개 소실) 다른 서버가 낸 엉뚱한 언어의 문장이 대신 나갔다. 게다가
    한국어 서버는 스페인어를 듣고 `메`·`홀라` 같은 한글 조각을 계속 냈고, 영어 서버는
    한 슬롯에 언어가 섞이면 두 번째 헤더를 찍고 그 뒤를 통째로 잃었다.

    서버가 자기 언어만 들으면 이 문제들이 생길 자리가 없다. 남는 문제는 "어느 오디오를
    누구에게" 인데, 그건 판정 시각의 문제라 시간축 위에서 풀 수 있다.

    **시간축.** 모든 서버는 같은 수의 샘플을 받는다(담당이 아니면 0). 그래서 서버가
    주는 `end` 는 프록시의 오디오 시각과 같고, 어느 서버가 실제 음성을 받은 구간을
    프록시가 정확히 알고 있으므로 무음 환각을 여기서 다시 거를 수 있다.

    **왜 DISPATCH_DELAY 만큼 늦게 보내나.** 발화 시작은 silero 가 잡은 뒤에야 알고
    (판정 갱신 주기 0.25초 포함 시작 후 약 0.35초), 그 앞 PRE_ROLL 을 실제 오디오로
    보내려면 그 시각을 아직 안 보냈어야 한다. 0.6초면 둘 다 만족한다. 판정 자체는
    발화 시작 후 0.3~1.0초(중앙 0.7초)에 서므로, 그때까지는 커서를 세워 두었다가
    (hold) 판정이 서면 한꺼번에 흘린다. ASR 은 실시간보다 빨리 따라잡는다.

    **발화 한복판 전환(`--lid-scan`)** 은 전환 후 약 1초 뒤에 확인되므로 그 사이 오디오는
    이미 옛 담당에게 갔다. 새 담당에게는 전환 지점부터 지금까지를 따로 보내 준다
    (SPLIT_BACKOFF 만큼 앞에서부터 — 지점 추정 오차 p90 0.62초라 첫 음절을 잃는 쪽이
    더 나쁘다). 옛 담당에게 이미 간 1초는 되돌릴 수 없어 그쪽 문장 끝에 낯선 말이
    한 조각 붙을 수 있다. 이 추가 전송만큼 그 서버의 시간축이 앞서므로 `extra` 로
    적어 두고 `end` 를 읽을 때 뺀다.
    """

    def __init__(self, tracker, names, route_of, fallback):
        self.tracker = tracker
        self.names = list(names)
        self.route_of = route_of
        self.fallback = fallback
        self._buf = np.zeros(0, dtype=np.int16)
        self._offset = 0.0          # _buf[0] 의 프록시 시각
        self.keep_sec = 30.0
        self.audio_sec = 0.0        # 받은 오디오 길이
        self.cursor = 0.0           # 여기까지 보냈다
        self.spans = {n: [] for n in self.names}   # [시작, 끝, 언어] — 실제 음성을 보낸 구간
        self.extra = {n: 0.0 for n in self.names}  # 공통 시간축 밖으로 더 보낸 길이
        self.last_owner = None
        self.last_lang = None
        self._last_speech_end = -1e9
        self.hold_since = None
        self._seen_splits = set()
        self._locks: list = []      # [시작, 끝, 언어] — 구간마다 한 번 정한 담당
        self._checked: set = set()  # 확정 판정과 대조를 끝낸 잠금 (id)
        self.revoked = {n: [] for n in self.names}   # 교정으로 무효가 된 (시작, 끝)

    def feed(self, pcm: bytes) -> None:
        x = np.frombuffer(pcm, dtype="<i2")
        self._buf = np.concatenate([self._buf, x])
        self.audio_sec += len(x) / SR
        keep = int(SR * self.keep_sec)
        if len(self._buf) > keep:
            drop = len(self._buf) - keep
            self._offset += drop / SR
            self._buf = self._buf[drop:]

    def _slice(self, a: float, b: float) -> bytes:
        i = max(0, int(round((a - self._offset) * SR)))
        j = max(i, min(int(round((b - self._offset) * SR)), len(self._buf)))
        return self._buf[i:j].tobytes()

    def _segment_at(self, t: float):
        """t 가 (앞뒤 여유를 포함한) 어느 발화 구간에 드는지."""
        for s, e, closed in self.tracker._last_segments:
            e_eff = e if closed else self.tracker.audio_sec
            if s - PRE_ROLL <= t < e_eff + POST_ROLL:
                return s, e_eff, closed
        return None

    def _note(self, name: str, a: float, b: float, lang) -> None:
        spans = self.spans[name]
        if spans and spans[-1][2] == lang and spans[-1][1] >= a - 1e-3:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b, lang])

    def _lock_for(self, s: float, e: float):
        """이 발화 구간에 이미 정해 둔 담당 언어.

        **시작점으로 찾는다, 겹침이 아니라.** 열린 구간의 끝은 "지금까지 받은 오디오"
        라서, 겹침으로 찾으면서 끝을 늘려 두면 다음 발화가 앞 발화의 잠금을 물려받는다.
        실측에서 그렇게 `How long have you studied Spanish?` 가 앞 한국어 발화의 ko 를
        물려받아 한국어 서버로 갔다. silero 가 경계를 조금씩 다르게 잡는 폭은 0.1초
        안쪽이라 시작점 0.4초 허용이면 같은 구간으로 모인다.
        """
        for lock in self._locks:
            if abs(lock[0] - s) < 0.4:
                lock[1] = max(lock[1], e if e < self.tracker.audio_sec else lock[1])
                return lock
        return None

    def _decide(self, s: float, e: float, t: float, lang):
        """구간의 담당 언어를 한 번 정하면 그 구간이 끝날 때까지 지킨다.

        **보낸 오디오는 되돌릴 수 없다.** 확정 판정(CONFIRM_SEC, 구간이 닫힌 뒤 2초를
        다시 듣는 것)이 조기 판정을 뒤집어도 앞부분은 이미 옛 담당에게 갔다. 그때
        뒷부분만 새 담당에게 보내면 한 문장이 두 서버에 반씩 갈려 둘 다 조각을 낸다.
        틀린 서버가 한 문장을 통째로 받는 쪽이 낫다. 예외는 `--lid-scan` 이 발화
        한복판에서 잡은 전환뿐이다 — 그건 실제로 언어가 바뀐 자리다.
        """
        lock = self._lock_for(s, e)
        if lock is None:
            lock = [s, e, lang]
            self._locks.append(lock)
            if len(self._locks) > 50:
                del self._locks[0]
            print(f"lock {s:.2f}s -> {lang} (heard {self.audio_sec - s:.2f}s)", flush=True)
        lang = lock[2]
        splits = [v for v in self.tracker.verdicts
                  if len(v) > 5 and v[5] and lock[0] - 0.5 <= v[0] <= t]
        if splits:
            lang = splits[-1][2]
        return self.route_of(lang) or self.fallback, lang

    def _owner_for(self, t: float, end: float, final: bool):
        """(담당 서버, 언어) 또는 None(무음), 또는 "hold"."""
        seg = self._segment_at(t)
        if seg is None:
            return None
        s, e, closed = seg
        if self._lock_for(s, e) is not None:
            return self._decide(s, e, t, None)
        v = self.tracker._find_overlap(s, e)
        if v is not None:
            # **구간의 판정을 그대로 쓴다.** 첫 조각은 PRE_ROLL 만큼 발화 시작보다
            # 앞이라, 그 시각으로 판정을 찾으면(lang_for_range) 이 구간과 안 겹쳐
            # "그 앞에서 끝난 마지막 판정" 으로 떨어진다 — 실측에서 오판 17건이 전부
            # 직전 발화의 언어였다.
            return self._decide(s, e, t, v[2])
        too_short = closed and (e - s) <= self.tracker.MIN_SPEECH_SEC
        if too_short:
            # 판정하지 않는 짧은 조각. 직전 발화가 가까우면 그 담당에게 (짧은 대답이
            # 흔하다), 아니면 기본 서버에.
            if self.last_lang and (s - self._last_speech_end) < 2.0:
                return self._decide(s, e, t, self.last_lang)
            return self._decide(s, e, t, None)
        max_hold = self.tracker.router.window_sec + MAX_HOLD_AFTER_WINDOW
        if final or (self.audio_sec - t) >= max_hold:
            if self.hold_since is not None:
                print(f"hold timeout at {t:.2f}s -> {self.last_lang or self.fallback}", flush=True)
            return self._decide(s, e, t, self.last_lang)
        return "hold"

    async def dispatch(self, send, final: bool = False) -> None:
        while True:
            t = self.cursor
            # **여유는 무음에서만 둔다.** 0.6초 뒤에서 보내는 이유는 발화 시작을 미리
            # 알고 PRE_ROLL 을 실제 오디오로 보내기 위해서다. 담당이 이미 정해진 발화
            # 안에서는 알 것이 없으므로 바로 보낸다 — 첫 partial 이 0.5초 빨라진다.
            seg = None if final else self._segment_at(t)
            in_locked = seg is not None and self._lock_for(seg[0], seg[1]) is not None
            delay = 0.0 if final else (LIVE_DELAY if in_locked else DISPATCH_DELAY)
            target = self.audio_sec - delay
            if t >= target - 1e-6:
                break
            end = min(target, t + PIECE)
            got = self._owner_for(t, end, final)
            if got == "hold":
                if self.hold_since is None:
                    self.hold_since = t
                break
            self.hold_since = None
            owner, lang = got if got else (None, None)
            piece = self._slice(t, end)
            for name in self.names:
                await send(name, piece if name == owner else bytes(len(piece)))
            if owner is not None and owner in self.spans:
                self._note(owner, t, end, lang)
                self.last_owner = owner
                if lang:
                    self.last_lang = lang
                self._last_speech_end = end
            self.cursor = end

    async def catch_up_splits(self, send) -> None:
        """발화 한복판에서 언어가 바뀐 판정이 새로 생겼으면 새 담당에게 그 뒤를 보낸다."""
        for v in self.tracker.verdicts:
            if len(v) < 6 or not v[5]:
                continue
            key = round(v[0], 2)
            if key in self._seen_splits:
                continue
            self._seen_splits.add(key)
            owner = self.route_of(v[2]) or self.fallback
            if owner not in self.spans:
                continue
            a = max(self._offset, v[0] - SPLIT_BACKOFF)
            b = self.cursor
            if b <= a:
                continue
            already = any(s <= a and b <= e for s, e, _l in self.spans[owner])
            if already:
                continue
            piece = self._slice(a, b)
            await send(owner, piece)
            self.extra[owner] += len(piece) / 2 / SR
            self._note(owner, a, b, v[2])
            print(f"split {v[0]:.2f}s -> {v[2]}: resend {a:.2f}~{b:.2f} to {owner}", flush=True)

    async def correct_locks(self, send) -> None:
        """확정 판정(구간 전체 로그확률)이 잠금과 다르면 그 발화를 맞는 서버로 다시 보낸다.

        **잠금은 1.5초만 듣고 정한 값이다.** 실제 녹음에서 7% 는 틀리고, 그 대부분이
        한국인 화자의 스페인어가 en 으로 가는 경우다. 구간이 닫힌 뒤 전체를 다시
        판정하면 그 오답이 사라지므로(30발화에서 100%), 다르면 발화 전체를 맞는 담당에게
        추가로 보내고 옛 담당의 그 구간 final 은 버리게 표시한다(`revoked`). 맞는 자막이
        발화 끝난 뒤 1~2초 늦게 뜨는 대신, 엉뚱한 언어의 자막은 안 뜬다.
        """
        for lock in self._locks:
            if id(lock) in self._checked:
                continue
            v = self.tracker._find_overlap(lock[0], max(lock[1], lock[0] + 0.1))
            if v is None or len(v) < 5 or not v[4]:
                continue                       # 아직 확정 판정 전
            self._checked.add(id(lock))
            new_lang = v[2]
            if new_lang == lock[2]:
                continue
            new_owner = self.route_of(new_lang) or self.fallback
            old_owner = self.route_of(lock[2]) or self.fallback
            if new_owner == old_owner or new_owner not in self.spans:
                continue
            a = max(self._offset, lock[0] - PRE_ROLL)
            b = min(self.cursor, (v[1] if v[1] is not None else lock[1]) + POST_ROLL)
            if b <= a:
                continue
            piece = self._slice(a, b)
            await send(new_owner, piece)
            self.extra[new_owner] += len(piece) / 2 / SR
            self._note(new_owner, a, b, new_lang)
            self.revoked[old_owner].append((a, b))
            lock[2] = new_lang
            print(f"correct {lock[0]:.2f}s {old_owner} -> {new_owner} ({new_lang}), "
                  f"resend {a:.2f}~{b:.2f}", flush=True)

    def is_revoked(self, name: str, key: float) -> bool:
        return any(a - 0.5 <= key <= b for a, b in self.revoked.get(name, ()))

    def to_proxy_time(self, name: str, server_sec) -> Optional[float]:
        if server_sec is None:
            return None
        return server_sec - self.extra.get(name, 0.0)

    def speech_in(self, name: str, a: float, b: float):
        """[a, b] 안에서 이 서버가 받은 실제 음성 길이와 그중 가장 긴 구간의 언어."""
        total, best, best_lang = 0.0, 0.0, None
        for s, e, lang in self.spans.get(name, ()):
            ov = min(b, e) - max(a, s)
            if ov > 0:
                total += ov
                if ov > best:
                    best, best_lang = ov, lang
        return total, best_lang


class Reorder:
    """언어 서버마다 커밋 시점이 달라 뒤집히는 final 순서를 바로잡는다.

    en 서버는 `<SEG>` 를 보자마자 커밋하고 베이스라인은 VAD 가 닫힌 뒤(800ms) 커밋한다.
    그래서 스페인어 뒤에 영어가 오면 영어 final 이 먼저 도착한다 — 실측에서 `Sure,` 가
    그보다 먼저 말한 `Hola, puedo sentarme aquí.` 위에 떴다.

    **기다리는 건 정말 앞선 발화가 있을 때만이다.** final 을 무조건 몇백 ms 붙들면 전부
    그만큼 늦어진다. 대신 이 final 이 속한 발화(그 서버에 보낸 음성 구간)의 시작 시각을
    키로 잡고, 다른 서버에 그보다 먼저 시작했는데 아직 final 을 내지 않은 구간이 있을
    때만 붙든다. 그 서버의 final 이 오거나 1.5초가 지나면 내보낸다. 구간이 있어도
    final 이 영영 안 올 수 있으므로(무음 환각으로 걸러짐, 조각) 시한이 있어야 한다.
    """

    WAIT_SEC = 1.5

    def __init__(self, disp, names, client):
        self.disp = disp
        self.client = client
        self.done = {n: -1.0 for n in names}   # 서버별로 마지막에 내보낸 final 의 키
        self.queue: list = []                   # [키, 순번, 서버, 메시지, 시한]
        self.seq = 0

    def key_for(self, name: str, b) -> float:
        """이 final 이 속한 발화의 시작 — 그 서버에 보낸 음성 구간 중 end 앞의 마지막 것."""
        if b is None:
            b = self.disp.cursor
        starts = [sp[0] for sp in self.disp.spans.get(name, ()) if sp[0] <= b + 0.05]
        return starts[-1] if starts else b

    def blocker(self, name: str, key: float):
        now = self.disp.audio_sec
        for other, spans in self.disp.spans.items():
            if other == name:
                continue
            for s0, s1, _lang in spans:
                if (s0 < key - 0.05 and s0 > self.done[other] + 1e-3
                        and (s1 - s0) >= 0.5 and (now - s1) < 3.0):
                    return other
        return None

    async def _forward(self, name: str, key: float, msg: str) -> None:
        self.done[name] = max(self.done[name], key)
        await self.client.send(msg)

    async def submit(self, name: str, key: float, msg: str) -> None:
        self.seq += 1
        other = self.blocker(name, key)
        if other is None and not self.queue:
            await self._forward(name, key, msg)
            return
        if other is not None:
            print(f"reorder: hold {name} key={key:.2f} behind {other}", flush=True)
        self.queue.append([key, self.seq, name, msg, time.monotonic() + self.WAIT_SEC])
        await self.drain()

    async def drain(self) -> None:
        """키 순서로, 막는 서버가 없어졌거나 시한이 지난 것부터 내보낸다."""
        while self.queue:
            self.queue.sort(key=lambda q: (q[0], q[1]))
            key, _seq, name, msg, deadline = self.queue[0]
            other = self.blocker(name, key)
            if other is not None and time.monotonic() < deadline:
                return
            self.queue.pop(0)
            if other is not None:
                print(f"reorder: timeout {name} key={key:.2f}", flush=True)
            await self._forward(name, key, msg)

    async def ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self.queue:
                await self.drain()


async def run_dual(client, start_raw, pending_binary):
    """라우팅 표의 모든 서버에 붙되, 오디오는 **판정 언어의 담당 서버에만** 보낸다.

    판정은 전사가 아니라 **오디오**로 한다. VerdictTracker 가 VAD 로 찾은 발화 구간의
    음성을 whisper-small 에 넣어 언어를 정하고, Dispatcher 가 그 구간의 오디오를 그
    언어의 서버에만 흘린다. 나머지 서버는 같은 길이의 무음을 받아 시간축을 맞춘다.
    각 서버는 자기 언어만 들으므로 final 을 고를 일이 없다 — 프록시가 보낸 음성이
    없는 구간의 final(무음 환각)만 거른다.

    핸드셰이크 응답(ready 등)은 한 서버 것만 넘긴다. 두 벌이 가면 클라이언트가
    같은 신호를 두 번 본다.
    """
    from lid_router import VerdictTracker

    # 붙을 서버들. 언어 코드로 된 것들에 더해, --rest 가 있으면 "*" 로 하나 더.
    servers = dict(ROUTES)
    if REST_UPSTREAM:
        servers[REST_KEY] = REST_UPSTREAM
    langs = list(servers)

    def narrow(msg_obj, server):
        """서버마다 자기가 맡은 언어만 담은 start/config 를 만든다.

        **서버는 langMap 의 키로 ASR 허용 언어(로짓 바이어스)를 정한다.** 웹에서
        ko·en·es 를 고르면 세 서버가 모두 셋을 허용받아, 맡지도 않은 언어를 답으로
        낼 수 있다. 실제로 베이스라인이 스페인어 'Me llamo Daniel' 을 ko 로 보고
        '메야모 다니엘.' 이라고 한글로 받아썼다. 원문과 번역이 같은 한글이 되어
        화면에는 같은 줄이 두 번 뜬다.

        ko 서버에는 ko 만, en 서버에는 en 만, --rest 에는 그 밖 언어만 남겨
        보낸다. 남길 게 없으면(예: 웹에서 ko·en 만 골랐는데 --rest 몫이 없음)
        원본을 그대로 보낸다 — 빈 langMap 을 주면 제한이 아예 풀린다.
        """
        lm = msg_obj.get("langMap")
        if not isinstance(lm, dict) or not lm:
            return None
        if server == REST_KEY:
            keys = [k for k in lm if k not in ROUTES]
        else:
            keys = [k for k in lm if k == server]
        if not keys or len(keys) == len(lm):
            return None
        out = dict(msg_obj)
        out["langMap"] = {k: lm[k] for k in keys}
        return json.dumps(out, ensure_ascii=False)

    try:
        _start = json.loads(start_raw)
    except Exception:
        _start = {}
    lang_map = _start.get("langMap") if isinstance(_start.get("langMap"), dict) else {}
    target_lang = _start.get("targetLang") or ""
    # 웹이 고른 소스 언어가 곧 LID 후보다. **여기에 라우팅 표를 더하면 안 된다.**
    # 종전에는 "서버가 있는 언어는 언제나 후보에 남긴다" 며 ko·en 을 강제로 넣었는데,
    # 그러면 사용자가 한국어를 꺼도 한국어가 답으로 나올 수 있다. 실제로 스페인어
    # 발화 '¿Dónde está el baño?' 가 ko 로 판정돼, 정확히 받아쓴 베이스라인과 en
    # 서버가 버려지고 ko 서버의 '돈데 스타일 반요.' 가 나갔다.
    # 고른 게 없을 때만(langMap 도 없고 lang 도 auto) 라우팅 표로 되돌아간다.
    allowed = set(lang_map) or {c for c in [_start.get("lang")] if c and c != "auto"}
    if not allowed:
        allowed = set(ROUTES)
    # step_sec 0.25: Dispatcher 가 발화 시작을 DISPATCH_DELAY 안에 알아야 한다.
    tracker = VerdictTracker(LID, allowed=allowed, step_sec=0.25, scan_win=LID_SCAN_WIN,
                             scan_hop=LID_SCAN_HOP, scan_confirm=LID_SCAN_CONFIRM,
                             early=LID_EARLY)

    ups = {}
    try:
        # **서버 하나가 죽어 있어도 세션은 연다.** 종전에는 --rest 가 안 떠 있으면
        # ConnectionRefusedError 로 세션이 통째로 끊겨 ready 조차 안 갔다.
        for lang in langs:
            try:
                ups[lang] = await websockets.connect(servers[lang], ping_interval=None,
                                                     max_size=None)
                await ups[lang].recv()         # 위쪽 hello 는 프록시가 이미 보냈다
                await ups[lang].send(narrow(_start, lang) or start_raw)
            except Exception as e:
                print(f"upstream {lang} ({servers[lang]}) 연결 실패: {e!r} — 없이 간다",
                      flush=True)
                ups.pop(lang, None)
        if not ups:
            print("붙을 서버가 하나도 없다", flush=True)
            return
        live = [l for l in langs if l in ups]
        primary = live[0]
        dead: set = set()
        # 판정이 아무 데도 안 맞을 때. rest 가 살아 있으면 그쪽, 없으면 기본 서버의 언어.
        fallback = REST_KEY if REST_KEY in ups else next(
            (l for l in live if ROUTES.get(l) == DEFAULT_UPSTREAM), primary)

        def route_of(verdict):
            """판정 언어를 붙을 서버 이름으로 바꾼다. 죽은 서버면 fallback."""
            if verdict is None:
                return None
            if verdict in ups and verdict not in dead:
                return verdict
            return fallback if fallback not in dead else next(
                (l for l in live if l not in dead), fallback)

        disp = Dispatcher(tracker, live, route_of, fallback)
        reorder = Reorder(disp, live, client)
        print(f"dual -> {', '.join(f'{k}:{servers[k]}' for k in live)} "
              f"| LID 후보 {sorted(allowed)} | 담당 서버에만 오디오 전달, 나머지는 무음",
              flush=True)

        async def send_to(name, data):
            if name in dead:
                return
            try:
                await ups[name].send(data)
            except Exception as e:
                dead.add(name)
                print(f"upstream {name} 전송 실패: {e!r}", flush=True)

        for chunk in pending_binary:
            tracker.feed(chunk)
            disp.feed(chunk)

        # 서버에 이미 알린 판정. 같은 구간을 두 번 알리지 않으려고 들고 있는다.
        hinted: dict = {}

        async def push_hints():
            """새로 선 판정을 담당 서버에 알린다 (--lang-hint).

            오디오를 담당에게만 보내는 지금 구조에서는 슬롯 자르기가 필요 없다 —
            언어가 바뀌면 옛 담당의 오디오가 무음으로 바뀌어 VAD 가 알아서 닫는다.
            남는 효과는 언어 태그를 판정으로 못박는 것뿐이라 cut 은 보내지 않는다.
            """
            for v in tracker.verdicts:
                key = round(v[0], 1)
                if hinted.get(key) == v[2]:
                    continue
                target = route_of(v[2])
                if target is None or target not in ups:
                    continue
                hinted[key] = v[2]
                await send_to(target, json.dumps(
                    {"type": "lang_hint", "lang": v[2],
                     "fromSec": max(0.0, v[0] - HINT_BACKOFF_SEC),
                     "cut": False, "force": LANG_HINT_FORCE}))

        async def pump_client():
            nonlocal lang_map, target_lang
            async for msg in client:
                data = None
                if isinstance(msg, (bytes, bytearray)):
                    raw = bytes(msg)
                    tracker.feed(raw)
                    disp.feed(raw)
                    await tracker.update()
                    if LANG_HINT:
                        await push_hints()
                    await disp.catch_up_splits(send_to)
                    await disp.correct_locks(send_to)
                    await disp.dispatch(send_to)
                    continue
                # **흐르는 중에 언어를 바꾸면 LID 후보도 따라가야 한다.**
                # 서버는 config 를 받아 로짓 바이어스를 갈지만(다음 슬롯부터),
                # 프록시는 start 만 보고 후보를 정해 두면 그대로 굳는다. 시연 중
                # 언어를 켜고 끄면 ASR 만 따라가고 판정은 안 따라가는 셈이다.
                try:
                    data = json.loads(msg)
                except Exception:
                    data = None
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "config":
                    if isinstance(data.get("langMap"), dict):
                        lang_map = data["langMap"]
                    if data.get("targetLang"):
                        target_lang = data["targetLang"]
                    new_allowed = set(lang_map) or {
                        c for c in [data.get("lang")] if c and c != "auto"}
                    if new_allowed and new_allowed != tracker.allowed:
                        print(f"config: LID 후보 {sorted(tracker.allowed)} -> "
                              f"{sorted(new_allowed)}", flush=True)
                        tracker.allowed = new_allowed
                if kind in ("finish", "stop"):
                    # 붙들어 둔 꼬리를 다 흘린 뒤에 끝내야 마지막 문장이 산다.
                    await tracker.update(force=True)
                    await disp.catch_up_splits(send_to)
                    await disp.dispatch(send_to, final=True)
                for _lang in list(ups):
                    if kind in ("start", "config"):
                        await send_to(_lang, narrow(data, _lang) or msg)
                    else:
                        await send_to(_lang, msg)

        async def pump_upstream(lang):
            try:
                await _pump_upstream(lang)
            except Exception as e:
                # 서버 하나가 죽어도 세션은 유지한다. 그 서버 몫의 언어는 route_of 가
                # fallback 으로 돌린다.
                dead.add(lang)
                print(f"upstream {lang} 끊김: {e!r} — 나머지로 계속", flush=True)
                await asyncio.Event().wait()

        async def _pump_upstream(lang):
            # 이 서버가 마지막으로 확정한 오디오 지점. 다음 final 의 시작점으로 쓴다 —
            # 서버가 final.start 를 안 채우기 때문이다.
            group_start = 0.0
            last_span = (0.0, 0.0)
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
                if kind == "partial":
                    # 지금 담당이 아닌 서버의 partial 은 옛 슬롯의 찌꺼기다.
                    if disp.last_owner == lang:
                        await client.send(msg)
                    continue
                b = disp.to_proxy_time(lang, _secs(data.get("end")))
                if b is not None and b > group_start:
                    # 같은 경계에서 여러 final 이 나오면(한 슬롯의 문장들) 같은 구간을
                    # 공유한다. 빈 구간으로 보면 두 번째 문장부터 전부 "무음" 으로 버려진다
                    # — 실측에서 `서울에 살아요`, `See you then` 이 그렇게 사라졌다.
                    span = (group_start, b)
                    group_start = b
                    last_span = span
                elif b is None:
                    span = (group_start, disp.cursor)
                else:
                    span = last_span
                heard, verdict_lang = disp.speech_in(lang, *span)
                if heard <= 0.05:
                    # 커밋 경계는 발화가 끝난 뒤에 찍히고, 한 슬롯의 문장들은 경계를
                    # 거의 같이 찍는다(dot 커밋 79.0, 그 뒤 VAD 커밋 79.2). 그 사이만 보면
                    # 진짜 문장도 "음성 없음" 이 된다 — 실측에서 `El coreano es difícil
                    # para mí.` 가 그렇게 버려졌다. 서버는 0 만 받은 슬롯을 디코딩하지
                    # 않으므로 환각은 실제 음성 뒤 꼬리에서만 나온다. 그래서 몇 초 안에
                    # 이 서버로 간 음성이 하나도 없을 때만 버린다.
                    b_eff = span[1]
                    heard, verdict_lang = disp.speech_in(lang, b_eff - SILENCE_LOOKBACK, b_eff)
                if heard <= 0.05:
                    print(f"drop {lang} [{span[0]:.1f}~{span[1]:.1f}] silence: "
                          f"{(data.get('original') or '')[:40]!r}", flush=True)
                    continue
                key = reorder.key_for(lang, b)
                if disp.is_revoked(lang, key):
                    print(f"drop {lang} key={key:.2f} revoked: "
                          f"{(data.get('original') or '')[:40]!r}", flush=True)
                    continue
                if lang in ROUTES:
                    verdict_lang = lang        # 이 서버는 자기 언어만 듣는다
                data, fixed = await fix_direction(data, verdict_lang, lang_map, target_lang)
                if fixed:
                    print(f"fix dir end={b} {lang}: "
                          f"{(data.get('original') or '')[:30]!r} -> "
                          f"{(data.get('translation') or '')[:30]!r}", flush=True)
                src = (verdict_lang or data.get("language") or "").lower()
                before = data.get("translations")
                data = await add_audience_translations(
                    data, src, (lang_map or {}).get(src) or target_lang)
                if fixed or data.get("translations") is not before:
                    msg = json.dumps(data, ensure_ascii=False)
                await reorder.submit(lang, key, msg)

        tasks = [asyncio.create_task(pump_client())]
        tasks += [asyncio.create_task(pump_upstream(l)) for l in live]
        tasks.append(asyncio.create_task(reorder.ticker()))
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
        mode = "발화별 담당 서버에만 전달" if DUAL else "스트림당 1회 라우팅"
        print(f"  음성 판정: {LID.model_name}, 창 {LID.window_sec}s — {mode}", flush=True)
    if TRANSLATE_URL:
        print(f"  번역 방향 교정: {TRANSLATE_URL}", flush=True)
    if LID_SCAN_WIN:
        print(f"  구간 안 스캔: 창 {LID_SCAN_WIN}s, 홉 {LID_SCAN_HOP}s, "
              f"연속 {LID_SCAN_CONFIRM}회 확인", flush=True)
    if LANG_HINT:
        how = "force_language" if LANG_HINT_FORCE else "로짓 바이어스"
        print(f"  판정 언어를 서버에 통보(lang_hint) — {how} + 슬롯 자르기", flush=True)
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
    ap.add_argument("--lid-model", default="openai/whisper-small")
    ap.add_argument("--lid-window", type=float, default=1.5,
                    help="발화 시작 후 몇 초를 듣고 판정할지. 원어민 낭독(226클립)은 "
                         "1.0s=98.2%%, 2.0s=99.1%% 지만 한국인 화자의 3개 국어 대화 "
                         "35발화에서는 0.5s=60%%, 1.0s=83%%, 1.5s=91%%, 2.0s=89%% 다. "
                         "오디오는 첫 판정의 서버로만 가므로 1.5s 를 기본으로 둔다")
    ap.add_argument("--lid-early", action="store_true",
                    help="발화 시작 0.3초부터 확신도가 높으면 창을 다 듣지 않고 확정한다. "
                         "끄면(기본) --lid-window 만큼 듣고 정한다 — 오디오는 첫 판정의 "
                         "서버로만 가므로 0.4초 더 듣는 쪽이 억양 있는 화자에게 안전하다")
    ap.add_argument("--lid-max-wait", type=float, default=5.0,
                    help="이만큼 들어도 판정이 안 서면 start.lang 규칙으로 되돌아간다")
    ap.add_argument("--lid-device", default="cuda")
    ap.add_argument("--lid-scan", action="store_true",
                    help="발화 구간 안을 창으로 계속 훑어 구간 한복판의 언어 전환도 "
                         "잡는다. 끄면 구간 앞머리에서 한 번 정하고 그 구간 내내 "
                         "그 값을 쓴다 — 숨 안 쉬고 언어를 바꾸면 못 따라간다")
    ap.add_argument("--lid-scan-window", type=float, default=1.5,
                    help="스캔 한 번이 보는 오디오 길이. 실측(합성 전환 120쌍) "
                         "1.5s 는 지연 중앙 1.07s·지점오차 p90 0.62s, "
                         "2.0s 는 헛플립이 0 에 가까운 대신 1.68s·1.09s")
    ap.add_argument("--lid-scan-hop", type=float, default=0.25,
                    help="스캔을 얼마마다 도는지. 짧을수록 연속 확인이 싸다")
    ap.add_argument("--lid-scan-confirm", type=int, default=2,
                    help="새 언어를 몇 번 연속 봐야 전환으로 인정할지. 1 이면 즉시 "
                         "인정해 헛플립이 쌍당 0.32건까지 는다")
    ap.add_argument("--lang-hint", action="store_true",
                    help="판정 언어를 lang_hint 로 서버에 보낸다. 서버는 그 언어만 "
                         "허용(로짓 바이어스)하고 슬롯을 잘라 앞 언어의 프리픽스를 "
                         "비운다. 서버가 이 메시지를 알아야 한다")
    ap.add_argument("--lang-hint-force", action="store_true",
                    help="lang_hint 를 바이어스 대신 force_language 로 적용한다. "
                         "프롬프트에 언어를 박아 넣는 방식이라 출력 형식이 바뀌고 "
                         "커밋 판정이 달라진다 — 실측에서 짧은 발화가 뭉개졌다")
    ap.add_argument("--targets", default=None,
                    help="청중 언어 목록(예: ko,en,es). 발화 하나를 이 언어들로 모두 "
                         "번역해 translations 로 보낸다. 서버는 발화당 목표 하나만 "
                         "내므로 나머지는 프록시가 --translate-url 로 채운다")
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
    if args.targets:
        AUDIENCE = [c.strip().lower() for c in args.targets.split(",") if c.strip()]
    LID_EARLY = args.lid_early
    LANG_HINT = args.lang_hint or args.lang_hint_force
    LANG_HINT_FORCE = args.lang_hint_force
    if args.lid_scan:
        LID_SCAN_WIN = args.lid_scan_window
        LID_SCAN_HOP = args.lid_scan_hop
        LID_SCAN_CONFIRM = args.lid_scan_confirm
    if args.lid or args.dual:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from lid_router import LidRouter

        LID = LidRouter(model_name=args.lid_model, window_sec=args.lid_window,
                        max_wait_sec=args.lid_max_wait, device=args.lid_device,
                        known_langs=ROUTES.keys())

    asyncio.run(main())
