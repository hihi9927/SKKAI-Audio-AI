#!/usr/bin/env python3
"""번역 호출 계측 + 재시도. **평가 서버 전용**(프로덕션은 건드리지 않는다).

왜 필요한가
-----------
프로덕션의 `google_translate_async` 는 예외를 통째로 삼키고 `("", "")` 를 돌려준다:

    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return "", ""

무료 gtx 엔드포인트(`translate.googleapis.com/translate_a/single?client=gtx`)는
비공식이라 할당량이 문서화돼 있지 않고, 몰아치면 429 나 HTML 차단 페이지를 준다.
그러면 `resp.json()` 이 터지고 → 빈 문자열이 반환되고 → **그 세그먼트는 "번역이 빈
커밋"이 된다.** 채점 단계에서는 정책이 아무 말도 못 만든 경우와 구분되지 않는다.
34만 건짜리 런이 조용히 오염된 채 끝나고, 며칠 뒤 "왜 BLEU 가 낮지" 를 하게 된다.

그래서 평가에서는 (1) 재시도로 일시적 실패를 흡수하고, (2) 그래도 실패한 건을
**세어서 기록**한다. call count 는 어차피 보고할 지표이기도 하다.

무엇을 세는가
-------------
- `calls`        실제로 나간 HTTP 요청 수 (재시도 포함)
- `commits`      번역이 요청된 커밋 수 (재시도 무관 — 정책의 호출 횟수)
- `retries`      재시도한 횟수
- `failed`       재시도를 다 쓰고도 실패한 커밋 수  ← **0 이 아니면 그 런은 의심해야 한다**
- `empty_result` 호출은 성공했는데 번역문이 빈 경우 (입력이 기호뿐인 등)
- `errors`       예외 종류별 횟수 (`HTTP429`, `TimeoutError`, `JSONDecodeError` …)

세그먼트 단위 귀속은 ContextVar 로 한다. asyncio 태스크는 생성 시점의 컨텍스트를
복사하므로, 커밋마다 다른 태스크에서 도는 번역들이 서로의 카운터를 오염시키지 않는다.

사용:

    import trans_guard
    trans_guard.install(base_server_module, retries=3, timeout=10.0)
    ...
    tok = trans_guard.begin_local()
    try:
        ...번역...
    finally:
        local = trans_guard.end_local(tok)   # {"calls":1,"retries":0,"failed":0} 또는 None
"""

from __future__ import annotations

import asyncio
import contextvars
import html
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 무료 위젯 엔드포인트. 인증이 없어 Google 이 구분할 수 있는 건 IP 뿐이고, 그래서
# 대량으로 쓰면 IP 단위로 429("Sorry..." HTML)를 맞는다. 실제로 2026-08 에 분절 실험이
# 6일간 30만 건을 호출해 막혔다. 평가 규모(주표만 4.5만 건)에는 맞지 않는다.
GTX_URL = "https://translate.googleapis.com/translate_a/single"

# 공식 Cloud Translation **Basic(v2)**. API 키로 인증한다(Advanced/v3 는 서비스 계정이
# 필요해 API 키로는 못 쓴다). 평문 번역은 v2/v3 가 같은 NMT 모델이고 요금도 같다.
V2_URL = "https://translation.googleapis.com/language/translate/v2"


class TranslateHTTPError(RuntimeError):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


@dataclass
class _Config:
    backend: str = "gtx"       # "gtx"(무료 위젯) | "v2"(Cloud Translation Basic) | "local"(MADLAD)
    api_key: Optional[str] = None
    local_model: str = "google/madlad400-3b-mt"
    local_device: str = "cuda"
    local_batch: int = 16      # 한 번에 GPU 로 보내는 최대 문장 수
    local_wait: float = 0.03   # 배치를 모으려고 기다리는 시간(초). 지연에 직접 더해진다
    local_max_new_tokens: int = 192
    retries: int = 3           # 총 시도 횟수(첫 시도 포함)
    backoff: float = 0.5       # 지수 백오프 기준(초)
    backoff_429: float = 5.0   # 429/403 은 더 길게 — 몰아치면 IP 째로 막힌다
    timeout: float = 10.0
    alert_rate: float = 0.005  # 실패율이 이 값을 넘으면 CRITICAL
    alert_min_calls: int = 200
    stats_path: Optional[Path] = None
    dump_every: int = 200      # 중간에 죽어도 통계가 남게 주기적으로 쓴다


@dataclass
class _Stats:
    commits: int = 0
    calls: int = 0
    ok: int = 0
    retries: int = 0
    failed: int = 0
    empty_result: int = 0
    empty_input: int = 0
    errors: Counter = field(default_factory=Counter)
    first_error_at: Optional[float] = None
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_alert: float = 0.0

    def as_dict(self) -> dict:
        el = max(time.time() - self.started_at, 1e-9)
        return {
            "commits": self.commits,
            "calls": self.calls,
            "ok": self.ok,
            "retries": self.retries,
            "failed": self.failed,
            "empty_result": self.empty_result,
            "empty_input": self.empty_input,
            "failure_rate": round(self.failed / self.commits, 6) if self.commits else 0.0,
            "calls_per_commit": round(self.calls / self.commits, 4) if self.commits else 0.0,
            "calls_per_sec": round(self.calls / el, 3),
            "errors": dict(self.errors),
            "repeat_guard": dict(_REP_STATS),
            "elapsed_sec": round(el, 1),
            "backend": _CFG.backend,
        }


_CFG = _Config()
_STATS = _Stats()
_orig_translate = None
_target_module = None

# 세그먼트(커밋) 단위 귀속용. None 이면 "상위에서 이미 집계 중"이라는 뜻이다.
_local: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "ast_trans_local", default=None
)


def begin_local():
    """이 커밋의 번역 호출 집계를 시작한다. 이미 상위가 집계 중이면 None 을 준다.

    base 의 `_correct_and_translate` 는 방향 자가교정 때 `_translate` 를 한 번 더
    부른다. 중첩해서 새 dict 를 깔면 안쪽 증가분이 바깥에 안 잡히므로, **가장 바깥
    한 곳만** 집계한다.
    """
    if _local.get() is not None:
        return None
    return _local.set({"calls": 0, "retries": 0, "failed": 0})


def end_local(token) -> Optional[dict]:
    if token is None:
        return None
    d = _local.get()
    _local.reset(token)
    return d


def _bump_local(key: str, n: int = 1) -> None:
    d = _local.get()
    if d is not None:
        d[key] = d.get(key, 0) + n


def snapshot() -> dict:
    return _STATS.as_dict()


def dump(path: Optional[Path] = None) -> None:
    p = path or _CFG.stats_path
    if not p:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    except Exception as exc:  # 통계 기록 실패가 런을 죽이면 안 된다
        logger.warning("번역 통계 기록 실패: %s", exc)


def log_summary(prefix: str = "[TRANS-STATS]") -> None:
    s = snapshot()
    logger.info(
        "%s 커밋 %d / 호출 %d (커밋당 %.2f) / 재시도 %d / 실패 %d (%.3f%%) / "
        "빈결과 %d / %.1f call/s / 오류 %s",
        prefix, s["commits"], s["calls"], s["calls_per_commit"], s["retries"],
        s["failed"], s["failure_rate"] * 100, s["empty_result"],
        s["calls_per_sec"], s["errors"] or "-",
    )


def _maybe_alert() -> None:
    """실패율이 임계를 넘으면 크게 운다. 런을 죽이지는 않는다 —
    90% 진행된 런을 끊는 것보다, 끝나고 통계를 보고 폐기 여부를 정하는 게 낫다."""
    if _STATS.commits < _CFG.alert_min_calls:
        return
    rate = _STATS.failed / _STATS.commits
    if rate <= _CFG.alert_rate:
        return
    now = time.time()
    if now - _STATS._last_alert < 30.0:
        return
    _STATS._last_alert = now
    logger.critical(
        "[TRANS-ALERT] 번역 실패율 %.2f%% (%d/%d) — 임계 %.2f%% 초과. "
        "rate-limit 을 의심할 것. 이 런의 BLEU/COMET 은 신뢰할 수 없다.",
        rate * 100, _STATS.failed, _STATS.commits, _CFG.alert_rate * 100,
    )


async def _call_gtx(session, text: str, target_lang: str, timeout) -> tuple[str, str]:
    params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
    async with session.get(GTX_URL, params=params,
                           headers={"User-Agent": "Mozilla/5.0"},
                           timeout=timeout) as resp:
        if resp.status != 200:
            # 본문을 읽어 커넥션을 정리한다(안 읽으면 keep-alive 재사용이 막힌다).
            await resp.read()
            raise TranslateHTTPError(resp.status)
        data = await resp.json(content_type=None)
    translated = "".join(item[0] for item in data[0] if item and item[0])
    detected = data[2] if len(data) > 2 else ""
    return translated, detected


async def _call_v2(session, text: str, target_lang: str, timeout) -> tuple[str, str]:
    """공식 Cloud Translation Basic(v2).

    `source` 를 넘기지 않고 자동 감지에 맡긴다 — gtx 의 `sl=auto` 와 같은 조건으로
    두어야 프로덕션 동작을 재현하고, `detectedSourceLanguage` 가 있어야 base 의
    방향 자가교정(`_maybe_fix_direction`)이 그대로 돈다.
    """
    body = {"q": text, "target": target_lang, "format": "text"}
    async with session.post(V2_URL, params={"key": _CFG.api_key}, json=body,
                            timeout=timeout) as resp:
        payload = await resp.json(content_type=None)
        if resp.status != 200:
            err = (payload or {}).get("error", {})
            logger.debug("v2 오류 본문: %s", str(err)[:300])
            raise TranslateHTTPError(resp.status)
    tr = payload["data"]["translations"][0]
    # format=text 면 이스케이프하지 않는 게 문서상 동작이지만, 실제로는 문장에 따라
    # `&#39;` 가 섞여 나오는 사례가 보고돼 있다. 남아 있으면 BLEU 가 깎이므로 푼다.
    return html.unescape(tr.get("translatedText") or ""), tr.get("detectedSourceLanguage") or ""


# ── 로컬 번역기(MADLAD-400-3B, greedy) ──────────────────────────────────────
# Cloud Translation 키가 막혔을 때의 대체 경로. 품질은 Google 보다 낮다
# (CometKiwi 0.8712 → greedy 0.8473, 실측 `core/meaning_segmentator/autoseg`).
# **번역기를 바꾸면 BLEU/COMET 이 통째로 달라지므로 v2 로 낸 결과와 섞으면 안 된다.**
#
# 두 가지를 반드시 지킨다.
#   1) 생성은 **executor 로 뺀다.** 이벤트 루프에서 돌리면 오디오 수신과 ASR 디코딩이
#      멈춘다 — 실시간 페이싱 하네스에서는 그대로 측정 오염이다.
#   2) **마이크로 배칭.** 서버는 커밋마다 `create_task` 로 번역을 발사하므로 동시 요청이
#      몰린다. 하나씩 돌리면 GPU 가 논다. `local_wait` 만큼 모아 한 번에 돌린다 —
#      그 대기는 지연에 그대로 더해지므로 크게 잡으면 안 된다.
_LOCAL_LOCK = threading.Lock()
_LOCAL_MODEL: dict = {}
_LOCAL_Q: list = []
_LOCAL_CV = threading.Condition()
_LOCAL_THREAD = None


def _local_model():
    with _LOCAL_LOCK:
        if "m" not in _LOCAL_MODEL:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(_CFG.local_model)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                _CFG.local_model, dtype=torch.float16).to(_CFG.local_device).eval()
            _LOCAL_MODEL["m"] = (tok, mdl)
            logger.info("[TRANS-GUARD] 로컬 번역기 로드 %s (%.1fs, device=%s)",
                        _CFG.local_model, time.time() - t0, _CFG.local_device)
        return _LOCAL_MODEL["m"]


# ── 반복 폭주(degeneration) 탐지·복구 ────────────────────────────────────────
# greedy 디코딩의 알려진 병증. 실측(ACL dev, MADLAD): `あの` 를 30회, `ニュースの分類に
# 関する情報を，` 를 5회 반복하는 커밋이 나온다. 커밋 수로는 ja 3.5% 뿐인데 **글자 수로는
# 18.4%** 라(한 커밋이 수백 자를 토해낸다) char 단위 채점을 통째로 오염시킨다:
# static@6s/ja 음수율 11.82% 중 61.5% 가 이 커밋들에서 나왔다.
#
# `no_repeat_ngram_size` 는 쓰지 않는다. 정답 번역 자체가 4-gram 을 자연스럽게 반복하기
# 때문이다 — 실측으로 gold 문장의 ja 10.5% / de 6.4% / zh 7.3% 가 반복 4-gram 을 갖는다.
# n=4 로 막으면 그 문장들을 **아예 생성할 수 없고**, 현재 출력에서도 폭주 20건을 잡으려다
# 정상 73건을 부순다(ja static@6s).
#
# 그래서 **사후 탐지 후 그 커밋만** 손본다. 정상 출력은 한 글자도 바뀌지 않으므로 이전
# 결과와의 연속성이 유지된다.
#   1) 같은 4~8자 부분열이 4회 이상 → 폭주로 판정
#   2) `repetition_penalty` 를 걸고 그 문장만 재생성
#   3) 그래도 폭주면 반복이 시작되는 지점에서 자른다(뜻은 앞부분에 남는다)
_REP_STATS = Counter()


def _is_degenerate(text: str) -> bool:
    s = re.sub(r"\s+", "", text or "")
    if len(s) < 16:
        return False
    for n in (4, 6, 8):
        for i in range(len(s) - n):
            if s.count(s[i:i + n]) >= 4:
                return True
    return False


def _truncate_repeat(text: str) -> str:
    """반복이 시작되는 자리에서 자른다. 못 찾으면 원문 그대로."""
    s = text or ""
    for n in (8, 6, 4):
        for i in range(len(s) - n):
            g = s[i:i + n]
            if s.count(g) >= 4:
                nxt = s.find(g, i + n)
                if nxt > 0:
                    return s[:nxt].rstrip()
    return s


def _local_generate(texts: list, targets: list) -> list:
    """MADLAD 는 타깃을 `<2xx>` 접두사로 지정한다. 배치 안에 타깃이 섞여도 된다."""
    import torch
    tok, mdl = _local_model()
    prompts = [f"<2{tg}> {tx}" for tx, tg in zip(texts, targets)]
    with torch.inference_mode():
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(_CFG.local_device)
        gen = mdl.generate(**enc, max_new_tokens=_CFG.local_max_new_tokens,
                           num_beams=1)          # greedy — 빔4 대비 29배 빠르고 품질 −0.008
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    if os.environ.get("AST_TRANS_ANTI_REPEAT") != "1":
        return outs

    bad = [i for i, o in enumerate(outs) if _is_degenerate(o)]
    if not bad:
        return outs
    _REP_STATS["detected"] += len(bad)
    # 폭주한 것만 페널티를 걸고 다시. 정상 출력은 건드리지 않는다.
    with torch.inference_mode():
        enc2 = tok([prompts[i] for i in bad], return_tensors="pt", padding=True,
                   truncation=True, max_length=384).to(_CFG.local_device)
        gen2 = mdl.generate(**enc2, max_new_tokens=_CFG.local_max_new_tokens,
                            num_beams=1, repetition_penalty=1.15)
    for k, i in enumerate(bad):
        cand = tok.decode(gen2[k], skip_special_tokens=True)
        if not _is_degenerate(cand):
            outs[i] = cand
            _REP_STATS["fixed_by_retry"] += 1
        else:
            outs[i] = _truncate_repeat(cand if len(cand) < len(outs[i]) else outs[i])
            _REP_STATS["truncated"] += 1
    return outs


def _local_worker():
    while True:
        with _LOCAL_CV:
            while not _LOCAL_Q:
                _LOCAL_CV.wait(timeout=0.5)
            if len(_LOCAL_Q) < _CFG.local_batch:
                _LOCAL_CV.wait(timeout=_CFG.local_wait)
            jobs, _LOCAL_Q[:] = _LOCAL_Q[:_CFG.local_batch], _LOCAL_Q[_CFG.local_batch:]
        if not jobs:
            continue
        try:
            outs = _local_generate([j[0] for j in jobs], [j[1] for j in jobs])
            err = None
        except Exception as exc:                  # noqa: BLE001
            # OOM 은 배치가 커서 나는 것이므로 **한 건씩** 다시 돌리면 대개 통과한다.
            # 그냥 실패로 두면 그 커밋의 번역이 빈 문자열이 되어 BLEU/COMET 이 오염된다
            # (실측 2026-08-30 punct/de: OOM 61건 → 실패 18건, 5발화 손상).
            if "CUDA out of memory" in str(exc) or exc.__class__.__name__ == "OutOfMemoryError":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                outs, err = [], None
                for j in jobs:
                    try:
                        outs.append(_local_generate([j[0]], [j[1]])[0])
                    except Exception as e2:       # noqa: BLE001
                        outs, err = None, e2
                        break
                if outs is not None:
                    logger.warning("[TRANS-GUARD] 로컬 번역 OOM — 배치 %d건을 1건씩 재처리해 복구",
                                   len(jobs))
            else:
                outs, err = None, exc
        for k, (_, _, box, ev) in enumerate(jobs):
            box.append(outs[k] if outs is not None else err)
            ev.set()


def _local_submit(text: str, target_lang: str) -> str:
    """워커 스레드에 넣고 결과를 기다린다(호출자는 executor 스레드다)."""
    global _LOCAL_THREAD
    with _LOCAL_LOCK:
        if _LOCAL_THREAD is None:
            _LOCAL_THREAD = threading.Thread(target=_local_worker, daemon=True)
            _LOCAL_THREAD.start()
    box: list = []
    ev = threading.Event()
    with _LOCAL_CV:
        _LOCAL_Q.append((text, target_lang, box, ev))
        _LOCAL_CV.notify()
    ev.wait()
    if isinstance(box[0], BaseException):
        raise box[0]
    return box[0]


async def _call_local(session, text: str, target_lang: str, timeout) -> tuple[str, str]:
    """`_call_v2` 와 같은 시그니처. session/timeout 은 쓰지 않는다.

    감지 언어는 빈 문자열을 돌려준다 — base 의 방향 자가교정(`_maybe_fix_direction`)은
    감지값이 비면 아무것도 하지 않으므로, 고정 방향(en→X) 평가에서는 그게 맞다.
    """
    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(None, _local_submit, text, target_lang)
    return (out or "").strip(), ""


async def _guarded_translate(session, text: str, target_lang: str) -> tuple[str, str]:
    """재시도 + 계측을 붙인 번역. 원본 `google_translate_async` 와 같은 시그니처/반환형."""
    import aiohttp

    if not text.strip() or not target_lang:
        with _STATS._lock:
            _STATS.empty_input += 1
        return "", ""

    with _STATS._lock:
        _STATS.commits += 1
    timeout = aiohttp.ClientTimeout(total=_CFG.timeout)
    call = {"v2": _call_v2, "local": _call_local}.get(_CFG.backend, _call_gtx)

    last_exc: Optional[BaseException] = None
    for attempt in range(1, _CFG.retries + 1):
        with _STATS._lock:
            _STATS.calls += 1
        _bump_local("calls")
        try:
            translated, detected = await call(session, text, target_lang, timeout)
            with _STATS._lock:
                _STATS.ok += 1
                if not translated:
                    _STATS.empty_result += 1
                n = _STATS.commits
            # 성공 경로에서도 주기적으로 기록한다. 실패 경로에만 두면 **정상 런에서는
            # 통계 파일이 아예 생기지 않아** 사후에 "이 런이 몇 건을 불렀나"를 못 센다.
            if _CFG.dump_every and n % _CFG.dump_every == 0:
                dump()
            return translated, detected
        except Exception as exc:  # noqa: BLE001 — 종류를 세서 기록하는 게 목적이다
            last_exc = exc
            name = (f"HTTP{exc.status}" if isinstance(exc, TranslateHTTPError)
                    else type(exc).__name__)
            with _STATS._lock:
                _STATS.errors[name] += 1
                if _STATS.first_error_at is None:
                    _STATS.first_error_at = time.time()
            if attempt >= _CFG.retries:
                break
            with _STATS._lock:
                _STATS.retries += 1
            _bump_local("retries")
            base = (_CFG.backoff_429
                    if isinstance(exc, TranslateHTTPError) and exc.status in (403, 429)
                    else _CFG.backoff)
            # 지터를 넣지 않으면 16 병렬이 같은 순간에 동시에 재시도해 파도를 만든다.
            await asyncio.sleep(base * (2 ** (attempt - 1)) * (0.5 + random.random()))

    with _STATS._lock:
        _STATS.failed += 1
    _bump_local("failed")
    logger.error("[TRANS-FAIL] %d회 시도 실패 (%s) text=%r",
                 _CFG.retries, last_exc, text[:60])
    _maybe_alert()
    dump()   # 실패는 드물어야 하므로 발생할 때마다 기록한다
    return "", ""


def install(base_module, **opts) -> None:
    """base 모듈의 `google_translate_async` 를 계측판으로 교체한다.

    base 안의 호출부(`_translate`, `google_translate_with_context_async`)와 FSL 의
    `base_server.google_translate_async` 는 모두 **모듈 속성을 호출 시점에 조회**하므로,
    속성 하나만 갈아끼우면 전 경로가 덮인다.
    """
    global _orig_translate, _target_module
    for k, v in opts.items():
        if v is None:
            continue
        if not hasattr(_CFG, k):
            raise TypeError(f"알 수 없는 설정: {k}")
        setattr(_CFG, k, Path(v) if k == "stats_path" else v)

    if _orig_translate is None:
        _orig_translate = base_module.google_translate_async
    _target_module = base_module
    base_module.google_translate_async = _guarded_translate
    if _CFG.backend == "v2" and not _CFG.api_key:
        raise RuntimeError(
            "backend='v2' 인데 API 키가 없습니다. .env 의 GOOGLE_TRANSLATE_API_KEY 를 확인하세요."
        )
    logger.info(
        "[TRANS-GUARD] 설치됨 — 백엔드 %s / 재시도 %d회 / 타임아웃 %.1fs / "
        "백오프 %.1fs(429 %.1fs) / 경보 임계 %.2f%% / 통계 %s",
        _CFG.backend, _CFG.retries, _CFG.timeout, _CFG.backoff, _CFG.backoff_429,
        _CFG.alert_rate * 100, _CFG.stats_path or "-",
    )


def uninstall() -> None:
    if _orig_translate is not None and _target_module is not None:
        _target_module.google_translate_async = _orig_translate
