#!/usr/bin/env python3
"""로컬 번역 모델을 단독 프로세스로 띄우는 HTTP 서버.

**왜 따로 띄우나.** 번역기를 ASR 서버 안(`--local-translation`)에 두면 ASR 서버를
여러 개 띄울 때 번역 모델도 그 수만큼 복제된다. madlad400-3b 는 인스턴스당 약
7.1GiB 라 24GiB GPU 에서는 ASR 두 개와 함께 올릴 수 없다. 번역기를 한 번만 올려
두고 ASR 서버들은 `--local-translation-url` 로 이 서버를 부른다.

    python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 \
        --model google/madlad400-3b-mt

    POST /translate  {"text": ..., "target": "ko", "source": "en", "context": [...]}
              ->     {"translation": ..., "source": "en"}
    GET  /health     -> {"status": "ok", "model": ..., "context": true}

``context`` 는 앞 발화 원문 리스트(오래된 것부터)다. **LLM 백엔드일 때만 쓰인다** —
NLLB·MADLAD 는 문장 단위 모델이라 문맥을 넣을 자리가 없고, 이어붙여 넣으면 번역이
망가진다(core/translator/local_translator.py 의 LLMTranslator 주석에 실측이 있다).

    python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 \
        --model Qwen/Qwen3-4B-Instruct-2507 --quant 4bit --context-window 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.translator.local_translator import LLMTranslator, make_translator  # noqa: E402

logger = logging.getLogger("local-translation-server")


async def handle_translate(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    text = (body.get("text") or "").strip()
    target = (body.get("target") or "").strip()
    source = (body.get("source") or "").strip() or None
    context = body.get("context") or None
    if isinstance(context, str):        # 한 문장만 온 경우도 받아 준다
        context = [context]
    if not text or not target:
        return web.json_response({"translation": "", "source": source or ""})

    translator = request.app["translator"]
    translated, src = await translator.translate(text, target, source, context=context)
    return web.json_response({"translation": translated, "source": src})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "model": request.app["model_name"],
        # 클라이언트가 문맥을 보낼 가치가 있는지 여기서 알 수 있다.
        "context": request.app["supports_context"],
        "context_window": request.app["context_window"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model", default="google/madlad400-3b-mt",
                        help="번역 모델 이름 또는 경로 (madlad / nllb / 지시형 LLM)")
    parser.add_argument("--quant", default="4bit", choices=["none", "4bit", "8bit"],
                        help="LLM 백엔드의 양자화. seq2seq 백엔드에서는 무시된다")
    parser.add_argument("--context-window", type=int, default=1,
                        help=(
                            "LLM 백엔드가 쓸 앞 발화 수 (기본 1). 대화셋 600 방향쌍 실측에서 "
                            "1턴이 +0.0076 COMET-DA 이고 2턴 이상은 더 얻는 게 없었다"))
    parser.add_argument("--device", default=None,
                        help="모델을 올릴 장치 (cuda / cpu). 미지정 시 자동")
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    translator = make_translator(
        model_name=args.model, device=args.device, num_beams=args.num_beams,
        quant=args.quant, context_window=args.context_window,
    )
    # 첫 요청에서 로딩 지연이 나지 않게 미리 올린다. 여기서 죽으면 서버를 띄우지
    # 않는다 — 빈 번역을 돌려주는 서버가 살아 있는 편이 더 나쁘다.
    translator.load()

    app = web.Application()
    app["translator"] = translator
    app["model_name"] = args.model
    app["supports_context"] = isinstance(translator, LLMTranslator)
    app["context_window"] = args.context_window if app["supports_context"] else 0
    app.router.add_post("/translate", handle_translate)
    app.router.add_get("/health", handle_health)

    logger.info(
        f"translation server on {args.host}:{args.port} (model={args.model}, "
        f"context={'on, window=' + str(args.context_window) if app['supports_context'] else 'off'})")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
