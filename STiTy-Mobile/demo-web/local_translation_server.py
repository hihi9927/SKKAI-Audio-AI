#!/usr/bin/env python3
"""로컬 번역 모델을 단독 프로세스로 띄우는 HTTP 서버.

**왜 따로 띄우나.** 번역기를 ASR 서버 안(`--local-translation`)에 두면 ASR 서버를
여러 개 띄울 때 번역 모델도 그 수만큼 복제된다. madlad400-3b 는 인스턴스당 약
7.1GiB 라 24GiB GPU 에서는 ASR 두 개와 함께 올릴 수 없다. 번역기를 한 번만 올려
두고 ASR 서버들은 `--local-translation-url` 로 이 서버를 부른다.

    python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 \
        --model google/madlad400-3b-mt

    POST /translate  {"text": ..., "target": "ko", "source": "en"}
              ->     {"translation": ..., "source": "en"}
    GET  /health     -> {"status": "ok", "model": ...}
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.translator.local_translator import make_translator  # noqa: E402

logger = logging.getLogger("local-translation-server")


async def handle_translate(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    text = (body.get("text") or "").strip()
    target = (body.get("target") or "").strip()
    source = (body.get("source") or "").strip() or None
    if not text or not target:
        return web.json_response({"translation": "", "source": source or ""})

    translator = request.app["translator"]
    translated, src = await translator.translate(text, target, source)
    return web.json_response({"translation": translated, "source": src})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "model": request.app["model_name"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model", default="google/madlad400-3b-mt",
                        help="번역 모델 이름 또는 경로 (madlad / nllb)")
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
    )
    # 첫 요청에서 로딩 지연이 나지 않게 미리 올린다. 여기서 죽으면 서버를 띄우지
    # 않는다 — 빈 번역을 돌려주는 서버가 살아 있는 편이 더 나쁘다.
    translator.load()

    app = web.Application()
    app["translator"] = translator
    app["model_name"] = args.model
    app.router.add_post("/translate", handle_translate)
    app.router.add_get("/health", handle_health)

    logger.info(f"translation server on {args.host}:{args.port} (model={args.model})")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
