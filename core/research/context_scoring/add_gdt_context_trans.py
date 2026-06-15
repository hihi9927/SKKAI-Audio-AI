"""
기존 eval JSON에 gdt_context_trans 필드 추가.

각 dialog 내에서 이전 N개 원문을 컨텍스트로 붙여 Google Translate 호출.
기존 파일을 in-place로 수정 (또는 --output으로 별도 저장).

Usage:
    python add_gdt_context_trans.py \
        --input evaluation/DailyTalk/results/eval_dailytalk_seg_en_p2_gpt_ctx.json \
        --context-window 5
"""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


async def google_translate(session: aiohttp.ClientSession, text: str, target: str) -> str:
    if not text.strip():
        return ""
    params = {"client": "gtx", "sl": "auto", "tl": target, "dt": "t", "q": text}
    async with session.get(
        "https://translate.googleapis.com/translate_a/single",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as resp:
        data = await resp.json(content_type=None)
        return "".join(item[0] for item in data[0] if item and item[0])


async def translate_with_context(
    session: aiohttp.ClientSession,
    text: str,
    target: str,
    context_originals: list[str],
) -> str:
    if not context_originals:
        return await google_translate(session, text, target)
    combined = "\n".join(context_originals + [text])
    result = await google_translate(session, combined, target)
    lines = [l.strip() for l in result.split("\n") if l.strip()]
    return lines[-1] if lines else result


async def process(input_path: Path, output_path: Path, context_window: int, target: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    async with aiohttp.ClientSession() as session:
        dialog_keys = [k for k in data if k != "stats"]
        total = sum(len(data[k]["data"]) for k in dialog_keys)
        done = 0

        for dialog_id in dialog_keys:
            segs = data[dialog_id]["data"]
            history: list[str] = []

            for seg in segs:
                text = seg.get("seg_text") or seg.get("text", "")
                ctx = history[-context_window:] if context_window > 0 else []
                translation = await translate_with_context(session, text, target, ctx)
                seg["gdt_context_trans"] = translation
                history.append(text)

                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{total} segments done...", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None, help="미지정 시 input 파일 in-place 수정")
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--target-lang", type=str, default="ko")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    asyncio.run(process(input_path, output_path, args.context_window, args.target_lang))


if __name__ == "__main__":
    main()
