#!/usr/bin/env python3
"""
Context-aware translation quality scorer.

Loads one or more metric.json result files, groups utterances by LibriSpeech
chapter, then asks GPT to rate how well each chapter's translations preserve
cross-sentence context (0-100). Final score is the mean of chapter scores.

Usage:
    python research/context_scoring/score_context.py \
        evaluation/LibriSpeech/results/.../metric.json \
        [evaluation/LibriSpeech/results/.../metric.json ...] \
        --output research/context_scoring/results/run_name.json

    Multiple metric.json paths → scores are computed independently per file
    and summarised together (useful for side-by-side comparison).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# GPT scoring
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional translation evaluator specialising in context-aware \
machine translation quality.

You will receive a sequence of English sentences from one chapter of an \
audiobook, paired with their Korean translations. These translations were \
produced sentence-by-sentence by a real-time speech translation system \
that does NOT have access to future sentences — each segment was translated \
the moment it was spoken.

Your task: score how well the translations maintain CONTEXTUAL CONSISTENCY \
across the entire chapter.

Evaluation criteria (weigh equally):

Linguistic criteria:
1. Coreference consistency – Are pronouns and referring expressions (he/she/it/\
they, demonstratives) translated consistently according to who or what has been \
established in prior segments?
2. Terminology consistency – Are recurring domain terms, proper nouns, and \
character names rendered the same way throughout?

Content-centred criteria:
3. Topical coherence – Does the translation preserve the chapter's subject and \
theme? When a topic shifts or deepens, does the translation reflect that \
progression rather than treating each sentence as an isolated fact?
4. Contextual word sense – When a word is ambiguous in isolation, is it \
translated according to the meaning established by the surrounding context \
(e.g., a technical term in a medical chapter vs. the same word in casual speech)?
5. Narrative/logical continuity – Are cause-effect relationships, temporal \
sequences, and references to earlier events in the chapter correctly carried \
forward? Does the translation make it clear that later sentences build on \
earlier ones?
6. Tone and register consistency – Is the formality level, emotional register, \
and narrative voice kept stable across the chapter? Sudden unexplained shifts \
in register should be penalised.

Scoring:
- Use any integer from 0 to 100. Do NOT round to multiples of 5 or 10.
- Base your score on the specific evidence you observe in this chapter; \
  do not default to a "safe" middle score.
- Deduct points proportionally to how often and how severely context breaks occur.

Reference points:
  90-100  Excellent – strong context awareness throughout, minor slips only
  70-89   Good – context generally well maintained, a few lapses
  50-69   Moderate – noticeable context loss in places
  30-49   Poor – frequent context breaks affecting comprehension
   0-29   Very poor – translations appear largely isolated from each other

Respond with ONLY a valid JSON object (no markdown, no extra text):
{"score": <integer 0-100>, "reasoning": "<two to three sentences citing specific evidence from the chapter>"}\
"""


async def score_chapter(
    client: AsyncOpenAI,
    chapter_id: str,
    pairs: list[tuple[str, str]],
    model: str,
    max_retries: int = 5,
) -> dict:
    """Score one chapter. Returns {"chapter_id", "score", "reasoning", "n_pairs"}."""
    lines = []
    for i, (src, tgt) in enumerate(pairs, start=1):
        lines.append(f"{i}. [{src}] → [{tgt}]")
    chapter_text = "\n".join(lines)

    user_content = f"Chapter: {chapter_id}\n\n{chapter_text}"

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            score = int(result.get("score", 0))
            score = max(0, min(100, score))
            return {
                "chapter_id": chapter_id,
                "score": score,
                "reasoning": result.get("reasoning", ""),
                "n_pairs": len(pairs),
            }
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = _parse_retry_after(msg, attempt)
                print(f"  [rate limit] {wait:.1f}s 대기 후 재시도 ({attempt + 1}/{max_retries})", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"  [error] chapter {chapter_id}: {e}", flush=True)
                raise

    raise RuntimeError(f"최대 재시도({max_retries}) 초과 — chapter {chapter_id}")


def _parse_retry_after(msg: str, attempt: int) -> float:
    m = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", msg)
    if m:
        secs = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
        return max(secs + 0.2, 1.0)
    return float(2 ** attempt)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_chapters(metric_json: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Returns {chapter_id: [(original_en, translation_ko), ...]} sorted by file_id.

    Each file's segment_metrics are appended in order within the file.
    Files within a chapter are sorted by file_id (lexicographic = utterance order).
    """
    with open(metric_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_results: list[dict] = data.get("raw_results", [])
    if not raw_results:
        print(f"  경고: {metric_json} 에 raw_results 가 없습니다.", file=sys.stderr)
        return {}

    # Group files by chapter
    chapter_files: dict[str, list[dict]] = {}
    for row in raw_results:
        fid = row.get("file_id", "")
        parts = fid.split("-")
        chapter_id = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else parts[0]
        chapter_files.setdefault(chapter_id, []).append(row)

    chapters: dict[str, list[tuple[str, str]]] = {}
    for chapter_id, rows in chapter_files.items():
        rows.sort(key=lambda r: r.get("file_id", ""))
        pairs: list[tuple[str, str]] = []
        for row in rows:
            for seg in row.get("segment_metrics") or []:
                src = (seg.get("text") or "").strip()
                tgt = (seg.get("translation") or "").strip()
                if src and tgt:
                    pairs.append((src, tgt))
        if pairs:
            chapters[chapter_id] = pairs

    return chapters


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

async def run_scoring(
    metric_paths: list[Path],
    model: str,
    output: Optional[Path],
    api_key: Optional[str],
    concurrency: int,
) -> None:
    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    all_run_results = []

    for metric_path in metric_paths:
        print(f"\n{'='*60}", flush=True)
        print(f"파일: {metric_path}", flush=True)
        chapters = load_chapters(metric_path)
        if not chapters:
            print("  챕터 데이터 없음, 건너뜀.", flush=True)
            continue

        print(f"  챕터 수: {len(chapters)}", flush=True)
        total_pairs = sum(len(v) for v in chapters.values())
        print(f"  총 세그먼트 쌍: {total_pairs}", flush=True)

        # Score chapters with bounded concurrency
        semaphore = asyncio.Semaphore(concurrency)
        chapter_results: list[dict] = []

        async def _score_one(ch_id: str, pairs: list[tuple[str, str]]) -> None:
            async with semaphore:
                print(f"  채점 중: {ch_id} ({len(pairs)}쌍) ...", flush=True)
                t0 = time.perf_counter()
                result = await score_chapter(client, ch_id, pairs, model)
                elapsed = time.perf_counter() - t0
                print(
                    f"  [{ch_id}] score={result['score']}  "
                    f"reason={result['reasoning'][:60]}  ({elapsed:.1f}s)",
                    flush=True,
                )
                chapter_results.append(result)

        tasks = [_score_one(ch_id, pairs) for ch_id, pairs in sorted(chapters.items())]
        await asyncio.gather(*tasks)

        chapter_results.sort(key=lambda r: r["chapter_id"])
        avg_score = mean(r["score"] for r in chapter_results)

        print(f"\n  ── 결과 요약 ──", flush=True)
        for r in chapter_results:
            print(f"  {r['chapter_id']:20s}  score={r['score']:3d}  pairs={r['n_pairs']}", flush=True)
        print(f"\n  평균 컨텍스트 점수: {avg_score:.2f} / 100", flush=True)

        all_run_results.append({
            "metric_json": str(metric_path),
            "chapter_scores": chapter_results,
            "avg_context_score": round(avg_score, 4),
            "n_chapters": len(chapter_results),
            "total_pairs": total_pairs,
        })

    # Side-by-side summary when comparing multiple files
    if len(all_run_results) > 1:
        print(f"\n{'='*60}", flush=True)
        print("비교 요약:", flush=True)
        for run in all_run_results:
            label = Path(run["metric_json"]).parent.name
            print(f"  {label:40s}  avg={run['avg_context_score']:.2f}", flush=True)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = [
            {
                "label": Path(run["metric_json"]).parent.name,
                "metric_json": run["metric_json"],
                "avg_context_score": run["avg_context_score"],
                "n_chapters": run["n_chapters"],
                "total_pairs": run["total_pairs"],
            }
            for run in all_run_results
        ]
        payload = {
            "model": model,
            "summary": summary,
            "runs": all_run_results,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {output}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT 기반 컨텍스트 번역 품질 채점 (LibriSpeech 챕터 단위)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "metric_jsons",
        nargs="+",
        type=Path,
        metavar="METRIC_JSON",
        help="채점할 metric.json 파일 경로 (여러 개 지정 시 비교 채점)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="채점에 사용할 OpenAI 모델 (기본: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 JSON 저장 경로 (미지정 시 저장 안 함)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API 키 (미지정 시 OPENAI_API_KEY 환경변수 사용)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="동시에 채점할 챕터 수 (기본: 3, rate limit 주의)",
    )

    args = parser.parse_args()

    for p in args.metric_jsons:
        if not p.exists():
            print(f"오류: 파일을 찾을 수 없음 — {p}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(
        run_scoring(
            metric_paths=args.metric_jsons,
            model=args.model,
            output=args.output,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
