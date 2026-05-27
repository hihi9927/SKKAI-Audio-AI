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
from datetime import date
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

# core.correct_and_trans.GPTTranslator — 프로덕션 서버와 동일한 번역 모듈
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from core.correct_and_trans import GPTTranslator
    _TRANSLATOR_AVAILABLE = True
except Exception as _e:
    _TRANSLATOR_AVAILABLE = False
    print(f"경고: GPTTranslator 로드 실패 ({_e})", file=sys.stderr)


# ---------------------------------------------------------------------------
# GPT translation (DailyTalk auto-translate)
# ---------------------------------------------------------------------------

async def _gpt_translate_dialogue(
    utterances: list[dict],
    model: str,
    api_key: Optional[str] = None,
) -> list[dict]:
    """대화 하나를 GPTTranslator(correct_and_trans)로 컨텍스트 유지 번역.

    - 프로덕션 서버와 동일한 프롬프트·컨텍스트 포맷 사용
    - <SEG> 포함 발화: 각 segment를 현재 컨텍스트 스냅샷으로 독립 번역
      (같은 발화 내 segment 간 결과는 컨텍스트에 영향 없음)
    - 발화 완료 후 (corrected, translation) 쌍을 컨텍스트에 추가
    """
    translator = GPTTranslator(api_key=api_key, model=model)
    context: list[tuple[str, str]] = []  # (corrected_original, translation)
    updated: list[dict] = []

    for utt in utterances:
        original = (utt.get("text") or "").strip()
        seg_text = (utt.get("seg_text") or original).strip()
        ctx_snapshot = context[-translator.max_context:] or None

        if "<SEG>" in seg_text:
            parts = [p.strip() for p in seg_text.split("<SEG>") if p.strip()]
            corrected_parts, trans_parts = [], []
            for part in parts:
                corrected, translation, _ = await translator.correct_and_translate(
                    text=part,
                    source_lang_name="English",
                    target_lang_code="ko",
                    context=ctx_snapshot,
                )
                corrected_parts.append(corrected)
                trans_parts.append(translation)
            combined_corrected = " ".join(corrected_parts)
            combined_translation = " ".join(trans_parts)
        else:
            combined_corrected, combined_translation, _ = await translator.correct_and_translate(
                text=original,
                source_lang_name="English",
                target_lang_code="ko",
                context=ctx_snapshot,
            )

        context.append((combined_corrected, combined_translation))

        new_utt = dict(utt)
        new_utt["gpt_seg_trans"] = combined_translation
        updated.append(new_utt)

    return updated


async def _ensure_dailytalk_translated(
    metric_path: Path,
    model: str,
    concurrency: int,
    api_key: Optional[str] = None,
) -> Path:
    """DailyTalk 파일에 gpt_seg_trans가 없으면 자동 번역 후 _gpt_ctx.json 경로 반환.

    - 이미 _gpt_ctx.json 파일이 있으면 재사용
    - DailyTalk 포맷이 아니면 원래 경로 그대로 반환
    """
    with open(metric_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if _detect_format(data) != "dailytalk":
        return metric_path

    ctx_path = metric_path.parent / f"{metric_path.stem}_gpt_ctx.json"

    if ctx_path.exists():
        print(f"  기존 번역 재사용: {ctx_path.name}", flush=True)
        return ctx_path

    dialogue_keys = sorted([k for k in data if k.isdigit()], key=int)
    needs_translation = any(
        not all("gpt_seg_trans" in u for u in data[k].get("data", []))
        for k in dialogue_keys
    )
    if not needs_translation:
        return metric_path

    if not _TRANSLATOR_AVAILABLE:
        print("  경고: GPTTranslator 사용 불가 — 번역 건너뜀", file=sys.stderr)
        return metric_path

    print(
        f"  gpt_seg_trans 없음 → GPTTranslator로 자동 번역 중 ({len(dialogue_keys)}개 대화)...",
        flush=True,
    )
    semaphore = asyncio.Semaphore(concurrency)
    output_data: dict = {k: v for k, v in data.items() if not k.isdigit()}
    completed = 0

    async def _translate_one(key: str) -> None:
        nonlocal completed
        async with semaphore:
            updated = await _gpt_translate_dialogue(
                data[key].get("data", []), model, api_key
            )
            output_data[key] = {"data": updated}
            completed += 1
            print(f"  번역 진행: {completed}/{len(dialogue_keys)}", flush=True)

    await asyncio.gather(*[_translate_one(k) for k in dialogue_keys])

    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  번역 저장: {ctx_path.name}", flush=True)
    return ctx_path


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

Important calibration: Because the system has no look-ahead, some degree of \
pronoun ambiguity and occasional terminology drift is structurally unavoidable. \
Do NOT penalise for information the system could not have known at translation \
time. Focus instead on whether the system actively exploits the context it \
already has (what was said before) and avoids unnecessary inconsistencies.

Your task: score how well the translations maintain CONTEXTUAL CONSISTENCY \
across the entire passage (audiobook chapter or spoken dialogue).

Evaluation criteria (4 criteria, weigh equally):

1. Cross-sentence coherence – Are pronouns, referring expressions, and \
narrative thread carried forward correctly using the context already available? \
Do cause-effect relationships and temporal sequences remain intact across \
segments?
2. Terminology consistency – Are recurring domain terms, proper nouns, and \
character names rendered the same way throughout the chapter?
3. Topical & contextual word sense – Does the translation preserve the \
chapter's subject and theme? When a word is ambiguous in isolation, is it \
translated according to the meaning established by prior context (e.g., a \
technical term in a medical chapter vs. the same word in casual speech)?
4. Tone and register consistency – Is the formality level and narrative voice \
kept stable across the chapter? Penalise only register shifts that are \
unexplained by the source text.

Scoring:
- Use any integer from 0 to 100. Do NOT round to multiples of 5 or 10.
- Base your score on the specific evidence you observe in this chapter; \
  do not default to a "safe" middle score.
- Deduct points proportionally to how often and how severely context breaks \
  occur, weighted by whether the system could have avoided the error given \
  the preceding context.

Reference points (calibrated for real-time streaming systems):
  80-100  Excellent – context actively exploited throughout; only minor or \
unavoidable slips
  60-79   Good – context mostly maintained with expected real-time limitations; \
a few avoidable lapses
  40-59   Moderate – noticeable context issues beyond what streaming constraints \
explain; comprehension occasionally affected
  20-39   Poor – frequent avoidable context breaks that hinder understanding \
of the chapter
   0-19   Very poor – translations appear largely isolated with little evidence \
of cross-sentence context use

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

def _detect_format(data: dict) -> str:
    """JSON 구조를 보고 포맷을 반환: 'librispeech' | 'dailytalk' | 'unknown'"""
    if "raw_results" in data:
        return "librispeech"
    numeric_keys = [k for k in data if k.isdigit()]
    if numeric_keys and isinstance(data.get(numeric_keys[0]), dict):
        if "data" in data[numeric_keys[0]]:
            return "dailytalk"
    return "unknown"


def _load_librispeech(data: dict, metric_json: Path) -> dict[str, list[tuple[str, str]]]:
    """LibriSpeech metric.json → {chapter_id: [(src_en, tgt_ko), ...]}"""
    raw_results: list[dict] = data.get("raw_results", [])
    if not raw_results:
        print(f"  경고: {metric_json} 에 raw_results 가 없습니다.", file=sys.stderr)
        return {}

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


def _load_dailytalk(
    data: dict,
    translation: str = "gpt",
) -> dict[str, list[tuple[str, str]]]:
    """DailyTalk eval JSON → {dialogue_NNNN: [(src_en, tgt_ko), ...]}

    translation:
      "gpt"  → gpt_seg_trans (GPT 컨텍스트 번역), 없으면 gdt_seg_trans 폴백
      "gdt"  → gdt_seg_trans (기존 번역) 고정
    """
    field = "gdt_seg_trans" if translation == "gdt" else "gpt_seg_trans"
    fallback = "gdt_seg_trans" if translation == "gpt" else None

    chapters: dict[str, list[tuple[str, str]]] = {}
    for key in sorted((k for k in data if k.isdigit()), key=int):
        utterances = data[key].get("data", [])
        pairs: list[tuple[str, str]] = []
        for utt in utterances:
            src = (utt.get("text") or "").strip()
            tgt = (utt.get(field) or (utt.get(fallback) if fallback else None) or "").strip()
            if src and tgt:
                pairs.append((src, tgt))
        if pairs:
            chapters[f"dialogue_{int(key):04d}"] = pairs
    return chapters


def load_chapters(
    metric_json: Path,
    translation: str = "gpt",
) -> dict[str, list[tuple[str, str]]]:
    """
    포맷 자동 감지 후 {unit_id: [(src_en, tgt_ko), ...]} 반환.

    지원 포맷:
    - LibriSpeech metric.json  ('raw_results' 키 존재)
    - DailyTalk eval JSON      (숫자 키 "0","1",... + 'data' 배열)

    translation: DailyTalk 전용 ("gpt" | "gdt")
    """
    with open(metric_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = _detect_format(data)
    if fmt == "librispeech":
        return _load_librispeech(data, metric_json)
    elif fmt == "dailytalk":
        return _load_dailytalk(data, translation)
    else:
        print(
            f"  경고: {metric_json} 의 포맷을 인식할 수 없습니다 "
            f"(raw_results 도 없고 숫자+data 구조도 없음).",
            file=sys.stderr,
        )
        return {}


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

async def run_scoring(
    metric_paths: list[Path],
    model: str,
    output: Optional[Path],
    api_key: Optional[str],
    concurrency: int,
    translation: str = "gpt",
) -> None:
    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    all_run_results = []

    for metric_path in metric_paths:
        print(f"\n{'='*60}", flush=True)
        print(f"파일: {metric_path}", flush=True)
        # DailyTalk + gpt 모드: gpt_seg_trans 없으면 자동 번역
        if translation == "gpt":
            metric_path = await _ensure_dailytalk_translated(
                metric_path, model, concurrency, api_key
            )
        chapters = load_chapters(metric_path, translation)
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
        total_pairs_scored = sum(r["n_pairs"] for r in chapter_results)
        avg_score = (
            sum(r["score"] * r["n_pairs"] for r in chapter_results) / total_pairs_scored
        )

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
        help=(
            "결과 JSON 저장 경로. "
            "미지정 시 research/context_scoring/results/ 아래에 자동 저장: "
            "단일 파일이면 <run_name>_YYYYMMDD.json, "
            "여러 파일이면 comparison_YYYYMMDD.json"
        ),
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
    parser.add_argument(
        "--translation",
        choices=["gpt", "gdt"],
        default="gpt",
        help=(
            "DailyTalk 전용 번역 소스 선택 (기본: gpt). "
            "gpt: GPT 컨텍스트 번역 (gpt_seg_trans, 없으면 자동 생성); "
            "gdt: 기존 번역 그대로 사용 (gdt_seg_trans)"
        ),
    )

    args = parser.parse_args()

    for p in args.metric_jsons:
        if not p.exists():
            print(f"오류: 파일을 찾을 수 없음 — {p}", file=sys.stderr)
            sys.exit(1)

    # --output 미지정 시 자동 경로 생성
    output = args.output
    if output is None:
        today = date.today().strftime("%Y%m%d")
        results_dir = Path(__file__).parent / "results"
        if len(args.metric_jsons) == 1:
            label = args.metric_jsons[0].parent.name  # e.g. "run_10"
            output = results_dir / f"{label}_{today}.json"
        else:
            output = results_dir / f"comparison_{today}.json"

    asyncio.run(
        run_scoring(
            metric_paths=args.metric_jsons,
            model=args.model,
            output=output,
            api_key=args.api_key,
            concurrency=args.concurrency,
            translation=args.translation,
        )
    )


if __name__ == "__main__":
    main()
