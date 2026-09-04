#!/usr/bin/env python3
"""아랍어 ASR 오프라인 평가 — WER + **모델이 실제로 찍는 문장부호 분포**.

왜 별도 스크립트인가
--------------------
`test_ast.py` 는 WebSocket 서버를 띄우고 BLEU 까지 내는 AST 하네스다. 지금 알고
싶은 건 두 가지뿐이라 서버가 필요 없다.

1. 요르단 구어(Casablanca)와 MSA(FLEURS ar_eg) 사이 WER 격차가 얼마나 되나.
2. **모델이 아랍어에 어떤 종결부호를 찍나.** dot-commit 은 ASR 출력의 부호에
   걸리므로, 레퍼런스 전사의 부호 관행이 아니라 모델 출력을 봐야 한다.
   Casablanca 요르단 test 레퍼런스는 `؟`122 `!`137 `.`2 로, ASCII 마침표를
   사실상 안 쓴다. 모델도 그런지는 별개 문제다.

아랍어 WER 정규화
-----------------
아랍어는 표기 변이가 커서 raw WER 이 실제 인식 품질을 과소평가한다. 표준 관행대로
정규화본을 함께 낸다 — 어느 쪽도 단독으로는 오해를 부르므로 **둘 다** 출력한다.

  - 타슈킬(diacritics) 제거: 발음기호는 보통 표기하지 않는다
  - 알레프 통일: أ إ آ ٱ → ا
  - 타 마르부타 ة → ه, 알레프 막수라 ى → ي
  - 타트윌(ـ) 제거, 문장부호 제거

사용:

    python evaluation/ast/asr_eval_arabic.py \
        --manifest evaluation/ast/manifests/casablanca_jordan_test.jsonl \
        --out evaluation/ast/results/asr_ar/casablanca_jordan_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer

# 종결부호 후보 — dot-commit 이 관심 갖는 것들. 아랍어 물음표 `؟`(U+061F)와
# 아랍어 쉼표 `،`(U+060C), 아랍어 세미콜론 `؛`(U+061B)을 ASCII 와 나란히 센다.
PUNCT_OF_INTEREST = ".!?،؛؟。！？,;:…"

_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = re.compile(r"ـ")
_PUNCT = re.compile(r"[^\w\s؀-ۿ]|[،؛؟٪-٭۔]")


def normalize_arabic(text: str) -> str:
    """WER 비교용 아랍어 정규화. 표기 변이를 접어 실제 인식 오류만 남긴다."""
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_punct_only(text: str) -> str:
    """부호만 떼고 표기 변이는 그대로 — raw WER 용."""
    text = _PUNCT.sub(" ", unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip()


def punct_stats(texts: list[str]) -> dict:
    """문장부호 총계 + 종결부호를 가진 발화 비율."""
    counter: collections.Counter = collections.Counter()
    for t in texts:
        counter.update(c for c in t if c in PUNCT_OF_INTEREST)
    # 종결부호(문장을 끝낼 수 있는 것)로 끝나는 발화 수
    terminal = ".!?؟。！？"
    ends_with_terminal = sum(1 for t in texts if t.rstrip() and t.rstrip()[-1] in terminal)
    has_any_terminal = sum(1 for t in texts if any(c in terminal for c in t))
    return {
        "counts": dict(counter),
        "utts_ending_with_terminal": ends_with_terminal,
        "utts_containing_terminal": has_any_terminal,
        "n_utts": len(texts),
    }


async def run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(args.repo_root).expanduser().resolve() / "Qwen3-ASR"))
    from qwen_asr import Qwen3ASRModel

    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]
    print(f"[로드] {len(items)}개 발화 — {args.manifest}")

    print(f"[모델] {args.model} (backend={args.backend})")
    if args.backend == "vllm":
        # transformers 백엔드는 GB10 에서 4.2초 오디오에 30초 넘게 걸린다(측정치).
        # 848 발화면 7시간대라 실용적이지 않아 vLLM 을 기본으로 쓴다.
        asr = Qwen3ASRModel.LLM(model=args.model, **json.loads(args.backend_kwargs))
    else:
        asr = Qwen3ASRModel.from_pretrained(args.model)

    hyps: list[str] = []
    refs: list[str] = []
    langs: list[str] = []
    per_utt = []

    t0 = time.time()
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        results = await asr.transcribe(
            audio=[b["wav"] for b in batch],
            language=args.language,          # None 이면 자동 감지
            return_time_stamps=False,
        )
        for b, r in zip(batch, results):
            hyps.append(r.text or "")
            refs.append(b["src_text"])
            langs.append(r.language or "")
            per_utt.append({
                "utt_id": b["utt_id"],
                "duration": b["duration"],
                "ref": b["src_text"],
                "hyp": r.text or "",
                "detected_language": r.language or "",
            })
        done = min(i + args.batch_size, len(items))
        if done % (args.batch_size * args.log_every) == 0 or done == len(items):
            el = time.time() - t0
            print(f"  {done}/{len(items)} — {el:.0f}s 경과, "
                  f"{el / done:.2f}s/발화", flush=True)

    # ---- WER ----
    raw_refs = [strip_punct_only(t) for t in refs]
    raw_hyps = [strip_punct_only(t) for t in hyps]
    nrm_refs = [normalize_arabic(t) for t in refs]
    nrm_hyps = [normalize_arabic(t) for t in hyps]

    # 빈 참조는 jiwer 가 0으로 나눈다 — 제외하고 개수를 남긴다.
    def paired(rs, hs):
        pairs = [(r, h) for r, h in zip(rs, hs) if r.strip()]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    rr, rh = paired(raw_refs, raw_hyps)
    nr, nh = paired(nrm_refs, nrm_hyps)

    summary = {
        "manifest": str(args.manifest),
        "model": args.model,
        "forced_language": args.language,
        "n_utts": len(items),
        "n_scored": len(rr),
        "audio_hours": round(sum(b["duration"] for b in items) / 3600, 3),
        "wer_raw": round(jiwer.wer(rr, rh) * 100, 2),
        "cer_raw": round(jiwer.cer(rr, rh) * 100, 2),
        "wer_normalized": round(jiwer.wer(nr, nh) * 100, 2),
        "cer_normalized": round(jiwer.cer(nr, nh) * 100, 2),
        "detected_languages": dict(collections.Counter(langs)),
        "punct_hypothesis": punct_stats(hyps),
        "punct_reference": punct_stats(refs),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "utterances": per_utt}, f,
                  ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"발화 {summary['n_utts']}개 / 오디오 {summary['audio_hours']}h")
    print(f"WER  raw {summary['wer_raw']}%   정규화 {summary['wer_normalized']}%")
    print(f"CER  raw {summary['cer_raw']}%   정규화 {summary['cer_normalized']}%")
    print(f"감지 언어: {summary['detected_languages']}")
    print(f"\n가설 문장부호: {summary['punct_hypothesis']['counts']}")
    print(f"  종결부호로 끝난 발화 "
          f"{summary['punct_hypothesis']['utts_ending_with_terminal']}/{len(hyps)}")
    print(f"참조 문장부호: {summary['punct_reference']['counts']}")
    print(f"\n결과: {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="아랍어 ASR WER + 문장부호 분포 측정")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--language", default="Arabic",
                   help="강제 언어. 자동 감지를 보려면 --language '' 로 비운다")
    p.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    p.add_argument("--backend-kwargs", default="{}",
                   help="vLLM 생성자에 넘길 JSON (예: '{\"gpu_memory_utilization\": 0.8}')")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    a = p.parse_args()
    if not a.language:
        a.language = None
    return a


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
