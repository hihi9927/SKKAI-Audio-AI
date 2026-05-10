"""
GPT API를 이용한 컨텍스트 없는 분절 단위 번역 스크립트

seg_text의 <SEG> 분절 단위를 각각 독립적으로 번역:
  - 모든 세그먼트를 컨텍스트 없이 독립 번역
  - 앞선 세그먼트의 원문·번역을 전달하지 않음

full_trans    : 전체 text를 GPT로 단일 번역 (COMET reference용)
seg_nc_trans  : 각 세그먼트 독립 번역을 공백으로 이어붙인 결과 (COMET hypothesis용)
"""

import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI


SEG_SYSTEM_PROMPT = (
    "You are a precise Korean-to-English translator specializing in spoken, conversational Korean.\n"
    "Rules:\n"
    "- Output ONLY the English translation. No explanations, notes, or alternatives.\n"
    "- The input may be a sentence fragment — translate exactly what is given. Do NOT complete or extend it.\n"
    "- Preserve the natural spoken register (casual/formal) exactly as in the source.\n"
    "- Korean filler words (음, 어, 그) and disfluencies: translate naturally or omit if they carry no meaning, "
    "consistent with spoken English convention.\n"
    "- Proper nouns (names, places, organizations): transliterate consistently.\n"
    "- Translate faithfully — do not add, omit, or infer meaning beyond what is stated."
)

FULL_SYSTEM_PROMPT = (
    "You are a precise Korean-to-English translator specializing in spoken, conversational Korean.\n"
    "Rules:\n"
    "- Output ONLY the English translation. No explanations, notes, or alternatives.\n"
    "- Translate the entire passage as one coherent unit.\n"
    "- Preserve the natural spoken register (casual/formal) exactly as in the source.\n"
    "- Korean filler words (음, 어, 그) and disfluencies: translate naturally or omit if they carry no meaning, "
    "consistent with spoken English convention.\n"
    "- Proper nouns (names, places, organizations): transliterate consistently.\n"
    "- Translate faithfully — do not add, omit, or infer meaning beyond what is stated."
)


def translate_full(client: OpenAI, text: str, model: str) -> str | None:
    try:
        response = client.responses.create(
            model=model,
            instructions=FULL_SYSTEM_PROMPT,
            input=text,
            reasoning={"effort": "none"},
            temperature=0,
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"  [full 번역 실패] {e}")
        return None


def translate_segments_without_context(
    client: OpenAI,
    segments: list[str],
    model: str,
    delay: float = 0.3,
) -> list[str | None]:
    """각 세그먼트를 컨텍스트 없이 독립적으로 번역."""
    translations: list[str | None] = []

    for i, seg in enumerate(segments):
        try:
            response = client.responses.create(
                model=model,
                instructions=SEG_SYSTEM_PROMPT,
                input=seg,
                reasoning={"effort": "none"},
                temperature=0,
            )
            t = response.output_text.strip()
            translations.append(t)
            print(f"    seg[{i+1}] KO: {seg}")
            print(f"    seg[{i+1}] EN: {t}")
        except Exception as e:
            print(f"    [seg[{i+1}] 번역 실패] {e}")
            translations.append(None)

        if delay > 0 and i < len(segments) - 1:
            time.sleep(delay)

    return translations


def main():
    _base        = Path(__file__).resolve().parent.parent.parent.parent / "evaluation" / "KsponSpeech"
    _results_dir = _base / "results"

    parser = argparse.ArgumentParser(description="GPT 컨텍스트 없는 분절 단위 번역")
    parser.add_argument("--input",     type=str, required=True,
                        help="results 디렉토리 내 파일명 (예: eval_clean_seg.json)")
    parser.add_argument("--output",    type=str, default=None,
                        help="출력 파일명 (기본: 입력과 동일 — 덮어씀)")
    parser.add_argument("--input-dir", type=str, default=str(_results_dir),
                        help="입력 파일 디렉토리")
    parser.add_argument("--model",     type=str, default="gpt-5.4-nano",
                        help="OpenAI 모델명")
    parser.add_argument("--api-key",   type=str, default=None,
                        help="OpenAI API 키 (미입력 시 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--delay",     type=float, default=0.3,
                        help="번역 요청 간 딜레이(초)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="이미 번역된 항목도 재처리")
    parser.add_argument("--skip-full", action="store_true",
                        help="full_trans 번역 건너뜀 (이미 있을 때 유용)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")

    client = OpenAI(api_key=api_key)

    input_path  = Path(args.input_dir) / args.input
    output_path = input_path if args.output is None else Path(args.input_dir) / args.output

    raw  = json.loads(input_path.read_text(encoding="utf-8"))
    data = raw["data"]

    total = len(data)
    for i, entry in enumerate(data):
        has_full   = bool(entry.get("gpt_full_trans"))
        has_seg_nc = bool(entry.get("gpt_seg_nc_trans"))

        if args.resume and has_full and has_seg_nc:
            print(f"[{i+1}/{total}] 건너뜀: {entry['file']}")
            continue

        if "seg_text" not in entry:
            print(f"[{i+1}/{total}] seg_text 없음, 스킵: {entry['file']}")
            continue

        print(f"[{i+1}/{total}] {entry['file']}")
        print(f"  원문: {entry['text']}")

        # ── gpt_full_trans ──────────────────────────────────────────────────
        if not args.skip_full and not (args.resume and has_full):
            entry["gpt_full_trans"] = translate_full(client, entry["text"], args.model)
            print(f"  gpt_full: {entry['gpt_full_trans']}")
            if args.delay > 0:
                time.sleep(args.delay)

        # ── gpt_seg_nc_trans ────────────────────────────────────────────────
        if not (args.resume and has_seg_nc):
            seg_text = entry["seg_text"]
            segments = [s.strip() for s in seg_text.split("<SEG>") if s.strip()]

            if len(segments) <= 1:
                entry["gpt_seg_nc_trans"] = entry.get("gpt_full_trans")
                print(f"  seg_nc: (분절 없음, gpt_full_trans 사용)")
            else:
                translations = translate_segments_without_context(
                    client, segments, args.model, args.delay
                )
                valid = [t for t in translations if t is not None]
                entry["gpt_seg_nc_trans"] = " ".join(valid) if valid else None
                print(f"  seg_nc: {entry['gpt_seg_nc_trans']}")

        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
