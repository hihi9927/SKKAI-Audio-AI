"""
GPT API를 이용한 점진적 컨텍스트 번역 스크립트

seg_text의 <SEG> 분절 단위를 순서대로 번역하되:
  - 첫 번째 세그먼트: 컨텍스트 없이 독립 번역
  - N번째 세그먼트: 앞선 1~N-1 세그먼트의 원문+번역을 컨텍스트로 제공
  - 이미 확정된 앞선 번역은 수정하지 않음

full_trans: 전체 text를 GPT로 단일 번역 (COMET reference용)
seg_trans : 각 세그먼트 번역을 공백으로 이어붙인 결과 (COMET hypothesis용)
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

SEG_CONTEXT_SYSTEM_PROMPT = (
    "You are a precise Korean-to-English translator specializing in spoken, conversational Korean.\n"
    "You will receive already-confirmed preceding segment translations as context, "
    "then a new segment to translate.\n"
    "Rules:\n"
    "- Output ONLY the English translation of the NEW segment. Nothing else.\n"
    "- The preceding translations are FINAL. Do NOT reproduce, paraphrase, or continue them.\n"
    "- The new segment may be a grammatical fragment — translate exactly what is given, do NOT complete it.\n"
    "- Match the register, terminology, and tone established in the preceding translations.\n"
    "- Preserve the natural spoken register (casual/formal) exactly as in the source.\n"
    "- Korean filler words (음, 어, 그) and disfluencies: translate naturally or omit if they carry no meaning, "
    "consistent with spoken English convention.\n"
    "- Proper nouns (names, places, organizations): transliterate consistently with preceding segments.\n"
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


def translate_segments_with_context(
    client: OpenAI,
    segments: list[str],
    model: str,
    delay: float = 0.3,
) -> list[str | None]:
    """
    세그먼트를 순서대로 번역.
    - segments[i]는 앞선 0~i-1 세그먼트의 원문+번역만 컨텍스트로 받음
    - 전체 원문(entry["text"])은 절대 GPT에 전달하지 않음
    - translations 리스트는 이 함수 호출마다 새로 초기화 → 파일(entry) 단위로 컨텍스트 리셋
    """
    translations: list[str | None] = []  # 파일 단위 초기화 — 이전 entry 컨텍스트 미전달

    for i, seg in enumerate(segments):
        if i == 0:
            system_prompt = SEG_SYSTEM_PROMPT
            user_content  = seg
        else:
            context_lines = []
            for j in range(i):
                if translations[j] is None:
                    continue  # 실패 세그먼트는 컨텍스트에서 제외
                context_lines.append(f"[{j+1}] KO: {segments[j]} → EN: {translations[j]}")
            context_str = "\n".join(context_lines)

            system_prompt = SEG_CONTEXT_SYSTEM_PROMPT
            user_content  = (
                f"=== Preceding segments (FINAL — do NOT reproduce or modify) ===\n"
                f"{context_str}\n\n"
                f"=== Translate ONLY this new segment ===\n"
                f"{seg}"
            )

        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_content,
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

    parser = argparse.ArgumentParser(description="GPT 점진적 컨텍스트 번역")
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
        has_full = bool(entry.get("gpt_full_trans"))
        has_seg  = bool(entry.get("gpt_seg_trans"))

        if args.resume and has_full and has_seg:
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

        # ── gpt_seg_trans ───────────────────────────────────────────────────
        if not (args.resume and has_seg):
            seg_text = entry["seg_text"]
            segments = [s.strip() for s in seg_text.split("<SEG>") if s.strip()]

            if len(segments) <= 1:
                # 분절 없으면 gpt_full_trans와 동일하게 처리
                entry["gpt_seg_trans"] = entry.get("gpt_full_trans")
                print(f"  seg : (분절 없음, gpt_full_trans 사용)")
            else:
                translations = translate_segments_with_context(
                    client, segments, args.model, args.delay
                )
                # None인 세그먼트는 제외하고 이어붙임
                valid = [t for t in translations if t is not None]
                entry["gpt_seg_trans"] = " ".join(valid) if valid else None
                print(f"  seg : {entry['gpt_seg_trans']}")

        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
