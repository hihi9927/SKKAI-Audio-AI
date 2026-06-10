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
import re
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


def _chat_translate(client: OpenAI, system_prompt: str, user_content: str, model: str, provider: str) -> tuple[str, float]:
    """Chat Completions API로 번역. (결과, 소요시간_초) 반환."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": ""},  # prefill: 설명 없이 번역만 출력 유도
        ],
        temperature=0,
    )
    if provider == "ollama":
        kwargs["extra_body"] = {"think": False}
    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    return response.choices[0].message.content.strip(), elapsed


def translate_full(client: OpenAI, text: str, model: str, provider: str = "openai") -> str | None:
    try:
        if provider == "ollama":
            result, elapsed = _chat_translate(client, FULL_SYSTEM_PROMPT, text, model, provider)
            print(f"    [full {elapsed:.2f}s]")
            return result
        t0 = time.perf_counter()
        response = client.responses.create(
            model=model,
            instructions=FULL_SYSTEM_PROMPT,
            input=text,
            reasoning={"effort": "none"},
            temperature=0,
        )
        print(f"    [full {time.perf_counter() - t0:.2f}s]")
        return response.output_text.strip()
    except Exception as e:
        print(f"  [full 번역 실패] {e}")
        return None


def translate_segments_with_context(
    client: OpenAI,
    segments: list[str],
    model: str,
    delay: float = 0.3,
    provider: str = "openai",
) -> list[str | None]:
    """
    세그먼트를 순서대로 번역.
    - segments[i]는 앞선 0~i-1 세그먼트의 원문+번역만 컨텍스트로 받음
    - 전체 원문(entry["text"])은 절대 GPT에 전달하지 않음
    - translations 리스트는 이 함수 호출마다 새로 초기화 → 파일(entry) 단위로 컨텍스트 리셋
    """
    translations: list[str | None] = []

    for i, seg in enumerate(segments):
        if i == 0:
            system_prompt = SEG_SYSTEM_PROMPT
            user_content  = seg
        else:
            context_lines = []
            for j in range(i):
                if translations[j] is None:
                    continue
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
            if provider == "ollama":
                t, elapsed = _chat_translate(client, system_prompt, user_content, model, provider)
            else:
                t0 = time.perf_counter()
                response = client.responses.create(
                    model=model,
                    instructions=system_prompt,
                    input=user_content,
                    reasoning={"effort": "none"},
                    temperature=0,
                )
                elapsed = time.perf_counter() - t0
                t = response.output_text.strip()
            translations.append(t)
            print(f"    seg[{i+1}] KO: {seg}")
            print(f"    seg[{i+1}] EN: {t}  [{elapsed:.2f}s]")
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
    parser.add_argument("--model",     type=str, default=None,
                        help="모델명 (openai 기본값: gpt-5.4-nano, ollama 기본값: exaone3.5:32b)")
    parser.add_argument("--provider",  type=str, default="openai", choices=["openai", "ollama"],
                        help="API 제공자 (openai 또는 ollama)")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama 호스트 (기본값: OLLAMA_HOST 환경변수 또는 localhost:11434)")
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

    if args.provider == "ollama":
        host = args.ollama_host or os.environ.get("OLLAMA_HOST", "localhost:11434")
        client = OpenAI(base_url=f"http://{host}/v1", api_key="ollama")
        model = args.model or "exaone3.5:32b"
        prefix = re.sub(r'[.\-]', '_', model.split(':')[0])
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        client = OpenAI(api_key=api_key)
        model = args.model or "gpt-5.4-nano"
        prefix = "gpt"

    field_full = f"{prefix}_full_trans"
    field_seg  = f"{prefix}_seg_trans"
    print(f"저장 필드: {field_full}, {field_seg}")

    input_path  = Path(args.input_dir) / args.input
    output_path = input_path if args.output is None else Path(args.input_dir) / args.output

    raw  = json.loads(input_path.read_text(encoding="utf-8"))
    data = raw["data"]

    total = len(data)
    for i, entry in enumerate(data):
        has_full = bool(entry.get(field_full))
        has_seg  = bool(entry.get(field_seg))

        if args.resume and has_full and has_seg:
            print(f"[{i+1}/{total}] 건너뜀: {entry['file']}")
            continue

        if "seg_text" not in entry:
            print(f"[{i+1}/{total}] seg_text 없음, 스킵: {entry['file']}")
            continue

        print(f"[{i+1}/{total}] {entry['file']}")
        print(f"  원문: {entry['text']}")

        # ── full_trans ──────────────────────────────────────────────────────
        if not args.skip_full and not (args.resume and has_full):
            entry[field_full] = translate_full(client, entry["text"], model, args.provider)
            print(f"  {field_full}: {entry[field_full]}")
            if args.delay > 0:
                time.sleep(args.delay)

        # ── seg_trans ───────────────────────────────────────────────────────
        if not (args.resume and has_seg):
            seg_text = entry["seg_text"]
            segments = [s.strip() for s in seg_text.split("<SEG>") if s.strip()]

            if len(segments) <= 1:
                entry[field_seg] = entry.get(field_full)
                print(f"  {field_seg}: (분절 없음, {field_full} 사용)")
            else:
                translations = translate_segments_with_context(
                    client, segments, model, args.delay, args.provider
                )
                valid = [t for t in translations if t is not None]
                entry[field_seg] = " ".join(valid) if valid else None
                print(f"  {field_seg}: {entry[field_seg]}")

        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
