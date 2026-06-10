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
import re


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


def _chat_translate(client: OpenAI, system_prompt: str, text: str, model: str, provider: str) -> tuple[str, float]:
    """Chat Completions API로 번역. (결과, 소요시간_초) 반환."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": text},
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


def translate_segments_without_context(
    client: OpenAI,
    segments: list[str],
    model: str,
    delay: float = 0.3,
    provider: str = "openai",
) -> list[str | None]:
    """각 세그먼트를 컨텍스트 없이 독립적으로 번역."""
    translations: list[str | None] = []

    for i, seg in enumerate(segments):
        try:
            if provider == "ollama":
                t, elapsed = _chat_translate(client, SEG_SYSTEM_PROMPT, seg, model, provider)
            else:
                t0 = time.perf_counter()
                response = client.responses.create(
                    model=model,
                    instructions=SEG_SYSTEM_PROMPT,
                    input=seg,
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

    parser = argparse.ArgumentParser(description="GPT 컨텍스트 없는 분절 단위 번역")
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
        # 필드 prefix: 모델명에서 태그(:) 제거 후 특수문자 → 언더스코어
        # 예) exaone3.5:32b → exaone3_5, qwen3:32b → qwen3
        prefix = re.sub(r'[.\-]', '_', model.split(':')[0])
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        client = OpenAI(api_key=api_key)
        model = args.model or "gpt-5.4-nano"
        prefix = "gpt"

    field_full   = f"{prefix}_full_trans"
    field_seg_nc = f"{prefix}_seg_nc_trans"
    print(f"저장 필드: {field_full}, {field_seg_nc}")

    input_path  = Path(args.input_dir) / args.input
    output_path = input_path if args.output is None else Path(args.input_dir) / args.output

    raw  = json.loads(input_path.read_text(encoding="utf-8"))
    data = raw["data"]

    total = len(data)
    for i, entry in enumerate(data):
        has_full   = bool(entry.get(field_full))
        has_seg_nc = bool(entry.get(field_seg_nc))

        if args.resume and has_full and has_seg_nc:
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

        # ── seg_nc_trans ────────────────────────────────────────────────────
        if not (args.resume and has_seg_nc):
            seg_text = entry["seg_text"]
            segments = [s.strip() for s in seg_text.split("<SEG>") if s.strip()]

            if len(segments) <= 1:
                entry[field_seg_nc] = entry.get(field_full)
                print(f"  {field_seg_nc}: (분절 없음, {field_full} 사용)")
            else:
                translations = translate_segments_without_context(
                    client, segments, model, args.delay, args.provider
                )
                valid = [t for t in translations if t is not None]
                entry[field_seg_nc] = " ".join(valid) if valid else None
                print(f"  {field_seg_nc}: {entry[field_seg_nc]}")

        output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n완료. 저장: {output_path}")


if __name__ == "__main__":
    main()
