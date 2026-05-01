# coding=utf-8
"""ASR 교정 + 번역을 단일 GPT API 호출로 처리하는 모듈.

프롬프트 설계는 core/meaning_segmentator/utils/gpt_trans.py의
SEG_SYSTEM_PROMPT / SEG_CONTEXT_SYSTEM_PROMPT를 기반으로,
멀티언어 지원 + ASR 교정 스텝을 추가한 형태.
"""

import asyncio
import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI


_LANG_CODE_TO_NAME: dict[str, str] = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
    "es": "Spanish", "fr": "French", "de": "German",
    "ti": "Tibetan",
}

# gpt_trans.py SEG_SYSTEM_PROMPT 기반 — 컨텍스트 없을 때 (첫 세그먼트)
_SYSTEM_PROMPT_NO_CONTEXT = """\
You are an expert ASR corrector and spoken-language translator specializing in conversational speech.
Perform two steps and respond with JSON only.

Step 1 — Minimal ASR correction (output as "corrected"):
- Fix homophones, obvious word repetitions, and clear misrecognitions ONLY.
- Do NOT change meaning, formalize speech, or add/remove content.
- When uncertain, keep the original unchanged.

Step 2 — Translate the corrected text to {target_name} (output as "translation"):
- The input may be a sentence fragment — translate exactly what is given. Do NOT complete or extend it.
- Preserve the natural spoken register (casual/formal) exactly as in the source.
- Filler words and disfluencies: translate naturally or omit if they carry no meaning, consistent with spoken {target_name} convention.
- Proper nouns (names, places, organizations): transliterate consistently.
- Translate faithfully — do not add, omit, or infer meaning beyond what is stated.

JSON format: {{"corrected": "<corrected original>", "translation": "<{target_name} translation>"}}\
"""

# gpt_trans.py SEG_CONTEXT_SYSTEM_PROMPT 기반 — 이전 세그먼트 컨텍스트 있을 때
_SYSTEM_PROMPT_WITH_CONTEXT = """\
You are an expert ASR corrector and spoken-language translator specializing in conversational speech.
You will receive already-confirmed preceding segment translations as context, then a new segment to correct and translate.
Perform two steps and respond with JSON only.

Step 1 — Minimal ASR correction (output as "corrected"):
- Fix homophones, obvious word repetitions, and clear misrecognitions ONLY.
- Do NOT change meaning, formalize speech, or add/remove content.
- When uncertain, keep the original unchanged.

Step 2 — Translate the corrected text to {target_name} (output as "translation"):
- Output ONLY the {target_name} translation of the NEW segment. Nothing else.
- The preceding translations are FINAL. Do NOT reproduce, paraphrase, or continue them.
- The new segment may be a grammatical fragment — translate exactly what is given, do NOT complete it.
- Match the register, terminology, and tone established in the preceding translations.
- Preserve the natural spoken register (casual/formal) exactly as in the source.
- Filler words and disfluencies: translate naturally or omit if they carry no meaning, consistent with spoken {target_name} convention.
- Proper nouns (names, places, organizations): transliterate consistently with preceding segments.
- Translate faithfully — do not add, omit, or infer meaning beyond what is stated.

JSON format: {{"corrected": "<corrected original>", "translation": "<{target_name} translation>"}}\
"""


class GPTTranslator:
    """ASR 교정과 번역을 단일 GPT API 호출로 처리.

    GPTCorrector(교정) + Google Translate(번역) 두 번의 API 호출을
    하나로 줄인다. JSON 응답으로 교정된 원문과 번역문을 함께 반환.

    context 파라미터로 이전 세그먼트의 (original, translation) 쌍을 넘기면
    일관된 용어·문체로 번역한다 (gpt_trans.py SEG_CONTEXT_SYSTEM_PROMPT 방식).

    Usage:
        translator = GPTTranslator(model="gpt-5.4-mini")
        corrected, translation = await translator.correct_and_translate(
            text="안녕하세요",
            source_lang_name="Korean",
            target_lang_code="en",
            context=[("이전 문장", "Previous sentence"), ...],
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.4-mini",
        max_retries: int = 3,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. "
                "api_key 인자 또는 OPENAI_API_KEY 환경변수를 설정하세요."
            )
        self._client = AsyncOpenAI(api_key=key)
        self._model = model
        self._max_retries = max_retries

    async def correct_and_translate(
        self,
        text: str,
        source_lang_name: str,
        target_lang_code: str,
        context: Optional[list[tuple[str, str]]] = None,
    ) -> tuple[str, str]:
        """ASR 텍스트를 교정하고 target_lang_code 언어로 번역.

        Args:
            text: ASR 출력 원문.
            source_lang_name: 원문 언어 이름 (예: "Korean"). 미감지 시 빈 문자열.
            target_lang_code: 번역 목표 언어 코드 (예: "en").
            context: 직전 세그먼트들의 (교정된 원문, 번역) 쌍. 최대 5개 권장.

        Returns:
            (corrected_original, translation)
        """
        if not text.strip() or not target_lang_code:
            return text, ""

        target_name = _LANG_CODE_TO_NAME.get(target_lang_code, target_lang_code)

        has_context = bool(context)

        system_prompt = (
            _SYSTEM_PROMPT_WITH_CONTEXT if has_context else _SYSTEM_PROMPT_NO_CONTEXT
        ).format(target_name=target_name)

        if has_context:
            src_label = source_lang_name or "Source"
            context_lines = "\n".join(
                f"[{i + 1}] {src_label}: {orig} → {target_name}: {trans}"
                for i, (orig, trans) in enumerate(context)
            )
            user_content = (
                f"=== Preceding segments (FINAL — do NOT reproduce or modify) ===\n"
                f"{context_lines}\n\n"
                f"=== Translate ONLY this new segment ===\n"
                f"{text}"
            )
        else:
            user_content = text

        for attempt in range(self._max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                result = json.loads(resp.choices[0].message.content)
                corrected = result.get("corrected", text).strip() or text
                translation = result.get("translation", "").strip()
                return corrected, translation
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = _parse_retry_after(msg, attempt)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"최대 재시도({self._max_retries}회) 초과: '{text[:40]}'")


def _parse_retry_after(msg: str, attempt: int) -> float:
    m = re.search(r"try again in (\d+(?:\.\d+)?)(m?s)", msg)
    if m:
        secs = float(m.group(1)) / 1000 if m.group(2) == "ms" else float(m.group(1))
        return max(secs + 0.2, 1.0)
    return float(2 ** attempt)
