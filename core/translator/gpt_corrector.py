# coding=utf-8
import asyncio
import os
import re
from typing import Optional

from openai import AsyncOpenAI


_SYSTEM_PROMPT_KO = """\
너는 실시간 음성 인식(ASR) 결과를 교정하는 전문가야.
입력은 ASR이 출력한 한국어 전사 텍스트이고, 네 목표는 음성 인식 오류를 최소한으로 수정해서 올바른 텍스트로 만드는 거야.

[교정 대상]
- 동음이의어 오류: 예) "안" ↔ "않", "이따" ↔ "있다", "되" ↔ "돼", "맞쳐" ↔ "맞춰"
- 받침 혼동: 예) "일거" → "읽어"
- 분명한 단어 반복: 예) "그 그 사람" → "그 사람"
- 문맥상 명확한 문법 오류 (단, 구어체 특성은 유지)

[교정 금지]
- 원문의 의미·뉘앙스 변경 금지
- 구어체 표현을 문어체로 바꾸는 것 금지
- 내용 추가·삭제 금지
- 확신이 없으면 원문 그대로 유지

[출력 규칙]
- 교정된 텍스트만 출력 (설명, 주석 절대 금지)
- 교정이 필요 없으면 원문을 그대로 출력\
"""

_SYSTEM_PROMPT_EN = """\
You are an expert at correcting Automatic Speech Recognition (ASR) output.
Your input is a transcribed English text from an ASR system. Minimally correct ASR errors only.

[Correction targets]
- Homophones: e.g., "their/there/they're", "to/too/two", "write/right"
- Obvious repeated words caused by ASR: e.g., "I I said" → "I said"
- Clear misrecognitions evident from context

[Do NOT]
- Change the meaning or nuance of the original
- Formalize informal/spoken language
- Add or remove content
- Correct if uncertain — output the original unchanged

[Output rules]
- Output only the corrected text (no explanations, no comments)
- If no correction is needed, output the original unchanged\
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "ko": _SYSTEM_PROMPT_KO,
    "en": _SYSTEM_PROMPT_EN,
}


class GPTCorrector:
    """ASR 전사 결과를 GPT API로 후보정하는 모듈.

    서버는 correct_text() 로 문자열을 직접 넘긴다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.4-mini",
        max_retries: int = 5,
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

    async def correct_text(self, text: str, language: str = "ko") -> str:
        """텍스트 문자열을 직접 받아 후보정된 텍스트를 반환 (단독 사용용)."""
        return await self._call_api(text, language)

    # ------------------------------------------------------------------

    async def _call_api(self, text: str, language: str) -> str:
        system_prompt = _SYSTEM_PROMPTS.get(language, _SYSTEM_PROMPT_KO)
        for attempt in range(self._max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = _parse_retry_after(msg, attempt)
                    print(f"  Rate limit — {wait:.1f}초 대기 후 재시도 ({attempt + 1}/{self._max_retries})")
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
