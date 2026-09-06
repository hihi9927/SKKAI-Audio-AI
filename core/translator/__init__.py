# coding=utf-8
"""번역 레이어 — GPT 교정/번역(`openai`)과 로컬 seq2seq 번역기(`transformers`).

세 모듈의 의존성이 서로 다르다. 로컬 번역 서버는 `openai` 없이 돌고 GPT 경로는
`transformers` 없이 도므로, 패키지를 import 하는 것만으로 남의 의존성을 요구하지
않도록 이름을 쓸 때 가져온다.
"""
from typing import TYPE_CHECKING

__all__ = ["GPTCorrector", "GPTTranslator"]

if TYPE_CHECKING:  # 타입 검사기에만 보이면 된다
    from .correct_and_trans import GPTTranslator
    from .gpt_corrector import GPTCorrector


def __getattr__(name):
    if name == "GPTCorrector":
        from .gpt_corrector import GPTCorrector

        return GPTCorrector
    if name == "GPTTranslator":
        from .correct_and_trans import GPTTranslator

        return GPTTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
