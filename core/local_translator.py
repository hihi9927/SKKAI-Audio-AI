"""로컬 번역 백엔드 (NLLB-200).

Google Translate 경로는 API 키가 없으면 무료 gtx 엔드포인트로 떨어지고, 호출이
몰리면 429 로 막혀 번역이 빈 채로 나간다. 이 모듈은 외부 호출 없이 로컬 GPU/CPU
에서 번역해 그 의존을 없앤다.

서버에서 ``--local-translation`` 으로 켠다. 모델은 처음 번역 요청 때 로드한다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# 앱 언어 코드 -> NLLB FLORES-200 코드
LANG_CODE_TO_FLORES = {
    "en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "ar": "arb_Arab",
    "pt": "por_Latn", "it": "ita_Latn", "ru": "rus_Cyrl", "vi": "vie_Latn",
    "th": "tha_Thai", "id": "ind_Latn", "hi": "hin_Deva", "tr": "tur_Latn",
    "nl": "nld_Latn", "pl": "pol_Latn", "sv": "swe_Latn", "da": "dan_Latn",
    "fi": "fin_Latn", "cs": "ces_Latn", "el": "ell_Grek", "ro": "ron_Latn",
    "hu": "hun_Latn", "fa": "pes_Arab", "ms": "zsm_Latn", "tl": "tgl_Latn",
    "mk": "mkd_Cyrl",
}

_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_ARABIC = re.compile(r"[؀-ۿ]")


def guess_lang_code(text: str) -> str:
    """문자 종류로 소스 언어를 추정한다.

    ASR 이 언어를 알려주므로 보통은 쓰이지 않는다. 언어 정보가 비어 왔을 때의
    마지막 수단이고, 라틴 문자는 구분이 안 되므로 en 으로 둔다.
    """
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _HAN.search(text):
        return "zh"
    if _CYRILLIC.search(text):
        return "ru"
    if _ARABIC.search(text):
        return "ar"
    return "en"


class NLLBTranslator:
    """NLLB-200 로컬 번역기.

    generate 는 GPU 를 점유하므로 락으로 한 번에 하나만 돌린다. 호출은
    ``asyncio.to_thread`` 로 넘겨 이벤트 루프를 막지 않는다.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
    ):
        # greedy(num_beams=1) 는 짧은 문장에서 EOS 를 일찍 내고 오역한다. 실측:
        #   ヘブライ人一家はほとんど都会で暮らしていました。
        #     beams=1 -> "이 두 사람은"                         (끊기고 오역)
        #     beams=4 -> "히브리 가족들은 대부분 도시에 살고 있었습니다."
        # 비용은 문장당 45ms -> 60ms 로, 파이프라인 전체 지연에서 무시할 수준이다.
        # min_new_tokens 로 길이를 강제하는 건 해법이 아니다 — 헛소리를 이어 붙인다.
        #
        # no_repeat_ngram_size 는 반복 루프를 막는다. 짧고 반복적인 입력에서 터진다:
        #   "아니, 아니."  ->  "No, no, no, no, no, ..." 가 상한까지 이어졌다.
        #   3 을 주면      ->  "No, no, not at all."
        # 정당한 반복은 살아남는다("네 네 네." -> "Yeah, yeah, yeah."). 다른 문장의
        # 결과는 바뀌지 않았고 속도 차이도 없다.
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._device = device
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    # ── 로딩 ──────────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self._device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            logger.info(f"[local-translate] loading {self.model_name} on {device} ({dtype})")
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, dtype=dtype)
            model.to(device)
            model.eval()
            self._tokenizer, self._model, self._device = tok, model, device
            logger.info("[local-translate] ready")

    def load(self) -> None:
        """서버 기동 시 미리 로드해 첫 번역의 지연을 없앤다."""
        self._ensure_loaded()

    # ── 번역 ──────────────────────────────────────────────────────────────────
    def _translate_sync(self, text: str, target_code: str, source_code: str) -> str:
        self._ensure_loaded()
        src = LANG_CODE_TO_FLORES.get(source_code)
        tgt = LANG_CODE_TO_FLORES.get(target_code)
        if not src or not tgt:
            logger.warning(f"[local-translate] 지원하지 않는 언어쌍 {source_code}->{target_code}")
            return ""
        if src == tgt:
            return text

        import torch

        with self._lock:
            tok, model = self._tokenizer, self._model
            tok.src_lang = src
            encoded = tok(text, return_tensors="pt", truncation=True, max_length=512)
            n_in = encoded["input_ids"].shape[1]
            inputs = {k: v.to(self._device) for k, v in encoded.items()}
            bos = tok.convert_tokens_to_ids(tgt)
            # 출력 길이를 입력에 비례해 묶는다. 반복이 새어 나가도 상한까지 가지 않는다.
            max_new = min(self.max_new_tokens, max(24, n_in * 3))
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    forced_bos_token_id=bos,
                    max_new_tokens=max_new,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                )
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    async def translate(
        self, text: str, target_code: str, source_code: Optional[str] = None
    ) -> tuple[str, str]:
        """(번역문, 소스 언어 코드) 를 돌려준다. Google 경로와 반환 형태를 맞췄다."""
        if not text.strip() or not target_code:
            return "", ""
        src = (source_code or "").strip() or guess_lang_code(text)
        try:
            translated = await asyncio.to_thread(self._translate_sync, text, target_code, src)
        except Exception as e:
            logger.warning(f"[local-translate] failed: {e!r}")
            return "", src
        return translated, src
