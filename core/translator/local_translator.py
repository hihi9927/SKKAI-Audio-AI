"""로컬 번역 백엔드 (NLLB-200 / MADLAD-400).

Google Translate 경로는 API 키가 없으면 무료 gtx 엔드포인트로 떨어지고, 호출이
몰리면 429 로 막혀 번역이 빈 채로 나간다. 이 모듈은 외부 호출 없이 로컬 GPU/CPU
에서 번역해 그 의존을 없앤다.

서버에서 ``--local-translation`` 으로 켜고, ``--local-translation-model`` 로 모델을
고른다. 이름에 ``madlad`` 가 들어가면 MADLAD, 아니면 NLLB 로 취급한다.

RTX 4090 실측 (num_beams=4, 시연 문장 9개):

===================================  ==========  ======  ==================================
모델                                 지연 중앙값  GPU     비고
===================================  ==========  ======  ==================================
facebook/nllb-200-distilled-600M     54ms        1.2GB   doorbell 을 놓치고 "문벨" 로 옮긴다
facebook/nllb-200-distilled-1.3B     91ms        2.6GB   doorbell 은 살리나 여전히 "문벨"
google/madlad400-3b-mt               195ms       ~6GB    "초인종". 문장이 가장 자연스럽다
===================================  ==========  ======  ==================================

MADLAD 를 쓰려면 ASR 쪽 ``--gpu-memory-utilization`` 을 낮춰 자리를 비워야 한다.
0.42 면 ASR 이 12GB 를 쓰고(KV 캐시 4.88GB, 45,696 토큰) 12.5GB 가 남는다.
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


class _Seq2SeqTranslator:
    """로컬 seq2seq 번역기 공통부.

    generate 는 GPU 를 점유하므로 락으로 한 번에 하나만 돌린다. 호출은
    ``asyncio.to_thread`` 로 넘겨 이벤트 루프를 막지 않는다.

    num_beams 와 no_repeat_ngram_size 기본값의 근거:

    - greedy(num_beams=1) 는 짧은 문장에서 EOS 를 일찍 내고 오역한다. 실측::

          ヘブライ人一家はほとんど都会で暮らしていました。
            beams=1 -> "이 두 사람은"                         (끊기고 오역)
            beams=4 -> "히브리 가족들은 대부분 도시에 살고 있었습니다."

      비용은 45ms -> 60ms 로, 파이프라인 전체 지연에서 무시할 수준이다.
      min_new_tokens 로 길이를 강제하는 건 해법이 아니다 — 헛소리를 이어 붙인다.

    - no_repeat_ngram_size 는 반복 루프를 막는다. 짧고 반복적인 입력에서 터진다::

          "아니, 아니."  ->  "No, no, no, no, no, ..." 가 상한까지 이어졌다.
          3 을 주면      ->  "No, no, not at all."

      정당한 반복은 살아남는다("네 네 네." -> "Yeah, yeah, yeah.").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._device = device
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._context_warned = False

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

    # ── 서브클래스가 채우는 부분 ────────────────────────────────────────────────
    def _build_input(self, text: str, target_code: str, source_code: str):
        """(모델에 넣을 문자열, generate 에 더할 인자) 를 돌려준다.

        지원하지 않는 언어쌍이면 None 을, 번역이 필요 없으면 (None, None) 대신
        _translate_sync 쪽에서 걸러지도록 각 구현이 알아서 처리한다.
        """
        raise NotImplementedError

    # ── 번역 ──────────────────────────────────────────────────────────────────
    def _translate_sync(self, text: str, target_code: str, source_code: str) -> str:
        self._ensure_loaded()
        if source_code == target_code:
            return text
        import torch

        with self._lock:
            tok, model = self._tokenizer, self._model
            # 입력 구성은 락 안에서 한다. NLLB 는 토크나이저의 src_lang 을 바꾸는데,
            # 락 밖에서 바꾸면 다른 호출이 그 사이에 tokenize 해 언어가 섞인다.
            built = self._build_input(text, target_code, source_code)
            if built is None:
                logger.warning(
                    f"[local-translate] 지원하지 않는 언어쌍 {source_code}->{target_code}")
                return ""
            model_input, extra = built
            encoded = tok(model_input, return_tensors="pt", truncation=True, max_length=512)
            n_in = encoded["input_ids"].shape[1]
            inputs = {k: v.to(self._device) for k, v in encoded.items()}
            # 출력 길이를 입력에 비례해 묶는다. 반복이 새어 나가도 상한까지 가지 않는다.
            max_new = min(self.max_new_tokens, max(24, n_in * 3))
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    **extra,
                )
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    async def translate(
        self, text: str, target_code: str, source_code: Optional[str] = None,
        context: Optional[list] = None,
    ) -> tuple[str, str]:
        """(번역문, 소스 언어 코드) 를 돌려준다. Google 경로와 반환 형태를 맞췄다.

        ``context`` 는 받되 쓰지 않는다. 이 백엔드는 문장 단위 모델이라 문맥을 넣을
        자리가 없고, 이어붙여 넣으면 번역이 망가진다(LLMTranslator 주석의 실측 참고).
        조용히 버리면 문맥이 반영되는 줄 알기 쉬우므로 한 번은 알린다.
        """
        if not text.strip() or not target_code:
            return "", ""
        if context and not self._context_warned:
            self._context_warned = True
            logger.warning(
                f"[local-translate] {self.model_name} 은 문맥을 쓸 수 없어 무시한다 — "
                "문맥이 필요하면 LLM 백엔드(예: Qwen/Qwen3-4B-Instruct-2507)를 쓸 것")
        src = (source_code or "").strip() or guess_lang_code(text)
        try:
            translated = await asyncio.to_thread(self._translate_sync, text, target_code, src)
        except Exception as e:
            logger.warning(f"[local-translate] failed: {e!r}")
            return "", src
        return translated, src


class NLLBTranslator(_Seq2SeqTranslator):
    """NLLB-200. 소스 언어를 토크나이저에 지정하고 타깃을 강제 BOS 로 준다."""

    def _build_input(self, text: str, target_code: str, source_code: str):
        src = LANG_CODE_TO_FLORES.get(source_code)
        tgt = LANG_CODE_TO_FLORES.get(target_code)
        if not src or not tgt:
            return None
        self._tokenizer.src_lang = src   # _translate_sync 의 락 안에서만 불린다
        return text, {"forced_bos_token_id": self._tokenizer.convert_tokens_to_ids(tgt)}


class MADLADTranslator(_Seq2SeqTranslator):
    """MADLAD-400. 타깃 언어를 입력 앞에 ``<2xx>`` 토큰으로 붙인다.

    소스 언어는 지정하지 않는다(모델이 알아서 본다). 그래서 ASR 이 언어를 잘못
    감지해도 NLLB 만큼 크게 망가지지 않는다.
    """

    def _build_input(self, text: str, target_code: str, source_code: str):
        tag = f"<2{target_code}>"
        if self._tokenizer.convert_tokens_to_ids(tag) == self._tokenizer.unk_token_id:
            return None
        return f"{tag} {text}", {}


class LLMTranslator:
    """지시형 LLM 을 번역기로 쓴다. **앞 문장을 문맥으로 받을 수 있는 유일한 백엔드다.**

    seq2seq 번역기(NLLB·MADLAD)는 문맥을 받는 자리가 없다. 서버의 ``--google-context``
    가 쓰는 이어붙이기(앞 문장들을 줄바꿈으로 붙여 한 번에 번역하고 마지막 줄만 취함)를
    로컬 모델에 얹어 봤지만 둘 다 실패했다 — Google 이 줄바꿈을 보존해 주기 때문에
    되는 트릭이지 일반적으로 통하는 게 아니다. 실측(대화 20건):

    - NLLB-1.3B: 줄 수 불일치 16/16. 문맥이 1줄일 땐 현재 문장을 빼먹고 문맥 문장만
      돌려줬고, 3줄일 땐 전부 한 줄로 합쳐 "마지막 줄"이 문맥 덩어리 전체가 됐다.
    - MADLAD-3B: 번역 대신 영어 잡음을 뱉고 지연이 2.4초까지 늘었다.

    그래서 문맥이 필요하면 LLM 을 쓴다. Qwen3-4B-Instruct 4bit 를 5개 언어 병렬
    대화셋(600 방향쌍, COMET-DA)으로 잰 결과:

    ====== ========= ============== ==========
    문맥    COMET-DA  문맥 없음 대비  지연 p50
    ====== ========= ============== ==========
    0턴     0.8827    —              170ms
    1턴     0.8903    +0.0076        170ms
    2턴     0.8899    +0.0073        167ms
    3턴     0.8899    +0.0072        170ms
    4턴     0.8890    +0.0063        170ms
    ====== ========= ============== ==========

    **이득은 1턴에서 포화한다.** 2턴 이상은 더 얻는 게 없다(95% CI 가 겹친다). 그래서
    ``context_window`` 기본값이 1이다. 지연은 문맥 깊이와 무관하다 — 프롬프트가 76에서
    99 토큰으로 늘어도 생성 시간이 지배해서 묻힌다(16턴/266토큰에서도 193ms).

    이득은 타깃이 영어일 때 가장 크다(+0.019~0.022). 한국어·일본어·중국어가 비워 둔
    주어와 목적어를 영어는 반드시 채워야 하는데 그 답이 앞 턴에만 있기 때문이다.
    ko/ja/zh 타깃은 +0.002~0.004 로 미미하다.

    VRAM 은 4bit 로 약 3.3GiB 다. 단 **카드가 비어 있을 때 올리면 4.4GiB 를 예약한다** —
    캐싱 할당자가 여유가 많으면 더 잡아 둔다. ASR 서버들이 자리를 잡은 뒤에 띄우고
    ``PYTORCH_ALLOC_CONF=expandable_segments:True`` 를 주는 편이 안전하다.
    """

    DEFAULT_LLM = "Qwen/Qwen3-4B-Instruct-2507"
    LANG_NAME = {
        "en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese", "es": "Spanish",
        "fr": "French", "de": "German", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
        "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "hi": "Hindi", "ar": "Arabic",
        "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    }
    SYSTEM = ("You are a translation engine for a live conversation. Use the earlier "
              "turns only to resolve pronouns, omitted subjects, gender agreement and "
              "politeness level. Output only the translation of the final line, with "
              "no explanation and no quotes.")

    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        device: Optional[str] = None,
        quant: str = "4bit",
        max_new_tokens: int = 200,
        context_window: int = 1,
        num_beams: int = 1,          # noqa: ARG002 — seq2seq 쪽과 인자 형태를 맞추기 위해 받는다
        **_ignored,
    ):
        self.model_name = model_name
        self.quant = quant
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self._device = device
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import gc

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[local-translate] loading {self.model_name} on {device} ({self.quant})")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.quant in ("4bit", "8bit") and device.startswith("cuda"):
                from transformers import BitsAndBytesConfig

                if self.quant == "4bit":
                    qc = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                else:
                    qc = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, quantization_config=qc, device_map=device,
                    dtype=torch.bfloat16)
            else:
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
                model.to(device)
            model.eval()
            self._model, self._device = model, device
            # 로딩 중 잡았다 놓은 블록을 반납한다. nvidia-smi 는 예약만 해 둔 것도
            # 사용량으로 보고하므로, 안 부르면 실제보다 1GiB 가까이 크게 잡힌다.
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            logger.info("[local-translate] ready")

    def load(self) -> None:
        """서버 기동 시 미리 로드해 첫 번역의 지연을 없앤다."""
        self._ensure_loaded()

    def _name(self, code: str) -> str:
        return self.LANG_NAME.get(code, code)

    def _build_prompt(self, text: str, target_code: str, source_code: str,
                      context: Optional[list] = None) -> str:
        tok = self._tokenizer
        if context:
            ctx_block = "\n".join(f"- {c}" for c in context)
            user = (f"Earlier turns in {self._name(source_code)}:\n{ctx_block}\n\n"
                    f"Translate this {self._name(source_code)} line into "
                    f"{self._name(target_code)}:\n{text}")
        else:
            user = (f"Translate the following {self._name(source_code)} sentence into "
                    f"{self._name(target_code)}.\n\n{text}")
        msgs = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user}]
        try:
            # Qwen3 계열은 생각 모드를 끄지 않으면 출력이 길어져 자막 지연이 커진다.
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _translate_sync(self, text: str, target_code: str, source_code: str,
                        context: Optional[list] = None) -> str:
        self._ensure_loaded()
        if source_code == target_code:
            return text
        import torch

        with self._lock:
            tok, model = self._tokenizer, self._model
            enc = tok(self._build_prompt(text, target_code, source_code, context),
                      return_tensors="pt")
            n_prompt = enc["input_ids"].shape[1]
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.eos_token_id)
            s = tok.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        if "</think>" in s:
            s = s.split("</think>")[-1].strip()
        s = s.strip().strip('"').strip()
        # 지시를 어기고 문맥까지 통째로 옮겨 오면 줄이 여러 개가 된다. 마지막 줄이
        # 현재 발화의 번역이다. (실측 600쌍에서는 한 번도 일어나지 않았다.)
        lines = [ln.strip(" -") for ln in s.split("\n") if ln.strip()]
        return lines[-1] if lines else s

    async def translate(
        self, text: str, target_code: str, source_code: Optional[str] = None,
        context: Optional[list] = None,
    ) -> tuple[str, str]:
        """(번역문, 소스 언어 코드). ``context`` 는 앞 발화 원문 리스트(오래된 것부터)."""
        if not text.strip() or not target_code:
            return "", ""
        src = (source_code or "").strip() or guess_lang_code(text)
        ctx = list(context or [])[-self.context_window:] if self.context_window else []
        try:
            translated = await asyncio.to_thread(
                self._translate_sync, text, target_code, src, ctx)
        except Exception as e:
            logger.warning(f"[local-translate] failed: {e!r}")
            return "", src
        return translated, src


def make_translator(model_name: str = DEFAULT_MODEL, **kwargs):
    """모델 이름으로 백엔드를 고른다.

    이름에 ``madlad`` 가 들어가면 MADLAD, 지시형 LLM 으로 보이면 LLMTranslator,
    나머지는 NLLB 로 친다. LLM 만 문맥을 쓸 수 있다(LLMTranslator 주석 참고).
    """
    lowered = model_name.lower()
    if "madlad" in lowered:
        return MADLADTranslator(model_name=model_name, **_seq2seq_kwargs(kwargs))
    if any(k in lowered for k in ("qwen", "instruct", "gemma", "llama", "mistral", "-it")):
        return LLMTranslator(model_name=model_name, **kwargs)
    return NLLBTranslator(model_name=model_name, **_seq2seq_kwargs(kwargs))


def _seq2seq_kwargs(kwargs: dict) -> dict:
    """seq2seq 번역기가 모르는 LLM 전용 인자를 걷어낸다."""
    return {k: v for k, v in kwargs.items() if k not in ("quant", "context_window")}


# ── 원격 번역기 ────────────────────────────────────────────────────────────────
# 번역 모델을 ASR 서버와 같은 프로세스에 올리면, ASR 서버를 여러 개 띄울 때
# 번역 모델도 그 수만큼 복제되어 VRAM 을 잡아먹는다(madlad400-3b 는 인스턴스당
# 약 7.1GiB). 번역기를 한 프로세스에 한 번만 올려 두고 나머지는 HTTP 로 부르게
# 한다. 인터페이스를 _Seq2SeqTranslator.translate 와 똑같이 맞췄으므로
# set_local_translator 에 그대로 꽂힌다.

class RemoteTranslator:
    """단독 번역 서버(STiTy-Mobile/demo-web/local_translation_server.py)를 HTTP 로 부르는 클라이언트."""

    def __init__(self, url: str, timeout: float = 20.0):
        self.url = url.rstrip("/") + "/translate"
        self.timeout = timeout
        self._session = None

    def load(self) -> None:
        """로컬 번역기와 호출 형태를 맞추기 위한 자리. 원격이라 올릴 게 없다."""
        return None

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def translate(
        self, text: str, target_code: str, source_code: Optional[str] = None,
        context: Optional[list] = None,
    ) -> tuple[str, str]:
        if not text.strip() or not target_code:
            return "", ""
        src = (source_code or "").strip() or guess_lang_code(text)
        try:
            session = await self._get_session()
            payload = {"text": text, "target": target_code, "source": src}
            # 문맥은 번역 서버가 LLM 백엔드일 때만 쓰인다. seq2seq 서버는 받아서
            # 버리므로(경고 한 번) 여기서 조건을 걸 필요가 없다.
            if context:
                payload["context"] = list(context)
            async with session.post(self.url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data.get("translation", ""), data.get("source", src)
        except Exception as e:
            logger.warning(f"[remote-translate] failed: {e!r}")
            return "", src
