"""Zhang 2020 Algorithm 1 이 요구하는 NMT 능력만 감싼다.

필요한 것 둘 — gtx 로는 어느 쪽도 안 된다:
    1. 강제 디코딩 (tgt_force = 직전까지 확정된 MU 번역)
    2. 전체 문장 번역의 beam top-N 후보 (엄격한 접두사 조건 완화, 논문 N=10)

NLLB-200-distilled-600M 하나로 de/ja/zh 를 덮는다. 타깃마다 다른 Marian 을 쓰면
모델 품질 차이가 타깃 간 비교를 오염시킨다 (en→ja Marian 은 특히 약하다).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"en": "eng_Latn", "de": "deu_Latn", "ja": "jpn_Jpan",
             "zh": "zho_Hans", "ko": "kor_Hang", "es": "spa_Latn"}


class Nmt:
    def __init__(self, src: str = "en", tgt: str = "de", device: str = "cuda",
                 model_name: str = MODEL, max_new_tokens: int = 128,
                 attentions: bool = False, attn_layer: int = 5):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.attn_layer = attn_layer
        self.tok = AutoTokenizer.from_pretrained(model_name, src_lang=NLLB_CODE[src])
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # 교차어텐션을 뽑으려면 sdpa 로는 안 되고 eager 여야 한다.
        kw = {"attn_implementation": "eager"} if attentions else {}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype, **kw).to(device).eval()
        self.tgt_id = self.tok.convert_tokens_to_ids(NLLB_CODE[tgt])
        self.start_id = self.model.config.decoder_start_token_id
        self.eos_id = self.tok.eos_token_id

    def _encode(self, text: str):
        return self.tok(text, return_tensors="pt", truncation=True,
                        max_length=256).to(self.device)

    def _decoder_prefix(self, forced: list[int] | None) -> torch.Tensor:
        ids = [self.start_id, self.tgt_id] + list(forced or [])
        return torch.tensor([ids], device=self.device)

    def _strip(self, seq: torch.Tensor) -> list[int]:
        """[start, lang, …, eos] → 가운데 본문 토큰만."""
        out = [int(t) for t in seq]
        out = out[2:] if len(out) >= 2 else out
        while out and out[-1] in (self.eos_id, self.tok.pad_token_id):
            out.pop()
        return out

    @torch.inference_mode()
    def full_candidates(self, text: str, n: int = 10) -> list[str]:
        """전체 문장 번역 상위 N 후보. 첫 번째가 beam 1위다."""
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(None),
            num_beams=n, num_return_sequences=n,
            max_new_tokens=self.max_new_tokens, do_sample=False)
        return [self.tok.decode(self._strip(s), skip_special_tokens=True) for s in out]

    @torch.inference_mode()
    def translate_prefix(self, text: str,
                         forced: list[int] | None = None) -> tuple[str, list[int]]:
        """소스 접두사를 강제 타깃 접두사 위에서 이어 디코딩한다 (greedy, 결정론적)."""
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(forced),
            num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens)
        ids = self._strip(out[0])
        return self.tok.decode(ids, skip_special_tokens=True), ids

    def _word_of_token(self, ids: torch.Tensor) -> list[int]:
        """소스 토큰 인덱스 → 어절 인덱스. `▁` 로 시작하면 새 어절이다.

        0번은 언어 코드, 마지막은 `</s>` 라 어절이 없다 (−1 로 둔다).
        """
        toks = self.tok.convert_ids_to_tokens(ids)
        out, w = [], -1
        for i, t in enumerate(toks):
            if i == 0 or t == self.tok.eos_token or t == "</s>":
                out.append(-1)
                continue
            if t.startswith("\u2581"):
                w += 1
            out.append(max(w, 0))
        return out

    @torch.inference_mode()
    def emit_with_alignment(self, text: str, forced: list[int] | None = None
                            ) -> list[tuple[int, int]]:
        """소스 접두사를 디코딩하며 **새 토큰마다 정렬된 소스 어절**을 함께 낸다.

        AlignAtt 이 요구하는 것이 이것뿐이다 — 토큰 i 의 교차어텐션 argmax 가 어느
        소스 어절을 가리키는가. 마지막 층은 `</s>` 로 쏠려(attention sink) 못 쓰므로
        `attn_layer`(기본 5 — 50문장 스윕에서 de 0.829 / ja 0.735 로 합산 최고)를
        쓰고 헤드는 평균낸다. 원 논문은 6층 중 4층 + 헤드 평균이다(아키텍처가 달라 층
        번호는 그대로 옮길 수 없다). 층 간 단조성 차이는 de 0.72~0.84 로 작다.
        """
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(forced),
            num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens,
            output_attentions=True, return_dict_in_generate=True)
        w_of = self._word_of_token(enc["input_ids"][0])
        n_src = len(w_of)
        seq = out.sequences[0]
        n_prefix = 2 + len(forced or [])
        new_ids = [int(t) for t in seq[n_prefix:]]
        res: list[tuple[int, int]] = []
        for step, tok_id in enumerate(new_ids):
            if step >= len(out.cross_attentions):
                break
            if tok_id in (self.eos_id, self.tok.pad_token_id):
                break
            a = out.cross_attentions[step][self.attn_layer].float().mean(1)[0, -1]
            # 언어코드(0)와 </s>(마지막)는 정렬 후보에서 뺀다.
            j = int(a[1:n_src - 1].argmax()) + 1
            res.append((tok_id, w_of[j]))
        return res
