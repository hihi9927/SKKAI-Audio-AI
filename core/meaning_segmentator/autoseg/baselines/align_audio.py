"""CTC 강제정렬로 소스 **단어 타임스탬프**를 뽑는다 — ms LAAL 을 실측으로 만들기 위한 것.

지금까지의 `laal_ms` 는 발화 길이(실측)를 어절 수로 **선형 보간**한 값이었다. 발화 안에서
말속도가 일정하다는 가정이 들어가는데, 실제로는 강세·휴지 때문에 그렇지 않다. 여기서
경계 시각을 실제 음성에서 재어 그 가정을 없앤다.

torchaudio 를 새로 깔지 않는다 (이 레포는 `transformers==4.57.6` 이 핀돼 있고 한 번
의존성을 깬 적이 있다). CTC Viterbi 정렬은 60줄이면 되므로 직접 구현한다.

정렬 모델은 `facebook/wav2vec2-base-960h` — 영어 전용이지만 소스가 en 뿐이라 충분하다.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

MODEL = "facebook/wav2vec2-base-960h"
_WS = re.compile(r"\s+")


class Aligner:
    def __init__(self, device: str = "cuda", model_name: str = MODEL):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.device = device
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.model.config.pad_token_id

    # ── 전사 정규화 ────────────────────────────────────────────────────────
    def _tokens_for(self, words: list[str]) -> tuple[list[int], list[int]]:
        """CTC 라벨 시퀀스와, 각 라벨이 몇 번째 어절에 속하는지."""
        ids, owner = [], []
        for wi, w in enumerate(words):
            up = re.sub(r"[^A-Z']", "", w.upper())
            if not up:
                continue
            if ids:                                   # 어절 사이 구분자
                ids.append(self.vocab["|"])
                owner.append(wi)
            for ch in up:
                ids.append(self.vocab.get(ch, self.vocab["|"]))
                owner.append(wi)
        return ids, owner

    # ── CTC Viterbi 강제정렬 ───────────────────────────────────────────────
    def _align(self, logprob: torch.Tensor, labels: list[int]) -> list[int]:
        """프레임별 최적 라벨 인덱스. blank 를 라벨 사이에 끼운 표준 CTC 격자."""
        ext = [self.blank]
        for l in labels:
            ext += [l, self.blank]
        T, S = logprob.shape[0], len(ext)
        NEG = -1e30
        dp = torch.full((S,), NEG, device=logprob.device)
        dp[0] = logprob[0, ext[0]]
        if S > 1:
            dp[1] = logprob[0, ext[1]]
        back = torch.zeros((T, S), dtype=torch.int8, device=logprob.device)
        for t in range(1, T):
            stay = dp
            prev = torch.cat([torch.tensor([NEG], device=dp.device), dp[:-1]])
            # 같은 라벨 반복이거나 blank 면 두 칸 건너뛰기 금지
            skip = torch.cat([torch.tensor([NEG, NEG], device=dp.device), dp[:-2]]) \
                if S > 2 else torch.full((S,), NEG, device=dp.device)
            if S > 2:
                bad = torch.tensor(
                    [True, True] + [ext[i] == self.blank or ext[i] == ext[i - 2]
                                    for i in range(2, S)], device=dp.device)
                skip = skip.masked_fill(bad, NEG)
            best = torch.stack([stay, prev, skip])
            idx = best.argmax(0)
            dp = best.gather(0, idx.unsqueeze(0)).squeeze(0) + \
                logprob[t, torch.tensor(ext, device=dp.device)]
            back[t] = idx.to(torch.int8)
        s = int(torch.argmax(dp[-2:]).item()) + (S - 2)
        path = [0] * T
        for t in range(T - 1, -1, -1):
            path[t] = s
            s -= int(back[t, s].item())
        return path

    @torch.inference_mode()
    def word_end_times(self, wav_path: Path, text: str) -> list[float] | None:
        """어절 i 가 **끝나는 시각(초)**. 정렬 실패 시 None."""
        sr, data = wavfile.read(wav_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim > 1:
            data = data.mean(1)
        dur = len(data) / sr
        words = _WS.split(text.strip())
        labels, owner = self._tokens_for(words)
        if not labels:
            return None

        x = self.proc(data, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        logits = self.model(x).logits[0].float()
        logprob = torch.log_softmax(logits, dim=-1)
        if logprob.shape[0] < len(labels):            # 프레임보다 라벨이 많으면 불가
            return None
        path = self._align(logprob, labels)

        frame_dur = dur / logprob.shape[0]
        ends = [0.0] * len(words)
        for t, s in enumerate(path):
            if s % 2 == 1:                            # 실제 라벨 (blank 아님)
                ends[owner[(s - 1) // 2]] = (t + 1) * frame_dur
        # 비어 있는 어절(전부 기호 등)은 앞 값을 물려받고, 단조 증가를 강제한다.
        run = 0.0
        for i, e in enumerate(ends):
            run = max(run, e)
            ends[i] = run
        ends[-1] = max(ends[-1], dur * 0.999)
        return ends
