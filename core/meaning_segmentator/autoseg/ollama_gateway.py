"""ollama 네이티브 /api/chat 로 autoseg.pipeline 의 Gateway 자리를 채우는 최소 shim.

OpenAI 호환 엔드포인트가 아니라 네이티브를 쓰는 이유는 `num_ctx` 때문이다 — 분절
프롬프트만 3k 토큰이라 ollama 기본 컨텍스트(4096)에서는 입력이 잘린다.
"""
from __future__ import annotations

import os
import threading

import httpx


class Gateway:
    def __init__(self, model: str = "llama3.3:70b", host: str | None = None,
                 timeout: float = 1800.0, temperature: float = 0.0,
                 num_ctx: int = 8192):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "localhost:11434")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = httpx.Client(base_url=f"http://{self.host}", timeout=timeout)
        self._lock = threading.Lock()
        self.calls = 0
        self.retries = 0
        self.empty = 0

    def chat(self, system: str, user: str, max_tokens: int = 4096,
             reasoning_effort: str | None = None, purpose: str = "", **kw) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": min(max_tokens, 2048)},
        }
        last = ""
        for _ in range(3):
            try:
                r = self._client.post("/api/chat", json=body)
                r.raise_for_status()
                last = (r.json()["message"]["content"] or "").strip()
            except Exception as e:                       # 네트워크·모델 로드 실패
                last = ""
                err = repr(e)[:200]
            with self._lock:
                self.calls += 1
                if purpose == "segment_retry":
                    self.retries += 1
            if last:
                return last
            with self._lock:
                self.empty += 1
        return last

    def close(self):
        self._client.close()
