# CLAUDE.md

**STiTy** — real-time multilingual speech translation (SKKU SKKAI Audio AI lab). Streaming ASR
(Qwen3-ASR) → translation → mobile or web client over WebSocket.

```
Audio in → ConversationManager (VAD / segmentation)
         → Qwen3-ASR (streaming)
         → commit trigger (SEG token / VAD / dot / always)
         → translation (Google Translate by default; GPT and local backends are opt-in flags)
         → client via WebSocket
```

Two things about that pipeline are easy to get wrong. **One server is one model**, and nothing in it
picks a model by language — the client chooses by which port it connects to, and the web demo's proxy
does that choosing for it. And the **local translation model belongs in one process** for the whole
machine (`--local-translation-url`), not one copy per ASR server.

## Commands

```bash
pip install -e ./Qwen3-ASR                 # or "./Qwen3-ASR[vllm]" for the vLLM backend

python Qwen3-ASR/examples/streaming_websocket_server.py       # production server
python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown   # eval server
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other --model "baseline(1.0.0)" --scope sample

cd STiTy-Mobile && npm install && npm start                   # mobile app
```

## Where things are

This file is the map; every measurement and every step-by-step procedure lives next to the code it
describes.

| Path | What, and what to read |
|---|---|
| `Qwen3-ASR/` | Vendored upstream ASR (QwenLM/Qwen3-ASR), tracked directly — **not** a submodule. The production WebSocket server is `examples/streaming_websocket_server.py`. [Qwen3-ASR/CLAUDE.md](Qwen3-ASR/CLAUDE.md) |
| `core/` | Translation layer (GPT, local, remote) and segmentation research — [core/CLAUDE.md](core/CLAUDE.md). Which local model to load and how much context it can use: [core/translator/LOCAL_TRANSLATION.md](core/translator/LOCAL_TRANSLATION.md) |
| `STiTy-Mobile/` | The React Native app ([MOBILE_APP.md](STiTy-Mobile/MOBILE_APP.md)) and the web demo — proxy, language routing, launch order, VRAM budget ([demo-web/CLAUDE.md](STiTy-Mobile/demo-web/CLAUDE.md)) |
| `evaluation/` | Benchmark harness, ASR and AST tracks — [evaluation/CLAUDE.md](evaluation/CLAUDE.md), full CLI in [TESTING_MANUAL.md](evaluation/TESTING_MANUAL.md) |
| `models/` | Finetuned weights: `Qwen3-ASR-1.7B-{ko-silence-v4c900,en-silence-c80}-merged/` and `Qwen3-ASR-1.7B-en-dailytalk-seg/` (the eval alias `finetuned(1.0.1)`) |
| [docs/RUNNING_THE_SERVERS.md](docs/RUNNING_THE_SERVERS.md) | Translation flags, sharing one GPU, `--vad-min-silence`, the worktree `PYTHONPATH` trap |
| [docs/WEBSOCKET_PROTOCOL.md](docs/WEBSOCKET_PROTOCOL.md) | Every message type, `langMap` semantics, `commitReason` values |

## Key configuration

- **Backend deps** (`Qwen3-ASR/pyproject.toml`): `transformers==4.57.6`, `accelerate`, `librosa`, `gradio`, `flask`; optional `vllm==0.14.0`. Eval extras in `evaluation/LibriSpeech/requirements.txt`; `openai` is imported by `core/` but pinned nowhere.
- **LLM model:** default `gpt-5.4-mini` in both `correct_and_trans.py` and `gpt_corrector.py`.
- **Protocol:** connect → `hello` → `start` → `ready`, then raw PCM up (s16le, 16 kHz, mono) and JSON `final` messages back.
