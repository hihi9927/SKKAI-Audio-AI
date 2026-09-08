# WebSocket message protocol

The contract between the ASR servers (`Qwen3-ASR/examples/streaming_websocket_server.py`,
`evaluation/streaming_websocket_server_ast.py`), the proxy
([STiTy-Mobile/demo-web/partial_demo/demo_proxy.py](../STiTy-Mobile/demo-web/partial_demo/demo_proxy.py))
and the clients (`STiTy-Mobile/src/context/WebSocketContext.tsx`, the web demo, the benchmark scripts).


**Handshake:** connect → server `hello` → client `start` → server `ready` → audio streams.
**Binary frames (client → server):** raw PCM, s16le, 16 kHz, mono, no wrapper.

## Client → Server

| `type` | Key fields | Purpose |
|---|---|---|
| `start` | `lang`, `targetLang`, `displayMode`, `langMap`? | Begin streaming (`lang` = code or `"auto"`) |
| `config` | `langMap`, `lang`?, `targetLang`? | Change translation direction mid-stream (no restart) |
| `stop` / `finish` | — | End and close / flush final segment but stay open |
| `log` / `tts_log` | — | Client-side telemetry appended to server logs |

## Server → Client

| `type` | Key fields | Purpose |
|---|---|---|
| `hello` | `message` (eval server adds `serverConfig`) | Connected |
| `ready` | `message` | Streaming initialized |
| `config_ok` | `lang`, `targetLang`, `langMap` | Echo of the applied `config` |
| `final` | `start`, `end`, `original`, `translation`, `language`, `commitReason` | Committed segment |

`langMap` maps a **detected** language code to its translation target (`{"ko":"en","ja":"ko"}`).
When present it wins over the `lang` ↔ `targetLang` pair rule in `_correct_and_translate`, so three or
more languages can each go somewhere different. Unknown or identity entries are dropped. Its **keys** —
the source languages — also replace the pair as the ASR `allowed_languages` set (targets are output only,
so they are not opened). That set is computed per stream slot, so a mid-stream `config` change switches
translation direction at once but reaches ASR only from the next commit on.

`commitReason`: `seg` (SEG token), `vad` (silence), `dot` (sentence-ending punctuation, needs `--enable-dot-commit`), `always` (every chunk, under `--always-commit`), `timeout`, `finish` (stream ended).

The eval server adds timing fields to `final` (`segmentId`, `audioStartSec`, `audioEndSec`, `fsl_sec`, `asr_inference_sec`, …) — benchmarking only, ignored by the app.
