# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STiTy** is a real-time multilingual speech translation system developed by SKKU SKKAI Audio AI lab. It pipelines ASR (Qwen3-ASR) → LLM correction → translation → mobile client via WebSocket.

## Commands

### Mobile App (`STiTy-Mobile/`)
```bash
npm install          # Install dependencies
npm start            # Expo dev server
npm run android      # Run on Android
npm run ios          # Run on iOS
```

### Python / Backend
```bash
# Install Qwen3-ASR
pip install -e ./Qwen3-ASR                  # transformers backend
pip install -e "./Qwen3-ASR[vllm]"          # with vLLM support

# Run the main WebSocket ASR server
python Qwen3-ASR/examples/streaming_websocket_server.py

# Run evaluation servers
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py

# Run benchmark evaluations
python evaluation/LibriSpeech/test_qwen3_librispeech.py
python evaluation/LibriSpeech/run_qwen_pipeline.py
```

## Architecture

### Pipeline

```
Audio Input
  → [A] ConversationManager (VAD / audio segmentation)
  → [ASR] Qwen3-ASR model (streaming, 52+ languages)
  → [B] CommitPolicy (decides when partial text becomes a committed sentence)
  → [C] GPTCorrector (LLM post-correction of ASR errors)
  → [Translation] Google Translate API
  → [Client] Mobile app via WebSocket
```

Core type definitions live in [core/types.py](core/types.py) — `AudioSegment → RecognizedToken → CommittedSentence → ValidatedSentence → TranslationResult`. Abstract module interfaces are in [core/modules.py](core/modules.py).

### Backend Key Files

| File | Role |
|---|---|
| `core/types.py` | Dataclass definitions for every pipeline stage |
| `core/modules.py` | Abstract base classes for pipeline modules |
| `core/llm_corrector/gpt_corrector.py` | Async OpenAI GPT corrector with retry/backoff |
| `core/meaning_segmentator/` | Sentence boundary detection utilities |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | Main production WebSocket server |
| `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py` | Eval server with forced-alignment logging |

The `Qwen3-ASR/` directory is a git submodule. Its `pyproject.toml` exposes CLI entry points: `qwen-asr-demo`, `qwen-asr-serve`, `qwen-asr-demo-streaming`.

### Mobile App (`STiTy-Mobile/`)

React Native + Expo (managed), TypeScript strict mode. Path alias `@/*` → `src/*`.

**Screens:** `HomeScreen` (language/mode selection) → `LoadingScreen` (server handshake) → `ConversationScreen` (live translation feed).

**Key hook:** `src/hooks/useWebSocket.ts` — manages WebSocket lifecycle, audio streaming, and message parsing. The server URL is currently hardcoded here (ngrok or LAN IP).

**Audio pipeline:** `react-native-live-audio-stream` → WebSocket binary frames → server → JSON translation results back.

Language color scheme: Purple `#8B5CF6` (Korean), Blue `#3B82F6` (English/Tibetan), Cyan `#06B6D4` (Indonesian). Design tokens are in `src/constants/theme.ts`.

## Key Configuration

- **Python deps:** `transformers==4.57.6`, `openai>=1.0.0`, `websockets>=12.0`, optional `vllm==0.14.0`
- **LLM corrector model:** configurable, default `gpt-5.4-mini` (check `gpt_corrector.py`)
- **Mobile target SDK:** configured in `app.json` and `eas.json`
- **Server connection:** hardcoded in `STiTy-Mobile/src/hooks/useWebSocket.ts` — update this when changing environments

## WebSocket Message Protocol

**Handshake sequence:** client connects → server sends `hello` → client sends `start` → server sends `ready` → audio streaming begins.

**Binary frames (Client → Server):** raw PCM audio, s16le, 16 kHz, mono. No wrapper.

### Client → Server (JSON)

| `type` | Key fields | Purpose |
|---|---|---|
| `start` | `lang`, `targetLang`, `displayMode` | Begin streaming (`lang`: language code or `"auto"`) |
| `stop` | — | End stream and close connection |
| `finish` | — | Flush final segment but keep connection open |
| `pair_host` | `roomId`, `myLang`, `targetLang`, `mode` | Create pairing room |
| `pair_join` | `roomId`, `myLang` | Join pairing room as guest |
| `pair_leave` | — | Exit pairing session |

### Server → Client (JSON)

| `type` | Key fields | Purpose |
|---|---|---|
| `hello` | `message` | Connection established |
| `ready` | `message` | Streaming initialized |
| `final` | `start`, `end`, `original`, `translation`, `language`, `commitReason` | Committed segment |
| `pair_hosted` | `roomId` | Host room created |
| `pair_connected` | `roomId`, `role`, `myLang`, `targetLang` | Pairing established (sent to both sides) |
| `pair_peer_left` | `roomId` | Peer disconnected |
| `pair_error` | `roomId`, `message` | Pairing failed |

**`final` message detail:**
```json
{
  "type": "final",
  "start": "0:00:02.50",
  "end": "0:00:05.75",
  "original": "Hello world",
  "translation": "안녕하세요 세계",
  "language": "en",
  "commitReason": "seg"
}
```
`commitReason` values: `"seg"` (model SEG token), `"vad"` (silence detected), `"finish"` (stream ended), `"dot"` (sentence-ending punctuation, optional).

The evaluation server (`streaming_websocket_server_fsl.py`) extends `final` with timing fields (`segmentId`, `audioStartSec`, `audioEndSec`, `fsl_sec`, `asr_inference_sec`, etc.) — these are for benchmarking only and not consumed by the mobile app.

## Evaluation Datasets

LibriSpeech, DailyTalk, KsponSpeech, CommonVoice. COMET (`unbabel-comet>=2.2.0`) is used for MT quality scoring. SmartTurn VAD is integrated for voice activity detection benchmarks.
