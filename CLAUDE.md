# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

**STiTy** — real-time multilingual speech translation (SKKU SKKAI Audio AI lab). Pipelines streaming ASR (Qwen3-ASR) → translation (optionally LLM-corrected) → mobile client over WebSocket.

## Commands

```bash
# Backend install
pip install -e ./Qwen3-ASR              # transformers backend
pip install -e "./Qwen3-ASR[vllm]"      # with vLLM

# Production WebSocket server
python Qwen3-ASR/examples/streaming_websocket_server.py

# Evaluation server + benchmark client (see evaluation/TESTING_MANUAL.md)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py --no-idle-shutdown
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other --model "baseline(1.0.0)" --scope sample

# Mobile app (STiTy-Mobile/)
npm install && npm start                # or: npm run android / npm run ios
```

## Architecture

### Pipeline

```
Audio in → ConversationManager (VAD / segmentation)
         → Qwen3-ASR (streaming)
         → commit trigger (SEG token / VAD / dot / always)
         → translation:  Google Translate (default)
                      or GPTTranslator — correction + translation in one call (--gpt-translation)
         → mobile client via WebSocket
```

Translation is **Google Translate by default** (`translate.googleapis.com` gtx endpoint, called inline in the server). LLM paths are opt-in flags and silently fall back to Google Translate when `OPENAI_API_KEY` is missing:

| Flag | Effect |
|---|---|
| `--gpt-translation` | `core/correct_and_trans.py` `GPTTranslator` — correction + translation in one call, `--context-window` sentences of history (default 5) |
| `--correction` | `core/llm_corrector/gpt_corrector.py` `GPTCorrector` — correction only, translation still Google |
| `--google-context` | Keeps Google Translate but feeds it prior sentences as context |

Pipeline dataclasses: [core/types.py](core/types.py) (`AudioSegment → RecognizedToken → CommittedSentence → ValidatedSentence → TranslationResult`). Abstract interfaces: [core/modules.py](core/modules.py) — signatures only, no implementations.

### Backend Key Files

| File | Role |
|---|---|
| `core/correct_and_trans.py` | `GPTTranslator` — correction + translation in one call (`--gpt-translation`) |
| `core/llm_corrector/gpt_corrector.py` | `GPTCorrector` — correction only, async with retry/backoff (`--correction`) |
| `core/types.py` / `core/modules.py` | Dataclasses / abstract base classes |
| `core/meaning_segmentator/utils/` | Research scripts: GPT `<SEG>` marking, context translation, COMET eval |
| `core/research/` | CIF & context-scoring experiments (not on the runtime path) |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | Production WebSocket server |
| `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py` | Eval server (wraps the above, adds FSL timing) |

`Qwen3-ASR/` is vendored upstream code (QwenLM/Qwen3-ASR) tracked directly — **not** a submodule. See [Qwen3-ASR/CLAUDE.md](Qwen3-ASR/CLAUDE.md).

### Mobile App (`STiTy-Mobile/`)

React Native 0.81 + Expo 54 (managed), TypeScript strict. Path alias `@/*` → `src/*`.

- **Screens:** `HomeScreen` (language/mode select) → `ConversationScreen` (live feed). Connection state is handled in-place, not by a separate loading screen.
- **`src/context/WebSocketContext.tsx`** holds the shared WS session; **`src/hooks/useWebSocket.ts`** owns the socket lifecycle and message parsing. `SERVER_URL` is hardcoded at the top of that hook — update it when changing environments.
- **Audio:** `react-native-live-audio-stream` → binary PCM frames → server → JSON `final` messages back. `useAudioRecording.web.ts` / `tts.web.ts` are web variants.
- **Languages** (`src/constants/languages.ts`): ko, ja, zh, es, en. Conversation modes `mode-1` (speaker), `mode-2` (one earphone), `mode-3` (both) — unrelated to the eval mode2/3/4 below.
- **Colors** (`src/constants/theme.ts`): gradient Purple `#8B5CF6` → Blue `#3B82F6` → Cyan `#06B6D4`; language labels reuse the same three.

## Key Configuration

- **Backend deps** (`Qwen3-ASR/pyproject.toml`): `transformers==4.57.6`, `accelerate`, `librosa`, `gradio`, `flask`; optional `vllm==0.14.0`. Eval extras in `evaluation/LibriSpeech/requirements.txt` (`websockets`, `jiwer`, …); `openai` is imported by `core/` but not pinned there.
- **LLM model:** default `gpt-5.4-mini` in both `correct_and_trans.py` and `gpt_corrector.py`.
- **Finetuned weights:** `models/Qwen3-ASR-1.7B-{en,ko}-silence-*-merged/` (eval scripts still reference the older `Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged` path, which is not present).

## WebSocket Message Protocol

**Handshake:** connect → server `hello` → client `start` → server `ready` → audio streams.
**Binary frames (client → server):** raw PCM, s16le, 16 kHz, mono, no wrapper.

### Client → Server

| `type` | Key fields | Purpose |
|---|---|---|
| `start` | `lang`, `targetLang`, `displayMode` | Begin streaming (`lang` = code or `"auto"`) |
| `stop` / `finish` | — | End and close / flush final segment but stay open |
| `pair_host` | `roomId`, `myLang`, `targetLang`, `mode` | Create pairing room |
| `pair_join` | `roomId`, `myLang` | Join as guest |
| `pair_leave` | — | Exit pairing |
| `log` / `tts_log` | — | Client-side telemetry appended to server logs |

### Server → Client

| `type` | Key fields | Purpose |
|---|---|---|
| `hello` | `message` (eval server adds `serverConfig`) | Connected |
| `ready` | `message` | Streaming initialized |
| `final` | `start`, `end`, `original`, `translation`, `language`, `commitReason` | Committed segment |
| `pair_hosted` / `pair_connected` / `pair_peer_left` / `pair_error` | `roomId`, … | Pairing lifecycle |

`commitReason`: `seg` (SEG token), `vad` (silence), `dot` (sentence-ending punctuation, needs `--enable-dot-commit`), `always` (every chunk, under `--always-commit`), `timeout`, `finish` (stream ended).

The eval server adds timing fields to `final` (`segmentId`, `audioStartSec`, `audioEndSec`, `fsl_sec`, `asr_inference_sec`, …) — benchmarking only, ignored by the app.

## Evaluation

Datasets under `evaluation/`: LibriSpeech, AMI (en); DailyTalk, KsponSpeech, KtelSpeech (ko); KokoroSpeech, ReazonSpeech (ja); AliMeeting, (zh)RAMC (zh); (es)CIEMPIESS (es); plus the standalone `smartturn/` VAD track.

Commit-policy modes compared in `evaluation/LibriSpeech/paper_result/ASR/`: **mode2** = always-commit, **mode3** = dot-commit with confirm gate, **mode4** = en-finetuned weights with SEG-only commit.

Full CLI reference: [evaluation/TESTING_MANUAL.md](evaluation/TESTING_MANUAL.md).
