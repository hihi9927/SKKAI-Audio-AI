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
| `--gpt-translation` | `core/translator/correct_and_trans.py` `GPTTranslator` — correction + translation in one call, `--context-window` sentences of history (default 5) |
| `--correction` | `core/translator/gpt_corrector.py` `GPTCorrector` — correction only, translation still Google |
| `--google-context` | Keeps Google Translate but feeds it prior sentences as context |
| `--local-translation` | Loads a local seq2seq translator (`core/translator/local_translator.py`) **into this process** — `--local-translation-model` picks MADLAD (`google/madlad400-3b-mt`) or NLLB |
| `--local-translation-url` | Calls a standalone translation server over HTTP instead of loading a model here (see below). Wins over `--local-translation` — giving a URL means "do not load a model in this process" |

### Running several servers on one GPU

Every translation call goes through `google_translate_async`, which delegates to the object set by
`set_local_translator`. `RemoteTranslator` in [core/translator/local_translator.py](core/translator/local_translator.py)
implements the same `translate(text, target, source) -> (translation, source_code)` interface over
HTTP, so `--local-translation-url` swaps it in at that one seam.

Keep the translation model in **one** process. With `--local-translation`, every ASR server loads its
own copy, and madlad400-3b costs about 7.1 GiB each — two ASR servers plus two translators do not fit
in 24 GiB.

```bash
# 1) translator once (about 7.1 GiB)
python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 --model google/madlad400-3b-mt

# 2) one ASR server per language, each pointing at it
PYTHONPATH=$PWD/Qwen3-ASR python Qwen3-ASR/examples/streaming_websocket_server.py \
  --model models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged --port 8766 \
  --gpu-memory-utilization 0.25 --enforce-eager --no-idle-shutdown \
  --local-translation-url http://127.0.0.1:8770
```

**vLLM's `--gpu-memory-utilization` is a fraction of the whole GPU claimed by that one process, and it
ignores what other processes already hold.** So the budget is simply
`translator + Σ(utilization × total) ≤ total` — you must leave room by hand; vLLM will not do it for you.
Measured on a 24564 MiB RTX 4090 with Qwen3-ASR-1.7B finetuned weights:

| Process | `--gpu-memory-utilization` | VRAM |
|---|---|---|
| `STiTy-Mobile/demo-web/local_translation_server.py` (madlad400-3b, fp16) | — | 7172 MiB |
| ASR (ko finetuned) | 0.25 | 6214 MiB — 3.87 GiB weights, 0.89 GiB KV cache (8288 tokens) |
| ASR (en finetuned) | 0.25 | 6214 MiB |
| **total** | | **19729 / 24564 MiB**, about 4.8 GiB spare |

0.25 is roughly the floor: the weights alone take 3.87 GiB, so anything below ~0.20 leaves no KV cache.
Going the other way, 0.55 gave one ASR server 13596 MiB and left no room for a second one.

One server is one model, and nothing in the pipeline picks a model by language — the client chooses
by which port it connects to. [STiTy-Mobile/demo-web/partial_demo/demo_proxy.py](STiTy-Mobile/demo-web/partial_demo/demo_proxy.py) does
that choosing for the web demo: it holds the upstream connection until the client's first `start`
arrives, routes on `start.lang` (falling back to a single-key `langMap`, then to `--default`), and sends
the `hello` itself so the handshake order the client expects is unchanged.

```bash
python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 --route ko=8766,en=8767 --default 8766
```

The route is fixed for the life of the stream — `start.lang` *is* the model choice. A later `config`
changes translation direction only. So a stream that leaves `lang` as `auto` to detect several
languages gets one model for all of them; per-utterance model switching needs a different design.

**`--lid` routes on the voice, not on what the client declared.** VAD finds where speech starts, then
whisper-base classifies the next `--lid-window` seconds, and that answer picks the upstream — so one
microphone carrying two languages still reaches the right model.
[STiTy-Mobile/demo-web/partial_demo/lid_router.py](STiTy-Mobile/demo-web/partial_demo/lid_router.py) holds the model and the measured
numbers behind those choices; the two that decide the design:

- **VAD first, always.** Feeding leading silence to a language classifier does not merely add noise, it
  inverts the answer — voxlingua107-ecapa scored 1.2% on 0.5s windows with the silence left in (chance
  is 50%) and 30.7% with it trimmed, because silence maps to one confident wrong label.
- **The ASR's own `language` field is not a routing signal.** On the same 166 clips at a 2s window,
  whisper-base got 100%, the ko finetune 93.4%, and the en finetune **53.0%** — finetuning wrecked its
  language ID, and it reports `en` while correctly transcribing Korean.

Waiting for the verdict costs nothing measurable: buffered audio is replayed to the upstream in full, ASR
catches up faster than realtime, and the first commit boundary lands later than the decision anyway
(first-subtitle latency 0.20s through the proxy vs 0.30s connecting directly). Short utterances do not
stall — once VAD sees the speech segment close, the router classifies what it has instead of waiting out
`--lid-max-wait` (a 0.6s "그렇지." routes at 1.6s).

**`--dual` decides per utterance instead of per stream.** `--lid` picks one server when the stream opens,
so a speaker who switches language mid-stream keeps the wrong model — the ko finetune writes English as
`헬로 나이스 미트 유.`, and the en finetune reports Korean as `en` so the translation comes back unchanged.
`--dual` sends the same audio to every server in the route table and lets `VerdictTracker` keep judging as
audio arrives, forwarding only the messages from the server that owns the language of that span.

**Select on the audio, never on the transcript.** Picking by script (Hangul vs Latin) breaks exactly where
it matters: when the ko model renders English in Hangul, its output *looks* Korean, so both servers' output
passes and the client sees the utterance twice. Judging the audio itself is independent of what either
model wrote. In a single stream alternating ko/en/ko/en, all eight committed segments came from the right
model with no duplicates.

Two guards keep the verdict list honest. Segments are matched by **overlap**, not by start time — silero
re-cuts the same utterance slightly differently as audio accumulates, which grew a 4-utterance stream to 29
verdicts before the fix and 10 after. And spans of 0.5s or less are not judged at all, since that is where
accuracy falls to 82.5%; a 0.4s sliver at the head of a Korean utterance had been coming back `en`.

**Locating a `final` on the verdict timeline is the delicate part**, because this server never fills
`final.start` — `segment_start_time` is initialised to 0.0 and never updated, so every final claims to start
at 0.0. Matching on `end` alone fails in both directions, and each failure was observed:

- With forward slack (`verdict.start <= end + 0.5`), the final of the *previous* utterance lands on the
  *next* utterance's verdict, because a commit boundary runs past the speech into the following silence. The
  correct output was dropped and the other server's rendering of the same audio passed in its place — one bug
  producing both a gap and a duplicate.
- With no span at all, a commit boundary that slides into the *middle* of the next utterance (when the pause
  is shorter than the server's 800ms VAD threshold) matches the wrong verdict, and since the other server's
  copy is dropped for being the wrong language, the whole utterance disappears.

So the proxy reconstructs the span itself: a final covers `(previous final's end from that same server, this
final's end)`, and the verdict with the largest overlap wins. Several finals sharing one boundary share the
span.

Deciding also has to wait for the verdict to exist. When it does not, falling back to the previous verdict is
wrong precisely at a language switch — the en server's `Hello,` rode the preceding English verdict out to the
client, and moments later the ko verdict arrived and let `안녕하세요.` through as well. `wait_for_verdict`
holds the message, but its condition must be **"is every speech segment that closed before this `end` judged
yet"**, not "does a verdict contain `end`": commit boundaries routinely land in silence, where no verdict will
ever contain them, and asking the wrong question made every final pay the full 1s timeout (median latency
1.16s instead of 0.22s).

**Routing the audio is not enough — the translation direction still comes from the ASR's self-report.** The
server derives its target from `client_lang_map[detected_source]`, and `detected` is the same label measured
at 53% for the en finetune. When a slot mixes languages (no VAD cut between them), the whole slot decodes
under the earlier language, so Korean speech arrives tagged `language English`, the target resolves to `ko`,
and the sentence is "translated" from Korean into Korean — `Oh, 반갑습니다.` came out as `오, 안녕하세요.`.
`--translate-url` lets the proxy re-translate against the verdict when the two disagree.

The root fix is to stop the slot from mixing languages at all: `init_streaming_state` already accepts
`language=` and turns it into the `language X<asr_text>` prompt prefix, but `_new_stream_slot` never passes it.
Cutting the slot on a language change and forcing the language there would fix the transcript, not just the
direction — forcing alone would not, since the previous language's text stays in the prefix. That work touches
the slot-switch path, which is the most exception-heavy code in the server, so it is deliberately left
separate from this change.

**Confidence buys time.** whisper returns a distribution over its language tokens, and taking only the argmax
throws away the part that says how sure it is. Measured over the same 166 clips (whisper-base, VAD-trimmed),
keeping only predictions above 0.8 turns short windows from unusable into decisive:

| window | argmax | at confidence ≥ 0.8 |
|---|---|---|
| 0.3s | 71.1% | 28.3% of clips, 97.9% correct |
| 0.5s | 82.5% | 45.8% of clips, 100% correct |
| 0.7s | 88.0% | 60.8% of clips, 98.0% correct |
| 1.0s | 92.8% | 76.5% of clips, 99.2% correct |

So `VerdictTracker.EARLY_STEPS` walks 0.3s@0.90 → 0.5s@0.85 → 0.7s@0.80 and settles as soon as one clears,
falling back to a 1.0s argmax otherwise. Over the 166 clips that rule scores **92.8% — identical to the fixed
1.0s window — while listening for a median of 0.70s**. It costs nothing extra to try several windows because
whisper pads every input to 30s, so a 0.3s clip and a 3s clip take the same 9ms. Ten of the twelve errors are
clips that never cleared a threshold and went the full 1.0s, which is the filter working as intended.

An early verdict is locked so a later, longer look cannot overturn it — but **only when the language is one
the route table knows**. Without that condition a Korean utterance was locked to `zh` at 0.5s and stayed
there. And a verdict outside the route table must fall back to the default server rather than matching
nothing: `zh` matched neither the ko nor the en upstream, so both copies were dropped and the utterance
vanished from the transcript entirely.

**Narrow what the LID may answer, but not how its confidence is measured.** Restricting the argmax to the
source languages the client actually selected (`start.langMap` keys, union the route table) removes the
answers nothing can serve — leaking Korean to `zh` or `tr` was 7 of the 12 errors. On the 166 clips that lifts
the staged rule from 92.8% to **97.0%** at the same median 0.70s of audio, and every remaining error is a
ko↔en confusion. What must *not* change is the confidence scale: re-softmaxing inside the narrowed set
concentrates the probabilities, so the same threshold means different things as the pool shrinks — at a 0.5s
window and a 0.8 threshold the surviving subset scored 100% over all languages, 97.1% over five, 95.5% over
just ko and en. Taking the argmax within the allowed set while reading its probability off the *full*
distribution keeps `EARLY_STEPS` calibrated no matter how many languages the client picked.

**A third server for everything else.** The finetunes mislabel languages they were not tuned for — Spanish
comes back tagged `ko` — so `--rest` points at a baseline Qwen3-ASR that takes any verdict outside the route
table:

```bash
python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 \
  --route ko=8766,en=8767 --default 8766 --rest 8768 --dual
```

Three ASR servers fit on the 24GiB card only if the translator shrinks. Measured: ko 6214MiB + en 6214MiB +
baseline 6830MiB + whisper-base 656MiB leaves no room for madlad400-3b's 7304MiB — the total overshoots by
213MiB, and dropping every server to `--gpu-memory-utilization 0.21` to compensate fails outright with
`Available KV cache memory: -0.05 GiB` (KV reaches zero at about 0.213, so 0.23 is the practical floor).
Running the translation server on `facebook/nllb-200-distilled-600M` instead costs 1570MiB and the whole set
lands at **21624 / 24564 MiB**. That trade is a memory decision, not a quality one: the repo's own CometKiwi
numbers are 0.8712 for Google and 0.8554 for madlad-3b, and NLLB-600M has never been measured here.

**Judge twice: once to route, once to be right.** The early thresholds settle on less than a second of audio,
which is enough to pick a server but not enough to be sure — a Spanish `Mi nombre es Daniel.` came back `ko`,
so the baseline and en servers both had the sentence right and both were dropped while the ko server's
`이름은 다니엘.` went out. Extending the evaluation set to 226 clips (60 Spanish added; every earlier number
here was ko/en only) shows where the accuracy actually is:

| audio heard | accuracy | errors |
|---|---|---|
| first 1.0s | 96.9% | ko→en 5, en→ko 1, es→ko 1 |
| first 2.0s | **98.7%** | ko→en 3 |
| 3s / 5s / whole segment | 98.7% | ko→en 3 |

Two seconds is where the gain stops, and it is free: a `final` only arrives after its segment has closed, so
by then the audio exists. `CONFIRM_SEC` re-judges every closed segment on its first 2s and overwrites the
early guess, and `settled()` now waits for that confirmation rather than the provisional value.

**The client's language selection is the whole pool — do not add the route table back.** An earlier version
unioned `ROUTES` into the allowed set "so a language with a server always stays available", which quietly
undid the selection: turning Korean off in the UI still let the LID answer `ko`. Spoken Spanish came back
`ko`, the baseline and en servers both had `¿Dónde está el baño?` right and were dropped, and the ko server's
`돈데 스타일 반요.` went out instead. The route table is now only the fallback for a stream that selected
nothing at all.

Every accuracy figure here is native speech (FLEURS, LibriSpeech, KsponSpeech). **Accented L2 speech is
unmeasured** and is where this fails in practice — a Korean speaker's Spanish is what surfaced the bug above.
`--lid-model openai/whisper-small` is the lever if it keeps happening: on the same 226 clips it reads 98.2%
at 1s and 99.1% at 2s against base's 96.9% / 98.7%, costing 884MiB instead of 656MiB with no measurable
latency change (the 30s padding dominates either way). The native-speech gain is small; the reason to prefer
it is accent robustness, which these clips cannot show.

**Narrow each upstream to the language it actually serves.** The ASR server turns `langMap`'s keys into a
`-100` logit bias on every other language name, so whatever the client selects, *every* server is allowed to
answer with it. With ko/en/es selected the baseline answered `ko` for spoken Spanish and wrote
`메야모 다니엘.` in Hangul — and because the proxy then re-translated that Hangul into Korean, the original
and the translation were the same string and the line appeared twice on screen. The proxy now rewrites
`start` and `config` per upstream: `{ko: en}` to the ko server, `{en: ko}` to the en server, and everything
outside the route table to `--rest`. A single allowed language makes the bias equivalent to forcing.

Equivalent, but not identical — `force_language` writes `language X<asr_text>` into the prompt and skips
detection entirely, while the bias still lets the model choose within what is left. The two coincide only
while each server ends up with exactly one language; select two non-ko/en sources and the baseline is back to
guessing between them. Forcing per utterance would need the verdict to reach the server, and forcing alone
would not be enough anyway: the slot's accumulated text stays in the prefix, so the model would be told to
continue an English sentence in Spanish. The slot has to be cut at the same time.

`--lang-hint` does exactly that, and is **off by default** because the trade is not one-sided. The proxy
sends `{"type": "lang_hint", "lang": "es", "fromSec": 13.1}` to whichever upstream owns a new verdict;
`_apply_lang_hint` sets `forced_language`, cuts the active slot so the previous language leaves the prefix,
and moves the audio from `fromSec` into the fresh slot. **Move `state.buffer` too, not just
`state.audio_accum`** — the accumulator only holds audio already consumed into chunks, and right after a
verdict it is routinely empty while the whole utterance sits in the buffer. Carrying only the accumulator
turned `Esto parece tener sentido…` into `de tener sentido…`.

**The hint should narrow the bias, not force the prompt.** Both only pin the language *tag* — neither
constrains the transcript's own characters, so a Spanish tag can still be followed by Hangul. But they differ in
what else they disturb. Forcing writes `language X<asr_text>` into the prompt, so the model never generates a
language name at all: the logit bias has nothing left to act on (hence the server's
`if state.allowed_languages and not state.force_language`), and the output arrives without the `language X`
prefix, which shifts the commit and SEG logic that reads that string. Biasing to a single language leaves the
format untouched and simply tightens what the current per-server pool already does.

Measured on the same ko/en/es stream:

| segment | hint off | **bias** | force |
|---|---|---|---|
| ko #1 | split into two | **one sentence** | one sentence + junk `그리고 그거` |
| en | `burgundy` | **`Burgundy`** | `Burgundy` |
| ko #2 | `아우랜더` | `아우랜더` | `아 우린 또` (what was said) |
| ko #3 | `어.` / `그지.` | **`어.` / `그지.`** | collapsed to `아니,` |
| junk fragments | 0 | **0** | 2 |

Bias keeps everything the off case got right and fixes the split sentence and the capitalisation; force trades
two junk fragments and a lost backchannel for one better line. So `--lang-hint` biases by default and
`--lang-hint-force` is the opt-in.

`--vad-min-silence` sets how much silence ends an utterance (default 800ms). Measured against 400ms on the
same audio: English segmentation identical (13), Korean **less** fragmented at 400 (16 → 14), latency within
0.05s either way, and en→ko chrF/BLEU identical at 34.0/13.9. Nothing recommends the change, so 800 stands.
Note the flag matters beyond the VAD itself — `_retry_vad_short_utterance` trims its tail by
`VAD_MIN_SILENCE_MS - 100ms`, a value that used to be hardcoded to 0.7s and would have cut 300ms of real
speech at 400ms. `--dual` routing does not depend on this setting at all: the proxy runs its own VAD.

**`PYTHONPATH` matters in a worktree.** `pip install -e ./Qwen3-ASR` pins `qwen_asr` to whichever
worktree it was installed from, so a second worktree's server silently imports the *other* tree's
package. The symptom is a type error rather than an import error, e.g.
`streaming_transcribe() got an unexpected keyword argument 'on_partial'`. Put this worktree's
`Qwen3-ASR/` on `PYTHONPATH` ahead of it.

### Backend Key Files

| File | Role |
|---|---|
| `core/translator/correct_and_trans.py` | `GPTTranslator` — correction + translation in one call (`--gpt-translation`) |
| `core/translator/gpt_corrector.py` | `GPTCorrector` — correction only, async with retry/backoff (`--correction`) |
| `core/translator/local_translator.py` | Local seq2seq translators (MADLAD / NLLB) plus `RemoteTranslator`, the HTTP client for the standalone server |
| `STiTy-Mobile/demo-web/local_translation_server.py` | Standalone translation server — loads the model once, serves `POST /translate` and `GET /health` |
| `core/meaning_segmentator/utils/` | Research scripts: GPT `<SEG>` marking, context translation, COMET eval |
| `core/research/` | CIF & context-scoring experiments (not on the runtime path) |
| `Qwen3-ASR/examples/streaming_websocket_server.py` | Production WebSocket server |
| `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py` | Eval server (wraps the above, adds FSL timing) |

`Qwen3-ASR/` is vendored upstream code (QwenLM/Qwen3-ASR) tracked directly — **not** a submodule. See [Qwen3-ASR/CLAUDE.md](Qwen3-ASR/CLAUDE.md).

### Mobile App (`STiTy-Mobile/`)

React Native 0.81 + Expo 54 (managed), TypeScript strict. Path alias `@/*` → `src/*`.

- **Screens:** `HomeScreen` is the only one registered in `App.tsx` — it carries both the language/mode picker and the live feed. Connection state is handled in-place, not by a separate loading screen.
- **`src/context/WebSocketContext.tsx`** owns the whole WS session: socket lifecycle, message parsing, and the hardcoded `RUNPOD_SERVER_URL` (`getServerUrl()` swaps in `<host>/asr` when `EXPO_PUBLIC_USE_LOCAL_ASR=1` on web). Update it there when changing environments.
- **Audio:** `react-native-live-audio-stream` → binary PCM frames → server → JSON `final` messages back. `useAudioRecording.web.ts` / `tts.web.ts` are web variants.
- **Languages** (`src/constants/languages.ts`): ko, ja, zh, es, en. Conversation modes `mode-1` (speaker), `mode-2` (one earphone), `mode-3` (both) — unrelated to the eval mode2/3/4 below.
- **Colors:** `HomeScreen` defines `LANG_COLORS` inline; there is no shared theme module.

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
| `start` | `lang`, `targetLang`, `displayMode`, `langMap`? | Begin streaming (`lang` = code or `"auto"`) |
| `config` | `langMap`, `lang`?, `targetLang`? | Change translation direction mid-stream (no restart) |
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
| `config_ok` | `lang`, `targetLang`, `langMap` | Echo of the applied `config` |
| `final` | `start`, `end`, `original`, `translation`, `language`, `commitReason` | Committed segment |
| `pair_hosted` / `pair_connected` / `pair_peer_left` / `pair_error` | `roomId`, … | Pairing lifecycle |

`langMap` maps a **detected** language code to its translation target (`{"ko":"en","ja":"ko"}`).
When present it wins over the `lang` ↔ `targetLang` pair rule in `_correct_and_translate`, so three or
more languages can each go somewhere different. Unknown or identity entries are dropped. Its **keys** —
the source languages — also replace the pair as the ASR `allowed_languages` set (targets are output only,
so they are not opened). That set is computed per stream slot, so a mid-stream `config` change switches
translation direction at once but reaches ASR only from the next commit on.

`commitReason`: `seg` (SEG token), `vad` (silence), `dot` (sentence-ending punctuation, needs `--enable-dot-commit`), `always` (every chunk, under `--always-commit`), `timeout`, `finish` (stream ended).

The eval server adds timing fields to `final` (`segmentId`, `audioStartSec`, `audioEndSec`, `fsl_sec`, `asr_inference_sec`, …) — benchmarking only, ignored by the app.

## Evaluation

Datasets under `evaluation/`: LibriSpeech (en); DailyTalk, KsponSpeech (ko); plus the separate AST track in
`ast/` (FLEURS, en→de/ko/ja/zh/es, scored by LAAL and BLEU).

Earlier benchmarks in other languages are no longer in the tree; their measured numbers are kept in
[evaluation/ARCHIVED_DATASETS_METRICS.md](evaluation/ARCHIVED_DATASETS_METRICS.md).

Commit-policy modes compared in `evaluation/LibriSpeech/paper_result/ASR/`: **mode2** = always-commit, **mode3** = dot-commit with confirm gate, **mode4** = en-finetuned weights with SEG-only commit.

Full CLI reference: [evaluation/TESTING_MANUAL.md](evaluation/TESTING_MANUAL.md).
