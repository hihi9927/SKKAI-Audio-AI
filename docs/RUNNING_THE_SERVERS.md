# Running the servers

Operating notes for the ASR servers and the standalone translation server. The measured VRAM budget of
the three-server web demo, its launch order, and everything the proxy does with languages live in
[STiTy-Mobile/demo-web/CLAUDE.md](../STiTy-Mobile/demo-web/CLAUDE.md).

## Translation flags

Translation is **Google Translate by default** (`translate.googleapis.com` gtx endpoint, called inline
in the server). The LLM and local paths are opt-in, and the GPT ones silently fall back to Google
Translate when `OPENAI_API_KEY` is missing:

| Flag | Effect |
|---|---|
| `--gpt-translation` | `core/translator/correct_and_trans.py` `GPTTranslator` — correction + translation in one call, `--context-window` sentences of history (default 5) |
| `--correction` | `core/translator/gpt_corrector.py` `GPTCorrector` — correction only, translation still Google |
| `--google-context` | Keeps Google Translate but feeds it prior sentences as context |
| `--local-translation` | Loads a local translator (`core/translator/local_translator.py`) **into this process** — `--local-translation-model` picks the backend by name |
| `--local-translation-url` | Calls a standalone translation server over HTTP instead of loading a model here. Wins over `--local-translation` — giving a URL means "do not load a model in this process" |
| `--local-translation-context` | Number of previous originals handed to the local translator (default 0 = off). **Only the LLM backend can use them** |

`Qwen/Qwen3-4B-Instruct-2507` at 4bit with one turn of context is the local translator for a
two-speaker stream; the multilingual demo stack runs `unsloth/gemma-3-4b-it` at 4bit with no context —
the measurements behind both are in [core/translator/LOCAL_TRANSLATION.md](../core/translator/LOCAL_TRANSLATION.md).

## One translation model for the whole machine

Every translation call goes through `google_translate_async`, which delegates to the object set by
`set_local_translator`. `RemoteTranslator` in [core/translator/local_translator.py](../core/translator/local_translator.py)
implements the same `translate(text, target, source, context=None) -> (translation, source_code)`
interface over HTTP, so `--local-translation-url` swaps it in at that one seam.

Keep the translation model in **one** process. With `--local-translation`, every ASR server loads its
own copy, and madlad400-3b costs about 7.1 GiB each — two ASR servers plus two translators do not fit
in 24 GiB.

## Sharing one GPU

**`--gpu-memory-utilization` is a fraction of the whole GPU, so the budget is yours to keep:
`translator + LID + Σ(utilization × total) ≤ total`. vLLM will not leave room for anything else.**
**Start the ASR servers one at a time** — vLLM charges every allocation made while it profiles to its
own KV cache, so three launched at once kill each other.

## Segmentation

`--vad-min-silence` sets how much silence ends an utterance (default 800ms). Measured against 400ms on the
same audio: English segmentation identical (13), Korean **less** fragmented at 400 (16 → 14), latency within
0.05s either way, and en→ko chrF/BLEU identical at 34.0/13.9. Nothing recommends the change, so 800 stands.
Note the flag matters beyond the VAD itself — `_retry_vad_short_utterance` trims its tail by
`VAD_MIN_SILENCE_MS - 100ms`, a value that used to be hardcoded to 0.7s and would have cut 300ms of real
speech at 400ms. `--dual` routing does not depend on this setting at all: the proxy runs its own VAD.

## Working in a git worktree

**`PYTHONPATH` matters in a worktree.** `pip install -e ./Qwen3-ASR` pins `qwen_asr` to whichever
worktree it was installed from, so a second worktree's server silently imports the *other* tree's
package. The symptom is a type error rather than an import error, e.g.
`streaming_transcribe() got an unexpected keyword argument 'on_partial'`. Put this worktree's
`Qwen3-ASR/` on `PYTHONPATH` ahead of it.
