# STiTy-Mobile/demo-web/

A translation server holding one model for the whole machine, and a proxy that puts several ASR
servers behind one port and picks per utterance which one the client hears.

- **Why the routing defaults are what they are — [LANGUAGE_ROUTING.md](LANGUAGE_ROUTING.md).** Read it
  before changing a default, a threshold, or the order of the decisions.
- Which local translation model — [core/translator/LOCAL_TRANSLATION.md](../../core/translator/LOCAL_TRANSLATION.md)
- `local_translation_server.py` serves `POST /translate` (optional `context`) and `GET /health`;
  `partial_demo/` holds `demo_proxy.py`, `lid_router.py` and the client (`web/show.html` is the demo page).

## Launching

From the repository root, in this order, and **start the ASR servers one at a time** — wait for
`Server listening` before the next. vLLM charges another process's allocations to its own KV-cache
profiling, so three started together die with `No available memory for the cache blocks`.

```bash
# 1) ASR servers, one at a time: ko 8766, en 8767, baseline 8768 (--rest)
PYTHONPATH=$PWD/Qwen3-ASR python Qwen3-ASR/examples/streaming_websocket_server.py \
  --model models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged --port 8766 \
  --gpu-memory-utilization 0.25 --enforce-eager --no-idle-shutdown \
  --local-translation-url http://127.0.0.1:8770 --local-translation-context 1
#    en: --model models/Qwen3-ASR-1.7B-en-silence-c80-merged --port 8767 | baseline: Qwen/Qwen3-ASR-1.7B 8768

# 2) the translator, after the ASR servers so its allocator sees the space that is left
PYTORCH_ALLOC_CONF=expandable_segments:True \
python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 \
  --model Qwen/Qwen3-4B-Instruct-2507 --quant 4bit --context-window 1

# 3) the proxy last — it holds the LID model
python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 \
  --route ko=8766,en=8767 --default 8766 --rest 8768 --dual --lid-scan \
  --lid-model openai/whisper-small --targets ko,en,es --translate-url http://127.0.0.1:8770
```

Use the **silence** finetune for en, not `en-dailytalk-seg`: that one is trained to mark `<SEG>`, so it
commits far more often — 41% of its finals in one live session were two words or fewer, and a fragment
translated alone comes out wrong (`Three days` → `세일`).

Open <http://localhost:8080> (dev) or `/show.html` (demo); forward port 8080 only. The stack is
**22.7 / 24.5 GiB** — 6.2 each for the ASR servers, 3.1 translator, 0.9 LID. 0.25 is roughly the
utilization floor, and the budget is yours to keep: vLLM leaves room for nothing else.

## The flags a multilingual demo needs

Each defaults to off. A demo where several languages go into one microphone needs all of them, and
leaving one out does not degrade gracefully — it drops utterances.

| Flag | What breaks without it |
|---|---|
| `--dual` | The model is chosen when the stream opens and keeps the whole session |
| `--rest 8768` | A verdict outside the route table matches no upstream, **both** copies are dropped, and the utterance vanishes |
| `--lid-scan` | One verdict covers a whole VAD segment; the half it does not fit disappears |
| `--lid-model openai/whisper-small` | base is accurate on native speech only — a Korean speaker's English routes to `ko`. small costs ~350 MiB more |
| `--targets ko,en,es` | One translation per utterance, whatever else is on screen |
| `--translate-url` | Direction stays whatever the ASR server declared, and the en finetune's declaration is a coin flip |

Every figure behind these defaults is native read speech; **accented L2 speech is unmeasured**, and
that is where this fails in practice.
