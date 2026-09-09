# STiTy-Mobile/demo-web/

A translation server holding one model for the whole machine, and a proxy that puts several ASR
servers behind one port and sends each utterance's audio to the server for its language.

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
  --local-translation-url http://127.0.0.1:8770 --no-translation --chunk-size 1.0
#    en: --model models/Qwen3-ASR-1.7B-en-silence-c80-merged --port 8767 | baseline: Qwen/Qwen3-ASR-1.7B 8768

# 2) the translator, after the ASR servers so its allocator sees the space that is left
PYTORCH_ALLOC_CONF=expandable_segments:True \
python STiTy-Mobile/demo-web/local_translation_server.py --port 8770 \
  --model unsloth/gemma-3-4b-it --quant 4bit --context-window 1

# 3) the proxy last — it holds the LID model
python STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080 \
  --route ko=8766,en=8767 --default 8766 --rest 8768 --dual --lid-scan \
  --lid-model openai/whisper-small --targets ko,en,es --translate-url http://127.0.0.1:8770 --context 1 \
  --pivot-via-en ar
```

`--chunk-size 1.0` (server default 2.0) shows the first partial about 0.4s sooner and doubles the partial
updates. It also makes the en finetune write a second `language English` header right after a `<SEG>` and
stop there; the server now cuts the slot on that pattern (`SEG-HEADER-RESET`) and re-decodes from the chunk
where the sentence began, so on the 143s test the two chunk sizes deliver the same sentences. It also
commits on commas more often (`Sure, <SEG> please sit down.`), and a comma fragment translated alone is
wrong (`Sure,` → `물론입니다.`); the server now holds a commit that ends in a comma and translates it
together with the next one (`FRAGMENT-DEFER` / `FRAGMENT-JOIN`), or alone when the slot closes first.

Use the **silence** finetune for en, not `en-dailytalk-seg`: that one is trained to mark `<SEG>`, so it
commits far more often — 41% of its finals in one live session were two words or fewer, and a fragment
translated alone comes out wrong (`Three days` → `세일`).

Open <http://localhost:8080> (dev) or `/show.html` (demo); forward port 8080 only. Another device
(a phone) mirrors the same subtitles with `/show.html?view=1`: it opens the `/view` socket, which the
proxy feeds with a copy of everything the microphone page receives, sends no audio, and takes its
language settings from the microphone page. A viewer starts blank; add `&replay=1` to get the last 8 finals first. The stack is
**23.0 / 24.5 GiB** — 6.1 each for the ASR servers, 3.5 translator, 0.9 LID. 0.25 is roughly the
utilization floor, and the budget is yours to keep: vLLM leaves room for nothing else.

The translator is gemma-3-4b-it: on the demo script it is the best model that fits the slot and the only
one that keeps Korean in 해요체. Context comes from the proxy (`--context 1`, the previous final of *any*
speaker, with its translation), not from the ASR servers — same-speaker context made translations worse,
cross-speaker context is +0.014 COMET — both measured in
[core/translator/LOCAL_TRANSLATION.md](../../core/translator/LOCAL_TRANSLATION.md).

## The flags a multilingual demo needs

Each defaults to off. A demo where several languages go into one microphone needs all of them, and
leaving one out does not degrade gracefully — it drops utterances.

| Flag | What breaks without it |
|---|---|
| `--dual` | The model is chosen when the stream opens and keeps the whole session |
| `--rest 8768` | A verdict outside the route table (Spanish) has no server of its own and goes to `--default`'s finetune, which writes it in Hangul |
| `--lid-scan` | One verdict covers a whole VAD segment; the half after a mid-utterance switch goes to the wrong server |
| `--lid-model openai/whisper-small` | base is accurate on native speech only — a Korean speaker's English routes to `ko`. small costs ~350 MiB more |
| `--lid-window 1.5` (default) | At 1.0s the verdict is 83% on Korean speakers' three-language talk, 91% at 1.5s; the utterance goes only where the first verdict says |
| `--targets ko,en,es` | Fallback audience when the page has not picked languages. Once the page sends `langMap`, the audience is its sources + targets + `targetLang` — a ko↔ar page gets ko and ar subtitles only, not the es column that a fixed `--targets` used to add |
| `--translate-url` | Direction stays whatever the ASR server declared; the baseline behind `--rest` still mislabels Spanish |
| `--context 1` | Every target is translated without the previous speaker's turn; `Mucho gusto` comes back `안녕하세요`, `In English, we say cake` as `케이크가 있어요` |
| `--pivot-via-en ar` | Arabic goes to Korean directly, and the 4B translator copies the Arabic through or invents a sentence in 15 of 224 FLEURS sentences (COMET 0.816); via English it is 0.831 and the copies are gone, for +0.55 s on the Korean subtitle. Measured for ko only — [results](../../evaluation/ast/results/fleurs_ar-ko_pivot_20260909/) |
| ASR `--no-translation` | Each server translates once more without context, the proxy throws that away — one wasted translator call per sentence and ~170 ms on every `final` |

Every figure behind these defaults is native read speech; **accented L2 speech is unmeasured**, and
that is where this fails in practice.
