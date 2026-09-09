# Local translation — model choice and context depth

Which model `--local-translation` / `local_translation_server.py` should load, and how many previous
turns to hand it. Every number here is measured; the code is
[local_translator.py](local_translator.py). The flags that switch these paths on are listed in the
repository-wide [CLAUDE.md](../../CLAUDE.md).

## Which local translator

Measured on a 5-language parallel set (en/ko/ja/zh/es), 600 directed pairs per candidate, scored with
`Unbabel/wmt22-comet-da` and `Unbabel/wmt22-cometkiwi-da`. VRAM is what nvidia-smi reports for the
process; latency is the median of a single sentence at `num_beams=4` (1 for the LLM):

| Model | VRAM | latency | COMET-DA | CometKiwi |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-4B-Instruct-2507` 4bit | 3.3 GiB | 170 ms | **0.8817** | **0.8377** |
| `google/madlad400-3b-mt` 4bit | 3.9 GiB | 436 ms | 0.8764 | 0.8336 |
| `facebook/nllb-200-distilled-1.3B` fp16 | 3.1 GiB | 124 ms | 0.8683 | 0.8254 |
| `facebook/nllb-200-distilled-1.3B` 4bit | 2.4 GiB | 188 ms | 0.8662 | 0.8241 |
| `facebook/nllb-200-distilled-600M` fp16 | 1.7 GiB | 80 ms | 0.8523 | 0.8136 |

Three results decide the choice.

**The distilled 1.3B beats the dense one** (0.8683 vs 0.8600) — picking `facebook/nllb-200-1.3B` by
name costs quality at the same size. **NLLB-1.3B loses nothing to 4-bit** (−0.0020, CI crosses zero)
but is 5× slower at 8-bit (887 ms), so 4-bit is the only quantization worth using there. And
**`mbart-large-50` and SeamlessM4T-v2 must not be used**: they collapse on non-English pairs, leaving
89 and 12 translations with none of the target script at all — Seamless renders
`죄송한데 한 번만 다시 말씀해 주시겠어요?` as `ごめんだけど一回だけ再言って 주시겠어요?`. Read only the
en column of a results table and they look fine.

MADLAD needs two things or it does not fit: `PYTORCH_ALLOC_CONF=expandable_segments:True` (4.5 GiB
without it) and keeping `DenseReluDense.wo` out of the quantization. transformers holds that layer in
fp32 against fp16 overflow, which is 2 GiB on madlad-3b; quantizing it to 4-bit destroys the model
(every output becomes noise like `basis scal потреб …`), while casting it to bf16 gives output
identical to fp32 at half the memory. `_shrink_t5_wo` in `local_translator.py` does that.

**Allocator reservation depends on free VRAM.** The same LLM measures 3.3 GiB when the ASR servers
already hold their share and 4.4 GiB on an empty card — PyTorch reserves more when there is room. Start
the translation server *after* the ASR servers and give it `PYTORCH_ALLOC_CONF=expandable_segments:True`.

## Measured again on the demo conversation (Korean speakers, ko/en/es)

The parallel set above is native, written text. The demo is Korean speakers switching between Korean,
English and Spanish, and there Qwen3-4B's Korean came out wrong in ways the set did not show: idioms
translated literally (`Nice to meet you` → `잘 알아가요`, `Please sit down` → `자리를 잡아주세요`,
`Mucho gusto` → `재미있게 보셨어요`), and Korean targets dropping to plain or written register
(`나는 루시아야. 메시코 출신이야`, `한국어는 나에게 어렵다`) while ko→es and es→en stayed natural.
Measured on the demo script — 42 sentences × 2 targets = 84 pairs, references written by hand, no
context, the prompt unchanged, GPU shared with nothing:

| model | COMET-DA | chrF | →ko | →en | →es | median | p90 | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-4B-Instruct-2507` 4bit | 0.912 | 61.6 | 0.906 | 0.951 | 0.877 | 144 ms | 225 ms | 3.75 GiB |
| `Qwen/Qwen3-4B-Instruct-2507` 8bit | 0.922 | 63.2 | 0.913 | 0.954 | 0.899 | 557 ms | 884 ms | 4.21 GiB |
| **`unsloth/gemma-3-4b-it` 4bit** | 0.920 | **67.3** | 0.917 | **0.958** | 0.885 | 172 ms | 249 ms | 4.00 GiB |
| `google/madlad400-3b-mt` bf16 | 0.921 | 64.9 | **0.932** | 0.929 | **0.902** | 146 ms | 234 ms | 6.53 GiB |
| `facebook/nllb-200-3.3B` fp16 | 0.911 | 61.9 | 0.916 | 0.932 | 0.885 | 84 ms | 112 ms | 6.28 GiB |

**The demo stack runs `unsloth/gemma-3-4b-it` at 4bit** (`google/gemma-3-4b-it` is the same weights
behind a gated download; the unsloth mirror is what is cached here). It is the best of the models that
fit the translator's slot (~3.5 GiB next to three ASR servers), and it is the only one that fixes the
register: 해요체 with 저 throughout (`제 이름은 루시아이고, 저는 멕시코에서 왔어요`, `저는 캐나다에서
왔어요`), `만나서 반갑습니다`, `케이크는 한국어로 어떻게 말해요?`. MADLAD scores higher into Korean but
needs 6.5 GiB and cannot take context; Qwen 8bit gains little and is 4× slower. Qwen3-8B and
Hunyuan-MT-7B are not measured yet — they need about 5.5 GiB at 4bit, which means lowering the ASR
servers' `--gpu-memory-utilization`, and their downloads run at ~1 MB/s on this line.

## Feeding the translator context

`--local-translation-context N` hands the previous N committed originals (same detected language only)
to the local translator. **Only the LLM backend can use them.** The concatenation trick behind
`--google-context` — join the previous lines with newlines, translate once, keep the last line — works
because Google preserves line breaks. Local seq2seq models do not: NLLB-1.3B mismatched the line count
16 times out of 16, dropping the current sentence entirely when given one context line and merging
everything into a single line when given three, so "the last line" becomes the whole context blob.
MADLAD emitted English noise instead of a translation and took 2.4 s. Both are accepted and ignored,
with one warning.

Depth measured on a 6-dialogue × 5-turn parallel set (600 directed pairs, Qwen3-4B-Instruct 4bit):

| context | COMET-DA | vs none | latency p50 | prompt tokens |
|---:|---:|---:|---:|---:|
| 0 turns | 0.8827 | — | 170 ms | 76 |
| **1 turn** | **0.8903** | **+0.0076** [+0.003, +0.013] | 170 ms | 90 |
| 2 turns | 0.8899 | +0.0073 | 167 ms | 98 |
| 3 turns | 0.8899 | +0.0072 | 170 ms | 99 |
| 4 turns | 0.8890 | +0.0063 | 170 ms | 99 |

**The gain saturates at one turn**, which is why `--context-window` on the translation server defaults
to 1. Depth is not limited by cost: latency is flat (193 ms even at 16 turns / 266 prompt tokens) and
VRAM grows 34 MiB over the same range. The gain lands almost entirely on English targets
(+0.019 vs +0.002~0.004 for ko/ja/zh) because ko/ja/zh drop subjects and objects that English must
supply, and the answer is only in the previous turn: `はい、どうぞ。` is "Yes, go ahead." alone and
"Yes, please sit." after `この席、空いていますか？`.

**In the demo stack the context is the same speaker's previous line, and there it hurts.** The +0.0076
above comes from a dialogue set where the previous turn is the *other* speaker. Behind the demo proxy
each ASR server hears one language, so `--local-translation-context 1` hands the translator the same
speaker's previous sentence — usually unrelated. Replaying the 143s demo recording with gemma, context
1 vs 0 changed 16 of 50 lines, and every difference that mattered was the context bleeding in: a
junk fragment `All` came back as the previous line's translation, `케이크에요.` became `The cake is
delicious.` after `여기 케이크도 맛있어요.`, `See you then.` became `그럼, 그래요.`. With 0 those are
`모든`, `It's cake.`, `그럼 그때 봐요.`. So the ASR servers hand the translator no context
(`--local-translation-context 0`).

**Context from the proxy, which sees every speaker in order, helps again.** `demo_proxy.py --context N`
keeps the last N finals across all servers — original, language, and the translation already made into
the target — and sends them as `{text, lang, translation}` items; `LLMTranslator._build_prompt` renders
those with their language and translation, and the proxy re-translates every target itself so the server's
context-free translation is replaced. Replaying the demo script in conversation order through gemma with
the model's own previous outputs as context:

| context (cross-speaker) | COMET-DA | →ko | →en | →es | median |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.915 | 0.910 | 0.956 | 0.881 | 162 ms |
| **1** | **0.929** | **0.928** | **0.959** | **0.900** | 162 ms |
| 2 | 0.922 | 0.927 | 0.957 | 0.883 | 150 ms |
| 3 | 0.923 | 0.929 | 0.957 | 0.885 | 151 ms |

One turn is the whole gain, as before, and it is the previous *speaker's* turn that carries it:
`Mucho gusto.` → `만나서 반갑습니다` instead of `안녕하세요` (+0.41), `저는 자주 와요` → `Vengo a menudo`
instead of `Yo voy a menudo` (+0.28), `In English, we say cake` → `케이크라고 해요` instead of
`케이크가 있어요` (+0.14, also on the live recording). The losses are small register shifts
(`See you then` → `Hasta luego`, −0.15). So the demo runs `--context 1` on the proxy and
`--context-window 1` on the translation server. One guard came out of the live replay: with a garbled
source the model imitated the context line's `원문 → 번역` shape (`Eso primero, besaki? → 어서, 좋아해요?`),
so `_translate_sync` keeps only what follows the last `→`.

**Score context experiments with reference-based COMET, not CometKiwi.** Kiwi sees only source and
translation, so information pulled from a previous turn reads as invented — the same runs scored
+0.0066 on COMET-DA and −0.0009 on CometKiwi.
