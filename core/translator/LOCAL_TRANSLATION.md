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

**Score context experiments with reference-based COMET, not CometKiwi.** Kiwi sees only source and
translation, so information pulled from a previous turn reads as invented — the same runs scored
+0.0066 on COMET-DA and −0.0009 on CometKiwi.
