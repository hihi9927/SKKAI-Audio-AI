# Language routing — the measurements behind the defaults

Why the proxy routes the way it does. Every number here was measured on this machine; the operating
side (what to run, which flags are mandatory) is in [CLAUDE.md](CLAUDE.md), and the code that carries
these rules is [partial_demo/lid_router.py](partial_demo/lid_router.py) and
[partial_demo/demo_proxy.py](partial_demo/demo_proxy.py).

Read this before changing a default, a threshold, or the order of the decisions. Most of the entries
below exist because the obvious alternative was tried first and produced a specific, reproducible
failure.

The route is fixed for the life of the stream — `start.lang` *is* the model choice. A later `config`
changes translation direction only. So a stream that leaves `lang` as `auto` to detect several
languages gets one model for all of them; per-utterance model switching needs a different design.

## Routing on the voice (`--lid`)

**`--lid` routes on the voice, not on what the client declared.** VAD finds where speech starts, then
whisper-base classifies the next `--lid-window` seconds, and that answer picks the upstream — so one
microphone carrying two languages still reaches the right model.
[partial_demo/lid_router.py](partial_demo/lid_router.py) holds the model and the measured
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

## Deciding per utterance (`--dual`)

**`--dual` decides per utterance instead of per stream.** `--lid` picks one server when the stream opens,
so a speaker who switches language mid-stream keeps the wrong model — the ko finetune writes English as
`헬로 나이스 미트 유.`, and the en finetune reports Korean as `en` so the translation comes back unchanged.
`--dual` stays connected to every server in the route table, lets `VerdictTracker` keep judging as audio
arrives, and sends each speech span **only to the server that owns its language** — the others receive the
same number of samples as digital silence, so every server's clock stays the proxy's clock.

**Judge on the audio, never on the transcript.** Picking by script (Hangul vs Latin) breaks exactly where
it matters: when the ko model renders English in Hangul, its output *looks* Korean. Judging the audio itself
is independent of what either model wrote.

Two guards keep the verdict list honest. Segments are matched by **overlap**, not by start time — silero
re-cuts the same utterance slightly differently as audio accumulates, which grew a 4-utterance stream to 29
verdicts before the fix and 10 after. And spans of 0.5s or less are not judged at all, since that is where
accuracy falls to 82.5%; a 0.4s sliver at the head of a Korean utterance had been coming back `en`.

## Feeding each server only its language

**Gate the input; do not pick among the outputs.** The first `--dual` sent every server all the audio and
chose, per `final`, which server's message to forward. That needs each `final` placed on the verdict
timeline, and the server only reports the commit time (`end`; `start` is always 0.0). The proxy took
`(previous final's end from that server, this end)` as the span, which is wrong whenever a commit is late
or several sentences commit at once. Measured on a 143s ko/en/es conversation through that design:

- 11 of about 40 utterances reached the screen in no correct form. `만나서 반갑습니다` committed with
  `end=17.54`, inside the following English verdict, so the ko copy was dropped — and the en copy was
  dropped for being Korean. `Hola, ¿puedo sentarme aquí?`, `My name is Ben`, `¿Es tu primera vez aquí?`,
  `In English we say cake` and `El coreano es difícil para mí` went the same way.
- The en server committed seven sentences in one burst at 46–48s; they shared one span, matched one
  English verdict, and all passed — `저는 지윤이에요`, `May I have Lucia?` (Me llamo Lucia), `So we're in
  Mexico.` (Soy de México) — while the baseline's correct Spanish was dropped.
- Servers hearing languages they were not tuned for produced the rest of the noise: the ko server wrote
  `메`, `홀라`, `아`, `올라면`, `암호 2분,` for Spanish; the en server, once a slot mixed Korean into English,
  wrote a second `language English` header and lost everything after it (55–63s, 71–74s).

Every one of those needs a server to hear audio that is not its language. `Dispatcher` in
[partial_demo/demo_proxy.py](partial_demo/demo_proxy.py) removes that: each 0.1s piece of audio goes to
the server that owns the language of its speech segment and to nobody else. Selection disappears — a
`final` from the ko server is Korean because the ko server heard nothing else.

**Dispatch runs 0.6s behind the microphone in silence** (`DISPATCH_DELAY`), 0.1s inside an utterance whose
owner is decided (`LIVE_DELAY`). The speech onset is known only after silero sees it and the tracker runs
(every 0.25s of audio), about 0.35s after the onset; to send the 0.25s before the onset (`PRE_ROLL`) as real
audio rather than silence, that time must not have been sent yet. Once the owner is locked there is nothing
left to learn, so the gap closes to 0.1s. The verdict
needs `--lid-window` of speech, so the cursor **holds** at the onset until it exists, then flushes — ASR
catches up faster than realtime. One second after the window should have filled it stops waiting and uses
the previous language; at `finish` it flushes everything.

**The first verdict is the only one that matters now, so it hears 1.5s, not 0.3–1.0s.** The early-confidence
steps (`EARLY_STEPS`) and the 1.0s window were measured on native read speech. On the 143s conversation —
three languages, Korean speakers — whisper-small narrowed to ko/en/es scores this on the head of each of
the 35 utterances (CPU, fp32, silero-cut):

| window | 0.5s | 0.7s | 1.0s | 1.5s | 2.0s |
|---|---|---|---|---|---|
| correct | 60% | 69% | 83% | **91%** | 89% |

So `--dual` decides on the full window (`--lid-early` turns the confidence steps back on) and the window
defaults to 1.5s. The cost is about 1.5s of hold at every utterance start. What 1.5s still gets wrong is
the accent itself: `Nice to meet you` and `Hola, me llamo Daniel` from a Korean speaker come back `ko` at
every window. Replaying the same file through `VerdictTracker` offline (CPU fp32, GPU fp16 and fp32 agree
to the verdict) reproduces the table, so a live run that routes worse than this is a proxy bug, not the
model — two were found that way: a lock that matched segments by overlap inherited the previous
utterance's language once the open segment's end grew, and the pre-roll piece looked up the verdict at a
time before the segment and fell back to the previous verdict.

**One owner per speech segment.** Sent audio cannot be recalled, so the language chosen at the first piece of
a segment stays for that segment. The 2s confirmation (`CONFIRM_SEC`) still runs in the tracker and shows in
the verdict log, but it cannot move audio: letting it redirect the remainder would split one sentence across
two servers, and two fragments are worse than one whole sentence from the wrong server. Only a `--lid-scan`
split inside the segment changes the owner mid-segment, because that is a real language change.

**A scan split arrives about a second late**, so that second already went to the old owner. The new owner
gets the audio from `SPLIT_BACKOFF` (0.3s) before the split point up to the cursor as an extra send; the
proxy records the extra length per server and subtracts it when reading that server's `end`. The old owner
keeps its leaked second — its sentence may end with a foreign fragment.

**Silence hallucinations are filtered with what the proxy knows it sent.** A `final` is dropped when the
proxy sent *that* server no speech in the 8s before the final's `end`. The tighter span
`(previous final's end, this end)` was tried first and threw away real sentences: a slot's sentences commit
at nearly the same boundary (a dot commit at 79.0, the VAD commit at 79.2), so the second one covered no
speech. The server side does the rest — a slot that has received only digital zeros is not decoded at all
(`real_audio` in the slot), so hallucinations can only come from the tail of real speech, where the known
silence phrases are dropped by name. Spans of 0.5s or less that the tracker never judges go to the
previous owner when the previous speech ended within 2s, otherwise to the default server.

## Translation direction (`--translate-url`)

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

## How sure the language ID is

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

## The third server (`--rest`)

**A third server for everything else.** The finetunes mislabel languages they were not tuned for — Spanish
comes back tagged `ko` — so `--rest` points at a baseline Qwen3-ASR that takes any verdict outside the route
table:

```bash
python partial_demo/demo_proxy.py 8080 \
  --route ko=8766,en=8767 --default 8766 --rest 8768 --dual
```

## Judging twice

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

Two seconds is where the gain stops. `CONFIRM_SEC` re-judges every closed segment on its first 2s and
overwrites the early guess in the verdict list. Since audio is now dispatched on the early guess and cannot
be recalled, the confirmation no longer changes where an utterance goes (see "One owner per speech segment"
above); it remains the number to read when judging whether the early thresholds are good enough.

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

## Telling each upstream which language it serves

**Narrow each upstream to the language it actually serves.** Whatever the client selects, *every* server is
otherwise allowed to answer with it: with ko/en/es selected the baseline answered `ko` for spoken Spanish and
wrote `메야모 다니엘.` in Hangul — and because the proxy then re-translated that Hangul into Korean, the
original and the translation were the same string and the line appeared twice on screen. The proxy rewrites
`start` and `config` per upstream: `{ko: en}` to the ko server, `{en: ko}` to the en server, and everything
outside the route table to `--rest`. The ASR server turns those `langMap` keys into its allowed-language set.

**Restricting by token bias does not hold — the model spells around it.** Blocking the disallowed language
names token by token is the obvious implementation and it fails: with only `Spanish` allowed, the model
dropped ` Korean`(16134) and emitted ` K`(730) + `orean`(45195) instead, so the header still read
`language Korean` and the parser still saw Korean. Measured on the live stack, the baseline reported en 45 /
ko 26 / es 3 across six connections that all received `langMap={'es': 'ko'}`. Blocking those pieces too is
not an option either — a global `-100` on ` K` would wreck `Korea` and `Kim` in English transcripts.

`qwen_asr/inference/lang_header_lock.py` constrains the *characters* instead, and only while the header is
being written: a candidate token passes if the header bytes so far plus that token still prefix
`language <allowed name>`. Every way of splitting the word fails at its first piece, and once the name is
complete the mask lifts so the transcript itself is untouched. Requests that do not carry
`extra_args={"allowed_languages": [...]}` are ignored, so the benchmark paths are unaffected.

Note what this does *not* do: it pins the language tag, not the script. The baseline can still write Spanish
in Hangul; that failure needs a different fix.

`force_language` is the heavier alternative — it writes `language X<asr_text>` into the prompt and skips
detection entirely, so there is no header left for the commit and SEG logic to read. And forcing alone would
not be enough for a mid-stream switch anyway: the slot's accumulated text stays in the prefix, so the model
would be told to continue an English sentence in Spanish. The slot has to be cut at the same time.

`--lang-hint` does that, and is **off by default** because the trade is not one-sided. The proxy sends
`{"type": "lang_hint", "lang": "es", "fromSec": 13.1, "cut": false}` to whichever upstream owns a new
verdict; `_apply_lang_hint` sets `forced_language` (or the bias, by default). The proxy no longer asks for a
slot cut: since a server only hears its own language, the previous language is never in its prefix, and the
audio that stops at a language change closes its slot through VAD. When the server does cut (`cut: true`),
it moves the audio from `fromSec` into the fresh slot. **Move `state.buffer` too, not just
`state.audio_accum`** — the accumulator only holds audio already consumed into chunks, and right after a
verdict it is routinely empty while the whole utterance sits in the buffer. Carrying only the accumulator
turned `Esto parece tener sentido…` into `de tener sentido…`.

**The hint should narrow the allowed set, not force the prompt.** Both only pin the language *tag* — neither
constrains the transcript's own characters, so a Spanish tag can still be followed by Hangul. But they differ in
what else they disturb. Forcing writes `language X<asr_text>` into the prompt, so the model never generates a
language name at all: the header constraint has nothing left to act on (hence the server's
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

## Switches inside one utterance (`--lid-scan`)

**A verdict per VAD segment cannot see a switch inside one.** The tracker judges the head of each speech
segment and never looks again, so a speaker who changes language without pausing gets one label for the whole
span — and the half that label does not fit disappears. Measured: Spanish followed with no gap by 4s of Korean
produced a single `final` carrying only the Spanish; the Korean never reached the client.

`--lid-scan` walks an absolute-time grid instead. Every `--lid-scan-hop` it classifies the preceding
`--lid-scan-window` of audio, and when the answer differs from the current language for `--lid-scan-confirm`
consecutive windows it splits the verdict there. VAD does not go away — it becomes the mask that keeps a
window from straddling silence, since silence still maps to one confident wrong label. Defaults are
1.5s / 0.25s / 2, and the flag is **off by default**: a spurious cut breaks a sentence in two, which is worse
than missing a switch, and natural (non-read) speech is unmeasured.

Measured over 120 synthetic switches (ko/en/es, all six orderings, VAD-trimmed halves butt-joined with a 30ms
crossfade). "Clean" means the label sequence changed exactly once; latency is from the true switch to the
first confirmed window ending:

| window | hop | confirm | clean | false flips/pair | latency median | \|error\| p90 |
|---|---|---|---|---|---|---|
| 1.0s | 0.25 | 2 | 72.5% | 0.47 | 0.77s | 0.57s |
| **1.5s** | **0.25** | **2** | **92.5%** | **0.12** | **1.07s** | **0.62s** |
| 1.5s | 0.25 | 3 | 95.8% | 0.07 | 1.32s | 0.88s |
| 2.0s | 0.25 | 2 | 95.8% | 0.04 | 1.35s | 0.79s |
| 2.0s | 0.50 | 2 | 100% | 0.00 | 1.68s | 1.09s |

Detection itself never failed (≥98.3% everywhere), so the choice is between how clean the label sequence is
and how fast it settles. 1.5s beats 2.0s on both latency and localisation and loses only 7.5 points of
cleanliness; 2.0s/0.25/2 is the conservative alternative. Requiring consecutive agreement is the cheap lever —
at 1.5s/0.25 it cuts false flips from 0.32 to 0.12 per pair for 0.27s. A shorter hop makes that confirmation
cheaper, so 0.25 beats 0.5 once confirmation is on; the raw "clean" figure looks worse at 0.25 only because
twice as many windows give a stray label twice as many chances to appear.

The precondition holds: a window taken from the middle of an utterance is as accurate as one from its head.
Over 180 clips (ko/en/es, 60 each, VAD span ≥3.5s, offsets stepped 0.5s), narrowed to the client's languages,
a 2.0s window scored 100% at the head and 99.9% elsewhere, and 1.5s scored 100% against 99.2%. Accuracy is
flat across offset, so there is no reason to prefer the head other than that it arrives first.

**A split moves the audio, not just the label.** From the confirmed split point the pieces go to the new
owner, and the ~1s that already went to the old owner is re-sent to the new one from 0.3s before the
estimated point (see "A scan split arrives about a second late" above).
