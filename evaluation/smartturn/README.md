# SmartTurn Track (Evaluation-only)

This folder is an isolated experiment path for SmartTurn-v3 integration under
`evaluation/`.

## Added entrypoints

- `./run_qwen_pipeline_smartturn.py`
- `./verify_architecture_smartturn.py`

## Purpose

- Keep baseline (`run_qwen_pipeline.py`, `verify_architecture_vad.py`) unchanged.
- Enable SmartTurn parameter tuning and endpoint validation in a separate track.

## Quick start

```bash
pip install -r evaluation/smartturn/requirements.txt
python evaluation/smartturn/run_qwen_pipeline_smartturn.py --port 8011
```

Run the original Qwen3 server (recommended path for parity with your previous tests):

```bash
python evaluation/smartturn/run_qwen3_streaming_server_smartturn.py \
  --host localhost --port 8765 \
  --model Qwen/Qwen3-ASR-1.7B \
  --no-idle-shutdown

# Optional SmartTurn tuning (no export needed)
#   --st-prob-mode speech|endpoint
#   --st-threshold-on 0.55 --st-threshold-off 0.35
#   --st-min-silence-ms 1400 --st-min-utterance-ms 2500
#   --st-end-cooldown-ms 800 --st-ema-alpha 0.15
```

Offline trace:

```bash
python evaluation/smartturn/verify_architecture_smartturn.py \
  --audio path/to/audio.wav \
  --output-json evaluation/smartturn/results/architecture_smartturn_trace.json
```

## Important note

`turn_detector.py` now tries SmartTurn runtime first and falls back to RMS when
SmartTurn is unavailable.

If your SmartTurn package exposes a custom scorer function, set:

```bash
set STITY_SMARTTURN_SCORE_HOOK=your_module:your_score_function
```

The function can return:
- float
- numpy scalar/array
- dict with one of: `prob`, `probability`, `speech_prob`, `turn_prob`, `score`

`verify_architecture_smartturn.py` prints `score_backend` so you can confirm
whether `smartturn` or `rms_fallback` was used.

For server runs, the SmartTurn shim is provided by:
- `evaluation/smartturn/silero_vad.py`
- `evaluation/smartturn/run_qwen3_streaming_server_smartturn.py`

If your SmartTurn output is endpoint probability (default), keep:
`STITY_SMARTTURN_PROB_MODE=endpoint`
