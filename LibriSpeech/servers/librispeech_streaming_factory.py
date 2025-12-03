"""
ASR factory for Streaming mode using SimulStreaming model
"""
from whisper_streaming.base import OnlineProcessorInterface, ASRBase
import argparse
import os
import sys
import logging
import torch

from simul_whisper.config import AlignAttConfig
from simul_whisper.simul_whisper import PaddedAlignAttWhisper

logger = logging.getLogger(__name__)

# Get parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def librispeech_streaming_args(parser):
    group = parser.add_argument_group('Whisper arguments (Streaming mode)')
    group.add_argument('--model_path', type=str,
                       default=os.path.join(parent_dir, 'SimulStreaming', 'large-v2.pt'),
                       help='The file path to the SimulStreaming .pt model or model name (e.g., "large-v2").')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.")
    group.add_argument("--decoder",type=str, default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=15.0,
                        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--audio_min_len', type=float, default=0.0,
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=32,
                        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
                            'until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model. ')

    group = parser.add_argument_group('Truncation of the last decoded word (from Simul-Whisper)')
    group.add_argument('--cif_ckpt_path', type=str,
                       default=os.path.join(parent_dir, 'SimulStreaming', 'cif_models', 'large-v2_cif.pt'),
                       help='The file path to the Simul-Whisper\'s CIF model checkpoint.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False,
                       help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. " \
                       ". If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. " \
                        "Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.")

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=0, help="Max context tokens for the model. Default is 0.")

    group = parser.add_argument_group("Speculative decoding")
    group.add_argument("--use_speculative_decoding", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable speculative decoding with distil-whisper for faster inference (only works with greedy decoder)")
    group.add_argument("--assistant_model_path", type=str, default="distil-whisper/distil-large-v2",
                       help="Path to assistant model or HuggingFace model ID (e.g., distil-whisper/distil-large-v2) for speculative decoding")
    group.add_argument("--num_assistant_tokens", type=int, default=5,
                       help="Number of tokens to predict ahead with assistant model (default: 5)")

    group = parser.add_argument_group("Audio enhancement")
    group.add_argument('--denoise', action="store_true", default=True,
                       help='Enable audio denoising using SpeechBrain MetricGAN+. Reduces background noise.')


def librispeech_streaming_factory(args):
    logger.setLevel(args.log_level)
    decoder = args.decoder
    if args.beams > 1:
        if decoder == "greedy":
            raise ValueError("Invalid 'greedy' decoder type for beams > 1. Use 'beam'.")
        elif decoder is None or decoder == "beam":
            decoder = "beam"
        else:
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")
    else:
        if decoder is None:
            decoder = "greedy"
        elif decoder not in ("beam","greedy"):
            raise ValueError("Invalid decoder type. Use 'beam' or 'greedy'.")

    # Validate speculative decoding settings
    if args.use_speculative_decoding:
        if decoder != "greedy":
            raise ValueError("Speculative decoding only works with greedy decoder (beams=1)")
        if not args.assistant_model_path:
            raise ValueError("--assistant_model_path is required when using speculative decoding")

    a = { v:getattr(args, v) for v in ["model_path", "cif_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens', "logdir",
                                       "use_speculative_decoding", "assistant_model_path", "num_assistant_tokens"
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")

    logger.info(f"[STREAMING MODE] Using model: {a['model_path']}")
    if a["use_speculative_decoding"]:
        logger.info(f"[STREAMING MODE] Speculative decoding enabled with assistant model: {a['assistant_model_path']}")
    logger.info(f"Arguments: {a}")
    logger.info(f"[STREAMING MODE] Audio enhancement - denoise: {args.denoise}")

    # Import the base ASR classes from librispeech_whisper
    from librispeech_whisper import LibriSpeechWhisperASR, LibriSpeechWhisperOnline

    asr = LibriSpeechWhisperASR(**a)
    return asr, LibriSpeechWhisperOnline(asr)
