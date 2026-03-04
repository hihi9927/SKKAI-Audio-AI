"""
ASR factory for Original mode - uses standard Whisper model (downloads if needed)
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

def librispeech_original_args(parser):
    group = parser.add_argument_group('Whisper arguments (Original mode)')
    group.add_argument('--model_path', type=str,
                       default='large-v2',  # Will download Whisper large-v2 model if not present
                       help='The file path to the Whisper .pt model or model name (e.g., "large-v2", "medium"). If not present, downloads automatically.')
    group.add_argument("--beams","-b", type=int, default=1, help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.")
    group.add_argument("--decoder",type=str, default=None, help="Override automatic selection of beam or greedy decoder. "
                        "If beams > 1 and greedy: invalid.")

    group = parser.add_argument_group('Audio buffer')
    group.add_argument('--audio_max_len', type=float, default=30.0,
                        help='Max length of the audio buffer, in seconds.')
    group.add_argument('--audio_min_len', type=float, default=0.0,
                        help='Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.')


    group = parser.add_argument_group('AlignAtt argument')
    group.add_argument('--frame_threshold', type=int, default=25,
                        help='Threshold for the attention-guided decoding. The AlignAtt policy will decode only ' \
                            'until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model. ')

    group = parser.add_argument_group('Truncation of the last decoded word (from Simul-Whisper)')
    group.add_argument('--cif_ckpt_path', type=str, default=None,
                       help='The file path to the Simul-Whisper\'s CIF model checkpoint. If not specified, uses default behavior.')
    group.add_argument("--never_fire", action=argparse.BooleanOptionalAction, default=False,
                       help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. " \
                       ". If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. " \
                        "Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.")

    group = parser.add_argument_group("Prompt and context")
    group.add_argument("--init_prompt",type=str, default=None, help="Init prompt for the model. It should be in the target language.")
    group.add_argument("--static_init_prompt",type=str, default=None, help="Do not scroll over this text. It can contain terminology that should be relevant over all document.")
    group.add_argument("--max_context_tokens",type=int, default=None, help="Max context tokens for the model. Default is 0.")


def librispeech_original_factory(args):
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

    a = { v:getattr(args, v) for v in ["model_path", "cif_ckpt_path", "frame_threshold", "audio_min_len", "audio_max_len", "beams", "task",
                                       "never_fire", 'init_prompt', 'static_init_prompt', 'max_context_tokens', "logdir"
                                       ]}
    a["language"] = args.lan
    a["segment_length"] = args.min_chunk_size
    a["decoder_type"] = decoder

    if args.min_chunk_size >= args.audio_max_len:
        raise ValueError("min_chunk_size must be smaller than audio_max_len")
    if args.audio_min_len > args.audio_max_len:
        raise ValueError("audio_min_len must be smaller than audio_max_len")

    logger.info(f"[ORIGINAL MODE] Using model: {a['model_path']}")
    logger.info(f"Arguments: {a}")

    # Import the base ASR classes from librispeech_whisper
    from librispeech_whisper import LibriSpeechWhisperASR, LibriSpeechWhisperOnline

    asr = LibriSpeechWhisperASR(**a)
    return asr, LibriSpeechWhisperOnline(asr)
