#!/usr/bin/env python3
"""
Diagnostic test for model hallucination issues
Tests model loading, inference, and hallucination detection
"""
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'SimulStreaming'))

import torch
import numpy as np

def test_pytorch_cuda():
    """Test PyTorch and CUDA setup"""
    logger.info("=" * 60)
    logger.info("PYTORCH AND CUDA DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("")

def test_model_loading():
    """Test model loading"""
    logger.info("=" * 60)
    logger.info("MODEL LOADING TEST")
    logger.info("=" * 60)

    try:
        from simul_whisper.config import AlignAttConfig
        from simul_whisper.simul_whisper import PaddedAlignAttWhisper

        model_path = os.path.join(parent_dir, 'SimulStreaming', 'large-v2.pt')
        cif_path = os.path.join(parent_dir, 'SimulStreaming', 'cif_models', 'large-v2_cif.pt')

        logger.info(f"Model path: {model_path}")
        logger.info(f"Model exists: {os.path.exists(model_path)}")
        logger.info(f"Model size: {os.path.getsize(model_path) / 1e9:.2f} GB")
        logger.info(f"CIF path: {cif_path}")
        logger.info(f"CIF exists: {os.path.exists(cif_path)}")

        cfg = AlignAttConfig(
            model_path=model_path,
            cif_ckpt_path=cif_path,
            language="auto",
            task="transcribe",
            frame_threshold=32,
            never_fire=False,
            use_speculative_decoding=False,
            logdir=None
        )

        logger.info("Creating model instance...")
        model = PaddedAlignAttWhisper(cfg)
        logger.info(f"Model loaded successfully!")
        logger.info(f"Model device: {model.model.device}")
        logger.info(f"Model dims: {model.model.dims}")
        logger.info("")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference(model):
    """Test inference on warmup audio"""
    logger.info("=" * 60)
    logger.info("INFERENCE TEST")
    logger.info("=" * 60)

    try:
        import torchaudio
        from simul_whisper.whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES

        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        logger.info(f"Warmup audio path: {warmup_path}")
        logger.info(f"Warmup audio exists: {os.path.exists(warmup_path)}")

        if os.path.exists(warmup_path):
            # Load audio
            audio, sr = torchaudio.load(warmup_path)
            logger.info(f"Audio shape: {audio.shape}, Sample rate: {sr}")

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)

            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)

            audio = audio.squeeze(0).numpy()
            logger.info(f"Audio duration: {len(audio) / 16000:.2f} seconds")

            # Compute mel spectrogram
            mel = log_mel_spectrogram(audio)
            logger.info(f"Mel spectrogram shape: {mel.shape}")

            # Pad to expected length
            mel = pad_or_trim(mel, N_FRAMES)
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).unsqueeze(0).to(model.model.device)
            else:
                mel = mel.unsqueeze(0).to(model.model.device)
            logger.info(f"Padded mel shape: {mel.shape}")

            # Detect language first
            logger.info("Detecting language...")
            from simul_whisper.whisper.decoding import detect_language
            with torch.no_grad():
                probs = detect_language(model.model, mel)
            detected_lang = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_lang}")

            # Transcribe using the model's decode method
            logger.info("Running full inference...")
            from simul_whisper.whisper.decoding import DecodingOptions

            options = DecodingOptions(language=detected_lang, without_timestamps=True)

            with torch.no_grad():
                result = model.model.decode(mel, options)

            logger.info(f"Transcription: {result.text}")
            logger.info("")

            # Check for hallucinations
            logger.info("=" * 60)
            logger.info("HALLUCINATION CHECK")
            logger.info("=" * 60)

            words = result.text.split()
            for i, word in enumerate(words):
                if len(word) > 1 and all(c == word[0] for c in word):
                    logger.warning(f"Potential hallucination at word {i}: '{word}' (repeated char: {word[0]})")

            logger.info("Inference test completed successfully!")
            logger.info("")

    except Exception as e:
        logger.error(f"Failed during inference: {e}")
        import traceback
        traceback.print_exc()

def test_memory_and_dtype():
    """Test memory usage and data types"""
    logger.info("=" * 60)
    logger.info("MEMORY AND DATA TYPE DIAGNOSTICS")
    logger.info("=" * 60)

    try:
        if torch.cuda.is_available():
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            logger.info(f"GPU Memory cached: {torch.cuda.memory_cached(0) / 1e9:.2f} GB if hasattr")

        logger.info("")

    except Exception as e:
        logger.error(f"Failed memory check: {e}")

if __name__ == "__main__":
    logger.info("\nSTARTING MODEL DIAGNOSTIC TEST\n")

    test_pytorch_cuda()
    test_memory_and_dtype()

    model = test_model_loading()

    if model is not None:
        test_inference(model)

    logger.info("=" * 60)
    logger.info("DIAGNOSTIC TEST COMPLETED")
    logger.info("=" * 60)
