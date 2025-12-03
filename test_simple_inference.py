#!/usr/bin/env python3
"""
Simple inference test for model hallucination diagnosis
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'SimulStreaming'))

import torch
import torchaudio

def main():
    logger.info("=" * 70)
    logger.info("SIMPLE INFERENCE TEST")
    logger.info("=" * 70)

    # Check GPU
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model using standard Whisper (not SimulStreaming-specific)
    from simul_whisper.whisper import load_model

    model_path = os.path.join(parent_dir, 'SimulStreaming', 'large-v2.pt')
    logger.info(f"\nLoading model from: {model_path}")
    logger.info(f"Model file size: {os.path.getsize(model_path) / 1e9:.2f} GB")

    try:
        model = load_model(name=model_path)
        model.eval()
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return

    # Load warmup audio
    audio_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
    logger.info(f"\nLoading audio from: {audio_path}")

    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(0)
        audio = audio.numpy()
        logger.info(f"✓ Audio loaded: {len(audio)/16000:.2f} seconds")
    except Exception as e:
        logger.error(f"✗ Failed to load audio: {e}")
        return

    # Run inference
    logger.info(f"\nRunning inference...")
    try:
        with torch.no_grad():
            result = model.transcribe(audio, language="en")

        logger.info(f"✓ Inference completed")
        logger.info(f"\nTranscription:")
        logger.info(f"Text: {result['text']}")
        logger.info(f"Language: {result.get('language', 'unknown')}")

        # Check for hallucinations
        logger.info(f"\nHallucination check:")
        words = result['text'].split()
        hallucinations = []
        for i, word in enumerate(words):
            if len(word) > 2 and all(c == word[0] for c in word):
                hallucinations.append((i, word))
                logger.warning(f"  Suspicious word {i}: '{word}'")

        if not hallucinations:
            logger.info("  No obvious hallucinations detected")
        else:
            logger.warning(f"  Found {len(hallucinations)} potential hallucinations")

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("\n" + "=" * 70)
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
