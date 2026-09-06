"""오디오 로더. 데이터셋마다 형식이 다른 유일한 입력 경로다."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


def load_soundfile(audio_path):
    """flac / wav 등 soundfile 이 읽는 형식 → float32 모노 16kHz."""
    import soundfile as sf
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        return audio
    except Exception as e:
        logger.error('Failed to load audio %s: %s', audio_path, e)
        return None


def load_raw_pcm(audio_path):
    """헤더 없는 raw PCM (s16le, 16kHz, mono) → float32 [-1, 1]."""
    try:
        audio_int16 = np.fromfile(audio_path, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32767.0
    except Exception as e:
        logger.error('Failed to load audio %s: %s', audio_path, e)
        return None
