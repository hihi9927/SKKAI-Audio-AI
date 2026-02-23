import sys
import numpy as np
from .types import AudioSegment, RecognizedToken, CommittedSentence
from .modules import SpeechRecognizer, CommitPolicy
from qwen_asr import Qwen3ASRModel

class Qwen3SpeechRecognizer(SpeechRecognizer):
    def __init__(self, model_path="Qwen/Qwen3-ASR-1.7B"):
        print(f"Qwen3 모델 로딩 중... ({model_path})", file=sys.stderr)
        
        # 메모리 부족 해결을 위한 설정값 추가
        self.model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=0.9,  # 메모리 활용도를 90%까지 상향
            max_model_len=4096,          # 65536에서 4096으로 대폭 하향 (스트리밍에 충분함)
            max_new_tokens=64,
        )
        
        self.state = self.model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )

    def transcribe(self, segment: AudioSegment) -> RecognizedToken:
        audio_data = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
        self.model.streaming_transcribe(audio_data, self.state)
        return RecognizedToken(
            text=self.state.text or "",
            time_range=segment.time_range,
            confidence=1.0,
            language=self.state.language or "",
            speaker=segment.speaker,
        )


class Qwen3CommitPolicy(CommitPolicy):
    def __init__(self):
        self.last_committed_text = ""

    def process_token(self, token: RecognizedToken) -> CommittedSentence:
        current_text = token.text
        new_words = self._extract_new_words(self.last_committed_text, current_text)
        if new_words:
            # [4번 과제] 확정 지점에 빗금(/) 출력
            print(f"{new_words} /", end=" ", flush=True, file=sys.stderr)
            self.last_committed_text += (" " if self.last_committed_text else "") + new_words
            return CommittedSentence(text=new_words)
        return None

    def _extract_new_words(self, old_text, new_text):
        old_words, new_words = old_text.strip().split(), new_text.strip().split()
        return " ".join(new_words[len(old_words):]) if len(new_words) > len(old_words) else ""