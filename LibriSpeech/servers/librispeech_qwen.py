import sys
import os
import numpy as np

from pipeline.types import AudioSegment, RecognizedToken, CommittedSentence
from pipeline.modules import SpeechRecognizer, CommitPolicy
from qwen_asr import Qwen3ASRModel

class Qwen3SpeechRecognizer(SpeechRecognizer):
    def __init__(self, model_path="Qwen/Qwen3-ASR-1.7B"):
        print(f"Qwen3 모델 로딩 중... ({model_path})", file=sys.stderr)
        
        self.model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=0.9, 
            max_model_len=8192,          
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
        self.sentence_counter = 0  

    def process_token(self, token: RecognizedToken) -> CommittedSentence:
        current_text = token.text
        new_words = self._extract_new_words(self.last_committed_text, current_text)
        
        if new_words:
            print(f"{new_words} /", end=" ", flush=True, file=sys.stderr)
            self.last_committed_text += (" " if self.last_committed_text else "") + new_words
            
            self.sentence_counter += 1  
            
            return CommittedSentence(
                sentence_id=self.sentence_counter,
                text=new_words,
                time_range=token.time_range,  
                language=token.language,      
                speaker=token.speaker,        
                commit_reason="streaming_update" 
            )
        return None

    def _extract_new_words(self, old_text, new_text):
        old_words, new_words = old_text.strip().split(), new_text.strip().split()
        return " ".join(new_words[len(old_words):]) if len(new_words) > len(old_words) else ""