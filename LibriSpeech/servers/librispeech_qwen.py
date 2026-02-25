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

# --- (기존 Qwen3SpeechRecognizer, Qwen3CommitPolicy 코드 유지) ---

def qwen_streaming_args(parser):
    """Qwen 모델 실행에 필요한 인자(Arguments)를 팀 표준 파서에 추가"""
    parser.add_argument(
        '--qwen-model-path', 
        type=str, 
        default="Qwen/Qwen3-ASR-1.7B", 
        help='Path to Qwen3 ASR model'
    )
    # 필요한 다른 하이퍼파라미터가 있다면 여기에 추가

def qwen_streaming_factory(args):
    """
    팀의 파이프라인 구조에 맞춰 Qwen 모듈들을 조립해서 반환하는 팩토리 함수
    """
    # 1. 모듈 인스턴스화
    recognizer = Qwen3SpeechRecognizer(model_path=args.qwen_model_path)
    policy = Qwen3CommitPolicy()
    
    # 2. 파이프라인 조립 후 반환
    # (주의: 기존 팀 코드가 dict, tuple, 혹은 특정 Pipeline 객체 중 어떤 형태로 
    # 반환받길 원하는지에 따라 이 부분의 형태를 맞춰줘야 해!)
    return {
        "recognizer": recognizer,
        "commit_policy": policy,
        # 만약 ConversationManager(VAD)나 StabilityFilter가 필수라면
        # 팀의 기본 모듈을 임포트해서 여기에 같이 넣어줘야 해.
    }

import argparse

def qwen_streaming_args(parser):
    """Qwen 모델 실행에 필요한 인자를 팀 표준 파서에 추가"""
    group = parser.add_argument_group('Qwen Streaming arguments')
    group.add_argument('--qwen_model_path', type=str,
                       default="Qwen/Qwen3-ASR-1.7B",
                       help='The file path to the Qwen3 ASR model.')
    # 필요하다면 unfixed_token_num 같은 하이퍼파라미터도 이곳에 추가 가능

def qwen_streaming_factory(args):
    """
    팀의 파이프라인 구조에 맞춰 Qwen 모듈들을 조립해서 반환하는 팩토리 함수
    """
    print(f"[STREAMING MODE] Using Qwen model: {args.qwen_model_path}")

    # 1. 모델 인스턴스화 (기존의 asr 역할)
    recognizer = Qwen3SpeechRecognizer(model_path=args.qwen_model_path)
    
    # 2. 확정 정책 인스턴스화 (기존의 OnlineProcessor 역할)
    policy = Qwen3CommitPolicy()
    
    # 3. 반드시 튜플 형태로 반환 (모델 객체, 실시간 처리 객체)
    return recognizer, policy