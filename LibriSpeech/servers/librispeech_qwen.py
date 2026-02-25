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
            gpu_memory_utilization=0.6, 
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


from pipeline.types import RecognizedToken, CommittedSentence
from pipeline.modules import CommitPolicy
import sys

class Qwen3CommitPolicy(CommitPolicy):
    def __init__(self, stability_threshold=5):
        # last_committed_text 대신 리스트로 관리하여 단어 개수 파악을 명확하게 함
        self.last_committed_words = []  
        self.uncommitted_words = []     # 아직 확정되지 않고 지켜보는 중인 단어들
        self.stability_counts = []      # uncommitted_words의 각 단어가 몇 번이나 유지되었는지 카운트
        self.stability_threshold = stability_threshold # 5번 대기
        self.sentence_counter = 0

    @property
    def last_committed_text(self):
        # 기존 run_qwen_streaming.py에서 policy.last_committed_text를 호출할 때 에러가 나지 않도록 프로퍼티로 제공
        return " ".join(self.last_committed_words)

    def process_token(self, token: RecognizedToken) -> CommittedSentence:
        current_words = token.text.strip().split()
        
        # 1. 이미 확정된 단어들은 제외하고, 현재 '미확정' 상태인 단어들만 추출
        num_committed = len(self.last_committed_words)
        new_uncommitted = current_words[num_committed:] if len(current_words) > num_committed else []

        # 2. 기존에 지켜보던 단어와 새로 들어온 단어를 비교하여 안정도(카운트) 계산
        updated_uncommitted = []
        updated_counts = []

        for i, word in enumerate(new_uncommitted):
            if i < len(self.uncommitted_words) and word == self.uncommitted_words[i]:
                # 단어가 이전 토큰과 똑같이 유지되었으면 카운트 +1
                updated_uncommitted.append(word)
                updated_counts.append(self.stability_counts[i] + 1)
            else:
                # 단어가 바뀌었거나 새로 추가된 단어라면 카운트를 1로 (재)설정
                updated_uncommitted.append(word)
                updated_counts.append(1)

        self.uncommitted_words = updated_uncommitted
        self.stability_counts = updated_counts

        # 3. 안정도 카운트가 threshold(5)에 도달한 단어들을 앞에서부터 차례로 확정(Commit)
        words_to_commit = []
        while self.stability_counts and self.stability_counts[0] >= self.stability_threshold:
            words_to_commit.append(self.uncommitted_words.pop(0))
            self.stability_counts.pop(0)

        # 4. 확정된 단어가 하나라도 있다면 출력 및 반환
        if words_to_commit:
            self.last_committed_words.extend(words_to_commit)
            committed_string = " ".join(words_to_commit)
            
            print(f"{committed_string} /", end=" ", flush=True, file=sys.stderr)
            self.sentence_counter += 1  
            
            return CommittedSentence(
                sentence_id=self.sentence_counter,
                text=committed_string,
                time_range=token.time_range,  
                language=token.language,      
                speaker=token.speaker,        
                commit_reason="stability_threshold_met" 
            )
            
        return None

    def _extract_new_words(self, old_text, new_text):
        # 스트리밍이 종료(finish)될 때 마지막으로 남은 단어들을 털어내기 위한 함수 (run_qwen_streaming.py 호환용)
        new_words = new_text.strip().split()
        return " ".join(new_words[len(self.last_committed_words):]) if len(new_words) > len(self.last_committed_words) else ""

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

    recognizer = Qwen3SpeechRecognizer(model_path=args.qwen_model_path)
    policy = Qwen3CommitPolicy(stability_threshold=5)
    
    # 두 모듈을 위에서 만든 어댑터로 감싸서 넘겨줌!
    online_processor = QwenOnlineProcessor(recognizer, policy)
    
    # (ASR 객체, OnlineProcessor 객체) 튜플 반환 (기존 로직과 동일)
    return recognizer, online_processor

import numpy as np
from pipeline.types import AudioSegment, SpeakerInfo, TimeRange

class QwenOnlineProcessor:
    """
    server_test.py가 요구하는 인터페이스(init, insert_audio_chunk, process_iter, finish)를 
    Qwen 모델과 CommitPolicy에 맞게 연결해주는 어댑터 클래스
    """
    def __init__(self, recognizer, policy):
        self.recognizer = recognizer
        self.policy = policy
        self.audio_buffer = []

    def init(self, offset=None):
        # VAD 파라미터 유연하게 받기
        self.audio_buffer = []
        self.offset = offset if offset is not None else 0.0

        # 모델 스트리밍 상태 초기화
        self.recognizer.state = self.recognizer.model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        self.policy.last_committed_words = []
        self.policy.uncommitted_words = []
        self.policy.stability_counts = []

    def insert_audio_chunk(self, audio):
        # 들어오는 오디오를 버퍼에 저장
        self.audio_buffer.append(audio)

    def process_iter(self):
        # 오디오 버퍼가 비어있으면 빈 딕셔너리 반환
        if not self.audio_buffer:
            return {}

        # 버퍼에 쌓인 오디오를 하나로 합침
        audio_chunk = np.concatenate(self.audio_buffer)
        self.audio_buffer = [] # 처리했으니 버퍼 비우기

        # Qwen3SpeechRecognizer가 기대하는 AudioSegment 형태로 변환
        segment = AudioSegment(
            speaker=SpeakerInfo(speaker_id=0, speaker_language=""),
            audio=audio_chunk,
            time_range=TimeRange(audio_start_time_ms=0, audio_end_time_ms=0) 
        )

        # 1. 모델 인식 (transcribe)
        token = self.recognizer.transcribe(segment)

        # 2. 5번 대기 로직을 거쳐 확정 (CommitPolicy)
        committed_sentence = self.policy.process_token(token)

        # 3. 확정된 텍스트가 있다면, server_test.py가 기대하는 딕셔너리 형태로 변환
        if committed_sentence and committed_sentence.text:
            return {
                'start': 0.0,  # Qwen은 현재 정확한 워드 레벨 타임스탬프를 제공하지 않으므로 임시값 할당
                'end': 0.0,
                'text': committed_sentence.text,
                'language': committed_sentence.language or 'ko' # 기본 언어 설정
            }
        
        # 확정된 단어가 아직 없으면(대기 중이면) 빈 딕셔너리 반환
        return {}

    def finish(self):
        # 스트리밍이 끝날 때 마지막으로 남은 버퍼 털어내기
        self.recognizer.model.finish_streaming_transcribe(self.recognizer.state)
        final_text = self.recognizer.state.text
        
        # 아직 확정 안 된 단어들 강제 추출
        remaining_words = self.policy._extract_new_words(self.policy.last_committed_text, final_text)
        
        if remaining_words:
            self.policy.last_committed_words.extend(remaining_words.split())
            return {
                'start': 0.0,
                'end': 0.0,
                'text': remaining_words,
                'language': 'ko'
            }
        return {}