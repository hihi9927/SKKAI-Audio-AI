from time import time
from typing import Any, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (ASRToken, Segment, PuncSegment, Silence,
                                          SilentSegment, SpeakerSegment,
                                          TimedText)


class TokensAlignment:

    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization
        self._tokens_index: int = 0
        self._diarization_index: int = 0
        self._translation_index: int = 0

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
        self.new_translation_buffer: Union[TimedText, str] = TimedText()
        self.new_tokens_buffer: List[Any] = []
        self.sep: str = sep if sep is not None else ' '
        self.beg_loop: Optional[float] = None

        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.diarization_buffer: List[ASRToken] = []

        self.last_punctuation = None
        self.last_uncompleted_punc_segment: PuncSegment = None
        self.unvalidated_tokens: PuncSegment = []

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = self.state.new_diarization, []
        self.new_translation, self.state.new_translation = self.state.new_translation, []
        self.new_tokens_buffer, self.state.new_tokens_buffer = self.state.new_tokens_buffer, []

        # 디버그: new_tokens 내용 확인
        if self.new_tokens:
            silence_count = sum(1 for t in self.new_tokens if hasattr(t, 'is_silence') and t.is_silence())
            token_texts = [getattr(t, 'text', '[SILENCE]') if not (hasattr(t, 'is_silence') and t.is_silence()) else '[SILENCE]' for t in self.new_tokens]
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[TokensAlignment] update() - new_tokens: {len(self.new_tokens)} items, {silence_count} silences, texts: {token_texts[:5]}...")

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        self.new_translation_buffer = self.state.new_translation_buffer

        # 메모리 최적화: 다이어리제이션이 꺼져 있으면 오래된 토큰 정리
        # 최근 1000개 토큰만 유지 (약 10-15분 분량)
        if not self.diarization and len(self.all_tokens) > 1000:
            self.all_tokens = self.all_tokens[-500:]

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        if segment.translation is None:
            segment.translation = ''
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                segment.translation += ts.text + (self.sep if ts.text else '')
            elif segment.translation:
                break


    def compute_punctuations_segments(self, tokens: Optional[List[ASRToken]] = None) -> List[PuncSegment]:
        """Group tokens into segments split by punctuation and explicit silence."""
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx: i+1],
                    )
                    segments.append(segment)
                    segment_start_idx = i+1

        final_segment = PuncSegment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments

    def compute_new_punctuations_segments(self) -> List[PuncSegment]:
        new_punc_segments = []
        segment_start_idx = 0
        self.unvalidated_tokens += self.new_tokens
        for i, token in enumerate(self.unvalidated_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i],
                    )
                if previous_segment:
                    new_punc_segments.append(previous_segment)
                segment = PuncSegment.from_tokens(
                    tokens=[token],
                    is_silence=True
                )
                new_punc_segments.append(segment)
                segment_start_idx = i+1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx: i+1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i+1

        self.unvalidated_tokens = self.unvalidated_tokens[segment_start_idx:]
        return new_punc_segments


    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker."""
        if not self.all_diarization_segments:
            return []
        merged = [self.all_diarization_segments[0]]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(segment)
        return merged


    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build segments when diarization is enabled and track overflow buffer."""
        diarization_buffer = ''
        punctuation_segments = self.compute_punctuations_segments()
        diarization_segments = self.concatenate_diar_segments()
        for punctuation_segment in punctuation_segments:
            if not punctuation_segment.is_silence():
                if diarization_segments and punctuation_segment.start >= diarization_segments[-1].end:
                    diarization_buffer += punctuation_segment.text
                else:
                    max_overlap = 0.0
                    max_overlap_speaker = 1
                    for diarization_segment in diarization_segments:
                        intersec = self.intersection_duration(punctuation_segment, diarization_segment)
                        if intersec > max_overlap:
                            max_overlap = intersec
                            max_overlap_speaker = diarization_segment.speaker + 1
                    punctuation_segment.speaker = max_overlap_speaker
        
        segments = []
        if punctuation_segments:
            segments = [punctuation_segments[0]]
            for segment in punctuation_segments[1:]:
                if segment.speaker == segments[-1].speaker:
                    if segments[-1].text:
                        segments[-1].text += segment.text
                    segments[-1].end = segment.end
                else:
                    segments.append(segment)

        return segments, diarization_buffer


    def get_lines(
            self,
            diarization: bool = False,
            translation: bool = False,
            current_silence: Optional[Silence] = None
        ) -> Tuple[List[Segment], str, Union[str, TimedText]]:
        """Return the formatted segments plus buffers, optionally with diarization/translation."""
        import logging
        logger = logging.getLogger(__name__)

        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ''
            for token in self.new_tokens:
                if token.is_silence():
                    logger.debug(f"[get_lines] SILENCE detected! current_line_tokens has {len(self.current_line_tokens)} tokens")
                    if self.current_line_tokens:
                        self.validated_segments.append(Segment().from_tokens(self.current_line_tokens))
                        self.current_line_tokens = []
                        logger.debug(f"[get_lines] Moved to validated_segments, now has {len(self.validated_segments)} segments")

                    # SilentSegment는 더 이상 추가하지 않음 (클라이언트에서 빈 라인으로 표시되므로)
                    # 대신 침묵 감지 시 이전 세그먼트를 완료 처리만 함
                else:
                    self.current_line_tokens.append(token)

            # 실시간 자막: 완료된 세그먼트는 유지하지 않음 (현재 진행 중인 문장만 출력)
            # 번역이 켜져 있을 때만 validated_segments 유지 (번역 정렬용)
            if not translation:
                self.validated_segments = []

            segments = []
            if self.current_line_tokens:
                segments.append(Segment().from_tokens(self.current_line_tokens))

        if current_silence:
            # 현재 침묵 중일 때도 SilentSegment를 추가하지 않음
            # 필요시 타임스탬프 정보만 마지막 세그먼트에 반영
            pass

        if translation:
            [self.add_translation(segment) for segment in segments if not segment.is_silence()]
        return segments, diarization_buffer, self.new_translation_buffer.text
