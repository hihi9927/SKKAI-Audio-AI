"""데이터셋 공용 평가 하네스.

데이터셋마다 클라이언트를 통째로 복사해 쓰던 것을 한 곳으로 모았다. 복사본 셋
(LibriSpeech / DailyTalk / KsponSpeech)은 이름이 같은 함수 21개가 2,451줄을 차지했는데
바이트까지 같은 것은 31줄뿐이었다 — 한쪽 수정이 다른 둘에 가지 않아 동작이 갈려 있었다.

기준 구현은 LibriSpeech 클라이언트다. 가장 최신이고 FSL 서버가 내려주는 타이밍 필드를
전부 다룬다.

데이터셋이 실제로 다른 부분은 셋뿐이고, 그것만 어댑터가 채운다:

    목록 만들기   find_* : [{file_id, path, reference, speaker_id?}, ...]
    오디오 읽기   audio.load_soundfile / audio.load_raw_pcm
    채점         scoring.calculate_wer(영어) / scoring.calculate_cer(한국어)
"""
