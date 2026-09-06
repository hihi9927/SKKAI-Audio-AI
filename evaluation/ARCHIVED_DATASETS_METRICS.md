# 삭제된 평가 데이터셋 지표 아카이브

2026-09-06 에 `evaluation/` 에서 AMI, KtelSpeech, AliMeeting, (zh)RAMC, (es)CIEMPIESS, KokoroSpeech, ReazonSpeech, smartturn 을 삭제했다. 데이터·로그·결과 디렉토리는 git 에 없었으므로 복구할 수 없다. **이 문서가 그 실행들에 대해 남은 전부다.**

남긴 것은 각 런의 요약 지표뿐이다. 발화 단위 결과(`raw_results`), 서버 로그, 플롯은 함께 사라졌다. 테스트 클라이언트 스크립트(`test_qwen3_*.py`)와 AMI `words/` XML 은 git 에 추적되어 있었으므로 삭제 커밋 이전 이력에서 복구할 수 있다.

## 표 읽는 법

- **오차**: 데이터셋별 주 메트릭(WER 또는 CER). 낮을수록 좋다.
- **FTL**: first token latency, 초. 세션 전체 평균이며 일부 런은 클라이언트 기준이라 값이 크다.
- **FSL**: 서버가 첫 확정 문장을 내보내기까지의 지연(초). 측정하지 않은 런은 `—`.
- **커밋**: 확정 사유 비율과 총 확정 수 — `vad`(침묵), `seg`(SEG 토큰), `dot`(문장부호), `finish`(스트림 종료).
- **정책**: 클라이언트 `--policy` 값. AMI/smartturn 의 옛 런은 결과 JSON 의 `policy_N` 키에서 읽었다.
- **모델·범위**: 결과가 실제로 저장된 경로에서 읽었다. `meta.json` 의 `cli_args` 가 경로와 어긋난 런은 비고에 `경로와 meta.json 불일치` 로 표시했다 — 같은 태그로 다른 조건을 덮어썼거나 스크립트에 잘못된 `--scope` 를 넘긴 흔적이다.


## AMI (영어, 다화자 회의)

AMI Meeting Corpus. 오디오는 `evaluation/AMI/AMI/`, 단어 XML 전사는 `evaluation/AMI/words/`. 클라이언트에 두 경로를 각각 넘겨야 했다 (`--ami-dir`, `--words-dir`).


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 3 | 2026-05-03 | 4 | 14.93% (WER) | 21.49 | — | vad 100% / n=568 | — |
| baseline(1.0.0) | full | run_02 | 3 | 2026-05-06 | 1 | 125.97% (WER) | 12.82 | — | seg 28%, vad 72% / n=224 | — |
| baseline(1.0.0) | full | run_03 | 3 | 2026-05-06 | 4 | 15.76% (WER) | 13.05 | — | seg 46%, vad 54% / n=2214 | — |
| baseline(1.0.0) | full | run_04 | 3 | 2026-05-09 | 4 | 15.01% (WER) | 21.17 | — | vad 100% / n=2158 | force_reason 살리고 full test. sample/run07 이후 진행 ES2004b 돌리다 끊겨서 WER 점수가 비정상적으로 높게 나옴. 이후 재실행 하느라 meta 정보가 좀 다름. |
| baseline(1.0.0) | full | run_05 | 3 | 2026-05-10 | 4 | 20.84% (WER) | 13.00 | — | dot 41%, vad 59% / n=919 | dot 발생 확인 |
| baseline(1.0.0) | full | run_06 | 3 | 2026-05-10 | 4 | 18.10% (WER) | 13.01 | — | dot 40%, vad 60% / n=935 | dot 발생 확인 |
| baseline(1.0.0) | full | run_07 | 3 | 2026-05-16 | 4 | 17.89% (WER) | 13.28 | 0.53 | dot 40%, vad 60% / n=937 | baseline(1.0.0) / sample / chunk 2.0s / dot-commit slot reset 적용 후 WER 측정 / 경로와 meta.json 불일치: meta.scope=sample |
| baseline(1.0.0) | full | run_08 | 3 | 2026-05-17 | 4 | 19.84% (WER) | 13.02 | 0.77 | dot 73%, vad 27% / n=2011 | baseline(1.0.0) / full / chunk 2.0s / dot-commit / FSL fix |
| baseline(1.0.0) | full | run_09 | 3 | 2026-05-17 | 4 | 63.12% (WER) | 13.11 | 1.21 | dot 66%, vad 34% / n=1174 | baseline(1.0.0) / full / chunk 2.0s / dot-commit / FSL fix |
| baseline(1.0.0) | full | run_10 | 3 | 2026-05-22 | — | — | — | — | — | (지표 없음 — 중단된 런) 번역 버그 수정 후 baseline(1.0.0) AMI full |
| baseline(1.0.0) | full | run_11 | 3 | 2026-05-23 | — | — | — | — | — | (지표 없음 — 중단된 런) 번역 버그 수정 후 baseline(1.0.0) AMI full |
| baseline(1.0.0) | full | run_12 | 3 | 2026-05-23 | 4 | 17.32% (WER) | 13.54 | 1.56 | dot 71%, vad 29% / n=1965 | 번역 버그 수정 후 baseline(1.0.0) AMI full |
| baseline(1.0.0) | full | run_13 | 3 | 2026-05-21 | 4 | 17.31% (WER) | 13.60 | 1.59 | dot 71%, vad 29% / n=1961 | baseline AMI benchmark |
| baseline(1.0.0) | sample | run_01 | 3 | 2026-05-09 | 1 | 15.71% (WER) | 12.84 | — | dot 66%, seg 7%, vad 27% / n=315 | force_reason 살리고 full test. sample/run07 이후 진행 |
| baseline(1.0.0) | sample | run_02 | 3 | 2026-05-08 | 1 | 26.60% (WER) | 12.84 | — | vad 100% / n=292 | baseline WER & dot 발생 X 문제 해결 |
| baseline(1.0.0) | sample | run_03 | 3 | 2026-05-08 | 1 | 26.60% (WER) | 12.87 | — | dot 71%, vad 29% / n=292 | baseline WER & dot 발생 X 문제 해결 |
| baseline(1.0.0) | sample | run_04 | 3 | 2026-05-09 | 1 | 26.60% (WER) | 12.86 | — | vad 100% / n=292 | force_reason 살리고 full test. sample/run07 이후 진행 |
| baseline(1.0.0) | sample | run_05 | 3 | 2026-05-09 | 1 | 26.60% (WER) | 12.87 | — | dot 71%, vad 29% / n=292 | baseline WER & dot 발생 X 문제 해결 |
| baseline(1.0.0) | sample | run_06 | 3 | 2026-05-09 | 1 | 26.60% (WER) | 12.84 | — | vad 100% / n=292 | baseline WER & dot 발생 X 문제 해결 |
| baseline(1.0.0) | sample | run_07 | 3 | 2026-05-09 | 1 | 15.12% (WER) | 12.83 | — | vad 100% / n=317 | baseline WER & dot 발생 X 문제 해결 |
| baseline(1.0.0) | sample | run_08 | 3 | 2026-05-16 | 4 | 17.89% (WER) | 13.28 | 0.53 | dot 40%, vad 60% / n=937 | baseline(1.0.0) / sample / chunk 2.0s / dot-commit slot reset 적용 후 WER 측정 |
| finetuned(1.0.1) | full | run_01 | 3 | 2026-05-01 | 4 | 27.78% (WER) | 6.05 | — | seg 80%, vad 20% / n=1796 | — |
| finetuned(1.0.1) | full | run_02 | 3 | 2026-05-02 | 4 | 24.18% (WER) | 5.68 | — | seg 80%, vad 20% / n=1743 | — |
| finetuned(1.0.1) | full | run_03 | 3 | 2026-05-10 | 4 | 27.26% (WER) | 5.95 | — | seg 81%, vad 19% / n=2205 | finetuning full test |
| finetuned(1.0.1) | full | run_04 | 3 | 2026-05-10 | 2 | 26.17% (WER) | 12.53 | — | seg 80%, vad 20% / n=801 | finetuning full test |
| finetuned(1.0.1) | full | run_05 | 3 | 2026-05-12 | 4 | 45.46% (WER) | 6.59 | — | seg 100% / n=2502 | VAD double-translation fix + SEG slot reset 적용 full test |
| finetuned(1.0.1) | full | run_06 | 3 | 2026-05-14 | 4 | 31.80% (WER) | 5.87 | — | seg 78%, vad 22% / n=2688 | updated server logic (SEG/VAD fix) + en-merged, same conditions as run_04 SEG 마지막에 있으면 slot 초기화해주는 로직 보완해서 돌림 |
| finetuned(1.0.1) | full | run_07 | 3 | 2026-05-16 | 4 | 31.79% (WER) | 5.66 | 1.37 | seg 78%, vad 22% / n=2688 | finetuned(1.0.1) / full / chunk 2.0s |
| finetuned(1.0.1) | full | run_08 | 3 | 2026-05-21 | 4 | 31.48% (WER) | 5.75 | 2.13 | seg 79%, vad 21% / n=2665 | finetuned(1.0.1) AMI benchmark |
| finetuned(1.0.1) | full | run_09 | 3 | 2026-05-22 | 4 | 31.44% (WER) | 5.66 | 2.03 | seg 79%, vad 21% / n=2648 | 1.0.1 모델로 실행됨 (finetuned_silence_gpt(1.0.4) run_06에서 이동 — 서버가 Qwen3-ASR-1.7B-en-merged로 떠있었음, OPENAI_API_KEY 없 |
| finetuned(1.0.1) | full | run_10 | 3 | 2026-05-23 | — | — | — | — | — | (지표 없음 — 중단된 런) 번역 버그 수정 후 finetuned(1.0.1) AMI full |
| finetuned(1.0.1) | full | run_11 | 3 | 2026-05-23 | 4 | 31.44% (WER) | 5.65 | 2.26 | seg 79%, vad 21% / n=2648 | 번역 버그 수정 후 finetuned(1.0.1) AMI full |
| finetuned_gpt_trans(1.0.2) | full | run_01 | 3 | 2026-05-17 | 4 | 30.46% (WER) | 5.65 | 2.53 | seg 70%, vad 30% / n=2038 | finetuned_gpt_trans(1.0.2) / full / chunk 2.0s / GPT translation ctx=5 |
| finetuned_silence(1.0.3) | full | run_01 | 3 | 2026-05-15 | 4 | 19.76% (WER) | 6.78 | 1.70 | seg 81%, vad 19% / n=2259 | silence 파인튜닝 모델 AMI 평가 / 경로와 meta.json 불일치: meta.model=silence(1.0.3) |
| finetuned_silence(1.0.3) | full | run_02 | 3 | 2026-05-17 | 4 | 19.76% (WER) | 6.66 | 1.61 | seg 81%, vad 19% / n=2259 | finetuned_silence(1.0.3) / full / chunk 2.0s |
| finetuned_silence(1.0.3) | full | run_03 | 3 | 2026-05-23 | 4 | 19.01% (WER) | 7.09 | 2.74 | seg 82%, vad 18% / n=2306 | 번역 버그 수정 후 finetuned_silence(1.0.3) AMI full |
| finetuned_silence(1.0.3) | full | run_04 | 3 | 2026-05-24 | 4 | 18.74% (WER) | 6.71 | 2.44 | seg 82%, vad 18% / n=2313 | 번역 버그 수정 후 finetuned_silence(1.0.3) AMI full |
| finetuned_silence_gpt(1.0.4) | full | run_01 | 3 | 2026-05-17 | 4 | 21.46% (WER) | 6.95 | 2.90 | seg 69%, vad 31% / n=1709 | finetuned_silence_gpt(1.0.4) / full / chunk 2.0s / GPT translation ctx=5 |
| finetuned_silence_gpt(1.0.4) | full | run_02 | 3 | 2026-05-23 | — | — | — | — | — | (지표 없음 — 중단된 런) 번역 버그 수정 후 finetuned_silence_gpt(1.0.4) AMI full |
| finetuned_silence_gpt(1.0.4) | full | run_03 | 3 | 2026-05-24 | 4 | 18.81% (WER) | 6.66 | 3.32 | seg 82%, vad 18% / n=2313 | 번역 버그+GPT버그 수정 후 finetuned_silence_gpt(1.0.4) AMI full |
| finetuned_silence_gpt(1.0.4) | full | run_04 | 3 | 2026-05-22 | — | — | — | — | — | (지표 없음 — 중단된 런) finetuned_silence_gpt AMI benchmark |
| finetuned_silence_gpt(1.0.4) | full | run_05 | 3 | 2026-05-22 | 4 | 18.74% (WER) | 6.80 | 2.37 | seg 82%, vad 18% / n=2313 | finetuned_silence_gpt AMI benchmark |
| finetuned_silence_gpt(1.0.4) | full | run_06 | 3 | 2026-05-22 | 4 | 31.44% (WER) | 5.66 | 2.03 | seg 79%, vad 21% / n=2648 | 번역 버그 수정 후 재실행 (lang=auto → targetLang 직접 번역) |
| finetuned_silence_gpt(1.0.4) | full | run_07 | 3 | 2026-05-22 | 4 | 18.74% (WER) | 6.68 | 2.29 | seg 82%, vad 18% / n=2313 | 1.0.4 재실행 (silence 모델 + GPT 번역 정상화) |


## KtelSpeech (한국어, 전화 대화)

전화 대역(협대역 8kHz) 음성. 수치를 LibriSpeech와 직접 비교하면 안 된다. `--data-dir` 아래 `KtelSpeech/`(오디오)와 `label/`(전사).


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| finetuned(1.0.1) | sample | run_01 | — | 2026-05-13 | 1 | 128.63% (CER) | 2.48 | — | seg 94%, vad 6% / n=35 | KtelSpeech 긴 발화 시 앞부분 청크 짤림 없는지 확인 용 |
| finetuned(1.0.1) | sample | run_02 | — | 2026-05-13 | 1 | 137.85% (CER) | 2.47 | — | seg 50%, vad 50% / n=4 | 할루시네이션 발생 확인 |
| finetuned(1.0.1) | sample | run_03 | — | 2026-05-13 | 1 | 85.22% (CER) | 5.79 | — | seg 95%, vad 5% / n=74 | 할루시네이션 발생 확인 |
| finetuned(1.0.1) | sample | run_04 | — | 2026-05-13 | 1 | 87.29% (CER) | 4.68 | — | seg 95%, vad 5% / n=75 | 할루시네이션 발생 확인 |
| finetuned(1.0.1) | sample | run_05 | — | 2026-05-14 | 1 | 85.22% (CER) | 4.68 | — | seg 95%, vad 5% / n=74 | 롤백 이후 테스트 |
| finetuned_silence(1.0.3) | full | run_01 | — | 2026-05-18 | 112 | 74.25% (CER) | 3.13 | — | seg 97%, vad 3% / n=9300 | finetuned_silence 1.0.3 첫 테스트 |
| finetuned_silence(1.0.3) | full | run_02 | — | 2026-05-25 | 28 | 55.68% (CER) | 3.13 | — | seg 98%, vad 2% / n=1782 | finetuned_silence 두 번째 테스트. 한국어 무한반복으로 slot이 터지는 문제 발생 여부 확인 |


## AliMeeting (중국어, 다화자 회의)

`--data-dir` 아래 `AliMeeting/`(혼합 WAV)와 `label/`(화자별 TextGrid).


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 3 | 2026-05-18 | 3 | 39.01% (CER) | 4.21 | 1.33 | dot 98%, vad 2% / n=2834 | slot switch(규연)이후 첫 테스트 |
| baseline(1.0.0) | full | run_02 | 3 | 2026-05-21 | — | — | — | — | — | (지표 없음 — 중단된 런) 0521 규연님 수정 이후 첫 테스트 |
| baseline(1.0.0) | sample | run_02 | 3 | 2026-05-21 | 1 | 34.16% (CER) | 2.40 | 1.10 | dot 99%, vad 1% / n=819 | 중국어 dot commit 해결 확인 |
| baseline(1.0.0) | sample | run_03 | 3 | 2026-05-16 | — | — | — | — | — | (지표 없음 — 중단된 런) 중국어 dot commit 해결 확인 |
| baseline(1.0.0) | sample | run_04 | 3 | 2026-05-17 | — | — | — | — | — | (지표 없음 — 중단된 런) 중국어 dot 커밋 flush_lock 의 cursor 탐색 부분 수정 후 확인 |
| baseline(1.0.0) | sample | run_05 | 3 | 2026-05-17 | — | — | — | — | — | (지표 없음 — 중단된 런) audio_accum 리밋 제한 문제 해결 후 테스트 |


## (zh)RAMC (중국어, 단문 발화)

발화 11,793개 / 화자 20명. `--data-dir` 아래 `RAMC/`(화자별 WAV)와 `label/TRANS.txt`(TSV).


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 3 | 2026-05-22 | 10323 | 3.72% (CER) | 6.16 | 1.51 | dot 7%, vad 93% / n=11166 | RAMC 첫 테스트 |
| baseline(1.0.0) | full | run_02 | 3 | 2026-05-23 | 4605 | 4.31% (CER) | 5.36 | 0.33 | dot 8%, vad 92% / n=5043 | RAMC trailing silence 500ms |
| baseline(1.0.0) | full | run_03 | 3 | 2026-05-24 | 1 | 19.18% (CER) | 87.45 | 3.07 | dot 100% / n=456 | vad 비활성화, 화자별 음성 이어붙어서 돌림 |


## (es)CIEMPIESS (스페인어, 단문 발화)

단문 1,000개, 서브셋 4개(train/read/fm/description). `--data-dir` 아래 `CIEMPIESS/{train,read,fm,description}/`와 `label/CIEMPIESS_test.{fileids,transcription}`.


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 3 | 2026-05-24 | 989 | 17.17% (WER) | 3.91 | 1.07 | dot 1%, vad 99% / n=1011 | ciem 첫 테스트 |


## KokoroSpeech (일본어, 단문 낭독)

tiny 서브셋 308개. `--data-dir` 아래 `KokoroSpeech/`(WAV)와 `metadata.csv`(파이프 구분: `ID|Transcription|Reading`).


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 3 | 2026-05-25 | 308 | 18.59% (CER) | 5.52 | 1.01 | dot 12%, vad 88% / n=411 | — |


## ReazonSpeech (일본어, 단문 클립)

평균 5초 내외 클립 350개. `--data-dir` 아래 `ReazonSpeech/`(WAV)와 `label/metadata.csv`. **실행 결과가 남아 있지 않다** — 스크립트만 있었고 `results/`가 없었다.


실행 기록 없음.


## smartturn (VAD 독립 트랙)

SmartTurn V3 턴 검출기를 silero VAD 자리에 끼워 넣는 실험. LibriSpeech test-other 오디오로 돌렸고 결과는 `results_json/*.json` 단일 파일로 떨어졌다. 이 트랙만은 `.gitignore` 대상이 아니어서 코드와 결과 JSON 이 모두 git 에 있었다 — 발화 단위 결과까지 삭제 커밋 이전 이력에서 되살릴 수 있다.


| 모델 | 범위 | 런 | 정책 | 날짜 | 파일수 | 오차 | FTL(s) | FSL(s) | 커밋 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | debug_base_1 | 3 | 2026-03-17 | 1 | 0.00% (WER) | 15.06 | — | — | — |
| — | — | debug_smartturn_8766 | 3 | 2026-03-17 | 1 | 0.00% (WER) | 15.06 | — | — | — |
| — | — | qwen3_test_other_smartturn_1shot | 3 | 2026-03-17 | 1 | 0.00% (WER) | 15.07 | — | — | — |
| — | — | qwen3_test_other_smartturn_fixcheck | 3 | 2026-03-17 | 1 | 48.48% (WER) | 15.06 | — | — | — |
| — | — | qwen3_test_other_smartturn_latest | 3 | 2026-03-17 | 10 | 66.10% (WER) | 7.96 | — | — | — |
| — | — | qwen3_test_other_smartturn_retry | 3 | 2026-03-17 | 7 | 85.61% (WER) | 8.96 | — | — | — |
| — | — | qwen3_test_other_smartturn_run | 3 | 2026-03-17 | 8 | 71.90% (WER) | 8.17 | — | — | — |
| — | — | qwen3_test_other_smartturn_top10 | 3 | 2026-03-17 | 9 | 74.56% (WER) | 8.52 | — | — | — |
| — | — | qwen3_test_other_smartturn_top10_v2 | 3 | 2026-03-17 | 9 | 67.66% (WER) | 8.50 | — | — | — |
| — | — | qwen3_test_other_smartturn_top10_v3 | 3 | 2026-03-17 | 7 | 68.03% (WER) | 9.12 | — | — | — |


## 모델 이름

| 라벨 | 가리키는 것 |
|---|---|
| `baseline(1.0.0)` | 파인튜닝 없는 Qwen3-ASR |
| `finetuned(1.0.1)` | 언어별 파인튜닝 가중치 |
| `finetuned_gpt_trans(1.0.2)` | 파인튜닝 가중치 + GPT 번역 경로 |
| `finetuned_silence(1.0.3)` | 침묵 구간을 넣어 재학습한 가중치 |
| `silence(1.0.3)` | 위와 같은 가중치, 라벨만 다르게 찍힌 초기 런 |
| `finetuned_silence_gpt(1.0.4)` | 1.0.3 + GPT 번역 경로 |


## 삭제 당시 상태

- 회수한 용량: 약 320MB (AMI 164MB / KtelSpeech 85MB / (zh)RAMC 61MB / DailyTalk 제외 나머지).
- AMI 164MB 중 162MB 가 `results/` 의 서버 로그였다. 오디오는 이미 리포에 없었다.
- 남긴 데이터셋: LibriSpeech(주 벤치마크), DailyTalk, KsponSpeech, ast(AST 트랙).

