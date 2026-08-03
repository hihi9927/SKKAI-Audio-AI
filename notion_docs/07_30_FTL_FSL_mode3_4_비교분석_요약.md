## 📅 날짜
2026-07-30

## 🔧 작업 내용

- mode3(rule-based/dot-commit, baseline `Qwen/Qwen3-ASR-1.7B`)와 mode4(seg-commit, finetuned `Qwen3-ASR-1.7B-ko-silence-v4c900-merged`)의 FTL(first_token_latency)/FSL(fsl_sec) 산출 기준 코드 조사.
  - FTL 계산식(`processing_start` ~ 첫 non-empty `final` 메시지 수신)은 두 모드 동일. 차이는 커밋 트리거 방식뿐: mode3는 문장부호(dot) 정규식, mode4는 모델이 학습된 `<SEG>` 토큰을 직접 뱉어야 커밋.
  - LibriSpeech 파일 `1688-142285-0002` 하나를 mode3/mode4 양쪽 결과에서 비교. mode3는 chunk0(2.0s)에서 바로 전체 문장 커밋, mode4는 chunk0+chunk1(4.035s)까지 가야 커밋 — "seg가 dot보다 먼저 나왔다"는 초기 추측은 데이터상 반대로 확인(seg가 더 늦게 커밋).
  - 청크당 encode/decode 시간(0.06~0.09초)은 두 모델이 비슷 → FTL 차이는 연산 속도 차이가 아니라 커밋 판단 정책(텍스트 마침표 vs 학습된 SEG 토큰) 차이로 결론.
- GPU(RTX 4090 1장) 위에서 mode4 단일 파일(`1688-142285-0002`) 재실행으로 위 결론 실측 검증.
  - 기존에 떠 있던 mode3 평가 서버(PID 1891625, 100개 fresh-start 재실행 중이었음)가 GPU 메모리 대부분 점유 → 완료 확인 후 사용자 승인 받아 종료, mode4 서버(`serve_mode4.sh`) 새로 기동.
  - `test_qwen3_librispeech.py --common-files`에 file_id 하나만 담은 JSON을 넘겨 단일 파일만 필터링해 실행 (`evaluation/LibriSpeech/paper_result/ASR/mode4/sample/single_1688-142285-0002/`).
  - 서버 로그(`[SEG]` 라인)로 실제 디코딩 중 `<SEG>` 토큰 등장 시점을 직접 확인: chunk0(audio_pos 2.0s) 처리 시점엔 `<SEG>` 없음 → chunk1 decode 시작(4.037s) 후 4.102s에 최초 등장. 커밋/flush 파이프라인 지연이 아니라 모델이 실제로 그 시점까지 SEG를 디코딩하지 않은 것으로 확인.
- 다른 파인튜닝 버전(`Doo12/Qwen3-ASR-1.7B-en-plus-merged`)을 HuggingFace에서 받아 동일 파일로 재테스트 완료.
- 원본 오디오(`1688-142285-0002.flac`) 자체에 긴 무음이 있는지 librosa RMS 분석으로 확인. 테스트 하네스가 `--trailing-silence-ms 5500`으로 인위적 무음을 뒤에 붙인다는 점도 함께 확인.
- SEG 파인튜닝(`ko-silence-v4c900`)이 왜 청크 경계에서 SEG를 주저하는지 학습 데이터/코드 관점에서 조사 (`Qwen3-ASR/finetuning/`, `evaluation/KsponSpeech/utils/generate_split_data.py` 등).
- **핵심 검증**: dot-commit이 실제로 오디오를 다 듣고 문장을 끝낸 것인지, 아니면 언어모델 prior로 앞서나가 문장을 완성(hallucination)한 것인지 확인하기 위해 원본 오디오의 앞 2.0초만 잘라 baseline(dot-commit) 서버에 단독으로 전송하는 실험 진행.
- 위 실험을 다른 파일 14개(1688 화자, duration 3.2~5.1s)로 확장. 각 파일을 `floor(duration/2.0)*2.0`초(직전 2초 청크 경계)로 잘라 단독 전송, 전체 오디오 기준 결과(mode3 run01)와 비교.
- 이 14개 파일에 대해 mode3(dot) vs mode4(seg) FTL을 직접 비교 (같은 test-dir/정렬 순서라 mode4 sample run01에 동일 file_id 존재).
- "dot-commit이 빠른 건 사실 의미 경계를 판단하는 게 아니라 청크 버퍼 끝나면 그냥 마침표를 찍어버리는 기계적 습성 아니냐"는 가설을 절삭 실험의 실제 출력 텍스트로 재검증.
- 긴 오디오 파일(15~20초, 세그먼트 7~10개) 4개를 골라 세그먼트 간격을 전수 확인 — 이 "2초마다 마침표" 패턴이 실제로 매 청크 예외 없이 반복되는지 확인.
- 이 패턴이 (A) 모델 자체의 디코딩 습성인지, (B) 서버 파이프라인이 인위적으로 마침표를 붙이거나 강제 커밋하는 버그인지 코드 레벨로 판별 (`Qwen3-ASR/qwen_asr/inference/sentence_boundary.py`, `qwen3_asr.py`, `examples/streaming_websocket_server.py` 전수 확인).

## 📊 결과 / 수치

| 항목 | mode3 (dot) | mode4 (seg) |
|---|---|---|
| 커밋 방식 | 문장부호 정규식 | 학습된 `<SEG>` 토큰 |
| 파일 1688-142285-0002 첫 커밋 시점(elapsed) | 2.203s | 4.238~4.239s (재현 2회 일치) |
| 첫 커밋까지 필요한 오디오 청크 | chunk0 (2.0s) | chunk0+chunk1 (4.035s) |
| sample run01 (100파일) FTL 평균 | 2.226s | 4.528s |
| sample run01 (100파일) WER | 8.38% | (기존 자료 기준, 이번 세션에서 재확인 안 함) |
| mode3 dot commit의 fsl_sec | 100% null (28개/이후 재실행 100개 모두) | 정상 산출됨 (예: 0.067s) |

| 항목 | 값 |
|---|---|
| 원본 오디오 실제 길이 | 2.835s (내부 긴 무음 없음, 최대 간격 0.14s) |
| trailing silence (하네스가 인위 추가) | 5.5s |
| `Doo12/Qwen3-ASR-1.7B-en-plus-merged` 테스트 WER | 180% (hallucination으로 무관한 문장 반복 생성) |
| baseline dot-commit에 앞 2.0초만 잘라 단독 입력 시 출력 (1688-142285-0002) | `"You don't mean that you thought me so silly."` (뒤 0.835초 안 들려줬는데도 완전한 문장 그대로 출력 — hallucination 확정) |
| 확장 테스트(14개 파일 절삭) 중 유의미하게 잘린(≥1초) 9개 결과 | **전부 정상**: 잘린 만큼만 부분 출력, hallucination 없음 (예: `1688-142285-0009` 1.54초 절삭 → "Why, it might have been." 로 정확히 끊김, "In the workhouse." 없음) |
| 절삭 폭 작은(<0.5초) 5개 | 전체 오디오 결과와 동일 출력 (해당 구간이 순수 trailing silence/숨소리였을 가능성 높아 hallucination 판단 불가) |

**14개 파일 dot(mode3) vs seg(mode4) FTL 비교** (모두 dur 3.19~5.06s):

| 파일 | dot_FTL | seg_FTL | 차이 | seg 필요 청크 수 |
|---|---|---|---|---|
| 0003 | 2.202 | 6.264 | +4.061 | 3 |
| 0004 | 2.204 | 6.278 | +4.074 | 3 |
| 0005 | 2.204 | 4.203 | +1.999 | 2 |
| 0009 | 2.204 | 4.340 | +2.136 | 2 |
| 0012 | 2.202 | 4.324 | +2.122 | 2 |
| 0025 | 2.202 | 4.205 | +2.002 | 2 |
| 0027 | 2.204 | 4.164 | +1.960 | 2 |
| 0042 | 2.202 | 4.263 | +2.061 | 2 |
| 0043 | 2.203 | 4.229 | +2.026 | 2 |
| 0044 | 2.202 | 4.208 | +2.006 | 2 |
| 0008 | 2.203 | 2.203 | 0.000 | 1 |
| 0019 | 2.203 | 2.203 | 0.000 | 1 |
| 0020 | 2.204 | 2.204 | 0.000 | 1 |
| 0038 | 2.203 | 2.203 | 0.000 | 1 |

10/14는 seg가 확실히 늦음(1.96~4.07초 차, chunk 2~3개 필요), 4/14는 dot과 완전히 동일(둘 다 chunk 1). **dot_FTL은 14개 전부 2.20~2.20초로 사실상 고정값** — 문장 길이/내용 무관하게 항상 첫 청크(2초)에서 커밋.

**"청크 끝나면 그냥 마침표 찍어버리는 것 아니냐" 가설 검증 — 절삭 출력에서 문법적으로 미완성인데 마침표 찍힌 사례:**
- `1688-142285-0012` (2.0초 절삭): `"No one came forwards to help the."` — "the" 뒤 목적어 없음, 미완성인데 마침표.
- `1688-142285-0042` (2.0초 절삭): `"But for a minute or two, she."` — 주어만 있고 동사 없음, 미완성인데 마침표.
- `1688-142285-0019` (2.0초 절삭): `"Not vicious. He never."` — "He never." 동사 보어 없음, 미완성인데 마침표.

**긴 파일(15~20초) 세그먼트 간격 전수 확인 — 4개 파일 전부 예외 없이 정확히 2.00초 간격:**

| 파일 | duration | 세그먼트 수 | 세그먼트 간격 |
|---|---|---|---|
| 1688-142285-0000 | 15.00s | 8 | 전부 2.00s (0→2→4→6→8→10→12→14→16) |
| 1688-142285-0031 | 16.80s | 9 | 전부 2.00s |
| 1688-142285-0040 | 20.05s | 10 | 전부 2.00s |
| 1688-142285-0011 | 14.05s | 7 | 전부 2.00s |

문법적으로 명백히 미완성인 지점에도 마침표 찍힌 예: `"But his, he."` `"Of strength and patience, to."` `"Rula."` `"Gambling, wild."` `"People's money to regain his."` — 34개 세그먼트 전수 확인, 예외 0건. 즉 dot-commit은 실질적으로 mode2(always-commit, 2초 고정 청킹)와 커밋 타이밍이 구분 안 되고, 차이는 청크 조각 끝에 마침표를 붙이느냐뿐.

## 🐛 발견된 문제 및 해결

- **mode3 dot commit의 `fsl_sec`가 항상 null**: char-ratio 기반 token index 역추적(`dot_char_ratio`)이 실패해 `fsl_sec` 미산출. `description.txt`에 "SEG-parity fix" 시도 흔적이 있었으나, 이번 세션 중 진행된 fresh-start 재실행(100개, run01) 완료 후에도 여전히 `avg_fsl_sec: N/A` — **근본 원인 미해결** (아래 미해결 작업 참고).
- **HuggingFace 다운로드 401 에러**: 이 GPU 서버의 CLI 토큰 캐시가 3월 19일자 stale 토큰이라 `Doo12/Qwen3-ASR-1.7B-en-plus-merged`(private repo) 접근 거부. `.env`의 `HF_TOKEN`으로 재인증해 해결(계정: Doo12).
- **`hf_transfer` 에러**: `.env`에 `HF_HUB_ENABLE_HF_TRANSFER=1`로 설정돼 있으나 해당 패키지 미설치 상태라 다운로드 실패. 해당 세션 한정으로 `HF_HUB_ENABLE_HF_TRANSFER=0`으로 override해서 해결.
- **GPU 자원 충돌**: RTX 4090 1장에서 mode3 서버(`--no-idle-shutdown`)가 이미 GPU 대부분을 점유한 채 계속 떠 있어 mode4 서버 로딩 시 OOM 발생. mode3 클라이언트 처리 완료(100/100, 연결 없음) 확인 후 서버 프로세스 종료로 해결.
- **`[SEG]` 로그 커버리지 구멍**: `_process_slot_updates`가 슬롯당 `<SEG>` 최초 등장만 로그로 남기는 가드(`if key not in self._slot_seg_detected`)가 있어, 한 generate 호출 안에서 세그먼트가 2개 이상 연속 커밋될 때 두번째 이후 SEG 등장 시점은 로그에 남지 않음. 이번 케이스는 타이밍 결론에 영향 없었으나 향후 유사 분석 시 로깅 보강이 필요할 수 있음.
- **`Doo12/Qwen3-ASR-1.7B-en-plus-merged`는 이 파이프라인(SEG-commit 스트리밍)에 부적합**: 동일 파일 테스트에서 WER 180%, trailing silence 구간에서 실제 음성과 무관한 문장("I'm sorry, but I can't help you.")을 반복 생성. 다른 목적(예: instruction/chat 튜닝)의 체크포인트일 가능성. HF 인증은 `.env`의 `HF_TOKEN`(계정 Doo12)으로 해결, `hf_transfer` 미설치로 인한 다운로드 실패는 `HF_HUB_ENABLE_HF_TRANSFER=0` override로 해결.
- **SEG 파인튜닝 모델이 청크 경계에서 SEG를 주저하는 이유(학습 데이터 관점)**: `evaluation/KsponSpeech/utils/generate_split_data.py`의 `trim_and_save_wav()`/`build_partial()` 확인 결과, 학습 데이터는 오디오가 forced-aligner 기준 "진짜 끝나는 지점"에서만 SEG를 라벨링하고 문장 중간에서 자른 데이터엔 SEG를 안 붙임. 즉 모델은 "청크 버퍼가 끝남 ≠ SEG"로 학습됨 → 실시간 스트리밍에서 청크 경계가 발화의 진짜 끝과 안 맞으면(이번 파일은 실제 끝 2.835s가 청크 경계 2.0s/4.035s 사이에 걸침) 다음 청크(무음 포함)까지 봐야 확신을 갖고 SEG를 찍음. 정황 증거: 개발자가 이미 `Qwen3-ASR/finetuning/utils/transcribe_finetuned.py`의 `--pad_silence` 옵션으로 이 현상을 실측(무음 유무에 따라 SEG 출력 10개 중 4개가 바뀜, 커밋 394fdf3) — "silence" 학습 기법 도입(5/17, 커밋 2f882cf)보다 3주 앞선 관찰. 단, "SEG 찍으려면 무음 몇 초 필요"라는 명시적 규칙/임계값은 코드에 없음. `-silence-`라는 이름 자체는 SEG 지연과 무관한 별개 증강(`generate_silence_dataset()`, 순수 무음/잡음 클립을 `text=""`로 학습 — "무음만 들리면 아무 말도 하지 마라")이었음.
- **baseline dot-commit의 hallucination은 확인되나, 흔한 현상은 아님(정정)**: `1688-142285-0002` 단일 케이스에서는 앞 2.0초만 잘라 보냈는데도 뒤쪽(0.835초, "so silly" 구간)까지 포함한 완전한 문장이 그대로 출력돼 hallucination이 확정됐음. 하지만 동일 실험을 다른 14개 파일로 확장한 결과, 유의미하게(≥1초) 잘린 9개 전부 **정확히 잘린 만큼만** 출력하고 hallucination이 재현되지 않음. 즉 이 케이스는 일반적 패턴이 아니라 예외적 사례로 보이며, 전체 유의미 절삭 테스트 10개 중 1개(10%)에서만 관찰됨 — dot-commit의 빠른 FTL이 "체계적인 hallucination" 때문이라는 결론은 과도한 일반화였고, "가끔 발생 가능한 리스크" 정도로 톤 다운 필요.
- **dot-commit은 의미 경계 감지기가 아니라 "청크 끝나면 마침표로 마무리짓는" 기계적 습성에 가까움**: 절삭 실험 출력 중 `"No one came forwards to help the."`, `"But for a minute or two, she."`, `"He never."` 등은 문법적으로 명백히 미완성(목적어/동사 없이 끊김)인데도 마침표가 찍혀 dot 정규식이 발동함. 즉 모델이 "이 지점에서 절/문장이 진짜 끝났다"를 판단하는 게 아니라, 청크 버퍼가 끝나면 그 시점까지 생성한 내용에 그냥 마침표를 붙여 마무리짓는 경향이 있는 것으로 보임. 이게 `1688-142285-0002`의 hallucination(내용을 이어붙여 완성)과 이번 사례(미완성 채로 마침표만 붙임)를 하나로 묶는 공통 원인 — "출력을 그럴듯하게 완결된 것처럼 보이게 만드는 습성"이며, dot_FTL이 파일 내용과 무관하게 항상 ~2.2초로 고정되는 이유이기도 함. 이는 dot-commit의 빠른 FTL이 신뢰할 만한 조기 종결이 아니라 구조적 아티팩트일 가능성을 시사함.
- **위 현상은 파이프라인 버그가 아니라 모델 자체의 디코딩 습성으로 확인됨**: `DOT_COMMIT_BOUNDARY_RE`([sentence_boundary.py:17-24](Qwen3-ASR/qwen_asr/inference/sentence_boundary.py#L17-L24))는 `.`/`!`/`?`/`<SEG>`만 매칭하고 느슨한 매칭 없음. `on_dot`은 vLLM 실제 디코딩 출력(`txt_p`, [qwen3_asr.py:854-877](Qwen3-ASR/qwen_asr/inference/qwen3_asr.py#L854-L877))에만 정규식을 돌리며, 서버가 텍스트에 인위적으로 "."을 추가하는 코드(`+= "."`, `.append(".")` 등)는 전수 grep 결과 없음. `max_new_tokens` 도달 시 강제로 dot 라벨링하는 fallback도 없음. 결정적으로 `[TRANSCRIBE-DECODING]` 로그(정규식/커밋 로직 타기 **전** 단계)에도 이미 마침표가 있어, 모델이 생성 단계에서부터 마침표를 만들어낸 것으로 확인됨. 원인 추정(미검증): 매 청크마다 누적 오디오 전체를 처음부터 다시 디코딩하는 구조라, 모델이 "지금까지 들은 게 전부"라고 착각해 그 시점에서 발화가 끝난 것처럼 마무리짓는 습성일 가능성.

## ⏭ 해결되지 않은 작업

- mode3 dot commit `fsl_sec` null의 근본 원인(왜 char-ratio 역추적이 실패하는지, `total_tokens` 혹은 `dot_char_ratio` 계산이 어디서 어긋나는지) 코드 추가 조사 필요.
- `[SEG]` 로그가 세그먼트별로 1회만 남도록 되어 있는 부분 로깅 개선 여부 결정 필요.
- dot-commit hallucination 발생률을 더 큰 샘플(현재 15개 중 1개, 10%대 관측)로 정밀 추정 필요. 어떤 조건(짧고 문법적으로 예측 가능한 문장, 특정 화제 등)에서 더 잘 발생하는지 패턴 파악 필요. `1688-142285-0002`("You don't mean that you thought me so silly")가 왜 유독 hallucination이 났는지(흔한 관용구/짧은 문장이라 LM prior가 강했을 가능성)도 별도 확인 여지 있음.
- "dot-commit = 청크 끝나면 마침표로 마무리짓는 기계적 습성" 가설을 더 큰 샘플로 정량 검증 필요 (전체 100개 중 몇 %가 문법적으로 미완성 상태에서 마침표 찍히는지). 확인되면 dot-commit 방식 자체(마침표 정규식 기반 커밋)의 신뢰성에 대한 재평가 필요 — 지금은 mode3(baseline) 전체 커밋이 이 습성에 의존하고 있음.
- "모델이 매 청크마다 실제로 마침표를 생성한다"는 결론은 이번 세션에서 확인한 15~20초 파일들 자체를 `[TRANSCRIBE-DECODING]` 로그 켜서 1:1 재검증한 게 아니라, 동일 코드 경로를 쓴 유사 truncation 실험 로그로 확인한 것 — 완전 확정하려면 해당 파일들로 직접 재현 필요. 또한 "왜" 모델이 이런 습성을 갖는지(재디코딩 구조 때문이라는 추정)는 코드로 증명된 게 아니라 정황적 해석임.
- SEG 학습 데이터에 "청크 경계 = 임의의 버퍼 컷"이라는 시나리오를 명시적으로 넣어(예: 문장 중간이 아니라 발화 끝 근처를 청크 크기 배수로 랜덤 컷) 재학습하면 청크 경계 SEG 지연이 줄어드는지 실험해볼 여지 있음 (v4 학습 런의 정확한 CLI 하이퍼파라미터는 체크포인트 디렉터리가 `.gitignore`돼 있어 리포에 없음).
