## 📅 날짜
2026-08-03

## 🔧 작업 내용

8/1 확정 게이트(`dot_commit_confirm`) 100파일 평가에서 남아 있던 **finish 커밋 9건 제거**. VAD 의존성 없이 텍스트 신호만으로 마지막 문장까지 dot으로 확정하는 것이 목표. 대상 9건만 뽑아 재평가(`--common-files`)로 검증.

**1. 원인 분석 (8/1 run01 서버 로그 실측)**

finish 9건은 두 부류로 갈렸다.

- **인용문 종료가 경계로 인식 안 됨 (5건: 0001, 0057, 0060, 0086, 0091)** — `DOT_COMMIT_BOUNDARY_RE`가 `\.(?:\s+|$)`라 마침표 바로 뒤가 공백이나 문자열 끝이어야 매치된다. 대화체는 마침표가 인용부호 안쪽에 찍혀(`... a shop boy."`) 매치 자체가 실패. 게이트 문제가 아니라 감지기 문제.
- **pending 왕복 (4건: 0000, 0067, 0070, 0074)** — `_on_dot` 콜백이 generate 루프 중간에 `_process_slot_updates`를 `chunk_end=False`로 호출하는데, 규칙 2는 `chunk_end`에서만 판정하므로 else로 떨어져 `pending_dot_text`를 중간 가설로 덮어썼다. 결과적으로 청크 종료 시점의 비교 대상이 "직전 청크의 프론티어"가 아니라 "같은 청크 중간값"이 되어 `sentence == pending`이 영원히 성립 못 함. 로그에 매 청크 두 문장 사이를 왕복하는 것이 그대로 찍혀 있었다.

**2. 서버 수정 (`Qwen3-ASR/examples/streaming_websocket_server.py`, `sentence_boundary.py`)**

- 종료 문장부호 뒤 닫는 따옴표/괄호를 흡수하는 `_CLOSERS` 추가. 소수점(`3.14`)·약어(`Mr.`) 제외는 유지.
- pending 등록을 `chunk_end`로 한정. 규칙 1(문맥 확정)은 루프 중간에서도 계속 커밋 가능해 지연 손해 없음.
- **규칙 2를 청크 경계집합 비교로 확장** — 후보 하나(`pending_dot_text`) 대신 직전 chunk_end 가설의 경계 문장 전체(`prev_boundary_sentences`)를 보관하고 집합 포함 여부로 확정. 한 청크에 경계가 여럿이면 각각 독립 판정되므로, 발화 마지막 문장이 앞 문장 커밋을 기다리다 오디오가 떨어지는 일이 없어진다.
- **슬롯 리셋 시 경계집합 carry** — 리셋은 슬롯 dict를 통째로 갈아끼워 게이트 상태를 날리는데, carry_audio는 같은 단어로 재디코딩되므로 직전 청크 증거는 유효하다. accum 스탬프는 -1로 재기준화.
- **규칙 3 정체 확정 신설** (`--dot-commit-stall-chunks`, 기본 1) — 오디오는 누적되는데 미커밋 가설 토큰 수가 N청크 연속 그대로면 발화 종료로 보고 커밋. 오디오 에너지를 안 보므로 VAD 의존성 없음. 단, 아래 조건으로 제한: 직전 청크 이후 커밋이 없었을 것(`committed_display` 길이 동일), frontier 경계일 것(`after`가 비어 있음).
- `finish` 경로는 안전망으로 존치. 마침표가 아예 안 찍힌 가설은 원리상 dot으로 커밋할 수 없다.
- `dot_commit_probe/gate.py` regex도 서버와 parity 맞춤. 단위테스트 6개 통과.

**3. 평가 하네스 수정 (`evaluation/LibriSpeech/servers/test_qwen3_librispeech.py`)**

`_recv`의 `wait_for(timeout=3.0 if send_done else 15.0)`는 timeout이 wait 진입 시점에 확정되고 대기 중 재평가되지 않는다. 이 서버는 ready 이후 첫 final까지 아무것도 안 보내므로 `_recv`가 15초 wait 하나에 앉아 있고, 수신 창이 실제 유휴 시간이 아니라 **오디오 길이의 계단 함수**가 됐다(`dur+trailing < 15s` → 창 15초 / `>= 15s` → 창 30초). 1초 폴링 + finish 후 유휴 8초 기준으로 교체.

## 📊 결과 / 수치

**finish 9건 재평가 (run04, mode3_confirm 동일 조건 / baseline 1.7B / no-vad / chunk 2.0s / trailing 5500ms / 하네스 수정 적용)**

| | before (8/1 run01) | after (run04) |
|---|---|---|
| finish | 9 | **0** |
| dot | 10 | 19 |
| WER | 4.76% | **3.70%** |
| 커밋당 토큰 | 9.32 | 9.26 |

| 파일 | before | after | WER |
|---|---|---|---|
| 0000 | dot, finish | dot×2 | 6.06% → **0.00%** |
| 0001 | finish | dot | 8.82% → 8.82% |
| 0057 | finish | dot | 0.00% → 0.00% |
| 0060 | finish | dot | 0.00% → 0.00% |
| 0067 | dot×2, finish | dot×3 | 0.00% → 0.00% |
| 0070 | dot×2, finish | dot×3 | 9.09% → 9.09% |
| 0074 | dot×3, finish | dot×4 | 7.41% → 7.41% |
| 0086 | finish | dot | 0.00% → 0.00% |
| 0091 | dot×2, finish | dot×3 | 7.14% → 7.14% |

**8건 동일 / 1건 개선**(0000: `And on.` → `Anon.`이 정답 `ANON`에 맞음) **/ 0건 악화.**

**확정 규칙 분포: context 6 / stable 13 / stall 0.** 규칙 3은 한 번도 발동하지 않았다 — 규칙 2 경계집합 확장 + 리셋 carry만으로 9건 전부 해결됐다. 규칙 3은 현재 순수 안전망이고 이 샘플에서 실효를 증명하지 못했다.

**FTL 9.98s → 13.04s는 지연 악화로 읽으면 안 된다.** LibriSpeech는 파일당 문장이 사실상 하나라 첫 커밋이 곧 마지막 커밋이고, 확정을 한 청크 기다리는 만큼 그대로 잡힌다. 게다가 이번 런은 GPU를 다른 세션과 공유해 청크당 디코딩이 0.1초대 → 2~3.6초로 늘었다. 지연 비교는 동일 부하에서 재측정 필요.

## 🐛 발견된 문제 및 해결

- **pending 등록을 chunk_end로 한정하자 교착 발생** — `_on_dot`이 `_process_slot_updates` 진입부에서 `last_text`를 먼저 갱신해버려 청크 종료 호출이 `current_text == last_text`로 조기 리턴됐다. 기존엔 on_dot이 pending도 같이 등록해줘서 `_recheck_pending`이 참이 되어 통과했는데, 등록을 chunk_end 전용으로 바꾸니 **조기 리턴 ↔ pending 미등록** 교착. 첫 검증 런에서 `DOT-PENDING` 0건, 파일 전체가 finish 1건으로 나와 수정 전보다 나빴다. → `dot_commit_confirm`이면 chunk_end 호출은 텍스트 동일 여부와 무관하게 항상 게이트 통과시키도록 변경. 규칙 3 카운터도 가설이 안 자란 청크에서 올라가야 하므로 어차피 이게 맞다.
- **규칙 3 정체 카운터 오발동으로 WER 7.69%까지 악화(run02)** — 카운터가 uncommitted 슬라이스의 토큰 수를 청크 간 비교하는데, 중간에 커밋이 일어나면 그 슬라이스가 가리키는 구간이 바뀐다. `'My dear," said Miss.'`와 `'"Pray don't."'`가 **둘 다 6토큰**이라 정체로 오판, 아직 안 끝난 `"Pray don't."`를 커밋해 0091이 7.14% → 50%로 터졌다(뒤 `go off on that idea`가 분리되고 순서까지 뒤집힘). → `committed_display` 길이 가드 + frontier 한정 추가.
- **평가 파일 1건(0057)이 조용히 유실** — 서버는 정상 커밋(`rule=stable`, `reason=dot`, finish 0건)했는데 `metric.json`에 안 들어가고 `Empty transcript` 경고 한 줄만 남았다. 위 하네스 계단 함수가 원인. 9파일 전부에 대해 `(dur+5.5s, final 도착 시각, 수신 창)`을 계산해 **9/9 예측 일치** 확인: 0057은 send 14.38s(창 15초)에 final 16.18s로 1.18초 초과, 0060은 send 15.28s(창 30초)에 final 16.59s로 통과 — 오디오 길이 0.9초 차이가 생사를 갈랐다. 8/1 원본 런에선 0057 final이 14.39s로 창 안쪽 **0.61초**, 원래부터 마진이 얇았고 GPU 경합이 방아쇠였다. → 하네스 수정 후 run04에서 9/9 정상 수집 확인.
- **CRLF 파일에 LF 줄 삽입** — 메인 서버 파일이 CRLF인데 편집 도구가 LF로 넣어 혼재. 매 편집 후 바이트 단위로 CRLF 재정규화해 diff를 실제 변경분만 유지.
- **vLLM EngineCore 좀비** — 서버 재시작 시 부모만 죽고 `VLLM::EngineCore` 자식이 살아남아 각각 30GB씩 점유. free가 26.4GiB로 떨어져 `gpu_memory_utilization 0.25` 요청이 거부됐다. 재시작 시 EngineCore까지 명시적으로 정리할 것.

## ⏭ 해결되지 않은 작업

- **규칙 3을 켤지 미결정.** 이 샘플에선 발동 0회라 실효 미증명이고, run02에서 오발동 전력이 있다. `--dot-commit-stall-chunks 0`으로 꺼도 run03 결과는 유지된다. 켠다면 오발동 조건을 더 좁힐지 판단 필요.
- **100파일 전체 재측정 미실시.** 이번 검증은 finish가 났던 9건만 대상이라, 나머지 91건에 회귀가 없는지는 확인 안 됐다. 특히 인용문 regex 확장은 모든 파일의 경계 판정에 영향을 준다.
- **지연 지표 재측정 필요.** GPU 경합 상태라 FTL/FSL 절대값이 8/1 런과 비교 불가.
- **다른 데이터셋 클라이언트의 동일 결함 여부 미확인.** `recv_timeout` 패턴은 LibriSpeech 클라이언트에만 있으나, 각자 다른 수신 루프를 쓰므로 같은 종류의 조용한 유실이 있는지는 따로 봐야 한다.
- 한국어·자유발화 미검증(8/1 문서에서 이어짐). 지금까지 전부 영어 낭독체 LibriSpeech.
