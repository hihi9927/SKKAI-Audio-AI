## 📅 날짜
2026-07-30

## 🔧 작업 내용

**1. mode3(no-vad + dot-commit) "마지막 문장 유실" 근본원인 조사**
- mode3 설정 그대로(baseline `Qwen/Qwen3-ASR-1.7B`, `--no-vad --enable-dot-commit`) 서버 띄우고 LibriSpeech 샘플 4개 직접 실행해 검증 → 4개 중 3개가 완전히 빈 transcript(`Empty transcript`)로 나옴. "마지막 문장만 유실"이 아니라 dot 트리거가 한 번도 안 걸린 발화는 통째로 유실되는 게 실제 증상이었음.
- 원인 두 가지를 분리해서 확인:
  1. **클라이언트 버그** (`test_qwen3_librispeech.py`): `finish` 메시지를 `process_single_file()`이 리턴한 *다음*(`process_batch` 쪽)에 보내고 있었음. 서버는 `finish` 받아야 마지막 uncommitted 텍스트를 flush하는데, 그 시점엔 클라이언트의 recv 루프가 이미 끝나있어 finish 트리거 커밋을 못 받음. 서버 로그로 서버는 정상 커밋(`reason=finish`)했는데 클라이언트만 놓치는 걸 직접 확인.
  2. **서버 dot-commit 정규식의 구조적 한계**: `\.\s+(?=\S)` 룩어헤드가 마침표 뒤에 실제 다음 단어가 와야만 매치되도록 되어 있어서, 발화의 **마지막 문장**은 뒤에 텍스트가 없으니 절대 dot으로 못 잡히고 항상 `finish` 커밋에만 의존하는 구조였음(설계 의도 — VAD/finish가 대신 잡도록).
- 부수 질문: 디코딩 출력 끝에 항상 보이던 "..."이 모델이 실제로 뱉는 토큰인지 확인 요청받음 → 라이브 로그(`TRANSCRIBE-DECODING`, `_raw_decoded`, `FINAL text=`) 전수 확인 결과 리터럴 `...` 단 한 번도 안 나옴. 모델 출력이 아니라 어딘가 디스플레이 레이어의 아티팩트로 결론(정확한 출처는 못 찾음).

**2. 클라이언트 finish-timing 버그 수정 + 재검증**
- `_send()` 내부(오디오+trailing silence 전송 완료 직후)에서 `finish` 전송하도록 이동, `process_batch`의 중복 전송 제거.
- `_recv()` 타임아웃 이원화: 스트리밍 중엔 기존과 동일 15초, `finish` 전송 후엔 3초(GPU 디코딩 없는 즉시 flush라 짧게 잡아도 안전, 매 파일마다 불필요한 지연 최소화).
- 같은 4개 + 2개 추가(총 6개) 샘플로 재검증 → 6/6 정상, `Empty transcript` 0건. 이전에 유실됐던 문장("Anon." → "And on.")이 `<finish>` 태그로 정상 도착하는 것 직접 확인.

**3. dot-commit을 `<SEG>`와 동급으로 취급하도록 구조 변경 (브랜치: `fix/dot-commit-seg-parity`)**
- TDD로 순수 함수 `count_dot_commit_boundaries()` 작성 (`Qwen3-ASR/qwen_asr/inference/sentence_boundary.py`) — 룩어헤드 없이 문자열 **끝** 마침표도 경계로 인정하되, 소수점(`3.14`)과 알려진 약어(Mr./Mrs./Dr./St./Jr./Sr./vs./No.)는 기존과 동일하게 계속 제외. 단위테스트 12개 작성(`Qwen3-ASR/tests/test_dot_commit_boundary.py`), RED 확인 후 구현 → GREEN.
- `qwen3_asr.py`의 `streaming_transcribe()`에 `on_dot` 콜백 추가 — 기존 `on_seg`와 완전히 동일한 방식으로 `model.generate()` 루프 안에서 마침표가 디코딩되는 **순간** 즉시 콜백 발화(원 질문이었던 "온점 디코딩 순간 포착" 요구사항 직접 구현).
- `streaming_websocket_server.py`: `_asr_streaming_transcribe`에 `_on_dot` 콜백을 `_on_seg`와 나란히 연결(`enable_dot_commit=True`일 때만 활성). `_process_slot_updates`의 dot 정규식을 새 `DOT_COMMIT_BOUNDARY_RE`로 교체.
- `streaming_websocket_server_dualbase.py`는 이번 수정 범위에서 제외(동일 로직 미적용 — mode3 eval 경로가 아니라서 스코프 밖으로 판단).

**4. mode3 100개 샘플 재검증 + mode2/mode4와 WER·FTL 비교**
- mode2/mode4와 동일 조건(`--limit 100`) 맞춰 재실행. 첫 시도에서 `--limit` 빠뜨려 test-other 전체(2939개)를 도는 실수 발견 → 즉시 중단하고 `--limit 100 --fresh-start`로 재실행.
- 100/100 완료, `Empty transcript` 0건, hypothesis가 마침표 없이 끝난 케이스 0건, commit_stats: `dot` 343건 / `finish` 0건 / `seg` 0건(baseline 모델이라 SEG는 원래 아예 안 나옴).

## 📊 결과 / 수치

| 항목 | mode2 (always-commit, baseline) | mode3 (dot-commit, baseline, 이번 수정 후) | mode4 (seg-commit, finetune) |
|---|---|---|---|
| WER | 8.44% | 8.38% | 5.40% |
| FTL(평균) | 2.068s | 2.226s | 4.528s |
| commit 수 (100파일) | 333 (전부 seg) | 343 (전부 dot) | 186 (전부 seg) |
| Empty transcript | - | 0/100 (수정 전 샘플 테스트에선 3/4) | - |

- WER: mode3(8.38%)가 mode2(8.44%)와 거의 동일 → dot-commit이 WER을 깎아먹는 게 아니라 baseline 모델 자체의 한계치. mode4가 낮은 건 파인튜닝 모델이라서지 커밋 방식 차이가 아님.
- FTL: 가설(dot이 SEG랑 비슷하거나 더 느릴 것)과 반대로 **dot이 SEG보다 약 2배 빠름**. 원인은 판정 속도 차이가 아니라 **커밋 단위 크기 차이** — 같은 100개 파일에서 dot은 343번(문장마다), SEG는 186번(모델이 학습한 "자연스러운 pause" 단위, 문장 여러 개를 묶어서 한 번)만 커밋. 첫 단위가 짧은 쪽이 구조적으로 항상 먼저 도착.

## 🐛 발견된 문제 및 해결

- **클라이언트 finish-timing 버그**: `finish` 전송이 recv 루프 종료 후에 이뤄져 finish-트리거 커밋을 못 받는 문제. → `_send()` 내부로 이동해 해결, 재검증 완료.
- **dot-commit 룩어헤드 구조적 한계**: 마지막 문장이 절대 dot으로 안 잡히던 문제. → `on_dot` 훅 + 룩어헤드 제거 regex로 해결, SEG와 동급 취급되도록 만듦. 100개 샘플 검증에서 `finish` 커밋 0건으로 확인.
- **100개 샘플 실험 중 `--limit` 누락**: 전체 데이터셋(2939개) 도는 중이었던 것 조기 발견 → 중단 후 재실행.
- **mode3 `avg_fsl_sec`가 null**: FSL 이벤트 서버(`streaming_websocket_server_fsl.py`)의 조기 타이밍 기록(`_slot_seg_detected`)이 `<SEG>` 문자열 존재 여부로만 게이팅돼 있어서, dot 트리거 경로는 이 기록을 안 탐. 근본 원인 미조사 상태 — 아래 후속 작업 참고.

## ⏭ 해결되지 않은 작업

- `avg_fsl_sec` null 근본 원인 조사 (FSL 이벤트 서버의 SEG 전용 게이팅을 dot에도 열어줄지 결정 필요).
- `fix/dot-commit-seg-parity` 브랜치 — 아직 커밋 안 함. 커밋/PR 진행 여부 결정 필요.
- SEG 가능한 파인튜닝 모델(mode4용 모델)에 dot-commit도 같이 켜서, 동일 모델 안에서 dot vs seg 진짜 헤드투헤드 latency 비교하는 후속 실험 (사용자가 관심 보였으나 아직 미실행).
- `streaming_websocket_server_dualbase.py`에 동일 fix 반영 여부 결정 필요.
