# dot_commit_probe — 커밋 정책 검증 하네스

`dot_commit_confirm`(확정 게이트) 설계·검증에 쓴 실험 도구. WebSocket 서버를 거치지 않고
`Qwen3ASRModel`을 직접 구동해 **커밋 정책만 바꿔가며** 같은 오디오를 비교한다.

전체 배경과 결과 해석은 [notion_docs/08_01_dot_commit_확정게이트_설계검증_요약.md](../../../notion_docs/08_01_dot_commit_확정게이트_설계검증_요약.md) 참조.

## 왜 별도 하네스가 필요했나

프로덕션 서버는 커밋이 발생하면 슬롯을 리셋해 `audio_accum`을 버린다. 그래서
"마침표가 다음 청크에서 살아남는가"라는 판별 신호 자체가 로그에 남지 않는다.
이 하네스는 리셋을 끄거나(수집 모드) 리셋을 서버와 동일하게 재현한 채(라이브 모드)
정책만 교체할 수 있게 해서 그 신호를 직접 측정한다.

## 구성

| 파일 | 역할 |
|---|---|
| `probe_collect.py` | 커밋·리셋 **없이** 누적 디코딩만 하며 청크별 가설 전체를 JSONL로 기록 |
| `gate.py` | 확정 게이트 순수 로직 (문맥 확정 / 합의 확정 / finish). 오프라인 리플레이용 |
| `test_gate.py` | `gate.py` 단위테스트 6개. 실측 시퀀스 기반 |
| `probe_live.py` | 슬롯 리셋·오디오 캐리오버·중복제거까지 서버와 동일하게 재현한 라이브 하네스 |
| `analyze.py` | `probe_collect` 결과 리플레이 — naive vs gate 비교, 마침표 생존율 |
| `score_live.py` | `probe_live` 결과 채점 — WER / 커밋 수 / 발화별 지연 |
| `results/` | 리포트에 인용된 수치의 원자료 JSONL |

정책 이름: `naive`(감지 즉시 커밋 = 게이트 이전 동작), `gate`(규칙 1+2+3), `gate1`(규칙 1과 finish만).

## 실행

프로젝트 루트에서 실행한다. GPU와 vLLM 필요(`stity` 환경).

```bash
# 1) 청크별 가설 수집 (커밋/리셋 없음) — 마침표 생존율 측정용
python evaluation/LibriSpeech/dot_commit_probe/probe_collect.py \
  --limit 60 --spread --enforce-eager \
  --out evaluation/LibriSpeech/dot_commit_probe/results/hyp_spread60.jsonl

# 2) 오프라인 리플레이 (naive vs gate)
python evaluation/LibriSpeech/dot_commit_probe/analyze.py \
  evaluation/LibriSpeech/dot_commit_probe/results/hyp_spread60.jsonl --show 3

# 3) 라이브 하네스 — 리셋 포함, 정책별 실제 커밋 시퀀스 생성
python evaluation/LibriSpeech/dot_commit_probe/probe_live.py \
  --limit 48 --spread --concat 4 --gap-sec 1.0 --enforce-eager \
  --policies naive gate \
  --out evaluation/LibriSpeech/dot_commit_probe/results/live_concat4.jsonl

# 4) 채점
python evaluation/LibriSpeech/dot_commit_probe/score_live.py \
  evaluation/LibriSpeech/dot_commit_probe/results/live_concat4.jsonl --gap 1.0

# 5) 게이트 단위테스트 (GPU 불필요)
python evaluation/LibriSpeech/dot_commit_probe/test_gate.py
```

주요 인자: `--spread`(화자 라운드로빈 샘플링), `--concat N --gap-sec S`(발화 N개를 무음 S초로
이어붙여 연속 발화 스트림 구성), `--chunk-sec`(청크 크기), `--policies`.

## 저장된 결과

| 파일 | 조건 |
|---|---|
| `hyp_spread60.jsonl` | 단일 발화 60개, 화자 분산, 커밋/리셋 없음 |
| `live_single.jsonl` | 단일 발화 48개, naive/gate |
| `live_concat4.jsonl` | 4발화 연속 12스트림, 무음 1.0s, chunk 2.0s, naive/gate |
| `live_chunk1.jsonl` | 위와 동일하나 chunk 1.0s, gate만 |
| `live_gap2.jsonl` | 4발화 연속, 무음 2.0s, chunk 2.0s, naive/gate |

## 주의

- 하네스는 커밋·리셋·중복제거만 재현한다. GPT 교정, 페어링, FSL 로깅, 슬롯 스위칭은 미반영이라
  프로덕션 서버와 완전히 동일하지 않다. 최종 검증은 `--dot-commit-confirm`을 켠 실제 서버로 할 것.
- `probe_collect.py`는 리셋을 하지 않으므로 누적 오디오가 30초를 넘기면 모델 출력이 무너진다
  (뒷부분 누락). 긴 스트림 실험에는 `probe_live.py`를 쓸 것.
- 지연 지표는 편집거리 정렬로 참조 단어별 전달 시각을 구해 계산한다. 단순 단어 수 누적
  방식은 ASR 삭제 오류에서 크게 어긋난다.
