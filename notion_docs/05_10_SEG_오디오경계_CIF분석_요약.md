# 05_10_SEG_오디오경계_CIF분석_요약

## 📅 날짜
2026-05-10

## 🔧 작업 내용

### SEG 토큰 처리 구조 분석

`Qwen3-ASR/examples/streaming_websocket_server.py` 및 `qwen_asr/inference/qwen3_asr.py` 코드를 기반으로 SEG 커밋의 실제 동작을 분석했다.

**핵심 발견:**

- SEG 감지 시 `_process_slot_updates()`가 호출되어 텍스트 커서(`committed_seg_count`, `committed_len`)만 이동한다. 오디오 버퍼는 전혀 잘리지 않는다.
- `streaming_transcribe()` 내부에서 `state.audio_accum`은 매 청크마다 concat으로만 누적되며, SEG 이벤트에 반응하지 않는다.
- VAD 묵음 감지 시에만 슬롯 스위치 + `_reset_stream_slot()`으로 `audio_accum`이 초기화된다.
- 모델은 매 청크 호출마다 `audio_accum` 전체를 입력으로 받기 때문에 SEG 이후에도 이전 오디오가 계속 재처리된다.

**SEG의 실제 역할:**

SEG는 오디오 경계가 아니라 UX 최적화 신호다. VAD를 기다리지 않고 확정된 텍스트 조각을 클라이언트에 먼저 push함으로써 체감 지연을 줄인다. 오디오 정확도나 ASR 품질과는 무관하다.

**오디오를 자르지 않는 이유:**

SEG는 텍스트 도메인 신호다. 모델이 `<SEG>` 토큰을 생성할 때 해당 위치가 오디오의 몇 번째 샘플인지 알 수 없다(word-level timestamp 미지원). VAD는 오디오 샘플 인덱스(`local_cut`, `target_sample`)를 직접 추적하기 때문에 자를 수 있다.

---

### CIF를 활용한 오디오 타임스탬프 역산 가능성 검토

`research/cif/docs/cif_latency_reduction.md` 문서를 함께 검토했다.

**질문:** 인코더 프레임 단위 출력을 이용해 SEG가 오디오 어느 지점인지 역산할 수 있는가?

**결론:** 역산 자체는 불가능하지만, CIF는 그 방향을 뒤집어 정방향으로 해결한다.

- 역방향 추적은 순환 문제: SEG 위치를 알려면 디코더가 이미 실행되어야 하므로, 그 결과로 오디오를 자르는 것은 이미 늦다.
- CIF(Continuous Integrate-and-Fire)는 인코더 출력 위에 weight predictor를 얹어 디코딩 전에 경계를 결정한다.
- 인코더의 다운샘플링 비율이 고정(8x)이므로, CIF가 frame N에서 fire하면 `N * 8 * 10ms = timestamp(초)`로 즉시 오디오 타임스탬프를 계산할 수 있다.
- 이 타임스탬프로 `audio_accum`을 슬라이싱하면 VAD 없이도 SEG 단위로 오디오를 잘라 다음 디코딩에 넘길 수 있다.

**현재 한계:**

파인튜닝 시 인코더는 완전 frozen이었다(디코더 LoRA + SEG 임베딩만 학습). SEG 경계 정보가 인코더 표현에 암묵적으로만 존재하므로, CIF weight predictor가 이를 추출할 수 있는지는 실험적으로 검증해야 한다. 불충분할 경우 인코더 LoRA를 2단계로 추가 학습하는 전략이 문서에 제안되어 있다.

---

### git-sync

- `git pull`: origin/main fast-forward (AMI baseline run_05, run_06 평가 결과 반영)
- 커밋 및 푸시 (`cd87b44`):
  - `.claude/commands/eval-run.md`: 평가 실험 자동화 슬래시 커맨드 신규 추가
  - `evaluation/LibriSpeech/servers/results/fsl/test/*.png`: 결과 플롯 3건 업데이트

## ⏭ 해결되지 않은 작업

- CIF weight predictor 학습 실험 미착수. 인코더 표현만으로 SEG 경계 추출 가능한지 검증 필요.
- `chunk_size_sec` 축소 시 레이턴시-품질 트레이드오프 실험 미진행.
- SEG 기반 `audio_accum` 슬라이싱 구현 미착수 (CIF 학습 이후 단계).
