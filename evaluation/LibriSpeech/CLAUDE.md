# evaluation/LibriSpeech/

영어 읽기 음성 ASR 벤치마크. 주요 메트릭은 **WER** 과 **FSL**(세그먼트 마지막 오디오 바이트부터
클라이언트가 `final` 을 받기까지). 이 디렉토리가 **평가 서버를 쥐고 있어서** 다른 데이터셋도 전부
여기 서버에 붙는다.

## 디렉토리

| 경로 | 역할 |
|---|---|
| `servers/` | `streaming_websocket_server_fsl.py`(평가 서버의 base — 띄우는 것은 `evaluation/streaming_websocket_server_ast.py` 다) + 클라이언트 `test_qwen3_librispeech.py` |
| `LibriSpeech/{test-other,test-clean,dev-clean,dev-other}/` | 오디오. git 미포함(`.gitignore`), OpenSLR 에서 받는다 |
| `paper_result/ASR/` | 논문용 mode2/3/4 측정. **git 추적 대상** |
| `results/` | 그 밖의 벤치마크 출력. git 추적하지 않는다 |
| `utils/` | 후처리·플롯 (`export_fsl_ftl_plots.py`, `compute_comet.py`, `export_results_xlsx.py`) |

## 빠른 시작

```bash
# 터미널 1
python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown

# 터미널 2
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 다화자 벤치마크

- `run_concurrent_chapters.py` — 챕터를 N개 클라이언트로 나눠 동시 실행. **현재 주 경로**이고
  `--num-clients` / `--model` / `--scope` / `--tag` 를 받는다.
- `run_concurrent_benchmark.py`, `run_multi_speaker_full.py` — 동시 1~10명 스윕. 인자가 없어 파일
  상단 상수를 고쳐야 한다. `PROJECT_ROOT` 는 파일 위치에서 유도되고, 인터프리터는 기본이
  현재 `sys.executable` 이며 `EVAL_PYTHON` / `EVAL_BIN_DIR` 로 덮어쓴다.

## 논문 측정 — mode2 / 3 / 4

커밋 정책 세 가지를 같은 자로 잰다. **mode2** = 매 청크 커밋(always-commit),
**mode3** = 문장부호 커밋 + 확정 게이트, **mode4** = 영어 파인튜닝 가중치 + SEG 단독 커밋.
전부 16클라이언트 동시 실행이다.

| 모드 | 범위 | 런 | 발화수 | WER | FSL(s) | FSL 정규화 | FTL(s) | 클라이언트 | makespan(s) |
|---|---|---|---|---|---|---|---|---|---|
| mode2 | full | concurrent16_run01 | 2939 | 10.98% | 0.06 | 0.06 | 2.07 | 16 | 2310 |
| mode2 | full | devclean_c16_run01 | 2703 | 7.47% | 0.07 | 0.07 | 2.07 | 16 | 2221 |
| mode2 | full | devother_c16_run01 | 2864 | 10.58% | 0.06 | 0.06 | 2.07 | 16 | 2206 |
| mode2 | full | testclean_c16_run01 | 2620 | 7.39% | 0.06 | 0.06 | 2.07 | 16 | 2186 |
| mode2 | full | testother_c16_run01 | 2939 | 10.97% | 0.06 | 0.06 | 2.07 | 16 | 2307 |
| mode3 | full | concurrent16_dedupfix_run01 | 2939 | 4.04% | 0.28 | 0.28 | 7.92 | 16 | 2307 |
| mode3 | full | concurrent16_run01 | 2939 | 5.74% | 0.29 | 0.29 | 7.92 | 16 | 2307 |
| mode3 | full | devclean_c16_run01 | 2703 | 2.22% | 0.29 | 0.29 | 8.55 | 16 | 2216 |
| mode3 | full | devother_c16_run01 | 2864 | 3.72% | 0.28 | 0.28 | 7.95 | 16 | 2200 |
| mode3 | full | testclean_c16_run01 | 2620 | 2.13% | 0.30 | 0.30 | 8.66 | 16 | 2184 |
| mode3 | full | testother_c16_run01 | 2939 | 3.91% | 0.29 | 0.29 | 7.91 | 16 | 2307 |
| mode4 | full | concurrent16_en_run01 | 2939 | 4.55% | 0.62 | 0.62 | 4.37 | 16 | 2307 |
| mode4 | full | concurrent16_run01 | 2939 | 13.47% | 0.27 | 0.27 | 4.76 | 16 | 2307 |
| mode4 | full | devclean_c16_run01 | 2703 | 2.19% | 0.71 | 0.71 | 4.63 | 16 | 2216 |
| mode4 | full | devother_c16_run01 | 2864 | 4.21% | 0.59 | 0.59 | 4.42 | 16 | 2200 |
| mode4 | full | testclean_c16_run01 | 2620 | 2.31% | 0.72 | 0.72 | 4.70 | 16 | 2184 |
| mode4 | full | testother_c16_run01 | 2939 | 4.59% | 0.62 | 0.62 | 4.38 | 16 | 2307 |

`mode4/full/concurrent16_run01` 의 13.47% 만 유독 높은데, 같은 조건을 영어 가중치로 다시 돌린
것이 `concurrent16_en_run01`(4.55%)이다. 두 줄을 나란히 읽지 말 것 — 가중치가 다르다.

원본은 `paper_result/ASR/mode*/[full|sample]/<run>/metric.json` 에 있고 `raw_results` 에 발화
단위 결과가 들어 있다. 스크립트는 `paper_result/ASR/scripts/` 다(`run_all_splits.sh` 가 진입점).

<details>
<summary>sample·파일럿 런 (게이트·중복제거·트레일링 실험)</summary>

| 모드 | 범위 | 런 | 발화수 | WER | FSL(s) | FSL 정규화 | FTL(s) | 클라이언트 | makespan(s) |
|---|---|---|---|---|---|---|---|---|---|
| mode2 | sample | concurrent16_pilot | 920 | 9.83% | 0.06 | 0.06 | 2.07 | 16 | 2572 |
| mode2 | sample | run01 | 3 | — | 0.68 | 0.68 | 3.24 | — | — |
| mode2 | sample | run02 | 100 | 8.44% | 0.06 | 0.06 | 2.07 | — | — |
| mode2 | sample | run03 | 100 | 8.44% | 0.06 | 0.06 | 2.07 | — | — |
| mode3 | sample | flushsplit_finish59 | 59 | 13.11% | 0.42 | 0.42 | 8.63 | — | — |
| mode3 | sample | fuzzydedup_100 | 100 | 22.80% | 0.37 | 0.37 | 8.09 | — | — |
| mode3 | sample | fuzzydedup_v2_100 | 100 | 7.97% | 0.38 | 0.38 | 8.05 | — | — |
| mode3 | sample | gate_concurrent4_v2 | — | — | — | — | — | 4 | 120 |
| mode3 | sample | nofinish_finish59 | 59 | 8.45% | 0.53 | 0.53 | 8.63 | — | — |
| mode3 | sample | norepdedup_finish59 | 59 | 8.52% | 0.52 | 0.52 | 8.63 | — | — |
| mode3 | sample | prefixstrip_finish59 | 59 | 9.78% | 0.44 | 0.44 | 8.63 | — | — |
| mode3 | sample | prerefactor_problem149 | 149 | 8.97% | 0.42 | 0.42 | 8.79 | — | — |
| mode3 | sample | refactor_problem149 | 149 | 8.97% | 0.42 | 0.42 | 8.79 | — | — |
| mode3 | sample | repcap_finish59 | 59 | 8.58% | 0.52 | 0.52 | 8.63 | — | — |
| mode3 | sample | resync_finish59 | 59 | 8.58% | 0.52 | 0.52 | 8.63 | — | — |
| mode3 | sample | rule4_finish59 | 59 | 13.11% | 0.44 | 0.44 | 8.63 | — | — |
| mode3 | sample | rule4all_finish59 | 59 | 13.11% | 0.44 | 0.44 | 8.63 | — | — |
| mode3 | sample | run01 | 100 | 8.38% | — | — | 2.23 | — | — |
| mode3 | sample | trailing5500_finish59 | 59 | 12.51% | 0.40 | 0.40 | 8.55 | — | — |
| mode3 | sample | trailing8000_finish59 | 59 | 11.98% | 0.41 | 0.41 | 8.84 | — | — |
| mode4 | sample | run01 | 100 | 5.40% | 0.20 | 0.20 | 4.53 | — | — |
| mode4 | sample | single_1688-142285-0002 | 1 | 0.00% | 0.07 | 0.07 | 4.24 | — | — |

</details>

## `results/` — 그 밖의 런 62개

`.gitignore` 에 넣어 추적하지 않는다. `metric.json` 이 발화 단위 결과를 담아 실행당 수 MB 이고
서버 로그는 그보다 크다. 지금까지 낸 수치는 아래가 기록이다.

<details>
<summary>표 펼치기</summary>

| 모델 | 범위 | 런 | 발화수 | WER | FTL(s) | FSL(s) | 청크 | 비고 |
|---|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | full | run_01 | 2939 | 3.90% | 7.78 | — | — | — |
| baseline(1.0.0) | full | run_02 | 2939 | 5.38% | 7.10 | — | — | — |
| baseline(1.0.0) | full | run_03 | 2939 | 4.97% | 7.19 | — | — | — |
| baseline(1.0.0) | full | run_04 | 290 | 7.08% | 6.67 | 0.64 | 2.0 | baseline(1.0.0) / full / chunk 2.0s / dot-commit / limit 548 |
| baseline(1.0.0) | full | run_05 | 548 | 7.47% | 6.63 | 0.60 | 2.0 | baseline(1.0.0) / full / chunk 2.0s / dot-commit / FSL fix / limit 548 |
| baseline(1.0.0) | full | run_06 | 548 | 11.92% | 6.52 | 0.54 | 2.0 | baseline(1.0.0) / full / chunk 2.0s / dot-commit / FSL fix |
| baseline(1.0.0) | full | run_07 | — | — | — | — | — | (지표 없음) baseline 548-sample benchmark |
| baseline(1.0.0) | full | run_08 | 548 | 7.70% | 7.23 | 1.20 | 2.0 | baseline 548-sample benchmark |
| baseline(1.0.0) | full | run_09 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_10 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_11 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_12 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_13 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_14 | 185 | 5.46% | 5.94 | 0.53 | 2.0 | 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_15 | 34 | 1.97% | 6.90 | 0.47 | 2.0 | 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | full | run_16 | 548 | 7.68% | 6.51 | 1.17 | 2.0 | 번역 버그 수정 후 baseline(1.0.0) 548-sample |
| baseline(1.0.0) | sample | run_01 | 50 | 4.53% | 5.99 | — | — | — |
| baseline(1.0.0) | sample | vad_dot_run01 | 10 | 6.78% | 2.40 | — | 2.0 | baseline(1.0.0) / sample / dot-commit ON + VAD ON / limit 10 / ASR only |
| baseline(1.0.0) | sample | vad_dot_run01_main | 1 | 3.03% | 16.23 | 0.32 | 2.0 | baseline(1.0.0) / sample / dot-commit ON + VAD ON / limit 10 / ASR only (main branch, no d |
| en-plus-merged | sample | single_1688-142285-0002 | 1 | 180.00% | 4.24 | 0.09 | 2.0 | en-plus-merged single-file rerun: 1688-142285-0002 timing re-measure |
| finetuned(1.0.1) | chunk_size_test | chunk0.25 | 548 | 50.97% | 2.23 | 0.85 | — | chunk_size=0.25s sweep |
| finetuned(1.0.1) | chunk_size_test | chunk0.5 | 548 | 22.05% | 3.48 | 0.77 | — | chunk_size=0.5s sweep |
| finetuned(1.0.1) | chunk_size_test | chunk1.0 | 548 | 13.39% | 4.13 | 0.80 | — | chunk_size=1.0s sweep |
| finetuned(1.0.1) | chunk_size_test | chunk1.5 | 548 | 13.16% | 4.45 | 0.93 | — | chunk_size=1.5s sweep |
| finetuned(1.0.1) | chunk_size_test | chunk2.0 | 548 | 10.68% | 4.67 | 0.83 | — | chunk size = 2.0 |
| finetuned(1.0.1) | full | run_01 | 2939 | 6.62% | 6.39 | — | — | — |
| finetuned(1.0.1) | full | run_02 | 2939 | 7.39% | 4.85 | — | — | — |
| finetuned(1.0.1) | full | run_03 | 548 | 16.08% | 4.78 | 0.92 | 2.0 | finetuned(1.0.1) / full / chunk 2.0s / limit 548 |
| finetuned(1.0.1) | full | run_04 | — | — | — | — | — | (지표 없음) finetuned 548-sample benchmark |
| finetuned(1.0.1) | full | run_05 | — | — | — | — | — | (지표 없음) finetuned 548-sample benchmark |
| finetuned(1.0.1) | full | run_06 | 548 | 28.58% | 5.31 | 1.71 | 2.0 | finetuned 548-sample benchmark |
| finetuned(1.0.1) | full | run_07 | — | — | — | — | — | (지표 없음) 번역 버그 수정 후 finetuned(1.0.1) 548-sample |
| finetuned(1.0.1) | full | run_08 | 548 | 19.73% | 4.68 | 1.13 | 2.0 | 번역 버그 수정 후 finetuned(1.0.1) 548-sample |
| finetuned(1.0.1) | sample | run_01 | — | — | — | — | — | (지표 없음)  |
| finetuned(1.0.1) | sample | run_02 | 1 | 15.15% | 4.56 | 1.69 | 2.0 | finetuned(1.0.1) / sample / chunk 2.0s / Google Translate / no correction / limit 1 |
| finetuned_gpt_trans(1.0.2) | full | run_01 | 548 | 23.85% | 6.38 | 1.94 | 2.0 | finetuned_gpt_trans(1.0.2) / full / chunk 2.0s / GPT translation ctx=5 |
| finetuned_gpt_trans(1.0.2) | gpt_corrected_test | context_10 | 31 | 5.58% | 6.50 | 3.19 | 2.0 | context window 10문장 |
| finetuned_gpt_trans(1.0.2) | gpt_corrected_test | context_20 | 548 | 13.40% | 5.93 | 2.65 | 2.0 | context window 20문장 |
| finetuned_gpt_trans(1.0.2) | gpt_corrected_test | context_5 | 548 | 12.74% | 5.65 | 1.55 | — | gpt 보정 및 번역 테스트 |
| finetuned_gpt_trans(1.0.2) | gpt_corrected_test | context_50 | 548 | 13.23% | 5.82 | 1.75 | 2.0 | context window 50문장 |
| finetuned_silence(1.0.3) | chunk_size_test | chunk0.25 | 547 | 14.93% | 3.71 | 1.61 | 0.2 | chunk_size=0.25s sweep |
| finetuned_silence(1.0.3) | chunk_size_test | chunk0.25_2 | 548 | 10.92% | 3.40 | 0.87 | 0.2 | chunk_size=0.25s sweep |
| finetuned_silence(1.0.3) | chunk_size_test | chunk0.5 | 548 | 9.82% | 3.74 | 1.38 | 0.5 | chunk_size=0.5s sweep |
| finetuned_silence(1.0.3) | chunk_size_test | chunk1.0 | 548 | 8.71% | 4.16 | 1.01 | 1.0 | chunk_size=1.0s sweep |
| finetuned_silence(1.0.3) | chunk_size_test | chunk1.5 | 548 | 9.61% | 4.53 | 1.16 | 1.5 | chunk_size=1.5s sweep |
| finetuned_silence(1.0.3) | chunk_size_test | chunk2.0 | 548 | 10.60% | 4.73 | 1.01 | 2.0 | chunk_size=2.0s sweep |
| finetuned_silence(1.0.3) | full | run_01 | 548 | 11.38% | 5.17 | 1.35 | 2.0 | finetuned_silence(1.0.3) / full / chunk 2.0s |
| finetuned_silence(1.0.3) | full | run_02 | — | — | — | — | — | (지표 없음) finetuned_silence 548-sample benchmark |
| finetuned_silence(1.0.3) | full | run_03 | 548 | 9.26% | 4.78 | 1.49 | 2.0 | finetuned_silence 548-sample benchmark |
| finetuned_silence(1.0.3) | full | run_04 | 548 | 8.66% | 4.60 | 1.25 | 2.0 | 번역 버그 수정 후 1.0.3 재실행 (GPT 없음) |
| finetuned_silence(1.0.3) | full | run_05 | 548 | 8.77% | 4.60 | 1.51 | 2.0 | 번역 버그 수정 후 finetuned_silence(1.0.3) 548-sample |
| finetuned_silence(1.0.3) | sample | run_01 | 548 | 11.62% | 5.18 | 1.85 | 2.0 | silence 파인튜닝 모델 LibriSpeech 평가 |
| finetuned_silence_gpt(1.0.4) | full | run_01 | 548 | 16.27% | 5.89 | 2.10 | 2.0 | finetuned_silence_gpt(1.0.4) / full / chunk 2.0s / GPT translation ctx=5 |
| finetuned_silence_gpt(1.0.4) | full | run_02 | — | — | — | — | — | (지표 없음) finetuned_silence_gpt(1.0.4) full run_02 - parallel ASR/trans |
| finetuned_silence_gpt(1.0.4) | full | run_03 | 548 | 9.00% | 4.67 | 1.35 | 2.0 | finetuned_silence + GPT correction+translation 548-sample benchmark |
| finetuned_silence_gpt(1.0.4) | full | run_04 | 548 | 8.87% | 4.67 | 1.34 | 2.0 | finetuned_silence_gpt full benchmark (GPT trans+correction) |
| finetuned_silence_gpt(1.0.4) | full | run_05 | 12 | 3.52% | 5.40 | 0.65 | 2.0 | 번역 버그 수정 후 재실행 (lang=auto → targetLang 직접 번역) |
| finetuned_silence_gpt(1.0.4) | full | run_06 | 548 | 8.70% | 4.60 | 1.26 | 2.0 | 번역 버그 수정 후 재실행 (lang=auto → targetLang 직접 번역) |
| finetuned_silence_gpt(1.0.4) | full | run_07 | 548 | 8.72% | 4.59 | 1.51 | 2.0 | 번역 버그 수정 후 finetuned_silence_gpt(1.0.4) 548-sample |
| finetuned_silence_gpt(1.0.4) | full | run_08 | 548 | 10.81% | 5.20 | 2.08 | 2.0 | GPT버그 수정 후 finetuned_silence_gpt(1.0.4) 548-sample |
| finetuned_silence_gpt(1.0.4) | full | run_09 | 548 | 10.65% | 4.78 | 1.84 | 2.0 | seg commit 후 남은 오디오 누락 문제 해결 |
| finetuned_silence_gpt(1.0.4) | full | run_10 | 548 | 8.65% | 4.74 | 1.86 | 2.0 | seg, vad path에서 remaining 검사 |

</details>

## FSL

`fsl_sec` 필드. 플롯은 `utils/export_fsl_ftl_plots.py` 로 뽑는다.
