# evaluation/KsponSpeech/

한국어 자유발화 벤치마크. 메트릭은 **CER**(문자 오류율). 오디오는 raw PCM(s16le, 16kHz, mono)
이고 클립마다 새 WebSocket 연결을 맺어 컨텍스트 오염을 막는다.

## 빠른 시작

```bash
# 터미널 1 — 공유 평가 서버
python evaluation/streaming_websocket_server_ast.py --no-idle-shutdown

# 터미널 2
python evaluation/KsponSpeech/test_qwen3_kspon.py \
  --data-json evaluation/KsponSpeech/transcribe/eval_clean_1000.json \
  --data-dir  evaluation/KsponSpeech/data/eval_clean \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

**오디오는 리포에 없다.** `--data-dir` 이 가리키는 `data/` 는 `.gitignore` 로 빠져 있고 현재
이 작업 트리에도 없다. 리포에 있는 음성은 `sample_data/` 의 40개뿐이다(아래).

## 파일

| 경로 | 무엇 |
|---|---|
| `transcribe/eval_clean.json` | 전체 clean 평가셋 |
| `transcribe/eval_clean_1000.json` | 1,000개 서브셋. 평가 기본값이자 autoseg(`runtime/data.py`)의 입력 |
| `transcribe/train.json` / `train2.json` | 파인튜닝용 전사. CIF 연구 코드와 `Qwen3-ASR/finetuning/utils/assemble_dataset.py` 가 읽는다 |
| `transcribe/split_train2.json` | 문장 중간에서 자른 partial 쌍. `convert_split_to_jsonl.py` 입력 |
| `sample_data/eval_clean/` | PCM 20개. 웹 데모의 한국어 샘플 — `STiTy-Mobile/demo-web/partial_demo/make_demo_wavs.py` 가 읽는다 |
| `sample_data/KsponSpeech_0001/` | PCM 20개. 참조 없음 — 형식 확인용 표본 |
| `utils/extract_trn_to_json.py` | 원본 `.trn` → JSON |

원본 전사 `transcribe/train.trn`(87MB)은 리포에 두지 않는다(`.gitignore:56`). 코드가 읽는 것은
파생본 `train.json` / `train2.json` 이다. 원본이 필요하면 AI-Hub 코퍼스에서 받아 아래로 변환한다.

```bash
python evaluation/KsponSpeech/utils/extract_trn_to_json.py \
  --trn transcribe/train.trn --folders 0001 0010 --output transcribe/train.json
```

## 측정 결과 (2026-06)

`results/` 는 이제 git 이 추적하지 않는다. 아래가 그때 낸 수치의 기록이다.
CER 은 `metric.json` 의 `overall.corpus_cer`, FTL/FSL 은 각각 첫 토큰 지연과 서버 FSL 평균이다.

| 모델 | 런 | 청크 | 발화수 | **CER** | FTL(s) | FSL(s) | 비고 |
|---|---|---|---|---|---|---|---|
| baseline(1.0.0) | run_01 | 1.0s | 953 | 17.60% | 6.00 | 2.25 | 1.0.4 세팅 미러, 파인튜닝과 GPU 동시 사용 |
| finetuned_silence(1.0.3) | run_01 | 2.0s | 996 | 18.48% | 4.21 | 1.62 | **서버 버그 이전 값** |
| finetuned_silence(1.0.3) | run_02 | 2.0s | 996 | 8.82% | 3.55 | 1.13 | order-fix + max-new-tokens 128 적용 |
| finetuned_silence(1.0.3) | run_03 | 1.0s | 999 | 9.61% | 3.09 | 1.07 | all-fix + chunk 1.0 |
| finetuned_silence(1.0.4) | run_01 | 2.0s | 324 | 11.01% | 4.04 | 1.18 | 중간에 끊긴 런(324/1000) |
| finetuned_silence(1.0.4) | run_02 | 2.0s | 814 | 9.62% | 4.02 | 1.20 | eval sort-fix 추가 |
| **finetuned_silence(1.0.4)** | **run_03** | 2.0s | 995 | **7.78%** | 3.85 | 1.07 | 최저 CER |
| finetuned_silence(1.0.4) | run_04 | 1.0s | 1000 | 7.98% | 3.75 | 0.99 | all-fix + chunk 1.0 |
| finetuned_silence(1.0.5) | run_01 | 1.0s | 1000 | 8.62% | 4.21 | 2.03 | v4c100 가중치 |

**18.48% → 8.82% 는 모델이 좋아진 게 아니라 서버 버그를 고친 결과다.** 같은
`ko-merged-silence` 가중치이고 바뀐 것은 순서 꼬임(drain + segid 정렬)과
`max-new-tokens 128` 두 가지다. 파인튜닝 가중치 간 비교는 run_02 이후끼리만 해야 한다.

가중치 계보: 1.0.3 = `ko-merged-silence`, 1.0.4 = `ko-silence-v3-merged`(ckpt-900),
1.0.5 = `ko-silence-v4c100-merged`.

## `results/` 는 추적하지 않는다

`.gitignore` 에 들어 있다 — 발화 단위 결과(`raw_results`)를 담은 `metric.json` 이 실행마다
수 MB 씩 쌓이고 서버 로그는 그보다 크다. 다른 데이터셋과 같은 정책이고, 지금까지의 수치는
위 표가 기록이다.
