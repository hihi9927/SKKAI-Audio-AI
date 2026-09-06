# STiTy ASR 평가 테스트 매뉴얼

## 1. 개요

테스트 결과는 `--model`, `--scope`, `--tag` 세 인자에 따라 아래 구조로 자동 저장됩니다.

```
evaluation/{Dataset}/results/
└── {model}/          ← --model  (대분류: 모델 종류)
    └── {scope}/      ← --scope  (소분류: 테스트 범위)
        └── {tag}/    ← --tag    (폴더명, 미지정 시 run_01, run_02 ... 자동 생성)
            ├── metric.json       # WER/CER 등 핵심 성능 지표
            ├── meta.json         # 실행에 사용된 모든 CLI 인자
            ├── description.txt   # --description으로 전달한 테스트 설명
            ├── plots/            # 샘플 파일별 타이밍 플롯 (PNG + LOG)
            └── logs/             # 서버/테스트 실행 로그
```

각 실행은 독립된 폴더에 저장되므로 덮어쓸 위험이 없습니다.  
`--tag`를 지정하면 동일 폴더에 이어서 저장(resume)됩니다.

---

## 2. 공통 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | `finetuned` | 대분류: 모델 종류 (예: `baseline(1.0.0)`, `finetuned(1.0.1)`) |
| `--scope` | `sample` | 소분류: `full`(전체 데이터셋) 또는 `sample`(일부) |
| `--tag` | 자동 생성 | 결과 폴더명. 지정 시 해당 폴더에 이어서 저장 |
| `--description` | — | 테스트 설명. `description.txt`에 저장 |
| `--host` | `localhost` | 서버 호스트 |
| `--port` | `8765` | 서버 포트 |
| `--limit` | — | 처리할 최대 파일 수 (미지정 시 전체) |
| `--chunk-size-ms` | `200` | 오디오 청크 크기 (ms) |
| `--send-interval-ms` | `200` | 청크 전송 간격 (ms, 실시간 페이싱) |
| `--trailing-silence-ms` | 스크립트별 상이 (LibriSpeech `8000`) | 오디오 끝에 추가할 묵음 길이 (VAD 트리거용) |
| `--target-lang` | `ko` | 번역 대상 언어 |
| `--fresh-start` | — | 기존 결과 무시하고 처음부터 재실행 |
| `--auto-server` | — | ASR 서버 자동 시작/종료 |
| `--server-model` | 미지정 시 `--model` 별칭이 가리키는 경로 | 자동 시작 시 사용할 모델 |

> `baseline` 계열 모델(`Qwen/Qwen3-ASR-1.7B`, `baseline`, `baseline(1.0.0)` 등)은 서버에서 `dot commit`이 기본 활성화됩니다.  
> baseline에서도 끄고 싶으면 `--disable-dot-commit`, finetuned 등에서 강제로 켜고 싶으면 `--enable-dot-commit`을 사용합니다.
>
> **`dot commit`이 켜지면 확정 게이트(`--dot-commit-confirm`)도 자동으로 켜집니다.** 즉 모드3 실행에는 별도
> 인자가 필요 없습니다. 게이트 없이 감지 즉시 커밋하던 예전 동작으로 돌리려면 `--no-dot-commit-confirm`을 씁니다.
> 게이트 상세는 [08_03 확정게이트 알고리즘 명세](../notion_docs/08_03_dot_commit_확정게이트_알고리즘_명세.md) 참조.

---

## 3. 데이터셋별 실행 방법

### 3.1 서버 실행 (공통)

서버 로그를 결과 폴더(`logs/server.log`)에 저장하려면 `--log-file`을 지정해야 합니다.  
run 폴더 이름은 테스트 스크립트 실행 시 결정되므로, **`--tag`로 폴더명을 미리 고정**한 뒤 서버와 테스트 스크립트에 같은 경로를 맞춰줍니다.  

baseline 서버는 `dot commit`이 기본으로 켜지므로 별도 인자를 붙이지 않아도 됩니다.  
반대로 baseline에서 `dot commit`을 끄고 싶다면 서버 실행 시 `--disable-dot-commit`을 추가합니다.

```bash
cd ~/STiTy && git pull

# Baseline 모델
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --no-idle-shutdown \
  --log-file "evaluation/LibriSpeech/results/baseline(1.0.0)/sample/run_01/logs/server.log"

# Finetuned 모델
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --model models/Qwen3-ASR-1.7B-en-silence-c80-merged \
  --no-idle-shutdown \
  --enforce-eager \
  --log-file "evaluation/LibriSpeech/results/finetuned(1.0.1)/sample/run_01/logs/server.log"
```

> **파인튜닝 가중치 경로:** 현재 저장소의 가중치는 `models/Qwen3-ASR-1.7B-en-silence-c80-merged`(영어)와
> `models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged`(한국어)입니다.
> 테스트 스크립트의 `--model "finetuned(1.0.1)"` 별칭은 아직 존재하지 않는 예전 경로
> (`Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged`)로 매핑되어 있으므로, 서버에는 위처럼 실제 경로를 직접 넘기세요.

> **`--log-file` 경로 규칙:** `evaluation/{Dataset}/results/{model}/{scope}/{tag}/logs/server.log`  
> 서버를 먼저 시작하고, 테스트 스크립트에서 동일한 `--model`, `--scope`, `--tag` 조합을 사용하면 같은 폴더에 결과가 모입니다.  
> 로그 저장이 필요 없으면 `--log-file`을 생략해도 됩니다.

---

### 3.2 LibriSpeech

```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model <baseline(1.0.0)|finetuned(1.0.1)> \
  --scope <full|sample> \
  [--tag <폴더명>] \
  [--description "테스트 설명"] \
  --target-lang ko \
  --trailing-silence-ms 8000 \
  --chunk-size-ms 200
```

> **`--trailing-silence-ms` 기본값이 5500 → 8000으로 바뀌었습니다.** dot-commit 확정은 누적 오디오가
> `chunk_size_sec` 배수에 도달할 때만 판정되므로, 무음이 짧으면 마지막 문장이 확정되기 전에 스트림이
> 끊겨 `finish` 커밋으로 빠집니다 (실측: 5500ms에서 0.11초 모자라 finish 발생, 8000ms에서 dot 확정).

`--auto-server`를 사용할 때도 같은 규칙이 적용됩니다. baseline 계열 모델은 `dot commit`이 자동 활성화되고, 필요하면 `--server-args "--disable-dot-commit"` 또는 `--server-args "--enable-dot-commit"`으로 override할 수 있습니다.

결과 위치: `evaluation/LibriSpeech/results/{model}/{scope}/{tag}/`

**예시 1 — finetuned, 전체 데이터셋**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "finetuned(1.0.1)" \
  --scope full \
  --description "finetuned 모델 전체 test-other 기준선 측정"
# → results/finetuned(1.0.1)/full/run_01/
```

**예시 2 — baseline, 일부 샘플**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "baseline(1.0.0)" \
  --scope sample \
  --limit 50
# → results/baseline(1.0.0)/sample/run_01/
```

**예시 3 — chunk size 변경 실험 (tag 지정)**
```bash
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
  --model "finetuned(1.0.1)" \
  --scope full \
  --tag chunk_1500ms \
  --chunk-size-ms 1500
# → results/finetuned(1.0.1)/full/chunk_1500ms/
```

---

### 3.3 KsponSpeech (자유발화, Korean)

KsponSpeech eval_clean 클립으로 구성됩니다.  
오디오: raw PCM (s16le, 16 kHz, mono). 클립마다 새 WebSocket 연결을 맺어 컨텍스트 오염을 방지합니다. 평가 지표: **CER** (문자 오류율).

```bash
# 1) 서버 시작 (별도 터미널)
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --no-idle-shutdown \
  --log-file "evaluation/KsponSpeech/results/baseline(1.0.0)/sample/run_01/logs/server.log"

# 2) 테스트 실행
python evaluation/KsponSpeech/test_qwen3_kspon.py \
  --data-json evaluation/KsponSpeech/transcribe/eval_clean_1000.json \
  --data-dir evaluation/KsponSpeech/data/eval_clean \
  --model "baseline(1.0.0)" \
  --scope sample \
  --tag run_01 \
  --description "테스트 설명" \
  --trailing-silence-ms 5500
```

> **참고:**
> - `--data-json`: 레이블 JSON 파일. `eval_clean_1000.json`(1000개, 빠른 실행) 또는 `eval_clean.json`(전체).
> - `--data-dir`: PCM 오디오 파일이 있는 디렉토리.
> - `--src-lang ko`가 기본값이므로 생략 가능합니다.
> - `--limit 100`으로 일부 클립만 빠르게 테스트할 수 있습니다.

결과 위치: `evaluation/KsponSpeech/results/{model}/{scope}/{tag}/`

---

### 삭제된 데이터셋

AMI, KtelSpeech, AliMeeting, (zh)RAMC, (es)CIEMPIESS, KokoroSpeech, ReazonSpeech, smartturn 은
2026-09-06 에 리포에서 지웠다. 실행 방법 문서도 함께 지웠고, 그 데이터셋들로 돌린 런의 요약 지표만
[ARCHIVED_DATASETS_METRICS.md](ARCHIVED_DATASETS_METRICS.md) 에 남아 있다. 테스트 클라이언트
스크립트(`test_qwen3_*.py`)는 삭제 커밋 이전 git 이력에서 복구할 수 있다.

---

## 4. 결과 파일 설명

| 파일 | 설명 |
|---|---|
| `metric.json` | 핵심 성능 지표 (WER/CER, FSL, latency 등) |
| `meta.json` | 실행 당시 모든 CLI 인자 + 타임스탬프. 재현에 사용 |
| `description.txt` | `--description`으로 전달한 이 테스트의 목적 설명 |
| `plots/` | 샘플 파일별 타이밍 분석 플롯 (PNG) 및 원본 로그 (LOG) |
| `logs/` | 서버 stdout/stderr 로그 |

---

## 5. 동시 접속 벤치마크

N명이 동시 접속할 때 WER / FSL이 어떻게 변하는지 측정합니다. 스크립트는 `evaluation/LibriSpeech/` 바로 아래에 있습니다.

| 스크립트 | 용도 |
|---|---|
| `run_concurrent_chapters.py` | 챕터를 N개 클라이언트로 나눠 동시 실행 (**현재 주 경로**, CLI 인자 지원) |
| `run_concurrent_benchmark.py` | 동시 1~10명 스윕 (인자 없음, 상단 상수 편집) |
| `run_multi_speaker_full.py` | full 스코프 다화자 실행 (인자 없음, 상단 상수 편집) |

```bash
python evaluation/LibriSpeech/run_concurrent_chapters.py \
  --num-clients 16 --model mode2 --scope full --tag c16_run01
```

> **주의:** `run_concurrent_benchmark.py`와 `run_multi_speaker_full.py`는 `PROJECT_ROOT = Path("/home/ubuntu/STiTy")`가 하드코딩되어 있습니다. 다른 머신에서는 파일 상단 상수를 고쳐야 합니다.

---

## 6. Resume (이어서 실행)

중단된 테스트를 이어서 실행하려면 `--tag`로 동일 폴더를 지정합니다.  
이미 처리된 파일은 `metric.json`을 기준으로 자동으로 건너뜁니다.

```bash
# 처음 실행 (run_01 자동 생성)
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full

# 중단 후 이어서 실행
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full --tag run_01

# 처음부터 다시 실행
python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
  --model "finetuned(1.0.1)" --scope full --tag run_01 --fresh-start
```

---

## 7. AST 트랙 (음성번역 — LAAL + BLEU)

ASR 트랙(WER/CER + FSL)과 **서버도 클라이언트도 다르다.** 지연은 LAAL(ms), 품질은 BLEU 로 잰다.
데이터셋은 manifest(JSONL)로 갈아 끼우므로 데이터셋별 클라이언트를 만들지 않는다.

전체 절차·지표 정의·주의사항: [ast/README.md](ast/README.md)

```bash
# 0) 의존성 (sacrebleu 추가됨)
pip install -r evaluation/ast/requirements.txt

# 1) 데이터 (FLEURS, 리포 밖에 둔다 — en 오디오 + 타깃 언어 참조 TSV, 약 280MB)
hf download google/fleurs --repo-type dataset \
  --include "data/en_us/test.tsv" "data/en_us/audio/test.tar.gz" \
            "data/de_de/test.tsv" "data/ko_kr/test.tsv" "data/ja_jp/test.tsv" \
            "data/cmn_hans_cn/test.tsv" "data/es_419/test.tsv" \
  --local-dir ~/datasets/fleurs

# 2) manifest 생성
python evaluation/ast/build_manifest_fleurs.py \
  --fleurs-root ~/datasets/fleurs --src en_us --tgt de_de \
  --out evaluation/ast/manifests/fleurs_en-de_test.jsonl --verify-audio

# 3) 서버 (별도 터미널)
python evaluation/streaming_websocket_server_ast.py \
  --model models/Qwen3-ASR-1.7B-en-silence-c80-merged --no-idle-shutdown

# 4) 테스트 실행
python evaluation/ast/test_ast.py \
  --manifest evaluation/ast/manifests/fleurs_en-de_test.jsonl \
  --dataset FLEURS --model "en-silence-c80" --scope full --tag run_01 \
  --src-lang en --target-lang de

# 5) 서버 종료 (pkill 은 vLLM EngineCore 를 남긴다)
bash evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh 8765
```

결과: `evaluation/ast/results/{dataset}/{model}/{scope}/{tag}/metric.json`
(`--tag` 재사용 시 이어서 실행, `--fresh-start` 로 초기화 — ASR 트랙과 동일)

### AST 전용 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--manifest` | (필수) | `build_manifest_*.py` 로 만든 JSONL |
| `--dataset` | `MuST-C` | 결과 디렉토리의 데이터셋 이름 (FLEURS 사용 시 `--dataset FLEURS`) |
| `--src-lang` | `en` | ASR 소스 언어. `auto` 로 두면 서버의 언어 제한이 꺼진다 |
| `--target-lang` | `de` | 번역 대상 |
| `--laal-unit` | `word` | LAAL 의 \|Y\| 단위 (de/en/es → word, zh/ja → char) |
| `--laal-cap-source` | 켬 | 비계산인지 지연을 소스 길이로 상한 |
| `--bleu-tokenize` | 타깃 언어로 결정 | sacrebleu 토크나이저 |
| `--strip-nonspeech` | 켬 | `(Laughter)` 등 이벤트 표기 제거 |

> 아래 네 인자는 **점수를 바꾼다.** 다른 값으로 낸 점수끼리는 비교할 수 없으므로 실험군 전체에서
> 고정해야 한다. 실제 사용값은 `meta.json` 과 `metric.json` 의 `summary` 에 자동 기록된다.

> **VAD off 로 돌릴 때 주의.** base 서버는 스트림 종료 시 남은 오디오의 최종 디코딩
> (`_asr_finish_streaming`)과 번역 태스크 드레인(`_drain_pending_gpt`)을 하지 않는다 —
> 둘 다 VAD 커밋 경로에만 있다. 그래서 마지막 미완성 청크의 음성이 **전사조차 되지 않는다.**
> 커밋 정책과 무관하므로 always/dot/seg 가 똑같이 겪는다. AST 평가 서버는 둘 다 보완했다.
> 실측(CoVoST2 200발화, 침묵 500ms): seg BLEU 27.62 → 36.71, dot 27.57 → 36.32.
> 침묵을 4000ms 로 늘려도 비슷해지지만 처리량이 13→8배속으로 떨어진다.

> 병렬 실행은 `--clients N`. 16 병렬 실측 8.0배속(침묵 4초 포함), 번역 실패 0건.

> FLEURS en→de test 전체는 346발화(0.95시간 오디오)로 실시간 페이싱 기준 **약 63분**이다.
> 반복 개발은 `--limit` 서브셋으로 한다. 타깃 언어는 `--tgt` 만 바꾸면 된다
> (`ko_kr` `ja_jp` `cmn_hans_cn` `es_419`) — 오디오는 소스 언어 것만 있으면 되므로 재다운로드가 필요 없다.
>
> MuST-C 는 배포처가 사라져 쓰지 않는다. 경위와 사본 확보 시 절차는 [ast/README.md](ast/README.md) 참조.
