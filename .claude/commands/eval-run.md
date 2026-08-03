평가 실험을 설정하고 서버 실행부터 결과 요약까지 자동으로 진행해줘. TESTING_MANUAL.md 규칙을 따라야 해.

---

## 0. 실행 환경 확인 (SSH 필요 여부)

**가장 먼저, 다른 어떤 질문보다 앞서 확인.** 이 세션의 Bash가 이미 GPU 서버 위에서 도는 경우(예: 사용자가 로컬 컴퓨터에서 이 GPU 서버로 이미 SSH 접속한 뒤 그 안에서 Claude Code를 실행 중인 경우)가 있음 — 이때 다시 `ssh <host_alias>`로 접속하려 하면 alias가 로컬 `~/.ssh/config`에는 없거나(현재 세션의 `~/.ssh/config`는 사용자의 로컬 머신 것과 다름) 엉뚱한 호스트로 연결됨.

```bash
hostname; pwd; ls -d ~/STiTy 2>/dev/null; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
```

- `~/STiTy`가 존재하고 `nvidia-smi`에 RTX 4090이 뜨면 → **이미 GPU 서버 위**. 이후 이 문서의 모든 `ssh <host_alias> "..."` 블록은 **SSH 래퍼 없이 따옴표 안 명령을 그대로 로컬 Bash로 실행**하고, `cd <remote_dir> &&`도 생략 (이미 저장소 루트에 있다고 가정, 아니면 `cd ~/STiTy`만 실행).
- 위 조건이 안 맞으면 → 원격. **3번(원격 서버 선택)**에서 SSH alias를 사용자에게 직접 확인 (`~/.ssh/config`에 `gpu` 같은 고정 alias가 없을 수 있으니 후보를 보여주고 물어봐줘. `ProxyJump`로 연결하는 alias일 수도 있음).
- 판단은 세션당 한 번만 하고 이후 반복하지 않음.

---

## 1. 시작 분기

먼저 물어봐줘:
- **새 실험 시작** → 2번부터 진행
- **진행 중인 테스트 확인** → 아래 진행 중 확인 절차 실행

### 진행 중인 테스트 확인

로컬(0번에서 판단)이면 바로, 원격이면 SSH로 테스트 로그 확인:
```bash
tail -30 /tmp/eval_test.log                    # 로컬인 경우
ssh <host_alias> "tail -30 /tmp/eval_test.log"  # 원격인 경우
```

- 로그에 `Results saved to ...` 메시지가 있거나 서버가 종료된 상태라면 → 완료.
  경로를 파싱해서 `git pull` 후 **7번(결과 요약)으로 이동**
- 아직 진행 중이면 → 현재 진행 상황(최근 로그 라인) 출력 후 종료

---

결과 폴더 구조:
```
evaluation/{Dataset}/results/{model}/{scope}/{tag}/
├── metric.json
├── meta.json
├── description.txt
├── plots/
└── logs/server.log
```

---

## 2. 데이터셋 선택

| 선택 | 데이터셋 | 언어 | 특이사항 |
|---|---|---|---|
| A | **LibriSpeech** | 영어 | WER + FSL 타이밍, GPT 번역/교정 옵션 있음 |
| B | **DailyTalk** | 한국어 | CER, GPT 옵션 없음 |
| C | **KtelSpeech** | 한국어 | CER, 전화 품질 음성 |
| D | **AMI** | 영어 | WER, 다화자 회의, ami-dir/words-dir 별도 지정 |
| E | **Paper mode (mode2/3/4)** | 영어 | LibriSpeech 고정, `paper_result/ASR/scripts/` 재현 스크립트 사용 |

A를 선택한 경우, 이어서 물어봐줘:
- **일반 실행** → 4번(실험 설정)부터 진행
- **Paper mode 재현** → E와 동일하게 아래 "2-1. Paper mode" 절차로 이동

### 2-1. Paper mode (mode2/3/4)

번역/GPT 옵션 없이 커밋 정책만 다른 3가지 고정 설정을 재현하는 전용 스크립트.
논문 벤치마크용이라 **4번(실험 설정) 전체를 건너뛰고** 아래만 물어봐줘.

| 모드 | 설명 | 모델 | 커밋 정책 |
|---|---|---|---|
| mode2 | always-commit | baseline `Qwen/Qwen3-ASR-1.7B` | `--always-commit` (2초 고정 청킹) |
| mode3 | rule-based / dot-commit | baseline `Qwen/Qwen3-ASR-1.7B` | `--enable-dot-commit` |
| mode4 | seg-commit | finetuned ko `Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged` | 기본값 (SEG 토큰 기반) |

물어볼 것: **모드(2/3/4)**, **scope**(기본 `sample`), **tag**(기본 `run01`), **limit**(선택).
chunk-size는 스크립트에 `200ms` 고정, 번역은 항상 off(`--target-lang ""`)라 물어볼 필요 없음.

이후 5번(서버 실행)·6번(테스트 실행)은 아래 "Paper mode 전용 절차"를 따르고,
일반 실행 절차(수동 커맨드 조립)는 건너뛴다.

---

## 3. 원격 서버 선택

0번에서 **이미 GPU 서버 위**로 판단됐으면 이 섹션은 건너뛴다 (SSH 불필요).

원격으로 판단된 경우: SSH alias는 `~/.ssh/config`에 고정된 이름이 없을 수 있으니 사용자에게 직접 물어봐줘 (`ProxyJump`를 쓰는 경우도 있음 — 예: 로컬 → EC2 relay → GPU 서버). 접속 후 아래를 확인:

| 항목 | 값 |
|---|---|
| 저장소 경로 | `~/STiTy` |
| conda env | `stity` (⚠ `qwen3-asr` 아님) |
| GPU | 1장 (RTX 4090) |
| 부팅 시 자동 기동 systemd 서비스 | 없음 — 서비스 중단 절차 불필요 |

---

## 4. 실험 설정

### 4-1. 모델 종류
- **baseline** — 기본 모델. 버전 기본값 `1.0.0`, dot commit 자동 활성화
- **finetuned-merged** — merge된 파인튜닝 모델. 버전 기본값 `1.0.1`, `--enforce-eager` 필요
- **finetuned-lora** — lora-ready 모델 + `--lora`. 버전 기본값 `1.0.1`, `--enforce-eager` 필요

버전은 변경 가능 (예: `1.0.2`).

테스트 스크립트 `--model` 인자값 (폴더 레이블):
- baseline → `"baseline({version})"` (예: `baseline(1.0.0)`)
- finetuned-merged / finetuned-lora → `"finetuned({version})"` (예: `finetuned(1.0.1)`)

### 4-2. 청크 크기
- `1.0초` / `2.0초 (기본값)` / `3.0초` / 직접 입력
- 서버 `--chunk-size <초>`, 테스트 `--chunk-size-ms <밀리초>`로 변환

### 4-3. GPT 번역 (LibriSpeech 전용)
- **on** → `--gpt-translation` + context_window 입력 (기본값: 5)
- **off** → Google Translate (기본값)

### 4-4. GPT 교정 (LibriSpeech 전용)
- **on** → `--correction`
- **off** → 기본값

### 4-5. 평가 범위 (scope)
자유 입력. 기본값: `sample`. 예: `full`, `sample`, `chapter` 등.

### 4-6. 실험 태그 (tag)
선택 입력. 비워두면 스크립트가 `run_01`, `run_02` ... 자동 생성.
서버 `--log-file` 경로를 맞추려면 태그를 미리 지정하는 것이 권장됨.

기본값 제안: 직전 `run_NN` 폴더 확인 후 다음 번호 제안 (예: `run_03`).
```bash
ls evaluation/{Dataset}/results/{model_label}/{scope}/ 2>/dev/null | sort | tail -1
```

### 4-7. 테스트 파일 수 (limit, 선택)
처리할 최대 파일 수. 비워두면 scope 전체 실행.
- 예: `50`, `100` (빠른 검증 시 유용)

### 4-8. description 자동 생성
사용자에게 묻지 말고, 앞서 수집한 설정을 바탕으로 자동 생성해서 `--description`으로 전달해줘.

형식 예시:
```
finetuned(1.0.1) / sample / chunk 2.0s / Google Translate / no correction
baseline(1.0.0) / full / chunk 1.0s / GPT translation ctx=5 / correction on
```

---

## 5. 서버 실행 (SSH/로컬 + tmux)

(0번에서 **이미 GPU 서버 위**로 판단됐다면, 아래 모든 `ssh <host_alias> "..."` 블록은 SSH 없이 따옴표 안 명령만 로컬로 실행하고 `cd <remote_dir> &&`도 생략.)

**환경 활성화 규칙 (머신마다 다름):** 파이썬 환경 이름을 가정하지 말 것.
- `~/STiTy/.venv`가 있으면 → venv 머신 (예: `skkai`). `. ~/STiTy/.venv/bin/activate`
- 없으면 → 원격 GPU 서버. `. ~/miniforge3/etc/profile.d/conda.sh && conda activate stity`

아래 tmux 블록들은 이 분기를 인라인으로 넣어 두었으니 그대로 쓰면 된다.
`conda activate stity`를 무조건 넣으면 venv 머신에서 `EnvironmentNameNotFound`로 `&&` 체인이 끊긴다.

**tmux:** `skkai`에는 conda-forge tmux가 `~/miniforge3/envs/tools`에 설치돼 있고 `~/.local/bin/tmux`
래퍼로 노출돼 있다 (`conda init`은 하지 않음 — base env가 PATH를 오염시키지 않게). 다른 머신에
tmux가 없으면 `conda install -y -n tools -c conda-forge tmux`로 설치 — sudo 불필요.

**참고:** `serve_mode*.sh` / `run_mode*.sh`는 자체 env 가드로 `.venv`를 자동 감지하고
`PYTHONPATH=`를 비워서 파이썬을 호출하므로, paper mode는 활성화 없이 `bash` 호출만으로도 동작한다.

**데이터셋과 관계없이 서버 스크립트는 공통:**
`evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`

**모델별 서버 명령:**

```bash
# baseline
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --chunk-size <n> \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"

# finetuned-merged
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --model ~/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged \
  --chunk-size <n> \
  --enforce-eager \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"

# finetuned-lora
python evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py \
  --model ~/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-lora-ready \
  --lora \
  --chunk-size <n> \
  --enforce-eager \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"
```

태그를 지정하지 않은 경우 `--log-file` 생략.

**Paper mode 서버 명령 (2-1에서 E/paper mode 선택 시, 위 명령 대신 사용):**

```bash
bash evaluation/LibriSpeech/paper_result/ASR/scripts/serve_mode<N>.sh <scope> <tag>
```

`<N>`은 2/3/4. 모델·커밋 정책은 스크립트에 고정돼 있어 별도 인자 불필요.
로그는 스크립트가 자체적으로 `paper_result/ASR/mode<N>/{scope}/{tag}/logs/server.log`에 저장하므로 `--log-file` 조립 불필요.

**Step 5-1. 기존 tmux 세션 정리**
```bash
ssh <host_alias> "tmux kill-session -t eval_server 2>/dev/null; true"
```

**Step 5-2. tmux로 서버 시작**
```bash
ssh <host_alias> "cd <remote_dir> && git pull && tmux new-session -d -s eval_server '<server_command> 2>&1 | tee /tmp/eval_server.log'"
```

**Step 5-3. 포트 준비 대기 (최대 120초)**
```bash
ssh <host_alias> "timeout 120 bash -c 'until nc -z localhost 8765; do sleep 3; done && echo SERVER_READY'"
```

**에러 처리:**
timeout 만료 시 서버 로그 40줄 출력해서 원인 진단 (OOM, CUDA 에러, 포트 충돌 등):
```bash
ssh <host_alias> "tail -40 /tmp/eval_server.log"
```
수정 가능하면 Step 5-1부터 재시도, 불가능하면 사용자에게 설명 후 중단.

---

## 6. 테스트 실행

(0번에서 로컬로 판단됐다면 여기도 SSH 없이 로컬로 실행 — 위 5번과 동일한 규칙.)

서버와 테스트 스크립트 모두 GPU 서버에서 실행. `--host localhost` 사용.
테스트도 tmux 세션(`eval_test`)으로 실행해 (원격인 경우) SSH 연결이 끊겨도 계속 돌아가게 해줘.

**Step 6-1. 기존 eval_test 세션 정리**
```bash
ssh <host_alias> "tmux kill-session -t eval_test 2>/dev/null; true"
```

**Step 6-2. tmux로 테스트 시작**

데이터셋별 명령을 tmux 세션으로 실행:

테스트 완료 후 자동으로 git push까지 체이닝해줘 (물리 GPU 서버라 원격 종료는 하지 않음).
tmux 명령 끝에 아래를 항상 추가:
```
&& git add -A && git commit -m "eval: <model_label> <dataset> <scope> <tag> 결과 추가" && git push
```

**LibriSpeech:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  '{ [ -x ~/STiTy/.venv/bin/python ] && . ~/STiTy/.venv/bin/activate || { . ~/miniforge3/etc/profile.d/conda.sh && conda activate stity; }; } && \
  python evaluation/LibriSpeech/servers/test_qwen3_librispeech.py \
    --test-dir evaluation/LibriSpeech/LibriSpeech/test-other \
    --model \"<model_label>\" --scope <scope> [--tag <tag>] \
    --description \"<auto_generated>\" [--limit <n>] \
    --host localhost --port 8765 --target-lang ko \
    --trailing-silence-ms 5500 --chunk-size-ms <ms> \
    [--gpt-translation --context-window <n>] [--correction] \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> LibriSpeech <scope> <tag> 결과 추가\" \
  && git push' \
  && echo TEST_STARTED"
```

**Paper mode (mode2/3/4):**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  '{ [ -x ~/STiTy/.venv/bin/python ] && . ~/STiTy/.venv/bin/activate || { . ~/miniforge3/etc/profile.d/conda.sh && conda activate stity; }; } && \
  bash evaluation/LibriSpeech/paper_result/ASR/scripts/run_mode<N>.sh <scope> <tag> [--limit <n>] \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: mode<N> LibriSpeech <scope> <tag> 결과 추가\" \
  && git push' \
  && echo TEST_STARTED"
```
`run_mode<N>.sh`가 시작 시 `check_server_config.py`로 서버 설정을 자체 검증하므로,
서버가 Step 5에서 올린 `serve_mode<N>.sh`와 같은 `<N>`인지만 맞춰주면 됨 (불일치 시 스크립트가 즉시 실패).

**DailyTalk:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  '{ [ -x ~/STiTy/.venv/bin/python ] && . ~/STiTy/.venv/bin/activate || { . ~/miniforge3/etc/profile.d/conda.sh && conda activate stity; }; } && \
  python evaluation/DailyTalk/test_qwen3_dailytalk.py \
    --host localhost --port 8765 --chunk-size-ms <ms> [--limit <n>] \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> DailyTalk 결과 추가\" \
  && git push' \
  && echo TEST_STARTED"
```

**KtelSpeech:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  '{ [ -x ~/STiTy/.venv/bin/python ] && . ~/STiTy/.venv/bin/activate || { . ~/miniforge3/etc/profile.d/conda.sh && conda activate stity; }; } && \
  python evaluation/KtelSpeech/test_qwen3_ktelspeech.py \
    --data-dir evaluation/KtelSpeech \
    --model \"<model_label>\" --scope <scope> [--tag <tag>] \
    --description \"<auto_generated>\" [--limit <n>] \
    --host localhost --port 8765 --trailing-silence-ms 5000 \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> KtelSpeech <scope> <tag> 결과 추가\" \
  && git push' \
  && echo TEST_STARTED"
```

**AMI:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  '{ [ -x ~/STiTy/.venv/bin/python ] && . ~/STiTy/.venv/bin/activate || { . ~/miniforge3/etc/profile.d/conda.sh && conda activate stity; }; } && \
  python evaluation/AMI/test_qwen3_ami.py \
    --ami-dir evaluation/AMI/AMI --words-dir evaluation/AMI/words \
    --model \"<model_label>\" --scope <scope> [--tag <tag>] \
    --description \"<auto_generated>\" [--limit <n>] \
    --host localhost --port 8765 --trailing-silence-ms 5000 \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> AMI <scope> <tag> 결과 추가\" \
  && git push' \
  && echo TEST_STARTED"
```

테스트는 백그라운드(tmux)에서 실행. 완료되면 자동으로 git push (서버 종료는 하지 않음, 물리 머신이므로 그대로 유지).
진행 상황 확인이 필요하면 다시 `/eval-run` → "진행 중인 테스트 확인" 선택.

---

## 7. 결과 요약

결과 파일은 GPU 서버에 있음 — 0번에서 로컬로 판단됐으면 그냥 `cat`/`ls`로 읽고, 원격이면 SSH로 읽어줘.

**태그 확인 (미지정 시):**
```bash
ssh <host_alias> "ls -t <remote_dir>/evaluation/{Dataset}/results/{model_label}/{scope}/ | head -1"
```

**metric.json 읽기:**
```bash
ssh <host_alias> "cat <remote_dir>/evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/metric.json"
```

**Paper mode (mode2/3/4)는 결과 경로가 다름** — `--results-root`로 `paper_result/ASR`를 지정해서 실행되므로:
```bash
ssh <host_alias> "cat <remote_dir>/evaluation/LibriSpeech/paper_result/ASR/mode<N>/{scope}/{tag}/metric.json"
```

**출력 지표:**
- WER (영어) / CER (한국어)
- COMET score (있으면)
- FSL: median / mean / P90
- encode_sec / decode_sec
- SEG / VAD / finish 커밋 비율

같은 model+scope 아래 다른 run이 있으면 나란히 비교.
결과 요약 후 `/notion-summary`나 `/git-sync` 실행할지 물어봐줘.
