평가 실험을 설정하고 원격 서버 실행부터 결과 요약까지 자동으로 진행해줘. TESTING_MANUAL.md 규칙을 따라야 해.

---

## 0. 시작 분기

먼저 물어봐줘:
- **새 실험 시작** → 1번부터 진행
- **진행 중인 테스트 확인** → 아래 진행 중 확인 절차 실행

### 진행 중인 테스트 확인

어느 서버에서 돌고 있는지 물어보고, SSH로 테스트 로그 확인:
```bash
ssh <host_alias> "tail -30 /tmp/eval_test.log"
```

- 로그에 `Results saved to ...` 메시지가 있거나 서버가 종료된 상태라면 → 완료.
  경로를 파싱해서 `git pull` 후 **6번(결과 요약)으로 이동**
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

## 1. 데이터셋 선택

| 선택 | 데이터셋 | 언어 | 특이사항 |
|---|---|---|---|
| A | **LibriSpeech** | 영어 | WER + FCL 타이밍, GPT 번역/교정 옵션 있음 |
| B | **DailyTalk** | 한국어 | CER, GPT 옵션 없음 |
| C | **KtelSpeech** | 한국어 | CER, 전화 품질 음성 |
| D | **AMI** | 영어 | WER, 다화자 회의, ami-dir/words-dir 별도 지정 |

---

## 2. 원격 서버 선택

먼저 AWS에서 현재 인스턴스 상태를 조회해줘:
```bash
aws ec2 describe-instances \
  --instance-ids i-0adab4f1c94585f9d i-0ecb954ca6ebde868 i-08319fccc65145965 \
  --query "Reservations[].Instances[].[InstanceId,PublicIpAddress,State.Name,Tags[?Key=='Name'].Value|[0]]" \
  --output table
```

조회 결과를 바탕으로 아래 표를 채워서 보여주고 선택받아줘:

| 별칭 | IP | 이름 | 상태 |
|---|---|---|---|
| `aws_dev` | 13.209.202.8 | Dev Server | (조회 결과) |
| `aws_app` | 15.165.227.114 | App Server | (조회 결과) |
| `aws_test` | 54.116.64.206 | Test Server | (조회 결과) |
| `super_com` | 115.145.230.36 | — | (SSH만 가능) |

SSH 접속 정보는 ~/.ssh/config 에서 자동으로 읽어 사용.
원격 STiTy 디렉토리도 물어봐줘 (기본값: `~/STiTy`).

### EC2 기동 (stopped 상태인 경우)

```bash
# 상태 확인
aws ec2 describe-instances --instance-ids <instance_id> \
  --query 'Reservations[0].Instances[0].State.Name' --output text

# stopped 이면 기동 및 대기
aws ec2 start-instances --instance-ids <instance_id>
aws ec2 wait instance-running --instance-ids <instance_id>
```
IP는 고정이므로 재조회 불필요. `super_com`은 이 단계 건너뜀.

**에러 처리:** start-instances 실패 시 에러 출력 후 중단.

### aws_app 전용: 자동 실행 서비스 중단

`aws_app`은 EC2 기동 시 `qwen-asr` systemd 서비스가 자동으로 모델을 실행하고,
클라이언트 연결이 1분간 없으면 EC2 자체가 자동 종료되는 구조.
평가 서버를 올리기 전에 반드시 이 서비스를 먼저 종료해줘:

```bash
ssh aws_app "sudo systemctl stop qwen-asr"
```

기동 직후 바로 실행해서 자동 종료 타이머가 발동하기 전에 처리해야 해.

---

## 3. 실험 설정

### 3-1. 모델 종류
- **baseline** — 기본 모델. 버전 기본값 `1.0.0`, dot commit 자동 활성화
- **finetuned-merged** — merge된 파인튜닝 모델. 버전 기본값 `1.0.1`, `--enforce-eager` 필요
- **finetuned-lora** — lora-ready 모델 + `--lora`. 버전 기본값 `1.0.1`, `--enforce-eager` 필요

버전은 변경 가능 (예: `1.0.2`).

테스트 스크립트 `--model` 인자값 (폴더 레이블):
- baseline → `"baseline({version})"` (예: `baseline(1.0.0)`)
- finetuned-merged / finetuned-lora → `"finetuned({version})"` (예: `finetuned(1.0.1)`)

### 3-2. 청크 크기
- `1.0초` / `2.0초 (기본값)` / `3.0초` / 직접 입력
- 서버 `--chunk-size <초>`, 테스트 `--chunk-size-ms <밀리초>`로 변환

### 3-3. GPT 번역 (LibriSpeech 전용)
- **on** → `--gpt-translation` + context_window 입력 (기본값: 5)
- **off** → Google Translate (기본값)

### 3-4. GPT 교정 (LibriSpeech 전용)
- **on** → `--correction`
- **off** → 기본값

### 3-5. 평가 범위 (scope)
자유 입력. 기본값: `sample`. 예: `full`, `sample`, `chapter` 등.

### 3-6. 실험 태그 (tag)
선택 입력. 비워두면 스크립트가 `run_01`, `run_02` ... 자동 생성.
서버 `--log-file` 경로를 맞추려면 태그를 미리 지정하는 것이 권장됨.

기본값 제안: 직전 `run_NN` 폴더 확인 후 다음 번호 제안 (예: `run_03`).
```bash
ls evaluation/{Dataset}/results/{model_label}/{scope}/ 2>/dev/null | sort | tail -1
```

### 3-7. 테스트 파일 수 (limit, 선택)
처리할 최대 파일 수. 비워두면 scope 전체 실행.
- 예: `50`, `100` (빠른 검증 시 유용)

### 3-8. description 자동 생성
사용자에게 묻지 말고, 앞서 수집한 설정을 바탕으로 자동 생성해서 `--description`으로 전달해줘.

형식 예시:
```
finetuned(1.0.1) / sample / chunk 2.0s / Google Translate / no correction
baseline(1.0.0) / full / chunk 1.0s / GPT translation ctx=5 / correction on
```

---

## 4. 서버 실행 (SSH + tmux)

**데이터셋과 관계없이 서버 스크립트는 공통:**
`evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py`

**모델별 서버 명령:**

```bash
# baseline
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --chunk-size <n> \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"

# finetuned-merged
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --model ~/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-en-merged \
  --chunk-size <n> \
  --enforce-eager \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"

# finetuned-lora
python evaluation/LibriSpeech/servers/streaming_websocket_server_fcl.py \
  --model ~/STiTy/Qwen3-ASR/finetuning/Qwen3-ASR-1.7B-lora-ready \
  --lora \
  --chunk-size <n> \
  --enforce-eager \
  --no-idle-shutdown \
  --log-file "evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/logs/server.log"
```

태그를 지정하지 않은 경우 `--log-file` 생략.

**Step 4-1. 기존 tmux 세션 정리**
```bash
ssh <host_alias> "tmux kill-session -t eval_server 2>/dev/null; true"
```

**Step 4-2. tmux로 서버 시작**
```bash
ssh <host_alias> "cd <remote_dir> && git pull && tmux new-session -d -s eval_server '<server_command> 2>&1 | tee /tmp/eval_server.log'"
```

**Step 4-3. 포트 준비 대기 (최대 120초)**
```bash
ssh <host_alias> "timeout 120 bash -c 'until nc -z localhost 8765; do sleep 3; done && echo SERVER_READY'"
```

**에러 처리:**
timeout 만료 시 서버 로그 40줄 출력해서 원인 진단 (OOM, CUDA 에러, 포트 충돌 등):
```bash
ssh <host_alias> "tail -40 /tmp/eval_server.log"
```
수정 가능하면 Step 4-1부터 재시도, 불가능하면 사용자에게 설명 후 중단.

---

## 5. 테스트 실행

서버와 테스트 스크립트 모두 원격 서버(EC2)에서 실행. `--host localhost` 사용.
테스트도 tmux 세션(`eval_test`)으로 실행해 SSH 연결이 끊겨도 계속 돌아가게 해줘.

**Step 5-1. 기존 eval_test 세션 정리**
```bash
ssh <host_alias> "tmux kill-session -t eval_test 2>/dev/null; true"
```

**Step 5-2. tmux로 테스트 시작**

데이터셋별 명령을 tmux 세션으로 실행:

테스트 완료 후 자동으로 git push → EC2 종료까지 체이닝해줘.
tmux 명령 끝에 아래를 항상 추가:
```
&& git add -A && git commit -m "eval: <model_label> <dataset> <scope> <tag> 결과 추가" && git push && sudo shutdown -h now
```
실패해도 (`||`) shutdown은 실행되지 않도록 `&&` 체이닝으로 연결 (push 실패 시 shutdown 건너뜀).

**LibriSpeech:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  'source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && \
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
  && git push \
  && sudo shutdown -h now' \
  && echo TEST_STARTED"
```

**DailyTalk:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  'source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && \
  python evaluation/DailyTalk/test_qwen3_dailytalk.py \
    --host localhost --port 8765 --chunk-size-ms <ms> [--limit <n>] \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> DailyTalk 결과 추가\" \
  && git push \
  && sudo shutdown -h now' \
  && echo TEST_STARTED"
```

**KtelSpeech:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  'source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && \
  python evaluation/KtelSpeech/test_qwen3_ktelspeech.py \
    --data-dir evaluation/KtelSpeech \
    --model \"<model_label>\" --scope <scope> [--tag <tag>] \
    --description \"<auto_generated>\" [--limit <n>] \
    --host localhost --port 8765 --trailing-silence-ms 5000 \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> KtelSpeech <scope> <tag> 결과 추가\" \
  && git push \
  && sudo shutdown -h now' \
  && echo TEST_STARTED"
```

**AMI:**
```bash
ssh <host_alias> "cd <remote_dir> && tmux new-session -d -s eval_test \
  'source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr && \
  python evaluation/AMI/test_qwen3_ami.py \
    --ami-dir evaluation/AMI/AMI --words-dir evaluation/AMI/words \
    --model \"<model_label>\" --scope <scope> [--tag <tag>] \
    --description \"<auto_generated>\" [--limit <n>] \
    --host localhost --port 8765 --trailing-silence-ms 5000 \
  2>&1 | tee /tmp/eval_test.log \
  && git add -A \
  && git commit -m \"eval: <model_label> AMI <scope> <tag> 결과 추가\" \
  && git push \
  && sudo shutdown -h now' \
  && echo TEST_STARTED"
```

테스트는 백그라운드(tmux)에서 실행. 완료되면 자동으로 git push 후 EC2 종료.
진행 상황 확인이 필요하면 다시 `/eval-run` → "진행 중인 테스트 확인" 선택.

---

## 6. 결과 요약

결과 파일은 원격 서버에 있으므로 SSH로 읽어줘.

**태그 확인 (미지정 시):**
```bash
ssh <host_alias> "ls -t <remote_dir>/evaluation/{Dataset}/results/{model_label}/{scope}/ | head -1"
```

**metric.json 읽기:**
```bash
ssh <host_alias> "cat <remote_dir>/evaluation/{Dataset}/results/{model_label}/{scope}/{tag}/metric.json"
```

**출력 지표:**
- WER (영어) / CER (한국어)
- COMET score (있으면)
- FCL: median / mean / P90
- encode_sec / decode_sec
- SEG / VAD / finish 커밋 비율

같은 model+scope 아래 다른 run이 있으면 나란히 비교.
결과 요약 후 `/notion-summary`나 `/git-sync` 실행할지 물어봐줘.
