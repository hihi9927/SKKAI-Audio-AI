# 07_19_RunPod_배포_세팅_가이드_요약

## 📅 날짜
2026-07-19

## 🔧 작업 내용

### 1. RunPod Pod 생성 기준
- **GPU**: A40 (46GB) 사용 확인. 프로덕션 서버가 vLLM 백엔드(`from vllm import SamplingParams`)를 쓰므로 Ampere 이상 필요. Qwen3-ASR-1.7B는 16GB로도 충분, 24GB+ 권장.
- **Container Disk**: 30GB+, **Volume Disk**: 20GB (HuggingFace 캐시 `~/.cache/huggingface` 영속화용).
- **Expose Port**: `8765`를 **HTTP 프록시**로 노출 → `wss://<podid>-8765.proxy.runpod.net` 형태로 자동 TLS(wss) 지원. ngrok 불필요.
- **Base Image 선택**: `vLLM latest` 템플릿은 부팅 시 자체 vLLM API 서버가 자동 실행되어 GPU 전체(44GB)를 점유 → STiTy 서버와 충돌. **PyTorch 2.8.0** 템플릿으로 교체 결정 (vLLM 0.14는 어차피 pip으로 직접 설치하며 torch 2.9.1을 함께 끌어옴).

### 2. Pod 초기 세팅 스크립트
```bash
apt-get update && apt-get install -y ffmpeg sox libsndfile1 git

git clone --recurse-submodules https://github.com/hihi9927/STiTy.git STiTy
cd STiTy

pip install -e "./Qwen3-ASR[vllm]"
apt-get remove -y python3-blinker
pip install websockets aiohttp silero-vad

ln -s $(which python3) /usr/local/bin/python   # python 커맨드 별칭
```

### 3. HF에 올린 파인튜닝 모델 가중치 다운로드
- 계정 `Doo12`(HF)에 private repo로 merged 모델 2개가 push되어 있음:
  - `Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged`
  - `Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged`
- private repo라 `HF_TOKEN` 없이는 401로 다운로드 실패. 로컬 `~/.cache/huggingface/token` 값을 RunPod에도 그대로 사용.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx   # RunPod 콘솔 Secret으로 등록 권장, 터미널 히스토리에 노출 금지
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged \
  --local-dir /workspace/models/Qwen3-ASR-1.7B-en-silence-c80-merged

hf download Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged \
  --local-dir /workspace/models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged
```
- 로컬엔 v2/v3/v4c100 등 다른 버전 체크포인트도 있지만 HF엔 이 두 개(en-c80, ko-v4c900)만 push됨. 다른 버전이 필요하면 로컬 GB10에서 먼저 `hf upload`로 올려야 함.
- en 모델(`en-silence-c80-merged`)은 아래 4번의 **옵션 B(dualbase)**를 쓸 때만 필요. 옵션 A(단일 모델)만 쓸 거면 ko 모델만 받으면 됨.

### 4. 서버 실행

**옵션 A: 단일 ko 모델로 en/ko 모두 처리 (권장, 검증됨)**

Qwen3-ASR가 원래 멀티링구얼 베이스라 ko 모델 하나로 영어도 인식 가능. 7/19 dualbase 라우팅 실험([[dualbase-lang-routing-experiment]])에서 dualbase(en+ko 두 엔진 라우팅) 대비 언어 정확도·지연 둘 다 더 좋았고, 현재 로컬 GB10 systemd 배포도 이 구성으로 운영 중.

```bash
python Qwen3-ASR/examples/streaming_websocket_server.py \
  --model /workspace/models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged \
  --chunk-size 1.0 \
  --host 0.0.0.0 --port 8765 --no-idle-shutdown
```

| 방식 | 언어 정확도 | 지연(중앙값) |
|---|---|---|
| dualbase 라우팅 (여러 시도) | 40~89% | +225~1589ms |
| **단일 ko 모델** | **89% (16/18)** | **638ms** |

**옵션 B: dualbase (en/ko 두 엔진 분리)**

```bash
python Qwen3-ASR/examples/streaming_websocket_server_dualbase.py \
  --model /workspace/models/Qwen3-ASR-1.7B-en-silence-c80-merged \
  --model-ko /workspace/models/Qwen3-ASR-1.7B-ko-silence-v4c900-merged \
  --host 0.0.0.0 --port 8765 --no-idle-shutdown
```
- 언어 전환 순간 라우팅이 1청크 지연되는 구조적 한계 있음 ([[dualbase-lang-routing-experiment]] 참고). en 특화 모델이 한국어를 확신에 차 오인식하는 케이스도 있어 현재는 옵션 A가 더 안정적.

- `--no-idle-shutdown`: 테스트 중 유휴 자동 종료 방지.
- `--disable-dot-commit --no-vad`는 안 켜도 됨(dot-commit/VAD 켜둬도 문제없음).
- 세션 끊김에도 서버가 죽지 않도록 `tmux new -s asr` 안에서 실행 권장.

### 5. 모바일 앱 서버 주소 연결
- `STiTy-Mobile/src/hooks/useWebSocket.ts:22`의 `SERVER_URL` 하드코딩 상수를 RunPod 프록시 주소로 교체:
  ```ts
  const SERVER_URL = 'wss://kg7hmbaupwv7e8-8765.proxy.runpod.net';
  ```
- Expo 개발 모드(`npm start`)면 저장만으로 핫 리로드. EAS로 만든 독립 APK/IPA는 URL이 빌드에 고정되므로 재빌드 또는 EAS Update 필요.

**참고: Expo 터널 URL**
```
https://m-b8iew-anonymous-8081.exp.direct
```
- `npx expo start --tunnel`(또는 web 모드에서 터널 옵션) 실행 시 Expo CLI가 발급하는 임시 공개 HTTPS 주소. 로컬 `localhost:8081` Metro 서버를 외부에서 접속 가능하도록 프록시([[mobile-web-build]] 참고).
- `getUserMedia`(마이크 녹음)는 secure context(HTTPS)가 필요한데, LAN IP로는 HTTP라 막힘 → 이 터널 주소를 쓰면 실기기/외부 브라우저에서도 마이크 권한이 정상 동작.
- RunPod 프록시(`wss://<podid>-8765.proxy.runpod.net`)와는 별개: 이건 **모바일 웹 프론트엔드**(Expo web) 접속용 주소이고, RunPod 쪽은 **ASR/번역 WebSocket 백엔드** 주소.
- Expo CLI가 세션마다 새로 발급하므로 재시작하면 서브도메인(`m-b8iew-anonymous`)이 바뀔 수 있어 고정 주소로 쓸 수 없음 — 임시 테스트/공유용.

### 6. 앱 없이 파이프라인 검증
- 레포에 이미 있는 진단 클라이언트 `evaluation/KsponSpeech/send_one.py` 사용 — 오디오 파일 하나를 서버에 보내고 ASR 원문 + 번역 결과를 콘솔에 출력.
- 테스트용 한국어 PCM 샘플이 git으로 이미 추적됨: `evaluation/KsponSpeech/sample_data/KsponSpeech_0001/*.pcm`.
```bash
python evaluation/KsponSpeech/send_one.py \
  --audio evaluation/KsponSpeech/sample_data/KsponSpeech_0001/KsponSpeech_000001.pcm \
  --host localhost --port 8765 --target-lang en
```

## 🐛 발견된 문제 및 해결

| 문제 | 원인 | 해결 |
|---|---|---|
| `Cannot uninstall blinker 1.4` (pip 설치 중단) | `blinker`가 apt/distutils로 시스템 설치되어 pip이 소유 파일을 특정 못 함 | `apt-get remove -y python3-blinker` 후 재설치, 또는 `--ignore-installed blinker` |
| `lmcache 0.5.1 requires huggingface_hub>=1.5.0 ...` 경고 | vLLM 0.14 설치 시 버전 다운그레이드로 인한 pip 의존성 경고 | 실제 설치는 정상 완료됨 (`Successfully installed`). `lmcache`는 프로젝트에서 미사용 기능이라 무시 가능 |
| `python: command not found` | vLLM/PyTorch 이미지에 `python3`만 있고 `python` 심볼릭 링크 없음 | `ln -s $(which python3) /usr/local/bin/python` |
| `Free memory on device cuda:0 (1.11/44.42 GiB)` 로 엔진 기동 실패 | RunPod `vLLM latest` 템플릿이 부팅 시 자동 실행한 vLLM 서버(PID 549)가 GPU 44GB 선점 | 자동 실행 프로세스 kill 시도 → 컨테이너 메인 프로세스라 연결 종료됨 → 근본 해결로 **PyTorch 템플릿으로 Pod 재생성** 결정 |
| `ModuleNotFoundError: No module named 'hf_transfer'` | PyTorch 이미지에 `HF_HUB_ENABLE_HF_TRANSFER=1`이 기본 설정되어 있으나 해당 패키지 미설치 | `pip install hf_transfer` (권장) 또는 `export HF_HUB_ENABLE_HF_TRANSFER=0` |
| Android 에뮬레이터 실행 실패 (`No Android connected device found`) | 로컬에 연결된 실기기/에뮬레이터 없음 | 실기기 USB 디버깅 연결 또는 EAS 클라우드 빌드(`eas build --profile preview`)로 APK 생성 후 직접 설치 |