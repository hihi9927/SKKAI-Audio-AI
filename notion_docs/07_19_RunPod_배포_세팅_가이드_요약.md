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

git clone --recurse-submodules <REPO_URL> STiTy
cd STiTy

pip install -e "./Qwen3-ASR[vllm]"
pip install websockets aiohttp silero-vad

ln -s $(which python3) /usr/local/bin/python   # python 커맨드 별칭
```

### 3. 서버 실행
```bash
python Qwen3-ASR/examples/streaming_websocket_server.py \
  --host 0.0.0.0 --port 8765 --no-idle-shutdown
```
- `--no-idle-shutdown`: 테스트 중 유휴 자동 종료 방지.
- 세션 끊김에도 서버가 죽지 않도록 `tmux new -s asr` 안에서 실행 권장.

### 4. 모바일 앱 서버 주소 연결
- `STiTy-Mobile/src/hooks/useWebSocket.ts:22`의 `SERVER_URL` 하드코딩 상수를 RunPod 프록시 주소로 교체:
  ```ts
  const SERVER_URL = 'wss://2ru6q7iwdkz4d5-8765.proxy.runpod.net';
  ```
- Expo 개발 모드(`npm start`)면 저장만으로 핫 리로드. EAS로 만든 독립 APK/IPA는 URL이 빌드에 고정되므로 재빌드 또는 EAS Update 필요.

### 5. 앱 없이 파이프라인 검증
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

## ⏭ 해결되지 않은 작업

- **Pod ID 변경 시 APK 무효화**: 현재 방식은 RunPod 프록시 주소(`<podid>-8765.proxy.runpod.net`)가 앱 빌드에 고정됨. Pod를 삭제/재생성하면 ID가 바뀌어 기존 APK가 연결 실패함 → 서버 주소를 앱 내에서 런타임 입력받는 방식으로 개선하거나, Pod를 계속 유지하는 운영 방식 필요.
- **빌드된 APK의 서버 주소 교체 가능 여부**: EAS로 이미 구운 APK에서 URL만 바꾸는 방법을 논의하던 중 세션 중단됨 — 다음 세션에서 이어서 확인 필요 (일반적으로 JS 번들에 값이 굳어 있어 재빌드나 OTA(EAS Update)가 필요함).
- **RunPod 자동 vLLM 서버 재발 방지**: 현재는 이미지를 PyTorch로 바꿔서 회피했으나, 만약 다시 vLLM 계열 템플릿을 쓸 경우 Container Start Command를 `sleep infinity`로 override하는 방법을 아직 실제 적용/검증하지 않음.
- **실기기 마이크 스트리밍 검증 미완료**: `send_one.py`로 서버-파이프라인 자체는 확인 가능하지만, 실제 모바일 앱에서 `react-native-live-audio-stream` 기반 실시간 오디오 캡처 → WebSocket 전송까지는 아직 실기기/APK로 테스트하지 못함.
