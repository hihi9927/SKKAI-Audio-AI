# Phase 1: HTTP API 기반 STT 서버 테스트 가이드

## 📋 구현 현황

✅ **서버 코드**: [server.py](server.py) - Flask 기반 HTTP API (포트: 8000) + ngrok 이용 (다른 LAN에서도 접속 가능)
✅ **클라이언트 코드**: [client.py](client.py) - 음성 녹음 + 서버 전송
✅ **웹사이트 구동**: [client.html](client.html) - 음성 녹음 + 서버 전송
✅ **앱 실행**: [STT자막.exe](STT자막.exe) - 음성 녹음 + 서버 전송

## 🚀 실행 방법

### 1단계: 필수 패키지 설치

```bash
# 기본 패키지 (이미 설치되어 있음)
pip install torch openai-whisper librosa deep-translator

# 추가 패키지 (Flask, pyaudio)
pip install flask requests pyaudio
```

**macOS에서 pyaudio 설치 시 에러 발생하면:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows에서 pyaudio 설치 시 에러 발생하면:**
```bash
pip install pipwin
pipwin install pyaudio
```

---

### 2-1단계: 서버 실행 (팀 컴퓨터)

```bash
# 로컬 서버 시작
python server.py
```

**정상 실행 시 출력:**
```
🤖 Whisper 모델 로딩 중...
✅ 모델 로드 완료 (device=cpu)
============================================================
🚀 STT 서버 시작
============================================================
📡 접속 주소: http://0.0.0.0:8000
🔗 Health Check: http://0.0.0.0:8000/health
🎤 STT 엔드포인트: http://0.0.0.0:8000/stt
============================================================
 * Serving Flask app 'server'
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://192.168.0.XXX:8000
```

### 2-2단계: ngrok 실행 (팀 컴퓨터)

```bash
# ngrok 서버 시작
ngrok http 8000
```

**📌 중요:** 
로컬 서버 접속시 `http://192.168.0.XXX:8000` 주소를 메모해두세요! (클라이언트에서 사용)

외부에서 연결시 (ngrok 서버로 접속)
'https://abc123.ngrok-free.app/stt' ngrok 활성화 주소를 메모해두세요!
---

### 3단계: 클라이언트 실행 (다른 컴퓨터/휴대폰)

#### 옵션 1: 같은 컴퓨터에서 테스트 (localhost)

```bash
# 5초 녹음 후 전송
python client.py 

# 10초 녹음
python client.py --duration 10
```

#### 옵션 2: 다른 컴퓨터에서 테스트 (타 LAN에서 접속할 경우)

```bash
# 서버 주소를 직접 지정
python client.py --server 'https://abc123.ngrok-free.app/stt'

# 10초 녹음
python client.py --server 'https://abc123.ngrok-free.app/stt' --duration 10
```

#### 옵션 3: 기존 오디오 파일 전송 (녹음 건너뛰기)

```bash
# audio_data 폴더의 파일 전송
python client.py --audio audio_data/test.wav --server http://192.168.0.10:8000/stt
```

---

#### 옵션 4: bash가 아닌 웹 사이트 or 앱에서 실행하고 싶을 때
client.html/ STT자막.exe 실행 후 서버 주소 란에
'https://abc123.ngrok-free.app/stt'를 입력해주세요.

#### MAC에서 실행시 exe파일이 실행되지 않으므로 따로 빌드해야함
pip install pyinstaller pywebview

pyinstaller --onefile --windowed --name="STT자막" app.py

## 🎯 실행 결과 예시

### 클라이언트 출력:

```
============================================================
🎙️  STT 클라이언트
============================================================
✅ 서버 연결 성공
   모델: base
   디바이스: cpu
🎤 녹음 시작... (5초)
   1초...
   2초...
   3초...
   4초...
   5초...
✅ 녹음 완료
📤 서버로 전송 중: http://192.168.0.10:8000/stt

============================================================
✅ STT 처리 완료
============================================================
🌍 감지 언어: ko
📝 원문: 안녕하세요. 테스트 음성입니다.
🌐 번역: Hello. This is a test voice.
============================================================

🧹 임시 파일 삭제: temp_recording.wav
```

### 서버 출력:

```
📥 오디오 파일 수신: temp_recording.wav
🎤 음성 인식 처리 중...
🌍 감지 언어: ko
📝 인식 결과: 안녕하세요. 테스트 음성입니다.
🌐 번역 완료: Hello. This is a test voice.
127.0.0.1 - - [15/Oct/2025 14:30:45] "POST /stt HTTP/1.1" 200 -
```

---

## ⚠️ 문제 해결

### 0. "Address already in use" (포트 사용 중) 에러

**원인:** macOS에서 포트 5000이 AirPlay Receiver에 의해 사용되고 있습니다.

**해결:**
- ✅ 이미 코드가 포트 8000으로 변경되어 있음
- 또는 AirPlay Receiver 비활성화: 시스템 환경설정 → 일반 → AirDrop 및 Handoff → "AirPlay 수신기" 끄기

### 1. "서버 연결 실패" 에러

**원인:** 서버가 실행 중이 아니거나 주소가 틀렸습니다.

**해결:**
- 서버가 실행 중인지 확인: `python server.py`
- 서버 IP 주소가 맞는지 확인
- 방화벽 설정 확인 (8000 포트 열기)

### 2. "No audio file provided" 에러

**원인:** 클라이언트에서 오디오 파일을 제대로 전송하지 못했습니다.

**해결:**
- 마이크 권한 확인 (macOS: 시스템 환경설정 → 보안 및 개인정보보호)
- pyaudio가 제대로 설치되었는지 확인

### 3. 번역 실패 "(번역 실패)"

**원인:** 인터넷 연결 문제 (GoogleTranslator는 인터넷 필요)

**해결:**
- 인터넷 연결 확인
- deep-translator 재설치: `pip install --upgrade deep-translator`

### 4. 모델 로딩이 너무 느림

**해결:**
- 더 작은 모델 사용: [server.py:21](server.py#L21)에서 `"base"` → `"tiny"` 변경
- GPU 사용 (CUDA 사용 가능한 경우 자동으로 GPU 사용)

### 5. macOS에서 방화벽 설정

```
시스템 환경설정 → 보안 및 개인정보보호 → 방화벽 → 방화벽 옵션
→ "Python"에 대한 수신 연결 허용
```

### 6. Windows에서 방화벽 설정

```
제어판 → Windows Defender 방화벽 → 고급 설정
→ 인바운드 규칙 → 새 규칙 → 포트 → TCP 8000 허용
```

---

## 🔍 서버 상태 확인

웹 브라우저나 curl로 서버 상태를 확인할 수 있습니다:

```bash
# 브라우저에서 접속
http://서버IP:8000/health

# 또는 터미널에서
curl http://서버IP:8000/health
```

**정상 응답:**
```json
{
  "status": "ok",
  "model": "base",
  "device": "cpu"
}
```

---

## 📱 휴대폰에서 테스트 (고급)

Python이 설치된 휴대폰은 드물기 때문에, 다음 단계로 **웹 기반 클라이언트**를 만들면 휴대폰 브라우저에서 바로 사용할 수 있습니다.

### 간단한 테스트 방법:

1. **Postman** 또는 **HTTP 테스트 앱** 사용
2. 휴대폰에서 오디오 녹음 후 서버로 POST 요청
3. 응답 JSON 확인

---

## 🎓 주요 명령어 정리

```bash
# 서버 실행 (팀 컴퓨터)
python server.py

# 클라이언트 실행 (기본 - 5초 녹음)
python client.py

# 클라이언트 실행 (서버 주소 지정)
python client.py --server http://192.168.0.10:8000/stt

# 클라이언트 실행 (10초 녹음)
python client.py --duration 10

# 클라이언트 실행 (기존 파일 전송)
python client.py --audio audio_data/test.wav --server http://192.168.0.10:8000/stt

# 서버 IP 확인 (macOS/Linux)
ifconfig | grep "inet "

# 서버 IP 확인 (Windows)
ipconfig
```

---

## 📊 Phase 1 검증 체크리스트

- [ ] 서버가 정상적으로 실행됨
- [ ] 클라이언트에서 마이크 녹음이 됨
- [ ] 같은 컴퓨터에서 테스트 성공 (localhost)
- [ ] 다른 컴퓨터에서 테스트 성공 (로컬 네트워크)
- [ ] 한국어 → 영어 번역 정상 작동
- [ ] 영어 → 한국어 번역 정상 작동
- [ ] 에러 핸들링 확인 (잘못된 파일, 서버 중단 등)

---

## 🚀 다음 단계: Phase 2

Phase 1이 정상 작동하면, 다음은 **WebSocket 기반 실시간 스트리밍**을 구현합니다.

**Phase 2에서 구현할 것:**
- 실시간 오디오 스트리밍 (녹음하면서 동시에 전송)
- 부분 결과 실시간 출력 (interim results)
- 더 낮은 지연시간 (latency)

---

## 💡 팁

1. **처음에는 같은 컴퓨터에서 테스트**하여 전체 흐름을 확인
2. **로그를 잘 확인**하면 문제 해결이 쉬움
3. **작은 모델(tiny)로 시작**해서 빠르게 테스트
4. **서버는 한 번만 실행**하고, 클라이언트를 여러 번 실행 가능
5. **Ctrl+C**로 서버 종료 가능

---

## 📞 추가 도움이 필요하면

에러 메시지와 함께 다음 정보를 제공하세요:
- 운영체제 (macOS/Windows/Linux)
- Python 버전: `python --version`
- 실행한 명령어
- 서버/클라이언트 출력 로그
