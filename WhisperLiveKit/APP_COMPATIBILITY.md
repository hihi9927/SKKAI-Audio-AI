# WhisperLiveKit + SimulStreaming App 호환성 가이드

이 문서는 SimulStreaming app을 WhisperLiveKit WebSocket 서버와 함께 사용할 때의 호환성 및 수정 사항을 설명합니다.

## 복사된 파일 목록

WhisperLiveKit의 `app/` 폴더에 다음 파일들이 SimulStreaming에서 완전히 복사되었습니다:

1. **index.html** - 메인 자막 표시 UI
2. **app.js** - 클라이언트 로직 (WebSocket, 오디오 처리)
3. **styles.css** - 메인 스타일시트
4. **settings.html** - 설정 창 UI
5. **settings.js** - 설정 창 로직
6. **settings-styles.css** - 설정 창 스타일시트

## 수정된 부분

### 1. 서버 URL 기본값 변경

**파일**: `app/app.js`, `app/settings.html`

```javascript
// 변경 전 (SimulStreaming)
SERVER_URL: 'wss://edra-raspiest-eagerly.ngrok-free.dev/ws'

// 변경 후 (WhisperLiveKit)
SERVER_URL: 'ws://localhost:8001'
```

**이유**: 로컬 개발을 위한 기본 설정

### 2. WebSocket 프로토콜 호환성

WhisperLiveKit WebSocket 서버는 SimulStreaming과 **완전히 동일한** 프로토콜을 사용합니다:

#### Client → Server 메시지

```json
// Start 메시지
{
  "type": "start",
  "lang": "auto",           // "auto", "ko", "en"
  "displayMode": "both"     // "translateOnly", "transcriptOnly", "both"
}

// Finish 메시지
{
  "type": "finish"
}

// Stop 메시지
{
  "type": "stop"
}
```

#### Server → Client 메시지

```json
// Hello 메시지
{
  "type": "hello",
  "message": "Connected to WhisperLiveKit Streaming Server"
}

// Ready 메시지
{
  "type": "ready",
  "message": "Server ready to receive audio"
}

// Partial 결과
{
  "type": "partial",
  "start": 1000,
  "end": 2000,
  "original": "현재 인식 중...",
  "last_translation": "Last translation...",
  "language": "ko"
}

// Final 결과
{
  "type": "final",
  "start": 1000,
  "end": 3000,
  "original": "완성된 문장",
  "polished": "Completed sentence",
  "language": "ko",
  "ko": "완성된 문장",
  "en": "Completed sentence"
}
```

## 기능 호환성 매트릭스

| 기능 | SimulStreaming | WhisperLiveKit | 호환성 |
|------|----------------|----------------|--------|
| WebSocket 연결 | ✅ | ✅ | ✅ 완벽 |
| 실시간 음성 인식 | ✅ | ✅ | ✅ 완벽 |
| 한국어 ↔ 영어 번역 | ✅ | ✅ | ✅ 완벽 |
| 3가지 표시 모드 | ✅ | ✅ | ✅ 완벽 |
| 언어 자동 감지 | ✅ | ✅ | ✅ 완벽 |
| Partial 결과 | ✅ | ✅ | ✅ 완벽 |
| 문장 완성 감지 | ✅ | ✅ | ✅ 완벽 |
| Space 키 녹음 토글 | ✅ | ✅ | ✅ 완벽 |
| Electron 통합 | ✅ | ✅ | ✅ 완벽 |
| 설정 창 | ✅ | ✅ | ✅ 완벽 |

## 동작 확인

### 1. 서버 시작
```bash
# WhisperLiveKit WebSocket 서버 시작
wlk-ws --host 0.0.0.0 --port 8001 --lan auto
```

### 2. 클라이언트 실행
```bash
# app 폴더에서 HTTP 서버 실행
cd app
python -m http.server 8080

# 브라우저에서 http://localhost:8080 접속
```

### 3. 확인 사항

#### ✅ 연결 확인
- 브라우저 콘솔에서 `✅ WS 연결 성공` 확인
- 그라데이션 바가 파란색으로 표시 (서버 연결 + 녹음 활성)

#### ✅ 음성 인식 확인
- 마이크에 대고 말하기
- 실시간으로 텍스트가 화면에 표시되는지 확인
- Partial 결과 (부분 결과) 표시 확인

#### ✅ 번역 확인
- 한국어로 말했을 때 영어 번역 표시 확인
- 영어로 말했을 때 한국어 번역 표시 확인

#### ✅ 표시 모드 전환
1. 패널 클릭하여 설정 창 열기
2. Display Mode 변경:
   - **Both**: 원문 + 번역 모두 표시
   - **Translation Only**: 번역만 표시
   - **Transcript Only**: 원문만 표시

#### ✅ 언어 힌트
1. 설정 창에서 Language Hint 변경:
   - **Auto Detect**: 자동 감지 (기본)
   - **English**: 영어로 강제 인식
   - **Korean**: 한국어로 강제 인식

## 주요 차이점

### SimulStreaming vs WhisperLiveKit

| 항목 | SimulStreaming | WhisperLiveKit |
|------|----------------|----------------|
| 백엔드 엔진 | SimulWhisper | WhisperLiveKit TranscriptionEngine |
| 모델 | Large-v3 | faster-whisper, mlx-whisper 등 |
| 번역 | Google Translator | Google Translator (동일) |
| 언어 지원 | 한국어, 영어 | 99개 언어 (Whisper 전체) |
| WebSocket 프로토콜 | 커스텀 | 동일한 프로토콜 (완벽 호환) |

## 트러블슈팅

### 문제 1: 서버 연결 실패
```
❌ WebSocket 연결 타임아웃
```

**해결**:
1. 서버가 실행 중인지 확인: `ps aux | grep wlk-ws`
2. 포트 확인: `netstat -an | grep 8001`
3. 방화벽 확인: Windows Defender 예외 추가

### 문제 2: 번역이 표시되지 않음
```
🟡 부분 결과: {original, lastTranslation}
last_translation이 비어있음
```

**해결**:
1. 인터넷 연결 확인 (Google Translate API 필요)
2. deep-translator 설치 확인: `pip show deep-translator`
3. 문장을 완성해서 말하기 (`.`, `!`, `?`로 끝나야 번역)

### 문제 3: 음성 인식이 안됨
```
⚠️ 이미 녹음 중
하지만 아무 텍스트도 표시되지 않음
```

**해결**:
1. 마이크 권한 확인 (브라우저 설정)
2. 오디오 청크 전송 확인 (콘솔에서 `📤 오디오 청크 전송` 메시지)
3. 서버 로그 확인 (`--log-level DEBUG` 옵션 사용)

## 성능 최적화

### 1. 오디오 청크 크기 조정

**기본값**: 500ms (8000 samples @ 16kHz)

**app.js 수정**:
```javascript
// 더 빠른 응답 (250ms)
const targetSamplesPerChunk = 4000;

// 더 안정적 (1000ms)
const targetSamplesPerChunk = 16000;
```

### 2. 서버 파라미터 튜닝

```bash
# 더 빠른 응답 (정확도 약간 감소)
wlk-ws --backend faster-whisper --beam-size 1

# 더 높은 정확도 (속도 약간 감소)
wlk-ws --backend faster-whisper --beam-size 5
```

## Electron 앱으로 패키징

SimulStreaming의 main.js를 사용하여 Electron 앱으로 패키징 가능합니다:

1. STiTy 프로젝트의 `main.js` 복사
2. `package.json` 설정
3. `npm run build` 실행

**주의**: main.js는 복사하지 않았으므로 필요시 직접 복사 필요

## 결론

SimulStreaming의 app 코드는 WhisperLiveKit WebSocket 서버와 **100% 호환**됩니다.

- ✅ 동일한 WebSocket 프로토콜
- ✅ 동일한 번역 기능
- ✅ 동일한 UI/UX
- ✅ 동일한 설정 옵션

WhisperLiveKit의 강력한 음성 인식 엔진과 SimulStreaming의 세련된 UI를 결합하여 최고의 실시간 자막 서비스를 제공합니다!
