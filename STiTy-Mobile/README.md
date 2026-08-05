# STiTy Mobile

실시간 음성 번역 모바일 앱 — React Native 0.81 + Expo 54 (TypeScript strict, 경로 별칭 `@/*` → `src/*`).

## 실행

```bash
npm install
npm start                 # Expo 개발 서버
npm run android           # 또는 npm run ios
```

## 서버 연결

`src/hooks/useWebSocket.ts` 상단의 `SERVER_URL` 상수를 환경에 맞게 수정합니다.

```typescript
const SERVER_URL = 'wss://<host>';       // ngrok / RunPod proxy
// LAN: 'ws://192.168.x.x:8001'
```

## 구조

```
App.tsx                          # 진입점 + 네비게이션
src/
├── screens/
│   ├── HomeScreen.tsx           # 언어 · 대화 모드 선택
│   └── ConversationScreen.tsx   # 실시간 번역 피드 (연결 상태도 여기서 표시)
├── context/WebSocketContext.tsx # 화면 간 공유되는 WS 세션
├── hooks/
│   ├── useWebSocket.ts          # 소켓 수명주기 · 메시지 파싱 · SERVER_URL
│   └── useAudioRecording.ts     # 오디오 녹음 (.web.ts 웹 변형 있음)
├── components/                  # GradientText · GradientButton · LanguageSelector · TranslationItem
├── utils/                       # audioRouting · serverUtils · tts (.web.ts 변형)
└── constants/
    ├── languages.ts             # 지원 언어 + 대화 모드
    └── theme.ts                 # 색상 · 타이포 토큰
```

## 언어 · 대화 모드

- 지원 언어: `ko` `ja` `zh` `es` `en` (`src/constants/languages.ts`)
- 대화 모드: `mode-1` 스피커 출력 · `mode-2` 한쪽만 이어폰 · `mode-3` 양쪽 이어폰

## 디자인 토큰

그라데이션 Purple `#8B5CF6` → Blue `#3B82F6` → Cyan `#06B6D4`. 언어 레이블 색도 같은 3색을 재사용합니다 (`theme.ts`).

## 오디오

`react-native-live-audio-stream`으로 PCM(s16le, 16 kHz, mono)을 캡처해 WebSocket 바이너리 프레임으로 전송하고, 서버는 JSON `final` 메시지로 원문+번역을 돌려줍니다. 프로토콜 상세는 루트 [CLAUDE.md](../CLAUDE.md) 참조.

Expo Go의 `expo-av`는 실시간 PCM 접근이 제한적이라 네이티브 모듈이 필요합니다 — dev client 또는 prebuild로 실행하세요.

권한: Android `RECORD_AUDIO`, iOS `NSMicrophoneUsageDescription`.

## 빌드

```bash
npx eas build --platform android      # 또는 ios
npx expo prebuild && cd android && ./gradlew assembleRelease   # 로컬 빌드
```

## 라이선스

MIT
