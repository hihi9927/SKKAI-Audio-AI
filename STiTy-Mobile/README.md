# STiTy Mobile

실시간 음성 번역 모바일 앱 — React Native 0.81 + Expo 54 (TypeScript strict, 경로 별칭 `@/*` → `src/*`).

## 실행

```bash
npm install
npm start                 # Expo 개발 서버
npm run android           # 또는 npm run ios
```

## 서버 연결

`src/context/WebSocketContext.tsx` 상단의 `RUNPOD_SERVER_URL` 상수를 환경에 맞게 수정합니다.
`getServerUrl()` 이 이 값을 쓰되, 웹에서 `EXPO_PUBLIC_USE_LOCAL_ASR=1` 이면 현재 호스트의
`/asr` 로 붙습니다.

```typescript
const RUNPOD_SERVER_URL = 'wss://<host>';   // ngrok / RunPod proxy
// LAN: 'ws://192.168.x.x:8001'
```

## 구조

```
App.tsx                          # 진입점 + 네비게이션 (등록된 화면은 HomeScreen 하나)
src/
├── screens/
│   └── HomeScreen.tsx           # 언어 · 대화 모드 선택 + 실시간 번역 피드 + 설정/약관
├── context/WebSocketContext.tsx # WS 세션 전부 — 소켓 수명주기 · 메시지 파싱 · RUNPOD_SERVER_URL
├── hooks/
│   └── useAudioRecording.ts     # 오디오 녹음 (.web.ts 웹 변형 있음)
├── utils/
│   ├── audioRouting.ts          # 대화 모드별 출력 라우팅
│   └── tts.ts                   # 음성 합성 (.web.ts 웹 변형 있음)
└── constants/
    └── languages.ts             # 지원 언어(LANGUAGES) + 대화 모드(CONVERSATION_MODES)
```

## 언어 · 대화 모드

- 지원 언어: `ko` `ja` `zh` `es` `en` (`src/constants/languages.ts`)
- 대화 모드: `mode-1` 스피커 출력 · `mode-2` 한쪽만 이어폰 · `mode-3` 양쪽 이어폰

## 색상

공용 테마 모듈은 없습니다. 언어별 말풍선·아바타 색은 `HomeScreen.tsx` 의 `LANG_COLORS`
에 인라인으로 있고(en/ko/ja/zh/es/fr/id/vi/th/de), 표에 없는 코드는 회색 `#909090` 으로
떨어집니다.

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
