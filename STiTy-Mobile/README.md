# STiTy Mobile

실시간 음성 번역 모바일 앱 - React Native (Expo)

## 화면 구성

### 1. 메인 화면 (Home)
- 나의 언어 선택
- 상대 언어 선택
- 대화 형식 선택
- 시작하기 버튼

### 2. 로딩 화면 (Loading)
- 서버 연결 상태 표시
- "로딩 중..." / "현재 이용자가 가득 차 이용이 불가합니다"
- 돌아가기 버튼

### 3. 대화 화면 (Conversation)
- 실시간 번역 결과 리스트
- 언어 레이블 (ko, id 등)
- 원문 + 번역문 표시
- 중지하기 / 재개하기 / 돌아가기 버튼

## 설치 및 실행

```bash
# 프로젝트 폴더로 이동
cd STiTy-Mobile

# 의존성 설치
npm install

# Expo 개발 서버 시작
npm start

# 또는 특정 플랫폼으로 실행
npm run android
npm run ios
```

## 서버 연결 설정

`src/hooks/useWebSocket.ts` 파일에서 서버 URL을 수정하세요:

```typescript
const SERVER_URL = 'wss://your-server-url.ngrok-free.dev/ws';
```

## 프로젝트 구조

```
STiTy-Mobile/
├── App.tsx                 # 앱 진입점 + 네비게이션
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx       # 메인 화면
│   │   ├── LoadingScreen.tsx    # 로딩 화면
│   │   └── ConversationScreen.tsx # 대화 화면
│   ├── components/
│   │   ├── GradientText.tsx     # 그라데이션 로고
│   │   ├── GradientButton.tsx   # 그라데이션 버튼
│   │   ├── LanguageSelector.tsx # 언어 선택기
│   │   └── TranslationItem.tsx  # 번역 항목
│   ├── hooks/
│   │   ├── useWebSocket.ts      # WebSocket 연결 훅
│   │   └── useAudioRecording.ts # 오디오 녹음 훅
│   └── constants/
│       ├── languages.ts         # 지원 언어 목록
│       └── theme.ts             # 테마 (색상, 폰트 등)
├── package.json
├── app.json                # Expo 설정
└── tsconfig.json
```

## 디자인 시스템

### 색상
- 보라색 (Purple): `#8B5CF6` - S, ko 레이블
- 파란색 (Blue): `#3B82F6` - Ti
- 청록색 (Cyan): `#06B6D4` - Ty, id 레이블

### 버튼 스타일
- 아웃라인 스타일 (기본)
- 그라데이션 텍스트

## 주의사항

### 실시간 오디오 스트리밍
Expo의 `expo-av`는 실시간 PCM 데이터 접근이 제한적입니다.
프로덕션 앱에서는 다음 중 하나를 고려하세요:

1. **Native Module 사용**: `react-native-audio-api` 또는 커스텀 네이티브 모듈
2. **Expo Dev Client**: 네이티브 코드 접근이 필요한 경우
3. **WebRTC 기반**: `react-native-webrtc`를 사용한 실시간 오디오 스트리밍

### 권한
- **Android**: `android.permission.RECORD_AUDIO`
- **iOS**: `NSMicrophoneUsageDescription`

## 빌드

```bash
# EAS Build (권장)
npx eas build --platform android
npx eas build --platform ios

# 로컬 빌드
npx expo prebuild
cd android && ./gradlew assembleRelease
```

## 라이선스

MIT License
