import { NativeModules, Platform } from 'react-native';

// MODE_IN_COMMUNICATION 상태에서 오디오 출력 경로를 제어한다.
// true  → 스피커 (이어폰 연결 여부 무관)
// false → 이어폰/헤드폰 (미연결 시 수화기)
export const setSpeakerphoneOn = (value: boolean): void => {
  if (Platform.OS === 'android') {
    NativeModules.RNLiveAudioStream?.setSpeakerphoneOn(value);
  }
};

// 유선/블루투스 이어폰·헤드폰 연결 여부 반환
export const isEarphoneConnected = (): Promise<boolean> => {
  if (Platform.OS === 'android') {
    return NativeModules.RNLiveAudioStream?.isEarphoneConnected() ?? Promise.resolve(false);
  }
  return Promise.resolve(false);
};

// 화면 완전 종료 시 호출 — 오디오 모드와 라우팅을 시스템 기본값으로 복원
export const releaseAudioMode = (): void => {
  if (Platform.OS === 'android') {
    NativeModules.RNLiveAudioStream?.releaseAudioMode();
  }
};
