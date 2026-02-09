import { useState, useCallback, useRef, useEffect } from 'react';
import { Platform, PermissionsAndroid } from 'react-native';
import LiveAudioStream from 'react-native-live-audio-stream';
import * as FileSystem from 'expo-file-system/legacy';

interface UseAudioRecordingOptions {
  onAudioData: (audioData: ArrayBuffer) => void;
  sampleRate?: number;
  bufferSize?: number;
}

interface UseAudioRecordingReturn {
  isRecordingActive: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  error: string | null;
}

const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 3200;

// ===== 모듈 레벨 싱글톤 상태 =====
// LiveAudioStream은 글로벌 싱글톤이므로 훅 인스턴스가 아닌 모듈 레벨에서 관리
let _nativeInitialized = false;  // native stream init 여부
let _streamRunning = false;      // start() 호출 여부

// 현재 활성 콜백 - 화면 전환 시 새 훅 인스턴스의 콜백으로 교체됨
let _activeCallback: ((base64Data: string) => void) | null = null;

// base64 → ArrayBuffer 변환
const base64ToArrayBuffer = (base64: string): ArrayBuffer => {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const buffer = new ArrayBuffer(len);
  const view = new Uint8Array(buffer);
  for (let i = 0; i < len; i++) {
    view[i] = binaryString.charCodeAt(i);
  }
  return buffer;
};

export const useAudioRecording = ({
  onAudioData,
  sampleRate = SAMPLE_RATE,
  bufferSize = BUFFER_SIZE,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onAudioDataRef = useRef(onAudioData);

  useEffect(() => {
    onAudioDataRef.current = onAudioData;
  }, [onAudioData]);

  // Android 권한 요청
  const requestPermissions = async (): Promise<boolean> => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: '마이크 권한',
            message: '실시간 음성 번역을 위해 마이크 권한이 필요합니다.',
            buttonPositive: '허용',
            buttonNegative: '거부',
          }
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          setError('마이크 권한이 필요합니다');
          return false;
        }
        return true;
      } catch (e) {
        console.error('Failed to request permissions:', e);
        setError('권한 요청 실패');
        return false;
      }
    }
    return true;
  };

  // LiveAudioStream 초기화 (싱글톤)
  const initializeStream = () => {
    // 이전 data 리스너 모두 제거 (화면 전환 시 중복 방지)
    try {
      LiveAudioStream.removeAllListeners?.('data');
    } catch (e) {
      // removeAllListeners 미지원 시 무시
    }

    // native stream은 한 번만 init
    if (!_nativeInitialized) {
      const tempWavPath = `${FileSystem.cacheDirectory}temp_audio.wav`;
      LiveAudioStream.init({
        sampleRate: sampleRate,
        channels: 1,
        bitsPerSample: 16,
        audioSource: 6,
        bufferSize: bufferSize,
        wavFile: tempWavPath,
      });
      _nativeInitialized = true;
      console.log(`LiveAudioStream native init (sampleRate: ${sampleRate}, bufferSize: ${bufferSize})`);
    }

    // 새 data 리스너 등록 (현재 훅 인스턴스의 콜백 사용)
    _activeCallback = (base64Data: string) => {
      try {
        const audioBuffer = base64ToArrayBuffer(base64Data);
        if (audioBuffer.byteLength > 0) {
          onAudioDataRef.current(audioBuffer);
        }
      } catch (e) {
        console.error('Error processing audio data:', e);
      }
    };
    LiveAudioStream.on('data', _activeCallback);
  };

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        throw new Error('No microphone permission');
      }

      initializeStream();

      // 이미 실행 중이면 start() 생략 (화면 전환 시 스트림 끊김 방지)
      if (!_streamRunning) {
        LiveAudioStream.start();
        _streamRunning = true;
        console.log('LiveAudioStream.start() called');
      } else {
        console.log('LiveAudioStream already running, listener swapped');
      }

      setIsRecordingActive(true);
      setError(null);
    } catch (e: any) {
      console.error('Failed to start recording:', e);
      setError(e.message || '녹음 시작 실패');
      setIsRecordingActive(false);
      throw e;
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<void> => {
    try {
      LiveAudioStream.stop();
      try {
        LiveAudioStream.removeAllListeners?.('data');
      } catch (e) { /* ignore */ }
      _activeCallback = null;
      _nativeInitialized = false;
      _streamRunning = false;
      setIsRecordingActive(false);
      console.log('LiveAudioStream fully stopped and cleaned up');
    } catch (e) {
      console.error('Error stopping recording:', e);
    }
  }, []);

  // auto-cleanup 제거: 화면 전환 시 싱글톤 스트림을 죽이지 않도록
  // 각 화면이 명시적으로 stopRecording()을 호출해야 함

  return {
    isRecordingActive,
    startRecording,
    stopRecording,
    error,
  };
};

export default useAudioRecording;
