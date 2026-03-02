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
// Zoom 등 통신 앱과 마이크 공유: audioSource 7 = VOICE_COMMUNICATION
// (기존 6 = VOICE_RECOGNITION은 Zoom이 독점 점유 시 차단됨)
const AUDIO_SOURCE = 7;
// 스트림 시작 후 데이터가 없으면 재시도하는 watchdog 시간 (ms)
const WATCHDOG_TIMEOUT_MS = 4000;

// ===== 모듈 레벨 싱글톤 상태 =====
let _nativeInitialized = false;
let _streamRunning = false;
let _activeCallback: ((base64Data: string) => void) | null = null;
let _watchdogTimer: ReturnType<typeof setTimeout> | null = null;
let _dataReceivedAfterStart = false;

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

const clearWatchdog = () => {
  if (_watchdogTimer !== null) {
    clearTimeout(_watchdogTimer);
    _watchdogTimer = null;
  }
};

// native 스트림 완전 정리 (강제 재초기화용)
const forceCleanupNative = () => {
  try {
    LiveAudioStream.stop();
  } catch (e) { /* ignore */ }
  try {
    (LiveAudioStream as any).removeAllListeners?.('data');
  } catch (e) { /* ignore */ }
  _activeCallback = null;
  _nativeInitialized = false;
  _streamRunning = false;
};

export const useAudioRecording = ({
  onAudioData,
  sampleRate = SAMPLE_RATE,
  bufferSize = BUFFER_SIZE,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onAudioDataRef = useRef(onAudioData);
  const mountedRef = useRef(true);

  useEffect(() => {
    onAudioDataRef.current = onAudioData;
  }, [onAudioData]);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

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
    try {
      (LiveAudioStream as any).removeAllListeners?.('data');
    } catch (e) { /* ignore */ }

    if (!_nativeInitialized) {
      const tempWavPath = `${FileSystem.cacheDirectory}temp_audio.wav`;
      LiveAudioStream.init({
        sampleRate: sampleRate,
        channels: 1,
        bitsPerSample: 16,
        audioSource: AUDIO_SOURCE,
        bufferSize: bufferSize,
        wavFile: tempWavPath,
      });
      _nativeInitialized = true;
      console.log(`LiveAudioStream native init (sampleRate: ${sampleRate}, audioSource: ${AUDIO_SOURCE})`);
    }

    _activeCallback = (base64Data: string) => {
      try {
        const audioBuffer = base64ToArrayBuffer(base64Data);
        if (audioBuffer.byteLength > 0) {
          _dataReceivedAfterStart = true;
          onAudioDataRef.current(audioBuffer);
        }
      } catch (e) {
        console.error('Error processing audio data:', e);
      }
    };
    LiveAudioStream.on('data', _activeCallback);
  };

  // Zoom 등으로 마이크가 차단된 경우 강제 재시도
  const retryAfterWatchdog = useCallback(async () => {
    if (!mountedRef.current) return;
    console.log('Watchdog: 오디오 데이터 없음 - 스트림 재시작 시도 (Zoom 마이크 충돌 가능성)');
    forceCleanupNative();
    // 짧은 대기 후 재초기화 (Zoom이 마이크를 잠시 놓아줄 수 있도록)
    await new Promise(resolve => setTimeout(resolve, 800));
    if (!mountedRef.current) return;
    try {
      _dataReceivedAfterStart = false;
      initializeStream();
      LiveAudioStream.start();
      _streamRunning = true;
      console.log('LiveAudioStream 재시작 완료');
      // 재시도 후에도 데이터가 없으면 에러 표시
      _watchdogTimer = setTimeout(() => {
        if (!_dataReceivedAfterStart && mountedRef.current) {
          console.warn('Watchdog: 재시도 후에도 오디오 없음 - Zoom이 마이크를 독점 중일 수 있음');
          if (mountedRef.current) {
            setError('마이크를 사용할 수 없습니다. Zoom 등 다른 앱이 마이크를 점유 중일 수 있습니다.');
          }
        }
      }, WATCHDOG_TIMEOUT_MS);
    } catch (e) {
      console.error('스트림 재시작 실패:', e);
    }
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        throw new Error('No microphone permission');
      }

      clearWatchdog();
      _dataReceivedAfterStart = false;
      initializeStream();

      if (!_streamRunning) {
        LiveAudioStream.start();
        _streamRunning = true;
        console.log('LiveAudioStream.start() called');
      } else {
        console.log('LiveAudioStream already running, listener swapped');
      }

      // Watchdog: 일정 시간 후 데이터가 없으면 재시도 (Zoom 마이크 충돌 대응)
      _watchdogTimer = setTimeout(() => {
        if (!_dataReceivedAfterStart) {
          retryAfterWatchdog();
        }
      }, WATCHDOG_TIMEOUT_MS);

      setIsRecordingActive(true);
      setError(null);
    } catch (e: any) {
      console.error('Failed to start recording:', e);
      setError(e.message || '녹음 시작 실패');
      setIsRecordingActive(false);
      throw e;
    }
  }, [retryAfterWatchdog]);

  const stopRecording = useCallback(async (): Promise<void> => {
    clearWatchdog();
    try {
      LiveAudioStream.stop();
      try {
        (LiveAudioStream as any).removeAllListeners?.('data');
      } catch (e) { /* ignore */ }
      _activeCallback = null;
      _nativeInitialized = false;
      _streamRunning = false;
      _dataReceivedAfterStart = false;
      setIsRecordingActive(false);
      console.log('LiveAudioStream fully stopped and cleaned up');
    } catch (e) {
      console.error('Error stopping recording:', e);
    }
  }, []);

  return {
    isRecordingActive,
    startRecording,
    stopRecording,
    error,
  };
};

export default useAudioRecording;
