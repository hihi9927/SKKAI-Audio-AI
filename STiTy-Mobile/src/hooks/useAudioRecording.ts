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
// 웹앱과 동일하게 연속 스트리밍 - 100ms마다 데이터 전송
// bufferSize = sampleRate * (ms / 1000) * bytesPerSample
// 16000 * 0.1 * 2 = 3200 bytes per 100ms
const BUFFER_SIZE = 3200;

export const useAudioRecording = ({
  onAudioData,
  sampleRate = SAMPLE_RATE,
  bufferSize = BUFFER_SIZE,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isInitializedRef = useRef(false);
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
    // iOS는 Info.plist에서 처리됨
    return true;
  };

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

  // LiveAudioStream 초기화
  const initializeStream = () => {
    if (isInitializedRef.current) return;

    // wavFile은 필수 옵션이지만 실시간 스트리밍에서는 사용하지 않음
    const tempWavPath = `${FileSystem.cacheDirectory}temp_audio.wav`;

    LiveAudioStream.init({
      sampleRate: sampleRate,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6, // VOICE_RECOGNITION (Android) - 노이즈 캔슬링 적용
      bufferSize: bufferSize,
      wavFile: tempWavPath,
    });

    // 오디오 데이터 이벤트 리스너
    LiveAudioStream.on('data', (base64Data: string) => {
      try {
        const audioBuffer = base64ToArrayBuffer(base64Data);
        if (audioBuffer.byteLength > 0) {
          // 웹앱과 동일하게 Int16 PCM을 직접 전송
          onAudioDataRef.current(audioBuffer);
        }
      } catch (e) {
        console.error('Error processing audio data:', e);
      }
    });

    isInitializedRef.current = true;
    console.log(`LiveAudioStream initialized (sampleRate: ${sampleRate}, bufferSize: ${bufferSize})`);
  };

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        throw new Error('No microphone permission');
      }

      initializeStream();
      LiveAudioStream.start();

      setIsRecordingActive(true);
      setError(null);
      console.log('LiveAudioStream recording started (continuous streaming mode)');

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
      setIsRecordingActive(false);
      console.log('LiveAudioStream recording stopped');
    } catch (e) {
      console.error('Error stopping recording:', e);
    }
  }, []);

  useEffect(() => {
    return () => {
      // 컴포넌트 언마운트 시 정리
      if (isRecordingActive) {
        LiveAudioStream.stop();
      }
    };
  }, [isRecordingActive]);

  return {
    isRecordingActive,
    startRecording,
    stopRecording,
    error,
  };
};

export default useAudioRecording;
