import { useState, useCallback, useRef, useEffect } from 'react';
import { Platform } from 'react-native';  // 리소스 해제 대기 시간용
import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';

interface UseAudioRecordingOptions {
  onAudioData: (audioData: ArrayBuffer) => void;
  sampleRate?: number;
  chunkDurationMs?: number;
}

interface UseAudioRecordingReturn {
  isRecordingActive: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  error: string | null;
}

const SAMPLE_RATE = 16000;
// 웹앱은 100ms 연속 스트리밍이지만, Expo는 stop-start 방식이라 갭이 발생
// 청크를 작게 해서 웹앱에 가깝게 만듦 (500ms)
// AAC 프라이밍 딜레이(~128ms)는 ffmpeg atrim으로 제거
const CHUNK_DURATION_MS = 500;
const WAV_HEADER_SIZE = 44;

export const useAudioRecording = ({
  onAudioData,
  sampleRate = SAMPLE_RATE,
  chunkDurationMs = CHUNK_DURATION_MS,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isRunningRef = useRef(false);
  const onAudioDataRef = useRef(onAudioData);

  useEffect(() => {
    onAudioDataRef.current = onAudioData;
  }, [onAudioData]);

  const requestPermissions = async (): Promise<boolean> => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        setError('마이크 권한이 필요합니다');
        return false;
      }
      return true;
    } catch (e) {
      console.error('Failed to request permissions:', e);
      setError('권한 요청 실패');
      return false;
    }
  };

  const configureAudio = async (): Promise<void> => {
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
      staysActiveInBackground: true,
      shouldDuckAndroid: true,
      playThroughEarpieceAndroid: false,
    });
  };

  // WAV 헤더 스킵 후 Int16 PCM 데이터 추출
  const extractPCMFromWav = (base64Data: string): ArrayBuffer | null => {
    try {
      const binaryString = atob(base64Data);
      const totalBytes = binaryString.length;

      if (totalBytes <= WAV_HEADER_SIZE) {
        return null;
      }

      const pcmLength = totalBytes - WAV_HEADER_SIZE;
      const buffer = new ArrayBuffer(pcmLength);
      const view = new Uint8Array(buffer);

      for (let i = 0; i < pcmLength; i++) {
        view[i] = binaryString.charCodeAt(WAV_HEADER_SIZE + i);
      }

      return buffer;
    } catch (e) {
      console.error('Failed to extract PCM:', e);
      return null;
    }
  };

  // base64 → ArrayBuffer 변환 (Android AAC 전송용)
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

  // 플랫폼별 녹음 옵션
  const recordingOptions: Audio.RecordingOptions = {
    android: {
      extension: '.aac',
      outputFormat: 6,  // AAC_ADTS
      audioEncoder: 3,  // AAC
      sampleRate,
      numberOfChannels: 1,
      bitRate: 128000,
    },
    ios: {
      extension: '.wav',
      outputFormat: Audio.IOSOutputFormat.LINEARPCM,
      audioQuality: Audio.IOSAudioQuality.HIGH,
      sampleRate,
      numberOfChannels: 1,
      bitRate: sampleRate * 16,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
    web: {
      mimeType: 'audio/webm',
      bitsPerSecond: sampleRate * 16,
    },
  };

  // 녹음 생성 (재시도 로직 포함)
  const createRecordingWithRetry = async (maxRetries = 3): Promise<Audio.Recording | null> => {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const { recording } = await Audio.Recording.createAsync(recordingOptions);
        return recording;
      } catch (e) {
        console.warn(`Recording create attempt ${attempt + 1} failed:`, e);
        if (attempt < maxRetries - 1) {
          await configureAudio();
          await new Promise<void>((resolve) => setTimeout(resolve, 300 * (attempt + 1)));
        }
      }
    }
    return null;
  };

  // 녹음 → 정지 → 데이터 추출 → 전송 반복 루프
  const recordingLoop = async () => {
    console.log(`Recording loop started (platform: ${Platform.OS})`);

    while (isRunningRef.current) {
      let recording: Audio.Recording | null = null;
      try {
        recording = await createRecordingWithRetry();
        if (!recording) {
          console.error('Failed to create recording after retries');
          await new Promise<void>((resolve) => setTimeout(resolve, 1000));
          continue;
        }

        // chunkDurationMs 만큼 녹음
        await new Promise<void>((resolve) => setTimeout(resolve, chunkDurationMs));

        // 정지 및 해제
        await recording.stopAndUnloadAsync();

        if (!isRunningRef.current) break;

        const uri = recording.getURI();
        if (uri) {
          const base64 = await FileSystem.readAsStringAsync(uri, {
            encoding: 'base64' as any,
          });

          // 플랫폼별 데이터 추출:
          // iOS: WAV → PCM (44바이트 헤더 스킵)
          // Android: AAC 파일 전체를 그대로 전송 (서버에서 ffmpeg 디코딩)
          const audioBuffer = Platform.OS === 'ios'
            ? extractPCMFromWav(base64)
            : base64ToArrayBuffer(base64);

          if (audioBuffer && audioBuffer.byteLength > 0) {
            console.log(`Sending audio: ${audioBuffer.byteLength} bytes (${Platform.OS})`);
            onAudioDataRef.current(audioBuffer);
          }

          await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
        }

        // 리소스 해제 대기 - Android에서 최소화하여 오디오 갭 줄임
        await new Promise<void>((resolve) => setTimeout(resolve, Platform.OS === 'android' ? 50 : 50));

      } catch (e) {
        console.error('Recording loop error:', e);
        if (recording) {
          try { await recording.stopAndUnloadAsync(); } catch (_) {}
        }
        await new Promise<void>((resolve) => setTimeout(resolve, 1000));
      }
    }

    console.log('Recording loop ended');
  };

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        throw new Error('No microphone permission');
      }

      await configureAudio();

      isRunningRef.current = true;
      setIsRecordingActive(true);
      setError(null);

      console.log('Recording started (chunk loop mode)');
      recordingLoop();

    } catch (e: any) {
      console.error('Failed to start recording:', e);
      setError(e.message || '녹음 시작 실패');
      isRunningRef.current = false;
      setIsRecordingActive(false);
      throw e;
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<void> => {
    isRunningRef.current = false;
    setIsRecordingActive(false);
    console.log('Recording stop requested');
  }, []);

  useEffect(() => {
    return () => {
      isRunningRef.current = false;
    };
  }, []);

  return {
    isRecordingActive,
    startRecording,
    stopRecording,
    error,
  };
};

export default useAudioRecording;
