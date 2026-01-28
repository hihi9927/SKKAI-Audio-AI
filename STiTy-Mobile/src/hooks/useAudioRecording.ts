import { useState, useCallback, useRef, useEffect } from 'react';
import { Audio } from 'expo-av';
import { Platform } from 'react-native';

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
const CHUNK_DURATION_MS = 500; // 500ms chunks like the desktop version

export const useAudioRecording = ({
  onAudioData,
  sampleRate = SAMPLE_RATE,
  chunkDurationMs = CHUNK_DURATION_MS,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

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
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });
    } catch (e) {
      console.error('Failed to configure audio:', e);
    }
  };

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      // Request permissions
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        throw new Error('No microphone permission');
      }

      // Configure audio mode
      await configureAudio();

      // Recording options optimized for speech
      const recordingOptions: Audio.RecordingOptions = {
        android: {
          extension: '.wav',
          outputFormat: Audio.AndroidOutputFormat.DEFAULT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
          sampleRate: sampleRate,
          numberOfChannels: 1,
          bitRate: sampleRate * 16,
        },
        ios: {
          extension: '.wav',
          outputFormat: Audio.IOSOutputFormat.LINEARPCM,
          audioQuality: Audio.IOSAudioQuality.HIGH,
          sampleRate: sampleRate,
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

      // Create and start recording
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync(recordingOptions);
      await recording.startAsync();

      recordingRef.current = recording;
      setIsRecordingActive(true);
      setError(null);

      console.log('Recording started');

      // Set up periodic chunk sending
      // Note: Expo Audio doesn't provide real-time PCM data access easily
      // This is a simplified implementation - for production, you might need
      // a native module or different approach

      intervalRef.current = setInterval(async () => {
        if (recordingRef.current) {
          try {
            const status = await recordingRef.current.getStatusAsync();
            if (status.isRecording && status.durationMillis > 0) {
              // In a real implementation, you would extract PCM data here
              // For now, we'll simulate sending audio chunks

              // Create a dummy audio chunk (16kHz, mono, 16-bit = 2 bytes per sample)
              // 500ms = 8000 samples = 16000 bytes
              const chunkSamples = Math.floor(sampleRate * (chunkDurationMs / 1000));
              const chunkBytes = chunkSamples * 2; // 16-bit = 2 bytes

              // Note: This is a placeholder. In production, you need actual audio data.
              // You might need to use a native module like react-native-audio-api
              // or expo-av's getStatusAsync to get actual audio chunks

              console.log(`Recording status: ${status.durationMillis}ms`);
            }
          } catch (e) {
            console.error('Error getting recording status:', e);
          }
        }
      }, chunkDurationMs);

    } catch (e: any) {
      console.error('Failed to start recording:', e);
      setError(e.message || '녹음 시작 실패');
      setIsRecordingActive(false);
      throw e;
    }
  }, [onAudioData, sampleRate, chunkDurationMs]);

  const stopRecording = useCallback(async (): Promise<void> => {
    try {
      // Clear the interval
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      // Stop and unload the recording
      if (recordingRef.current) {
        const status = await recordingRef.current.getStatusAsync();
        if (status.isRecording) {
          await recordingRef.current.stopAndUnloadAsync();
        }

        // Get the URI of the recorded audio
        const uri = recordingRef.current.getURI();
        console.log('Recording stopped, URI:', uri);

        // In a production app, you might want to send the final chunk here
        // or process the complete recording

        recordingRef.current = null;
      }

      setIsRecordingActive(false);
      console.log('Recording stopped');

    } catch (e: any) {
      console.error('Failed to stop recording:', e);
      setError(e.message || '녹음 중지 실패');
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (recordingRef.current) {
        recordingRef.current.stopAndUnloadAsync().catch(() => {});
      }
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
