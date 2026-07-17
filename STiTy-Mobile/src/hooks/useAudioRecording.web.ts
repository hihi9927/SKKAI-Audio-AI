import { useCallback, useEffect, useRef, useState } from 'react';

// 웹(브라우저) 전용 마이크 캡처 구현.
// 네이티브의 react-native-live-audio-stream 대신 getUserMedia + Web Audio API를 사용해
// 서버가 기대하는 PCM s16le / 16kHz / mono 청크를 생성한다.
// Metro 웹 번들러가 useAudioRecording.ts 대신 이 파일(.web.ts)을 자동 선택한다.

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

const TARGET_SAMPLE_RATE = 16000;
// ScriptProcessor 버퍼 크기(2의 거듭제곱). 48kHz 기준 약 85ms.
const PROCESSOR_BUFFER = 4096;

// Float32(-1~1) 모노 → 16kHz로 다운샘플 후 s16le PCM(ArrayBuffer)로 변환
const downsampleToInt16 = (
  input: Float32Array,
  inputRate: number,
  outputRate: number,
): ArrayBuffer => {
  let samples: Float32Array;
  if (outputRate >= inputRate) {
    samples = input;
  } else {
    const ratio = inputRate / outputRate;
    const newLength = Math.round(input.length / ratio);
    samples = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const start = Math.round(i * ratio);
      const end = Math.min(Math.round((i + 1) * ratio), input.length);
      let sum = 0;
      let count = 0;
      for (let j = start; j < end; j++) {
        sum += input[j];
        count++;
      }
      samples[i] = count > 0 ? sum / count : input[Math.min(start, input.length - 1)];
    }
  }

  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); // little-endian
  }
  return buffer;
};

export const useAudioRecording = ({
  onAudioData,
  sampleRate = TARGET_SAMPLE_RATE,
}: UseAudioRecordingOptions): UseAudioRecordingReturn => {
  const [isRecordingActive, setIsRecordingActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onAudioDataRef = useRef(onAudioData);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mutedGainRef = useRef<GainNode | null>(null);

  useEffect(() => {
    onAudioDataRef.current = onAudioData;
  }, [onAudioData]);

  const cleanup = useCallback(() => {
    try {
      processorRef.current?.disconnect();
    } catch { /* ignore */ }
    try {
      sourceRef.current?.disconnect();
    } catch { /* ignore */ }
    try {
      mutedGainRef.current?.disconnect();
    } catch { /* ignore */ }
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch { /* ignore */ }
    try {
      audioCtxRef.current?.close();
    } catch { /* ignore */ }
    processorRef.current = null;
    sourceRef.current = null;
    mutedGainRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;
  }, []);

  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      if (!navigator?.mediaDevices?.getUserMedia) {
        throw new Error('이 브라우저는 마이크 입력을 지원하지 않습니다');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      streamRef.current = stream;

      const Ctx: typeof AudioContext =
        window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new Ctx();
      // 자동재생 정책으로 suspended 상태일 수 있어 명시적으로 재개
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioCtx.createScriptProcessor(PROCESSOR_BUFFER, 1, 1);
      processorRef.current = processor;

      const inputRate = audioCtx.sampleRate;
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const channel = e.inputBuffer.getChannelData(0);
        const pcm = downsampleToInt16(channel, inputRate, sampleRate);
        if (pcm.byteLength > 0) {
          onAudioDataRef.current(pcm);
        }
      };

      // 마이크 → 프로세서 연결. 프로세서가 콜백을 발화하려면 destination까지
      // 연결돼야 하므로, 하울링 방지를 위해 gain 0인 노드를 거쳐 출력에 연결.
      const mutedGain = audioCtx.createGain();
      mutedGain.gain.value = 0;
      mutedGainRef.current = mutedGain;
      source.connect(processor);
      processor.connect(mutedGain);
      mutedGain.connect(audioCtx.destination);

      setIsRecordingActive(true);
      setError(null);
    } catch (e: any) {
      cleanup();
      const msg =
        e?.name === 'NotAllowedError'
          ? '마이크 권한이 필요합니다'
          : e?.message || '녹음 시작 실패';
      setError(msg);
      setIsRecordingActive(false);
      throw e;
    }
  }, [sampleRate, cleanup]);

  const stopRecording = useCallback(async (): Promise<void> => {
    cleanup();
    setIsRecordingActive(false);
  }, [cleanup]);

  return {
    isRecordingActive,
    startRecording,
    stopRecording,
    error,
  };
};

export default useAudioRecording;
