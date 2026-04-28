import { Platform } from 'react-native';
import Tts from 'react-native-tts';

// Android에서 Google TTS 엔진 패키지명
const GOOGLE_TTS_ENGINE = 'com.google.android.tts';

const BCP47: Record<string, string> = {
  ko: 'ko-KR', en: 'en-US', ja: 'ja-JP', zh: 'zh-CN',
  id: 'id-ID', vi: 'vi-VN', th: 'th-TH', es: 'es-ES', fr: 'fr-FR', de: 'de-DE',
};

export const toBCP47 = (code: string): string => BCP47[code] ?? code;

let engineReady = false;

// Android에서 Google TTS 엔진으로 초기화.
// Samsung 기기의 기본 Samsung TTS 대신 Google TTS를 강제 사용해
// 언어 팩 미설치 시 엉뚱한 언어로 fallback되는 문제 방지.
export const initTtsEngine = async () => {
  if (engineReady) return;
  if (Platform.OS === 'android') {
    try {
      await Tts.setDefaultEngine(GOOGLE_TTS_ENGINE);
    } catch {
      // Google TTS 미설치 기기면 기본 엔진 사용
    }
  }
  engineReady = true;
};

export const ttsSpeak = (
  text: string,
  langCode: string,
  rate: number,
  onDone: () => void,
  onError: () => void,
  useVoiceStream = false,
) => {
  const lang = toBCP47(langCode);
  const stream = useVoiceStream ? 'STREAM_VOICE_CALL' : 'STREAM_MUSIC';
  Tts.setDefaultLanguage(lang)
    .then(() => {
      Tts.setDefaultRate(rate, true);
      const listeners = [
        Tts.addEventListener('tts-finish', () => {
          listeners.forEach(l => l.remove());
          onDone();
        }),
        Tts.addEventListener('tts-cancel', () => {
          listeners.forEach(l => l.remove());
          onError();
        }),
        Tts.addEventListener('tts-error', () => {
          listeners.forEach(l => l.remove());
          onError();
        }),
      ];
      Tts.speak(text, { KEY_PARAM_STREAM: stream });
    })
    .catch(onError);
};

export const ttsStop = () => {
  Tts.stop();
};
