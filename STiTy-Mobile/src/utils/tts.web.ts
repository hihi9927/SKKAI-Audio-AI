// 웹(브라우저) 전용 TTS 구현.
// 네이티브의 react-native-tts 대신 브라우저 Web Speech API(speechSynthesis)를 사용한다.
// Metro 웹 번들러가 tts.ts 대신 이 파일(.web.ts)을 자동 선택한다.

const BCP47: Record<string, string> = {
  ko: 'ko-KR', en: 'en-US', ja: 'ja-JP', zh: 'zh-CN',
  id: 'id-ID', vi: 'vi-VN', th: 'th-TH', es: 'es-ES', fr: 'fr-FR', de: 'de-DE',
};

export const toBCP47 = (code: string): string => BCP47[code] ?? code;

const hasSynth = (): boolean =>
  typeof window !== 'undefined' && typeof window.speechSynthesis !== 'undefined';

// 일부 브라우저(Chrome)는 voices 목록을 비동기로 로드한다. 미리 한 번 깨워둔다.
export const initTtsEngine = async (): Promise<void> => {
  if (!hasSynth()) return;
  try {
    // voices가 아직 비어 있으면 onvoiceschanged 이벤트로 채워지길 기다림 (best-effort)
    if (window.speechSynthesis.getVoices().length === 0) {
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 500);
        window.speechSynthesis.onvoiceschanged = () => {
          clearTimeout(timer);
          resolve();
        };
      });
    }
  } catch {
    // 무시
  }
};

const pickVoice = (lang: string): SpeechSynthesisVoice | undefined => {
  if (!hasSynth()) return undefined;
  const voices = window.speechSynthesis.getVoices();
  // 정확히 일치(ko-KR) → 언어 접두사 일치(ko) 순으로 탐색
  const prefix = lang.split('-')[0].toLowerCase();
  return (
    voices.find((v) => v.lang.toLowerCase() === lang.toLowerCase()) ||
    voices.find((v) => v.lang.toLowerCase().startsWith(prefix))
  );
};

export const ttsSpeak = (
  text: string,
  langCode: string,
  rate: number,
  onDone: () => void,
  onError: () => void,
  _useVoiceStream = false,
): void => {
  if (!hasSynth() || !text) {
    onError();
    return;
  }
  try {
    const lang = toBCP47(langCode);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    // Web Speech rate 범위는 0.1~10, 기본 1. 네이티브의 빠른 발화 의도를 반영해 약간 가속.
    utterance.rate = Math.max(0.5, Math.min(2, rate));
    const voice = pickVoice(lang);
    if (voice) utterance.voice = voice;
    utterance.onend = () => onDone();
    utterance.onerror = () => onError();
    window.speechSynthesis.speak(utterance);
  } catch {
    onError();
  }
};

export const ttsStop = (): void => {
  if (!hasSynth()) return;
  try {
    window.speechSynthesis.cancel();
  } catch {
    // 무시
  }
};
