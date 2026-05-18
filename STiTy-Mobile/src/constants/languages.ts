export interface Language {
  code: string;
  name: string;
  nativeName: string;
  /** 다른 언어에서 이 언어를 부르는 이름 (key: 언어 code) */
  translations: Record<string, string>;
}

export const LANGUAGES: Language[] = [
  { code: 'ko', name: 'Korean',   nativeName: '한국어', translations: { ko: '한국어', en: 'Korean',   ja: '韓国語', zh: '韩语',   es: 'Coreano'  } },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', translations: { ko: '일본어', en: 'Japanese', ja: '日本語', zh: '日语',   es: 'Japonés'  } },
  { code: 'zh', name: 'Chinese',  nativeName: '中文',   translations: { ko: '중국어', en: 'Chinese',  ja: '中国語', zh: '中文',   es: 'Chino'    } },
  { code: 'es', name: 'Spanish',  nativeName: 'Español',translations: { ko: '스페인어',en: 'Spanish',  ja: 'スペイン語',zh: '西班牙语',es: 'Español'  } },
  { code: 'en', name: 'English',  nativeName: 'English',translations: { ko: '영어',   en: 'English',  ja: '英語',   zh: '英语',   es: 'Inglés'   } },
];

export const CONVERSATION_MODES = [
  { id: 'mode-1', name: 'Speaker output', description: 'Share one device; everyone hears translations through the speaker', earphones: 0 },
  { id: 'mode-2', name: 'Earphone only', description: 'One person wears an earphone and hears the peer\'s translations', earphones: 1 },
  { id: 'mode-3', name: 'Both', description: 'Both users wear earphones and hear all translations', earphones: 2 },
];

export const getLanguageByCode = (code: string): Language | undefined => {
  return LANGUAGES.find(lang => lang.code === code);
};

/** 기본 표시: 자국어 이름(code) */
export const formatLanguageDisplay = (lang: Language): string => {
  return `${lang.nativeName}(${lang.code})`;
};

/** 나의 언어 기준으로 표시: 예) myLang=ko, target=en → 영어(en) */
export const formatLanguageAs = (lang: Language, asLangCode: string): string => {
  const translated = lang.translations[asLangCode] || lang.nativeName;
  return `${translated}(${lang.code})`;
};
