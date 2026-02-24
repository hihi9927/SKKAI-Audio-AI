export interface Language {
  code: string;
  name: string;
  nativeName: string;
  /** 다른 언어에서 이 언어를 부르는 이름 (key: 언어 code) */
  translations: Record<string, string>;
}

export const LANGUAGES: Language[] = [
  { code: 'ko', name: 'Korean', nativeName: '한국어', translations: { ko: '한국어', en: 'Korean', ja: '韓国語', zh: '韩语', id: 'Korea', vi: 'Tiếng Hàn', th: 'เกาหลี', es: 'Coreano', fr: 'Coréen', de: 'Koreanisch' } },
  { code: 'en', name: 'English', nativeName: 'English', translations: { ko: '영어', en: 'English', ja: '英語', zh: '英语', id: 'Inggris', vi: 'Tiếng Anh', th: 'อังกฤษ', es: 'Inglés', fr: 'Anglais', de: 'Englisch' } },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', translations: { ko: '일본어', en: 'Japanese', ja: '日本語', zh: '日语', id: 'Jepang', vi: 'Tiếng Nhật', th: 'ญี่ปุ่น', es: 'Japonés', fr: 'Japonais', de: 'Japanisch' } },
  { code: 'zh', name: 'Chinese', nativeName: '中文', translations: { ko: '중국어', en: 'Chinese', ja: '中国語', zh: '中文', id: 'Mandarin', vi: 'Tiếng Trung', th: 'จีน', es: 'Chino', fr: 'Chinois', de: 'Chinesisch' } },
  { code: 'id', name: 'Indonesian', nativeName: 'Indonesia', translations: { ko: '인도네시아어', en: 'Indonesian', ja: 'インドネシア語', zh: '印尼语', id: 'Indonesia', vi: 'Tiếng Indonesia', th: 'อินโดนีเซีย', es: 'Indonesio', fr: 'Indonésien', de: 'Indonesisch' } },
  { code: 'vi', name: 'Vietnamese', nativeName: 'Tiếng Việt', translations: { ko: '베트남어', en: 'Vietnamese', ja: 'ベトナム語', zh: '越南语', id: 'Vietnam', vi: 'Tiếng Việt', th: 'เวียดนาม', es: 'Vietnamita', fr: 'Vietnamien', de: 'Vietnamesisch' } },
  { code: 'th', name: 'Thai', nativeName: 'ไทย', translations: { ko: '태국어', en: 'Thai', ja: 'タイ語', zh: '泰语', id: 'Thailand', vi: 'Tiếng Thái', th: 'ไทย', es: 'Tailandés', fr: 'Thaï', de: 'Thailändisch' } },
  { code: 'es', name: 'Spanish', nativeName: 'Español', translations: { ko: '스페인어', en: 'Spanish', ja: 'スペイン語', zh: '西班牙语', id: 'Spanyol', vi: 'Tiếng Tây Ban Nha', th: 'สเปน', es: 'Español', fr: 'Espagnol', de: 'Spanisch' } },
  { code: 'fr', name: 'French', nativeName: 'Français', translations: { ko: '프랑스어', en: 'French', ja: 'フランス語', zh: '法语', id: 'Prancis', vi: 'Tiếng Pháp', th: 'ฝรั่งเศส', es: 'Francés', fr: 'Français', de: 'Französisch' } },
  { code: 'de', name: 'German', nativeName: 'Deutsch', translations: { ko: '독일어', en: 'German', ja: 'ドイツ語', zh: '德语', id: 'Jerman', vi: 'Tiếng Đức', th: 'เยอรมัน', es: 'Alemán', fr: 'Allemand', de: 'Deutsch' } },
];

export const CONVERSATION_MODES = [
  { id: 'mode-1', name: '화면으로 보기', description: '모두 내 화면을 보고 대화', earphones: 0 },
  { id: 'mode-2', name: '이어폰으로 듣기', description: '상대는 내 화면을 보거나 본인 이어폰으로 참여 - 추천', earphones: 1 },
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
