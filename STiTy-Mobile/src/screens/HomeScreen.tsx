import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  AppState,
  TouchableOpacity,
  Modal,
  Animated,
  StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Path, Rect, Ellipse } from 'react-native-svg';
import { LANGUAGES, Language, CONVERSATION_MODES } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';
import { useAudioRecording } from '../hooks/useAudioRecording';
import { initTtsEngine, ttsSpeak, ttsStop } from '../utils/tts';
import { setSpeakerphoneOn, releaseAudioMode } from '../utils/audioRouting';

// ─── Types ────────────────────────────────────────────────────────────────────
interface TranscriptionEntry {
  id: string;
  language: string;
  text: string;
  translatedText: string;
  timestamp: number;
}

// ─── Fixed card colors (from prototype) ──────────────────────────────────────
const MY_CARD_COLOR   = '#6080C8';
const PEER_CARD_COLOR = '#9BB4D4';
const MODE_ACTIVE_COLOR = '#C8B830';

// ─── Bubble color palette (by detected language) ──────────────────────────────
const LANG_COLORS: Record<string, { bubble: string; avatar: string }> = {
  en: { bubble: '#6080C8', avatar: '#9BB4D4' },
  ko: { bubble: '#7A9030', avatar: '#A8B870' },
  ja: { bubble: '#C87060', avatar: '#E0A898' },
  zh: { bubble: '#9060C8', avatar: '#B898E0' },
  es: { bubble: '#C8A030', avatar: '#E0C878' },
  fr: { bubble: '#308898', avatar: '#78B8C8' },
  id: { bubble: '#30A070', avatar: '#70C0A0' },
  vi: { bubble: '#B85050', avatar: '#E09080' },
  th: { bubble: '#5080C0', avatar: '#88B0E0' },
  de: { bubble: '#C09050', avatar: '#E0C080' },
};
const getLangColor = (code: string) =>
  LANG_COLORS[code] ?? { bubble: '#909090', avatar: '#B8B8B8' };

// ─── Storage keys ─────────────────────────────────────────────────────────────
const SK = { MY: 'stity_myLang', TARGET: 'stity_targetLang', MODE: 'stity_mode' };

// ─── Helpers ──────────────────────────────────────────────────────────────────
const langToCode = (lang: string): string => {
  const map: Record<string, string> = {
    Korean: 'ko', English: 'en', Japanese: 'ja', Chinese: 'zh',
    Indonesian: 'id', Vietnamese: 'vi', Thai: 'th',
    French: 'fr', German: 'de', Spanish: 'es',
  };
  return map[lang] ?? lang.toLowerCase().substring(0, 2);
};

const translateText = async (text: string, sl: string, tl: string): Promise<string> => {
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sl}&tl=${tl}&dt=t&q=${encodeURIComponent(text)}`;
    const res = await fetch(url);
    const data = await res.json();
    return data[0].map((i: any) => i[0]).join('');
  } catch {
    return '';
  }
};

// ─── Sub-components ───────────────────────────────────────────────────────────

// Logo: S(blue) T(blue) i(light-blue, normal) T(olive) y(pale-yellow)
const STiTyLogo = () => (
  <Text style={S.logo}>
    <Text style={{ color: '#6080C8' }}>ST</Text>
    <Text style={{ color: '#9BB4D4' }}>i</Text>
    <Text style={{ color: '#7A9030' }}>T</Text>
    <Text style={{ color: '#A8B040' }}>y</Text>
  </Text>
);

const StatusDot = ({ active }: { active: boolean }) => {
  const scale = useRef(new Animated.Value(1)).current;
  const animRef = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    if (active) {
      animRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(scale, { toValue: 1.8, duration: 700, useNativeDriver: true }),
          Animated.timing(scale, { toValue: 1, duration: 700, useNativeDriver: true }),
        ])
      );
      animRef.current.start();
    } else {
      animRef.current?.stop();
      scale.setValue(1);
    }
    return () => { animRef.current?.stop(); };
  }, [active]);

  return (
    <View style={S.dotWrap}>
      {active && <Animated.View style={[S.dotRing, { transform: [{ scale }] }]} />}
      <View style={[S.dot, { backgroundColor: active ? '#22c55e' : '#ccc' }]} />
    </View>
  );
};

const WaveBar = ({ delay }: { delay: number }) => {
  const h = useRef(new Animated.Value(3)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(h, { toValue: 14, duration: 380, delay, useNativeDriver: false }),
        Animated.timing(h, { toValue: 3, duration: 380, useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, []);
  return <Animated.View style={[S.waveBar, { height: h }]} />;
};

const RecordingBar = () => (
  <View style={S.recBar}>
    <View style={S.recDot} />
    <Text style={S.recLabel}>Listening...</Text>
    <View style={S.waveRow}>
      {[0, 80, 160, 240, 320, 240, 160, 80, 0].map((d, i) => (
        <WaveBar key={i} delay={d} />
      ))}
    </View>
  </View>
);

// SVG illustration from prototype
const IdleIllustration = () => (
  <Svg width={110} height={96} viewBox="0 0 160 140">
    <Circle cx="80" cy="68" r="54" fill="#E8E0A0" opacity="0.35" />
    <Circle cx="38" cy="28" r="2.5" fill="#E8E0A0" />
    <Circle cx="122" cy="22" r="2" fill="#9BB4D4" />
    <Circle cx="130" cy="45" r="1.5" fill="#E8E0A0" />
    <Circle cx="30" cy="50" r="1.5" fill="#9BB4D4" />
    <Circle cx="110" cy="15" r="3" fill="#E8E0A0" opacity="0.7" />
    <Path d="M50 20 Q58 10 68 18 Q56 20 50 20Z" fill="#E8E0A0" />
    <Ellipse cx="80" cy="118" rx="54" ry="14" fill="#7A9030" opacity="0.25" />
    <Ellipse cx="80" cy="115" rx="44" ry="10" fill="#7A9030" opacity="0.4" />
    <Rect x="32" y="98" width="5" height="14" rx="2" fill="#7A9030" opacity="0.6" />
    <Ellipse cx="34.5" cy="94" rx="9" ry="10" fill="#7A9030" opacity="0.7" />
    <Rect x="119" y="100" width="4" height="12" rx="2" fill="#7A9030" opacity="0.6" />
    <Ellipse cx="121" cy="96" rx="7" ry="8" fill="#7A9030" opacity="0.7" />
    <Circle cx="58" cy="90" r="10" fill="#9BB4D4" />
    <Circle cx="58" cy="76" r="8" fill="#E8E0A0" />
    <Circle cx="55.5" cy="75.5" r="1.2" fill="#6080C8" />
    <Circle cx="60.5" cy="75.5" r="1.2" fill="#6080C8" />
    <Path d="M56 79 Q58 81 60 79" stroke="#6080C8" strokeWidth="1.2" strokeLinecap="round" fill="none" />
    <Circle cx="102" cy="90" r="10" fill="#6080C8" />
    <Circle cx="102" cy="76" r="8" fill="#E8E0A0" />
    <Circle cx="99.5" cy="75.5" r="1.2" fill="#7A9030" />
    <Circle cx="104.5" cy="75.5" r="1.2" fill="#7A9030" />
    <Path d="M100 79 Q102 81 104 79" stroke="#7A9030" strokeWidth="1.2" strokeLinecap="round" fill="none" />
    <Rect x="30" y="52" width="34" height="18" rx="9" fill="white" stroke="#9BB4D4" strokeWidth="1.5" />
    <Path d="M52 70 L56 76 L46 70Z" fill="white" stroke="#9BB4D4" strokeWidth="1.5" strokeLinejoin="round" />
    <Circle cx="41" cy="61" r="2" fill="#9BB4D4" />
    <Circle cx="48" cy="61" r="2" fill="#9BB4D4" />
    <Circle cx="55" cy="61" r="2" fill="#9BB4D4" />
    <Rect x="96" y="52" width="34" height="18" rx="9" fill="white" stroke="#6080C8" strokeWidth="1.5" />
    <Path d="M108 70 L104 76 L114 70Z" fill="white" stroke="#6080C8" strokeWidth="1.5" strokeLinejoin="round" />
    <Circle cx="105" cy="61" r="2" fill="#6080C8" />
    <Circle cx="113" cy="61" r="2" fill="#6080C8" />
    <Circle cx="121" cy="61" r="2" fill="#6080C8" />
  </Svg>
);

const LangPickerSheet = ({
  visible, title, selectedCode, excludeCode, onSelect, onClose, myLangCode,
}: {
  visible: boolean;
  title: string;
  selectedCode: string;
  excludeCode: string;
  onSelect: (lang: Language) => void;
  onClose: () => void;
  myLangCode: string;
}) => {
  const filtered = LANGUAGES.filter(l => l.code !== excludeCode);
  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={S.pickerOverlay}>
        <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={onClose} />
        <View style={S.pickerSheet}>
          <Text style={S.pickerTitle}>{title}</Text>
          <ScrollView>
            {filtered.map(lang => {
              const name = lang.translations[myLangCode] || lang.nativeName;
              const selected = lang.code === selectedCode;
              return (
                <TouchableOpacity
                  key={lang.code}
                  style={[S.pickerOption, selected && S.pickerOptionSel]}
                  onPress={() => { onSelect(lang); onClose(); }}
                >
                  <Text style={[S.pickerOptionTxt, selected && S.pickerOptionTxtSel]}>
                    {name} ({lang.code})
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
          <TouchableOpacity style={S.pickerCancel} onPress={onClose}>
            <Text style={S.pickerCancelTxt}>Cancel</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const BubbleItem = ({ entry, myLangCode }: { entry: TranscriptionEntry; myLangCode: string }) => {
  const isMine = entry.language === myLangCode;
  const clr = getLangColor(entry.language);
  return (
    <View style={[S.bubbleRow, isMine && S.bubbleRowMine]}>
      <View style={[S.avatar, { backgroundColor: clr.avatar }]}>
        <Text style={S.avatarTxt}>{entry.language.toUpperCase()}</Text>
      </View>
      <View style={[
        S.bubble,
        {
          backgroundColor: clr.bubble,
          borderBottomRightRadius: isMine ? 4 : 18,
          borderBottomLeftRadius: isMine ? 18 : 4,
        },
      ]}>
        <Text style={S.bubbleMain}>{entry.text}</Text>
        {!!entry.translatedText && (
          <Text style={S.bubbleSub}>{entry.translatedText}</Text>
        )}
      </View>
    </View>
  );
};

// ─── Main screen ──────────────────────────────────────────────────────────────
export const HomeScreen: React.FC<{ navigation: any }> = () => {
  const insets = useSafeAreaInsets();

  type SessionState = 'idle' | 'recording' | 'paused';
  const [sessionState, setSessionState] = useState<SessionState>('idle');
  const sessionStateRef = useRef<SessionState>('idle');

  const [myLang, setMyLang] = useState<Language>(LANGUAGES[1]);   // en
  const [targetLang, setTargetLang] = useState<Language>(LANGUAGES[0]); // ko
  const [modeId, setModeId] = useState('mode-1');
  const [loaded, setLoaded] = useState(false);

  const [showSetup, setShowSetup] = useState(false);
  const [picker, setPicker] = useState<null | 'my' | 'peer'>(null);
  const [isTTSMuted, setIsTTSMuted] = useState(false);

  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isInitializing, setIsInitializing] = useState(false);
  const [sessionError, setSessionError] = useState('');

  const scrollRef = useRef<ScrollView>(null);
  const entryIdRef = useRef(0);
  const ttsQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef(false);

  const modeRef = useRef(modeId);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);
  const isTTSMutedRef = useRef(isTTSMuted);

  const { isConnected, connect, sendAudio, disconnect, addMessageListener, sendMessage, serverStatus, probeServer } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const sendAudioRef = useRef(sendAudio);
  const sendMessageRef = useRef(sendMessage);
  const serverStatusRef = useRef(serverStatus);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);
  useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);
  useEffect(() => { serverStatusRef.current = serverStatus; }, [serverStatus]);
  useEffect(() => { isTTSMutedRef.current = isTTSMuted; }, [isTTSMuted]);
  useEffect(() => { modeRef.current = modeId; }, [modeId]);
  useEffect(() => { myLangRef.current = myLang; }, [myLang]);
  useEffect(() => { targetLangRef.current = targetLang; }, [targetLang]);
  useEffect(() => { sessionStateRef.current = sessionState; }, [sessionState]);

  const { startRecording, stopRecording } = useAudioRecording({
    onAudioData: (audioData) => {
      if (isConnectedRef.current && sessionStateRef.current === 'recording') {
        sendAudioRef.current(audioData);
      }
    },
  });

  // ── 초기화 ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    void initTtsEngine();
    probeServer();

    AsyncStorage.multiGet([SK.MY, SK.TARGET, SK.MODE]).then(pairs => {
      const [myVal, targetVal, modeVal] = pairs.map(p => p[1]);
      if (myVal) {
        const f = LANGUAGES.find(l => l.code === myVal);
        if (f) { setMyLang(f); myLangRef.current = f; }
      }
      if (targetVal) {
        const f = LANGUAGES.find(l => l.code === targetVal);
        if (f) { setTargetLang(f); targetLangRef.current = f; }
      }
      if (modeVal) {
        const f = CONVERSATION_MODES.find(m => m.id === modeVal);
        if (f) { setModeId(f.id); modeRef.current = f.id; }
      }
      setLoaded(true);
    }).catch(() => setLoaded(true));

    return () => {
      stopRecording();
      disconnect();
      ttsQueueRef.current = [];
      isSpeakingRef.current = false;
      ttsStop();
      releaseAudioMode();
    };
  }, []);

  // ── AppState ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    const prevRef = { current: AppState.currentState };
    const sub = AppState.addEventListener('change', next => {
      const prev = prevRef.current;
      prevRef.current = next;
      if ((next === 'background' || next === 'inactive') && sessionStateRef.current === 'recording') {
        stopRecording();
        disconnect();
        sessionStateRef.current = 'paused';
        setSessionState('paused');
      } else if (next === 'active' && (prev === 'background' || prev === 'inactive') && sessionStateRef.current === 'paused') {
        doResume();
      }
    });
    return () => sub.remove();
  }, []);

  // ── Message listener ─────────────────────────────────────────────────────────
  const handleMessageRef = useRef((msg: any) => {
    const text = (msg.original || '').trim();
    if (!text || msg.type !== 'final') return;
    const lang = langToCode(msg.language || 'auto');
    const serverTrans = (msg.translation || '').trim();
    addTranscription(lang, text, serverTrans || undefined);
  });

  useEffect(() => {
    return addMessageListener((msg: any) => handleMessageRef.current(msg));
  }, [addMessageListener]);

  // ── Auto-scroll ───────────────────────────────────────────────────────────────
  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: false }), 50);
  }, [transcriptions.length]);

  // ── TTS ───────────────────────────────────────────────────────────────────────
  // mode-1: speaker, TTS for all languages
  // mode-2: earphone, TTS only for peer language
  // mode-3: if TTS lang == myLang → earphone (I hear peer's translation)
  //         if TTS lang != myLang → speaker (peer hears my translation)
  const processNextTTS = useCallback(() => {
    if (ttsQueueRef.current.length === 0) { isSpeakingRef.current = false; return; }
    isSpeakingRef.current = true;
    const next = ttsQueueRef.current.shift()!;
    const ttsStart = new Date().toISOString();

    let isEarphone: boolean;
    if (modeRef.current === 'mode-1') {
      isEarphone = false;
    } else if (modeRef.current === 'mode-2') {
      isEarphone = true;
    } else {
      // mode-3: earphone when TTS is in my language, speaker when it's in peer's language
      isEarphone = next.lang === myLangRef.current.code;
    }
    setSpeakerphoneOn(!isEarphone);
    ttsSpeak(
      next.text, next.lang, 1.3,
      () => {
        sendMessageRef.current({ type: 'tts_log', text: next.text, lang: next.lang, start: ttsStart, end: new Date().toISOString() });
        processNextTTS();
      },
      () => processNextTTS(),
      isEarphone,
    );
  }, []);

  const speakTranslation = (text: string, lang: string) => {
    if (isTTSMutedRef.current || !text) return;
    ttsQueueRef.current.push({ text, lang });
    if (!isSpeakingRef.current) processNextTTS();
  };

  const shouldPlayTTS = (detectedLang: string) => {
    if (isTTSMutedRef.current) return false;
    if (modeRef.current === 'mode-1' || modeRef.current === 'mode-3') return true;
    if (modeRef.current === 'mode-2') return detectedLang !== myLangRef.current.code;
    return false;
  };

  const getTTSTarget = (detectedLang: string) =>
    detectedLang !== myLangRef.current.code ? myLangRef.current.code : targetLangRef.current.code;

  const getTransTarget = (detectedLang: string) =>
    detectedLang === myLangRef.current.code ? targetLangRef.current.code : myLangRef.current.code;

  const addTranscription = (lang: string, text: string, serverTrans?: string) => {
    entryIdRef.current += 1;
    const id = String(entryIdRef.current);
    setTranscriptions(prev => [...prev, { id, language: lang, text, translatedText: serverTrans ?? '', timestamp: Date.now() }]);
    if (serverTrans) {
      if (shouldPlayTTS(lang)) speakTranslation(serverTrans, getTTSTarget(lang));
      return;
    }
    const tTarget = getTransTarget(lang);
    if (tTarget && tTarget !== lang) {
      translateText(text, lang, tTarget).then(translated => {
        if (!translated) return;
        setTranscriptions(prev => prev.map(e => e.id === id ? { ...e, translatedText: translated } : e));
        if (shouldPlayTTS(lang)) speakTranslation(translated, getTTSTarget(lang));
      });
    }
  };

  // ── Session actions ───────────────────────────────────────────────────────────
  const stopTTS = () => {
    ttsQueueRef.current = [];
    isSpeakingRef.current = false;
    ttsStop();
  };

  // mode-2: earphone default, mode-1 & mode-3: speaker default
  // (mode-3 switches dynamically per TTS item inside processNextTTS)
  const applySpeakerRouting = (id = modeRef.current) =>
    setSpeakerphoneOn(id !== 'mode-2');

  const doStart = async () => {
    if (myLang.code === targetLang.code) {
      Alert.alert('Error', 'My language and peer language must be different.');
      return;
    }
    setSessionError('');
    try {
      if (serverStatusRef.current !== 'ready') {
        setIsInitializing(true);
        await probeServer();
      }
      await connect({ lang: myLang.code, targetLang: targetLang.code });
      await startRecording();
      applySpeakerRouting();
      sessionStateRef.current = 'recording';
      setSessionState('recording');
      setShowSetup(false);
    } catch (e: any) {
      setSessionError(e?.message || 'Connection failed');
    } finally {
      setIsInitializing(false);
    }
  };

  const doStop = async () => {
    stopTTS();
    await stopRecording();
    sessionStateRef.current = 'paused';
    setSessionState('paused');
  };

  const doResume = async () => {
    setSessionError('');
    try {
      if (!isConnectedRef.current) {
        if (serverStatusRef.current !== 'ready') {
          setIsInitializing(true);
          await probeServer();
        }
        await connect({ lang: myLangRef.current.code, targetLang: targetLangRef.current.code });
      }
      await startRecording();
      applySpeakerRouting();
      sessionStateRef.current = 'recording';
      setSessionState('recording');
    } catch (e: any) {
      setSessionError(e?.message || 'Connection failed');
    } finally {
      setIsInitializing(false);
    }
  };

  const doBack = () => {
    Alert.alert('End conversation', 'Do you want to end the conversation?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'End', style: 'destructive',
        onPress: async () => {
          stopTTS();
          await stopRecording();
          disconnect();
          setTranscriptions([]);
          releaseAudioMode();
          sessionStateRef.current = 'idle';
          setSessionState('idle');
          setShowSetup(false);
        },
      },
    ]);
  };

  // ── Settings updates ──────────────────────────────────────────────────────────
  const updateMyLang = (lang: Language) => {
    setMyLang(lang); myLangRef.current = lang;
    AsyncStorage.setItem(SK.MY, lang.code);
  };
  const updateTargetLang = (lang: Language) => {
    setTargetLang(lang); targetLangRef.current = lang;
    AsyncStorage.setItem(SK.TARGET, lang.code);
  };
  const updateMode = (id: string) => {
    setModeId(id); modeRef.current = id;
    AsyncStorage.setItem(SK.MODE, id);
    if (sessionStateRef.current !== 'idle') {
      stopTTS();
      applySpeakerRouting(id);
    }
  };

  if (!loaded) return null;

  const isIdle = sessionState === 'idle';
  const isRecording = sessionState === 'recording';
  const isPaused = sessionState === 'paused';
  const setupVisible = isIdle || showSetup;
  const modeObj = CONVERSATION_MODES.find(m => m.id === modeId) ?? CONVERSATION_MODES[0];
  const bottomPad = Math.max(insets.bottom, 16);

  return (
    <SafeAreaView style={S.container} edges={['top', 'left', 'right']}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      {/* ── Header ── */}
      <View style={S.header}>
        <STiTyLogo />
        <View style={S.headerRight}>
          {!isIdle && (
            <TouchableOpacity onPress={() => setShowSetup(v => !v)} style={S.iconBtn}>
              <Ionicons name="settings-outline" size={18} color="#ccc" />
            </TouchableOpacity>
          )}
          <StatusDot active={isRecording} />
          <TouchableOpacity
            style={S.iconBtn}
            onPress={() => setIsTTSMuted(v => {
              const next = !v;
              isTTSMutedRef.current = next;
              if (next) stopTTS();
              return next;
            })}
          >
            <Ionicons
              name={isTTSMuted ? 'volume-mute-outline' : 'volume-high-outline'}
              size={20}
              color={isTTSMuted ? '#C87060' : '#7A9030'}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* ── Setup panel ── */}
      {setupVisible && (
        <View style={S.setupPanel}>
          {/* Language cards — fixed colors per prototype */}
          <View style={S.langRow}>
            <TouchableOpacity
              style={[S.langCard, { backgroundColor: MY_CARD_COLOR }]}
              onPress={() => setPicker('my')}
              activeOpacity={0.85}
            >
              <Text style={S.langCardLabel}>MY LANGUAGE</Text>
              <Text style={S.langCardValue}>{myLang.name}</Text>
              <Text style={S.langCardArrow}>▾</Text>
            </TouchableOpacity>

            <Text style={S.langSwap}>⇄</Text>

            <TouchableOpacity
              style={[S.langCard, { backgroundColor: PEER_CARD_COLOR }]}
              onPress={() => setPicker('peer')}
              activeOpacity={0.85}
            >
              <Text style={S.langCardLabel}>PEER LANGUAGE</Text>
              <Text style={S.langCardValue}>
                {targetLang.translations[myLang.code] ?? targetLang.name}
              </Text>
              <Text style={S.langCardArrow}>▾</Text>
            </TouchableOpacity>
          </View>

          {/* Mode segment — 3 options */}
          <View style={S.modeSection}>
            <Text style={S.modeSectionLabel}>MODE</Text>
            <View style={S.modeSegment}>
              {CONVERSATION_MODES.map(m => (
                <TouchableOpacity
                  key={m.id}
                  style={[S.modeOption, modeId === m.id && S.modeOptionActive]}
                  onPress={() => updateMode(m.id)}
                  activeOpacity={0.8}
                >
                  <Text style={[S.modeOptionTxt, modeId === m.id && S.modeOptionTxtActive]}>
                    {m.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {!isIdle && (
            <TouchableOpacity style={S.collapseBtn} onPress={() => setShowSetup(false)}>
              <Text style={S.collapseBtnTxt}>Collapse ▲</Text>
            </TouchableOpacity>
          )}
        </View>
      )}

      {/* ── Session chips ── */}
      {!isIdle && !showSetup && (
        <TouchableOpacity style={S.chips} onPress={() => setShowSetup(true)} activeOpacity={0.8}>
          <View style={[S.chip, { backgroundColor: MY_CARD_COLOR + '22' }]}>
            <Text style={[S.chipTxt, { color: MY_CARD_COLOR }]}>{myLang.name}</Text>
          </View>
          <Text style={S.chipSep}>↔</Text>
          <View style={[S.chip, { backgroundColor: PEER_CARD_COLOR + '44' }]}>
            <Text style={[S.chipTxt, { color: '#6070A0' }]}>
              {targetLang.translations[myLang.code] ?? targetLang.name}
            </Text>
          </View>
          <View style={[S.chip, { backgroundColor: '#f0f0f0' }]}>
            <Text style={[S.chipTxt, { color: '#777' }]}>{modeObj.name}</Text>
          </View>
        </TouchableOpacity>
      )}

      {/* ── Error banner ── */}
      {!!sessionError && (
        <View style={S.errorBanner}>
          <Text style={S.errorBannerTxt}>{sessionError}</Text>
        </View>
      )}

      {/* ── Conversation area ── */}
      <ScrollView
        ref={scrollRef}
        style={S.convArea}
        contentContainerStyle={[
          S.convContent,
          (isIdle || transcriptions.length === 0) && S.convContentFlex,
        ]}
      >
        {isIdle && (
          <View style={S.idleWrap}>
            <IdleIllustration />
            <Text style={S.idleTxt}>Configure and tap Start</Text>
          </View>
        )}

        {!isIdle && transcriptions.length === 0 && (
          <View style={S.idleWrap}>
            <Text style={{ fontSize: 32 }}>🎙</Text>
            <Text style={S.idleTxt}>Start speaking</Text>
          </View>
        )}

        {transcriptions.map(entry => (
          <BubbleItem key={entry.id} entry={entry} myLangCode={myLang.code} />
        ))}
      </ScrollView>

      {/* ── Recording bar ── */}
      {isRecording && <RecordingBar />}

      {/* ── Loading overlay ── */}
      {isInitializing && (
        <View style={S.initOverlay}>
          <Text style={S.initTxt}>Connecting...</Text>
        </View>
      )}

      {/* ── Bottom bar ── */}
      <View style={[S.bottomBar, { paddingBottom: bottomPad + 8 }]}>
        {isIdle && (
          <TouchableOpacity
            style={[S.btn, S.btnOutline, serverStatus !== 'ready' && S.btnDisabled]}
            onPress={doStart}
            activeOpacity={0.85}
            disabled={serverStatus !== 'ready' || isInitializing}
          >
            <Text style={[S.btnTxt, { color: '#7A9030' }]}>
              {serverStatus === 'ready'
                ? 'Start'
                : serverStatus === 'connecting'
                  ? 'Connecting to server...'
                  : 'Starting server...'}
            </Text>
          </TouchableOpacity>
        )}

        {isRecording && (
          <>
            <TouchableOpacity style={[S.btn, S.btnStop]} onPress={doStop} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#ef4444' }]}>Stop</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnGhost]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#999' }]}>✕</Text>
            </TouchableOpacity>
          </>
        )}

        {isPaused && (
          <>
            <TouchableOpacity style={[S.btn, S.btnOutline]} onPress={doResume} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#7A9030' }]}>Resume</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnOutlineCyan]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#6080C8' }]}>Back</Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      {/* ── Language pickers ── */}
      <LangPickerSheet
        visible={picker === 'my'}
        title="My Language"
        selectedCode={myLang.code}
        excludeCode={targetLang.code}
        onSelect={updateMyLang}
        onClose={() => setPicker(null)}
        myLangCode={myLang.code}
      />
      <LangPickerSheet
        visible={picker === 'peer'}
        title="Peer Language"
        selectedCode={targetLang.code}
        excludeCode={myLang.code}
        onSelect={updateTargetLang}
        onClose={() => setPicker(null)}
        myLangCode={myLang.code}
      />
    </SafeAreaView>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const S = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },

  // Header
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingTop: 14, paddingBottom: 2 },
  logo: { fontSize: 28, fontWeight: '900', letterSpacing: -1 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  iconBtn: { padding: 4 },
  dotWrap: { width: 16, height: 16, alignItems: 'center', justifyContent: 'center' },
  dotRing: { position: 'absolute', width: 16, height: 16, borderRadius: 8, backgroundColor: 'rgba(34,197,94,0.2)' },
  dot: { width: 8, height: 8, borderRadius: 4 },

  // Setup panel
  setupPanel: { paddingHorizontal: 20, paddingTop: 18 },
  langRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  langCard: { flex: 1, borderRadius: 16, paddingTop: 12, paddingBottom: 14, paddingHorizontal: 14 },
  langCardLabel: { fontSize: 9, fontWeight: '700', color: 'rgba(255,255,255,0.7)', letterSpacing: 0.5, marginBottom: 5 },
  langCardValue: { fontSize: 16, fontWeight: '700', color: '#fff' },
  langCardArrow: { position: 'absolute', top: 10, right: 12, color: 'rgba(255,255,255,0.6)', fontSize: 12 },
  langSwap: { fontSize: 20, color: '#ccc' },
  modeSection: { marginBottom: 4 },
  modeSectionLabel: { fontSize: 9, fontWeight: '700', color: '#aaa', letterSpacing: 0.5, marginBottom: 6 },
  modeSegment: { flexDirection: 'row', backgroundColor: '#f0f0f0', borderRadius: 12, padding: 3, gap: 2 },
  modeOption: { flex: 1, paddingVertical: 8, borderRadius: 10, alignItems: 'center' },
  modeOptionActive: { backgroundColor: '#fff', shadowColor: '#000', shadowOpacity: 0.08, shadowRadius: 4, elevation: 2 },
  modeOptionTxt: { fontSize: 11, fontWeight: '600', color: '#999', textAlign: 'center', lineHeight: 14 },
  modeOptionTxtActive: { color: MODE_ACTIVE_COLOR },
  collapseBtn: { alignItems: 'flex-end', paddingTop: 8, paddingBottom: 2 },
  collapseBtnTxt: { fontSize: 12, color: '#aaa', fontWeight: '600' },

  // Session chips
  chips: { flexDirection: 'row', paddingHorizontal: 20, paddingTop: 12, paddingBottom: 2, alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  chip: { borderRadius: 100, paddingVertical: 3, paddingHorizontal: 10 },
  chipTxt: { fontSize: 11, fontWeight: '600' },
  chipSep: { fontSize: 11, color: '#aaa' },

  // Error
  errorBanner: { backgroundColor: '#fff0f0', marginHorizontal: 16, marginTop: 6, padding: 10, borderRadius: 12 },
  errorBannerTxt: { color: '#ef4444', fontSize: 12, textAlign: 'center' },

  // Conversation
  convArea: { flex: 1 },
  convContent: { paddingVertical: 12 },
  convContentFlex: { flexGrow: 1 },
  idleWrap: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 40, gap: 12 },
  idleTxt: { fontSize: 13, color: '#ccc', textAlign: 'center' },

  // Bubbles
  bubbleRow: { flexDirection: 'row', paddingHorizontal: 16, paddingVertical: 5, gap: 8, alignItems: 'flex-end' },
  bubbleRowMine: { flexDirection: 'row-reverse' },
  avatar: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  avatarTxt: { fontSize: 8, fontWeight: '700', color: '#fff' },
  bubble: { maxWidth: '72%', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 18 },
  bubbleMain: { fontSize: 15, fontWeight: '600', color: '#fff', lineHeight: 21 },
  bubbleSub: { fontSize: 11, color: 'rgba(255,255,255,0.65)', marginTop: 3 },

  // Recording bar
  recBar: { flexDirection: 'row', alignItems: 'center', marginHorizontal: 16, marginBottom: 8, backgroundColor: 'rgba(96,128,200,0.07)', borderRadius: 16, paddingVertical: 12, paddingHorizontal: 16, gap: 10 },
  recDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#ef4444' },
  recLabel: { flex: 1, fontSize: 12, fontWeight: '600', color: '#6080C8' },
  waveRow: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  waveBar: { width: 3, borderRadius: 2, backgroundColor: '#6080C8' },

  // Loading overlay
  initOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(255,255,255,0.9)', alignItems: 'center', justifyContent: 'center' },
  initTxt: { fontSize: 16, fontWeight: '600', color: '#7A9030' },

  // Bottom bar
  bottomBar: { flexDirection: 'row', paddingHorizontal: 20, paddingTop: 10 , gap: 10 },
  btn: { flex: 1, height: 52, borderRadius: 100, alignItems: 'center', justifyContent: 'center' },
  btnTxt: { fontSize: 15, fontWeight: '600' },
  btnOutline: { borderWidth: 2, borderColor: '#7A9030', backgroundColor: '#fff' },
  btnOutlineCyan: { borderWidth: 2, borderColor: '#6080C8', backgroundColor: '#fff' },
  btnStop: { backgroundColor: '#fff0f0' },
  btnGhost: { flex: 0, width: 52, backgroundColor: '#f5f5f5' },
  btnDisabled: { opacity: 0.45 },

  // Language picker
  pickerOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', justifyContent: 'flex-end' },
  pickerSheet: { backgroundColor: '#fff', borderTopLeftRadius: 24, borderTopRightRadius: 24, paddingTop: 20, paddingHorizontal: 20, paddingBottom: 32, maxHeight: '75%' },
  pickerTitle: { fontSize: 13, fontWeight: '700', color: '#aaa', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 },
  pickerOption: { paddingVertical: 13, paddingHorizontal: 14, borderRadius: 12 },
  pickerOptionSel: { backgroundColor: '#eaf2fb' },
  pickerOptionTxt: { fontSize: 15, fontWeight: '600', color: '#1a1a1a' },
  pickerOptionTxtSel: { color: '#6080C8' },
  pickerCancel: { marginTop: 8, paddingVertical: 14, borderRadius: 14, backgroundColor: '#f5f5f5', alignItems: 'center' },
  pickerCancelTxt: { fontSize: 15, fontWeight: '600', color: '#aaa' },
});

export default HomeScreen;
