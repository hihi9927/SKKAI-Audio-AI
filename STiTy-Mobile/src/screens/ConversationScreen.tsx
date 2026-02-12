import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Alert,
  AppState,
  TouchableOpacity,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import * as Speech from 'expo-speech';
import { Ionicons } from '@expo/vector-icons';
import { TranslationItem } from '../components/TranslationItem';
import { GradientButton } from '../components/GradientButton';
import { COLORS, FONTS, SPACING } from '../constants/theme';
import { Language, CONVERSATION_MODES } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';
import { useAudioRecording } from '../hooks/useAudioRecording';

type RootStackParamList = {
  Home: undefined;
  Loading: { myLang: Language; targetLang: Language; mode: string };
  Conversation: { myLang: Language; targetLang: Language; mode: string; initialMessage?: any };
};

type ConversationScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Conversation'>;
type ConversationScreenRouteProp = RouteProp<RootStackParamList, 'Conversation'>;

interface ConversationScreenProps {
  navigation: ConversationScreenNavigationProp;
  route: ConversationScreenRouteProp;
}

interface TranscriptionEntry {
  id: string;
  language: string;
  text: string;
  translatedText: string;
  timestamp: number;
}

const langToCode = (lang: string): string => {
  const map: Record<string, string> = {
    'Korean': 'ko', 'English': 'en', 'Japanese': 'ja',
    'Chinese': 'zh', 'Indonesian': 'id', 'Vietnamese': 'vi',
    'Thai': 'th', 'French': 'fr', 'German': 'de', 'Spanish': 'es',
  };
  return map[lang] || lang.toLowerCase().substring(0, 2);
};

const MAX_VISIBLE = 4;

// Google Translate 무료 API
const translateText = async (text: string, sourceLang: string, targetLang: string): Promise<string> => {
  try {
    const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${sourceLang}&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;
    const res = await fetch(url);
    const data = await res.json();
    // 응답 형식: [[["translated text","source text",null,null,10]],null,"ko"]
    return data[0].map((item: any) => item[0]).join('');
  } catch (e) {
    console.error('Translation failed:', e);
    return '';
  }
};

export const ConversationScreen: React.FC<ConversationScreenProps> = ({ navigation, route }) => {
  const { myLang, targetLang, initialMessage } = route.params;
  const [currentMode, setCurrentMode] = useState(route.params.mode);
  const [displayText, setDisplayText] = useState<{ lang: string; text: string } | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [showFullTranscript, setShowFullTranscript] = useState(false);

  // 모드에 따라 TTS 자동 설정: mode-1, mode-2는 TTS 켜짐
  const isTTSEnabled = currentMode === 'mode-1' || currentMode === 'mode-2';
  const isTTSEnabledRef = useRef(isTTSEnabled);

  const { isConnected, connect, sendAudio, disconnect, addMessageListener } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const isPausedRef = useRef(isPaused);
  const sendAudioRef = useRef(sendAudio);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);
  useEffect(() => { isTTSEnabledRef.current = isTTSEnabled; }, [isTTSEnabled]);

  // 모드 ref (handleMessage에서 접근)
  const modeRef = useRef(currentMode);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);

  useEffect(() => { modeRef.current = currentMode; }, [currentMode]);

  // TTS로 말한 텍스트를 기억 (마이크 피드백 방지)
  const lastSpokenTextRef = useRef('');

  const speakTranslation = (text: string, lang: string) => {
    if (!isTTSEnabledRef.current || !text) return;
    Speech.stop();
    lastSpokenTextRef.current = text;
    Speech.speak(text, { language: lang, rate: 1.0 });
  };

  const { startRecording, stopRecording, isRecordingActive } = useAudioRecording({
    onAudioData: (audioData) => {
      if (isConnectedRef.current && !isPausedRef.current) {
        sendAudioRef.current(audioData);
      }
    },
  });

  const processedInitialRef = useRef(false);
  const entryIdRef = useRef(0);

  // 모드별 TTS 재생 판단
  const shouldPlayTTS = (detectedLang: string): boolean => {
    if (!isTTSEnabledRef.current) return false;
    const currentMode = modeRef.current;
    if (currentMode === 'mode-0') return false; // 0 이어폰: TTS 없음
    if (currentMode === 'mode-1') {
      // 1 이어폰: 상대 언어 감지 시에만 TTS (이어폰 착용자에게 내 언어로 번역)
      return detectedLang !== myLangRef.current.code;
    }
    // mode-2: 양방향 TTS
    return true;
  };

  // TTS 재생 시 대상 언어 결정
  const getTTSTargetLang = (detectedLang: string): string => {
    // 감지된 언어의 반대 언어로 번역 읽기
    if (detectedLang === myLangRef.current.code) return targetLangRef.current.code;
    return myLangRef.current.code;
  };

  // 번역 대상 언어 결정 (감지 언어 → 반대 언어)
  const getTranslationTarget = (detectedLang: string): string => {
    if (detectedLang === myLangRef.current.code) return targetLangRef.current.code;
    return myLangRef.current.code;
  };

  const addTranscription = (lang: string, text: string, serverTranslation?: string) => {
    entryIdRef.current += 1;
    const id = entryIdRef.current.toString();
    const translTarget = getTranslationTarget(lang);

    setTranscriptions((prev) => [...prev, {
      id,
      language: lang,
      text,
      translatedText: serverTranslation || '',
      timestamp: Date.now(),
    }]);

    // 서버 번역이 있으면 TTS 재생 (모드 조건 확인)
    if (serverTranslation && shouldPlayTTS(lang)) {
      speakTranslation(serverTranslation, getTTSTargetLang(lang));
    }

    // 서버 번역이 없으면 클라이언트에서 번역
    if (!serverTranslation) {
      if (translTarget && translTarget !== lang) {
        translateText(text, lang, translTarget).then((translated) => {
          if (translated) {
            setTranscriptions((prev) =>
              prev.map((item) => item.id === id ? { ...item, translatedText: translated } : item)
            );
            if (shouldPlayTTS(lang)) {
              speakTranslation(translated, getTTSTargetLang(lang));
            }
          }
        });
      }
    }
  };

  const handleMessage = useRef((message: any) => {
    const text = (message.original || '').trim();
    if (!text) return;

    // TTS로 말한 텍스트가 마이크에 잡힌 경우 무시
    // 단어 단위로 50% 이상 겹치면 TTS 피드백으로 판단
    if (lastSpokenTextRef.current) {
      const spokenWords = lastSpokenTextRef.current.toLowerCase().split(/\s+/);
      const incomingWords = text.toLowerCase().split(/\s+/);
      const matchCount = spokenWords.filter(w => incomingWords.includes(w)).length;
      if (spokenWords.length > 0 && matchCount / spokenWords.length >= 0.5) {
        return;
      }
    }

    const lang = langToCode(message.language || 'auto');

    if (message.type === 'partial') {
      // partial: 화면에 표시하지 않음 (서버가 final로 분절해서 보냄)
    } else if (message.type === 'final') {
      // final: 서버가 분절한 확정 문장 → 바로 표시
      const serverTranslation = (message.translation || '').trim();
      addTranscription(lang, text, serverTranslation || undefined);
      setDisplayText(null);
    }
  });

  useEffect(() => {
    if (initialMessage && !processedInitialRef.current) {
      processedInitialRef.current = true;
      handleMessage.current(initialMessage);
    }
  }, []);

  // LoadingScreen에서 넘어올 때 LiveAudioStream이 정리되므로 다시 시작
  useEffect(() => {
    startRecording();
    return () => {
      stopRecording();
      disconnect();
      Speech.stop();
    };
  }, []);

  // 앱이 백그라운드로 갈 때 정지, 돌아오면 일시정지 상태로 전환
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      if (nextAppState === 'background' || nextAppState === 'inactive') {
        if (!isPausedRef.current) {
          stopRecording();
          disconnect();
          setIsPaused(true);
        }
      }
    });
    return () => subscription.remove();
  }, []);

  // 리스너 패턴으로 메시지 수신 (React 배칭으로 인한 메시지 유실 방지)
  useEffect(() => {
    const unsubscribe = addMessageListener((message: any) => {
      handleMessage.current(message);
    });
    return unsubscribe;
  }, [addMessageListener]);

  const handleStopResume = async () => {
    if (isPaused) {
      setDisplayText(null);
      // 연결이 끊어졌으면 재연결
      if (!isConnectedRef.current) {
        await connect({ lang: myLang.code, targetLang: targetLang.code });
      }
      await startRecording();
      setIsPaused(false);
    } else {
      await stopRecording();
      setIsPaused(true);
    }
  };

  const handleGoBack = () => {
    Alert.alert(
      '대화 종료',
      '대화를 종료하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '종료',
          style: 'destructive',
          onPress: async () => {
            await stopRecording();
            disconnect();
            setTranscriptions([]);
            setDisplayText(null);
            navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* 상단 좌측: 모드 변경 버튼 */}
      <TouchableOpacity
        style={styles.topLeftButton}
        onPress={() => {
          const idx = CONVERSATION_MODES.findIndex(m => m.id === currentMode);
          const next = CONVERSATION_MODES[(idx + 1) % CONVERSATION_MODES.length];
          setCurrentMode(next.id);
          // 모드 변경 시 TTS 중단
          Speech.stop();
          lastSpokenTextRef.current = '';
        }}
      >
        <Ionicons
          name={currentMode === 'mode-0' ? 'phone-landscape-outline' : currentMode === 'mode-1' ? 'ear-outline' : 'headset-outline'}
          size={22}
          color={COLORS.textMuted}
        />
        <Text style={styles.modeLabel}>
          {CONVERSATION_MODES.find(m => m.id === currentMode)?.name || ''}
        </Text>
      </TouchableOpacity>

      {/* 상단 우측: 전체 대화 내역 보기 */}
      <TouchableOpacity
        style={styles.topRightButton}
        onPress={() => setShowFullTranscript((prev) => !prev)}
      >
        <Ionicons
          name={showFullTranscript ? 'document-text' : 'document-text-outline'}
          size={28}
          color={showFullTranscript ? COLORS.gradientMiddle : COLORS.textMuted}
        />
      </TouchableOpacity>

      <ScrollView style={styles.transcriptionArea} contentContainerStyle={styles.transcriptionContent}>
        {currentMode === 'mode-2' && !showFullTranscript ? (
          // 모드 3 (2 이어폰): 최소 UI - 상태만 표시
          <View style={styles.emptyContainer}>
            <Ionicons name="headset-outline" size={48} color={COLORS.textMuted} />
            <Text style={[styles.emptyText, { marginTop: SPACING.md }]}>
              {!isPaused ? '대화 중...' : '대기 중...'}
            </Text>
          </View>
        ) : !displayText && transcriptions.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {!isPaused ? '말씀해 주세요...' : '대기 중...'}
            </Text>
          </View>
        ) : (
          <>
            {(() => {
              // 모드별 표시할 항목 필터링
              let filtered = transcriptions;

              if (currentMode === 'mode-1' && !showFullTranscript) {
                // 모드 2 (1 이어폰): 내 언어(myLang) 감지 항목만 화면에 표시
                // (상대방이 화면을 보므로, 내가 말한 것의 번역을 보여줌)
                filtered = filtered.filter(item => item.language === myLang.code);
              }

              // 전체 내역이 아니면 최근 MAX_VISIBLE개만
              if (!showFullTranscript) {
                filtered = filtered.slice(-MAX_VISIBLE);
              }

              // mode-0, mode-1 + 문서 아이콘 꺼짐: 번역만 표시
              const onlyTranslation = (currentMode === 'mode-0' || currentMode === 'mode-1') && !showFullTranscript;

              return filtered.map((item) => (
                <TranslationItem
                  key={item.id}
                  sourceLang={item.language}
                  targetLang={getTranslationTarget(item.language)}
                  sourceText={item.text}
                  targetText={item.translatedText}
                  isLatest={false}
                  translationOnly={onlyTranslation}
                />
              ));
            })()}
            {displayText && (() => {
              // 모드별 실시간 partial 표시 조건
              if (currentMode === 'mode-1' && !showFullTranscript) {
                // 모드 2: 내 언어 감지 시에만 화면 표시
                if (displayText.lang !== myLang.code) return null;
              }
              if (currentMode === 'mode-2' && !showFullTranscript) {
                return null; // 모드 3: 최소 UI
              }
              return (
                <TranslationItem
                  key="live"
                  sourceLang={displayText.lang}
                  targetLang=""
                  sourceText={displayText.text}
                  targetText=""
                  isLatest={true}
                />
              );
            })()}
          </>
        )}
      </ScrollView>

      <View style={styles.buttonContainer}>
        {isPaused ? (
          <View style={styles.buttonRow}>
            <GradientButton
              title="재개하기"
              onPress={handleStopResume}
              style={styles.button}
            />
            <GradientButton
              title="돌아가기"
              onPress={handleGoBack}
              style={styles.button}
            />
          </View>
        ) : (
          <GradientButton
            title="중지하기"
            onPress={handleStopResume}
          />
        )}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  topLeftButton: {
    position: 'absolute',
    top: 50,
    left: 20,
    zIndex: 10,
    flexDirection: 'row',
    alignItems: 'center',
    padding: SPACING.sm,
    gap: 6,
  },
  modeLabel: {
    fontSize: FONTS.sizes.sm,
    color: COLORS.textMuted,
    fontWeight: '500',
  },
  topRightButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    padding: SPACING.sm,
  },
  transcriptionArea: {
    flex: 1,
    paddingHorizontal: SPACING.md,
  },
  transcriptionContent: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingTop: SPACING.xxl,
    paddingBottom: SPACING.md,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: SPACING.xxl,
  },
  emptyText: {
    fontSize: FONTS.sizes.lg,
    color: COLORS.textMuted,
  },
  buttonContainer: {
    paddingHorizontal: SPACING.xl,
    paddingTop: SPACING.md,
    paddingBottom: 80,
    alignItems: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: SPACING.lg,
  },
  button: {
    minWidth: 120,
  },
});

export default ConversationScreen;
