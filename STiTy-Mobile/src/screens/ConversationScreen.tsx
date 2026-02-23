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
  ActivityIndicator,
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
  const [showFullTranscript, setShowFullTranscript] = useState(route.params.mode === 'mode-2');

  // ── 연결/초기화 상태 관리 ──
  // 'connecting': WebSocket 연결 중
  // 'waiting': 연결 완료, 첫 전사 대기 중 (아무말이나 해주세요)
  // 'ready': 첫 전사 도착, 정상 대화 모드
  // 'error': 연결 실패
  const [sessionStatus, setSessionStatus] = useState<'connecting' | 'ready' | 'error'>('connecting');
  const [connectionError, setConnectionError] = useState<string>('');

  // 모드에 따라 TTS 자동 설정
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

  const modeRef = useRef(currentMode);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);

  useEffect(() => { modeRef.current = currentMode; }, [currentMode]);
  useEffect(() => {
    if (currentMode === 'mode-2') {
      setShowFullTranscript(true);
    }
  }, [currentMode]);

  const ttsQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef(false);

  const processNextTTS = () => {
    if (ttsQueueRef.current.length === 0) {
      isSpeakingRef.current = false;
      return;
    }
    isSpeakingRef.current = true;
    const next = ttsQueueRef.current.shift()!;
    Speech.speak(next.text, {
      language: next.lang,
      rate: 1.0,
      onDone: processNextTTS,
      onError: processNextTTS,
      onStopped: () => { isSpeakingRef.current = false; },
    });
  };

  const speakTranslation = (text: string, lang: string) => {
    if (!isTTSEnabledRef.current || !text) return;
    ttsQueueRef.current.push({ text, lang });
    if (!isSpeakingRef.current) {
      processNextTTS();
    }
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

  const shouldPlayTTS = (detectedLang: string): boolean => {
    if (!isTTSEnabledRef.current) return false;
    const currentMode = modeRef.current;
    if (currentMode === 'mode-0') return false;
    if (currentMode === 'mode-1' || currentMode === 'mode-2') {
      return detectedLang !== myLangRef.current.code;
    }
    return false;
  };

  const getTTSTargetLang = (detectedLang: string): string => {
    if (detectedLang !== myLangRef.current.code) return myLangRef.current.code;
    return targetLangRef.current.code;
  };

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

    if (serverTranslation && shouldPlayTTS(lang)) {
      speakTranslation(serverTranslation, getTTSTargetLang(lang));
    }

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

    const lang = langToCode(message.language || 'auto');

    if (message.type === 'partial') {
      // partial: 화면에 표시하지 않음
    } else if (message.type === 'final') {
      const serverTranslation = (message.translation || '').trim();
      addTranscription(lang, text, serverTranslation || undefined);
      setDisplayText(null);

      // ── 첫 전사 도착 시 힌트 오버레이 제거 ──
      setSessionStatus((prev) => {
        if (prev === 'connecting') return 'ready';
        return prev;
      });
    }
  });

  // ── 화면 진입 시 WebSocket 연결 + 녹음 시작 ──
  useEffect(() => {
    const initSession = async () => {
      setSessionStatus('connecting');
      setConnectionError('');

      try {
        await connect({
          lang: myLang.code,
          targetLang: targetLang.code,
        });
        await startRecording();
        setSessionStatus('ready');
      } catch (error: any) {
        console.error('Connection failed:', error);
        setSessionStatus('error');
        setConnectionError(error.message || '연결에 실패했습니다');
      }
    };

    initSession();

    return () => {
      stopRecording();
      disconnect();
      ttsQueueRef.current = [];
      isSpeakingRef.current = false;
      Speech.stop();
    };
  }, []);

  // initialMessage 처리 (LoadingScreen에서 넘어온 경우 호환)
  useEffect(() => {
    if (initialMessage && !processedInitialRef.current) {
      processedInitialRef.current = true;
      handleMessage.current(initialMessage);
    }
  }, []);

  // 앱 백그라운드 처리
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

  // 메시지 리스너
  useEffect(() => {
    const unsubscribe = addMessageListener((message: any) => {
      handleMessage.current(message);
    });
    return unsubscribe;
  }, [addMessageListener]);

  const handleStopResume = async () => {
    if (isPaused) {
      setDisplayText(null);
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
    // 에러 상태에서는 확인 없이 바로 돌아가기
    if (sessionStatus === 'error') {
      stopRecording();
      disconnect();
      navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
      return;
    }

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
            ttsQueueRef.current = [];
            isSpeakingRef.current = false;
            setTranscriptions([]);
            setDisplayText(null);
            navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
          },
        },
      ]
    );
  };

  const handleRetry = async () => {
    setSessionStatus('connecting');
    setConnectionError('');
    try {
      await connect({
        lang: myLang.code,
        targetLang: targetLang.code,
      });
      await startRecording();
      setSessionStatus('ready');
    } catch (error: any) {
      setSessionStatus('error');
      setConnectionError(error.message || '연결에 실패했습니다');
    }
  };

  // ── 연결 중 / 대기 중 / 에러 오버레이 ──
  if (sessionStatus === 'connecting' || sessionStatus === 'error') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.overlayContainer}>
          {sessionStatus === 'error' ? (
            <>
              <Text style={styles.errorText}>
                {connectionError || '현재 이용자가 가득 차 이용이 불가합니다'}
              </Text>
              <View style={styles.overlayButtonRow}>
                <GradientButton title="재시도" onPress={handleRetry} />
                <GradientButton title="돌아가기" onPress={handleGoBack} />
              </View>
            </>
          ) : (
            <>
              {sessionStatus === 'connecting' && (
                <>
                  <ActivityIndicator size="large" color={COLORS.gradientMiddle} style={{ marginBottom: SPACING.lg }} />
                  <Text style={styles.overlayHintText}>말을 시작해주세요</Text>
                </>
              )}
              <View style={styles.overlayBackButton}>
                <GradientButton title="돌아가기" onPress={handleGoBack} />
              </View>
            </>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // ── 정상 대화 UI (sessionStatus === 'ready') ──
  return (
    <SafeAreaView style={styles.container}>
      {/* 상단 좌측: 모드 변경 버튼 */}
      <TouchableOpacity
        style={styles.topLeftButton}
        onPress={() => {
          const idx = CONVERSATION_MODES.findIndex(m => m.id === currentMode);
          const next = CONVERSATION_MODES[(idx + 1) % CONVERSATION_MODES.length];
          setCurrentMode(next.id);
          ttsQueueRef.current = [];
          isSpeakingRef.current = false;
          Speech.stop();
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
      {currentMode !== 'mode-2' && (
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
      )}

      <ScrollView style={styles.transcriptionArea} contentContainerStyle={styles.transcriptionContent}>
        {currentMode === 'mode-2' && !showFullTranscript ? (
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
              let filtered = transcriptions;

              if (currentMode === 'mode-1' && !showFullTranscript) {
                filtered = filtered.filter(item => item.language === myLang.code);
              }

              if (!showFullTranscript) {
                filtered = filtered.slice(-MAX_VISIBLE);
              }

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
              if (currentMode === 'mode-1' && !showFullTranscript) {
                if (displayText.lang !== myLang.code) return null;
              }
              if (currentMode === 'mode-2' && !showFullTranscript) {
                return null;
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
  // ── 오버레이 (연결 중 / 대기 중 / 에러) ──
  overlayContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: SPACING.xl,
  },
  overlayHintText: {
    fontSize: FONTS.sizes.xl,
    fontWeight: '600',
    color: COLORS.textPrimary,
    textAlign: 'center',
  },
  overlayBackButton: {
    position: 'absolute',
    bottom: '28%',
  },
  overlayButtonRow: {
    flexDirection: 'row',
    gap: SPACING.lg,
    marginTop: SPACING.xl,
  },
  errorText: {
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
    textAlign: 'center',
  },
  // ── 대화 UI ──
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
