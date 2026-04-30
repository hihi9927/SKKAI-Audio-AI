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
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { initTtsEngine, ttsSpeak, ttsStop } from '../utils/tts';
import { setSpeakerphoneOn, releaseAudioMode } from '../utils/audioRouting';
import { Ionicons } from '@expo/vector-icons';
import { TranslationItem } from '../components/TranslationItem';
import { GradientButton } from '../components/GradientButton';
import { COLORS, FONTS, SPACING } from '../constants/theme';
import { Language, CONVERSATION_MODES } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';
import { useAudioRecording } from '../hooks/useAudioRecording';

type RootStackParamList = {
  Home: undefined;
  Conversation: { myLang: Language; targetLang: Language; mode: string };
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
  const { myLang, targetLang } = route.params;
  const [currentMode, setCurrentMode] = useState(route.params.mode);
  const [displayText, setDisplayText] = useState<{ lang: string; text: string } | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [isTTSMuted, setIsTTSMuted] = useState(false);
  const isTTSMutedRef = useRef(isTTSMuted);
  const scrollRef = useRef<ScrollView>(null);

  // ── 연결/초기화 상태 관리 ──
  const [sessionStatus, setSessionStatus] = useState<'ready' | 'error'>('ready');
  const [connectionError, setConnectionError] = useState<string>('');


  const { isConnected, connect, sendAudio, disconnect, addMessageListener, sendMessage, serverStatus, probeServer } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const isPausedRef = useRef(isPaused);
  const serverStatusRef = useRef(serverStatus);

  const [isInitializing, setIsInitializing] = useState(false);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressAnimRunning = useRef<Animated.CompositeAnimation | null>(null);
  const sendAudioRef = useRef(sendAudio);
  const sendMessageRef = useRef(sendMessage);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { serverStatusRef.current = serverStatus; }, [serverStatus]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);
  useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);
  useEffect(() => { isTTSMutedRef.current = isTTSMuted; }, [isTTSMuted]);

  const modeRef = useRef(currentMode);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);

  useEffect(() => { modeRef.current = currentMode; }, [currentMode]);

  const ttsQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef(false);

  // serverStatus에 따라 진행 바 애니메이션
  useEffect(() => {
    if (serverStatus === 'connecting') {
      progressAnim.setValue(0);
      progressAnimRunning.current = Animated.timing(progressAnim, {
        toValue: 0.85,
        duration: 80000,
        useNativeDriver: false,
      });
      progressAnimRunning.current.start();
    } else if (serverStatus === 'ready') {
      progressAnimRunning.current?.stop();
      Animated.timing(progressAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: false,
      }).start();
    }
  }, [serverStatus]);

  // TTS 엔진 초기화 (Samsung 기기 대응)
  useEffect(() => {
    void initTtsEngine();
  }, []);

  const processNextTTS = () => {
    if (ttsQueueRef.current.length === 0) {
      isSpeakingRef.current = false;
      return;
    }
    isSpeakingRef.current = true;
    const next = ttsQueueRef.current.shift()!;
    const ttsStart = new Date().toISOString();

    const isEarphoneMode = modeRef.current === 'mode-2';
    // 매 TTS마다 오디오 모드 설정: 스피커=MODE_IN_COMMUNICATION, 이어폰=MODE_NORMAL(BT A2DP 활성화)
    setSpeakerphoneOn(!isEarphoneMode);
    ttsSpeak(next.text, next.lang, 1.3,
      () => {
        sendMessageRef.current({ type: 'tts_log', text: next.text, lang: next.lang, start: ttsStart, end: new Date().toISOString() });
        processNextTTS();
      },
      () => processNextTTS(),
      isEarphoneMode,
    );
  };

  const speakTranslation = (text: string, lang: string) => {
    if (isTTSMutedRef.current || !text) return;
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

  const entryIdRef = useRef(0);

  const shouldPlayTTS = (detectedLang: string): boolean => {
    if (isTTSMutedRef.current) return false;
    if (modeRef.current === 'mode-1') return true;
    if (modeRef.current === 'mode-2') return detectedLang !== myLangRef.current.code;
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
      // serverTranslation === text 이면 서버가 번역 대신 원문을 그대로 반환한 것(예: zh→zh)
      // 이 경우 undefined로 처리해 클라이언트 측 Google Translate 폴백을 실행
      const validTranslation = (serverTranslation && serverTranslation !== text) ? serverTranslation : undefined;
      addTranscription(lang, text, validTranslation);
      setDisplayText(null);

    }
  });

  // ── 화면 진입 시 WebSocket 연결 + 녹음 시작 ──
  useEffect(() => {
    const initSession = async () => {
      setConnectionError('');
      try {
        if (serverStatusRef.current !== 'ready') {
          setIsInitializing(true);
          await probeServer();
        }
        await connect({ lang: myLang.code, targetLang: targetLang.code });
        await startRecording();
        setSpeakerphoneOn(modeRef.current !== 'mode-2');
      } catch (error: any) {
        console.error('Connection failed:', error);
        setSessionStatus('error');
        setConnectionError(error.message || '연결에 실패했습니다');
      } finally {
        setIsInitializing(false);
      }
    };

    initSession();

    return () => {
      stopRecording();
      disconnect();
      ttsQueueRef.current = [];
      isSpeakingRef.current = false;
      ttsStop();
      releaseAudioMode();
    };
  }, []);

  // 앱 백그라운드/포그라운드 처리
  useEffect(() => {
    const prevStateRef = { current: AppState.currentState };

    const subscription = AppState.addEventListener('change', (nextAppState) => {
      const prevState = prevStateRef.current;
      prevStateRef.current = nextAppState;

      if (nextAppState === 'background' || nextAppState === 'inactive') {
        if (!isPausedRef.current) {
          stopRecording();
          disconnect();
          setIsPaused(true);
        }
      } else if (nextAppState === 'active' && (prevState === 'background' || prevState === 'inactive')) {
        if (isPausedRef.current) {
          setDisplayText(null);
          const doReconnect = async () => {
            try {
              if (serverStatusRef.current !== 'ready') {
                setIsInitializing(true);
                await probeServer();
              }
              await connect({ lang: myLang.code, targetLang: targetLang.code });
              await startRecording();
              setSpeakerphoneOn(modeRef.current !== 'mode-2');
              setIsPaused(false);
            } catch (error: any) {
              setSessionStatus('error');
              setConnectionError(error.message || '연결에 실패했습니다');
            } finally {
              setIsInitializing(false);
            }
          };
          doReconnect();
        }
      }
    });
    return () => subscription.remove();
  }, []);

  // 새 항목 추가 시 맨 아래로 스크롤
  useEffect(() => {
    setTimeout(() => {
      scrollRef.current?.scrollToEnd({ animated: false });
    }, 50);
  }, [transcriptions.length]);

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
      setSpeakerphoneOn(modeRef.current !== 'mode-2');
      setIsPaused(false);
    } else {
      ttsQueueRef.current = [];
      isSpeakingRef.current = false;
      ttsStop();
      await stopRecording();
      setIsPaused(true);
    }
  };

  const handleGoBack = () => {
    // 에러 상태에서는 확인 없이 바로 돌아가기
    if (sessionStatus === 'error') {
      stopRecording();
      disconnect();
      releaseAudioMode();
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
            releaseAudioMode();
            navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
          },
        },
      ]
    );
  };

  const handleRetry = async () => {
    setSessionStatus('ready');
    setConnectionError('');
    try {
      await connect({
        lang: myLang.code,
        targetLang: targetLang.code,
      });
      await startRecording();
    } catch (error: any) {
      setSessionStatus('error');
      setConnectionError(error.message || '연결에 실패했습니다');
    }
  };

  // ── 로딩 오버레이 (서버 시작 중 / 연결 중) ──
  if (isInitializing) {
    const isServerStarting = serverStatus !== 'connecting' && serverStatus !== 'ready';
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.overlayContainer}>
          {isServerStarting ? (
            <>
              <Text style={styles.overlayHintText}>잠시만요...</Text>
              <Text style={[styles.errorText, { marginTop: 8 }]}>서버를 시작하고 있어요</Text>
            </>
          ) : (
            <>
              <Text style={styles.overlayHintText}>연결 중...</Text>
              <View style={styles.progressTrack}>
                <Animated.View style={[styles.progressFill, {
                  width: progressAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: ['0%', '100%'],
                  }),
                }]}>
                  <LinearGradient
                    colors={['#8E54E9', '#4776E6', '#00CFEF']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={StyleSheet.absoluteFill}
                  />
                </Animated.View>
              </View>
            </>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // ── 에러 오버레이 ──
  if (sessionStatus === 'error') {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.overlayContainer}>
          <Text style={styles.errorText}>
            {connectionError || '현재 이용자가 가득 차 이용이 불가합니다'}
          </Text>
          <View style={styles.overlayButtonRow}>
            <GradientButton title="재시도" onPress={handleRetry} />
            <GradientButton title="돌아가기" onPress={handleGoBack} />
          </View>
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
          setSpeakerphoneOn(next.id !== 'mode-2');
          ttsQueueRef.current = [];
          isSpeakingRef.current = false;
          ttsStop();
        }}
      >
        <Ionicons
          name={currentMode === 'mode-1' ? 'phone-landscape-outline' : 'headset-outline'}
          size={22}
          color={COLORS.textMuted}
        />
        <Text style={styles.modeLabel}>
          {CONVERSATION_MODES.find(m => m.id === currentMode)?.name || ''}
        </Text>
      </TouchableOpacity>

      {/* 상단 우측: mode-1일 때 스피커 토글 */}
      {currentMode === 'mode-1' && (
        <View style={styles.topRightButtons}>
          <TouchableOpacity onPress={() => {
            setIsTTSMuted((prev) => {
              const next = !prev;
              isTTSMutedRef.current = next;
              if (next) { ttsQueueRef.current = []; isSpeakingRef.current = false; ttsStop(); }
              return next;
            });
          }}>
            <Ionicons
              name={isTTSMuted ? 'volume-mute-outline' : 'volume-high-outline'}
              size={20}
              color={isTTSMuted ? COLORS.gradientMiddle : COLORS.textMuted}
            />
          </TouchableOpacity>
        </View>
      )}

      <ScrollView
        ref={scrollRef}
        style={styles.transcriptionArea}
        contentContainerStyle={styles.transcriptionContent}
      >
        {!displayText && transcriptions.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {!isPaused ? '말씀해 주세요...' : '대기 중...'}
            </Text>
          </View>
        ) : (
          <>
            {transcriptions.map((item) => (
              <TranslationItem
                key={item.id}
                sourceLang={item.language}
                targetLang={getTranslationTarget(item.language)}
                sourceText={item.text}
                targetText={item.translatedText}
                isLatest={false}
                translationOnly={true}
              />
            ))}
            {displayText && (
              <TranslationItem
                key="live"
                sourceLang={displayText.lang}
                targetLang=""
                sourceText={displayText.text}
                targetText=""
                isLatest={true}
              />
            )}
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
  topRightButtons: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: SPACING.sm,
  },
  transcriptionArea: {
    flex: 1,
    marginTop: 90,
    paddingHorizontal: SPACING.md,
  },
  transcriptionContent: {
    flexGrow: 1,
    justifyContent: 'flex-end',
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
  progressTrack: {
    width: '75%',
    height: 6,
    backgroundColor: COLORS.border,
    borderRadius: 3,
    marginTop: SPACING.xl,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
    overflow: 'hidden',
  },
});

export default ConversationScreen;
