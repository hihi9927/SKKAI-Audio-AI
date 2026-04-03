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
  const { myLang, targetLang } = route.params;
  const [currentMode, setCurrentMode] = useState(route.params.mode);
  const [displayText, setDisplayText] = useState<{ lang: string; text: string } | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [showFullTranscript, setShowFullTranscript] = useState(false);
  const fullTranscriptScrollRef = useRef<ScrollView>(null);

  // ── 연결/초기화 상태 관리 ──
  // 'connecting': WebSocket 연결 중
  // 'waiting': 연결 완료, 첫 전사 대기 중 (아무말이나 해주세요)
  // 'ready': 첫 전사 도착, 정상 대화 모드
  // 'error': 연결 실패
  const [sessionStatus, setSessionStatus] = useState<'ready' | 'error'>('ready');
  const [connectionError, setConnectionError] = useState<string>('');

  // 모드에 따라 TTS 자동 설정
  const isTTSEnabled = currentMode === 'mode-2';
  const isTTSEnabledRef = useRef(isTTSEnabled);

  const { isConnected, connect, sendAudio, disconnect, addMessageListener, sendMessage } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const isPausedRef = useRef(isPaused);
  const sendAudioRef = useRef(sendAudio);
  const sendMessageRef = useRef(sendMessage);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);
  useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);
  useEffect(() => { isTTSEnabledRef.current = isTTSEnabled; }, [isTTSEnabled]);

  const modeRef = useRef(currentMode);
  const myLangRef = useRef(myLang);
  const targetLangRef = useRef(targetLang);

  useEffect(() => { modeRef.current = currentMode; }, [currentMode]);

  const ttsQueueRef = useRef<{ text: string; lang: string }[]>([]);
  const isSpeakingRef = useRef(false);
  const ttsTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Samsung TTS 등 일부 엔진은 'ko' 같은 짧은 코드를 인식 못 해 silent fail함.
  // BCP-47 full tag로 변환해 호환성 확보.
  const toBCP47 = (code: string): string => {
    const map: Record<string, string> = {
      ko: 'ko-KR', en: 'en-US', ja: 'ja-JP', zh: 'zh-CN',
      id: 'id-ID', vi: 'vi-VN', th: 'th-TH',
      es: 'es-ES', fr: 'fr-FR', de: 'de-DE',
    };
    return map[code] ?? code;
  };

  const processNextTTS = () => {
    if (ttsQueueRef.current.length === 0) {
      isSpeakingRef.current = false;
      return;
    }
    isSpeakingRef.current = true;
    const next = ttsQueueRef.current.shift()!;
    const ttsStart = new Date().toISOString();

    // Samsung TTS 등 일부 엔진이 onDone/onError를 호출하지 않는 경우 대비.
    // 예상 재생 시간(글자 수 × 60ms, 최소 3초) 후 강제로 다음 항목 진행.
    const estimatedMs = Math.max(3000, next.text.length * 60);
    if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
    ttsTimeoutRef.current = setTimeout(() => {
      if (isSpeakingRef.current) {
        isSpeakingRef.current = false;
        processNextTTS();
      }
    }, estimatedMs);

    Speech.speak(next.text, {
      language: toBCP47(next.lang),
      rate: 1.3,
      onDone: () => {
        if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
        sendMessageRef.current({ type: 'tts_log', text: next.text, lang: next.lang, start: ttsStart, end: new Date().toISOString() });
        processNextTTS();
      },
      onError: () => {
        if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
        processNextTTS();
      },
      onStopped: () => {
        if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
        isSpeakingRef.current = false;
        processNextTTS();
      },
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

  const entryIdRef = useRef(0);

  const shouldPlayTTS = (detectedLang: string): boolean => {
    if (!isTTSEnabledRef.current) return false;
    if (modeRef.current === 'mode-2') {
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

    }
  });

  // ── 화면 진입 시 WebSocket 연결 + 녹음 시작 ──
  useEffect(() => {
    const initSession = async () => {
      setConnectionError('');

      try {
        await connect({
          lang: myLang.code,
          targetLang: targetLang.code,
        });
        await startRecording();
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
      if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
      Speech.stop();
    };
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

  // 전체 내역 열릴 때 + 새 항목 추가 시 맨 아래로 스크롤
  useEffect(() => {
    if (showFullTranscript) {
      setTimeout(() => {
        fullTranscriptScrollRef.current?.scrollToEnd({ animated: false });
      }, 50);
    }
  }, [showFullTranscript, transcriptions.length]);

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
            if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
            setTranscriptions([]);
            setDisplayText(null);
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
          ttsQueueRef.current = [];
          isSpeakingRef.current = false;
          if (ttsTimeoutRef.current) clearTimeout(ttsTimeoutRef.current);
          Speech.stop();
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

      {/* 상단 우측: 전체 대화 내역 보기 */}
      <TouchableOpacity
        style={styles.topRightButton}
        onPress={() => setShowFullTranscript((prev) => !prev)}
      >
        <Ionicons
          name={showFullTranscript ? 'document-text' : 'document-text-outline'}
          size={20}
          color={showFullTranscript ? COLORS.gradientMiddle : COLORS.textMuted}
        />
      </TouchableOpacity>

      <ScrollView
        ref={showFullTranscript ? fullTranscriptScrollRef : undefined}
        style={styles.transcriptionArea}
        contentContainerStyle={[
          styles.transcriptionContent,
          showFullTranscript && { justifyContent: 'flex-start' },
        ]}
      >
        {!displayText && transcriptions.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {!isPaused ? '말씀해 주세요...' : '대기 중...'}
            </Text>
          </View>
        ) : (
          <>
            {(() => {
              let filtered = transcriptions;

              if (currentMode === 'mode-2' && !showFullTranscript) {
                filtered = filtered.filter(item => item.language === myLang.code);
              }

              if (!showFullTranscript) {
                filtered = filtered.slice(-MAX_VISIBLE);
              }

              const onlyTranslation = currentMode === 'mode-1' && !showFullTranscript;

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
              if (currentMode === 'mode-2' && !showFullTranscript) {
                if (displayText.lang !== myLang.code) return null;
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
