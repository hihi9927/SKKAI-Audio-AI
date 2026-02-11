import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  Alert,
  AppState,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { TranslationItem } from '../components/TranslationItem';
import { GradientButton } from '../components/GradientButton';
import { COLORS, FONTS, SPACING } from '../constants/theme';
import { Language } from '../constants/languages';
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

const MAX_VISIBLE = 2;

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
  const [displayText, setDisplayText] = useState<{ lang: string; text: string } | null>(null);
  const [transcriptions, setTranscriptions] = useState<TranscriptionEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);

  const { isConnected, connect, sendAudio, disconnect, addMessageListener } = useWebSocketContext();
  const isConnectedRef = useRef(isConnected);
  const isPausedRef = useRef(isPaused);
  const sendAudioRef = useRef(sendAudio);

  useEffect(() => { isConnectedRef.current = isConnected; }, [isConnected]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { sendAudioRef.current = sendAudio; }, [sendAudio]);

  const { startRecording, stopRecording, isRecordingActive } = useAudioRecording({
    onAudioData: (audioData) => {
      if (isConnectedRef.current && !isPausedRef.current) {
        sendAudioRef.current(audioData);
      }
    },
  });

  const processedInitialRef = useRef(false);
  const entryIdRef = useRef(0);
  // 이미 확정(commit)된 텍스트 길이를 추적
  const committedLenRef = useRef(0);

  const addTranscription = (lang: string, text: string, serverTranslation?: string) => {
    entryIdRef.current += 1;
    const id = entryIdRef.current.toString();
    setTranscriptions((prev) => [...prev, {
      id,
      language: lang,
      text,
      translatedText: serverTranslation || '',
      timestamp: Date.now(),
    }].slice(-MAX_VISIBLE));

    // 서버 번역이 없으면 클라이언트에서 번역 (partial 추출 문장용)
    if (!serverTranslation) {
      const tgtCode = targetLang.code;
      if (tgtCode && tgtCode !== lang) {
        translateText(text, lang, tgtCode).then((translated) => {
          if (translated) {
            setTranscriptions((prev) =>
              prev.map((item) => item.id === id ? { ...item, translatedText: translated } : item)
            );
          }
        });
      }
    }
  };

  // partial 텍스트에서 확정된 문장을 추출
  // "안녕하세요. 제가 말을" → committedLen 이후 부분에서
  // 온점+공백+텍스트 패턴이 있으면 온점까지를 확정
  const extractCommittedSentences = (fullText: string, lang: string): string => {
    const uncommitted = fullText.slice(committedLenRef.current);
    // 온점/물음표/느낌표 뒤에 공백+텍스트가 있는 마지막 위치를 찾음
    const sentenceEndPattern = /[.?!。？！]\s+/g;
    let lastSplitIndex = -1;
    let match;
    while ((match = sentenceEndPattern.exec(uncommitted)) !== null) {
      // 온점 뒤에 실제 텍스트가 더 있는 경우만 확정
      const afterMatch = uncommitted.slice(match.index + match[0].length);
      if (afterMatch.trim().length > 0) {
        lastSplitIndex = match.index + match[0].length;
      }
    }

    if (lastSplitIndex > 0) {
      // 확정할 텍스트 (여러 문장일 수 있음)
      const confirmedPart = uncommitted.slice(0, lastSplitIndex).trim();
      // 개별 문장으로 분리해서 각각 추가
      const sentences = confirmedPart.split(/(?<=[.?!。？！])\s+/).filter(s => s.trim());
      for (const sentence of sentences) {
        addTranscription(lang, sentence);
      }
      committedLenRef.current += lastSplitIndex;
      // 남은 partial 텍스트
      return uncommitted.slice(lastSplitIndex).trim();
    }

    // 확정할 게 없으면 uncommitted 전체가 partial
    return uncommitted;
  };

  const handleMessage = useRef((message: any) => {
    const text = (message.original || '').trim();
    if (!text) return;
    const lang = langToCode(message.language || 'auto');

    if (message.type === 'partial') {
      const remaining = extractCommittedSentences(text, lang);
      if (remaining) {
        setDisplayText({ lang, text: remaining });
      } else {
        setDisplayText(null);
      }
    } else if (message.type === 'final') {
      // final: 아직 확정 안 된 나머지 텍스트를 모두 확정
      const uncommitted = text.slice(committedLenRef.current).trim();
      if (uncommitted) {
        // 서버 번역 결과가 있으면 사용
        const serverTranslation = (message.translation || '').trim();
        addTranscription(lang, uncommitted, serverTranslation || undefined);
      }
      committedLenRef.current = 0;
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
      <ScrollView style={styles.transcriptionArea} contentContainerStyle={styles.transcriptionContent}>
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
                targetLang={targetLang.code}
                sourceText={item.text}
                targetText={item.translatedText}
                isLatest={false}
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
