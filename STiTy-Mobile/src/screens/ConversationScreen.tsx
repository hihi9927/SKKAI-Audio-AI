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

const MAX_VISIBLE = 3;

export const ConversationScreen: React.FC<ConversationScreenProps> = ({ navigation, route }) => {
  const { myLang, initialMessage } = route.params;
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

  const handleMessage = useRef((message: any) => {
    const text = (message.original || '').trim();
    if (!text) return;
    const lang = langToCode(message.language || 'auto');

    if (message.type === 'partial') {
      // 웹앱과 동일: original 텍스트를 그대로 표시
      setDisplayText({ lang, text });
    } else if (message.type === 'final') {
      // final: 확정된 텍스트를 기록에 추가
      entryIdRef.current += 1;
      setTranscriptions((prev) => [...prev, {
        id: entryIdRef.current.toString(),
        language: lang,
        text,
        timestamp: Date.now(),
      }].slice(-MAX_VISIBLE));
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
        await connect({ lang: myLang.code });
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
                targetLang=""
                sourceText={item.text}
                targetText=""
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
    paddingTop: SPACING.xl,
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
