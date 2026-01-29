import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  FlatList,
  Alert,
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

interface TranslationEntry {
  id: string;
  sourceLang: string;
  targetLang: string;
  sourceText: string;
  targetText: string;
  timestamp: number;
}

export const ConversationScreen: React.FC<ConversationScreenProps> = ({ navigation, route }) => {
  const { myLang, targetLang, mode, initialMessage } = route.params;
  const [translations, setTranslations] = useState<TranslationEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const { isConnected, sendAudio, lastMessage, disconnect } = useWebSocketContext();
  const { startRecording, stopRecording, isRecordingActive } = useAudioRecording({
    onAudioData: (audioData) => {
      if (isConnected && !isPaused) {
        sendAudio(audioData);
      }
    },
  });

  // LoadingScreen에서 이미 연결+녹음이 시작된 상태로 진입

  // 초기 메시지 처리 (로딩 화면에서 받은 첫 번째 출력)
  const processedInitialRef = useRef(false);

  useEffect(() => {
    if (initialMessage && !processedInitialRef.current) {
      processedInitialRef.current = true;
      handleIncomingMessage(initialMessage);
    }
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
      disconnect();
    };
  }, []);

  useEffect(() => {
    // initialMessage와 동일한 메시지면 중복 처리 방지
    if (lastMessage && lastMessage !== initialMessage) {
      handleIncomingMessage(lastMessage);
    }
  }, [lastMessage]);

  const handleIncomingMessage = (message: any) => {
    if (message.type === 'final' && message.original) {
      // 서버가 보내는 필드: original, translation, language
      const detectedLang = message.language || myLang.code;
      // 번역 대상 언어: 감지된 언어가 내 언어면 상대 언어로, 아니면 내 언어로
      const translationTargetLang = detectedLang === myLang.code ? targetLang.code : myLang.code;

      const newEntry: TranslationEntry = {
        id: Date.now().toString(),
        sourceLang: detectedLang,
        targetLang: translationTargetLang,
        sourceText: message.original,
        targetText: message.translation || '',  // 서버는 'translation' 필드 사용
        timestamp: Date.now(),
      };

      setTranslations((prev) => [newEntry, ...prev]);

      setTimeout(() => {
        flatListRef.current?.scrollToOffset({ offset: 0, animated: true });
      }, 100);
    }
  };

  const handleStopResume = async () => {
    if (isPaused) {
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
            navigation.navigate('Home');
          },
        },
      ]
    );
  };

  const renderTranslationItem = ({ item, index }: { item: TranslationEntry; index: number }) => (
    <TranslationItem
      sourceLang={item.sourceLang}
      targetLang={item.targetLang}
      sourceText={item.sourceText}
      targetText={item.targetText}
      isLatest={index === 0}
    />
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.translationContainer}>
        {translations.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {!isPaused ? '말씀해 주세요...' : '대기 중...'}
            </Text>
          </View>
        ) : (
          <FlatList
            ref={flatListRef}
            data={translations}
            renderItem={renderTranslationItem}
            keyExtractor={(item) => item.id}
            inverted={false}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={styles.listContent}
            ItemSeparatorComponent={() => <View style={styles.separator} />}
          />
        )}
      </View>

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
  translationContainer: {
    flex: 1,
    paddingHorizontal: SPACING.md,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: FONTS.sizes.lg,
    color: COLORS.textMuted,
  },
  listContent: {
    paddingVertical: SPACING.lg,
  },
  separator: {
    height: SPACING.lg,
  },
  buttonContainer: {
    paddingHorizontal: SPACING.xl,
    alignItems: 'center',
    position: 'absolute',
    bottom: '28%',
    alignSelf: 'center',
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
