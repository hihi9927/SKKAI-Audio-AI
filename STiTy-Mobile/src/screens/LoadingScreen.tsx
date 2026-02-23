import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { GradientButton } from '../components/GradientButton';
import { COLORS, FONTS, SPACING } from '../constants/theme';
import { Language } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';
import { useAudioRecording } from '../hooks/useAudioRecording';

// Qwen3-ASR 전용 - 번역 없이 ASR만
type RootStackParamList = {
  Home: undefined;
  Loading: { myLang: Language; targetLang: Language; mode: string };  // HomeScreen 호환
  Conversation: { myLang: Language; targetLang: Language; mode: string; initialMessage?: any };  // HomeScreen 호환
};

type LoadingScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Loading'>;
type LoadingScreenRouteProp = RouteProp<RootStackParamList, 'Loading'>;

interface LoadingScreenProps {
  navigation: LoadingScreenNavigationProp;
  route: LoadingScreenRouteProp;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({ navigation, route }) => {
  const { myLang, targetLang, mode } = route.params;
  const [status, setStatus] = useState<'connecting' | 'waiting' | 'error' | 'success'>('connecting');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const { connect, disconnect, sendAudio, addMessageListener } = useWebSocketContext();
  const hasNavigated = useRef(false);
  const firstFinalMessageRef = useRef<any>(null);

  const { startRecording, stopRecording } = useAudioRecording({
    onAudioData: (audioData) => {
      sendAudio(audioData);
    },
  });

  useEffect(() => {
    // 리스너로 첫 메시지 감지 (lastMessage 잔존값 문제 방지)
    const unsubscribe = addMessageListener((message: any) => {
      if (!hasNavigated.current && message.type === 'final') {
        firstFinalMessageRef.current = message;
        setStatus('success');
        hasNavigated.current = true;

        setTimeout(() => {
          navigation.replace('Conversation', {
            myLang,
            targetLang,
            mode,
            initialMessage: firstFinalMessageRef.current,
          });
        }, 300);
      }
    });

    connectToServer();

    return () => {
      unsubscribe();
      if (!hasNavigated.current) {
        stopRecording();
        disconnect();
      }
    };
  }, []);

  const connectToServer = async () => {
    setStatus('connecting');
    setErrorMessage('');

    try {
      // Qwen3-ASR 서버에 연결
      await connect({
        lang: myLang.code,  // 'auto', 'ko', 'en' 등
        targetLang: targetLang.code,  // 번역 대상 언어
      });

      // 연결 성공 → 녹음 시작, 첫 출력 대기
      await startRecording();
      setStatus('waiting');

    } catch (error: any) {
      setStatus('error');
      setErrorMessage(error.message || '연결에 실패했습니다');
    }
  };

  const handleGoBack = () => {
    stopRecording();
    disconnect();
    navigation.goBack();
  };

  const handleRetry = () => {
    connectToServer();
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {(status === 'connecting' || status === 'waiting') && (
          <>
            <Text style={styles.loadingText}>로딩 중 ...</Text>
            <Text style={styles.hintText}>말을 시작해주세요</Text>
            <ActivityIndicator
              size="large"
              color={COLORS.gradientMiddle}
              style={styles.spinner}
            />
          </>
        )}

        {status === 'error' && (
          <>
            <Text style={styles.errorText}>
              {errorMessage || '현재 이용자가 가득 차 이용이 불가합니다'}
            </Text>
          </>
        )}

        {status === 'success' && (
          <>
            <Text style={styles.successText}>연결 성공!</Text>
            <Text style={styles.hintText}>대화 화면으로 이동합니다...</Text>
          </>
        )}

        <View style={styles.buttonContainer}>
          {status === 'error' ? (
            <View style={styles.buttonRow}>
              <GradientButton
                title="재시도"
                onPress={handleRetry}
              />
              <GradientButton
                title="돌아가기"
                onPress={handleGoBack}
              />
            </View>
          ) : (
            <GradientButton
              title="돌아가기"
              onPress={handleGoBack}
            />
          )}
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: SPACING.xl,
  },
  loadingText: {
    fontSize: FONTS.sizes.xl,
    fontWeight: '600',
    color: COLORS.gradientMiddle,
    marginBottom: SPACING.sm,
  },
  hintText: {
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
    marginBottom: SPACING.lg,
  },
  spinner: {
    marginVertical: SPACING.xl,
  },
  errorText: {
    fontSize: FONTS.sizes.md,
    color: COLORS.textPrimary,
    textAlign: 'center',
    marginBottom: SPACING.xl,
  },
  successText: {
    fontSize: FONTS.sizes.xl,
    fontWeight: '600',
    color: COLORS.success,
    marginBottom: SPACING.sm,
  },
  buttonContainer: {
    position: 'absolute',
    bottom: '28%',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: SPACING.lg,
  },
});

export default LoadingScreen;
