import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  StatusBar,
  Alert,
  TouchableOpacity,
  Modal,
  FlatList,
  Animated,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import MaskedView from '@react-native-masked-view/masked-view';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LANGUAGES, Language, CONVERSATION_MODES, formatLanguageDisplay, formatLanguageAs } from '../constants/languages';
import { useWebSocketContext } from '../context/WebSocketContext';

type RootStackParamList = {
  Home: undefined;
  Conversation: { myLang: Language; targetLang: Language; mode: string };
};

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

interface HomeScreenProps {
  navigation: HomeScreenNavigationProp;
}

const STORAGE_KEYS = {
  MY_LANG: 'stity_myLang',
  TARGET_LANG: 'stity_targetLang',
  MODE: 'stity_mode',
};

const GradientText: React.FC<{ text: string; style?: any }> = ({ text, style }) => {
  return (
    <MaskedView
      maskElement={<Text style={[{ fontSize: 16 }, style]}>{text}</Text>}
    >
      <LinearGradient
        colors={['#8E54E9', '#4776E6', '#00CFEF']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
      >
        <Text style={[{ fontSize: 16 }, style, { opacity: 0 }]}>{text}</Text>
      </LinearGradient>
    </MaskedView>
  );
};

const LanguagePickerModal: React.FC<{
  visible: boolean;
  onClose: () => void;
  onSelect: (lang: Language) => void;
  selectedCode: string;
  excludeCode?: string;
  title: string;
  displayAsLangCode?: string;
}> = ({ visible, onClose, onSelect, selectedCode, excludeCode, title, displayAsLangCode }) => {
  const filtered = excludeCode
    ? LANGUAGES.filter(l => l.code !== excludeCode)
    : LANGUAGES;

  const getDisplayName = (item: Language) => {
    if (displayAsLangCode) {
      return formatLanguageAs(item, displayAsLangCode);
    }
    return `${item.nativeName}(${item.code})`;
  };

  return (
    <Modal
      animationType="slide"
      transparent
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={modalStyles.overlay}>
        <View style={modalStyles.content}>
          <View style={modalStyles.header}>
            <Text style={modalStyles.title}>{title}</Text>
            <TouchableOpacity onPress={onClose} style={modalStyles.closeBtn}>
              <Text style={modalStyles.closeText}>✕</Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={filtered}
            keyExtractor={(item) => item.code}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={[
                  modalStyles.item,
                  item.code === selectedCode && modalStyles.itemSelected,
                ]}
                onPress={() => { onSelect(item); onClose(); }}
              >
                <Text style={[
                  modalStyles.itemText,
                  item.code === selectedCode && modalStyles.itemTextSelected,
                ]}>
                  {getDisplayName(item)}
                </Text>
              </TouchableOpacity>
            )}
            ItemSeparatorComponent={() => <View style={modalStyles.sep} />}
          />
        </View>
      </View>
    </Modal>
  );
};

const ModePickerModal: React.FC<{
  visible: boolean;
  onClose: () => void;
  onSelect: (mode: typeof CONVERSATION_MODES[0]) => void;
  selectedId: string;
}> = ({ visible, onClose, onSelect, selectedId }) => {
  return (
    <Modal
      animationType="slide"
      transparent
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={modalStyles.overlay}>
        <View style={modalStyles.content}>
          <View style={modalStyles.header}>
            <Text style={modalStyles.title}>대화방식 선택</Text>
            <TouchableOpacity onPress={onClose} style={modalStyles.closeBtn}>
              <Text style={modalStyles.closeText}>✕</Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={CONVERSATION_MODES}
            keyExtractor={(item) => item.id}
            contentContainerStyle={modalStyles.modeListContent}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={[
                  modalStyles.item,
                  item.id === selectedId && modalStyles.itemSelected,
                ]}
                onPress={() => { onSelect(item); onClose(); }}
              >
                <Text style={[
                  modalStyles.itemText,
                  item.id === selectedId && modalStyles.itemTextSelected,
                ]}>
                  {item.name}
                </Text>
                <Text style={modalStyles.itemDesc}>
                  {item.description}
                </Text>
              </TouchableOpacity>
            )}
            ItemSeparatorComponent={() => <View style={modalStyles.sep} />}
          />
        </View>
      </View>
    </Modal>
  );
};

export const HomeScreen: React.FC<HomeScreenProps> = ({ navigation }) => {
  const [myLanguage, setMyLanguage] = useState<Language>(LANGUAGES[0]);
  const [targetLanguage, setTargetLanguage] = useState<Language>(LANGUAGES[7]);
  const [conversationMode, setConversationMode] = useState(CONVERSATION_MODES[0]);
  const [loaded, setLoaded] = useState(false);

  const [myLangModal, setMyLangModal] = useState(false);
  const [targetLangModal, setTargetLangModal] = useState(false);
  const [modeModal, setModeModal] = useState(false);

  const { serverStatus, probeServer } = useWebSocketContext();
  const progressAnim = useRef(new Animated.Value(0)).current;
  const slowAnimRef = useRef<Animated.CompositeAnimation | null>(null);

  // 마운?????�버 probe ?�작
  useEffect(() => {
    probeServer();
  }, []);

  // serverStatus 변?�에 ?�른 ?�니메이??
  useEffect(() => {
    if (serverStatus === 'connecting') {
      progressAnim.setValue(0);
      slowAnimRef.current = Animated.timing(progressAnim, {
        toValue: 0.8,
        duration: 1 * 60 * 1000 + 40 * 1000,
        useNativeDriver: false,
      });
      slowAnimRef.current.start();
    } else if (serverStatus === 'ready') {
      if (slowAnimRef.current) slowAnimRef.current.stop();
      Animated.timing(progressAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: false,
      }).start();
    } else if (serverStatus === 'error') {
      if (slowAnimRef.current) slowAnimRef.current.stop();
    }
  }, [serverStatus]);

  useEffect(() => {
    const loadSettings = async () => {
      try {
        const [savedMyLang, savedTargetLang, savedMode] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEYS.MY_LANG),
          AsyncStorage.getItem(STORAGE_KEYS.TARGET_LANG),
          AsyncStorage.getItem(STORAGE_KEYS.MODE),
        ]);

        if (savedMyLang) {
          const found = LANGUAGES.find(l => l.code === savedMyLang);
          if (found) setMyLanguage(found);
        }
        if (savedTargetLang) {
          const found = LANGUAGES.find(l => l.code === savedTargetLang);
          if (found) setTargetLanguage(found);
        }
        if (savedMode) {
          const found = CONVERSATION_MODES.find(m => m.id === savedMode);
          if (found) setConversationMode(found);
        }
      } catch (e) {
        console.error('Failed to load settings:', e);
      }
      setLoaded(true);
    };
    loadSettings();
  }, []);

  const updateMyLanguage = (lang: Language) => {
    setMyLanguage(lang);
    AsyncStorage.setItem(STORAGE_KEYS.MY_LANG, lang.code);
  };

  const updateTargetLanguage = (lang: Language) => {
    setTargetLanguage(lang);
    AsyncStorage.setItem(STORAGE_KEYS.TARGET_LANG, lang.code);
  };

  const updateMode = (mode: typeof CONVERSATION_MODES[0]) => {
    setConversationMode(mode);
    AsyncStorage.setItem(STORAGE_KEYS.MODE, mode.id);
  };

  const handleStart = () => {
    if (myLanguage.code === targetLanguage.code) {
      Alert.alert('오류', '내 언어와 상대 언어가 같을 수 없습니다.');
      return;
    }
    navigation.navigate('Conversation', {
      myLang: myLanguage,
      targetLang: targetLanguage,
      mode: conversationMode.id,
    });
  };

  if (!loaded) return null;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      <View style={styles.content}>
        {/* Logo */}
        <View style={styles.logoSection}>
          <GradientText text="STiTy" style={styles.logoText} />
        </View>

        {/* Selection Rows */}
        <View style={styles.selectionSection}>
          {/* ?�의 ?�어 */}
          <View style={styles.row}>
            <Text style={styles.label}>내 언어</Text>
            <TouchableOpacity
              style={styles.valueBox}
              onPress={() => setMyLangModal(true)}
            >
              <Text style={styles.valueText}>{formatLanguageDisplay(myLanguage)}</Text>
              <Text style={styles.arrow}>⌄</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.divider} />

          {/* ?��? ?�어 */}
          <View style={styles.row}>
            <Text style={styles.label}>상대 언어</Text>
            <TouchableOpacity
              style={styles.valueBox}
              onPress={() => setTargetLangModal(true)}
            >
              <Text style={styles.valueText}>{formatLanguageAs(targetLanguage, myLanguage.code)}</Text>
              <Text style={styles.arrow}>⌄</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.divider} />

          {/* ?�???�식 */}
          <View style={styles.row}>
            <Text style={styles.label}>대화방식</Text>
            <TouchableOpacity
              style={styles.valueBox}
              onPress={() => setModeModal(true)}
            >
              <Text style={styles.valueText}>{conversationMode.name}</Text>
              <Text style={styles.arrow}>⌄</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* ?�버 로딩 �?/ ?�작?�기 버튼 */}
        {serverStatus === 'ready' ? (
          <TouchableOpacity onPress={handleStart} activeOpacity={0.8}>
            <LinearGradient
              colors={['#8E54E9', '#4776E6', '#00CFEF']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.startBtnGradient}
            >
              <View style={styles.startBtnInner}>
                <GradientText text="시작하기" style={styles.startText} />
              </View>
            </LinearGradient>
          </TouchableOpacity>
        ) : serverStatus === 'error' ? (
          <View style={styles.serverErrorContainer}>
            <Text style={styles.serverErrorText}>서버에 연결할 수 없습니다</Text>
            <TouchableOpacity onPress={probeServer} activeOpacity={0.8}>
              <LinearGradient
                colors={['#8E54E9', '#4776E6', '#00CFEF']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.startBtnGradient}
              >
                <View style={styles.startBtnInner}>
                  <GradientText text="재시도" style={styles.startText} />
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.loadingContainer}>
            <View style={styles.progressTrack}>
              <Animated.View
                style={[
                  styles.progressFill,
                  {
                    width: progressAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: ['0%', '100%'],
                    }),
                  },
                ]}
              >
                <LinearGradient
                  colors={['#8E54E9', '#4776E6', '#00CFEF']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={StyleSheet.absoluteFill}
                />
              </Animated.View>
            </View>
            <Text style={styles.loadingText}>서버 연결 중..</Text>
          </View>
        )}
      </View>

      {/* Modals */}
      <LanguagePickerModal
        visible={myLangModal}
        onClose={() => setMyLangModal(false)}
        onSelect={updateMyLanguage}
        selectedCode={myLanguage.code}
        excludeCode={targetLanguage.code}
        title="내 언어 선택"
      />
      <LanguagePickerModal
        visible={targetLangModal}
        onClose={() => setTargetLangModal(false)}
        onSelect={updateTargetLanguage}
        selectedCode={targetLanguage.code}
        excludeCode={myLanguage.code}
        title="상대 언어 선택"
        displayAsLangCode={myLanguage.code}
      />
      <ModePickerModal
        visible={modeModal}
        onClose={() => setModeModal(false)}
        onSelect={updateMode}
        selectedId={conversationMode.id}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 60,
  },
  logoSection: {
    marginBottom: 56,
  },
  logoText: {
    fontSize: 48,
    fontWeight: 'bold',
    letterSpacing: 48 * 0.06,
  },
  selectionSection: {
    width: '100%',
    marginBottom: 78,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
  },
  label: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1F2937',
    width: 80,
  },
  valueBox: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  valueText: {
    fontSize: 15,
    color: '#6B7280',
  },
  arrow: {
    fontSize: 14,
    color: '#9CA3AF',
    marginLeft: 8,
  },
  divider: {
    height: 1,
    backgroundColor: '#E5E7EB',
    marginLeft: 80,
  },
  startBtnGradient: {
    borderRadius: 28,
    padding: 1.5,
  },
  startBtnInner: {
    backgroundColor: '#FFFFFF',
    borderRadius: 26.5,
    paddingVertical: 10,
    paddingHorizontal: 36,
    alignItems: 'center',
  },
  startText: {
    fontSize: 17,
    fontWeight: 'bold',
  },
  loadingContainer: {
    alignItems: 'center',
    gap: 10,
  },
  progressTrack: {
    width: 200,
    height: 6,
    backgroundColor: '#E5E7EB',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  loadingText: {
    fontSize: 13,
    color: '#9CA3AF',
  },
  serverErrorContainer: {
    alignItems: 'center',
    gap: 12,
  },
  serverErrorText: {
    fontSize: 13,
    color: '#9CA3AF',
  },
});

const modalStyles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  content: {
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '70%',
    paddingBottom: 32,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  title: {
    fontSize: 17,
    fontWeight: '600',
    color: '#1F2937',
  },
  closeBtn: {
    padding: 8,
  },
  closeText: {
    fontSize: 18,
    color: '#9CA3AF',
  },
  item: {
    paddingVertical: 14,
    paddingHorizontal: 24,
  },
  itemSelected: {
    backgroundColor: '#F3F4F6',
  },
  itemText: {
    fontSize: 16,
    color: '#1F2937',
  },
  itemTextSelected: {
    color: '#4776E6',
    fontWeight: '600',
  },
  itemDesc: {
    fontSize: 13,
    color: '#9CA3AF',
    marginTop: 4,
  },
  modeListContent: {
    paddingBottom: 14,
  },
  sep: {
    height: 1,
    backgroundColor: '#F3F4F6',
    marginHorizontal: 24,
  },
});

export default HomeScreen;
