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
  Easing,
  StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Path, Rect, Ellipse, Text as SvgText } from 'react-native-svg';
import { LinearGradient } from 'expo-linear-gradient';
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

// ─── UI string localizations ──────────────────────────────────────────────────
const UI_STRINGS: Record<string, {
  peerLanguage: string; mode: string;
  modes: Record<string, string>;
  collapse: string; configureAndStart: string; startSpeaking: string;
  listening: string; start: string; connectingServer: string; startingServer: string;
  stop: string; resume: string; back: string; cancel: string;
  endTitle: string; endMsg: string; end: string;
  errTitle: string; langMustDiffer: string; peerPickerTitle: string; connectionFailed: string;
  waitingForSlot: string; capacityFullMsg: string; capacityFreeMsg: string;
}> = {
  en: { peerLanguage: 'PEER LANGUAGE', mode: 'MODE', modes: { 'mode-1': 'Speaker', 'mode-2': 'Earphone', 'mode-3': 'Both' }, collapse: 'Collapse ▲', configureAndStart: 'Configure and tap Start', startSpeaking: 'Start speaking', listening: 'Listening...', start: 'Start', connectingServer: 'Connecting to server...', startingServer: 'Starting server...', stop: 'Stop', resume: 'Resume', back: 'Back', cancel: 'Cancel', endTitle: 'End conversation', endMsg: 'Do you want to end the conversation?', end: 'End', errTitle: 'Error', langMustDiffer: 'My language and peer language must be different.', peerPickerTitle: 'Peer Language', connectionFailed: 'Connection failed', waitingForSlot: 'Waiting for a slot…', capacityFullMsg: 'Server is full. Start will activate when a slot frees up.', capacityFreeMsg: 'A slot just opened — you can start now.' },
  ko: { peerLanguage: '상대방 언어', mode: '모드', modes: { 'mode-1': '스피커', 'mode-2': '이어폰', 'mode-3': '둘 다' }, collapse: '접기 ▲', configureAndStart: '설정 후 시작을 누르세요', startSpeaking: '말씀해 주세요', listening: '듣는 중...', start: '시작', connectingServer: '서버 연결 중...', startingServer: '서버 시작 중...', stop: '중지', resume: '재개', back: '종료', cancel: '취소', endTitle: '대화 종료', endMsg: '대화를 종료하시겠습니까?', end: '종료', errTitle: '오류', langMustDiffer: '내 언어와 상대방 언어가 달라야 합니다.', peerPickerTitle: '상대방 언어', connectionFailed: '연결 실패', waitingForSlot: '슬롯 대기 중…', capacityFullMsg: '서버가 꽉 찼습니다. 슬롯이 생기면 시작됩니다.', capacityFreeMsg: '슬롯이 생겼어요 — 지금 시작하세요!' },
  ja: { peerLanguage: '相手の言語', mode: 'モード', modes: { 'mode-1': 'スピーカー', 'mode-2': 'イヤホン', 'mode-3': '両方' }, collapse: '閉じる ▲', configureAndStart: '設定して開始をタップ', startSpeaking: '話してください', listening: '聴いています...', start: '開始', connectingServer: 'サーバー接続中...', startingServer: 'サーバー起動中...', stop: '停止', resume: '再開', back: '終了', cancel: 'キャンセル', endTitle: '会話を終了', endMsg: '会話を終了しますか？', end: '終了', errTitle: 'エラー', langMustDiffer: '自分と相手の言語が異なる必要があります。', peerPickerTitle: '相手の言語', connectionFailed: '接続失敗', waitingForSlot: 'スロット待機中…', capacityFullMsg: 'サーバーが満員です。空き次第、開始できます。', capacityFreeMsg: 'スロットが空きました — 開始できます。' },
  zh: { peerLanguage: '对方语言', mode: '模式', modes: { 'mode-1': '扬声器', 'mode-2': '耳机', 'mode-3': '两者' }, collapse: '收起 ▲', configureAndStart: '设置后点击开始', startSpeaking: '请开始说话', listening: '正在聆听...', start: '开始', connectingServer: '正在连接服务器...', startingServer: '正在启动服务器...', stop: '停止', resume: '继续', back: '结束', cancel: '取消', endTitle: '结束对话', endMsg: '确定要结束对话吗？', end: '结束', errTitle: '错误', langMustDiffer: '我的语言和对方语言必须不同。', peerPickerTitle: '对方语言', connectionFailed: '连接失败', waitingForSlot: '等待空位…', capacityFullMsg: '服务器已满，有空位时即可开始。', capacityFreeMsg: '有空位了 — 现在可以开始。' },
  es: { peerLanguage: 'IDIOMA DEL OTRO', mode: 'MODO', modes: { 'mode-1': 'Altavoz', 'mode-2': 'Auricular', 'mode-3': 'Ambos' }, collapse: 'Contraer ▲', configureAndStart: 'Configura y pulsa Iniciar', startSpeaking: 'Empieza a hablar', listening: 'Escuchando...', start: 'Iniciar', connectingServer: 'Conectando al servidor...', startingServer: 'Iniciando servidor...', stop: 'Detener', resume: 'Reanudar', back: 'Volver', cancel: 'Cancelar', endTitle: 'Finalizar conversación', endMsg: '¿Quieres finalizar la conversación?', end: 'Finalizar', errTitle: 'Error', langMustDiffer: 'Mi idioma y el del otro deben ser distintos.', peerPickerTitle: 'Idioma del otro', connectionFailed: 'Conexión fallida', waitingForSlot: 'Esperando espacio…', capacityFullMsg: 'Servidor lleno. Inicio disponible al liberarse un espacio.', capacityFreeMsg: '¡Hay espacio libre — puedes iniciar!' },
  fr: { peerLanguage: "LANGUE DE L'AUTRE", mode: 'MODE', modes: { 'mode-1': 'Haut-parleur', 'mode-2': 'Écouteur', 'mode-3': 'Les deux' }, collapse: 'Réduire ▲', configureAndStart: 'Configurez et appuyez sur Démarrer', startSpeaking: 'Commencez à parler', listening: 'En écoute...', start: 'Démarrer', connectingServer: 'Connexion au serveur...', startingServer: 'Démarrage du serveur...', stop: 'Arrêter', resume: 'Reprendre', back: 'Retour', cancel: 'Annuler', endTitle: 'Terminer la conversation', endMsg: 'Voulez-vous terminer la conversation ?', end: 'Terminer', errTitle: 'Erreur', langMustDiffer: "Ma langue et celle de l'autre doivent être différentes.", peerPickerTitle: "Langue de l'autre", connectionFailed: 'Connexion échouée', waitingForSlot: 'En attente…', capacityFullMsg: "Serveur plein. Démarrage possible dès qu'un créneau se libère.", capacityFreeMsg: 'Un créneau est libre — commencez !' },
  id: { peerLanguage: 'BAHASA MITRA', mode: 'MODE', modes: { 'mode-1': 'Speaker', 'mode-2': 'Earphone', 'mode-3': 'Keduanya' }, collapse: 'Tutup ▲', configureAndStart: 'Atur dan ketuk Mulai', startSpeaking: 'Mulai berbicara', listening: 'Mendengarkan...', start: 'Mulai', connectingServer: 'Menghubungkan ke server...', startingServer: 'Memulai server...', stop: 'Berhenti', resume: 'Lanjutkan', back: 'Kembali', cancel: 'Batal', endTitle: 'Akhiri percakapan', endMsg: 'Apakah Anda ingin mengakhiri percakapan?', end: 'Akhiri', errTitle: 'Kesalahan', langMustDiffer: 'Bahasa saya dan bahasa mitra harus berbeda.', peerPickerTitle: 'Bahasa Mitra', connectionFailed: 'Koneksi gagal', waitingForSlot: 'Menunggu slot…', capacityFullMsg: 'Server penuh. Mulai tersedia saat ada slot kosong.', capacityFreeMsg: 'Slot tersedia — mulai sekarang!' },
  vi: { peerLanguage: 'NGÔN NGỮ ĐỐI TÁC', mode: 'CHẾ ĐỘ', modes: { 'mode-1': 'Loa ngoài', 'mode-2': 'Tai nghe', 'mode-3': 'Cả hai' }, collapse: 'Thu gọn ▲', configureAndStart: 'Cài đặt và nhấn Bắt đầu', startSpeaking: 'Bắt đầu nói', listening: 'Đang nghe...', start: 'Bắt đầu', connectingServer: 'Đang kết nối máy chủ...', startingServer: 'Đang khởi động máy chủ...', stop: 'Dừng', resume: 'Tiếp tục', back: 'Quay lại', cancel: 'Hủy', endTitle: 'Kết thúc cuộc trò chuyện', endMsg: 'Bạn có muốn kết thúc cuộc trò chuyện không?', end: 'Kết thúc', errTitle: 'Lỗi', langMustDiffer: 'Ngôn ngữ của tôi và đối tác phải khác nhau.', peerPickerTitle: 'Ngôn ngữ đối tác', connectionFailed: 'Kết nối thất bại', waitingForSlot: 'Đang chờ chỗ trống…', capacityFullMsg: 'Máy chủ đầy. Sẽ bắt đầu khi có chỗ trống.', capacityFreeMsg: 'Có chỗ trống rồi — bắt đầu ngay!' },
  th: { peerLanguage: 'ภาษาของคู่สนทนา', mode: 'โหมด', modes: { 'mode-1': 'ลำโพง', 'mode-2': 'หูฟัง', 'mode-3': 'ทั้งสอง' }, collapse: 'ย่อ ▲', configureAndStart: 'ตั้งค่าแล้วแตะเริ่มต้น', startSpeaking: 'เริ่มพูด', listening: 'กำลังฟัง...', start: 'เริ่มต้น', connectingServer: 'กำลังเชื่อมต่อเซิร์ฟเวอร์...', startingServer: 'กำลังเริ่มเซิร์ฟเวอร์...', stop: 'หยุด', resume: 'ดำเนินการต่อ', back: 'กลับ', cancel: 'ยกเลิก', endTitle: 'สิ้นสุดการสนทนา', endMsg: 'คุณต้องการสิ้นสุดการสนทนาหรือไม่?', end: 'สิ้นสุด', errTitle: 'ข้อผิดพลาด', langMustDiffer: 'ภาษาของฉันและคู่สนทนาต้องแตกต่างกัน', peerPickerTitle: 'ภาษาของคู่สนทนา', connectionFailed: 'การเชื่อมต่อล้มเหลว', waitingForSlot: 'รอช่อง…', capacityFullMsg: 'เซิร์ฟเวอร์เต็ม จะเริ่มได้เมื่อมีช่องว่าง', capacityFreeMsg: 'มีช่องว่างแล้ว — เริ่มได้เลย!' },
  de: { peerLanguage: 'SPRACHE DES ANDEREN', mode: 'MODUS', modes: { 'mode-1': 'Lautsprecher', 'mode-2': 'Kopfhörer', 'mode-3': 'Beide' }, collapse: 'Einklappen ▲', configureAndStart: 'Einstellen und Start tippen', startSpeaking: 'Anfangen zu sprechen', listening: 'Zuhören...', start: 'Start', connectingServer: 'Verbinde mit Server...', startingServer: 'Server wird gestartet...', stop: 'Stopp', resume: 'Fortsetzen', back: 'Zurück', cancel: 'Abbrechen', endTitle: 'Gespräch beenden', endMsg: 'Möchten Sie das Gespräch beenden?', end: 'Beenden', errTitle: 'Fehler', langMustDiffer: 'Meine Sprache und die des anderen müssen unterschiedlich sein.', peerPickerTitle: 'Sprache des anderen', connectionFailed: 'Verbindung fehlgeschlagen', waitingForSlot: 'Warte auf Slot…', capacityFullMsg: 'Server voll. Start möglich, sobald ein Slot frei wird.', capacityFreeMsg: 'Ein Slot ist frei — jetzt starten!' },
};

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

// Logo: S(#6080C8) T(#6080C8) i(#9BB4D4) T(#7A9030) y(#E8E0A0) — Prototype.html 기준
const STiTyLogo = () => (
  <Text style={S.logo}>
    <Text style={{ color: '#6080C8' }}>ST</Text>
    <Text style={{ color: '#9BB4D4' }}>i</Text>
    <Text style={{ color: '#7A9030' }}>T</Text>
    <Text style={{ color: '#E8E0A0' }}>y</Text>
  </Text>
);

const StatusDot = ({ active }: { active: boolean }) => {
  const ringOpacity = useRef(new Animated.Value(0.3)).current;
  const animRef = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    if (active) {
      animRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(ringOpacity, { toValue: 0.06, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(ringOpacity, { toValue: 0.3, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ])
      );
      animRef.current.start();
    } else {
      animRef.current?.stop();
      ringOpacity.setValue(0.3);
    }
    return () => { animRef.current?.stop(); };
  }, [active]);

  return (
    <View style={S.dotWrap}>
      {active && <Animated.View style={[S.dotRing, { opacity: ringOpacity }]} />}
      <View style={[S.dot, { backgroundColor: active ? '#22c55e' : '#ccc' }]} />
    </View>
  );
};

const WaveBar = ({ delay }: { delay: number }) => {
  const h = useRef(new Animated.Value(3)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(h, { toValue: 14, duration: 400, delay, useNativeDriver: false }),
        Animated.timing(h, { toValue: 3, duration: 400, useNativeDriver: false }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, []);
  return <Animated.View style={[S.waveBar, { height: h }]} />;
};

const RecordingBar = ({ label }: { label: string }) => (
  <View style={S.recBar}>
    <View style={S.recDot} />
    <Text style={S.recLabel}>{label}</Text>
    <View style={S.waveRow}>
      {[0, 100, 200, 300, 400, 300, 200, 100, 0].map((d, i) => (
        <WaveBar key={i} delay={d} />
      ))}
    </View>
  </View>
);

// Mode-based idle illustrations (from prototype)
const SpeakerIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#E8E0A0" opacity="0.3" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Rect x="84" y="70" width="12" height="28" rx="2.5" fill="#1a1a1a" />
    <Rect x="85.4" y="74" width="9.2" height="20" rx="1" fill="#6080C8" />
    <Rect x="87.5" y="71.8" width="5" height="1.2" rx="0.6" fill="#555" />
    <Path d="M87 84 Q88.5 81 90 84 Q91.5 87 93 84" stroke="#E8E0A0" strokeWidth="1" fill="none" strokeLinecap="round" />
    <Path d="M80 76 Q75 84 80 92" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M75 72 Q67 84 75 96" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Path d="M100 76 Q105 84 100 92" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M105 72 Q113 84 105 96" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
  </Svg>
);

const EarphoneIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#9BB4D4" opacity="0.22" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Circle cx="74" cy="76" r="1.6" fill="#9BB4D4" />
    <Circle cx="84" cy="70" r="2.2" fill="#E8E0A0" />
    <Circle cx="96" cy="70" r="2.2" fill="#E8E0A0" />
    <Circle cx="106" cy="76" r="1.6" fill="#6080C8" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="43" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#6080C8" strokeWidth="1.3" />
    <Rect x="42" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#6080C8" strokeWidth="1" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Ellipse cx="137" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#7A9030" strokeWidth="1.3" />
    <Rect x="136" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#7A9030" strokeWidth="1" />
  </Svg>
);

const BothIllustration = () => (
  <Svg width={150} height={125} viewBox="0 0 180 150">
    <Circle cx="90" cy="82" r="62" fill="#E8E0A0" opacity="0.25" />
    <Ellipse cx="90" cy="132" rx="58" ry="7" fill="#7A9030" opacity="0.22" />
    <Ellipse cx="55" cy="114" rx="14" ry="15" fill="#9BB4D4" />
    <Circle cx="55" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="59" cy="92" r="1.5" fill="#6080C8" />
    <Circle cx="64" cy="92" r="1.5" fill="#6080C8" />
    <Path d="M58 97 Q61 99 64 97" stroke="#6080C8" strokeWidth="1.3" strokeLinecap="round" fill="none" />
    <Rect x="44" y="64" width="22" height="11" rx="5.5" fill="#6080C8" />
    <Path d="M53 75 L55 78 L57 75 Z" fill="#6080C8" />
    <SvgText x="55" y="72.5" textAnchor="middle" fontSize="7.5" fontWeight="800" fill="#fff" letterSpacing="0.5">ME</SvgText>
    <Ellipse cx="43" cy="94" rx="2.6" ry="3.6" fill="#fff" stroke="#6080C8" strokeWidth="1.3" />
    <Rect x="42" y="97.5" width="2" height="4.5" rx="1" fill="#fff" stroke="#6080C8" strokeWidth="1" />
    <Rect x="84" y="76" width="12" height="28" rx="2.5" fill="#1a1a1a" />
    <Rect x="85.4" y="80" width="9.2" height="20" rx="1" fill="#6080C8" />
    <Rect x="87.5" y="77.8" width="5" height="1.2" rx="0.6" fill="#555" />
    <Path d="M87 90 Q88.5 87 90 90 Q91.5 93 93 90" stroke="#E8E0A0" strokeWidth="1" fill="none" strokeLinecap="round" />
    <Path d="M100 82 Q105 90 100 98" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.75" />
    <Path d="M105 78 Q113 90 105 102" stroke="#6080C8" strokeWidth="1.4" fill="none" strokeLinecap="round" opacity="0.4" />
    <Ellipse cx="125" cy="114" rx="14" ry="15" fill="#6080C8" />
    <Circle cx="125" cy="92" r="13" fill="#E8E0A0" />
    <Circle cx="116" cy="92" r="1.5" fill="#7A9030" />
    <Circle cx="121" cy="92" r="1.5" fill="#7A9030" />
    <Path d="M116 97 Q119 99 122 97" stroke="#7A9030" strokeWidth="1.3" strokeLinecap="round" fill="none" />
  </Svg>
);

const ModeIllustration = ({ modeId }: { modeId: string }) => {
  if (modeId === 'mode-2') return <EarphoneIllustration />;
  if (modeId === 'mode-3') return <BothIllustration />;
  return <SpeakerIllustration />;
};

// Pulsing dot for capacity notice
const CapacityPulse = ({ isOk }: { isOk: boolean }) => {
  const ringOpacity = useRef(new Animated.Value(0.2)).current;
  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(ringOpacity, { toValue: 0.05, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(ringOpacity, { toValue: 0.2, duration: 750, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, []);
  const color = isOk ? '#7A9030' : '#C8A030';
  return (
    <View style={{ width: 14, height: 14, alignItems: 'center', justifyContent: 'center' }}>
      <Animated.View style={{ position: 'absolute', width: 14, height: 14, borderRadius: 7, backgroundColor: color, opacity: ringOpacity }} />
      <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: color }} />
    </View>
  );
};

const LangPickerSheet = ({
  visible, title, selectedCode, excludeCode, onSelect, onClose, myLangCode,
  cancelLabel = 'Cancel', forceEnglish = false,
}: {
  visible: boolean;
  title: string;
  selectedCode: string;
  excludeCode: string;
  onSelect: (lang: Language) => void;
  onClose: () => void;
  myLangCode: string;
  cancelLabel?: string;
  forceEnglish?: boolean;
}) => {
  const { bottom } = useSafeAreaInsets();
  const filtered = LANGUAGES.filter(l => l.code !== excludeCode);
  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={S.pickerOverlay}>
        <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={onClose} />
        <View style={[S.pickerSheet, { paddingBottom: Math.max(bottom, 16) + 16 }]}>
          <Text style={S.pickerTitle}>{title}</Text>
          <ScrollView>
            {filtered.map(lang => {
              const name = forceEnglish
                ? lang.name
                : (lang.translations[myLangCode] || lang.nativeName);
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
            <Text style={S.pickerCancelTxt}>{cancelLabel}</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
};

const BubbleItem = ({ entry, myLangCode, targetLangCode }: {
  entry: TranscriptionEntry; myLangCode: string; targetLangCode: string;
}) => {
  const isMine = entry.language === myLangCode;
  const isPeer = entry.language === targetLangCode;
  const clr = isMine
    ? { bubble: '#6080C8', avatar: '#9BB4D4', text: '#fff', subText: 'rgba(255,255,255,0.65)' }
    : isPeer
    ? { bubble: '#f2f2f2', avatar: '#E8E0A0', text: '#1a1a1a', subText: '#aaa' }
    : { ...getLangColor(entry.language), text: '#fff', subText: 'rgba(255,255,255,0.65)' };

  const slideAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(slideAnim, {
      toValue: 1,
      duration: 300,
      easing: Easing.bezier(0.4, 0, 0.2, 1),
      useNativeDriver: true,
    }).start();
  }, []);

  return (
    <Animated.View style={[S.bubbleRow, isMine && S.bubbleRowMine, {
      opacity: slideAnim,
      transform: [{ translateY: slideAnim.interpolate({ inputRange: [0, 1], outputRange: [12, 0] }) }],
    }]}>
      <View style={[S.avatar, { backgroundColor: clr.avatar }]}>
        <Text style={[S.avatarTxt, { color: isMine ? '#fff' : '#7A9030' }]}>
          {entry.language.toUpperCase()}
        </Text>
      </View>
      <View style={[
        S.bubble,
        {
          backgroundColor: clr.bubble,
          borderBottomRightRadius: isMine ? 4 : 18,
          borderBottomLeftRadius: isMine ? 18 : 4,
        },
      ]}>
        <Text style={[S.bubbleMain, { color: clr.text }]}>{entry.text}</Text>
        {!!entry.translatedText && (
          <Text style={[S.bubbleSub, { color: clr.subText }]}>{entry.translatedText}</Text>
        )}
      </View>
    </Animated.View>
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
  const [serverCapacity, setServerCapacity] = useState<{ active: number; max: number } | null>(null);
  const [justFreed, setJustFreed] = useState(false);
  const prevServerFullRef = useRef(false);
  const [connectPct, setConnectPct] = useState(0);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressAnimRef = useRef<Animated.CompositeAnimation | null>(null);

  // chips / setup 패널 애니메이션
  const chipsAnim = useRef(new Animated.Value(0)).current;
  const setupOpacityAnim = useRef(new Animated.Value(1)).current;
  const setupHeightAnim = useRef(new Animated.Value(1000)).current;
  const setupNaturalHeight = useRef(1000);
  const isSetupAnimatingRef = useRef(false);
  const isSetupVisibleRef = useRef(true);
  const setupAnimRef = useRef<Animated.CompositeAnimation | null>(null);

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

  // ── Server connect progress ───────────────────────────────────────────────────
  useEffect(() => {
    const id = progressAnim.addListener(({ value }) => setConnectPct(Math.round(value * 100)));
    return () => progressAnim.removeListener(id);
  }, []);

  useEffect(() => {
    progressAnimRef.current?.stop();
    if (serverStatus === 'ec2-starting') {
      progressAnim.setValue(0);
    } else if (serverStatus === 'connecting') {
      progressAnim.setValue(0);
      progressAnimRef.current = Animated.timing(progressAnim, { toValue: 0.85, duration: 80000, useNativeDriver: false });
      progressAnimRef.current.start();
    } else if (serverStatus === 'ready') {
      Animated.timing(progressAnim, { toValue: 1, duration: 400, useNativeDriver: false }).start();
    }
  }, [serverStatus]);

  // ── Setup panel ↔ Chips 전환 애니메이션 (maxHeight + opacity parallel) ────────
  const prevSetupVisibleRef = useRef(true);
  useEffect(() => {
    const visible = sessionState === 'idle' || showSetup;
    if (visible === prevSetupVisibleRef.current) return;
    prevSetupVisibleRef.current = visible;
    isSetupVisibleRef.current = visible;

    setupAnimRef.current?.stop();
    isSetupAnimatingRef.current = true;

    if (visible) {
      setupHeightAnim.setValue(0);
      setupOpacityAnim.setValue(0);
      // non-idle 상태에서는 collapse 버튼이 추가돼 높이가 늘어나므로 여유 확보
      const showTargetH = sessionStateRef.current !== 'idle'
        ? setupNaturalHeight.current + 48
        : setupNaturalHeight.current;
      setupAnimRef.current = Animated.parallel([
        Animated.timing(setupHeightAnim, {
          toValue: showTargetH,
          duration: 420,
          easing: Easing.bezier(0.4, 0, 0.2, 1),
          useNativeDriver: false,
        }),
        Animated.timing(setupOpacityAnim, {
          toValue: 1,
          duration: 340,
          easing: Easing.ease,
          useNativeDriver: false,
        }),
      ]);
      setupAnimRef.current.start(() => { isSetupAnimatingRef.current = false; });
      Animated.timing(chipsAnim, { toValue: 0, duration: 250, easing: Easing.ease, useNativeDriver: true }).start();
    } else {
      setupAnimRef.current = Animated.parallel([
        Animated.timing(setupHeightAnim, {
          toValue: 0,
          duration: 380,
          easing: Easing.bezier(0.4, 0, 0.2, 1),
          useNativeDriver: false,
        }),
        Animated.timing(setupOpacityAnim, {
          toValue: 0,
          duration: 260,
          easing: Easing.ease,
          useNativeDriver: false,
        }),
      ]);
      setupAnimRef.current.start(() => { isSetupAnimatingRef.current = false; });
      Animated.timing(chipsAnim, { toValue: 1, duration: 380, easing: Easing.ease, useNativeDriver: true }).start();
    }
  }, [sessionState, showSetup]);

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
      } else if (next === 'active' && (prev === 'background' || prev === 'inactive') && sessionStateRef.current === 'idle') {
        probeServer(); // keepalive 끊겼으면 재기동, 살아있으면 즉시 return
      }
    });
    return () => sub.remove();
  }, []);

  // ── keepalive 실패 감지: serverStatus 'ready'→'idle' 전환 시 자동 재기동 ────────
  const prevServerStatusRef = useRef<string>(serverStatus);
  useEffect(() => {
    const prev = prevServerStatusRef.current;
    prevServerStatusRef.current = serverStatus;
    if (serverStatus === 'idle' && prev !== 'idle' && sessionStateRef.current === 'idle') {
      probeServer();
    }
  }, [serverStatus]);

  // ── Message listener ─────────────────────────────────────────────────────────
  const handleMessageRef = useRef((msg: any) => {
    const text = (msg.original || '').trim();
    if (!text || msg.type !== 'final') return;
    const lang = langToCode(msg.language || 'auto');
    const serverTrans = (msg.translation || '').trim();
    addTranscription(lang, text, serverTrans || undefined);
  });

  useEffect(() => {
    return addMessageListener((msg: any) => {
      if (msg.type === 'capacity' && typeof msg.active === 'number' && typeof msg.max === 'number') {
        setServerCapacity({ active: msg.active, max: msg.max });
      }
      handleMessageRef.current(msg);
    });
  }, [addMessageListener]);

  // ── Server capacity: full→available transition 감지 ───────────────────────────
  const serverFull = serverCapacity !== null && serverCapacity.active >= serverCapacity.max;
  useEffect(() => {
    if (prevServerFullRef.current && !serverFull) {
      setJustFreed(true);
      const t = setTimeout(() => setJustFreed(false), 2500);
      prevServerFullRef.current = serverFull;
      return () => clearTimeout(t);
    }
    prevServerFullRef.current = serverFull;
  }, [serverFull]);

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
      isEarphone, // true → STREAM_VOICE_CALL(이어폰), false → STREAM_MUSIC(스피커)
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
      const s = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
      Alert.alert(s.errTitle, s.langMustDiffer);
      return;
    }
    const s = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
    setSessionError('');
    try {
      // 서버가 준비 안 됐거나 오류 상태면 재시작
      if (serverStatusRef.current !== 'ready') {
        await probeServer(serverStatusRef.current === 'error');
      }
      if (serverStatusRef.current !== 'ready') {
        throw new Error(s.connectionFailed);
      }
      try {
        await connect({ lang: myLang.code, targetLang: targetLang.code });
      } catch {
        // connect 실패 = serverStatus는 'ready'였지만 서버 실제로 죽어있음 → 강제 재시작
        await probeServer(true);
        if (serverStatusRef.current !== 'ready') {
          throw new Error(s.connectionFailed);
        }
        await connect({ lang: myLang.code, targetLang: targetLang.code });
      }
      await startRecording();
      applySpeakerRouting();
      sessionStateRef.current = 'recording';
      setSessionState('recording');
      setShowSetup(false);
    } catch (e: any) {
      setSessionError(e?.message ?? s.connectionFailed);
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
        try {
          await connect({ lang: myLangRef.current.code, targetLang: targetLangRef.current.code });
        } catch {
          // connect 실패 = 서버가 이미 종료됨. 강제로 서버 재시작 후 재연결
          setIsInitializing(true);
          await probeServer(true);
          await connect({ lang: myLangRef.current.code, targetLang: targetLangRef.current.code });
        }
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
    const s = UI_STRINGS[myLangRef.current.code] ?? UI_STRINGS.en;
    Alert.alert(s.endTitle, s.endMsg, [
      { text: s.cancel, style: 'cancel' },
      {
        text: s.end, style: 'destructive',
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

  const ui = UI_STRINGS[myLang.code] ?? UI_STRINGS.en;
  const isIdle = sessionState === 'idle';
  const isRecording = sessionState === 'recording';
  const isPaused = sessionState === 'paused';
  const modeObj = CONVERSATION_MODES.find(m => m.id === modeId) ?? CONVERSATION_MODES[0];
  const bottomPad = Math.max(insets.bottom, 20);

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
              color={isTTSMuted ? '#ccc' : '#7A9030'}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* ── Setup panel (maxHeight + opacity parallel animation) ── */}
      <Animated.View
        style={{ overflow: 'hidden', maxHeight: setupHeightAnim, opacity: setupOpacityAnim }}
        onLayout={(e) => {
          const h = e.nativeEvent.layout.height;
          if (h > 10 && !isSetupAnimatingRef.current && isSetupVisibleRef.current) {
            if (Math.abs(h - setupNaturalHeight.current) > 2) {
              setupNaturalHeight.current = h;
              setupHeightAnim.setValue(h);
            }
          }
        }}
      >
        <View style={S.setupPanel}>
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

            <TouchableOpacity
              style={S.langSwapBtn}
              onPress={() => {
                const tmp = myLangRef.current;
                updateMyLang(targetLangRef.current);
                updateTargetLang(tmp);
              }}
              activeOpacity={0.7}
            >
              <Text style={S.langSwap}>⇄</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[S.langCard, { backgroundColor: PEER_CARD_COLOR }]}
              onPress={() => setPicker('peer')}
              activeOpacity={0.85}
            >
              <Text style={S.langCardLabel}>{ui.peerLanguage}</Text>
              <Text style={S.langCardValue}>
                {targetLang.translations[myLang.code] ?? targetLang.name}
              </Text>
              <Text style={S.langCardArrow}>▾</Text>
            </TouchableOpacity>
          </View>

          <View style={S.modeSection}>
            <Text style={S.modeSectionLabel}>{ui.mode}</Text>
            <View style={S.modeSegment}>
              {CONVERSATION_MODES.map(m => (
                <TouchableOpacity
                  key={m.id}
                  style={[S.modeOption, modeId === m.id && S.modeOptionActive]}
                  onPress={() => updateMode(m.id)}
                  activeOpacity={0.8}
                >
                  <Text style={[S.modeOptionTxt, modeId === m.id && S.modeOptionTxtActive]} numberOfLines={1} adjustsFontSizeToFit>
                    {ui.modes[m.id] ?? m.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {!isIdle && (
            <TouchableOpacity style={S.collapseBtn} onPress={() => setShowSetup(false)}>
              <Text style={S.collapseBtnTxt}>{ui.collapse}</Text>
            </TouchableOpacity>
          )}
        </View>
      </Animated.View>

      {/* ── Session chips (opacity 애니메이션) ── */}
      {!isIdle && !showSetup && (
        <Animated.View style={{
          opacity: chipsAnim,
          transform: [{ translateY: chipsAnim.interpolate({ inputRange: [0, 1], outputRange: [-6, 0] }) }],
        }}>
          <TouchableOpacity style={S.chips} onPress={() => setShowSetup(true)} activeOpacity={0.8}>
            <View style={[S.chip, { backgroundColor: '#f0f4e8' }]}>
              <Text style={[S.chipTxt, { color: '#7A9030' }]}>{myLang.name}</Text>
            </View>
            <Text style={S.chipSep}>↔</Text>
            <View style={[S.chip, { backgroundColor: '#eaf2fb' }]}>
              <Text style={[S.chipTxt, { color: '#6080C8' }]}>
                {targetLang.translations[myLang.code] ?? targetLang.name}
              </Text>
            </View>
            <View style={[S.chip, { backgroundColor: '#f0f0f0' }]}>
              <Text style={[S.chipTxt, { color: '#777' }]}>{ui.modes[modeObj.id] ?? modeObj.name}</Text>
            </View>
          </TouchableOpacity>
        </Animated.View>
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
            <ModeIllustration modeId={modeId} />
            <Text style={S.idleTxt}>{ui.configureAndStart}</Text>
            {__DEV__ && (
              <View style={{ flexDirection: 'row', gap: 6, marginTop: 10 }}>
                <TouchableOpacity
                  style={S.dbgBtn}
                  onPress={() => setServerCapacity({ active: 20, max: 20 })}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#996600' }]}>Full</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[S.dbgBtn, { backgroundColor: '#D4F0D4', borderColor: '#70C070' }]}
                  onPress={() => {
                    setServerCapacity({ active: 20, max: 20 });
                    setTimeout(() => setServerCapacity({ active: 19, max: 20 }), 150);
                  }}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#336633' }]}>Freed</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[S.dbgBtn, { backgroundColor: '#f0f0f0', borderColor: '#ccc' }]}
                  onPress={() => {
                    prevServerFullRef.current = false;
                    setJustFreed(false);
                    setServerCapacity(null);
                  }}
                >
                  <Text style={[S.dbgBtnTxt, { color: '#777' }]}>Clear</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}

        {!isIdle && transcriptions.length === 0 && (
          <View style={S.idleWrap}>
            <LinearGradient
              colors={['rgba(122,144,48,0.1)', 'rgba(6,182,212,0.1)']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={S.micIconWrap}
            >
              <Ionicons name="mic" size={26} color="#7A9030" />
            </LinearGradient>
            <Text style={S.idleTxt}>{ui.startSpeaking}</Text>
          </View>
        )}

        {transcriptions.map(entry => (
          <BubbleItem key={entry.id} entry={entry} myLangCode={myLang.code} targetLangCode={targetLang.code} />
        ))}
      </ScrollView>

      {/* ── Recording bar ── */}
      {isRecording && <RecordingBar label={ui.listening} />}

      {/* ── Loading overlay ── */}
      {isInitializing && (
        <View style={S.initOverlay}>
          <Text style={S.initTxt}>
            {serverStatus === 'ec2-starting'
              ? `${ui.startingServer} (${connectPct}%)`
              : serverStatus === 'connecting'
              ? `${ui.connectingServer} (${connectPct}%)`
              : ui.connectingServer}
          </Text>
        </View>
      )}

      {/* ── Capacity notice (idle only) ── */}
      {isIdle && (serverFull || justFreed) && (
        <View style={[S.capacityNotice, !serverFull && justFreed && S.capacityNoticeOk]}>
          <CapacityPulse isOk={!serverFull && justFreed} />
          <Text style={[S.capText, !serverFull && justFreed && S.capTextOk]}>
            {serverFull ? ui.capacityFullMsg : ui.capacityFreeMsg}
          </Text>
          {serverCapacity && (
            <View style={[S.capBadge, !serverFull && justFreed && S.capBadgeOk]}>
              <Text style={[S.capBadgeTxt, !serverFull && justFreed && S.capBadgeTxtOk]}>
                {Math.round((serverCapacity.active / serverCapacity.max) * 100)}%
              </Text>
            </View>
          )}
        </View>
      )}

      {/* ── Bottom bar ── */}
      <View style={[S.bottomBar, { paddingBottom: bottomPad + 8 }]}>
        {isIdle && (
          <TouchableOpacity
            style={[S.btn, S.btnOutline,
              serverFull && S.btnFull,
              (serverStatus === 'ec2-starting' || serverStatus === 'connecting') && S.btnDisabled]}
            onPress={doStart}
            activeOpacity={0.85}
            disabled={serverStatus === 'ec2-starting' || serverStatus === 'connecting' || isInitializing || serverFull}
          >
            <Text style={[S.btnTxt, { color: serverFull ? '#bbb' : '#7A9030' }]}>
              {serverFull
                ? ui.waitingForSlot
                : serverStatus === 'ready' || serverStatus === 'idle' || serverStatus === 'error'
                ? ui.start
                : serverStatus === 'ec2-starting'
                ? ui.startingServer
                : `${ui.connectingServer} (${connectPct}%)`}
            </Text>
          </TouchableOpacity>
        )}

        {isRecording && (
          <>
            <TouchableOpacity style={[S.btn, S.btnStop]} onPress={doStop} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#ef4444' }]}>{ui.stop}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnGhost]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#999' }]}>✕</Text>
            </TouchableOpacity>
          </>
        )}

        {isPaused && (
          <>
            <TouchableOpacity style={[S.btn, S.btnPaused]} onPress={doResume} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#aaa' }]}>{ui.resume}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[S.btn, S.btnPaused]} onPress={doBack} activeOpacity={0.85}>
              <Text style={[S.btnTxt, { color: '#aaa' }]}>{ui.back}</Text>
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
        cancelLabel="Cancel"
        forceEnglish
      />
      <LangPickerSheet
        visible={picker === 'peer'}
        title={ui.peerPickerTitle}
        selectedCode={targetLang.code}
        excludeCode={myLang.code}
        onSelect={updateTargetLang}
        onClose={() => setPicker(null)}
        myLangCode={myLang.code}
        cancelLabel={ui.cancel}
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
  dotWrap: { width: 20, height: 20, alignItems: 'center', justifyContent: 'center' },
  dotRing: { position: 'absolute', width: 20, height: 20, borderRadius: 10, backgroundColor: '#22c55e' },
  dot: { width: 8, height: 8, borderRadius: 4 },

  // Setup panel
  setupPanel: { paddingHorizontal: 20, paddingTop: 18 },
  langRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 14 },
  langCard: { flex: 1, borderRadius: 16, paddingTop: 12, paddingBottom: 14, paddingHorizontal: 14 },
  langCardLabel: { fontSize: 9, fontWeight: '700', color: 'rgba(255,255,255,0.7)', letterSpacing: 0.5, marginBottom: 5 },
  langCardValue: { fontSize: 16, fontWeight: '700', color: '#fff' },
  langCardArrow: { position: 'absolute', top: 10, right: 12, color: 'rgba(255,255,255,0.6)', fontSize: 12 },
  langSwapBtn: { padding: 8 },
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
  micIconWrap: { width: 56, height: 56, borderRadius: 28, alignItems: 'center', justifyContent: 'center' },
  idleTxt: { fontSize: 13, color: '#bbb', textAlign: 'center' },

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
  btnPaused: { borderWidth: 2, borderColor: '#ccc', backgroundColor: 'transparent' },
  btnStop: { backgroundColor: '#fff0f0' },
  btnGhost: { flex: 0, width: 52, backgroundColor: '#f5f5f5' },
  btnDisabled: { opacity: 0.45 },

  // Capacity notice
  capacityNotice: { flexDirection: 'row', alignItems: 'center', gap: 10, marginHorizontal: 20, marginBottom: 4, paddingVertical: 10, paddingHorizontal: 14, backgroundColor: '#fffaf0', borderWidth: 1, borderColor: '#f3e6c4', borderRadius: 14 },
  capacityNoticeOk: { backgroundColor: '#f3faf0', borderColor: '#d8ebc8' },
  capText: { flex: 1, fontSize: 12, fontWeight: '600', color: '#7A6010', lineHeight: 18 },
  capTextOk: { color: '#56711F' },
  capBadge: { paddingVertical: 3, paddingHorizontal: 8, borderRadius: 100, backgroundColor: 'rgba(200,160,48,0.12)' },
  capBadgeOk: { backgroundColor: 'rgba(122,144,48,0.12)' },
  capBadgeTxt: { fontSize: 12, fontWeight: '700', color: '#7A6010' },
  capBadgeTxtOk: { color: '#56711F' },
  btnFull: { borderColor: '#e2e2e2' },

  // Debug (dev-only)
  dbgBtn: { paddingVertical: 4, paddingHorizontal: 10, borderRadius: 8, backgroundColor: '#FFF3CD', borderWidth: 1, borderColor: '#F0C040' },
  dbgBtnTxt: { fontSize: 11, fontWeight: '600' as const },

  // Language picker
  pickerOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', justifyContent: 'flex-end' },
  pickerSheet: { backgroundColor: '#fff', borderTopLeftRadius: 24, borderTopRightRadius: 24, paddingTop: 20, paddingHorizontal: 20, maxHeight: '75%' },
  pickerTitle: { fontSize: 13, fontWeight: '700', color: '#aaa', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 },
  pickerOption: { paddingVertical: 13, paddingHorizontal: 14, borderRadius: 12 },
  pickerOptionSel: { backgroundColor: '#eaf2fb' },
  pickerOptionTxt: { fontSize: 15, fontWeight: '600', color: '#1a1a1a' },
  pickerOptionTxtSel: { color: '#6080C8' },
  pickerCancel: { marginTop: 8, paddingVertical: 14, borderRadius: 14, backgroundColor: '#f5f5f5', alignItems: 'center' },
  pickerCancelTxt: { fontSize: 15, fontWeight: '600', color: '#aaa' },
});

export default HomeScreen;
