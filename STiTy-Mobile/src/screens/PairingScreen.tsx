import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import QRCode from 'react-native-qrcode-svg';
import { GradientButton } from '../components/GradientButton';
import { COLORS, FONTS, SPACING } from '../constants/theme';
import { getLanguageByCode, Language } from '../constants/languages';

type RootStackParamList = {
  Home: undefined;
  Loading: { myLang: Language; targetLang: Language; mode: string };
  Pairing: { myLang: Language; targetLang: Language; mode: string };
  Conversation: { myLang: Language; targetLang: Language; mode: string };
};

type PairingScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Pairing'>;
type PairingScreenRouteProp = RouteProp<RootStackParamList, 'Pairing'>;

interface PairingScreenProps {
  navigation: PairingScreenNavigationProp;
  route: PairingScreenRouteProp;
}

interface PairQrPayload {
  type: 'stity-pair-v2';
  roomId: string;
}

interface PairConnectedMessage {
  type: 'pair_connected';
  roomId: string;
  mode: string;
  myLang: string;
  targetLang: string;
}

const SERVER_URL = 'wss://supervigorously-unforded-lavone.ngrok-free.dev';

const createRoomId = (): string =>
  `room-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

export const PairingScreen: React.FC<PairingScreenProps> = ({ navigation, route }) => {
  const { myLang, targetLang } = route.params;
  const roomIdRef = useRef(createRoomId());
  const wsRef = useRef<WebSocket | null>(null);
  const pendingJoinRoomRef = useRef<string | null>(null);
  const hasNavigatedRef = useRef(false);
  const [scannerOpen, setScannerOpen] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [statusText, setStatusText] = useState('');
  const handledScanRef = useRef(false);

  const qrValue = useMemo(() => {
    const payload: PairQrPayload = {
      type: 'stity-pair-v2',
      roomId: roomIdRef.current,
    };
    return JSON.stringify(payload);
  }, []);

  const sendJson = (payload: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  };

  const hostRoom = () => {
    sendJson({
      type: 'pair_host',
      roomId: roomIdRef.current,
      myLang: myLang.code,
      targetLang: targetLang.code,
      mode: 'mode-2',
    });
    setStatusText('');
  };

  const attachSocketListeners = (ws: WebSocket) => {
    ws.onopen = () => {
      const joinRoom = pendingJoinRoomRef.current;
      if (joinRoom) {
        sendJson({ type: 'pair_join', roomId: joinRoom, myLang: myLang.code });
        setStatusText('상대방과 연결 중...');
      } else {
        hostRoom();
      }
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== 'string') return;
      let data: any = null;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === 'hello') return;

      if (data.type === 'pair_hosted') {
        setStatusText('');
        return;
      }

      if (data.type === 'pair_error') {
        const message = data.message === 'room_not_found'
          ? '유효하지 않은 QR 코드입니다.'
          : '페어링 중 오류가 발생했습니다.';
        setStatusText(message);
        Alert.alert('페어링 오류', message);
        pendingJoinRoomRef.current = null;
        hostRoom();
        handledScanRef.current = false;
        return;
      }

      if (data.type === 'pair_peer_left') {
        setStatusText('상대 연결이 끊겼습니다. 다시 스캔해 주세요.');
        return;
      }

      if (data.type === 'pair_connected') {
        const msg = data as PairConnectedMessage;
        const nextMy = getLanguageByCode(msg.myLang);
        const nextTarget = getLanguageByCode(msg.targetLang);
        if (!nextMy || !nextTarget || hasNavigatedRef.current) return;
        hasNavigatedRef.current = true;
        setScannerOpen(false);
        navigation.replace('Conversation', {
          myLang: nextMy,
          targetLang: nextTarget,
          mode: 'mode-2',
        });
      }
    };

    ws.onclose = () => {
      if (!hasNavigatedRef.current) {
        setStatusText('서버 연결할 수 없습니다');
      }
    };

    ws.onerror = () => {
      if (!hasNavigatedRef.current) {
        setStatusText('서버 연결할 수 없습니다');
      }
    };
  };

  const connectSocket = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(SERVER_URL);
    wsRef.current = ws;
    attachSocketListeners(ws);
  };

  useEffect(() => {
    connectSocket();
    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'pair_leave' }));
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const closeScanner = () => {
    handledScanRef.current = false;
    setScannerOpen(false);
  };

  const onPressScan = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('카메라 권한 필요', 'QR 코드를 스캔하려면 카메라 권한이 필요합니다.');
        return;
      }
    }
    handledScanRef.current = false;
    setScannerOpen(true);
  };

  const requestJoinRoom = (roomId: string) => {
    pendingJoinRoomRef.current = roomId;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      sendJson({ type: 'pair_join', roomId, myLang: myLang.code });
      setStatusText('상대방과 연결 중...');
      return;
    }
    wsRef.current?.close();
    wsRef.current = null;
    connectSocket();
  };

  const handleScannedData = (raw: string) => {
    try {
      const payload = JSON.parse(raw) as PairQrPayload;
      if (payload.type !== 'stity-pair-v2' || !payload.roomId) {
        throw new Error('invalid');
      }
      if (payload.roomId === roomIdRef.current) {
        throw new Error('self');
      }

      closeScanner();
      requestJoinRoom(payload.roomId);
    } catch (error: any) {
      const message =
        error?.message === 'self'
          ? '내 QR 코드는 스캔할 수 없습니다.'
          : 'STiTy 2이어폰 연결용 QR 코드가 아닙니다.';
      Alert.alert('잘못된 QR 코드', message);
      handledScanRef.current = false;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <View style={styles.qrBlock}>
          <Text style={styles.title}>상대와 연동</Text>
          <View style={styles.qrContainer}>
            <QRCode value={qrValue} size={176} />
          </View>
          {statusText ? <Text style={styles.statusText}>{statusText}</Text> : null}
        </View>

        <View style={styles.buttonRow}>
          <GradientButton title="스캔하기" onPress={onPressScan} />
          <GradientButton title="돌아가기" onPress={() => navigation.goBack()} />
        </View>
      </View>

      {scannerOpen && (
        <View style={styles.scannerOverlay}>
          <CameraView
            style={StyleSheet.absoluteFillObject}
            barcodeScannerSettings={{ barcodeTypes: ['qr'] }}
            onBarcodeScanned={({ data }: { data: string }) => {
              if (handledScanRef.current) return;
              handledScanRef.current = true;
              handleScannedData(data);
            }}
          />
          <View style={styles.scannerBottom}>
            <Text style={styles.scannerHint}>상대 QR 코드를 화면 중앙에 맞춰주세요.</Text>
            <TouchableOpacity onPress={closeScanner} style={styles.cancelButton}>
              <Text style={styles.cancelText}>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
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
  qrBlock: {
    alignItems: 'center',
  },
  title: {
    position: 'absolute',
    top: -90,
    fontSize: FONTS.sizes.xl,
    fontWeight: '700',
    color: COLORS.textPrimary,
  },
  qrContainer: {
    backgroundColor: '#FFFFFF',
    padding: SPACING.md,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: SPACING.lg,
  },
  statusText: {
    fontSize: FONTS.sizes.sm,
    color: COLORS.textSecondary,
    marginTop: SPACING.lg,
  },
  buttonRow: {
    position: 'absolute',
    bottom: 140,
    flexDirection: 'row',
    gap: SPACING.md,
  },
  scannerOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000000',
  },
  scannerBottom: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 88,
    alignItems: 'center',
    paddingHorizontal: SPACING.xl,
  },
  scannerHint: {
    color: '#FFFFFF',
    fontSize: FONTS.sizes.md,
    textAlign: 'center',
    marginBottom: SPACING.md,
  },
  cancelButton: {
    borderWidth: 1,
    borderColor: '#FFFFFF',
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  cancelText: {
    color: '#FFFFFF',
    fontSize: FONTS.sizes.md,
    fontWeight: '600',
  },
});

export default PairingScreen;
