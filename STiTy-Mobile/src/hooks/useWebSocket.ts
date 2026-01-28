import { useState, useCallback, useRef, useEffect } from 'react';

interface WebSocketConfig {
  myLang: string;
  targetLang: string;
  mode: string;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: any;
  error: string | null;
  connect: (config: WebSocketConfig) => Promise<void>;
  disconnect: () => void;
  sendAudio: (audioData: ArrayBuffer) => void;
  sendMessage: (message: object) => void;
}

// ===== 서버 URL 설정 =====
// ngrok 터널 사용 시: 터미널에서 `ngrok http 8001` 실행 후 URL 입력
// 예: 'wss://abcd-1234.ngrok-free.app'
// 로컬 네트워크 사용 시: 'ws://192.168.x.x:8001'
const SERVER_URL = 'wss://sportless-postpituitary-ludie.ngrok-free.dev';

export const useWebSocket = (): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const configRef = useRef<WebSocketConfig | null>(null);

  const connect = useCallback(async (config: WebSocketConfig): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }

        configRef.current = config;
        const ws = new WebSocket(SERVER_URL);
        ws.binaryType = 'arraybuffer';

        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('연결 시간이 초과되었습니다'));
        }, 10000);

        ws.onopen = () => {
          console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
          try {
            if (typeof event.data === 'string') {
              const data = JSON.parse(event.data);
              console.log('Received message:', data.type);

              // 서버가 hello를 보내면 연결 성공 → start 메시지 전송
              if (data.type === 'hello') {
                clearTimeout(timeout);
                setIsConnected(true);
                setError(null);

                const startMessage = {
                  type: 'start',
                  lang: config.myLang,
                  targetLang: config.targetLang,
                  displayMode: 'both',
                };
                ws.send(JSON.stringify(startMessage));
                resolve();
                return;
              }

              setLastMessage(data);
            }
          } catch (e) {
            console.error('Failed to parse message:', e);
          }
        };

        ws.onerror = (event) => {
          clearTimeout(timeout);
          console.error('WebSocket error:', event);
          setError('연결 오류가 발생했습니다');
          setIsConnected(false);
          reject(new Error('서버에 연결할 수 없습니다'));
        };

        ws.onclose = (event) => {
          clearTimeout(timeout);
          console.log('WebSocket closed:', event.code, event.reason);
          setIsConnected(false);

          if (event.code !== 1000) {
            setError('연결이 끊어졌습니다');
          }
        };

        wsRef.current = ws;

      } catch (err) {
        console.error('Failed to create WebSocket:', err);
        setError('연결을 시작할 수 없습니다');
        reject(err);
      }
    });
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'stop' }));
        } catch (e) {
          console.error('Failed to send stop message:', e);
        }
      }

      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    setIsConnected(false);
    setLastMessage(null);
  }, []);

  const sendAudio = useCallback((audioData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(audioData);
      } catch (e) {
        console.error('Failed to send audio:', e);
      }
    }
  }, []);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (e) {
        console.error('Failed to send message:', e);
      }
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
    sendAudio,
    sendMessage,
  };
};

export default useWebSocket;
