import React, { createContext, useContext, useState, useCallback, useRef } from 'react';

// ===== Qwen3-ASR 서버 설정 =====
interface WebSocketConfig {
  lang: string;  // 언어 코드: 'auto', 'ko', 'en', 'zh', 'ja' 등
}

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: any;
  error: string | null;
  connect: (config: WebSocketConfig) => Promise<void>;
  disconnect: () => void;
  sendAudio: (audioData: ArrayBuffer) => void;
  sendMessage: (message: object) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// ===== Qwen3-ASR 서버 URL 설정 =====
// ngrok 터널 사용 시: `ngrok http 8765` 실행 후 URL 입력
// 로컬 네트워크 사용 시: 'ws://192.168.x.x:8765'
const SERVER_URL = 'ws://localhost:8765';

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(async (config: WebSocketConfig): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }

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

              if (data.type === 'hello') {
                clearTimeout(timeout);
                setIsConnected(true);
                setError(null);

                // Qwen3-ASR start 메시지
                const startMessage = {
                  type: 'start',
                  lang: config.lang,  // 'auto', 'ko', 'en', 'zh', 'ja' 등
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
    setError(null);
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

  return (
    <WebSocketContext.Provider value={{
      isConnected, lastMessage, error,
      connect, disconnect, sendAudio, sendMessage,
    }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocketContext = (): WebSocketContextType => {
  const ctx = useContext(WebSocketContext);
  if (!ctx) throw new Error('useWebSocketContext must be used within WebSocketProvider');
  return ctx;
};
