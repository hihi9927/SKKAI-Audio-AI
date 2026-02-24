import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

// ===== Qwen3-ASR 서버 설정 =====
interface WebSocketConfig {
  lang: string;  // 언어 코드: 'auto', 'ko', 'en', 'zh', 'ja' 등
  targetLang?: string;  // 번역 대상 언어 코드
}

export type ServerStatus = 'idle' | 'connecting' | 'ready' | 'error';

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: any;
  error: string | null;
  serverStatus: ServerStatus;
  connect: (config: WebSocketConfig) => Promise<void>;
  disconnect: () => void;
  sendAudio: (audioData: ArrayBuffer) => void;
  sendMessage: (message: object) => void;
  addMessageListener: (listener: (msg: any) => void) => () => void;
  probeServer: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// ===== Qwen3-ASR 서버 URL 설정 =====
const SERVER_URL = 'wss://supervigorously-unforded-lavone.ngrok-free.dev';

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const probeWsRef = useRef<WebSocket | null>(null);
  const isProbingRef = useRef(false);
  const serverStatusRef = useRef<ServerStatus>('idle');
  const listenersRef = useRef<Set<(msg: any) => void>>(new Set());

  useEffect(() => { serverStatusRef.current = serverStatus; }, [serverStatus]);

  const probeServer = useCallback(() => {
    if (isProbingRef.current || serverStatusRef.current === 'ready') return;
    isProbingRef.current = true;
    setServerStatus('connecting');

    if (probeWsRef.current && probeWsRef.current.readyState !== WebSocket.CLOSED) {
      probeWsRef.current.close();
    }

    try {
      const ws = new WebSocket(SERVER_URL);
      ws.binaryType = 'arraybuffer';
      let helloReceived = false;

      const timeout = setTimeout(() => {
        if (!helloReceived) {
          ws.close();
          setServerStatus('error');
          isProbingRef.current = false;
        }
      }, 30000);

      ws.onmessage = (event) => {
        try {
          if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            if (data.type === 'hello') {
              clearTimeout(timeout);
              helloReceived = true;
              setServerStatus('ready');
              isProbingRef.current = false;
              ws.close(1000, 'Probe complete');
            }
          }
        } catch (e) {}
      };

      ws.onerror = () => {
        clearTimeout(timeout);
        if (!helloReceived) {
          setServerStatus('error');
          isProbingRef.current = false;
        }
      };

      ws.onclose = () => {
        if (!helloReceived) {
          setServerStatus('error');
          isProbingRef.current = false;
        }
      };

      probeWsRef.current = ws;
    } catch (err) {
      setServerStatus('error');
      isProbingRef.current = false;
    }
  }, []);

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
                  lang: config.lang,
                  targetLang: config.targetLang || '',
                  translate: !!config.targetLang,
                };
                ws.send(JSON.stringify(startMessage));
                resolve();
                return;
              }

              setLastMessage(data);
              listenersRef.current.forEach(l => l(data));
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

  const addMessageListener = useCallback((listener: (msg: any) => void) => {
    listenersRef.current.add(listener);
    return () => { listenersRef.current.delete(listener); };
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
      isConnected, lastMessage, error, serverStatus,
      connect, disconnect, sendAudio, sendMessage, addMessageListener, probeServer,
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
