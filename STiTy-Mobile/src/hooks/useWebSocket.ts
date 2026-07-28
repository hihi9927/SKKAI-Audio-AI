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

// ===== Server URL =====
// ngrok example: 'wss://abcd-1234.ngrok-free.app'
// LAN example: 'ws://192.168.x.x:8001'
const SERVER_URL = 'wss://h2b9loruk400f8-8765.proxy.runpod.net';

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
          reject(new Error('Connection timeout'));
        }, 10000);

        ws.onopen = () => {
          console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
          try {
            if (typeof event.data === 'string') {
              const data = JSON.parse(event.data);
              console.log('Received message:', data.type);

              // When the server sends "hello", consider connection ready and start streaming.
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
          setError('Connection error occurred');
          setIsConnected(false);
          reject(new Error('Failed to connect to server'));
        };

        ws.onclose = (event) => {
          clearTimeout(timeout);
          console.log('WebSocket closed:', event.code, event.reason);
          setIsConnected(false);

          if (event.code !== 1000) {
            setError('Connection closed unexpectedly');
          }
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Failed to create WebSocket:', err);
        setError('Failed to initialize connection');
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
