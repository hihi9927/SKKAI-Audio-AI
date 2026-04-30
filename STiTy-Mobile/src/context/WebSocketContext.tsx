import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { startServer } from '../utils/serverUtils';

interface WebSocketConfig {
  lang: string;
  targetLang?: string;
}

export type ServerStatus = 'idle' | 'ec2-starting' | 'connecting' | 'ready' | 'error';

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
  probeServer: (force?: boolean) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

const SERVER_URL = 'wss://edra-raspiest-eagerly.ngrok-free.dev';
const PROBE_SOCKET_TIMEOUT_MS = 15000;
const PROBE_RETRY_INTERVAL_MS = 10000;
const MAX_WEBSOCKET_PROBE_WAIT_MS = 4 * 60 * 1000;
const KEEPALIVE_RECONNECT_MS = 3000;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus>('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const probeWsRef = useRef<WebSocket | null>(null);
  const keepAliveWsRef = useRef<WebSocket | null>(null);
  const keepAliveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const keepAliveEnabledRef = useRef(false);
  const isProbingRef = useRef(false);
  const serverStatusRef = useRef<ServerStatus>('idle');
  const listenersRef = useRef<Set<(msg: any) => void>>(new Set());
  const keepAliveFailCountRef = useRef(0);
  const probeAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    serverStatusRef.current = serverStatus;
  }, [serverStatus]);

  const clearKeepAliveTimer = useCallback(() => {
    if (keepAliveTimerRef.current) {
      clearTimeout(keepAliveTimerRef.current);
      keepAliveTimerRef.current = null;
    }
  }, []);

  const stopKeepAlive = useCallback(() => {
    keepAliveEnabledRef.current = false;
    keepAliveFailCountRef.current = 0;
    clearKeepAliveTimer();
    if (keepAliveWsRef.current) {
      keepAliveWsRef.current.close(1000, 'Stop keepalive');
      keepAliveWsRef.current = null;
    }
  }, [clearKeepAliveTimer]);

  const startKeepAlive = useCallback(() => {
    keepAliveEnabledRef.current = true;

    const connectKeepAlive = () => {
      if (!keepAliveEnabledRef.current) return;
      if (wsRef.current) return; // real session is active
      if (
        keepAliveWsRef.current &&
        (keepAliveWsRef.current.readyState === WebSocket.OPEN ||
          keepAliveWsRef.current.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }

      const ws = new WebSocket(SERVER_URL);
      ws.binaryType = 'arraybuffer';
      keepAliveWsRef.current = ws;

      ws.onopen = () => {
        keepAliveFailCountRef.current = 0;
      };

      ws.onmessage = () => {
        // keepalive socket just needs to stay connected
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onclose = () => {
        if (keepAliveWsRef.current === ws) {
          keepAliveWsRef.current = null;
        }
        if (keepAliveEnabledRef.current && !wsRef.current) {
          keepAliveFailCountRef.current += 1;
          if (keepAliveFailCountRef.current >= 3 && serverStatusRef.current === 'ready') {
            setServerStatus('idle');
            keepAliveFailCountRef.current = 0;
          }
          clearKeepAliveTimer();
          keepAliveTimerRef.current = setTimeout(connectKeepAlive, KEEPALIVE_RECONNECT_MS);
        }
      };
    };

    connectKeepAlive();
  }, [clearKeepAliveTimer]);

  const probeServer = useCallback(async (force = false) => {
    if (isProbingRef.current) return;
    if (!force && serverStatusRef.current === 'ready' && keepAliveWsRef.current?.readyState === WebSocket.OPEN) return;
    isProbingRef.current = true;
    stopKeepAlive();
    setServerStatus('ec2-starting');

    probeAbortRef.current?.abort();
    const abort = new AbortController();
    probeAbortRef.current = abort;

    if (probeWsRef.current && probeWsRef.current.readyState !== WebSocket.CLOSED) {
      probeWsRef.current.close();
    }

    try {
      await startServer(abort.signal);
      if (abort.signal.aborted) return;
      setServerStatus('connecting');

      const tryProbe = (): Promise<void> =>
        new Promise<void>((resolve, reject) => {
          const ws = new WebSocket(SERVER_URL);
          ws.binaryType = 'arraybuffer';
          probeWsRef.current = ws;

          let settled = false;
          let helloReceived = false;

          const settle = (fn: () => void) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);
            fn();
          };

          const timeout = setTimeout(() => {
            if (!helloReceived) {
              ws.close();
              settle(() => reject(new Error('timeout')));
            }
          }, PROBE_SOCKET_TIMEOUT_MS);

          ws.onmessage = event => {
            try {
              if (typeof event.data !== 'string') return;
              const data = JSON.parse(event.data);
              if (data.type === 'hello') {
                helloReceived = true;
                ws.close(1000, 'Probe complete');
                settle(() => resolve());
              }
            } catch {
              // ignore
            }
          };

          ws.onerror = () => {
            if (!helloReceived) settle(() => reject(new Error('error')));
          };

          ws.onclose = () => {
            if (!helloReceived) settle(() => reject(new Error('closed')));
          };
        });

      const deadline = Date.now() + MAX_WEBSOCKET_PROBE_WAIT_MS;
      let connected = false;

      while (Date.now() < deadline) {
        try {
          await tryProbe();
          connected = true;
          break;
        } catch {
          await sleep(PROBE_RETRY_INTERVAL_MS);
        }
      }

      if (!connected) throw new Error('Server connection timeout');
      setServerStatus('ready');
      startKeepAlive();
    } catch (e: any) {
      if (e?.name === 'AbortError') return; // app closed mid-probe, silently exit
      setServerStatus('error');
      stopKeepAlive();
    } finally {
      isProbingRef.current = false;
    }
  }, [startKeepAlive, stopKeepAlive]);

  const connect = useCallback(async (config: WebSocketConfig): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }

        stopKeepAlive();
        const ws = new WebSocket(SERVER_URL);
        ws.binaryType = 'arraybuffer';

        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('Connection timed out'));
        }, 15000);

        ws.onopen = () => {
          console.log('WebSocket connected');
        };

        ws.onmessage = event => {
          try {
            if (typeof event.data !== 'string') return;
            const data = JSON.parse(event.data);
            console.log('Received message:', data.type);

            if (data.type === 'hello') {
              clearTimeout(timeout);
              setIsConnected(true);
              setError(null);

              ws.send(
                JSON.stringify({
                  type: 'start',
                  lang: config.lang,
                  targetLang: config.targetLang || '',
                  translate: !!config.targetLang,
                })
              );
              resolve();
              return;
            }

            setLastMessage(data);
            listenersRef.current.forEach(listener => listener(data));
          } catch (e) {
            console.error('Failed to parse message:', e);
          }
        };

        ws.onerror = event => {
          clearTimeout(timeout);
          console.error('WebSocket error:', event);
          setError('Connection error');
          setIsConnected(false);
          reject(new Error('Cannot connect to server'));
        };

        ws.onclose = event => {
          clearTimeout(timeout);
          console.log('WebSocket closed:', event.code, event.reason);
          setIsConnected(false);
          if (event.code !== 1000) {
            setError('Connection lost');
          }
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Failed to create WebSocket:', err);
        setError('Cannot start connection');
        reject(err);
      }
    });
  }, [stopKeepAlive]);

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
    if (serverStatusRef.current === 'ready') {
      startKeepAlive();
    }
  }, [startKeepAlive]);

  useEffect(() => {
    return () => {
      probeAbortRef.current?.abort();
      stopKeepAlive();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Provider unmount');
      }
      if (probeWsRef.current) {
        probeWsRef.current.close(1000, 'Provider unmount');
      }
    };
  }, [stopKeepAlive]);

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
    return () => {
      listenersRef.current.delete(listener);
    };
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
    <WebSocketContext.Provider
      value={{
        isConnected,
        lastMessage,
        error,
        serverStatus,
        connect,
        disconnect,
        sendAudio,
        sendMessage,
        addMessageListener,
        probeServer,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocketContext = (): WebSocketContextType => {
  const ctx = useContext(WebSocketContext);
  if (!ctx) throw new Error('useWebSocketContext must be used within WebSocketProvider');
  return ctx;
};
