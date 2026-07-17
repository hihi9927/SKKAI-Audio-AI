import { Platform } from 'react-native';

const SERVER_STARTER_URL =
  'https://r61duqe7w4.execute-api.ap-northeast-2.amazonaws.com/default/Serverstarter_Speech_AI';

const POLL_INTERVAL_MS = 10000;
const MAX_SERVER_START_WAIT_MS = 4 * 60 * 1000;

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('Aborted', 'AbortError')); }, { once: true });
  });

export const startServer = async (signal?: AbortSignal): Promise<void> => {
  // 웹: 이 환경의 로컬 서버가 항상 떠 있으므로 AWS 서버스타터(CORS 차단 대상)를 건너뜀.
  if (Platform.OS === 'web') return;

  const deadline = Date.now() + MAX_SERVER_START_WAIT_MS;

  while (Date.now() < deadline) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    const res = await fetch(SERVER_STARTER_URL, { signal });
    const data = await res.json();

    if (data.status === 'ready') return;

    if (data.status === 'starting' || data.status === 'stopping') {
      await sleep(POLL_INTERVAL_MS, signal);
      continue;
    }

    throw new Error('Failed to start server');
  }

  throw new Error('Server start timeout');
};
