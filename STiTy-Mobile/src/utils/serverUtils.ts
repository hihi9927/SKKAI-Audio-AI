const SERVER_STARTER_URL =
  'https://r61duqe7w4.execute-api.ap-northeast-2.amazonaws.com/default/Serverstarter_Speech_AI';

const POLL_INTERVAL_MS = 10000;
const MAX_SERVER_START_WAIT_MS = 10 * 60 * 1000;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const startServer = async (): Promise<void> => {
  const deadline = Date.now() + MAX_SERVER_START_WAIT_MS;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(SERVER_STARTER_URL);
      const data = await res.json();
      if (data.status === 'ready') return;
    } catch {
      // 네트워크 오류 등 과도기 상태 → 재시도
    }
    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error('Server start timeout');
};
