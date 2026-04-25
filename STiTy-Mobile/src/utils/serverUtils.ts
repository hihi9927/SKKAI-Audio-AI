const SERVER_STARTER_URL =
  'https://r61duqe7w4.execute-api.ap-northeast-2.amazonaws.com/default/Serverstarter_Speech_AI';

const POLL_INTERVAL_MS = 10000;
const MAX_SERVER_START_WAIT_MS = 4 * 60 * 1000;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const startServer = async (): Promise<void> => {
  const deadline = Date.now() + MAX_SERVER_START_WAIT_MS;

  while (Date.now() < deadline) {
    const res = await fetch(SERVER_STARTER_URL);
    const data = await res.json();

    if (data.status === 'ready') return;

    if (data.status === 'starting' || data.status === 'stopping') {
      await sleep(POLL_INTERVAL_MS);
      continue;
    }

    throw new Error('Failed to start server');
  }

  throw new Error('Server start timeout');
};
