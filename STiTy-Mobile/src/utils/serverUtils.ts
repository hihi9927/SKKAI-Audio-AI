const SERVER_STARTER_URL = 'https://r61duqe7w4.execute-api.ap-northeast-2.amazonaws.com/default/Serverstarter_Speech_AI';

export const startServer = async (): Promise<void> => {
  const res = await fetch(SERVER_STARTER_URL);
  const data = await res.json();

  if (data.status === 'ready') return;

  // starting 또는 stopping: 최대 3분 폴링
  if (data.status === 'starting' || data.status === 'stopping') {
    for (let i = 0; i < 18; i++) {  // 최대 3분
      await new Promise(r => setTimeout(r, 10000));
      const check = await fetch(SERVER_STARTER_URL);
      const result = await check.json();
      if (result.status === 'ready') return;
      if (result.status === 'starting') continue;
      if (result.status === 'stopping') continue;
    }
    throw new Error('서버 시작 시간 초과');
  }

  throw new Error('서버를 시작할 수 없습니다');
};
