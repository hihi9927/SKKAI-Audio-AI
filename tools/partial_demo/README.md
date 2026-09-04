# partial 스트리밍 데모

토큰 단위 `partial` 메시지가 실제로 어떻게 보이는지 브라우저에서 확인하는 도구다.
모바일 앱(`STiTy-Mobile/src/screens/HomeScreen.tsx`)의 디자인을 그대로 옮겨 놓았으므로,
앱에 반영하기 전에 화면 동작을 눈으로 먼저 본다.

```
tools/partial_demo/
├── run_server.sh          ASR 서버 기동 (워크트리 코드 + 로컬 번역)
├── demo_proxy.py          페이지와 WebSocket 을 한 포트로 묶는 프록시
├── web/index.html         데모 페이지
├── web/partial_test.wav   테스트 음성 (git 에 넣지 않는다)
├── make_test_wav.py       LibriSpeech 로 테스트 음성 만들기
└── partial_ws_client.py   CLI 검증 클라이언트 (자동 판정 포함)
```

## 띄우기

오래 도는 작업이므로 tmux 로 띄운다. 백그라운드 실행은 세션이 끊기면 같이 죽는다.

```bash
cd /path/to/repo
tmux new-session -d -s partialtest -c . "bash tools/partial_demo/run_server.sh"
tmux new-session -d -s partialweb  -c . "python -u tools/partial_demo/demo_proxy.py 8080"
```

서버가 뜨는 데 2~3분 걸린다. `tools/partial_demo/server.log` 에 `Warmup complete` 가 찍히면 준비된 것이다.

그다음 <http://localhost:8080> 를 연다. 원격 개발이면 **8080 하나만** 포워딩하면 된다 —
페이지가 같은 포트의 `/ws` 로 붙고 프록시가 8766 으로 중계한다.

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PYTHON` | `/home/mobility/STiTy/.venv/bin/python` | vLLM 이 설치된 파이썬 |
| `PORT` | `8766` | ASR 서버 포트 |

## 페이지 동작

- **테스트 음성** — `web/partial_test.wav` 를 실시간 속도로 전송한다.
- **마이크** — 브라우저 마이크를 그대로 보낸다. localhost 는 secure context 라 권한만 주면 된다.
- 미확정 `partial` 은 흐린 말풍선에 등간격으로 한 글자씩 찍힌다. 서버는 통째로 교체할
  전체 문자열을 보내고, 타자 연출은 순수하게 표시 쪽 처리다.
- `final` 이 오면 새 말풍선을 만들지 않고 그 말풍선을 그대로 확정시킨다(진해지고 텍스트만
  번역/원문으로 교체). 목록을 다시 그리지 않으므로 화면이 튀지 않는다.
- 말풍선 아래 작은 글씨로 `partial · seq=N · +T초` / `final · vad · +T초` 가 찍힌다.

## 테스트 음성 다시 만들기

LibriSpeech 오디오는 저장소에 없다. 데이터가 있는 경로를 넘긴다.

```bash
python tools/partial_demo/make_test_wav.py tools/partial_demo/web/partial_test.wav 3 \
  /path/to/evaluation/LibriSpeech/LibriSpeech/test-other
```

발화 사이에 1.2초 침묵을 넣는다. VAD 커밋과 슬롯 리셋을 실제로 태워서, 그 지점에
빈 `partial`(화면 비우기 신호)이 나가는지 보기 위해서다.

## CLI 로 검증

브라우저 없이 프로토콜만 확인한다. `final` 직후 빈 `partial` 이 오는지, 마지막 메시지가
빈 `partial` 인지, `seq` 가 단조 증가하는지, 메타 헤더(`language ...`)가 새지 않는지 판정한다.

```bash
python tools/partial_demo/partial_ws_client.py \
  --url ws://127.0.0.1:8766 --wav tools/partial_demo/web/partial_test.wav

# 무음만 넣었을 때 partial/final 이 안 나오는지
python tools/partial_demo/partial_ws_client.py \
  --url ws://127.0.0.1:8766 --wav silence.wav --expect-silence
```

## 번역

`run_server.sh` 는 `--local-translation` 으로 띄운다. `core/local_translator.py` 의
NLLB-200-distilled-600M 을 로컬 GPU 에 올려 번역하므로 외부 호출이 없고, API 키 없이
쓰던 무료 gtx 엔드포인트의 429 를 안 만난다.
