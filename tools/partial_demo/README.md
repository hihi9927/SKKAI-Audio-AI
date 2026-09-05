# partial 스트리밍 데모

토큰 단위 `partial` 메시지가 실제로 어떻게 보이는지 브라우저에서 확인하는 도구다.
모바일 앱(`STiTy-Mobile/src/screens/HomeScreen.tsx`)의 디자인을 그대로 옮겨 놓았으므로,
앱에 반영하기 전에 화면 동작을 눈으로 먼저 본다.

페이지는 둘이다. `index.html` 은 앱 화면을 그대로 옮긴 **개발 검증용**,
`show.html` 은 검은 바탕에 자막만 띄우는 **시연용**이다.

```
tools/partial_demo/
├── run_server.sh          ASR 서버 기동 (워크트리 코드 + 로컬 번역)
├── demo_proxy.py          페이지와 WebSocket 을 한 포트로 묶는 프록시
├── web/index.html         개발 검증용 페이지 (앱 화면 그대로)
├── web/show.html          시연용 자막 페이지
├── web/partial_test*.wav  테스트 음성 (git 에 넣지 않는다)
├── make_test_wav.py       LibriSpeech 로 테스트 음성 만들기 (index.html 용)
├── make_demo_wavs.py      en/ja/ko 테스트 음성 만들기 (show.html 용)
├── mock_show_server.py    ASR 없이 show.html 만 확인하는 목 서버
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

그다음 <http://localhost:8080> (개발 검증용) 또는 <http://localhost:8080/show.html>
(시연용) 을 연다. 원격 개발이면 **8080 하나만** 포워딩하면 된다 — 페이지가 같은 포트의
`/ws` 로 붙고 프록시가 8766 으로 중계한다.

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

## 시연용 페이지 (`show.html`)

검은 바탕에 자막만 뜬다. 확정된 문장은 화면 세로 한가운데에 최대 밝기로 잠시 머물고,
다음 문장이 오면 위로 밀리며 흐려진다.

- **머무는 시간을 보장한다.** 확정된 문장은 최소 `hold`(기본 2600ms) 동안 그 자리에
  있는다. 너무 빨리 지나가면 청중이 못 읽는다. 그동안 들어온 `partial` 은 버퍼에,
  `final` 은 대기열에 쌓였다가 시간이 끝나면 순서대로 풀린다.
- 밀리면 `hold` 와 타자 예산을 backlog 에 비례해 줄여 실시간을 따라잡는다(하한 `holdmin`).
  대기열에서 풀린 `final` 도 그냥 튀어나오지 않고 짧아진 예산으로 한 번 흘린 뒤 확정된다.
- 전사(`partial`)는 원문이라 **작은 글씨**로 흐르고, 확정되면 번역이 큰 글씨로 자리를 잡는다.
- 발화는 잡혔는데 아직 디코딩 결과가 없으면 `•••` 이 다음 문장이 뜰 자리 아래에 나온다.

다국어 배치 두 가지를 `1` / `2` 키(또는 우상단 버튼)로 바꾼다. 선택은 localStorage 에 남는다.

| 키 | 배치 |
|---|---|
| `1` | 한 칸에 모아 쓰고 문장 앞에 언어 태그를 붙인다. 태그 색이 언어마다 다르다 |
| `2` | 언어별로 칸을 나눠 세로로 배치한다. 칸은 언어가 처음 나온 순서로 생긴다 |

`R` 은 화면 초기화. `EN`/`JA`/`KO` 버튼은 해당 언어의 테스트 음성을 흘린다(그 언어를
핸드셰이크에 실어 보낸다). `마이크` 는 `src` 설정(기본 `auto`)을 쓴다.

동작 값은 쿼리 스트링으로 덮어쓴다 — 시연장에서 값만 바꿔 다시 열면 된다.

| 파라미터 | 기본 | 뜻 |
|---|---|---|
| `layout` | `tag` | `tag` / `lane` |
| `hold` | `2600` | 확정 문장이 머무는 최소 시간(ms) |
| `holdmin` | `900` | 밀렸을 때 줄일 수 있는 하한(ms) |
| `type` | `1300` | 한 덩어리를 다 타이핑하는 목표 시간(ms) |
| `src` / `tgt` | `auto` / `ko` | `start` 핸드셰이크의 입력/목표 언어 |

### ASR 없이 화면만 보기

GPU 없이 배치와 페이싱만 볼 때 쓴다. 대본을 실제 발화 속도로 흘리고, 뒷부분 세 문장은
0.2~0.3초 간격으로 몰아쳐 유지 시간과 대기열이 도는지 보여 준다.

```bash
tmux new-session -d -s showmock -c . \
  "python -u tools/partial_demo/mock_show_server.py 8090 --loop"
```

<http://localhost:8090/show.html> 를 열고 `EN`/`JA`/`KO` 중 아무거나 누르면 시작한다
(목 서버는 무음 wav 를 내주므로 테스트 음성 파일이 없어도 된다).

## 테스트 음성 다시 만들기

LibriSpeech 오디오는 저장소에 없다. 데이터가 있는 경로를 넘긴다.

```bash
python tools/partial_demo/make_test_wav.py tools/partial_demo/web/partial_test.wav 3 \
  /path/to/evaluation/LibriSpeech/LibriSpeech/test-other
```

발화 사이에 1.2초 침묵을 넣는다. VAD 커밋과 슬롯 리셋을 실제로 태워서, 그 지점에
빈 `partial`(화면 비우기 신호)이 나가는지 보기 위해서다.

시연용(`show.html`)은 언어별 파일을 따로 쓴다. 경로가 스크립트 안에 박혀 있어 인자 없이 돈다.

```bash
python tools/partial_demo/make_demo_wavs.py                 # en, ja, ko 모두
python tools/partial_demo/make_demo_wavs.py --lang ja --n 4
```

`web/partial_test_<lang>.wav` 와, 무엇이 나와야 하는지 대조할 `web/partial_test_<lang>.txt`
가 함께 나온다. 출처는 언어마다 다르다.

| 언어 | 출처 | 비고 |
|---|---|---|
| en, ja | FLEURS dev (`/home/mobility/datasets/fleurs/data/{en_us,ja_jp}`) | 낭독체. 구두점 있는 원문 전사가 붙는다 |
| ko | KsponSpeech `sample_data/eval_clean` | FLEURS `ko_kr` 에는 오디오가 없다. 대화체라 나머지 둘과 결이 다르다 |

FLEURS TSV 는 본문에 따옴표가 그대로 들어 있어 `csv.QUOTE_NONE` 으로 읽는다.
기본 파서로 읽으면 여러 행이 한 필드로 붙어 수천 어절짜리 잔해가 섞인다.

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
