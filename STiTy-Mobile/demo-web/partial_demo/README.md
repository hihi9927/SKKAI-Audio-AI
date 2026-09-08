# partial 스트리밍 데모

토큰 단위 `partial` 메시지가 실제로 어떻게 보이는지 브라우저에서 확인하는 도구다.
모바일 앱(`STiTy-Mobile/src/screens/HomeScreen.tsx`)의 디자인을 그대로 옮겨 놓았으므로,
앱에 반영하기 전에 화면 동작을 눈으로 먼저 본다.

페이지는 둘이다. `index.html` 은 앱 화면을 그대로 옮긴 **개발 검증용**,
`show.html` 은 검은 바탕에 자막만 띄우는 **시연용**이다.

```
STiTy-Mobile/demo-web/partial_demo/
├── run_server.sh          ASR 서버 기동 (워크트리 코드 + 로컬 번역)
├── demo_proxy.py          페이지와 WebSocket 을 한 포트로 묶는 프록시
├── web/index.html         개발 검증용 페이지 (앱 화면 그대로)
├── web/show.html          시연용 자막 페이지
├── web/stity_logo.png     시연 페이지 왼쪽 위 로고 (assets/logo 에서 배경을 지워 줄인 것)
├── web/partial_test*.wav  테스트 음성 (git 에 넣지 않는다)
├── make_test_wav.py       LibriSpeech 로 테스트 음성 만들기 (index.html 용)
├── make_demo_wavs.py      en/ja/ko 테스트 음성 만들기 (show.html 용)
├── mock_show_server.py    ASR 없이 show.html 만 확인하는 목 서버
└── partial_ws_client.py   CLI 검증 클라이언트 (자동 판정 포함). 프록시에 붙일 때는
                           --url ws://127.0.0.1:8080/ws --lang-map ko=en,en=ko,es=ko
```

## 띄우기

오래 도는 작업이므로 tmux 로 띄운다. 백그라운드 실행은 세션이 끊기면 같이 죽는다.

```bash
cd /path/to/repo
tmux new-session -d -s partialtest -c . "bash STiTy-Mobile/demo-web/partial_demo/run_server.sh"
tmux new-session -d -s partialweb  -c . "python -u STiTy-Mobile/demo-web/partial_demo/demo_proxy.py 8080"
```

서버가 뜨는 데 2~3분 걸린다. `STiTy-Mobile/demo-web/partial_demo/server.log` 에 `Warmup complete` 가 찍히면 준비된 것이다.

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

- **앞선 문장을 흐리게 하지 않는다.** 밀려 올라가도 계속 읽히므로 최신 문장을 붙잡아 둘
  이유가 없다. 그래서 `hold` 가 기본 0 이고 문장은 오는 대로 곧바로 뜬다. 지연이 없다.
- `?fade=1` 을 주면 깊이만큼 흐려지는 대신 `hold` 가 2600ms 로 붙는다. 흐려진 문장은
  못 읽으니 최신 문장을 그만큼 붙잡아 둬야 하고, **그 붙잡는 시간이 곧 지연이다.**
  그동안 들어온 `partial` 은 버퍼에, `final` 은 대기열에 쌓였다가 순서대로 풀린다.
  밀리면 `hold` 와 타자 예산을 backlog 에 비례해 줄여 따라잡는다(하한 `holdmin`).
- **한 문장은 두 칸으로 고정이다.** 위가 번역(큰 글씨), 아래가 원문(작은 글씨)이고 태그가
  그 원문 왼쪽에 붙는다. 전사는 처음부터 아래 칸에 뜨고 번역은 위 칸이 비워 둔 자리에
  들어온다 — 그래서 번역이 도착해도 **이미 떠 있던 글자가 움직이지 않는다.** 줄 너비도
  글자 길이와 무관하게 고정이라 문장마다 좌우로 흔들리지 않는다.
- 전사(`partial`)는 원문이라 작은 글씨로 흐른다. `final` 이 오면 **기다리지 않고 바로**
  번역을 큰 글씨로 올린다(지연 우선). `?order=strict` 를 주면 전사를 끝까지 친 뒤
  `trans`(기본 500ms)만큼 두고 번역으로 넘어간다 — 순서는 지켜지지만 그만큼 번역이 늦게 뜬다.
- 발화는 잡혔는데 아직 디코딩 결과가 없으면 `●●●` 이 다음 문장이 뜰 자리 아래에 나온다.
- 왼쪽 위에 로고(`web/stity_logo.png`)가 뜬다. 조작부가 사라져도 남는다.

배치 다섯 가지를 `1`~`5` 키로 바꾼다. `Ctrl+L` 은 목표별 칸의 가로(`4`)와 세로(`5`)를 오간다.
앞의 둘은 **말한 언어**로 나누고, 뒤의 둘은 한 발화를 **여러 언어로 동시에** 띄우는 안이다.

| 키 | 배치 | 무엇을 보여 주나 |
|---|---|---|
| `1` | 언어 태그 | 한 칸에 모아 쓰고 원문 왼쪽에 태그. 태그 칸은 132px 고정이라 글자가 길든 짧든 자리가 같다 |
| `2` | 언어별 칸 | 말한 언어마다 가로 띠. 칸은 언어가 처음 나온 순서로 생긴다 |
| `3` | 목표 쌓기 | 목표마다 한 줄씩 쌓고 원문을 맨 아래 둔다. 왼쪽 칸은 번역 줄에서는 목표 언어 코드, 원문 줄에서는 화자 태그다 |
| `4` | 목표별 칸 | 언어마다 가로 띠. 칸이 곧 언어라 그 칸에는 그 언어 문장만 쭉 흐른다. 말한 언어의 칸에는 원문이 그대로 들어간다 |
| `5` | 목표별 세로 칸 | 같은 것을 세로 기둥으로 세운다. **칸 이름이 기둥 아래**에 붙고 글은 아래에서 위로 쌓인다 — 새 문장이 늘 이름 바로 위에 뜬다. 기둥이 좁아 글자와 줄 수를 그만큼 줄여 잡는다 |

가로로 칸을 쓰는 배치(`2`·`4`)는 **칸 이름을 왼쪽에 크게 고정한다.** 칸마다 같은 자리·같은
크기라 어느 칸인지 한눈에 잡힌다. 글 뭉치는 그 이름과 같은 높이에 오고, 이름 폭만큼 오른쪽에도
빈 자리를 둬서 글의 가운데가 화면 가운데에 온다.

`4` 와 `5` 는 칸을 채우는 규칙이 완전히 같다(`isLaneTarget`). 다른 것은 CSS 의 기하뿐이다 —
어느 쪽이 시연장 화면 비율에 맞는지 보고 고른다. 옆으로 긴 화면이면 `4`, 세로로 긴 화면이나
언어가 셋 이하면 `5` 가 한 언어당 줄을 더 담는다.

**태그도 문장 뭉치의 세로 가운데에 놓는다**(`1`·`2`). 번역이 두 줄로 늘어나도 태그와 문장의
중심이 같은 높이에 남는다. 쌓기(`3`)만 예외인데, 왼쪽 칸을 줄마다 목표 언어 코드가 쓰고 있어
태그가 줄을 가로지르면 겹친다 — 거기서는 원문 줄에 그대로 둔다.

`3`·`4` 는 `final` 이 목표별 번역을 **여럿** 실어 와야 제 모습이 나온다. 서버는 아직
목표 하나(`translation`)만 보내므로, 지금 이 둘을 보려면 목 서버로 띄운다(아래 참고).
목표가 하나뿐이면 두 배치 모두 한 줄짜리로 줄어들 뿐 깨지지는 않는다.

**`3` 은 칸을 먼저 잡고 글자를 나중에 채운다.** 목표 수만큼 줄을 미리 비워 두므로
번역이 도착해도 이미 떠 있던 글자가 움직이지 않는다.

**`4` 는 칸끼리 줄을 맞추지 않는다.** 맞추려면 그 언어로 할 말이 없는 발화 자리에도 빈 줄을
넣어야 하는데, 그러면 읽을 것보다 여백이 많아진다. 그래서 칸마다 자기 문장만 이어 붙이고
줄 수도 칸마다 따로 센다. 글자를 조금 줄이고 아래 여백을 깎아 한 칸에 다섯 줄이 들어간다
(1600×900 기준). 더 넣고 싶으면 설정 패널의 **줄 수** 나 `?depth=N` 으로 올린다.

### 표시 목표 (`T` 키 / 설정 패널)

목표를 다 띄울지, 하나만 띄울지 고른다.

| 방식 | 화면 |
|---|---|
| **전부** | 목표를 모두 띄운다 (기본) |
| **하나 고정** | 고른 목표 하나만. 큰 화면은 한 언어만 띄우고 나머지는 각자 기기로 보는 안이다. `?pick=ja` |
| **자동 순환** | 몇 초마다 목표를 돌려 가며 하나씩. 간격은 `?rotate=4000`(ms) |

`?aud=ko,en,ja,zh` 로 청중 언어를 미리 정해 두면 첫 문장부터 칸을 다 잡고 시작한다.
그러지 않으면 새 목표가 처음 올 때마다 칸이 하나씩 늘면서 화면이 한 번씩 덜컹인다.

### 단축키

시연 화면에는 버튼이 없다. 조작은 키로 한다.

| 키 | 하는 일 |
|---|---|
| `Ctrl+Space` | 마이크 시작·정지 |
| `Ctrl+Enter` | 대본 리허설 재생·정지 (`web/mock_script.json`, 서버 없이 돈다) |
| `C` | 설정 패널 |
| `R` | 화면 비우기. **스트림은 계속 돈다** — 끊는 건 `Ctrl+Space` 다 |
| `1`~`5` | 배치 전환 |
| `Ctrl+L` | 목표별 칸의 가로(`4`)↔세로(`5`) 전환 |
| `T` | 표시 목표 방식 순환 |

`Ctrl` 조합은 위 둘만 잡고 나머지는 브라우저에 넘긴다. 안 그러면 `Ctrl+R` 새로고침과
`Ctrl+C` 복사를 페이지가 가로챈다.

소리는 내지 않는다. 화면에 글만 띄운다.

### 설정 패널 (`C`)

시연 도중에 바꿔야 하는 값 둘을 셀렉트 박스로 뺐다. 고르는 즉시 적용된다.

| 항목 | 무엇이 바뀌나 |
|---|---|
| **표시 문장 개수** | 화면에 남기는 줄 수(`자동`, 1~8). 화면만 다시 그린다. `자동` 은 배치마다 다르다 — 태그 배치 4줄(`anchor=center` 면 3줄), 칸 배치는 칸 높이에 몇 줄이 들어가는지 재서 정한다 |
| **사용할 소스 언어** | 이번 시연에서 **말할** 언어를 칩으로 켜고 끈다(기본 ko·en·ja). 켠 것만 아래 목표 목록에 나오고, 그대로 ASR 허용 언어가 된다 |
| **소스 언어별 번역 목표** | 켠 언어마다 어디로 번역할지. 서버로 `config` 메시지를 보낸다 — 스트림을 끊지 않으므로 **다음 확정 문장부터** 바뀐 방향으로 나온다 |
| **표시 목표** | 목표를 전부 띄울지, 하나만 띄울지(`전부` / `하나 고정` / `자동 순환`). `T` 키와 같다 |
| **태그 글자** | 태그에 띄울 글자를 언어마다 직접 적는다(24자까지). `발표자 A` 처럼 화자를 가리키는 말로 바꿔 쓰는 자리다. 비우면 다시 언어 코드가 뜬다. 화면에만 쓰므로 서버로 나가지 않고, 이미 떠 있는 줄까지 그 자리에서 바뀐다 |

소스 언어를 껐다 켜도 그 언어에 고르던 목표는 그대로 돌아온다. 하나도 안 남으면 서버가
쌍 규칙으로 되돌아가므로 마지막 하나는 꺼지지 않는다.

**소스 언어를 넓힐수록 ASR 오감지가 는다.** 목표 언어는 출력일 뿐이라 허용 목록에 넣지
않는다 — `ja→ko` 만 켜면 ASR 은 일본어만 듣는다.

고른 값은 localStorage 에 남고, 쿼리 스트링(`?langs=ko,en&depth=6&map=ko:en&tags=ko:발표자 A`)이
있으면 그쪽이 이긴다.

줄 수를 늘릴 때 주의: CSS 의 `.d0`~`.d7` 만큼(8줄)까지만 보인다. `anchor=center` 는 최신 줄이
화면 한가운데 오므로 위로 남길 자리가 좁아 4줄만 넘어가도 위가 잘린다.

### 번역 방향은 언어쌍이다

서버는 임의 다국어가 아니라 **"내 언어(`lang`) ↔ 상대 언어(`targetLang`)"** 쌍으로
방향을 정한다(`_correct_and_translate`).

| 상황 | 번역 목표 |
|---|---|
| `lang=auto` | 무조건 `targetLang` |
| 감지 언어 == `lang` | `targetLang` |
| 감지 언어가 그 밖(제3언어 포함) | **`lang`** |

그래서 `lang=auto` 로 두면 안 된다. `targetLang=ko` 인데 한국어를 말하면 번역이
원문 그대로 나온다. 기본값을 `src=ko`, `tgt=en` 으로 둔 이유다.

**단 `langMap` 이 있으면 이 쌍 규칙보다 그쪽이 먼저다.** 쌍으로는 제3언어가 전부 `lang` 으로
되돌아가므로 셋 이상을 각각 다른 곳에 보낼 수 없다. 설정 패널의 소스 언어 칩과 목표 목록이
이 매핑을 만들고, `start` 와 `config` 에 실려 나간다(`{"ko":"en","ja":"ko",...}`). 서버는
`parse_lang_map` 으로 모르는 코드와 자기 자신으로 가는 항목을 버린다.

매핑이 있으면 ASR 의 `allowed_languages` 는 **매핑의 키(소스 언어)만** 으로 좁혀진다.
다만 그 목록은 스트림 슬롯을 만들 때 계산하므로, 흐르는 중에 바꾸면 **번역 방향은 바로**,
ASR 허용 언어는 다음 커밋부터 반영된다.

| 발화 | 번역 |
|---|---|
| 한국어 | 영어 |
| 영어 | 한국어 |
| 일본어·그 밖 | 한국어 |

ASR 자체의 언어 제한은 별개다. `--no-restrict-languages` 로 띄우면 `lang` 을 구체
언어로 줘도 인식은 제한되지 않는다(`_new_stream_slot` 의 `restrict_languages` 분기).
`lang` 은 번역 방향 결정에만 쓰인다.

번역이 원문과 같으면 아래 줄에 겹쳐 쓰지 않는다.

동작 값은 쿼리 스트링으로 덮어쓴다 — 시연장에서 값만 바꿔 다시 열면 된다.

| 파라미터 | 기본 | 뜻 |
|---|---|---|
| `layout` | `tag` | `tag` / `lane` |
| `fade` | (없음) | `1` 이면 앞선 문장을 깊이만큼 흐리게 하고 `hold` 를 2600ms 로 켠다 |
| `hold` | `0` (`fade=1` 이면 `2600`) | 확정 문장이 머무는 최소 시간(ms) |
| `holdmin` | `900` | 밀렸을 때 줄일 수 있는 하한(ms). `hold=0` 이면 적용하지 않는다 |
| `type` | `1300` | 한 덩어리를 다 타이핑하는 목표 시간(ms) |
| `order` | (없음) | `strict` 면 전사를 완주시킨 뒤 번역으로 넘어간다 |
| `trans` | `500` | `order=strict` 에서 전사를 다 친 뒤 두는 시간(ms) |
| `src` / `tgt` | `ko` / `en` | 내 언어 / 상대 언어. 위 표대로 방향이 갈린다 |
| `langs` | `ko,en,ja` | 사용할 소스 언어. 없이 `map` 만 주면 그 키가 소스 언어가 된다 |
| `map` | `ko:en,en:ko,ja:ko,zh:ko,es:ko` | 소스 언어별 번역 목표. 쌍 규칙보다 먼저다 |
| `depth` | (배치 기본) | 화면에 남기는 줄 수(1~8). `0` 이면 자동 |
| `anchor` | (없음) | `center` 면 최신 문장이 화면 세로 한가운데 뜬다 |
| `layout` | `laneT` | 배치. `tag` `lane` `stack` `laneT` `laneV` |
| `aud` | (없음) | 청중 언어. 목표 칸을 첫 문장 전에 다 잡아 둔다 |
| `show` | `all` | 표시 목표 방식. `all` `one` `rotate` |
| `pick` | (없음) | 그 목표 하나만 띄운다(`show=one` 과 같다) |
| `rotate` | `4000` | 자동 순환 간격(ms) |
| `auto` | (없음) | 열자마자 그 입력으로 시작한다. `ko` `en` `ja` `mix` `mic` |

### ASR 없이 화면만 보기

GPU 없이 배치와 페이싱만 볼 때 쓴다. 대본은 `web/mock_script.json` 에 있다. 실제 발화
속도로 흐르고, 뒷부분 세 문장은 0.2~0.3초 간격으로 몰아쳐 유지 시간과 대기열이 도는지
보여 준다.

**서버 없이 보려면 어느 포트에서 열든 `Ctrl+Enter`.** show.html 이 같은 대본을 직접 읽어
서버가 보내 온 것처럼 화면에 흘린다. 프록시(8080)에 붙은 채로도 되고, 마이크·WebSocket 을
쓰지 않는다. 아래 목 서버는 WebSocket 을 타는 경로까지 같이 볼 때 쓴다.

```bash
tmux new-session -d -s showmock -c . \
  "python -u STiTy-Mobile/demo-web/partial_demo/mock_show_server.py 8090 --loop"
```

<http://localhost:8090/show.html> 를 열고 `Ctrl+Space` 를 누르면 시작한다(목 서버는 무음
wav 를 내주므로 테스트 음성 파일이 없어도 된다). `?auto=ko` 를 붙이면 누르지 않아도 열자마자
시작한다.

**목 서버의 `final` 은 목표별 번역을 다 실어 보낸다.** 대본은 ko·en·ja 세 언어로 말하고
청중도 그 셋이라, 발화마다 나머지 두 언어 번역이 함께 나간다(`translations`). 서버는 아직
목표 하나만 보내므로, 목표가 여럿인 배치(`3`·`4`)와 표시 목표 방식은 여기서만 실제 모양으로
볼 수 있다. 차례로 보려면:

```
http://localhost:8090/show.html?auto=ko&layout=stack      # 3 · 목표 쌓기
http://localhost:8090/show.html?auto=ko&layout=laneT      # 4 · 목표별 칸
http://localhost:8090/show.html?auto=ko&layout=laneV      # 5 · 목표별 세로 칸
http://localhost:8090/show.html?auto=ko&pick=ja           # 하나 고정
http://localhost:8090/show.html?auto=ko&show=rotate       # 자동 순환
```

**무음 wav 길이는 대본에서 계산한다.** 페이지는 wav 를 다 보내면 `finish` 를 보내고 8초 뒤
연결을 닫으므로, 무음 wav 가 대본보다 짧으면 뒷부분이 통째로 잘린다(예전 44바이트짜리로는
8초까지만 보였다). `--loop` 로 두 바퀴 이상 보려면 페이지를 다시 연다.

서버를 목표 여럿으로 바꾸려면 `final` 에 `translations`(목표별 번역)를 더하고,
`langMap` 이 목표를 여럿 받게 넓혀야 한다 — 지금은 `parse_lang_map` 이 소스 하나에
목표 하나로 정규화한다. 번역 호출은 목표마다 한 번씩이므로 `asyncio.gather` 로 묶어야
지연이 합이 아니라 최댓값이 된다.

## 테스트 음성 다시 만들기

LibriSpeech 오디오는 저장소에 없다. 데이터가 있는 경로를 넘긴다.

```bash
python STiTy-Mobile/demo-web/partial_demo/make_test_wav.py STiTy-Mobile/demo-web/partial_demo/web/partial_test.wav 3 \
  /path/to/evaluation/LibriSpeech/LibriSpeech/test-other
```

발화 사이에 1.2초 침묵을 넣는다. VAD 커밋과 슬롯 리셋을 실제로 태워서, 그 지점에
빈 `partial`(화면 비우기 신호)이 나가는지 보기 위해서다.

시연용(`show.html`)은 언어별 파일을 따로 쓴다. 경로가 스크립트 안에 박혀 있어 인자 없이 돈다.

```bash
python STiTy-Mobile/demo-web/partial_demo/make_demo_wavs.py                 # en, ja, ko 모두
python STiTy-Mobile/demo-web/partial_demo/make_demo_wavs.py --lang ja --n 4
```

`web/partial_test_<lang>.wav` 와, 무엇이 나와야 하는지 대조할 `web/partial_test_<lang>.txt`
가 함께 나온다. 출처는 언어마다 다르다.

| 언어 | 출처 | 비고 |
|---|---|---|
| en, ja | FLEURS dev (`$FLEURS_ROOT/{en_us,ja_jp}`, 기본 `~/datasets/fleurs/data`) | 낭독체. 구두점 있는 원문 전사가 붙는다 |
| ko | KsponSpeech `sample_data/eval_clean` | FLEURS `ko_kr` 에는 오디오가 없다. 대화체라 나머지 둘과 결이 다르다 |
| mix | 위 셋을 ko/en/ja 순으로 번갈아 | 언어별 칸 배치 확인용 |

FLEURS TSV 는 본문에 따옴표가 그대로 들어 있어 `csv.QUOTE_NONE` 으로 읽는다.
기본 파서로 읽으면 여러 행이 한 필드로 붙어 수천 어절짜리 잔해가 섞인다.

## CLI 로 검증

브라우저 없이 프로토콜만 확인한다. `final` 직후 빈 `partial` 이 오는지, 마지막 메시지가
빈 `partial` 인지, `seq` 가 단조 증가하는지, 메타 헤더(`language ...`)가 새지 않는지 판정한다.

```bash
python STiTy-Mobile/demo-web/partial_demo/partial_ws_client.py \
  --url ws://127.0.0.1:8766 --wav STiTy-Mobile/demo-web/partial_demo/web/partial_test.wav

# 무음만 넣었을 때 partial/final 이 안 나오는지
python STiTy-Mobile/demo-web/partial_demo/partial_ws_client.py \
  --url ws://127.0.0.1:8766 --wav silence.wav --expect-silence
```

## 번역

`run_server.sh` 는 `--local-translation` 으로 띄운다. `core/translator/local_translator.py` 의
NLLB-200-distilled-600M 을 로컬 GPU 에 올려 번역하므로 외부 호출이 없고, API 키 없이
쓰던 무료 gtx 엔드포인트의 429 를 안 만난다.

번역기는 `--local-translation-model` 로 고른다. 이름에 `madlad` 가 들어가면 MADLAD,
아니면 NLLB 로 취급한다(`core/translator/local_translator.py` 의 `make_translator`).
`run_server.sh` 는 `TRANSLATION_MODEL` 환경변수로 받는다.

RTX 4090 실측 (num_beams=4, 시연 문장 9개):

| 모델 | 지연 중앙값 | GPU | 비고 |
|---|---|---|---|
| `facebook/nllb-200-distilled-600M` | 54ms | 1.2GB | doorbell 을 놓치고 "문벨" 로 옮긴다 |
| `facebook/nllb-200-distilled-1.3B` | 91ms | 2.6GB | doorbell 은 살리나 여전히 "문벨" |
| **`google/madlad400-3b-mt`** (기본) | 195ms | ~6GB | "초인종". 문장이 가장 자연스럽다 |

MADLAD 를 쓰려면 ASR 쪽 `--gpu-memory-utilization` 을 낮춰 자리를 비워야 한다.
`0.42` 면 ASR 이 12GB 를 쓰고(KV 캐시 4.88GB, 45,696 토큰) 12.5GB 가 남는다.
시연은 동시 접속이 한 명이라 KV 캐시가 줄어도 문제되지 않는다.

MADLAD 는 소스 언어를 지정하지 않고 타깃만 `<2ko>` 처럼 앞에 붙인다. 그래서 ASR 이
언어를 잘못 감지해도 NLLB 만큼 크게 망가지지 않는다.

`num_beams` 는 4 다. greedy(1) 로 두면 짧은 문장에서 EOS 를 일찍 내고 오역한다.

```
ヘブライ人一家はほとんど都会で暮らしていました。
  beams=1 -> 이 두 사람은                              (끊기고 오역)
  beams=4 -> 히브리 가족들은 대부분 도시에 살고 있었습니다.
```

문장당 45ms → 60ms 로 15ms 더 든다. 파이프라인 전체 지연에서 무시할 수준이다.
`min_new_tokens` 로 길이를 강제하는 건 해법이 아니다 — 헛소리를 이어 붙인다
(`이 두 사람은 모두 이 도시에서 살고 있었습니다.`).

`no_repeat_ngram_size` 는 3 이다. 짧고 반복적인 입력에서 반복 루프가 터진다.

```
아니, 아니.
  없을 때 -> No, no, no, no, no, ... (상한까지)
  3 을 주면 -> No, no, not at all.
```

정당한 반복은 살아남는다 (`네 네 네.` → `Yeah, yeah, yeah.`). 출력 길이도 입력에
비례해 묶어(`max(24, 입력토큰*3)`) 반복이 새어 나가도 멀리 못 간다.
