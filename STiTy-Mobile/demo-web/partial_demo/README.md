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
- 전사(`partial`)는 원문이라 작은 글씨로 흐른다. `final` 이 오면 **기다리지 않고 바로**
  번역을 큰 글씨로 올린다(지연 우선). 전사가 사라지는 것은 아니고 번역 아래 작은 줄로
  남는다. `?order=strict` 를 주면 전사를 끝까지 친 뒤 `trans`(기본 500ms)만큼 두고
  번역으로 넘어간다 — 순서는 지켜지지만 그만큼 번역이 늦게 뜬다.
- 발화는 잡혔는데 아직 디코딩 결과가 없으면 `•••` 이 다음 문장이 뜰 자리 아래에 나온다.

다국어 배치 두 가지를 `1` / `2` 키(또는 우상단 버튼)로 바꾼다. 선택은 localStorage 에 남는다.

| 키 | 배치 |
|---|---|
| `1` | 한 칸에 모아 쓰고 문장 앞에 언어 태그를 붙인다. 태그 색이 언어마다 다르다 |
| `2` | 언어별로 칸을 나눠 세로로 배치한다. 칸은 언어가 처음 나온 순서로 생긴다 |

`R` 은 화면 초기화, `C`(또는 `⚙ 설정`)는 설정 패널. `EN`/`JA`/`KO`/`MIX` 버튼은 해당 테스트 음성을 흘린다.
`MIX` 는 ko/en/ja 를 번갈아 붙인 것이라 **언어별 칸 배치를 확인할 때 쓴다** — 한 언어만
들어오면 칸이 하나뿐이라 배치 2 가 배치 1 과 구분되지 않는다.
음성 언어와 무관하게 세션 언어쌍(`src`/`tgt`)은 그대로 쓴다 — 아래 참고.

`🔊 소리`(또는 `S` 키)를 켜면 테스트 음성을 스피커로도 재생한다. 소리와 자막을 같이 보면
자막이 얼마나 빨리 뜨는지 체감할 수 있다. 전송과 같은 시점에 재생하므로 대략 맞물린다.
브라우저가 자동 재생을 막으면 화면을 한 번 클릭하고 다시 누르면 된다. 설정은
localStorage 에 남는다. 마이크 경로에서는 쓰지 않는다(에코).

### 설정 패널 (`⚙ 설정` / `C`)

시연 도중에 바꿔야 하는 값 둘을 셀렉트 박스로 뺐다. 고르는 즉시 적용된다.

| 항목 | 무엇이 바뀌나 |
|---|---|
| **표시 문장 개수** | 화면에 남기는 줄 수(`자동`, 1~8). 화면만 다시 그린다. `자동` 은 배치 기본값 — 태그 배치 4줄(`anchor=center` 면 3줄), 칸 배치 3줄 |
| **사용할 소스 언어** | 이번 시연에서 **말할** 언어를 칩으로 켜고 끈다(기본 ko·en·ja). 켠 것만 아래 목표 목록에 나오고, 그대로 ASR 허용 언어가 된다 |
| **소스 언어별 번역 목표** | 켠 언어마다 어디로 번역할지. 서버로 `config` 메시지를 보낸다 — 스트림을 끊지 않으므로 **다음 확정 문장부터** 바뀐 방향으로 나온다 |

소스 언어를 껐다 켜도 그 언어에 고르던 목표는 그대로 돌아온다. 하나도 안 남으면 서버가
쌍 규칙으로 되돌아가므로 마지막 하나는 꺼지지 않는다.

**소스 언어를 넓힐수록 ASR 오감지가 는다.** 목표 언어는 출력일 뿐이라 허용 목록에 넣지
않는다 — `ja→ko` 만 켜면 ASR 은 일본어만 듣는다.

고른 값은 localStorage 에 남고, 쿼리 스트링(`?langs=ko,en&depth=6&map=ko:en`)이 있으면 그쪽이 이긴다.

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

### ASR 없이 화면만 보기

GPU 없이 배치와 페이싱만 볼 때 쓴다. 대본을 실제 발화 속도로 흘리고, 뒷부분 세 문장은
0.2~0.3초 간격으로 몰아쳐 유지 시간과 대기열이 도는지 보여 준다.

```bash
tmux new-session -d -s showmock -c . \
  "python -u STiTy-Mobile/demo-web/partial_demo/mock_show_server.py 8090 --loop"
```

<http://localhost:8090/show.html> 를 열고 `EN`/`JA`/`KO` 중 아무거나 누르면 시작한다
(목 서버는 무음 wav 를 내주므로 테스트 음성 파일이 없어도 된다).

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
