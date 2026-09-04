# evaluation/ast/ — AST(음성번역) 평가 트랙

지연은 **LAAL(ms)**, 품질은 **BLEU** 로 잰다. 서버는 `evaluation/streaming_websocket_server_ast.py`
하나를 모든 AST 데이터셋이 공유하고, 데이터셋은 manifest(JSONL)로 갈아 끼운다.

```
evaluation/
├── streaming_websocket_server_ast.py   공용 AST 평가 서버 (FSL 서버 상속)
└── ast/
    ├── metrics_ast.py             LAAL / BLEU 계산 (순수 함수, ms 단위)
    ├── check_metrics_ast.py       metrics 자체 검증 (단독 실행)
    ├── build_manifest_fleurs.py   FLEURS → manifest   ← 주 데이터셋
    ├── build_manifest_mustc.py    MuST-C → manifest   (배포처 소멸, 아래 참고)
    ├── test_ast.py                manifest 기반 공용 클라이언트
    ├── manifests/                 생성된 manifest (git 미추적 — 절대경로를 담는다)
    └── results/{dataset}/{model}/{scope}/{tag}/
```

## 설치

```bash
pip install -r evaluation/ast/requirements.txt
```

## 데이터 — FLEURS

FLoRes-101 문장을 103개 언어로 낭독한 **n-way 병렬** 코퍼스. 문장 id 가 언어 간에 공유되므로
`소스 언어 오디오` + `타깃 언어 전사` 를 붙이면 그대로 음성번역 평가쌍이 된다.
**ko/ja/zh/es 를 모두 커버**하므로 제품이 실제로 서비스하는 방향을 직접 평가할 수 있다.

```bash
# 오디오는 소스 언어만 받으면 된다. 타깃은 참조 텍스트(TSV)만 필요하다.
hf download google/fleurs --repo-type dataset \
  --include "data/en_us/test.tsv" "data/en_us/audio/test.tar.gz" \
            "data/de_de/test.tsv" "data/ko_kr/test.tsv" "data/ja_jp/test.tsv" \
            "data/cmn_hans_cn/test.tsv" "data/es_419/test.tsv" \
  --local-dir ~/datasets/fleurs
```

리포 밖(`~/datasets/fleurs`)에 둔다. en_us test 오디오까지 포함해 약 280MB.
언어 코드: `en_us de_de ko_kr ja_jp cmn_hans_cn(zh) es_419`.

낭독체라 TED 실연설보다 쉽고, 비교 대상 문헌은 IWSLT 계열이 아니라 Whisper/SeamlessM4T
계열이다. 대신 **소스 언어를 바꿀 수 있다** — ko→en 도 `--src ko_kr --tgt en_us` 로 바로 된다
(단 ko 오디오를 따로 받아야 하고 ASR 모델도 ko 가중치를 써야 한다).

### MuST-C 는 왜 안 쓰나

원래 이 트랙은 MuST-C `tst-COMMON` 기준으로 설계했으나 **배포처가 사라졌다**:
`mustc.fbk.eu` 는 공인 DNS 에서 NXDOMAIN, `ict.fbk.eu/must-c` 는 FBK 일반 페이지로 리다이렉트,
FBK MT 그룹 Resources 페이지에 언급 없음, HuggingFace 미러는 전부 text-only(오디오 없음).
IWSLT 2026 동시통역 트랙도 MuST-C 를 더 이상 쓰지 않는다(CoVoST v2 / Europarl-ST / VoxPopuli).

`build_manifest_mustc.py` 는 그대로 두었다 — 사본을 구하면 아래 구조로 풀고 그 스크립트를 쓰면 된다.

```
~/datasets/mustc/en-de/data/tst-COMMON/{wav/, txt/{*.yaml,*.en,*.de}}
```

## 실행

```bash
# 1) manifest 생성 (오디오는 자르지 않고 offset/duration 만 담는다)
python evaluation/ast/build_manifest_fleurs.py \
    --fleurs-root ~/datasets/fleurs --src en_us --tgt de_de \
    --out evaluation/ast/manifests/fleurs_en-de_test.jsonl --verify-audio

# 2) 서버 (터미널 1)
#    카드를 다른 작업과 나눠 쓰면 --gpu-memory-utilization 을 반드시 낮출 것 (아래 참조)
python evaluation/streaming_websocket_server_ast.py \
    --model models/Qwen3-ASR-1.7B-en-silence-c80-merged --no-idle-shutdown \
    --gpu-memory-utilization 0.5

# 3) 클라이언트 (터미널 2)
python evaluation/ast/test_ast.py \
    --manifest evaluation/ast/manifests/fleurs_en-de_test.jsonl \
    --dataset FLEURS --src-lang en --target-lang de \
    --model "en-silence-c80" --scope full --tag run_01

# 4) 서버 종료 — pkill 은 vLLM EngineCore 를 남긴다
bash evaluation/LibriSpeech/paper_result/ASR/scripts/stop_server.sh 8765
```

en→de test 전체는 346발화 / 0.95시간 오디오 → 실시간 페이싱 기준 **약 63분**.
`--limit` 으로 서브셋, `--tag` 재사용으로 중단 지점부터 재개.

### GPU 를 혼자 쓰는 게 아니면 `--gpu-memory-utilization` 을 낮춰라

vLLM 기본값은 **0.8** 이고, 이건 "모델이 필요한 양"이 아니라 **"남는 걸 다 잡아둔다"** 는
뜻이다. 24GB 카드에서 19.2GB 를 선점한다 — 1.7B 모델이 실제로 쓰는 양과 무관하다.

실제 사고 (2026-08-28 00:21): `repro110` ASR 서버(EngineCore pid 149462)가 19.28GiB 를
잡은 상태에서, 같은 카드에서 4시간째 돌던 autoseg 루프(CometKiwi + NLI, 4.1GB)가 16MiB
할당에 실패해 CUDA OOM 으로 죽었다. 남은 여유가 9.5MiB 였다.

```
0.5  →  12GB.  1.7B·max_len 4096·스트리밍(배치 1)에는 충분하고 12GB 를 비워 둔다
0.8  →  19GB.  카드를 혼자 쓸 때만
```

`run_acl6060.sh` / `run_covost2.sh` / `run_three_axes.sh` 는 **기본 0.5** 로 돈다.
`GPU_UTIL=0.8 bash run_acl6060.sh ...` 로 올릴 수 있다. 임시 스크립트를 손으로 쓸 때도
이 인자를 빼먹지 말 것 — 위 사고가 정확히 그렇게 났다.

## 지표

| 지표 | 정의 |
|---|---|
| `laal_ms` | 비계산인지 LAAL. d = `decisionAudioSec`(커밋을 결정한 순간까지 읽은 소스 오디오). 정책만 평가하므로 GPU 가 달라도 재현된다. **주지표.** |
| `laal_ca_ms` | 계산인지 LAAL. d = 클라이언트가 `final` 을 받은 실시간 경과. 실제 체감 지연. |
| `bleu` | sacrebleu corpus BLEU. 발화별 세그먼트 번역을 이어붙인 것 vs 참조. |

```
LAAL = (1/τ) · Σ_{i=1..τ} [ d_i − (i−1) · T / max(|Y_hyp|, |Y_ref|) ]
τ    = min{ i : d_i ≥ T },  없으면 |Y_hyp|
```

AL 과 다른 점은 분모가 `max(|Y_hyp|, |Y_ref|)` 라는 것 하나다. AL 은 분모가 `|Y_hyp|` 라서
짧게 생성할수록 지연이 작게 나오는 구멍이 있다. 세그먼트 단위 커밋이므로 한 세그먼트의
모든 타깃 단어는 같은 d 를 공유한다(chunk-level SimulST 의 표준 처리).

### 점수를 바꾸는 설정 — 반드시 고정하고 기록할 것

`meta.json` 과 `metric.json.summary` 에 자동으로 남는다. 다른 값으로 낸 점수끼리는 비교 불가다.

| 설정 | 기본 | 효과 |
|---|---|---|
| `--laal-unit` | `word` | LAAL 의 \|Y\| 단위. de/en/es 는 word, **zh/ja 는 char** |
| `--laal-cap-source` | 켬 | 비계산인지 지연을 소스 길이 T 로 상한. AL/LAAL 정의상 읽기는 T 를 넘을 수 없고, 뒤에 붙인 묵음은 하네스가 만든 것이라 지연으로 세면 안 된다 |
| `--bleu-tokenize` | 타깃 언어로 결정(de→`13a`) | sacrebleu 토크나이저. ja/ko 는 mecab 미설치 시 `char` 로 폴백하며 signature 에 남는다 |
| `--strip-nonspeech` | 켬 | `(Laughter)` / `(Gelächter)` 같은 이벤트 표기 제거 |

빈 번역(가설 없음)도 BLEU 계산에 **포함**한다. 어려운 발화를 버려서 점수를 올리는 길을 막는다.

### VAD off 로 돌릴 때 — 스트림 종료 처리가 점수를 좌우한다

base 서버는 스트림 종료(`finish`) 시 **두 가지를 하지 않는다.** 둘 다 VAD 커밋 경로에만 있다.

1. `_asr_finish_streaming()` — 마지막 미완성 청크에 남은 오디오의 최종 디코딩.
   **없으면 그 구간은 전사조차 되지 않는다** (커밋이 늦는 게 아니라 텍스트가 없다).
2. `_drain_pending_gpt()` — 백그라운드로 발사된 번역 태스크 완결.
   없으면 `<SEG>` 는 박혔는데 `final` 이 안 나가고 발화가 통째로 사라진다.

`--no-vad` 면 두 경로 모두 안 타므로 손실이 그대로 드러난다. **커밋 정책과 무관한 문제라
always/dot/seg 가 똑같이 겪는다.** `ASTStreamingHandler.finish_streaming` 이 둘 다 보완한다.

CoVoST2 en-de 200발화(단일 클립), `--no-vad --chunk-size 2.0`, 침묵 500ms 실측:

| 축 | 최종 디코딩 | 빈 가설 | BLEU | LAAL | 처리량(16병렬) |
|---|---|---|---|---|---|
| seg | 끔 | 12 | 27.62 | 3282 ms | 12.2배속 |
| seg | **켬** | 7 | **36.71** | 3188 ms | 13.1배속 |
| dot | 끔 | 8 | 27.57 | 3728 ms | 13.2배속 |
| dot | **켬** | 1 | **36.32** | 3640 ms | 12.9배속 |

침묵을 4000ms 로 늘리면 같은 효과를 얻지만(빈 가설 1, BLEU 37.42) **처리량이 8.0배속으로
떨어지고** 짧은 발화의 `laal_ca_ms` 에 침묵 대기가 섞인다. 최종 디코딩 쪽이 낫다 —
지연은 오히려 소폭 줄고(3282→3188ms) 침묵은 500ms 로 충분하다.

A/B 는 `AST_NO_FINISH_DECODE=1` 환경변수로 최종 디코딩을 꺼서 재현한다.

**규칙: VAD off 에서는 `--trailing-silence-ms` 를 청크 크기의 2배 이상**(2초 청크 → 4000ms)으로 준다.
세 축(always/dot/seg)을 비교할 때는 **모두 같은 값**을 써야 공평하다 — always 는 침묵이 필요 없고
dot 은 마침표 토큰이 필요하지만 seg 만 침묵이 필요하기 때문이다.

부작용: 침묵을 실시간으로 흘려보내므로 처리량이 떨어지고(실측 12.3→8.0배속), 짧은 발화의
`laal_ca_ms` 에 침묵 대기가 포함된다. `laal_ms`(NCA)는 소스 길이로 상한이 걸려 영향받지 않는다.

### 발화 밀림(late final) — 원인과 해법

증상: 어떤 발화가 `final` 을 한 건도 못 받고(빈 가설), 그 내용이 **다음 발화**의 결과에
섞여 나온다. seg 축에서 특히 컸다(9,424발화 중 162건).

원인은 base 의 **flush 태스크 핸들 덮어쓰기**다:

```python
self._gpt_flush_task = asyncio.create_task(self._flush_pending_gpt_tasks())  # 청크마다 대입
```

이전 flush 가 아직 돌고 있는데 다음 청크가 같은 변수에 새 핸들을 덮어쓰면 앞의 핸들은
사라진다. `_drain_pending_gpt()` 는 **마지막 핸들만** await 하므로 앞의 flush 를 놓치고,
그게 스트림 종료 뒤에 끝나면 그 final 이 다음 발화로 밀려 나간다.

해법 두 가지를 `ASTStreamingHandler` 에 넣었다:

1. **실행 중 개수 세기** — `_flush_pending_gpt_tasks` 를 감싸 in-flight 카운터를 두고,
   드레인이 카운터가 0 이 될 때까지 기다린다. 핸들이 덮어써져도 전부 기다린다.
2. **발화 단위 못박기(uttId)** — 클라이언트가 `start` 에 `uttId` 를 싣고, 서버는 **커밋이
   결정되는 순간**(번역 진입 시점) 그 값을 찍어 `final` 로 돌려준다. 전송 시점 발화와
   다르면 `[AST-LATE]` 로 경고하고, 클라이언트는 그 final 을 이번 발화 가설에서 제외한다
   (`n_foreign_finals`). 밀림이 남더라도 **오염은 막힌다.**

실측(CoVoST2, seg 축, 침묵 500ms):

| | 빈 가설 | 경계 넘은 final | BLEU |
|---|---|---|---|
| 실패 확정 40발화 · 못만 박음 | 29 | 29 | 0.96 |
| 실패 확정 40발화 · **+ 중첩 인지 드레인** | **3** | **0** | **26.79** |
| 일반 200발화 · 수정 전 | 3~5 | — | 36.5~37.0 |
| 일반 200발화 · **수정 후** | **1** | **0** | **37.85** |

남은 빈 가설은 레이스가 아니라 **거의 무음이거나(−55, −73 dBFS) 1초 남짓한 초단문**이다.

> **프로덕션에도 같은 버그가 있다.** `_on_vad_commit` 의 드레인도 마지막 핸들만 기다리므로,
> 번역이 겹치는 순간 마지막 문장이 늦게 가거나 다음 발화에 붙을 수 있다. 수정은 base 에
> 같은 카운터를 넣는 것이지만 모든 벤치마크에 영향을 주므로 여기서는 평가 서버만 고쳤다.

### 병렬 실행

`--clients N` 으로 WebSocket 연결을 N 개 띄운다(각 연결이 큐에서 발화를 하나씩 가져간다).
오디오 로드와 채점은 별도 스레드로 빼서 다른 워커의 실시간 페이싱을 방해하지 않는다.
16 병렬 실측 8.0배속(침묵 4초 포함). 요약 로그의 `실시간 대비 N배속` 으로 서버 병목을 본다.

번역 호출은 **부분적으로 병렬**이다 — SEG/dot 커밋은 `asyncio.create_task` 로 발사돼 ASR
디코딩과 겹쳐 돌지만, always/vad/finish 커밋은 직접 await 라 그 연결 안에서는 순차다.
`asr_lock` 은 연결마다 하나라 연결 간에는 직렬화되지 않는다(실제 병목은 vLLM 엔진).
16 병렬에서 번역 실패는 0건이었다.

### 검산

요약의 `검산(세그먼트)` 줄을 볼 것. 세그먼트별 `수신시각 − 결정시점` 은 계산 비용,
즉 FSL 과 같아야 한다.

- **집계 LAAL 끼리 빼서 검산하면 안 된다.** LAAL 은 τ 에서 잘리고 타깃 단어 수로 가중되는
  반면 평균 FSL 은 세그먼트 균등 가중이라, 배선이 멀쩡해도 두 값은 다르다.
- SEG 커밋의 `fsl_sec` 은 토큰비율로 역추정한 audio_end 기준이라 결정시점 기준인 이 차이와
  수백 ms~수 초 어긋나는 게 정상이다 (`max_abs_seg_fsl_residual_ms` 가 크게 잡히는 이유).
  평균 잔차(`mean_seg_fsl_residual_ms`)가 수백 ms 안쪽이면 정상으로 본다.

## 해석 주의

BLEU 는 **번역 백엔드(Google/GPT)의 품질**이 상당 부분을 차지한다. 번역 백엔드를 고정한 채
**커밋/분절 정책을 비교**하는 용도로 쓸 것 — 정책이 달라지면 LAAL 과 BLEU 가 함께 움직이므로
두 축을 같이 보고해야 의미가 있다(LAAL–BLEU 곡선).

## 다른 데이터셋 붙이기

`build_manifest_*.py` 를 하나 더 쓰면 된다. 클라이언트는 manifest 만 본다:

```json
{"utt_id": "...", "wav": "/abs/path.wav", "offset": 0.0, "duration": 9.87,
 "src_lang": "en", "tgt_lang": "de", "src_text": "...", "tgt_text": "...",
 "speaker_id": "...", "talk_id": "..."}
```

`offset`/`duration` 은 클라이언트가 로드할 때 자른다 — 오디오를 한 벌 더 만들지 않는다.
