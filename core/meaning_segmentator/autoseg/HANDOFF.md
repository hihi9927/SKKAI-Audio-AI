# autoseg 핸드오프 — 2026-08-10 (2차 갱신)

## 현재 상태

- 지표 체계 v2.1 구현·검증 완료, 커밋 `80a27af` (`feat/autoseg-prompt-loop`) 푸시됨
- **GPU 문제는 해소됐다** — 4090 이 완전히 비어 있고, 실측 사용량은 3.7GB (CometKiwi 2.1 + NLI 1.6)
- **run04 는 다시 중단됐다. 이번엔 GPU 가 아니라 Letsur 게이트웨이 계정의 COST 한도 초과다.**
- run01~03 의 effective/contradiction/consistency 수치는 구 집계라 **새 런과 비교 불가**

## 지금 막고 있는 것 — 게이트웨이 계정 한도 (사람 조치 필요)

```
HTTP 429 {"type":"usage_limit_exceeded","detail":"COST limit exceeded"}
```

우리 `--budget 60` 이 아니라 **계정 자체의 한도**다. 토큰 1개짜리 호출도 같은 429 로 죽는다.
22:33 에는 정상 응답했고(스모크 $0.000096) 22:43 에 한도에 닿았으니, 시작 시점에 이미
잔액이 거의 없었다. **한도를 올리거나 충전하기 전에는 autoseg 의 어떤 것도 진행 불가**
(분절·프로파일러·판정자·비평가가 전부 게이트웨이를 탄다).

번역기(`--translator google`)는 게이트웨이와 무관하고 정상이다 — gtx 엔드포인트 직접 확인.

## 환경은 이제 준비돼 있다 (이 머신)

이 머신의 어떤 파이썬에도 `comet` 이 없었다. 기존 `.venv` 에 넣으면 numpy 2.2.6→1.26.4,
protobuf 7.35.1→4.25.9 로 내려가 **vllm 0.14.0 ASR 평가 스택이 깨진다**(pip dry-run 확인).
그래서 **별도 venv 를 팠다**:

```bash
/home/mobility/STiTy/.venv-autoseg    # python 3.10.12, torch 2.13.0+cu130, unbabel-comet 2.2.7
```

- 실행 시 `PYTHONPATH` 를 비울 것 (ROS humble 간섭 — `[[stity-venv-and-ros-pythonpath]]` 참조).
  루프 자체는 `PYTHONPATH=.` 가 필요하므로 리포지토리 루트에서 돌리면 된다.
- `.env` 의 `HF_HUB_ENABLE_HF_TRANSFER=1` 때문에 `hf_transfer` 도 설치했다 (없으면 다운로드가 죽는다).
- HF 인증 확인됨 — 게이트 모델 `Unbabel/wmt22-cometkiwi-da` 접근 가능 (계정 `Doo12`).
- 백엔드 스모크: CometKiwi 0.862(정상 번역), NLI contra 조기방출 1.0 vs 안전 0.001.
- **NLI 관문 재확인 통과** (`judge_check.py --skip-judge`, 순위 위반 0/6) — torch/transformers
  빌드가 바뀐 새 환경에서도 기록된 결과를 재현한다. 게이트웨이가 없어도 도는 유일한 관문이다.

재개 명령 (게이트웨이 복구 후):

```bash
cd /home/mobility/STiTy && set -a && . ./.env && set +a && \
PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.loop \
    --dataset kspon-train --src-lang Korean --tgt-lang English \
    --pair-id ko-en --run-id run04 --translator google \
    --iterations 6 --train 60 --train-pool 120 --dev 150 --test 150 --budget 60
```

`runs/ko-en/run04/` 는 iter_00 까지 만들어져 있다. 이어서 돌리면 언어 프로파일·prompt_v0·
캐시를 재사용한다. 처음부터 다시 하려면 `--fresh`.

## run04 iter0 에서 실제로 나온 것 — 읽고 시작할 것

```
[iter 0] train fmt=0.90(1st 0.68)  score=0.0000
```

**score 0 은 버그가 아니다.** 60행 중 6행이 `text_modified`(채점 차단 규칙)에 걸려
채점 가능 비율이 0.90 → `skip_translation_below=0.95` 게이트에 막혀 **번역을 통째로
건너뛰었다**(`loop.py:221`). 그래서 `full_trans` 가 전부 None, `by_T` 가 빈 dict 다.
설계대로 동작한 것이다.

문제는 **이 상태가 구조적으로 반복된다**는 것이다:

- 실패한 6행은 **전부 빈 문자열**(`seg_len=0`)이 돌아왔다.
- 길이 편중이 뚜렷하다 — 실패 행 중앙값 30어절 vs 전체 12어절.
  글자수로는 실패 `[66, 86, 87, 125, 140, 193]`, 성공은 **최대 86**.
  즉 **87자 초과 문장은 3개 중 3개 전부 실패**했고, 그 아래는 확률적으로 섞인다.
- run04 에서 `kspon`(eval_clean_1000) → `kspon-train` 으로 바꾼 것이 원인 조건이다.
  꼬리가 훨씬 길다: p99 28→54어절, max 36→72어절.

**원인은 아직 확정되지 않았다.** `pipeline.py:54-60` 의 주석은 예산 부족 시 thinking 이
content 를 다 먹어 빈 문자열이 나온다고 기록하고 있고(과거 1024 에서 실측), 지금 값은
8192 다. 다만 그 주석의 실측치(103자 → 사고 918~1464 토큰, 약 9~14 tok/자)를 그대로
외삽하면 193자라도 ~2700 토큰이라 8192 를 못 채운다. 따라서 **단순 길이 비례 절단으로는
설명되지 않는다** — 반복·비유창 발화(`그냥 그냥 그건 그러고…`)에서 thinking 이 루프를
도는 쪽이 더 그럴듯하지만, 확인된 바 없다.

그래서 이번에 **원인을 로그에 남기게 고쳤다** (아래). 다음 런에서 이게 답을 준다.

### 이번에 고친 것 (`gateway.py`)

`chat()` 이 `finish_reason` 을 버리고 `content or ""` 만 반환하고 있었다. 그 결과
**절단으로 빈 출력이 난 건과 모델이 진짜 빈 답을 준 건이 구분되지 않았고**, 하위 검증기가
둘 다 "모델이 텍스트를 고쳐 씀"(`text_modified`)으로 오진했다. 원인이 로그에 한 줄도
남지 않았다.

- `Usage.truncated` — `finish_reason == "length"` 응답 수를 센다 (`snapshot()` 에 포함)
- `chat()` — content 가 비었는데 `finish_reason == "length"` 면 stderr 에 경고

오프라인 검증만 했다 (`_post` 모킹 + `Usage.add` 단위 확인). **게이트웨이 실호출로는
확인 못 했다** — 계정 한도 때문. 순수 진단용 변경이라 동작 경로는 바꾸지 않는다.

`SEG_MAX_TOKENS` 는 **일부러 8192 그대로 뒀다.** 원인이 절단이라는 증거가 아직 없는데
올리면 사고 토큰 과금만 늘고 원인은 여전히 안 보인다. 다음 런의 경고 유무를 보고 정할 것.

### 그다음 갈림길 (게이트웨이 복구 후 첫 판단)

fmt 이 0.95 를 못 넘으면 루프는 **매 이터레이션 score 0 을 받고 아무것도 채택하지 못한다.**
핸드오프의 질문 1번("iter0 이후 채택이 일어나는가")을 물어볼 수조차 없다. 셋 중 하나를 골라야 한다:

1. 경고가 `length` 로 찍히면 → `SEG_MAX_TOKENS` 를 올린다 (가장 깨끗한 해결)
2. 경고가 없으면 → 빈 출력의 진짜 원인을 따로 봐야 한다 (모델 거부? 반복 루프?)
3. 어느 쪽도 아니면 → 데이터 쪽 결정이 필요하다. `kspon-train` 에 길이 상한을 둘지
   (실험 정의가 바뀐다), `skip_translation_below` 를 낮출지 (게이트의 의미가 약해진다).
   **이건 설계 판단이라 사람이 정할 것.**

## run04 에서 볼 것 (원래 목록, 유효)

1. **채택이 iter0 이후에도 일어나는가** (새 임계 + dev 150 검출력)
2. **순위정렬 Spearman 추이** — 오르면 [Priority Rules] 조향이 작동한다는 뜻
3. `reference_suspect_rate` — 높으면 Google 오라클 의심
4. 완료 후 비교군:
   ```bash
   PYTHONPATH=. .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.eval_prompt \
       --prompt core/meaning_segmentator/autoseg/human_prompts/ko_human_current.txt \
       --run-id ko-en/run04 --split test --label human_current --no-priority
   ```
   → 곡선(우리 4점 + 무분절 상한선 + 기계 + 사람) 완성

## 미해결 (우선순위순)

0. **게이트웨이 계정 한도 해제** — 이게 풀리기 전엔 나머지가 전부 대기 (사람)
1. **gold 참조 교차검증** — consistency 가 주축이 된 지금 논문 방어 급소. AIHub 한-영 신청 필요 (사람)
2. `adequacy_cases.json` 문안 사람 확정 (10분)
3. 잡음 바닥 c₀ 를 목적함수에서 차감할지 — run04 Spearman 추이 보고 결정
   (주의: `noise_floor.py` 는 `*_rows.json` 이 필요한데 gitignore 라 이 클론엔 없다.
   run03 재검을 다시 하려면 그 런을 다시 돌려야 한다)
4. use_context=False 재채점 (평가 vs 운영 서버 번역 모드 불일치, 설계 §12.5)
5. ms-LAAL — KsponSpeech 오디오 강제정렬 (설계 §13-3)
6. 비영어 타깃 consistency — NLI 음역 맹점 (설계 §13-8)

## 문서 위치

설계: [../AUTO_PROMPT_LOOP_DESIGN.md](../AUTO_PROMPT_LOOP_DESIGN.md) (v2.1 반영) /
사용법: [README.md](README.md) / 관문 결과: `../runs/validity_nli/`, `../runs/adequacy_validity/`,
`../runs/ko-en/run03/noise_floor_test.json` / run04 로그: `../runs/run04.log`
