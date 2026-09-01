# experiment — autoseg 프롬프트 루프 실험 모음

`en-multi/run13` 계열(영어 소스)과 `{de,ja,zh}-en/run02`(비영어 소스 → 영어)을 한자리에
모은 **인덱스**다. 두 묶음이 같은 설정으로 돌아 나란히 읽을 수 있다.

## 이 폴더는 심볼릭 링크다

실제 산출물은 전부 `core/meaning_segmentator/runs/` 아래 그대로 있다. 복사도 이동도
아니다. 물리적으로 옮기지 않은 이유가 셋이다:

- `../autoseg/loop.py:1230` 이 `runs/<pair_id>/<run_id>` 를 하드코딩한다. 옮기면 새 런이
  다시 `runs/` 에 떨어져 구조가 반쪽이 된다 — 코드에 `--runs-root` 를 먼저 내야 한다.
- `tools/covost2_label/*.sh` 7개가 `runs/en-multi/run13/best_prompt.txt` 를 가리키고,
  `MULTI2EN_DATASET.md` 가 `runs/...` 경로로 런을 인용한다.
- `{de,ja,zh}-en/run02` 는 **지금 돌고 있다.** 쓰는 중인 디렉토리는 못 옮긴다.

그래서 링크만 건다. 링크는 `../runs/...` 상대경로라 레포를 통째로 옮겨도 안 깨진다.

## 공통 설정

`en-multi/run13` 이 기준이고 `run02` 세 트랙이 이를 그대로 복제한다.

```
--model gpt-5-mini  --agent-reasoning-effort none  --seg-reasoning-effort none
--iterations 5  --train 40  --dev 265  --test 100   (= 405, 매니페스트 전량)
--patience 5  --budget 25  --workers 24
--translate-backend local  (google/madlad400-3b-mt)
--adequacy-backend cometkiwi  --consistency-backend nli  --adopt-se-mult 0.5
```

`--min-gap` / `--t-grid` / `--t-floor` 는 **인자로 안 준다.** `MIN_GAP_MS(1200) × 코퍼스
발화속도`로 언어마다 유도되고, 그 속도는 강제정렬 산출물
(`evaluation/ast/manifests/*_unittimes.json`)에서 실측으로 온다. run13 의 config 에 찍힌
`min_gap: 3` 도 인자가 아니라 유도 결과다 (`loop.py` 가 `args.min_gap` 에 되쓴다).

`--tgt-lang` 은 검증 타깃이 아니다. 검증 타깃은 기본 풀
(English, Korean, Japanese, Chinese, Spanish, German)에서 **소스 언어만 뺀** 5개이고,
목적함수는 타깃별 z-정규화 effective 의 평균이다. `--target-aware` 런만 타깃이 1개다.

## 런 목록

| 링크 | 방향 | 데이터셋 | 타깃 | min_gap / t-grid | 상태 | 비용 |
|---|---|---|---|---|---|---|
| [en-multi-run13](en-multi-run13) | en → 5개 | `fleurs-en-multi` | 다중 (대표 Korean) | 3 / [4,6,12] | 완료 | $10.72 |
| [en-multi-run13ta-de](en-multi-run13ta-de) | en → de | `fleurs-en-multi` | German (`--target-aware`) | 3 / [4,6,12] | 완료 | $11.87 |
| [en-multi-run13ta-ja](en-multi-run13ta-ja) | en → ja | `fleurs-en-multi-ja` | Japanese (`--target-aware`) | 3 / [4,6,12] | 완료 | $18.07 |
| [en-multi-run13ta-zh](en-multi-run13ta-zh) | en → zh | `fleurs-en-multi` ⚠ | Chinese (`--target-aware`) | 3 / [4,6,12] | 완료 | $15.28 |
| [de-en-run](de-en-run) | de → 5개 | `fleurs-de-en` | 다중 (대표 English) | 3 / [4,6,12] | 완료 — iter 0~5, **iter_05 채택** | $16.86 |
| [ja-en-run](ja-en-run) | ja → 5개 | `fleurs-ja-en` | 다중 (대표 English) | 7 / [9,14,27] | **예산 초과 중단** — iter 0~1 완료, iter_02 도중 사망 | $25.01 |
| [zh-en-run](zh-en-run) | zh → 5개 | `fleurs-zh-en` | 다중 (대표 English) | 6 / [8,12,24] | 실행 중 (iter_03) — 같은 벽 예상 | 예산 $25 |

run13 계열 4런 합계 **$55.94**. run02 세 트랙은 순차로 돌고 트랙당 상한이 $25 다. 링크 이름에는 런 번호를 안 붙였다 (`de-en-run` → `../runs/de-en/run02`) — 트랙당
최신 런 하나만 가리키는 자리이고, 실제 번호는 링크가 가리키는 경로와 `config.json` 에 있다.
`logs/` 쪽 이름은 `runs/` 의 실제 파일명 그대로 둔다.

`ja`/`zh` 의 min_gap 이 큰 것은 언어 특성이 아니라 **단위가 다르기 때문**이다. 띄어쓰기가
없는 언어는 어절이 아니라 문자를 세므로 초당 단위 수가 크다 (de 2.40 / ja 5.74 / zh 4.73).

## ⚠ x2en 세 트랙은 이터레이션 수가 다르다 — 나란히 읽을 때 반드시 적을 것

de-en 은 6이터를 다 돌았지만 ja-en 은 2, zh-en 은 4에서 예산 상한에 걸려 멈췄다.
**프롬프트 품질 차이가 아니라 예산이 끊은 지점의 차이다.** 개정 기회를 덜 받은 트랙이
불리하게 보이는 것은 당연하므로, 트랙 간 절대 비교는 하지 말 것.

원인은 `segment_retry` 다. 트랙별 1차 포맷 통과율과 재시도 비중:

| 트랙 | 1차 통과율 | 재시도 비중 | 이터당 단가 |
|---|---|---|---|
| de-en | 0.90 → 0.81 | $5.83 / $14.25 (41%) | ~$3.5 |
| ja-en | 0.50 → **0.07** | $10.65 (68%) | ~$9.8 |
| zh-en | 0.28 | $10.38 (59%) | ~$6.5 |

ja/zh 는 띄어쓰기가 없어 **어절이 아니라 글자를 센다.** 그래서 요구 경계 수가 많고
지배적 위반이 `too_few_tags` 다 (ja iter_02: 41건 중 33건). 재시도 1콜이 사고 토큰
10~11k 를 태운다.

여기에 겹친 것이 **목적함수가 재시도 비용을 못 본다**는 점이다. ja-en iter_01 이 채택한
개정본은 1차 통과율을 0.50 → 0.07 로 무너뜨렸는데 dev Δ 가 +0.036 이라 채택됐고, 그
채택이 이터 단가를 $5.86 → $9.77 로 올려 iter_02 에서 벽에 부딪혔다. 루프는 점수를 돈으로
사고 있다는 사실을 모른다.

`--seg-reasoning-effort none` 은 "사고 끔"이 아니라 **파라미터를 빼라(= 모델 기본값)** 이고
(`gateway.py:39`), gpt-5-mini 기본값은 사고 켜짐이다 (콜당 10~17k). run13 도 같은 설정이라
비교 가능성은 깨지지 않았지만, 비용을 줄이려면 이게 첫 손잡이다 — 대신 run13 과 나란히
못 읽게 된다.

### 멈춘 트랙을 이어 돌리는 법

예산 가드는 프로세스마다 0 에서 시작하므로 `--budget` 을 새로 주면 그만큼 더 돈다.
`tools/autoseg_x2en/run_chain.sh` 를 다시 돌리면 `history.json` 을 보고 알아서
`--resume` 을 붙인다. ja-en 은 `iter_02/prompt.txt` 가 남아 있어 개정본을 안 잃었고,
분절 캐시에 죽은 iter_02 가 이미 낸 호출이 들어 있어 재개분이 그만큼 싸다.

`tools/autoseg_x2en/resume_rest.sh` 는 체인이 끝나기를 기다렸다가 멈춘 트랙을
`--resume` 으로 끝까지 돌린다 (ja $45 / zh $30 상한 — 폭주 가드이지 목표액이 아니다).
`--iterations 5` 는 그대로라 세 트랙이 "iter_00 + 실제 개정 4회" 로 맞춰진다.
이터를 이미 다 채운 트랙이면 `loop.py` 가 `start_it >= args.iterations` 로 최종 평가만
돌므로, 이 스크립트 하나가 `--final-only` 까지 겸한다.

최종 곡선만 급하면 `--final-only` 를 직접 줘도 된다. **그 경로는 `history.json` /
`best_prompt.txt` / `config.json` 을 안 건드리므로** (`loop.py:1358`, `1727`) 곡선을 먼저
뽑아 두고 나중에 `--resume` 으로 이터를 더 돌아도 충돌하지 않는다.

## ⚠ run13ta-zh 는 나머지 둘과 같은 자가 아니다

`--target-aware` 는 타깃 프로파일을 **매니페스트의 정답 번역**에서 측정으로 만든다.
`run13ta-zh` 는 2026-08-31 00:32 에 시작했는데 en-zh 매니페스트는 같은 날 13:35 에
만들어졌다. 그래서 en-de 매니페스트로 돌았고 로그에 이렇게 남아 있다:

```
[target-aware] 매니페스트에 Chinese(zh) 정답 번역이 없다 (tgt_lang=de)
               — 타깃 프로파일은 모델 사전지식으로 간다
```

`run13ta-ja` 는 en-ja 매니페스트를 제대로 받았다 (`타깃 표본 20쌍 ... 매니페스트 tgt_lang=ja`).
채점은 참조 없는 QE 라 점수 자체는 오염되지 않지만, **prompt_v0 가 다른 근거에서 나왔으므로
de/ja 와 나란히 놓고 "타깃별 차이"라고 읽으면 안 된다.** zh 를 같은 자로 맞추려면
`--dataset fleurs-en-multi-zh` 로 재실행해야 한다.

## 상태 보기

```bash
# 진행 중인 런
tail -f core/meaning_segmentator/experiment/logs/de-en_run02.log
cat core/meaning_segmentator/experiment/logs/*_run02.launch.log        # 시작 시각 / 종료 코드

# 끝난 런
cat core/meaning_segmentator/experiment/en-multi-run13/final_report.md
PYTHONPATH=. python -m core.meaning_segmentator.autoseg.cost_report --run-id en-multi/run13
```

## 런을 추가할 때

```bash
ln -sfn ../runs/<pair>/<run> core/meaning_segmentator/experiment/<pair>-<run>
```

위 표에 한 줄 추가할 것. 이 README 가 인덱스의 전부다.
