# evaluation/DailyTalk/

한국어 대화 음성 벤치마크. DailyTalk 데이터셋 사용. 메트릭: CER(문자 오류율).

## 데이터셋 분할

`transcribe/` 디렉토리의 JSON 파일들이 평가 드라이버 역할:
- `test(1008).json` — 전체 테스트 셋
- `toy(200).json` — 빠른 확인용 서브셋

## 빠른 시작

```bash
python evaluation/DailyTalk/test_qwen3_dailytalk.py \
  --data-dir evaluation/DailyTalk/transcribe/test\(1008\).json \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## CER 사용 이유

한국어는 단어 분절 없이 WER을 계산하기 어려워 문자 단위 CER을 사용.

## 결과 위치

`evaluation/DailyTalk/results/{model}/{scope}/{tag}/`
