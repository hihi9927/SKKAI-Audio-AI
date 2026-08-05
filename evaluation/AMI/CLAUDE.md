# evaluation/AMI/

다화자 영어 회의 음성 벤치마크. AMI Meeting Corpus 사용. 메트릭: WER.

## 데이터 레이아웃 주의사항

오디오는 `evaluation/AMI/AMI/`에, 단어 수준 XML 전사는 `evaluation/AMI/words/`에 있다.  
`--ami-dir evaluation/AMI`만 넘기면 "No audio files found" 오류 발생. 반드시 두 경로 모두 지정.

`evaluation/AMI/`는 `.gitignore`로 제외되어 있고 스크립트와 `words/` XML만 force-add로 추적된다. 오디오(`AMI/`)는 별도로 내려받아야 한다.

## 빠른 시작

```bash
python evaluation/AMI/test_qwen3_ami.py \
  --ami-dir evaluation/AMI/AMI \
  --words-dir evaluation/AMI/words \
  --model "baseline(1.0.0)" --scope sample --tag run_01
```

## 서버

공유 평가 서버: `evaluation/LibriSpeech/servers/streaming_websocket_server_fsl.py`

## 결과 위치

`evaluation/AMI/results/{model}/{scope}/{tag}/`

## AMI 데이터 형식

`words/` 디렉토리: `{sessionId}.{speakerId}.words.xml` 형식. 테스트 스크립트가 세션 단위로 화자를 병합.
