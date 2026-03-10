# Sentence Segmentation Guide

WhisperLiveKit의 문장 분절 기능 사용 가이드입니다.

## 개요

문장 분절 기능을 사용하면 온점, 물음표, 한국어/영어 종결어미가 나올 때 디코딩을 중단할 수 있습니다.
- **SimulStreaming**: 실시간 디코딩 중 문장 경계에서 중단
- **LocalAgreement**: 토큰을 문장으로 그룹화할 때 개선된 감지

## 주요 기능

- ✅ 온점, 물음표 등 문장 부호 감지
- ✅ 한국어 종결어미 감지 (`다`, `요`, `습니다` 등)
- ✅ 영어 종결 패턴 감지 (선택사항)
- ✅ Token-level 실시간 감지
- ✅ 기존 코드와 완전 호환 (기본값: 비활성화)

## 사용 방법

### CLI 옵션 (test_server.py)

**가장 간단한 방법**: test_server 실행 시 CLI 옵션으로 활성화

```bash
# 한국어 문장 분절 활성화
python -m whisperlivekit.test_server \
  --host localhost \
  --port 8001 \
  --model large-v3 \
  --lan ko \
  --backend-policy simulstreaming \
  --enable-sentence-segmentation \
  --segmentation-mode korean \
  --min-tokens-before-break 3

# 영어 문장 부호만 감지
python -m whisperlivekit.test_server \
  --host localhost \
  --port 8001 \
  --model large-v3 \
  --lan en \
  --backend-policy simulstreaming \
  --enable-sentence-segmentation \
  --segmentation-mode punctuation
```

### Python 코드 (SimulStreaming)

```python
from whisperlivekit.simul_whisper.backend import SimulStreamingASR, SimulStreamingOnlineProcessor

# 문장 분절 활성화
asr = SimulStreamingASR(
    model_size="large-v3",
    lan="ko",
    enable_sentence_segmentation=True,  # 문장 분절 켜기
    segmentation_mode="korean",         # "punctuation", "korean", "full" 중 선택
    min_tokens_before_break=3,          # 최소 3개 토큰 후 중단 허용
)

processor = SimulStreamingOnlineProcessor(asr)
# 이후 사용법은 기존과 동일
```

### Python 코드 (LocalAgreement)

```python
from whisperlivekit.local_agreement.backends import WhisperASR
from whisperlivekit.local_agreement.online_asr import OnlineASRProcessor

asr = WhisperASR(lan="ko", model_size="large-v3")

# 문장 분절 옵션 설정
asr.enable_sentence_segmentation = True
asr.segmentation_mode = "korean"

processor = OnlineASRProcessor(asr)
# 이후 사용법은 기존과 동일
```

### 기존 방식 사용 (문장 분절 비활성화)

```python
# 기본값이 False이므로 아무것도 설정하지 않으면 기존 동작 유지
asr = SimulStreamingASR(model_size="large-v3", lan="ko")

# 또는 명시적으로
asr = SimulStreamingASR(
    model_size="large-v3",
    lan="ko",
    enable_sentence_segmentation=False
)
```

## 설정 옵션

### `enable_sentence_segmentation` (bool, 기본값: False)

문장 분절 기능 활성화 여부

```python
enable_sentence_segmentation=True   # 활성화
enable_sentence_segmentation=False  # 비활성화 (기본값)
```

### `segmentation_mode` (str, 기본값: "full")

어떤 감지 방법을 사용할지 선택

- **`"off"`**: 비활성화
- **`"punctuation"`**: 온점, 물음표 등 문장 부호만 감지
- **`"korean"`**: 문장 부호 + 한국어 종결어미 (한국어 추천)
- **`"full"`**: 모든 감지 방법 (문장 부호 + 한국어 + 영어)

```python
segmentation_mode="punctuation"  # 온점, 물음표만
segmentation_mode="korean"       # 온점 + 한국어 종결어미 (추천)
segmentation_mode="full"         # 모든 감지 방법
```

### `min_tokens_before_break` (int, 기본값: 3)

문장 중단을 허용하기 전 최소 토큰 수. 너무 일찍 중단되는 것을 방지합니다.

```python
min_tokens_before_break=3   # 최소 3개 토큰 필요
min_tokens_before_break=5   # 더 보수적
min_tokens_before_break=1   # 경계 감지 즉시 중단
```

## 감지 패턴

### 문장 종결 부호
- 영어: `.`, `!`, `?`
- 한국어/중국어: `。`, `！`, `？`

### 한국어 종결어미
- 기본형: `다`, `요`, `까`, `니`, `죠`, `네`, `지`
- 격식체: `습니다`, `입니다`, `됩니다`, `합니다`
- 비격식체: `해요`, `가요`, `와요`, `네요`, `군요`, `겠어요`
- 기타: `어`, `아`, `지`, `죠`, `게`, `는데`, `거든`

### 영어 종결어 (선택사항)
- `right`, `okay`, `ok`, `yeah`, `yes`, `no`, `please`

## 사용 예시

### 예시 1: 한국어 대화 전사

```python
asr = SimulStreamingASR(
    model_size="large-v3",
    lan="ko",
    enable_sentence_segmentation=True,
    segmentation_mode="korean",
    min_tokens_before_break=3,
)
```

**입력 오디오**: "안녕하세요. 오늘 날씨가 좋네요. 어디 가실 거예요?"

**출력** (3개의 분절된 청크):
1. "안녕하세요."
2. "오늘 날씨가 좋네요."
3. "어디 가실 거예요?"

### 예시 2: 영어 공식 발표

```python
asr = SimulStreamingASR(
    model_size="large-v3",
    lan="en",
    enable_sentence_segmentation=True,
    segmentation_mode="punctuation",
    min_tokens_before_break=5,
)
```

**입력 오디오**: "Good morning. Today we will discuss the quarterly results. Please turn to page five."

**출력** (3개의 분절된 청크):
1. "Good morning."
2. "Today we will discuss the quarterly results."
3. "Please turn to page five."

## 언제 사용하면 좋을까?

### 사용 권장
- ✅ 실시간 전사에서 자연스러운 중단점이 필요할 때
- ✅ 한국어/영어 대화형 음성을 전사할 때
- ✅ 지나치게 긴 전사 청크를 방지하고 싶을 때
- ✅ 번역 등 후처리를 위해 문장 단위 경계가 필요할 때

### 사용 비권장
- ❌ 연속적이고 끊김 없는 전사가 필요할 때
- ❌ 오디오에 문장 부호가 거의 없을 때
- ❌ 기술 문서 등 특이한 문장 구조를 처리할 때
- ❌ 최대한의 디코딩 유연성이 필요할 때

## 문제 해결

### 문제: 분절이 너무 짧음
**해결**: `min_tokens_before_break`를 늘리거나 `segmentation_mode="punctuation"` 사용

### 문제: 분절이 너무 김
**해결**: `min_tokens_before_break`를 줄이거나 오디오에 적절한 문장 부호가 있는지 확인

### 문제: 한국어 종결어미가 감지되지 않음
**해결**: `segmentation_mode="korean"` 또는 `"full"` 사용, Whisper 모델이 올바른 한국어 종결어미를 생성하는지 확인

### 문제: 분절이 전혀 안 됨
**해결**: `enable_sentence_segmentation=True`로 설정되었는지 확인, 로그에서 감지 메시지 확인

## 성능 고려사항

- **최소 오버헤드**: 문장 분절은 거의 무시할 수 있는 수준의 처리 시간만 추가
- **모델 변경 없음**: 동일한 Whisper 모델 사용, 중단 시점만 제어
- **완전 호환**: 비활성화 시 원래 동작과 완전히 동일

## 구현 파일

- [`sentence_segmentation.py`](../whisperlivekit/sentence_segmentation.py) - 핵심 분절 로직
- [`simul_whisper/config.py`](../whisperlivekit/simul_whisper/config.py) - SimulStreaming 설정
- [`simul_whisper/simul_whisper.py`](../whisperlivekit/simul_whisper/simul_whisper.py) - SimulStreaming 통합
- [`local_agreement/online_asr.py`](../whisperlivekit/local_agreement/online_asr.py) - LocalAgreement 통합
