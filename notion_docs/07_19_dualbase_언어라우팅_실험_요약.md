# 07_19_dualbase_언어라우팅_실험_요약

## 📅 날짜
2026-07-19

## 🔧 작업 내용

### 1. 원인 분석 — dualbase 모델 라우팅 구조적 지연
`streaming_websocket_server_dualbase.py`의 `_active_asr`/`_get_lora_request`가 `_canonical_lang`(=`state.language`)로 en/ko 두 vLLM 엔진 중 하나를 고르는데, `state.language`는 직전 청크 디코딩 결과에서만 갱신됨. 언어가 전환되는 순간엔 최소 1청크가 이전 언어 엔진으로 잘못 디코딩됨. 실제 운영 로그에서 "language Korean" 태그가 본문에 그대로 새는 케이스, en-en/ko-ko 무의미 번역 케이스를 확인.

### 2. Bug fix — last_text_lang → state.language 전파 (`fix/dualbase-slot-language-carry` 브랜치, 미커밋)
DOT-SLOT-SWITCH/FORCE-SLOT-SWITCH가 같은 발화를 기술적으로만 리셋할 때 `last_text_lang`을 새 슬롯에 넘겨주면서도 `state.language`엔 반영을 안 하던 버그 수정. 9줄 변경, 부작용 없음.

### 3. LID(오디오 언어감지) 슬롯-시작 1회 프로토타입 — 기각
faster-whisper tiny/base, speechbrain VoxLingua107 3종을 같은 22개 ko/en 클립으로 비교. tiny 302ms/77.8%, base 682ms/88.9%, VoxLingua107 225ms/83.3%. 셋 다 "0.2초 이내 + 90%+" 기준 미달로 보류.

### 4. 커밋시점 한국어 재전사(refine) 프로토타입 (`feat/dualbase-ko-refine-commit` 브랜치, 미커밋) — 기각
SEG로 문장이 완전히 끝난 시점에 원본 오디오를 asr_ko로 재전사해 최종 텍스트를 교체하는 방식. 실제 dualbase 서버(en+ko 병합 체크포인트)로 스모크 테스트한 결과 문장당 평균 1296ms 추가 지연(최대 2380ms) 확인. 추가로 audio_accum이 슬롯 리셋 없이 여러 문장에 걸쳐 누적될 때 재전사 대상 오디오가 이전 문장들과 섞이는 정렬 버그 발견. 지연·구현 위험 모두 과도해 폐기.

### 5. 조기 abort + 재라우팅 프로토타입 — 기각
스트리밍 부분출력의 "language X" 태그가 현재 라우팅된 엔진과 어긋나는 순간 generate()를 abort하고 올바른 엔진으로 재시작하는 방식. en+ko 실제 엔진으로 10개 시나리오 테스트 결과 naive 대비 정확도 개선 0%(4/10 → 4/10), 지연은 오히려 1.58배 증가. en 특화 모델이 한국어 오디오에도 "English"를 흔들림 없이 자신 있게 오보고하는 게 원인 — 감지할 신호 자체가 없음.

### 6. 듀얼 디코딩(en+ko 동시) + logprob 비교 — 참고용, 미채택
매 청크를 두 엔진에 동시에(asyncio.gather) 태워서 평균 logprob이 높은 쪽을 채택. 정확도 9/12(75%), 지연 오버헤드 1.58배(2배가 아님 — 공유 GPU에서도 부분 병렬 확인). 짧은 문장에서 두 모델 간 logprob 비교가 불안정해 이후 검증 안 함.

### 7. 단일 한국어 모델 검증 — 채택 방향
현재 실제 배포된 systemd 설정(dualbase 아닌 `streaming_websocket_server.py` + ko 병합 단일 모델)을 그대로 재현해 테스트. 18개 ko/en 클립 기준 언어태그 일치율 89%(16/18), 지연 중앙값 638ms — 시도한 모든 dualbase 라우팅 방식보다 정확도·지연 둘 다 우수. 라우팅 자체가 없어 "완전히 다른 모델이 오디오를 통째로 오인식"하는 파국적 실패가 구조적으로 불가능해짐.

### 8. 텍스트 스크립트 힌트로 잔여 라벨 오류 보정 — 검증 완료, 미적용
단일모델 테스트의 잔여 2건 실패 중 1건("텍스트는 맞는데 language 태그만 틀림")은 한글/로마자 문자비율 휴리스틱으로 교정 가능함을 확인(89%→94%). 모델 호출 없는 순수 함수라 비용 거의 0. 프로덕션 서버(`streaming_websocket_server.py`의 `_correct_and_translate`)에 아직 미적용.

### 9. 빠른 언어교차(무음 간격 0) 연속 스트리밍 테스트
단일 ko 모델에 무음 간격 없이 ko/en 10조각을 이어붙여 실제 streaming_transcribe를 청크 단위로 반복 호출. SEG 리셋을 실제 서버와 동일하게 반영해도, 언어 전환 직후 1~2단어가 이전 언어 스크립트로 음차되었다가("Welcome to Mansion Hotel"→"웰컴 투 맨션 호텔", "Okay"→"오케이") 곧 스스로 정상 언어로 복귀하는 패턴을 재현 확인. 1초 미만의 짧은 발화가 인접 세그먼트에 흡수되어 유실되는 것도 확인.

## 📊 결과 / 수치

| 방식 | 언어 정확도 | 지연(중앙값 기준) |
|---|---|---|
| dualbase + LID (tiny/base/VoxLingua107) | 77.8~88.9% | +225~682ms |
| dualbase + 커밋시점 재전사 | — | +1296ms |
| dualbase + 조기abort 재라우팅 | 40%(개선 없음) | 1370ms |
| dualbase + 듀얼디코딩(en+ko 동시) | 75% | 1589ms |
| **단일 ko 모델(현재 배포 설정)** | **89%(16/18)** | **638ms** |
| 단일 ko 모델 + 스크립트 힌트 보정 | **94%(17/18)** | 추가 지연 없음 |

## 🐛 발견된 문제 및 해결

| 문제 | 원인 | 해결/결론 |
|---|---|---|
| dualbase en-en/ko-ko 무의미 번역 | `state.language`가 직전 청크 기준이라 언어 전환 시 라우팅 지연 | 근본적으로는 라우팅 구조 자체의 한계. `fix/dualbase-slot-language-carry`로 일부 케이스(DOT/FORCE-SLOT-SWITCH) 완화 |
| "language Korean" 태그가 본문에 유출 | 스트리밍 롤백 중 `<asr_text>` 스페셜 토큰이 청크 경계에 걸려 파싱 실패 | 원인만 특정, 미수정 |
| 재전사 시 오디오-텍스트 정렬 오류 | audio_accum이 슬롯 리셋 없이 여러 문장에 걸쳐 누적되는 경우를 가정 못 함 | 재전사 접근 자체를 폐기하며 해소 |
| en 특화 모델이 한국어를 "English"로 확신에 차 오보고 | en/ko 두 체크포인트의 파인튜닝 정도 비대칭 | 근본 원인만 특정(재학습 필요), 우회책 없음 |
| 텍스트는 맞는데 language 라벨만 틀림 | `state.language` 태그 파싱 신뢰도 문제 | 스크립트(한글/로마자) 휴리스틱으로 검증 완료, 프로덕션 미적용 |
