현재 브랜치의 변경사항을 커밋하고 원격에 푸시해줘. 아래 절차를 순서대로 실행해줘.

---

## 1. git pull (먼저 원격 동기화)

`git pull` 실행. 충돌 없으면 다음 단계로 진행.

**pull 충돌 발생 시:**
- 충돌 파일 목록을 보여주고 파일별로 선택하게 해줘:
  - `ours` — 내 변경사항 유지 (`git checkout --ours`)
  - `theirs` — 원격 변경사항 채택 (`git checkout --theirs`)
  - `manual` — 충돌 내용 직접 보여주고 합칠 방법 물어봐줘
- 해결 후 `git add` + `git commit` 으로 머지 커밋 완료

---

## 2. 대용량 파일 / 모델 가중치 자동 gitignore 등록

스테이징 전에 아래 기준으로 추적되지 않은 파일을 검사하고, 해당하는 항목은 `.gitignore`에 자동 추가해줘.

**자동 추가 대상:**
- **50MB 초과** 파일 (단일 파일 기준)
- **확장자 기준:** `.pt`, `.pth`, `.safetensors`, `.gguf`, `.ckpt`, `.bin`, `.h5`, `.pkl`, `.npy`, `.npz`, `.arrow`, `.parquet`
- **폴더명 기준:** 경로에 `checkpoint`, `weights`, `ckpt`, `pretrained`, `model_cache` 가 포함된 폴더

이미 `.gitignore`에 있는 패턴은 중복 추가 금지.
추가된 항목이 있으면 어떤 패턴을 추가했는지 알려줘.

---

## 3. 커밋

1. `git status`로 변경사항 확인
2. 변경 내용을 분석해 **Conventional Commits** 형식으로 한국어 커밋 메시지 자동 생성

**형식:**
```
<type>(<scope>): <한국어 요약>

- 세부 변경사항 bullet
```

**type 선택 기준:**
- `feat` — 새 기능, 새 모듈, 새 스크립트 추가
- `fix` — 버그 수정
- `refactor` — 동작 변화 없는 코드 구조 변경
- `perf` — 성능 개선 (속도, 메모리 등)
- `test` — 테스트 추가 또는 수정
- `docs` — 문서, 주석, CLAUDE.md 변경
- `chore` — 설정 파일, .gitignore, 패키지, CI 등

**scope 선택 기준** (변경된 영역에 따라):
- `asr` — Qwen3-ASR, streaming_websocket_server
- `fsl` — evaluation FSL 서버 및 테스트
- `eval` — 평가 스크립트, metric, context scoring
- `core` — core/ 모듈 (corrector, translator 등)
- `mobile` — STiTy-Mobile
- `config` — 설정 파일, .gitignore, .mcp.json, CLAUDE.md
- scope가 여러 개면 가장 대표적인 것 하나만 사용

3. 스테이징 목록을 보여주고 확인 받은 뒤 커밋

**스테이징 제외 (절대 포함 금지):** `.env`, `*.pem`, `*secret*`, `*credential*`

---

## 4. git push

푸시 완료 후 결과 요약 출력.
