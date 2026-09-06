업무 보고 문서를 Confluence STiTy 스페이스에 만들어줘.

문서는 `문서 정리 > <n>차 업무 분담 > 보고 문서` 폴더에 들어가고(계획 문서는 옆의
`계획 문서` 폴더에 있다), 제목은
`[n차][대범주][소범주][세분화 업무명] 보고 문서`, 구분은 **라벨**로 한다.

계획 문서와 달리 **세분화 업무명이 한 칸 더 붙는다.** 보고는 세분화된 업무 하나마다
쓰는 것이라서다. 그 이름은 계획 문서의 `업무 세분화 내용` 항목(= 그 업무의 Jira 이슈 요약)
문구를 **그대로** 가져와야 문서와 이슈가 짝지어진다.

**형식을 외워서 쓰지 말 것.** 표 구조는 [업무 보고 작성 가이드](https://stity.atlassian.net/wiki/spaces/STiTy/pages/327683)
에서 매번 읽어온다.

---

## 1. 현재 형식 확인 (항상 먼저)

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --show-format report
```

출력된 항목 이름을 그대로 `fields` 키로 쓴다. 계획 문서와 다른 점은 보통 세 가지다 —
목표 기한 대신 **완료 일자**, **파생된 계획 문서** 링크, 그리고 실제로 한 일을 적는
**업무 내용 정리**. 업무 세분화 목록은 없다.
다만 이것도 가이드가 바뀌면 달라지니 **출력된 것을 믿을 것.**

## 2. 값 모으기

이 세션에서 실제로 한 작업을 근거로 채워줘. 추측해서 채우지 말 것.

| 항목 | 설명 |
|---|---|
| `round` | 몇 차 업무 분담인지. 숫자만 |
| `major` / `minor` | 대범주 / 소범주. **대응하는 계획 문서와 같아야 한다** |
| `task` | 세분화 업무명. 제목 넷째 칸. **보고 문서에는 반드시 필요하다** — 빠지면 스크립트가 멈춘다 |
| `jira` | Jira 이슈 키. 없으면 생략 |

`task` 는 계획 문서의 `업무 세분화 내용` 에 적힌 문구를 그대로 쓴다. 헷갈리면 그 업무의
Jira 이슈 요약을 확인해줘 — 이슈는 그 문구로 만들어졌다.

파생된 계획 문서의 제목은 **실제로 있는지 먼저 확인**해줘. 없는 제목을 넣으면 링크가 깨진 채로
올라간다:

```bash
set -a; . ./.env; set +a
curl -s -u "$ATLASSIAN_EMAIL:$ATLASSIAN_API_TOKEN" -H "Accept: application/json" \
  --data-urlencode 'cql=space=STiTy and label="계획"' --get \
  "https://stity.atlassian.net/wiki/rest/api/content/search?limit=20"
```

## 3. JSON 만들기

`fields` 값의 `type` 은 계획 스킬과 같고(`text` / `list` / `links` / `pagelink`),
보고 전용으로 `markdown` 이 하나 더 있다. 계획 문서 링크처럼 문서 하나만 걸 때는 `pagelink`.

### 업무 내용 정리는 `markdown` 으로 쓴다

한 문단으로 뭉쳐 쓰지 말고 **md 쓰듯 소제목으로 내용을 갈라서** 써줘. 소제목은 배경 /
한 일 / 결과 / 남은 것처럼 실제 업무 흐름에 맞게 정하면 된다 — 정해진 이름은 없다.

`value` 는 md 문자열 하나다. JSON 이라 줄바꿈은 `\n` 으로 넣는다. 지원하는 문법:

| 문법 | 결과 |
|---|---|
| `# 소제목` | 칸 안의 소제목 (`#` 이 깊을수록 아래 단계) |
| `- 항목` / `1. 항목` | 글머리 / 번호 목록. 두 칸 들여쓰면 중첩된다 |
| `` `코드` ``, `**굵게**` | 인라인 코드, 굵게 |
| ` ```python ` … ` ``` ` | 코드 블록 (Confluence 코드 매크로) |

표와 링크 문법은 안 된다. 다른 Confluence 문서를 걸 때는 `links` / `pagelink` 칸을 쓴다.

수치는 **실제로 측정한 값만** 적어줘. 안 재본 값을 그럴듯하게 적지 말 것.

```json
{
  "kind": "report",
  "round": "1",
  "major": "시연 대비",
  "minor": "아랍어 지원",
  "task": "아랍어 DOT commit 기준 정리",
  "jira": "STITY-7",
  "fields": {
    "업무 완료 일자": {"type": "text", "value": "2026-09-07"},
    "작성자": {"type": "text", "value": "정다현"},
    "업무 카테고리": {"type": "text", "value": "범용"},
    "파생된 계획 문서": {"type": "pagelink", "value": "[1차][시연 대비][아랍어 지원] 계획 문서"},
    "업무 내용 정리": {"type": "markdown", "value": "# 배경\n아랍어는 문장부호가 `؟` 라서 기존 dot commit 이 안 걸린다.\n\n# 한 일\n- `--enable-dot-commit` 에 아랍어 부호 추가\n  - `؟` 물음표\n  - `۔` 마침표\n- 회귀 테스트 12건 통과\n\n# 결과\n1. FSL 평균 **0.42초** 단축\n2. 잘못 끊긴 문장 3건이 0건\n\n# 남은 것\n분절 품질 평가는 다음 차수로 넘긴다."}
  }
}
```

## 4. 올리기

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/report.json --dry-run
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/report.json
```

동작 규칙은 계획 스킬과 같다 — 폴더 자동 생성, 비워둔 칸은 경고, 같은 제목이 이미 있으면
마지막 태그(보고 문서는 세분화 업무명) 옆에 `[2]` 처럼 번호를 붙여 새 문서로 만든다
(`seq` 로 직접 지정, `--no-autonumber` 로 끔).

## 5. 마무리

올라간 주소를 출력해줘. 내용 확인은 사용자가 Confluence 에서 직접 한다.

업무 내용은 이제 별도 페이지가 아니라 `업무 내용 정리` 칸 안에 바로 들어간다. 따로 정리한
문서가 이미 있으면 그 문서를 대신 걸지 말고, 요지를 md 로 옮겨 적고 링크는 본문에 문장으로
덧붙여줘.
