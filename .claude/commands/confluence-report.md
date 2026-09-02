업무 보고 문서를 Confluence STiTy 스페이스에 만들어줘.

계획 문서와 **같은 폴더**(`문서 정리 > <n>차 업무 분담`)에 들어가고, 제목은
`[n차][대범주][소범주] 보고 문서`, 구분은 **라벨**로 한다.

**형식을 외워서 쓰지 말 것.** 표 구조는 [업무 보고 작성 가이드](https://stity.atlassian.net/wiki/spaces/STiTy/pages/327683)
에서 매번 읽어온다.

---

## 1. 현재 형식 확인 (항상 먼저)

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --show-format report
```

출력된 항목 이름을 그대로 `fields` 키로 쓴다. 계획 문서와 다른 점은 보통 두 가지다 —
목표 기한 대신 **완료 일자**, 그리고 **파생된 계획 문서** 링크. 업무 세분화 목록은 없다.
다만 이것도 가이드가 바뀌면 달라지니 **출력된 것을 믿을 것.**

## 2. 값 모으기

이 세션에서 실제로 한 작업을 근거로 채워줘. 추측해서 채우지 말 것.

| 항목 | 설명 |
|---|---|
| `round` | 몇 차 업무 분담인지. 숫자만 |
| `major` / `minor` | 대범주 / 소범주. **대응하는 계획 문서와 같아야 한다** |
| `jira` | Jira 이슈 키. 없으면 생략 |

파생된 계획 문서의 제목은 **실제로 있는지 먼저 확인**해줘. 없는 제목을 넣으면 링크가 깨진 채로
올라간다:

```bash
set -a; . ./.env; set +a
curl -s -u "$ATLASSIAN_EMAIL:$ATLASSIAN_API_TOKEN" -H "Accept: application/json" \
  --data-urlencode 'cql=space=STiTy and label="계획"' --get \
  "https://stity.atlassian.net/wiki/rest/api/content/search?limit=20"
```

## 3. JSON 만들기

`fields` 값의 `type` 은 계획 스킬과 같다 — `text` / `list` / `links` / `pagelink`.
계획 문서 링크처럼 문서 하나만 걸 때는 `pagelink` 를 쓴다.

```json
{
  "kind": "report",
  "round": "1",
  "major": "시연 대비",
  "minor": "아랍어 지원",
  "jira": "STITY-7",
  "fields": {
    "업무 완료 일자": {"type": "text", "value": "2026-09-07"},
    "작성자": {"type": "text", "value": "정다현"},
    "업무 카테고리": {"type": "text", "value": "범용"},
    "파생된 계획 문서": {"type": "pagelink", "value": "[1차][시연 대비][아랍어 지원] 계획 문서"},
    "업무 내용 정리 링크": {"type": "links", "value": []}
  }
}
```

## 4. 올리기

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/report.json --dry-run
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/report.json
```

동작 규칙은 계획 스킬과 같다 — 폴더 자동 생성, 같은 제목이면 멈춤, 비워둔 칸은 경고.

## 5. 마무리

올라간 주소를 출력해줘. 내용 확인은 사용자가 Confluence 에서 직접 한다.

**업무 내용 자체를 정리한 문서**(`업무 내용 정리 링크` 에 넣을 페이지)는 이 스킬이 만들지 않는다.
따로 작성해 올린 뒤 제목만 넣어줘.
