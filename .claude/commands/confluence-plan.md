업무 계획 문서를 Confluence STiTy 스페이스에 만들어줘.

문서는 `문서 정리 > <n>차 업무 분담` 폴더에 들어가고, 제목은 `[n차][대범주][소범주] 계획 문서`,
계획/보고 구분은 **라벨**로 한다.

**형식을 외워서 쓰지 말 것.** 표 구조는 [업무 계획 작성 가이드](https://stity.atlassian.net/wiki/spaces/STiTy/pages/294931)
에서 매번 읽어온다. 가이드에 칸이 늘거나 이름이 바뀌면 그게 그대로 반영된다.

---

## 1. 현재 형식 확인 (항상 먼저)

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --show-format plan
```

출력된 항목 이름이 그대로 JSON 의 `fields` 키가 된다. **이름을 임의로 바꾸거나 줄이면 그 칸은
비어서 올라간다.** 가이드에 없는 이름을 넣으면 무시되고 경고만 뜬다.

## 2. 값 모으기

이미 대화에서 알 수 있는 값은 다시 묻지 말고 채워줘. 모르는 것만 물어봐줘.

제목과 라벨에 쓰는 값 세 개는 가이드와 무관하게 항상 필요하다.

| 항목 | 설명 |
|---|---|
| `round` | 몇 차 업무 분담인지. 숫자만 (예: `1`) |
| `major` | 대범주 — **시연 대비 / 미래지향연구 / 기존개선** 중 하나 |
| `minor` | 소범주 — 해당 차수 업무 분담 페이지에 쓰인 표현이 있으면 그대로 |
| `jira` | Jira 이슈 키 (예: `STITY-7`). 없으면 생략 |

대범주·소범주가 헷갈리면 해당 차수의 업무 분담 페이지를 먼저 읽고 거기 `[대범주] [소범주]` 표기를
그대로 가져와줘:

```bash
set -a; . ./.env; set +a
curl -s -u "$ATLASSIAN_EMAIL:$ATLASSIAN_API_TOKEN" -H "Accept: application/json" \
  --data-urlencode 'cql=space=STiTy and title~"업무 분담"' --get \
  "https://stity.atlassian.net/wiki/rest/api/content/search?limit=20"
```

## 3. JSON 만들기

스크래치패드에 저장해줘. 리포 안에 두지 말 것.

`fields` 의 각 값은 `type` 과 `value` 를 갖는다.

| `type` | 쓸 때 | `value` |
|---|---|---|
| `text` | 날짜, 이름, 카테고리 같은 한 줄 | 문자열 |
| `list` | 번호 매긴 목록 (예: 업무 세분화 내용) | 문자열 배열 |
| `links` | 다른 Confluence 문서 여럿 링크 | **페이지 제목** 배열 (URL 아님) |
| `pagelink` | 문서 하나 링크 | 페이지 제목 |

```json
{
  "kind": "plan",
  "round": "1",
  "major": "시연 대비",
  "minor": "아랍어 지원",
  "jira": "STITY-7",
  "fields": {
    "계획 수립 일자": {"type": "text", "value": "2026-09-02"},
    "작성자": {"type": "text", "value": "정다현"},
    "업무 카테고리": {"type": "text", "value": "범용"},
    "목표 기한": {"type": "text", "value": "2026-09-07"},
    "업무 세분화 내용": {"type": "list", "value": ["아랍어 DOT commit 기준 정리", "파인튜닝 데이터 준비"]},
    "업무 내용 정리 링크": {"type": "links", "value": []}
  }
}
```

`업무 카테고리` 는 **Jira 에서 부여한 업무 유형과 같은 값**을 넣어야 한다.

## 4. 올리기

```bash
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/plan.json --dry-run
PYTHONPATH= .venv/bin/python .claude/confluence/confluence_doc.py --json <경로>/plan.json
```

`--dry-run` 으로 먼저 확인해줘. **"값이 없어 비워둔 항목" 경고가 뜨면 그냥 넘어가지 말고**
사용자에게 그 칸을 물어봐줘 — 가이드에 새로 생긴 칸일 수 있다.

- `<n>차 업무 분담` 폴더가 없으면 `문서 정리` 아래에 자동으로 만든다
- 같은 제목이 이미 있으면 **덮어쓰지 않고 멈춘다.** 기존 문서를 고칠지 제목을 바꿀지 물어봐줘

## 5. 마무리

올라간 주소를 출력해줘. 내용 확인은 사용자가 Confluence 에서 직접 한다.
