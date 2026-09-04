#!/usr/bin/env python3
"""업무 계획/보고 문서를 Confluence STiTy 스페이스에 만든다.

가이드 문서(업무 계획/보고 작성 가이드)의 표 구조를 그대로 재현한다.
스킬(.claude/commands/confluence-plan.md, confluence-report.md)이 JSON 을 만들어
이 스크립트에 넘기는 방식이다. 인증은 .env 의 ATLASSIAN_EMAIL/ATLASSIAN_API_TOKEN.

  python .claude/confluence/confluence_doc.py --json payload.json
  python .claude/confluence/confluence_doc.py --json payload.json --dry-run
"""
import argparse
import html
import json
import os
import re
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]

# storage 형식은 ac:/ri: 접두사를 그대로 요구한다. 등록하지 않으면 ns0:/ns1: 로 나가서
# Confluence 가 매크로와 페이지 링크를 알아보지 못한다.
ET.register_namespace("ac", "http://atlassian.com/content")
ET.register_namespace("ri", "http://atlassian.com/resource/identifier")
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))


def load_env() -> tuple[str, str]:
    """.env 에서 이메일과 토큰을 읽는다. 환경변수가 이미 있으면 그쪽을 쓴다."""
    env = {}
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    email = os.environ.get("ATLASSIAN_EMAIL") or env.get("ATLASSIAN_EMAIL")
    token = os.environ.get("ATLASSIAN_API_TOKEN") or env.get("ATLASSIAN_API_TOKEN")
    if not email or not token:
        sys.exit("ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN 이 .env 에 있어야 한다.")
    return email, token


class Confluence:
    def __init__(self, email: str, token: str):
        self.base = CONFIG["base_url"]
        self.client = httpx.Client(auth=(email, token), timeout=30.0,
                                   headers={"Accept": "application/json"})

    def _check(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            sys.exit(f"Confluence API 실패 {r.status_code}: {r.text[:600]}")
        return r.json() if r.content else {}

    def find_folder(self, title: str) -> str | None:
        """폴더 이름은 스페이스 안에서 유일하므로 전체에서 찾는다.

        부모 아래만 뒤지면, 같은 이름이 다른 자리에 있을 때 생성이 400 으로 막힌다.
        """
        r = self.client.get(f"{self.base}/rest/api/search", params={
            "cql": f'space={CONFIG["space_key"]} and type=folder', "limit": 100,
        })
        for item in self._check(r).get("results", []):
            if item.get("title") == title:
                return item["content"]["id"]
        return None

    def create_folder(self, parent_id: str, title: str) -> str:
        r = self.client.post(f"{self.base}/api/v2/folders", json={
            "spaceId": CONFIG["space_id"], "title": title, "parentId": parent_id,
        })
        return self._check(r)["id"]

    def get_storage(self, page_id: str) -> str:
        r = self.client.get(f"{self.base}/rest/api/content/{page_id}",
                            params={"expand": "body.storage"})
        return self._check(r)["body"]["storage"]["value"]

    def find_page(self, title: str) -> str | None:
        r = self.client.get(f"{self.base}/rest/api/content",
                            params={"spaceKey": CONFIG["space_key"], "title": title, "limit": 1})
        results = self._check(r).get("results", [])
        return results[0]["id"] if results else None

    def create_page(self, parent_id: str, title: str, body: str) -> dict:
        r = self.client.post(f"{self.base}/api/v2/pages", json={
            "spaceId": CONFIG["space_id"], "status": "current", "title": title,
            "parentId": parent_id,
            "body": {"representation": "storage", "value": body},
        })
        return self._check(r)

    def add_labels(self, page_id: str, labels: list[str]) -> None:
        if not labels:
            return
        r = self.client.post(f"{self.base}/rest/api/content/{page_id}/label",
                             json=[{"prefix": "global", "name": n} for n in labels])
        self._check(r)


def esc(s: str) -> str:
    return html.escape(str(s or ""), quote=True)


def page_link(title: str) -> str:
    """다른 Confluence 페이지를 인라인 카드로 건다."""
    return (f'<ac:link ac:card-appearance="inline">'
            f'<ri:page ri:content-title="{esc(title)}" />'
            f'<ac:link-body>{esc(title)}</ac:link-body></ac:link>')


def jira_macro(key: str) -> str:
    j = CONFIG["jira"]
    return ('<ac:structured-macro ac:name="jira" ac:schema-version="1">'
            f'<ac:parameter ac:name="key">{esc(key)}</ac:parameter>'
            f'<ac:parameter ac:name="serverId">{j["server_id"]}</ac:parameter>'
            f'<ac:parameter ac:name="server">{esc(j["server"])}</ac:parameter>'
            '</ac:structured-macro>')


class Jira:
    """계획의 세분화된 업무를 이슈로 만든다. Confluence 와 같은 토큰을 쓴다."""

    def __init__(self, email: str, token: str):
        self.base = CONFIG["jira"]["base_url"]
        self.client = httpx.Client(auth=(email, token), timeout=30.0,
                                   headers={"Accept": "application/json"})

    def _check(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            sys.exit(f"Jira API 실패 {r.status_code}: {r.text[:600]}")
        return r.json() if r.content else {}

    def find_by_summary(self, project: str, summary: str) -> str | None:
        """같은 요약의 이슈가 이미 있으면 그 키를 준다. 두 번 돌려도 중복이 안 생기게."""
        jql = f'project="{project}" and summary~"\\"{summary}\\"" order by created desc'
        r = self.client.get(f"{self.base}/rest/api/3/search/jql",
                            params={"jql": jql, "maxResults": 5, "fields": "summary"})
        for issue in self._check(r).get("issues", []):
            if issue["fields"]["summary"].strip() == summary.strip():
                return issue["key"]
        return None

    def assignable(self, project: str) -> list[dict]:
        r = self.client.get(f"{self.base}/rest/api/3/user/assignable/search",
                            params={"project": project, "maxResults": 50})
        return [u for u in self._check(r) if u.get("active")]

    def account_id(self, project: str, who: str) -> str:
        """사람 이름을 accountId 로 바꾼다.

        표시 이름이 계정 아이디인 사람이 있어 부분 검색은 못 믿는다. 배정 가능한
        사람 목록에서 정확히 일치하는 이름만 받는다. accountId 를 직접 줘도 된다.
        """
        if ":" in who:
            return who
        users = self.assignable(project)
        hits = [u for u in users if u.get("displayName", "").strip() == who.strip()]
        if len(hits) == 1:
            return hits[0]["accountId"]
        names = ", ".join(u.get("displayName", "?") for u in users)
        if not hits:
            sys.exit(f"'{who}' 라는 담당자를 찾지 못했다. 배정 가능한 사람: {names}")
        sys.exit(f"'{who}' 가 여러 명이다. accountId 로 직접 지정할 것: {names}")

    def create(self, project: str, issuetype: str, summary: str, description: str = "",
               assignee_id: str | None = None) -> str:
        fields = {
            "project": {"key": project},
            "issuetype": {"name": issuetype},
            "summary": summary,
        }
        if assignee_id:
            fields["assignee"] = {"accountId": assignee_id}
        if description:
            fields["description"] = {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph",
                             "content": [{"type": "text", "text": description}]}],
            }
        r = self.client.post(f"{self.base}/rest/api/3/issue", json={"fields": fields})
        return self._check(r)["key"]


INLINE_RE = re.compile(r"`[^`]+`|\*\*[^*]+\*\*")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def inline(text: str) -> str:
    """줄 안의 `코드` 와 **굵게** 만 살린다. 나머지는 그대로 escape."""
    out, pos = [], 0
    for m in INLINE_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(f"<code>{esc(tok[1:-1])}</code>")
        else:
            out.append(f"<strong>{esc(tok[2:-2])}</strong>")
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def code_macro(text: str, lang: str = "") -> str:
    param = f'<ac:parameter ac:name="language">{esc(lang)}</ac:parameter>' if lang else ""
    return ('<ac:structured-macro ac:name="code" ac:schema-version="1">' + param +
            f"<ac:plain-text-body>{esc(text)}</ac:plain-text-body>"
            "</ac:structured-macro>")


def build_list(items: list[tuple[int, bool, str]], idx: int, depth: int) -> tuple[str, int]:
    """(들여쓰기 깊이, 번호목록 여부, 글) 목록을 <ul>/<ol> 로 접는다. 중첩도 살린다."""
    ordered = items[idx][1]
    tag = "ol" if ordered else "ul"
    parts = ['<ol start="1">' if ordered else "<ul>"]
    while idx < len(items):
        d, o, text = items[idx]
        if d < depth or (d == depth and o != ordered):
            break
        if d > depth:
            sub, idx = build_list(items, idx, d)
            parts[-1] = parts[-1][: -len("</li>")] + sub + "</li>"
            continue
        parts.append(f"<li><p>{inline(text)}</p></li>")
        idx += 1
    parts.append(f"</{tag}>")
    return "".join(parts), idx


def take_list(lines: list[str], i: int) -> tuple[str, int]:
    items = []
    while i < len(lines):
        m = LIST_RE.match(lines[i])
        if not m:
            break
        items.append((len(m.group(1)) // 2, m.group(2)[0] not in "-*+", m.group(3).strip()))
        i += 1
    return build_list(items, 0, items[0][0])[0], i


def markdown(text: str) -> str:
    """md 로 쓴 업무 내용을 storage 형식으로 바꾼다.

    소제목(#), 목록(- / 1.), 코드블록(```), 문단, 인라인 `코드`·**굵게** 를 지원한다.
    표나 링크 문법은 지원하지 않는다 — 링크는 links/pagelink 칸을 쓴다.
    """
    lines = str(text or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    i = 0
    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()
        if not stripped:
            flush()
            i += 1
            continue
        m = HEADING_RE.match(stripped)
        if m:
            flush()
            # 가이드의 절 제목이 h3 이라 칸 안의 소제목은 그 아래 단계부터 쓴다.
            lv = min(len(m.group(1)) + 3, 6)
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            out.append(code_macro("\n".join(buf), lang))
            i += 1
            continue
        if LIST_RE.match(line):
            flush()
            block, i = take_list(lines, i)
            out.append(block)
            continue
        para.append(stripped)
        i += 1
    flush()
    return "".join(out) or "<p />"


def cell(value, kind: str) -> str:
    """JSON 의 값 하나를 storage 형식 칸 내용으로 바꾼다."""
    if kind == "list":
        if not value:
            return "<p />"
        return '<ol start="1">' + "".join(
            f"<li><p>{esc(v)}</p></li>" for v in value) + "</ol>"
    if kind == "links":
        if not value:
            return "<p />"
        return '<ol start="1">' + "".join(
            f"<li><p>{page_link(v)}</p></li>" for v in value) + "</ol>"
    if kind == "pagelink":
        return f"<p>{page_link(value)}</p>" if value else "<p />"
    if kind == "markdown":
        return markdown(value)
    if kind == "jira":
        return f"<p>{jira_macro(value)}</p>" if value else "<p />"
    return f"<p>{esc(value)}</p>"


def parse_storage(xhtml: str) -> ET.Element:
    """storage 형식을 파싱한다. ac:/ri: 접두사와 &nbsp; 때문에 손질이 필요하다."""
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)([a-zA-Z]+);",
                  lambda m: html.unescape(f"&{m.group(2)};"), xhtml)
    wrapper = ('<root xmlns:ac="http://atlassian.com/content" '
               'xmlns:ri="http://atlassian.com/resource/identifier">'
               f"{text}</root>")
    return ET.fromstring(wrapper)


def row_labels(root: ET.Element) -> list[str]:
    """템플릿 표의 왼쪽 항목 이름들을 순서대로 뽑는다."""
    names = []
    for tr in root.iter("tr"):
        th = tr.find("th")
        if th is not None:
            name = "".join(th.itertext()).strip()
            if name:
                names.append(name)
    return names


def headings(root: ET.Element) -> list[str]:
    return ["".join(h.itertext()).strip()
            for lv in ("h1", "h2", "h3", "h4") for h in root.iter(lv)]


def render(template: str, fields: dict, jira_keys: list[str]) -> tuple[str, list[str]]:
    """가이드 본문을 틀로 삼아 값을 끼워 넣는다.

    표의 왼쪽 항목 이름으로 짝을 맞춘다. 값이 없는 칸은 가이드의 예시 문구를 그대로
    두면 안 되므로 비운다. Jira 매크로는 키만 갈아끼운다.
    Jira 매크로는 넘겨받은 이슈 키 개수만큼 복제한다.
    반환값의 두 번째는 JSON 에 값이 없어서 비워둔 항목 목록이다.
    """
    root = parse_storage(template)
    missing = []

    for tr in root.iter("tr"):
        th, td = tr.find("th"), tr.find("td")
        if th is None or td is None:
            continue
        name = "".join(th.itertext()).strip()
        if not name:
            continue
        spec = fields.get(name)
        if spec is None:
            missing.append(name)
            filled = "<p />"
        else:
            filled = cell(spec.get("value"), spec.get("type", "text"))
        for child in list(td):
            td.remove(child)
        td.text = None
        for node in parse_storage(filled):
            td.append(node)

    body = "".join(ET.tostring(child, encoding="unicode") for child in root)
    body = re.sub(r"\sxmlns:(ac|ri)=\"[^\"]*\"", "", body)

    # 가이드 맨 위의 제목 작성 안내 문단은 실제 문서에 들어가면 안 된다.
    first_heading = re.search(r"<h[1-6][ >]", body)
    if first_heading:
        body = body[first_heading.start():]

    return expand_jira(body, jira_keys), missing


MAX_SEQ = 50

MACRO_RE = re.compile(r'<ac:structured-macro\s+ac:name="jira".*?</ac:structured-macro>', re.S)
# 매크로를 품고 있는 문단째로 잡는다. 문단 안에서 매크로만 바꾸면 이슈들이
# 한 줄에 나란히 붙어버려서, 번호 목록으로 갈아끼우려면 문단을 통째로 걷어내야 한다.
MACRO_PARA_RE = re.compile(
    r'<p[^>]*>(?:(?!</p>).)*?<ac:structured-macro\s+ac:name="jira".*?</ac:structured-macro>'
    r'(?:(?!</p>).)*?</p>', re.S)


def expand_jira(body: str, keys: list[str]) -> str:
    """가이드의 Jira 매크로 하나를 이슈 개수만큼 복제해 번호 목록으로 만든다.

    키가 없으면 매크로를 지운다. 가이드의 예시 키(STITY-7)가 그대로 남으면
    엉뚱한 이슈가 문서에 붙는다.
    """
    m = MACRO_RE.search(body)
    if not m:
        return body

    target_re = MACRO_PARA_RE if MACRO_PARA_RE.search(body) else MACRO_RE
    if not keys:
        return target_re.sub(lambda _: "", body, count=1)

    template = m.group(0)
    items = []
    for key in keys:
        one = re.sub(r'(<ac:parameter ac:name="key">)[^<]*(</ac:parameter>)',
                     lambda mm: mm.group(1) + esc(key) + mm.group(2), template)
        one = re.sub(r'ac:(local-id|macro-id)="[^"]*"',
                     lambda mm: f'ac:{mm.group(1)}="{uuid.uuid4()}"', one)
        items.append(f"<li><p>{one}</p></li>")
    listed = '<ol start="1">' + "".join(items) + "</ol>"
    return target_re.sub(lambda _: listed, body, count=1)


def build_title(d: dict, kind_ko: str, seq: int | None) -> str:
    """제목을 만든다. 같은 태그가 이미 있으면 마지막 태그 옆에 [2], [3] 을 붙인다.

    가이드의 제목 형식이 종류마다 다르다. 보고는 세분화된 업무 하나마다 쓰는 것이라
    세분화 업무명이 한 칸 더 붙는다.
    """
    tail = f"[{seq}]" if seq else ""
    parts = [f'{d["round"]}차', d["major"], d["minor"]]
    if d.get("kind") == "report":
        parts.append(d["task"])
    return "".join(f"[{p}]" for p in parts) + f"{tail} {kind_ko} 문서"


def resolve_title(cf: "Confluence", d: dict, kind_ko: str, autonumber: bool) -> str:
    """쓸 수 있는 제목을 고른다.

    JSON 에 seq 를 주면 그 번호를 그대로 쓴다. 안 주면 같은 제목이 이미 있는지 보고
    비어 있는 다음 번호를 찾는다. 번호가 붙는 자리는 소범주 태그 바로 옆이다.
    """
    seq = d.get("seq")
    if seq:
        return build_title(d, kind_ko, int(seq))

    title = build_title(d, kind_ko, None)
    if not autonumber or not cf.find_page(title):
        return title

    for n in range(2, MAX_SEQ + 1):
        candidate = build_title(d, kind_ko, n)
        if not cf.find_page(candidate):
            return candidate
    sys.exit(f"같은 태그의 문서가 {MAX_SEQ} 개를 넘었다. seq 로 직접 번호를 지정할 것.")


def slug(s: str) -> str:
    """Confluence 라벨은 공백을 못 쓴다."""
    return str(s).replace(" ", "")


def resolve_jira(spec, fields: dict, major: str, dry_run: bool) -> tuple[list[str], list[str]]:
    """문서에 붙일 Jira 이슈 키를 정한다.

    spec 이 문자열이면 기존 이슈 하나를 그대로 쓴다. 객체면 create_from 이 가리키는
    항목의 목록 하나하나를 이슈로 만들고, keys 로 준 기존 이슈를 뒤에 덧붙인다.
    같은 요약의 이슈가 이미 있으면 새로 만들지 않고 재사용한다.
    """
    if not spec:
        return [], []
    if isinstance(spec, str):
        return [spec], []

    keys = list(spec.get("keys", []))
    source = spec.get("create_from")
    if not source:
        return keys, []

    items = (fields.get(source) or {}).get("value") or []
    if not items:
        sys.exit(f"jira.create_from 이 가리키는 '{source}' 항목에 값이 없다.")

    project = spec.get("project") or CONFIG["jira"]["default_project"]
    # 대범주와 Jira 업무 유형을 같은 값으로 맞춘다. 아직 그 유형이 프로젝트에 없으면
    # 이슈 생성이 400 으로 막히므로, 없을 때 쓸 유형을 config 에 따로 둔다.
    issuetype = (spec.get("issuetype")
                 or CONFIG["jira"]["type_by_major"].get(major)
                 or (fields.get("업무 카테고리") or {}).get("value")
                 or CONFIG["jira"]["fallback_issuetype"])
    note = spec.get("description", "")

    default_who = spec.get("assignee")
    by_item = spec.get("assignee_by_item", {})

    made, new_keys = [], []
    jira = None if dry_run else Jira(*load_env())
    cache: dict[str, str] = {}
    for item in items:
        who = by_item.get(item, default_who)
        label = f" -> {who}" if who else " -> (담당자 없음)"
        if dry_run:
            made.append(f"[{project}/{issuetype}] {item}{label}")
            continue
        existing = jira.find_by_summary(project, item)
        if existing:
            new_keys.append(existing)
            made.append(f"{existing}  (이미 있어서 재사용, 담당자는 건드리지 않음) {item}")
            continue
        if who and who not in cache:
            cache[who] = jira.account_id(project, who)
        key = jira.create(project, issuetype, item, note,
                          cache.get(who) if who else None)  # 유형이 없으면 여기서 멈춘다
        new_keys.append(key)
        made.append(f"{key}  (새로 만듦){label} {item}")
    return new_keys + keys, made


def main() -> None:
    ap = argparse.ArgumentParser(
        description="업무 계획/보고 문서를 Confluence 에 만든다. 형식은 가이드 문서에서 그때그때 읽는다.")
    ap.add_argument("--json", help="문서 내용을 담은 JSON 파일")
    ap.add_argument("--show-format", choices=sorted(CONFIG["guide"]),
                    help="가이드에서 현재 형식(항목 이름)만 읽어 출력. JSON 을 짜기 전에 먼저 볼 것")
    ap.add_argument("--list-users", action="store_true",
                    help="Jira 이슈 담당자로 지정할 수 있는 사람 목록을 출력")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 제목·라벨·본문만 출력")
    ap.add_argument("--no-autonumber", action="store_true",
                    help="같은 제목이 있어도 번호를 붙이지 않고 그냥 멈춘다")
    args = ap.parse_args()

    if args.list_users:
        project = CONFIG["jira"]["default_project"]
        print(f"{project} 이슈에 배정 가능한 사람 (이 이름을 그대로 assignee 에 쓸 것):")
        for u in Jira(*load_env()).assignable(project):
            print(f"  - {u['displayName']}")
        return

    cf = Confluence(*load_env())

    if args.show_format:
        guide_id = CONFIG["guide"][args.show_format]
        root = parse_storage(cf.get_storage(guide_id))
        print(f"가이드 페이지: {guide_id}")
        print("절 구성 :", " / ".join(headings(root)) or "(없음)")
        print("채워야 하는 항목 (이 이름을 JSON 의 fields 키로 그대로 쓸 것):")
        for name in row_labels(root):
            print(f"  - {name}")
        return

    if not args.json:
        sys.exit("--json 또는 --show-format 중 하나가 필요하다.")

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    kind = d.get("kind")
    if kind not in ("plan", "report"):
        sys.exit('kind 는 "plan" 또는 "report" 여야 한다.')
    for field in ("round", "major", "minor"):
        if not d.get(field):
            sys.exit(f"필수 항목 누락: {field}")

    kind_ko = "계획" if kind == "plan" else "보고"
    if kind == "report" and not d.get("task"):
        sys.exit("보고 문서에는 세분화 업무명(task)이 필요하다.\n"
                 "가이드 제목 형식: [n차][대범주][소범주][세분화 업무명] 보고 문서")
    title = resolve_title(cf, d, kind_ko, autonumber=not args.no_autonumber)
    labels = [kind_ko, f'{d["round"]}차', slug(d["major"]), slug(d["minor"])]
    folder_title = f'{d["round"]}차 업무 분담'

    template = cf.get_storage(CONFIG["guide"][kind])
    fields = d.get("fields", {})
    jira_keys, planned = resolve_jira(d.get("jira"), fields, d["major"], args.dry_run)
    if planned:
        head = "만들 이슈 (dry-run 이라 아직 안 만듦)" if args.dry_run else "만든 이슈"
        print(f"{head}:")
        for line in planned:
            print(f"  {line}")
    body, missing = render(template, fields, jira_keys)

    unused = [k for k in d.get("fields", {}) if k not in row_labels(parse_storage(template))]
    if unused:
        print(f"경고: 가이드에 없는 항목이라 무시됨 — {', '.join(unused)}", file=sys.stderr)
    if missing:
        print(f"경고: 값이 없어 비워둔 항목 — {', '.join(missing)}", file=sys.stderr)

    if args.dry_run:
        base = build_title(d, kind_ko, None)
        if title != base:
            print(f"참고  : '{base}' 가 이미 있어 번호를 붙였다")
        print(f"제목  : {title}")
        print(f"폴더  : {folder_title} (없으면 생성)")
        print(f"라벨  : {labels}")
        print(f"본문  : {len(body)}자")
        print(body)
        return

    if cf.find_page(title):
        sys.exit(f"같은 제목의 페이지가 이미 있다: {title}\n"
                 "덮어쓰지 않는다. 제목을 바꾸거나 기존 페이지를 직접 수정할 것.")

    folder_id = cf.find_folder(folder_title) or cf.create_folder(
        CONFIG["docs_folder_id"], folder_title)
    page = cf.create_page(folder_id, title, body)
    cf.add_labels(page["id"], labels)

    print(f"만들어짐: {title}")
    print(f"폴더    : {folder_title} ({folder_id})")
    print(f"라벨    : {', '.join(labels)}")
    print(f"주소    : {CONFIG['base_url'] + page['_links']['webui']}")


if __name__ == "__main__":
    main()
