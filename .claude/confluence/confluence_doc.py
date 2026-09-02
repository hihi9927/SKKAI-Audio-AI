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


def render(template: str, fields: dict, jira: str | None) -> tuple[str, list[str]]:
    """가이드 본문을 틀로 삼아 값을 끼워 넣는다.

    표의 왼쪽 항목 이름으로 짝을 맞춘다. 값이 없는 칸은 가이드의 예시 문구를 그대로
    두면 안 되므로 비운다. Jira 매크로는 키만 갈아끼운다.
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

    # Jira 매크로의 이슈 키 교체 (매크로 자체가 가이드에 있을 때만)
    if jira:
        body = re.sub(r'(<ac:parameter ac:name="key">)[^<]*(</ac:parameter>)',
                      lambda m: m.group(1) + esc(jira) + m.group(2), body)
    return body, missing


def slug(s: str) -> str:
    """Confluence 라벨은 공백을 못 쓴다."""
    return str(s).replace(" ", "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="업무 계획/보고 문서를 Confluence 에 만든다. 형식은 가이드 문서에서 그때그때 읽는다.")
    ap.add_argument("--json", help="문서 내용을 담은 JSON 파일")
    ap.add_argument("--show-format", choices=sorted(CONFIG["guide"]),
                    help="가이드에서 현재 형식(항목 이름)만 읽어 출력. JSON 을 짜기 전에 먼저 볼 것")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 제목·라벨·본문만 출력")
    args = ap.parse_args()

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
    title = f'[{d["round"]}차][{d["major"]}][{d["minor"]}] {kind_ko} 문서'
    labels = [kind_ko, f'{d["round"]}차', slug(d["major"]), slug(d["minor"])]
    folder_title = f'{d["round"]}차 업무 분담'

    template = cf.get_storage(CONFIG["guide"][kind])
    body, missing = render(template, d.get("fields", {}), d.get("jira"))

    unused = [k for k in d.get("fields", {}) if k not in row_labels(parse_storage(template))]
    if unused:
        print(f"경고: 가이드에 없는 항목이라 무시됨 — {', '.join(unused)}", file=sys.stderr)
    if missing:
        print(f"경고: 값이 없어 비워둔 항목 — {', '.join(missing)}", file=sys.stderr)

    if args.dry_run:
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
