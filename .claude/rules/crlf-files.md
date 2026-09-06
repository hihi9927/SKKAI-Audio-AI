# CRLF 파일은 줄바꿈을 보존해서 편집한다

`Qwen3-ASR/examples/streaming_websocket_server.py` 는 저장소에 **CRLF 줄바꿈**으로
들어가 있다. 지금까지 확인된 CRLF 파일은 이것 하나다.

## 왜

파이썬으로 `open(p).read()` 한 뒤 `open(p,'w').write(s)` 하면 universal newlines 가
CRLF 를 LF 로 바꿔 저장한다. **한 줄만 고쳐도 파일 전체가 바뀐 것으로 잡힌다** —
실측으로 3137줄 삭제 / 3152줄 추가짜리 diff 가 났다. 이렇게 되면 리뷰에서 무엇이
진짜 변경인지 보이지 않고, 같은 파일을 만지는 다른 작업과 충돌한다.

이 파일은 프로덕션 WebSocket 서버라 손댈 일이 잦다는 점이 문제를 키운다.

## 지킬 것

- 스크립트로 편집할 때는 `open(p, newline='')` 으로 **읽고 쓴다**. 둘 다 해야 한다.
- 여러 줄짜리 문자열을 찾아 바꿀 때는 패턴의 `\n` 도 `\r\n` 으로 바꿔야 매칭된다.
  안 그러면 "찾는 문자열이 없다" 로 조용히 실패한다.
- 편집 후 확인: `python -c "d=open(파일,'rb').read(); print(d.count(b'\r\n'), d.count(b'\n'))"`
  두 값이 같아야 한다.
- 이미 LF 로 바꿔버렸으면 바이너리로 열어 `b.replace(b'\n', b'\r\n')` 로 되돌린다.
  `git diff --ignore-cr-at-eol` 로 실제 내용 차이만 먼저 보면 판단이 빠르다.
- Edit 도구는 이 문제가 없다. 한두 군데 고칠 때는 그쪽이 안전하다.
