"""실행 중인 평가 서버의 설정이 기대값과 일치하는지 확인하는 범용 pre-flight 검증 스크립트.

paper_result 아래 ASR/AST, mode2/3/4 등 모든 실험 launch 스크립트가 공용으로 사용한다.
서버의 hello.serverConfig 필드를 이 스크립트가 늘어날 때마다 --expect key=value 를
추가하기만 하면 되므로, 새 모드/새 데이터셋이 생겨도 이 파일은 수정하지 않는다.

사용 예:
    python check_server_config.py --host localhost --port 8765 \\
        --expect model=Qwen/Qwen3-ASR-1.7B \\
        --expect always_commit=true \\
        --expect enable_dot_commit=false
"""
import argparse
import asyncio
import json
import sys

import websockets


async def fetch_server_config(ws_url: str, timeout: float = 8.0):
    async with websockets.connect(ws_url, ping_interval=None, open_timeout=10) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(msg, str):
            data = json.loads(msg)
            if data.get("type") == "hello":
                return data.get("serverConfig") or {}
    raise RuntimeError("서버로부터 hello.serverConfig를 받지 못했습니다.")


def _coerce(expected_str: str, actual_value):
    """actual_value의 타입에 맞춰 expected 문자열을 캐스팅한다."""
    if isinstance(actual_value, bool):
        return expected_str.strip().lower() in ("true", "1", "yes")
    if isinstance(actual_value, int) and not isinstance(actual_value, bool):
        return int(expected_str)
    if isinstance(actual_value, float):
        return float(expected_str)
    return expected_str


def parse_expect_args(pairs: list[str]) -> dict[str, str]:
    expected = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--expect 값은 key=value 형식이어야 합니다: {pair!r}")
        key, value = pair.split("=", 1)
        expected[key.strip()] = value.strip()
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--expect", action="append", default=[], metavar="KEY=VALUE",
        help="serverConfig에서 확인할 key=value. 여러 번 지정 가능.",
    )
    args = parser.parse_args()

    if not args.expect:
        print("검증할 --expect 항목이 없습니다. 아무 것도 확인하지 않고 통과합니다.")
        return

    expected = parse_expect_args(args.expect)
    ws_url = f"ws://{args.host}:{args.port}"

    try:
        actual = asyncio.run(fetch_server_config(ws_url))
    except Exception as e:
        print(f"[FAIL] 서버 설정을 가져오지 못했습니다 ({ws_url}): {e}")
        sys.exit(1)

    mismatches = []
    for key, expected_str in expected.items():
        if key not in actual:
            mismatches.append(f"  - {key}: 서버 serverConfig에 이 필드가 없음 (실제 필드: {sorted(actual)})")
            continue
        actual_value = actual[key]
        expected_value = _coerce(expected_str, actual_value)
        if actual_value != expected_value:
            mismatches.append(f"  - {key}: 기대값={expected_value!r} / 실제값={actual_value!r}")

    if mismatches:
        print(f"[FAIL] 서버 설정이 기대값과 다릅니다 ({ws_url}):")
        print("\n".join(mismatches))
        sys.exit(1)

    print(f"[OK] 서버 설정 검증 통과 ({ws_url}): {expected}")


if __name__ == "__main__":
    main()
