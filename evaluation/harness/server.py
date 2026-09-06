"""평가 서버 기동과 프로토콜 핸드셰이크.

`ServerManager` 는 LibriSpeech 판을 기준으로 하되 KsponSpeech 판이 갖고 있던 개선 둘을
얹었다 — `--log-file` 이 있으면 stderr 를 같은 파일에 이어 붙이고(트레이스백 보존),
기동 타임아웃을 360초로 늘렸다(vLLM 워밍업이 180초를 넘는 경우가 있다).
"""
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)


class ServerManager:
    def __init__(self, server_script, host='localhost', port=8765, model='Qwen/Qwen3-ASR-1.7B'):
        self.server_script = server_script
        self.host = host
        self.port = port
        self.model = model
        self.process = None
        self._stderr_fh = None

    def start_server(self, additional_args=None, startup_timeout=360):
        if self.process is not None:
            self.stop_server()

        cmd = [
            sys.executable,
            str(self.server_script),
            '--host', self.host,
            '--port', str(self.port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]
        if additional_args:
            cmd.extend(additional_args)

        # --log-file 이 있으면 stderr 를 같은 파일에 이어 붙인다. 없으면 버린다.
        # stdout 은 항상 버린다 — vLLM 이 stdout 으로 쏟아내므로 PIPE 로 받아 두고 읽지
        # 않으면 파이프가 차서 서버가 멈춘다.
        log_file = next((cmd[i + 1] for i, a in enumerate(cmd) if a == '--log-file'), None)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            self._stderr_fh = open(log_file, 'a', encoding='utf-8')
            stderr_target = self._stderr_fh
        else:
            stderr_target = subprocess.DEVNULL

        logger.info('Starting Qwen3 server...')
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_target)

        if not self._wait_for_server_ready(timeout=startup_timeout):
            self.stop_server()
            return False

        logger.info('Server started (PID: %s)', self.process.pid)
        return True

    def _wait_for_server_ready(self, timeout=360):
        ws_url = f'ws://{self.host}:{self.port}'
        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                logger.error('Server exited unexpectedly while waiting for readiness.')
                return False

            async def _probe():
                try:
                    async with websockets.connect(ws_url, ping_interval=None, open_timeout=3) as ws:
                        msg = await asyncio.wait_for(ws.recv(), timeout=4)
                        if isinstance(msg, str):
                            return json.loads(msg).get('type') == 'hello'
                except Exception:
                    return False
                return False

            if asyncio.run(_probe()):
                return True
            time.sleep(1)
        logger.error('Server readiness timeout reached.')
        return False

    def stop_server(self):
        if self.process is None:
            return
        logger.info('Stopping server (PID: %s)', self.process.pid)
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.process = None
            if self._stderr_fh:
                self._stderr_fh.close()
                self._stderr_fh = None

async def recv_type(ws, expected_types, timeout=8.0, ignore_types=None):
    if isinstance(expected_types, str):
        expected_types = {expected_types}
    else:
        expected_types = set(expected_types)

    ignore_types = set(ignore_types or [])
    end_at = time.time() + timeout

    while time.time() < end_at:
        msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end_at - time.time()))
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        msg_type = data.get('type', '')

        if msg_type in ignore_types:
            continue
        if msg_type in expected_types:
            return data
    raise TimeoutError(f'Expected message types {sorted(expected_types)} not received in {timeout}s')

async def fetch_server_config(ws_url):
    """서버의 hello 메시지에서 serverConfig 필드를 읽어 반환."""
    try:
        async with websockets.connect(ws_url, ping_interval=None, open_timeout=10) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=8)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get('type') == 'hello':
                    return data.get('serverConfig')
    except Exception as e:
        logger.warning('서버 config 수집 실패: %s', e)
    return None

async def run_protocol_smoke(ws_url, lang='auto'):
    logger.info('Running protocol smoke test...')

    # start/ready/finish/restart/stop path
    async with websockets.connect(ws_url, ping_interval=None, ping_timeout=None) as ws:
        await recv_type(ws, 'hello', timeout=8)

        await ws.send(json.dumps({'type': 'start', 'lang': lang, 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)

        await ws.send((b'\x00\x00' * 1600))  # 100 ms silence (int16 mono)
        await ws.send(json.dumps({'type': 'finish'}))

        await ws.send(json.dumps({'type': 'start', 'lang': lang, 'targetLang': ''}))
        await recv_type(ws, 'ready', timeout=20)

        await ws.send(json.dumps({'type': 'stop'}))

    logger.info('Protocol smoke test passed.')
