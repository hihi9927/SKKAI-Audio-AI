#!/usr/bin/env python3
"""
Qwen3 server protocol integration tests (real server, no mocking).

Usage examples:
1) External server mode (default)
   - Start server separately, then:
     python Qwen3-ASR/tests/test_streaming_architecture.py

2) Auto-server mode
   - set QWEN3_TEST_AUTO_SERVER=1
   - optionally set QWEN3_SERVER_SCRIPT / QWEN3_SERVER_MODEL / QWEN3_SERVER_URL
   - then run this file.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import unittest

import websockets


DEFAULT_SERVER_URL = os.environ.get('QWEN3_SERVER_URL', 'ws://localhost:8765')
DEFAULT_SERVER_SCRIPT = os.environ.get('QWEN3_SERVER_SCRIPT', 'Qwen3-ASR/examples/streaming_websocket_server.py')
DEFAULT_SERVER_MODEL = os.environ.get('QWEN3_SERVER_MODEL', 'Qwen/Qwen3-ASR-1.7B')
AUTO_SERVER = os.environ.get('QWEN3_TEST_AUTO_SERVER', '0') == '1'


def _parse_ws_url(url):
    # expected format ws://host:port
    if not url.startswith('ws://'):
        raise ValueError(f'Only ws:// URLs are supported: {url}')
    host_port = url[len('ws://'):]
    if '/' in host_port:
        host_port = host_port.split('/', 1)[0]
    host, port_str = host_port.split(':', 1)
    return host, int(port_str)


class _ServerProcess:
    def __init__(self, script, url, model):
        self.script = script
        self.url = url
        self.model = model
        self.proc = None

    def start(self):
        host, port = _parse_ws_url(self.url)
        cmd = [
            sys.executable,
            self.script,
            '--host', host,
            '--port', str(port),
            '--model', self.model,
            '--no-idle-shutdown',
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        if not self.wait_ready(180):
            self.stop()
            raise RuntimeError('Auto-started server failed readiness probe')

    def wait_ready(self, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                return False

            async def _probe():
                try:
                    async with websockets.connect(DEFAULT_SERVER_URL, ping_interval=None, open_timeout=3) as ws:
                        msg = await asyncio.wait_for(ws.recv(), timeout=4)
                        if isinstance(msg, str):
                            return json.loads(msg).get('type') == 'hello'
                except Exception:
                    return False
                return False

            if asyncio.run(_probe()):
                return True
            time.sleep(1)
        return False

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None


async def _recv_type(ws, expected, timeout=10.0, ignore=None):
    if isinstance(expected, str):
        expected = {expected}
    else:
        expected = set(expected)

    ignore = set(ignore or [])
    end_at = time.time() + timeout

    while time.time() < end_at:
        msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, end_at - time.time()))
        if not isinstance(msg, str):
            continue
        data = json.loads(msg)
        msg_type = data.get('type', '')
        if msg_type in ignore:
            continue
        if msg_type in expected:
            return data

    raise TimeoutError(f'Timeout waiting for one of {sorted(expected)}')


class Qwen3ProtocolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    server = None

    @classmethod
    def setUpClass(cls):
        if AUTO_SERVER:
            if not os.path.exists(DEFAULT_SERVER_SCRIPT):
                raise unittest.SkipTest(f'Auto-server script not found: {DEFAULT_SERVER_SCRIPT}')
            cls.server = _ServerProcess(DEFAULT_SERVER_SCRIPT, DEFAULT_SERVER_URL, DEFAULT_SERVER_MODEL)
            try:
                cls.server.start()
            except Exception as e:
                raise unittest.SkipTest(str(e))
        else:
            probe = _ServerProcess(DEFAULT_SERVER_SCRIPT, DEFAULT_SERVER_URL, DEFAULT_SERVER_MODEL)
            if not probe.wait_ready(timeout=8):
                raise unittest.SkipTest(
                    f'Qwen3 server is not reachable at {DEFAULT_SERVER_URL}. '
                    'Run server first or set QWEN3_TEST_AUTO_SERVER=1.'
                )

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.stop()

    async def test_start_ready_finish_restart_stop(self):
        async with websockets.connect(DEFAULT_SERVER_URL, ping_interval=None, ping_timeout=None) as ws:
            await _recv_type(ws, 'hello', timeout=8)

            await ws.send(json.dumps({'type': 'start', 'lang': 'auto', 'targetLang': ''}))
            await _recv_type(ws, 'ready', timeout=25, ignore={'partial', 'final'})

            await ws.send(b'\x00\x00' * 3200)  # 200 ms silence
            await ws.send(json.dumps({'type': 'finish'}))

            await ws.send(json.dumps({'type': 'start', 'lang': 'auto', 'targetLang': ''}))
            await _recv_type(ws, 'ready', timeout=25, ignore={'partial', 'final'})

            await ws.send(json.dumps({'type': 'stop'}))

if __name__ == '__main__':
    unittest.main(verbosity=2)
