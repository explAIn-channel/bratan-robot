"""Supervisor: the bus hub, the mind and life in-process, body/eyes/voice as child processes.

Usage: python -m bratan.main [--config path] [--no-eyes] [--no-voice] [--no-body]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

from .common import bus as busmod
from .common.config import HOME, ROOT, STATE, load, pin, write_json

CHILDREN = ('body', 'eyes', 'voice')
LOG_LIMIT = 4 << 20


class Child:
    def __init__(self, name, config_path):
        self.name, self.config_path = name, config_path
        self.process = None
        self.started = 0.
        self.failures = 0
        self.log = HOME / 'logs' / f'{name}.log'

    async def run(self, stopping):
        """Keep the child alive; back off when it keeps crashing."""
        while not stopping.is_set():
            self.log.parent.mkdir(parents=True, exist_ok=True)
            if self.log.exists() and self.log.stat().st_size > LOG_LIMIT:
                os.replace(self.log, self.log.with_suffix('.log.1'))
            handle = open(self.log, 'ab', buffering=0)
            args = [sys.executable, '-u', '-m', f'bratan.{self.name}.main']
            if self.config_path:
                args += ['--config', str(self.config_path)]
            self.process = await asyncio.create_subprocess_exec(
                *args, cwd=str(ROOT), stdout=handle, stderr=handle, env={**os.environ, 'PYTHONPATH': str(ROOT)})
            handle.close()
            self.started = time.monotonic()
            await self.process.wait()
            if stopping.is_set():
                break
            self.failures = self.failures + 1 if time.monotonic() - self.started < 30 else 0
            try:
                await asyncio.wait_for(stopping.wait(), min(30., .5 * 2 ** min(self.failures, 6)))
            except asyncio.TimeoutError:
                pass

    def stop(self):
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()


async def amain(args):
    config = load(args.config)
    pin('mind', config)
    STATE.mkdir(parents=True, exist_ok=True)
    hub = busmod.Hub()
    await hub.start()
    loop = asyncio.get_running_loop()
    from .mind.life import Life
    life = Life(hub, loop, config)
    life.start()

    def control(topic, data):
        if topic == 'mind.autonomy':
            config['autonomy'] = bool(data.get('enabled'))
            override = HOME / 'config.json'
            current = {}
            if override.exists():
                import json
                current = json.loads(override.read_text(encoding='utf-8'))
            current['autonomy'] = config['autonomy']
            write_json(override, current)
            if not config['autonomy']:
                hub.publish('body.stop', {'reason': 'autonomy_off'})
    hub.subscribe(('mind.autonomy',), control)

    children = [Child(name, args.config) for name in CHILDREN if not getattr(args, f'no_{name}')]
    stopping = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopping.set)
    tasks = [loop.create_task(child.run(stopping)) for child in children]
    await stopping.wait()
    for child in children:
        child.stop()
    life.close()
    await asyncio.wait(tasks, timeout=5.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    for name in CHILDREN:
        parser.add_argument(f'--no-{name}', action='store_true')
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == '__main__':
    main()
