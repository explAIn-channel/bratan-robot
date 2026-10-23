"""Mind + body + bus together offline: a scripted model decides, the simulated body drives.

python -m tests.smoke_mind
"""
import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ['BRATAN_STATE'] = str(TMP / 'state')
os.environ['BRATAN_SECRETS'] = str(TMP / 'secrets')
os.environ['BRATAN_BUS'] = 'tcp:127.0.0.1:18765'
(TMP / 'secrets').mkdir()
for name in ('openrouter-api-key', 'openai-live-key'):
    (TMP / 'secrets' / name).write_text('test')

from bratan.common import bus as busmod            # noqa: E402
from bratan.common.config import DEFAULTS          # noqa: E402
from bratan.mind import life as lifemod            # noqa: E402
from bratan.mind.llm import LLM                    # noqa: E402
from tests.run_sim import DATA, DOCK, make         # noqa: E402

SCRIPT = [
    {'thought': 'Зарядился. Загляну на кухню.', 'do': {'tool': 'go', 'args': {'to': 'Кухня'}}, 'face': 'curious'},
    {'thought': 'Я на кухне. Оглянусь.', 'do': {'tool': 'look_around', 'args': {}}},
    {'thought': 'Пора домой.', 'do': {'tool': 'rest', 'args': {}}},
]
calls = []


def fake_chat(self, system, user, **kw):
    calls.append(user[-300:])
    if 'просит или спрашивает' in user:
        return 'Привет! Всё хорошо.\n###\n{"thought": "поболтали", "do": null, "face": "happy"}'
    step = SCRIPT[min(len(calls) - 1, len(SCRIPT) - 1)] if len(calls) <= len(SCRIPT) else {'thought': 'отдыхаю', 'do': {'tool': 'wait', 'args': {'seconds': 600}}}
    return json.dumps(step, ensure_ascii=False)


LLM.chat = fake_chat
LLM.warm = lambda self: None
lifemod.ChargeLimit.poll = lambda self: {}


async def main():
    config = dict(DEFAULTS, rooms_json=str(DATA / 'rooms.json'), autonomy=True, quiet_start_hour=3, quiet_end_hour=4,
                  departure_voltage=15.5)
    hub = busmod.Hub(os.environ['BRATAN_BUS'])
    await hub.start()
    loop = asyncio.get_running_loop()
    life = lifemod.Life(hub, loop, config)
    life.face = lambda state: None
    life.start()
    seen = []
    hub.subscribe(('body.event',), lambda t, d: seen.append(d))
    body, link, events = make(DOCK)
    body.bus = busmod.Client('body', ('body.do', 'body.stop', 'eyes.seen'), body.on_message, os.environ['BRATAN_BUS'])
    body.bus.start()
    body.emit = lambda kind, **data: body.bus.publish('body.event', {'kind': kind, 't': time.time(), **data})
    life.world.event('departure_ready', 'зарядился')
    started = time.time()
    while time.time() - started < 260:
        await asyncio.sleep(1)
        done = [e for e in seen if e.get('kind') == 'skill' and e.get('status') in ('done', 'failed')]
        if len(done) >= 3 or (link.charging and len(done) >= 2):
            break
    print('model calls', len(calls))
    for e in seen:
        print(' ', round(e.get('t', 0) % 1000, 1), e.get('kind'), e.get('skill', ''), e.get('status', ''), e.get('reason', ''), e.get('id', ''))
    for t, k, d in reversed(life.archive.events(0, ('mind_step', 'refused'))):
        print('  ', round(t % 1000, 1), k, d.get('trigger'), d.get('effect') or d.get('text'))
    print('journal', [j['thought'] for j in life.state.get('journal', [])])
    print('room now', life.world.room, 'charging', link.charging)
    body.stopping.set()


asyncio.run(main())
