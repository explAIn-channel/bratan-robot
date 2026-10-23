"""On the phone: step prompt latency when only the moment changes between calls (prefix cache)."""
import json, shutil, sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from pathlib import Path
from bratan.common.config import load, secret
from bratan.mind.brain import Mind
from bratan.mind.llm import LLM
from bratan.mind.memory import Archive, MindState
from bratan.mind.world import World
work = Path('/tmp/promptcache'); work.mkdir(exist_ok=True)
for n in ('mind.json', 'memory.sqlite'):
    shutil.copy(f'/opt/robotbrain/bratan/state/{n}', work / n)
c = load(None)
st, ar = MindState(work / 'mind.json'), Archive(work / 'memory.sqlite')
w = World(json.load(open(c['rooms_json'])), 'Таффа', ar)
w.body = {'docked': True, 'voltage': 16.4, 'pose': [0, 0, 0]}
llm = LLM(c['llm_base_url'], secret(c['llm_key']), c['llm_fast'], fallback=c.get('llm_smart'), providers=c.get('llm_providers', ()))
m = Mind(st, w, ar, llm, c, lambda *a, **k: None)
p0 = m.prompt(time.time(), 'heartbeat', False)
print('chars', len(p0), '| first 300:', p0[:120].replace('\n', ' '))
for i, trig in enumerate(['heartbeat', 'person_appeared', 'heartbeat', 'action_done', 'heartbeat']):
    w.event('test', f'событие {i}')
    p = m.prompt(time.time(), trig, False)
    common = next((k for k in range(min(len(p), len(p0))) if p[k] != p0[k]), min(len(p), len(p0)))
    t = time.monotonic()
    llm.chat('Ты — внутренний голос живого робота. Отвечай ровно одним JSON-объектом.', p, max_tokens=500,
             temperature=.85, json_mode=True, timeout=20.)
    print(f'{trig:16s} call {time.monotonic() - t:.2f}s first {llm.stats["last_first_token_s"]}s  same prefix {common} chars')
    time.sleep(3)
