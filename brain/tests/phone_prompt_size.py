"""On the phone: size of the mind's step prompt by section (copies of live memory, nothing written)."""
import json, shutil, sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from pathlib import Path
from bratan.common.config import load, secret
from bratan.mind.brain import Mind
from bratan.mind.llm import LLM
from bratan.mind.memory import Archive, MindState
from bratan.mind.world import World
work = Path('/tmp/promptsize'); work.mkdir(exist_ok=True)
for n in ('mind.json', 'memory.sqlite'):
    shutil.copy(f'/opt/robotbrain/bratan/state/{n}', work / n)
c = load(None)
st, ar = MindState(work / 'mind.json'), Archive(work / 'memory.sqlite')
w = World(json.load(open(c['rooms_json'])), 'Таффа', ar)
w.body = {'docked': True, 'voltage': 16.4, 'pose': [0, 0, 0]}
llm = LLM(c['llm_base_url'], secret(c['llm_key']), c['llm_fast'], fallback=c.get('llm_smart'))
m = Mind(st, w, ar, llm, c, lambda *a, **k: None)
p = m.prompt(time.time(), 'heartbeat', False)
print('total chars', len(p))
sections = {}
cur = 'head'
for line in p.split('\n'):
    head = line.split(':', 1)[0] if line.isupper() or (line[:1].isupper() and line.split(':', 1)[0].isupper()) else None
    if head and len(head) < 60:
        cur = head
    sections[cur] = sections.get(cur, 0) + len(line) + 1
for k, v in sorted(sections.items(), key=lambda kv: -kv[1])[:14]:
    print(f'{v:6d} {k[:50]}')
for i in range(3):
    t = time.monotonic()
    llm.chat('Ты — внутренний голос живого робота. Отвечай ровно одним JSON-объектом.', p, max_tokens=500,
             temperature=.85, json_mode=True, timeout=20.)
    print('step call', round(time.monotonic() - t, 2), 'first token', llm.stats['last_first_token_s'])
