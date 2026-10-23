"""On the phone: run a reflection and a night dream against COPIES of the robot's memory (live state untouched)."""
import json, shutil, sys, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/app')
from pathlib import Path
from bratan.common.config import load, secret
from bratan.mind.brain import Mind
from bratan.mind.llm import LLM
from bratan.mind.memory import Archive, MindState
from bratan.mind.world import World

work = Path('/tmp/dreamtest')
work.mkdir(exist_ok=True)
for name in ('mind.json', 'memory.sqlite'):
    shutil.copy(f'/opt/robotbrain/bratan/state/{name}', work / name)
config = load(None)
state = MindState(work / 'mind.json')
archive = Archive(work / 'memory.sqlite')
world = World(json.load(open(config['rooms_json'])), 'Таффа', archive)
world.body = {'docked': True, 'voltage': 16.4, 'pose': [0, 0, 0]}
llm = LLM(config['llm_base_url'], secret(config['llm_key']), config['llm_fast'])
mind = Mind(state, world, archive, llm, config, lambda *a, **k: None)
t = time.monotonic()
print('REFLECT:', mind.reflect(time.time()), f'({time.monotonic() - t:.1f}s)')
print('SELF:', state.get('self'))
t = time.monotonic()
record = mind.dream(time.time())
print(f'DREAM ({time.monotonic() - t:.1f}s, model {config.get("llm_smart")}):')
print(json.dumps(record, ensure_ascii=False, indent=1))
print('PROMPT LINES:', mind.dream_lines(time.time()))
