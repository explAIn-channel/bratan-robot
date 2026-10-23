"""Phone: print the robot's life as it happens (mind thoughts, actions, voice, body events, errors)."""
import json, sys, time
sys.path.insert(0, '/opt/robotbrain/bratan/app')
from bratan.common import bus


def show(topic, d):
    t = time.strftime('%H:%M:%S')
    if topic == 'mind.step':
        if d.get('trigger') == 'conversation':
            line = f"👂 «{d.get('heard','')[:120]}» → «{d.get('reply','')[:160]}» {d.get('latency_s')}s" + (f" ⚙{d.get('do')}" if d.get('do') else '')
        else:
            line = f"🧠[{d.get('trigger')} {d.get('latency_s')}s] {str(d.get('thought'))[:170]}"
            if d.get('say'): line += f" | 🗣{d['say']}"
            if d.get('do'): line += f" | ⚙{d['do']}"
            if d.get('effect'): line += f" → {str(d['effect'])[:100]}"
    elif topic == 'body.event':
        if d.get('kind') == 'skill' and d.get('status') == 'started':
            return
        line = f"   body {d.get('kind')} {d.get('skill','')} {d.get('status','')} {d.get('reason','')}"
    elif topic == 'voice.event':
        if d.get('kind') in ('heard', 'said'):
            return
        line = f"   voice {d.get('kind')} " + json.dumps({k: v for k, v in d.items() if k not in ('kind', 't')}, ensure_ascii=False)[:160]
    elif topic in ('eyes.error',):
        line = f"   ⚠ {topic} {d}"
    else:
        line = f"   {topic} {str(d)[:160]}"
    print(t, line, flush=True)


bus.Client('watch', ('mind.step', 'body.event', 'voice.event', 'eyes.error', 'voice.ask'), show).start()
end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 1e9)
while time.time() < end:
    time.sleep(1)
