import time
from tests.run_sim import make
body, link, events = make((9.48, -1.08, -1.0))
time.sleep(3)
body.on_message('body.do', {'skill': 'go', 'args': {'xy': [6.02, -2.88], 'tolerance': .5}, 'id': 'hall'})
start = time.monotonic(); last = None
while time.monotonic() - start < 100:
    s = body.skill
    li = getattr(body, '_last_intent', None)
    line = (s.stage if s else None, s.info.get('door', {}).get('phase') if s else None, body.last_reason,
            (round(li.v, 2), round(li.w, 2)) if li else None)
    if line != last:
        print(f'{time.monotonic()-start:5.1f}', line, tuple(round(v, 2) for v in link.true_pose), flush=True)
        last = line
    if any(k == 'skill' and d.get('id') == 'hall' and d.get('status') in ('done', 'failed') for _, k, d in events):
        print([d for _, k, d in events if k == 'skill'][-1]); break
    time.sleep(.1)
body.stopping.set()
