import time
from tests.run_sim import make
from bratan.common.geometry import relative
body, link, events = make((9.48, -1.08, -1.0))
time.sleep(3)
body.on_message('body.do', {'skill': 'go', 'args': {'xy': [6.02, -2.88], 'tolerance': .5}, 'id': 'hall'})
last = None
start = time.monotonic()
while time.monotonic() - start < 90:
    s = body.skill
    line = None
    if s is not None:
        door = s.info.get('door')
        line = (s.stage, door and door.get('phase'), body.last_reason)
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    if line != last or bumps:
        p = link.true_pose
        extra = ''
        if s is not None and s.info.get('door'):
            d = s.info['door']
            extra = f" door c={tuple(round(v,2) for v in d['center'])} ax={d['axis']:.2f} w={d.get('width')} rel={tuple(round(v,2) for v in relative((*d['center'], d['axis']), p))}"
        print(f'{time.monotonic()-start:5.1f}', line, tuple(round(v, 2) for v in p), 'bumps', bumps, extra, flush=True)
        last = line
        if bumps > 3:
            break
    if any(k == 'skill' and d.get('id') == 'hall' and d.get('status') in ('done', 'failed') for _, k, d in events):
        print('result', [d for _, k, d in events if k == 'skill'][-1])
        break
    time.sleep(.1)
body.stopping.set()
