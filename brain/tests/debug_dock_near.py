"""Docking from 1.3 m in front of the dock inside the bedroom (07.10 22:24: circled for 4 min). Records stages."""
import sys
import time

from bratan.common.geometry import relative
from tests.run_sim import DOCK, make

start = tuple(float(v) for v in sys.argv[1:4]) if len(sys.argv) > 3 else (1.21, .34, .17)
body, link, events = make(start)
time.sleep(3)
body.on_message('body.do', {'skill': 'dock', 'args': {}, 'id': 'dock'})
time.sleep(.5)
t0 = time.monotonic()
last = None
while time.monotonic() - t0 < 150:
    skill = body.skill
    if skill is None:
        break
    nav = skill.info.get('nav')
    stage = (skill.stage, nav.stage if nav is not None and skill.stage == 'route' else None)
    if stage != last:
        pose = link.true_pose
        print(f'{time.monotonic() - t0:6.1f}s {stage} true {tuple(round(v, 2) for v in pose)} '
              f'blocked {body.last_reason} extra {skill.info.get("extra")}', flush=True)
        last = stage
    time.sleep(.05)
print('result', [(k, d.get('status'), d.get('reason')) for _, k, d in events if k == 'skill'][-2:],
      'charging', link.charging, f'{time.monotonic() - t0:.0f}s')
body.stopping.set()
