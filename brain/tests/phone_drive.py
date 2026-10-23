"""On the phone: run motion skills one after another through the bus and log what happened."""
import json, sys, time, threading
sys.path.insert(0, '/opt/robotbrain/bratan/app')
from bratan.common import bus

state = {}
events = []
lock = threading.Lock()


def on(topic, data):
    with lock:
        if topic == 'body.state':
            state.update(data)
            samples.append((time.time(), data.get('pose'), data.get('localized'), data.get('blocked'), data.get('twist'),
                            (data.get('skill') or {}).get('door_phase'), (data.get('skill') or {}).get('doors_crossed')))
        else:
            events.append((time.time(), topic, data))


samples = []
client = bus.Client('drive-test', ('body.state', 'body.event'), on).start()
client.wait_connected(5)
time.sleep(1)
plan = json.loads(sys.argv[1])
for n, (skill, args, timeout) in enumerate(plan):
    sid = f't{n}-{skill}'
    started = time.time()
    with lock:
        mark = len(events)
        smark = len(samples)
    client.publish('body.do', {'skill': skill, 'args': args, 'id': sid, 'origin': 'owner'})
    result = None
    while time.time() - started < timeout:
        time.sleep(.2)
        with lock:
            for _, topic, e in events[mark:]:
                if e.get('kind') == 'skill' and e.get('id') == sid and e.get('status') in ('done', 'failed', 'cancelled'):
                    result = e
        if result:
            break
    if result is None:
        client.publish('body.stop', {'reason': 'test_timeout'})
    with lock:
        window = samples[smark:]
        evs = [(round(t - started, 1), e.get('kind'), e.get('skill', ''), e.get('status', ''), e.get('reason', ''))
               for t, _, e in events[mark:] if e.get('kind') != 'skill' or e.get('id', '').startswith(sid)]
    lost = sum(1 for s in window if not s[2])
    blocked = {}
    for s in window:
        if s[3]:
            blocked[s[3]] = blocked.get(s[3], 0) + 1
    speeds = [abs(s[4][0]) for s in window if s[4]]
    phases = []
    for s_ in window:
        if s_[5] and (not phases or phases[-1] != s_[5]):
            phases.append(s_[5])
    crossed = max([s_[6] or 0 for s_ in window] or [0])
    print(json.dumps({'skill': skill, 'args': args, 'status': (result or {}).get('status', 'timeout'),
                      'reason': (result or {}).get('reason'), 'seconds': round(time.time() - started, 1),
                      'end_pose': state.get('pose'), 'room': state.get('room'), 'localized': state.get('localized'),
                      'unlocalized_samples': lost, 'samples': len(window), 'blocked': blocked,
                      'max_speed': round(max(speeds), 3) if speeds else 0,
                      'loop_max_ms': state.get('loop_max_ms'), 'door_phases': phases, 'doors_crossed': crossed,
                      'events': evs[:12]}, ensure_ascii=False), flush=True)
    if (result or {}).get('status') != 'done' and skill != 'look_around':
        break
    time.sleep(1.5)
