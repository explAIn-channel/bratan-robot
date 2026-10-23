"""End-to-end body scenarios in the simulator (real time): python -m tests.run_sim [scenario]."""
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault('BRATAN_STATE', tempfile.mkdtemp())
from bratan.body.main import Body          # noqa: E402
from bratan.common.config import DEFAULTS  # noqa: E402
from bratan.common.geometry import compose, relative  # noqa: E402
from tests.sim import SimLink              # noqa: E402

DATA = Path(__file__).parent / 'data'
DOCK = (-0.0887, -0.0557, 0.1896)


def carve_passages(link):
    """Open the owner's taught lines in the simulated flat.

    They are proven free in the real one (07.10 23:52: through the kitchen door +-1 cm), but the 5 cm map copy
    has wall cells 11 cm from the kitchen line where the real jamb is further away."""
    import numpy as np
    from bratan.common.config import read_json
    from bratan.common.geometry import footprint_clearance, inverse, transform
    grid = link.grid
    for passage in read_json(DEFAULTS['passages_json'])['passages']:
        x, y, h = passage['start']
        for d in np.arange(-float(passage.get('approach', .4)), float(passage['length']) + .01, .02):
            pose = (x + d * math.cos(h), y + d * math.sin(h), h)
            r0, c0 = grid.cell(pose)
            cells = [(r, c) for r in range(r0 - 7, r0 + 8) for c in range(c0 - 7, c0 + 8) if grid.inside((r, c))]
            local = transform(inverse(pose), np.array([grid.xy(cell) for cell in cells]))
            # a cell any part of which a perimeter sample could touch: within half its diagonal of the body
            for cell, gap in zip(cells, footprint_clearance(local)):
                if gap < grid.resolution * .72:
                    link.occupied[cell] = False


def make(start, **sim):
    config = dict(DEFAULTS, map_yaml=str(DATA / 'apartment-final-20261006-dock.yaml'),
                  rooms_json=str(DATA / 'rooms.json'), rooms_labels=str(DATA / 'rooms-labels.npy'),
                  dock_pose=list(DOCK))
    link = SimLink(config['map_yaml'], start, DOCK, **sim)
    carve_passages(link)
    body = Body(config, bus_address='tcp:127.0.0.1:9', link=link)
    events = []
    body.emit = lambda kind, **data: events.append((round(time.monotonic(), 2), kind, data))
    thread = threading.Thread(target=body.run, daemon=True)
    thread.start()
    return body, link, events


def do(body, skill, **args):
    body.on_message('body.do', {'skill': skill, 'args': args, 'id': skill})


def wait_skill(body, events, name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for _, kind, data in events:
            if kind == 'skill' and data.get('id') == name and data.get('status') in ('done', 'failed', 'cancelled'):
                return data
        time.sleep(.1)
    return None


def report(body, link, label):
    pose = relative((0, 0, 0), link.true_pose)
    est = compose(body.map_from_odom, link.odom)
    print(f'{label}: true {tuple(round(v, 3) for v in pose)} est {tuple(round(v, 3) for v in est)} '
          f'err {math.dist(pose[:2], est[:2]):.3f} m localized={body.localized} loop_max={body.loop_max*1000:.1f} ms', flush=True)


def scenario_trip_and_dock():
    body, link, events = make(compose(DOCK, (.0, 0, 0)))
    time.sleep(3)
    report(body, link, 'docked start')
    started = time.monotonic()
    do(body, 'go', xy=[8.02, 1.53])
    result = wait_skill(body, events, 'go', 150)
    print('go corridor:', result, f'{time.monotonic()-started:.1f}s')
    report(body, link, 'after trip')
    started = time.monotonic()
    do(body, 'dock')
    result = wait_skill(body, events, 'dock', 200)
    print('dock:', result, f'{time.monotonic()-started:.1f}s', 'charging', link.charging)
    report(body, link, 'after dock')
    print('events', [(k, d.get('status') or '', d.get('reason') or '') for _, k, d in events][-12:])
    body.stopping.set()
    return result and result['status'] == 'done' and link.charging


def scenario_wrong_dock():
    """19.10: the dock had been learned 0.4 m and 24° off the map (its panorama stored with that wrong pose); every
    undocking began «lost», a wrong place was «found», he drove to a wall. On the dock the map itself now checks the
    dock and learns it anew; the trip and the way home work."""
    from bratan.body import localize
    from bratan.body import scan as scanmod
    from bratan.common.config import STATE, write_json
    wrong = compose(DOCK, (-.25, .3, .42))
    probe = SimLink(str(DATA / 'apartment-final-20261006-dock.yaml'), DOCK, DOCK)
    ref = localize.DockReference()
    ref.record(wrong, scanmod.points(probe._raycast(DOCK)[0]))
    write_json(STATE / 'dock-reference.json', ref.as_dict())
    write_json(STATE / 'location.json', {'pose': list(wrong), 'dock': list(wrong), 'boot': None})
    body, link, events = make(DOCK)
    time.sleep(12)
    corrected = [d for _, k, d in events if k == 'dock_corrected']
    print('wrong_dock: dock now', tuple(round(v, 3) for v in body.dock), 'corrected', corrected,
          f'off the truth {math.dist(body.dock[:2], DOCK[:2]):.3f} m', flush=True)
    do(body, 'go', xy=[1.53, .38])
    trip = wait_skill(body, events, 'go', 90)
    report(body, link, 'after the trip')
    do(body, 'dock')
    home = wait_skill(body, events, 'dock', 150)
    print('wrong_dock: trip', trip and trip['status'], 'dock', home and (home['status'], home.get('reason')),
          'charging', link.charging, 'lost', sum(1 for _, k, _ in events if k == 'lost'), flush=True)
    body.stopping.set()
    return bool(corrected) and math.dist(body.dock[:2], DOCK[:2]) < .05 and bool(home) and home['status'] == 'done' \
        and link.charging


def scenario_kidnap():
    body, link, events = make((1.53, 0.38, 0.))
    time.sleep(2)
    do(body, 'wait', seconds=1)
    time.sleep(3)
    link.kidnap((6.54, 1.38, 2.0))
    time.sleep(1)
    do(body, 'relocalize')
    result = wait_skill(body, events, 'relocalize', 60)
    print('relocalize:', result)
    report(body, link, 'after kidnap')
    body.stopping.set()
    return result and result['status'] == 'done'




def scenario_doors():
    body, link, events = make((9.48, -1.08, -1.0))
    time.sleep(3)
    ok = True
    phases = []
    for name, xy in (('hall', [6.02, -2.88]), ('kitchen', [9.48, -1.08]), ('corridor', [8.02, 1.53]),
                     ('bedroom', [1.53, 0.38])):
        start = time.monotonic()
        body.on_message('body.do', {'skill': 'go', 'args': {'xy': xy, 'tolerance': .5}, 'id': name})
        seen = set()
        while time.monotonic() - start < 150:
            skill = body.skill
            if skill is not None and skill.stage == 'door':
                door = skill.info.get('door') or {}
                seen.add(door.get('phase'))
            if any(k == 'skill' and d.get('id') == name and d.get('status') in ('done', 'failed') for _, k, d in events):
                break
            time.sleep(.05)
        result = [d for _, k, d in events if k == 'skill' and d.get('id') == name and d.get('status') != 'started']
        bumps = sum(1 for _, k, d in events if k == 'bumper')
        print(name, (result[-1]['status'], result[-1].get('reason')) if result else 'timeout', f'{time.monotonic()-start:.0f}s', 'door phases', sorted(p for p in seen if p),
              'bumps', bumps, flush=True)
        ok = ok and bool(result) and result[-1]['status'] == 'done'
    report(body, link, 'end')
    body.stopping.set()
    return ok


def scenario_doorway_person():
    """07.10 18:14: someone stands in the bedroom doorway for a few seconds — wait, then go (no give-up)."""
    body, link, events = make((1.2, .1, 2.8))
    time.sleep(3)
    grid = link.grid
    r0, c0 = grid.cell((2.95, 1.6))
    r1, c1 = grid.cell((3.1, .65))
    before = link.occupied.copy()
    with link.lock:
        link.occupied[r0:r1 + 1, c0:c1 + 1] = True              # a person across the doorway
    time.sleep(2)
    start = time.monotonic()
    body.on_message('body.do', {'skill': 'go', 'args': {'xy': [6.54, 1.38], 'tolerance': .3}, 'id': 'through'})
    time.sleep(5)
    with link.lock:
        link.occupied[:] = before                                # ... and steps aside
    result = wait_skill(body, events, 'through', 90)
    backed = any(k == 'skill' and d.get('id') == 'through' and d.get('status') == 'failed' for _, k, d in events)
    print('doorway_person:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'failed on the way' if backed else '', flush=True)
    report(body, link, 'after doorway')
    body.stopping.set()
    return bool(result) and result['status'] == 'done'


def scenario_dock_chair():
    """07.10 18:16: a chair caster 19 cm beside the line in front of the dock; the robot comes from the corridor."""
    leg = (.31, .16)
    body, link, events = make((4.5, 1.2, 3.1), extra_walls=[(leg[0] - .03, leg[1] - .03, leg[0] + .03, leg[1] + .03)])
    time.sleep(3)
    start = time.monotonic()
    do(body, 'dock')
    result = wait_skill(body, events, 'dock', 240)
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    print('dock_chair:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'charging', link.charging, 'bumps', bumps, flush=True)
    body.stopping.set()
    # Either docked, or gave up with a reason the mind can say — never waiting in silence for ever.
    return bool(result) and bumps == 0


def scenario_dock_bounce():
    """09.10 18:17: one short touch of the springy contacts, then the robot stands 3 cm off them; the skill stood in
    «charging» with no contact for good (5 min and counting, the battery not charging)."""
    body, link, events = make(compose(DOCK, (.8, .05, 0.)), bounce=1)
    time.sleep(3)
    start = time.monotonic()
    do(body, 'dock')
    result = wait_skill(body, events, 'dock', 90)
    stage = (body.skill.stage if body.skill else None)
    print('dock_bounce:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'charging', link.charging, 'stage now', stage, 'bounces left', link.bounce, flush=True)
    body.stopping.set()
    return bool(result) and result['status'] == 'done' and link.charging


def scenario_dock_flicker():
    """09.10 20:20:55-59: the contact flickered — three re-presses in 4 s, then «contact_unstable»; a fresh
    come-round docked at once. Four bounces: the skill comes round itself and docks."""
    body, link, events = make(compose(DOCK, (.8, .05, 0.)), bounce=4)
    time.sleep(3)
    start = time.monotonic()
    do(body, 'dock')
    result = wait_skill(body, events, 'dock', 150)
    print('dock_flicker:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'charging', link.charging, 'bounces left', link.bounce, 'retry', body.skill and body.skill.info.get('retry'),
          flush=True)
    body.stopping.set()
    return bool(result) and result['status'] == 'done' and link.charging


def scenario_slow_pivot():
    """10.10 evening: the wheels broke into a turn slowly (0.03-0.09 rad/s after 1.2 s of 0.42 asked) and every such
    pivot was taken for rubbing — three 10 cm back-offs in a row, then a run-up. Bedroom -> hall with slow pivots."""
    body, link, events = make((2.4, .8, 3.1), spin_accel=.07)        # facing away: a half turn first
    time.sleep(3)
    start = time.monotonic()
    do(body, 'go', xy=[6.0, -2.4])
    stages, last = [], None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        stage = body.skill.stage if body.skill else None
        if stage != last and stage:
            stages.append(stage)
            last = stage
        if any(k == 'skill' and d.get('id') == 'go' and d.get('status') in ('done', 'failed') for _, k, d in events):
            break
        time.sleep(.05)
    result = [d for _, k, d in events if k == 'skill' and d.get('id') == 'go' and d.get('status') != 'started']
    backs = stages.count('backing') + stages.count('runup_back')
    print('slow_pivot:', result[-1]['status'] if result else 'running', f'{time.monotonic() - start:.0f}s',
          'back-offs', backs, 'stages', stages, flush=True)
    body.stopping.set()
    return bool(result) and result[-1]['status'] == 'done' and backs == 0


def scenario_kitchen_corner():
    """07.10 23:21: from the corridor into the kitchen — an L-turn with a threshold at the narrow corridor end;
    three live attempts at finding a way failed. Since 23:52 the owner's taught lines (body/passages.json)."""
    body, link, events = make((6.0, 1.2, 0.))
    time.sleep(3)
    ok = True
    for name, xy in (('kitchen', [9.48, -1.08]), ('back', [6.0, 1.2])):
        start = time.monotonic()
        body.on_message('body.do', {'skill': 'go', 'args': {'xy': xy, 'tolerance': .4}, 'id': name})
        stages = set()
        while time.monotonic() - start < 120:
            skill = body.skill
            if skill is not None:
                stages.add(skill.stage)
            if any(k == 'skill' and d.get('id') == name and d.get('status') in ('done', 'failed') for _, k, d in events):
                break
            time.sleep(.05)
        result = [d for _, k, d in events if k == 'skill' and d.get('id') == name and d.get('status') != 'started']
        bumps = sum(1 for _, k, d in events if k == 'bumper')
        print(name, (result[-1]['status'], result[-1].get('reason')) if result else 'timeout',
              f'{time.monotonic() - start:.0f}s', 'stages', sorted(stages), 'bumps', bumps, flush=True)
        ok = ok and bool(result) and result[-1]['status'] == 'done' and bumps == 0
    report(body, link, 'end')
    body.stopping.set()
    return ok


def scenario_kitchen_retry():
    """Owner's rule 07.10: held up in the doorway — back along the same line to its start and go again."""
    body, link, events = make((6.0, 1.2, 0.))
    time.sleep(3)
    start = time.monotonic()
    body.on_message('body.do', {'skill': 'go', 'args': {'xy': [9.48, -1.08], 'tolerance': .4}, 'id': 'kitchen'})
    grid = link.grid
    r0, c0 = grid.cell((8.70, .20))
    r1, c1 = grid.cell((8.90, .05))
    before = link.occupied.copy()
    stages, blocked_at = [], None
    while time.monotonic() - start < 150:
        skill = body.skill
        if skill is not None and (not stages or stages[-1] != skill.stage):
            stages.append(skill.stage)
        pose = link.true_pose
        if blocked_at is None and skill is not None and skill.stage == 'passage_run' and pose[1] < .75:
            with link.lock:
                link.occupied[r0:r1 + 1, c0:c1 + 1] = True             # someone steps into the doorway ...
            blocked_at = time.monotonic()
        if blocked_at is not None and time.monotonic() - blocked_at > 6.:
            with link.lock:
                link.occupied[:] = before                               # ... and goes away
        if any(k == 'skill' and d.get('id') == 'kitchen' and d.get('status') in ('done', 'failed') for _, k, d in events):
            break
        time.sleep(.05)
    result = [d for _, k, d in events if k == 'skill' and d.get('id') == 'kitchen' and d.get('status') != 'started']
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    print('kitchen_retry:', (result[-1]['status'], result[-1].get('reason')) if result else 'timeout',
          f'{time.monotonic() - start:.0f}s', 'stages', stages, 'bumps', bumps, flush=True)
    body.stopping.set()
    return bool(result) and result[-1]['status'] == 'done' and 'passage_back' in stages and bumps == 0


def scenario_chair_behind():
    """08.10 11:30: a chair leg right at the tail (2.7 cm), free ahead; no maneuver found, the trip home failed
    twice and «rest» was then refused. Now: out forward a little, plan again, home."""
    from bratan.common.geometry import compose as comp
    pose = (0.64, -0.72, -0.88)
    leg = comp(pose, (-.27, -.07, 0.))          # its 5 cm map cells end ~3 cm from the tail
    body, link, events = make(pose, extra_walls=[(leg[0] - .025, leg[1] - .025, leg[0] + .025, leg[1] + .025)])
    time.sleep(3)
    start = time.monotonic()
    do(body, 'dock')
    result = wait_skill(body, events, 'dock', 200)
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    print('chair_behind:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'charging', link.charging, 'bumps', bumps, flush=True)
    body.stopping.set()
    return bool(result) and result['status'] == 'done' and link.charging and bumps == 0


def scenario_follow_corner():
    """«Братан, поезжай за мной»: the person walks out of the bedroom into the corridor and turns out of sight.

    The robot must follow while it sees them, then drive to where they were last seen (not spin and give up)."""
    import math as m
    body, link, events = make((1.2, .3, 0.))
    time.sleep(3)
    path = [(1.9, .35), (2.6, .7), (3.05, 1.1), (3.6, 1.2), (4.3, 1.2), (5.0, 1.2)]
    body.on_message('body.do', {'skill': 'track', 'args': {'who': 'человек', 'mode': 'follow', 'seconds': 90,
                                                           'distance': .7}, 'id': 'follow'})
    start = time.monotonic()
    last_seen = None
    while time.monotonic() - start < 80:
        t = time.monotonic() - start
        leg = min(len(path) - 1.001, t / 3.)          # 3 s per leg, then stands still in the corridor
        i = int(leg)
        f = leg - i
        px = path[i][0] + f * (path[i + 1][0] - path[i][0])
        py = path[i][1] + f * (path[i + 1][1] - path[i][1])
        pose = link.true_pose
        bearing = m.atan2(py - pose[1], px - pose[0]) - pose[2]
        bearing = m.atan2(m.sin(bearing), m.cos(bearing))
        distance = m.hypot(px - pose[0], py - pose[1])
        visible = abs(bearing) < .45 and distance < 4. and t < 17.          # gone round the corner after 17 s
        if visible:
            last_seen = (px, py)
            body.on_message('eyes.seen', {'mono': time.monotonic(), 'items': [
                {'label': 'person', 'track': 7, 'bearing': bearing, 'distance': distance, 'map_xy': [px, py]}]})
        if any(k == 'skill' and d.get('id') == 'follow' and d.get('status') in ('done', 'failed') for _, k, d in events):
            break
        time.sleep(.1)
    pose = link.true_pose
    gap = m.dist(pose[:2], last_seen) if last_seen else 9.
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    result = [d for _, k, d in events if k == 'skill' and d.get('id') == 'follow' and d.get('status') != 'started']
    print('follow_corner: last seen', last_seen and tuple(round(v, 2) for v in last_seen), 'robot',
          tuple(round(v, 2) for v in pose), f'gap {gap:.2f} m', 'bumps', bumps,
          (result[-1]['status'], result[-1].get('reason')) if result else 'running', flush=True)
    body.stopping.set()
    return gap < 1.0 and bumps == 0


def scenario_perform():
    """11.10 (GrowBot's «Disney mode»): a little number in the bedroom — a bow, a half turn, back 20 cm (away from the start), a half turn
    back, a spin, back 20 cm (home again). Every step is cued, nothing is bumped, he ends near where he began, and his own
    motion is never felt as hands."""
    body, link, events = make((2.4, .8, 3.1))
    time.sleep(3)
    start, began = time.monotonic(), link.true_pose
    do(body, 'perform', steps=[{'emote': 'bow', 'say': 'Встречайте!', 'pause': 1}, {'turn': math.pi},
                               {'move': -.2, 'say': 'Опа'}, {'turn': -math.pi}, {'emote': 'spin'}, {'move': -.2}])
    result = wait_skill(body, events, 'perform', 90)
    time.sleep(2.5)
    cues = [d['index'] for _, k, d in events if k == 'cue']
    bumps = sum(1 for _, k, d in events if k == 'bumper')
    felt = [d for _, k, d in events if k == 'handled']
    gap = math.dist(began[:2], link.true_pose[:2])
    print('perform:', result and (result['status'], result.get('reason')), f'{time.monotonic() - start:.0f}s',
          'cues', cues, 'bumps', bumps, 'felt', felt, f'end {gap:.2f} m from start', flush=True)
    body.stopping.set()
    return bool(result) and result['status'] == 'done' and cues == list(range(6)) and not bumps and not felt \
        and gap < .2


def scenario_hands():
    """11.10: standing still, someone pushes him back 10 cm by hand: felt once as «оттолкнули назад»."""
    body, link, events = make((2.4, .8, 3.1))
    time.sleep(4)
    for _ in range(25):                          # 10 cm over a second, the wheels turned by the push
        with link.lock:
            link.odom = compose(link.odom, (-.004, 0., 0.))
            link.true_pose = compose(link.true_pose, (-.004, 0., 0.))
        time.sleep(.04)
    time.sleep(1.5)
    felt = [d for _, k, d in events if k == 'handled']
    print('hands:', felt, flush=True)
    body.stopping.set()
    return len(felt) == 1 and felt[0]['how'] == 'pushed' and felt[0]['back']


if __name__ == '__main__':
    names = sys.argv[1:] or ['trip_and_dock', 'kidnap']
    ok = all(globals()['scenario_' + n]() for n in names)
    print('ALL OK' if ok else 'FAILED')
