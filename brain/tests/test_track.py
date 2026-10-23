"""The Track skill's look-round with a fake body (no robot): python -m pytest tests/test_track.py

09.10: «найди меня» shivered on the spot — a new random turn direction every tick, 0.17 rad in 8 s.
"""
import math

from bratan.body import skills


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Ctx:
    """Heading follows the asked turn exactly, unless `blocked(w)` says the guard stops it."""

    def __init__(self, blocked=lambda w: False, seen=None):
        self.now, self.dt = 0., .02
        self.odom = [0., 0., 0.]
        self.twist = [0., 0.]
        self.blocked = blocked
        self.seen = seen or (lambda ctx: None)

    def find(self, who, track=None):
        return self.seen(self)

    def step(self, skill):
        intent = skill.tick(self) or skills.STOP
        w = 0. if self.blocked(intent.w) else intent.w
        self.twist = [intent.v, w]
        self.odom[2] = wrap(self.odom[2] + w * self.dt)
        self.now += self.dt
        return intent


def make(mode='search', seconds=40., **args):
    skill = skills.Track({'who': 'Сергей', 'mode': mode, 'seconds': seconds, **args}, id='t', origin='owner')
    skill.kind, skill.started = 'track', 0.
    return skill


def run(ctx, skill, limit=60.):
    turns = []
    while skill.status is None and ctx.now < limit:
        turns.append(ctx.step(skill).w)
    return turns


def test_a_search_turns_one_way_in_steps_and_gives_up_after_a_full_turn():
    ctx, skill = Ctx(), make()
    turns = run(ctx, skill)
    assert skill.status == 'failed' and skill.reason == 'not_found'
    assert len({math.copysign(1., w) for w in turns if w}) == 1             # one way round
    looks = sum(1 for a, b in zip(turns, turns[1:]) if a and not b)
    assert looks >= 10                                                     # stood still to look ~every 0.55 rad
    assert 15. < ctx.now < 32.                                             # a whole turn with looks, not 40 s


def test_a_blocked_turn_goes_the_other_way_then_gives_up():
    ctx, skill = Ctx(blocked=lambda w: w > 0), make(bearing=.5)          # starts left, the left is blocked
    turns = run(ctx, skill)
    assert skill.reason == 'not_found' and min(turns) < 0                 # went round to the right instead
    ctx, skill = Ctx(blocked=lambda w: True), make()
    run(ctx, skill)
    assert skill.status == 'failed' and skill.reason == 'turn_blocked' and ctx.now < 4.


def test_seek_looks_round_comes_up_and_is_done_when_there():
    target = 1.6                                                          # someone off to the left

    def seen(ctx):
        bearing = wrap(target - ctx.odom[2])
        if abs(bearing) < .44:
            return {'label': 'person', 'track': 7, 'distance': .85, 'bearing_now': bearing}
        return None
    ctx, skill = Ctx(seen=seen), make('seek', seconds=45., distance=.8, bearing=.4)
    run(ctx, skill)
    assert skill.status == 'done' and skill.reason == 'arrived'
    assert abs(wrap(target - ctx.odom[2])) < .3 and ctx.now < 12.


def test_search_is_done_as_soon_as_they_are_in_view():
    ctx, skill = Ctx(seen=lambda ctx: {'label': 'dog', 'track': 3, 'distance': 1.5, 'bearing_now': .1}), make()
    ctx.step(skill)
    assert skill.status == 'done' and skill.reason == 'found'


# ---------------------------------------------------------------- something moving while we drive (lidar vs map)
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from bratan.body.grid import Grid  # noqa: E402
from bratan.body.motion import MovingWatch  # noqa: E402

MAP = Path(__file__).parent / 'data' / 'apartment-final-20261006-dock.yaml'


def blob(x, y, n=5):
    """Returns of a pair of legs / a dog across the beams."""
    return [(x, y - .1 + .2 * i / (n - 1)) for i in range(n)]


def test_something_walking_by_is_seen_while_driving_clutter_is_not():
    grid = Grid(MAP)
    watch = MovingWatch(grid)
    events = []
    rng = np.random.default_rng(3)
    for step in range(16):                                        # 2.7 s at 6 scans a second, robot driving on
        t = step / 6.
        pose = (.2 + .2 * t, .3, 0.)
        clutter = blob(1.6 + rng.normal(0, .02), .8 + rng.normal(0, .02))    # a bag the map lacks, jittering
        walker = blob(1.9, -.1 + .45 * t) if t < 2.2 else []                  # someone crossing ahead-left
        seen = watch.update(pose, np.array(clutter + walker), t)
        if seen:
            events.append((t, seen))
    assert len(events) == 1                                       # one event for one walker, none for the bag
    t, (bearing, distance) = events[0]
    assert t < 1.5 and abs(bearing) < 1.2 and 1. < distance < 2.            # seen within 1.5 s, ahead


def test_a_wall_is_never_a_walker():
    grid = Grid(MAP)
    watch = MovingWatch(grid)
    v, _, _ = grid.sample(np.array([[-.3, -.5]]))
    for step in range(12):
        pose = (.2 + .05 * step, .3, 0.)
        assert watch.update(pose, np.array(blob(-.3, -.5)), step / 6.) is None or v[0] > .2


def test_a_few_frames_without_the_dog_are_not_losing_it():
    # 10.10 evening: Taffa in some frames, not all; 0.6 s without her and the skill stood, then drove to her map
    # point while she ran on. Now it keeps turning after her the way she was going.
    dog = {'angle': .3}

    def seen(ctx):
        dog['angle'] += .4 * ctx.dt                              # she runs off to the left, 0.4 rad/s
        if int(ctx.now / .2) % 4 == 3:                           # every fourth fifth of a second she is missed
            return None
        if 2. < ctx.now < 3.:                                    # and once for a whole second
            return None
        return {'label': 'dog', 'track': 3, 'distance': 2., 'bearing_now': wrap(dog['angle'] - ctx.odom[2])}
    ctx, skill = Ctx(seen=seen), make('follow', seconds=6., who='собака', distance=.8)
    run(ctx, skill, limit=6.)
    assert 'chase' not in skill.info and 'sweep' not in skill.info          # never gave her up
    assert abs(wrap(dog['angle'] - ctx.odom[2])) < .35                       # still facing her
