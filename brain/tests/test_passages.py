"""Taught passages (07.10, the kitchen doorway): get onto the owner's line, drive it, back up and retry."""
import json
import math
from pathlib import Path

from bratan.body import skills
from bratan.common.geometry import wrap

PASSAGES = json.loads((Path(skills.__file__).parent / 'passages.json').read_text(encoding='utf-8'))['passages']
IN, OUT = PASSAGES


class Ctx:
    """Ideal unicycle: a held heading is reached at 3 rad/s (capped like the board), nothing is in the way
    unless `wall_at` (metres along the line) stops it."""

    def __init__(self, pose, wall_at=None):
        self.pose = self.odom = pose
        self.now = 0.
        self.passages = PASSAGES
        self.localized = True
        self.wall_at = wall_at
        self.blocked_since = None

    def blocked_for(self):
        return self.now - self.blocked_since if self.blocked_since is not None else 0.

    def step(self, intent, passage, dt=.05):
        w = intent.w
        if intent.hold:
            w = max(-.42, min(.42, 3. * wrap(intent.heading - self.odom[2])))
        x, y, h = self.pose
        nxt = (x + intent.v * math.cos(h) * dt, y + intent.v * math.sin(h) * dt, wrap(h + w * dt))
        along = skills.Navigate._on_passage(passage, nxt)[0]
        if self.wall_at is not None and intent.v > 0 and along > self.wall_at:
            self.blocked_since = self.blocked_since if self.blocked_since is not None else self.now
        else:
            self.blocked_since = None
            self.pose = self.odom = nxt
        self.now += dt


def navigate_through(passage, start, goal, wall_at=None, seconds=40.):
    nav = skills.Navigate({'xy': list(goal)})
    nav.info['route'] = [start[:2], tuple(passage['gate']), tuple(goal)]
    ctx = Ctx(start, wall_at)
    assert nav._passage_start(ctx) and nav.stage == 'passage_approach'
    stages, worst = [], 0.
    nav._plan = lambda ctx: setattr(nav, 'stage', 'planning')       # the run is over: back to plain routing
    while ctx.now < seconds and nav.stage.startswith('passage') and nav.status is None:
        if nav.stage == 'passage_approach':
            nav._passage(ctx)
            if nav.stage == 'passage_approach':                    # the sub-route to the line is not simulated
                nav._passage_run_start(ctx)
        intent = nav._passage(ctx)
        if not stages or stages[-1] != nav.stage:
            stages.append(nav.stage)
        ctx.step(intent, passage)
        along, left = nav._on_passage(passage, ctx.pose)
        if along > passage.get('tight_from', 0.):
            worst = max(worst, abs(left))
    return nav, ctx, stages, worst


def test_route_through_the_gate_takes_the_passage_only_in_its_direction():
    nav = skills.Navigate({'xy': [9.48, -1.08]})
    nav.info['route'] = [(8.42, 1.32), (8.7, .3), (9.0, -.5), (9.48, -1.08)]
    assert nav._passage_start(Ctx((8.42, 1.32, -1.2)))
    assert nav.info['passage_name'] == IN['name']
    other = skills.Navigate({'xy': [6.0, 1.2]})                    # along the corridor: not through the door
    other.info['route'] = [(8.42, 1.32), (7.0, 1.3), (6.0, 1.2)]
    assert not other._passage_start(Ctx((8.42, 1.32, 3.1)))
    sub = skills.Navigate({'xy': [9.48, -1.08], 'no_passages': True})
    sub.info['route'] = nav.info['route']
    assert not sub._passage_start(Ctx((8.42, 1.32, -1.2)))


def test_drives_the_taught_line_both_ways():
    # 23:47: the robot's own stop by the entry was 12 cm off the line.
    nav, ctx, stages, worst = navigate_through(IN, (8.424, 1.317, -1.211), (9.48, -1.08))
    assert nav.stage == 'planning' and nav.info['passages_done'] == [IN['name']]
    assert worst < .03
    nav, ctx, stages, worst = navigate_through(OUT, (8.88, -.67, 2.6), (8.02, 1.53))
    assert nav.stage == 'planning' and worst < .03


def test_held_up_backs_to_the_start_and_tries_again_then_says_why():
    # Owner 07.10: «если не доезжает — отъезжать на ту же точку назад и пробовать ещё раз», backing, not turning.
    nav, ctx, stages, worst = navigate_through(IN, (8.2841, 1.3488, -1.1926), (9.48, -1.08), wall_at=1.0,
                                                 seconds=120.)
    assert stages.count('passage_back') == skills.PASSAGE_TRIES
    assert nav.status == 'failed' and nav.reason == 'passage_blocked'
    along, left = skills.Navigate._on_passage(IN, ctx.pose)
    assert abs(along) < .05 and abs(left) < .02                    # backed out of the doorway before giving up
    assert abs(wrap(ctx.pose[2] - IN['start'][2])) < .05           # reversing, never turned round


def test_backing_returns_to_the_taught_start():
    nav = skills.Navigate({'xy': [9.48, -1.08]})
    nav.info['route'] = [(8.2841, 1.3488), tuple(IN['gate']), (9.48, -1.08)]
    ctx = Ctx((8.2841, 1.3488, -1.1926), wall_at=1.0)
    nav._passage_start(ctx)
    nav._passage_run_start(ctx)
    while nav.stage != 'passage_back' and ctx.now < 30:
        ctx.step(nav._passage(ctx), IN)
    while nav.stage == 'passage_back' and ctx.now < 60:
        ctx.step(nav._passage(ctx), IN)
    along, left = skills.Navigate._on_passage(IN, ctx.pose)
    assert nav.stage == 'passage_run' and abs(along) < .05 and abs(left) < .02


def test_body_emotes_end_where_they_began():
    for kind in skills.EMOTES:
        emote = skills.Emote({'kind': kind})
        ctx = Ctx((1., 1., .3))
        emote.started = 0.
        while emote.status is None and ctx.now < 30.:
            intent = emote.tick(ctx)
            if intent.hold:
                ctx.step(intent, IN)
            else:
                x, y, h = ctx.pose
                nxt = (x + intent.v * math.cos(h) * .05, y + intent.v * math.sin(h) * .05, wrap(h + intent.w * .05))
                ctx.pose = ctx.odom = nxt
                ctx.now += .05
        assert emote.status == 'done', kind
        assert abs(wrap(ctx.pose[2] - .3)) < .08, kind
        assert math.dist(ctx.pose[:2], (1., 1.)) < .05, kind


def test_from_beside_the_door_in_the_kitchen_the_way_home_takes_the_taught_line():
    # 09.10 11:25: in the kitchen 1 m east of the line, level with its gate; «past the doorway» was judged by the
    # distance along the line only, so the plain route went across the doorway and wedged in the L-turn (4 min).
    route = [(9.76, .13), (9.28, .12), (8.98, .08), (8.62, .33), (8.53, .58), (8.28, .98), (6.0, 1.2), (-.09, -.06)]
    nav = skills.Navigate({'xy': [-.09, -.06]})
    nav.info['route'] = route
    assert nav._passage_start(Ctx((9.76, .13, 2.8)))
    assert nav.info['passage_name'] == OUT['name'] and nav.stage == 'passage_approach'
    on_line = skills.Navigate({'xy': [-.09, -.06]})                # really on the line beyond the gate: done
    on_line.info['route'] = [(8.62, .56), (8.28, .98), (6.0, 1.2), (-.09, -.06)]
    assert not on_line._passage_start(Ctx((8.62, .56, 1.7)))
