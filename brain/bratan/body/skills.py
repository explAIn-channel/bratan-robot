"""Motion skills. Each `tick(ctx)` returns the wanted (v, w, hold, heading) or None.

A skill never talks to the board: the body loop passes its intent through the
Guard and the link. Skills end by setting `status` to 'done' or 'failed' with a
short `reason`. Speeds are the owner's: cruise .24 m/s, turns .42 rad/s.
"""
from __future__ import annotations

import math
import random

import numpy as np
from dataclasses import dataclass, field

from ..common.geometry import (BODY_RADIUS, compose, lookahead_point, path_length, relative,
                               remaining_path, transform, wrap, inverse)
from . import maneuver
from .planner import NoRoute

CRUISE = .24
TURN = .42
TURN_MIN = .10
PIVOT_ABOVE = .55          # rad of heading error: turn in place first
VERIFY_NEAR_M, VERIFY_FAR_M = 1.0, 2.6   # a thing coming up ahead is in good view between these distances
VERIFY_BEARING = .35       # rad off the bow: well inside the camera's 0.89 rad
VERIFY_FRAMES = 10         # frames it should have been in, never seen: it is not there any more
VERIFY_LIGHT = 35
TRACK_COAST_S = 1.5        # a target unseen this long is lost; until then the body keeps after it
SWEEP_STEP = .55           # rad turned between looks while searching (the camera sees ~0.9 rad across)
SWEEP_HOLD_S = .45         # then still for a moment: sharp frames (a dim room blurs while turning)
SWEEP_STUCK_S = 1.5        # asked to turn, the heading did not move (the guard: a bed, a chair leg)
RUNUP_BACK_M = .45         # door thresholds (owner 06.10.2026): back off, then take it at full speed
RUNUP_CHARGE_M = .75
RUNUP_MAX_BEARING = .45
RUNUP_PER_TARGET = 3
RUNUP_BACK_MAX_M = .7
RUNUP_FAN = .7             # after backing off, look for a clear straight line within this of the route
RUNUP_CLEAR_M = .8
THROUGH_SPEED = .24
GOAL_TOLERANCE = .12
STALL_S = 4.
NO_ROUTE_WAIT_S = 8.       # a shut passage is often shut only for a moment: look again before giving up
DOCK_ALIGN_TOL = .035
DOCK_ALIGN_W = .35
DOCK_LATERAL = .02
DOCK_BLOCKED_S = 6.        # the guard refusing this long while lining up at the dock: come round again
CORNER_TURN = .6           # rad: a sharper bend of the route in a narrow place is taken stop-turn-go
CORNER_TIGHT = BODY_RADIUS + .15
CORNER_LOOK_M = 1.5
CORNER_REACHED_M = .08
DOCK_PRESS_V = .05         # the last push onto the springy contacts (slower stalls short of them)
DOCK_PRESS_S = 1.2
RELOC_BACK_V = .06          # lost and wedged: back out at this speed ...
RELOC_BACK_STEP_M = .25     # ... this far at a time ...
RELOC_BACK_TOTAL_M = .6     # ... and no more than this in one search
PASSAGE_GATE_M = .35       # a route through a taught doorway passes this near its gate
PASSAGE_STEER = 3.         # rad of heading per metre off the taught line (converges within ~0.4 m)
PASSAGE_STEER_MAX = .4
PASSAGE_PIVOT = .3         # further off the line's heading than this: face it on the spot first
PASSAGE_STUCK_S = 2.5      # a run held up this long (guard refusing, or no headway: the threshold) ...
PASSAGE_BACK_V = .12       # ... backs along the same line to its start and goes again (owner's rule, 07.10)
PASSAGE_SIDE_M = .35       # this close to a taught line counts as on it
PASSAGE_TRIES = 3          # runs per passage; then say why instead of grinding at the jamb


@dataclass
class Intent:
    v: float = 0.
    w: float = 0.
    hold: bool = False
    heading: float = 0.      # board odometry yaw for hold


STOP = Intent()


def point_segment_distance(p, a, b):
    """Metres from point p to the segment a-b."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    length2 = ax * ax + ay * ay
    t = 0. if length2 < 1e-12 else max(0., min(1., ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / length2))
    return math.hypot(p[0] - a[0] - t * ax, p[1] - a[1] - t * ay)


def turn_toward(error, limit=TURN, gain=1.6, minimum=TURN_MIN):
    return math.copysign(min(limit, max(minimum, abs(error) * gain)), error)


@dataclass
class Skill:
    args: dict
    id: str = ''
    origin: str = 'mind'
    kind: str = ''
    stage: str = 'start'
    status: str | None = None
    reason: str = ''
    started: float = 0.
    info: dict = field(default_factory=dict)

    lidar_needed = True

    def done(self, reason=''):
        self.status, self.reason = 'done', reason
        return STOP

    def fail(self, reason):
        self.status, self.reason = 'failed', reason
        return STOP

    def public(self):
        plain = {k: v for k, v in self.info.items()
                 if isinstance(v, (int, float, str, bool)) and k not in ('progress_at', 'last_yaw')}
        nested = next((v for v in self.info.values() if isinstance(v, Skill)), None)
        if nested is not None:
            plain['sub'] = f'{nested.kind}:{nested.stage}'
        return {'id': self.id, 'kind': self.kind, 'stage': self.stage, 'origin': self.origin,
                'args': self.args, 'status': self.status, 'reason': self.reason, **plain}

    def tick(self, ctx) -> Intent:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------- simple moves
class Rotate(Skill):
    kind = 'rotate'

    def tick(self, ctx):
        if self.stage == 'start':
            self.info['turned'] = 0.
            self.info['last_yaw'] = ctx.odom[2]
            self.stage = 'turning'
            self.info['progress_at'] = ctx.now
        delta = wrap(ctx.odom[2] - self.info['last_yaw'])
        self.info['turned'] += delta
        self.info['last_yaw'] = ctx.odom[2]
        if abs(delta) > .01:
            self.info['progress_at'] = ctx.now
        elif ctx.now - self.info['progress_at'] > 3.:
            return self.fail('turn_blocked')
        error = float(self.args.get('angle', 0.)) - self.info['turned']
        if abs(error) < .03:
            return self.done()
        return Intent(0., turn_toward(error, gain=1.5, minimum=.08), abs(error) < math.pi / 2,
                      wrap(ctx.odom[2] + error))


class Drive(Skill):
    kind = 'drive'

    def tick(self, ctx):
        distance = float(self.args.get('distance', 0.))
        if self.stage == 'start':
            self.info['origin'] = ctx.odom
            self.stage = 'driving'
        progress = relative(self.info['origin'], ctx.odom)[0]
        left = distance - progress
        if abs(left) < .012:
            return self.done()
        if ctx.now - self.started > 5. + abs(distance) / .05:
            return self.fail('blocked')        # the guard held it: say so instead of waiting for ever
        top = float(self.args.get('speed', CRUISE))
        speed = math.copysign(min(top, max(.05, 1.2 * abs(left))), left)
        return Intent(speed, 0., True, self.info['origin'][2])


class Wait(Skill):
    kind = 'wait'
    lidar_needed = False

    def tick(self, ctx):
        if ctx.now - self.started >= float(self.args.get('seconds', 1.)):
            return self.done()
        return STOP


# ---------------------------------------------------------------- route driving
class Navigate(Skill):
    """Plan, then follow the route with pure pursuit; doors are crossed straight."""
    kind = 'go'

    def target(self, ctx):
        return tuple(self.args['xy'])

    def tick(self, ctx):
        if not ctx.localized:
            return self._lost(ctx)
        self.info.pop('lost_since', None)
        if self.stage == 'start':
            self._plan(ctx)
        if self.stage == 'planning':
            future = self.info['future']
            if not future.done():
                return STOP
            try:
                self.info['route'] = list(future.result())
                self.info['doors'] = [d for d in getattr(future.result(), 'doors', ())
                                      if not any(math.dist(d['center'], x) < .4 for x in self.info.get('crossed', []))]
                self.info['revision'] = ctx.grid.revision
                self.stage = 'driving'
                self.info['progress'] = (ctx.now, ctx.pose)
                self.info.pop('no_route_since', None)
                self._passage_start(ctx)
            except NoRoute as error:
                return self._no_route(ctx, str(error))
        if self.stage == 'waiting':
            if ctx.now >= self.info['replan_at']:
                self._plan(ctx)
            return STOP
        if self.stage == 'backing':
            return self._backing(ctx)
        if self.stage in ('maneuver_plan', 'maneuver'):
            return self._maneuver(ctx)
        if self.stage in ('runup_back', 'runup_aim', 'runup_charge'):
            return self._runup(ctx)
        if self.stage in ('passage_approach', 'passage_run', 'passage_back'):
            return self._passage(ctx)
        if self.stage == 'door':
            return self._door(ctx)
        if self.stage == 'final_yaw':
            error = wrap(self.args['yaw'] - ctx.pose[2])
            if abs(error) < .04:
                return self.arrived(ctx)
            return Intent(0., turn_toward(error), True, wrap(ctx.odom[2] + error))
        if self.stage == 'driving':
            return self._drive(ctx)
        return STOP

    def _plan(self, ctx):
        self.info['future'] = ctx.plan(ctx.pose, self.target(ctx), float(self.args.get('tolerance', 0.)))
        self.stage = 'planning'

    def _no_route(self, ctx, reason):
        # Someone standing in the doorway, the dog, a door on the swing (07.10 18:14: «no route», and the
        # same route was free 25 s later): stand and look again for a while, without backing off.
        if reason in ('no_route', 'goal_has_no_body_clearance'):
            since = self.info.setdefault('no_route_since', ctx.now)
            if ctx.now - since < NO_ROUTE_WAIT_S:
                self.info['replan_at'] = ctx.now + 1.5
                self.stage = 'waiting'
                return STOP
        self.info.pop('no_route_since', None)
        attempts = self.info['plan_failures'] = self.info.get('plan_failures', 0) + 1
        # Backing off helps only from a tight spot (the start inside the walls' margin). A shut door is not
        # opened by it — 08.10 10:06 two back-offs right after undocking put the robot back on the charger.
        cell = ctx.grid.cell(ctx.pose)
        tight = ctx.grid.inside(cell) and ctx.grid.clearance[cell] < BODY_RADIUS + .05
        wedged = reason in ('no_maneuver', 'blocked')
        if attempts <= 2 and (tight or wedged) and ctx.clear(-.08, 0., 2.):
            self.info['back_origin'] = ctx.odom
            self.stage = 'backing'
            return STOP
        if attempts <= 2 and (tight or wedged) and ctx.clear(.08, 0., 2.):
            # Nothing room behind (a chair leg at the tail, 08.10 11:30) but free ahead: out forward a little, then
            # plan again from there.
            self.info['back_origin'] = ctx.odom
            self.info['back_distance'] = -.15
            self.stage = 'backing'
            return STOP
        return self.fail(reason)

    def _lost(self, ctx):
        since = self.info.setdefault('lost_since', ctx.now)
        if ctx.now - since > 25.:
            return self.fail('lost')
        return STOP                 # the body loop runs recovery while we wait

    def _backing(self, ctx):
        """Back off (or, with a negative distance, edge forward) a little, then plan again."""
        progress = relative(self.info['back_origin'], ctx.odom)[0]
        distance = self.info.get('back_distance', .15)
        direction = -1. if distance > 0 else 1.
        if direction * progress >= abs(distance) or not ctx.clear(.08 * direction, 0., 1.):
            self.info.pop('back_distance', None)
            self._plan(ctx)
            return STOP
        return Intent(.08 * direction, 0., True, self.info['back_origin'][2])

    def _drive(self, ctx):
        route = self.info['route']
        pose = ctx.pose
        goal = route[-1]
        distance = math.dist(pose[:2], goal)
        if distance < GOAL_TOLERANCE:
            if 'yaw' in self.args:
                self.stage = 'final_yaw'
                return STOP
            return self.arrived(ctx)
        # Replan when the map changed under the remaining route.
        if ctx.grid.revision != self.info['revision']:
            self.info['revision'] = ctx.grid.revision
            if not ctx.route_clear(remaining_path(pose, route)):
                self._plan(ctx)
                return STOP
        if self._slipping(ctx) and self._runup_start(ctx, route):
            return STOP
        # Stall: no 10 cm / 15 deg of progress for STALL_S.
        t0, p0 = self.info['progress']
        if math.dist(pose[:2], p0[:2]) > .25:
            self.info['maneuvers'] = 0                 # real headway: tight-spot attempts start over
        if math.dist(pose[:2], p0[:2]) > .10 or abs(wrap(pose[2] - p0[2])) > .25:
            self.info['progress'] = (ctx.now, pose)
        elif ctx.now - t0 > STALL_S:
            if self._runup_start(ctx, route, aim=True):
                return STOP
            self.info['stalls'] = self.info.get('stalls', 0) + 1
            if self.info['stalls'] > 3:
                return self.fail('blocked')
            self.info['progress'] = (ctx.now, pose)
            ctx.mark_blocked_ahead()
            return self._no_route(ctx, 'blocked')
        for door in list(self.info.get('doors', ())):
            # Robot in the door's frame: x < 0 before the opening, |y| sideways.
            local = relative((*door['center'], door['axis']), pose)
            if -1.3 < local[0] < -.05 and abs(local[1]) < .8:
                self._begin_door(ctx, door)
                if self.stage == 'door':
                    return self._door(ctx)
            if local[0] > .3 and math.dist(pose[:2], door['center']) < .8:
                self.info['doors'].remove(door)            # just came through it
        ahead = remaining_path(pose, route)
        # 07.10: the mere presence of 'pivot_stuck' (set on the very first tick of any turn, before the wheels
        # spin up) sent every turn of more than ~30° into a parking maneuver and, after two, into a run-up —
        # the slow «тыркается» and the circling by the dock. Only a pivot that really does not turn counts.
        if self._tight(ctx) and (ctx.blocked_for() > .4 or self._pivot_stuck(ctx)):
            self.info.pop('pivot_stuck', None)
            self.info.pop('pivot_yaw', None)
            if self.info.get('maneuvers', 0) >= 2 and self._runup_start(ctx, route, aim=True):
                self.info['maneuvers'] = 0
                return STOP
            return self._plan_maneuver(ctx, ahead)
        look = .25 + .5 * min(1., ctx.twist[0] / CRUISE)
        point = lookahead_point(ahead, look)
        cap = None
        corner = self.info.get('corner_target')
        if corner is not None and (ctx.now - corner[3] > 25. or math.dist(pose[:2], corner[1]) > 1.6):
            corner = self.info['corner_target'] = None       # given up on (blocked, replanned far away)
        if corner is None:
            corner = self._tight_corner(ctx, ahead)
            if corner is not None:
                corner = self.info['corner_target'] = (*corner, ctx.now)   # held until reached: no hopping
        if corner is not None:
            # A sharp bend in a narrow place (07.10: corridor end -> kitchen door, an L-turn 0.6 m wide):
            # pure pursuit cut it and put the nose into the jamb. Drive to the widest spot by the bend,
            # turn on the spot onto the next leg, then go on — the way a careful driver parks round a corner.
            vertex, pivot, out, _ = corner
            self.info['corner'] = f'{pivot[0]:.2f},{pivot[1]:.2f}'
            reach = math.dist(pose[:2], pivot)
            if reach > CORNER_REACHED_M:
                if reach < .8:
                    point = pivot            # aim at the turning spot itself: no cutting the bend
                cap = min(CRUISE, .06 + .45 * reach)
            else:
                error = wrap(out - pose[2])
                if abs(error) > .07:
                    return Intent(0., turn_toward(error), True, wrap(ctx.odom[2] + error))
                self.info.setdefault('corners', []).append(vertex)
                self.info.pop('corner', None)
                self.info['corner_target'] = None
        local = relative(pose, (*point, 0.))
        error = math.atan2(local[1], local[0])
        if abs(error) > PIVOT_ABOVE:
            command = turn_toward(error)
            # A pivot that does not turn is rubbing on something: back off a little and try again.
            if abs(ctx.twist[1]) < .3 * abs(command):
                self.info.setdefault('pivot_stuck', ctx.now)
                self.info.setdefault('pivot_yaw', ctx.odom[2])
                if self._pivot_stuck(ctx) and ctx.clear(-.08, 0., 1.5):
                    self.info.pop('pivot_stuck', None)
                    self.info.pop('pivot_yaw', None)
                    self.info['back_origin'] = ctx.odom
                    self.info['back_distance'] = .10
                    self.stage = 'backing'
                    return STOP
            else:
                self.info.pop('pivot_stuck', None)
                self.info.pop('pivot_yaw', None)
            return Intent(0., command, True, wrap(ctx.odom[2] + error))
        self.info.pop('pivot_stuck', None)
        self.info.pop('pivot_yaw', None)
        left = path_length(ahead)
        speed = min(float(self.args.get('speed', CRUISE)), .08 + .6 * left)
        speed *= max(.35, math.cos(error) ** 2)
        if ctx.near_obstacle(.45):
            speed = min(speed, .12)
        if cap is not None:
            speed = min(speed, cap)
        # Pure pursuit curvature, clipped to the turn limit.
        curvature = 2 * local[1] / max(.05, local[0] ** 2 + local[1] ** 2)
        w = max(-TURN, min(TURN, speed * curvature))
        return Intent(speed, w)

    # ------------------------------------------------------------ taught passages
    # 07.10: the corridor end -> kitchen door is an L-turn with a threshold, 8 cm to spare on the right line;
    # a minute of maneuvers and run-ups found no way in. The owner put the robot where it is always taken
    # from: driven straight from there at full speed it goes through clean, both ways. So such a doorway is
    # not searched for: the robot gets onto the taught line and drives it, correcting by the map pose.

    @staticmethod
    def _on_passage(passage, xy):
        """(along, left of) the taught line, metres from its start."""
        x, y, h = passage['start']
        c, s = math.cos(h), math.sin(h)
        dx, dy = xy[0] - x, xy[1] - y
        return dx * c + dy * s, -dx * s + dy * c

    def _passage_start(self, ctx):
        """Switch to a taught passage if the fresh route goes through one (in its direction)."""
        if self.args.get('no_passages') or not getattr(ctx, 'passages', None):
            return False
        route = self.info['route']
        if len(route) < 2:
            return False
        done = self.info.setdefault('passages_done', [])
        for passage in ctx.passages:
            name = passage.get('name', '')
            if name in done:
                continue
            gate = self._on_passage(passage, passage['gate'])[0]
            along, left = self._on_passage(passage, ctx.pose)
            # Already past the doorway — on the line beyond its gate — or the goal is before it: nothing to take.
            # 09.10 11:25: standing in the kitchen 1 m beside the line, level with the gate, counted as «past»; the
            # plain route went across the doorway at a slant and wedged in the L-turn for four minutes.
            if ((along > gate - .15 and abs(left) < PASSAGE_SIDE_M)
                    or self._on_passage(passage, route[-1])[0] < gate + .2):
                continue
            h = passage['start'][2]
            through = False
            for a, b in zip(route, route[1:]):
                step = math.dist(a, b)
                if step < 1e-6 or point_segment_distance(passage['gate'], a, b) > PASSAGE_GATE_M:
                    continue
                through = (b[0] - a[0]) * math.cos(h) + (b[1] - a[1]) * math.sin(h) > .5 * step
                if through:
                    break
            if not through:
                continue
            self.info['passage'] = dict(passage)
            self.info['passage_name'] = name
            for key in ('passage_nav', 'passage_from', 'passage_runs', 'passage_headway'):
                self.info.pop(key, None)
            self.stage = 'passage_approach'
            return True
        return False

    def _passage(self, ctx):
        passage = self.info['passage']
        pose = ctx.pose
        along, left = self._on_passage(passage, pose)
        x, y, h = passage['start']
        if self.stage == 'passage_approach':
            back = float(passage.get('approach', .4))
            # Near the line and before its narrow part with room to settle onto it (e**-3d): go now.
            settle = math.log(max(abs(left), .02) / .02) / PASSAGE_STEER
            if abs(left) < .2 and -back - .3 < along and along + settle < float(passage.get('tight_from', 0.)):
                return self._passage_run_start(ctx)
            nav = self.info.get('passage_nav')
            if nav is None:
                point = [x - back * math.cos(h), y - back * math.sin(h)]
                nav = self.info['passage_nav'] = Navigate({'xy': point, 'no_passages': True,
                                                           'speed': self.args.get('speed', CRUISE)})
                nav.started = ctx.now
            intent = nav.tick(ctx)
            if nav.status == 'done':
                return self._passage_run_start(ctx)
            if nav.status in ('failed', 'cancelled'):
                return self._passage_give_up(ctx, 'approach_' + nav.reason)
            return intent
        steer = max(-PASSAGE_STEER_MAX, min(PASSAGE_STEER_MAX, PASSAGE_STEER * left))
        if self.stage == 'passage_back':
            # Straight back along the same line (backing, the heading that closes the gap is mirrored).
            if along <= self.info['passage_back_to'] + .02 or ctx.now - self.info['passage_back_since'] > 15.:
                if self.info.get('passage_runs', 0) >= PASSAGE_TRIES:
                    return self.fail('passage_blocked')      # out of the doorway first, then say why
                return self._passage_run_start(ctx)
            if ctx.blocked_for() > 3 * PASSAGE_STUCK_S:
                return self.fail('passage_blocked')          # boxed in both ways: say so
            turn = wrap(h + steer - pose[2])
            return Intent(-PASSAGE_BACK_V, 0., True, wrap(ctx.odom[2] + turn))
        # Run: along the line, steering back onto it by the map pose (not a held odometry heading: the
        # threshold kicks it a few degrees, 23:45 it ended 5 cm off the line).
        goal = self.info['route'][-1]
        if math.dist(pose[:2], goal) < GOAL_TOLERANCE:
            return self.arrived(ctx)
        if along >= float(passage['length']) - .03:
            self.info['passages_done'].append(self.info.pop('passage_name', ''))
            self.info.pop('passage', None)
            self._plan(ctx)
            return STOP
        turn = wrap(h - steer - pose[2])
        heading = wrap(ctx.odom[2] + turn)
        if abs(turn) > PASSAGE_PIVOT:
            self.info['passage_headway'] = (ctx.now, along)     # turning onto the line is not being stuck
            if ctx.blocked_for() > PASSAGE_STUCK_S:
                return self._passage_retry(ctx, 'turn_blocked')
            return Intent(0., turn_toward(turn), True, heading)
        headway = self.info.get('passage_headway')
        if headway is None or along > headway[1] + .03:
            self.info['passage_headway'] = headway = (ctx.now, along)
        if ctx.blocked_for() > PASSAGE_STUCK_S or ctx.now - headway[0] > PASSAGE_STUCK_S:
            return self._passage_retry(ctx, 'blocked' if ctx.blocked_for() else 'stalled')
        return Intent(float(passage.get('speed', CRUISE)), 0., True, heading)

    def _passage_run_start(self, ctx):
        self.info.pop('passage_nav', None)
        self.info.pop('passage_headway', None)
        self.info['passage_since'] = ctx.now
        if 'passage_from' not in self.info:
            self.info['passage_from'] = min(0., self._on_passage(self.info['passage'], ctx.pose)[0])
        self.stage = 'passage_run'
        return STOP

    def _passage_retry(self, ctx, reason):
        """Did not get through: reverse (no turning) along the line to where the run began, then go again."""
        self.info['passage_runs'] = self.info.get('passage_runs', 0) + 1
        self.info['passage_failed'] = f"{self.info.get('passage_name', '')}:{reason}"
        self.info['passage_back_to'] = self.info.get('passage_from', 0.)
        self.info['passage_back_since'] = ctx.now
        self.stage = 'passage_back'
        return STOP

    def _passage_give_up(self, ctx, reason):
        """Could not even get onto the line: plain planning; the passage is tried once more on the next plan."""
        name = self.info.pop('passage_name', '')
        self.info.pop('passage', None)
        self.info.pop('passage_nav', None)
        tries = self.info.setdefault('passage_tries', {})
        tries[name] = tries.get(name, 0) + 1
        self.info['passage_failed'] = f'{name}:{reason}'
        if tries[name] >= 2:
            self.info.setdefault('passages_done', []).append(name)
        self._plan(ctx)
        return STOP

    def _tight_corner(self, ctx, ahead):
        """(vertex, pivot point, outgoing heading) of the next sharp bend in a narrow place within reach.

        The pivot point is the roomiest cell next to the bend (a turn on the spot needs BODY_RADIUS free).
        """
        done = self.info.get('corners', [])
        grid = ctx.grid
        clearance = grid.clearance
        prev = tuple(ctx.pose[:2])
        walked = 0.
        points = [tuple(p) for p in ahead[1:]]
        for i in range(len(points) - 1):
            vertex, nxt = points[i], points[i + 1]
            step = math.dist(prev, vertex)
            walked += step
            if walked > CORNER_LOOK_M:
                return None
            if step < .05:
                prev = vertex
                continue
            h_in = math.atan2(vertex[1] - prev[1], vertex[0] - prev[0])
            h_out = math.atan2(nxt[1] - vertex[1], nxt[0] - vertex[0])
            prev = vertex
            if abs(wrap(h_out - h_in)) < CORNER_TURN or any(math.dist(vertex, c) < .2 for c in done):
                continue
            cell = grid.cell(vertex)
            if not grid.inside(cell) or clearance[cell] >= CORNER_TIGHT:
                continue
            # The pivot: the cell nearest the bend with room to turn (a little margin), else the roomiest one.
            k = int(math.ceil(.25 / grid.resolution))
            r0, c0 = cell
            near, roomy = None, None
            for r in range(max(0, r0 - k), min(grid.height, r0 + k + 1)):
                for c in range(max(0, c0 - k), min(grid.width, c0 + k + 1)):
                    room = float(clearance[r, c])
                    if room < BODY_RADIUS + .01:
                        continue
                    xy = grid.xy((r, c))
                    gap = math.dist(xy, vertex)
                    if room >= BODY_RADIUS + .03 and (near is None or gap < near[0]):
                        near = (gap, xy)
                    if roomy is None or room - .3 * gap > roomy[0]:
                        roomy = (room - .3 * gap, xy)
            if near is None and roomy is None:
                return None                        # no room to turn here: the maneuver planner takes it
            pivot = (near or roomy)[1]
            return vertex, pivot, math.atan2(nxt[1] - pivot[1], nxt[0] - pivot[0])
        return None

    def _slipping(self, ctx):
        """Wheels turning forward but the measured pose not moving: a threshold or a rug edge."""
        if ctx.twist[0] < .05:
            self.info.pop('slip', None)
            return False
        slip = self.info.get('slip')
        if slip is None or math.dist(ctx.pose[:2], slip[1][:2]) > .04:
            self.info['slip'] = (ctx.now, ctx.pose)
            return False
        return ctx.now - slip[0] > 1.5

    def _runup_start(self, ctx, route, heading=None, aim=False):
        """Back off, (optionally) face the route beyond the obstacle, then take it at full speed."""
        key = str(self.target(ctx))
        runups = self.info.setdefault('runups', {})
        if runups.get(key, 0) >= RUNUP_PER_TARGET or not ctx.scan_fresh:
            return False
        if heading is None:
            ahead = remaining_path(ctx.pose, route)
            point = lookahead_point(ahead, .7)
            bearing = math.atan2(point[1] - ctx.pose[1], point[0] - ctx.pose[0])
            if not aim and abs(wrap(bearing - ctx.pose[2])) > RUNUP_MAX_BEARING:
                return False
            heading = bearing if aim else ctx.pose[2]
        if not ctx.clear(-.08, 0., 2.):
            return False
        runups[key] = runups.get(key, 0) + 1
        self.info['runup'] = {'origin': ctx.odom, 'started': ctx.now, 'map_heading': heading,
                              'back_heading': ctx.odom[2], 'route_heading': heading}
        self.info.pop('slip', None)
        self.stage = 'runup_back'
        return True

    def _runup(self, ctx):
        run = self.info['runup']
        progress = relative(run['origin'], ctx.odom)[0]
        ctx.live_only(True)
        if self.stage == 'runup_back':
            # Back off far enough that turning towards the route is free (the jamb by a front corner
            # stopped the aim half-way, 23:15) — up to RUNUP_BACK_MAX_M if the room behind allows.
            turn = 1. if wrap(run['route_heading'] - ctx.pose[2]) > 0 else -1.
            pivot_free = ctx.clear(0., turn * .4, 2.)
            far_enough = progress <= -RUNUP_BACK_M and pivot_free
            if far_enough or progress <= -RUNUP_BACK_MAX_M or ctx.now - run['started'] > 9. or ctx.blocked_for() > .5:
                self.stage = 'runup_aim'
                run.update(origin=ctx.odom, started=ctx.now)
                # Backed off: pick the straight line through the gap (not along a jamb), seen live now.
                run['map_heading'] = self._clear_heading(ctx, run['route_heading'])
                return STOP
            return Intent(-.12, 0., True, run['back_heading'])
        if self.stage == 'runup_aim':
            error = wrap(run['map_heading'] - ctx.pose[2])
            if abs(error) < .05 or ctx.now - run['started'] > 6. or ctx.blocked_for() > 1.:
                self.stage = 'runup_charge'
                run.update(origin=ctx.odom, started=ctx.now, heading=wrap(ctx.odom[2] + error))
                return STOP
            return Intent(0., turn_toward(error, minimum=.1), True, wrap(ctx.odom[2] + error))
        if progress >= RUNUP_CHARGE_M or ctx.now - run['started'] > 6. or ctx.blocked_for() > .3:
            ctx.live_only(False)
            self.info.pop('runup', None)
            self.stage = 'driving'
            self.info['progress'] = (ctx.now, ctx.pose)
            if self.info.get('door'):
                self.stage = 'door'
                self.info['door']['phase'] = 'through'
                self.info['door']['origin'] = self.info['door'].get('origin', ctx.odom)
            return STOP
        return Intent(THROUGH_SPEED, 0., True, run['heading'])

    def _clear_heading(self, ctx, wanted):
        """The heading nearest `wanted` (within RUNUP_FAN) whose straight run of RUNUP_CLEAR_M is free."""
        from .guard import swept_hit
        points = ctx.points()
        if not len(points):
            return wanted
        offsets = sorted((k * .087 for k in range(-int(RUNUP_FAN / .087), int(RUNUP_FAN / .087) + 1)), key=abs)
        base = wrap(wanted - ctx.pose[2])            # wanted direction in the robot frame
        best, best_margin = None, -1.
        for offset in offsets:
            angle = base + offset
            c, s = math.cos(-angle), math.sin(-angle)
            rotated = points @ np.array([[c, s], [-s, c]])          # the world turned so that `angle` is ahead
            if swept_hit(rotated, THROUGH_SPEED, 0., RUNUP_CLEAR_M / THROUGH_SPEED, .03):
                continue
            # Prefer the middle of the gap: the most room on both sides of the line.
            ahead = rotated[(rotated[:, 0] > 0) & (rotated[:, 0] < RUNUP_CLEAR_M + .2)]
            margin = float(np.min(np.abs(ahead[:, 1]))) if len(ahead) else 1.
            score = margin - .15 * abs(offset)
            if score > best_margin:
                best, best_margin = angle, score
        return wanted if best is None else wrap(ctx.pose[2] + best)

    def _plan_maneuver(self, ctx, ahead):
        """Search a short parking-style sequence to a route point beyond the tight spot."""
        target = lookahead_point(ahead, 1.0)
        further = lookahead_point(ahead, 1.3)
        heading = math.atan2(further[1] - target[1], further[0] - target[0]) \
            if math.dist(target, further) > .05 else ctx.pose[2]
        goal = relative(ctx.pose, (target[0], target[1], heading))
        points = ctx.points()
        self.info['maneuver_future'] = ctx.submit(maneuver.plan, points, goal)
        self.info['maneuver_origin'] = ctx.odom
        self.info['maneuvers'] = self.info.get('maneuvers', 0) + 1
        self.stage = 'maneuver_plan'
        return STOP

    def _maneuver(self, ctx):
        if self.stage == 'maneuver_plan':
            future = self.info['maneuver_future']
            if not future.done():
                return STOP
            moves = future.result()
            if not moves:
                self.stage = 'driving'
                self.info['progress'] = (ctx.now, ctx.pose)
                return self._no_route(ctx, 'no_maneuver')
            # Receding horizon: drive the first few moves, then look again.
            self.info['moves'] = list(moves[:4])
            self.info['move_started'] = None
            self.stage = 'maneuver'
        moves = self.info['moves']
        if not moves:
            self.stage = 'driving'
            self.info['progress'] = (ctx.now, ctx.pose)
            return STOP
        v, w, seconds = moves[0]
        if self.info['move_started'] is None:
            self.info['move_started'] = ctx.now
        if ctx.now - self.info['move_started'] >= seconds:
            moves.pop(0)
            self.info['move_started'] = None
            return STOP
        if ctx.blocked_for() > .8:
            self.info['moves'] = []
            self.stage = 'driving'
            return STOP
        return Intent(v, w)

    PIVOT_STUCK_S = 2.
    PIVOT_MOVED_RAD = .05        # ~3°: a pivot that turned this much is spinning up, not rubbing

    def _pivot_stuck(self, ctx):
        """A turn on the spot commanded for a while without the heading moving (rubbing on something).

        Not the wheel speed: 10.10 the wheels often spun up slowly (0.03-0.09 rad/s after 1.2 s of 0.42 asked) while
        the heading did turn, and each such pivot became a back-off, three in a row at the same spot."""
        since = self.info.get('pivot_stuck')
        if since is None or ctx.now - since <= self.PIVOT_STUCK_S:
            return False
        start = self.info.get('pivot_yaw')
        return start is None or abs(wrap(ctx.odom[2] - start)) < self.PIVOT_MOVED_RAD

    def _tight(self, ctx):
        """Walls within reach of the corners, or the guard has been refusing: steer by search."""
        cell = ctx.grid.cell(ctx.pose)
        tight = ctx.grid.inside(cell) and ctx.grid.clearance[cell] < BODY_RADIUS + .10
        return tight or ctx.blocked_for() > .3

    def _begin_door(self, ctx, door):
        """Square up on the opening's axis before entering, then drive straight through its middle."""
        self.info['doors'].remove(door)
        center, axis = door['center'], door['axis']
        entry = None
        for back in (.55, .45, .65, .35):
            point = (center[0] - back * math.cos(axis), center[1] - back * math.sin(axis))
            if ctx.grid.body_fits(point, margin=.03):
                entry = point
                break
        if entry is None:
            return                                # no room to square up: drive it normally
        self.info['door'] = {'center': center, 'axis': axis, 'entry': entry, 'phase': 'entry',
                             'started': ctx.now, 'width': door.get('width'), 'retries': 0}
        self.stage = 'door'

    def _door(self, ctx):
        door = self.info['door']
        self.info['door_phase'] = door['phase']
        self.info['doors_crossed'] = len(self.info.get('crossed', []))
        pose = ctx.pose
        frame = (*door['center'], door['axis'])
        if ctx.now - door['started'] > 40.:
            return self._door_done(ctx)
        if door['phase'] == 'entry':
            on_axis = relative(frame, pose)
            if abs(on_axis[1]) < .04 and abs(wrap(door['axis'] - pose[2])) < .08 and on_axis[0] < -.15:
                return self._through(ctx, door, on_axis)
            local = relative(pose, (*door['entry'], 0.))
            distance = math.hypot(local[0], local[1])
            if distance < .04:
                door['phase'] = 'align'
                return STOP
            error = math.atan2(local[1], local[0])
            if abs(error) > 2.6 and distance < .25:          # overshot a little: back onto it
                return Intent(-.06, 0., True, ctx.odom[2])
            if abs(error) > .12:
                return Intent(0., turn_toward(error), True, wrap(ctx.odom[2] + error))
            return Intent(min(.2, .05 + .8 * distance), max(-.3, min(.3, 1.5 * error)))
        if door['phase'] == 'align':
            error = wrap(door['axis'] - pose[2])
            if abs(error) < .04 or (ctx.blocked_for() > 1.5 and abs(error) < .3):
                return self._through(ctx, door, relative(frame, pose))
            if ctx.blocked_for() > 2.:
                return self._door_done(ctx)          # cannot turn here: let the route follower handle it
            return Intent(0., turn_toward(error, minimum=.08), True, wrap(ctx.odom[2] + error))
        # through: straight along the axis (the board's gyro holds the heading), gentle centring
        progress = relative(door['origin'], ctx.odom)[0]
        if progress >= door['length']:
            return self._door_done(ctx)
        on_axis = relative(frame, pose)
        heading = wrap(ctx.odom[2] + wrap(door['axis'] - pose[2]) - max(-.12, min(.12, 1.2 * on_axis[1])))
        if ctx.blocked_for() > 2.:
            if on_axis[0] > -.05:
                return self._door_done(ctx)          # the body is through the opening
            door['retries'] += 1
            if door['retries'] > 1:
                return self._door_done(ctx)
            door['phase'] = 'entry'
            return STOP
        if self._slipping(ctx):
            target_heading = door['axis']
            self.stage = 'driving'
            if self._runup_start(ctx, None, heading=target_heading):
                return STOP
            self.stage = 'door'
        return Intent(THROUGH_SPEED, 0., True, heading)

    def _through(self, ctx, door, on_axis):
        door['phase'] = 'through'
        door['origin'] = ctx.odom
        door['length'] = -on_axis[0] + .3     # body centre 0.3 m past the opening, then the route resumes
        return STOP

    def _door_done(self, ctx):
        door = self.info.pop('door', None)
        if door:
            self.info.setdefault('crossed', []).append(door['center'])
        self.stage = 'driving'
        self.info['progress'] = (ctx.now, ctx.pose)
        return STOP

    def arrived(self, ctx):
        return self.done()


class GoToRoom(Navigate):
    kind = 'go_room'

    def target(self, ctx):
        return tuple(self.args['xy'])


class Approach(Navigate):
    """Go to a viewpoint `distance` from a map point, then face it."""
    kind = 'approach'

    def _plan(self, ctx):
        target = tuple(self.args['xy'])
        self.info['future'] = ctx.plan_ring(ctx.pose, target, float(self.args.get('distance', .6)))
        self.stage = 'planning'

    def arrived(self, ctx):
        target = self.args['xy']
        error = wrap(math.atan2(target[1] - ctx.pose[1], target[0] - ctx.pose[0]) - ctx.pose[2])
        if abs(error) < .06:
            return self.done()
        self.stage = 'face'
        return Intent(0., turn_toward(error), True, wrap(ctx.odom[2] + error))

    def tick(self, ctx):
        if self.stage == 'face':
            return self.arrived(ctx)
        if self.missing(ctx):
            return self.fail('not_there')
        return super().tick(ctx)

    def missing(self, ctx):
        """A thing that can be taken away (`label`) is checked on the way, like a hand reaching for a cup: in good
        view and light it should be seen; ten frames there and never once — it is gone, no use driving on
        (09.10, the owner; IntentNav, OpenBelief-Nav: a remembered object is a hypothesis until confirmed)."""
        label = self.args.get('label')
        if not label or self.info.get('confirmed') or not hasattr(ctx, 'sees'):
            return False
        target = self.args['xy']
        dx, dy = target[0] - ctx.pose[0], target[1] - ctx.pose[1]
        bearing = wrap(math.atan2(dy, dx) - ctx.pose[2])
        if not VERIFY_NEAR_M <= math.hypot(dx, dy) <= VERIFY_FAR_M or abs(bearing) > VERIFY_BEARING:
            return False
        seen = ctx.sees(label, bearing)
        if seen is None:
            return False
        frame, hit, light = seen
        if (light is not None and light < VERIFY_LIGHT) or frame == self.info.get('verify_frame'):
            return False
        self.info['verify_frame'] = frame
        if hit:
            self.info['confirmed'] = True
            return False
        self.info['missed'] = self.info.get('missed', 0) + 1
        return self.info['missed'] >= VERIFY_FRAMES


# ---------------------------------------------------------------- docking
class Undock(Skill):
    kind = 'undock'

    def tick(self, ctx):
        if self.stage == 'start':
            self.info['origin'] = ctx.odom
            self.stage = 'leaving'
        if not ctx.scan_fresh:
            return STOP
        progress = relative(self.info['origin'], ctx.odom)[0]
        if progress >= float(self.args.get('distance', .22)):
            return self.done()
        if ctx.now - self.started > 12.:
            return self.fail('undock_blocked')
        return Intent(.10, 0., True, self.info['origin'][2])


class Dock(Skill):
    """Return to the charger: staging point, square up, reverse onto the contacts."""
    kind = 'dock'

    def tick(self, ctx):
        cfg = ctx.config
        if ctx.charging:
            return self._contact(ctx)
        self.info.pop('charge_since', None)
        if self.info.pop('resting', None):
            # Contact lost while settling: push back onto the contacts.
            self.lidar_needed = True
            self.info['reseat'] = self.info.get('reseat', 0) + 1
            if self.info['reseat'] > 3:
                return self.fail('contact_unstable')
            self.stage = 'approach'
            self.info['approach_started'] = ctx.now
            self.info['best'] = (ctx.now, 9.)
        elif self.stage == 'charging':
            # A touch that never held: 09.10 18:17 the springy contacts threw the robot 3 cm off after a moment,
            # and with no branch for «charging» without contact it stood there, not charging, for good. Press again.
            self.info['reseat'] = self.info.get('reseat', 0) + 1
            if self.info['reseat'] > 3:
                # Still flickering (20:20:55-59: three presses in 4 s): line up afresh — the mind's new come-round
                # docked at once (20:21:20) — instead of failing the owner's «на базу».
                self.info['reseat'] = 0
                return self._retry(ctx, 'contact_unstable')
            self.stage = 'press'
            self.info['press_started'] = ctx.now
            self.info.setdefault('press_heading', ctx.odom[2])
        dock = ctx.dock
        staging = compose(dock, (cfg['dock_staging_distance'] + self.info.get('extra', 0.), self.info.get('offset', 0.), 0.))
        if self.stage == 'start':
            if not ctx.localized:
                since = self.info.setdefault('lost_since', ctx.now)
                return STOP if ctx.now - since < 30. else self.fail('lost')
            x, y, yaw = relative(dock, ctx.pose)
            detour = self.info.pop('detour', False)
            if not detour and .05 <= x <= .45 and abs(y) <= .03 and abs(yaw) <= .1:
                self.stage = 'align'
            elif not detour and x < 1.0 and abs(y) < .3 and math.dist(ctx.pose[:2], staging[:2]) < .6:
                self.stage = 'position'
            else:
                self.info['nav'] = Navigate({'xy': staging[:2], 'tolerance': .1 if detour else .3, 'speed': CRUISE})
                self.info['nav'].started = ctx.now
                self.stage = 'route'
        if self.stage == 'route':
            nav = self.info['nav']
            intent = nav.tick(ctx)
            if nav.status == 'done':
                self.stage = 'position'
                return STOP
            if nav.status == 'failed':
                extra = self.info['extra'] = self.info.get('extra', 0.) + .1
                if extra > .4:
                    return self.fail('dock_unreachable:' + nav.reason)
                self.stage = 'start'
                return STOP
            return intent
        pose = self._pose(ctx)
        if self.stage in ('position', 'align') and ctx.blocked_for() > DOCK_BLOCKED_S:
            return self._detour(ctx)
        if self.stage == 'position':
            local = relative(pose, (staging[0], staging[1], pose[2]))
            distance = math.hypot(local[0], local[1])
            if distance < .02:
                self.stage = 'align'
                return STOP
            backwards = local[0] < 0
            bearing = wrap(math.atan2(local[1], local[0]) - (math.pi if backwards else 0.))
            w = max(-.54, min(.54, 1.7 * bearing))
            v = 0. if abs(bearing) > .45 else min(.10, max(.045, .7 * distance))
            return Intent(-v if backwards else v, w)
        if self.stage == 'align':
            x, y, _ = relative(dock, pose)
            error = wrap(dock[2] - pose[2])
            if abs(error) < DOCK_ALIGN_TOL:
                if abs(y) > DOCK_LATERAL and abs(x) > .25 and self.info.get('squares', 0) < 2:
                    self.info['squares'] = self.info.get('squares', 0) + 1
                    self.stage = 'position'
                    return STOP
                self.stage = 'approach'
                self.info['approach_started'] = ctx.now
                self.info['best'] = (ctx.now, 9.)
                return STOP
            # The bulk of the half-turn at the normal turn speed; the last ~30° stay slow and precise.
            limit = TURN if abs(error) > .5 else DOCK_ALIGN_W
            return Intent(0., turn_toward(error, limit=limit, gain=1.5, minimum=.08), True,
                          wrap(ctx.odom[2] + error))
        if self.stage == 'approach':
            x, y, yaw = relative(dock, pose)
            remaining = cfg['dock_overtravel'] + x
            if ctx.now - self.info['approach_started'] > 30.:
                return self._retry(ctx, 'approach_timeout')
            best_t, best = self.info['best']
            if remaining < best - .02:
                self.info['best'] = (ctx.now, remaining)
            elif ctx.now - best_t > 8.:
                return self._retry(ctx, 'no_progress')
            if ctx.dock_fix and abs(y) > .025 and remaining < .20:
                return self._retry(ctx, 'lateral_miss')
            if remaining <= .01:
                # The contacts are springy: the last centimetres at 2.5 cm/s stalled short of them and every
                # stall cost a full come-round (07.10 22:35: three tries). Press straight back a little more.
                self.stage = 'press'
                self.info['press_started'] = ctx.now
                self.info['press_heading'] = ctx.odom[2]
                return STOP
            correction = math.atan2(y, max(.25, abs(x))) * min(1., abs(x) / .12)
            error = wrap(correction - yaw)
            return Intent(-min(.08, max(.04, remaining * .7)), max(-.15, min(.15, 1.7 * error)))
        if self.stage == 'press':
            if ctx.now - self.info['press_started'] < DOCK_PRESS_S:
                return Intent(-DOCK_PRESS_V, 0., True, self.info['press_heading'])
            self.stage = 'contact_wait'
            self.info['wait_started'] = ctx.now
            return STOP
        if self.stage == 'contact_wait':
            x, _, _ = relative(dock, pose)
            if ctx.dock_fix and cfg['dock_overtravel'] + x > .02:
                self.stage = 'approach'
                self.info['best'] = (ctx.now, cfg['dock_overtravel'] + x)    # progress counts from here
                return STOP
            if ctx.now - self.info['wait_started'] > 2.:
                return self._retry(ctx, 'no_contact')
            return STOP
        return STOP

    def _pose(self, ctx):
        """Near the dock trust the charging panorama; between fixes, odometry from the last fix."""
        if ctx.dock_fix is not None:
            self.info['anchor'] = (ctx.dock_fix, ctx.dock_fix_odom)
        anchor = self.info.get('anchor')
        if anchor and ctx.now - anchor[0][3] < 10.:
            fix, odom = anchor
            return compose(fix[:3], relative(odom, ctx.odom))
        return ctx.pose

    def _detour(self, ctx):
        """Something stands by the dock (07.10: a chair caster held it 2 min 16 s in silence): square up
        further out, where turning is free, and back in straight from there; say so if even that fails."""
        extra = self.info['extra'] = self.info.get('extra', 0.) + .15
        if extra > .5:
            return self.fail('dock_blocked')
        self.info['detour'] = True
        self.info.pop('squares', None)
        self.stage = 'start'
        return STOP

    def _retry(self, ctx, reason):
        self.info['retry'] = reason                 # telemetry: why the last come-round happened
        attempt = self.info['attempt'] = self.info.get('attempt', 0) + 1
        if attempt >= 3:
            return self.fail('dock_failed:' + reason)
        self.info['offset'] = .04 if attempt == 1 else -.04
        self.info.pop('squares', None)
        self.stage = 'position'
        return STOP

    def _contact(self, ctx):
        """Qualify the charge twice: on contact, then again in the resting state (lidar off).

        Leaving navigation mode can drop a marginal contact (the old system saw it too):
        success is only declared once the charge holds in exactly the state it will rest in.
        """
        still = max(abs(ctx.twist[0]), abs(ctx.twist[1])) < .005
        since = self.info.setdefault('charge_since', ctx.now)
        if not still:
            self.info['charge_since'] = ctx.now
        self.stage = 'charging'
        if ctx.now - since >= 3.:
            if not self.info.get('resting'):
                self.info['resting'] = True
                self.lidar_needed = False
                self.info['charge_since'] = ctx.now
                return STOP
            return self.done()
        return STOP


# ---------------------------------------------------------------- looking and people
# Emotions with the body (owner 08.10: «трястись от радости»). Targets, not timed commands: the wheels answer
# ~0.3 s late, so 0.22 s pulses barely moved the robot (08.10 11:25, twist 0.07 of 0.55 rad/s). Each emote is a
# list of headings (radians from the start) or distances (metres along the start heading) to reach in turn;
# a target counts as reached when close or after STEP_S, the guard checks every tick.
EMOTES = {
    'wiggle': ('turn', (.22, -.22, .22, -.22, 0.)),          # joy: a happy wag
    'shake': ('turn', (.3, -.3, 0.)),                         # no
    'shiver': ('turn', (.09, -.09, .09, -.09, .09, 0.)),      # fright
    'spin': ('spin', (math.pi, 2 * math.pi)),                  # delight: a whole twirl
    'bounce': ('move', (.06, 0., .06, 0.)),                    # excitement
    'nod': ('move', (.035, 0., .035, 0.)),                     # yes
    'bow': ('move', (.07, .07, 0.)),                           # a bow: forward, a pause, back
    # A tremble (owner 08.10: «супер мелкие движения налево-направо»): full-rate turn pulses flipping faster
    # than the wheels can follow (they answer ~0.3 s late), so the body only shivers in place.
    'tremble': ('pulse', (.008, 8)),                           # (swing metres, direction changes)
}


class Emote(Skill):
    """One body emotion (EMOTES), ending at the starting heading and place."""
    kind = 'emote'
    STEP_S = 1.3

    def tick(self, ctx):
        mode, targets = EMOTES.get(str(self.args.get('kind', 'wiggle')), (None, ()))
        if mode is None:
            return self.fail('unknown_emote')
        if self.stage == 'start':
            self.info.update(origin=ctx.odom, index=0, step_at=ctx.now, turned=0., last=ctx.odom[2])
            self.stage = 'moving'
        if ctx.blocked_for() > .6:
            return self.done('no_room')          # something close by: an emotion is not worth a bump
        origin = self.info['origin']
        self.info['turned'] += wrap(ctx.odom[2] - self.info['last'])
        self.info['last'] = ctx.odom[2]
        if mode == 'pulse':
            flips_wanted, limit_s = int(self.args.get('flips', targets[1])), float(self.args.get('seconds', 4.))
            along = relative(origin, ctx.odom)[0]
            if self.stage == 'moving':
                # Bang-bang around the spot on odometry: past +1.5 cm go back, past -1.5 cm go forward. Timed
                # pulses crept 8-33 cm forward (the board answers ~0.35 s late and not symmetrically, 08.10).
                swing = float(self.args.get('swing', targets[0]))
                sign = self.info.get('sign', 1.)
                if sign > 0 and along > swing or sign < 0 and along < -swing:
                    sign = -sign
                    self.info['flips'] = self.info.get('flips', 0) + 1
                self.info['sign'] = sign
                if self.info.get('flips', 0) < flips_wanted and ctx.now - self.info['step_at'] < limit_s:
                    return Intent(float(self.args.get('speed', .08)) * sign, 0.)   # 08.10: +-2 cm, 5 flips/4 s
                self.stage = 'return'
            if abs(along) > .01 and ctx.now - self.info['step_at'] < limit_s + 4.:
                return Intent(math.copysign(min(.15, max(.05, 3. * abs(along))), -along), 0., True, origin[2])
            error = wrap(origin[2] - ctx.odom[2])
            if abs(error) < .05 or ctx.now - self.info['step_at'] > limit_s + 6.:
                return self.done()
            return Intent(0., turn_toward(error), True, origin[2])
        index = self.info['index']
        if index >= len(targets):
            return self.done()
        target = targets[index]
        if mode == 'move':
            error = target - relative(origin, ctx.odom)[0]
            reached = abs(error) < .008
            speed = math.copysign(min(.2, max(.05, 3. * abs(error))), error)
            intent = Intent(speed, 0., True, origin[2])
        else:
            error = (target - self.info['turned']) if mode == 'spin' else wrap(origin[2] + target - ctx.odom[2])
            reached = abs(error) < .05
            intent = Intent(0., math.copysign(min(.6, max(.18, 4. * abs(error))), error))
        if 'limit' not in self.info:
            # A big step (a half twirl) gets its time: size over a fair speed, plus the wheels' slow start.
            self.info['limit'] = max(self.STEP_S, abs(error) / (.1 if mode == 'move' else .45) + .6)
        if reached or ctx.now - self.info['step_at'] > self.info['limit']:
            self.info['index'], self.info['step_at'] = index + 1, ctx.now
            self.info.pop('limit', None)
            return STOP
        return intent


class Dance(Skill):
    """Sway to music: turns left and right on the spot around the starting heading, then face it again."""
    kind = 'dance'
    AMPLITUDE = .25          # rad either way
    HALF_PERIOD_S = 1.       # within the board's .6 rad/s

    def tick(self, ctx):
        if self.stage == 'start':
            self.info['origin'] = ctx.odom[2]
            self.stage = 'dancing'
        origin = self.info['origin']
        if ctx.blocked_for() > 1.5:
            return self.done('no_room')          # something by the corners: a dance is not worth a bump
        if self.stage == 'dancing' and ctx.now - self.started < float(self.args.get('seconds', 12.)):
            phase = ((ctx.now - self.started) / self.HALF_PERIOD_S) % 2.
            target = wrap(origin + (self.AMPLITUDE if phase < 1. else -self.AMPLITUDE))
            error = wrap(target - ctx.odom[2])
            return Intent(0., max(-.55, min(.55, 3. * error)))
        self.stage = 'finishing'
        error = wrap(origin - ctx.odom[2])
        if abs(error) < .05:
            return self.done()
        return Intent(0., turn_toward(error), True, origin)


class Perform(Skill):
    """A little number (GrowBot's «Disney mode», 11.10): up to MAX_STEPS steps, each a turn, a short move, an
    emotion or a pause. Each step is cued to the mind as it starts (info['cues'] -> body.event 'cue'), so its words,
    sound and face come in the same second as the motion, not a thought later. A step lasts at least its pause."""
    kind = 'perform'
    MAX_STEPS = 8
    MOTION = ('turn', 'move', 'emote')

    @property
    def lidar_needed(self):
        # Words and sounds only: no reason to leave the dock or spin the lidar.
        return any(isinstance(s, dict) and any(k in s for k in self.MOTION) for s in self.args.get('steps') or [])

    def child_for(self, step):
        if 'turn' in step:
            return Rotate({'angle': float(step['turn'])})
        if 'move' in step:
            return Drive({'distance': float(step['move'])})
        if 'emote' in step:
            return Emote({'kind': str(step['emote'])})
        return Wait({'seconds': 0.})

    def tick(self, ctx):
        steps = [s for s in (self.args.get('steps') or []) if isinstance(s, dict)][:self.MAX_STEPS]
        if self.stage == 'start':
            self.stage, self.info['index'] = 'step', -1
        child = self.info.get('child')
        if child is not None:
            if child.status is None:
                intent = child.tick(ctx)
                if child.status is None:
                    return intent
            if child.status == 'failed':
                return self.fail(child.reason)
            step = steps[self.info['index']]
            try:
                pause = max(0., min(4., float(step.get('pause', 0.))))
            except (TypeError, ValueError):
                pause = 0.
            if ctx.now - self.info['step_at'] < pause:
                return STOP                          # the words of this step are still being said
            self.info['child'] = None
        index = self.info['index'] + 1
        if index >= len(steps):
            return self.done()
        self.info['index'], self.info['step_at'] = index, ctx.now
        self.info.setdefault('cues', []).append(index)
        child = self.info['child'] = self.child_for(steps[index])
        child.started = ctx.now
        intent = child.tick(ctx)
        return intent if child.status is None else STOP


class LookAround(Rotate):
    kind = 'look_around'

    def tick(self, ctx):
        if 'angle' not in self.args:
            self.args['angle'] = random.choice((1, -1)) * 2 * math.pi
        return super().tick(ctx)


class Track(Skill):
    """Keep a person/dog in view (watch), follow it, come up to it (seek: look round for it first, done once
    there), or search for it by turning (done when it is in view)."""
    kind = 'track'

    def tick(self, ctx):
        mode = self.args.get('mode', 'watch')
        if ctx.now - self.started > float(self.args.get('seconds', 40.)):
            return self.done('time') if self.info.get('seen') else self.fail('not_seen')
        target = ctx.find(self.args.get('who', 'person'), self.info.get('track'))
        if target is not None and self.info.get('chase'):
            self.info.pop('chase', None)             # found again on the way: back to following
        if target is not None:
            self.info.pop('sweep', None)             # lost again later: a fresh look round from there
            if target.get('map_xy'):
                self.info['last_xy'] = target['map_xy']
            self.info['seen'] = self.info.get('seen', 0) + 1
            self.info['track'] = target.get('track')
            bearing = target['bearing_now']
            self.info['last_bearing'] = bearing
            heading = wrap(ctx.odom[2] + bearing)
            previous = self.info.get('last_heading')
            if previous is not None and ctx.now - self.info.get('last_seen', -9.) < 1.:
                # Which way it is moving across the view (rad/s, smoothed): where to look for it next.
                rate = wrap(heading - previous) / max(.05, ctx.now - self.info['last_seen'])
                self.info['drift'] = .7 * self.info.get('drift', 0.) + .3 * max(-1., min(1., rate))
            self.info['last_heading'] = heading
            self.info['last_seen'] = ctx.now
            if mode == 'search':
                return self.done('found')
            braking = bearing - ctx.twist[1] * .15
            error = math.copysign(max(0., abs(braking) - .06), braking)
            w = max(-.6, min(.6, 2.4 * error))
            if w * ctx.twist[1] < 0 and abs(ctx.twist[1]) > .05:
                w = 0.
            v = 0.
            distance = target.get('distance')
            if mode in ('follow', 'seek') and distance is not None:
                gap = distance - float(self.args.get('distance', .7))
                advancing = gap > (.04 if self.info.get('advancing') else .10)
                self.info['advancing'] = advancing
                if advancing:
                    v = min(CRUISE, .35 * gap) * max(0., 1 - abs(bearing) / .45)
                self.info['last_gap'] = gap
            if mode == 'seek':
                # Come up to them and stay: there (or too close to judge the distance) and facing them a moment.
                if not v and abs(bearing) < .3:
                    self.info['there_since'] = self.info.get('there_since') or ctx.now
                    if ctx.now - self.info['there_since'] > (.8 if distance is not None else 2.5):
                        return self.done('arrived')
                else:
                    self.info.pop('there_since', None)
            return Intent(v, w)
        # Lost or not yet seen: sweep toward where it was last seen.
        last = self.info.get('last_seen')
        if last is not None and ctx.now - last < TRACK_COAST_S and self.info.get('last_heading') is not None:
            # A frame or two without it is not «lost» (the eyes see it in some frames, not all): keep turning to
            # where it was heading and go on that way, as a dog keeps after a cat behind a chair.
            lead = self.info.get('drift', 0.) * min(1., ctx.now - last)
            error = wrap(self.info['last_heading'] + lead - ctx.odom[2])
            w = max(-.6, min(.6, 2.4 * error)) if abs(error) > .06 else 0.
            v = 0.
            if mode in ('follow', 'seek') and self.info.get('advancing') and abs(error) < .45:
                v = min(.15, .35 * max(0., self.info.get('last_gap', 0.))) * (1 - abs(error) / .45)
            return Intent(v, w)
        # Following and the person walked off (round a corner, through a door): drive to where they were
        # last seen and look there, like a dog would — not just spin on the spot and give up.
        if mode in ('follow', 'seek') and self.info.get('last_xy') and not self.info.get('chased'):
            chase = self.info.get('chase')
            if chase is None:
                chase = self.info['chase'] = Navigate({'xy': self.info['last_xy'], 'tolerance': .5})
                chase.started = ctx.now
            intent = chase.tick(ctx)
            if chase.status:
                self.info.pop('chase', None)
                self.info['chased'] = True           # arrived (or could not): now look around there
                return STOP
            return intent
        return self.sweep(ctx)

    def sweep(self, ctx):
        """Look round for it: turn a step, stand still a moment for sharp frames, turn on — one way round (09.10:
        a new random direction every tick made it shiver on the spot, 0.17 rad in 8 s); the other way if
        something keeps it from turning; at most one full turn."""
        sweep = self.info.get('sweep')
        if sweep is None:
            side = self.info.get('last_bearing') or self.args.get('bearing')
            sweep = self.info['sweep'] = {'dir': math.copysign(1., side) if side else random.choice((1., -1.)),
                                          'yaw': ctx.odom[2], 'turned': 0., 'step': 0., 'hold_until': 0.,
                                          'moved_at': ctx.now, 'reversed': False}
        delta = abs(wrap(ctx.odom[2] - sweep['yaw']))
        sweep['yaw'] = ctx.odom[2]
        sweep['turned'] += delta
        sweep['step'] += delta
        if delta > .004:
            sweep['moved_at'] = ctx.now
        if sweep['turned'] > 2 * math.pi:
            return self.fail('not_found') if not self.info.get('seen') else self.done('lost_sight')
        if ctx.now < sweep['hold_until']:
            sweep['moved_at'] = ctx.now
            return STOP
        if sweep['step'] >= SWEEP_STEP:
            sweep['step'] = 0.
            sweep['hold_until'] = ctx.now + SWEEP_HOLD_S
            return STOP
        if ctx.now - sweep['moved_at'] > SWEEP_STUCK_S:
            if sweep['reversed']:
                return self.fail('turn_blocked')
            sweep.update(dir=-sweep['dir'], reversed=True, moved_at=ctx.now)
        return Intent(0., TURN * sweep['dir'])


class Explore(Skill):
    """Visit a few unvisited viewpoints in a room (or nearby) and look around at each."""
    kind = 'explore'

    def tick(self, ctx):
        if self.stage == 'start':
            self.info['visited'] = 0
            self.stage = 'choose'
        if self.stage == 'choose':
            if self.info['visited'] >= int(self.args.get('viewpoints', 3)):
                return self.done()
            goal = ctx.viewpoint(self.args.get('room'))
            if goal is None:
                return self.done('nothing_new') if self.info['visited'] else self.fail('no_viewpoint')
            self.info['nav'] = Navigate({'xy': goal, 'tolerance': .3})
            self.info['nav'].started = ctx.now
            self.stage = 'going'
        if self.stage == 'going':
            nav = self.info['nav']
            intent = nav.tick(ctx)
            if nav.status == 'done':
                self.info['look'] = LookAround({'angle': random.choice((1, -1)) * math.pi * 1.5})
                self.info['look'].started = ctx.now
                self.stage = 'looking'
                return STOP
            if nav.status == 'failed':
                ctx.forget_viewpoint(nav.args['xy'])
                if self.args.get('room') and nav.reason.startswith('no_route'):
                    # 08.10 01:35: the hall was shut; four viewpoints behind the same door meant four waits and
                    # back-offs (the last one reversed it onto the dock). No way in is no way in: say so at once.
                    return self.fail('no_route')
                self.info['failures'] = self.info.get('failures', 0) + 1
                if self.info['failures'] > 3:
                    return self.fail('blocked')
                self.stage = 'choose'
                return STOP
            return intent
        if self.stage == 'looking':
            look = self.info['look']
            intent = look.tick(ctx)
            if look.status:
                self.info['visited'] += 1
                self.stage = 'choose'
                return STOP
            return intent
        return STOP


class Relocalize(Skill):
    """Stand still for a global search; if it fails, turn to a new view and retry."""
    kind = 'relocalize'

    def tick(self, ctx):
        if self.stage == 'start':
            # Always search: a stale "localized" flag can survive a few scans after being carried.
            self.info['views'] = 0
            self.stage = 'searching'
            ctx.request_recovery()
            return STOP
        if self.stage == 'searching':
            if ctx.recovering:
                return STOP
            if ctx.localized:
                return self.done()
            self.info['views'] += 1
            if self.info['views'] > 6:
                return self.fail('lost')
            self.info['turn_origin'] = ctx.odom[2]
            self.stage = 'turning'
        if self.stage == 'turning':
            turned = abs(wrap(ctx.odom[2] - self.info['turn_origin']))
            if turned <= math.pi / 3 and not ctx.clear(0., .36, .5) and \
                    self.info.get('backed', 0.) < RELOC_BACK_TOTAL_M and ctx.clear(-RELOC_BACK_V, 0., .8):
                # No room to turn (10.10 11:13: nose against a panel by the speaker — six views without a turn,
                # lost for two hours): back out straight the way we came, then look again from there.
                self.info['back_origin'] = ctx.odom
                self.info['back_started'] = ctx.now
                self.stage = 'backing'
                return STOP
            if turned > math.pi / 3 or not ctx.clear(0., .36, .5):
                self.stage = 'searching'
                ctx.request_recovery()
                return STOP
            return Intent(0., .36)
        if self.stage == 'backing':
            origin = self.info['back_origin']
            moved = math.dist(ctx.odom[:2], origin[:2])
            if moved >= RELOC_BACK_STEP_M or ctx.now - self.info['back_started'] > 6. or \
                    not ctx.clear(-RELOC_BACK_V, 0., .8):
                self.info['backed'] = self.info.get('backed', 0.) + moved
                self.stage = 'searching'
                ctx.request_recovery()
                return STOP
            return Intent(-RELOC_BACK_V, 0., True, origin[2])
        return STOP


SKILLS = {cls.kind: cls for cls in (Rotate, Drive, Wait, Navigate, GoToRoom, Approach, Undock, Dock,
                                    LookAround, Track, Explore, Relocalize, Dance, Emote, Perform)}
