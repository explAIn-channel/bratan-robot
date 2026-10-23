#!/usr/bin/env python3
"""The flat as a place to practise driving, fast enough to learn in.

Imitation has been measured and is finished: cloning the teacher produced a
network that a one-line "repeat the last command" baseline beats, and the
teacher itself is eight lines of geometry.  Copying it harder cannot help.  So
this is the environment for learning from consequence instead -- reach the
goal, keep off the walls, move smoothly -- where the ceiling is the task
rather than somebody else's controller.

Rays cross exact grid boundaries, so thin occupied cells cannot disappear
between range samples. The route is refreshed when observed obstacles change.
The observation is byte-for-byte the one the robot builds on
the phone, so a policy trained here can be shipped without a translation
layer -- and, more to the point, without a translation layer's bugs.
"""
from __future__ import annotations

import collections
import copy
import math
import sys
from pathlib import Path

import numpy as np

V5 = str(Path(__file__).resolve().parents[2])
if V5 not in sys.path:
    sys.path.insert(0, V5)
from scipy.ndimage import distance_transform_edt

import sim_geometry as core
from features import (HISTORY_DELAYS_S, LASER_YAW_RAD as _FEATURE_LASER_YAW,
                      build_features, pool_lidar)
from simulate_expert_dataset import (BEAMS, BODY_RADIUS_M, COLLISION_FOOTPRINT_M, DT, Flat,
                                     RANGE_MAX, RANGE_MIN, SENSOR_RANGE_MAX,
                                     descend, outline_gap, sensor_window,
                                     to_base_link, wrap)
from closed_loop_eval import ARRIVED_M
from dynamic_world import World

MAX_LINEAR = 0.25
#: Capped to what the robot has actually been shown to do.
#:
#: 0.70 was the bridge's ceiling, not a measured capability. Scan
#: matching on 2026-08-30 found commanded turns above roughly a
#: quarter radian a second going erratic -- one commanded 90 came
#: out at 46 degrees and its mirror at 142 -- while 0.10 to 0.25
#: behaved. A policy that learns to lean on authority the chassis
#: does not have would arrive on the robot already wrong.
#:
#: Still owed: 0.25 was measured before the turning fusion fix. It
#: wants confirming on the robot now that turns land at 0.99 of
#: what they are asked for.
MAX_ANGULAR = 0.25
#: The same rotation `features` uses, not a second copy. Keeping a second
#: copy meant the simulator cast its rays in one frame while the features
#: rotated them out of another the moment the measured value replaced the
#: guessed one.
LASER_YAW_RAD = _FEATURE_LASER_YAW

#: Rent, per step, for still being on the road.  It is what stops a policy
#: from wandering pleasantly for the whole budget.
STEP_COST = 0.02

#: What hitting something costs, over and above the rent below.
#:
#: It has to exceed the arrival bonus and be worth more than any detour the
#: flat can ask for -- 30 is two and a half metres of progress, and the
#: longest way round an obstacle here is shorter than that.  Set it lower and
#: the policy will correctly work out that a collision is cheaper than going
#: around, which is a true statement about the reward and a terrible one
#: about a robot that shares a corridor with people.
COLLISION_PENALTY = 30.0

#: Reaching the goal.
ARRIVAL_BONUS = 15.0

#: What a unit of progress along the route is worth.
#:
#: Progress used to be the reduction in the straight-line distance to the
#: goal, and on an empty flat that is almost the same thing.  With a chair in
#: the way it is the opposite of the thing being asked for: stepping sideways
#: to go round does not shorten the straight line, and stepping back to get
#: round lengthens it, so every metre of avoidance was billed to the policy
#: while driving into the obstacle was rewarded right up to the impact.  That
#: is a fair description of what it learned -- 150 updates with collisions
#: stuck near 90 % in the crowded scenario and no trend.
#:
#: Measured down the planner's own cost field instead, a detour that stays
#: inside the corridor costs only the time it takes.  The weight is a scale
#: conversion, not a preference: the wavefront charges 1.20 units a metre on
#: this map, so 10.0 a unit is the 12.0 a metre that came before it.
PROGRESS_PER_COST = 10.0

#: Where the robot starts being charged for shaving past things, and how
#: steeply.
#:
#: This is the inflation layer, moved into the reward because the planner does
#: not have one.  Real Nav2 prices proximity as a cost field and its routes
#: come out down the middle of a corridor; this wavefront searches a binary
#: mask and its routes come out along the wall, so if nothing else expresses a
#: preference for room, the policy has no reason to keep any.
#:
#: What it replaces looked at the forward 60 degrees of a scan clipped at
#: 0.15 m and charged at most 0.115 a step.  A robot tracking a wall with its
#: flank was therefore charged nothing at all, right up to the moment it hit
#: it -- and on an empty flat the policy settled at 35 % collisions and stopped
#: improving.  Measuring the real gap all the way round, and pricing it hard
#: enough to matter, is what turns "do not hit things" into a gradient the
#: policy can follow rather than a cliff it discovers by falling off.
CLEARANCE_COMFORT_M = 0.35
CLEARANCE_PENALTY = 2.0

#: The same idea again, for things that are moving.
#:
#: The band above is ten centimetres wide -- 0.35 m of comfort against a
#: 0.249 m body -- which is ample for furniture, because the only thing
#: closing that gap is the robot itself at 0.25 m/s.  A person in this
#: simulator walks at up to 1.4 m/s and a cat runs at 3.0, and they cross ten
#: centimetres inside a single 0.1 s tick.  So on everything that moves, the
#: distance penalty cannot fire in time to teach anything, and the only signal
#: left is the collision itself: sparse, terminal, and arriving after the
#: decision that caused it.
#:
#: Measured on the trained policy, that is exactly how it behaves -- it
#: collides in 79 % of crowded episodes while barely ever timing out, so it is
#: driving into people rather than hesitating near them.  And the scenario is
#: not the problem: walking the world forward without a robot, the route is
#: obstructed at some point in 95-97 % of walking and crowded episodes but
#: still obstructed at the end in only 5 % and 20 %.  Almost every one of
#: these is winnable by waiting a moment.
#:
#: Pricing closing speed rather than distance gives that a gradient.  The
#: horizon is what the robot needs to do something about it: at 0.25 m/s it
#: covers 0.375 m in 1.5 s, which is enough to stop or turn away.
#:
#: The first value tried here was 1.0, and it taught caution at the cost of
#: the job: collisions fell in every scenario -- static 35.8 % to 27.5 %,
#: standing 32.5 % to 27.5 %, crowded 79.2 % to 76.7 % -- and arrivals fell
#: with them, static 57.5 % to 46.7 %, while crowded timeouts rose from 1.7 %
#: to 5.8 %. That is a robot that has learned to hesitate rather than to
#: yield. At 1.0 the penalty is more than three times the 0.30 that a
#: full-speed step of progress earns, so near anything at all the arithmetic
#: says stop. Half of that leaves it comparable to progress when contact is
#: imminent and clearly weaker further out, which is the trade being asked
#: for: give way, then carry on.
#:
#: Halving it was a null result, and the honest record of that belongs here:
#: averaged over the six scenarios, 0.5 arrived 48.5 % of the time and hit
#: something 40.6 %, against 50.6 % and 38.7 % at 1.0 -- worse on both counts,
#: by less than the spread between runs. The knob is not linear and the effect
#: it was meant to tune is smaller than the noise around it, so the value goes
#: back to the one that measured better and the next attempt at this should
#: change the shape of the signal rather than its size.
APPROACH_HORIZON_S = 1.5
APPROACH_PENALTY = 1.0

class FastFlat(Flat):
    """The map, exact grid rays and a distance field for proximity rewards."""

    def __init__(self, yaml_path: Path,
                 clearance: float = Flat.PLAN_CLEARANCE_M):
        # Deferring to the parent's constant rather than repeating a number:
        # this signature carried its own 0.20 and silently won over the value
        # the base class had been changed to, so a run launched after the
        # change was still planning on the old margin.
        super().__init__(yaml_path, clearance=clearance)
        # Distance from every cell to the nearest cell a beam cannot pass.
        # The parent treats anything not known-free as blocking, and so must
        # this: unknown space stops a ray in the recorded data too.
        self.blocked_distance = (
            distance_transform_edt(self.free).astype(np.float32)
            * self.resolution)

    def _distance_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        columns = np.rint((xs - self.origin[0]) / self.resolution).astype(np.int32)
        rows = np.rint(self.height - 1
                - (ys - self.origin[1]) / self.resolution).astype(np.int32)
        inside = ((rows >= 0) & (rows < self.height)
                  & (columns >= 0) & (columns < self.width))
        out = np.zeros(xs.shape, dtype=np.float32)
        out[inside] = self.blocked_distance[rows[inside], columns[inside]]
        return out

    def raycast(self, x: float, y: float, heading: float) -> np.ndarray:
        """Return the first occupied cell boundary, including oblique grazes.

        Distance-transform samples measure centre-to-centre distance. Using
        them as surface distances jumped into or over occupied cells, hiding
        a thin obstacle or reporting the wall behind its actual near face.
        """
        angles = heading + np.arange(BEAMS) * (2.0 * math.pi / BEAMS)
        cos, sin = np.cos(angles), np.sin(angles)
        resolution = self.resolution
        column = math.floor((x-self.origin[0])/resolution + .5)
        row = math.floor(self.height-1-(y-self.origin[1])/resolution + .5)
        columns, rows = np.full(BEAMS, column), np.full(BEAMS, row)
        step_x, step_y = np.sign(cos).astype(int), -np.sign(sin).astype(int)
        centre_x, centre_y = self.metres(row, column)
        with np.errstate(divide='ignore', invalid='ignore'):
            next_x = (centre_x + step_x*resolution/2 - x) / cos
            next_y = (centre_y - step_y*resolution/2 - y) / sin
            delta_x, delta_y = resolution/np.abs(cos), resolution/np.abs(sin)
        next_x = np.where(cos == 0., np.inf, next_x)
        next_y = np.where(sin == 0., np.inf, next_y)
        travel, measured = np.zeros(BEAMS), np.full(BEAMS, np.inf)
        live = np.ones(BEAMS, dtype=bool)
        # Each step crosses an x or y boundary. This bound covers even a
        # diagonal ray's whole physical horizon without a guessed iteration cap.
        for _ in range(2*math.ceil(SENSOR_RANGE_MAX/resolution) + 3):
            inside = ((rows >= 0) & (rows < self.height) &
                      (columns >= 0) & (columns < self.width))
            blocked = ~inside
            blocked[inside] = ~self.free[rows[inside], columns[inside]]
            hit = live & blocked
            measured[hit] = travel[hit]
            live &= ~hit
            distance = np.minimum(next_x, next_y)
            live &= distance <= SENSOR_RANGE_MAX
            if not live.any():
                break
            cross_x, cross_y = live & (next_x <= next_y), live & (next_y <= next_x)
            travel[live] = distance[live]
            columns[cross_x] += step_x[cross_x]
            rows[cross_y] += step_y[cross_y]
            next_x[cross_x] += delta_x[cross_x]
            next_y[cross_y] += delta_y[cross_y]
        return sensor_window(measured)


class DriveEnv:
    """One drive across the flat, observed exactly as the robot observes it."""

    #: A drive may take this many times its own straight-line distance before
    #: it counts as a failure.  Three is generous for a flat: the longest
    #: detour here is around one wall.
    STEP_ALLOWANCE = 3.0

    #: ...but never less than this, so a two-metre hop still has room to
    #: turn around and correct itself.
    MIN_STEPS = 400

    def __init__(self, flat: FastFlat, rng: np.random.Generator,
                 max_steps: int | None = None, scenario: str = "clear",
                 encoding: str | None = None):
        """``max_steps=None`` sizes each episode to the drive it was given.

        A fixed cap is a distance limit wearing a disguise: at 0.25 m/s on a
        0.1 s tick, 400 steps is exactly ten metres of travel.  That was ample
        on the 12 m2 survey this was built against, and is less than one
        crossing of the real flat -- where it stopped measuring the driver and
        started measuring the map.  Worse, as a training signal it teaches
        that distant goals are not worth starting, because they cannot be
        finished.

        So the budget follows the task: three times the straight-line distance
        to the goal, floored so short hops keep room to recover.  Passing an
        integer keeps the old fixed behaviour, which is what reproduces the
        runs recorded before this existed.
        """
        self.flat = flat
        self.rng = rng
        depth = (1 if not HISTORY_DELAYS_S else
                 int(round(max(HISTORY_DELAYS_S) / DT)) + 2)
        self._views = collections.deque(maxlen=depth)
        self.fixed_steps = max_steps
        self.max_steps = max_steps if max_steps is not None else self.MIN_STEPS
        self.standable = np.argwhere(flat.plannable)
        # Occupants enter observations through the scan. Replanning sees only
        # these measured returns, never the world's hidden body coordinates.
        self.world = World(scenario=scenario, rng=rng)
        # Which range encoding this episode is observed through. ``None`` is
        # the module default, which is what the board implements.
        self.encoding = encoding

    def _random_pose(self):
        row, column = self.standable[self.rng.integers(len(self.standable))]
        x, y = self.flat.metres(int(row), int(column))
        return x, y

    def reset(self):
        for _ in range(60):
            self.x, self.y = self._random_pose()
            goal = self._random_pose()
            if math.hypot(goal[0] - self.x, goal[1] - self.y) < 1.0:
                continue
            self.route = core.Wavefront(self.flat.field,
                                        clearance=self.flat.clearance)
            if not self.route.plan(goal):
                continue
            if not math.isfinite(self.route.cost_at((self.x, self.y))):
                continue
            self.goal = goal
            break
        else:
            raise RuntimeError("no reachable start/goal pair on this map")

        self.yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.linear = 0.0
        self.angular = 0.0
        self.steps = 0
        self.numpy = np
        self.obstacles = core.LocalMap(self)
        self.replans = 0
        self.no_route = False
        self.remaining = math.hypot(self.goal[0] - self.x,
                                    self.goal[1] - self.y)
        self.route_cost = self.route.cost_at((self.x, self.y))
        if self.fixed_steps is None:
            # Time to drive it, plus time to point at it. The budget used to
            # count only the distance, as though turning were free, and every
            # episode starts at a random heading: on average a quarter turn
            # before the robot may even set off.
            #
            # That was harmless while the two rates were far apart and became
            # a trap on 2026-08-30, when MAX_ANGULAR was cut from 0.70 to the
            # 0.25 the chassis is actually good for. The same quarter turn
            # went from 2.2 seconds to 6.3, taken out of a budget that had not
            # moved, and timeouts across the static scenarios jumped to four
            # episodes in ten. The policy was not failing to arrive; it was
            # being asked to arrive in someone else's time.
            bearing = math.atan2(self.goal[1] - self.y, self.goal[0] - self.x)
            turn = abs(wrap(bearing - self.yaw))
            seconds = (self.remaining / MAX_LINEAR) + (turn / MAX_ANGULAR)
            reach = self.STEP_ALLOWANCE * seconds / DT
            self.max_steps = max(self.MIN_STEPS, int(math.ceil(reach)))
        self.initial_world_rejections = []
        for _ in range(60):
            self.world.reset(self)
            certificate = self._initial_collision_certificate()
            if certificate is None:
                break
            self.initial_world_rejections.append(certificate)
        else:
            raise RuntimeError('no initial world without a proven unavoidable collision')
        # A drive starts with no past; the first frame is its own history.
        self._views.clear()
        observation = self._observe()
        self.gap = self._body_gap()
        self.outline = self._outline_gap()
        return observation

    def _initial_collision_certificate(self):
        """Sufficient proof of contact for every admissible initial control.

        Absence of a certificate means unknown, not guaranteed solvability.
        Keep the scene unless its entire reachable centre box is covered by
        collision regions. Never forecast across robot-relative spawning.
        """
        if self.steps != 0 or self.linear != 0. or self.angular != 0.:
            raise ValueError('initial collision check requires a stopped reset state')
        if not any(math.isfinite(b.x) and (b.vx or b.vy) for b in self.world.bodies):
            return None
        horizon = min(round(2./DT), self.max_steps,
            min((b.appears_at-1 for b in self.world.bodies if not math.isfinite(b.x)),
                default=round(2./DT)))
        preview = copy.copy(self)
        preview.world = copy.deepcopy(self.world)  # Includes an independent RNG.
        origin = np.array([self.x, self.y])
        rotation = np.array([[math.cos(self.yaw), -math.sin(self.yaw)],
                             [math.sin(self.yaw), math.cos(self.yaw)]])
        row, col = self.flat.cell(self.x, self.y)
        span = math.ceil((MAX_LINEAR*horizon*DT+BODY_RADIUS_M+self.flat.resolution)
                         / self.flat.resolution)+2
        top, bottom = max(0, row-span), min(self.flat.height, row+span+1)
        left, right = max(0, col-span), min(self.flat.width, col+span+1)
        rows, columns = np.nonzero(~self.flat.free[top:bottom, left:right])
        walls = np.column_stack(self.flat.metres(rows+top, columns+left))
        wall_radii = np.full(len(walls), self.flat.resolution/math.sqrt(2))
        polygon = np.asarray(COLLISION_FOOTPRINT_M)
        planes = np.asarray(core.guard_polygon_edges(COLLISION_FOOTPRINT_M))
        xs, ys = np.meshgrid(np.linspace(polygon[:, 0].min(), polygon[:, 0].max(), 9),
                             np.linspace(polygon[:, 1].min(), polygon[:, 1].max(), 7))
        disks = np.vstack((np.column_stack((xs.ravel(), ys.ravel())), [[0., 0.]]))
        directions = np.arctan2(disks[:, 1], disks[:, 0])[:, None] - np.arctan2(planes[:, 1], planes[:, 0])[None]
        directions = np.arctan2(np.sin(directions), np.cos(directions))
        lengths = np.linalg.norm(disks, axis=1)[:, None]
        side = yaw_rate = yaw_bound = 0.
        for tick in range(1, horizon+1):
            angle = MAX_ANGULAR*(tick-1)*DT
            if angle >= math.pi/2:
                break  # The nonnegative forward bound no longer applies.
            travel = MAX_LINEAR*tick*DT  # Allows even instantaneous acceleration.
            side += MAX_LINEAR*DT*math.sin(angle)
            yaw_rate += (MAX_ANGULAR-yaw_rate)*-math.expm1(-DT/core.TURN_COAST_S)
            yaw_bound += yaw_rate*DT
            if self.remaining-ARRIVED_M <= travel:
                break  # A controller could finish before the claimed contact.
            preview.world.step(preview)
            circles = [b.circles() for b in preview.world.bodies if math.isfinite(b.x)]
            centres = np.concatenate([item[0] for item in circles])
            radii = np.concatenate([item[1] for item in circles]).astype(float)
            if np.all(outline_gap(self.x, self.y, self.yaw, centres) > radii):
                continue  # Stopping is one admissible collision-free pose here.
            # Disks contained in the footprint at EVERY attainable yaw. For
            # each outward plane, maximize its rotated normal's dot product.
            maximum = lengths*np.where(np.abs(directions) <= yaw_bound, 1.,
                np.maximum(np.cos(directions-yaw_bound), np.cos(directions+yaw_bound)))
            sizes = np.min(planes[None, :, 2]-maximum, axis=1)-1e-6
            valid = sizes > 1e-5
            points = (np.concatenate((walls, centres))-origin) @ rotation
            cover_centres = (points[:, None]-disks[None, valid]).reshape(-1, 2)
            cover_radii = (np.concatenate((wall_radii, radii))[:, None]+sizes[None, valid]).ravel()
            # ponytail: 8x8 conservative tiles may miss an impossible scene;
            # refine only if a retained counterexample needs a tighter proof.
            cell = np.array([travel, 2*side])/8
            gx, gy = np.meshgrid((np.arange(8)+.5)*cell[0], -side+(np.arange(8)+.5)*cell[1])
            grid = np.column_stack((gx.ravel(), gy.ravel()))
            margin = float((cover_radii[None]-np.linalg.norm(
                grid[:, None]-cover_centres[None], axis=2)).max(axis=1).min()-np.linalg.norm(cell)/2)
            if margin > 1e-5:
                return dict(tick=tick, margin_m=margin, centre_bounds_m=[[0., -side], [travel, side]],
                            yaw_bound_rad=yaw_bound)
        return None

    def _body_gap(self) -> float:
        """Distance from the robot's centre to the nearest thing it can hit.

        This replaces asking the scan, and the reason is not a refinement.
        The beams are clipped at the sensor's own 0.15 m minimum range, and a
        collision was declared below 0.12 m -- a threshold the scan could not
        reach even in principle.  The branch never once executed: every
        `"collision": 0.0` this project has recorded, in every training run,
        was an unreachable comparison rather than a robot that missed the
        furniture.  Measured properly on the same runs, the body outline was
        overlapping something on 9.6 % of steps.

        The simulator knows where the walls and the occupants are, so it
        answers from that.  The scan keeps its job -- it is what the policy
        sees -- and stops being asked a question it is the wrong instrument
        for.
        """
        row, column = self.flat.cell(self.x, self.y)
        if not (0 <= row < self.flat.height and 0 <= column < self.flat.width):
            return 0.0
        walls = float(self.flat.blocked_distance[row, column])
        return min(walls, self.world.nearest_surface(self.x, self.y))

    def _outline_gap(self) -> float:
        """Room left between the body outline and the nearest thing it can hit.

        Zero or less means the outline is over something.  `_body_gap` above
        answers the same question about the robot's centre and is kept for the
        comfort term, which wants a smooth proximity signal rather than a
        verdict; this one decides whether a drive ended.

        Blocked cells are squares, so a cell counts as touched when its centre
        comes within half the cell diagonal of the outline. This contains
        every corner of the occupied square at any robot yaw.  Occupants are already circles and are asked
        directly, because an occupant's radius is its own.
        """
        resolution = self.flat.resolution
        row, column = self.flat.cell(self.x, self.y)
        if not (0 <= row < self.flat.height and 0 <= column < self.flat.width):
            return 0.0
        span = int(math.ceil(BODY_RADIUS_M / resolution)) + 1
        top, bottom = max(0, row - span), min(self.flat.height, row + span + 1)
        left, right = max(0, column - span), min(self.flat.width, column + span + 1)
        window = self.flat.free[top:bottom, left:right]
        best = math.inf
        rows, columns = np.nonzero(~window)
        if rows.size:
            xs = self.flat.origin[0] + (columns + left) * resolution
            ys = self.flat.origin[1] + (self.flat.height - 1 - (rows + top)) * resolution
            gaps = outline_gap(self.x, self.y, self.yaw,
                               np.stack([xs, ys], axis=1))
            best = float(gaps.min()) - math.sqrt(.5) * resolution
        for body in self.world.bodies:
            if not math.isfinite(body.x):
                continue
            centres, radii = body.circles()
            gaps = outline_gap(self.x, self.y, self.yaw,
                               np.asarray(centres, dtype=float))
            best = min(best, float(np.min(gaps - np.asarray(radii, dtype=float))))
        # Capped so an empty window reports a number rather than infinity:
        # the approach term differences this between steps, and inf - inf
        # is a nan that would silently poison the reward.
        return min(best, float(RANGE_MAX))

    def _observe(self):
        self.scan = self.flat.raycast(self.x, self.y, self.yaw + LASER_YAW_RAD)
        self.scan = self.world.fold_into_scan(
            self.x, self.y, self.yaw + LASER_YAW_RAD, self.scan, 0.0)
        self.scan = sensor_window(self.scan)
        now = self.steps * DT
        angles = self.yaw + LASER_YAW_RAD + np.arange(BEAMS) * (2*np.pi/BEAMS)
        valid = np.isfinite(self.scan) & (self.scan <= 1.5)
        xs = self.x + self.scan[valid]*np.cos(angles[valid])
        ys = self.y + self.scan[valid]*np.sin(angles[valid])
        for x, y, distance in zip(xs, ys, self.scan[valid]):
            row, col = self.flat.cell(float(x), float(y))
            if (0 <= row < self.flat.height and 0 <= col < self.flat.width
                    and self.flat.blocked_distance[row, col] >= .16):
                self.obstacles.mark(float(x), float(y), float(distance), now)
        # Same merge, capacity and TTL as the live local map. Without measured
        # scan-motion geometry, rays do not prematurely clear remembered points.
        self.obstacles._clear((self.x, self.y, self.yaw), now)
        if self.steps % round(1. / DT) == 0:
            if self.route.update_obstacles(self.obstacles.points):
                self.no_route = not (self.route.plan(self.goal) and
                    math.isfinite(self.route.cost_at((self.x, self.y), look=7)))
                # Reuse the live planner; a changed route is not physical progress.
                self.route_cost = self.route.cost_at((self.x, self.y))
                self.replans += 1
        path = [] if self.no_route else descend(self.route, self.flat, self.x, self.y)
        self.local = (to_base_link(path, self.x, self.y, self.yaw)
                      if len(path) > 1 else [(0., 0.)])
        history = self._remember_scan()
        return build_features(self.scan, RANGE_MIN, RANGE_MAX, self.local,
                              self.linear, self.angular, angle_min=0.0,
                              angle_increment=2.0 * math.pi / BEAMS,
                              laser_yaw=LASER_YAW_RAD,
                              encoding=self.encoding,
                              history=history)

    def _remember_scan(self):
        """Keep the pooled views the observation reaches back for.

        The delays are stated in seconds and turned into frames here, so the
        board -- which runs its scan at its own rate -- can honour the same
        contract without matching this simulator's step.

        Early in a drive the buffer is shallower than the longest delay, and
        the oldest view stands in.  That reads as "this is as long as anything
        has been watched", which is exactly true and is what makes the first
        seconds of an episode say "nothing has been standing here a while".
        """
        if not HISTORY_DELAYS_S:
            return None
        pooled = pool_lidar(self.scan, RANGE_MIN, RANGE_MAX, angle_min=0.0,
                            angle_increment=2.0 * math.pi / BEAMS,
                            laser_yaw=_FEATURE_LASER_YAW,
                            encoding=self.encoding)
        self._views.append(pooled)
        views = []
        for delay in HISTORY_DELAYS_S:
            back = int(round(delay / DT))
            index = max(0, len(self._views) - 1 - back)
            views.append(self._views[index])
        return views

    def step(self, action: np.ndarray):
        """Apply a twist, advance the body, and say what it was worth.

        The chassis does not jump to a commanded speed and neither does this:
        the same first-order lag the recorded runs show is applied here, so a
        policy that learns to slam the controls learns that it does not work.
        """
        target_linear = float(np.clip(action[0], 0.0, MAX_LINEAR))
        target_angular = float(np.clip(action[1], -MAX_ANGULAR, MAX_ANGULAR))

        previous_linear, previous_angular = self.linear, self.angular
        self.linear += (target_linear - self.linear) * 0.5
        self.angular += (target_angular - self.angular) * -math.expm1(-DT / core.TURN_COAST_S)

        self.x += self.linear * math.cos(self.yaw) * DT
        self.y += self.linear * math.sin(self.yaw) * DT
        self.yaw = wrap(self.yaw + self.angular * DT)
        self.steps += 1
        self.world.step(self)

        remaining = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.remaining = remaining
        # Arrival is still judged by the straight line -- being within 0.30 m
        # of the goal is a fact about where the robot is, not about the route
        # it took -- but what earns reward is progress down the plan.
        cost = self.route.cost_at((self.x, self.y))
        if math.isfinite(cost) and math.isfinite(self.route_cost):
            progress = self.route_cost - cost
        else:
            progress = 0.0
        if math.isfinite(cost):
            self.route_cost = cost

        observation = self._observe()
        previous_gap = self.gap
        previous_outline = self.outline
        self.gap = self._body_gap()
        self.outline = self._outline_gap()

        # Getting closer is the whole job, so it carries most of the reward.
        reward = PROGRESS_PER_COST * progress - STEP_COST
        # Smoothness is not decoration: a policy that chatters the wheels
        # reads as a machine and wears the gearbox.
        reward -= 0.6 * abs(self.linear - previous_linear)
        reward -= 0.15 * abs(self.angular - previous_angular)
        # Shaving past furniture is a bad habit even when it works.
        if self.gap < CLEARANCE_COMFORT_M:
            reward -= CLEARANCE_PENALTY * (CLEARANCE_COMFORT_M - self.gap)
        # Closing on something is priced whatever the distance, because the
        # distance band cannot catch anything that moves.  Only approach is
        # charged: opening the gap, or holding it, costs nothing, so waiting
        # for somebody to pass is free where pushing through is not.
        closing = (previous_outline - self.outline) / DT
        free = self.outline
        if closing > 0.0 and free > 0.0:
            contact_in = free / closing
            if contact_in < APPROACH_HORIZON_S:
                reward -= APPROACH_PENALTY * (1.0 - contact_in /
                                              APPROACH_HORIZON_S)

        done = False
        outcome = None
        if self.outline <= 0.0:
            # The rent for the rest of the episode is charged here, and the
            # reason is not bookkeeping.  A collision ends the episode, and
            # ending the episode stops the per-step cost -- so on any drive
            # whose budget outlasts the penalty, dying is a way to stop
            # paying.  This flat is fifteen metres across, so a long drive is
            # budgeted at 1200 steps and costs 24 in rent against a 12-point
            # penalty: crashing was cheaper than continuing, and a policy
            # trained against that learns to bank whatever progress it can and
            # then hit something.  It was doing exactly that -- reward rising
            # while arrivals fell.
            #
            # Charging the remaining rent makes the choice honest: whether it
            # crashes now or idles to the end it pays the same for the time,
            # and the penalty is then a penalty rather than a discount.
            reward -= COLLISION_PENALTY + STEP_COST * (self.max_steps - self.steps)
            done, outcome = True, "collision"
        elif remaining <= ARRIVED_M:
            reward += ARRIVAL_BONUS
            done, outcome = True, "arrived"
        elif self.steps >= self.max_steps:
            done, outcome = True, "timeout"

        return observation, float(reward), done, outcome
