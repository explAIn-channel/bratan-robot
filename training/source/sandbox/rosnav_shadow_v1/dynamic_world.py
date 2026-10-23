#!/usr/bin/env python3
"""The things in a flat that the map does not know about.

The stock map is a map of the walls.  It is drawn once, at night, by a robot
that had the place to itself, and it is right about the walls for months.  It
is wrong about the flat from the moment somebody stands up.

So the simulator needs the rest: a bag left in a doorway, a chair pulled out,
a person standing in the corridor, a person walking down it, a dog that was
asleep and is now not.  None of these are on the map, none of them are in the
global plan, and all of them are the reason the policy exists -- the planner
routes around geometry it was told about, and the network's whole job is the
geometry that arrives afterwards.

Two things here are load-bearing.

The first is height.  The lidar sits 8 cm off the floor, so it does not see a
person; it sees two ankles with a gap between them, and the gap is wide enough
for a beam to go through and report the wall behind.  A simulator that models
people as solid cylinders trains a network on an obstacle that does not exist
in this flat, and the first real person it meets will read as two thin posts
with clear space between -- which is an invitation to drive between somebody's
feet.  So creatures are built from legs.

The second is that the plan stays ignorant.  The route is planned on the
static map and is never told about any of this, exactly as Nav2 plans on the
saved map on the real robot.  If the simulator quietly re-planned around a
walking person the network would learn to trust the plan, which is the one
habit that gets it kicked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

#: Every scenario the world can be asked for.
SCENARIOS = ("clear", "static", "standing", "walking", "sudden", "crowded")

#: A creature's legs, as the lidar meets them at 8 cm: radius and half-gap.
LEG_RADIUS_M = {"person": 0.055, "dog": 0.045, "cat": 0.030}
LEG_GAP_M = {"person": 0.20, "dog": 0.34, "cat": 0.20}

#: How fast each thing moves when it is moving, in metres a second.
SPEED_MPS = {"person": (0.6, 1.4), "dog": (0.8, 2.2), "cat": (1.2, 3.0)}

#: Obstacles must narrow the way, not seal it.
#:
#: A 0.64 m armchair dropped onto the route of a 0.6 m corridor does not teach
#: avoidance; it teaches that the drive was over before it started. Trained
#: against a mix where a quarter of the episodes were unwinnable and most of
#: the rest left a gap too narrow for the chassis, the policy did not learn to
#: go round -- it lost the plan-following it already had, falling from 80 % to
#: 45 % on the empty flat while the crowded scenario stayed at 2 %.
#:
#: So a placement is kept only if a body-width path to the goal survives it.
#: What that leaves is the situation worth practising: the way through is
#: still there and it is tighter than the planner thinks.
PASSABILITY_ATTEMPTS = 6

#: Nothing may appear closer than this to the robot.  A creature spawned on
#: top of the bumper is not a test of driving, it is a coin toss, and training
#: against coin tosses teaches the network that collisions are unavoidable.
MIN_SPAWN_M = 1.0
#: The chassis, for deciding whether a gap is still a gap, is *not* restated
#: here.  It was, and the copy went stale: when the measured chassis moved
#: from 0.175 m to 0.249 m the placement check kept letting through anything a
#: robot seven centimetres narrower could pass, which is precisely the
#: unwinnable-episode mix the note above says destroys a run.  ``_admits``
#: reads the one definition in simulate_expert_dataset instead.
#: ...nor close enough to the goal to make arriving impossible.
GOAL_CLEARANCE_M = 0.6


@dataclass
class Body:
    """One thing in the room: a set of circles that move together."""

    offsets: np.ndarray          # (n, 2) circle centres, relative to (x, y)
    radii: np.ndarray            # (n,) radius of each circle
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    kind: str = "object"
    appears_at: int = 0
    heading: float = 0.0

    def circles(self, *, x=None, y=None, heading=None) -> tuple[np.ndarray, np.ndarray]:
        heading = self.heading if heading is None else heading
        cos, sin = math.cos(heading), math.sin(heading)
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        centres = self.offsets @ rotation.T
        centres = centres + np.array([self.x if x is None else x,
                                     self.y if y is None else y], dtype=np.float32)
        return centres, self.radii


def _legs(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """A creature as its lidar cross-section: two legs, side by side."""
    half = LEG_GAP_M[kind] * 0.5
    radius = LEG_RADIUS_M[kind]
    offsets = np.array([[0.0, half], [0.0, -half]], dtype=np.float32)
    return offsets, np.full(2, radius, dtype=np.float32)


def make_creature(kind: str, x: float, y: float, heading: float,
                  speed: float) -> Body:
    offsets, radii = _legs(kind)
    return Body(offsets=offsets, radii=radii, x=x, y=y,
                vx=speed * math.cos(heading), vy=speed * math.sin(heading),
                kind=kind, heading=heading)


def make_object(rng: np.random.Generator, x: float, y: float) -> Body:
    """A bag, a box, or a chair -- the last being four legs, not a block."""
    which = rng.integers(3)
    if which == 0:                                    # a bag or a bin
        offsets = np.zeros((1, 2), dtype=np.float32)
        radii = np.array([rng.uniform(0.10, 0.20)], dtype=np.float32)
    elif which == 1:                                  # a box against the wall
        offsets = np.zeros((1, 2), dtype=np.float32)
        radii = np.array([rng.uniform(0.20, 0.32)], dtype=np.float32)
    else:                                             # a chair, seen as legs
        half = rng.uniform(0.16, 0.22)
        offsets = np.array([[half, half], [half, -half],
                            [-half, half], [-half, -half]], dtype=np.float32)
        radii = np.full(4, 0.022, dtype=np.float32)
    return Body(offsets=offsets, radii=radii, x=x, y=y, kind="object",
                heading=float(rng.uniform(-math.pi, math.pi)))


@dataclass
class World:
    """Everything in the flat that the saved map has never heard of."""

    scenario: str = "clear"
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0))
    bodies: list[Body] = field(default_factory=list)
    steps: int = 0

    # -- setting the scene ------------------------------------------------

    def reset(self, env) -> None:
        self.bodies = []
        self.steps = 0
        if self.scenario == "clear":
            return
        route = self._route_points(env)
        if self.scenario == "static":
            for _ in range(int(self.rng.integers(1, 4))):
                self._place_object(env, route)
        elif self.scenario == "standing":
            self._place_creature(env, route, moving=False)
        elif self.scenario == "walking":
            for _ in range(int(self.rng.integers(1, 3))):
                self._place_creature(env, route, moving=True)
        elif self.scenario == "sudden":
            self._arm_ambush(env)
        elif self.scenario == "crowded":
            for _ in range(int(self.rng.integers(1, 3))):
                self._place_object(env, route)
            self._place_creature(env, route, moving=True)
            if self.rng.random() < 0.5:
                self._place_creature(env, route, moving=False)
            if self.rng.random() < 0.5:
                self._arm_ambush(env)
        else:
            raise ValueError(f"unknown scenario: {self.scenario}")

    def _route_points(self, env) -> list[tuple[float, float]]:
        """Where the robot is actually going, so the obstacles are in the way.

        Scattering obstacles uniformly over the flat mostly decorates rooms the
        robot will never enter.  What tests a driver is the thing standing on
        its route, so that is where these go.

        The route has to be walked to its end, and that is the part worth
        stating.  Descending a fixed eighty cells covers four metres, which was
        the whole map this simulator was first written against; on a real flat
        it puts every obstacle in the opening seconds and leaves the remaining
        ten metres empty.  The metrics then look excellent and mean almost
        nothing -- a policy scores full marks on "crowded" by driving past one
        chair and then down an empty corridor.
        """
        from simulate_expert_dataset import descend
        route = getattr(env, "route", None)
        if route is None or getattr(route, "cost", None) is None:
            # No plan to stand in the way of.  Scatter instead of raising:
            # an unplanned episode is still a legitimate place to practise,
            # and a world that refuses to build would take the run down.
            return []
        reach = math.hypot(env.goal[0] - env.x, env.goal[1] - env.y)
        # Cells, with slack for the detours the straight line does not know
        # about; descend stops of its own accord once it reaches the goal.
        points = int(min(600, max(80, 3.0 * reach / env.flat.resolution)))
        path = descend(route, env.flat, env.x, env.y, points=points)
        return [point for point in path
                if math.hypot(point[0] - env.x, point[1] - env.y) > MIN_SPAWN_M
                and math.hypot(point[0] - env.goal[0],
                               point[1] - env.goal[1]) > GOAL_CLEARANCE_M]

    def _pick(self, env, route):
        if route:
            return route[int(self.rng.integers(len(route)))]
        for _ in range(40):
            row, column = env.standable[self.rng.integers(len(env.standable))]
            x, y = env.flat.metres(int(row), int(column))
            if (math.hypot(x - env.x, y - env.y) > MIN_SPAWN_M and
                    math.hypot(x - env.goal[0],
                               y - env.goal[1]) > GOAL_CLEARANCE_M):
                return x, y
        return None

    def _place_object(self, env, route) -> None:
        for _ in range(PASSABILITY_ATTEMPTS):
            spot = self._pick(env, route)
            if spot is None:
                return
            body = make_object(self.rng, spot[0], spot[1])
            if self._admits(env, body):
                self.bodies.append(body)
                return

    def _place_creature(self, env, route, *, moving: bool) -> None:
        spot = self._pick(env, route)
        if spot is None:
            return
        kind = "person" if self.rng.random() < 0.7 else "dog"
        low, high = SPEED_MPS[kind]
        speed = float(self.rng.uniform(low, high)) if moving else 0.0
        for _ in range(PASSABILITY_ATTEMPTS):
            heading = float(self.rng.uniform(-math.pi, math.pi))
            body = make_creature(kind, spot[0], spot[1], heading, speed)
            if self._walkable(env, body.x, body.y, body=body) and (speed > 0.0 or self._admits(env, body)):
                # Somebody walking will move out of the doorway on their own;
                # only the ones who stay put have to leave room.
                self.bodies.append(body)
                return
            spot = self._pick(env, route)
            if spot is None:
                return

    def _arm_ambush(self, env) -> None:
        """Something that is not there yet, and then is.

        A cat out of a doorway is not an obstacle-avoidance problem in the
        usual sense: by the time it exists there is no room to plan, only to
        stop.  It spawns ahead of wherever the robot has got to, which is why
        it carries a delay rather than a position.
        """
        kind = "cat" if self.rng.random() < 0.6 else "dog"
        low, high = SPEED_MPS[kind]
        body = make_creature(kind, math.nan, math.nan,
                             float(self.rng.uniform(-math.pi, math.pi)),
                             float(self.rng.uniform(low, high)))
        # Anywhere in the drive, not just its first eight seconds.  A cat that
        # only ever appears at the start is a lesson about starting.
        latest = max(40, int(0.85 * getattr(env, "max_steps", 400)))
        body.appears_at = int(self.rng.integers(15, latest))
        self.bodies.append(body)

    def _admits(self, env, body: Body) -> bool:
        """Would the robot still have a way to the goal past this?"""
        if not self._walkable(env, body.x, body.y, body=body):
            return False
        flat = env.flat
        gap = getattr(flat, "blocked_distance", None)
        if gap is None:
            return True
        from scipy import ndimage
        from simulate_expert_dataset import BODY_RADIUS_M as BODY_CLEARANCE_M
        blocked = np.zeros(gap.shape, dtype=bool)
        circles = [item.circles() for item in [*self.bodies, body] if math.isfinite(item.x)]
        centres = np.concatenate([item[0] for item in circles])
        radii = np.concatenate([item[1] for item in circles])
        resolution = flat.resolution
        for (cx, cy), radius in zip(centres, radii):
            row, column = flat.cell(float(cx), float(cy))
            reach = int(math.ceil((float(radius) + BODY_CLEARANCE_M)
                                  / resolution))
            rows = slice(max(0, row - reach), min(flat.height, row + reach + 1))
            columns = slice(max(0, column - reach),
                            min(flat.width, column + reach + 1))
            grid_r, grid_c = np.ogrid[rows, columns]
            near = (((grid_r - row) ** 2 + (grid_c - column) ** 2)
                    * resolution ** 2) <= (float(radius) + BODY_CLEARANCE_M) ** 2
            blocked[rows, columns] |= near
        passable = (gap >= BODY_CLEARANCE_M) & ~blocked
        labels, count = ndimage.label(passable)
        if count == 0:
            return False
        start = labels[flat.cell(env.x, env.y)]
        goal = labels[flat.cell(*env.goal)]
        return bool(start) and start == goal

    # -- letting it move --------------------------------------------------

    def step(self, env) -> None:
        self.steps += 1
        for body in self.bodies:
            if self.steps == body.appears_at and not math.isfinite(body.x):
                self._spawn_ahead(env, body)
            if not math.isfinite(body.x):
                continue
            if body.vx == 0.0 and body.vy == 0.0:
                continue
            from simulate_expert_dataset import DT
            nx = body.x + body.vx * DT
            ny = body.y + body.vy * DT
            heading = math.atan2(body.vy, body.vx)
            yaw_delta = math.atan2(math.sin(heading-body.heading), math.cos(heading-body.heading))
            samples = max(1, math.ceil(math.hypot(nx-body.x, ny-body.y)/(env.flat.resolution*.5)),
                          math.ceil(abs(yaw_delta)/.1))
            if all(self._walkable(env, body.x+(nx-body.x)*fraction,
                                 body.y+(ny-body.y)*fraction, body=body,
                                 heading=body.heading+yaw_delta*fraction)
                   for fraction in np.linspace(0., 1., samples+1)):
                body.x, body.y = nx, ny
                body.heading = heading
            else:
                # Off a wall rather than through it, and with a turn rather
                # than a bounce, so a person keeps looking like a person.
                turn = float(self.rng.uniform(math.pi * 0.5, math.pi * 1.5))
                speed = math.hypot(body.vx, body.vy)
                heading = math.atan2(body.vy, body.vx) + turn
                body.vx, body.vy = speed * math.cos(heading), speed * math.sin(heading)

    def _spawn_ahead(self, env, body: Body) -> None:
        """Put the ambush just beyond the bumper, off to one side."""
        for _ in range(12):
            bearing = env.yaw + float(self.rng.uniform(-0.9, 0.9))
            reach = float(self.rng.uniform(MIN_SPAWN_M, 1.8))
            x = env.x + reach * math.cos(bearing)
            y = env.y + reach * math.sin(bearing)
            heading = bearing + math.pi * 0.5 * (1 if self.rng.random() < 0.5 else -1)
            if self._walkable(env, x, y, body=body, heading=heading):
                body.x, body.y = x, y
                # Crossing the bow is the hard case, so it runs across.
                speed = math.hypot(body.vx, body.vy)
                body.vx, body.vy = speed * math.cos(heading), speed * math.sin(heading)
                body.heading = heading
                return

    @staticmethod
    def _walkable(env, x: float, y: float, *, body=None, heading=None) -> bool:
        row, column = env.flat.cell(x, y)
        if not (0 <= row < env.flat.height and 0 <= column < env.flat.width):
            return False
        if not env.flat.free[row, column]:
            return False
        if body is None:
            return True
        centres, radii = body.circles(x=x, y=y, heading=heading)
        resolution = env.flat.resolution
        for (cx, cy), radius in zip(centres, radii):
            row, column = env.flat.cell(float(cx), float(cy))
            reach = int(math.ceil(float(radius)/resolution)) + 1
            if (row-reach < 0 or column-reach < 0 or
                    row+reach >= env.flat.height or column+reach >= env.flat.width):
                return False
            rows, columns = np.mgrid[row-reach:row+reach+1, column-reach:column+reach+1]
            wx, wy = env.flat.metres(rows, columns)
            covered = np.hypot(wx-cx, wy-cy) <= float(radius)+resolution/math.sqrt(2)
            if not np.all(env.flat.free[rows[covered], columns[covered]]):
                return False
        return True

    # -- being hit --------------------------------------------------------

    def nearest_surface(self, x: float, y: float) -> float:
        """Distance from a point to the nearest occupant's surface.

        Asked by the collision test, which cannot use the scan: a beam is
        clipped at the sensor's 0.15 m floor, so a leg pressed against the
        robot and a leg 0.15 m away return the same number.
        """
        best = math.inf
        for body in self.bodies:
            if not math.isfinite(body.x):
                continue
            centres, radii = body.circles()
            for centre, radius in zip(centres, radii):
                gap = math.hypot(centre[0] - x, centre[1] - y) - float(radius)
                best = min(best, gap)
        return best

    # -- being seen -------------------------------------------------------

    def fold_into_scan(self, x: float, y: float, heading: float,
                       ranges: np.ndarray, range_min: float) -> np.ndarray:
        """Shorten each beam to whatever it meets first.

        Analytic ray-circle intersection rather than stamping the bodies onto
        the occupancy grid: a person is 11 cm wide and a cell is 5 cm, so a
        rasterised leg is either two cells or none depending on where it
        stands, and the network would be learning the rounding.
        """
        live = [body for body in self.bodies if math.isfinite(body.x)]
        if not live:
            return ranges
        centres = []
        radii = []
        for body in live:
            body_centres, body_radii = body.circles()
            centres.append(body_centres)
            radii.append(body_radii)
        centres = np.concatenate(centres).astype(np.float32)
        radii = np.concatenate(radii).astype(np.float32)

        beams = ranges.shape[0]
        angles = heading + np.arange(beams, dtype=np.float32) * (2.0 * math.pi / beams)
        direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        to_centre = centres[None, :, :] - np.array([x, y], dtype=np.float32)
        along = np.einsum("bd,bcd->bc", direction, np.broadcast_to(
            to_centre, (beams,) + centres.shape))
        squared = (to_centre ** 2).sum(axis=-1)
        gap = along ** 2 - (squared - radii[None, :] ** 2)
        hit = gap >= 0.0
        root = np.sqrt(np.maximum(gap, 0.0))
        entry = along - root
        entry = np.where(hit & (entry > range_min), entry, np.inf)
        nearest = entry.min(axis=1)
        return np.minimum(ranges, nearest).astype(np.float32)
