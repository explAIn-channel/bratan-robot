#!/usr/bin/env python3
"""Generate training data by driving the real planner around the real map.

The recorded dataset lives on the robot and the robot is not always
reachable, but everything needed to make more of it is here: the flat's own
occupancy grid, the measured chassis calibration, and the wavefront planner
that actually drives the machine.  So the expert is not a hand-written
control law invented for the occasion -- it is the same
``sim_geometry.Wavefront`` the robot follows, steering a
differential-drive model over the same map.

Two details make the data faithful rather than merely plausible.  The scan is
cast in the *laser* frame, rotated by the calibrated mount yaw, exactly as the
hardware delivers it, so the feature builder's correction is exercised end to
end.  And the path handed to the policy is produced the way the robot's is --
by descending the planner's cost field from where the robot stands -- rather
than being a straight line to the goal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

V5 = str(Path(__file__).resolve().parents[2])
if V5 not in sys.path:
    sys.path.insert(0, V5)

from features import (FEATURE_DIM, LASER_YAW_RAD, PATH_POINTS,  # noqa: E402
                      build_features)
import sim_geometry as core                            # noqa: E402
from sim_map import DistanceMap              # noqa: E402

BEAMS = 360
# The hardware acceptance window and the policy encoding window are different
# contracts. A missing physical return is encoded as the policy horizon; a
# return outside the sensor window must never be clipped into a fake wall.
SENSOR_RANGE_MIN = 0.20
SENSOR_RANGE_MAX = 8.0
RANGE_MAX = 12.0
RANGE_MIN = 0.15
STEP_M = 0.025
DT = 0.1
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
# Planner clearance keeps its calibrated shell envelope. Collision checking
# uses the full printed contour, including the front paws and side ears.
BODY_RADIUS_M = 0.26734535916422786
FOOTPRINT_FRONT_M = 0.2259172
FOOTPRINT_REAR_M = 0.1827
FOOTPRINT_HALF_WIDTH_M = 0.1429509
COLLISION_FOOTPRINT_M = json.loads(
    (Path(V5) / "navigation/calibration.json").read_text(encoding="utf-8"))["guard_footprint_m"]
_COLLISION_PLANES = np.asarray(core.guard_polygon_edges(COLLISION_FOOTPRINT_M))
_COLLISION_VERTICES = np.asarray(COLLISION_FOOTPRINT_M)
_COLLISION_SEGMENTS = np.roll(_COLLISION_VERTICES, -1, axis=0) - _COLLISION_VERTICES
_COLLISION_LENGTHS_SQ = np.sum(_COLLISION_SEGMENTS**2, axis=1)
BODY_HALF_WIDTH_M = float(np.max(np.abs(_COLLISION_VERTICES[:, 1])))
if np.max(np.linalg.norm(_COLLISION_VERTICES, axis=1)) > BODY_RADIUS_M:
    raise ValueError("collision_contour_exceeds_planner_search_radius")


def sensor_window(ranges: np.ndarray) -> np.ndarray:
    """Apply the live Delta-2D acceptance window, preserving missing returns."""
    values = np.asarray(ranges, dtype=np.float32)
    accepted = (np.isfinite(values) & (values >= SENSOR_RANGE_MIN) &
                (values <= SENSOR_RANGE_MAX))
    return np.where(accepted, values, np.inf).astype(np.float32)


def outline_gap(x: float, y: float, yaw: float, points: np.ndarray) -> np.ndarray:
    """Exact point distance to the calibrated convex contour; inside is zero."""
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = points[:, 0] - x, points[:, 1] - y
    local = np.column_stack((cosine*dx + sine*dy, -sine*dx + cosine*dy))
    offset = local[:, None, :] - _COLLISION_VERTICES
    fraction = np.clip(np.sum(offset * _COLLISION_SEGMENTS, axis=2)
                       / _COLLISION_LENGTHS_SQ, 0., 1.)
    nearest = offset - fraction[:, :, None] * _COLLISION_SEGMENTS
    distance = np.sqrt(np.min(np.sum(nearest**2, axis=2), axis=1))
    inside = np.all(local @ _COLLISION_PLANES[:, :2].T
                    <= _COLLISION_PLANES[:, 2] + 1e-12, axis=1)
    return np.where(inside, 0., distance)


def wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class Flat:
    """The map, with the two questions a simulator actually asks of it."""

    #: How much room the global plan keeps between the robot's centre and
    #: anything solid.
    #:
    #: The real stack refuses to plan inside 0.205 m (`robot_radius: 0.19` plus
    #: `footprint_padding: 0.015` in `nav2_overlay.yaml`); this is that limit
    #: rounded up to the next grid cell.  The 0.20 m it replaces was a quarter
    #: of a cell above the body, and this wavefront -- unlike the real planner,
    #: which has an inflation layer pushing routes towards the middle of a
    #: corridor -- takes the shortest path through the binary mask, so it
    #: hugged that limit the whole way.  A controller then had 2.5 cm to absorb
    #: every error: the eight-line reference driver arrived 25 % of the time on
    #: an *empty* flat, which measured the margin rather than the driver.  At
    #: 0.25 m it arrives 60 %.
    #:
    #: 0.30 m would suit the reference driver better still (92 %) and is
    #: refused: it narrows the doorway into the bottom-left room past the
    #: point of admitting the robot, and a simulator that quietly deletes a
    #: room is not a simulator of this flat.  Retried on 2026-08-30 for a
    #: different reason and refused again for the same one, now measured:
    #: at 0.30 the plan reaches 70 % of the floor the chassis fits on.
    #:
    #: What changed on 2026-08-30 is which body dimension this is judged
    #: against.  The note above said the chassis needs 0.175 m; that was the
    #: round donor's radius, and this robot is 406 by 232 mm.  It sweeps
    #: 0.249 m turning on the spot -- from the drive axle, which is not the
    #: middle of the body -- but presents only its 0.116 m half width to a
    #: doorway it drives straight through.  A planned route asks the second
    #: question, so that is what the margin here is measured from.
    #:
    #: Still owed: nothing yet checks that a route leaves room to turn round
    #: at the far end, which is where the spin radius genuinely applies.
    PLAN_CLEARANCE_M = 0.25

    def __init__(self, yaml_path: Path, clearance: float = PLAN_CLEARANCE_M):
        self.field = DistanceMap(yaml_path)
        self.free = np.asarray(self.field.free, dtype=bool)
        # The planner and the physics have to agree about what a room is.
        #
        # `DistanceMap` is built for localisation, where an obstacle is
        # something a beam came back from: it measures distance to cells at or
        # below 65, so unknown space -- 205, everything the lidar never saw --
        # counts as open. The raycaster and the collision test here take the
        # opposite and correct view for driving: unknown is not floor.
        #
        # Left as it came, the wavefront planned across 37,803 cells on this
        # map while only 9,485 were floor. Three quarters of its route space
        # was the unmapped world outside the walls, so the shortest path
        # between two rooms went straight through a wall and out of the
        # building -- and the robot, following its plan faithfully, drove into
        # the wall. Every episode was scored against a route that could not be
        # driven.
        from scipy.ndimage import distance_transform_edt
        self.field.field = (distance_transform_edt(self.free).astype(np.float32)
                            * self.field.resolution)
        self.height, self.width = self.free.shape
        self.resolution = float(self.field.resolution)
        self.origin = self.field.origin
        # Cells a robot body centre may legally occupy.
        from scipy.ndimage import binary_erosion
        # Ceiling, not rounding, and the epsilon is not superstition:
        # 0.175 / 0.05 evaluates to 3.4999999999999996, which `round` takes
        # down to 3.  That eroded the map by 0.15 m instead of 0.175 -- a
        # simulator quietly driving a robot 5 cm narrower than the real one,
        # and handing out start poses a real chassis would not fit in.
        radius = max(1, math.ceil(BODY_RADIUS_M / self.resolution - 1e-9))
        size = 2 * radius + 1
        rows, columns = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = (rows ** 2 + columns ** 2) <= radius ** 2
        self.standable = binary_erosion(self.free, structure=disc)
        self.clearance = clearance
        # Where an episode may start or finish.  Sampling from `standable`
        # alone offers poses the planner will not route to, which the reset
        # loop then has to discard one at a time.
        self.plannable = self.standable & (self.field.field >= clearance)

    def cell(self, x: float, y: float):
        column = int(round((x - self.origin[0]) / self.resolution))
        row = int(round(self.height - 1 - (y - self.origin[1]) / self.resolution))
        return row, column

    def metres(self, row: int, column: int):
        """Grid sample coordinates shared with DistanceMap and Wavefront.

        A private floor/truncate convention shifted physics relative to the
        planner by half a cell and sometimes selected its neighbouring cell.
        """
        x = self.origin[0] + column * self.resolution
        y = self.origin[1] + (self.height - 1 - row) * self.resolution
        return x, y

    def raycast(self, x: float, y: float, heading: float) -> np.ndarray:
        """One full turn of beams from ``(x, y)``, in the frame of ``heading``."""
        angles = heading + np.arange(BEAMS) * (2.0 * math.pi / BEAMS)
        # Cast close enough to know that a surface exists, then apply the
        # sensor window. This keeps collision physics independent: a wall at
        # 0.16 m is real but produces the same missing return as the board.
        steps = np.arange(STEP_M, SENSOR_RANGE_MAX + STEP_M,
                          STEP_M, dtype=np.float32)
        xs = x + np.cos(angles)[:, None] * steps[None, :]
        ys = y + np.sin(angles)[:, None] * steps[None, :]
        columns = np.rint((xs - self.origin[0]) / self.resolution).astype(np.int32)
        rows = np.rint(self.height - 1 - (ys - self.origin[1]) / self.resolution).astype(np.int32)
        inside = ((rows >= 0) & (rows < self.height) &
                  (columns >= 0) & (columns < self.width))
        blocked = np.ones_like(inside)
        blocked[inside] = ~self.free[rows[inside], columns[inside]]
        first = np.argmax(blocked, axis=1)
        hit = blocked.any(axis=1)
        ranges = np.where(
            hit, steps[np.clip(first, 0, steps.size - 1)], np.inf)
        return sensor_window(ranges)


def descend(route, flat: Flat, x: float, y: float, points: int = 30):
    """Walk down the planner's cost field, the way the robot's plan is made."""
    return route.path(x, y, points)


def to_base_link(path, x: float, y: float, yaw: float):
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    out = []
    for point_x, point_y in path:
        dx, dy = point_x - x, point_y - y
        out.append((dx * cosine - dy * sine, dx * sine + dy * cosine))
    return out


def expert_command(bearing: float, ranges: np.ndarray,
                   remaining: float, reach: float = 0.30):
    """The teacher: turn toward the planner, slow for what is close ahead."""
    angular = float(np.clip(1.6 * bearing, -MAX_ANGULAR, MAX_ANGULAR))
    ahead = np.concatenate([ranges[:30], ranges[-30:]])
    clearance = float(np.min(ahead))
    linear = MAX_LINEAR - 0.30 * abs(angular)
    if clearance < 0.8:
        linear *= max(0.15, (clearance - 0.25) / 0.55)
    if abs(bearing) > math.radians(50):
        linear = 0.0
    # Do not out-drive the steering.  Working off a heading error of
    # `bearing` takes |bearing| / MAX_ANGULAR seconds, and whatever is
    # travelled past the aiming point in that time arrives at the corner
    # still pointing the wrong way.  This cost nothing while the cap was
    # 0.70 rad/s, where full speed turned inside 0.36 m; at the measured
    # 0.25 the radius is a full metre, wider than the rooms, and the
    # teacher drove into walls in four episodes out of five on an *empty*
    # flat -- which measured the cap, not the driver.
    turn_seconds = abs(bearing) / MAX_ANGULAR
    if turn_seconds > 1e-3:
        linear = min(linear, reach / turn_seconds)
    linear = float(np.clip(linear, 0.0, MAX_LINEAR))
    if remaining < 0.25:
        linear *= remaining / 0.25
    return linear, angular


def episode(flat: Flat, rng: np.random.Generator, max_steps: int = 400):
    """One drive from a random legal pose to a random reachable goal."""
    standable = np.argwhere(flat.standable)
    if len(standable) < 20:
        return None
    start_row, start_column = standable[rng.integers(len(standable))]
    goal_row, goal_column = standable[rng.integers(len(standable))]
    x, y = flat.metres(int(start_row), int(start_column))
    goal = flat.metres(int(goal_row), int(goal_column))
    if math.hypot(goal[0] - x, goal[1] - y) < 0.8:
        return None
    yaw = float(rng.uniform(-math.pi, math.pi))

    route = core.Wavefront(flat.field, clearance=flat.clearance)
    if not route.plan(goal):
        return None
    if not math.isfinite(route.cost_at((x, y))):
        return None

    features, labels = [], []
    linear = angular = 0.0
    for _ in range(max_steps):
        remaining = math.hypot(goal[0] - x, goal[1] - y)
        if remaining < 0.20:
            break
        scan = flat.raycast(x, y, yaw + LASER_YAW_RAD)
        if float(np.min(scan)) < 0.12:
            break                                     # would have hit something
        path = descend(route, flat, x, y)
        if len(path) < 2:
            break
        local = to_base_link(path, x, y, yaw)
        aim = local[min(6, len(local) - 1)]
        bearing = math.atan2(aim[1], aim[0])
        target_linear, target_angular = expert_command(
            bearing, scan, remaining, math.hypot(aim[0], aim[1]))
        try:
            vector = build_features(scan, RANGE_MIN, RANGE_MAX, local,
                                    linear, angular,
                                    angle_min=0.0,
                                    angle_increment=2.0 * math.pi / BEAMS,
                                    laser_yaw=LASER_YAW_RAD)
        except ValueError:
            break
        features.append(vector)
        labels.append((target_linear, target_angular))

        # First-order lag, so the recorded command and the motion that follows
        # are related the way they are on the robot rather than instantaneous.
        linear += (target_linear - linear) * 0.5
        angular += (target_angular - angular) * 0.6
        x += linear * math.cos(yaw) * DT
        y += linear * math.sin(yaw) * DT
        yaw = wrap(yaw + angular * DT)
    if len(features) < 20:
        return None
    return (np.asarray(features, dtype=np.float32),
            np.asarray(labels, dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    flat = Flat(args.map)
    rng = np.random.default_rng(args.seed)
    rows, total, attempts = [], 0, 0
    while len(rows) < args.episodes and attempts < args.episodes * 12:
        attempts += 1
        result = episode(flat, rng)
        if result is None:
            continue
        x, y = result
        index = len(rows)
        name = f"sim-{index:03d}.npz"
        np.savez_compressed(args.output / name, x=x, y=y,
                            timestamp_ns=(np.arange(len(x), dtype=np.int64)
                                          * int(DT * 1e9)))
        digest = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        rows.append({"episode": f"sim-{index:03d}", "file": name, "status": "ok",
                     "sha256": digest, "samples": int(len(x))})
        total += int(len(x))
        if index % 10 == 0:
            print(f"  {index} episodes, {total} samples", flush=True)

    manifest = {"schema": "robotbrain.rosnav-shadow.dataset.v1",
                "samples": total, "rows": rows,
                "note": "expert rollouts of the production wavefront planner "
                        "over the flat's own map"}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "samples": total,
                      "attempts": attempts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
