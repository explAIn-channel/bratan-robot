"""Tight-place maneuvers: a small Hybrid-A* over (x, y, heading) with the exact body outline.

Used when plain route following cannot move (a doorway entered at an angle, a table
leg by the path). The search runs in the robot's own frame against the points the
lidar has measured recently, with short forward/backward arcs and turns on the spot,
like parking a car. The result is a list of (v, w, seconds) moves.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..common.geometry import FOOTPRINT, twist_pose, compose, wrap

CELL = .04
HALF = 1.8                      # search window: +-1.8 m around the robot
HEADINGS = 24
MARGIN = .02
# (v, w, seconds): 12 cm arcs forward, 8 cm back, 20-degree turns on the spot
MOVES = [(.12, 0., 1.), (.12, .35, 1.), (.12, -.35, 1.),
         (-.08, 0., 1.), (-.08, .35, 1.), (-.08, -.35, 1.),
         (0., .35, 1.), (0., -.35, 1.)]


def _perimeter(step=.03):
    points = []
    for a, b in zip(FOOTPRINT, np.roll(FOOTPRINT, -1, axis=0)):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(n):
            points.append(a + (b - a) * k / n)
    return np.array(points)


PERIMETER = _perimeter()


class Field:
    def __init__(self, points):
        size = int(2 * HALF / CELL)
        occupied = np.zeros((size, size), dtype=bool)
        if len(points):
            ij = np.floor((np.asarray(points) + HALF) / CELL).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < size) & (ij[:, 1] >= 0) & (ij[:, 1] < size)
            occupied[ij[ok, 0], ij[ok, 1]] = True
        self.size = size
        self.distance = distance_transform_edt(~occupied) * CELL if occupied.any() else np.full((size, size), 9.)

    def clear(self, pose, margin=MARGIN):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        x = PERIMETER[:, 0] * c - PERIMETER[:, 1] * s + pose[0]
        y = PERIMETER[:, 0] * s + PERIMETER[:, 1] * c + pose[1]
        i = np.floor((x + HALF) / CELL).astype(int)
        j = np.floor((y + HALF) / CELL).astype(int)
        if (i < 0).any() or (j < 0).any() or (i >= self.size).any() or (j >= self.size).any():
            return False
        # A point exactly on a cell centre is up to half a cell from the surface it marks.
        return bool(np.all(self.distance[i, j] >= margin + CELL * .5))


def plan(points, goal, *, goal_tolerance=.12, max_nodes=6000, deadline_s=1.5):
    """Moves from (0,0,0) to within goal_tolerance of goal (x, y[, heading]) in the robot frame."""
    started = time.monotonic()
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points):
        # Points inside the body where it stands are not obstacles one could avoid: remembered returns smeared by
        # odometry, or something already touching. They made every move «unsafe» (08.10 11:30: the robot backed
        # against a chair leg could not plan its way out, then neither the trip home).
        from ..common.geometry import footprint_clearance
        points = points[footprint_clearance(points) > .005]
    field = Field(points)
    if not field.clear((0., 0., 0.), margin=0.):
        start_margin = 0.
    else:
        start_margin = MARGIN
    gx, gy = goal[0], goal[1]
    goal_heading = goal[2] if len(goal) > 2 else None

    def key(pose):
        return (int(round(pose[0] / CELL)), int(round(pose[1] / CELL)),
                int(round(pose[2] / (2 * math.pi / HEADINGS))) % HEADINGS)

    def heuristic(pose):
        h = math.hypot(gx - pose[0], gy - pose[1])
        if goal_heading is not None:
            h += .1 * abs(wrap(goal_heading - pose[2]))
        return h

    start = (0., 0., 0.)
    frontier = [(heuristic(start), 0., 0, start)]
    parents = {key(start): (None, None)}
    poses = {key(start): start}
    counter = 0
    while frontier and counter < max_nodes and time.monotonic() - started < deadline_s:
        _, cost, _, pose = heapq.heappop(frontier)
        if math.hypot(gx - pose[0], gy - pose[1]) < goal_tolerance and (
                goal_heading is None or abs(wrap(goal_heading - pose[2])) < .4):
            moves = []
            k = key(pose)
            while parents[k][0] is not None:
                parent, move = parents[k]
                moves.append(move)
                k = parent
            return list(reversed(moves))
        for move in MOVES:
            v, w, t = move
            nxt = compose(pose, twist_pose(v, w, t))
            k = key(nxt)
            if k in parents:
                continue
            # Check the middle of the move as well as its end.
            mid = compose(pose, twist_pose(v, w, t / 2))
            margin = start_margin if cost < .3 else MARGIN
            if not (field.clear(mid, margin) and field.clear(nxt, margin)):
                continue
            step = abs(v) * t + .08 * abs(w) * t + (.12 if v < 0 else 0.) + (.05 if not v else 0.)
            parents[k] = (key(pose), move)
            counter += 1
            heapq.heappush(frontier, (cost + step + heuristic(nxt), cost + step, counter, nxt))
    return None
