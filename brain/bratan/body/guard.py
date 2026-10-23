"""Last check before every wheel command: bumper, front IR, swept body vs measured points.

Everything is in the base frame at command time. Lidar points come from the
latest scan moved by the odometry since that scan; camera obstacles are map discs
moved into the base frame. A command passes only if the body swept along the
commanded arc for the horizon stays clear.
"""
from __future__ import annotations

import math

import numpy as np

from ..common.geometry import BODY_RADIUS, compose, inside_footprint, inverse, transform, twist_pose

SWEEP_TIMES = (.1, .25, .45, .7)
PIVOT_MARGIN_M = .015
NEAR_M = 1.2


def swept_hit(points, v, w, horizon=.7, margin=.0):
    """True if the footprint hits any base-frame point along the (v, w) arc."""
    if not len(points) or (not v and not w):
        return False
    reach = BODY_RADIUS + abs(v) * horizon + .05
    near = points[np.einsum('ij,ij->i', points, points) <= reach * reach]
    if not len(near):
        return False
    for t in SWEEP_TIMES:
        if t > horizon + 1e-6:
            break
        body = twist_pose(v, w, t)
        local = transform(inverse(body), near)
        if inside_footprint(local, margin).any():
            return True
    return False


MEMORY_S = 3.0           # recent scans kept: a jamb that slid into the lidar's 0.2 m blind zone stays known
WALL_REACH_M = .7


class Guard:
    def __init__(self, ir_mask=6, ir_active_low=True):
        self.ir_mask, self.ir_active_low = ir_mask, ir_active_low
        self.points = np.zeros((0, 2))      # base frame at self.odom
        self.odom = (0., 0., 0.)
        self.history = []                   # (stamp, odom, near points in that base frame)
        self.walls = None                   # (row/col -> xy) of structural walls, set by the body
        self._cache = (None, None)
        self.live_only = False              # during a run-up: wheel slip makes remembered points unreliable
        self.discs = []                      # (x, y, radius) in map frame
        self.ignore = None                   # callable(points_base) -> mask, e.g. dock base during docking

    def update_scan(self, points_base, scan_odom, stamp=0.):
        self.points, self.odom = np.asarray(points_base, dtype=float), tuple(scan_odom)
        near = self.points[np.einsum('ij,ij->i', self.points, self.points) < 1.0] if len(self.points) else self.points
        self.history = [h for h in self.history if stamp - h[0] < MEMORY_S][-8:] + [(stamp, tuple(scan_odom), near)]

    def nearby(self, odom, map_pose):
        """Live points + recent near points + map walls, all in the current base frame."""
        key = (tuple(round(v, 3) for v in odom), None if map_pose is None else tuple(round(v, 3) for v in map_pose),
               self.live_only)
        if self._cache[0] == key:
            return self._cache[1]
        parts = []
        if len(self.points):
            parts.append(transform(compose(inverse(odom), self.odom), self.points))
        for _, old_odom, near in ([] if self.live_only else self.history[:-1]):
            if len(near):
                parts.append(transform(compose(inverse(odom), old_odom), near))
        if self.walls is not None and map_pose is not None:
            xy = self.walls
            d = xy - np.asarray(map_pose[:2])
            close = xy[np.einsum('ij,ij->i', d, d) < WALL_REACH_M ** 2]
            if len(close):
                parts.append(transform(inverse(map_pose), close))
        points = np.vstack(parts) if parts else np.zeros((0, 2))
        self._cache = (key, points)
        return points

    def check(self, body, odom, map_pose, v, w, horizon=.7):
        """Return a stop reason or None."""
        if not v and not w:
            return None
        bumper = int(body.get('bumper', 0) or 0)
        if bumper and not (v < 0 and not w):
            return 'bumper'                      # only a straight retreat may leave contact
        ir = int(body.get('ir', 15)) ^ (15 if self.ir_active_low else 0)
        if v > 0 and ir & self.ir_mask:
            return 'front_ir'
        points = self.nearby(odom, map_pose)
        if self.ignore is not None and len(points):
            points = points[~self.ignore(points)]
        if self.discs and map_pose is not None:
            extra = []
            to_base = inverse(map_pose)
            for x, y, r in self.discs:
                bx, by, _ = compose(to_base, (x, y, 0.))
                if math.hypot(bx, by) < NEAR_M + r:
                    k = max(4, int(2 * math.pi * r / .04))
                    angles = np.linspace(0, 2 * math.pi, k, endpoint=False)
                    extra.append(np.column_stack((bx + r * np.cos(angles), by + r * np.sin(angles))))
                    extra.append([[bx, by]])
            if extra:
                points = np.vstack([points] + [np.asarray(e, dtype=float) for e in extra])
        horizon = min(horizon, max(.45, .25 + abs(v) * 2.))
        # Turning in place sweeps the corners sideways: keep 1.5 cm for screws and wheels.
        margin = PIVOT_MARGIN_M if abs(v) < .03 else 0.
        if swept_hit(points, v, w, horizon, margin):
            # For the telemetry: the nearest point in the swept area (base frame), what stopped us.
            if len(points):
                d = np.einsum('ij,ij->i', points, points)
                self.last_hit = [round(float(x), 3) for x in points[int(np.argmin(d))]] + [round(v, 2), round(w, 2)]
            return 'obstacle'
        return None
