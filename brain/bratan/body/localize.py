"""Scan-to-map localization: Gauss-Newton on the wall distance field.

`match` refines a predicted pose in ~1-3 ms (analytic Jacobian, robust weights),
so it runs on every lidar turn. `recover` searches orientations at the predicted
position and then the whole free floor when the robot is lost or was carried.
`DockReference` matches against the panorama recorded while charging, which pins
the final centimetres of docking without depending on map quality.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..common.geometry import compose, relative, transform, wrap
from .grid import FIELD_CAP

OUTLIER_M = .30
TRACK_MEAN_MAX = .16           # tracking never accepts a worse mean than this
TRACK_HELD_FRACTION = .38      # the pose held: a view with this much on walls counts (mean < .14) ...
TRACK_HELD_STRAY = .48         # ... or this much with stray points from someone close by (mean < .16)
HUBER_M = .05


@dataclass
class Match:
    ok: bool
    pose: tuple
    mean: float          # mean clipped residual (m)
    fraction: float      # share of points within 10 cm of a wall
    correction: float    # metres moved from the prediction
    turn: float          # radians turned from the prediction
    points: int


def _subsample(points, n=240):
    step = max(1, len(points) // n)
    return points[::step]


def refine(grid, seed, points, iterations=12):
    """Gauss-Newton with Huber weights; returns (pose, residuals)."""
    x, y, yaw = seed
    for _ in range(iterations):
        c, s = math.cos(yaw), math.sin(yaw)
        wx = points[:, 0] * c - points[:, 1] * s + x
        wy = points[:, 0] * s + points[:, 1] * c + y
        value, gx, gy = grid.sample(np.column_stack((wx, wy)))
        dyaw = gx * (-points[:, 0] * s - points[:, 1] * c) + gy * (points[:, 0] * c - points[:, 1] * s)
        weight = np.where(value < OUTLIER_M, np.minimum(1., HUBER_M / np.maximum(value, 1e-6)), 0.)
        jac = np.column_stack((gx, gy, dyaw))
        jw = jac * weight[:, None]
        h = jw.T @ jac + np.diag((1e-3, 1e-3, 1e-3))
        g = jw.T @ value
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        # Trust region: no single step jumps farther than the cell scale.
        step[:2] = np.clip(step[:2], -.08, .08)
        step[2] = float(np.clip(step[2], -.12, .12))
        x, y, yaw = x + step[0], y + step[1], wrap(yaw + step[2])
        if abs(step[0]) + abs(step[1]) < 5e-4 and abs(step[2]) < 5e-4:
            break
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.column_stack((points[:, 0] * c - points[:, 1] * s + x, points[:, 0] * s + points[:, 1] * c + y))
    value, _, _ = grid.sample(world)
    return (float(x), float(y), float(yaw)), value


def score(predicted, pose, residual, n, search, yaw_search, min_fraction=.45):
    mean = float(np.minimum(residual, OUTLIER_M).mean())
    fraction = float((residual <= .10).mean())
    correction = math.dist(pose[:2], predicted[:2])
    turn = abs(wrap(pose[2] - predicted[2]))
    ok = mean < .14 and fraction >= min_fraction and correction < search and turn < yaw_search
    return Match(ok, pose, mean, fraction, correction, turn, n)


def match(grid, predicted, scan_points, search=.30, yaw_search=.40):
    if len(scan_points) < 40:
        return Match(False, tuple(predicted), .5, 0., 0., 0., len(scan_points))
    pts = _subsample(scan_points)
    pose, residual = refine(grid, predicted, pts)
    result = score(predicted, pose, residual, len(pts), search, yaw_search)
    if not result.ok and result.mean < TRACK_MEAN_MAX and result.correction < .10 and result.turn < .10:
        # The pose held, but the view is poor: furniture hides the walls (10.10, the bedroom doorway: 42-45 % of the
        # scan on walls, pose within 1-4 cm of the next sure fix) or a person close by adds stray points (22:05,
        # 24 scans in a row at mean 0.140-0.148 with 51-55 % on walls). Lost every 10-30 s before; replayed on 4548
        # recorded scans the poses so accepted were never more than 6 cm from the next sure fix.
        if ((result.mean < .14 and result.fraction >= TRACK_HELD_FRACTION) or result.fraction >= TRACK_HELD_STRAY):
            result = Match(True, result.pose, result.mean, result.fraction, result.correction, result.turn, result.points)
    return result


def recover(grid, predicted, scan_points, deadline_s=6., cancelled=lambda: False):
    """Best unambiguous pose: local orientations first, then a global floor search."""
    if len(scan_points) < 60:
        return None
    pts = _subsample(scan_points, 200)
    candidates = []
    for k in range(12):
        seed = (predicted[0], predicted[1], wrap(predicted[2] + k * math.pi / 6))
        pose, residual = refine(grid, seed, pts)
        candidates.append(score(seed, pose, residual, len(pts), 1., 1.))
    best = _choose(candidates)
    if best is not None:
        return best
    started = time.monotonic()
    coarse = _subsample(pts, 60)
    free = np.argwhere((grid.pixels >= 250)[::4, ::4]) * 4           # 20 cm lattice of floor
    if not len(free):
        return None
    centres = np.column_stack((grid.origin[0] + (free[:, 1] + .5) * grid.resolution,
                               grid.origin[1] + (grid.height - free[:, 0] - .5) * grid.resolution))
    scored = []
    for yaw in np.arange(-math.pi, math.pi, math.radians(10)):
        if cancelled() or time.monotonic() - started > deadline_s:
            break
        c, s = math.cos(yaw), math.sin(yaw)
        rotated = coarse @ np.array([[c, s], [-s, c]])
        world = (centres[:, None, :] + rotated[None]).reshape(-1, 2)
        value = grid.sample(world)[0].reshape(len(centres), len(coarse))
        cost = np.minimum(value, OUTLIER_M).mean(axis=1)
        for i in np.argsort(cost)[:4]:
            scored.append((float(cost[i]), (float(centres[i, 0]), float(centres[i, 1]), float(yaw))))
    scored.sort()
    for _, seed in scored[:24]:
        if cancelled():
            break
        pose, residual = refine(grid, seed, pts, iterations=20)
        candidates.append(score(seed, pose, residual, len(pts), 1., 1.))
    return _choose(candidates, strict=True)


RECOVER_MIN_FRACTION = .65    # a re-found pose must explain most of the scan (C27, 07.10)
RECOVER_MAX_MEAN = .10


def _choose(candidates, strict=False):
    # Near a wrong guess (carried, wheels slipped) a pose 3 m off fitted 47 % of the scan and was taken
    # (22 of 147 recoveries in the 07.10 recording); the true poses fit 62-95 %. With 65 %: 0 wrong.
    good = sorted((c for c in candidates if c.ok and c.fraction >= RECOVER_MIN_FRACTION), key=lambda c: c.mean)
    if not good:
        return None
    best = good[0]
    # Reject when a clearly different pose explains the scan almost as well.
    for other in candidates:
        if (other.mean < best.mean + .02 and other.fraction > .45 and
                (math.dist(other.pose[:2], best.pose[:2]) > .25 or abs(wrap(other.pose[2] - best.pose[2])) > .35)):
            return None
    if strict and (best.mean > RECOVER_MAX_MEAN or best.fraction < RECOVER_MIN_FRACTION):
        return None
    return best


def densify(points, gap=.25, spacing=.02):
    """Join neighbouring returns of one surface so far walls are lines, not dots."""
    points = np.asarray(points, dtype=float)
    order = np.argsort(np.arctan2(points[:, 1], points[:, 0]))
    ordered = points[order]
    out = [ordered]
    nxt = np.roll(ordered, -1, axis=0)
    span = np.linalg.norm(nxt - ordered, axis=1)
    for a, b, d in zip(ordered, nxt, span):
        if spacing < d < gap:
            f = np.arange(spacing, d, spacing)[:, None] / d
            out.append(a + f * (b - a))
    return np.vstack(out)


def normals(points, tree, k=7):
    """Unit surface normals from the local covariance of each point's neighbours."""
    _, index = tree.query(points, k=min(k, len(points)))
    neighbours = points[index] - points[index].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', neighbours, neighbours)
    _, vectors = np.linalg.eigh(cov)
    return vectors[:, :, 0]


class DockReference:
    """Charging panorama: scan points stored in the dock frame (dock = origin)."""

    def __init__(self, points=None, dock=None):
        self.points = None if points is None else np.asarray(points, dtype=float)
        self.dock = None if dock is None else tuple(dock)
        self.tree = None if self.points is None else cKDTree(self.points)
        self.normals = None if self.points is None else normals(self.points, self.tree)
        self.samples = []

    @property
    def ready(self):
        return self.tree is not None and len(self.points) >= 100

    def record(self, dock, scan_points):
        """Called while charging and still: the scan *is* the dock frame view."""
        if len(scan_points) < 100:
            return False
        self.points = densify(scan_points)
        self.dock = tuple(dock)
        self.tree = cKDTree(self.points)
        self.normals = normals(self.points, self.tree)
        self.samples.clear()
        return True

    def match(self, predicted, scan_points, stamp, odom):
        """Pose of the robot in the map via the dock panorama, or None.

        Three consecutive fits must agree on the dock-to-odometry transform
        (1.5 cm / 1.5 deg) before the result is trusted.
        """
        if not self.ready or len(scan_points) < 100:
            return None
        seed = relative(self.dock, predicted)
        if math.hypot(seed[0], seed[1]) > 1.2 or abs(seed[2]) > .5:
            self.samples.clear()
            return None
        pts = _subsample(scan_points, 240)
        x, y, yaw = seed
        for _ in range(12):
            c, s_ = math.cos(yaw), math.sin(yaw)
            world = pts @ np.array([[c, s_], [-s_, c]]) + (x, y)
            distance, index = self.tree.query(world, distance_upper_bound=.25)
            near = np.isfinite(distance)
            if near.sum() < 60:
                self.samples.clear()
                return None
            normal = self.normals[index[near]]
            delta = world[near] - self.points[index[near]]
            residual = np.einsum('ij,ij->i', normal, delta)
            local = pts[near]
            dyaw = normal[:, 0] * (-local[:, 0] * s_ - local[:, 1] * c) + normal[:, 1] * (local[:, 0] * c - local[:, 1] * s_)
            jac = np.column_stack((normal, dyaw))
            weight = np.minimum(1., .02 / np.maximum(np.abs(residual), 1e-6))
            jw = jac * weight[:, None]
            step = -np.linalg.solve(jw.T @ jac + np.eye(3) * 1e-4, jw.T @ residual)
            step = np.clip(step, (-.05, -.05, -.06), (.05, .05, .06))
            x, y, yaw = x + step[0], y + step[1], wrap(yaw + step[2])
            if np.abs(step).max() < 3e-4:
                break
        world = transform((x, y, yaw), pts)
        distance = self.tree.query(world)[0]
        inliers = distance < .04
        fraction = float(inliers.mean())
        rms = float(np.sqrt(np.mean(distance[inliers] ** 2))) if inliers.any() else 1.
        if fraction < .6 or rms > .02 or math.hypot(x - seed[0], y - seed[1]) > .3 or abs(wrap(yaw - seed[2])) > .3:
            self.samples.clear()
            return None
        pose = compose(self.dock, (x, y, yaw))
        anchor = compose(pose, (0, 0, 0))
        # Consensus on the fixed map-from-odometry transform across fits.
        map_from_odom = compose(anchor, relative(odom, (0., 0., 0.)))
        self.samples = [s for s in self.samples if stamp - s[0] < 1.5][-2:] + [(stamp, map_from_odom)]
        ref = self.samples[0][1]
        consistent = all(math.dist(m[:2], ref[:2]) < .015 and abs(wrap(m[2] - ref[2])) < .026
                         for _, m in self.samples)
        if not consistent:
            self.samples = self.samples[-1:]
            return None
        if len(self.samples) < 3:
            return None
        return Match(True, pose, rms, fraction, math.dist(pose[:2], predicted[:2]),
                     abs(wrap(pose[2] - predicted[2])), len(pts))

    def as_dict(self):
        if not self.ready:
            return None
        return {'dock': list(self.dock), 'points': np.round(self.points, 4).tolist()}

    @classmethod
    def from_dict(cls, value):
        if not value:
            return cls()
        return cls(value['points'], value['dock'])
