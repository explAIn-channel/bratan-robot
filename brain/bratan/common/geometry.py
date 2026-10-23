"""SE(2) poses, the robot footprint and fast vectorized point tests.

A pose is a tuple (x, y, yaw). Points are (N, 2) float arrays.
"""
from __future__ import annotations

import math

import numpy as np

TAU = 2 * math.pi


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def relative(a, b):
    """Pose b expressed in the frame of pose a."""
    return compose(inverse(a), b)


def transform(pose, points) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + (pose[0], pose[1])


def twist_pose(v: float, w: float, seconds: float):
    """Exact displacement along a constant (v, w) arc."""
    half = w * seconds * .5
    distance = v * seconds * (math.sin(half) / half if half else 1.)
    return (distance * math.cos(half), distance * math.sin(half), 2 * half)


def interpolate(samples, stamp):
    """Pose at `stamp` from a time-sorted sequence of (t, pose); None outside it."""
    if not samples or stamp < samples[0][0] or stamp > samples[-1][0]:
        return None
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= stamp:
            lo = mid
        else:
            hi = mid
    (t0, p0), (t1, p1) = samples[lo], samples[hi]
    f = 0. if t1 == t0 else (stamp - t0) / (t1 - t0)
    f = min(1., max(0., f))
    return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]),
            wrap(p0[2] + f * wrap(p1[2] - p0[2])))


# Assembled body outline (base frame, metres): 408.6 x 285.9 mm, lidar at origin.
FOOTPRINT = np.array([[-.1827, -.1168], [.100, -.1429509], [.145, -.1429509],
                      [.2259172, -.1168], [.2259172, .1168], [.145, .1429509],
                      [.100, .1429509], [-.1827, .1168]])
BODY_RADIUS = float(np.linalg.norm(FOOTPRINT, axis=1).max())   # ~.248 m
HALF_WIDTH = .1429509
_EDGES = np.roll(FOOTPRINT, -1, axis=0) - FOOTPRINT
_NORMALS = np.stack((_EDGES[:, 1], -_EDGES[:, 0]), axis=1)       # outward for CCW
_NORMALS /= np.linalg.norm(_NORMALS, axis=1, keepdims=True)
_OFFSETS = np.einsum('ij,ij->i', _NORMALS, FOOTPRINT)
if np.any(np.einsum('ij,ij->i', _NORMALS, FOOTPRINT.mean(axis=0) - FOOTPRINT) > 0):
    _NORMALS, _OFFSETS = -_NORMALS, -_OFFSETS


def inside_footprint(points, margin: float = 0.) -> np.ndarray:
    """Boolean mask: point lies inside the body outline grown by `margin`."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return np.zeros(0, dtype=bool)
    return np.all(points @ _NORMALS.T - _OFFSETS <= margin, axis=1)


def footprint_clearance(points) -> np.ndarray:
    """Signed distance of points to the body outline (negative inside)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    delta = points[:, None, :] - FOOTPRINT[None]
    t = np.clip(np.einsum('nki,ki->nk', delta, _EDGES) / np.einsum('ki,ki->k', _EDGES, _EDGES), 0, 1)
    nearest = delta - t[..., None] * _EDGES[None]
    distance = np.sqrt(np.min(np.einsum('nki,nki->nk', nearest, nearest), axis=1))
    return np.where(inside_footprint(points), -distance, distance)


def segment_projection(path, point):
    """(index, closest point, distance) of `point` on a polyline."""
    best = (math.inf, 0, tuple(path[0]))
    x, y = point[0], point[1]
    for i in range(len(path) - 1):
        ax, ay = path[i]
        dx, dy = path[i + 1][0] - ax, path[i + 1][1] - ay
        f = min(1., max(0., ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy + 1e-12)))
        px, py = ax + f * dx, ay + f * dy
        d = math.hypot(x - px, y - py)
        if d < best[0]:
            best = (d, i, (px, py))
    return best[1], best[2], best[0]


def remaining_path(pose, path):
    """The part of a polyline still ahead of the robot, starting at its projection."""
    if len(path) < 2:
        return [tuple(p) for p in path]
    index, point, _ = segment_projection(path, pose)
    return [point] + [tuple(p) for p in path[index + 1:]]


def path_length(path) -> float:
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def lookahead_point(path, distance):
    """Point `distance` metres along a polyline (last point if shorter)."""
    left = distance
    for i in range(len(path) - 1):
        step = math.dist(path[i], path[i + 1])
        if step >= left and step > 0:
            f = left / step
            return (path[i][0] + f * (path[i + 1][0] - path[i][0]),
                    path[i][1] + f * (path[i + 1][1] - path[i][1]))
        left -= step
    return tuple(path[-1])
