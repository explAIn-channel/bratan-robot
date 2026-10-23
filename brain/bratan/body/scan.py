"""Lidar scan -> base-frame points. 360 one-degree bins, Delta-2D, mounted 86 deg off the bow."""
from __future__ import annotations

import json
import math

import numpy as np

from ..common.config import ASSETS
from ..common.geometry import inside_footprint

_MOUNT = json.loads((ASSETS / 'lidar_mounting.json').read_text())
LASER_YAW = float(_MOUNT['yaw_rad'])
LASER_HEIGHT = float(_MOUNT['height_m'])
RANGE_MIN, RANGE_MAX = .2, 8.
_ANGLES = np.arange(360) * (2 * math.pi / 360) + LASER_YAW
_COS, _SIN = np.cos(_ANGLES), np.sin(_ANGLES)


def clean_ranges(ranges, qualities=None, quality_max=110, near_max=.80):
    """Zero isolated weak near returns (glare speckles); keep everything else.

    07.10, sun through an open door: phantom returns at 0.33-0.54 m always toward the window, quality
    40-100 and alone or in pairs; real surfaces near the robot measured 140-180. Those phantoms held the
    robot dancing on a free spot. Measured in TEST_JOURNAL C24.
    """
    r = np.asarray(ranges, dtype=float)
    r = np.where(np.isfinite(r), r, 0.)
    if qualities is None or len(qualities) != len(r):
        return r
    q = np.asarray(qualities, dtype=float)
    valid = r >= RANGE_MIN
    support = np.ones(len(r), dtype=int)
    for k in (-2, -1, 1, 2):
        n = np.roll(r, k)
        support += (n >= RANGE_MIN) & (np.abs(n - r) <= .08)
    speckle = valid & (r <= near_max) & (q > 0) & (q <= quality_max) & (support < 3)
    return np.where(speckle, 0., r)


def points(ranges) -> np.ndarray:
    """Valid returns as (N, 2) base-frame points, the robot's own body removed."""
    r = np.asarray(ranges, dtype=float)
    valid = (r >= RANGE_MIN) & (r <= RANGE_MAX)
    xy = np.column_stack((r[valid] * _COS[valid], r[valid] * _SIN[valid]))
    return xy[~inside_footprint(xy, .005)]


def bearing_index(bearing_base: float) -> int:
    """Scan bin whose ray points along a base-frame bearing."""
    return int(round(math.degrees(bearing_base - LASER_YAW))) % 360
