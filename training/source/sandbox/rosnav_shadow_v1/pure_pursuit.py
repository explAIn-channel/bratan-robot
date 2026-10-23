#!/usr/bin/env python3
"""A geometric controller, so the learned policy has an honest opponent.

The trained model was only ever compared against a baseline that outputs
nothing at all.  Beating "do nothing" by 65 % is the floor, not evidence, and
without this file we could not say whether a 145 KB network earns its place
over twenty lines of geometry.

This is pure pursuit: aim at a point some way along the path, turn with the
curvature of the arc that reaches it, and slow down for whatever is close.  It
reads exactly the feature vector the network reads -- the same pooled LiDAR,
the same five path points, the same velocity -- so neither side gets to see
something the other cannot.

Its handful of constants are fitted on the training episodes and reported on
the test episodes, which is the discipline the network was held to.
"""
from __future__ import annotations

import numpy as np

from features import LIDAR_BINS, PATH_POINTS, PATH_SCALE_M, SCAN_VIEWS

#: Sectors counted from straight ahead that count as "in front of me".
FORWARD_SECTORS = 9


def path_from_features(features: np.ndarray) -> np.ndarray:
    """The five path points, back in metres, in the robot's own frame."""
    start = LIDAR_BINS * SCAN_VIEWS
    flat = features[..., start:start + PATH_POINTS * 2]
    return flat.reshape(*flat.shape[:-1], PATH_POINTS, 2) * PATH_SCALE_M


def forward_clearance(features: np.ndarray) -> np.ndarray:
    """Smallest normalised range within a cone around straight ahead.

    Sector zero is centred on forward and sectors run around the circle, so
    the cone is the first few sectors and the last few together.
    """
    lidar = features[..., :LIDAR_BINS]
    head = lidar[..., :FORWARD_SECTORS + 1]
    tail = lidar[..., LIDAR_BINS - FORWARD_SECTORS:]
    return np.minimum(head.min(axis=-1), tail.min(axis=-1))


def predict(features: np.ndarray, *, lookahead_m: float, max_linear_mps: float,
            curvature_gain: float, max_angular_radps: float,
            clearance_floor: float, align_rad: float,
            align_gain: float) -> np.ndarray:
    """Return ``(linear, angular)`` for one or many feature rows.

    Differential drives do not steer like cars, and neither does the planner
    this is competing with: when the path leaves at a steep angle it turns on
    the spot first and drives afterwards.  Nearly two fifths of the expert's
    samples are stationary for exactly that reason.  A pursuit controller that
    always rolls forward would be losing to a behaviour it was never allowed
    to have, so it gets the same turn-then-go rule.
    """
    single = features.ndim == 1
    rows = features[None, :] if single else features

    path = path_from_features(rows)
    distance = np.linalg.norm(path, axis=-1)

    # The aim point: the first one far enough away, else the furthest we have.
    far_enough = distance >= lookahead_m
    index = np.where(far_enough.any(axis=-1),
                     far_enough.argmax(axis=-1),
                     distance.argmax(axis=-1))
    take = np.arange(rows.shape[0])
    aim = path[take, index]
    aim_distance = np.maximum(distance[take, index], 1e-3)

    heading = np.arctan2(aim[:, 1], aim[:, 0])
    curvature = 2.0 * aim[:, 1] / (aim_distance ** 2)

    # Rolling: turn with the arc, slow for corners and for whatever is close.
    room = np.clip((forward_clearance(rows) - clearance_floor)
                   / max(1e-6, 1.0 - clearance_floor), 0.0, 1.0)
    straightness = 1.0 / (1.0 + np.abs(curvature))
    linear = max_linear_mps * room * straightness
    angular = np.clip(curvature_gain * curvature,
                      -max_angular_radps, max_angular_radps)

    # Turning on the spot: no forward motion at all until roughly aligned.
    misaligned = np.abs(heading) > align_rad
    linear = np.where(misaligned, 0.0, linear)
    angular = np.where(misaligned,
                       np.clip(align_gain * heading,
                               -max_angular_radps, max_angular_radps),
                       angular)

    # A path that has run out is a goal reached, not a reason to keep going.
    arrived = distance.max(axis=-1) < 0.02
    linear = np.where(arrived, 0.0, linear)
    angular = np.where(arrived, 0.0, angular)

    out = np.stack([linear, angular], axis=-1).astype(np.float32)
    return out[0] if single else out


DEFAULTS = {"lookahead_m": 0.45, "max_linear_mps": 0.18,
            "curvature_gain": 0.28, "max_angular_radps": 0.25,
            "clearance_floor": 0.02, "align_rad": 0.25, "align_gain": 1.0}
