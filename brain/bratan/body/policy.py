"""The trained navigation actor (assets/navigation/actor.onnx).

Input contract (228 floats, fixed by training): three 72-sector lidar views (now,
0.3 s ago, 2 s ago; min-pooled, log-encoded 0.15..12 m, sector 0 straight ahead),
five path points sampled over the next metre (base frame, /2 m), and the measured
twist (v/0.5, w/2). Output: (v, w).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..common.config import ASSETS
from .scan import LASER_YAW

BINS = 72
DELAYS = (.3, 2.)
R_MIN, R_MAX = .15, 12.
_LOG_SPAN = math.log(R_MAX / R_MIN)
_SECTOR = np.floor(((np.arange(360) * (2 * math.pi / 360) + LASER_YAW + math.pi / BINS) % (2 * math.pi))
                   / (2 * math.pi / BINS)).astype(int).clip(0, BINS - 1)


def pool(ranges) -> np.ndarray:
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where((r > 0) & np.isfinite(r), r, R_MAX).clip(R_MIN, R_MAX)
    pooled = np.full(BINS, R_MAX, dtype=np.float32)
    np.minimum.at(pooled, _SECTOR, r)
    return (np.log(pooled / R_MIN) / _LOG_SPAN).clip(0, 1).astype(np.float32)


def sample_path(local, count=5, lookahead=1.):
    xy = np.asarray(local, dtype=np.float32).reshape(-1, 2)
    if len(xy) == 1:
        out = np.repeat(xy, count, axis=0)
    else:
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.concatenate(([0.], np.cumsum(seg)))
        total = float(cum[-1])
        if total < 1e-6:
            out = np.repeat(xy[-1:], count, axis=0)
        else:
            d = np.linspace(min(total, lookahead) / count, min(total, lookahead), count)
            out = np.column_stack((np.interp(d, cum, xy[:, 0]), np.interp(d, cum, xy[:, 1])))
    return np.clip(out / 2., -1, 1).reshape(-1).astype(np.float32)


class Actor:
    def __init__(self, model=ASSETS / 'navigation/actor.onnx', threads=1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[-1] != BINS * 3 + 12:
            raise ValueError('actor_contract_mismatch')
        self.history = deque()

    def observe(self, stamp, ranges):
        """Feed every scan (keeps the 2 s history the actor needs)."""
        if self.history and (stamp <= self.history[-1][0] or stamp - self.history[-1][0] > .75):
            self.history.clear()
        self.history.append((stamp, pool(ranges)))
        while len(self.history) > 2 and self.history[1][0] < stamp - DELAYS[-1] - .75:
            self.history.popleft()

    @property
    def ready(self):
        return bool(self.history) and self.history[-1][0] - self.history[0][0] >= DELAYS[-1]

    def command(self, ranges, local_path, twist, max_speed=.24):
        now = pool(ranges)
        past = []
        stamp = self.history[-1][0] if self.history else 0.
        for delay in DELAYS:
            view = now
            for t, sample in reversed(self.history):
                if t <= stamp - delay + 1e-9:
                    view = sample
                    break
            past.append(view)
        vector = np.concatenate([now] + past + [sample_path(local_path),
                                np.array([np.clip(twist[0] / .5, -1, 1), np.clip(twist[1] / 2., -1, 1)], dtype=np.float32)])
        out = np.asarray(self.session.run(None, {self.input: vector[None].astype(np.float32)})[0]).reshape(-1)
        v, w = float(out[0]), float(out[1])
        if not (math.isfinite(v) and math.isfinite(w)):
            return 0., 0.
        if v > max_speed:                     # keep the learned curvature at a lower top speed
            w *= max_speed / v
            v = max_speed
        return max(0., v), float(np.clip(w, -.25, .25))
