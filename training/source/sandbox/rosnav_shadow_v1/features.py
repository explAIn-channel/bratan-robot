"""Shared feature extraction for the offline trainer and ROS 2 shadow node."""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence

import numpy as np


LIDAR_BINS = 72
#: Rotation of the scan frame inside ``base_link``, measured 2026-08-19 over
#: three straight 1.2 m runs that drifted 4.4-5.1 cm per metre sideways while
#: the heading barely changed.  The production motion stack applies it; this
#: pipeline did not, so its LiDAR arrived rotated against the plan, which is
#: expressed in ``base_link``.  Two frames 2.7 degrees apart is exactly the
#: kind of mismatch a policy resolves as a standing steering offset.
#: Read from navigation/lidar_mounting.json, the same file the robot uses,
#: rather than held as a number here. It was 0.0473 rad -- 2.7 degrees --
#: inferred in August from straight runs that drifted sideways while the
#: heading barely changed. That is a weak witness for a frame rotation and it
#: was wrong by most of a right angle: measured 4 September, twice and two
#: ways, the laser sits about 86 degrees off the bow. Sector zero is supposed
#: to be straight ahead, so every policy fitted here read a world turned 84
#: degrees against the one it steered in.
def _mounting_yaw_rad() -> float:
    import json
    from pathlib import Path as _Path
    here = _Path(__file__).resolve()
    for candidate in (here.parents[2] / "navigation" / "lidar_mounting.json",
                      _Path("/opt/robotbrain/v5/navigation/lidar_mounting.json")):
        try:
            return float(json.loads(candidate.read_text())["yaw_rad"])
        except Exception:
            continue
    raise RuntimeError(
        "lidar_mounting.json not found; refusing to guess the laser yaw, "
        "because guessing it is what trained every policy before "
        "4 September on a world turned 84 degrees")


LASER_YAW_RAD = _mounting_yaw_rad()
PATH_POINTS = 5
PATH_LOOKAHEAD_M = 1.0

#: How a range in metres becomes a number the network reads.
#:
#: ``"linear"`` divides by the 12 m window, and it is what the board's
#: `features.rs` implements, what every parity fixture pins, and what the
#: weights currently on the robot were fitted against.  It is the default here
#: for exactly that reason: changing it silently would make the claim that the
#: two sides agree false without anything failing.
#:
#: Its weakness is where driving is decided.  Between 0.15 m and 0.35 m -- the
#: band in which a robot 0.175 m in radius either goes round something or hits
#: it -- the linear scale moves from 0.0125 to 0.0292, which is 1.7 % of its
#: range, while a wall eight metres away sits at 0.67.  The network is asked
#: to resolve the difference between a scrape and a clear pass out of a
#: hundredth of an input.
#:
#: ``"log"`` spends the scale where the decisions are: the same band takes
#: 19.3 % of it, an elevenfold expansion, and the far field compresses to
#: where nothing is decided anyway.  It is monotone and still fills [0, 1], so
#: nothing downstream changes shape.
#:
#: Adopting it is not a switch: the board's Rust twin, three generated parity
#: fixtures and the weights themselves would all have to be redone together,
#: and the ARM binary rebuilt and redeployed.  So it is offered, measured, and
#: only then argued for.
RANGE_ENCODINGS = ("linear", "log")
RANGE_ENCODING = "log"
PATH_SCALE_M = 2.0
LINEAR_SCALE_MPS = 0.5
ANGULAR_SCALE_RADPS = 2.0
#: How far back the extra scan views reach, in seconds.
#:
#: A single frame cannot say whether anything is moving, and the policy paid
#: for it: measured 2026-08-31 on identical placements, freezing the people in
#: the `walking` scenario took arrivals from 0.425 to 0.787 and collisions from
#: 0.550 to 0.100 -- the whole deficit was motion, and the network had no way
#: to see any.  It was treating a person walking into its path as furniture.
#:
#: 0.3 s answers "is this moving, and which way".  2.0 s answers the question
#: the owner actually posed: something that has held still that long is not
#: about to pass, so it should be driven around rather than waited for.
#:
#: An empty tuple restores the 84-number vector exactly, which is what the
#: board's `features.rs`, the three parity fixtures and the weights currently
#: on the robot are all pinned to.  Do not change this without regenerating
#: that set together -- the input width changes, so no weight survives it.
HISTORY_DELAYS_S: tuple[float, ...] = (0.3, 2.0)
HISTORY_MAXIMUM_GAP_S = 0.75
SCAN_VIEWS = 1 + len(HISTORY_DELAYS_S)
FEATURE_DIM = LIDAR_BINS * SCAN_VIEWS + PATH_POINTS * 2 + 2


class TemporalScanBuffer:
    """Causal timestamped history for the temporal policy contract.

    Runtime advice is withheld until the longest requested delay exists. A
    timestamp regression or a receive gap clears the history, so a view from a
    previous stream generation can never be presented as recent motion.
    """

    def __init__(self, delays: Sequence[float] = HISTORY_DELAYS_S, *,
                 maximum_gap_s: float = HISTORY_MAXIMUM_GAP_S):
        self.delays = tuple(float(delay) for delay in delays)
        if (any(not math.isfinite(delay) or delay <= 0.0
                for delay in self.delays) or
                tuple(sorted(self.delays)) != self.delays):
            raise ValueError("history delays must be finite, positive and sorted")
        self.maximum_gap_s = float(maximum_gap_s)
        if not math.isfinite(self.maximum_gap_s) or self.maximum_gap_s <= 0.0:
            raise ValueError("maximum history gap must be finite and positive")
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self.last_timestamp: float | None = None
        self.reset_count = 0
        self.last_reset_reason = "startup"

    @property
    def ready(self) -> bool:
        if not self.delays:
            return bool(self._samples)
        return bool(self._samples and
                    self._samples[-1][0] - self._samples[0][0] >=
                    self.delays[-1])

    def reset(self, reason: str) -> None:
        self._samples.clear()
        self.last_timestamp = None
        self.reset_count += 1
        self.last_reset_reason = str(reason)

    def append(self, timestamp_s: float, view: np.ndarray) -> list[np.ndarray] | None:
        stamp = float(timestamp_s)
        value = np.asarray(view, dtype=np.float32)
        if not math.isfinite(stamp) or value.shape != (LIDAR_BINS,) or not np.all(
                np.isfinite(value)):
            raise ValueError("invalid temporal scan sample")
        if self.last_timestamp is not None:
            delta = stamp - self.last_timestamp
            if delta <= 0.0:
                self.reset("timestamp_not_increasing")
            elif delta > self.maximum_gap_s:
                self.reset("receive_gap_too_large")
        self._samples.append((stamp, value.copy()))
        self.last_timestamp = stamp
        if self.delays:
            oldest_needed = stamp - self.delays[-1] - self.maximum_gap_s
            while (len(self._samples) > 1 and
                   self._samples[1][0] <= oldest_needed):
                self._samples.popleft()
        if not self.delays:
            return []
        if not self.ready:
            return None
        history: list[np.ndarray] = []
        for delay in self.delays:
            target = stamp - delay
            candidate = None
            for sample_stamp, sample in reversed(self._samples):
                if sample_stamp <= target + 1e-9:
                    candidate = (sample_stamp, sample)
                    break
            if candidate is None or target - candidate[0] > self.maximum_gap_s:
                self.reset("history_selection_gap")
                return None
            history.append(candidate[1].copy())
        return history


def pool_lidar(
    ranges: Iterable[float],
    range_min: float,
    range_max: float,
    bins: int = LIDAR_BINS,
    angle_min: float = 0.0,
    angle_increment: float | None = None,
    laser_yaw: float = LASER_YAW_RAD,
    encoding: str | None = None,
) -> np.ndarray:
    """Conservative min-pooled LiDAR sectors, expressed in ``base_link``.

    Each beam is placed in the sector its true body-frame angle falls in
    rather than by its position in the array, so the sectors line up with the
    plan and with the robot's own forward axis.  Sector zero is centred on
    straight ahead, which is also what makes the mirror below exact.
    """
    values = np.asarray(list(ranges), dtype=np.float32)
    if values.size < bins:
        raise ValueError(f"LiDAR has {values.size} beams, fewer than {bins} bins")
    usable_max = float(range_max) if math.isfinite(range_max) and range_max > 0 else 12.0
    usable_min = max(float(range_min), 0.01) if math.isfinite(range_min) else 0.01
    values = np.nan_to_num(values, nan=usable_max, posinf=usable_max, neginf=usable_min)
    values = np.clip(values, usable_min, usable_max)
    step = (2.0 * math.pi / values.size if angle_increment is None
            else float(angle_increment))
    angles = float(angle_min) + float(laser_yaw) + np.arange(values.size) * step
    width = 2.0 * math.pi / bins
    sector = np.floor(((angles + width / 2.0) % (2.0 * math.pi)) / width).astype(np.int64)
    np.clip(sector, 0, bins - 1, out=sector)
    pooled = np.full(bins, usable_max, dtype=np.float32)
    np.minimum.at(pooled, sector, values)
    return encode_ranges(pooled, usable_min, usable_max, encoding)


def encode_ranges(pooled: np.ndarray, usable_min: float, usable_max: float,
                  encoding: str | None = None) -> np.ndarray:
    """Turn pooled metres into the [0, 1] the network reads."""
    encoding = RANGE_ENCODING if encoding is None else encoding
    if encoding == "linear":
        return (pooled / usable_max).astype(np.float32)
    if encoding == "log":
        floor = max(usable_min, 1e-3)
        span = math.log(usable_max / floor)
        scaled = np.log(np.maximum(pooled, floor) / floor) / span
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown range encoding: {encoding}")


def sample_path_xy(points: Sequence[Sequence[float]], count: int = PATH_POINTS,
                   lookahead_m: float | None = PATH_LOOKAHEAD_M) -> np.ndarray:
    """Sample the first metre by arc length; explicit None replays old features."""
    if not points:
        raise ValueError("Path is empty")
    if lookahead_m is not None and (not math.isfinite(lookahead_m) or lookahead_m <= 0):
        raise ValueError("invalid path lookahead")
    xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(xy) == 1:
        sampled = np.repeat(xy, count, axis=0)
    else:
        segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)
        total = float(cumulative[-1])
        if total < 1e-6:
            sampled = np.repeat(xy[-1:], count, axis=0)
        else:
            horizon = total if lookahead_m is None else min(total, lookahead_m)
            distances = np.linspace(horizon / count, horizon, count, dtype=np.float32)
            sampled = np.column_stack(
                (
                    np.interp(distances, cumulative, xy[:, 0]),
                    np.interp(distances, cumulative, xy[:, 1]),
                )
            ).astype(np.float32)
    return np.clip(sampled / PATH_SCALE_M, -1.0, 1.0).reshape(-1)


def build_features(
    ranges: Iterable[float],
    range_min: float,
    range_max: float,
    path_points: Sequence[Sequence[float]],
    linear_velocity: float,
    angular_velocity: float,
    angle_min: float = 0.0,
    angle_increment: float | None = None,
    laser_yaw: float = LASER_YAW_RAD,
    encoding: str | None = None,
    history: Sequence[np.ndarray] | None = None,
    path_lookahead_m: float | None = PATH_LOOKAHEAD_M,
) -> np.ndarray:
    lidar = pool_lidar(ranges, range_min, range_max, angle_min=angle_min,
                       angle_increment=angle_increment, laser_yaw=laser_yaw,
                       encoding=encoding)
    if SCAN_VIEWS > 1:
        # A caller with no history yet -- the first frames of a drive -- gets
        # the present repeated, which reads as "nothing has moved".  That is
        # the truthful default: at the first frame nothing has been seen to
        # move, and it keeps the vector well-formed rather than zero-padded,
        # which would read as an obstacle pressed against the robot.
        if history is None:
            past = [lidar] * (SCAN_VIEWS - 1)
        else:
            past = [np.asarray(view, dtype=np.float32) for view in history]
            if len(past) != SCAN_VIEWS - 1:
                raise ValueError(
                    f"Expected {SCAN_VIEWS - 1} past views, got {len(past)}")
            for view in past:
                if view.shape != (LIDAR_BINS,):
                    raise ValueError(f"Past view has shape {view.shape}")
        lidar = np.concatenate([lidar] + past)
    path = sample_path_xy(path_points, lookahead_m=path_lookahead_m)
    velocity = np.asarray(
        [
            np.clip(linear_velocity / LINEAR_SCALE_MPS, -1.0, 1.0),
            np.clip(angular_velocity / ANGULAR_SCALE_RADPS, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    result = np.concatenate((lidar, path, velocity)).astype(np.float32)
    if result.shape != (FEATURE_DIM,):
        raise AssertionError(f"Unexpected feature shape: {result.shape}")
    return result

def mirror_features(features: np.ndarray) -> np.ndarray:
    """Reflect one observation about the robot's forward axis.

    The scan is pooled by beam index over a full turn starting at zero, so
    sector ``i`` spans ``[5i, 5i+5)`` degrees and its reflection spans
    ``[355-5i, 360-5i)`` -- sector ``bins-1-i``.  Reversing the sectors is
    therefore the mirror.  A path point mirrors by negating its lateral
    offset, and a rotation mirrors by changing sign.

    Training on the mirror of every sample is what forces the policy to be
    left-right symmetric.  Without it the model inherits whichever direction
    the recorded runs happened to favour: the first policy drifted visibly
    right on a straight one-metre path.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {features.shape[-1]}")
    out = features.copy()
    # Sector 0 straddles straight ahead, so it is its own mirror and the rest
    # reverse around it: sector ``i`` reflects to ``bins - i``.  Every scan
    # view mirrors the same way; reflecting only the newest would hand the
    # network a present that disagreed with its own past.
    for view in range(SCAN_VIEWS):
        start = view * LIDAR_BINS
        block = out[..., start:start + LIDAR_BINS]
        out[..., start:start + LIDAR_BINS] = np.concatenate(
            (block[..., :1], block[..., :0:-1]), axis=-1)
    scans = LIDAR_BINS * SCAN_VIEWS
    path = out[..., scans:scans + PATH_POINTS * 2]
    path[..., 1::2] *= -1.0                      # lateral offsets
    out[..., -1] *= -1.0                         # angular velocity
    return out


def mirror_labels(labels: np.ndarray) -> np.ndarray:
    """Mirror a (linear, angular) command: forward is unchanged, turn flips."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape[-1] != 2:
        raise ValueError(f"Expected 2 outputs, got {labels.shape[-1]}")
    out = labels.copy()
    out[..., 1] *= -1.0
    return out
