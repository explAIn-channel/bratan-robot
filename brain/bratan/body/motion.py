"""Something moving around the robot, seen by the lidar all round (the camera sees only ahead).

LOVOT and EMO turn to movement beside and behind them; Братан could not — his camera looks ahead and the
lidar was used only to drive. Standing still, every new scan is compared beam by beam with the median of
the last second: a run of adjacent beams that suddenly got much shorter (legs, Taffa coming close) or longer
(they walked off) is motion at that bearing. Only while standing still (no ego-motion), never on the dock.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .scan import LASER_YAW

CHANGE_M = .25           # a beam this much shorter than its recent median: something came into it
MIN_BEAMS = 2            # adjacent changed beams (a leg at 1.5 m spans ~3 one-degree beams)
MAX_RANGE = 4.
HISTORY = 6              # scans in the reference (~1 s at 6 Hz)
GAP_S = 4.               # one motion event per this many seconds


class MotionWatch:
    def __init__(self):
        self.history: deque = deque(maxlen=HISTORY)
        self.last_event = -1e9

    def reset(self):
        self.history.clear()

    def update(self, ranges, still, now):
        """`ranges`: 360 cleaned ranges (0 = no return). Returns (bearing_base, distance) or None."""
        r = np.asarray(ranges, dtype=float)
        if not still or len(r) != 360:
            self.history.clear()
            return None
        if len(self.history) < HISTORY:
            self.history.append(r)
            return None
        reference = np.median(np.stack(self.history), axis=0)
        self.history.append(r)
        if now - self.last_event < GAP_S:
            return None
        valid = (r > .2) & (r < MAX_RANGE)
        came = valid & ((reference <= .2) | (reference - r > CHANGE_M))      # new return in front of the old one
        best = None
        run = []
        for i in list(range(360)) + [0]:
            if came[i % 360]:
                run.append(i % 360)
                continue
            if len(run) >= MIN_BEAMS and (best is None or len(run) > len(best)):
                best = run
            run = []
        if best is None or len(best) > 40:        # a whole wall changing is a bump of the lidar, not a walker
            return None
        index = best[len(best) // 2]
        distance = float(np.median(r[best]))
        bearing = math.atan2(math.sin(math.radians(index) + LASER_YAW), math.cos(math.radians(index) + LASER_YAW))
        self.last_event = now
        return bearing, distance


UNEXPLAINED_M = .2       # this far from everything on the map: not the house (a walker, a dog, a bag)
NEAR_M = 3.5             # only what is this close (the localization holds well there, and it matters)
CLUSTER_GAP_M = .15      # neighbouring returns this close are one blob
CLUSTER_MIN = 3
CLUSTER_MAX_M = .8       # a person's legs or a dog; larger is a wall the map lacks, not a walker
ASSOCIATE_M = .6         # a blob this close to one of the last scan is the same blob
WALKED_M = .35           # ...and when it has gone this far within TRACK_S, it is moving
TRACK_S = 2.
LOST_S = .6


class MovingWatch:
    """Something moving near the robot while it drives (MotionWatch above is blind then: it compares raw scans).

    Each localized scan in map coordinates: the returns the map does not explain — further than UNEXPLAINED_M from
    every wall and piece of furniture it knows — are grouped into small blobs and followed from scan to scan. A blob
    that walks WALKED_M within TRACK_S is something moving; one that stays put is clutter and says nothing. The
    localization takes the robot's own motion out, like the efference copy that keeps our own head turn from looking
    like the world moving (Kismet switched its motion detector off for 600 ms after every eye movement instead)."""

    def __init__(self, grid):
        self.grid = grid
        self.tracks = []         # {'xy', 't', 'start', 'start_t', 'hits', 'fired'}
        self.last_event = -1e9

    def reset(self):
        self.tracks = []

    def blobs(self, pose, points_map):
        points_map = np.asarray(points_map, dtype=float).reshape(-1, 2)
        if not len(points_map):
            return []
        value, _, _ = self.grid.sample(points_map)
        near = np.hypot(points_map[:, 0] - pose[0], points_map[:, 1] - pose[1]) < NEAR_M
        runs, run = [], []
        for i in np.flatnonzero((value > UNEXPLAINED_M) & near):     # in beam order
            if run and math.dist(points_map[i], points_map[run[-1]]) > CLUSTER_GAP_M:
                runs.append(run)
                run = []
            run.append(i)
        if run:
            runs.append(run)
        out = []
        for run in runs:
            pts = points_map[run]
            if len(run) >= CLUSTER_MIN and np.ptp(pts[:, 0]) < CLUSTER_MAX_M and np.ptp(pts[:, 1]) < CLUSTER_MAX_M:
                out.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        return out

    def update(self, pose, points_map, now):
        """`pose`: the robot in the map; `points_map`: this scan's returns in the map. Returns (bearing_base,
        distance) of something that has just been seen moving, or None."""
        self.tracks = [t for t in self.tracks if now - t['t'] < LOST_S]
        moving = None
        for xy in self.blobs(pose, points_map):
            track = min(self.tracks, key=lambda t: math.dist(t['xy'], xy), default=None)
            if track is None or math.dist(track['xy'], xy) > ASSOCIATE_M or track['t'] == now:
                self.tracks.append({'xy': xy, 't': now, 'start': xy, 'start_t': now, 'hits': 1, 'fired': False})
                continue
            track.update(xy=xy, t=now, hits=track['hits'] + 1)
            if now - track['start_t'] > TRACK_S:
                track.update(start=xy, start_t=now)                     # slow drift is not walking: start over
            elif (not track['fired'] and track['hits'] >= 3 and math.dist(track['start'], xy) > WALKED_M
                  and now - self.last_event >= GAP_S):
                track['fired'] = True
                moving = xy
        if moving is None:
            return None
        self.last_event = now
        bearing = math.atan2(moving[1] - pose[1], moving[0] - pose[0]) - pose[2]
        return math.atan2(math.sin(bearing), math.cos(bearing)), math.dist(moving, pose[:2])
