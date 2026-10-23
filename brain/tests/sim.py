"""Simulated board for offline tests: differential drive, slipping odometry,
ray-cast lidar on the real apartment map, charging contacts at the dock."""
from __future__ import annotations

import math
import random
import threading
import time

import numpy as np

from bratan.body.grid import Grid
from bratan.body.link import Command
from bratan.body.scan import LASER_YAW
from bratan.common.geometry import compose, relative, wrap


class SimLink:
    def __init__(self, map_yaml, pose, dock, *, slip=.02, rate=1., scan_hz=6., seed=1, extra_walls=(), bounce=0,
                 spin_accel=.8):
        self.grid = Grid(map_yaml)
        self.occupied = self.grid.structural.copy()
        for x0, y0, x1, y1 in extra_walls:          # furniture absent from the map
            r0, c0 = self.grid.cell((x0, y1))
            r1, c1 = self.grid.cell((x1, y0))
            self.occupied[max(0, r0):r1 + 1, max(0, c0):c1 + 1] = True
        self.true_pose = tuple(pose)
        self.odom = (0., 0., 0.)
        self.dock = tuple(dock)
        self.rng = random.Random(seed)
        self.slip, self.rate, self.scan_period = slip, rate, 1. / scan_hz
        self.v = self.w = 0.
        self.cmd = Command()
        self.cmd_at = 0.
        self.lock = threading.Lock()
        self.fresh = threading.Event()
        self.body, self.scan, self.connected = {}, {}, False
        self.rtt = .003
        self.offset = 0.
        self.seq = self.scan_seq = 0
        self.bumper_events = 0
        self.stopping = threading.Event()
        self.t0 = time.monotonic()
        self.charging = False
        self.bounce, self.touch_t = bounce, None    # springy contacts: touches that throw the robot back off them
        self.spin_accel = spin_accel                # rad/s² for a turn from standstill (10.10: the real one ~0.07)
        self.thread = threading.Thread(target=self._run, daemon=True)

    # BoardLink interface ---------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def command(self, command):
        with self.lock:
            self.cmd, self.cmd_at = command, time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.body, self.scan, self.connected

    def close(self):
        self.stopping.set()

    # simulation ---------------------------------------------------------------
    def kidnap(self, pose):
        with self.lock:
            self.true_pose = tuple(pose)

    def _collides(self, pose):
        from bratan.common.geometry import FOOTPRINT, transform
        outline = transform(pose, FOOTPRINT)
        for a, b in zip(outline, np.roll(outline, -1, axis=0)):
            for f in np.linspace(0, 1, 6):
                p = a + f * (b - a)
                r, c = self.grid.cell(p)
                if not self.grid.inside((r, c)) or self.occupied[r, c]:
                    return True
        return False

    def _raycast(self, pose):
        angles = pose[2] + LASER_YAW + np.radians(np.arange(360))
        d = np.arange(.2, 8., .02)
        xs = pose[0] + np.cos(angles)[:, None] * d[None]
        ys = pose[1] + np.sin(angles)[:, None] * d[None]
        rows, cols = self.grid.cells(np.column_stack((xs.ravel(), ys.ravel())))
        inside = (rows >= 0) & (rows < self.grid.height) & (cols >= 0) & (cols < self.grid.width)
        hit = ~inside
        hit[inside] = self.occupied[rows[inside], cols[inside]]
        hit = hit.reshape(360, len(d))
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        ranges = np.where(any_hit, d[first] + np.array([self.rng.gauss(0, .008) for _ in range(360)]), 0.)
        quality = np.where(any_hit, 120, 0)
        return ranges, quality

    def _run(self):
        dt = .02
        last_scan = 0.
        sim_t = 0.
        dock_x, dock_y = self.dock[0], self.dock[1]
        while not self.stopping.is_set():
            wall = time.monotonic()
            with self.lock:
                cmd, age = self.cmd, wall - self.cmd_at
            if age > .75:
                cmd = Command(lidar=cmd.lidar)
            tv, tw = cmd.v, cmd.w
            if cmd.hold and tv:
                tw += 1.8 * wrap(cmd.heading - self.odom[2])
            elif cmd.hold:
                error = wrap(cmd.heading - self.odom[2])
                tw = 0. if abs(error) < .015 else max(-abs(cmd.w), min(abs(cmd.w), 1.8 * error))
            tw = max(-.6, min(.6, tw))
            self.v += max(-.4 * dt, min(.4 * dt, tv - self.v))
            accel = self.spin_accel if abs(self.v) < .02 and abs(self.w) < .1 else .8   # slow to break away
            self.w += max(-accel * dt, min(accel * dt, tw - self.w))
            step = (self.v * dt * (1 - self.slip * self.rng.random()), 0., self.w * dt)
            candidate = compose(self.true_pose, step)
            bumper = 0
            if self._collides(candidate):
                bumper = 6
                self.v = self.w = 0.
                if not getattr(self, '_bump', False):
                    self.bumper_events += 1
                self._bump = True
            else:
                self._bump = False
                self.true_pose = candidate
            self.odom = compose(self.odom, (self.v * dt, 0., self.w * dt * (1 + .01 * self.rng.gauss(0, 1))))
            local = relative(self.dock, self.true_pose)
            contact = -.06 < local[0] < .01 and abs(local[1]) < .03 and abs(local[2]) < .08
            if contact and self.bounce:
                # 09.10 18:17: a moment of contact, then the robot stood 3 cm off the contacts
                self.touch_t = self.touch_t if self.touch_t is not None else sim_t
                if sim_t - self.touch_t > .15:
                    self.bounce -= 1
                    self.touch_t = None
                    self.true_pose = compose(self.true_pose, (.03, 0., 0.))
                    self.v = 0.
                    contact = False
            if contact and self.v < 0:
                self.v = 0.
            self.charging = contact
            sim_t += dt
            stamp = wall
            with self.lock:
                self.seq += 1
                self.body = {'type': 'body', 'boot': 'sim', 'seq': self.seq, 't': stamp, 'odom': list(self.odom),
                             'twist': [self.v, self.w], 'bumper': bumper, 'ir': 15, 'bumper_events': self.bumper_events,
                             'charging': contact, 'docked': contact, 'voltage': 16.0, 'lidar_pwm': cmd.lidar,
                             'inhibit_motion': False, 'start_button': 0, 'dock_button': 0, 'touch_events': 0}
                self.connected = True
            if cmd.lidar and sim_t - last_scan >= self.scan_period:
                last_scan = sim_t
                ranges, quality = self._raycast(self.true_pose)
                self.scan_seq += 1
                with self.lock:
                    self.scan = {'type': 'scan', 'boot': 'sim', 'seq': self.scan_seq, 't': stamp,
                                 'ranges': ranges.tolist(), 'qualities': quality.tolist()}
            self.fresh.set()
            spent = time.monotonic() - wall
            time.sleep(max(0., dt / self.rate - spent))
