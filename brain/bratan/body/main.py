"""Body process: the only owner of motion. 50 Hz loop, skills at 25 Hz.

Bus in:  body.do {skill, args, id, origin}   start a skill (replaces the current one)
         body.stop {}                         stop now
         eyes.seen {...}                      detections (for tracking)
         eyes.obstacles {discs}               camera obstacles in map frame
Bus out: body.state (10 Hz, retained)  body.event {kind, ...}
"""
from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..common import bus as busmod
from ..common.config import STATE, load, pin, read_json, secret, write_json
from ..common.geometry import compose, interpolate, inverse, relative, transform, wrap
from . import doors as doorsmod, localize, planner, scan as scanmod
from .motion import MotionWatch, MovingWatch
from .grid import Grid
from .guard import Guard
from .feel import Feel
from .link import BoardLink, Command
from .skills import SKILLS, STOP, Intent

LOOP_S = .02
SKILL_EVERY = 2                 # skills at 25 Hz
STATE_EVERY = 5                 # state at 10 Hz
BODY_MAX_AGE = .2
SCAN_MAX_AGE = 1.
LOST_AFTER = 4                  # consecutive failed scan matches
LOCATION_FILE = STATE / 'location.json'
DOORS_FILE = STATE / 'doors.json'
DOCK_REF_FILE = STATE / 'dock-reference.json'
DOCK_LEARN_JUMP_M, DOCK_LEARN_JUMP_RAD = .15, .15   # the dock refined while charging moves by less than this at once…
DOCK_SURE_MEAN, DOCK_SURE_WALLS = .09, .70           # …unless the map itself says so clearly (19.10 on the dock: .069, 78 %)
DOCK_VERIFY_S = 30.                                  # on the dock the lidar runs this long to check the dock on the map


class Body:
    def __init__(self, config, bus_address=busmod.DEFAULT_ADDRESS, link=None):
        self.config = config
        self.link = link or BoardLink(config['board_host'], config['board_port'], secret(config['board_key']))
        self.grid = Grid(config['map_yaml'])
        self.guard = Guard(config['front_ir_mask'], config['front_ir_active_low'])
        # Map walls are not added to the guard: 5 cm cells make narrow passages narrower than they are.
        # The lidar's own recent near points cover its 0.2 m blind zone instead.
        self.actor = None
        saved = read_json(LOCATION_FILE, {}) or {}
        self.dock = tuple(saved.get('dock') or config['dock_pose'])
        self.dock_ref = localize.DockReference.from_dict(read_json(DOCK_REF_FILE))
        self.rooms = read_json(config['rooms_json'], {}) or {}
        self.passages = (read_json(config.get('passages_json') or '', {}) or {}).get('passages', [])
        self.room_labels = None
        try:
            self.room_labels = np.load(config['rooms_labels'])
        except (OSError, ValueError):
            pass
        # Doors between rooms and whether they are open (from the static map, before any lidar evidence).
        self.motion = MotionWatch()
        self.feel = Feel()
        self.moving = None                 # MovingWatch, once the map is loaded (things moving while we drive)
        self.doors = doorsmod.Doors(self.grid, self.room_labels,
                                    doorsmod.find(self.grid, self.room_labels, self.rooms),
                                    saved=read_json(DOORS_FILE, []) or [])
        self.map_from_odom = (0., 0., 0.)
        self.saved_pose = saved.get('pose')
        self.saved_boot = saved.get('boot')
        self.localized = False
        self.fails = 0
        self.boot = None
        self.odom_history = deque(maxlen=400)
        self.scan_seq = None
        self.scan_points = np.zeros((0, 2))
        self.scan_ranges = np.zeros(360)
        self.scan_t = -1e9
        self.dock_fix = None
        self.dock_fix_odom = None
        self.dock_samples = deque(maxlen=10)
        self.dock_verified = False         # the dock checked on the map by the lidar since the robot last docked
        self.verify_until = time.monotonic() + DOCK_VERIFY_S
        self.was_charging = None
        self.skill = None
        self.last_reason = None
        self.detections = []
        self.detections_t = -1e9
        self.detections_seq = 0            # one per processed camera frame
        self.eyes_light = None             # mean brightness of that frame (0..255)
        self.viewpoints_failed = []
        self.workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix='body-work')
        self.recovery = None
        self.events = deque(maxlen=64)
        self.lock = threading.Lock()
        self.inbox = deque(maxlen=32)
        self.bus = busmod.Client('body', ('body.do', 'body.stop', 'eyes.seen', 'eyes.obstacles', 'mind.lidar', 'body.dump'),
                                 self.on_message, bus_address)
        self.ticks = 0
        self.loop_max = 0.
        self.lidar_wanted = False
        self.blocked_since = None
        self.dump_scans = 0
        self.stopping = threading.Event()

    # ----------------------------------------------------------------- bus
    def on_message(self, topic, data):
        if topic == 'body.dump':
            # Diagnostics only (navigation tests 07.10): the next N scans with odometry and the match.
            self.dump_scans = int((data or {}).get('scans', 10))
        elif topic == 'body.stop':
            with self.lock:
                self.inbox.appendleft(('stop', data or {}))
        elif topic == 'body.do':
            with self.lock:
                self.inbox.append(('do', data or {}))
        elif topic == 'eyes.seen':
            items = data.get('items', [])
            for item in items:
                item['t'] = data.get('mono', time.monotonic())
            self.detections, self.detections_t = items, time.monotonic()
            self.detections_seq += 1
            self.eyes_light = (data.get('stats') or {}).get('light')
        elif topic == 'eyes.obstacles':
            discs = [tuple(d) for d in data.get('discs', []) if len(d) == 3]
            self.guard.discs = discs
            self.grid.set_visual(discs)

    def emit(self, kind, **data):
        self.bus.publish('body.event', {'kind': kind, 't': time.time(), **data})

    # ----------------------------------------------------------------- run
    def run(self):
        self.link.start()
        self.bus.start()
        try:
            from .policy import Actor
            self.actor = Actor()
        except Exception as error:          # the actor is optional; pure pursuit drives
            print('actor_unavailable', type(error).__name__, error, flush=True)
        next_tick = time.monotonic()
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                self.tick(started)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.link.command(Command())
                if self.skill:
                    self.finish('failed', f'error:{type(error).__name__}')
            elapsed = time.monotonic() - started
            self.loop_max = max(self.loop_max * .999, elapsed)
            next_tick += LOOP_S
            delay = next_tick - time.monotonic()
            if delay < -.1:
                next_tick = time.monotonic()
            self.link.fresh.wait(max(0., min(delay, LOOP_S)))
            self.link.fresh.clear()

    def tick(self, now):
        body, scan, connected = self.link.snapshot()
        self.ticks += 1
        if not connected or not body:
            self.link.command(Command())
            if self.skill and self.ticks % 25 == 0 and now - self.skill.started > 3:
                self.finish('failed', 'board_disconnected')
            return
        if body.get('boot') != self.boot:
            self._new_boot(body)
        odom = tuple(body['odom'])
        if not self.odom_history or body['t'] > self.odom_history[-1][0]:
            self.odom_history.append((body['t'], odom))
        charging = self._contact(bool(body.get('charging') or body.get('docked')), now)
        self._charging_edge(charging, odom)
        if charging and not self.localized and now - self.scan_t > 3. and self.recovery is None:
            # On the charger with the lidar resting: the charge contact *is* the dock pose.
            self.map_from_odom = compose(self.dock, inverse(odom))
            self.localized = True
        if scan and scan.get('boot') == self.boot and scan.get('seq') != self.scan_seq:
            self._scan(scan, charging)
        pose = compose(self.map_from_odom, odom)
        self._inbox(now)
        intent = STOP
        if self.skill and self.ticks % SKILL_EVERY == 0:
            ctx = Ctx(self, body, odom, pose, now, charging)
            intent = self.skill.tick(ctx)
            self._last_intent = intent
            for index in self.skill.info.pop('cues', None) or ():
                self.emit('cue', id=self.skill.id, index=index)      # a step of a number starts: its words now
            if self.skill.status:
                self.finish(self.skill.status, self.skill.reason)
                intent = STOP
        elif self.skill:
            intent = getattr(self, '_last_intent', STOP)
        command = self._command(intent, body, odom, pose, now, charging)
        self.link.command(command)
        feeling = self.feel.update(body, command.moving, charging, now)
        if feeling:
            self.emit('handled', **feeling)                          # hands turned or pushed me
        if body.get('bumper_events', 0) > getattr(self, '_bumps', body.get('bumper_events', 0)):
            self.emit('bumper')
            if self.skill and intent.v > 0:
                self.finish('failed', 'bumper')
        self._bumps = body.get('bumper_events', 0)
        if self.ticks % STATE_EVERY == 0:
            self._publish(body, odom, pose, charging, now)
        if self.ticks % 100 == 0:
            self._save(pose)

    # ------------------------------------------------------------- pieces
    def _new_boot(self, body):
        first = self.boot is None
        self.boot = body['boot']
        self.odom_history.clear()
        self.scan_seq = None
        odom = tuple(body['odom'])
        if first and self.saved_boot == self.boot and self.saved_pose:
            self.map_from_odom = compose(tuple(self.saved_pose), inverse(odom))
        elif body.get('charging') or body.get('docked'):
            self.map_from_odom = compose(self.dock, inverse(odom))
        elif self.saved_pose:
            self.map_from_odom = compose(tuple(self.saved_pose), inverse(odom))
        self.localized = False
        self.fails = 0
        if self.skill:
            self.finish('failed', 'board_restarted')

    CONTACT_LOSS_S = .6

    def _contact(self, raw, now):
        """Charge contact with a short memory: one dropped sample (seen at a full battery,
        06.10 21:43 and 21:47) is not leaving the dock and must not start a reseat."""
        if raw:
            self.contact_lost_at = None
            return True
        if not self.was_charging:
            return False
        if getattr(self, 'contact_lost_at', None) is None:
            self.contact_lost_at = now
        return now - self.contact_lost_at < self.CONTACT_LOSS_S

    def _charging_edge(self, charging, odom):
        if charging and not self.was_charging:
            self.dock_verified = False                   # on the dock again: checked on the map once more
            self.verify_until = time.monotonic() + DOCK_VERIFY_S
            self.dock_samples.clear()
        if self.was_charging is False and charging:
            here = compose(self.map_from_odom, odom)
            if not self.localized or math.dist(here[:2], self.dock[:2]) > .5:
                # Charge contact is the dock: a carried robot restarts from there.
                self.map_from_odom = compose(self.dock, inverse(odom))
                self.localized = False
            self.emit('docked')
        elif self.was_charging and not charging:
            self.emit('undocked')
            here = compose(self.map_from_odom, odom)
            if self.skill is None and math.dist(here[:2], self.dock[:2]) < .3:
                # Nobody asked to leave: the contact slipped. Reseat at once.
                with self.lock:
                    self.inbox.append(('do', {'skill': 'dock', 'origin': 'system', 'id': 'reseat'}))
        self.was_charging = charging

    def _scan(self, scan, charging):
        self.scan_seq = scan['seq']
        scan_odom = interpolate(self.odom_history, scan['t'])
        if scan_odom is None:
            scan_odom = self.odom_history[-1][1] if self.odom_history else (0., 0., 0.)
        ranges = scanmod.clean_ranges(scan['ranges'], scan.get('qualities'))
        pts = scanmod.points(ranges)
        self.scan_points, self.scan_ranges, self.scan_t = pts, ranges, scan['t']
        self.guard.update_scan(pts, scan_odom, scan['t'])
        if self.actor is not None:
            self.actor.observe(scan['t'], ranges)
        twist = self.link.body.get('twist', (1., 1.))
        still = (not charging and (self.skill is None or self.skill.kind == 'wait')
                 and abs(twist[0]) < .01 and abs(twist[1]) < .02)
        moved = self.motion.update(ranges, still, scan['t'])
        if moved is not None:
            bearing, distance = moved
            self.emit('motion', bearing=round(bearing, 3), distance=round(distance, 2))
        if self.recovery is not None:
            self._dump(scan, scan_odom, None, None, charging)
            return
        predicted = compose(self.map_from_odom, scan_odom)
        result = localize.match(self.grid, predicted, pts)
        near_dock = math.dist(predicted[:2], self.dock[:2]) < 1.2
        fix = None
        # Just docked (or started on the dock): the map alone checks the dock first — the panorama would only confirm
        # the dock it is stored with (19.10: a dock learned 0.4 m off kept itself so).
        checking = charging and not self.dock_verified
        if near_dock and self.dock_ref.ready and not checking and (charging or (self.skill and self.skill.kind == 'dock')):
            fix = self.dock_ref.match(predicted, pts, scan['t'], scan_odom)
        if fix is not None:
            self.dock_fix = (*fix.pose, scan['t'])
            self.dock_fix_odom = scan_odom
            result = fix
        self._dump(scan, scan_odom, predicted, result, charging, dock_fix=fix is not None)
        if result.ok:
            self.map_from_odom = compose(result.pose, inverse(scan_odom))
            at_dock = self.skill is not None and self.skill.kind == 'dock' and near_dock
            if not still and not charging and not at_dock:      # the docking dance by the charger: not news
                # Driving: what moves beside us is told apart from the house by the map (09.10, the owner:
                # «на ходу соображать» — the still-only watch above is blind while moving).
                if self.moving is None:
                    self.moving = MovingWatch(self.grid)
                seen = self.moving.update(result.pose, transform(result.pose, pts), scan['t'])
                if seen is not None:
                    self.emit('motion', bearing=round(seen[0], 3), distance=round(seen[1], 2), moving=True)
            if not self.localized:
                self.emit('found', pose=list(result.pose))
            self.localized = True
            self.fails = 0
            if not charging and self.scan_seq % 2 == 0:
                self.grid.observe(result.pose, pts)
                self.doors.observe(result.pose, transform(result.pose, pts))
                self._door_news()
            if charging:
                self._dock_learning(result.pose, scan_odom, pts, map_sure=fix is None and result.mean < DOCK_SURE_MEAN
                                    and result.fraction >= DOCK_SURE_WALLS)
        else:
            self.fails += 1
            if self.fails >= LOST_AFTER and self.localized:
                self.localized = False
                self.emit('lost')
            if not self.localized and self.fails % 15 == LOST_AFTER and (not charging or not self.dock_verified):
                self.request_recovery()

    def _door_news(self):
        if not self.doors.changes:
            return
        for door, state in self.doors.changes:
            self.emit('door', rooms=list(door.rooms), state=state)
        self.doors.changes.clear()
        write_json(DOORS_FILE, self.doors.public())

    def _dump(self, scan, scan_odom, predicted, result, charging, dock_fix=False):
        if self.dump_scans <= 0:
            return
        self.dump_scans -= 1
        record = {'t': scan['t'], 'odom': [round(v, 4) for v in scan_odom], 'charging': charging,
                  'localized': self.localized, 'skill': self.skill.kind if self.skill else None,
                  'ranges': [round(float(v), 3) for v in scan['ranges']],
                  'q': [int(v) for v in (scan.get('qualities') or [])]}
        if predicted is not None:
            record['predicted'] = [round(v, 4) for v in predicted]
        if result is not None:
            record['match'] = {'ok': bool(result.ok), 'pose': [round(v, 4) for v in result.pose],
                               'mean': round(float(result.mean), 4), 'fraction': round(float(result.fraction), 3),
                               'dock_fix': dock_fix}
        self.bus.publish('body.scan_dump', record)

    def _dock_learning(self, pose, scan_odom, pts, map_sure=False):
        """While charging and still: refine the dock pose and record its panorama. `map_sure` — this pose is the map's
        own clear answer (not the dock panorama's): only such a pose may move the dock far."""
        body = self.link.body
        still = max(map(abs, body.get('twist', (1, 1)))) < .005
        if not still:
            self.dock_samples.clear()
            return
        self.dock_samples.append((*pose, bool(map_sure)))
        if len(self.dock_samples) < 10:
            return
        samples = np.asarray([s[:3] for s in self.dock_samples])
        sure = all(s[3] for s in self.dock_samples)
        xy = np.median(samples[:, :2], axis=0)
        yaw = math.atan2(np.sin(samples[:, 2]).mean(), np.cos(samples[:, 2]).mean())
        if (np.max(np.linalg.norm(samples[:, :2] - xy, axis=1)) < .03 and
                max(abs(wrap(p - yaw)) for p in samples[:, 2]) < .02):
            dock = (float(xy[0]), float(xy[1]), float(yaw))
            moved = math.dist(dock[:2], self.dock[:2]) > .01 or abs(wrap(dock[2] - self.dock[2])) > .01
            far = math.dist(dock[:2], self.dock[:2]) > DOCK_LEARN_JUMP_M or \
                abs(wrap(dock[2] - self.dock[2])) > DOCK_LEARN_JUMP_RAD
            if self.dock_ref.ready and far and not sure:
                # Far from the dock by a pose that is not the map's clear answer: the pose may be wrong, not the dock.
                self.dock_samples.clear()
                return
            # 19.10: the dock had been learned 0.4 m and 24° off the map (every undocking began «lost», a wrong place
            # was «found», he drove to a wall); on the dock the lidar laid on the map at once (78 % of the scan on the
            # walls) 0.4 m from it — the map is right: the dock and its panorama are learned again there.
            if moved or not self.dock_ref.ready:
                if far:
                    self.emit('dock_corrected', old=[round(v, 3) for v in self.dock], dock=[round(v, 3) for v in dock])
                self.dock = dock
                if self.dock_ref.record(dock, pts):
                    write_json(DOCK_REF_FILE, self.dock_ref.as_dict())
                self.emit('dock_learned', dock=list(dock))
            self.dock_samples.clear()
            self.dock_verified = True

    def request_recovery(self):
        if self.recovery is not None or not len(self.scan_points):
            return
        predicted = compose(self.map_from_odom, self.odom_history[-1][1]) if self.odom_history else self.dock
        pts, scan_odom = self.scan_points.copy(), self.guard.odom
        self.recovery = self.workers.submit(localize.recover, self.grid, predicted, pts)

        def done(future):
            try:
                result = future.result()
            except Exception:
                result = None
            if result is None:
                self.localized = False
            else:
                self.map_from_odom = compose(result.pose, inverse(scan_odom))
                self.localized = True
                self.fails = 0
                self.emit('found', pose=list(result.pose), recovered=True)
            self.recovery = None
        self.recovery.add_done_callback(done)

    def _inbox(self, now):
        with self.lock:
            items = list(self.inbox)
            self.inbox.clear()
        for kind, data in items:
            if kind == 'stop':
                if self.skill:
                    self.finish('cancelled', data.get('reason', 'stop'))
                continue
            name = data.get('skill')
            cls = SKILLS.get(name)
            if cls is None:
                self.emit('skill', id=data.get('id'), skill=name, status='failed', reason='unknown_skill')
                continue
            if self.skill:
                self.finish('cancelled', 'replaced')
            if self.link.body.get('inhibit_motion') and name not in ('wait',):
                self.emit('skill', id=data.get('id'), skill=name, status='failed', reason='motion_inhibited')
                continue
            skill = cls(dict(data.get('args') or {}), id=data.get('id') or uuid.uuid4().hex[:12],
                        origin=data.get('origin', 'mind'))
            skill.kind = name
            skill.started = now
            charging = bool(self.was_charging)
            if charging and name not in ('dock', 'wait', 'relocalize', 'undock') and skill.lidar_needed:
                # Leave the dock first; the requested skill follows.
                undock = SKILLS['undock']({'distance': self.config['undock_distance']}, id=skill.id + '-u',
                                          origin=skill.origin)
                undock.kind, undock.started = 'undock', now
                undock.info['then'] = skill
                skill = undock
            self.skill = skill
            self.emit('skill', id=skill.id, skill=skill.kind, status='started')

    def finish(self, status, reason=''):
        self.guard.live_only = False
        skill = self.skill
        if skill is None:
            return
        follow = skill.info.get('then') if status == 'done' else None
        self.skill = None
        self._last_intent = STOP
        if follow is not None:
            follow.started = time.monotonic()
            self.skill = follow
            return
        self.emit('skill', id=skill.id.removesuffix('-u'), skill=skill.kind, status=status, reason=reason,
                  origin=skill.origin)

    def _command(self, intent, body, odom, pose, now, charging):
        # On the dock the lidar rests — after the dock is checked on the map (DOCK_VERIFY_S at most after docking).
        verifying = charging and not self.dock_verified and now < self.verify_until
        if self.skill is not None:
            lidar = self.config['lidar_pwm'] if self.skill.lidar_needed or not charging or verifying else 0
        else:
            lidar = 0 if charging and not verifying else self.config['lidar_pwm']
        if not intent.v and not intent.w:
            # Not asking to move: the guard is not refusing anything. Without this reset the stale
            # «blocked for N s» cut every run-up stage at its first tick (23:10: backed 10 cm, charged 0).
            self.blocked_since = None
            return Command(lidar=lidar)
        if now - body['t'] > BODY_MAX_AGE:
            self.last_reason = 'body_stale'
            return Command(lidar=lidar)
        if now - self.scan_t > SCAN_MAX_AGE:
            self.last_reason = 'scan_stale'
            return Command(lidar=lidar)
        ignore = None
        if self.skill and self.skill.kind == 'dock' and self.skill.stage in ('approach', 'press', 'contact_wait', 'align'):
            bounds = (-.28, .06, .18)
            dock = self.dock

            def ignore(points_base, pose=pose):
                local = transform(relative(dock, pose), points_base)
                return (local[:, 0] >= bounds[0]) & (local[:, 0] <= bounds[1]) & (np.abs(local[:, 1]) <= bounds[2])
        self.guard.ignore = ignore
        reason = self.guard.check(body, odom, pose if self.localized else None, intent.v, intent.w)
        self.last_reason = reason
        if reason and not self.blocked_since:
            self.blocked_since = now
        elif not reason:
            self.blocked_since = None
        if reason:
            return Command(lidar=lidar)
        heading = intent.heading if intent.hold else 0.
        return Command(intent.v, intent.w, intent.hold, heading, lidar)

    def room_of(self, xy):
        if self.room_labels is None or not self.rooms.get('rooms'):
            return None
        row, col = self.grid.cell(xy)
        if not (0 <= row < self.room_labels.shape[0] and 0 <= col < self.room_labels.shape[1]):
            return None
        label = int(self.room_labels[row, col])
        for room in self.rooms['rooms']:
            if room.get('label') == label:
                return room.get('name')
        return None

    def _publish(self, body, odom, pose, charging, now):
        twist = body.get('twist', [0, 0])
        self.bus.publish('body.state', {
            't': time.time(), 'mono': now, 'pose': [round(v, 4) for v in pose], 'localized': self.localized,
            'room': self.room_of(pose) if self.localized else None,
            'guard_hit': getattr(self.guard, 'last_hit', None) if self.blocked_since else None,
            'docked': charging, 'charging': bool(body.get('charging')), 'voltage': body.get('voltage'),
            'twist': twist, 'moving': bool(max(abs(twist[0]), abs(twist[1])) > .01),
            'skill': self.skill.public() if self.skill else None, 'blocked': self.last_reason,
            'intent': [round(getattr(self, '_last_intent', STOP).v, 3), round(getattr(self, '_last_intent', STOP).w, 3)],
            'lidar': body.get('lidar_pwm'), 'inhibit': body.get('inhibit_motion'),
            'dock': [round(v, 4) for v in self.dock], 'loop_max_ms': round(self.loop_max * 1000, 1),
            'rtt_ms': round((self.link.rtt or 0) * 1000, 1), 'recovering': self.recovery is not None,
            'bumper': body.get('bumper'), 'buttons': [body.get('start_button'), body.get('dock_button')],
            'touch_events': body.get('touch_events'),
            'doors': self.doors.public(),
        })

    def _save(self, pose):
        if self.localized:
            write_json(LOCATION_FILE, {'pose': list(pose), 'dock': list(self.dock), 'boot': self.boot,
                                       't': time.time()})

    # ------------------------------------------------------- ctx helpers
    def plan(self, start, goal, tolerance=0.):
        clearance = self.grid.clearance.copy()
        return self.workers.submit(planner.plan, self.grid, start, goal, clearance, tolerance)

    def plan_ring(self, start, target, distance):
        clearance = self.grid.clearance.copy()
        return self.workers.submit(planner.ring_goal, self.grid, start, target, distance, clearance)

    def viewpoint(self, room=None):
        """A reachable, rarely visited floor point (in `room` if given)."""
        self.grid.decay_visits()
        clearance = self.grid.clearance
        mask = clearance >= .35
        if room and self.room_labels is not None:
            labels = [r['label'] for r in self.rooms.get('rooms', []) if r.get('name') == room]
            if labels and self.room_labels.shape == mask.shape:
                mask &= np.isin(self.room_labels, labels)
        cells = np.argwhere(mask)
        if not len(cells):
            return None
        pose = compose(self.map_from_odom, self.odom_history[-1][1])
        best = None
        for row, col in cells[::max(1, len(cells) // 300)]:
            xy = self.grid.xy((row, col))
            distance = math.dist(xy, pose[:2])
            if distance < .6 or any(math.dist(xy, f) < .5 for f in self.viewpoints_failed[-20:]):
                continue
            score = -int(self.grid.visited[row, col]) * .3 - abs(distance - 1.8) * .5 + float(clearance[row, col]) + random.random() * .3
            if best is None or score > best[0]:
                best = (score, xy)
        return best[1] if best else None


class Ctx:
    """What a skill may read and ask during one tick."""

    def __init__(self, owner: Body, body, odom, pose, now, charging):
        self.owner, self.body, self.odom, self.pose, self.now = owner, body, odom, pose, now
        self.charging = charging
        self.localized = owner.localized
        self.grid = owner.grid
        self.config = owner.config
        self.dock = owner.dock
        self.twist = tuple(body.get('twist', (0., 0.)))
        self.dt = LOOP_S * SKILL_EVERY
        self.scan_fresh = now - owner.scan_t < SCAN_MAX_AGE
        fix = owner.dock_fix
        self.dock_fix = fix if fix is not None and now - fix[3] < .75 else None
        self.dock_fix_odom = owner.dock_fix_odom
        self.recovering = owner.recovery is not None
        self.passages = owner.passages

    def plan(self, start, goal, tolerance=0.):
        return self.owner.plan(start, goal, tolerance)

    def plan_ring(self, start, target, distance):
        return self.owner.plan_ring(start, target, distance)

    def route_clear(self, route):
        return planner.route_clear(self.grid, route)

    def clear(self, v, w, seconds):
        return self.owner.guard.check(self.body, self.odom, self.pose if self.localized else None, v, w, seconds) is None

    def points(self):
        return self.owner.guard.nearby(self.odom, self.pose if self.localized else None)

    def live_only(self, value):
        self.owner.guard.live_only = bool(value)

    def submit(self, fn, *args):
        return self.owner.workers.submit(fn, *args)

    def near_obstacle(self, distance):
        pts = self.owner.guard.points
        if not len(pts):
            return False
        ahead = pts[(pts[:, 0] > 0) & (np.abs(pts[:, 1]) < .2)]
        return bool(len(ahead) and ahead[:, 0].min() < .23 + distance)

    def blocked_for(self):
        """Seconds the guard has been refusing motion continuously."""
        since = self.owner.blocked_since
        return self.now - since if since else 0.

    def mark_blocked_ahead(self):
        pass        # live lidar evidence already marks what stopped us

    def request_recovery(self):
        self.owner.request_recovery()

    def viewpoint(self, room=None):
        return self.owner.viewpoint(room)

    def forget_viewpoint(self, xy):
        self.owner.viewpoints_failed.append(tuple(xy))

    def sees(self, label, bearing, tolerance=.3):
        """The freshest frame: (frame number, is a `label` there within `tolerance` of `bearing` (rad, now),
        its light), or None without a fresh frame."""
        if self.now - self.owner.detections_t > .8:
            return None
        hit = False
        for item in self.owner.detections:
            if item.get('label') != label:
                continue
            capture_odom = interpolate(self.owner.odom_history, item.get('t', 0)) or self.odom
            if abs(wrap(item.get('bearing', 0.) + capture_odom[2] - self.odom[2] - bearing)) < tolerance:
                hit = True
                break
        return self.owner.detections_seq, hit, self.owner.eyes_light

    def find(self, who, track=None):
        """Freshest matching detection with bearing corrected for our turn since capture."""
        if self.now - self.owner.detections_t > .8:
            return None
        who = (who or 'person').casefold()
        wanted = {'собака': 'dog', 'собаку': 'dog', 'пёс': 'dog', 'пес': 'dog', 'dog': 'dog',
                  'человек': 'person', 'кто-то': 'person', 'person': 'person', '': 'person'}.get(who)
        best = unnamed = None
        for item in self.owner.detections:
            label, name = item.get('label'), (item.get('name') or '').casefold()
            # Locked on: the same track stays the target even when the face (and so the name) turns away.
            if track is not None and item.get('track') == track and label in ('person', 'dog'):
                best = item
                break
            if wanted:
                if label != wanted:
                    continue
            elif name != who and not (item.get('pet') and wanted is None and label == 'dog'):
                # 08.10 01:34 «подъеду к Сергею», at night: the face is rarely recognised in the dark or from the
                # side, so the person was in view without a name and the robot «did not see» him. A person
                # whose face is not recognised may be the one asked for — never one recognised as someone else.
                if label == 'person' and not name and (unnamed is None or
                                                       abs(item.get('bearing', 0)) < abs(unnamed.get('bearing', 0))):
                    unnamed = item
                continue
            if track is not None and item.get('track') == track:
                best = item
                break
            if best is None or abs(item.get('bearing', 0)) < abs(best.get('bearing', 0)):
                best = item
        best = best or unnamed
        if best is None:
            return None
        capture_odom = interpolate(self.owner.odom_history, best.get('t', 0)) or self.odom
        return {**best, 'bearing_now': wrap(best.get('bearing', 0.) + capture_odom[2] - self.odom[2])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    config = load(args.config)
    pin('body', config)
    try:
        os.nice(-10)
    except (OSError, AttributeError):
        pass
    body = Body(config)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: body.stopping.set())
    try:
        body.run()
    finally:
        body.link.close()


if __name__ == '__main__':
    main()
