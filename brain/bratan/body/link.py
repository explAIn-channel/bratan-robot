"""TCP link to the board's `body` program: sensor frames in, velocity intent out.

Board protocol (one text line each way):
  phone -> board: AUTH <key> | SET <seq> <v> <w> <hold> <heading> <lidar_pwm> | STOP <seq>
  board -> phone: {"type":"body",...} at 50 Hz, {"type":"scan",...} per lidar turn

Safety lives on both ends: the board stops the wheels if no SET arrives for its
command timeout; here a motion command older than COMMAND_LEASE_S is replaced by a
stop, and feedback older than FEEDBACK_TIMEOUT_S drops the connection.
"""
from __future__ import annotations

import json
import math
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

COMMAND_LEASE_S = .25
FEEDBACK_TIMEOUT_S = .75
SEND_PERIOD_S = .04
MAX_V = .24      # board linear limit
MAX_W = .6       # board angular limit (body-calibration angular_limit)


@dataclass(frozen=True)
class Command:
    v: float = 0.
    w: float = 0.
    hold: bool = False       # board-side heading hold (gyro) toward `heading`
    heading: float = 0.      # board odometry yaw
    lidar: int = 0           # lidar motor PWM 0..100

    @property
    def moving(self):
        return bool(self.v or self.w)


class BoardLink:
    def __init__(self, host, port, key):
        self.address, self.key = (host, port), key
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.fresh = threading.Event()          # set on every new body frame
        self.body: dict = {}
        self.scan: dict = {}
        self.scans = deque(maxlen=4)
        self.connected = False
        self.error = ''
        self.offset = None                      # phone_monotonic - board_clock
        self.rtt = None
        self.reconnects = 0
        self._command = Command()
        self._command_at = 0.
        self._serial = 0
        self.thread = threading.Thread(target=self._run, name='board-link', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def command(self, command: Command):
        with self.lock:
            if command != self._command:
                self._serial += 1
            self._command = command
            self._command_at = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.body, self.scan, self.connected

    def _current_command(self, now):
        with self.lock:
            command, at = self._command, self._command_at
        if command.moving and now - at > COMMAND_LEASE_S:
            command = Command(lidar=command.lidar)
        return command

    def _run(self):
        while not self.stopping.is_set():
            try:
                self._session()
            except (OSError, ValueError, KeyError) as error:
                self.error = f'{type(error).__name__}:{error}'[:120]
            with self.lock:
                self.connected = False
            self.reconnects += 1
            self.stopping.wait(.3)

    def _session(self):
        with socket.create_connection(self.address, timeout=2.) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(f'AUTH {self.key}\n'.encode())
            buffer = b''
            seq = int(time.time() * 1000) % 1_000_000_000   # monotonic across reconnects
            sent_at: dict[int, float] = {}
            samples = deque(maxlen=64)
            last_feedback = time.monotonic()
            last_sent = (None, 0.)
            next_send = 0.
            while not self.stopping.is_set():
                now = time.monotonic()
                if now >= next_send:
                    command = self._current_command(now)
                    # Resend while moving (board watchdog), else on change or 1 Hz.
                    if command != last_sent[0] or command.moving or now - last_sent[1] > 1.:
                        seq += 1
                        # The board rejects (and disconnects on) out-of-range values.
                        v = max(-MAX_V, min(MAX_V, command.v))
                        w = max(-MAX_W, min(MAX_W, command.w))
                        heading = math.atan2(math.sin(command.heading), math.cos(command.heading))
                        sock.sendall((f'SET {seq} {v:.5f} {w:.5f} {int(command.hold)} '
                                      f'{heading:.5f} {max(0, min(100, int(command.lidar)))}\n').encode())
                        sent_at[seq] = now
                        if len(sent_at) > 128:
                            sent_at.pop(next(iter(sent_at)))
                        last_sent = (command, now)
                    next_send = now + SEND_PERIOD_S
                ready, _, _ = select.select([sock], [], [], max(0., next_send - time.monotonic()))
                if not ready:
                    if time.monotonic() - last_feedback > FEEDBACK_TIMEOUT_S:
                        raise ConnectionError('feedback_timeout')
                    continue
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError('board_closed')
                buffer += chunk
                if len(buffer) > 1 << 18:
                    raise ValueError('frame_too_large')
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    frame = json.loads(line)
                    received = time.monotonic()
                    kind = frame.get('type')
                    if kind == 'body':
                        ack = frame.get('command_seq')
                        if ack in sent_at:
                            start = sent_at.pop(ack)
                            samples.append((received - start, (received + start) / 2 - frame['board_s']))
                        offset = min(samples)[1] if samples else received - frame['board_s']
                        frame['t'] = frame['sample_s'] + offset
                        frame['received'] = received
                        last_feedback = received
                        with self.lock:
                            self.offset = offset
                            self.rtt = min(samples)[0] if samples else None
                            self.body = frame
                            self.connected = True
                        self.fresh.set()
                    elif kind == 'scan':
                        offset = self.offset if self.offset is not None else received - frame['board_s']
                        frame['t'] = frame['board_s'] + offset
                        frame['received'] = received
                        if len(frame.get('ranges', ())) != 360:
                            continue
                        with self.lock:
                            self.scan = frame
                            self.scans.append(frame)

    def close(self):
        self.command(Command())
        time.sleep(.1)
        self.stopping.set()
        self.thread.join(2.)


def finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
