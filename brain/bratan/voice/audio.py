"""Microphone in, speaker out, through the face app.

Preferred: the media channel (TCP 127.0.0.1:8767, one connection, framed):
  client -> app : b'AUTH <token>\\n', then frames [type:u8][length:u32 BE][payload]
      1 = speaker PCM (16 kHz s16le mono) appended to the live stream
      2 = speaker stop (drop everything queued)
      3 = control JSON
  app -> client : 10 = microphone frame [boottime_ns:u64 BE][PCM 20 ms]
                  11 = status JSON (queued_ms, playing, level)
Fallback: HTTP polling of /audio/stream.pcm and /audio/live/* (older face app).
"""
from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
import uuid

import numpy as np

from ..face import client as face

RATE = 16000
FRAME = 640                       # 20 ms of s16le mono
MEDIA_PORT = 8767


def as_heard(frame):
    """The frame as the microphone gave it: the body's ears pass a cleaned frame with the raw one beside it (voice/
    main.py EarsFrame). Vosk and the loudness gates take the raw one — the thresholds were set on it, and on the
    cleaned ears «стой» was made out of other words (18.10 A/B: two false stops) — Microsoft takes the cleaned one."""
    return getattr(frame, 'raw', None) or frame


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.
    return float(np.sqrt(np.mean(samples * samples)))


class Audio:
    """Delivers (monotonic, 20 ms frame) tuples into `frames`; plays PCM."""

    def __init__(self):
        self.frames: queue.Queue = queue.Queue(maxsize=500)
        self.sock = None
        self.mode = None
        self.send_lock = threading.Lock()
        self.status = {'queued_ms': 0, 'playing': False}
        self.stopping = threading.Event()
        self.playback_id = None
        self.sequence = 0
        self.speaker_until = 0.          # monotonic end of what we have sent to play
        self.stopped_at = 0.             # monotonic: the last time playing was cut off
        # Whether the phone's microphone feeds `frames` now (17.10: the body's array may hear instead, voice/array.py);
        # None — always. Frames from either source go out in time order.
        self.gate = None
        self.tap = None                  # sees every phone frame, heard or not (the A/B recording, C51)
        self.played_tap = None           # (monotonic start, pcm) of everything we send to the speaker (C52)
        self.last_offered = 0.
        self.offer_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='audio', daemon=True)

    def start(self):
        self.thread.start()
        return self

    # ------------------------------------------------------------ input
    def _run(self):
        while not self.stopping.is_set():
            try:
                self._media()
            except OSError:
                pass
            if self.stopping.is_set():
                break
            try:
                self._poll()
            except Exception:
                self.stopping.wait(.5)

    def _offer(self, mono, frame):
        if self.tap is not None:
            self.tap(mono, frame)
        if self.gate is not None and not self.gate():
            return                                   # the array hears now
        self.offer(mono, frame)

    def offer(self, mono, frame):
        """Put one 20 ms frame (or a gap marker, None) to the listeners, whatever its source."""
        with self.offer_lock:
            mono = max(mono, self.last_offered + 1e-3)   # two sources: never back in time
            self.last_offered = mono
            try:
                self.frames.put_nowait((mono, frame))
            except queue.Full:
                while not self.frames.empty():
                    self.frames.get_nowait()
                self.frames.put_nowait((mono, None))      # a gap marker resets decoders

    def _media(self):
        sock = socket.create_connection(('127.0.0.1', MEDIA_PORT), timeout=1.)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(f'AUTH {face.token("microphone.read")} {face.token("speaker.write")}\n'.encode())
        sock.settimeout(2.)
        self.sock, self.mode = sock, 'media'
        buffer = b''
        offset = None
        try:
            while not self.stopping.is_set():
                chunk = sock.recv(16384)
                if not chunk:
                    raise ConnectionError('media_closed')
                buffer += chunk
                while len(buffer) >= 5:
                    kind = buffer[0]
                    length = struct.unpack('>I', buffer[1:5])[0]
                    if len(buffer) < 5 + length:
                        break
                    payload, buffer = buffer[5:5 + length], buffer[5 + length:]
                    if kind == 10 and length > 8:
                        boot_ns = struct.unpack('>Q', payload[:8])[0]
                        if offset is None:
                            offset = time.monotonic() - time.clock_gettime(getattr(time, 'CLOCK_BOOTTIME', 7))
                        pcm = payload[8:]
                        for i in range(0, len(pcm) - FRAME + 1, FRAME):
                            self._offer(boot_ns / 1e9 + offset + i / (2 * RATE), pcm[i:i + FRAME])
                    elif kind == 11:
                        self.status = json.loads(payload)
        finally:
            self.sock, self.mode = None, None
            sock.close()

    def _poll(self):
        self.mode = 'http'
        meta = face.get_json('/audio/cursor', 'microphone.read')
        cursor = int(meta['cursor'])
        pending = b''
        while not self.stopping.is_set():
            started = time.monotonic()
            with face.request(f'/audio/stream.pcm?since={cursor}&max_bytes=32000', 'microphone.read',
                              timeout=2.) as response:
                data = response.read()
                end = int(response.headers.get('X-RobotBrain-Cursor-End', cursor))
                dropped = response.headers.get('X-RobotBrain-Cursor-Dropped') == 'true'
            cursor = end
            if dropped:
                pending = b''
                self._offer(time.monotonic(), None)
            pending += data
            now = time.monotonic()
            count = len(pending) // FRAME
            for i in range(count):
                self._offer(now - (count - i) * .02, pending[i * FRAME:(i + 1) * FRAME])
            pending = pending[count * FRAME:]
            if self.stopping.wait(max(0., .06 - (time.monotonic() - started))):
                break

    # ------------------------------------------------------------ output
    def play(self, pcm: bytes):
        """Append 16 kHz PCM to the live speaker stream (non-blocking)."""
        if not pcm:
            return
        volume = getattr(self, 'volume', 1.)
        now = time.monotonic()
        if volume != 1.:
            # The speaker's own level on top of each voice's (18.10, the owner: «голос с динамика процентов на 30
            # тише»; quieter also leaves more of a person's voice under ours for the barge-in). Whole samples only: a
            # piece of odd length keeps its last byte for the next piece of the same stream (a second later — stale).
            carry = getattr(self, 'carry', b'') if now - getattr(self, 'carry_at', 0.) < 1. else b''
            pcm = carry + pcm
            whole = len(pcm) - len(pcm) % 2
            pcm, self.carry, self.carry_at = pcm[:whole], pcm[whole:], now
            if not pcm:
                return
            x = np.frombuffer(pcm, '<i2').astype(np.float32) * volume
            pcm = np.clip(x, -32768, 32767).astype('<i2').tobytes()
        start = max(self.speaker_until, now)
        self.speaker_until = start + len(pcm) / (2 * RATE)
        self.cut = False
        if self.played_tap is not None:
            # What we say and when (the queue's own time line): the echo canceller's reference (C52).
            self.played_tap(start, pcm)
        if self.mode == 'media' and self.sock is not None:
            with self.send_lock:
                try:
                    self.sock.sendall(b'\x01' + struct.pack('>I', len(pcm)) + pcm)
                    return
                except OSError:
                    pass
        self._http_play(pcm)

    def _http_play(self, pcm):
        if self.playback_id is None:
            self.playback_id = 'bratan-' + uuid.uuid4().hex
            self.sequence = 0
            face.post(f'/audio/live/start?playback_id={self.playback_id}', 'speaker.write')
        for i in range(0, len(pcm), 3200):
            face.post(f'/audio/live/append?playback_id={self.playback_id}&sequence={self.sequence}',
                      'speaker.write', pcm[i:i + 3200])
            self.sequence += 1

    def stop_playing(self):
        self.stopped_at = time.monotonic()   # cut off: the sound still dies away in the room (the ears' null, C52)
        self.speaker_until = 0.
        self.carry = b''
        self.cut = True                      # the app's «playing» is not believed until something new is played
        if self.mode == 'media' and self.sock is not None:
            with self.send_lock:
                try:
                    self.sock.sendall(b'\x02' + struct.pack('>I', 0))
                    return
                except OSError:
                    pass
        if self.playback_id:
            try:
                face.post(f'/audio/live/stop?playback_id={self.playback_id}', 'speaker.write')
            except Exception:
                pass
            self.playback_id = None

    def control(self, value: dict):
        if self.mode == 'media' and self.sock is not None:
            payload = json.dumps(value).encode()
            with self.send_lock:
                try:
                    self.sock.sendall(b'\x03' + struct.pack('>I', len(payload)) + payload)
                except OSError:
                    pass

    @property
    def speaking(self):
        # Cut off: the speaker is quiet in 0.1-0.2 s, but the app's «playing» comes late — until something new is
        # played it is not believed (19.10: «Подожди, а сколько сейчас времени?» cut me off, I went on «speaking» for
        # seconds and his question was gone from the frames kept before I listened).
        playing = bool(self.status.get('playing')) and not getattr(self, 'cut', False)
        return time.monotonic() < self.speaker_until + .3 or playing
