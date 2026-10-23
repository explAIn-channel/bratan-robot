"""Is someone speaking: Silero VAD (v6, ONNX — the copy faster-whisper ships, MIT) over the 20 ms microphone frames.

18.10: the body's ears (voice/array.py) hear the room's own noise only 12-20 dB below a voice (the phone's microphone
filters its noise to almost nothing). Gates that compared loudness with that noise misbehaved: the name's gate was
open 21-79 % of the time in a quiet room, and the name check, whose floor was fixed below the ears' noise, never heard
the end of a phrase — «Братан, привет» was answered after the 8 s limit, merged with what came next. A speech detector
does not care how loud the room is (Silero: far better than WebRTC's VAD or a level threshold; < 1 ms per 32 ms).

`Silero.feed(frame)` → the probability of speech in the newest 32 ms; `Talk` turns probabilities into speech/silence
with hysteresis, counted in 20 ms frames as the gates before it did.
"""
from __future__ import annotations

import numpy as np

CHUNK = 512                  # Silero's window at 16 kHz: 32 ms
CONTEXT = 64                 # samples of the previous window it looks at
START = .5                   # speech from here
STOP = .35                   # silence below; between the two the state holds


class Silero:
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        """A gap in the sound or another microphone: start afresh (the detector remembers the last second or so)."""
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.prob = 0.

    def feed(self, frame: bytes) -> float:
        self.pending = np.concatenate([self.pending, np.frombuffer(frame, '<i2').astype(np.float32) / 32768.])
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            window = np.concatenate([self.context, chunk])[None, :]
            out, self.h, self.c = self.session.run(None, {'input': window, 'h': self.h, 'c': self.c})
            self.context = chunk[-CONTEXT:]
            self.prob = float(np.asarray(out).reshape(-1)[-1])
        return self.prob


class Talk:
    """Speech or silence frame by frame from the detector's probabilities (hysteresis: on at START, off below STOP)."""

    def __init__(self, speaking=False):
        self.on = speaking
        self.silence = 0             # frames of silence in a row
        self.voiced = 0              # frames of speech in all

    def feed(self, prob):
        if prob >= START:
            self.on = True
        elif prob < STOP:
            self.on = False
        if self.on:
            self.silence = 0
            self.voiced += 1
        else:
            self.silence += 1
        return self.on
