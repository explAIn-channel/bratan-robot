"""A gentle stationary-noise suppressor for the body's ears (C52, 18.10).

The ears hear the room's own hum 12-20 dB below a voice: tones near 94, 188, 281, 375, 469, 563, 750, 844 Hz up to
~5 kHz and a broadband floor — the same with and without the Wi-Fi streams (measured 18.10), not locked to the board's
clock (93.67 / 375.14 / 844.00 Hz), so it cannot simply be subtracted.

Classic short-time spectral suppression: a minimum-tracking noise estimate per frequency (down at once, up slowly — a
voice is short, the hum steady) and a Wiener gain on the decision-directed a-priori SNR (Ephraim & Malah 1984). The
gain never goes below FLOOR_DB: deep suppression leaves «musical noise» that hurts recognizers more than the hum.
Streamed: 20 ms in, 20 ms out, 16 ms later.
"""
from __future__ import annotations

import numpy as np

N = 512                         # analysis window (32 ms)
HOP = 256                       # half overlap: sqrt-Hann in and out adds back to the input
PRIME = 192                     # samples of silence out first: with 320-sample frames in and 256-sample hops out
                                # the output never runs dry (the delay is fixed: N - HOP + PRIME = 28 ms)
DELAY = N - HOP + PRIME
FLOOR_DB = -12.                 # never attenuate more than this
ALPHA_DD = .96                  # decision-directed smoothing of the a-priori SNR
SMOOTH = .8                     # power smoothing before the minimum tracking
WINDOW_HOPS = 125               # the noise is the lowest smoothed power of the last 2 s (minimum statistics, Martin
BIAS = 1.7                      # 2001) times the bias of a minimum: a voice pauses within 2 s, the hum does not


class Denoiser:
    def __init__(self, floor_db=FLOOR_DB):
        self.window = np.sqrt(np.hanning(N + 1)[:N]).astype(np.float32)
        self.floor = np.float32(10 ** (floor_db / 20.))
        self.reset()

    def reset(self):
        self.frame = np.zeros(N, np.float32)          # the analysis window: the last N samples
        self.pending = np.zeros(0, np.float32)        # samples in, not yet a hop
        self.overlap = np.zeros(N, np.float32)        # overlap-add of the synthesis
        self.ready = np.zeros(PRIME, np.float32)      # samples out, waiting
        self.noise = None
        self.smooth = None
        self.prev_clean = None
        self.history = None                           # smoothed power of the last WINDOW_HOPS hops
        self.hops = 0

    def process(self, frame: bytes) -> bytes:
        """int16 PCM in → as many int16 samples out (the first ones are silence while it starts)."""
        x = np.frombuffer(frame, '<i2').astype(np.float32)
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= HOP:
            hop, self.pending = self.pending[:HOP], self.pending[HOP:]
            self.frame = np.concatenate([self.frame[HOP:], hop])
            self._hop()
        need = len(x)
        if len(self.ready) < need:                    # frames of another size than 20 ms: pad, never stall
            self.ready = np.concatenate([self.ready, np.zeros(need - len(self.ready), np.float32)])
        out, self.ready = self.ready[:need], self.ready[need:]
        return np.clip(np.round(out), -32768, 32767).astype('<i2').tobytes()

    def _hop(self):
        spec = np.fft.rfft(self.frame * self.window)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        if self.smooth is None:
            self.smooth = power.copy()
            self.prev_clean = np.zeros_like(power)
            self.history = np.tile(power, (WINDOW_HOPS, 1))
        self.smooth = SMOOTH * self.smooth + (1 - SMOOTH) * power
        self.history[self.hops % WINDOW_HOPS] = self.smooth
        self.hops += 1
        self.noise = BIAS * self.history.min(axis=0) + 1.
        post = power / self.noise
        prio = ALPHA_DD * self.prev_clean / self.noise + (1 - ALPHA_DD) * np.maximum(post - 1., 0.)
        gain = np.maximum(prio / (1. + prio), self.floor)
        self.prev_clean = gain * gain * power
        y = np.fft.irfft(spec * gain, N).astype(np.float32) * self.window
        self.overlap += y
        self.ready = np.concatenate([self.ready, self.overlap[:HOP]])
        self.overlap = np.concatenate([self.overlap[HOP:], np.zeros(HOP, np.float32)])


def denoise_track(pcm: np.ndarray, floor_db=FLOOR_DB) -> np.ndarray:
    """A whole int16 track at once (for comparisons on recordings): same length, the 16 ms delay taken out."""
    d = Denoiser(floor_db)
    out = []
    for i in range(0, len(pcm) - 319, 320):
        out.append(np.frombuffer(d.process(pcm[i:i + 320].astype('<i2').tobytes()), '<i2'))
    y = np.concatenate(out + [np.zeros(len(pcm) - 320 * len(out), np.int16)])
    return np.concatenate([y[DELAY:], np.zeros(DELAY, np.int16)]).astype(np.int16)
