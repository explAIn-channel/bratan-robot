"""My own voice taken out of the ears by space (C52, 18.10 evening; the owner: «подожди, а сколько времени» over my
story was not heard at all).

The speaker sits in the body at one place: my voice reaches the three microphones with one fixed pattern of delays and
levels per frequency — its covariance R (3 × 3 per frequency) is learned from my own speech while I speak. The filter
per frequency is the direction that carries the least of that pattern (the smallest eigenvector of R, unit norm): my
voice is taken out as far as three microphones allow, a person anywhere else in the room is kept. Measured on the
owner's live test (work/ushi/tools/null_eval.py, test3): my voice −11…−12 dB, his +8…+12 dB over it; Microsoft heard
«подожди а сколько сейчас времени» through my story where the plain microphone gave only my own words — and over my
speech alone it heard nothing at all. All three microphones run on the board's one clock: no playback reference and no
clock drift (what kept a playback echo canceller on the ears at ~2 dB).

Learning only from what is mine: once the pattern is known, a frequency of a window is learned only if the filter takes
it out about as deeply as usual — a person talking over me (the very moment the filter is for) stays out of the
pattern; the «usual» follows the windows themselves, so a pattern grown stale (another room) is still relearned.

Streaming: windows of 512 with a 256 hop, square-root Hann in and out (they add up to one), the output 256 samples
behind the input. While I am silent the output is one microphone as it is (the same filter bank with that microphone's
weights: exact, and a switch to the null and back fades over one window). The learned pattern is kept in a file and
loaded at start: the first sentence after a restart is already clean.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

RATE = 16000
N, HOP = 512, 256
F = N // 2 + 1
LEARN = .004                  # per window of my speech: the pattern's memory is ~250 windows (~4 s of my voice)
VOICED = 1e8                  # a window this loud (mean power per bin and microphone; my voice ~60-70 dB on the ears at
                              # the speaker's .7, the room ~46) is learned from — my pauses are left out
READY = 200                   # windows learned (~3 s of my voice) before the pattern is used
FIT_DB = 6.                   # a frequency is mine if the filter leaves at most this much more of it than usual…
# …«usual»: the 30th percentile per frequency, followed slowly — up at most 0.03 dB a window (~2 dB a second of my
# speech): a person talking over me for a few seconds stays out, a pattern grown stale is relearned in seconds
FIT_SHARE, FIT_STEP_DB = .3, .1
# …but only while the filter is deep: grown shallow, it learns every frequency of my voice again (19.10, the owner's
# test: overnight my voice came to the microphones 1-2 samples otherwise, the filter took out 5 dB instead of 17 and
# this care kept it from relearning — Microsoft heard my own words through it)
# (by a slower measure of the depth: a person talking over me for a few seconds must not switch the care off). On the
# owner's recordings: his real interruption (test3) kept −0.8 → −0.8 dB; today's stale pattern (test4) back to −11.6 dB
# within one answer (with the care always on: −6.6)
GATE_FROM_DB, SLOW_STEP_DB = -10., .05
DEPTH_STEP_DB = .1            # how deep my voice is taken out (speech band, the median over windows), slowly followed
GOOD_DB = 6.                  # the null is used once it takes at least this much of my voice out…
KEEP_DB = 3.                  # …and kept down to this (a person talking over me raises the measure for a while)
RECOMPUTE_EVERY = 16          # the filters follow the pattern every this many windows learned (~0.3 s of my voice)
SAVE_S = 120.                 # and the pattern is written down this often while it changes
BAND = slice(int(300 / RATE * N), int(3400 / RATE * N) + 1)


class RobotNull:
    def __init__(self, path=None, channels=3, seed=None):
        """`path` — the learned pattern, kept; `seed` — the pattern to start from while there is none yet (learned on
        the owner's tests in the release's assets, work/ushi/tools/null_stream_eval.py --save)."""
        self.channels = channels
        self.path = Path(path) if path else None
        self.seed = Path(seed) if seed else None
        self.window = np.sqrt(np.hanning(N + 1)[:N])           # periodic: √hann² at 50 % overlap sums to one
        self.R = np.repeat(np.eye(channels, dtype=complex)[None] * 1e3, F, axis=0)
        self.learned = 0                                       # windows of my voice learned from
        self.fit = np.full(F, -10.)                            # dB of a window of mine the filter leaves, per frequency
        self.depth = None                                      # dB of my voice the filter leaves (speech band)
        self.trusted = False                                   # deep enough to be used (with hysteresis)
        self.W = np.zeros((F, channels), complex)
        self.W[:, 0] = 1.
        # One microphone as it is, through the same filter bank: weights 1 on it, 0 on the others.
        self.picks = np.repeat(np.eye(channels, dtype=complex)[:, None, :], F, axis=1)
        self.computed_learned = 0
        self.saved_at = time.monotonic()
        self.dirty = False
        self.cleaned = 0                                       # windows given out with my voice taken out
        self.reset()
        self.load()

    # ------------------------------------------------------------------ the pattern
    def load(self):
        source = next((p for p in (self.path, self.seed) if p is not None and p.exists()), None)
        if source is None:
            return
        try:
            with np.load(source) as data:
                if data['R'].shape != self.R.shape:
                    return
                self.R = data['R'].astype(complex)
                self.learned = int(data['learned'])
                if 'fit' in data and data['fit'].shape == self.fit.shape:
                    self.fit = data['fit'].astype(float)
                if 'depth' in data and np.isfinite(float(data['depth'])):
                    self.depth = self.slow = float(data['depth'])
                    self.trusted = self.depth <= -GOOD_DB
            self._compute()
        except Exception:
            pass                                               # a bad file: learn afresh

    def save(self):
        if self.path is None or not self.dirty:
            return
        try:
            tmp = self.path.with_name(self.path.stem + '.tmp.npz')
            np.savez(tmp, R=self.R, learned=self.learned, fit=self.fit,
                     depth=np.nan if self.depth is None else self.depth)
            tmp.replace(self.path)
            self.dirty = False
        except OSError:
            pass

    @property
    def ready(self):
        return self.learned >= READY

    @property
    def good(self):
        """Known and deep enough to listen through while I speak."""
        return bool(self.ready and self.trusted)

    def summary(self):
        return {'learned': int(self.learned), 'ready': bool(self.ready), 'good': self.good, 'cleaned': int(self.cleaned),
                'depth_db': None if self.depth is None else round(float(self.depth), 1)}

    def _compute(self):
        vals, vecs = np.linalg.eigh(self.R)                    # ascending eigenvalues per frequency
        w = vecs[:, :, 0]                                      # the direction with the least of my voice
        w = w * np.exp(-1j * np.angle(w[:, :1]))               # microphone 1's phase as the reference
        self.W = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9)
        self.computed_learned = self.learned

    def _learn(self, spec):
        power = spec.real ** 2 + spec.imag ** 2                # F × channels
        mean = power.mean(axis=1)
        if mean.mean() <= VOICED:
            return
        if self.ready:
            z = np.einsum('fc,fc->f', self.W.conj(), spec)
            left = z.real ** 2 + z.imag ** 2
            depth = float(10 * np.log10((left[BAND].sum() + 1.) / (mean[BAND].sum() + 1.)))
            self.depth = depth if self.depth is None else \
                self.depth + (DEPTH_STEP_DB if depth > self.depth else -DEPTH_STEP_DB)
            self.trusted = self.depth <= (-KEEP_DB if self.trusted else -GOOD_DB)
            slow = getattr(self, 'slow', None)
            self.slow = depth if slow is None else slow + (SLOW_STEP_DB if depth > slow else -SLOW_STEP_DB)
            left_db = 10 * np.log10((left + 1.) / (mean + 1.))
            mine = left_db < self.fit + FIT_DB
            self.fit += np.where(left_db < self.fit, -FIT_STEP_DB * (1 - FIT_SHARE), FIT_STEP_DB * FIT_SHARE)
            rate = LEARN * mine if self.slow <= GATE_FROM_DB else np.full(F, LEARN)
        else:
            rate = np.full(F, LEARN)
        outer = spec[:, :, None] * spec[:, None, :].conj()
        self.R += rate[:, None, None] * (outer - self.R)
        self.learned += 1
        self.dirty = True

    # ------------------------------------------------------------------ streaming
    def feed(self, first, block, learn=False, clean=False, pick=0):
        """The next samples of the microphones: `block` [n × channels] int16 from the board's frame `first` (any n; a
        gap or a jump starts the stream over). `learn` — I am speaking now; `clean` — my voice may be in the room: the
        output is my voice taken out (once the pattern is good), else microphone `pick` as it is. Returns (the board's
        frame of the first sample out or None, int16 samples) — HOP behind the input."""
        if self.in_end is not None and first != self.in_end:
            self.reset()
        x = np.concatenate([self.pending, np.asarray(block, np.float64).reshape(-1, self.channels)])
        self.in_end = first + len(block)
        start = self.in_end - len(x)                           # the board's frame of x[0]
        hops = len(x) // HOP
        self.pending = x[hops * HOP:]
        null = bool(clean) and self.good
        W = self.W if null else self.picks[min(max(int(pick), 0), self.channels - 1)]
        out, out_first = [], None
        for j in range(hops):
            self.inp = np.concatenate([self.inp[HOP:], x[j * HOP:(j + 1) * HOP]])
            spec = np.fft.rfft(self.inp * self.window[:, None], axis=0)          # F × channels
            if learn:
                self._learn(spec)
            y = np.fft.irfft(np.einsum('fc,fc->f', W.conj(), spec), N) * self.window
            self.out = np.concatenate([self.out[HOP:], np.zeros(HOP)]) + y
            if self.skip:
                self.skip -= 1                                 # the first hop out lies before the stream began
                continue
            if out_first is None:
                out_first = start + (j + 1) * HOP - N
            out.append(self.out[:HOP].copy())
            self.cleaned += null
        if self.learned - self.computed_learned >= RECOMPUTE_EVERY:
            self._compute()
        now = time.monotonic()
        if self.dirty and now - self.saved_at >= SAVE_S:
            self.saved_at = now
            self.save()
        if not out:
            return None, np.zeros(0, np.int16)
        return out_first, np.clip(np.rint(np.concatenate(out)), -32768, 32767).astype(np.int16)

    def reset(self):
        """A gap or a new stream: the windows start over (the learned pattern stays)."""
        self.inp = np.zeros((N, self.channels))                # the last N input samples
        self.out = np.zeros(N)                                 # overlap-add accumulator
        self.pending = np.zeros((0, self.channels))            # input not yet a whole hop
        self.in_end = None                                     # the board's frame after the last input
        self.skip = 1
