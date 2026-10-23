"""Three microphones as one voice (C52, 18.10): delay-and-sum toward the speaker.

Each 32 ms window: the delays of М2 and М3 against the loudest microphone come from the cross-correlation of the
whole phrase (its syllable starts — voice/array.py's way), every channel is shifted by its delay in the frequency domain
(fractional samples) and they are added with weights by their own voice-to-noise ratio (maximal-ratio combining): the
uncorrelated part of the ears' noise adds up slower than the voice. The body shadows the far microphones, so a channel
much quieter than the best one adds noise, not voice — its weight goes down by its SNR.

Used to compare on A/B recordings (bratan.voice.ab --beam): is the sum heard better than the best single microphone?
"""
from __future__ import annotations

import numpy as np

RATE = 16000
N, HOP = 512, 256
MAX_LAG = 30


def _gcc(a, b, band):
    """Delay of b against a (samples, fractional to 1/8) over a whole stretch."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1. / RATE)
    cross = B * np.conj(A)
    cross = cross / (np.abs(cross) + 1e-9)
    cross[(f < band[0]) | (f > band[1])] = 0
    corr = np.fft.irfft(cross, n * 8)
    lim = MAX_LAG * 8
    window = np.concatenate([corr[-lim:], corr[:lim + 1]])
    return (int(np.argmax(window)) - lim) / 8.


def delays(x, band=(300., 3500.)):
    """Delays (samples) of each channel against channel `ref` (the loudest) over the stretch x [n × 3]."""
    level = x.astype(np.float64).std(axis=0)
    ref = int(np.argmax(level))
    d = np.zeros(x.shape[1])
    for ch in range(x.shape[1]):
        if ch != ref:
            d[ch] = _gcc(x[:, ref].astype(np.float64), x[:, ch].astype(np.float64), band)
    return d, ref


def delay_and_sum(x, d=None, noise=None, weights='snr'):
    """x [n × 3] int16 → one int16 track: channels shifted by their delays (to the reference) and summed.
    weights: 'snr' (by each channel's voice over its noise), 'equal'; noise — per-channel noise rms (else from the
    quietest 20 % of 32 ms windows)."""
    x = x.astype(np.float64)
    n = len(x)
    if d is None:
        d, _ = delays(x)
    if noise is None:
        frames = x[:n // N * N].reshape(-1, N, x.shape[1])
        rms = np.sqrt((frames ** 2).mean(axis=1))
        noise = np.percentile(rms, 20, axis=0) + 1e-6
    signal = np.sqrt(np.maximum(x.std(axis=0) ** 2 - noise ** 2, 1e-6))
    if weights == 'snr':
        w = signal / noise ** 2                    # maximal-ratio combining for independent noise
    else:
        w = np.ones(x.shape[1])
    w = w / w.sum()
    m = 1 << int(np.ceil(np.log2(n + 2 * MAX_LAG)))
    f = np.fft.rfftfreq(m, 1. / RATE)
    out = np.zeros(m // 2 + 1, complex)
    for ch in range(x.shape[1]):
        spec = np.fft.rfft(x[:, ch], m)
        out += w[ch] * spec * np.exp(2j * np.pi * f * d[ch] / RATE)     # advance a late channel by its delay
    y = np.fft.irfft(out, m)[:n]
    # keep the reference's loudness: the sum of partly correlated channels is quieter than any of them
    scale = x.std(axis=0).max() / (y.std() + 1e-9)
    return np.clip(y * scale, -32768, 32767).astype(np.int16)


class Beam:
    """The same sum live, window by window (the robot's voice from the three microphones, ears.voice_mix = 'sum').

    The delays are learned from the voice itself: each window's cross-spectrum against the reference microphone is
    averaged over the voiced windows (~0.5 s), whitened (PHAT) and its peak is the delay; a peak that does not stand out
    from the rest leaves that microphone out. Nothing is known yet, the speaker moved or the reference changed — the
    reference microphone alone, exactly the voice as before. The output lags the input by LAG samples (2 ms): a late
    microphone is advanced by reading it further on."""

    LAG = MAX_LAG + 2
    FFT = 1024
    BAND = (300., 3500.)
    SMOOTH = .85                 # cross-spectrum memory per voiced window (~0.2 s)
    VOICED_DB = 6.               # the reference this far over its noise: a voice to learn the delays from
    SHARP = 4.                   # the delay's peak over the mean of the correlation: trusted
    FAR_DB = 10.                 # a microphone this much worse than the reference by SNR only adds its noise

    def __init__(self, channels=3):
        self.channels = channels
        self.freq = np.fft.rfftfreq(self.FFT, 1. / RATE)
        self.band = (self.freq >= self.BAND[0]) & (self.freq <= self.BAND[1])
        self.reset()

    def reset(self):
        self.tail = np.zeros((2 * self.LAG, self.channels))
        self.cross = np.zeros((self.channels, len(self.freq)), complex)
        self.learned = 0
        self.ref = None
        self.d = np.zeros(self.channels)
        self.use = np.zeros(self.channels, bool)
        self.noise = self.voice = None
        self.last_mix = None

    def _learn(self, x, ref):
        spectra = np.fft.rfft(x, self.FFT, axis=0)
        cross = spectra * np.conj(spectra[:, ref])[:, None]
        self.cross = self.SMOOTH * self.cross + (1 - self.SMOOTH) * cross.T
        self.learned += 1
        for ch in range(self.channels):
            if ch == ref:
                continue
            c = self.cross[ch] / (np.abs(self.cross[ch]) + 1e-9)
            c[~self.band] = 0
            corr = np.fft.irfft(c, self.FFT * 8)
            lim = MAX_LAG * 8
            window = np.concatenate([corr[-lim:], corr[:lim + 1]])
            peak = int(np.argmax(window))
            sharp = window[peak] / (np.abs(window).mean() + 1e-12)
            self.d[ch] = (peak - lim) / 8.
            self.use[ch] = self.learned >= 4 and sharp >= self.SHARP

    def _mix(self, ref):
        """Per microphone: (weight, gain, delay) for the voice now. Each channel is brought to the reference's voice
        level (gain), then weighted by its own voice-to-noise ratio at that level (maximal-ratio combining); weights
        add up to 1, so the voice stays as loud as on the reference and only the noise goes down."""
        weights, gains = np.zeros(self.channels), np.ones(self.channels)
        weights[ref] = self.voice[ref] / self.noise[ref]
        for ch in range(self.channels):
            if ch == ref or not self.use[ch]:
                continue
            gains[ch] = np.sqrt(self.voice[ref] / self.voice[ch])
            snr = self.voice[ref] / (self.noise[ch] * gains[ch] ** 2)
            if 10 * np.log10(snr / weights[ref]) >= -self.FAR_DB:
                weights[ch] = snr
        return weights / weights.sum(), gains, np.where(np.arange(self.channels) == ref, 0., self.d)

    def _render(self, whole, t, mix):
        weights, gains, delays = mix
        out = np.zeros(len(t))
        for ch in np.flatnonzero(weights):
            at = t + delays[ch]                             # a late microphone is read further on
            i = np.floor(at).astype(int)
            u = at - i
            p0, p1, p2, p3 = (whole[np.clip(i + k, 0, len(whole) - 1), ch] for k in (-1, 0, 1, 2))
            # Catmull-Rom between p1 and p2: keeps the highs a linear blend of two samples would dull
            value = p1 + .5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3 + u * (3 * (p1 - p2) + p3 - p0)))
            out += weights[ch] * gains[ch] * value
        return out

    def feed(self, block, ref):
        """`block` [n × channels] int16, consecutive with the last one; `ref` — the microphone the voice is taken
        from now (ArrayLink._pick). Returns n int16 samples (LAG behind the input). The mix moves from the last
        window's to this one's across the window: a step between two windows would be a click."""
        x = block.astype(np.float64)
        energy = (x ** 2).mean(axis=0) + 1.
        if self.noise is None:
            self.noise, self.voice = energy.copy(), energy.copy()
        self.noise = np.where(energy < self.noise, energy, self.noise * 1.002)
        snr = 10 * np.log10(energy / self.noise)
        if ref != self.ref:
            self.cross[:] = 0
            self.learned, self.ref = 0, ref
            self.use[:] = False
        if snr[ref] >= self.VOICED_DB:
            self.voice = .9 * self.voice + .1 * np.maximum(energy - self.noise, 1.)
            self._learn(x, ref)
        whole = np.concatenate([self.tail, x])
        self.tail = whole[-2 * self.LAG:]
        n = len(x)
        t = np.arange(n) + self.LAG                     # output sample i is input sample i - LAG (in `whole`)
        new = self._mix(ref)
        old = getattr(self, 'last_mix', None) or new
        self.last_mix = new
        if np.count_nonzero(new[0]) == 1 and np.count_nonzero(old[0]) == 1 and old[0][ref] == 1.:
            return whole[t, ref].astype(np.int16)          # nothing to add: the reference itself, sample for sample
        fade = np.arange(n) / n
        out = (1 - fade) * self._render(whole, t, old) + fade * self._render(whole, t, new)
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
