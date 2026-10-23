"""Someone talking over the robot (C52 barge-in): predict the loudness of the robot's own voice in the ears from what
it sent to the speaker (…-played.wav), frame by frame, and see how far the ears go above that prediction when a person
speaks too. Energy, not the waveform: it survives the two clocks' drift and the room's tail that keep a waveform echo
canceller at ~2 dB (aec_try.py).

  python tools/doubletalk.py recordings/ab/barge1 recordings/ab/barge2 ...
"""
import sys
import wave

import numpy as np

RATE, FRAME = 16000, 320
TAPS = 25                     # 0.5 s of the room's tail in the envelope model
DELAY = 8                     # frames (160 ms) from the speaker to the ears


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)


def band_energy(x):
    n = len(x) // FRAME
    frames = x[:n * FRAME].reshape(n, FRAME) * np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(frames, 512, axis=1)) ** 2
    f = np.fft.rfftfreq(512, 1 / RATE)
    return spec[:, (f >= 300) & (f <= 3400)].sum(axis=1) + 1.


def design(played_e, n):
    """Rows: the played energy DELAY..DELAY+TAPS frames earlier (log-free linear envelope model)."""
    cols = [np.concatenate([np.zeros(DELAY + k), played_e])[:n] for k in range(TAPS)]
    return np.stack(cols, 1)


LAGS = range(4, 26)           # 80–500 ms: each sentence leaves the queue with its own delay
TAIL = .75                    # the room's tail per 20 ms frame in the envelope (~0.15 s to fall by 1/e... ~1.4 dB a frame)


def tracked(pe, ae, window=50):
    """Predicted echo energy with the delay found anew every frame: the lag whose log-envelope of what we sent best
    matches the ears over the last second (the queue's time is not the speaker's: a new sentence starts 0.1–0.4 s
    off), the room's tail as one decaying echo, the gain fitted on the go from frames where only we spoke."""
    n = len(pe)
    tail = np.zeros(n)
    acc = 0.
    for t in range(n):
        acc = max(pe[t], acc * TAIL)
        tail[t] = acc
    lp, la = np.log10(tail + 1.), np.log10(ae)
    pred = np.full(n, np.nan)
    lag, gain = 8, None
    for t in range(n):
        if t >= window and t % 5 == 0 and tail[t - window:t].max() > 1e5:
            best = max(LAGS, key=lambda d: np.corrcoef(lp[t - window - d:t - d], la[t - window:t])[0, 1]
                       if t - window - d >= 0 else -1)
            lag = best
        src = tail[t - lag] if t - lag >= 0 else 0.
        if gain is None and src > 1e6:
            gain = ae[t] / src
        if gain is not None:
            pred[t] = gain * src
            if src > 1e6:
                ratio = ae[t] / src
                gain = np.exp(.98 * np.log(gain) + .02 * np.log(min(ratio, gain * 4)))   # follows the echo, not a voice
    return pred


def main2(stems):
    for stem in stems:
        p, a = load(stem + '-played.wav'), load(stem + '-array.wav')
        n = min(len(p), len(a)) // FRAME
        pe, ae = band_energy(p)[:n], band_energy(a)[:n]
        noise = np.percentile(ae, 10)
        pred = tracked(pe, ae)
        pred = np.where(np.isnan(pred), 0., pred) + noise
        excess = 10 * np.log10(ae / pred)
        smooth = np.convolve(excess, np.ones(10) / 10, 'same')
        robot = np.array([pe[max(0, t - 25):t].max() > 1e5 if t else False for t in range(n)])
        line = ''.join('#' if v > 9 else '+' if v > 6 else '.' if r else ' ' for v, r in zip(smooth, robot))
        print(f'\n{stem}: while the robot speaks — excess p50 {np.percentile(smooth[robot], 50):.1f} dB, '
              f'p95 {np.percentile(smooth[robot], 95):.1f}, p99 {np.percentile(smooth[robot], 99):.1f}')
        for i in range(0, len(line), 100):
            print(f'{i * .02:5.0f} s |{line[i:i + 100]}|')


def main(stems):
    data = {}
    for stem in stems:
        p, a = load(stem + '-played.wav'), load(stem + '-array.wav')
        n = min(len(p), len(a)) // FRAME
        pe, ae = band_energy(p)[:n], band_energy(a)[:n]
        data[stem] = (pe, ae)
    # fit on frames where the robot speaks, from the recordings with nobody else (the last ones given: barge3/4)
    train = [s for s in stems if s.endswith(('barge3', 'barge4'))] or stems
    X = np.concatenate([design(data[s][0], len(data[s][0])) for s in train])
    y = np.concatenate([data[s][1] for s in train])
    noise = np.median(np.concatenate([data[s][1] for s in train]))
    speaking = X.sum(axis=1) > 1e3 * TAPS
    h, *_ = np.linalg.lstsq(X[speaking], y[speaking] - noise, rcond=None)
    h = np.maximum(h, 0)
    print('echo envelope gain over 0.5 s:', np.round(h[:8] / h.max(), 2), '…')
    for stem in stems:
        pe, ae = data[stem]
        pred = design(pe, len(pe)) @ h + noise
        excess = 10 * np.log10(ae / pred)
        smooth = np.convolve(excess, np.ones(10) / 10, 'same')       # 0.2 s
        robot = design(pe, len(pe)).sum(axis=1) > 1e3 * TAPS
        line = ''.join('#' if v > 9 else '+' if v > 6 else '.' if r else ' ' for v, r in zip(smooth, robot))
        print(f'\n{stem}: while the robot speaks — excess p50 {np.percentile(smooth[robot], 50):.1f} dB, '
              f'p95 {np.percentile(smooth[robot], 95):.1f}, p99 {np.percentile(smooth[robot], 99):.1f}, '
              f'max {smooth[robot].max():.1f}')
        for i in range(0, len(line), 100):                          # 2 s a row: # >9 dB, + >6 dB over the echo
            print(f'{i * .02:5.0f} s |{line[i:i + 100]}|')


if __name__ == '__main__':
    main(sys.argv[1:])
