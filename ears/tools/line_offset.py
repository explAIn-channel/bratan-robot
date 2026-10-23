"""The delay between the two I2S lines (М1 on one, М2/М3 on the other) in an A/B recording: the robot's own voice
(played) cross-correlated between microphone pairs — whole samples, peak and its strength.
python tools/line_offset.py STEM [STEM ...]"""
import sys
import wave

import numpy as np

RATE = 16000
for stem in sys.argv[1:]:
    d = np.load(stem + '-multi.npz')
    x, t0 = d['x'].astype(np.float64), float(d['t0'])
    with wave.open(stem + '-played.wav') as w:
        played = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    mask = np.zeros(len(x), bool)
    for i in range(0, len(x) - 512, 512):
        a = int((t0 + i / RATE) * RATE)
        if 0 <= a < len(played) - 512 and 10 * np.log10((played[a:a + 512] ** 2).mean() + 1) > 50:
            mask[i:i + 512] = True
    y = x[mask]
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        A, B = np.fft.rfft(y[:, a], 2 * len(y)), np.fft.rfft(y[:, b], 2 * len(y))
        c = np.fft.irfft(A.conj() * B / (np.abs(A * B) + 1e-9), 2 * len(y))
        lags = np.r_[np.arange(0, 64), np.arange(-64, 0)]
        vals = np.r_[c[:64], c[-64:]]
        k = int(np.argmax(vals))
        out.append(f'М{a + 1}-М{b + 1}: {lags[k]:+d} ({vals[k]:.2f})')
    print(stem.split('\\')[-1].split('/')[-1], f'{mask.sum() / RATE:.0f} s of robot:', '  '.join(out))
