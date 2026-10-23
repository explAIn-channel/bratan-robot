"""10 с трёх микрофонов в тишине (подписка как у панели, без голоса) → recordings/quiet-<время>.npz и спектр шума:
уровни по полосам, узкие пики (гул и его гармоники), похожесть микрофонов между собой.

  python tools/quiet_capture.py [секунд]
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import csd, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'bratan'))
from bratan.voice import array  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.
link = array.ArrayLink('192.0.2.85', voice=None).start()
time.sleep(seconds + 1.)
got = link.ring.latest(seconds)
link.stop()
if got is None:
    sys.exit('нет звука с платы')
x, valid, _ = got
out = ROOT / 'recordings' / f'quiet-{time.strftime("%Y%m%d-%H%M%S")}.npz'
np.savez_compressed(out, x=x.astype(np.int16), valid=valid)
print('записано', out, f'целых {valid.mean():.0%}')
x = x[valid].astype(np.float64)
x -= x.mean(axis=0)
f, p = welch(x, 16000, nperseg=4096, axis=0)
print('rms дБFS16:', np.round(20 * np.log10(x.std(axis=0) / 32768), 1))
bands = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
for lo, hi in bands:
    m = (f >= lo) & (f < hi)
    db = 10 * np.log10(p[m].sum(axis=0) * (f[1] - f[0]) + 1e-9)
    coh = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        _, pab = csd(x[:, a], x[:, b], 16000, nperseg=4096)
        coh.append((np.abs(pab[m]) ** 2 / (p[m, a] * p[m, b])).mean())
    print(f'{lo:5d}-{hi:<5d} Гц  уровни {np.round(db, 1)}  похожесть {np.round(coh, 2)}')
avg = p.mean(axis=1)
smooth = np.convolve(avg, np.ones(21) / 21, mode='same')
ratio = 10 * np.log10(avg / (smooth + 1e-12))
top = sorted(np.argsort(ratio)[::-1][:16])
print('пики (Гц, дБ над соседями):', [(round(float(f[i]), 1), round(float(ratio[i]), 1)) for i in top])
