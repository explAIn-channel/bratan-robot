"""Щелчки с известной стороны (по умолчанию — спереди, в метре): совпадают ли линии по времени и что говорит
направление. Тем же модулем, что у робота (три микрофона одним пакетом, subm).
    python tools/snaps_front.py [секунд] [--side 0]    (сторона в градусах: 0 — спереди, 90 — слева, −90 — справа)

Ждём для щелчка спереди (модель корпуса: М1 спереди x=182, М2/М3 на боках x=−84): М1 раньше М2 и М3 на ~12 отсчётов,
М2 и М3 вместе (±2). Сдвиг линий (М1 на одном I2S, М2/М3 на другом) виден как общая добавка к задержкам М1→М2 и М1→М3.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'bratan'))
from bratan.voice import array  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else 20.
side = float(sys.argv[sys.argv.index('--side') + 1]) if '--side' in sys.argv else 0.
ip = (ROOT / '.ushi-ip').read_text().strip()
link = array.ArrayLink(ip, voice=None).start()
print(f'пишу {seconds:.0f} с — щёлкай', flush=True)
time.sleep(seconds + .5)
got = link.ring.latest(seconds)
link.stop()
if got is None:
    sys.exit('звук не пришёл')
x, valid, _ = got
print(f'дошло {valid.mean() * 100:.1f} % звука', flush=True)
FS = array.RATE
y = x - x.mean(axis=0)
F = np.fft.rfft(y, axis=0)
f = np.fft.rfftfreq(len(y), 1 / FS)
F[(f < 1000) | (f > 6000)] = 0
h = np.fft.irfft(F, len(y), axis=0)
win = 80
nw = len(y) // win
env = np.abs(h[:nw * win]).reshape(nw, win, 3).max(axis=1)
db = 20 * np.log10(env + 1)
rise = db - np.median(db, axis=0)
loud = rise.max(axis=1)
events, i = [], 0
while i < nw:
    if loud[i] > 18:
        j = min(nw, i + 40)
        events.append(i + int(np.argmax(loud[i:j])) * 1)
        i = j
    else:
        i += 1


def gcc(a, b, maxlag=40, interp=8):
    n = 2048
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    fr = np.fft.rfftfreq(n, 1 / FS)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    R[(fr < 800) | (fr > 6000)] = 0
    r = np.fft.irfft(R, n * interp)
    r = np.concatenate([r[-maxlag * interp:], r[:maxlag * interp + 1]])
    k = int(np.argmax(r))
    return (k - maxlag * interp) / interp, float(r[k] / (np.abs(r).mean() + 1e-12))


rows = []
for k in events:
    c0 = k * win
    if not valid[max(0, c0 - 400):c0 + 800].all():
        continue
    seg = h[max(0, c0 - 160):c0 + 480]
    e = np.abs(seg)
    onset = []
    for ch in range(3):
        thr = max(np.median(e[:120, ch]) * 6, e[:, ch].max() * .3)
        idx = np.where(e[:, ch] > thr)[0]
        onset.append(int(idx[0]) if len(idx) else -1)
    t12, s12 = gcc(seg[:, 0], seg[:, 1])
    t13, s13 = gcc(seg[:, 0], seg[:, 2])
    t23, s23 = gcc(seg[:, 1], seg[:, 2])
    lv = rise[k]
    rows.append((t12, t13, t23))
    on = np.array(onset) - min(onset)
    print(f'{c0 / FS:5.1f} с: начало М1 {on[0]:4d} М2 {on[1]:4d} М3 {on[2]:4d} | задержки М1→М2 {t12:+6.2f} '
          f'М1→М3 {t13:+6.2f} М2→М3 {t23:+6.2f} | над фоном М1 {lv[0]:4.1f} М2 {lv[1]:4.1f} М3 {lv[2]:4.1f}', flush=True)
direction = array.Direction(link.ring)
if rows:
    r = np.array(rows)
    med = np.median(r, axis=0)
    print(f'\nмедианы: М1→М2 {med[0]:+.2f}, М1→М3 {med[1]:+.2f}, М2→М3 {med[2]:+.2f} отсчётов (щелчков {len(r)})')
    if abs(side) < 1:
        shift = (med[0] + med[1]) / 2 - 12.4
        print(f'ожидали для «спереди» М1→М2 ≈ М1→М3 ≈ +12,4: сдвиг линий ≈ {shift:+.1f} отсчётов '
              f'({"линии в такт" if abs(shift) < 4 else "ЛИНИИ РАЗЪЕХАЛИСЬ"})')
else:
    print('щелчков не поймано — громче или ближе')
