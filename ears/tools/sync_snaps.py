"""Синхронность двух линий: М1 (I2S0, D33) и М3 (I2S1, D32) лежат дырочками вплотную, владелец щёлкает пальцами
весь тест. Скрипт много раз перезапускает I2S и пишет по 0,75 с; в каждой записи со щелчком считает задержку М1→М3.
Микрофоны рядом: правильная задержка ≈ 0 (меньше отсчёта). Постоянный сдвиг в целые отсчёты — не беда (вычтем),
а разный от перезапуска к перезапуску — беда.   python tools/sync_snaps.py [записей]"""
import sys
import time

import numpy as np
import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from board import open_board  # noqa: E402
from serial.tools import list_ports

dev = next((p.device for p in list_ports.comports() if 'CP210' in (p.description or '')), None)
if not dev:
    sys.exit('плата не видна по USB')
n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 30
RESTART = '--no-restart' not in sys.argv     # без перезапусков: держится ли сдвиг, пока приём идёт
p = open_board(dev)
time.sleep(.3)
p.write(b's')
time.sleep(.3)


def capture():
    p.reset_input_buffer()
    p.write(b'c')
    end = time.time() + 6
    while time.time() < end:
        line = p.readline().decode('utf-8', 'replace').strip()
        if line.startswith('CAP '):
            frames, channels, rate = map(int, line.split()[1:4])
            p.timeout = 4
            raw = p.read(frames * channels * 2)
            p.timeout = .3
            if len(raw) == frames * channels * 2:
                return np.frombuffer(raw, '<i2').reshape(frames, channels).astype(float), rate
    return None, None


def lag(a, b, max_lag=24):
    """Задержка b относительно a (отсчёты, дробная): GCC-PHAT с параболой у пика."""
    n2 = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spec = np.fft.rfft(b, n2) * np.conj(np.fft.rfft(a, n2))
    spec /= np.abs(spec) + 1e-12
    cc = np.fft.irfft(spec, n2)
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
    k = int(np.argmax(cc))
    shift = 0.
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        shift = .5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return k - max_lag + shift, float(cc[k] / (np.abs(cc).mean() + 1e-12))


found = []
for k in range(n):
    if RESTART:
        p.write(b'r')                               # каждый раз заново: выравнивание должно повторяться
        time.sleep(.4)
    x, rate = capture()
    if x is None:
        print(f'{k + 1:2d}: нет записи', flush=True)
        continue
    # Живой канал на каждой линии, в каком бы слоте ни был микрофон (L/R мог не попасть на землю).
    line1 = [c for c in (0, 1) if x[:, c].any()]
    line2 = [c for c in (2, 3) if x[:, c].any()]
    if not line1 or not line2:
        print(f'{k + 1:2d}: молчит линия {"D33 (М1/М2)" if not line1 else "D32 (М3/М4)"}', flush=True)
        continue
    x = x - x.mean(axis=0)
    a, b = x[:, line1[0]], x[:, line2[0]]
    energy = np.convolve(a ** 2 + b ** 2, np.ones(int(rate * .005)), 'same')
    centre = int(np.argmax(energy))
    loud = energy[centre] / (np.median(energy) + 1e-9)
    if loud < 30:
        print(f'{k + 1:2d}: щелчка не было', flush=True)
        continue
    half = int(rate * .06)
    seg = slice(max(0, centre - half), centre + half)
    d, sharp = lag(a[seg], b[seg])
    found.append(d)
    print(f'{k + 1:2d}: щелчок, М3 отстаёт от М1 на {d:+6.2f} отсч. ({d / rate * 1e6:+6.0f} мкс), чёткость {sharp:4.1f}',
          flush=True)
p.write(b'l')
if found:
    f = np.array(found)
    print(f'\nщелчков поймано: {len(f)}; задержки: {np.round(f, 2).tolist()}')
    print(f'среднее {f.mean():+.2f} отсч., разброс {np.ptp(f):.2f} отсч.')
    # Итог по большинству: один слабый щелчок с эхом даёт выброс, он не должен решать (15.10: -1,70 при чёткости 7,8
    # среди четырёх -1,1 — прежняя проверка по размаху и округлению сказала «не держится»).
    med = float(np.median(f))
    near = np.abs(f - med) <= .35
    print(f'медиана {med:+.2f} отсч.; рядом с ней (±0,35) {int(near.sum())} из {len(f)}; '
          f'разброс у них {np.ptp(f[near]):.2f} отсч.')
    where = 'после каждого перезапуска' if RESTART else 'пока приём идёт без перезапусков'
    if near.mean() >= .75:
        print(f'ИТОГ: сдвиг между линиями постоянный {where}: {med:+.2f} отсч. ({med / 16:+.3f} мс) — вычитается')
    else:
        print(f'ИТОГ: сдвиг меняется {where} — так не вычесть')
else:
    print('\nни одного щелчка не поймано')
