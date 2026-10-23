"""Сколько градусов ошибки даёт каждая расстановка микрофонов на корпусе R5 (в плоскости 150 мм от пола).

Корпус на этой высоте — прямоугольник x −155…+182, y −116…+116 (перёд +x, лево +y). Звук — плоская волна с
направления θ. Микрофон, который «видит» волну, слышит её напрямую; микрофон в тени — через огибание корпуса по
стенке от ближайшего освещённого угла (путь длиннее, сигнал слабее → шум задержки вдвое больше).
Оценка — перебор направлений по 1° по задержкам всех пар (как SRP по таблице). Шум задержки 20 мкс у освещённого,
40 мкс у затенённого, 5 % грубых промахов (эхо). Сдвиг между линиями I2S считаем вычтенным (калибровка при включении).
python tools/layout_sim.py
"""
import numpy as np

C = 343e3                      # мм/с
X0, X1, Y0, Y1 = -155., 182., -116., 116.
CORNERS = np.array([[X1, Y1], [X1, Y0], [X0, Y0], [X0, Y1]])
PERIM = 2 * ((X1 - X0) + (Y1 - Y0))


def s_of(p):
    """Положение точки на периметре (мм от угла перёд-лево по часовой при взгляде сверху)."""
    x, y = p
    if abs(x - X1) < 1e-6:
        return Y1 - y
    if abs(y - Y0) < 1e-6:
        return (Y1 - Y0) + (X1 - x)
    if abs(x - X0) < 1e-6:
        return (Y1 - Y0) + (X1 - X0) + (y - Y0)
    return 2 * (Y1 - Y0) + (X1 - X0) + (x - X0)


def normal(p):
    x, y = p
    if abs(x - X1) < 1e-6: return np.array([1., 0.])
    if abs(x - X0) < 1e-6: return np.array([-1., 0.])
    if abs(y - Y1) < 1e-6: return np.array([0., 1.])
    return np.array([0., -1.])


def arrival(p, u):
    """Время прихода плоской волны с направления u (единичный вектор «откуда») в точку p на стенке, и тень ли."""
    p = np.asarray(p, float)
    if normal(p) @ u > 0:
        return -(p @ u) / C, False
    best = np.inf
    sp = s_of(p)
    for q in CORNERS:
        n_ok = (q @ u)            # угол освещён, если он «выступает» к источнику; берём все углы, минимум и так верный
        along = abs(s_of(q) - sp)
        along = min(along, PERIM - along)
        best = min(best, (-(q @ u) + along) / C)
        _ = n_ok
    return best, True


LAYOUTS = {
    '4 мик (план): перёд-пр, пр.бок, зад, лев.бок': [(182, -80), (-88, Y0), (X0, 0), (-88, Y1)],
    '3 мик A: перёд-пр, пр.бок, зад (как сейчас)': [(182, -80), (-88, Y0), (X0, 0)],
    '3 мик B: перёд-пр, пр.бок, лев.бок': [(182, -80), (-88, Y0), (-88, Y1)],
    '3 мик C: бока спереди (x=150) + зад': [(150, Y0), (150, Y1), (X0, 0)],
    '3 мик D: перёд-лев, пр.бок сзади, зад-лев.бок': [(182, 80), (-88, Y0), (-130, Y1)],
}
for k, v in LAYOUTS.items():
    LAYOUTS[k] = [(x if abs(y) < 116 else x, y) for x, y in v]

GRID = np.radians(np.arange(360))
rng = np.random.default_rng(1)


def table(mics):
    t = np.zeros((len(GRID), len(mics)))
    sh = np.zeros_like(t, dtype=bool)
    for i, th in enumerate(GRID):
        u = np.array([np.cos(th), np.sin(th)])
        for j, m in enumerate(mics):
            t[i, j], sh[i, j] = arrival(m, u)
    return t, sh


def evaluate(mics, trials=40, sig_lit=20e-6, sig_sh=40e-6, outlier=.05):
    t, sh = table(mics)
    pairs = [(a, b) for a in range(len(mics)) for b in range(a + 1, len(mics))]
    ref = np.stack([t[:, b] - t[:, a] for a, b in pairs], axis=1)         # таблица задержек пар
    errs = np.zeros((len(GRID), trials))
    for i in range(len(GRID)):
        for k in range(trials):
            meas = []
            w = []
            for a, b in pairs:
                s = np.hypot(sig_sh if sh[i, a] else sig_lit, sig_sh if sh[i, b] else sig_lit)
                d = t[i, b] - t[i, a] + rng.normal(0, s)
                if rng.random() < outlier:
                    d = rng.uniform(-1.2e-3, 1.2e-3)                       # эхо: пик не там
                meas.append(d)
                w.append(1 / s ** 2)
            meas, w = np.array(meas), np.array(w)
            # робастная стоимость (как SRP: промах не тянет за собой всё)
            r = (ref - meas) ** 2 * w
            cost = np.minimum(r, 9.).sum(axis=1)
            j = int(np.argmin(cost))
            e = np.degrees(np.angle(np.exp(1j * (GRID[j] - GRID[i]))))
            errs[i, k] = abs(e)
    return errs


def report(layouts):
    print(f'{"расстановка":56s}  медиана  90%   >30°   худшие сектора (90%-ошибка, °)')
    for name, mics in layouts.items():
        e = evaluate(mics)
        per_dir = np.percentile(e, 90, axis=1)
        worst = np.argsort(per_dir)[-3:][::-1]
        sectors = ', '.join(f'{int(d)}°:{per_dir[d]:.0f}' for d in worst)
        print(f'{name:56s}  {np.median(e):5.1f}  {np.percentile(e, 90):5.1f}  {(e > 30).mean() * 100:4.1f}%   {sectors}')
        # по секторам: перёд (−45..45), лево (45..135), зад (135..225), право (225..315)
        for lo, label in ((-45, 'перёд'), (45, 'лево'), (135, 'зад'), (225, 'право')):
            idx = [(lo + d) % 360 for d in range(90)]
            print(f'      {label:6s}: медиана {np.median(e[idx]):4.1f}°, 90% {np.percentile(e[idx], 90):5.1f}°, '
                  f'>30° {(e[idx] > 30).mean() * 100:4.1f}%')


# 16.10, владелец: «чтобы ничего не сверлить» — готовые проёмы корпуса. Малый вырез спереди внизу на боку
# (x 102…148, высота 66…96) и три наклонные прорези сзади внизу на боках (x −112…−55, высота 55…87).
NO_DRILL = {
    'сверлить B: перёд-пр, пр.бок, лев.бок (для сравнения)': [(182, -80), (-88, Y0), (-88, Y1)],
    'без сверления 1: вырез спереди слева + прорези обоих боков': [(125, Y1), (-85, Y0), (-85, Y1)],
    'без сверления 2: вырезы спереди с обеих сторон + прорези слева': [(125, Y1), (125, Y0), (-85, Y1)],
    'без сверления 3: вырез спереди слева + прорези справа + задний проём': [(125, Y1), (-85, Y0), (X0, 0)],
}

if __name__ == '__main__':
    import sys
    report(NO_DRILL if '--no-drill' in sys.argv else LAYOUTS)
