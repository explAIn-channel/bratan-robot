"""Калибровка «ушей» из сохранённых записей (recordings/calib-*.npz) — тем же модулем, что у робота
(bratan/voice/array.py): угол модели по каждому «Братан», ошибка без поправки и с поправкой «оставь одну сторону»,
и новая таблица ears-calibration.json (версия 2: fit + taus).
    python tools/calib_rebuild.py [маска файлов] [--write]"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'bratan'))
from bratan.voice import array  # noqa: E402

mask = next((a for a in sys.argv[1:] if not a.startswith('--')), 'calib-*.npz')
files = sorted(glob.glob(str(ROOT / 'recordings' / mask)))
rows = []
for path in files:
    d = np.load(path)
    ring = array.Ring()
    ring.put(1, 0, d['x'].astype(np.int16), 0.)
    feat = array.Direction(ring).features(seconds=3., history=3.)
    name = Path(path).name
    if feat is None:
        print(f'{name}: начал слогов нет — пропуск')
        continue
    fit, residual = array.model_fit(feat['taus'])
    deg = float(d['deg'])
    err = abs(array._wrap_deg(fit - deg))
    rows.append((deg, fit, feat['taus'], name))
    print(f'{name}: сторона {deg:+5.0f}°  модель {fit:+5.0f}°  ошибка {err:3.0f}°  невязка {residual:4.1f}  '
          f'начал {feat["onsets"]}')
if not rows:
    sys.exit('нет записей')
errs = [abs(array._wrap_deg(f - d)) for d, f, _, _ in rows]
print(f'\nбез поправки: медиана {np.median(errs):.0f}°, худшая {max(errs):.0f}°, ≤30°: {sum(e <= 30 for e in errs)} из {len(errs)}')
loo = []
for i, (deg, fit, taus, name) in enumerate(rows):
    cal = array.Calibration()
    for j, (d2, f2, t2, _) in enumerate(rows):
        if j != i and abs(array._wrap_deg(d2 - deg)) > 1:           # та же сторона другой записью — тоже вон
            cal.add(d2, f2, t2)
    if cal.ready():
        loo.append(abs(array._wrap_deg(cal.correct(fit) - deg)))
if loo:
    print(f'с поправкой (оставь одну сторону): медиана {np.median(loo):.0f}°, худшая {max(loo):.0f}°, '
          f'≤30°: {sum(e <= 30 for e in loo)} из {len(loo)}')
if '--write' in sys.argv:
    cal = array.Calibration(ROOT / 'ears-calibration.json')
    cal.entries = []
    for deg, fit, taus, _ in rows:
        cal.add(deg, fit, taus)
    cal.save()
    print('записано:', ROOT / 'ears-calibration.json', f'({len(cal.entries)} точек)')
