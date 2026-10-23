"""Как компас ведёт себя по ходу звука: запись трёх микрофонов (calib-*.npz) прокручивается кусками по 32 мс,
и каждые 0,1 с считается направление тем же модулем, что у робота (окно 1 с, как у панели). Видно, куда
уходит стрелка на хвосте, когда звук затихает.

  python tools/dir_timeline.py "recordings/calib-*.npz" [--loo] [--quiet]
--loo — поправка калибровки без той стороны, что проверяется (иначе запись проверяет сама себя).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'bratan'))
from bratan.voice import array  # noqa: E402

CAL = ROOT / 'ears-calibration.json'


def wrap(d):
    return (d + 180.) % 360. - 180.


def timeline(path, calibration, step_s=.1, window_s=1.):
    data = np.load(path)
    x, valid, deg = data['x'], data['valid'], float(data['deg'])
    ring = array.Ring()
    direction = array.Direction(ring, calibration)
    if hasattr(direction, 'reset'):
        direction.reset()
    out = []
    block = 512
    next_at = step_s
    for first in range(0, len(x) - block + 1, block):
        part = x[first:first + block].copy()
        part[~valid[first:first + block]] = 0
        ring.put(1, first, part, (first + block) / array.RATE)
        t = (first + block) / array.RATE
        if t + 1e-9 >= next_at:
            next_at += step_s
            found = direction.estimate(window_s)
            level = 10 * np.log10(np.mean(x[max(0, first + block - 1600):first + block].astype(float) ** 2) + 1)
            out.append((t, level, found))
    return deg, out


def fingerprints(paths):
    """Отпечаток каждой записи проверяемым расчётом (как calib_rebuild): сторона, угол модели, задержки."""
    rows = []
    for path in paths:
        d = np.load(path)
        ring = array.Ring()
        ring.put(1, 0, d['x'].astype(np.int16), 0.)
        feat = array.Direction(ring).features(seconds=3., history=3.)
        if feat is not None:
            rows.append({'deg': float(d['deg']), 'fit': array.model_fit(feat['taus'])[0],
                         'taus': [float(v) for v in feat['taus']], 'path': path})
    return rows


def main(argv):
    args = [a for a in argv if not a.startswith('--')]
    pattern = args[0] if args else str(ROOT / 'recordings' / 'calib-*.npz')
    loo, quiet = '--loo' in argv, '--quiet' in argv
    paths = sorted(glob.glob(pattern))
    if loo:
        entries = fingerprints(paths)                 # поправка — тем же расчётом, что проверяется
    else:
        entries = json.loads(CAL.read_text(encoding='utf-8')).get('entries', []) if CAL.exists() else []
    summary = []
    for path in paths:
        deg = float(np.load(path)['deg'])
        cal = array.Calibration()
        cal.entries = [e for e in entries if not (loo and round(e['deg']) == round(deg))]
        deg, out = timeline(path, cal)
        confident = [(t, f) for t, _, f in out if f and f['confidence'] >= .3]
        errors = [abs(wrap(f['deg'] - deg)) for _, f in confident]
        last = confident[-1][1] if confident else None
        summary.append((Path(path).name, deg, len(confident), np.median(errors) if errors else None,
                        max(errors) if errors else None, abs(wrap(last['deg'] - deg)) if last else None))
        if quiet:
            continue
        print(f'\n{Path(path).name}  истина {deg:+.0f}°')
        for t, level, f in out:
            if f is None:
                print(f'  {t:4.1f} с  {level:5.1f} дБ   —')
            else:
                err = wrap(f['deg'] - deg)
                mark = '  <-- мимо' if abs(err) > 45 and f['confidence'] >= .3 else ''
                print(f'  {t:4.1f} с  {level:5.1f} дБ  {f["deg"]:+6.0f}°  ош {err:+5.0f}  ув {f["confidence"]:.2f}  '
                      f'начал {f["onsets"]:3d}  невязка {f["residual"]:4.1f}{mark}')
    print('\nзапись                                   истина  увер.  медиана  худшая  последняя')
    for name, deg, n, med, worst, last in summary:
        fmt = lambda v: '   —  ' if v is None else f'{v:5.0f}°'
        print(f'{name:40s} {deg:+5.0f}°  {n:4d}  {fmt(med)}   {fmt(worst)}   {fmt(last)}')
    lasts = [s[5] for s in summary if s[5] is not None]
    worsts = [s[4] for s in summary if s[4] is not None]
    if lasts:
        print(f'последняя уверенная: медиана {np.median(lasts):.0f}°, мимо (>45°) {sum(v > 45 for v in lasts)} из '
              f'{len(lasts)}; худшая за звук: медиана {np.median(worsts):.0f}°, мимо {sum(v > 45 for v in worsts)}')


if __name__ == '__main__':
    main(sys.argv[1:])
