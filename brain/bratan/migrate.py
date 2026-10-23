"""One-time import of what the old system (r285) knew: run on the phone before the first start.

python -m bratan.migrate   (idempotent: existing new state is kept unless --force)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from .common.config import STATE, read_json, write_json
from .mind.memory import Archive

OLD = Path('/opt/robotbrain/state')
OLD_MEMORY = OLD / 'integrated/memory.sqlite'
SKIP_LABELS = {'person', 'dog', 'wall', 'door', 'unknown', 'floor', 'ceiling', 'window'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    report = {}

    mind_path = STATE / 'mind.json'
    if args.force or not mind_path.exists():
        clock = read_json(OLD / 'psyche/integrated-clock.json', {}) or {}
        old = clock.get('mind', {})
        character = read_json(OLD / 'psyche/character.json', {}) or {}
        album = read_json(OLD / 'album/index.json', {}) or {}
        mind = {key: old[key] for key in ('journal', 'notes', 'dialogue', 'said', 'wants', 'plans', 'skills', 'self')
                if old.get(key)}
        mind['character'] = {'traits': character.get('traits', {}),
                             'lessons': (character.get('preferences') or {}).get('lessons', [])[-12:]}
        mind['album'] = [{'id': p['id'], 'path': str(OLD / 'album' / p['file']), 't': p.get('at', time.time()),
                          'caption': p.get('caption', '')} for p in album.get('photos', [])]
        mind['pets'] = (clock.get('social') or {}).get('pets', {})
        write_json(mind_path, mind)
        report['mind'] = {k: len(v) if hasattr(v, '__len__') else v for k, v in mind.items()}

    if OLD_MEMORY.exists():
        con = sqlite3.connect(f'file:{OLD_MEMORY}?mode=ro', uri=True)
        meta = dict(con.execute('SELECT key, value FROM meta').fetchall())
        location = json.loads(meta.get('location', '{}'))
        if location.get('dock') and (args.force or not (STATE / 'location.json').exists()):
            write_json(STATE / 'location.json', {'pose': location.get('pose'), 'dock': location['dock'],
                                                 'boot': location.get('body_boot'), 't': time.time()})
            report['dock'] = location['dock']
        reference = location.get('dock_reference') or {}
        if reference.get('points') and (args.force or not (STATE / 'dock-reference.json').exists()):
            from .body.localize import densify
            import numpy as np
            points = densify(np.asarray(reference['points'], dtype=float))
            write_json(STATE / 'dock-reference.json', {'dock': reference['pose'],
                                                        'points': np.round(points, 4).tolist()})
            report['dock_reference_points'] = len(points)
        archive = Archive(STATE / 'memory.sqlite')
        if args.force or not archive.objects(min_count=1, limit=1):
            rows = con.execute('SELECT label, x, y, seen, observations FROM objects WHERE observations >= 5 '
                               'AND x IS NOT NULL ORDER BY seen').fetchall()
            imported = 0
            for label, x, y, seen, count in rows:
                if label in SKIP_LABELS:
                    continue
                archive.saw(label, (x, y), None, seen, count=min(int(count), 20))
                imported += 1
            report['objects_imported'] = imported
            report['objects_kept'] = len(archive.objects(min_count=1, limit=10000))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
