"""Liveness numbers per day from the robot's archive (09.10 audit, section 9): read-only, no model.

python tools/liveness.py <memory.sqlite> [days]      e.g. a copy from E:/RobotBrain-soul/*.tgz
On the phone: python tools/liveness.py /opt/robotbrain/bratan/state/memory.sqlite 3

Per local day: time off the dock and trips, thoughts (and how many at night), thoughts that repeat the one
before, thoughts woken by his own refusal, words he wanted to say and the code refused, talks and how often the
speaker was known, the answer time, use of his own «freedom» tools, starts of the program (each a small amnesia).
"""
import difflib
import json
import re
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Europe/Moscow')
QUIET = (23, 10)
FREEDOM = ('want', 'self', 'new_skill', 'use_skill', 'remember_place', 'plan', 'note')


def quiet(ts):
    hour = datetime.fromtimestamp(ts, TZ).hour
    return hour >= QUIET[0] or hour < QUIET[1]


def similar(a, b):
    a, b = re.sub(r'[^а-яa-z ]', '', a.casefold()), re.sub(r'[^а-яa-z ]', '', b.casefold())
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.


def day(db, start, end):
    rows = db.execute('SELECT t, kind, data FROM events WHERE t >= ? AND t < ? ORDER BY t', (start, end)).fetchall()
    out, trips, left = {}, [], None
    steps, talks = [], []
    counts = {'quiet_note': 0, 'objects_tidy': 0, 'mind_empty': 0}
    for t, kind, data in rows:
        if kind == 'undocked':
            left = t
        elif kind == 'docked' and left is not None:
            trips.append(t - left)
            left = None
        elif kind == 'mind_step':
            steps.append((t, json.loads(data)))
        elif kind == 'converse':
            talks.append(json.loads(data))
        elif kind in counts:
            counts[kind] += 1
    thoughts = [(t, str(s.get('thought') or '')) for t, s in steps]
    repeats = sum(1 for (_, a), (_, b) in zip(thoughts, thoughts[1:]) if similar(a, b) > .5)
    says = [s for _, s in steps if s.get('say')]
    known = [x for x in talks if x.get('who') not in (None, '', 'человек')]
    latencies = [x['latency_s'] for x in talks if isinstance(x.get('latency_s'), (int, float))]
    tools = [((s.get('do') or {}).get('tool')) for _, s in steps]
    out['вне базы, мин'] = round(sum(trips) / 60)
    out['выездов'] = len(trips)
    out['выезд, медиана мин'] = round(statistics.median(trips) / 60, 1) if trips else 0
    out['мыслей'] = len(steps)
    out['мыслей ночью'] = sum(1 for t, _ in steps if quiet(t))
    out['повторы, %'] = round(100 * repeats / max(1, len(thoughts) - 1))
    out['повод — свой отказ, %'] = round(100 * sum(1 for _, s in steps if s.get('trigger') == 'refused') / max(1, len(steps)))
    out['реплик хотел / отклонено'] = f'{len(says)} / {counts["quiet_note"]}'
    out['разговорных ответов'] = len(talks)
    out['собеседник известен, %'] = round(100 * len(known) / max(1, len(talks)))
    out['ответ, медиана с'] = round(statistics.median(latencies), 1) if latencies else None
    out['инструменты свободы'] = sum(1 for x in tools if x in FREEDOM)
    out['радость в мыслях'] = sum(1 for _, x in thoughts if re.search(r'\bрад\b|рада\b|люблю|весел', x.casefold()))
    out['пустых мыслей'] = counts['mind_empty']
    out['запусков программы'] = counts['objects_tidy']
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/opt/robotbrain/bratan/state/memory.sqlite'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    last = db.execute('SELECT MAX(t) FROM events').fetchone()[0] or time.time()
    today = datetime.fromtimestamp(last, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(days - 1, -1, -1):
        start = today - timedelta(days=back)
        numbers = day(db, start.timestamp(), (start + timedelta(days=1)).timestamp())
        print(start.strftime('%d.%m'), '  '.join(f'{k}: {v}' for k, v in numbers.items()))


if __name__ == '__main__':
    main()
