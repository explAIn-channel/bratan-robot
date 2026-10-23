"""What Братан remembers.

Working memory (`MindState`, one JSON file): thoughts, wants, plans, own skills,
self-description, recent dialogue and speech, notes, character, people, rooms.
Long-term (`Archive`, SQLite): every event and dialogue line, and the objects
seen around the flat (label, place, when) for "what did you see / where is".
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from pathlib import Path

from ..common.config import read_json, write_json

JOURNAL_KEPT = 40
DIALOGUE_KEPT = 24
SAID_KEPT = 30


class MindState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = read_json(self.path, {}) or {}
        self.dirty = False
        self.saved_at = 0.

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setdefault(self, key, default):
        self.dirty = True
        return self.data.setdefault(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.dirty = True

    def append(self, key, value, keep):
        items = self.data.setdefault(key, [])
        items.append(value)
        del items[:-keep]
        self.dirty = True

    def save(self, force=False):
        if self.dirty and (force or time.monotonic() - self.saved_at > 5):
            write_json(self.path, self.data)
            self.dirty = False
            self.saved_at = time.monotonic()

    # convenience -----------------------------------------------------------
    def think(self, now, thought, did=None):
        entry = {'t': now, 'thought': thought}
        if did:
            entry['did'] = did
        self.append('journal', entry, JOURNAL_KEPT)

    def heard(self, now, who, text):
        self.append('dialogue', {'t': now, 'who': who, 'text': text}, DIALOGUE_KEPT)

    def spoke(self, now, to, text):
        self.append('said', {'t': now, 'to': to, 'text': text}, SAID_KEPT)


class Archive:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript('''
                PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS events(t REAL, kind TEXT, data TEXT);
                CREATE INDEX IF NOT EXISTS events_t ON events(t);
                CREATE TABLE IF NOT EXISTS objects(id INTEGER PRIMARY KEY, label TEXT, x REAL, y REAL,
                    room TEXT, first REAL, seen REAL, count INTEGER, name TEXT);
                CREATE INDEX IF NOT EXISTS objects_label ON objects(label);
                CREATE TABLE IF NOT EXISTS sightings(t REAL, who TEXT, room TEXT);
                CREATE INDEX IF NOT EXISTS sightings_who ON sightings(who, t);
            ''')
            columns = {row[1] for row in self.db.execute('PRAGMA table_info(objects)')}
            # gone: hidden, not deleted (08.10: a wrong «it is not there» erased the bed and the sofa for good);
            # place: what it lay on («на столе»), for «где ключи?».
            for column, kind in (('gone', 'REAL'), ('place', 'TEXT')):
                if column not in columns:
                    self.db.execute(f'ALTER TABLE objects ADD COLUMN {column} {kind}')
            self.db.commit()

    def event(self, kind, data, t=None):
        with self.lock:
            self.db.execute('INSERT INTO events VALUES(?,?,?)',
                            (t or time.time(), kind, json.dumps(data, ensure_ascii=False, default=str)[:4000]))
            self.db.commit()

    def events(self, since, kinds=None, limit=200):
        query = 'SELECT t, kind, data FROM events WHERE t >= ?'
        args = [since]
        if kinds:
            query += ' AND kind IN (%s)' % ','.join('?' * len(kinds))
            args += list(kinds)
        query += ' ORDER BY t DESC LIMIT ?'
        args.append(limit)
        with self.lock:
            rows = self.db.execute(query, args).fetchall()
        return [(t, k, json.loads(d)) for t, k, d in rows]

    def saw(self, label, xy, room, now, name=None, merge_m=.6, count=1, place=None):
        """Remember an object at a place; the same label nearby updates the old entry (a hidden one comes back).

        Returns True when this is a thing never seen here before (for curiosity about new things)."""
        with self.lock:
            rows = self.db.execute('SELECT id, x, y FROM objects WHERE label=? AND (name IS ? OR name=?)',
                                   (label, name, name)).fetchall()
            near = [(math.dist((x, y), xy), i) for i, x, y in rows if x is not None]
            near = [n for n in near if n[0] < merge_m]
            if near:
                _, oid = min(near)
                self.db.execute('UPDATE objects SET x=(x*0.7+?*0.3), y=(y*0.7+?*0.3), seen=MAX(seen, ?), '
                                'count=count+?, room=COALESCE(?, room), gone=NULL, place=COALESCE(?, place) WHERE id=?',
                                (xy[0], xy[1], now, count, room, place, oid))
                new = False
            else:
                self.db.execute('INSERT INTO objects(label,x,y,room,first,seen,count,name,place) VALUES(?,?,?,?,?,?,?,?,?)',
                                (label, xy[0], xy[1], room, now, now, count, name, place))
                new = True
            self.db.commit()
        return new

    def tidy(self, room_at, moving_labels):
        """Moving things out of the object memory; rooms for objects remembered without one."""
        with self.lock:
            removed = self.db.execute(f'UPDATE objects SET count=0, gone=? WHERE gone IS NULL AND label IN '
                                      f'({",".join("?" * len(moving_labels))})', (time.time(), *moving_labels)).rowcount
            rows = self.db.execute('SELECT id, x, y FROM objects WHERE room IS NULL AND x IS NOT NULL').fetchall()
            placed = 0
            for oid, x, y in rows:
                room = room_at((x, y))
                if room:
                    self.db.execute('UPDATE objects SET room=? WHERE id=?', (room, oid))
                    placed += 1
            self.db.commit()
        return removed, placed, len(rows)

    FIELDS = ('id', 'label', 'x', 'y', 'room', 'seen', 'count', 'name', 'place')

    def fade(self, oid, amount=3):
        """Looked right at the place and it is not there: trust the memory less; hidden at zero.

        Returns the hidden object (dict) or None. The owner: a thing taken away must not stay for ever —
        but nothing is deleted: a thing seen there again simply comes back (saw clears `gone`).
        """
        with self.lock:
            row = self.db.execute(f'SELECT {",".join(self.FIELDS)} FROM objects WHERE id=? AND gone IS NULL',
                                  (oid,)).fetchone()
            if row is None:
                return None
            item = dict(zip(self.FIELDS, row))
            if (item['count'] or 0) - amount <= 0:
                self.db.execute('UPDATE objects SET count=0, gone=? WHERE id=?', (time.time(), oid))
                self.db.commit()
                return item
            self.db.execute('UPDATE objects SET count=count-? WHERE id=?', (amount, oid))
            self.db.commit()
        return None

    def forget(self, oid):
        """A wrong entry (it turned out to be something else): hidden, kept for the record."""
        with self.lock:
            self.db.execute('UPDATE objects SET count=0, gone=? WHERE id=?', (time.time(), oid))
            self.db.commit()

    def object(self, oid):
        with self.lock:
            row = self.db.execute(f'SELECT {",".join(self.FIELDS)} FROM objects WHERE id=? AND gone IS NULL',
                                  (oid,)).fetchone()
        return dict(zip(self.FIELDS, row)) if row else None

    def objects(self, *, min_count=3, limit=40, near=None, labels=None):
        query = f'SELECT {",".join(self.FIELDS)} FROM objects WHERE count>=? AND gone IS NULL'
        args = [min_count]
        if labels:
            query += ' AND label IN (%s)' % ','.join('?' * len(labels))
            args += list(labels)
        with self.lock:
            rows = self.db.execute(query + ' ORDER BY seen DESC LIMIT 400', args).fetchall()
        items = [dict(zip(self.FIELDS, r)) for r in rows]
        if near is not None:
            items.sort(key=lambda o: math.dist((o['x'], o['y']), near))
        return items[:limit]

    def day(self, since):
        """What I did since `since`: places, things approached, who was seen, talks (for «как прошёл день»)."""
        went, approached, explored = [], [], []
        for _, _, step in reversed(self.events(since, ('mind_step',), limit=400)):
            effect = str(step.get('effect') or '')
            tool = ((step.get('do') or {}).get('tool'))
            for prefix, bucket in (('еду в «', went), ('подъезжаю: ', approached)):
                if prefix in effect:
                    name = effect.split(prefix, 1)[1].split('»')[0].split(';')[0].strip()
                    if name and name not in bucket:
                        bucket.append(name)
            if tool == 'explore_room':
                room = str((step.get('do') or {}).get('args', {}).get('room') or '').strip()
                if room and room not in explored:
                    explored.append(room)
        looked = []
        for _, _, done in reversed(self.events(since, ('action_done',), limit=200)):
            text = str(done.get('text') or '')
            if 'рассмотрел вблизи — ' in text:
                looked.append(' '.join(text.split('рассмотрел вблизи — ', 1)[1].split())[:110])
        with self.lock:
            seen = self.db.execute('SELECT who, COUNT(*) FROM sightings WHERE t>=? GROUP BY who', (since,)).fetchall()
        talks = len(self.events(since, ('conversation_start',), limit=200))
        parts = []
        if went:
            parts.append('ездил в: ' + ', '.join(went))
        if approached:
            parts.append('подъезжал рассмотреть: ' + ', '.join(approached))
        if explored:
            parts.append('осматривал комнаты: ' + ', '.join(explored))
        if looked:
            parts.append('что увидел вблизи: ' + ' | '.join(looked[-3:]))
        if seen:
            parts.append('видел: ' + ', '.join(f'{who} ({n} раз)' for who, n in seen))
        if talks:
            parts.append(f'разговаривал с людьми {talks} раз')
        return '; '.join(parts) or 'пока ничего особенного'

    def sighting(self, who, room, now):
        with self.lock:
            self.db.execute('INSERT INTO sightings VALUES(?,?,?)', (now, who, room))
            self.db.commit()

    def last_seen(self, who):
        with self.lock:
            row = self.db.execute('SELECT t, room FROM sightings WHERE who=? ORDER BY t DESC LIMIT 1', (who,)).fetchone()
        return row

    def habits(self, who, days=7):
        """Where and at what hours someone is usually seen (for plans and curiosity)."""
        since = time.time() - days * 86400
        with self.lock:
            rows = self.db.execute('SELECT t, room FROM sightings WHERE who=? AND t>=? ORDER BY t', (who, since)).fetchall()
        return rows

    def talks_with(self, who, since, default_who=None):
        """How many times this person talked with me (for attachment: the ones who talk to me are close).

        Talks recorded before 07.10 carry no speaker; `default_who` (the owner — the only one who had
        talked to the robot by then) is credited with those."""
        with self.lock:
            rows = self.db.execute("SELECT data FROM events WHERE kind='converse' AND t>=?", (since,)).fetchall()
        count = 0
        for (data,) in rows:
            try:
                speaker = json.loads(data).get('who', default_who)
            except ValueError:
                continue
            if speaker == who:
                count += 1
        return count

    DIARY_SOUNDS = {'doorbell': 'звонили в дверь', 'knock': 'стучали', 'dog': 'лаял Таффа', 'bang': 'что-то грохнуло',
                    'alarm': 'звенел будильник или телефон', 'crying': 'кто-то плакал', 'music': 'играла музыка',
                    'laughter': 'смеялись', 'vacuum': 'гудел пылесос', 'door': 'хлопала дверь'}

    def diary(self, since, now=None):
        """What happened at home since a moment (Astro's activity log, in words): «что было, пока меня не было?»."""
        now = time.time() if now is None else now
        rows = self.events(since, ('heard_sound', 'door_opened', 'door_closed', 'person_appeared', 'dog_appeared',
                                   'game_played', 'mind_step', 'object_gone', 'docked', 'battery_low'), limit=600)
        sounds, doors, people, games, deeds = {}, [], [], [], []
        for t, kind, data in reversed(rows):
            clock = time.strftime('%H:%M', time.localtime(t))
            if kind == 'heard_sound' and data.get('sound') in self.DIARY_SOUNDS:
                # Where from, when the ears placed it in another room (C52).
                where = f' со стороны «{data["toward"]}»' if data.get('toward') else ''
                sounds.setdefault(data['sound'], []).append(clock + where)
            elif kind in ('door_opened', 'door_closed'):
                doors.append(f'{clock} {data.get("text", "")}')
            elif kind == 'person_appeared':
                who = data.get('who') or 'человек'
                if who == 'человек' and data.get('unknown_face'):
                    who = 'незнакомое лицо'
                if who not in [p.split(' ', 1)[1] for p in people]:
                    people.append(f'{clock} {who}')
            elif kind == 'game_played':
                games.append(f'{clock} {data.get("text", "")}')
            elif kind == 'object_gone':
                deeds.append(f'{clock} {data.get("text", "")}')
        parts = []
        for kind, times in sounds.items():
            shown = ', '.join(times[:4]) + (f' и ещё {len(times) - 4}' if len(times) > 4 else '')
            parts.append(f'{self.DIARY_SOUNDS[kind]} ({shown})')
        if doors:
            parts.append('двери: ' + '; '.join(doors[-4:]))
        if people:
            parts.append('видел: ' + ', '.join(people[:6]))
        if games:
            parts.append('игры: ' + '; '.join(games[-3:]))
        if deeds:
            parts.append('заметил: ' + '; '.join(deeds[-2:]))
        return '; '.join(parts) or 'дома было тихо, ничего особенного'

    def kinds_by(self, who, since, kinds):
        """How many events of each kind a person caused (praise, touch, games) — for attachment."""
        with self.lock:
            rows = self.db.execute('SELECT kind, data FROM events WHERE t>=? AND kind IN (%s)' % ','.join('?' * len(kinds)),
                                   (since, *kinds)).fetchall()
        counts = {}
        for kind, data in rows:
            try:
                if json.loads(data).get('who') == who:
                    counts[kind] = counts.get(kind, 0) + 1
            except ValueError:
                continue
        return counts

    def search(self, words, limit=12):
        """Recall: recent events/dialogue lines containing any of the words."""
        words = [w for w in words if len(w) >= 3][:4]
        if not words:
            return []
        clause = ' OR '.join('data LIKE ?' for _ in words)
        with self.lock:
            rows = self.db.execute(f'SELECT t, kind, data FROM events WHERE {clause} ORDER BY t DESC LIMIT ?',
                                   [f'%{w}%' for w in words] + [limit]).fetchall()
        return [(t, k, json.loads(d)) for t, k, d in rows]
