"""Storage guard: the phone's memory must never fill up (owner, 06.10).

Every few minutes: how much is free on the phone, and the robot's own growing data is kept in
bounds — the supervisor log, trip videos (the PC takes them), the archive of events, the album.
Only data this runtime itself produces is ever removed; anything else is reported, not touched.

Levels: ok (> LOW_BYTES free), low (the mind is told; trip videos are capped harder),
critical (< CRITICAL_BYTES: no new trip videos at all).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

LOW_BYTES = 6 << 30
CRITICAL_BYTES = 2 << 30
MAIN_LOG_LIMIT = 8 << 20
MAIN_LOG_KEEP = 1 << 20
EVENTS_KEEP_DAYS = 120
KEEP_KINDS = ('dream', 'reflection', 'converse', 'learned_face', 'conversation_start')   # kept for good
ALBUM_KEEP = 400
TRIPS = Path('/proc/1/root/data/media/0/Android/data/ai.robotbrain.face/files/trips')
TRIP_KEEP_DAYS = 3


def free_bytes(path='/opt/robotbrain'):
    return shutil.disk_usage(path).free


def level_of(free):
    return 'critical' if free < CRITICAL_BYTES else 'low' if free < LOW_BYTES else 'ok'


def trim_log(path: Path, limit=MAIN_LOG_LIMIT, keep=MAIN_LOG_KEEP):
    """Keep the tail of an append-only log in place (the writer keeps its file handle)."""
    try:
        if path.stat().st_size <= limit:
            return 0
        with open(path, 'rb') as handle:
            handle.seek(-keep, 2)
            tail = handle.read()
        with open(path, 'r+b') as handle:
            handle.write(tail)
            handle.truncate(len(tail))
        return limit - keep
    except OSError:
        return 0


def prune_trips(now, keep_days=TRIP_KEEP_DAYS):
    """Trip videos older than a few days: the PC agent has long taken them."""
    removed = 0
    try:
        for path in TRIPS.glob('trip-*.mp4'):
            if now - path.stat().st_mtime > keep_days * 86400:
                removed += path.stat().st_size
                path.unlink()
    except OSError:
        pass
    return removed


def prune_album(state, keep=ALBUM_KEEP):
    album = state.get('album', [])
    if len(album) <= keep:
        return 0
    removed = 0
    for photo in album[:-keep]:
        try:
            path = Path(photo.get('path', ''))
            if path.is_file():
                removed += path.stat().st_size
                path.unlink()
        except OSError:
            pass
    state.set('album', album[-keep:])
    return removed


def prune_archive(archive, now, days=EVENTS_KEEP_DAYS):
    with archive.lock:
        cursor = archive.db.execute(
            f'DELETE FROM events WHERE t < ? AND kind NOT IN ({",".join("?" * len(KEEP_KINDS))})',
            (now - days * 86400, *KEEP_KINDS))
        archive.db.execute('DELETE FROM sightings WHERE t < ?', (now - days * 86400,))
        archive.db.commit()
        return cursor.rowcount
