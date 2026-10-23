"""Doors between rooms: where they are (from the room map) and whether they are open (from the lidar).

08.10: with the bedroom door shut the mind concluded «the hall and the kitchen are closed» and kept trying
rooms behind the same door; it could not tell which door was shut. A real animal knows its doors.

Where: rooms touch on the room-label raster along their doorways; each separate contact between two rooms is
a door (its band: the contact cells and 20 cm around them, free floor away from the jambs).

Open or shut, from what each localized scan sees, standing in one of the two rooms:
- returns that land deep in the other room (rays get there only through the doorway): open;
- returns on the doorway's band, and none in the other room: a door leaf there — shut.
A state changes only after it was seen twice, ~1 s apart (a person passing is not a door).
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import label as connected

SEEN_AGAIN_S = .8            # a new state counts after a second sighting this much later
THROUGH_HITS = 3             # returns deep in the other room: the doorway is open
LEAF_HITS = 4                # returns on the doorway band with none beyond: shut
DEEP_M = .5                  # «deep»: this far from the doorway
BAND_M = .2
JAMB_M = .1                  # band cells this close to a wall are the jambs, not the door leaf


class Door:
    def __init__(self, labels, rooms, center, size, band):
        self.labels = labels                # (room label, room label)
        self.rooms = rooms                  # (room name, room name)
        self.center = center
        self.size = size                    # extent of the contact, metres
        self.band = band                    # bool raster of the doorway band
        self.state = None                   # 'open' | 'shut' | None (not seen yet)
        self.seen = 0.                      # when the state was last confirmed
        self.changed = 0.
        self.pending = None                 # (state, first time seen)

    def public(self):
        return {'rooms': list(self.rooms), 'state': self.state, 'seen': round(self.seen, 1),
                'center': [round(v, 2) for v in self.center]}


def find(grid, labels, rooms):
    """Doors from the room-label raster: every separate contact between two rooms."""
    names = {r['label']: r['name'] for r in rooms.get('rooms', []) if 'label' in r}
    if labels is None or labels.shape != grid.pixels.shape or not names:
        return []
    free = ~grid.structural & (grid.pixels >= 250)
    to_wall = distance_transform_edt(~grid.structural) * grid.resolution
    height, width = labels.shape
    contacts = {}
    for dr, dc in ((0, 1), (1, 0)):
        a, b = labels[:height - dr, :width - dc], labels[dr:, dc:]
        both = (a != b) & (a > 0) & (b > 0) & free[:height - dr, :width - dc] & free[dr:, dc:]
        for r, c in zip(*np.nonzero(both)):
            key = tuple(sorted((int(a[r, c]), int(b[r, c]))))
            contacts.setdefault(key, []).extend([(r, c), (r + dr, c + dc)])
    grow = max(1, int(round(BAND_M / grid.resolution)))
    doors = []
    for (la, lb), cells in contacts.items():
        if la not in names or lb not in names:
            continue
        mask = np.zeros(labels.shape, dtype=bool)
        rows, cols = zip(*cells)
        mask[list(rows), list(cols)] = True
        # Pieces of one doorway split by a pixel are one door: join what is within the band's reach.
        groups, count = connected(binary_dilation(mask, iterations=grow), structure=np.ones((3, 3)))
        for k in range(1, count + 1):
            part = mask & (groups == k)
            if part.sum() < 6:
                continue
            band = binary_dilation(part, iterations=grow) & free & (to_wall >= JAMB_M)
            rr, cc = np.nonzero(part)
            xy = np.array([grid.xy((int(r), int(c))) for r, c in zip(rr, cc)])
            doors.append(Door((la, lb), (names[la], names[lb]), tuple(xy.mean(axis=0).tolist()),
                              round(float(np.ptp(xy, axis=0).max()), 2), band))
    return doors


class Doors:
    """Door states from what each localized scan sees."""

    def __init__(self, grid, labels, doors, saved=None):
        self.grid, self.labels, self.doors = grid, labels, doors
        for door in doors:
            for item in (saved or []):
                if tuple(item.get('rooms', ())) == door.rooms and item.get('state') in ('open', 'shut'):
                    door.state, door.seen = item['state'], float(item.get('seen', 0.))
        self.changes = []                  # (door, state) to announce

    def observe(self, pose, hits, now=None):
        """`hits`: (N, 2) map points of one localized scan taken at `pose`."""
        if not self.doors or self.labels is None or not len(hits):
            return
        now = time.time() if now is None else now
        grid = self.grid
        here = grid.cell(pose)
        if not grid.inside(here):
            return
        mine = int(self.labels[here])
        rows, cols = grid.cells(hits)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        rows, cols, hits = rows[inside], cols[inside], hits[inside]
        where = self.labels[rows, cols]
        for door in self.doors:
            if mine not in door.labels:
                continue                   # from a third room this doorway cannot be seen
            other = door.labels[1] if mine == door.labels[0] else door.labels[0]
            far = np.hypot(hits[:, 0] - door.center[0], hits[:, 1] - door.center[1]) > DEEP_M
            through = int(((where == other) & far).sum())
            leaf = int(door.band[rows, cols].sum())
            if through >= THROUGH_HITS:
                seen = 'open'
            elif leaf >= LEAF_HITS and through == 0 and math.dist(pose[:2], door.center) < 4.:
                seen = 'shut'
            else:
                continue
            if seen == door.state:
                door.seen, door.pending = now, None
                continue
            if door.pending is None or door.pending[0] != seen:
                door.pending = (seen, now)
                continue
            if now - door.pending[1] >= SEEN_AGAIN_S:
                old = door.state
                door.state, door.seen, door.changed, door.pending = seen, now, now, None
                if old is not None:
                    self.changes.append((door, seen))

    def public(self):
        return [d.public() for d in self.doors]
