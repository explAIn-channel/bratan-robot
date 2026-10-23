"""Where the sounds of the house come from (C52, 18.10): the body's ears give a direction (voice/array.py), the map gives
the rest — a direction from where I stand is a ray across the rooms.

A sound from another room reaches me through its doorway (or muffled, through a wall): the ray points at the door, not at
the source, so a direction that meets a doorway of my room means «со стороны» the room behind it. Heard again from
another spot, two rays cross where it is (robots map sound sources from rays heard at different poses: Sasaki et al.
2006, Rauschenberger & Wagner 2016). A sound gone quiet is still worth going to: where and what it was is kept, as the
semantic audio-visual navigation agents keep a goal after the sound stops (Chen et al., CVPR 2021).
"""
from __future__ import annotations

import math
import time
from collections import deque

RAY_MAX_M = 8.               # no room here is longer
STEP_M = .05                 # the map's own cell
WALL_PASS_M = .6             # looking past a wall for the room behind it
DOOR_CONE = .35              # rad: a doorway this close to the direction (plus its own half-width) — it came through it
DOOR_HALF_M = .4
RAY_FRESH_S = 90.            # a direction is worth going by this long after it was heard
CROSS_MIN = math.radians(15.)  # two rays crossing at a smaller angle place nothing
CROSS_BASE_M = .6            # ... nor do two heard from almost the same spot
CROSS_FRESH_S = 300.
SURE = .3                    # the ears' own confidence below this: no direction at all (voice/array.py)
MOVING_RAD_S = .15           # turning faster than this while hearing: the direction is smeared — not a ray


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def trace(room_map, room_names, xy, heading):
    """Walk the ray from `xy` along `heading` over the room raster: the rooms it crosses before the first wall, the
    free length up to that wall, and the room just behind it (None outside the map)."""
    grid, labels = room_map
    crossed, free, behind = [], RAY_MAX_M, None
    dx, dy = math.cos(heading), math.sin(heading)
    for k in range(1, int(RAY_MAX_M / STEP_M) + 1):
        d = k * STEP_M
        cell = grid.cell((xy[0] + d * dx, xy[1] + d * dy))
        if not grid.inside(cell):
            free = d
            break
        if grid.structural[cell]:
            free = d
            for j in range(1, int(WALL_PASS_M / STEP_M) + 1):
                past = grid.cell((xy[0] + (d + j * STEP_M) * dx, xy[1] + (d + j * STEP_M) * dy))
                if not grid.inside(past):
                    break
                name = room_names.get(int(labels[past]))
                if name and not grid.structural[past]:
                    behind = name
                    break
            break
        name = room_names.get(int(labels[cell]))
        if name and (not crossed or crossed[-1] != name):
            crossed.append(name)
    return crossed, free, behind


class SoundMap:
    """The rays of the last minutes, what they point at, and where two of one sound cross."""

    def __init__(self, world):
        self.world = world
        self.rays: deque = deque(maxlen=120)

    # ------------------------------------------------------------------ one ray
    def add(self, kind, words, bearing, conf, t=None):
        """A direction the ears gave → a ray on the map (dict), or None when it cannot be placed (unsure, no pose,
        lost on the map, turning while it was heard)."""
        world = self.world
        body = world.body or {}
        pose = body.get('pose')
        if bearing is None or conf is None or conf < SURE or not pose or body.get('localized') is False:
            return None
        twist = body.get('twist') or (0., 0.)
        if abs(float(twist[1])) > MOVING_RAD_S:
            return None
        t = time.time() if t is None else t
        heading = _wrap(float(pose[2]) + float(bearing))
        ray = {'t': t, 'kind': kind, 'words': words, 'x': float(pose[0]), 'y': float(pose[1]),
               'heading': round(heading, 3), 'bearing': round(float(bearing), 3), 'conf': round(float(conf), 2),
               'here': world.room}
        ray.update(self.toward(ray))
        self.rays.append(ray)
        return ray

    def toward(self, ray):
        """What the ray points at: {'toward': room, 'via': 'door'|'open'|'wall'|'here', 'free': m, 'door': xy}."""
        world = self.world
        here, xy, heading = ray.get('here'), (ray['x'], ray['y']), ray['heading']
        if world.room_map is None:
            return {'toward': None, 'via': None, 'free': None}
        names = {room.get('label'): name for name, room in world.rooms.items()}
        crossed, free, behind = trace(world.room_map, names, xy, heading)
        # Through a doorway of my room: the sound comes from the room behind it.
        best = None
        for door in world.doors or []:
            rooms, center = door.get('rooms') or [], door.get('center')
            if not center or here not in rooms or len(rooms) != 2:
                continue
            distance = math.dist(xy, center)
            off = abs(_wrap(math.atan2(center[1] - xy[1], center[0] - xy[0]) - heading))
            if off <= DOOR_CONE + math.atan2(DOOR_HALF_M, max(distance, .3)) and (best is None or off < best[0]):
                best = (off, rooms[1] if rooms[0] == here else rooms[0], center)
        if best is not None:
            return {'toward': best[1], 'via': 'door', 'free': round(free, 2), 'door': [round(v, 2) for v in best[2]]}
        beyond = [name for name in crossed if name != here]
        if beyond:
            return {'toward': beyond[0], 'via': 'open', 'free': round(free, 2)}
        if behind and behind != here:
            return {'toward': behind, 'via': 'wall', 'free': round(free, 2)}
        return {'toward': here, 'via': 'here', 'free': round(free, 2)}

    # ------------------------------------------------------------------ what the rays say together
    def snapshot(self):
        """The rays as they are now: the loop adds them while the thoughts read them (a copy in one C call does not
        see a half-added ray — iterating the deque itself raised «deque mutated during iteration» in the ears, 17.10)."""
        return tuple(self.rays)

    def latest(self, kinds=None, within=RAY_FRESH_S, now=None):
        now = time.time() if now is None else now
        for ray in reversed(self.snapshot()):
            if now - ray['t'] > within:
                return None
            if kinds is None or ray['kind'] in kinds:
                return ray
        return None

    def cross(self, kinds, within=CROSS_FRESH_S, now=None):
        """Where two rays of the same sound heard from different spots meet: {'xy', 'room', 'rays'} or None."""
        now = time.time() if now is None else now
        rays = [r for r in self.snapshot() if r['kind'] in kinds and now - r['t'] <= within]
        best = None
        for i in range(len(rays) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                a, b = rays[i], rays[j]
                point = intersect(a, b)
                if point is None:
                    continue
                score = a['conf'] * b['conf']
                if best is None or score > best[0]:
                    best = (score, point, (a, b))
        if best is None:
            return None
        xy = best[1]
        return {'xy': [round(xy[0], 2), round(xy[1], 2)], 'room': self.world.room_at(xy), 'rays': 2}

    @staticmethod
    def words(ray):
        """«со стороны «Коридор»» when the ray goes out through a doorway or an opening, else ''. Only that is said:
        on the dock by the wall most sounds of the house (kitchen, music, voices behind the door) were heard «справа»
        (18.10) — reflections; «здесь» or «из-за стены» would be guesses."""
        if ray and ray.get('toward') and ray.get('via') in ('door', 'open'):
            return f'со стороны «{ray["toward"]}»'
        return ''


def intersect(a, b):
    """The point two rays meet ahead of both, or None (too parallel, too close together, behind, too far)."""
    if math.dist((a['x'], a['y']), (b['x'], b['y'])) < CROSS_BASE_M:
        return None
    if abs(_wrap(a['heading'] - b['heading'])) < CROSS_MIN or abs(abs(_wrap(a['heading'] - b['heading'])) - math.pi) < CROSS_MIN:
        return None
    ax, ay, bx, by = a['x'], a['y'], b['x'], b['y']
    da = (math.cos(a['heading']), math.sin(a['heading']))
    db = (math.cos(b['heading']), math.sin(b['heading']))
    det = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(det) < 1e-6:
        return None
    rx, ry = bx - ax, by - ay
    ta = (rx * (-db[1]) - ry * (-db[0])) / det
    tb = (da[0] * ry - da[1] * rx) / det
    if ta < .3 or tb < .3 or ta > RAY_MAX_M or tb > RAY_MAX_M:
        return None
    return ax + ta * da[0], ay + ta * da[1]
