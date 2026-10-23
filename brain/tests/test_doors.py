"""Doors between rooms seen by the lidar (08.10: the mind could not tell which door was shut)."""
import json
from pathlib import Path

import numpy as np

from bratan.body import doors as doorsmod
from bratan.body import scan as scanmod
from bratan.body.grid import Grid
from bratan.common.geometry import transform
from tests.sim import SimLink

DATA = Path(__file__).parent / 'data'
MAP = DATA / 'apartment-final-20261006-dock.yaml'
DOCK = (-0.0887, -0.0557, 0.1896)
ROOMS = json.loads((DATA / 'rooms.json').read_text(encoding='utf-8'))


def setup():
    grid = Grid(MAP)
    labels = np.load(DATA / 'rooms-labels.npy')
    return grid, labels, doorsmod.Doors(grid, labels, doorsmod.find(grid, labels, ROOMS))


def look(doors, pose, extra_walls=(), times=(0., 1., 2.)):
    sim = SimLink(MAP, pose, DOCK, extra_walls=extra_walls)
    for t in times:
        ranges, _ = sim._raycast(pose)
        doors.observe(pose, transform(pose, scanmod.points(ranges)), now=1000. + t)


def bedroom_door(doors):
    return next(d for d in doors.doors if set(d.rooms) == {'Спальня и кабинет', 'Коридор'})


def test_the_three_doors_of_the_flat():
    _, _, doors = setup()
    pairs = {frozenset(d.rooms) for d in doors.doors}
    assert pairs == {frozenset({'Коридор', 'Спальня и кабинет'}), frozenset({'Коридор', 'Кухня'}),
                     frozenset({'Коридор', 'Зал'})}


def test_open_bedroom_door_is_seen_open_and_a_shut_one_shut():
    _, _, doors = setup()
    pose = (2.0, .7, .2)                                   # in the bedroom, facing the corridor door
    look(doors, pose)
    door = bedroom_door(doors)
    assert door.state == 'open'
    # The door leaf across the doorway (x ~3.1, y 0.6-1.35).
    look(doors, pose, extra_walls=[(3.08, .55, 3.13, 1.4)], times=(3., 4., 5.))
    assert door.state == 'shut' and doors.changes[-1] == (door, 'shut')


def test_a_door_out_of_sight_keeps_its_state():
    _, _, doors = setup()
    door = bedroom_door(doors)
    look(doors, (2.0, .7, .2))
    assert door.state == 'open'
    # From the corridor's far end the bedroom door cannot be judged: nothing changes.
    look(doors, (8.0, 1.4, 0.), extra_walls=[(3.08, .55, 3.13, 1.4)], times=(3., 4., 5.))
    assert door.state == 'open'
