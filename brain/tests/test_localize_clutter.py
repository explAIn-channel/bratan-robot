"""Scan matching in clutter, offline: python -m pytest tests/test_localize_clutter.py

10.10: in the bedroom doorway furniture hid the walls — 42-45 % of the scan on walls while the pose held within
1-4 cm, then a good 59-61 % fit 0.35 m past the drifted prediction was refused — and the robot called itself
lost every 10-30 s. tests/data/scans-1010-doorway.json: such scans as recorded on the robot (all refused then).
"""
import json
from pathlib import Path

import numpy as np

from bratan.body import localize, scan as scanmod
from bratan.body.grid import Grid
from tests.sim import SimLink

DATA = Path(__file__).parent / 'data'
GRID = Grid(str(DATA / 'apartment-final-20261006-dock.yaml'))
SCANS = json.loads((DATA / 'scans-1010-doorway.json').read_text(encoding='utf-8'))


def matched(record):
    pts = scanmod.points(scanmod.clean_ranges(record['ranges'], record.get('q')))
    return localize.match(GRID, tuple(record['predicted']), pts)


def test_a_cluttered_doorway_view_is_not_lost():
    for record in SCANS['held']:
        result = matched(record)
        assert result.ok and result.correction < .1, record['fraction']


def test_someone_close_by_does_not_make_him_lose_the_place():
    # 10.10 22:04: 24 scans in a row at mean 0.140-0.148 with 51-55 % on walls, the pose within 3 cm.
    for record in SCANS['stray']:
        result = matched(record)
        assert result.ok and .14 <= result.mean < .16 and result.correction < .1


def test_a_view_of_furniture_only_is_still_lost():
    pose = (2.6, 1.0, 3.1)
    link = SimLink(str(DATA / 'apartment-final-20261006-dock.yaml'), pose, (-.09, -.06, .19))
    ranges, _ = link._raycast(pose)
    ranges = np.where(ranges > .6, np.random.default_rng(3).uniform(.35, .55, len(ranges)), ranges)
    assert not localize.match(GRID, pose, scanmod.points(ranges)).ok


def test_a_jump_is_not_taken_while_tracking():
    # A fit 0.35 m from the prediction (a corridor looks alike further along) is for the search, not for tracking:
    # replayed on 4548 recorded scans such a jump was once 33 cm off the next sure fix.
    for record in SCANS['strong']:
        assert not matched(record).ok
