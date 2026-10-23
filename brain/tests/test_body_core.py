import math
import time
from pathlib import Path

import numpy as np
import pytest

from bratan.body import localize, planner
from bratan.body.grid import Grid
from bratan.body import scan as scanmod
from bratan.common.geometry import FOOTPRINT, compose, inside_footprint, footprint_clearance
from tests.sim import SimLink

DATA = Path(__file__).parent / 'data'
MAP = DATA / 'apartment-final-20261006-dock.yaml'
DOCK = (-0.0887, -0.0557, 0.1896)


@pytest.fixture(scope='module')
def grid():
    return Grid(MAP)


def scan_at(grid, pose, seed=0):
    sim = SimLink(MAP, pose, DOCK, seed=seed)
    ranges, _ = sim._raycast(pose)
    return scanmod.points(ranges)


def test_footprint_tests():
    assert inside_footprint([[0, 0]]).all()
    assert not inside_footprint([[.3, 0], [0, .2], [-.2, 0]]).any()
    d = footprint_clearance([[.3, 0], [0, 0]])
    assert d[0] == pytest.approx(.3 - .2259172, abs=1e-6) and d[1] < 0


def test_match_recovers_offset_quickly(grid):
    truth = (1.53, 0.38, 0.4)
    pts = scan_at(grid, truth)
    started = time.perf_counter()
    for dx, dy, dyaw in ((.15, -.1, .2), (-.2, .1, -.25), (.05, .05, .05)):
        result = localize.match(grid, (truth[0] + dx, truth[1] + dy, truth[2] + dyaw), pts)
        assert result.ok, result
        assert math.dist(result.pose[:2], truth[:2]) < .04
        assert abs(result.pose[2] - truth[2]) < .03
    assert (time.perf_counter() - started) / 3 < .05


def test_global_recovery(grid):
    truth = (6.54, 1.38, -1.0)
    pts = scan_at(grid, truth, seed=3)
    result = localize.recover(grid, (0., 0., 0.), pts, deadline_s=20.)
    assert result is not None and math.dist(result.pose[:2], truth[:2]) < .1


def test_plan_between_rooms(grid):
    started = time.perf_counter()
    route = planner.plan(grid, (1.53, 0.38), (8.02, 1.53))
    spent = time.perf_counter() - started
    assert len(route) >= 2 and route[-1] == (8.02, 1.53)
    print('plan s', spent, 'points', len(route), 'doors', route.doors)
    assert spent < 1.5


def test_dock_reference_consensus(grid):
    pts = scan_at(grid, DOCK, seed=5)
    ref = localize.DockReference()
    assert ref.record(DOCK, pts)
    near = compose(DOCK, (.3, .02, .05))
    live = scan_at(grid, near, seed=6)
    results = [ref.match(compose(near, (.03, -.02, .03)), live, t, (0., 0., 0.)) for t in (1., 1.2, 1.4)]
    assert results[-1] is not None
    assert math.dist(results[-1].pose[:2], near[:2]) < .02


def test_find_by_name_takes_an_unrecognised_face_never_someone_else():
    """08.10 01:34: «подъеду к Сергею» at night — in view, face not recognised — counted as not seen."""
    import types
    from bratan.body.main import Ctx
    owner = types.SimpleNamespace(detections_t=10., odom_history=[], detections=[
        {'label': 'person', 'name': 'Марина', 'bearing': .0, 't': 10., 'track': 1},
        {'label': 'person', 'name': None, 'bearing': .3, 't': 10., 'track': 2}])
    ctx = Ctx.__new__(Ctx)
    ctx.owner, ctx.now, ctx.odom = owner, 10.2, (0., 0., 0.)
    assert ctx.find('Сергей')['track'] == 2
    assert ctx.find('Марина')['track'] == 1
    owner.detections = owner.detections[:1]
    assert ctx.find('Сергей') is None
