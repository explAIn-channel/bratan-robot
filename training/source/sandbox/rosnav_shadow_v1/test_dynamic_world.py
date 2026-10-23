#!/usr/bin/env python3
"""What the robot must see, and must not, when the flat has people in it.

An obstacle that never reaches the scan is worse than no obstacle at all: the
training run still costs the hours, the metrics still improve, and the number
that improves is measuring an empty flat.  So each of these asserts against
the beams rather than against the bookkeeping.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from simulate_expert_dataset import (  # noqa: E402
    BODY_RADIUS_M as BODY_CLEARANCE_M, SENSOR_RANGE_MAX, SENSOR_RANGE_MIN,
    sensor_window, core,
)
from dynamic_world import (  # noqa: E402
    GOAL_CLEARANCE_M, LEG_GAP_M, LEG_RADIUS_M,
    MIN_SPAWN_M, SCENARIOS, Body, World, make_creature, make_object,
)

BEAMS = 360
RANGE_MIN, RANGE_MAX = 0.15, 12.0


def clear_scan() -> np.ndarray:
    return np.full(BEAMS, RANGE_MAX, dtype=np.float32)


def beam_towards(bearing_rad: float) -> int:
    """Index of the beam pointing this way, for a robot facing +x."""
    return int(round(bearing_rad / (2.0 * math.pi / BEAMS))) % BEAMS


class SeeingTests(unittest.TestCase):
    def test_live_sensor_window_drops_near_and_far_returns_without_clipping(self):
        world = World(scenario="static")
        near = make_creature("person", 0.0, 0.21, 0.0, 0.0)
        near.offsets = np.zeros((1, 2), dtype=np.float32)
        near.radii = np.array([0.05], dtype=np.float32)
        world.bodies = [near]
        raw = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), 0.0)
        side = beam_towards(math.pi / 2.0)
        self.assertLess(float(raw[side]), SENSOR_RANGE_MIN)
        self.assertTrue(math.isinf(float(sensor_window(raw)[side])))

        far = make_creature("person", SENSOR_RANGE_MAX + 0.2, 0.0, 0.0, 0.0)
        far.offsets = np.zeros((1, 2), dtype=np.float32)
        far.radii = np.array([0.05], dtype=np.float32)
        world.bodies = [far]
        raw = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), 0.0)
        self.assertGreater(float(raw[0]), SENSOR_RANGE_MAX)
        self.assertTrue(math.isinf(float(sensor_window(raw)[0])))

    def test_a_bag_ahead_shortens_the_forward_beam_to_its_near_face(self):
        world = World(scenario="static")
        body = make_creature("person", 3.0, 0.0, 0.0, 0.0)
        # Replace the legs with one disc of a known size, to check the maths.
        body.offsets = np.zeros((1, 2), dtype=np.float32)
        body.radii = np.array([0.25], dtype=np.float32)
        world.bodies = [body]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        self.assertAlmostEqual(float(scan[0]), 2.75, places=2)

    def test_nothing_behind_the_obstacle_is_touched(self):
        world = World(scenario="static")
        world.bodies = [make_creature("person", 3.0, 0.0, 0.0, 0.0)]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        self.assertEqual(float(scan[180]), RANGE_MAX)

    def test_a_person_reads_as_two_legs_with_daylight_between_them(self):
        """The whole reason creatures are not cylinders.

        At 8 cm the lidar meets ankles.  If the gap between them closed up,
        the network would be trained against an obstacle this flat does not
        contain -- and would meet its first real person as two thin posts
        with an inviting space in the middle.
        """
        world = World(scenario="standing")
        world.bodies = [make_creature("person", 2.0, 0.0, 0.0, 0.0)]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        straight_ahead = float(scan[0])
        self.assertEqual(straight_ahead, RANGE_MAX,
                         "the gap between the ankles closed up")
        half_gap = LEG_GAP_M["person"] * 0.5
        left = beam_towards(math.atan2(half_gap, 2.0))
        self.assertLess(float(scan[left]), 2.1)
        self.assertGreater(float(scan[left]), 1.9)

    def test_a_leg_is_the_width_a_leg_should_be(self):
        world = World(scenario="standing")
        world.bodies = [make_creature("person", 2.0, 0.0, 0.0, 0.0)]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        struck = np.flatnonzero(scan < RANGE_MAX)
        self.assertTrue(struck.size, "no beam met the person at all")
        # Two legs, each about 2*r wide at 2 m, is a handful of beams -- not
        # the forty-odd a solid body would take.
        self.assertLess(struck.size, 24)
        span = 2.0 * LEG_RADIUS_M["person"] / 2.0
        self.assertGreater(struck.size, 2 * math.degrees(span) / 2)

    def test_a_clear_flat_leaves_every_beam_alone(self):
        world = World(scenario="clear")
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        self.assertTrue(np.all(scan == RANGE_MAX))

    def test_an_obstacle_behind_the_robot_is_seen_behind_it(self):
        world = World(scenario="static")
        body = make_creature("person", -2.0, 0.0, 0.0, 0.0)
        body.offsets = np.zeros((1, 2), dtype=np.float32)
        body.radii = np.array([0.3], dtype=np.float32)
        world.bodies = [body]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        self.assertAlmostEqual(float(scan[180]), 1.7, places=2)
        self.assertEqual(float(scan[0]), RANGE_MAX)

    def test_a_body_the_robot_is_inside_does_not_wrap_around(self):
        """A negative intersection is behind the sensor and is not a return."""
        world = World(scenario="static")
        body = make_creature("person", 0.0, 0.0, 0.0, 0.0)
        body.offsets = np.zeros((1, 2), dtype=np.float32)
        body.radii = np.array([0.05], dtype=np.float32)
        world.bodies = [body]
        scan = world.fold_into_scan(0.0, 0.0, 0.0, clear_scan(), RANGE_MIN)
        self.assertTrue(np.all(scan == RANGE_MAX))


class FakeFlat:
    """Open floor, so placement rules can be checked without a map."""

    resolution = 0.05
    height = width = 400
    origin = (0., 0.)

    def __init__(self):
        self.free = np.ones((self.height, self.width), dtype=bool)
        self.standable = self.free

    def cell(self, x, y):
        return int(self.height - 1 - y / self.resolution), int(x / self.resolution)

    def metres(self, row, column):
        return column * self.resolution, (self.height - 1 - row) * self.resolution


class FakeEnv:
    def __init__(self):
        self.flat = FakeFlat()
        self.x, self.y, self.yaw = 5.0, 5.0, 0.0
        self.goal = (12.0, 5.0)
        self.standable = np.argwhere(self.flat.standable)
        self.route = None


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()

    def distances(self, world, point):
        out = []
        for body in world.bodies:
            if not math.isfinite(body.x):
                continue
            centres, _radii = body.circles()
            for centre in centres:
                out.append(math.hypot(centre[0] - point[0], centre[1] - point[1]))
        return out

    def test_nothing_spawns_on_the_bumper(self):
        """Collisions the driver could not have avoided are not a lesson."""
        for seed in range(40):
            for scenario in ("static", "standing", "walking", "crowded"):
                world = World(scenario=scenario,
                              rng=np.random.default_rng(seed))
                world.reset(self.env)
                for distance in self.distances(world, (self.env.x, self.env.y)):
                    self.assertGreater(distance, MIN_SPAWN_M - 0.4,
                                       f"{scenario} spawned on top of the robot")

    def test_nothing_sits_on_the_goal(self):
        for seed in range(40):
            world = World(scenario="crowded", rng=np.random.default_rng(seed))
            world.reset(self.env)
            for distance in self.distances(world, self.env.goal):
                self.assertGreater(distance, GOAL_CLEARANCE_M - 0.4,
                                   "an obstacle made arriving impossible")

    def test_the_ambush_is_absent_until_its_moment(self):
        world = World(scenario="sudden", rng=np.random.default_rng(1))
        world.reset(self.env)
        self.assertEqual(len(world.bodies), 1)
        body = world.bodies[0]
        self.assertGreater(body.appears_at, 0)
        self.assertFalse(math.isfinite(body.x))
        scan = world.fold_into_scan(self.env.x, self.env.y, 0.0,
                                    clear_scan(), RANGE_MIN)
        self.assertTrue(np.all(scan == RANGE_MAX),
                        "the ambush was visible before it existed")
        for _ in range(body.appears_at):
            world.step(self.env)
        self.assertTrue(math.isfinite(world.bodies[0].x),
                        "the ambush never arrived")

    def test_a_walker_stays_on_the_floor(self):
        world = World(scenario="walking", rng=np.random.default_rng(2))
        world.reset(self.env)
        for _ in range(400):
            world.step(self.env)
            for body in world.bodies:
                if math.isfinite(body.x):
                    self.assertTrue(World._walkable(self.env, body.x, body.y),
                                    "somebody walked through a wall")

    def test_a_standing_person_stands_still(self):
        world = World(scenario="standing", rng=np.random.default_rng(4))
        world.reset(self.env)
        before = [(body.x, body.y) for body in world.bodies]
        for _ in range(50):
            world.step(self.env)
        self.assertEqual(before, [(body.x, body.y) for body in world.bodies])

    def test_a_clear_flat_is_empty(self):
        world = World(scenario="clear", rng=np.random.default_rng(5))
        world.reset(self.env)
        self.assertEqual(world.bodies, [])

    def test_every_named_scenario_can_actually_be_built(self):
        for scenario in SCENARIOS:
            world = World(scenario=scenario, rng=np.random.default_rng(6))
            world.reset(self.env)
        with self.assertRaises(ValueError):
            World(scenario="mayhem").reset(self.env)

    def test_the_populated_scenarios_actually_populate(self):
        for scenario in ("static", "standing", "walking", "sudden", "crowded"):
            found = 0
            for seed in range(10):
                world = World(scenario=scenario,
                              rng=np.random.default_rng(seed))
                world.reset(self.env)
                found += len(world.bodies)
            self.assertGreater(found, 0, f"{scenario} put nothing in the flat")


class FakeRoute(core.Wavefront):
    """A cost field that falls towards the goal, so `descend` can walk it."""

    def __init__(self, flat, goal):
        self.map = flat
        rows, columns = np.indices((flat.height, flat.width))
        goal_row, goal_column = flat.cell(*goal)
        self.cost = np.hypot(rows - goal_row,
                             columns - goal_column).astype(np.float32)


class ReachTests(unittest.TestCase):
    """Obstacles have to stand along the whole drive, not just its start.

    This is the failure that hides: with the route walked only a fixed eighty
    cells, every scenario places its people in the opening four metres and a
    policy scores full marks on "crowded" by clearing one chair and then
    driving down an empty corridor for another ten.  Nothing errors, nothing
    looks wrong, and the headline number is meaningless.
    """

    def setUp(self):
        self.env = FakeEnv()
        self.env.goal = (16.0, 5.0)          # 11 m away: a real crossing
        self.env.route = FakeRoute(self.env.flat, self.env.goal)
        self.env.max_steps = 1300

    def fractions(self, scenario, seeds=30):
        out = []
        span = math.hypot(self.env.goal[0] - self.env.x,
                          self.env.goal[1] - self.env.y)
        for seed in range(seeds):
            world = World(scenario=scenario, rng=np.random.default_rng(seed))
            world.reset(self.env)
            for body in world.bodies:
                if math.isfinite(body.x):
                    out.append(math.hypot(body.x - self.env.x,
                                          body.y - self.env.y) / span)
        return np.array(out)

    def test_the_route_is_walked_past_its_first_few_metres(self):
        points = World(scenario="static")._route_points(self.env)
        self.assertTrue(points)
        furthest = max(math.hypot(x - self.env.x, y - self.env.y)
                       for x, y in points)
        self.assertGreater(furthest, 8.0,
                           "the route was only walked a few metres out")

    def test_obstacles_reach_the_far_half_of_the_drive(self):
        for scenario in ("static", "standing", "walking", "crowded"):
            fractions = self.fractions(scenario)
            self.assertTrue(fractions.size, f"{scenario} placed nothing")
            self.assertGreater(
                (fractions > 0.5).mean(), 0.25,
                f"{scenario} keeps its obstacles near the start")

    def test_the_ambush_can_come_late_in_a_long_drive(self):
        latest = 0
        for seed in range(40):
            world = World(scenario="sudden", rng=np.random.default_rng(seed))
            world.reset(self.env)
            latest = max(latest, world.bodies[0].appears_at)
        self.assertGreater(latest, 400,
                           "the ambush only ever springs in the first seconds")


class CorridorFlat:
    """One corridor, walls either side, so sealing it is unambiguous."""

    resolution = 0.05

    def __init__(self, width_m: float):
        self.height, self.width = 200, 200
        free = np.zeros((self.height, self.width), dtype=bool)
        half = int(round(width_m / 2 / self.resolution))
        middle = self.height // 2
        free[middle - half:middle + half + 1, 10:190] = True
        self.free = free
        from scipy import ndimage
        self.blocked_distance = (ndimage.distance_transform_edt(free)
                                 .astype(np.float32) * self.resolution)
        self.standable = self.blocked_distance >= BODY_CLEARANCE_M

    def cell(self, x, y):
        return int(self.height - 1 - y / self.resolution), int(x / self.resolution)

    def metres(self, row, column):
        return (column + 0.5) * self.resolution, (self.height - 1 - row - 0.5) * self.resolution


class CorridorEnv:
    def __init__(self, width_m):
        self.flat = CorridorFlat(width_m)
        middle = self.flat.height // 2
        self.x, self.y = self.flat.metres(middle, 20)
        self.goal = self.flat.metres(middle, 180)
        self.yaw = 0.0
        self.standable = np.argwhere(self.flat.standable)
        self.route = None


class PassabilityTests(unittest.TestCase):
    """An obstacle must narrow the way, never close it.

    A thing that seals the corridor does not teach going round -- it teaches
    that the drive was lost before it began. Trained against a mix where a
    quarter of the episodes could not be won, the policy gave up the
    plan-following it already had (80 % to 45 % on an empty flat) without
    learning anything about obstacles.
    """

    def blocker(self, radius, x, y):
        return Body(offsets=np.zeros((1, 2), dtype=np.float32),
                    radii=np.array([radius], dtype=np.float32), x=x, y=y)

    def test_a_body_that_seals_the_corridor_is_refused(self):
        env = CorridorEnv(0.80)
        world = World(scenario="static")
        middle_y = env.y
        self.assertFalse(
            world._admits(env, self.blocker(0.40, 5.0, middle_y)),
            "an obstacle wider than the corridor was accepted")

    def test_two_individually_passable_objects_must_not_seal_the_route_together(self):
        env = CorridorEnv(1.20)
        world = World(scenario="static")
        first = self.blocker(.15, 5., env.y+.35)
        second = self.blocker(.15, 5., env.y-.35)
        self.assertTrue(world._admits(env, first))
        self.assertTrue(world._admits(env, second))
        world.bodies.append(first)
        self.assertFalse(world._admits(env, second))

    def test_body_and_swept_translation_cannot_pass_through_a_wall(self):
        env = CorridorEnv(.80)
        body = make_creature('person', 5., env.y+.32, 0., 1.)
        self.assertTrue(World._walkable(env, body.x, body.y))
        self.assertFalse(World._walkable(env, body.x, body.y, body=body))
        env = FakeEnv()
        _, column = env.flat.cell(5., 5.)
        env.flat.free[:,column] = False
        body = self.blocker(.03, 4.8, 5.)
        body.vx = 3.
        world = World(scenario='walking', bodies=[body])
        self.assertTrue(World._walkable(env, 5.1, 5., body=body))
        world.step(env)
        self.assertEqual((body.x,body.y),(4.8,5.))

    def test_a_body_that_leaves_room_is_accepted(self):
        env = CorridorEnv(1.20)
        world = World(scenario="static")
        self.assertTrue(
            world._admits(env, self.blocker(0.15, 5.0, env.y + 0.35)))

    def test_the_gap_it_leaves_must_fit_the_chassis(self):
        """0.05 m of daylight is not a way past a 0.35 m robot."""
        env = CorridorEnv(0.80)
        world = World(scenario="static")
        self.assertFalse(
            world._admits(env, self.blocker(0.30, 5.0, env.y - 0.10)))

    def test_placement_actually_uses_the_check(self):
        env = CorridorEnv(0.80)
        for seed in range(15):
            world = World(scenario="static", rng=np.random.default_rng(seed))
            world.reset(env)
            for body in world.bodies:
                self.assertTrue(world._admits(env, body),
                                "a sealing placement was kept")


class ObjectTests(unittest.TestCase):
    def test_a_chair_is_legs_and_a_box_is_not(self):
        rng = np.random.default_rng(0)
        shapes = {len(make_object(rng, 0.0, 0.0).radii) for _ in range(60)}
        self.assertIn(1, shapes)
        self.assertIn(4, shapes, "no chair was ever produced")


if __name__ == "__main__":
    unittest.main()
