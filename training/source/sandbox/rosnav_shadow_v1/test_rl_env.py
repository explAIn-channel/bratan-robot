#!/usr/bin/env python3
"""The environment's two silent failures, held open so they cannot close again.

Both were found by disbelieving a good number rather than by anything failing.

The first: for the whole life of this project the simulator reported zero
collisions, in every run, for every policy.  Not because the driving was
clean -- because the test could not fire.  A beam is clipped at the sensor's
0.15 m minimum range and a collision was declared below 0.12 m, so the
comparison was false by construction and the twelve-point penalty was never
once applied.  Measured against the map instead, the body outline was inside
something on 9.6 % of steps.

The second: ``0.175 / 0.05`` is 3.4999999999999996, so rounding it gave a
three-cell body and the simulator drove a robot 5 cm narrower than the one in
the hall.  That is how start poses came to be handed out inside walls.

Neither had a symptom.  Both had a green metric.
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

from dynamic_world import World, make_creature  # noqa: E402
from rl_env import (COLLISION_PENALTY, MAX_ANGULAR, MAX_LINEAR, STEP_COST,  # noqa: E402
                    DriveEnv, FastFlat)
from simulate_expert_dataset import (BODY_HALF_WIDTH_M, BODY_RADIUS_M,  # noqa: E402
                                     DT, Flat, FOOTPRINT_FRONT_M,
                                     FOOTPRINT_REAR_M, RANGE_MIN,
                                     outline_gap, wrap)

from features import LIDAR_BINS, SCAN_VIEWS  # noqa: E402

MAP = HERE.parents[3] / "assets/map/map.yaml"
SCENARIOS = ("clear", "static", "standing", "walking", "sudden", "crowded")


def flat():
    if not MAP.is_file():                       # pragma: no cover
        raise unittest.SkipTest(f"training map missing: {MAP}")
    return FastFlat(MAP)


class RaySurfaceTests(unittest.TestCase):
    def test_returns_first_cell_surface_without_skipping_thin_obstacles(self):
        from scipy.ndimage import distance_transform_edt
        from simulate_expert_dataset import BEAMS, sensor_window
        arena = FastFlat.__new__(FastFlat)
        arena.resolution, arena.origin = .05, (-2., -2.)
        arena.height = arena.width = 81
        arena.free = np.ones((81, 81), dtype=bool)
        arena.free[[0, -1], :] = arena.free[:, [0, -1]] = False
        arena.free[15:65, 60] = False
        arena.free[30, 44] = arena.free[52, 37] = False
        arena.blocked_distance = distance_transform_edt(arena.free) * arena.resolution
        rows, columns = np.nonzero(~arena.free)
        centres = np.array([arena.metres(int(r), int(c)) for r, c in zip(rows, columns)])
        for x, y, heading in ((.013, .021, 0.), (-.31, .42, .437),
                              (.811, -.213, -.172), (-.47, -.19, 1.2)):
            angles = heading + np.arange(BEAMS) * math.tau / BEAMS
            directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
            # Independent analytic ray/square intersections, including isolated
            # single cells, oblique grazes and returns inside the sensor floor.
            with np.errstate(divide='ignore', invalid='ignore'):
                a = (centres[None] - .025 - [x, y]) / directions[:, None]
                b = (centres[None] + .025 - [x, y]) / directions[:, None]
            entry = np.maximum(np.minimum(a, b).max(axis=2), 0.)
            leave = np.maximum(a, b).min(axis=2)
            expected = sensor_window(np.where(leave >= entry, entry, np.inf).min(axis=1))
            actual = arena.raycast(x, y, heading)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        arena.free[:] = True
        arena.resolution, arena.origin = .5, (-20., -20.)
        arena.blocked_distance = distance_transform_edt(arena.free) * arena.resolution
        self.assertTrue(np.isposinf(arena.raycast(0., 0., .437)).all())
        self.assertAlmostEqual(float(arena.raycast(19.5, 0., 0.)[0]), .75, places=6)
        self.assertTrue(np.isposinf(arena.raycast(20.15, 0., 0.)[0]))


class InitialSceneTests(unittest.TestCase):
    def test_only_certified_impossible_worlds_are_resampled(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        arena = FastFlat.__new__(FastFlat)
        arena.resolution, arena.origin = .1, (-5., -5.)
        arena.height = arena.width = 101
        arena.free = np.ones((101, 101), dtype=bool)
        arena.plannable = arena.free
        arena.blocked_distance = np.full(arena.free.shape, 5., dtype=np.float32)
        arena.clearance = .25
        arena.field = SimpleNamespace(field=arena.blocked_distance, free=arena.free,
            resolution=arena.resolution, origin=arena.origin, height=101, width=101)
        env = DriveEnv(arena, np.random.default_rng(19), max_steps=100, scenario='walking')
        places = iter(((0., 0.), (0., 1.5)))
        env._random_pose = lambda: next(places)
        placements = []
        def place(world, robot):
            heading = robot.yaw + (math.pi if not placements else math.pi/2)
            body = make_creature('person', robot.x+1.1*math.cos(robot.yaw),
                robot.y+1.1*math.sin(robot.yaw), heading, 1.4)
            world.bodies, world.steps = [body], 0
            placements.append((body.x, body.y))
        with patch.object(World, 'reset', place):
            env.reset()
        self.assertEqual(len(env.initial_world_rejections), 1)
        self.assertGreater(env.initial_world_rejections[0]['margin_m'], 1e-5)
        self.assertEqual(len(placements), 2)
        self.assertEqual(env.world.scenario, 'walking')
        self.assertEqual(len(env.world.bodies), 1)
        self.assertEqual(env.world.steps, 0, 'forecast must not advance the delivered world')
        self.assertEqual((env.world.bodies[0].x, env.world.bodies[0].y), placements[-1])
        self.assertEqual(env.goal, (0., 1.5))
        state = repr(env.rng.bit_generator.state)
        self.assertIsNone(env._initial_collision_certificate())
        self.assertEqual(repr(env.rng.bit_generator.state), state)
        # No proof may cross a robot-relative ambush: its future pose depends
        # on the very control sequence the bound quantifies over.
        placements.clear()
        place(env.world, env)
        self.assertIsNotNone(env._initial_collision_certificate())
        ambush = make_creature('dog', math.nan, math.nan, 0., 1.)
        ambush.appears_at = 1
        env.world.bodies.append(ambush)
        self.assertIsNone(env._initial_collision_certificate())
        places = iter(((0., 0.), (0., 1.5)))
        env._random_pose = lambda: next(places)
        with patch.object(DriveEnv, '_initial_collision_certificate', return_value={'tick': 1}):
            with self.assertRaisesRegex(RuntimeError, 'proven unavoidable collision'):
                env.reset()
        self.assertEqual(len(env.initial_world_rejections), 60)


class CollisionIsReachableTests(unittest.TestCase):
    """The guard against the dead branch coming back."""

    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def test_physics_and_planner_use_the_same_map_cells(self):
        import sim_geometry as core
        route = core.Wavefront(self.flat.field)
        rng = np.random.default_rng(9807)
        for _ in range(80):
            row = int(rng.integers(1, self.flat.height - 1))
            column = int(rng.integers(1, self.flat.width - 1))
            centre = self.flat.metres(row, column)
            self.assertEqual(route._cell(centre), (row, column))
            for offset in (-.4, .0, .4):
                x = centre[0] + offset * self.flat.resolution
                y = centre[1] - offset * self.flat.resolution
                self.assertEqual(self.flat.cell(x, y), route._cell((x, y)))
                distance = self.flat._distance_at(np.array([x]), np.array([y]))[0]
                self.assertEqual(distance, self.flat.blocked_distance[row, column])

    def test_the_test_is_not_asking_a_clipped_sensor(self):
        """State the trap as arithmetic, so nobody re-introduces it.

        Any collision threshold at or below the sensor's minimum range is
        unreachable, whatever the robot does.
        """
        self.assertGreater(BODY_RADIUS_M, RANGE_MIN,
                           "the collision threshold is inside the sensor's "
                           "blind spot and can never be reached")

    def test_driving_into_a_wall_is_a_collision(self):
        """The whole point: it has to be possible to fail."""
        env = DriveEnv(self.flat, np.random.default_rng(3), scenario="clear")
        collided = 0
        for _ in range(12):
            env.reset()
            # Drive flat out, straight ahead, ignoring the plan entirely.
            for _ in range(env.max_steps):
                _obs, _reward, done, outcome = env.step(
                    np.array([MAX_LINEAR, 0.0], dtype=np.float32))
                if done:
                    collided += int(outcome == "collision")
                    break
        self.assertGreater(collided, 6,
                           "charging blindly across the flat never once "
                           "counted as hitting anything")

    def test_a_person_the_robot_reaches_is_a_collision(self):
        """A leg inside the outline must end the drive even though the beam,
        clipped at 0.15 m, reports the same number it would from 0.15 m away."""
        env = DriveEnv(self.flat, np.random.default_rng(4), scenario="clear")
        env.reset()
        env.world.bodies = [make_creature(
            "person", env.x + 0.05, env.y, 0.0, 0.0)]
        _obs, _reward, done, outcome = env.step(
            np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(done)
        self.assertEqual(outcome, "collision")

    def test_open_floor_is_not_a_collision(self):
        env = DriveEnv(self.flat, np.random.default_rng(5), scenario="clear")
        env.reset()
        self.assertGreaterEqual(env.gap, BODY_RADIUS_M)

    def test_turn_response_matches_measured_chassis_time_constant(self):
        from sim_geometry import TURN_COAST_S
        env = DriveEnv(self.flat, np.random.default_rng(5), scenario="clear")
        env.reset()
        env.angular = 0.0
        for _ in range(5):
            env.step(np.array([0.0, .15]))
        self.assertAlmostEqual(env.angular, .15 * (1 - math.exp(-5 * DT / TURN_COAST_S)))

    def test_the_body_is_not_the_circle_it_spins_in(self):
        """A doorframe alongside the robot is not a collision.

        The robot sweeps 0.251632 m turning on the spot and presents 0.116 m to
        the side.  Colliding on the sweep circle declared contact in every
        doorway the chassis actually drives through: measured on 2026-08-30 it
        cost the reference driver four episodes in five on a flat with nothing
        in it, and it is why the planner could not be given more margin without
        deleting a room.
        """
        beside = np.array([[0.0, 0.20]])
        self.assertLess(BODY_HALF_WIDTH_M, 0.20,
                        "the fixture must sit clear of the robot's side")
        self.assertGreater(BODY_RADIUS_M, 0.20,
                           "and inside the circle it spins in, or the test "
                           "distinguishes nothing")
        self.assertGreater(outline_gap(0.0, 0.0, 0.0, beside)[0], 0.0)

    def test_the_outline_reaches_further_in_front_of_the_axle(self):
        """The axle is 45 percent forward from the tail, so the nose is longer.

        Stated as behaviour rather than as arithmetic on the constants: a point
        the same distance fore and aft is contact only at the front.
        """
        self.assertGreater(FOOTPRINT_FRONT_M, FOOTPRINT_REAR_M)
        span = 0.5 * (FOOTPRINT_FRONT_M + FOOTPRINT_REAR_M)
        self.assertGreater(outline_gap(0.0, 0.0, 0.0,
                                       np.array([[-span, 0.0]]))[0], 0.0)
        self.assertEqual(outline_gap(0.0, 0.0, 0.0,
                                     np.array([[span, 0.0]]))[0], 0.0)

    def test_collision_contour_includes_paws_and_preserves_empty_corners(self):
        # Nose and ear samples outside the former shell rectangle are occupied.
        local = np.array([[.233, 0.], [.12, .145], [0., 0.],
                          [.247, 0.], [.247, .130], [-.20, 0.], [.20, .145]])
        expected = [0., 0., 0., .010, math.sqrt(.0002), .014]
        for yaw in (0., .7, -2.4):
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                                 [math.sin(yaw), math.cos(yaw)]])
            world = local @ rotation.T + [2., -3.]
            gaps = outline_gap(2., -3., yaw, world)
            np.testing.assert_allclose(gaps[:6], expected, atol=1e-12)
            self.assertGreater(gaps[6], 0., 'sloping nose is not its bounding box')
        self.assertEqual(outline_gap(0., 0., 0., np.empty((0, 2))).shape, (0,))

    def test_occupied_grid_cell_corner_cannot_pass_through_the_nose(self):
        from types import SimpleNamespace
        free = np.ones((101, 101), dtype=bool)
        free[43, 63] = False  # Centre (.246, .129); lower corner (.236, .119).
        env = DriveEnv.__new__(DriveEnv)
        env.x = env.y = env.yaw = 0.
        env.flat = SimpleNamespace(resolution=.02, origin=(-1.014, -1.011),
            height=101, width=101, free=free, cell=lambda x, y: (50, 50))
        env.world = SimpleNamespace(bodies=[])
        self.assertEqual(outline_gap(0., 0., 0., np.array([[.236, .119]]))[0], 0.)
        centre_gap = outline_gap(0., 0., 0., np.array([[.246, .129]]))[0]
        self.assertGreater(centre_gap, .01, 'half-side radius misses this occupied corner')
        self.assertLessEqual(env._outline_gap(), 0.)


class DyingIsNeverCheapTests(unittest.TestCase):
    """A collision must never be the economical way out of a long drive.

    The episode ends on a collision, and ending the episode stops the per-step
    cost.  On this flat a long drive is budgeted at over a thousand steps, so
    the rent it saves by dying was larger than the penalty it paid -- and the
    policy found that out before we did: through 150 updates the reward climbed
    while arrivals fell, which is what optimising an exploit looks like from
    the outside.
    """

    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def test_the_penalty_outweighs_the_rent_it_saves(self):
        env = DriveEnv(self.flat, np.random.default_rng(12), scenario="clear")
        worst = 0.0
        for _ in range(30):
            env.reset()
            worst = max(worst, STEP_COST * env.max_steps)
        self.assertGreater(
            COLLISION_PENALTY, worst,
            "on the longest drive this map offers, crashing saves more in "
            "step cost than it pays in penalty")

    def test_a_collision_is_charged_the_rest_of_the_episode(self):
        """Stated against the reward itself, not against the constants."""
        env = DriveEnv(self.flat, np.random.default_rng(13), scenario="clear")
        env.reset()
        env.world.bodies = [make_creature("person", env.x + 0.05, env.y,
                                          0.0, 0.0)]
        remaining = env.max_steps - env.steps
        _obs, reward, done, outcome = env.step(
            np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(done)
        self.assertEqual(outcome, "collision")
        self.assertLessEqual(
            reward, -(COLLISION_PENALTY + STEP_COST * (remaining - 1)),
            "the collision was not charged for the time it did not serve")


class BodySizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def test_the_eroded_map_admits_the_whole_chassis(self):
        """Every sampled centre must fit the full swept radius."""
        rows, columns = np.nonzero(self.flat.standable)
        self.assertTrue(rows.size, "nothing on this map is standable")
        worst = float(self.flat.blocked_distance[rows, columns].min())
        self.assertGreaterEqual(
            worst, BODY_RADIUS_M,
            f"a standable cell has only {worst:.3f} m of clearance")

    def test_no_episode_begins_inside_something(self):
        for scenario in SCENARIOS:
            env = DriveEnv(self.flat, np.random.default_rng(5),
                           scenario=scenario)
            for _ in range(30):
                env.reset()
                self.assertGreaterEqual(
                    env.gap, BODY_RADIUS_M,
                    f"{scenario} started the robot inside an obstacle")


class ReachTheWholeFlatTests(unittest.TestCase):
    """The plan's margin must not cost a room.

    The margin is a real trade.  Too little and the route runs along the
    inscribed limit the whole way, leaving a controller 2.5 cm to absorb every
    error -- the reference driver arrived a quarter of the time on an empty
    flat.  Too much and the doorway into the bottom-left room stops admitting
    the robot at all, and the simulator silently becomes a simulator of a
    smaller flat that happens to be easy to drive.
    """

    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def test_the_plan_can_still_reach_what_the_body_can_reach(self):
        from scipy import ndimage
        body = self.flat.blocked_distance >= BODY_RADIUS_M
        plan = self.flat.blocked_distance >= self.flat.clearance
        reach = []
        for mask in (body, plan):
            labels, count = ndimage.label(mask)
            sizes = ndimage.sum(mask, labels, range(1, count + 1))
            reach.append(float(sizes.max()) * self.flat.resolution ** 2)
        body_area, plan_area = reach
        self.assertGreater(
            plan_area / body_area, 0.80,
            f"planning at {self.flat.clearance:.2f} m reaches only "
            f"{plan_area:.1f} m2 of the {body_area:.1f} m2 the chassis fits "
            "in -- a room has been sealed off")

    def test_the_margin_is_wider_than_a_single_cell(self):
        """Otherwise one step of tracking error is the whole budget."""
        # Against the half width, which is what the robot presents to a gap
        # it drives through. Comparing with the spin radius asks whether it
        # could pirouette in the doorway, which is a different question and
        # one no planned route needs to answer.
        self.assertGreaterEqual(
            self.flat.clearance - BODY_HALF_WIDTH_M, self.flat.resolution,
            "the plan leaves the controller less than one cell to work in")


class EpisodeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def test_a_long_drive_is_given_time_to_finish(self):
        """400 steps is ten metres of travel, and this flat is longer."""
        env = DriveEnv(self.flat, np.random.default_rng(8), scenario="clear")
        generous = 0
        for _ in range(40):
            env.reset()
            needed = env.remaining / (MAX_LINEAR * 0.1)
            self.assertGreaterEqual(env.max_steps, needed)
            generous += int(env.max_steps > 400)
        self.assertGreater(generous, 0,
                           "no drive on this map ever needed more than the "
                           "old fixed budget, which cannot be right")

    def test_a_fixed_budget_is_still_honoured_when_asked_for(self):
        env = DriveEnv(self.flat, np.random.default_rng(9), max_steps=250)
        env.reset()
        self.assertEqual(env.max_steps, 250)

    def _budget(self, yaw):
        env = DriveEnv(self.flat, np.random.default_rng(3))
        env.reset()
        env.yaw = yaw
        # Far enough that MIN_STEPS does not absorb the difference: the floor
        # is 400 steps and a two-metre hop sits under it either way.
        env.goal = (env.x + 6.0, env.y)
        env.remaining = 6.0
        bearing = math.atan2(env.goal[1] - env.y, env.goal[0] - env.x)
        turn = abs(wrap(bearing - env.yaw))
        seconds = env.remaining / MAX_LINEAR + turn / MAX_ANGULAR
        return max(env.MIN_STEPS,
                   int(math.ceil(env.STEP_ALLOWANCE * seconds / DT)))

    def test_facing_away_buys_more_time_than_facing_the_goal(self):
        """The clock has to pay for turning, not only for driving.

        It did not, and cutting MAX_ANGULAR to what the chassis can actually
        do turned that into four timeouts in ten across the static scenarios:
        the quarter turn every episode opens with went from 2.2 seconds to
        6.3, taken out of a budget that had not moved.
        """
        self.assertGreater(self._budget(math.pi), self._budget(0.0))

    def test_the_turn_is_worth_what_the_turn_rate_says(self):
        extra = self._budget(math.pi) - self._budget(0.0)
        expected = DriveEnv.STEP_ALLOWANCE * (math.pi / MAX_ANGULAR) / DT
        self.assertAlmostEqual(extra, expected, delta=2)


class ScanReplanTests(unittest.TestCase):
    def test_measured_return_changes_route_then_expires_without_truth_or_double_history(self):
        import collections
        from types import SimpleNamespace
        from scipy.ndimage import distance_transform_edt
        from rl_env import LASER_YAW_RAD, BEAMS, core
        free = np.ones((61, 61), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        field = SimpleNamespace(free=free, field=distance_transform_edt(free)*.05,
            resolution=.05, origin=(-1.5, -1.5), height=61, width=61)
        arena = FastFlat.__new__(FastFlat)
        arena.field, arena.free, arena.blocked_distance = field, free, field.field
        arena.resolution, arena.origin, arena.height, arena.width = .05, field.origin, 61, 61
        arena.clearance = .25
        scan = np.full(BEAMS, np.inf, dtype=np.float32)
        arena.raycast = lambda *args: scan.copy()
        class ObservedOnly:
            def fold_into_scan(self, x, y, heading, ranges, start):
                return ranges
            @property
            def bodies(self):
                raise AssertionError('planner read hidden actor coordinates')
        env = DriveEnv.__new__(DriveEnv)
        env.flat, env.world = arena, ObservedOnly()
        env.x = env.y = env.yaw = env.linear = env.angular = 0.
        env.steps, env.replans, env.no_route = 0, 0, False
        env.goal, env.encoding, env.numpy = (1.1, 0.), None, np
        env.obstacles = core.LocalMap(env)
        env._views = collections.deque(maxlen=22)
        env.route = core.Wavefront(field, clearance=.25)
        self.assertTrue(env.route.plan(env.goal))
        before = env._observe()
        self.assertEqual(len(env._views), 1)
        self.assertEqual(env.obstacles.points, [])
        scan[round(-LASER_YAW_RAD*BEAMS/(2*math.pi)) % BEAMS] = .6
        env.steps = 10
        after = env._observe()
        self.assertEqual(len(env._views), 2)
        self.assertEqual(len(env.obstacles.points), 1)
        cell = arena.cell(*env.obstacles.points[0]["xy"])
        self.assertFalse(env.route.free[cell])
        self.assertTrue(arena.free[cell], 'temporary observation changed the saved map')
        self.assertFalse(np.allclose(before[216:226], after[216:226]))
        self.assertEqual(env.route_cost, env.route.cost_at((0., 0.)))
        scan[:] = np.inf
        env.steps = 260  # Exactly25seconds: retain a disappeared return until TTL.
        env._observe()
        self.assertEqual(len(env.obstacles.points), 1)
        env.steps = 270
        env._observe()
        self.assertEqual(env.obstacles.points, [])
        self.assertTrue(env.route.free[cell])
        self.assertEqual(env.replans, 2)


class ThePastIsVisibleTests(unittest.TestCase):
    """The observation has to say whether anything is moving.

    Measured 2026-08-31 on identical placements: freezing the people in the
    `walking` scenario took the policy from 0.425 arrivals to 0.787 and its
    collisions from 0.550 to 0.100.  The entire deficit was motion, and the
    single-frame observation gave the network no way to see any -- it met a
    person walking into its path with the same answer it gave a wardrobe.

    These pin the signal itself rather than a score, because a score can be
    recovered by luck and this cannot: a robot holding still must see nothing
    change around a stationary person, and must see something change around a
    walking one.
    """

    @classmethod
    def setUpClass(cls):
        cls.flat = flat()

    def _views_after(self, scenario, steps=25):
        still = np.array([0.0, 0.0], dtype=np.float32)
        for seed in range(11, 111):
            env = DriveEnv(self.flat, np.random.default_rng(seed),
                           scenario=scenario)
            observation = env.reset()
            for _ in range(steps):
                observation, _reward, done, _outcome = env.step(still)
                if done:
                    break
            if not done:
                return [observation[i * LIDAR_BINS:(i + 1) * LIDAR_BINS]
                        for i in range(SCAN_VIEWS)]
        self.fail("no deterministic fixture survived until history filled")

    def test_a_standing_person_leaves_every_view_identical(self):
        views = self._views_after("standing")
        for older in views[1:]:
            np.testing.assert_allclose(views[0], older, rtol=0, atol=0)

    def test_a_walking_person_moves_between_the_views(self):
        views = self._views_after("walking")
        moved = max(float(np.abs(views[0] - older).max()) for older in views[1:])
        self.assertGreater(moved, 0.01,
                           "a walking person left no trace between the scan "
                           "views, so the policy cannot tell it from furniture")


if __name__ == "__main__":
    unittest.main()
