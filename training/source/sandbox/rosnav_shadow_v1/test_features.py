import unittest

import numpy as np

from features import (FEATURE_DIM, LIDAR_BINS, PATH_POINTS, SCAN_VIEWS,
                      TemporalScanBuffer, build_features,
                      encode_ranges, mirror_features, mirror_labels,
                      pool_lidar, sample_path_xy)
from symmetrize import mirror_plan


class FeatureTests(unittest.TestCase):
    def test_lidar_min_pool_is_conservative(self):
        """Sectors are placed by body-frame angle, not by array position.

        With two sectors the first is centred on forward and spans the front
        half, so the beams at 0 and 270 degrees share it and the beams at 90
        and 180 share the rear sector."""
        result = pool_lidar([1.0, 2.0, 3.0, 4.0], 0.05, 4.0, bins=2)
        # Through the encoding rather than as literals: which beam wins its
        # sector is what this test is about, and writing `[0.25, 0.5]` made it
        # also assert the range scale, so it failed when only the scale moved.
        np.testing.assert_allclose(
            result, encode_ranges(np.float32([1.0, 2.0]), 0.05, 4.0), rtol=1e-6)

    def test_invalid_lidar_values_are_bounded(self):
        result = pool_lidar([np.nan, np.inf, -np.inf, 2.0], 0.1, 4.0, bins=2)
        # A missing return reads as the horizon, a negative one as touching.
        np.testing.assert_allclose(
            result, encode_ranges(np.float32([2.0, 0.1]), 0.1, 4.0), rtol=1e-6)

    def test_path_sampling_follows_arc_length(self):
        result = sample_path_xy([(0.0, 0.0), (1.0, 0.0)], count=5).reshape(-1, 2)
        np.testing.assert_allclose(result[:, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(result[:, 1], 0.0)

    def test_near_turns_do_not_disappear_before_the_first_path_sample(self):
        left = [(0., 0.), (.25, .2), (.5, 0.), (5., 0.)]
        right = [(x, -y) for x, y in left]
        np.testing.assert_allclose(sample_path_xy(left, lookahead_m=None),
                                   sample_path_xy(right, lookahead_m=None))
        near_left = sample_path_xy(left).reshape(-1, 2)
        near_right = sample_path_xy(right).reshape(-1, 2)
        self.assertGreater(near_left[0, 1], .05)
        np.testing.assert_allclose(near_right, near_left * [1., -1.])
        np.testing.assert_allclose(sample_path_xy([(0.,0.), (8.,0.)]),
                                   [.1,0.,.2,0.,.3,0.,.4,0.,.5,0.], atol=1e-7)

    def test_complete_feature_shape(self):
        result = build_features(
            np.linspace(0.1, 6.0, 360), 0.1, 6.0, [(0.0, 0.0), (1.0, 0.2)], 0.1, 0.2
        )
        self.assertEqual(result.shape, (FEATURE_DIM,))
        self.assertEqual(result.dtype, np.float32)

    def test_pursuit_reads_path_after_all_temporal_scan_views(self):
        from pure_pursuit import path_from_features
        path = [(0.0, 0.0), (1.0, 0.2)]
        features = build_features([2.0] * 360, .15, 12.0, path, .1, .2)
        expected = sample_path_xy(path).reshape(PATH_POINTS, 2) * 2.0
        np.testing.assert_allclose(path_from_features(features), expected)
        np.testing.assert_allclose(path_from_features(features[None])[0], expected)

    def test_pursuit_defaults_are_complete_and_respect_chassis_turn_limit(self):
        from pure_pursuit import DEFAULTS, predict
        features = build_features([2.0] * 360, .15, 12.0, [(0.0, 0.0), (0.0, 1.0)], 0.0, 0.0)
        action = predict(features, **DEFAULTS)
        self.assertEqual(float(action[0]), 0.0)
        self.assertLessEqual(abs(float(action[1])), .25)


class MirrorTests(unittest.TestCase):
    def _sample(self, lateral: float, angular: float) -> np.ndarray:
        ranges = np.linspace(0.4, 6.0, 360, dtype=np.float32)
        return build_features(
            ranges, 0.15, 12.0,
            [[0.4, lateral], [0.8, 2 * lateral], [1.2, 3 * lateral]],
            0.18, angular)

    def test_mirroring_twice_returns_the_original(self):
        original = self._sample(0.25, 0.4)
        restored = mirror_features(mirror_features(original))
        np.testing.assert_allclose(restored, original, rtol=0, atol=0)
        labels = np.asarray([0.2, -0.35], dtype=np.float32)
        np.testing.assert_allclose(mirror_labels(mirror_labels(labels)), labels)

    def test_exported_mirror_plan_matches_the_canonical_transform(self):
        order, sign = mirror_plan()
        probes = np.random.default_rng(20260831).normal(
            size=(7, FEATURE_DIM)).astype(np.float32)
        planned = probes[:, order] * sign
        np.testing.assert_array_equal(planned, mirror_features(probes))

    def test_exported_mirror_plan_is_an_exact_involution(self):
        order, sign = mirror_plan()
        np.testing.assert_array_equal(order[order], np.arange(FEATURE_DIM))
        np.testing.assert_array_equal(sign * sign[order],
                                      np.ones(FEATURE_DIM, dtype=np.float32))

    def test_mirror_flips_turn_and_lateral_offset_but_not_forward(self):
        features = self._sample(0.25, 0.4)
        mirrored = mirror_features(features)
        scans = LIDAR_BINS * SCAN_VIEWS
        path = features[scans:scans + PATH_POINTS * 2]
        path_mirrored = mirrored[scans:scans + PATH_POINTS * 2]
        np.testing.assert_allclose(path_mirrored[0::2], path[0::2])   # forward
        np.testing.assert_allclose(path_mirrored[1::2], -path[1::2])  # lateral
        self.assertAlmostEqual(float(mirrored[-1]), -float(features[-1]))
        self.assertAlmostEqual(float(mirrored[-2]), float(features[-2]))
        np.testing.assert_allclose(mirror_labels(np.asarray([0.2, 0.5], np.float32)),
                                   np.asarray([0.2, -0.5], np.float32))

    def test_a_symmetric_scene_on_a_straight_path_maps_to_itself(self):
        """The fixed point of the mirror is exactly the case the policy got
        wrong: symmetric surroundings, dead-ahead goal, no rotation.  If this
        does not hold, augmentation cannot cancel a turn bias.

        The scene has to be symmetric about *forward*, which is the axis the
        mirror reflects in.  The first version of this test built one that was
        symmetric about a line half a beam off forward and then fed it through
        the 2.7 degree mounting yaw, so it was never a fixed point at all --
        it merely came within 0.02 of one, which under a linear range scale
        was small enough to look like rounding.  Under a logarithmic scale the
        same geometric offset reads five times larger, and the test failed for
        a reason that had nothing to do with the mirror.

        So: an exactly symmetric scene, `laser_yaw=0`, and a tolerance at
        float precision.  What the mounting yaw does to scene symmetry is a
        fact about the sensor, not about this code.
        """
        half = np.linspace(0.5, 5.0, 181, dtype=np.float32)
        ranges = np.concatenate([half, half[-2:0:-1]])
        self.assertEqual(ranges.size, 360)
        features = build_features(ranges, 0.15, 12.0,
                                  [[0.5, 0.0], [1.0, 0.0]], 0.2, 0.0,
                                  laser_yaw=0.0)
        mirrored = mirror_features(features)
        scans = LIDAR_BINS * SCAN_VIEWS
        lidar_gap = float(np.max(np.abs(mirrored[:scans] - features[:scans])))
        self.assertLess(lidar_gap, 1e-6)
        np.testing.assert_allclose(mirrored[scans:], features[scans:],
                                   atol=1e-7)


class RangeEncodingTests(unittest.TestCase):
    """The default is a promise to the board, not a preference.

    Flipping this constant on its own would make every parity fixture describe
    a pipeline nobody runs, and nothing would fail: the Rust tests would keep
    passing against their own frozen copies while the host quietly computed
    something else.

    It has been flipped once, to `log`, and the flip was six files moving
    together -- `features.py`, `features.rs`, `weights.rs`, both parity
    fixtures and the ARM binary. This literal is the reminder that the next
    flip costs the same; `test_range_encoding_agreement.py` is what actually
    checks the six agree, and `features.rs` refuses to compile if the weights
    disagree with it.
    """

    def test_the_default_is_what_the_board_implements(self):
        from features import RANGE_ENCODING
        self.assertEqual(
            RANGE_ENCODING, "log",
            "changing this default requires features.rs, both parity "
            "fixtures, the weights and the ARM binary to move with it")

    def test_log_expands_the_band_where_driving_is_decided(self):
        import math
        from features import encode_ranges
        import numpy as np
        near = np.array([0.15, 0.35], dtype=np.float32)
        linear = encode_ranges(near, 0.15, 12.0, "linear")
        log = encode_ranges(near, 0.15, 12.0, "log")
        self.assertLess(float(linear[1] - linear[0]), 0.05)
        self.assertGreater(float(log[1] - log[0]), 0.15)

    def test_both_encodings_are_monotone_and_bounded(self):
        from features import encode_ranges
        import numpy as np
        metres = np.linspace(0.15, 12.0, 200, dtype=np.float32)
        for encoding in ("linear", "log"):
            out = encode_ranges(metres, 0.15, 12.0, encoding)
            self.assertTrue(np.all(np.diff(out) >= -1e-6), encoding)
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_an_unknown_encoding_is_refused(self):
        from features import encode_ranges
        import numpy as np
        with self.assertRaises(ValueError):
            encode_ranges(np.array([1.0], dtype=np.float32), 0.15, 12.0, "sqrt")


class TemporalHistoryTests(unittest.TestCase):
    @staticmethod
    def view(value):
        return np.full(LIDAR_BINS, value, dtype=np.float32)

    def test_jittered_five_hertz_stream_selects_only_causal_views(self):
        history = TemporalScanBuffer()
        selected = None
        stamps = (0.0, .19, .39, .58, .78, .97, 1.17, 1.36,
                  1.56, 1.75, 1.95, 2.14)
        for stamp in stamps:
            selected = history.append(stamp, self.view(stamp))
        self.assertTrue(history.ready)
        self.assertIsNotNone(selected)
        self.assertLessEqual(float(selected[0][0]), stamps[-1] - .3)
        self.assertLessEqual(float(selected[1][0]), stamps[-1] - 2.0)
        self.assertGreater(float(selected[0][0]), 1.0)

    def test_warmup_withholds_history_until_two_seconds_exist(self):
        history = TemporalScanBuffer()
        for stamp in (0.0, .2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8):
            self.assertIsNone(history.append(stamp, self.view(stamp)))
        self.assertIsNotNone(history.append(2.0, self.view(2.0)))

    def test_gap_and_timestamp_regression_reset_the_generation(self):
        history = TemporalScanBuffer()
        for stamp in np.arange(0.0, 2.01, .2):
            selected = history.append(float(stamp), self.view(stamp))
        self.assertIsNotNone(selected)
        self.assertIsNone(history.append(3.0, self.view(3.0)))
        self.assertEqual(history.last_reset_reason, "receive_gap_too_large")
        self.assertIsNone(history.append(2.9, self.view(2.9)))
        self.assertEqual(history.last_reset_reason, "timestamp_not_increasing")


if __name__ == "__main__":
    unittest.main()
