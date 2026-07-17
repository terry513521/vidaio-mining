"""Tests for soft feature normalization."""

from __future__ import annotations

import unittest

from feature_extractor import (
    _EDGE_DENSITY_MID,
    _HF_ENERGY_MID,
    _MOTION_P90_MID,
    _NOISE_LEVEL_MID,
    _TEXTURE_MID,
    soft_level,
)
from interp_search import _feature_levels, propose_feature_nvenc_baseline


class SoftLevelTests(unittest.TestCase):
    def test_midpoint_maps_to_half(self) -> None:
        self.assertAlmostEqual(soft_level(_MOTION_P90_MID, _MOTION_P90_MID), 0.5)
        self.assertAlmostEqual(soft_level(_NOISE_LEVEL_MID, _NOISE_LEVEL_MID), 0.5)
        self.assertAlmostEqual(soft_level(_TEXTURE_MID, _TEXTURE_MID), 0.5)

    def test_asymptote_below_one(self) -> None:
        self.assertLess(soft_level(100.0, _MOTION_P90_MID), 1.0)
        self.assertGreater(soft_level(100.0, _MOTION_P90_MID), 0.99)

    def test_known_clip_not_saturated(self) -> None:
        # Values from the user's terminal run / calibration demo.
        motion = soft_level(0.1343, _MOTION_P90_MID)
        noise = soft_level(0.0439, _NOISE_LEVEL_MID)
        texture = soft_level(4.248, _TEXTURE_MID)
        edge = soft_level(0.0634, _EDGE_DENSITY_MID)
        hf = soft_level(4.759, _HF_ENERGY_MID)
        for name, val in (
            ("motion", motion),
            ("noise", noise),
            ("texture", texture),
            ("edge", edge),
            ("hf", hf),
        ):
            self.assertLess(val, 0.99, msg=name)
            self.assertGreater(val, 0.20, msg=name)
        # Near corpus median → near 0.5
        self.assertAlmostEqual(motion, 0.48, delta=0.05)
        self.assertAlmostEqual(noise, 0.48, delta=0.05)


class BaselineDifferentiationTests(unittest.TestCase):
    def test_low_vs_high_noise_differs(self) -> None:
        low = {
            "motion_level": 0.48,
            "texture_level": 0.50,
            "noise_level_norm": 0.40,
            "edge_level": 0.40,
            "cut_level": 0.2,
            "fps": 30.0,
            "segment_count": 6.0,
            "duration": 30.0,
        }
        high = {**low, "noise_level_norm": 0.60}
        low_ov, _ = propose_feature_nvenc_baseline(low)
        high_ov, _ = propose_feature_nvenc_baseline(high)
        self.assertNotEqual(low_ov.get("nvenc_aq_strength"), high_ov.get("nvenc_aq_strength"))
        self.assertEqual(high_ov.get("nvenc_aq_strength"), 4)
        self.assertFalse(high_ov.get("nvenc_temporal_aq"))

    def test_legacy_fallback_uses_soft(self) -> None:
        lvl = _feature_levels({"motion_p90": 0.145, "noise_level": 0.048, "texture": 4.22})
        self.assertAlmostEqual(lvl["motion"], 0.5, places=2)
        self.assertAlmostEqual(lvl["noise"], 0.5, places=2)
        self.assertAlmostEqual(lvl["texture"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
