"""Unit tests for feature → libx265 non-CRF params."""

from __future__ import annotations

import unittest

from interp_search import format_x265_params, propose_feature_x265_params
from recipes import DEFAULT_MEDIUM, describe_feature_x265_baseline, select_recipes
from request import CompressionRequest


def _feats(**overrides: float) -> dict[str, float]:
    base = {
        "motion_level": 0.50,
        "texture_level": 0.50,
        "edge_level": 0.50,
        "noise_level_norm": 0.45,
        "cut_level": 0.20,
        "worst_difficulty": 0.40,
        "hard_fraction": 0.30,
        "volatility": 0.30,
        "segment_count": 6.0,
        "duration": 30.0,
        "fps": 30.0,
    }
    base.update(overrides)
    return base


class FeatureX265ParamsTests(unittest.TestCase):
    def test_high_noise_lower_aq_than_high_texture(self) -> None:
        noisy, _ = propose_feature_x265_params(
            _feats(noise_level_norm=0.60, texture_level=0.40, edge_level=0.40)
        )
        textured, _ = propose_feature_x265_params(
            _feats(
                noise_level_norm=0.40,
                texture_level=0.89,
                edge_level=0.68,
                worst_difficulty=0.60,
                hard_fraction=1.0,
            )
        )
        self.assertLess(float(noisy["aq-strength"]), float(textured["aq-strength"]))
        self.assertEqual(noisy["aq-mode"], 1)
        self.assertEqual(textured["aq-mode"], 1)

    def test_aq_mode_from_fleet_like_features(self) -> None:
        v1_like, r1 = propose_feature_x265_params(
            _feats(
                motion_level=0.73,
                texture_level=0.89,
                edge_level=0.68,
                worst_difficulty=0.60,
                hard_fraction=1.0,
                volatility=0.2,
            )
        )
        v4_like, r2 = propose_feature_x265_params(
            _feats(
                motion_level=0.69,
                texture_level=0.85,
                edge_level=0.50,
                worst_difficulty=0.53,
                hard_fraction=1.0,
                volatility=0.07,
            )
        )
        self.assertEqual(v1_like["aq-mode"], 1)
        self.assertTrue(any("aq-mode=1" in r for r in r1))
        self.assertEqual(v4_like["aq-mode"], 2)
        self.assertTrue(any("aq-mode=2" in r for r in r2))

    def test_fleet_like_aq_strength_additive(self) -> None:
        v1_like, _ = propose_feature_x265_params(
            _feats(
                motion_level=0.73,
                texture_level=0.89,
                edge_level=0.68,
                noise_level_norm=0.23,
                worst_difficulty=0.60,
                hard_fraction=1.0,
                volatility=0.2,
            )
        )
        v4_like, _ = propose_feature_x265_params(
            _feats(
                motion_level=0.69,
                texture_level=0.85,
                edge_level=0.50,
                noise_level_norm=0.14,
                worst_difficulty=0.53,
                hard_fraction=1.0,
                volatility=0.07,
            )
        )
        self.assertGreaterEqual(float(v1_like["aq-strength"]), float(v4_like["aq-strength"]))
        self.assertEqual(float(v1_like["aq-strength"]), 1.4)
        self.assertGreaterEqual(float(v4_like["aq-strength"]), 1.2)

    def test_flat_regions_reduce_aq_strength(self) -> None:
        textured, _ = propose_feature_x265_params(
            _feats(texture_level=0.55, edge_level=0.50, flatness=0.20)
        )
        flat, _ = propose_feature_x265_params(
            _feats(texture_level=0.55, edge_level=0.50, flatness=0.72)
        )
        self.assertGreater(float(textured["aq-strength"]), float(flat["aq-strength"]))

    def test_high_motion_more_bframes_and_lookahead(self) -> None:
        high, _ = propose_feature_x265_params(_feats(motion_level=0.55))
        low, _ = propose_feature_x265_params(
            _feats(motion_level=0.40, noise_level_norm=0.40)
        )
        self.assertGreater(int(high["bframes"]), int(low["bframes"]))
        self.assertGreater(int(high["rc-lookahead"]), int(low["rc-lookahead"]))

    def test_never_sets_crf(self) -> None:
        params, _ = propose_feature_x265_params(_feats())
        self.assertNotIn("crf", {k.lower() for k in params})
        s = format_x265_params(params)
        self.assertNotIn("crf=", s.lower())

    def test_format_rejects_crf_key(self) -> None:
        with self.assertRaises(ValueError):
            format_x265_params({"aq-mode": 3, "crf": 28})

    def test_keyint_from_segments(self) -> None:
        params, reasons = propose_feature_x265_params(
            _feats(segment_count=6.0, duration=30.0, fps=30.0)
        )
        # avg seg 5s → 150 frames, clamped to [30, 60]
        self.assertEqual(params["keyint"], 60)
        self.assertTrue(any("keyint" in r for r in reasons))

    def test_select_recipes_uses_feature_params(self) -> None:
        recipes = select_recipes(
            _feats(noise_level_norm=0.60),
            85,
            max_recipes=1,
            preset="fast",
            feature_baseline=True,
        )
        self.assertEqual(len(recipes), 1)
        self.assertIn("aq-strength=0.8", recipes[0].params)
        self.assertNotIn("crf=", recipes[0].params.lower())

    def test_flag_off_keeps_static_defaults(self) -> None:
        recipes = select_recipes(
            _feats(noise_level_norm=0.60, motion_level=0.70),
            85,
            max_recipes=1,
            preset="medium",
            feature_baseline=False,
        )
        self.assertEqual(recipes[0].params, DEFAULT_MEDIUM.params)

    def test_request_flag_default_true(self) -> None:
        req = CompressionRequest(input_path="x.mp4", encoder="libx265", preset="medium")
        self.assertTrue(req.libx265_feature_baseline)

    def test_describe_includes_params_line(self) -> None:
        lines = describe_feature_x265_baseline(_feats())
        self.assertTrue(any(line.startswith("x265-params=") for line in lines))
        joined = " ".join(lines).lower()
        self.assertNotIn("crf=", joined)


if __name__ == "__main__":
    unittest.main()
