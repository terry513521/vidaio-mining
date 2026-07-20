#!/usr/bin/env python3
"""Unit tests for feature-driven / survey VBR preprocess selection."""

from __future__ import annotations

import unittest

from encoder import SURVEY_PREPROCESS_SWEEP, _PREPROCESS_FILTERS
from recipes import (
    PreprocessScoreView,
    choose_best_preprocess,
    choose_preprocess_ab_winner,
    propose_preprocess_from_features,
    resolve_vbr_preprocess,
    survey_preprocess_candidates,
)
from request import CompressionRequest


class ProposePreprocessTests(unittest.TestCase):
    def test_average_noise_is_none(self) -> None:
        # Soft-norm ~0.5 is corpus-average; light threshold is 0.52.
        preset, reason = propose_preprocess_from_features({"noise_level_norm": 0.50})
        self.assertEqual(preset, "none")
        self.assertIn("none", reason)

    def test_mid_noise_is_hqdn3d_light(self) -> None:
        preset, reason = propose_preprocess_from_features({"noise_level_norm": 0.53})
        self.assertEqual(preset, "hqdn3d_light")
        self.assertIn("hqdn3d_light", reason)

    def test_high_noise_textured_prefers_bilateral(self) -> None:
        preset, _ = propose_preprocess_from_features(
            {
                "noise_level_norm": 0.60,
                "motion_level": 0.3,
                "texture_level": 0.6,
            }
        )
        self.assertEqual(preset, "bilateral_light")

    def test_high_noise_temporal_prefers_atadenoise(self) -> None:
        preset, reason = propose_preprocess_from_features(
            {
                "noise_level_norm": 0.60,
                "motion_level": 0.70,
                "texture_level": 0.30,
            }
        )
        self.assertEqual(preset, "atadenoise_light")
        self.assertIn("atadenoise_light", reason)

    def test_clean_edge_rich_prefers_unsharp(self) -> None:
        preset, reason = propose_preprocess_from_features(
            {
                "noise_level_norm": 0.35,
                "edge_level": 0.70,
                "flatness": 0.2,
            }
        )
        self.assertEqual(preset, "unsharp_mild")
        self.assertIn("unsharp_mild", reason)

    def test_clean_flat_prefers_contrast(self) -> None:
        preset, reason = propose_preprocess_from_features(
            {
                "noise_level_norm": 0.35,
                "edge_level": 0.2,
                "flatness": 0.60,
            }
        )
        self.assertEqual(preset, "contrast_mild")
        self.assertIn("contrast_mild", reason)

    def test_empty_features_is_none(self) -> None:
        preset, _ = propose_preprocess_from_features({})
        self.assertEqual(preset, "none")


class SurveySweepTests(unittest.TestCase):
    def test_survey_presets_exist(self) -> None:
        for name in SURVEY_PREPROCESS_SWEEP:
            self.assertIn(name, _PREPROCESS_FILTERS)

    def test_brave_presets_exist(self) -> None:
        from encoder import BRAVE_PREPROCESS_SWEEP

        for name in BRAVE_PREPROCESS_SWEEP:
            self.assertIn(name, _PREPROCESS_FILTERS)

    def test_sweep_candidates_include_survey_set(self) -> None:
        cands, reason = survey_preprocess_candidates({}, sweep=True)
        labels = {c or "none" for c in cands}
        for name in SURVEY_PREPROCESS_SWEEP:
            self.assertIn(name, labels)
        self.assertIn("sweep", reason)

    def test_brave_candidates_include_micro_set(self) -> None:
        from encoder import BRAVE_PREPROCESS_SWEEP

        cands, reason = survey_preprocess_candidates({}, brave=True)
        labels = {c or "none" for c in cands}
        for name in BRAVE_PREPROCESS_SWEEP:
            self.assertIn(name, labels)
        self.assertIn("brave", reason)

    def test_auto_candidates_are_none_plus_pick(self) -> None:
        cands, _ = survey_preprocess_candidates(
            {"noise_level_norm": 0.53}, sweep=False
        )
        self.assertEqual(cands[0], None)
        self.assertEqual(cands[-1], "hqdn3d_light")


class ResolveVbrPreprocessTests(unittest.TestCase):
    def test_explicit_wins_over_auto(self) -> None:
        preset, reason, cands = resolve_vbr_preprocess(
            explicit="unsharp_mild",
            preprocess_auto=True,
            features={"noise_level_norm": 0.1},
        )
        self.assertEqual(preset, "unsharp_mild")
        self.assertEqual(cands, ["unsharp_mild"])
        self.assertIn("explicit", reason)

    def test_explicit_none(self) -> None:
        preset, _, cands = resolve_vbr_preprocess(
            explicit="none",
            preprocess_auto=True,
            features={"noise_level_norm": 0.9},
        )
        self.assertIsNone(preset)
        self.assertEqual(cands, [None])

    def test_auto_disabled(self) -> None:
        preset, reason, cands = resolve_vbr_preprocess(
            explicit=None,
            preprocess_auto=False,
            features={"noise_level_norm": 0.9},
        )
        self.assertIsNone(preset)
        self.assertEqual(cands, [None])
        self.assertIn("preprocess_auto=false", reason)

    def test_sweep_returns_many_candidates(self) -> None:
        _, _, cands = resolve_vbr_preprocess(
            explicit=None,
            preprocess_auto=True,
            features={"noise_level_norm": 0.50},
            preprocess_sweep=True,
        )
        self.assertGreaterEqual(len(cands), len(SURVEY_PREPROCESS_SWEEP))

    def test_explicit_brave(self) -> None:
        from encoder import BRAVE_PREPROCESS_SWEEP

        _, reason, cands = resolve_vbr_preprocess(
            explicit="brave",
            preprocess_auto=False,
            features={"noise_level_norm": 0.50},
        )
        self.assertIn("brave", reason)
        self.assertGreaterEqual(len(cands), len(BRAVE_PREPROCESS_SWEEP))


class ChooseBestPreprocessTests(unittest.TestCase):
    def test_keeps_highest_s_f_with_gates(self) -> None:
        winner, reason = choose_best_preprocess(
            [
                (None, PreprocessScoreView(0.40, 86.0, True)),
                ("hqdn3d_light", PreprocessScoreView(0.45, 85.5, True)),
                ("unsharp_mild", PreprocessScoreView(0.60, 90.0, False)),
                ("contrast_mild", PreprocessScoreView(0.42, 87.0, True)),
            ]
        )
        self.assertEqual(winner, "hqdn3d_light")
        self.assertIn("hqdn3d_light", reason)

    def test_ab_wrapper_still_works(self) -> None:
        winner, _ = choose_preprocess_ab_winner(
            none_score=PreprocessScoreView(0.50, 86.0, True),
            denoise_score=PreprocessScoreView(0.40, 87.0, True),
            denoise_preset="bilateral_light",
        )
        self.assertIsNone(winner)


class RequestPreprocessFlagsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        req = CompressionRequest(input_path="x.mp4")
        self.assertTrue(req.preprocess_auto)
        self.assertTrue(req.preprocess_ab)
        self.assertFalse(req.preprocess_sweep)
        self.assertIsNone(req.preprocess)

    def test_new_presets_validate(self) -> None:
        for name in (
            "bilateral_light",
            "unsharp_mild",
            "contrast_mild",
            "unsharp_micro",
            "cas_micro",
            "denoise_unsharp",
        ):
            req = CompressionRequest(input_path="x.mp4", preprocess=name)
            self.assertEqual(req.preprocess, name)


if __name__ == "__main__":
    unittest.main()
