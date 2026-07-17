"""Unit tests for ab-av1-style CRF search."""

from __future__ import annotations

import unittest

from crf_search import (
    QualityConverter,
    encoded_percent_size,
    round_crf_for_encode,
    search_crf,
    vmaf_lerp_q,
)
from crf_search import CrfAttempt
from scene_samples import ab_av1_sample_count, resolve_sample_plan


class QualityConverterTests(unittest.TestCase):
    def test_q_round_trip(self) -> None:
        conv = QualityConverter(crf_increment=0.1)
        self.assertEqual(conv.q(33.5), 335)
        self.assertAlmostEqual(conv.crf(335), 33.5)

    def test_min_max_q(self) -> None:
        conv = QualityConverter(crf_increment=1.0)
        min_q, max_q = conv.min_max_q(22, 40)
        self.assertEqual(min_q, 22)
        self.assertEqual(max_q, 40)


class VmafLerpTests(unittest.TestCase):
    def test_interpolates_between_brackets(self) -> None:
        # worse: q=30 vmaf 87, better: q=28 vmaf 91, target 89
        q = vmaf_lerp_q(89.0, 30, 87.0, 28, 91.0)
        self.assertGreater(q, 28)
        self.assertLess(q, 30)


class SearchCrfTests(unittest.TestCase):
    def test_finds_crf_near_target(self) -> None:
        truth = {28: 92.0, 30: 89.5, 32: 86.0, 34: 82.0}

        def evaluate(crf: float) -> CrfAttempt:
            nearest = min(truth.keys(), key=lambda k: abs(k - crf))
            vmaf = truth[nearest]
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=vmaf,
                per_sample_vmaf=[vmaf],
                encode_percent=50.0,
                encode_ok=True,
            )

        result = search_crf(
            evaluate,
            min_vmaf=89.0,
            crf_min=26,
            crf_max=36,
            crf_increment=1.0,
            max_encoded_percent=80.0,
            max_runs=8,
        )
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.crf)
        assert result.crf is not None
        self.assertGreaterEqual(result.mean_vmaf, 89.0)
        self.assertLessEqual(round_crf_for_encode(result.crf), 31)

    def test_rejects_oversized_encode(self) -> None:
        def evaluate(crf: float) -> CrfAttempt:
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=92.0,
                per_sample_vmaf=[92.0],
                encode_percent=95.0,
                encode_ok=True,
            )

        result = search_crf(
            evaluate,
            min_vmaf=89.0,
            crf_min=30,
            crf_max=40,
            crf_increment=1.0,
            max_encoded_percent=80.0,
            max_runs=3,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "no_good_crf")

    def test_initial_crf_seeds_first_probe(self) -> None:
        probed: list[float] = []

        def evaluate(crf: float) -> CrfAttempt:
            probed.append(crf)
            # High enough VMAF + small size → accept immediately near seed.
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=89.05,
                per_sample_vmaf=[89.05],
                encode_percent=40.0,
                encode_ok=True,
            )

        result = search_crf(
            evaluate,
            min_vmaf=89.0,
            crf_min=20,
            crf_max=40,
            crf_increment=2.0,
            max_encoded_percent=80.0,
            initial_crf=34.0,
            max_runs=4,
        )
        self.assertTrue(result.ok)
        self.assertEqual(probed[0], 34.0)
        self.assertEqual(result.crf, 34.0)

    def test_near_threshold_steps_instead_of_lerp(self) -> None:
        """When VMAF is close to target, next CRF is ±1, not a long lerp jump."""
        probed: list[float] = []
        # Slightly above target at CRF 30 → should try 31 next (fine step).
        table = {30: 86.5, 31: 85.2, 32: 83.0, 40: 70.0}

        def evaluate(crf: float) -> CrfAttempt:
            probed.append(crf)
            key = int(round(crf))
            vmaf = table.get(key, 80.0)
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=vmaf,
                per_sample_vmaf=[vmaf],
                encode_percent=40.0,
                encode_ok=True,
            )

        result = search_crf(
            evaluate,
            min_vmaf=85.0,
            crf_min=22,
            crf_max=42,
            crf_increment=1.0,
            max_encoded_percent=80.0,
            initial_crf=30.0,
            near_vmaf_band=2.0,
            max_runs=6,
        )
        self.assertGreaterEqual(len(probed), 2)
        self.assertEqual(probed[0], 30.0)
        self.assertEqual(probed[1], 31.0)  # not a big jump toward max
        self.assertTrue(result.ok)


class SamplePlanTests(unittest.TestCase):
    def test_ab_av1_sample_count_from_duration(self) -> None:
        self.assertEqual(ab_av1_sample_count(3600, sample_every_sec=720), 5)
        self.assertEqual(ab_av1_sample_count(300, samples=3), 3)

    def test_encoded_percent_size(self) -> None:
        self.assertAlmostEqual(encoded_percent_size(1000, 400), 40.0)

    def test_resolve_sample_plan_caps_by_scenes(self) -> None:
        count, seconds, frames = resolve_sample_plan(
            600,
            scene_count=4,
            sample_seconds=3.0,
            fps=30.0,
            sample_every_sec=720,
            scene_max_samples=8,
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(seconds, 3.0)
        self.assertEqual(frames, 90)

    def test_resolve_sample_plan_respects_min_samples(self) -> None:
        count, seconds, frames = resolve_sample_plan(
            30,
            scene_count=6,
            sample_seconds=3.0,
            fps=30.0,
            sample_every_sec=720,
            min_samples=5,
            scene_max_samples=8,
        )
        self.assertEqual(count, 5)
        self.assertAlmostEqual(seconds, 3.0)
        self.assertEqual(frames, 90)


if __name__ == "__main__":
    unittest.main()
