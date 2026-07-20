"""Unit tests for CRF-mode pack + Phase B/C."""

from __future__ import annotations

import unittest

from crf_mode import (
    aq_strength_candidates,
    build_crf_mode_params,
    is_crf_mode_feasible,
    run_crf_mode_phase_bc,
    select_crf_mode_aq_mode,
    CrfModeTrial,
)
from param_tune import get_param


class TestCrfModeAqMode(unittest.TestCase):
    def test_aq_mode_1_when_clean_high_texture_few_cuts(self) -> None:
        mode, reason = select_crf_mode_aq_mode(
            {
                "texture_level": 0.85,
                "noise_level_norm": 0.10,
                "cut_count": 2,
            }
        )
        self.assertEqual(mode, 1)
        self.assertIn("aq-mode=1", reason)

    def test_aq_mode_2_otherwise(self) -> None:
        mode, _ = select_crf_mode_aq_mode(
            {
                "texture_level": 0.55,
                "noise_level_norm": 0.51,
                "cut_count": 5,
            }
        )
        self.assertEqual(mode, 2)


class TestCrfModePack(unittest.TestCase):
    def test_fixed_pack_keys(self) -> None:
        params, mode, _ = build_crf_mode_params(
            {"texture_level": 0.5, "noise_level_norm": 0.5, "cut_count": 5}
        )
        self.assertEqual(mode, 2)
        self.assertEqual(get_param(params, "ref"), "6")
        self.assertEqual(get_param(params, "rd"), "6")
        self.assertEqual(get_param(params, "bframes"), "12")
        self.assertEqual(get_param(params, "aq-strength"), "1")
        self.assertEqual(get_param(params, "rc-lookahead"), "50")
        self.assertEqual(get_param(params, "aq-mode"), "2")


class TestAqWalk(unittest.TestCase):
    def test_stronger_then_weaker_skips_baseline(self) -> None:
        vals = aq_strength_candidates(aq_min=0.2, aq_max=2.4, aq_step=0.2, baseline=1.0)
        self.assertEqual(vals[0], 1.2)
        self.assertEqual(vals[-1], 0.2)
        self.assertNotIn(1.0, vals)
        self.assertIn(2.4, vals)


class TestPhaseBC(unittest.TestCase):
    def test_accepts_better_aq_and_headroom_ladder(self) -> None:
        base_params, _, _ = build_crf_mode_params(
            {"texture_level": 0.5, "noise_level_norm": 0.5, "cut_count": 5}
        )
        calls: list[tuple[float, str]] = []

        def evaluate(crf: float, params: str) -> CrfModeTrial:
            calls.append((crf, params))
            aq = float(get_param(params, "aq-strength") or 1.0)
            la = int(float(get_param(params, "rc-lookahead") or 50))
            # Baseline-like: crf=30 aq=1 → s_f=0.50 vmaf=87
            # Better aq=1.2 at crf=30 → s_f=0.55 vmaf=88 (headroom → try +1/+2)
            # crf=31 aq=1.2 → s_f=0.60 vmaf=86
            # crf=32 aq=1.2 → s_f=0.58 vmaf=84 (no accept if thr=85 and C gate)
            vmaf = 87.0
            s_f = 0.50
            rate = 0.050
            if la == 40 and abs(crf - 31) < 1e-6 and abs(aq - 1.2) < 1e-6:
                vmaf, s_f, rate = 86.6, 0.61, 0.044
            elif abs(aq - 1.2) < 1e-6 and abs(crf - 30) < 1e-6:
                vmaf, s_f, rate = 88.0, 0.55, 0.048
            elif abs(aq - 1.2) < 1e-6 and abs(crf - 31) < 1e-6:
                vmaf, s_f, rate = 86.5, 0.60, 0.045
            elif abs(aq - 1.2) < 1e-6 and abs(crf - 32) < 1e-6:
                vmaf, s_f, rate = 84.0, 0.58, 0.042
            return CrfModeTrial(
                crf=crf,
                aq_strength=aq,
                rc_lookahead=la,
                params=params,
                s_f=s_f,
                vmaf=vmaf,
                compression_rate=rate,
                compression_ratio=1.0 / rate,
                path=f"/tmp/c{crf}_a{aq}_l{la}.mp4",
                ok=vmaf > 0,
                reason="ok",
            )

        state = run_crf_mode_phase_bc(
            base_crf=30.0,
            initial_params=base_params,
            initial_s_f=0.50,
            initial_vmaf=87.0,
            initial_path="/tmp/base.mp4",
            initial_compression_rate=0.050,
            initial_compression_ratio=20.0,
            evaluate=evaluate,
            vmaf_threshold=85.0,
            max_compression_rate=0.055,  # exercise optional size gate in this unit test
            lookahead_sweep=(40, 60),
            default_lookahead=50,
            job_id="t",
        )
        self.assertGreaterEqual(state.s_f, 0.61)
        self.assertEqual(state.aq_strength, 1.2)
        self.assertEqual(state.rc_lookahead, 40)
        self.assertTrue(any(abs(c[0] - 31.0) < 1e-6 for c in calls))
        self.assertTrue(any(get_param(p, "rc-lookahead") == "40" for _, p in calls))

    def test_crf_down_when_aq_breaks_vmaf(self) -> None:
        """Stronger AQ tanks VMAF → compensate with CRF−."""
        base_params, _, _ = build_crf_mode_params(
            {"texture_level": 0.5, "noise_level_norm": 0.5, "cut_count": 5}
        )
        calls: list[float] = []

        def evaluate(crf: float, params: str) -> CrfModeTrial:
            calls.append(crf)
            aq = float(get_param(params, "aq-strength") or 1.0)
            # Only exercise aq=1.2 path; other AQs return mediocre feasible scores.
            if abs(aq - 1.2) < 1e-6 and abs(crf - 30) < 1e-6:
                vmaf, s_f, rate = 83.0, 0.40, 0.040  # under thr, better size
            elif abs(aq - 1.2) < 1e-6 and abs(crf - 29) < 1e-6:
                vmaf, s_f, rate = 86.0, 0.58, 0.048  # recovered
            elif abs(aq - 1.2) < 1e-6 and abs(crf - 28) < 1e-6:
                vmaf, s_f, rate = 88.0, 0.52, 0.055
            else:
                vmaf, s_f, rate = 86.0, 0.45, 0.060
            return CrfModeTrial(
                crf=crf,
                aq_strength=aq,
                rc_lookahead=50,
                params=params,
                s_f=s_f,
                vmaf=vmaf,
                compression_rate=rate,
                compression_ratio=1.0 / rate,
                path=f"/tmp/c{crf}_a{aq}.mp4",
                ok=True,
                reason="ok",
            )

        state = run_crf_mode_phase_bc(
            base_crf=30.0,
            initial_params=base_params,
            initial_s_f=0.50,
            initial_vmaf=87.0,
            initial_path="/tmp/base.mp4",
            initial_compression_rate=0.050,
            initial_compression_ratio=20.0,
            evaluate=evaluate,
            vmaf_threshold=85.0,
            max_compression_rate=None,
            lookahead_sweep=(),
            default_lookahead=50,
            job_id="down",
        )
        self.assertTrue(any(abs(c - 29.0) < 1e-6 for c in calls))
        self.assertGreaterEqual(state.s_f, 0.58)
        self.assertEqual(state.aq_strength, 1.2)
        self.assertAlmostEqual(state.crf, 29.0)

    def test_feasible_requires_compression_rate(self) -> None:
        trial = CrfModeTrial(
            crf=30,
            aq_strength=1.0,
            rc_lookahead=50,
            params="",
            s_f=0.9,
            vmaf=90.0,
            compression_rate=0.06,
            ok=True,
        )
        self.assertFalse(
            is_crf_mode_feasible(trial, vmaf_threshold=85.0, max_compression_rate=0.055)
        )
        trial.compression_rate = 0.05
        self.assertTrue(
            is_crf_mode_feasible(trial, vmaf_threshold=85.0, max_compression_rate=0.055)
        )


if __name__ == "__main__":
    unittest.main()
