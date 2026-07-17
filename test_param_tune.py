"""Unit tests for sequential libx265 param tuning."""

from __future__ import annotations

import unittest

from param_tune import (
    feature_guided_candidates,
    run_param_tune_loop,
    set_param,
    should_try_crf_bump,
    TuneTrialResult,
)


class ParamTuneHelpersTests(unittest.TestCase):
    def test_set_param_overrides_key(self) -> None:
        out = set_param("aq-mode=2:rd=5:ref=4", "aq-mode", 1)
        self.assertIn("aq-mode=1", out)
        self.assertIn("rd=5", out)

    def test_feature_guided_aq_mode_prefers_1_on_high_texture(self) -> None:
        cands = feature_guided_candidates(
            "aq-mode", 2, {"texture_level": 0.9, "noise_level_norm": 0.2}
        )
        self.assertEqual(cands[0], 1)

    def test_crf_bump_headroom(self) -> None:
        self.assertTrue(should_try_crf_bump(88.0, vmaf_threshold=85.0, headroom=2.0))
        self.assertFalse(should_try_crf_bump(86.5, vmaf_threshold=85.0, headroom=2.0))


class ParamTuneLoopTests(unittest.TestCase):
    def test_picks_higher_s_f_and_optional_crf_bump(self) -> None:
        base = "aq-mode=2:aq-strength=1.0:rd=5:ref=4:bframes=4:rc-lookahead=40"
        calls: list[tuple[int, str]] = []

        def evaluate(crf: int, params: str) -> TuneTrialResult:
            calls.append((crf, params))
            # Better aq-mode=1, and with headroom allow CRF+1 to win.
            s_f = 0.40
            vmaf = 86.0
            if "aq-mode=1" in params:
                s_f = 0.50
                vmaf = 90.0
            if crf >= 31 and "aq-mode=1" in params:
                s_f = 0.55
                vmaf = 87.0
            return TuneTrialResult(
                ok=True,
                crf=crf,
                params=params,
                s_f=s_f,
                vmaf=vmaf,
                path=f"/tmp/c{crf}.mp4",
            )

        state = run_param_tune_loop(
            initial_crf=30,
            initial_params=base,
            initial_s_f=0.40,
            initial_vmaf=86.0,
            initial_path="/tmp/c30.mp4",
            features={"texture_level": 0.9, "motion_level": 0.4},
            evaluate=evaluate,
            vmaf_threshold=85.0,
            crf_max=42,
            max_trials=25,
            no_improve_stop=10,
            vmaf_headroom=2.0,
            keys=("aq-mode",),
            max_rounds=1,
        )
        self.assertGreaterEqual(state.s_f, 0.55)
        self.assertEqual(state.crf, 31)
        self.assertIn("aq-mode=1", state.params)
        self.assertGreaterEqual(state.trials, 1)
        self.assertTrue(any(c[0] == 31 for c in calls))

    def test_stops_after_no_improve_streak(self) -> None:
        base = "aq-mode=2:aq-strength=1.0:rd=5:ref=4:bframes=4:rc-lookahead=40"

        def evaluate(crf: int, params: str) -> TuneTrialResult:
            return TuneTrialResult(
                ok=True,
                crf=crf,
                params=params,
                s_f=0.40,  # never improves
                vmaf=86.0,
                path="/tmp/x.mp4",
            )

        state = run_param_tune_loop(
            initial_crf=30,
            initial_params=base,
            initial_s_f=0.40,
            initial_vmaf=86.0,
            initial_path="/tmp/x.mp4",
            features={},
            evaluate=evaluate,
            vmaf_threshold=85.0,
            crf_max=42,
            max_trials=25,
            no_improve_stop=3,
            vmaf_headroom=2.0,
            max_rounds=3,
        )
        self.assertLessEqual(state.trials, 25)
        self.assertGreaterEqual(state.no_improve_streak, 3)
        self.assertEqual(state.s_f, 0.40)


if __name__ == "__main__":
    unittest.main()
