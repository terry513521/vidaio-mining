"""Unit tests for VBR-mode pack + preprocess arms + param ladder."""

from __future__ import annotations

import unittest

from vbr_mode import (
    VbrModeTrial,
    aq_strength_grid,
    build_vbr_mode_params,
    run_vbr_param_ladder,
    run_vbr_preprocess_arms,
)


def _trial(
    pp,
    params: str,
    *,
    s_f: float,
    vmaf: float = 90.0,
    ok: bool = True,
) -> VbrModeTrial:
    return VbrModeTrial(
        preprocess=pp,
        aq_strength=1.0,
        rd=6,
        bframes=8,
        rc_lookahead=50,
        params=params,
        ok=ok,
        s_f=s_f,
        vmaf=vmaf,
        compression_rate=0.1,
        compression_ratio=10.0,
        path=f"/tmp/{pp or 'none'}.mp4",
        reason="ok" if ok else "fail",
    )


class VbrModeTests(unittest.TestCase):
    def test_aq_grid_inclusive(self) -> None:
        grid = aq_strength_grid(aq_min=0.2, aq_max=2.6, aq_step=0.2)
        self.assertEqual(grid[0], 0.2)
        self.assertEqual(grid[-1], 2.6)
        self.assertIn(1.0, grid)
        self.assertEqual(len(grid), 13)

    def test_build_params_freeze_pack(self) -> None:
        params, aq_mode, _ = build_vbr_mode_params(
            {"texture_level": 0.5, "noise_level_norm": 0.5, "cut_count": 5},
            aq_strength=1.2,
            rd=5,
            bframes=12,
            rc_lookahead=40,
        )
        self.assertEqual(aq_mode, 2)
        self.assertIn("aq-strength=1.2", params)
        self.assertIn("rd=5", params)
        self.assertIn("bframes=12", params)
        self.assertIn("rc-lookahead=40", params)
        self.assertIn("ref=6", params)
        self.assertIn("keyint=60", params)

    def test_preprocess_stops_when_vmaf_clears(self) -> None:
        order: list[str] = []

        def evaluate(pp, params: str) -> VbrModeTrial:
            tag = pp or "none"
            order.append(tag)
            # none already clears threshold — later arms must not run.
            return _trial(pp, params, s_f=0.40, vmaf=90.0)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_preprocess_arms(
            candidates=["hqdn3d_light", "unsharp_micro"],
            baseline_params=baseline,
            evaluate=evaluate,
            vmaf_threshold=85,
        )
        self.assertEqual(order, ["none"])
        self.assertEqual(state.preprocess, None)
        self.assertTrue(state.stopped_early)
        self.assertAlmostEqual(state.s_f, 0.40, places=4)

    def test_preprocess_keeps_best_while_below_threshold(self) -> None:
        order: list[str] = []

        def evaluate(pp, params: str) -> VbrModeTrial:
            tag = pp or "none"
            order.append(tag)
            table = {
                "none": (0.40, 80.0),
                "hqdn3d_light": (0.35, 82.0),
                "unsharp_micro": (0.55, 83.0),
            }
            s_f, vmaf = table[tag]
            return _trial(pp, params, s_f=s_f, vmaf=vmaf)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_preprocess_arms(
            candidates=["hqdn3d_light", "unsharp_micro"],
            baseline_params=baseline,
            evaluate=evaluate,
            vmaf_threshold=85,
        )
        self.assertEqual(order[0], "none")
        self.assertEqual(len(order), 3)
        self.assertEqual(state.preprocess, "unsharp_micro")
        self.assertFalse(state.stopped_early)
        self.assertAlmostEqual(state.s_f, 0.55, places=4)
        self.assertAlmostEqual(state.vmaf, 83.0, places=2)

    def test_ladder_stops_on_first_clear(self) -> None:
        calls: list[str] = []

        def evaluate(pp, params: str) -> VbrModeTrial:
            calls.append(params)
            # Climb below thr until aq=0.4 clears; must not continue to rd/bf/la.
            if "aq-strength=0.4" in params:
                return _trial(pp, params, s_f=0.50, vmaf=90.0)
            return _trial(pp, params, s_f=0.41, vmaf=80.0)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_param_ladder(
            preprocess=None,
            initial_params=baseline,
            initial_s_f=0.40,
            initial_vmaf=80.0,
            initial_path="/tmp/none.mp4",
            initial_compression_rate=0.1,
            initial_compression_ratio=10.0,
            evaluate=evaluate,
            vmaf_threshold=85,
            aq_min=0.2,
            aq_max=0.6,
            aq_step=0.2,
            rd_sweep=(4, 6),
            bframes_sweep=(6, 8),
            lookahead_sweep=(40, 50),
        )
        self.assertTrue(state.stopped_early)
        self.assertIn("aq-strength=0.4", state.params)
        self.assertAlmostEqual(state.vmaf, 90.0, places=2)
        labels = [h["label"] for h in state.history]
        self.assertTrue(any(l.startswith("aq=") for l in labels))
        self.assertFalse(any(l.startswith("rd=") for l in labels))
        self.assertFalse(any(l.startswith("bf=") for l in labels))
        self.assertFalse(any(l.startswith("la=") for l in labels))

    def test_ladder_skips_when_initial_already_clears(self) -> None:
        calls = {"n": 0}

        def evaluate(pp, params: str) -> VbrModeTrial:
            calls["n"] += 1
            return _trial(pp, params, s_f=0.5, vmaf=90.0)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_param_ladder(
            preprocess=None,
            initial_params=baseline,
            initial_s_f=0.40,
            initial_vmaf=90.0,
            initial_path="/tmp/none.mp4",
            initial_compression_rate=0.1,
            initial_compression_ratio=10.0,
            evaluate=evaluate,
            vmaf_threshold=85,
            aq_min=0.2,
            aq_max=0.6,
            aq_step=0.2,
            rd_sweep=(4, 6),
            bframes_sweep=(6, 8),
            lookahead_sweep=(40, 50),
        )
        self.assertEqual(calls["n"], 0)
        self.assertTrue(state.stopped_early)
        self.assertEqual(state.trials, 0)
        self.assertAlmostEqual(state.s_f, 0.40, places=4)

    def test_below_threshold_is_kept_not_failed(self) -> None:
        """Regression: vmaf 91.24 with thr 93 must seed state, not 'all failed'."""
        order: list[str] = []

        def evaluate(pp, params: str) -> VbrModeTrial:
            tag = pp or "none"
            order.append(tag)
            return _trial(pp, params, s_f=0.3182, vmaf=91.24)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_preprocess_arms(
            candidates=["contrast_mild"],
            baseline_params=baseline,
            evaluate=evaluate,
            vmaf_threshold=93,
            job_id="v4",
        )
        self.assertGreater(state.s_f, 0)
        self.assertTrue(state.path)
        self.assertNotEqual(state.preprocess_reason, "all preprocess arms failed")
        self.assertFalse(state.stopped_early)
        self.assertAlmostEqual(state.vmaf, 91.24, places=2)
        # Still below thr → try other preprocess arms (do not abort on first).
        self.assertEqual(order, ["none", "contrast_mild"])

        ladder = run_vbr_param_ladder(
            preprocess=state.preprocess,
            initial_params=state.params,
            initial_s_f=float(state.s_f),
            initial_vmaf=float(state.vmaf),
            initial_path=str(state.path),
            initial_compression_rate=float(state.compression_rate),
            initial_compression_ratio=float(state.compression_ratio),
            evaluate=evaluate,
            vmaf_threshold=93,
            aq_min=1.0,
            aq_max=1.2,
            aq_step=0.2,
            rd_sweep=(6,),
            bframes_sweep=(8,),
            lookahead_sweep=(50,),
        )
        # Ladder must run (not skip-as-failed); still below thr so no early stop
        # unless a trial clears — here none clear, so history may be non-empty.
        self.assertGreater(ladder.s_f, 0)
        self.assertTrue(ladder.path)

    def test_below_threshold_updates_then_clearing_stops(self) -> None:
        def evaluate(pp, params: str) -> VbrModeTrial:
            if "aq-strength=0.2" in params:
                return _trial(pp, params, s_f=0.55, vmaf=82.0)  # better, still below
            if "aq-strength=0.4" in params:
                return _trial(pp, params, s_f=0.50, vmaf=90.0)  # clears — stop
            if "aq-strength=0.6" in params:
                return _trial(pp, params, s_f=0.99, vmaf=95.0)  # must not run
            return _trial(pp, params, s_f=0.40, vmaf=80.0)

        baseline, _, _ = build_vbr_mode_params({})
        state = run_vbr_param_ladder(
            preprocess=None,
            initial_params=baseline,
            initial_s_f=0.40,
            initial_vmaf=80.0,
            initial_path="/tmp/x.mp4",
            initial_compression_rate=0.1,
            initial_compression_ratio=10.0,
            evaluate=evaluate,
            vmaf_threshold=85,
            aq_min=0.2,
            aq_max=0.6,
            aq_step=0.2,
            rd_sweep=(4, 6),
            bframes_sweep=(6, 8),
            lookahead_sweep=(40, 50),
        )
        self.assertTrue(state.stopped_early)
        self.assertIn("aq-strength=0.4", state.params)
        self.assertAlmostEqual(state.vmaf, 90.0, places=2)
        # Cleared at 0.4 even though s_f is lower than the below-thr 0.55 step.
        self.assertAlmostEqual(state.s_f, 0.50, places=4)
        labels = [h["label"] for h in state.history]
        self.assertIn("aq=0.2", labels)
        self.assertIn("aq=0.4", labels)
        self.assertNotIn("aq=0.6", labels)


if __name__ == "__main__":
    unittest.main()
