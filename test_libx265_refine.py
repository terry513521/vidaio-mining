"""Unit tests for NVENC → libx265 refine helpers."""

from __future__ import annotations

import time
import unittest

from interp_search import (
    CqObservation,
    cq_seed_from_features,
    map_cq_to_crf,
    propose_handoff_cqs,
    propose_vmaf_anchored_crfs,
    round1_cqs,
    round1_feature_cqs,
)
from recipes import candidate_crfs
from request import CompressionRequest
from scoring import ScoreResult
from search import (
    TrialResult,
    _is_better_trial,
    abr_refine_bitrate_candidates,
    affordable_refine_candidates,
    refine_search_deadline,
)


def _score(s_f: float, vmaf: float = 90.0) -> ScoreResult:
    return ScoreResult(
        s_f=s_f,
        vmaf=vmaf,
        compression_rate=0.1,
        compression_ratio=10.0,
        compression_component=0.5,
        quality_component=0.5,
        reason="ok",
        validation_errors=[],
        vmaf_base=vmaf,
        vmaf_delta=0.0,
        passed_encoding_gates=True,
        passed_vmaf_delta_gate=True,
    )


def _trial(
    *,
    s_f: float,
    encoder: str,
    stage: str,
    crf: int | None = 28,
    encode_ok: bool = True,
) -> TrialResult:
    return TrialResult(
        recipe="t",
        mode="RC",
        crf=crf,
        bitrate=None,
        path="/tmp/x.mp4",
        score=_score(s_f),
        encode_ok=encode_ok,
        stage=stage,
        encoder=encoder,
    )


class AbrRefineCandidatesTests(unittest.TestCase):
    def test_default_three_at_or_below_anchor(self) -> None:
        cands = abr_refine_bitrate_candidates(
            4.0, floor_mbps=0.5, cap_mbps=8.0, count=3
        )
        self.assertEqual(cands, [3.4, 3.68, 4.0])
        self.assertTrue(all(c <= 4.0 for c in cands))

    def test_clamps_to_floor_and_cap(self) -> None:
        cands = abr_refine_bitrate_candidates(
            0.2, floor_mbps=0.5, cap_mbps=1.0, count=3
        )
        self.assertTrue(min(cands) >= 0.5)
        self.assertTrue(max(cands) <= 1.0)
        self.assertEqual(len(cands), 3)

    def test_single_candidate(self) -> None:
        self.assertEqual(
            abr_refine_bitrate_candidates(2.5, floor_mbps=0.5, cap_mbps=8.0, count=1),
            [2.5],
        )


class RcRefineCandidatesTests(unittest.TestCase):
    def test_feature_seed_band_not_nvenc_cq(self) -> None:
        # NVENC CQ 28 must not become the x265 CRF identity map.
        nvenc_cq = 28
        seed = 18  # threshold/feature-derived seed for VMAF 85
        cands = candidate_crfs(seed, 8, 40, count=3, spread=2)
        self.assertEqual(cands, [14, 16, 18])
        self.assertNotIn(nvenc_cq, cands)


class DeadlineReservationTests(unittest.TestCase):
    def test_reserves_refine_time(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="hevc_nvenc",
            preset="p5",
            libx265_refine=True,
            libx265_refine_time_sec=60.0,
            time_budget_sec=180.0,
        )
        overall = time.time() + 180.0
        search_dl = refine_search_deadline(req, overall)
        # Search budget ≈ 120s (180 - 60), allow small scheduling slack.
        left = search_dl - time.time()
        self.assertGreater(left, 110.0)
        self.assertLess(left, 125.0)

    def test_no_reserve_when_disabled(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="hevc_nvenc",
            preset="p5",
            libx265_refine=False,
            time_budget_sec=180.0,
        )
        overall = time.time() + 180.0
        self.assertAlmostEqual(refine_search_deadline(req, overall), overall, places=2)

    def test_disabled_for_libx265_encoder(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="libx265",
            preset="medium",
            libx265_refine=True,
            libx265_refine_time_sec=60.0,
        )
        self.assertFalse(req.refine_with_libx265_enabled)


class WinnerFallbackTests(unittest.TestCase):
    def test_higher_s_f_wins(self) -> None:
        nvenc = _trial(s_f=0.3, encoder="hevc_nvenc", stage="full")
        x265 = _trial(s_f=0.8, encoder="libx265", stage="libx265_refine")
        self.assertTrue(_is_better_trial(x265, nvenc))

    def test_nvenc_kept_when_refine_worse(self) -> None:
        nvenc = _trial(s_f=0.5, encoder="hevc_nvenc", stage="full")
        x265 = _trial(s_f=0.2, encoder="libx265", stage="libx265_refine")
        self.assertFalse(_is_better_trial(x265, nvenc))

    def test_failed_refine_does_not_win(self) -> None:
        nvenc = _trial(s_f=0.4, encoder="hevc_nvenc", stage="full")
        x265 = _trial(
            s_f=0.0, encoder="libx265", stage="libx265_refine", encode_ok=False
        )
        self.assertFalse(_is_better_trial(x265, nvenc))


def _obs(cq: int, s_f: float, vmaf: float, ok: bool = True) -> CqObservation:
    return CqObservation(
        cq=cq,
        vmaf=vmaf,
        compression_rate=0.1,
        compression_ratio=10.0,
        s_f=s_f,
        encode_ok=ok,
    )


# Recorded Round-1 observations from the reported run (CQ 22,26,29,32,36):
# s_f climbs with CQ and VMAF stays above the 85 threshold, so the best
# measured point is CQ 36 near the compression cliff.
_ROUND1 = [
    _obs(22, 0.30, 93.0),
    _obs(26, 0.36, 91.0),
    _obs(29, 0.40, 90.0),
    _obs(32, 0.44, 88.0),
    _obs(36, 0.47, 87.0),
]


class HandoffCqTests(unittest.TestCase):
    def test_measured_best_first(self) -> None:
        handoff = propose_handoff_cqs(
            _ROUND1, count=3, crf_min=22, crf_max=40, vmaf_threshold=85.0
        )
        self.assertGreaterEqual(len(handoff), 1)
        self.assertEqual(handoff[0].cq, 36)
        self.assertEqual(handoff[0].reason, "measured_best")

    def test_includes_higher_interpolated_cq(self) -> None:
        handoff = propose_handoff_cqs(
            _ROUND1, count=4, crf_min=22, crf_max=40, vmaf_threshold=85.0
        )
        cqs = [h.cq for h in handoff]
        # Beyond the measured best, we expect at least one higher (>=36) CQ probe.
        self.assertTrue(any(c >= 36 for c in cqs[1:]))
        self.assertEqual(len(set(cqs)), len(cqs))

    def test_empty_observations_returns_empty(self) -> None:
        self.assertEqual(
            propose_handoff_cqs(
                [], count=3, crf_min=22, crf_max=40, vmaf_threshold=85.0
            ),
            [],
        )

    def test_skips_failed_observations(self) -> None:
        obs = [_obs(22, 0.0, 0.0, ok=False), _obs(30, 0.4, 90.0)]
        handoff = propose_handoff_cqs(
            obs, count=2, crf_min=22, crf_max=40, vmaf_threshold=85.0
        )
        self.assertEqual(handoff[0].cq, 30)


class CqToCrfMappingTests(unittest.TestCase):
    """Legacy fixed-offset helpers still behave; refine uses VMAF anchoring."""

    def test_fixed_offset(self) -> None:
        self.assertEqual(map_cq_to_crf(36, -6, 16, 40), 30)
        self.assertEqual(map_cq_to_crf(22, -6, 16, 40), 16)

    def test_clamps_to_bounds(self) -> None:
        self.assertEqual(map_cq_to_crf(18, -6, 16, 40), 16)
        self.assertEqual(map_cq_to_crf(50, -6, 16, 40), 40)


class VmafAnchoredCrfTests(unittest.TestCase):
    def test_matches_nvenc_quality_not_fixed_offset(self) -> None:
        # Recorded Round-1 shape: best at CQ 33, neg≈90.6 (headroom above 85).
        obs = [
            _obs(25, 0.283, 96.15),
            _obs(27, 0.292, 95.68),
            _obs(29, 0.304, 95.01),
            _obs(31, 0.323, 93.09),
            _obs(33, 0.351, 90.63),
        ]
        props = propose_vmaf_anchored_crfs(
            obs,
            count=3,
            nvenc_cq_min=22,
            nvenc_cq_max=40,
            crf_min=16,
            crf_max=40,
            vmaf_threshold=85.0,
            spread=2,
        )
        crfs = [p.crf for p in props]
        self.assertGreaterEqual(len(crfs), 1)
        # Range-map CQ 33 in [22,40] → CRF in [16,40] ≈ 31 — not offset -6 → 27.
        self.assertIn(31, crfs)
        self.assertNotIn(27, crfs)
        self.assertTrue(all(c >= 29 for c in crfs))

    def test_includes_near_gate_push(self) -> None:
        props = propose_vmaf_anchored_crfs(
            _ROUND1,
            count=3,
            nvenc_cq_min=22,
            nvenc_cq_max=40,
            crf_min=16,
            crf_max=40,
            vmaf_threshold=85.0,
            spread=2,
        )
        crfs = [p.crf for p in props]
        self.assertEqual(len(set(crfs)), len(crfs))
        # Best CQ 36 → range-mapped CRF high in the band; plus push/gate probes.
        self.assertTrue(max(crfs) >= 30)

    def test_empty_observations(self) -> None:
        self.assertEqual(
            propose_vmaf_anchored_crfs(
                [],
                count=3,
                nvenc_cq_min=22,
                nvenc_cq_max=40,
                crf_min=16,
                crf_max=40,
                vmaf_threshold=85.0,
            ),
            [],
        )


class AffordableCandidatesTests(unittest.TestCase):
    def test_full_set_when_budget_ample(self) -> None:
        n = affordable_refine_candidates(
            600.0, 3, workers=3, sec_per_candidate=90.0
        )
        self.assertEqual(n, 3)

    def test_reduces_under_short_budget(self) -> None:
        # 100s budget, 1 worker, 90s/candidate -> 1 batch -> 1 candidate.
        n = affordable_refine_candidates(
            100.0, 3, workers=1, sec_per_candidate=90.0
        )
        self.assertEqual(n, 1)

    def test_never_below_one(self) -> None:
        n = affordable_refine_candidates(
            5.0, 3, workers=1, sec_per_candidate=90.0
        )
        self.assertEqual(n, 1)

    def test_parallel_workers_increase_throughput(self) -> None:
        # 100s, 3 workers -> 1 batch of 3 = full requested set.
        n = affordable_refine_candidates(
            100.0, 3, workers=3, sec_per_candidate=90.0
        )
        self.assertEqual(n, 3)


class HandoffConfigTests(unittest.TestCase):
    def test_handoff_active_for_nvenc_rc(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="hevc_nvenc",
            preset="p5",
            codec_mode="RC",
            libx265_refine=True,
        )
        self.assertTrue(req.nvenc_x265_handoff)

    def test_handoff_inactive_for_abr(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="hevc_nvenc",
            preset="p5",
            codec_mode="ABR",
            target_bitrate="8M",
            libx265_refine=True,
        )
        self.assertFalse(req.nvenc_x265_handoff)

    def test_handoff_inactive_without_refine(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            encoder="hevc_nvenc",
            preset="p5",
            codec_mode="RC",
            libx265_refine=False,
        )
        self.assertFalse(req.nvenc_x265_handoff)

    def test_x265_bounds_default_to_crf_bounds(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4", crf_min=20, crf_max=38
        )
        self.assertEqual(req.x265_crf_floor, 20)
        self.assertEqual(req.x265_crf_ceiling, 38)

    def test_x265_bounds_override(self) -> None:
        req = CompressionRequest(
            input_path="x.mp4",
            crf_min=22,
            crf_max=40,
            libx265_crf_min=16,
            libx265_crf_max=36,
        )
        self.assertEqual(req.x265_crf_floor, 16)
        self.assertEqual(req.x265_crf_ceiling, 36)


class RequestConfigTests(unittest.TestCase):
    def test_request_json_enables_refine(self) -> None:
        req = CompressionRequest.from_json("request.json")
        self.assertTrue(req.libx265_refine)
        self.assertTrue(req.refine_with_libx265_enabled)
        self.assertTrue(req.nvenc_x265_handoff)
        self.assertTrue(req.libx265_feature_baseline)
        self.assertEqual(req.libx265_refine_preset, "superfast")
        self.assertEqual(req.libx265_refine_candidates, 1)
        self.assertEqual(req.libx265_cq_to_crf_offset, 0)

    def test_encoder_field_serialized(self) -> None:
        from search import SearchResult

        best = _trial(s_f=0.5, encoder="libx265", stage="libx265_refine")
        result = SearchResult(best=best, trials=[best], strategy="x+libx265_refine")
        payload = result.to_dict()
        self.assertEqual(payload["best"]["encoder"], "libx265")
        self.assertEqual(payload["trials"][0]["encoder"], "libx265")


class FeatureRound1CqTests(unittest.TestCase):
    def test_hard_content_seeds_lower_than_easy(self) -> None:
        hard = {
            "motion_level": 0.70,
            "texture_level": 0.65,
            "edge_level": 0.60,
            "noise_level_norm": 0.55,
            "cut_level": 0.50,
            "worst_difficulty": 0.80,
            "hard_fraction": 0.60,
            "volatility": 0.70,
            "fps": 30.0,
        }
        easy = {
            "motion_level": 0.30,
            "texture_level": 0.30,
            "edge_level": 0.30,
            "noise_level_norm": 0.30,
            "cut_level": 0.10,
            "worst_difficulty": 0.20,
            "hard_fraction": 0.10,
            "volatility": 0.10,
            "fps": 30.0,
        }
        hard_seed, _ = cq_seed_from_features(
            hard, vmaf_threshold=85, crf_min=22, crf_max=40
        )
        easy_seed, _ = cq_seed_from_features(
            easy, vmaf_threshold=85, crf_min=22, crf_max=40
        )
        self.assertLess(hard_seed, easy_seed)

    def test_higher_vmaf_threshold_seeds_lower_cq(self) -> None:
        feats = {
            "motion_level": 0.50,
            "texture_level": 0.50,
            "edge_level": 0.50,
            "noise_level_norm": 0.50,
            "cut_level": 0.20,
            "fps": 30.0,
        }
        seed85, _ = cq_seed_from_features(
            feats, vmaf_threshold=85, crf_min=22, crf_max=40
        )
        seed93, _ = cq_seed_from_features(
            feats, vmaf_threshold=93, crf_min=22, crf_max=40
        )
        self.assertLess(seed93, seed85)

    def test_crf_start_overrides_features(self) -> None:
        feats = {
            "motion_level": 0.70,
            "texture_level": 0.70,
            "edge_level": 0.70,
            "noise_level_norm": 0.70,
            "cut_level": 0.70,
            "fps": 30.0,
        }
        seed, reason = cq_seed_from_features(
            feats, vmaf_threshold=85, crf_min=22, crf_max=40, crf_start=28
        )
        self.assertEqual(seed, 28)
        self.assertIn("crf_start", reason)

    def test_upward_band_climbs_toward_cliff(self) -> None:
        feats = {
            "motion_level": 0.50,
            "texture_level": 0.50,
            "edge_level": 0.50,
            "noise_level_norm": 0.50,
            "cut_level": 0.20,
            "worst_difficulty": 0.50,
            "hard_fraction": 0.40,
            "volatility": 0.40,
            "fps": 30.0,
        }
        cqs, seed, reason = round1_feature_cqs(
            feats,
            count=5,
            crf_min=22,
            crf_max=40,
            vmaf_threshold=85.0,
            spread=2,
        )
        self.assertEqual(len(cqs), 5)
        self.assertEqual(len(set(cqs)), 5)
        self.assertIn(seed, cqs)
        self.assertTrue(all(22 <= c <= 40 for c in cqs))
        self.assertIn("upward_band", reason)
        # Content-sensitive seed sits mid-band (not stuck at cliff top).
        self.assertGreaterEqual(seed, 24)
        self.assertLessEqual(seed, 34)
        self.assertGreaterEqual(max(cqs), seed)
        above_or_at = sum(1 for c in cqs if c >= seed)
        self.assertGreaterEqual(above_or_at, 3)
        linspace = round1_cqs(22, 40, 5)
        self.assertNotEqual(cqs, linspace)

    def test_recorded_features_probe_past_cq33(self) -> None:
        # Soft-norm levels from a hard mashup: seed should be safer (lower)
        # than the old cliff-aiming seed (~35+), leaving room to climb.
        feats = {
            "motion_level": 0.4809,
            "texture_level": 0.5017,
            "edge_level": 0.5716,
            "noise_level_norm": 0.4775,
            "cut_level": 0.3337,
            "worst_difficulty": 0.5973,
            "hard_fraction": 0.8331,
            "volatility": 0.5328,
            "fps": 30.0,
        }
        cqs, seed, _ = round1_feature_cqs(
            feats,
            count=5,
            crf_min=22,
            crf_max=40,
            vmaf_threshold=85.0,
            spread=2,
        )
        self.assertGreaterEqual(seed, 22)
        self.assertLessEqual(seed, 32)
        self.assertTrue(all(22 <= c <= 40 for c in cqs))
        self.assertGreaterEqual(max(cqs), seed)

    def test_no_features_falls_back_to_linspace(self) -> None:
        cqs, _, reason = round1_feature_cqs(
            None,
            count=5,
            crf_min=22,
            crf_max=40,
            vmaf_threshold=85.0,
            spread=2,
        )
        self.assertEqual(cqs, round1_cqs(22, 40, 5))
        self.assertIn("linspace_fallback", reason)

    def test_clamps_to_bounds(self) -> None:
        feats = {
            "motion_level": 0.90,
            "texture_level": 0.90,
            "edge_level": 0.90,
            "noise_level_norm": 0.90,
            "cut_level": 0.90,
            "worst_difficulty": 0.95,
            "hard_fraction": 0.90,
            "volatility": 0.90,
            "fps": 30.0,
        }
        cqs, seed, _ = round1_feature_cqs(
            feats,
            count=5,
            crf_min=22,
            crf_max=40,
            vmaf_threshold=93.0,
            spread=2,
        )
        self.assertTrue(all(22 <= c <= 40 for c in cqs))
        self.assertTrue(22 <= seed <= 40)


if __name__ == "__main__":
    unittest.main()
