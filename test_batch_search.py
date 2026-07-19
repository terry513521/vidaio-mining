"""Unit tests for fleet batch / serial CQ search."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_search import fleet_jobs_from_request, load_fleet_jobs
from interp_search import (
    estimate_primary_x265_crf,
    next_serial_cq_probe,
    observations_from_trials,
    round1_feature_cqs,
    CqObservation,
)
from request import CompressionRequest


def _obs(cq: int, s_f: float, vmaf: float) -> CqObservation:
    return CqObservation(
        cq=cq,
        vmaf=vmaf,
        compression_rate=0.1,
        compression_ratio=10.0,
        s_f=s_f,
        encode_ok=True,
    )


class SerialCqProbeTests(unittest.TestCase):
    def test_walks_probe_plan_one_per_round(self) -> None:
        feats = {
            "motion_level": 0.5,
            "texture_level": 0.5,
            "edge_level": 0.5,
            "noise_level_norm": 0.45,
            "cut_level": 0.2,
            "fps": 30.0,
        }
        plan, _, _ = round1_feature_cqs(
            feats, count=5, crf_min=22, crf_max=40, vmaf_threshold=85.0, spread=2
        )
        used: set[int] = set()
        picked: list[int] = []
        for r in range(1, 6):
            cq, _ = next_serial_cq_probe(
                [],
                round_idx=r,
                max_rounds=5,
                probe_plan=plan,
                features=feats,
                crf_min=22,
                crf_max=40,
                vmaf_threshold=85.0,
                spread=2,
                crf_start=None,
                used=used,
            )
            self.assertIsNotNone(cq)
            assert cq is not None
            used.add(cq)
            picked.append(cq)
        self.assertEqual(len(set(picked)), 5)
        self.assertEqual(picked, plan)

    def test_estimate_primary_crf_near_match(self) -> None:
        obs = [
            _obs(33, 0.35, 90.6),
            _obs(35, 0.39, 87.5),
            _obs(37, 0.09, 84.0),
        ]
        crf, _ = estimate_primary_x265_crf(
            obs,
            nvenc_cq_min=22,
            nvenc_cq_max=40,
            crf_min=16,
            crf_max=40,
            vmaf_threshold=85.0,
        )
        self.assertIsNotNone(crf)
        assert crf is not None
        self.assertGreaterEqual(crf, 28)


class FleetManifestTests(unittest.TestCase):
    def test_load_manifest_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "jobs.txt"
            manifest.write_text(
                "/a/v1.mp4\n"
                "/a/v2.mp4\t/custom/out.mp4\n"
                "# comment\n"
                "\n",
                encoding="utf-8",
            )
            jobs = load_fleet_jobs(str(manifest), limit=0)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[1].output_path, "/custom/out.mp4")


class FleetRequestTests(unittest.TestCase):
    def test_fleet_defaults(self) -> None:
        req = CompressionRequest(input_path="x.mp4")
        self.assertEqual(req.fleet_batch_size, 5)
        self.assertEqual(req.fleet_gpu_slots, 1)

    def test_assign_fleet_gpu_slots(self) -> None:
        from batch_search import FleetVideoJob, assign_fleet_gpu_slots

        jobs = [
            FleetVideoJob(job_id="a", input_path="a.mp4", output_path="oa.mp4", work_dir="wa"),
            FleetVideoJob(job_id="b", input_path="b.mp4", output_path="ob.mp4", work_dir="wb"),
            FleetVideoJob(job_id="c", input_path="c.mp4", output_path="oc.mp4", work_dir="wc"),
        ]
        assign_fleet_gpu_slots(jobs, 1)
        self.assertTrue(jobs[0].use_gpu)
        self.assertFalse(jobs[1].use_gpu)
        self.assertFalse(jobs[2].use_gpu)
        assign_fleet_gpu_slots(jobs, 2)
        self.assertTrue(jobs[0].use_gpu)
        self.assertEqual(jobs[0].gpu_device, 0)
        self.assertTrue(jobs[1].use_gpu)
        self.assertEqual(jobs[1].gpu_device, 1)
        self.assertFalse(jobs[2].use_gpu)
        self.assertIsNone(jobs[2].gpu_device)

    def test_http_jobs_are_parsed_and_materialized(self) -> None:
        req = CompressionRequest.from_dict(
            {
                "work_dir": "/tmp/fleet",
                "jobs": [
                    {
                        "id": "clip/1",
                        "input_url": "https://download.invalid/input.mp4",
                        "upload_url": "https://upload.invalid/output.mp4?signature=x",
                    }
                ],
            }
        )
        jobs = fleet_jobs_from_request(req)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "clip/1")
        self.assertTrue(jobs[0].input_path.endswith("clip_1/input.mp4"))
        self.assertEqual(jobs[0].upload_url, req.jobs[0]["upload_url"])
        self.assertFalse(req.skip_transfer)

    def test_local_jobs_skip_transfer(self) -> None:
        req = CompressionRequest.from_dict(
            {
                "work_dir": "/tmp/fleet",
                "jobs": [
                    {
                        "id": "local1",
                        "input_path": "/data/a.mp4",
                        "output_path": "/out/a.mp4",
                    }
                ],
            }
        )
        self.assertTrue(req.skip_transfer)
        self.assertEqual(req.download_reserve_sec, 0.0)
        self.assertEqual(req.upload_reserve_sec, 0.0)
        jobs = fleet_jobs_from_request(req)
        self.assertEqual(jobs[0].input_path, "/data/a.mp4")
        self.assertEqual(jobs[0].output_path, "/out/a.mp4")
        self.assertEqual(jobs[0].input_url, "")

    def test_http_jobs_reject_non_http_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_url"):
            CompressionRequest.from_dict(
                {
                    "jobs": [
                        {
                            "input_url": "file:///tmp/input.mp4",
                            "upload_url": "https://upload.invalid/out",
                        }
                    ]
                }
            )

    def test_sla_profile(self) -> None:
        config = Path(__file__).with_name("request.json")
        req = CompressionRequest.from_json(str(config))
        self.assertEqual(req.time_budget_sec, 380)
        self.assertEqual(req.crf_candidates, 2)
        self.assertEqual(req.encoder, "libx265")
        self.assertEqual(req.libx265_refine_preset, "medium")
        self.assertEqual(req.libx265_profile, "main")
        self.assertIsNone(req.libx265_params)
        self.assertEqual(req.proxy_vmaf_margin, 2.0)
        self.assertEqual(req.proxy_mashup_push_ceiling, 0.55)
        self.assertEqual(req.proxy_seconds_per_segment, 2.0)
        self.assertEqual(req.fleet_gpu_slots, 1)
        self.assertTrue(req.skip_transfer)
        self.assertEqual(len(req.jobs), 5)


class RuleAnchoredCrfTests(unittest.TestCase):
    def test_bracket_picks_passing_candidate(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=27, vmaf=91.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=29, vmaf=87.0, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=29,
            candidates=[27, 29],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
        )
        self.assertIsNotNone(crf)
        assert crf is not None
        self.assertIn("bracket", reason)
        self.assertIn(crf, {27, 29, 28, 30})

    def test_all_pass_above_proxy_target_picks_high_crf(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=30, vmaf=92.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=32, vmaf=91.5, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=30,
            candidates=[30, 32],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
            proxy_vmaf_margin=2.0,
        )
        self.assertEqual(crf, 32)
        self.assertIn("high_crf", reason)

    def test_all_pass_excess_margin_nudges_up(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=30, vmaf=93.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=32, vmaf=92.5, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=30,
            candidates=[30, 32],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
            proxy_vmaf_margin=2.0,
        )
        self.assertEqual(crf, 33)
        self.assertIn("margin", reason)

    def test_bracket_lean_up_when_high_probe_near_threshold(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=27, vmaf=91.5, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=29, vmaf=88.5, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=29,
            candidates=[27, 29],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
            proxy_vmaf_margin=2.0,
        )
        self.assertEqual(crf, 29)
        self.assertIn("lean_up", reason)

    def test_mashup_blocks_upward_push(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=30, vmaf=93.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=32, vmaf=92.5, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        feats = {
            "worst_difficulty": 0.7,
            "hard_fraction": 0.8,
            "volatility": 0.5,
        }
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=30,
            candidates=[30, 32],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
            proxy_vmaf_margin=2.0,
            mashup_push_ceiling=0.55,
            features=feats,
        )
        self.assertEqual(crf, 32)
        self.assertIn("best_s_f", reason)

    def test_asymmetric_hard_content_uses_down_band(self) -> None:
        from interp_search import CqObservation, pick_rule_anchored_crf

        obs = [
            CqObservation(cq=30, vmaf=84.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.0, encode_ok=True),
            CqObservation(cq=32, vmaf=82.0, compression_rate=0.15, compression_ratio=6.7, s_f=0.0, encode_ok=True),
        ]
        crf, reason = pick_rule_anchored_crf(
            obs,
            seed=30,
            candidates=[30, 32],
            crf_min=16,
            crf_max=40,
            vmaf_threshold=89,
        )
        self.assertEqual(crf, 29)
        self.assertIn("all_fail", reason)

    def test_asymmetric_hard_content_uses_down_band(self) -> None:
        feats = {
            "motion_level": 0.8,
            "texture_level": 0.5,
            "edge_level": 0.5,
            "noise_level_norm": 0.3,
            "cut_level": 0.2,
            "worst_difficulty": 0.7,
            "hard_fraction": 0.8,
            "volatility": 0.5,
            "fps": 30.0,
        }
        cqs, seed, reason = round1_feature_cqs(
            feats, count=2, crf_min=22, crf_max=40, vmaf_threshold=89, spread=2
        )
        self.assertEqual(cqs, [seed - 2, seed])
        self.assertIn("hard_down_band", reason)

    def test_moderate_hard_motion_uses_up_band(self) -> None:
        feats = {
            "motion_level": 0.7,
            "texture_level": 0.5,
            "edge_level": 0.5,
            "noise_level_norm": 0.3,
            "cut_level": 0.2,
            "worst_difficulty": 0.55,
            "hard_fraction": 0.5,
            "volatility": 0.2,
            "fps": 30.0,
        }
        cqs, seed, reason = round1_feature_cqs(
            feats, count=2, crf_min=22, crf_max=40, vmaf_threshold=89, spread=2
        )
        self.assertEqual(cqs, [seed, seed + 2])
        self.assertIn("hard_up_band", reason)

    def test_asymmetric_easy_content_uses_up_band(self) -> None:
        feats = {
            "motion_level": 0.2,
            "texture_level": 0.2,
            "edge_level": 0.2,
            "noise_level_norm": 0.2,
            "cut_level": 0.1,
            "worst_difficulty": 0.2,
            "hard_fraction": 0.1,
            "volatility": 0.1,
            "fps": 30.0,
        }
        cqs, seed, reason = round1_feature_cqs(
            feats, count=2, crf_min=22, crf_max=40, vmaf_threshold=89, spread=2
        )
        self.assertEqual(cqs, [seed, seed + 2])
        self.assertIn("easy_up_band", reason)


class InterpCrfTests(unittest.TestCase):
    def test_estimate_interp_crf_from_two_probes(self) -> None:
        from interp_search import CqObservation, estimate_interp_crf

        obs = [
            CqObservation(cq=32, vmaf=92.0, compression_rate=0.2, compression_ratio=5.0, s_f=0.5, encode_ok=True),
            CqObservation(cq=34, vmaf=86.0, compression_rate=0.15, compression_ratio=6.7, s_f=0.55, encode_ok=True),
        ]
        crf, reason = estimate_interp_crf(obs, crf_min=16, crf_max=40, vmaf_threshold=89)
        self.assertIsNotNone(crf)
        assert crf is not None
        self.assertIn("interp_vmaf", reason)
        self.assertGreaterEqual(crf, 32)
        self.assertLessEqual(crf, 34)


class Libx265ParamsTests(unittest.TestCase):
    def test_merge_x265_params_override_wins(self) -> None:
        from interp_search import merge_x265_params

        base = "aq-mode=3:rd=5:ref=4"
        merged = merge_x265_params(base, "rd=6:ref=5")
        self.assertEqual(merged, "aq-mode=3:rd=6:ref=5")

    def test_request_accepts_x265_params_alias(self) -> None:
        req = CompressionRequest.from_dict(
            {
                "encoder": "libx265",
                "profile": "main",
                "x265_params": "scenecut=50",
            }
        )
        self.assertEqual(req.libx265_profile, "main")
        self.assertEqual(req.libx265_params, "scenecut=50")

    def test_proxy_scoring_skips_pair_gates(self) -> None:
        from scoring import score_candidate, validate_validator_gates

        ref = "/tmp/test_proxy_ref.mp4"
        dist = "/tmp/test_probe_crf27.mp4"
        if not Path(ref).is_file() or not Path(dist).is_file():
            self.skipTest("proxy fixture not built")
        paired = validate_validator_gates(ref, dist)
        self.assertFalse(paired.ok)
        score = score_candidate(
            ref,
            dist,
            89,
            vmaf_n_subsample=6,
            vmaf_docker_gpus=False,
            pair_gates=False,
            timeout=90,
        )
        self.assertNotEqual(score.reason, "encoding_gate_failed")


class CompressUtilTests(unittest.TestCase):
    def test_measure_compression(self) -> None:
        from compress_util import measure_compression

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.bin"
            out = Path(td) / "out.bin"
            src.write_bytes(b"x" * 1000)
            out.write_bytes(b"y" * 250)
            rate, ratio = measure_compression(str(src), str(out))
            self.assertAlmostEqual(rate, 0.25)
            self.assertAlmostEqual(ratio, 4.0)


if __name__ == "__main__":
    unittest.main()
