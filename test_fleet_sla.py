"""Mocked end-to-end tests for the 180-second fleet SLA path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from batch_search import fleet_jobs_from_request, run_fleet_sla
from crf_search import CrfSearchResult
from encoder import EncodeResult
from fleet_io import TransferResult
from proxy import ProxyBuildResult
from request import CompressionRequest
from scoring import EncodeValidation, ScoreResult
from search import TrialResult


def _score(s_f: float = 0.4, vmaf: float = 90.0) -> ScoreResult:
    return ScoreResult(
        s_f=s_f,
        vmaf=vmaf,
        compression_rate=0.1,
        compression_ratio=10.0,
        compression_component=0.8,
        quality_component=0.7,
        reason="success",
        validation_errors=[],
        vmaf_base=vmaf + 0.5,
        vmaf_delta=0.5,
        passed_encoding_gates=True,
        passed_vmaf_delta_gate=True,
    )


def _final_score_trial(trial: TrialResult) -> TrialResult:
    """Attach a real full-file score to an encoded SLA candidate (tests/mocks)."""
    return TrialResult(
        recipe=trial.recipe,
        mode=trial.mode,
        crf=trial.crf,
        bitrate=trial.bitrate,
        path=trial.path,
        score=_score(s_f=0.42, vmaf=88.5),
        encode_ok=True,
        stage="sla_final",
        encoder=trial.encoder,
        encode_sec=trial.encode_sec,
        score_sec=1.5,
        elapsed_sec=trial.encode_sec + 1.5,
    )


def _set_feature_scenes(job, *, duration: float = 30.0, difficulty: float = 0.4) -> None:
    job.features = {
        "motion_level": 0.4,
        "texture_level": 0.5,
        "edge_level": 0.4,
        "noise_level_norm": 0.3,
        "cut_level": 0.2,
        "fps": 30.0,
        "duration": duration,
        "avg_segment_duration": duration,
    }
    job.segments = [
        {
            "index": 0,
            "start_sec": 0.0,
            "end_sec": duration,
            "duration": duration,
            "difficulty": difficulty,
        }
    ]


def _probe_trial(path: str, cq: int = 33, *, vmaf: float | None = None) -> TrialResult:
    vmaf_score = vmaf if vmaf is not None else 88.0 + (40 - cq) * 0.5
    return TrialResult(
        recipe="medium",
        mode="RC",
        crf=cq,
        bitrate=None,
        path=path,
        score=_score(vmaf=vmaf_score),
        encode_ok=True,
        stage="sla_scene_crf_search",
        encoder="libx265",
        encode_sec=1.0,
        score_sec=1.0,
        elapsed_sec=2.0,
    )


class FleetSlaTests(unittest.TestCase):
    def _template(self, work_dir: str) -> CompressionRequest:
        return CompressionRequest.from_dict(
            {
                "work_dir": work_dir,
                "vmaf_threshold": 85,
                "encoder": "libx265",
                "libx265_refine_preset": "superfast",
                "time_budget_sec": 180,
                "download_reserve_sec": 25,
                "final_encode_reserve_sec": 90,
                "upload_reserve_sec": 20,
                "probe_min_budget_sec": 10,
                "crf_candidates": 2,
                "crf_min": 22,
                "crf_max": 40,
                "sample_frames": 8,
                "vmaf_n_subsample": 6,
                "fleet_batch_size": 2,
                "jobs": [
                    {
                        "id": "a",
                        "input_url": "https://download.invalid/a.mp4",
                        "upload_url": "https://upload.invalid/a.mp4?sig=1",
                    },
                    {
                        "id": "b",
                        "input_url": "https://download.invalid/b.mp4",
                        "upload_url": "https://upload.invalid/b.mp4?sig=2",
                    },
                ],
            }
        )

    def test_one_probe_then_x265_upload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            template = self._template(td)
            jobs = fleet_jobs_from_request(template)

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_download(url, dest, *, deadline, chunk_size=1024 * 1024):
                path = Path(dest)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"input-bytes")
                return TransferResult(True, str(path), 11, 0.01, url=url)

            def fake_search(evaluate, **kwargs):
                return CrfSearchResult(True, 30.0, 90.0, reason="test")

            def fake_encode(input_path, output_path, **kwargs):
                label = b"probe" if "probe_s" in output_path else b"final-x265"
                Path(output_path).write_bytes(label)
                return EncodeResult(True, output_path, 0, "", [])

            def fake_upload(source, url, *, deadline, content_type="video/mp4", chunk_size=1024 * 1024):
                return TransferResult(True, str(source), Path(source).stat().st_size, 0.01, url=url)

            with patch("batch_search.download_to_path", side_effect=fake_download), patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf", side_effect=fake_search
            ), patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ), patch(
                "batch_search._score_sla_final",
                side_effect=lambda req, job, trial, *, deadline: _final_score_trial(trial),
            ), patch(
                "batch_search.upload_presigned_put", side_effect=fake_upload
            ):
                results = run_fleet_sla(template, jobs)

            self.assertEqual(len(results), 2)
            for job, result in zip(jobs, results):
                self.assertEqual(job.error, "")
                self.assertTrue(job.uploaded)
                self.assertIsNotNone(result.best)
                assert result.best is not None
                self.assertEqual(result.best.encoder, "libx265")
                self.assertEqual(result.strategy, "fleet_sla_x265_full_crf")
                self.assertEqual(job.chosen_crf, 30)

    def test_final_encode_failure_marks_job_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            template = self._template(td)
            template.jobs = template.jobs[:1]
            jobs = fleet_jobs_from_request(template)

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=20.0, difficulty=0.5)
                job.features.update(
                    {
                        "motion_level": 0.5,
                        "edge_level": 0.5,
                        "noise_level_norm": 0.4,
                        "cut_level": 0.1,
                    }
                )

            def fake_download(url, dest, *, deadline, chunk_size=1024 * 1024):
                path = Path(dest)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"input")
                return TransferResult(True, str(path), 5, 0.01, url=url)

            def fake_search(evaluate, **kwargs):
                return CrfSearchResult(True, 30.0, 90.0, reason="test")

            def fake_encode(input_path, output_path, **kwargs):
                if "final_x265" in output_path:
                    return EncodeResult(False, output_path, 1, "x265 failed", [])
                Path(output_path).write_bytes(b"probe")
                return EncodeResult(True, output_path, 0, "", [])

            with patch("batch_search.download_to_path", side_effect=fake_download), patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf", side_effect=fake_search
            ), patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ), patch(
                "batch_search.upload_presigned_put"
            ):
                results = run_fleet_sla(template, jobs)

            self.assertFalse(jobs[0].uploaded)
            self.assertIn("final encode failed", jobs[0].error)
            self.assertIsNone(results[0].best)

    def test_local_path_skips_download_and_upload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"local-input")
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "libx265_refine_preset": "superfast",
                    "time_budget_sec": 180,
                    "final_encode_reserve_sec": 90,
                    "probe_min_budget_sec": 10,
                    "crf_candidates": 2,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "local",
                            "input_path": str(src),
                            "output_path": str(out),
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_search(evaluate, **kwargs):
                return CrfSearchResult(True, 30.0, 90.0, reason="test")

            def fake_encode(input_path, output_path, **kwargs):
                Path(output_path).write_bytes(b"final-x265")
                return EncodeResult(True, output_path, 0, "", [])

            with patch("batch_search.download_to_path") as download_mock, patch(
                "batch_search.upload_presigned_put"
            ) as upload_mock, patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf", side_effect=fake_search
            ), patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ), patch(
                "batch_search._score_sla_final",
                side_effect=lambda req, job, trial, *, deadline: _final_score_trial(trial),
            ):
                results = run_fleet_sla(template, jobs)

            download_mock.assert_not_called()
            upload_mock.assert_not_called()
            self.assertTrue(jobs[0].uploaded)
            self.assertTrue(Path(jobs[0].output_path).is_file())
            self.assertEqual(Path(jobs[0].output_path).read_bytes(), b"final-x265")
            self.assertEqual(results[0].best.encoder, "libx265")
            self.assertEqual(jobs[0].stage_timings.get("download"), 0.0)
            self.assertEqual(jobs[0].stage_timings.get("upload"), 0.0)
            self.assertIn("final_score", jobs[0].stage_timings)
            self.assertEqual(results[0].best.score.reason, "success")
            self.assertEqual(results[0].best.score.compression_rate, 0.1)

    def test_fixed_crf_skips_search(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"local-input")
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "libx265_refine_preset": "superfast",
                    "time_budget_sec": 180,
                    "final_encode_reserve_sec": 90,
                    "probe_min_budget_sec": 10,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "fixed",
                            "input_path": str(src),
                            "output_path": str(out),
                            "crf": 28,
                            "libx265_params": "aq-mode=2:aq-strength=0.9",
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)
            encode_crfs: list[int] = []
            encode_params: list[str] = []

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_encode(input_path, output_path, **kwargs):
                encode_crfs.append(int(kwargs.get("crf") or -1))
                encode_params.append(str(kwargs.get("params") or ""))
                Path(output_path).write_bytes(b"fixed-encode")
                return EncodeResult(True, output_path, 0, "", [])

            def fake_score(*args, **kwargs):
                return _score(s_f=0.5, vmaf=88.0)

            with patch("batch_search.download_to_path") as download_mock, patch(
                "batch_search.upload_presigned_put"
            ) as upload_mock, patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf"
            ) as search_mock, patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.score_candidate", side_effect=fake_score
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ):
                results = run_fleet_sla(template, jobs)

            search_mock.assert_not_called()
            download_mock.assert_not_called()
            upload_mock.assert_not_called()
            self.assertEqual(jobs[0].error, "")
            self.assertEqual(jobs[0].chosen_crf, 28)
            self.assertEqual(encode_crfs, [28])  # one fixed encode; reused as final
            self.assertIn("aq-mode=2", encode_params[0])
            self.assertIn("aq-strength=0.9", encode_params[0])
            self.assertTrue(Path(jobs[0].output_path).is_file())
            self.assertEqual(Path(jobs[0].output_path).read_bytes(), b"fixed-encode")
            self.assertEqual(results[0].best.crf, 28)
            self.assertEqual(results[0].best.stage, "sla_final")

    def test_fixed_vbr_skips_crf_search_and_tunes_params(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"local-input")
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "codec_mode": "ABR",
                    "target_bitrate": "8M",
                    "libx265_refine_preset": "superfast",
                    "libx265_params": "aq-mode=2:aq-strength=1.0:rd=5:ref=4:bframes=4:rc-lookahead=40",
                    "param_tune": True,
                    "param_tune_max_trials": 3,
                    "param_tune_no_improve_stop": 10,
                    "param_tune_max_rounds": 1,
                    "vbr_mode_tune": False,
                    "time_budget_sec": 0,
                    "final_encode_reserve_sec": 0,
                    "probe_min_budget_sec": 0,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "vbr",
                            "input_path": str(src),
                            "output_path": str(out),
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)
            encode_calls: list[dict] = []
            score_n = {"n": 0}

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)
                job.features["texture_level"] = 0.9

            def fake_encode(input_path, output_path, **kwargs):
                encode_calls.append(
                    {
                        "crf": kwargs.get("crf"),
                        "bitrate": kwargs.get("bitrate"),
                        "codec_mode": kwargs.get("codec_mode"),
                        "params": str(kwargs.get("params") or ""),
                    }
                )
                Path(output_path).write_bytes(b"vbr-encode")
                return EncodeResult(True, output_path, 0, "", [])

            def fake_score(*args, **kwargs):
                score_n["n"] += 1
                # First score = baseline; later trials get higher s_f when aq-mode=1.
                params = ""
                # score_candidate doesn't get params; infer from encode order.
                s_f = 0.40 + 0.01 * score_n["n"]
                return _score(s_f=s_f, vmaf=88.0)

            with patch("batch_search.download_to_path") as download_mock, patch(
                "batch_search.upload_presigned_put"
            ) as upload_mock, patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf"
            ) as search_mock, patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.score_candidate", side_effect=fake_score
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ):
                results = run_fleet_sla(template, jobs)

            search_mock.assert_not_called()
            download_mock.assert_not_called()
            upload_mock.assert_not_called()
            self.assertEqual(jobs[0].error, "")
            self.assertEqual(jobs[0].chosen_bitrate, "8M")
            self.assertIsNone(jobs[0].chosen_crf)
            self.assertGreaterEqual(jobs[0].param_tune_trials, 1)
            self.assertTrue(all(c["bitrate"] == "8M" for c in encode_calls))
            self.assertTrue(all(c["crf"] is None for c in encode_calls))
            self.assertTrue(all(c["codec_mode"] == "ABR" for c in encode_calls))
            self.assertEqual(results[0].strategy, "fleet_sla_x265_vbr_param_tune")
            self.assertEqual(results[0].best.bitrate, "8M")
            self.assertEqual(results[0].best.stage, "sla_final")
            self.assertTrue(Path(jobs[0].output_path).is_file())
            best_path = Path(jobs[0].work_dir) / "best.json"
            self.assertTrue(best_path.is_file())
            import json

            best = json.loads(best_path.read_text(encoding="utf-8"))
            self.assertEqual(best["bitrate"], "8M")
            self.assertEqual(best["target_bitrate"], "8M")

    def test_vbr_mode_proxy_ladder_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"local-input")
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "codec_mode": "ABR",
                    "target_bitrate": "5M",
                    "libx265_refine_preset": "superfast",
                    "param_tune": True,
                    "vbr_mode_tune": True,
                    "vbr_mode_aq_min": 1.0,
                    "vbr_mode_aq_max": 1.2,
                    "vbr_mode_aq_step": 0.2,
                    "vbr_mode_rd_sweep": [5, 6],
                    "vbr_mode_bframes_sweep": [8, 12],
                    "vbr_mode_lookahead_sweep": [50, 60],
                    "preprocess_auto": False,
                    "preprocess_ab": False,
                    "preprocess": None,
                    "time_budget_sec": 0,
                    "final_encode_reserve_sec": 0,
                    "probe_min_budget_sec": 0,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "vbrm",
                            "input_path": str(src),
                            "output_path": str(out),
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)
            encode_inputs: list[str] = []

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_proxy(input_path, output_path, windows, **kwargs):
                Path(output_path).write_bytes(b"proxy-ref")
                return ProxyBuildResult(
                    ok=True,
                    path=str(output_path),
                    windows=list(windows),
                    total_seconds=2.0,
                )

            def fake_encode(input_path, output_path, **kwargs):
                encode_inputs.append(str(input_path))
                Path(output_path).write_bytes(b"vbr-encode")
                return EncodeResult(True, output_path, 0, "", [])

            score_n = {"n": 0}

            def fake_score(*args, **kwargs):
                score_n["n"] += 1
                return _score(s_f=0.40 + 0.005 * score_n["n"], vmaf=88.0)

            with patch("batch_search.download_to_path"), patch(
                "batch_search.upload_presigned_put"
            ), patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.build_proxy_reference", side_effect=fake_proxy
            ), patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.score_candidate", side_effect=fake_score
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ), patch(
                "batch_search._measured_bitrate_mbps", return_value=5.0
            ), patch(
                "batch_search.measure_compression", return_value=(0.1, 10.0)
            ):
                results = run_fleet_sla(template, jobs)

            self.assertEqual(jobs[0].error, "", msg=jobs[0].error)
            self.assertIsNotNone(results[0].best)
            self.assertEqual(results[0].strategy, "fleet_sla_x265_vbr_mode")
            # none + baseline already clears VMAF → stop without param ladder.
            self.assertEqual(jobs[0].param_tune_trials, 1)
            labels = [str(h.get("label") or "") for h in jobs[0].param_tune_history]
            self.assertTrue(any(l.startswith("pp=") for l in labels))
            self.assertFalse(any(l.startswith("aq=") for l in labels))
            # Proxy search encodes use proxy path; final encode uses source.
            self.assertTrue(any("vbr_mode_proxy" in p for p in encode_inputs))
            self.assertTrue(any(str(src) == p for p in encode_inputs))
            self.assertEqual(results[0].best.stage, "sla_final")
            self.assertIsNotNone(jobs[0].best_params)
            self.assertIn("aq-strength=", jobs[0].best_params or "")

    def test_abr_requires_target_bitrate_or_rate(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            CompressionRequest.from_dict(
                {
                    "encoder": "libx265",
                    "codec_mode": "ABR",
                    "jobs": [
                        {
                            "id": "x",
                            "input_path": "/tmp/x.mp4",
                            "output_path": "/tmp/y.mp4",
                        }
                    ],
                }
            )
        msg = str(ctx.exception)
        self.assertTrue(
            "target_bitrate" in msg or "target_compression_rate" in msg,
            msg,
        )

    def test_abr_accepts_target_compression_rate(self) -> None:
        req = CompressionRequest.from_dict(
            {
                "encoder": "libx265",
                "codec_mode": "ABR",
                "target_compression_rate": 0.035,
                "jobs": [
                    {
                        "id": "x",
                        "input_path": "/tmp/x.mp4",
                        "output_path": "/tmp/y.mp4",
                    }
                ],
            }
        )
        self.assertEqual(req.target_compression_rate, 0.035)
        self.assertTrue(req.vbr_fallback_to_crf)

    def test_score_sla_final_uses_source_size_and_dual_vmaf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            out = Path(td) / "out.mp4"
            src.write_bytes(b"x" * 1000)
            out.write_bytes(b"y" * 100)
            job = fleet_jobs_from_request(
                CompressionRequest.from_dict(
                    {
                        "work_dir": td,
                        "jobs": [
                            {
                                "id": "t",
                                "input_path": str(src),
                                "output_path": str(out),
                            }
                        ],
                    }
                )
            )[0]
            trial = TrialResult(
                recipe="default_superfast",
                mode="RC",
                crf=32,
                bitrate=None,
                path=str(out),
                score=_score(),
                encode_ok=True,
                stage="sla_final",
                encoder="libx265",
                encode_sec=1.0,
            )
            req = CompressionRequest.from_dict(
                {
                    "work_dir": td,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {"id": "t", "input_path": str(src), "output_path": str(out)}
                    ],
                }
            )

            def fake_score_candidate(reference_path, distorted_path, vmaf_threshold, **kwargs):
                self.assertEqual(reference_path, job.input_path)
                self.assertEqual(distorted_path, str(out))
                rate = out.stat().st_size / src.stat().st_size
                return ScoreResult(
                    s_f=0.55,
                    vmaf=91.0,
                    compression_rate=rate,
                    compression_ratio=1.0 / rate,
                    compression_component=0.8,
                    quality_component=0.7,
                    reason="success",
                    validation_errors=[],
                    vmaf_base=92.0,
                    vmaf_delta=1.0,
                    passed_encoding_gates=True,
                    passed_vmaf_delta_gate=True,
                )

            from batch_search import _score_sla_final
            import time

            with patch("batch_search.score_candidate", side_effect=fake_score_candidate), patch(
                "batch_search._measured_bitrate_mbps", return_value=2.5
            ):
                scored = _score_sla_final(
                    req, job, trial, deadline=time.monotonic() + 30
                )
            self.assertAlmostEqual(scored.score.compression_rate, 0.1, places=5)
            self.assertEqual(scored.score.vmaf, 91.0)
            self.assertEqual(scored.score.vmaf_base, 92.0)
            self.assertGreaterEqual(scored.score_sec, 0.0)
            self.assertEqual(scored.measured_bitrate_mbps, 2.5)

    def test_reserve_budget_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserves exceed"):
            CompressionRequest.from_dict(
                {
                    "time_budget_sec": 50,
                    "download_reserve_sec": 25,
                    "final_encode_reserve_sec": 20,
                    "upload_reserve_sec": 20,
                    "probe_min_budget_sec": 10,
                    "jobs": [
                        {
                            "input_url": "https://a/x",
                            "upload_url": "https://b/y",
                        }
                    ],
                }
            )




    def test_vbr_derives_bitrate_from_compression_rate(self) -> None:
        from compress_util import bitrate_for_compression_rate, parse_bitrate_mbps

        # 100_000_000 bytes, 30s, rate 0.035 → bits=28e6 → bps≈933333 → ~0.933M
        br = bitrate_for_compression_rate(
            source_bytes=100_000_000,
            duration_sec=30.0,
            compression_rate=0.035,
        )
        mbps = parse_bitrate_mbps(br)
        self.assertIsNotNone(mbps)
        self.assertAlmostEqual(mbps, 0.933333, places=3)

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            # ~10MB placeholder (size only; encode is mocked)
            src.write_bytes(b"x" * 10_000_000)
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "codec_mode": "ABR",
                    "target_compression_rate": 0.035,
                    "vbr_fallback_to_crf": False,
                    "libx265_refine_preset": "superfast",
                    "libx265_params": "aq-mode=2:aq-strength=1.0:rd=5:ref=4:bframes=4:rc-lookahead=40",
                    "param_tune": False,
                    "time_budget_sec": 0,
                    "final_encode_reserve_sec": 0,
                    "probe_min_budget_sec": 0,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "rate",
                            "input_path": str(src),
                            "output_path": str(out),
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)
            encode_calls: list[dict] = []

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_encode(input_path, output_path, **kwargs):
                encode_calls.append(
                    {
                        "bitrate": kwargs.get("bitrate"),
                        "codec_mode": kwargs.get("codec_mode"),
                        "crf": kwargs.get("crf"),
                    }
                )
                Path(output_path).write_bytes(b"vbr-encode")
                return EncodeResult(True, output_path, 0, "", [])

            with patch("batch_search.download_to_path"), patch(
                "batch_search.upload_presigned_put"
            ), patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf"
            ) as search_mock, patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.score_candidate",
                return_value=_score(s_f=0.5, vmaf=88.0),
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ):
                results = run_fleet_sla(template, jobs)

            search_mock.assert_not_called()
            self.assertEqual(jobs[0].error, "")
            self.assertIsNotNone(jobs[0].chosen_bitrate)
            self.assertTrue(all(c["codec_mode"] == "ABR" for c in encode_calls))
            self.assertTrue(all(c["crf"] is None for c in encode_calls))
            # 10MB / 30s / 0.035 → ~0.093M
            mbps = parse_bitrate_mbps(jobs[0].chosen_bitrate)
            self.assertIsNotNone(mbps)
            self.assertAlmostEqual(mbps, 0.093333, places=3)
            self.assertEqual(results[0].strategy, "fleet_sla_x265_fixed_vbr")

    def test_vbr_falls_back_to_crf_when_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"local-input")
            out = Path(td) / "out.mp4"
            work = Path(td) / "work"
            template = CompressionRequest.from_dict(
                {
                    "work_dir": str(work),
                    "skip_transfer": True,
                    "encoder": "libx265",
                    "codec_mode": "ABR",
                    "target_bitrate": "1M",
                    "vbr_fallback_to_crf": True,
                    "scene_crf_search": True,
                    "libx265_refine_preset": "superfast",
                    "libx265_params": "aq-mode=2:aq-strength=1.0:rd=5:ref=4:bframes=4:rc-lookahead=40",
                    "param_tune": False,
                    "time_budget_sec": 0,
                    "final_encode_reserve_sec": 0,
                    "probe_min_budget_sec": 0,
                    "vmaf_threshold": 85,
                    "jobs": [
                        {
                            "id": "fb",
                            "input_path": str(src),
                            "output_path": str(out),
                        }
                    ],
                }
            )
            jobs = fleet_jobs_from_request(template)
            modes: list[str] = []

            def fake_extract(job, sample_frames, *, deadline):
                _set_feature_scenes(job, duration=30.0, difficulty=0.4)

            def fake_encode(input_path, output_path, **kwargs):
                modes.append(str(kwargs.get("codec_mode")))
                Path(output_path).write_bytes(b"enc")
                return EncodeResult(True, output_path, 0, "", [])

            def fake_score(*args, **kwargs):
                # VBR probe stays below threshold so fallback triggers.
                return _score(s_f=0.05, vmaf=82.0)

            def fake_search(evaluate, **kwargs):
                return CrfSearchResult(True, 30.0, 90.0, reason="test")

            with patch("batch_search.download_to_path"), patch(
                "batch_search.upload_presigned_put"
            ), patch(
                "batch_search._extract_sla_features", side_effect=fake_extract
            ), patch(
                "batch_search.search_crf", side_effect=fake_search
            ), patch(
                "batch_search.encode_hevc", side_effect=fake_encode
            ), patch(
                "batch_search.score_candidate", side_effect=fake_score
            ), patch(
                "batch_search.validate_hevc_output",
                return_value=EncodeValidation(True, [], {}),
            ), patch(
                "batch_search._score_sla_final",
                side_effect=lambda req, job, trial, *, deadline: _final_score_trial(trial),
            ):
                results = run_fleet_sla(template, jobs)

            self.assertEqual(jobs[0].error, "")
            self.assertTrue(jobs[0].vbr_fell_back_to_crf)
            self.assertIsNone(jobs[0].chosen_bitrate)
            self.assertEqual(jobs[0].chosen_crf, 30)
            self.assertIn("ABR", modes)
            self.assertIn("RC", modes)
            self.assertTrue(
                results[0].strategy.startswith("fleet_sla_x265_vbr_fallback_crf")
            )


class ScoringTimeoutTests(unittest.TestCase):
    def test_score_candidate_timeout_returns_zero(self) -> None:
        from scoring import score_candidate

        with patch(
            "scoring.validate_validator_gates",
            side_effect=TimeoutError("candidate scoring deadline exhausted"),
        ):
            result = score_candidate(
                "ref.mp4",
                "dist.mp4",
                85,
                timeout=0.01,
            )
        self.assertEqual(result.s_f, 0.0)
        self.assertIn("scoring_timeout", result.reason)



class SkipPublishedTests(unittest.TestCase):
    def test_filter_pending_skips_published_and_seeds_work(self) -> None:
        from batch_search import (
            FleetVideoJob,
            filter_pending_fleet_jobs,
            is_usable_best_payload,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pub = root / "published_results" / "85" / "v1"
            pub.mkdir(parents=True)
            best = {
                "job_id": "v1",
                "crf": 30,
                "s_f": 0.5,
                "error": None,
                "libx265_params": "aq-mode=1",
            }
            self.assertTrue(is_usable_best_payload(best))
            (pub / "best.json").write_text(json.dumps(best), encoding="utf-8")
            (pub / "result.json").write_text("{}", encoding="utf-8")

            work = root / "work_fleet" / "85"
            jobs = [
                FleetVideoJob(
                    job_id="v1",
                    input_path="/ephemeral/videos/1.mp4",
                    output_path=str(root / "out" / "v1.mp4"),
                    work_dir=str(work / "v1"),
                ),
                FleetVideoJob(
                    job_id="v2",
                    input_path="/ephemeral/videos/2.mp4",
                    output_path=str(root / "out" / "v2.mp4"),
                    work_dir=str(work / "v2"),
                ),
            ]
            pending, skipped = filter_pending_fleet_jobs(
                jobs,
                threshold=85,
                published_root=str(root / "published_results"),
                skip_complete=True,
                seed_work=True,
            )
            self.assertEqual([j.job_id for j in pending], ["v2"])
            self.assertEqual([j.job_id for j in skipped], ["v1"])
            self.assertTrue((work / "v1" / "best.json").is_file())
            self.assertTrue((work / "v1" / "result.json").is_file())

    def test_force_keeps_all_jobs(self) -> None:
        from batch_search import FleetVideoJob, filter_pending_fleet_jobs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pub = root / "published_results" / "85" / "v1"
            pub.mkdir(parents=True)
            (pub / "best.json").write_text(
                json.dumps({"job_id": "v1", "crf": 30, "s_f": 0.5, "error": None}),
                encoding="utf-8",
            )
            jobs = [
                FleetVideoJob(
                    job_id="v1",
                    input_path="/ephemeral/videos/1.mp4",
                    output_path=str(root / "out" / "v1.mp4"),
                    work_dir=str(root / "work" / "v1"),
                )
            ]
            pending, skipped = filter_pending_fleet_jobs(
                jobs,
                threshold=85,
                published_root=str(root / "published_results"),
                skip_complete=False,
            )
            self.assertEqual(len(pending), 1)
            self.assertEqual(skipped, [])

if __name__ == "__main__":
    unittest.main()
