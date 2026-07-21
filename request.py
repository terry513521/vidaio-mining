"""Compression challenge request body (dict / JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from encode_mode import is_abr_mode, is_rc_mode, normalize_codec_mode, nvenc_rc_ok_for_abr
from encoder import _PREPROCESS_FILTERS

_X265_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
}
_NVENC_PRESETS = {f"p{i}" for i in range(1, 8)}
_LIBX265_PROFILES = {"main", "main10", "mainstillpicture", "rext"}


@dataclass
class CompressionRequest:
    """Parameters a validator-like caller would send to the compressor."""

    input_path: str = ""
    output_path: str = "compressed.mp4"
    # Fleet job list. Each entry is either:
    #   remote: {id, input_url, upload_url}  (presigned HTTP PUT)
    #   local:  {id, input_path, output_path?}
    # Optional per-job overrides: crf, libx265_params, target_bitrate,
    # target_compression_rate.
    jobs: list[dict[str, Any]] = field(default_factory=list)
    # Local testing: skip HTTP download/upload even if URL fields exist.
    skip_transfer: bool = False

    # Vidaio-style challenge knobs
    vmaf_threshold: int = 89  # 85 | 89 | 93
    codec: str = "hevc"  # HEVC-only for now
    # RC = constant quality (search CQ/CRF). ABR = average bitrate (search -b:v).
    # Aliases: CRF/CQ → RC; VBR/BITRATE → ABR. NVENC rate-control flag is nvenc_rc.
    codec_mode: str = "RC"  # RC | ABR
    target_bitrate: Optional[str] = None  # e.g. "8M" when ABR
    # When ABR and target_bitrate is unset: derive -b:v so expected
    # compression_rate ≈ this value (e.g. 0.035). Per-job override allowed.
    target_compression_rate: Optional[float] = None
    # After VBR probe+tune, if VMAF < threshold (or score unusable), fall back
    # to the CRF search path for that job.
    vbr_fallback_to_crf: bool = True

    # Encoder backend: CPU libx265 or GPU hevc_nvenc
    encoder: str = "libx265"  # libx265 | hevc_nvenc

    # Optional: fast NVENC search, then a small libx265 refine before output.
    # Only applies when encoder=hevc_nvenc. Winner is highest s_f across both.
    libx265_refine: bool = False
    libx265_refine_preset: str = "medium"  # libx265 preset for refine encodes
    libx265_refine_candidates: int = 3
    libx265_refine_crf_spread: int = 2  # RC refine CRF spacing around seed
    libx265_refine_max_workers: int = 2
    libx265_refine_time_sec: float = 60.0  # reserved from time_budget_sec
    # Legacy CQ→CRF offset (unused by VMAF-anchored refine; kept for overrides/tests).
    libx265_cq_to_crf_offset: int = 0
    libx265_crf_min: Optional[int] = None  # x265 CRF floor (defaults to crf_min)
    libx265_crf_max: Optional[int] = None  # x265 CRF ceiling (defaults to crf_max)
    # Budget guards for the x265 refine stage.
    libx265_refine_min_budget_sec: float = 30.0  # skip refine below this
    libx265_refine_sec_per_candidate: float = 90.0  # est. per-encode cost

    # Search / runtime
    # 0 = unlimited (no wall-clock deadline for fleet SLA waves).
    time_budget_sec: float = 600.0
    max_search_steps: int = 8  # ABR step budget; RC uses crf_candidates
    max_recipes: int = 1
    max_workers: int = 3
    # Real-world fleet: process N videos in parallel, one CQ probe per video
    # per wave (no parallel CQ sweep on a single video).
    fleet_batch_size: int = 5
    # How many fleet jobs may use GPU (NVENC + libvmaf_cuda) at once; rest are CPU-only.
    fleet_gpu_slots: int = 1
    serial_cq_search: bool = False  # one CQ per round per video (set True for fleet)
    # Hard end-to-end SLA reserves. Ignored when time_budget_sec=0 (unlimited).
    download_reserve_sec: float = 25.0
    final_encode_reserve_sec: float = 90.0
    upload_reserve_sec: float = 20.0
    probe_min_budget_sec: float = 20.0
    # libx265: ultrafast..placebo; hevc_nvenc: p1..p7 (or same x265 names, mapped)
    preset: str = "medium"
    crf_candidates: int = 3
    crf_spread: int = 2
    vbr_max_ratio_to_target: float = 1.1
    vbr_min_mbps_floor: float = 0.5
    crf_min: int = 8
    crf_max: int = 40
    crf_start: Optional[int] = None  # optional search seed; else feature/recipe default
    # Fixed CRF: if set, skip CRF search — one full encode + dual VMAF at this CRF.
    # Control quality knobs via libx265_params / libx265_feature_baseline / preset.
    crf: Optional[int] = None
    # CQ/CRF search strategy: parallel_grid (spread around seed) or
    # interp_answer (feature-seeded round1 + interpolated answer-based round2).
    search_strategy: str = "parallel_grid"  # parallel_grid | interp_answer
    search_rounds: int = 2  # used by interp_answer (typically 2)
    # Round 2 split (interp_answer + hevc_nvenc): tune NVENC at best CQ, then refine CQ.
    round2_nvenc_param_trials: int = 3
    round2_cq_trials: int = 2
    # Before Round 1: set NVENC baseline from features (never touches CQ).
    nvenc_feature_baseline: bool = True
    # When building libx265 recipes (primary or refine): set -x265-params from
    # features (never touches CRF).
    libx265_feature_baseline: bool = True
    # libx265 ffmpeg knobs (ignored for hevc_nvenc). libx265_params overlays
    # feature-derived params (request keys win on conflict).
    libx265_profile: str = "main"  # main | main10 | mainstillpicture | rext
    libx265_params: Optional[str] = None  # colon-joined -x265-params overlay
    # Optional preprocess (survey set: denoise / bilateral / mild sharpen / contrast).
    # none | hqdn3d_light | hqdn3d_med | atadenoise_light | bilateral_light |
    # unsharp_mild | contrast_mild
    preprocess: Optional[str] = None
    # When preprocess is unset: pick from features (VBR path).
    preprocess_auto: bool = True
    # Compare multiple candidates at the same bitrate and keep the best dual-VMAF
    # / s_f result (VBR path). With a single auto pick this is none vs pick;
    # with preprocess_sweep it evaluates the full survey set.
    preprocess_ab: bool = True
    # Try the full VMAF-NEG survey preprocess set at fixed bitrate (costly).
    preprocess_sweep: bool = False
    # Round 2: add one measured light-denoise trial at the locked best CQ.
    round2_preprocess_trial: bool = False
    round2_preprocess: str = "hqdn3d_light"

    # NVENC-only knobs (ignored for libx265). RC mode maps crf → -cq.
    nvenc_tune: str = "hq"  # hq | ll | ull | lossless
    nvenc_rc: str = "vbr"  # vbr | vbr_hq | cbr | cbr_hq | constqp
    nvenc_multipass: str = "qres"  # disabled | qres | fullres
    nvenc_spatial_aq: bool = True
    nvenc_temporal_aq: bool = True
    nvenc_aq_strength: int = 8  # 1..15
    nvenc_rc_lookahead: int = 0  # 0 = omit (ffmpeg default); else -rc-lookahead N
    nvenc_bf: int = 0  # max B-frames (-bf); 0 is ffmpeg default
    nvenc_gop: Optional[int] = None  # None = omit -g; else GOP size
    nvenc_b_ref_mode: str = "disabled"  # disabled | each | middle
    nvenc_gpu: int = 0
    nvenc_hwaccel: bool = False  # CUDA decode before NVENC (optional)

    # Scene-based CRF search (feature-extractor cuts + samples + ab-av1 bisection).
    scene_crf_search: bool = True
    # Legacy pick caps; fleet path samples every scene then concats.
    scene_max_samples: int = 12
    # Legacy; unused when cuts come from feature extraction.
    scene_detect_downscale: int = 1
    crf_search_increment: float = 0.1
    crf_search_max_runs: int = 12
    crf_search_max_encoded_percent: float = 80.0
    crf_search_thorough: bool = False
    # When |probe VMAF - threshold| ≤ this, step CRF by ±1 instead of interpolating.
    crf_search_near_vmaf_band: float = 2.0
    crf_search_samples: Optional[int] = None
    crf_search_sample_every_sec: float = 720.0
    # Legacy min-sample count; fleet path no longer picks a subset of scenes.
    crf_search_min_samples: int = 1
    # Frame subsampling for dual-model VMAF during scene CRF search.
    crf_search_vmaf_n_subsample: int = 4
    crf_search_probe_preset: str = "ultrafast"
    # After CRF search: sequential libx265 param tune maximizing s_f.
    # Skipped in fixed-CRF manual mode (when crf is set).
    # When crf_mode_tune=True (default), CRF path uses aq×CRF Phase B/C instead.
    param_tune: bool = True
    param_tune_max_trials: int = 25
    param_tune_no_improve_stop: int = 10
    param_tune_vmaf_headroom: float = 2.0  # try CRF+1 when vmaf - threshold >= this
    param_tune_max_rounds: int = 3
    # CRF mode: fixed pack (ref=16, rd=6, bframes=12, aq=1, la=50) + aq-mode rule,
    # then Phase B aq-strength walk with CRF+1/+2 ladder, then Phase C la∈{40,60}.
    crf_mode_tune: bool = True
    crf_mode_pack: bool = True
    crf_mode_aq_min: float = 0.2
    crf_mode_aq_max: float = 2.4
    crf_mode_aq_step: float = 0.2
    crf_mode_vmaf_headroom: float = 2.0
    crf_mode_compensate_steps: int = 2  # CRF± steps after each AQ trial
    # Challenger must also beat this size rate to replace best (None = disabled).
    crf_mode_max_compression_rate: Optional[float] = None
    crf_mode_lookahead_default: int = 50
    crf_mode_lookahead_sweep: tuple[int, ...] = (40, 60)
    # VBR mode: proxy (2s/scene) preprocess arms + aq→rd→bframes→lookahead ladder.
    # When True (default on ABR), replaces legacy sequential param_tune for VBR.
    vbr_mode_tune: bool = True
    vbr_mode_aq_min: float = 0.2
    vbr_mode_aq_max: float = 2.6
    vbr_mode_aq_step: float = 0.2
    vbr_mode_rd_sweep: tuple[int, ...] = (4, 5, 6)
    vbr_mode_bframes_sweep: tuple[int, ...] = (6, 8, 12)
    vbr_mode_lookahead_sweep: tuple[int, ...] = (40, 50, 60)
    vbr_mode_proxy_seconds_per_scene: float = 2.0
    # Cap parallel fleet jobs during VBR-mode proxy search (OOM guard).
    # Each job runs libx265 + VMAF; 5-wide waves easily OOM on ~50GB hosts.
    vbr_mode_max_parallel: int = 2
    # Legacy proxy search (mashup seconds-per-segment). Used when scene_crf_search=false.
    use_proxy: bool = True
    proxy_seconds_per_segment: float = 2.5
    proxy_max_seconds: float = 15.0
    proxy_min_window_seconds: float = 0.5
    proxy_lossless: bool = True
    # Target proxy VMAF ≈ vmaf_threshold + margin (full-file usually scores lower).
    proxy_vmaf_margin: float = 2.0
    # Do not push CRF up when feature mashup score is at/above this (v1-style risk).
    proxy_mashup_push_ceiling: float = 0.55

    # Feature / VMAF
    # Legacy quick-sample budget; fleet/SLA features use full-video scan instead.
    sample_frames: int = 60
    # Scene CRF search: fixed-duration clip from each selected scene midpoint.
    sample_seconds_per_scene: float = 3.0
    vmaf_n_subsample: int = 1
    vmaf_n_threads: int = 4
    # Encode uses native ffmpeg_bin; local VMAF scoring is Docker-only by default
    # (validator image vmaf_ffmpeg). Pass vmaf_backend=native only as a fallback.
    vmaf_backend: str = "docker"  # docker | native
    vmaf_docker_image: str = "vmaf_ffmpeg"
    vmaf_docker_gpus: bool = False  # True → libvmaf_cuda (needs NVIDIA Container Toolkit)

    work_dir: str = "work"
    keep_candidates: bool = False

    ffmpeg_bin: Optional[str] = None
    ffprobe_bin: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.codec = self.codec.lower().strip()
        self.codec_mode = self.codec_mode.upper().strip()

        if self.codec not in {"hevc", "h265", "x265"}:
            raise ValueError(f"Only HEVC is supported currently, got codec={self.codec!r}")

        self.codec = "hevc"

        self.codec_mode = normalize_codec_mode(self.codec_mode)

        if self.vmaf_threshold not in {85, 89, 93}:
            # Allow custom, but warn via normalization to nearest typical set
            if not (0 < self.vmaf_threshold <= 100):
                raise ValueError(f"vmaf_threshold out of range: {self.vmaf_threshold}")

        if self.codec_mode == "ABR":
            if self.target_bitrate is not None:
                self.target_bitrate = str(self.target_bitrate).strip() or None
            if self.target_compression_rate is not None:
                rate = float(self.target_compression_rate)
                if not (0.0 < rate < 1.0):
                    raise ValueError(
                        "target_compression_rate must be in (0, 1), "
                        f"got {self.target_compression_rate!r}"
                    )
                self.target_compression_rate = rate
            job_has_bitrate = any(
                isinstance(j, dict)
                and (
                    str(j.get("target_bitrate") or "").strip()
                    or j.get("target_compression_rate") is not None
                )
                for j in (self.jobs or [])
            )
            if (
                not self.target_bitrate
                and self.target_compression_rate is None
                and not job_has_bitrate
            ):
                raise ValueError(
                    "ABR/VBR requires target_bitrate, target_compression_rate, "
                    "or a per-job bitrate/rate override"
                )

        if self.crf_min > self.crf_max:
            raise ValueError("crf_min must be <= crf_max")
        if self.crf is not None:
            self.crf = int(self.crf)
            if not (0 <= self.crf <= 51):
                raise ValueError("crf must be in 0..51 when set")
        if self.crf_start is not None:
            self.crf_start = int(self.crf_start)
            if not (0 <= self.crf_start <= 51):
                raise ValueError("crf_start must be in 0..51 when set")
        if self.vbr_max_ratio_to_target <= 0:
            raise ValueError("vbr_max_ratio_to_target must be > 0")
        if self.vbr_min_mbps_floor <= 0:
            raise ValueError("vbr_min_mbps_floor must be > 0")
        if self.crf_candidates < 1:
            raise ValueError("crf_candidates must be >= 1")
        if self.crf_spread < 1:
            raise ValueError("crf_spread must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.fleet_batch_size < 1:
            raise ValueError("fleet_batch_size must be >= 1")
        if self.fleet_gpu_slots < 0:
            raise ValueError("fleet_gpu_slots must be >= 0")
        if self.time_budget_sec < 0:
            raise ValueError("time_budget_sec must be >= 0 (0 = unlimited)")
        for name in (
            "download_reserve_sec",
            "final_encode_reserve_sec",
            "upload_reserve_sec",
            "probe_min_budget_sec",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        normalized_jobs: list[dict[str, Any]] = []
        for index, raw in enumerate(self.jobs):
            if not isinstance(raw, dict):
                raise ValueError(f"jobs[{index}] must be an object")
            job_id = str(raw.get("id") or f"video-{index + 1}").strip()
            input_url = str(raw.get("input_url") or "").strip()
            upload_url = str(raw.get("upload_url") or "").strip()
            input_path = str(raw.get("input_path") or "").strip()
            output_path = str(raw.get("output_path") or "").strip()
            overrides: dict[str, Any] = {}
            if "crf" in raw and raw.get("crf") is not None:
                crf = int(raw["crf"])
                if not (0 <= crf <= 51):
                    raise ValueError(f"jobs[{index}].crf must be in 0..51 when set")
                overrides["crf"] = crf
            params = raw.get("libx265_params", raw.get("x265_params"))
            if params is not None:
                params = str(params).strip()
                if params:
                    overrides["libx265_params"] = params
            job_bitrate = raw.get("target_bitrate", raw.get("bitrate"))
            if job_bitrate is not None:
                job_bitrate = str(job_bitrate).strip()
                if job_bitrate:
                    overrides["target_bitrate"] = job_bitrate
            if raw.get("target_compression_rate") is not None:
                job_rate = float(raw["target_compression_rate"])
                if not (0.0 < job_rate < 1.0):
                    raise ValueError(
                        f"jobs[{index}].target_compression_rate must be in (0, 1)"
                    )
                overrides["target_compression_rate"] = job_rate
            has_remote = bool(input_url or upload_url)
            has_local = bool(input_path)
            if has_remote and has_local:
                raise ValueError(
                    f"jobs[{index}] must use either remote URLs or local paths, not both"
                )
            if has_remote:
                if not input_url.startswith(("http://", "https://")):
                    raise ValueError(f"jobs[{index}].input_url must be HTTP(S)")
                if not upload_url.startswith(("http://", "https://")):
                    raise ValueError(f"jobs[{index}].upload_url must be HTTP(S)")
                normalized_jobs.append(
                    {
                        "id": job_id,
                        "input_url": input_url,
                        "upload_url": upload_url,
                        **overrides,
                    }
                )
            elif has_local:
                normalized_jobs.append(
                    {
                        "id": job_id,
                        "input_path": str(Path(input_path).expanduser()),
                        "output_path": (
                            str(Path(output_path).expanduser()) if output_path else ""
                        ),
                        **overrides,
                    }
                )
            else:
                raise ValueError(
                    f"jobs[{index}] needs input_url+upload_url or input_path"
                )
        self.jobs = normalized_jobs
        if self.jobs and all("input_path" in job for job in self.jobs):
            # Pure local batch: no network I/O budget needed.
            self.skip_transfer = True
        if self.skip_transfer:
            self.download_reserve_sec = 0.0
            self.upload_reserve_sec = 0.0
        reserved = (
            float(self.download_reserve_sec)
            + float(self.final_encode_reserve_sec)
            + float(self.upload_reserve_sec)
            + float(self.probe_min_budget_sec)
        )
        if float(self.time_budget_sec) > 0 and reserved > float(self.time_budget_sec):
            raise ValueError(
                "download/final/upload/probe reserves exceed time_budget_sec "
                f"({reserved:.1f}s > {self.time_budget_sec:.1f}s)"
            )

        self.search_strategy = self.search_strategy.lower().strip()
        if self.search_strategy not in {"parallel_grid", "interp_answer"}:
            raise ValueError(
                f"search_strategy must be parallel_grid or interp_answer, "
                f"got {self.search_strategy!r}"
            )
        if self.search_rounds < 1:
            raise ValueError("search_rounds must be >= 1")
        if self.round2_nvenc_param_trials < 0:
            raise ValueError("round2_nvenc_param_trials must be >= 0")
        if self.round2_cq_trials < 0:
            raise ValueError("round2_cq_trials must be >= 0")
        if self.round2_nvenc_param_trials + self.round2_cq_trials < 1:
            raise ValueError(
                "round2_nvenc_param_trials + round2_cq_trials must be >= 1"
            )

        self.encoder = self.encoder.lower().strip()
        if self.encoder in {"x265"}:
            self.encoder = "libx265"
        if self.encoder in {"nvenc", "nvenc_hevc"}:
            self.encoder = "hevc_nvenc"
        if self.encoder not in {"libx265", "hevc_nvenc"}:
            raise ValueError(
                f"encoder must be libx265 or hevc_nvenc, got {self.encoder!r}"
            )

        self.preset = self.preset.lower().strip()
        if self.encoder == "libx265":
            if self.preset not in _X265_PRESETS:
                raise ValueError(
                    f"preset must be one of {sorted(_X265_PRESETS)}, got {self.preset!r}"
                )
        else:
            if self.preset not in _X265_PRESETS | _NVENC_PRESETS:
                raise ValueError(
                    f"NVENC preset must be p1..p7 or a libx265 preset name, got {self.preset!r}"
                )

        self.libx265_refine_preset = self.libx265_refine_preset.lower().strip()
        if self.libx265_refine_preset not in _X265_PRESETS:
            raise ValueError(
                "libx265_refine_preset must be one of "
                f"{sorted(_X265_PRESETS)}, got {self.libx265_refine_preset!r}"
            )
        if self.libx265_refine_candidates < 1:
            raise ValueError("libx265_refine_candidates must be >= 1")
        if self.libx265_refine_crf_spread < 1:
            raise ValueError("libx265_refine_crf_spread must be >= 1")
        if self.libx265_refine_max_workers < 1:
            raise ValueError("libx265_refine_max_workers must be >= 1")
        if self.libx265_refine_time_sec < 0:
            raise ValueError("libx265_refine_time_sec must be >= 0")
        if self.libx265_refine_min_budget_sec < 0:
            raise ValueError("libx265_refine_min_budget_sec must be >= 0")
        if self.libx265_refine_sec_per_candidate <= 0:
            raise ValueError("libx265_refine_sec_per_candidate must be > 0")
        if self.libx265_crf_min is not None and not (0 <= self.libx265_crf_min <= 51):
            raise ValueError("libx265_crf_min must be in 0..51")
        if self.libx265_crf_max is not None and not (0 <= self.libx265_crf_max <= 51):
            raise ValueError("libx265_crf_max must be in 0..51")
        if (
            self.libx265_crf_min is not None
            and self.libx265_crf_max is not None
            and self.libx265_crf_min > self.libx265_crf_max
        ):
            raise ValueError("libx265_crf_min must be <= libx265_crf_max")
        if self.libx265_refine and self.encoder != "hevc_nvenc":
            # No-op when already on libx265; keep flag but skip at runtime.
            pass

        self.libx265_profile = self.libx265_profile.lower().strip()
        if self.libx265_profile not in _LIBX265_PROFILES:
            raise ValueError(
                "libx265_profile must be one of "
                f"{sorted(_LIBX265_PROFILES)}, got {self.libx265_profile!r}"
            )
        if self.libx265_params is not None:
            self.libx265_params = str(self.libx265_params).strip() or None

        self.nvenc_tune = self.nvenc_tune.lower().strip()
        if self.nvenc_tune not in {"hq", "ll", "ull", "lossless"}:
            raise ValueError(
                f"nvenc_tune must be hq|ll|ull|lossless, got {self.nvenc_tune!r}"
            )
        self.nvenc_rc = self.nvenc_rc.lower().strip()
        if self.nvenc_rc not in {"vbr", "vbr_hq", "cbr", "cbr_hq", "constqp"}:
            raise ValueError(
                f"nvenc_rc must be vbr|vbr_hq|cbr|cbr_hq|constqp, got {self.nvenc_rc!r}"
            )
        if self.codec_mode == "ABR" and self.encoder == "hevc_nvenc":
            if not nvenc_rc_ok_for_abr(self.nvenc_rc):
                raise ValueError(
                    "ABR mode with hevc_nvenc requires nvenc_rc vbr|cbr|cbr_hq|vbr_hq "
                    f"(bitrate target), not {self.nvenc_rc!r}"
                )
        if self.codec_mode == "RC" and self.encoder == "hevc_nvenc":
            if self.nvenc_rc in {"cbr", "cbr_hq"}:
                raise ValueError(
                    "RC mode with hevc_nvenc expects quality-driven nvenc_rc "
                    f"(vbr|vbr_hq|constqp), not {self.nvenc_rc!r}"
                )
        self.nvenc_multipass = self.nvenc_multipass.lower().strip()
        if self.nvenc_multipass not in {"disabled", "qres", "fullres"}:
            raise ValueError(
                f"nvenc_multipass must be disabled|qres|fullres, got {self.nvenc_multipass!r}"
            )
        if not (1 <= int(self.nvenc_aq_strength) <= 15):
            raise ValueError("nvenc_aq_strength must be in 1..15")
        self.nvenc_rc_lookahead = int(self.nvenc_rc_lookahead)
        if self.nvenc_rc_lookahead < 0:
            raise ValueError("nvenc_rc_lookahead must be >= 0")
        self.nvenc_bf = int(self.nvenc_bf)
        if self.nvenc_bf < 0:
            raise ValueError("nvenc_bf must be >= 0")
        if self.nvenc_gop is not None:
            self.nvenc_gop = int(self.nvenc_gop)
            if self.nvenc_gop < 1:
                raise ValueError("nvenc_gop must be >= 1 when set")
        self.nvenc_b_ref_mode = self.nvenc_b_ref_mode.lower().strip()
        if self.nvenc_b_ref_mode not in {"disabled", "each", "middle"}:
            raise ValueError(
                f"nvenc_b_ref_mode must be disabled|each|middle, got {self.nvenc_b_ref_mode!r}"
            )
        if self.preprocess is not None:
            self.preprocess = self.preprocess.lower().strip() or None
        if self.preprocess is not None and self.preprocess not in _PREPROCESS_FILTERS:
            raise ValueError(
                f"preprocess must be one of {sorted(_PREPROCESS_FILTERS)}, got {self.preprocess!r}"
            )
        self.round2_preprocess = self.round2_preprocess.lower().strip()
        _denoise_only = {
            k
            for k, v in _PREPROCESS_FILTERS.items()
            if k != "none" and v is not None and (
                k.startswith("hqdn3d") or k.startswith("atadenoise")
            )
        }
        if self.round2_preprocess not in _denoise_only:
            raise ValueError(
                "round2_preprocess must be a denoise preset "
                f"({sorted(_denoise_only)}), "
                f"got {self.round2_preprocess!r}"
            )
        if int(self.nvenc_gpu) < 0:
            raise ValueError("nvenc_gpu must be >= 0")

        if self.proxy_seconds_per_segment <= 0:
            raise ValueError("proxy_seconds_per_segment must be > 0")
        if self.scene_max_samples < 1:
            raise ValueError("scene_max_samples must be >= 1")
        if float(self.sample_seconds_per_scene) <= 0:
            raise ValueError("sample_seconds_per_scene must be > 0")
        if int(self.sample_frames) < 1:
            raise ValueError("sample_frames must be >= 1")
        if int(self.scene_detect_downscale) < 1:
            raise ValueError("scene_detect_downscale must be >= 1")
        if float(self.crf_search_increment) <= 0:
            raise ValueError("crf_search_increment must be > 0")
        if int(self.crf_search_max_runs) < 2:
            raise ValueError("crf_search_max_runs must be >= 2")
        if not (0.0 < float(self.crf_search_max_encoded_percent) <= 100.0):
            raise ValueError("crf_search_max_encoded_percent must be in (0, 100]")
        if float(self.crf_search_near_vmaf_band) < 0:
            raise ValueError("crf_search_near_vmaf_band must be >= 0")
        if int(self.param_tune_max_trials) < 1:
            raise ValueError("param_tune_max_trials must be >= 1")
        if int(self.param_tune_no_improve_stop) < 1:
            raise ValueError("param_tune_no_improve_stop must be >= 1")
        if float(self.param_tune_vmaf_headroom) < 0:
            raise ValueError("param_tune_vmaf_headroom must be >= 0")
        if int(self.param_tune_max_rounds) < 1:
            raise ValueError("param_tune_max_rounds must be >= 1")
        if float(self.crf_mode_aq_step) <= 0:
            raise ValueError("crf_mode_aq_step must be > 0")
        if float(self.crf_mode_aq_min) > float(self.crf_mode_aq_max):
            raise ValueError("crf_mode_aq_min must be <= crf_mode_aq_max")
        if float(self.crf_mode_vmaf_headroom) < 0:
            raise ValueError("crf_mode_vmaf_headroom must be >= 0")
        if int(self.crf_mode_compensate_steps) < 1:
            raise ValueError("crf_mode_compensate_steps must be >= 1")
        if self.crf_mode_max_compression_rate is not None:
            rate = float(self.crf_mode_max_compression_rate)
            if not (0.0 < rate < 1.0):
                raise ValueError(
                    "crf_mode_max_compression_rate must be in (0, 1) or None, "
                    f"got {self.crf_mode_max_compression_rate!r}"
                )
            self.crf_mode_max_compression_rate = rate
        if int(self.crf_mode_lookahead_default) < 1:
            raise ValueError("crf_mode_lookahead_default must be >= 1")
        if isinstance(self.crf_mode_lookahead_sweep, list):
            self.crf_mode_lookahead_sweep = tuple(int(x) for x in self.crf_mode_lookahead_sweep)
        else:
            self.crf_mode_lookahead_sweep = tuple(int(x) for x in self.crf_mode_lookahead_sweep)
        if float(self.vbr_mode_aq_step) <= 0:
            raise ValueError("vbr_mode_aq_step must be > 0")
        if float(self.vbr_mode_aq_min) > float(self.vbr_mode_aq_max):
            raise ValueError("vbr_mode_aq_min must be <= vbr_mode_aq_max")
        if float(self.vbr_mode_proxy_seconds_per_scene) <= 0:
            raise ValueError("vbr_mode_proxy_seconds_per_scene must be > 0")
        if int(self.vbr_mode_max_parallel) < 1:
            raise ValueError("vbr_mode_max_parallel must be >= 1")
        if isinstance(self.vbr_mode_rd_sweep, list):
            self.vbr_mode_rd_sweep = tuple(int(x) for x in self.vbr_mode_rd_sweep)
        else:
            self.vbr_mode_rd_sweep = tuple(int(x) for x in self.vbr_mode_rd_sweep)
        if isinstance(self.vbr_mode_bframes_sweep, list):
            self.vbr_mode_bframes_sweep = tuple(int(x) for x in self.vbr_mode_bframes_sweep)
        else:
            self.vbr_mode_bframes_sweep = tuple(int(x) for x in self.vbr_mode_bframes_sweep)
        if isinstance(self.vbr_mode_lookahead_sweep, list):
            self.vbr_mode_lookahead_sweep = tuple(int(x) for x in self.vbr_mode_lookahead_sweep)
        else:
            self.vbr_mode_lookahead_sweep = tuple(int(x) for x in self.vbr_mode_lookahead_sweep)
        if not self.vbr_mode_rd_sweep:
            raise ValueError("vbr_mode_rd_sweep must be non-empty")
        if not self.vbr_mode_bframes_sweep:
            raise ValueError("vbr_mode_bframes_sweep must be non-empty")
        if not self.vbr_mode_lookahead_sweep:
            raise ValueError("vbr_mode_lookahead_sweep must be non-empty")
        if float(self.crf_search_sample_every_sec) <= 0:
            raise ValueError("crf_search_sample_every_sec must be > 0")
        if int(self.crf_search_min_samples) < 1:
            raise ValueError("crf_search_min_samples must be >= 1")
        if int(self.crf_search_vmaf_n_subsample) < 1:
            raise ValueError("crf_search_vmaf_n_subsample must be >= 1")
        self.crf_search_probe_preset = self.crf_search_probe_preset.lower().strip()
        if self.crf_search_probe_preset not in _X265_PRESETS:
            raise ValueError(
                "crf_search_probe_preset must be one of "
                f"{sorted(_X265_PRESETS)}, got {self.crf_search_probe_preset!r}"
            )
        if self.crf_search_samples is not None and int(self.crf_search_samples) < 1:
            raise ValueError("crf_search_samples must be >= 1 when set")
        if self.proxy_max_seconds <= 0:
            raise ValueError("proxy_max_seconds must be > 0")
        if self.proxy_min_window_seconds <= 0:
            raise ValueError("proxy_min_window_seconds must be > 0")
        if float(self.proxy_vmaf_margin) < 0:
            raise ValueError("proxy_vmaf_margin must be >= 0")
        if not (0.0 <= float(self.proxy_mashup_push_ceiling) <= 1.0):
            raise ValueError("proxy_mashup_push_ceiling must be in 0..1")

        self.vmaf_backend = self.vmaf_backend.lower().strip()
        if self.vmaf_backend not in {"docker", "native"}:
            raise ValueError(f"vmaf_backend must be docker or native, got {self.vmaf_backend!r}")
        self.vmaf_docker_image = (self.vmaf_docker_image or "vmaf_ffmpeg").strip()

        self.input_path = str(Path(self.input_path).expanduser()) if self.input_path else ""
        self.output_path = str(Path(self.output_path).expanduser())
        self.work_dir = str(Path(self.work_dir).expanduser())

    @property
    def is_rc(self) -> bool:
        return is_rc_mode(self.codec_mode)

    @property
    def is_abr(self) -> bool:
        return is_abr_mode(self.codec_mode)

    @property
    def refine_with_libx265_enabled(self) -> bool:
        """True when NVENC search should reserve budget for a libx265 refine."""
        return bool(self.libx265_refine) and self.encoder == "hevc_nvenc"

    @property
    def nvenc_x265_handoff(self) -> bool:
        """True for the NVENC RC → x265 CQ-mapping refine path.

        In this mode NVENC Round 2 is suppressed; Round-1 CQ answers are mapped
        to x265 CRFs instead of running further NVENC parameter trials.
        """
        return self.refine_with_libx265_enabled and self.is_rc

    @property
    def x265_crf_floor(self) -> int:
        """Resolved x265 CRF floor for the handoff mapping."""
        return self.crf_min if self.libx265_crf_min is None else self.libx265_crf_min

    @property
    def x265_crf_ceiling(self) -> int:
        """Resolved x265 CRF ceiling for the handoff mapping."""
        return self.crf_max if self.libx265_crf_max is None else self.libx265_crf_max

    def ensure_input_exists(self) -> None:
        path = Path(self.input_path)
        if not path.is_file():
            raise FileNotFoundError(f"input_path not found: {self.input_path}")
        self.input_path = str(path.resolve())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressionRequest":
        data = dict(data)
        if "x265_params" in data and "libx265_params" not in data:
            data["libx265_params"] = data.pop("x265_params")
        if "profile" in data and "libx265_profile" not in data:
            data["libx265_profile"] = data.pop("profile")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if "extra" in payload and isinstance(payload["extra"], dict):
            payload["extra"] = {**payload["extra"], **extra}
        else:
            payload["extra"] = extra
        return cls(**payload)

    @classmethod
    def from_json(cls, path_or_text: str) -> "CompressionRequest":
        p = Path(path_or_text)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(path_or_text)
        if not isinstance(data, dict):
            raise ValueError("Request JSON must be an object")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
