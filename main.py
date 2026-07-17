"""
HEVC compression entrypoint (mashup-aware strategy).

Strategy:
  1) Features nudge CRF seed only
  2) Default medium preset recipe
  3) Build proxy from ~2.5s mid-windows of each segment
  4) Encode 3 CRF candidates on the proxy in parallel (bitrate-ratio s_f)
  5) Encode full file once at the winning CRF for the true s_f

Encode uses native ffmpeg: libx265 (CPU) or hevc_nvenc (GPU). Local
evaluation VMAF is Dockerized (default image: vmaf_ffmpeg).

Examples:
  python main.py -r request.json
  python main.py --input mixed.mp4 --encoder hevc_nvenc --preset p5
  python main.py --input mixed.mp4 --vmaf-docker-gpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from feature_extractor import HEVCFeatureExtractor, format_feature_report
from interp_search import apply_feature_nvenc_baseline
from logutil import log
from request import CompressionRequest
from search import run_search


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HEVC compressor optimized for Vidaio interleaved ~30s mashups"
    )

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--request", "-r", help="Path to JSON request body")
    src.add_argument("--json", "-j", help="Inline JSON request body")

    parser.add_argument("--input", "-i", help="Input video path")
    parser.add_argument("--output", "-o", default="compressed_hevc.mp4", help="Output path")
    parser.add_argument("--vmaf-threshold", type=int, default=89, choices=[85, 89, 93])
    parser.add_argument("--codec", default="hevc")
    parser.add_argument(
        "--codec-mode",
        default="RC",
        choices=["RC", "ABR", "CRF", "VBR", "rc", "abr", "crf", "vbr"],
        help="RC=constant quality (CQ/CRF search); ABR=average bitrate (-b:v search)",
    )
    parser.add_argument(
        "--encoder",
        default="libx265",
        choices=["libx265", "hevc_nvenc", "x265", "nvenc", "nvenc_hevc"],
        help="Encode backend: libx265 (CPU) or hevc_nvenc (GPU)",
    )
    parser.add_argument(
        "--libx265-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After hevc_nvenc search, run a small libx265 refine (max s_f wins)",
    )
    parser.add_argument(
        "--libx265-refine-preset",
        default="medium",
        help="libx265 preset for refine encodes (default: medium)",
    )
    parser.add_argument(
        "--libx265-refine-candidates",
        type=int,
        default=3,
        help="Number of libx265 refine candidates",
    )
    parser.add_argument(
        "--libx265-refine-crf-spread",
        type=int,
        default=2,
        help="RC refine CRF spacing around feature seed",
    )
    parser.add_argument(
        "--libx265-refine-max-workers",
        type=int,
        default=2,
        help="CPU workers for libx265 refine",
    )
    parser.add_argument(
        "--libx265-refine-time-sec",
        type=float,
        default=60.0,
        help="Seconds reserved from time_budget for libx265 refine",
    )
    parser.add_argument(
        "--libx265-feature-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set libx265 -x265-params from features (never touches CRF)",
    )
    parser.add_argument("--target-bitrate", default=None, help='ABR target, e.g. "8M"')

    parser.add_argument("--time-budget-sec", type=float, default=3000.0)
    parser.add_argument("--max-search-steps", type=int, default=5, help="ABR search step budget")
    parser.add_argument("--max-recipes", type=int, default=1, help="1=primary preset only; 2=+slow")
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx265: ultrafast..placebo; NVENC: p1..p7 (or same names, mapped)",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel encode workers")
    parser.add_argument("--crf-candidates", type=int, default=3, help="Number of CRF trials")
    parser.add_argument("--crf-spread", type=int, default=2, help="CRF spacing around seed")
    parser.add_argument("--vbr-max-ratio-to-target", type=float, default=1.1)
    parser.add_argument("--vbr-min-mbps-floor", type=float, default=0.5)
    parser.add_argument("--crf-min", type=int, default=22)
    parser.add_argument("--crf-max", type=int, default=36)
    parser.add_argument("--crf-start", type=int, default=None)
    parser.add_argument(
        "--search-strategy",
        default="parallel_grid",
        choices=["parallel_grid", "interp_answer"],
        help="CQ/CRF search: parallel_grid or interp_answer (2-round interpolate)",
    )
    parser.add_argument(
        "--search-rounds",
        type=int,
        default=2,
        help="Rounds for interp_answer (default 2)",
    )

    parser.add_argument("--nvenc-tune", default="hq", choices=["hq", "ll", "ull", "lossless"])
    parser.add_argument(
        "--nvenc-rc",
        default="vbr",
        choices=["vbr", "vbr_hq", "cbr", "cbr_hq", "constqp"],
    )
    parser.add_argument(
        "--nvenc-multipass",
        default="qres",
        choices=["disabled", "qres", "fullres"],
    )
    parser.add_argument(
        "--nvenc-spatial-aq",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--nvenc-temporal-aq",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nvenc-aq-strength", type=int, default=8)
    parser.add_argument(
        "--preprocess",
        default=None,
        choices=["none", "hqdn3d_light", "hqdn3d_med", "atadenoise_light"],
        help="Optional safe denoise preprocess before encode (denoise-only)",
    )
    parser.add_argument(
        "--round2-preprocess-trial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add one measured light-denoise trial in Round 2 (reports NEG/base/delta)",
    )
    parser.add_argument("--nvenc-gpu", type=int, default=0)
    parser.add_argument(
        "--nvenc-hwaccel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA hwaccel decode before NVENC",
    )

    parser.add_argument(
        "--use-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Two-phase: proxy CRF select then one full encode (default on)",
    )
    parser.add_argument(
        "--proxy-seconds-per-segment",
        type=float,
        default=2.5,
        help="Seconds sampled from the middle of each segment",
    )
    parser.add_argument(
        "--proxy-max-seconds",
        type=float,
        default=15.0,
        help="Hard cap on total proxy duration",
    )
    parser.add_argument(
        "--proxy-min-window-seconds",
        type=float,
        default=0.5,
        help="Minimum window length kept when trimming the proxy",
    )

    parser.add_argument("--sample-frames", type=int, default=40)
    parser.add_argument("--vmaf-n-subsample", type=int, default=1)
    parser.add_argument("--vmaf-n-threads", type=int, default=2)
    parser.add_argument(
        "--vmaf-backend",
        default="docker",
        choices=["docker", "native"],
        help="Local eval VMAF: docker (default, validator-parity) or native fallback",
    )
    parser.add_argument(
        "--vmaf-docker-image",
        default="vmaf_ffmpeg",
        help="Docker image with libvmaf ffmpeg (default: vmaf_ffmpeg)",
    )
    parser.add_argument(
        "--vmaf-docker-gpus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --gpus all and use libvmaf_cuda (validator-style)",
    )

    parser.add_argument("--work-dir", default="work")
    parser.add_argument(
        "--keep-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ffmpeg-bin", default="/usr/bin/ffmpeg")
    parser.add_argument("--ffprobe-bin", default="/usr/bin/ffprobe")

    parser.add_argument(
        "--print-request",
        action="store_true",
        help="Print normalized request and exit",
    )
    parser.add_argument(
        "--dump-features",
        action="store_true",
        help="Extract features, print report, and exit (no encode)",
    )
    return parser.parse_args(argv)


def load_request(args: argparse.Namespace) -> CompressionRequest:
    if args.request:
        return CompressionRequest.from_json(args.request)
    if args.json:
        return CompressionRequest.from_json(args.json)

    if not args.input:
        raise SystemExit("Provide --input (or --request / --json)")

    return CompressionRequest(
        input_path=args.input,
        output_path=args.output,
        vmaf_threshold=args.vmaf_threshold,
        codec=args.codec,
        codec_mode=args.codec_mode,
        encoder=args.encoder,
        libx265_refine=args.libx265_refine,
        libx265_refine_preset=args.libx265_refine_preset,
        libx265_refine_candidates=args.libx265_refine_candidates,
        libx265_refine_crf_spread=args.libx265_refine_crf_spread,
        libx265_refine_max_workers=args.libx265_refine_max_workers,
        libx265_refine_time_sec=args.libx265_refine_time_sec,
        libx265_feature_baseline=args.libx265_feature_baseline,
        target_bitrate=args.target_bitrate,
        time_budget_sec=args.time_budget_sec,
        max_search_steps=args.max_search_steps,
        max_recipes=args.max_recipes,
        max_workers=args.max_workers,
        preset=args.preset,
        crf_candidates=args.crf_candidates,
        crf_spread=args.crf_spread,
        vbr_max_ratio_to_target=args.vbr_max_ratio_to_target,
        vbr_min_mbps_floor=args.vbr_min_mbps_floor,
        crf_min=args.crf_min,
        crf_max=args.crf_max,
        crf_start=args.crf_start,
        search_strategy=args.search_strategy,
        search_rounds=args.search_rounds,
        nvenc_tune=args.nvenc_tune,
        nvenc_rc=args.nvenc_rc,
        nvenc_multipass=args.nvenc_multipass,
        nvenc_spatial_aq=args.nvenc_spatial_aq,
        nvenc_temporal_aq=args.nvenc_temporal_aq,
        nvenc_aq_strength=args.nvenc_aq_strength,
        preprocess=args.preprocess,
        round2_preprocess_trial=args.round2_preprocess_trial,
        nvenc_gpu=args.nvenc_gpu,
        nvenc_hwaccel=args.nvenc_hwaccel,
        use_proxy=args.use_proxy,
        proxy_seconds_per_segment=args.proxy_seconds_per_segment,
        proxy_max_seconds=args.proxy_max_seconds,
        proxy_min_window_seconds=args.proxy_min_window_seconds,
        sample_frames=args.sample_frames,
        vmaf_n_subsample=args.vmaf_n_subsample,
        vmaf_n_threads=args.vmaf_n_threads,
        vmaf_backend=args.vmaf_backend,
        vmaf_docker_image=args.vmaf_docker_image,
        vmaf_docker_gpus=args.vmaf_docker_gpus,
        work_dir=args.work_dir,
        keep_candidates=args.keep_candidates,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    req = load_request(args)

    if args.print_request:
        print(json.dumps(req.to_dict(), indent=2))
        return 0

    req.ensure_input_exists()

    if args.dump_features:
        full = HEVCFeatureExtractor(req.input_path).extract_full()
        print(format_feature_report(full))
        dump_path = Path(req.work_dir) / "features.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
        log(f"Wrote {dump_path}")
        return 0

    log("Request:")
    print(json.dumps(req.to_dict(), indent=2))

    log("[1/3] Extracting segment-aware features...")
    full = HEVCFeatureExtractor(req.input_path).extract_full()
    features = full["summary"]
    segments = full.get("segments") or []
    for key in (
        "segment_count",
        "cut_rate",
        "hard_fraction",
        "worst_difficulty",
        "difficulty_p90",
        "motion_p90",
        "motion_level",
        "texture_level",
        "noise_level_norm",
        "texture",
        "volatility",
    ):
        if key in features:
            log(f"  {key:24} {features[key]:.4f}")
    log(f"  segments                 {len(segments)}")

    if req.encoder == "hevc_nvenc" and req.nvenc_feature_baseline:
        log("[1b/3] Applying feature → NVENC baseline (CQ untouched)...")
        for line in apply_feature_nvenc_baseline(req, features):
            log(f"  {line}")
    elif req.encoder == "libx265" and req.libx265_feature_baseline:
        from recipes import describe_feature_x265_baseline

        log("[1b/3] Applying feature → libx265 params (CRF untouched)...")
        for line in describe_feature_x265_baseline(features):
            log(f"  {line}")

    log("[2/3] Searching HEVC (maximize s_f)...")
    result = run_search(req, features, segments=segments)

    log("[3/3] Done")
    print(json.dumps(result.to_dict(), indent=2))

    summary_path = Path(req.work_dir) / "result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log(f"Wrote {summary_path}")

    if result.best is None or result.best.score.s_f <= 0:
        log("No positive-scoring candidate found.", file=sys.stderr)
        return 1

    mbps = result.best.measured_bitrate_mbps
    bitrate_txt = f"{mbps:.2f}Mbps" if mbps is not None else "n/a"
    log(
        f"Best: strategy={result.strategy} recipe={result.best.recipe} "
        f"encoder={result.best.encoder} "
        f"crf={result.best.crf} bitrate={bitrate_txt} stage={result.best.stage} "
        f"neg={result.best.score.vmaf:.2f} "
        f"base={result.best.score.vmaf_base:.2f} "
        f"delta={result.best.score.vmaf_delta:.2f} "
        f"s_f={result.best.score.s_f:.4f} "
        f"compression_ratio={result.best.score.compression_ratio:.2f}x "
        f"encode={result.best.encode_sec:.1f}s score={result.best.score_sec:.1f}s "
        f"candidate={result.best.elapsed_sec:.1f}s "
        f"elapsed={result.elapsed_sec:.1f}s -> {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
