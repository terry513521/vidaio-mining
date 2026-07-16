"""
HEVC compression entrypoint (mashup-aware strategy).

Strategy:
  1) Features nudge CRF seed only
  2) Default medium preset recipe
  3) Build proxy from ~2.5s mid-windows of each segment
  4) Encode 3 CRF candidates on the proxy in parallel (bitrate-ratio s_f)
  5) Encode full file once at the winning CRF for the true s_f

Encode uses native ffmpeg (libx265, yuv420p). VMAF defaults to Docker
(vmaf_ffmpeg) for validator-parity.

Examples:
  python main.py --input mixed.mp4 --vmaf-threshold 89
  python main.py --input mixed.mp4 --crf-candidates 3 --crf-spread 2 --max-workers 3
  python main.py --input mixed.mp4 --no-use-proxy   # full-file parallel only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from feature_extractor import HEVCFeatureExtractor
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
    parser.add_argument("--codec-mode", default="CRF", choices=["CRF", "VBR", "crf", "vbr"])
    parser.add_argument("--target-bitrate", default=None, help='VBR target, e.g. "5M"')

    parser.add_argument("--time-budget-sec", type=float, default=3000.0)
    parser.add_argument("--max-search-steps", type=int, default=5, help="VBR search step budget")
    parser.add_argument("--max-recipes", type=int, default=1, help="1=primary preset only; 2=+slow")
    parser.add_argument(
        "--preset",
        default="medium",
        choices=[
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
        ],
        help="libx265 preset (default: medium)",
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
        help="VMAF via vmaf_ffmpeg Docker (default) or native ffmpeg with libvmaf",
    )
    parser.add_argument(
        "--vmaf-docker-image",
        default="vidaio-compression-eval",
        help="Docker image with libvmaf ffmpeg (local: vidaio-compression-eval)",
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
        "motion_mean",
        "texture",
        "volatility",
    ):
        if key in features:
            log(f"  {key:24} {features[key]:.4f}")
    log(f"  segments                 {len(segments)}")

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

    log(
        f"Best: strategy={result.strategy} recipe={result.best.recipe} "
        f"crf={result.best.crf} bitrate={result.best.bitrate} stage={result.best.stage} "
        f"vmaf={result.best.score.vmaf:.2f} s_f={result.best.score.s_f:.4f} "
        f"elapsed={result.elapsed_sec:.1f}s -> {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
