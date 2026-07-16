"""
HEVC compression entrypoint (mashup-aware strategy).

Strategy:
  1) Features nudge CRF seed only
  2) Default quality (+ optional balance) recipes
  3) Full-file CRF search: VMAF-lerp walk toward threshold
  4) Winner chosen by Vidaio-style s_f (VMAF NEG + compression rate)

Examples:
  python main.py --input mixed.mp4 --vmaf-threshold 89
  python main.py --input mixed.mp4 --output out.mp4 --codec-mode CRF
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from feature_extractor import HEVCFeatureExtractor
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
    parser.add_argument("--max-search-steps", type=int, default=5)
    parser.add_argument("--max-recipes", type=int, default=2)
    parser.add_argument("--vbr-max-ratio-to-target", type=float, default=1.1)
    parser.add_argument("--vbr-min-mbps-floor", type=float, default=0.5)
    parser.add_argument("--crf-min", type=int, default=22)
    parser.add_argument("--crf-max", type=int, default=36)
    parser.add_argument("--crf-start", type=int, default=None)

    parser.add_argument("--sample-frames", type=int, default=40)
    parser.add_argument("--vmaf-n-subsample", type=int, default=1)
    parser.add_argument("--vmaf-n-threads", type=int, default=2)

    parser.add_argument("--work-dir", default="work")
    parser.add_argument(
        "--keep-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ffmpeg-bin", default=r"C:\ffmpeg\bin\ffmpeg.exe")
    parser.add_argument("--ffprobe-bin", default=r"C:\ffmpeg\bin\ffprobe.exe")

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
        vbr_max_ratio_to_target=args.vbr_max_ratio_to_target,
        vbr_min_mbps_floor=args.vbr_min_mbps_floor,
        crf_min=args.crf_min,
        crf_max=args.crf_max,
        crf_start=args.crf_start,
        sample_frames=args.sample_frames,
        vmaf_n_subsample=args.vmaf_n_subsample,
        vmaf_n_threads=args.vmaf_n_threads,
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

    print("Request:")
    print(json.dumps(req.to_dict(), indent=2))

    print("\n[1/3] Extracting segment-aware features...")
    features = HEVCFeatureExtractor(req.input_path).extract()
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
            print(f"  {key:24} {features[key]:.4f}")

    print("\n[2/3] Searching HEVC on full file (maximize s_f)...")
    result = run_search(req, features)

    print("\n[3/3] Done")
    print(json.dumps(result.to_dict(), indent=2))

    summary_path = Path(req.work_dir) / "result.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")

    if result.best is None or result.best.score.s_f <= 0:
        print("No positive-scoring candidate found.", file=sys.stderr)
        return 1

    print(
        f"Best: strategy={result.strategy} recipe={result.best.recipe} "
        f"crf={result.best.crf} bitrate={result.best.bitrate} stage={result.best.stage} "
        f"vmaf={result.best.score.vmaf:.2f} s_f={result.best.score.s_f:.4f} "
        f"-> {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
