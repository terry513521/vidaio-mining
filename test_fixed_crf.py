#!/usr/bin/env python3
"""Encode one video with fixed CRF + libx265_params, then print Vidaio score.

Examples:
  python test_fixed_crf.py \
    --input /ephemeral/videos/1.mp4 \
    --crf 32 \
    --preset fast \
    --params "aq-mode=1:aq-strength=0.8:rd=4:ref=6:bframes=8:rc-lookahead=30:keyint=60:min-keyint=1:scenecut=50" \
    --vmaf-threshold 85 \
    --gpu

  python test_fixed_crf.py --input /ephemeral/videos/1.mp4 --from-best published_results/85/v1/best.json --preset medium
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from encoder import encode_hevc
from scoring import score_candidate

_X265_PRESETS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, help="Source video path")
    p.add_argument("--output", "-o", default="", help="Output mp4 (default: work/fixed_test.mp4)")
    p.add_argument("--crf", type=int, default=None, help="Fixed libx265 CRF")
    p.add_argument("--params", default="", help="libx265 -x265-params string")
    p.add_argument(
        "--preset",
        "-p",
        default="fast",
        choices=_X265_PRESETS,
        help="libx265 preset (default: fast)",
    )
    p.add_argument("--profile", default="main", help="libx265 profile (default: main)")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument("--from-best", default="", help="Load crf/params/threshold from a best.json")
    p.add_argument("--gpu", action="store_true", help="Use Docker libvmaf_cuda (GPU 0)")
    p.add_argument("--gpu-device", type=int, default=0, help="GPU device index when --gpu")
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--vmaf-n-threads", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    crf = args.crf
    params = args.params
    preset = args.preset
    threshold = args.vmaf_threshold
    if args.from_best:
        best = json.loads(Path(args.from_best).read_text(encoding="utf-8"))
        crf = int(best["crf"]) if crf is None else crf
        params = params or str(best.get("libx265_params") or "")
        threshold = int(best.get("vmaf_threshold") or threshold)
        # Optional fields if present in best/result payloads.
        if best.get("preset") and args.preset == "fast":
            preset = str(best["preset"])

    if crf is None:
        raise SystemExit("need --crf or --from-best with a crf field")
    if not params:
        raise SystemExit("need --params or --from-best with libx265_params")

    out = Path(args.output) if args.output else Path("work") / "fixed_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"input : {input_path}")
    print(f"output: {out}")
    print(f"crf   : {crf}")
    print(f"params: {params}")
    print(f"preset: {preset}  profile={args.profile}")
    print(f"vmaf  : threshold={threshold} gpu={args.gpu} device={args.gpu_device}")
    print("-" * 60)

    t0 = time.monotonic()
    enc = encode_hevc(
        str(input_path),
        str(out),
        preset=preset,
        params=params,
        codec_mode="RC",
        crf=crf,
        encoder="libx265",
        libx265_profile=args.profile,
        progress_reference_path=str(input_path),
        progress_label=f"CRF{crf}/{preset}",
    )
    encode_sec = time.monotonic() - t0
    if not enc.ok:
        raise SystemExit(f"encode failed: {enc.error or enc.stderr[-1000:]}")

    t1 = time.monotonic()
    score = score_candidate(
        str(input_path),
        str(out),
        threshold,
        vmaf_n_subsample=args.vmaf_n_subsample,
        vmaf_n_threads=args.vmaf_n_threads,
        vmaf_backend="docker",
        vmaf_docker_image="vmaf_ffmpeg",
        vmaf_docker_gpus=bool(args.gpu),
        vmaf_gpu_device=args.gpu_device if args.gpu else None,
        codec_mode="RC",
    )
    score_sec = time.monotonic() - t1

    in_size = input_path.stat().st_size
    out_size = out.stat().st_size
    print("-" * 60)
    print(f"encode_sec        : {encode_sec:.1f}")
    print(f"score_sec         : {score_sec:.1f}")
    print(f"size_in           : {in_size / (1024 * 1024):.2f} MiB")
    print(f"size_out          : {out_size / (1024 * 1024):.2f} MiB")
    print(f"vmaf (NEG)        : {score.vmaf:.6f}")
    print(f"vmaf_base         : {score.vmaf_base}")
    print(f"vmaf_delta        : {score.vmaf_delta}")
    print(f"compression_rate  : {score.compression_rate:.6f}  (out/in)")
    print(f"compression_ratio : {score.compression_ratio:.4f}x  (in/out)")
    print(f"s_f               : {score.s_f:.6f}")
    print(f"reason            : {score.reason}")
    print(f"gates             : enc={score.passed_encoding_gates} delta={score.passed_vmaf_delta_gate}")
    if score.validation_errors:
        print(f"validation_errors : {score.validation_errors}")
    return 0 if score.s_f > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
