#!/usr/bin/env python3
"""Encode one video with fixed VBR (-b:v) + libx265_params, then print Vidaio score.

No CRF search and no param tune — one encode at the given bitrate (or a bitrate
derived from --target-compression-rate), then dual VMAF / s_f.

Supports VMAF-NEG survey / brave preprocess presets. Use ``--preprocess brave``
to try micro-enhancement + denoise→sharpen combos aimed at higher NEG under
the |base−neg|≤3 gate. ``--twopass`` enables libx265 2-pass ABR.

Examples:
  python test_fixed_vbr.py \\
    --input ../video/1.mp4 \\
    --target-compression-rate 0.04 \\
    --params "aq-mode=1:aq-strength=0.8:rd=6:ref=6:bframes=8:rc-lookahead=40:keyint=60:min-keyint=1:scenecut=50" \\
    --preprocess brave --gpu --preset fast
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from compress_util import bitrate_for_compression_rate
from encoder import (
    BRAVE_PREPROCESS_SWEEP,
    SURVEY_PREPROCESS_SWEEP,
    _PREPROCESS_FILTERS,
    encode_hevc,
)
from recipes import (
    PreprocessScoreView,
    choose_best_preprocess,
    resolve_vbr_preprocess,
)
from scoring import ScoreResult, probe_video, score_candidate

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

_PREPROCESS_CHOICES = [
    "auto",
    "sweep",
    "brave",
    *sorted(_PREPROCESS_FILTERS.keys()),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, help="Source video path")
    p.add_argument(
        "--output",
        "-o",
        default="",
        help="Output mp4 (default: work/fixed_vbr_test.mp4)",
    )
    p.add_argument(
        "--bitrate",
        "-b",
        default="",
        help="Fixed average bitrate for -b:v (e.g. 2M, 1500k)",
    )
    p.add_argument(
        "--target-compression-rate",
        type=float,
        default=None,
        help="If set (and --bitrate unset), derive -b:v so expected size ≈ rate * source",
    )
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
    p.add_argument(
        "--from-best",
        default="",
        help="Load params/threshold (and bitrate if present) from a best.json",
    )
    p.add_argument(
        "--preprocess",
        default="brave",
        choices=_PREPROCESS_CHOICES,
        help=(
            "Preset, auto (feature pick + A/B vs none), sweep "
            f"(survey: {', '.join(SURVEY_PREPROCESS_SWEEP)}), or brave "
            f"(micro set: {', '.join(BRAVE_PREPROCESS_SWEEP)}; default: brave)"
        ),
    )
    p.add_argument(
        "--preprocess-ab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare multiple candidates and keep the best (default: true)",
    )
    p.add_argument(
        "--twopass",
        action="store_true",
        help="libx265 2-pass ABR (better rate allocation at fixed -b:v)",
    )
    p.add_argument(
        "--features-json",
        default="",
        help="Optional features JSON (default: video_features/<stem>.json if present)",
    )
    p.add_argument("--gpu", action="store_true", help="Use Docker libvmaf_cuda (GPU 0)")
    p.add_argument("--gpu-device", type=int, default=0, help="GPU device index when --gpu")
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--vmaf-n-threads", type=int, default=40)
    return p.parse_args()


def _parse_bitrate_mbps(value: str) -> Optional[float]:
    text = str(value or "").strip().lower().replace(" ", "")
    if not text:
        return None
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([kmg]?)", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or "m"
    if unit == "g":
        return number * 1000.0
    if unit == "m":
        return number
    if unit == "k":
        return number / 1000.0
    return number


def _resolve_bitrate(
    input_path: Path,
    *,
    bitrate: str,
    target_compression_rate: Optional[float],
) -> str:
    explicit = str(bitrate or "").strip()
    if explicit:
        return explicit
    if target_compression_rate is None:
        raise SystemExit("need --bitrate or --target-compression-rate")
    rate = float(target_compression_rate)
    if not (0.0 < rate < 1.0):
        raise SystemExit(f"target_compression_rate must be in (0, 1), got {rate}")

    probe = probe_video(str(input_path))
    fmt = probe.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise SystemExit(f"unable to probe duration for {input_path}")

    return bitrate_for_compression_rate(
        source_bytes=input_path.stat().st_size,
        duration_sec=duration,
        compression_rate=rate,
    )


def _load_features(input_path: Path, features_json: str) -> dict[str, Any]:
    candidates: list[Path] = []
    if features_json:
        candidates.append(Path(features_json))
    stem = input_path.stem
    candidates.append(Path("video_features") / f"{stem}.json")
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if "noise_level_norm" in data or "motion_level" in data:
            return data
        for key in ("features", "global"):
            nested = data.get(key)
            if isinstance(nested, dict) and (
                "noise_level_norm" in nested or "motion_level" in nested
            ):
                return dict(nested)
        if isinstance(data.get("features"), dict):
            return dict(data["features"])
    return {}


def _score_view(score: ScoreResult) -> PreprocessScoreView:
    gates_ok = (
        float(score.vmaf) > 0
        and bool(score.passed_encoding_gates)
        and bool(score.passed_vmaf_delta_gate)
    )
    return PreprocessScoreView(
        s_f=float(score.s_f or 0.0),
        vmaf=float(score.vmaf or 0.0),
        gates_ok=gates_ok,
    )


def _encode_score(
    *,
    input_path: Path,
    out: Path,
    bitrate: str,
    params: str,
    preset: str,
    profile: str,
    preprocess: Optional[str],
    threshold: int,
    target_mbps: float,
    args: argparse.Namespace,
    label: str,
) -> tuple[Optional[ScoreResult], float, float]:
    t0 = time.monotonic()
    enc = encode_hevc(
        str(input_path),
        str(out),
        preset=preset,
        params=params,
        codec_mode="ABR",
        crf=None,
        bitrate=bitrate,
        encoder="libx265",
        preprocess=preprocess,
        libx265_profile=profile,
        twopass=bool(getattr(args, "twopass", False)),
        progress_reference_path=str(input_path),
        progress_label=label,
    )
    encode_sec = time.monotonic() - t0
    if not enc.ok:
        err = getattr(enc, "error", None) or enc.stderr_tail
        print(f"encode failed ({preprocess or 'none'}): {err[-500:]}")
        return None, encode_sec, 0.0

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
        codec_mode="ABR",
        target_bitrate_mbps=target_mbps,
    )
    return score, encode_sec, time.monotonic() - t1


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    bitrate = str(args.bitrate or "").strip()
    params = args.params
    preset = args.preset
    threshold = args.vmaf_threshold
    target_rate = args.target_compression_rate

    if args.from_best:
        best = json.loads(Path(args.from_best).read_text(encoding="utf-8"))
        params = params or str(best.get("libx265_params") or "")
        threshold = int(best.get("vmaf_threshold") or threshold)
        if not bitrate:
            bitrate = str(best.get("bitrate") or best.get("target_bitrate") or "").strip()
        if target_rate is None and best.get("target_compression_rate") is not None:
            target_rate = float(best["target_compression_rate"])
        if best.get("preset") and args.preset == "fast":
            preset = str(best["preset"])

    if not params:
        raise SystemExit("need --params or --from-best with libx265_params")

    bitrate = _resolve_bitrate(
        input_path,
        bitrate=bitrate,
        target_compression_rate=target_rate,
    )
    target_mbps = _parse_bitrate_mbps(bitrate)
    if target_mbps is None or target_mbps <= 0:
        raise SystemExit(f"invalid bitrate: {bitrate!r}")

    features = _load_features(input_path, args.features_json)
    sweep = args.preprocess == "sweep"
    brave = args.preprocess == "brave"
    if args.preprocess in {"auto", "sweep", "brave"}:
        proposed, reason, candidates = resolve_vbr_preprocess(
            explicit=args.preprocess if args.preprocess != "auto" else None,
            preprocess_auto=True,
            features=features,
            preprocess_sweep=sweep,
            preprocess_brave=brave,
        )
    else:
        proposed, reason, candidates = resolve_vbr_preprocess(
            explicit=args.preprocess,
            preprocess_auto=False,
            features=features,
            preprocess_sweep=False,
            preprocess_brave=False,
        )

    unique_cands: list[Optional[str]] = []
    for c in candidates:
        if c not in unique_cands:
            unique_cands.append(c)
    do_multi = bool(args.preprocess_ab or sweep or brave) and len(unique_cands) > 1

    out = Path(args.output) if args.output else Path("work") / "fixed_vbr_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"input : {input_path}")
    print(f"output: {out}")
    print(f"mode  : ABR/VBR{' 2-pass' if args.twopass else ''}")
    print(f"bitrate: {bitrate}  ({target_mbps:.3f} Mbps)")
    if target_rate is not None and not str(args.bitrate or "").strip():
        print(f"derived from target_compression_rate={target_rate}")
    print(f"params: {params}")
    print(f"preset: {preset}  profile={args.profile}")
    print(
        f"preprocess: {proposed or 'none'} ({reason})"
        + (
            f" [candidates={','.join(c or 'none' for c in unique_cands)}]"
            if do_multi
            else ""
        )
    )
    print(f"vmaf  : threshold={threshold} gpu={args.gpu} device={args.gpu_device}")
    print("-" * 60)

    encode_sec = 0.0
    score_sec = 0.0
    score: Optional[ScoreResult] = None
    chosen = proposed
    keep_reason = reason

    if do_multi:
        trial_paths: dict[Optional[str], Path] = {}
        trial_scores: dict[Optional[str], ScoreResult] = {}
        trial_views: list[tuple[Optional[str], PreprocessScoreView]] = []
        for cand in unique_cands:
            tag = cand or "none"
            cand_out = out.with_name(out.stem + f"_{tag}" + out.suffix)
            sc, e_sec, s_sec = _encode_score(
                input_path=input_path,
                out=cand_out,
                bitrate=bitrate,
                params=params,
                preset=preset,
                profile=args.profile,
                preprocess=cand,
                threshold=threshold,
                target_mbps=target_mbps,
                args=args,
                label=f"VBR {bitrate}/{preset} preprocess={tag}",
            )
            encode_sec += e_sec
            score_sec += s_sec
            if sc is None:
                trial_views.append((cand, PreprocessScoreView(0.0, 0.0, False)))
                continue
            trial_paths[cand] = cand_out
            trial_scores[cand] = sc
            view = _score_view(sc)
            trial_views.append((cand, view))
            print(
                f"  candidate {tag}: "
                f"vmaf_neg={sc.vmaf:.3f} "
                f"vmaf_base={sc.vmaf_base if sc.vmaf_base is not None else float('nan'):.3f} "
                f"delta={sc.vmaf_delta if sc.vmaf_delta is not None else float('nan'):.3f} "
                f"s_f={sc.s_f:.4f} gates={view.gates_ok}"
            )
        if not trial_scores:
            raise SystemExit("all preprocess candidates failed")
        chosen, keep_reason = choose_best_preprocess(trial_views)
        print(keep_reason)
        if chosen not in trial_scores:
            chosen = next(iter(trial_scores))
            keep_reason = f"fallback to {chosen or 'none'}"
        score = trial_scores[chosen]
        src = trial_paths[chosen]
        if src.resolve() != out.resolve():
            if out.exists():
                out.unlink()
            src.replace(out)
    else:
        score, encode_sec, score_sec = _encode_score(
            input_path=input_path,
            out=out,
            bitrate=bitrate,
            params=params,
            preset=preset,
            profile=args.profile,
            preprocess=proposed,
            threshold=threshold,
            target_mbps=target_mbps,
            args=args,
            label=f"VBR {bitrate}/{preset}",
        )
        if score is None:
            raise SystemExit("encode/score failed")
        chosen = proposed

    in_size = input_path.stat().st_size
    out_size = out.stat().st_size if out.is_file() else 0
    print("-" * 60)
    print(f"encode_sec        : {encode_sec:.1f}")
    print(f"score_sec         : {score_sec:.1f}")
    print(f"preprocess        : {chosen or 'none'}")
    print(f"preprocess_reason : {keep_reason}")
    print(f"size_in           : {in_size / (1024 * 1024):.2f} MiB")
    print(f"size_out          : {out_size / (1024 * 1024):.2f} MiB")
    print(f"vmaf (NEG)        : {score.vmaf:.6f}")
    print(f"vmaf_base         : {score.vmaf_base}")
    print(f"vmaf_delta        : {score.vmaf_delta}")
    print(f"compression_rate  : {score.compression_rate:.6f}  (out/in)")
    print(f"compression_ratio : {score.compression_ratio:.4f}x  (in/out)")
    print(f"s_f               : {score.s_f:.6f}")
    print(f"reason            : {score.reason}")
    print(
        f"gates             : enc={score.passed_encoding_gates} "
        f"delta={score.passed_vmaf_delta_gate}"
    )
    if score.validation_errors:
        print(f"validation_errors : {score.validation_errors}")
    return 0 if score.s_f > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
