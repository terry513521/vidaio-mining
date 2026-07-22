#!/usr/bin/env python3
"""Extract 3-level video features and save one JSON per video.

Level 1 — global statistics (clip means / rates)
Level 2 — per-segment statistics
Level 3 — distribution statistics (max, p95, variance, ratios, histogram)

Examples:
  python extract_video_features.py
  python extract_video_features.py --input /ephemeral/videos --out /ephemeral/videos/features
  python extract_video_features.py --input /ephemeral/videos --limit 5 --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from feature_extractor import HEVCFeatureExtractor

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("/ephemeral/videos")
DEFAULT_OUT = DEFAULT_INPUT / "features"

# Metrics aggregated across segments for Level 3.
_DIST_METRICS = (
    "motion_mean",
    "motion_p90",
    "motion_max",
    "texture",
    "edge_density",
    "noise_level",
    "high_freq_energy",
    "flatness",
    "entropy",
    "difficulty",
)

_HIST_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metric_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "p95": 0.0,
            "variance": 0.0,
            "std": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "p95": float(np.percentile(arr, 95)),
        "variance": float(np.var(arr)),
        "std": float(np.std(arr)),
    }


def _difficulty_histogram(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "bins": _HIST_BINS,
            "counts": [0] * (len(_HIST_BINS) - 1),
            "fractions": [0.0] * (len(_HIST_BINS) - 1),
        }
    arr = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=_HIST_BINS)
    total = max(int(np.sum(counts)), 1)
    fractions = [float(c) / total for c in counts.tolist()]
    return {
        "bins": _HIST_BINS,
        "counts": [int(c) for c in counts.tolist()],
        "fractions": fractions,
    }


def build_global(summary: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Level 1: clip-level / global statistics."""
    return {
        "mean_motion": _safe_float(summary.get("motion_mean")),
        "mean_texture": _safe_float(summary.get("texture")),
        "mean_noise": _safe_float(summary.get("noise_level")),
        "mean_edge": _safe_float(summary.get("edge_density")),
        "mean_entropy": _safe_float(summary.get("entropy")),
        "mean_hf_energy": _safe_float(summary.get("high_freq_energy")),
        "mean_flatness": _safe_float(summary.get("flatness")),
        "mean_luma": _safe_float(summary.get("luma_mean")),
        "mean_difficulty": _safe_float(summary.get("difficulty_mean")),
        "motion_p90": _safe_float(summary.get("motion_p90")),
        "motion_level": _safe_float(summary.get("motion_level")),
        "texture_level": _safe_float(summary.get("texture_level")),
        "noise_level_norm": _safe_float(summary.get("noise_level_norm")),
        "edge_level": _safe_float(summary.get("edge_level")),
        "segment_count": int(_safe_float(summary.get("segment_count"))),
        "cut_count": int(_safe_float(summary.get("cut_count"))),
        "cut_rate": _safe_float(summary.get("cut_rate")),
        "hard_fraction": _safe_float(summary.get("hard_fraction")),
        "worst_difficulty": _safe_float(summary.get("worst_difficulty")),
        "volatility": _safe_float(summary.get("volatility")),
        "duration": _safe_float(summary.get("duration")),
        "fps": _safe_float(summary.get("fps")),
        "width": int(_safe_float(summary.get("width"))),
        "height": int(_safe_float(summary.get("height"))),
        "n_segments": len(segments),
    }


def build_segment_stats(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Level 2: per-segment statistics (compact view + raw fields)."""
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "index": int(seg.get("index", 0)),
                "start_sec": _safe_float(seg.get("start_sec")),
                "end_sec": _safe_float(seg.get("end_sec")),
                "duration": _safe_float(seg.get("duration")),
                "frame_count": int(_safe_float(seg.get("frame_count"))),
                "motion": _safe_float(seg.get("motion_mean")),
                "motion_p90": _safe_float(seg.get("motion_p90")),
                "motion_max": _safe_float(seg.get("motion_max")),
                "texture": _safe_float(seg.get("texture")),
                "edge": _safe_float(seg.get("edge_density")),
                "noise": _safe_float(seg.get("noise_level")),
                "entropy": _safe_float(seg.get("entropy")),
                "hf_energy": _safe_float(seg.get("high_freq_energy")),
                "flatness": _safe_float(seg.get("flatness")),
                "si": _safe_float(seg.get("si")),
                "ti": _safe_float(seg.get("ti")),
                "luma_mean": _safe_float(seg.get("luma_mean")),
                "sat_mean": _safe_float(seg.get("sat_mean")),
                "difficulty": _safe_float(seg.get("difficulty")),
            }
        )
    return out


def build_distributions(
    segments: list[dict[str, Any]],
    *,
    high_thr: float = 0.45,
    low_thr: float = 0.20,
) -> dict[str, Any]:
    """Level 3: distribution statistics over segments."""
    metrics: dict[str, list[float]] = {key: [] for key in _DIST_METRICS}
    difficulties: list[float] = []
    durations: list[float] = []

    for seg in segments:
        dur = max(_safe_float(seg.get("duration")), 0.0)
        durations.append(dur)
        for key in _DIST_METRICS:
            metrics[key].append(_safe_float(seg.get(key)))
        difficulties.append(_safe_float(seg.get("difficulty")))

    total_dur = float(sum(durations)) or 1.0
    high_mask = [d >= high_thr for d in difficulties]
    low_mask = [d <= low_thr for d in difficulties]
    high_dur = float(sum(dur for dur, flag in zip(durations, high_mask) if flag))
    low_dur = float(sum(dur for dur, flag in zip(durations, low_mask) if flag))

    dists = {key: _metric_distribution(vals) for key, vals in metrics.items()}
    return {
        "thresholds": {"high_complexity": high_thr, "low_complexity": low_thr},
        "metrics": dists,
        "high_complexity_ratio": high_dur / total_dur,
        "low_complexity_ratio": low_dur / total_dur,
        "high_complexity_segment_ratio": (
            float(sum(high_mask)) / max(len(difficulties), 1)
        ),
        "low_complexity_segment_ratio": (
            float(sum(low_mask)) / max(len(difficulties), 1)
        ),
        "scene_complexity_histogram": _difficulty_histogram(difficulties),
        "worst_segment_index": (
            int(np.argmax(np.asarray(difficulties, dtype=np.float64)))
            if difficulties
            else None
        ),
    }


def extract_features(
    video_path: Path,
    *,
    scene_detector: str = "content",
    scene_threshold: Optional[float] = None,
    min_scene_len_sec: float = 0.4,
    use_builtin_cuts: bool = False,
    samples_per_segment: int = 16,
) -> dict[str, Any]:
    """Extract features; default cuts from PySceneDetect ContentDetector.

    Set ``use_builtin_cuts=True`` to use the legacy histogram/diff cut detector
    inside ``HEVCFeatureExtractor`` instead of PySceneDetect.
    """
    started = time.perf_counter()
    scene_spans: Optional[list[tuple[float, float]]] = None
    detect_meta: dict[str, Any] = {"detector": "builtin"}

    if not use_builtin_cuts:
        from scene_detect import detect_scenes

        det = detect_scenes(
            str(video_path),
            detector="adaptive" if str(scene_detector).lower().startswith("adapt") else "content",
            threshold=scene_threshold,
            min_scene_len_sec=float(min_scene_len_sec),
        )
        scene_spans = det.spans
        detect_meta = {
            "detector": det.detector or scene_detector,
            "ok": bool(det.ok),
            "error": det.error,
            "detect_sec": round(float(det.detect_sec), 3),
            "n_scenes": len(det.scenes),
        }

    full = HEVCFeatureExtractor(
        str(video_path),
        samples_per_segment=int(samples_per_segment),
    ).extract_full(scene_spans=scene_spans)
    segments = full.get("segments") or []
    summary = full.get("summary") or {}
    meta = full.get("meta") or {}
    meta = {
        **meta,
        "scene_detect": detect_meta,
        "samples_per_segment": int(samples_per_segment),
        "si_ti_agg": "mean",
    }
    elapsed = time.perf_counter() - started

    return {
        "video": video_path.name,
        "path": str(video_path.resolve()),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 3),
        "meta": meta,
        "cut_times_sec": full.get("cut_times_sec") or [],
        "global": build_global(summary, segments),
        "segments": build_segment_stats(segments),
        "distributions": build_distributions(segments),
        # Keep raw extractor summary for compatibility with recipes/search.
        "summary_raw": summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract and save per-video feature JSON")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Video directory (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory for feature JSON (default: {DEFAULT_OUT})",
    )
    p.add_argument("--limit", type=int, default=0, help="Max videos (0 = all)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output JSON already exists",
    )
    p.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob under --input (default: *.mp4)",
    )
    p.add_argument(
        "--scene-detector",
        choices=["content", "adaptive", "builtin"],
        default="content",
        help=(
            "Cut detector: content=PySceneDetect ContentDetector (default), "
            "adaptive=AdaptiveDetector, builtin=legacy histogram/diff cuts"
        ),
    )
    p.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        help="Optional detector threshold (ContentDetector default 27)",
    )
    p.add_argument(
        "--min-scene-len",
        type=float,
        default=0.4,
        help="Minimum scene length in seconds (default: 0.4)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input
    out_dir: Path = args.out
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    videos = sorted(input_dir.glob(args.pattern), key=_natural_key)
    videos = [p for p in videos if p.is_file()]
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        print(f"No videos matched {args.pattern} in {input_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    skipped = 0
    failed = 0

    for index, video in enumerate(videos, start=1):
        dest = out_dir / f"{video.stem}.json"
        if dest.is_file() and not args.force:
            skipped += 1
            print(f"[{index}/{len(videos)}] skip {video.name} (exists)")
            continue
        print(f"[{index}/{len(videos)}] extract {video.name} …", flush=True)
        try:
            payload = extract_features(
                video,
                scene_detector=str(args.scene_detector),
                scene_threshold=args.scene_threshold,
                min_scene_len_sec=float(args.min_scene_len),
                use_builtin_cuts=(str(args.scene_detector) == "builtin"),
            )
            dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            ok += 1
            g = payload["global"]
            print(
                f"  -> {dest.name}  segs={g['segment_count']}  "
                f"mot={g['mean_motion']:.4f} tex={g['mean_texture']:.4f}  "
                f"hard={payload['distributions']['high_complexity_ratio']:.3f}  "
                f"{payload['elapsed_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            err_path = out_dir / f"{video.stem}.error.json"
            err_path.write_text(
                json.dumps(
                    {"video": video.name, "error": str(exc)},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"  ERROR {video.name}: {exc}", file=sys.stderr, flush=True)

    manifest = {
        "input_dir": str(input_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "total": len(videos),
        "ok": ok,
        "skipped": skipped,
        "failed": failed,
        "videos": [p.name for p in videos],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Done: {ok} saved, {skipped} skipped, {failed} failed -> {out_dir}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
