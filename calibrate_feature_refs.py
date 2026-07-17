#!/usr/bin/env python3
"""Sweep raw features across a video corpus and suggest soft-norm midpoints.

Usage:
  python calibrate_feature_refs.py [--dir s3_videos/compression] [--limit N] [-o out.json]

Prints percentiles for motion_p90, noise_level, texture, high_freq_energy,
edge_density, cut_rate and recommends soft-curve midpoints (≈ median / p50,
with p75 shown for sensitivity). Soft norm: level = x / (x + mid) → mid maps to 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from feature_extractor import HEVCFeatureExtractor

RAW_KEYS = (
    "motion_p90",
    "motion_mean",
    "noise_level",
    "texture",
    "high_freq_energy",
    "edge_density",
    "cut_rate",
    "flatness",
    "duration",
    "fps",
    "width",
    "height",
)

# Keys used as soft-norm midpoints in feature_extractor.
MIDPOINT_KEYS = (
    "motion_p90",
    "noise_level",
    "texture",
    "high_freq_energy",
    "edge_density",
)


def _percentile_report(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = [0, 10, 25, 50, 75, 90, 95, 100]
    out = {f"p{q}": float(np.percentile(arr, q)) for q in qs}
    out["mean"] = float(np.mean(arr))
    out["std"] = float(np.std(arr))
    out["n"] = float(arr.size)
    return out


def soft_level(x: float, mid: float) -> float:
    """Saturating map: mid → 0.5, asymptote → 1."""
    if mid <= 0:
        return 0.0
    return float(x / (x + mid))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="s3_videos/compression",
        help="Directory of .mp4 files (recursive)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max videos (0=all)")
    parser.add_argument(
        "-o",
        "--output",
        default="work_fast/feature_calibration.json",
        help="JSON dump path",
    )
    parser.add_argument(
        "--suggest-percentile",
        type=float,
        default=50.0,
        help="Percentile used as soft-norm midpoint (default p50)",
    )
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    videos = sorted(root.rglob("*.mp4"))
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        print(f"No mp4 files under {root}", file=sys.stderr)
        return 1

    print(f"Sweeping {len(videos)} videos under {root}")
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for i, path in enumerate(videos, 1):
        t1 = time.time()
        try:
            full = HEVCFeatureExtractor(str(path)).extract_full()
            summary = full.get("summary") or {}
            row = {"path": str(path), "ok": True}
            for key in RAW_KEYS:
                row[key] = float(summary.get(key, 0.0) or 0.0)
            # Also dump current (possibly saturated) levels for comparison.
            for key in (
                "motion_level",
                "texture_level",
                "noise_level_norm",
                "edge_level",
                "cut_level",
            ):
                row[key] = float(summary.get(key, 0.0) or 0.0)
            rows.append(row)
            elapsed = time.time() - t1
            print(
                f"  [{i}/{len(videos)}] {path.name}: "
                f"motion_p90={row['motion_p90']:.4f} "
                f"noise={row['noise_level']:.4f} "
                f"texture={row['texture']:.3f} "
                f"hf={row['high_freq_energy']:.3f} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            rows.append({"path": str(path), "ok": False, "error": str(exc)})
            print(f"  [{i}/{len(videos)}] {path.name}: FAILED {exc}", file=sys.stderr)

    ok_rows = [r for r in rows if r.get("ok")]
    stats: dict[str, dict[str, float]] = {}
    for key in RAW_KEYS:
        stats[key] = _percentile_report([float(r[key]) for r in ok_rows])

    suggest_q = float(args.suggest_percentile)
    suggestions: dict[str, float] = {}
    for key in MIDPOINT_KEYS:
        rep = stats.get(key) or {}
        if not rep:
            continue
        # Prefer exact percentile key when available; else interpolate via numpy.
        vals = [float(r[key]) for r in ok_rows]
        mid = float(np.percentile(vals, suggest_q))
        suggestions[key] = mid

    # Show how the known saturating clip would score under new soft mids.
    demo: dict[str, Any] = {}
    if suggestions:
        demo_vals = {
            "motion_p90": 0.1343,
            "noise_level": 0.0439,
            "texture": 4.248,
            "high_freq_energy": 4.759,
            "edge_density": 0.0634,
        }
        demo = {
            "example_raw": demo_vals,
            "soft_levels": {
                k: soft_level(demo_vals[k], suggestions[k])
                for k in demo_vals
                if k in suggestions
            },
            "old_hard_clamp_levels": {
                "motion": min(0.1343 / 0.08, 1.0),
                "noise": min(0.0439 / 0.012, 1.0),
                "texture": min(4.248 / 2.0, 1.0),
                "hf": min(4.759 / 2.0, 1.0),
                "edge": min(0.0634 / 0.12, 1.0),
            },
        }

    payload = {
        "dir": str(root),
        "n_videos": len(videos),
        "n_ok": len(ok_rows),
        "elapsed_sec": time.time() - t0,
        "suggest_percentile": suggest_q,
        "suggested_midpoints": suggestions,
        "suggested_constants": {
            "_MOTION_P90_MID": suggestions.get("motion_p90"),
            "_NOISE_LEVEL_MID": suggestions.get("noise_level"),
            "_TEXTURE_MID": suggestions.get("texture"),
            "_HF_ENERGY_MID": suggestions.get("high_freq_energy"),
            "_EDGE_DENSITY_MID": suggestions.get("edge_density"),
        },
        "stats": stats,
        "demo_known_clip": demo,
        "rows": rows,
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Suggested soft-norm midpoints "
          f"(p{suggest_q:g} → level 0.5) ===")
    for name, val in payload["suggested_constants"].items():
        if val is None:
            continue
        print(f"  {name} = {val:.6g}")
    print("\nPercentile tables:")
    for key in MIDPOINT_KEYS:
        rep = stats.get(key) or {}
        if not rep:
            continue
        print(
            f"  {key:18} "
            f"p10={rep.get('p10', 0):.4f} p25={rep.get('p25', 0):.4f} "
            f"p50={rep.get('p50', 0):.4f} p75={rep.get('p75', 0):.4f} "
            f"p90={rep.get('p90', 0):.4f} max={rep.get('p100', 0):.4f}"
        )
    if demo:
        print("\nDemo (terminal clip) soft levels vs old hard clamp:")
        print(f"  soft: {demo['soft_levels']}")
        print(f"  old:  {demo['old_hard_clamp_levels']}")
    print(f"\nWrote {out_path} ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
