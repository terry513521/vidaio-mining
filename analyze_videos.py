"""
Run segment-aware feature extraction on eval / local videos.

Examples:
  python analyze_videos.py
  python analyze_videos.py --video ..\eval_samples\4k_permuted_30s_001.mp4
  python analyze_videos.py --video ..\eval_samples\4k_permuted_30s_001.mp4 --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extractor import HEVCFeatureExtractor
from recipes import adjust_crf_for_volatility, crf_seed_for_threshold, select_recipes

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_samples"

DEFAULT_VIDEOS = [
    EVAL / "4k_permuted_30s_001.mp4",
    EVAL / "8k_permuted_30s_001.mp4",
]


def _print_segment(seg: dict, idx: int) -> None:
    print(
        f"  [{idx:02d}] {seg['start_sec']:6.2f}-{seg['end_sec']:6.2f}s  "
        f"dur={seg['duration']:.2f}s  "
        f"diff={seg['difficulty']:.3f}  "
        f"mot={seg['motion_mean']:.3f}/{seg['motion_p90']:.3f}  "
        f"tex={seg['texture']:.3f}  "
        f"ent={seg['entropy']:.3f}  "
        f"edge={seg['edge_density']:.3f}  "
        f"noise={seg['noise_level']:.3f}  "
        f"hf={seg['high_freq_energy']:.3f}  "
        f"flat={seg['flatness']:.3f}  "
        f"Y={seg['luma_mean']:.3f}±{seg['luma_std']:.3f}  "
        f"sat={seg['sat_mean']:.3f}"
    )


def analyze_one(path: Path) -> dict:
    print(f"\n=== {path.name} ===")
    print("Detecting cuts + extracting per-segment features...")

    extractor = HEVCFeatureExtractor(str(path))
    full = extractor.extract_full()
    summary = full["summary"]
    segments = full["segments"]
    meta = full["meta"]

    print(
        f"  meta: {meta['width']}x{meta['height']}  "
        f"{meta['fps']:.2f}fps  {meta['duration']:.2f}s  "
        f"frames={meta['frame_count']}"
    )
    print(f"  segments: {len(segments)}  cuts: {int(summary['cut_count'])}  "
          f"cut_rate={summary['cut_rate']:.3f}/s")

    print("\n  Per-segment:")
    for i, seg in enumerate(segments):
        _print_segment(seg, i)

    print("\n  Clip summary:")
    keys = [
        "segment_count",
        "cut_count",
        "cut_rate",
        "hard_fraction",
        "worst_difficulty",
        "difficulty_mean",
        "difficulty_p90",
        "duration_weighted_difficulty",
        "motion_mean",
        "motion_std",
        "motion_p90",
        "texture",
        "texture_std",
        "entropy",
        "edge_density",
        "noise_level",
        "high_freq_energy",
        "flatness",
        "luma_mean",
        "luma_std",
        "sat_mean",
        "cut_density",
        "volatility",
    ]
    for key in keys:
        print(f"    {key:32} {summary[key]:.4f}")

    seed85 = adjust_crf_for_volatility(crf_seed_for_threshold(85), summary)
    seed89 = adjust_crf_for_volatility(crf_seed_for_threshold(89), summary)
    seed93 = adjust_crf_for_volatility(crf_seed_for_threshold(93), summary)
    recipes = select_recipes(summary, 89, max_recipes=2)

    mashup_like = (
        summary["cut_rate"] >= 0.15
        or summary["hard_fraction"] >= 0.25
        or summary["volatility"] >= 0.35
        or summary["segment_count"] >= 4
    )

    print(f"\n  CRF seeds: 85->{seed85}  89->{seed89}  93->{seed93}")
    print(f"  Mashup-like? {mashup_like}")
    print("  Recipes:")
    for recipe in recipes:
        print(f"    - {recipe.name}  preset={recipe.preset}  crf_start={recipe.crf_start}")

    return {
        "path": str(path),
        "meta": meta,
        "cut_times_sec": full["cut_times_sec"],
        "segments": segments,
        "summary": summary,
        "crf_seeds": {"85": seed85, "89": seed89, "93": seed93},
        "mashup_like": mashup_like,
        "recipes": [
            {
                "name": r.name,
                "preset": r.preset,
                "crf_start": r.crf_start,
                "params": r.params,
            }
            for r in recipes
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment-aware video feature analysis")
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help="Video path (repeatable). Default: eval_samples 4k+8k",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EVAL / "analysis_segments.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = [Path(v) for v in args.video] if args.video else list(DEFAULT_VIDEOS)

    results: dict = {}
    for path in videos:
        if not path.is_file():
            print(f"SKIP missing: {path}")
            continue
        results[path.name] = analyze_one(path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
