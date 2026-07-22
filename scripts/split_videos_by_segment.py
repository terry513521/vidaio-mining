#!/usr/bin/env python3
"""Split whole videos into per-segment clips for ML oracle / sweep datasets.

Uses scene/feature segments from ``extract_video_features`` (same cuts as the
encoder pipeline). Writes one subdirectory per source video under the output
root, plus per-video ``manifest.json``, ``features.json``, and global
``manifest.jsonl``.

Per-segment ML features saved (one scalar each, whole-segment summary):
  si, ti, noise, flatness, duration_sec, luma_mean, sat_mean

Defaults:
  source      : ../raw videos          (workspace/raw videos)
  destination : ../segmented videos    (workspace/segmented videos)

Example:
  python3 scripts/split_videos_by_segment.py

  python3 scripts/split_videos_by_segment.py \\
    --input "../raw videos" \\
    --output "../segmented videos" \\
    --workers 4 --resume

  python3 scripts/split_videos_by_segment.py --limit 3
  python3 scripts/split_videos_by_segment.py --copy   # fast, not frame-accurate

Segment trims default to frame-accurate libx264 ``-crf 0`` (same as
``test_zones_zonefile_score._trim_frame_range`` / competition VMAF refs).
Use ``--copy`` for stream copy, or ``--reencode`` for lossy CRF trims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary

# One scalar per segment for ML (~500 rows); see SI/TI (ITU-T P.910).
SEGMENT_ML_FEATURE_KEYS = (
    "si",
    "ti",
    "noise",
    "flatness",
    "duration_sec",
    "luma_mean",
    "sat_mean",
)


def _segment_ml_features(seg: dict[str, Any]) -> dict[str, float]:
    return {key: float(seg.get(key, 0.0) or 0.0) for key in SEGMENT_ML_FEATURE_KEYS}


def _features_json_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    names = payload.get("feature_names")
    return names == list(SEGMENT_ML_FEATURE_KEYS)


def _cached_features_need_refresh(
    feat: dict[str, Any],
    *,
    scene_detector: str = "content",
) -> bool:
    segs = feat.get("segments")
    if not isinstance(segs, list) or not segs:
        return True
    first = segs[0]
    if not isinstance(first, dict):
        return True
    if any(k not in first for k in ("si", "ti", "luma_mean", "sat_mean")):
        return True
    # Re-extract if cached cuts were not produced by the requested detector.
    meta = feat.get("meta") if isinstance(feat.get("meta"), dict) else {}
    sd = meta.get("scene_detect") if isinstance(meta.get("scene_detect"), dict) else {}
    want = (
        "builtin"
        if scene_detector == "builtin"
        else (
            "adaptive"
            if str(scene_detector).lower().startswith("adapt")
            else "content"
        )
    )
    have = str(sd.get("detector") or "")
    if want == "builtin":
        # Old caches with no scene_detect meta are treated as builtin.
        if bool(have) and have != "builtin":
            return True
    elif have != want:
        return True
    # Refresh when sample count protocol changed (default now 16).
    samples = int(meta.get("samples_per_segment") or 0)
    if samples and samples < 16:
        return True
    if not samples:
        return True
    return False


def _segments_from_features(feat: dict[str, Any]) -> list[dict[str, Any]]:
    segs = feat.get("segments")
    if not isinstance(segs, list) or not segs:
        raise ValueError("features JSON has no segments[]")
    meta = feat.get("meta") if isinstance(feat.get("meta"), dict) else {}
    fps = float(meta.get("fps") or feat.get("global", {}).get("fps") or 30.0)
    frame_count = int(meta.get("frame_count") or 0)
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            continue
        start_sec = float(seg.get("start_sec", 0.0) or 0.0)
        end_sec = float(seg.get("end_sec", start_sec) or start_sec)
        start_f = int(round(start_sec * fps))
        end_f = int(round(end_sec * fps))
        if frame_count > 0:
            start_f = max(0, min(start_f, frame_count))
            end_f = max(start_f + 1, min(end_f, frame_count))
        else:
            start_f = max(0, start_f)
            end_f = max(start_f + 1, end_f)
        out.append(
            {
                "index": int(seg.get("index", i)),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_f,
                "end_frame": end_f,
                "frame_count": max(1, end_f - start_f),
                "duration_sec": max(0.01, end_sec - start_sec),
                "si": float(seg.get("si", 0.0) or 0.0),
                "ti": float(seg.get("ti", 0.0) or 0.0),
                "noise": float(seg.get("noise", 0.0) or 0.0),
                "flatness": float(seg.get("flatness", 0.0) or 0.0),
                "luma_mean": float(seg.get("luma_mean", 0.0) or 0.0),
                "sat_mean": float(seg.get("sat_mean", 0.0) or 0.0),
            }
        )
    if not out:
        raise ValueError("no usable segments")
    if frame_count > 0 and out[-1]["end_frame"] < frame_count:
        out[-1]["end_frame"] = frame_count
        out[-1]["frame_count"] = out[-1]["end_frame"] - out[-1]["start_frame"]
        out[-1]["duration_sec"] = out[-1]["frame_count"] / max(fps, 1e-6)
    return out


def _trim_segment(
    src: Path,
    dst: Path,
    *,
    start_frame: int,
    end_frame: int,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str,
    copy_streams: bool,
    lossless: bool,
    crf: int,
    preset: str,
) -> None:
    if copy_streams:
        duration = max(0.01, end_sec - start_sec)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-an",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ]
    else:
        last = max(start_frame, end_frame - 1)
        vf = f"select=between(n\\,{start_frame}\\,{last}),setpts=PTS-STARTPTS"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]
        if lossless:
            cmd.extend(["-crf", "0"])
        else:
            cmd.extend(["-crf", str(int(crf))])
        cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(
            f"trim failed [{start_frame},{end_frame}): "
            + ((proc.stderr or proc.stdout or "")[-800:])
        )


def _load_or_extract_features(
    video: Path,
    *,
    features_dir: Path,
    use_cached: bool,
    force_features: bool,
    scene_detector: str = "content",
    scene_threshold: Optional[float] = None,
    min_scene_len_sec: float = 0.4,
) -> dict[str, Any]:
    feat_path = features_dir / f"{video.stem}.json"

    def _extract() -> dict[str, Any]:
        return extract_features(
            video,
            scene_detector=scene_detector,
            scene_threshold=scene_threshold,
            min_scene_len_sec=float(min_scene_len_sec),
            use_builtin_cuts=(scene_detector == "builtin"),
            samples_per_segment=16,
        )

    if use_cached and feat_path.is_file() and not force_features:
        feat = json.loads(feat_path.read_text(encoding="utf-8"))
        if _cached_features_need_refresh(feat, scene_detector=scene_detector):
            feat = _extract()
            features_dir.mkdir(parents=True, exist_ok=True)
            feat_path.write_text(json.dumps(feat, indent=2), encoding="utf-8")
        return feat
    feat = _extract()
    features_dir.mkdir(parents=True, exist_ok=True)
    feat_path.write_text(json.dumps(feat, indent=2), encoding="utf-8")
    return feat


def _segment_clip_name(seg: dict[str, Any]) -> str:
    idx = int(seg["index"])
    sf = int(seg["start_frame"])
    ef = int(seg["end_frame"])
    return f"seg{idx:02d}_f{sf}-{ef}.mp4"


def _write_video_outputs(
    *,
    video: Path,
    out_dir: Path,
    features_dir: Path,
    seg_rows: list[dict[str, Any]],
    trim_mode: str,
    lossless: bool,
    crf: int,
    preset: str,
) -> None:
    features_payload = {
        "video_stem": video.stem,
        "source_video": str(video.resolve()),
        "feature_names": list(SEGMENT_ML_FEATURE_KEYS),
        "segments": [
            {
                "index": row["index"],
                "file": row["file"],
                "features": _segment_ml_features(row),
            }
            for row in seg_rows
        ],
    }
    (out_dir / "features.json").write_text(
        json.dumps(features_payload, indent=2), encoding="utf-8"
    )

    manifest_payload = {
        "source_video": str(video.resolve()),
        "video_stem": video.stem,
        "output_dir": str(out_dir.resolve()),
        "features_path": str((features_dir / f"{video.stem}.json").resolve()),
        "segment_features_path": str((out_dir / "features.json").resolve()),
        "n_segments": len(seg_rows),
        "trim_mode": trim_mode,
        "lossless": bool(lossless),
        "trim_crf": 0 if lossless else (int(crf) if trim_mode == "reencode" else None),
        "trim_preset": preset if trim_mode == "reencode" else None,
        "feature_names": list(SEGMENT_ML_FEATURE_KEYS),
        "segments": seg_rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )


def split_one_video(
    video: Path,
    *,
    output_root: Path,
    features_dir: Path,
    use_cached_features: bool,
    force_features: bool,
    ffmpeg_bin: str,
    copy_streams: bool,
    lossless: bool,
    crf: int,
    preset: str,
    resume: bool,
    scene_detector: str = "content",
    scene_threshold: Optional[float] = None,
    min_scene_len_sec: float = 0.4,
) -> dict[str, Any]:
    t0 = time.monotonic()
    feat = _load_or_extract_features(
        video,
        features_dir=features_dir,
        use_cached=use_cached_features,
        force_features=force_features,
        scene_detector=scene_detector,
        scene_threshold=scene_threshold,
        min_scene_len_sec=min_scene_len_sec,
    )
    segments = _segments_from_features(feat)
    out_dir = output_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    features_path = out_dir / "features.json"
    if resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = [_segment_clip_name(s) for s in segments]
        have = [s.get("file") for s in existing.get("segments") or []]
        clips_ok = have == expected and all(
            (out_dir / name).is_file() for name in expected
        )
        if clips_ok and _features_json_current(features_path):
            return {
                "video": video.name,
                "stem": video.stem,
                "output_dir": str(out_dir),
                "n_segments": len(segments),
                "skipped": True,
                "elapsed_sec": time.monotonic() - t0,
            }

    clips_only = (
        resume
        and manifest_path.is_file()
        and all((out_dir / _segment_clip_name(s)).is_file() for s in segments)
    )

    trim_mode = (
        "lossless" if lossless else ("reencode" if not copy_streams else "copy")
    )

    seg_rows: list[dict[str, Any]] = []
    for seg in segments:
        fname = _segment_clip_name(seg)
        clip = out_dir / fname
        if not (clips_only and clip.is_file() and clip.stat().st_size > 0):
            _trim_segment(
                video,
                clip,
                start_frame=int(seg["start_frame"]),
                end_frame=int(seg["end_frame"]),
                start_sec=float(seg["start_sec"]),
                end_sec=float(seg["end_sec"]),
                ffmpeg_bin=ffmpeg_bin,
                copy_streams=copy_streams,
                lossless=lossless,
                crf=crf,
                preset=preset,
            )
        ml = _segment_ml_features(seg)
        seg_rows.append(
            {
                "index": int(seg["index"]),
                "file": fname,
                "path": str(clip.resolve()),
                "start_frame": int(seg["start_frame"]),
                "end_frame": int(seg["end_frame"]),
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "duration_sec": float(seg["duration_sec"]),
                "frame_count": int(seg["frame_count"]),
                "size_bytes": int(clip.stat().st_size),
                **ml,
                "features": ml,
            }
        )

    _write_video_outputs(
        video=video,
        out_dir=out_dir,
        features_dir=features_dir,
        seg_rows=seg_rows,
        trim_mode=trim_mode,
        lossless=lossless,
        crf=crf,
        preset=preset,
    )
    return {
        "video": video.name,
        "stem": video.stem,
        "output_dir": str(out_dir),
        "n_segments": len(seg_rows),
        "skipped": False,
        "elapsed_sec": time.monotonic() - t0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=WORKSPACE / "raw videos",
        help="Source directory of whole videos",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=WORKSPACE / "segmented videos",
        help="Destination root (one subdir per video)",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=ROOT / "video_features",
        help="Cache extracted feature JSON here",
    )
    p.add_argument(
        "--use-cached-features",
        action="store_true",
        default=True,
        help="Reuse video_features/<stem>.json when present (default: true)",
    )
    p.add_argument(
        "--no-cached-features",
        action="store_false",
        dest="use_cached_features",
        help="Always re-extract features",
    )
    p.add_argument(
        "--force-features",
        action="store_true",
        help="Re-extract features even when cached JSON exists",
    )
    p.add_argument("--pattern", default="*.mp4")
    p.add_argument("--limit", type=int, default=0, help="Max videos (0=all)")
    p.add_argument("--workers", type=int, default=1, help="Parallel videos")
    p.add_argument("--resume", action="store_true", help="Skip complete videos")
    p.add_argument(
        "--lossless",
        action="store_true",
        default=True,
        help=(
            "Frame-accurate libx264 -crf 0 (default; matches "
            "test_zones_zonefile_score._trim_frame_range)"
        ),
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Stream-copy trims (fast; keyframe-aligned; not competition-aligned)",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Frame-accurate libx264 trim with --crf (lossy; not recommended)",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=12,
        help="Trim quality with --reencode (default: 12)",
    )
    p.add_argument("--preset", default="ultrafast", help="x264 preset for trim")
    p.add_argument(
        "--scene-detector",
        choices=["content", "adaptive", "builtin"],
        default="content",
        help=(
            "Cut detector (default: content = PySceneDetect ContentDetector). "
            "Use adaptive for AdaptiveDetector, builtin for legacy histogram cuts."
        ),
    )
    p.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        help="Optional PySceneDetect threshold (ContentDetector default 27)",
    )
    p.add_argument(
        "--min-scene-len",
        type=float,
        default=0.4,
        help="Minimum scene length in seconds (default: 0.4)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input
    output_root: Path = args.output
    if not input_dir.is_dir():
        raise SystemExit(f"input not found: {input_dir}")

    videos = sorted(input_dir.glob(args.pattern))
    videos = [p for p in videos if p.is_file()]
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"no videos matched {args.pattern} in {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_binary("ffmpeg", None)
    workers = max(1, int(args.workers))

    # Default = lossless (competition-aligned). --copy / --reencode override.
    if args.copy and args.reencode:
        raise SystemExit("use only one of --copy or --reencode")
    if args.copy:
        copy_streams, lossless = True, False
        trim_desc = "stream copy (keyframe-aligned; not competition-aligned)"
    elif args.reencode:
        copy_streams, lossless = False, False
        trim_desc = f"reencode crf={args.crf} preset={args.preset} (frame-accurate)"
    else:
        copy_streams, lossless = False, True
        trim_desc = "lossless crf=0 (frame-accurate; matches competition VMAF trim)"

    print("=" * 88)
    print(f"input      : {input_dir}")
    print(f"output     : {output_root}")
    print(f"videos     : {len(videos)}")
    print(f"workers    : {workers}")
    print(f"trim       : {trim_desc}")
    print(f"features   : {args.features_dir}  cached={args.use_cached_features}")
    print(f"seg feats  : {', '.join(SEGMENT_ML_FEATURE_KEYS)}")
    print(
        f"scenes     : detector={args.scene_detector}"
        + (
            f" thr={args.scene_threshold}"
            if args.scene_threshold is not None
            else ""
        )
        + f" min_len={args.min_scene_len}s"
    )
    print("=" * 88)

    results: list[dict[str, Any]] = []
    t0 = time.monotonic()

    def _run(video: Path) -> dict[str, Any]:
        return split_one_video(
            video,
            output_root=output_root,
            features_dir=args.features_dir,
            use_cached_features=bool(args.use_cached_features),
            force_features=bool(args.force_features),
            ffmpeg_bin=ffmpeg_bin,
            copy_streams=copy_streams,
            lossless=lossless,
            crf=int(args.crf),
            preset=str(args.preset),
            resume=bool(args.resume),
            scene_detector=str(args.scene_detector),
            scene_threshold=args.scene_threshold,
            min_scene_len_sec=float(args.min_scene_len),
        )

    if workers == 1:
        for i, video in enumerate(videos, start=1):
            try:
                row = _run(video)
            except Exception as exc:
                print(f"[{i}/{len(videos)}] FAIL {video.name}: {exc}", flush=True)
                continue
            results.append(row)
            tag = "skip" if row.get("skipped") else f"{row['n_segments']} segs"
            print(
                f"[{i}/{len(videos)}] {video.name}  {tag}  {row['elapsed_sec']:.1f}s",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_run, v): v for v in videos}
            done = 0
            for fut in as_completed(futs):
                video = futs[fut]
                done += 1
                try:
                    row = fut.result()
                except Exception as exc:
                    print(f"[{done}/{len(videos)}] FAIL {video.name}: {exc}", flush=True)
                    continue
                results.append(row)
                tag = "skip" if row.get("skipped") else f"{row['n_segments']} segs"
                print(
                    f"[{done}/{len(videos)}] {video.name}  {tag}  {row['elapsed_sec']:.1f}s",
                    flush=True,
                )

    manifest_jsonl = output_root / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda r: r.get("stem", "")):
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    n_seg = sum(int(r.get("n_segments") or 0) for r in results)
    n_skip = sum(1 for r in results if r.get("skipped"))
    print("=" * 88)
    print(f"done       : {len(results)}/{len(videos)} videos  {n_seg} segment clips")
    print(f"skipped    : {n_skip} videos (already complete)")
    print(f"manifest   : {manifest_jsonl}")
    print(f"wall_sec   : {time.monotonic() - t0:.1f}")
    print("=" * 88)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
