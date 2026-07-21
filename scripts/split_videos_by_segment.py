#!/usr/bin/env python3
"""Split whole videos into per-segment clips for ML oracle / sweep datasets.

Uses scene/feature segments from ``extract_video_features`` (same cuts as the
encoder pipeline). Writes one subdirectory per source video under the output
root, plus a per-video ``manifest.json`` and global ``manifest.jsonl``.

Defaults:
  source      : ../raw videos          (workspace/raw videos)
  destination : ../segmented videos    (workspace/segmented videos)

Example:
  python3 scripts/split_videos_by_segment.py

  python3 scripts/split_videos_by_segment.py \\
    --input "../raw videos" \\
    --output "../segmented videos" \\
    --workers 4 --resume

  python3 scripts/split_videos_by_segment.py --limit 3 --lossless
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
                "difficulty": float(seg.get("difficulty", 0.0) or 0.0),
                "motion": float(seg.get("motion", 0.0) or 0.0),
                "motion_p90": float(seg.get("motion_p90", 0.0) or 0.0),
                "texture": float(seg.get("texture", 0.0) or 0.0),
                "edge": float(seg.get("edge", 0.0) or 0.0),
                "noise": float(seg.get("noise", 0.0) or 0.0),
                "entropy": float(seg.get("entropy", 0.0) or 0.0),
                "flatness": float(seg.get("flatness", 0.0) or 0.0),
                "luma_mean": float(seg.get("luma_mean", 0.0) or 0.0),
                "hf_energy": float(seg.get("hf_energy", 0.0) or 0.0),
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
    ffmpeg_bin: str,
    lossless: bool,
    crf: int,
    preset: str,
) -> None:
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
) -> dict[str, Any]:
    feat_path = features_dir / f"{video.stem}.json"
    if use_cached and feat_path.is_file() and not force_features:
        return json.loads(feat_path.read_text(encoding="utf-8"))
    feat = extract_features(video)
    features_dir.mkdir(parents=True, exist_ok=True)
    feat_path.write_text(json.dumps(feat, indent=2), encoding="utf-8")
    return feat


def _segment_clip_name(seg: dict[str, Any]) -> str:
    idx = int(seg["index"])
    sf = int(seg["start_frame"])
    ef = int(seg["end_frame"])
    return f"seg{idx:02d}_f{sf}-{ef}.mp4"


def split_one_video(
    video: Path,
    *,
    output_root: Path,
    features_dir: Path,
    use_cached_features: bool,
    force_features: bool,
    ffmpeg_bin: str,
    lossless: bool,
    crf: int,
    preset: str,
    resume: bool,
) -> dict[str, Any]:
    t0 = time.monotonic()
    feat = _load_or_extract_features(
        video,
        features_dir=features_dir,
        use_cached=use_cached_features,
        force_features=force_features,
    )
    segments = _segments_from_features(feat)
    out_dir = output_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = [_segment_clip_name(s) for s in segments]
        have = [s.get("file") for s in existing.get("segments") or []]
        if have == expected and all((out_dir / name).is_file() for name in expected):
            return {
                "video": video.name,
                "stem": video.stem,
                "output_dir": str(out_dir),
                "n_segments": len(segments),
                "skipped": True,
                "elapsed_sec": time.monotonic() - t0,
            }

    seg_rows: list[dict[str, Any]] = []
    for seg in segments:
        fname = _segment_clip_name(seg)
        clip = out_dir / fname
        if not (resume and clip.is_file() and clip.stat().st_size > 0):
            _trim_segment(
                video,
                clip,
                start_frame=int(seg["start_frame"]),
                end_frame=int(seg["end_frame"]),
                ffmpeg_bin=ffmpeg_bin,
                lossless=lossless,
                crf=crf,
                preset=preset,
            )
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
                "features": {
                    k: seg[k]
                    for k in (
                        "difficulty",
                        "motion",
                        "motion_p90",
                        "texture",
                        "edge",
                        "noise",
                        "entropy",
                        "flatness",
                        "luma_mean",
                        "hf_energy",
                    )
                },
            }
        )

    payload = {
        "source_video": str(video.resolve()),
        "video_stem": video.stem,
        "output_dir": str(out_dir.resolve()),
        "features_path": str((features_dir / f"{video.stem}.json").resolve()),
        "n_segments": len(seg_rows),
        "lossless": bool(lossless),
        "trim_crf": 0 if lossless else int(crf),
        "trim_preset": preset,
        "global_features": feat.get("global") or {},
        "segments": seg_rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        help="Trim with libx264 -crf 0 (large files, best for VMAF oracle)",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=12,
        help="Trim quality when not --lossless (default: 12)",
    )
    p.add_argument("--preset", default="ultrafast", help="x264 preset for trim")
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

    print("=" * 88)
    print(f"input      : {input_dir}")
    print(f"output     : {output_root}")
    print(f"videos     : {len(videos)}")
    print(f"workers    : {workers}")
    print(
        f"trim       : {'lossless crf=0' if args.lossless else f'crf={args.crf} preset={args.preset}'}"
    )
    print(f"features   : {args.features_dir}  cached={args.use_cached_features}")
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
            lossless=bool(args.lossless),
            crf=int(args.crf),
            preset=str(args.preset),
            resume=bool(args.resume),
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
