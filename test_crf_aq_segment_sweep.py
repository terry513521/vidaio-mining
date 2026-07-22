#!/usr/bin/env python3
"""Per-segment CRF × aq-strength grid sweep (parallel encode + VMAF).

Uses feature segments (scene cuts) from the video. For each segment:
  1. Trim reference clip once
  2. Sweep CRF [28,38] step 1 × aq-strength [0.4,2.4] step 0.2
  3. Encode + score VMAF on that segment only
  4. Save trials.jsonl / results.csv / optional 3D plots per segment

Default: 121 trials × N segments, 24 total workers (4 per segment × up to 6 segments).

Example:
  python3 test_crf_aq_segment_sweep.py \\
    --input ../raw\\ videos/d7cbca62-b96c-4370-804f-23a930ea3455.mp4 \\
    --workers 24 --segment-workers 4 --preset slow --use-cached-features

  python3 test_crf_aq_segment_sweep.py -i video.mp4 --workers 24 --segment-workers 4 --resume --plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary
from encoder import encode_hevc
from compress_util import measure_compression
from interp_search import format_x265_params, parse_x265_params, propose_feature_x265_params
from scoring import ScoreResult, score_candidate
from test_crf_aq_sweep import (
    TrialResult,
    _append_csv,
    _append_jsonl,
    _build_float_grid,
    _build_int_grid,
    _completed_keys,
    _gpu_score_lock,
    _resolve_base_params,
)


def _source_segment_packet_bytes(
    source: Path,
    segments: list[dict[str, Any]],
    *,
    ffprobe_bin: Optional[str] = None,
) -> dict[int, int]:
    """Sum source video packet bytes per segment (competition-aligned size-in).

    Matches ``test_zones_zonefile_score._zone_packet_bytes`` / ``_attach_zone_compression``.
    Do NOT use lossless VMAF ref file sizes — those inflate compression_ratio ~8–9×.
    """
    probe = resolve_binary("ffprobe", ffprobe_bin)
    proc = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,size",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe packets failed: {(proc.stderr or '')[-500:]}")

    fps_proc = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    rate = (fps_proc.stdout or "30/1").strip() or "30/1"
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / max(1e-9, float(den))
    else:
        fps = float(rate)

    pkt_sizes: list[tuple[int, int]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            sz = int(float(parts[1]))
        except ValueError:
            continue
        pkt_sizes.append((int(round(t * fps)), sz))

    out: dict[int, int] = {}
    for seg in segments:
        sf = int(seg["start_frame"])
        ef = int(seg["end_frame"])
        total = sum(sz for fi, sz in pkt_sizes if sf <= fi < ef)
        out[int(seg["index"])] = int(total)
    return out


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_feature_payload(data: dict[str, Any], *, source: Path) -> dict[str, Any]:
    """Accept extract_video_features JSON or segmented-videos manifest.json."""
    segs = data.get("segments")
    if not isinstance(segs, list) or not segs:
        raise SystemExit(f"features have no segments[]: {source}")

    # Already full extractor format
    if isinstance(data.get("meta"), dict) or "summary_raw" in data or "global" in data:
        return data

    # segmented videos/<stem>/manifest.json (or features.json with frame bounds)
    first = segs[0] if isinstance(segs[0], dict) else {}
    if "start_frame" in first or "start_sec" in first:
        # Infer meta from segment extents when possible
        end_f = max(int(s.get("end_frame") or 0) for s in segs if isinstance(s, dict))
        end_sec = max(float(s.get("end_sec") or 0.0) for s in segs if isinstance(s, dict))
        fps = 30.0
        if end_sec > 0 and end_f > 0:
            fps = float(end_f) / end_sec
        return {
            "video": data.get("video_stem") or source.stem,
            "path": data.get("source_video") or "",
            "meta": {
                "fps": fps,
                "frame_count": end_f,
                "duration": end_sec,
            },
            "global": {},
            "segments": segs,
            "feature_source": str(source),
        }

    raise SystemExit(f"unrecognized features schema: {source}")


def _feature_cache_candidates(
    input_path: Path,
    *,
    features_dir: Path,
    features_arg: str = "",
) -> list[Path]:
    stem = input_path.stem
    cands: list[Path] = []
    if features_arg:
        cands.append(Path(features_arg))
    cands.extend(
        [
            features_dir / f"{stem}.json",
            Path("video_features") / f"{stem}.json",
            ROOT.parent / "segmented videos" / stem / "manifest.json"
            if (ROOT := Path(__file__).resolve().parent)
            else Path(),
            _workspace_root() / "segmented videos" / stem / "manifest.json",
            Path("..") / "segmented videos" / stem / "manifest.json",
            Path("../segmented videos") / stem / "features.json",
            _workspace_root() / "segmented videos" / stem / "features.json",
        ]
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for p in cands:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_features(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    features_dir = Path(args.features_dir)
    want_cache = bool(args.features or args.use_cached_features)

    if want_cache:
        for path in _feature_cache_candidates(
            input_path,
            features_dir=features_dir,
            features_arg=str(args.features or ""),
        ):
            if not path.is_file():
                continue
            print(f"features   : cached {path}", flush=True)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return _normalize_feature_payload(raw, source=path)
        if args.features:
            raise SystemExit(f"features not found: {args.features}")
        print(
            "features   : cache miss "
            f"(looked under {features_dir}/ and segmented videos/); "
            "re-extracting …",
            flush=True,
        )
    else:
        print(f"features   : re-extracting from {input_path} …", flush=True)

    t0 = time.monotonic()
    data = extract_features(input_path)
    print(
        f"features   : extracted {len(data.get('segments') or [])} segments "
        f"in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    if args.save_features:
        features_dir.mkdir(parents=True, exist_ok=True)
        out = features_dir / f"{input_path.stem}.json"
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"features   : saved {out}", flush=True)
    return data


def _segments_from_features(feat: dict[str, Any]) -> list[dict[str, Any]]:
    segs = feat.get("segments")
    if not isinstance(segs, list) or not segs:
        raise SystemExit("features JSON has no segments[]")
    meta = feat.get("meta") if isinstance(feat.get("meta"), dict) else {}
    fps = float(meta.get("fps") or feat.get("global", {}).get("fps") or 30.0)
    frame_count = int(meta.get("frame_count") or 0)
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            continue
        start_sec = float(seg.get("start_sec", 0.0) or 0.0)
        end_sec = float(seg.get("end_sec", start_sec) or start_sec)
        # Prefer explicit frame bounds (segmented-videos manifest)
        if "start_frame" in seg and "end_frame" in seg:
            start_f = int(seg["start_frame"])
            end_f = int(seg["end_frame"])
        else:
            start_f = int(round(start_sec * fps))
            end_f = int(round(end_sec * fps))
        if frame_count > 0:
            start_f = max(0, min(start_f, frame_count))
            end_f = max(start_f + 1, min(end_f, frame_count))
        else:
            start_f = max(0, start_f)
            end_f = max(start_f + 1, end_f)
        feats = seg.get("features") if isinstance(seg.get("features"), dict) else {}
        out.append(
            {
                "index": int(seg.get("index", i)),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_f,
                "end_frame": end_f,
                "frame_count": max(1, end_f - start_f),
                "difficulty": float(seg.get("difficulty", 0.0) or 0.0),
                "motion": float(seg.get("motion", 0.0) or 0.0),
                "texture": float(seg.get("texture", 0.0) or 0.0),
                "edge": float(seg.get("edge", 0.0) or 0.0),
                "noise": float(
                    seg.get("noise", feats.get("noise", 0.0)) or 0.0
                ),
                "si": float(seg.get("si", feats.get("si", 0.0)) or 0.0),
                "ti": float(seg.get("ti", feats.get("ti", 0.0)) or 0.0),
                "flatness": float(
                    seg.get("flatness", feats.get("flatness", 0.0)) or 0.0
                ),
                "luma_mean": float(
                    seg.get("luma_mean", feats.get("luma_mean", 0.0)) or 0.0
                ),
                "sat_mean": float(
                    seg.get("sat_mean", feats.get("sat_mean", 0.0)) or 0.0
                ),
            }
        )
    if not out:
        raise SystemExit("no usable segments")
    if frame_count > 0 and out[-1]["end_frame"] < frame_count:
        out[-1]["end_frame"] = frame_count
        out[-1]["frame_count"] = out[-1]["end_frame"] - out[-1]["start_frame"]
    return out


def _trim_segment(
    src: Path,
    dst: Path,
    *,
    start_frame: int,
    end_frame: int,
    ffmpeg_bin: str,
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
        "ultrafast",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(
            f"trim failed [{start_frame},{end_frame}): "
            + ((proc.stderr or proc.stdout or "")[-800:])
        )


def _segment_dir(work_dir: Path, seg: dict[str, Any]) -> Path:
    return work_dir / f"segment_{int(seg['index']):02d}"


def _parse_segment_filter(text: str, n_segments: int) -> list[int]:
    if not text.strip():
        return list(range(n_segments))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_segments:
            raise SystemExit(f"segment index {idx} out of range 0..{n_segments - 1}")
        if idx not in out:
            out.append(idx)
    return out


def _trial_to_segment_row(seg: dict[str, Any], trial: TrialResult) -> dict[str, Any]:
    row = asdict(trial)
    row.update(
        {
            "segment_index": int(seg["index"]),
            "start_frame": int(seg["start_frame"]),
            "end_frame": int(seg["end_frame"]),
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "difficulty": float(seg["difficulty"]),
            "motion": float(seg["motion"]),
            "texture": float(seg["texture"]),
            "edge": float(seg["edge"]),
            "noise": float(seg["noise"]),
        }
    )
    return row


def _load_segment_rows(trials_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not trials_path.is_file():
        return rows
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _segment_ref_path(seg_dir: Path, seg: dict[str, Any]) -> Path:
    return seg_dir / f"ref_seg{int(seg['index']):02d}.mp4"


def _ensure_segment_ref(
    source: Path,
    ref_path: Path,
    seg: dict[str, Any],
    *,
    ffmpeg_bin: str,
) -> None:
    """Create segment reference trim if missing (re-created after work/ cleanup)."""
    if ref_path.is_file() and ref_path.stat().st_size > 0:
        return
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"trim       : seg[{seg['index']}] frames={seg['start_frame']}-{seg['end_frame']} "
        f"(ref → {ref_path.name})",
        flush=True,
    )
    _trim_segment(
        source,
        ref_path,
        start_frame=int(seg["start_frame"]),
        end_frame=int(seg["end_frame"]),
        ffmpeg_bin=ffmpeg_bin,
    )


def _run_one_segment(
    *,
    trial_idx: int,
    crf: int,
    aq: float,
    source_path: Path,
    ref_path: Path,
    seg: dict[str, Any],
    out_path: Path,
    base_params: dict[str, str],
    preset: str,
    profile: str,
    preprocess: Optional[str],
    vmaf_threshold: int,
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    keep_encode: bool,
    ffmpeg_bin: str,
    source_segment_bytes: int,
) -> TrialResult:
    """Encode segment from full source (ss/t), score VMAF vs segment ref trim.

    Compression uses **source packet bytes** for this segment (competition formula),
    not the lossless VMAF reference file size.
    """
    _ensure_segment_ref(source_path, ref_path, seg, ffmpeg_bin=ffmpeg_bin)
    start_sec = float(seg["start_sec"])
    duration = max(0.01, float(seg["end_sec"]) - start_sec)
    src_bytes = max(0, int(source_segment_bytes))

    params = dict(base_params)
    params["aq-strength"] = f"{round(float(aq), 3):g}"
    params_str = format_x265_params(params)

    t0 = time.monotonic()
    enc = encode_hevc(
        str(source_path),
        str(out_path),
        preset=preset,
        params=params_str,
        codec_mode="RC",
        crf=int(crf),
        encoder="libx265",
        preprocess=preprocess,
        libx265_profile=profile,
        ss=start_sec,
        t=duration,
        progress_reference_path=str(ref_path),
        progress_reference_bytes=src_bytes if src_bytes > 0 else None,
        progress_label=f"seg{seg['index']} CRF{crf}/aq{aq:.1f}",
    )
    encode_sec = time.monotonic() - t0
    if not enc.ok:
        return TrialResult(
            trial_idx=trial_idx,
            crf=int(crf),
            aq_strength=round(float(aq), 3),
            params=params_str,
            encode_ok=False,
            encode_sec=encode_sec,
            score_sec=0.0,
            vmaf_neg=0.0,
            vmaf_base=None,
            vmaf_delta=None,
            compression_rate=1.0,
            compression_ratio=1.0,
            s_f=0.0,
            reason="encode_failed",
            gates_ok=False,
            passed_encoding_gates=False,
            passed_vmaf_delta_gate=False,
            size_out_bytes=0,
            output_path=str(out_path),
            error=(enc.stderr_tail or "")[-400:],
        )

    t1 = time.monotonic()
    size_out = out_path.stat().st_size if out_path.is_file() else 0
    # Competition: rate = out/in, ratio = in/out using SOURCE segment bytes.
    if src_bytes > 0:
        rate, _ratio = measure_compression(
            str(ref_path),
            str(out_path),
            reference_bytes=src_bytes,
        )
        rate_override: Optional[float] = float(rate)
    else:
        rate_override = None

    def _score() -> ScoreResult:
        return score_candidate(
            str(ref_path),
            str(out_path),
            vmaf_threshold,
            vmaf_n_subsample=vmaf_n_subsample,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_backend="docker",
            vmaf_docker_image="vmaf_ffmpeg",
            vmaf_docker_gpus=bool(use_gpu),
            vmaf_gpu_device=gpu_device if use_gpu else None,
            vmaf_gpu_prefer=bool(use_gpu),
            codec_mode="RC",
            compression_rate_override=rate_override,
        )

    if use_gpu:
        with _gpu_score_lock:
            score = _score()
    else:
        score = _score()
    score_sec = time.monotonic() - t1

    if not keep_encode and out_path.is_file():
        try:
            out_path.unlink()
        except OSError:
            pass

    gates_ok = bool(score.passed_encoding_gates and score.passed_vmaf_delta_gate)
    return TrialResult(
        trial_idx=trial_idx,
        crf=int(crf),
        aq_strength=round(float(aq), 3),
        params=params_str,
        encode_ok=True,
        encode_sec=encode_sec,
        score_sec=score_sec,
        vmaf_neg=float(score.vmaf),
        vmaf_base=None if score.vmaf_base is None else float(score.vmaf_base),
        vmaf_delta=None if score.vmaf_delta is None else float(score.vmaf_delta),
        compression_rate=float(score.compression_rate),
        compression_ratio=float(score.compression_ratio),
        s_f=float(score.s_f),
        reason=str(score.reason),
        gates_ok=gates_ok,
        passed_encoding_gates=bool(score.passed_encoding_gates),
        passed_vmaf_delta_gate=bool(score.passed_vmaf_delta_gate),
        size_out_bytes=int(size_out),
        output_path=str(out_path),
        error="",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True)
    p.add_argument(
        "--work-dir",
        default="",
        help="Output root (default: work/crf_aq_segment_sweep/<stem>)",
    )
    p.add_argument("--features", default="")
    p.add_argument("--use-cached-features", action="store_true")
    p.add_argument(
        "--save-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--features-dir", default="video_features")
    p.add_argument(
        "--segments",
        default="",
        help="Comma segment indices to run (default: all)",
    )
    p.add_argument("--crf-min", type=int, default=28)
    p.add_argument("--crf-max", type=int, default=38)
    p.add_argument("--crf-step", type=int, default=1)
    p.add_argument("--aq-min", type=float, default=0.4)
    p.add_argument("--aq-max", type=float, default=2.4)
    p.add_argument("--aq-step", type=float, default=0.2)
    p.add_argument("--params", default="")
    p.add_argument("--preset", "-p", default="slow")
    p.add_argument("--profile", default="main")
    p.add_argument("--preprocess", default="none")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument(
        "--vmaf-n-threads",
        type=int,
        default=0,
        help="libvmaf threads per score job (0=auto from --workers)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Total worker budget (default: 24)",
    )
    p.add_argument(
        "--segment-workers",
        type=int,
        default=4,
        help="Workers per segment grid sweep (default: 4; 24/4 → 6 segments in parallel)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-encodes", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-html", action="store_true")
    p.add_argument("--plot-gates-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    feat = _resolve_features(args, input_path)
    segments = _segments_from_features(feat)
    seg_indices = _parse_segment_filter(args.segments, len(segments))
    segments = [s for s in segments if int(s["index"]) in seg_indices]

    total_workers = max(1, int(args.workers))
    segment_workers = max(1, int(args.segment_workers))
    if segment_workers > total_workers:
        raise SystemExit(
            f"--segment-workers ({segment_workers}) cannot exceed --workers ({total_workers})"
        )
    parallel_segments = max(1, total_workers // segment_workers)
    if total_workers % segment_workers != 0:
        print(
            f"WARNING    : workers={total_workers} not divisible by segment-workers="
            f"{segment_workers}; using {parallel_segments} segments in parallel "
            f"({parallel_segments * segment_workers} active slots)",
            flush=True,
        )
    vmaf_n_threads = int(args.vmaf_n_threads)
    if vmaf_n_threads <= 0:
        # Keep CPU libvmaf from oversubscribing when many jobs run at once.
        vmaf_n_threads = max(2, min(6, 48 // total_workers))
    crfs = _build_int_grid(args.crf_min, args.crf_max, args.crf_step)
    aqs = _build_float_grid(args.aq_min, args.aq_max, args.aq_step)
    grid = [(crf, aq) for crf in crfs for aq in aqs]

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("work") / "crf_aq_segment_sweep" / input_path.stem
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        import shutil

        for child in work_dir.glob("segment_*"):
            if child.is_dir():
                shutil.rmtree(child)
        for name in ("summary.json", "all_trials.jsonl"):
            p = work_dir / name
            if p.exists():
                p.unlink()

    base_params = _resolve_base_params(args, input_path)
    base_params.pop("aq-strength", None)
    preprocess = None if str(args.preprocess).lower().strip() in {"", "none"} else str(
        args.preprocess
    ).lower().strip()

    ffmpeg_bin = resolve_binary("ffmpeg", None)

    print("source_bytes: probing packet sizes per segment …", flush=True)
    source_bytes_by_seg = _source_segment_packet_bytes(input_path, segments)
    for seg in segments:
        idx = int(seg["index"])
        sb = int(source_bytes_by_seg.get(idx, 0))
        print(
            f"  seg[{idx}] frames={seg['start_frame']}-{seg['end_frame']}  "
            f"source_pkt={sb / 1e6:.2f}MB",
            flush=True,
        )

    print("=" * 88)
    print(f"input      : {input_path}")
    print(f"work_dir   : {work_dir}")
    print(f"segments   : {[int(s['index']) for s in segments]}")
    print(f"grid/seg   : {len(grid)} points  (CRF {crfs[0]}..{crfs[-1]}, AQ {aqs[0]}..{aqs[-1]})")
    print(f"total      : {len(grid) * len(segments)} trials max")
    print(
        f"workers    : total={total_workers}  per_segment={segment_workers}  "
        f"parallel_segments={parallel_segments}"
    )
    print(f"preset     : {args.preset}")
    print(
        f"vmaf       : thr={args.vmaf_threshold} backend={'GPU' if args.gpu else 'CPU'} "
        f"threads/job={vmaf_n_threads}"
    )
    print("comp_ratio : source packet bytes / encoded size (competition formula)")
    print("=" * 88)

    # Build job list grouped by segment.
    jobs_by_segment: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for seg in segments:
        seg_dir = _segment_dir(work_dir, seg)
        seg_dir.mkdir(parents=True, exist_ok=True)
        trials_path = seg_dir / "trials.jsonl"
        csv_path = seg_dir / "results.csv"
        if args.force:
            for p in (trials_path, csv_path, seg_dir / "summary.json"):
                if p.exists():
                    p.unlink()
        done = _completed_keys(trials_path) if args.resume else set()
        ref_path = _segment_ref_path(seg_dir, seg)
        encodes_dir = seg_dir / "encodes"
        encodes_dir.mkdir(parents=True, exist_ok=True)
        key_to_idx = {(int(c), round(float(a), 4)): i for i, (c, a) in enumerate(grid)}
        for crf, aq in grid:
            key = (int(crf), round(float(aq), 4))
            if key in done:
                continue
            jobs_by_segment[int(seg["index"])].append(
                (
                    key_to_idx[key],
                    seg,
                    int(crf),
                    float(aq),
                    ref_path,
                    encodes_dir / f"crf{crf}_aq{aq:.1f}.mp4",
                    trials_path,
                )
            )

    jobs = [j for seg_jobs in jobs_by_segment.values() for j in seg_jobs]

    if not jobs:
        print("nothing to do (all segment grid points already done)", flush=True)
    else:
        t_wall0 = time.monotonic()
        completed = 0
        progress_lock = threading.Lock()

        def _job(
            item: tuple[Any, ...],
        ) -> dict[str, Any]:
            trial_idx, seg, crf, aq, ref_path, out_path, trials_path = item
            trial = _run_one_segment(
                trial_idx=trial_idx,
                crf=crf,
                aq=aq,
                source_path=input_path,
                ref_path=ref_path,
                seg=seg,
                out_path=out_path,
                base_params=base_params,
                preset=args.preset,
                profile=args.profile,
                preprocess=preprocess,
                vmaf_threshold=args.vmaf_threshold,
                vmaf_n_threads=vmaf_n_threads,
                vmaf_n_subsample=args.vmaf_n_subsample,
                use_gpu=bool(args.gpu),
                gpu_device=args.gpu_device,
                keep_encode=bool(args.keep_encodes),
                ffmpeg_bin=ffmpeg_bin,
                source_segment_bytes=int(
                    source_bytes_by_seg.get(int(seg["index"]), 0)
                ),
            )
            row = _trial_to_segment_row(seg, trial)
            csv_path = trials_path.parent / "results.csv"
            _append_jsonl(trials_path, row)
            _append_csv(csv_path, trial, write_header=False)
            return row

        def _run_segment(seg_idx: int, seg_jobs: list[tuple[Any, ...]]) -> None:
            nonlocal completed
            if seg_jobs:
                seg = seg_jobs[0][1]
                ref_path = seg_jobs[0][4]
                _ensure_segment_ref(
                    input_path, ref_path, seg, ffmpeg_bin=ffmpeg_bin
                )
            with ThreadPoolExecutor(max_workers=segment_workers) as pool:
                futs = [pool.submit(_job, j) for j in seg_jobs]
                for fut in as_completed(futs):
                    row = fut.result()
                    with progress_lock:
                        completed += 1
                        c = completed
                        elapsed = time.monotonic() - t_wall0
                        rate = c / max(elapsed, 1e-6)
                        eta = (len(jobs) - c) / max(rate, 1e-9)
                    print(
                        f"[{c}/{len(jobs)}] seg={row['segment_index']} "
                        f"crf={row['crf']} aq={row['aq_strength']:.1f}  "
                        f"vmaf={row['vmaf_neg']:.2f}  ratio={row['compression_ratio']:.2f}x  "
                        f"s_f={row['s_f']:.4f}  gates={row['gates_ok']}  "
                        f"reason={row['reason']}  "
                        f"ETA={eta/60:.1f}m",
                        flush=True,
                    )

        pending_segments = sorted(jobs_by_segment.keys())
        with ThreadPoolExecutor(max_workers=parallel_segments) as outer:
            outer_futs = [
                outer.submit(_run_segment, seg_idx, jobs_by_segment[seg_idx])
                for seg_idx in pending_segments
            ]
            for fut in as_completed(outer_futs):
                fut.result()

    # Per-segment summaries + global summary.
    segment_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for seg in segments:
        seg_dir = _segment_dir(work_dir, seg)
        trials_path = seg_dir / "trials.jsonl"
        rows = _load_segment_rows(trials_path)
        ok_rows = [r for r in rows if r.get("encode_ok")]
        all_rows.extend(ok_rows)
        best = max(ok_rows, key=lambda r: float(r.get("s_f") or 0.0), default=None)
        best_gated = max(
            (r for r in ok_rows if r.get("gates_ok")),
            key=lambda r: float(r.get("s_f") or 0.0),
            default=None,
        )
        seg_summary = {
            "segment_index": int(seg["index"]),
            "start_frame": int(seg["start_frame"]),
            "end_frame": int(seg["end_frame"]),
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "difficulty": float(seg["difficulty"]),
            "motion": float(seg["motion"]),
            "texture": float(seg["texture"]),
            "edge": float(seg["edge"]),
            "n_trials": len(rows),
            "n_ok": len(ok_rows),
            "best_s_f": best,
            "best_gated_s_f": best_gated,
            "trials_jsonl": str(trials_path),
        }
        (seg_dir / "summary.json").write_text(
            json.dumps(seg_summary, indent=2), encoding="utf-8"
        )
        segment_summaries.append(seg_summary)
        if best is not None:
            print(
                f"seg[{seg['index']}] best: crf={best['crf']} aq={best['aq_strength']} "
                f"s_f={best['s_f']:.4f} vmaf={best['vmaf_neg']:.2f} "
                f"ratio={best['compression_ratio']:.2f}x",
                flush=True,
            )

    summary = {
        "input": str(input_path),
        "work_dir": str(work_dir),
        "mode": "per_segment",
        "segments": segment_summaries,
        "grid": {
            "crf_min": args.crf_min,
            "crf_max": args.crf_max,
            "crf_step": args.crf_step,
            "aq_min": args.aq_min,
            "aq_max": args.aq_max,
            "aq_step": args.aq_step,
            "n_points_per_segment": len(grid),
        },
        "vmaf_threshold": args.vmaf_threshold,
        "workers": total_workers,
        "segment_workers": segment_workers,
        "parallel_segments": parallel_segments,
        "vmaf_n_threads": vmaf_n_threads,
        "n_trials_ok_total": len(all_rows),
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    all_trials_path = work_dir / "all_trials.jsonl"
    with all_trials_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            for row in _load_segment_rows(_segment_dir(work_dir, seg) / "trials.jsonl"):
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("=" * 88)
    print(f"summary    : {summary_path}")
    print(f"all_trials : {all_trials_path}")

    if args.plot:
        try:
            from plot_crf_aq_sweep import plot_sweep_3d

            for seg in segments:
                seg_dir = _segment_dir(work_dir, seg)
                title = (
                    f"{input_path.name}  seg[{seg['index']}] "
                    f"frames {seg['start_frame']}-{seg['end_frame']}"
                )
                outputs = plot_sweep_3d(
                    work_dir=seg_dir,
                    gates_only=bool(args.plot_gates_only),
                    html=bool(args.plot_html),
                    title=title,
                )
                for kind, path in outputs.items():
                    print(f"plot seg[{seg['index']}] {kind}: {path}")
        except Exception as exc:
            print(f"plot failed: {exc}", flush=True)
            raise SystemExit(1) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
