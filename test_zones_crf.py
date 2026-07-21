#!/usr/bin/env python3
"""Test x265 zones: one encode, different QP (q=) per segment, report segment VMAF.

IMPORTANT: Official libx265 ``zones=`` accepts only ``q=`` (force QP) or ``b=``
(bitrate multiplier). ``crf=`` inside zones is INVALID and silently ignored
(ffmpeg logs: \"Invalid value for zones\"). This script uses ``q=``.

Separate from oracle/fleet search. One continuous encode — no concat/splice.

Manual QP per zone:
  python3 test_zones_crf.py \\
    --input ../video/1.mp4 \\
    --base-crf 28 \\
    --segment-qps 22,28,26,24,30,28 \\
    --params \"aq-mode=1:aq-strength=0.9:rd=6:ref=5:bframes=6:rc-lookahead=40:keyint=60:min-keyint=1:scenecut=40\" \\
    --gpu --verify-zones

``--segment-crfs`` is accepted as an alias for ``--segment-qps`` (same integers → q=).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from encoder import encode_hevc
from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary
from scoring import NEG_MODEL, BASE_MODEL, compute_vmaf, score_candidate


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
    p.add_argument("--output", "-o", default="", help="Output mp4 (default: work/zones_test.mp4)")
    p.add_argument(
        "--features",
        default="",
        help="Optional features JSON path (implies --use-cached-features)",
    )
    p.add_argument(
        "--use-cached-features",
        action="store_true",
        help="Load video_features/<stem>.json instead of re-extracting",
    )
    p.add_argument(
        "--save-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When re-extracting, write video_features/<stem>.json (default: yes)",
    )
    p.add_argument(
        "--features-dir",
        default="video_features",
        help="Directory for cached/saved feature JSON (default: video_features)",
    )
    p.add_argument(
        "--base-crf",
        type=int,
        required=True,
        help="Global ffmpeg -crf baseline (fallback; zones use q= QP overrides)",
    )
    p.add_argument(
        "--segment-qps",
        default="",
        help="Manual QP per zone, comma-separated (maps to zones=...,q=N). Preferred.",
    )
    p.add_argument(
        "--segment-crfs",
        default="",
        help="Alias for --segment-qps (same numbers become q=, NOT crf= in zones)",
    )
    p.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prompt for each zone QP after features (default: on if QPs omitted)",
    )
    p.add_argument(
        "--auto-qp",
        action="store_true",
        help="Optional: auto QPs from difficulty (not recommended; prefer manual)",
    )
    p.add_argument(
        "--auto-crf",
        action="store_true",
        help=argparse.SUPPRESS,  # backward alias for --auto-qp
    )
    p.add_argument(
        "--crf-span",
        type=float,
        default=4.0,
        help="Only with --auto-qp: harder segments get lower QP by up to this span",
    )
    p.add_argument(
        "--verify-zones",
        action="store_true",
        help="After encode, check per-zone sizes correlate with QP (zones actually applied)",
    )
    p.add_argument("--params", default="", help="Base -x265-params (zones appended)")
    p.add_argument("--preset", "-p", default="fast", choices=_X265_PRESETS)
    p.add_argument("--profile", default="main")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument("--gpu", action="store_true", help="Prefer Docker libvmaf_cuda")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--vmaf-n-threads", type=int, default=40)
    p.add_argument(
        "--skip-segment-vmaf",
        action="store_true",
        help="Only run whole-file VMAF (faster)",
    )
    p.add_argument(
        "--segment-base-vmaf",
        action="store_true",
        help="Also compute base-model VMAF per segment (slower)",
    )
    p.add_argument(
        "--result-json",
        default="",
        help="Write full result JSON (default: <output>.zones.json)",
    )
    return p.parse_args()


def _load_cached_features(input_path: Path, features_arg: str, features_dir: Path) -> dict[str, Any]:
    if features_arg:
        path = Path(features_arg)
    else:
        path = features_dir / f"{input_path.stem}.json"
    if not path.is_file():
        raise SystemExit(f"features not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid features JSON: {path}")
    print(f"features   : cached {path}", flush=True)
    return data


def _extract_fresh_features(
    input_path: Path,
    *,
    save: bool,
    features_dir: Path,
) -> dict[str, Any]:
    print(f"features   : re-extracting from {input_path} …", flush=True)
    t0 = time.monotonic()
    data = extract_features(input_path)
    elapsed = time.monotonic() - t0
    n_seg = len(data.get("segments") or [])
    print(
        f"features   : extracted {n_seg} segments in {elapsed:.1f}s",
        flush=True,
    )
    if save:
        features_dir.mkdir(parents=True, exist_ok=True)
        out_path = features_dir / f"{input_path.stem}.json"
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"features   : saved {out_path}", flush=True)
    return data


def _resolve_features(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    features_dir = Path(args.features_dir)
    use_cached = bool(args.use_cached_features) or bool(args.features)
    if use_cached:
        return _load_cached_features(input_path, args.features, features_dir)
    return _extract_fresh_features(
        input_path,
        save=bool(args.save_features),
        features_dir=features_dir,
    )


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
                "end_frame": end_f,  # exclusive for x265 zones / trim
                "frame_count": max(1, end_f - start_f),
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
        raise SystemExit("no usable segments in features")
    # Ensure last segment reaches declared frame_count when available.
    if frame_count > 0 and out[-1]["end_frame"] < frame_count:
        out[-1]["end_frame"] = frame_count
        out[-1]["frame_count"] = out[-1]["end_frame"] - out[-1]["start_frame"]
    return out


def _parse_manual_qps(
    segment_qps: str,
    n: int,
) -> list[int]:
    """Parse exact zone QPs — no min/max clamp (use the values you pass)."""
    vals = [int(x.strip()) for x in segment_qps.split(",") if x.strip()]
    if len(vals) != n:
        raise SystemExit(
            f"segment QPs has {len(vals)} values but video has {n} zones. "
            f"Example: --segment-qps {','.join(['28'] * n)}"
        )
    return vals


def _auto_qps_from_difficulty(
    segments: list[dict[str, Any]],
    *,
    base_qp: int,
    qp_span: float,
) -> list[int]:
    diffs = [float(s["difficulty"]) for s in segments]
    d_min = min(diffs)
    d_max = max(diffs)
    span = max(0.0, float(qp_span))
    out: list[int] = []
    for d in diffs:
        if d_max <= d_min + 1e-9:
            qp = float(base_qp)
        else:
            t = (d - d_min) / (d_max - d_min)
            # Harder → lower QP (more bits).
            qp = float(base_qp) + span * (0.5 - t)
        out.append(int(round(qp)))
    return out


def _prompt_manual_qps(
    segments: list[dict[str, Any]],
    *,
    base_qp: int,
) -> list[int]:
    """Ask for one QP per zone after printing zone features."""
    if not sys.stdin.isatty():
        n = len(segments)
        raise SystemExit(
            "Need manual zone QPs. Pass --segment-qps (non-interactive stdin).\n"
            f"  Example: --segment-qps {','.join([str(base_qp)] * n)}"
        )

    print("=" * 72)
    print("MANUAL ZONE QP ENTRY (maps to zones=...,q=N)")
    print(f"  {len(segments)} zones  |  default (Enter) = {base_qp}")
    print("  no clamp — any integer QP you type is used as-is")
    print("  NOTE: this is force-QP (q=), not CRF")
    print("=" * 72)

    out: list[int] = []
    for seg in segments:
        print("-" * 72)
        _print_zone_features(
            {
                **seg,
                "qp": int(base_qp),
                "vmaf_neg": None,
                "vmaf_base": None,
                "vmaf_delta": None,
            },
            live=False,
        )
        while True:
            raw = input(
                f"  → QP for zone[{seg['index']}] (Enter={base_qp}): "
            ).strip()
            if raw == "":
                qp = int(base_qp)
                break
            try:
                qp = int(raw)
            except ValueError:
                print("    invalid integer, try again")
                continue
            break
        out.append(qp)
        print(f"  zone[{seg['index']}] q={qp}", flush=True)

    print("-" * 72)
    print(f"manual QPs  : {out}")
    print(f"re-run tip  : --segment-qps {','.join(str(c) for c in out)}")
    print("-" * 72)
    return out


def _resolve_zone_qps(
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[int]:
    n = len(segments)
    manual = (args.segment_qps or "").strip() or (args.segment_crfs or "").strip()
    if manual:
        src = "--segment-qps" if (args.segment_qps or "").strip() else "--segment-crfs (alias→q=)"
        print(f"qp mode    : manual ({src}, no clamp)", flush=True)
        return _parse_manual_qps(manual, n)

    if args.auto_qp or args.auto_crf:
        print("qp mode    : auto from difficulty", flush=True)
        return _auto_qps_from_difficulty(
            segments,
            base_qp=args.base_crf,
            qp_span=args.crf_span,
        )

    interactive = True if args.interactive is None else bool(args.interactive)
    if not interactive:
        raise SystemExit(
            "Pass --segment-qps for manual zone QPs, or omit it for interactive prompts,\n"
            "or use --auto-qp for difficulty-based QPs.\n"
            f"  Example: --segment-qps {','.join([str(args.base_crf)] * n)}"
        )
    print("qp mode    : manual (interactive, no clamp)", flush=True)
    return _prompt_manual_qps(
        segments,
        base_qp=args.base_crf,
    )


def _strip_zones(params: str) -> str:
    parts = [p for p in (params or "").split(":") if p and not p.startswith("zones=")]
    return ":".join(parts)


def _build_zones_param(segments: list[dict[str, Any]], qps: list[int]) -> str:
    """Official x265 zones: start,end,q=N — end exclusive; abutting ranges."""
    chunks: list[str] = []
    for seg, qp in zip(segments, qps):
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        if end_f <= start_f:
            continue
        chunks.append(f"{start_f},{end_f},q={int(qp)}")
    if not chunks:
        raise SystemExit("empty zones string")
    return "zones=" + "/".join(chunks)


def _zone_output_sizes(
    distorted: Path,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sum video packet sizes per zone (true bitstream spend, not re-encode proxy)."""
    ffprobe_bin = resolve_binary("ffprobe", None)
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time,size,flags",
        "-of",
        "csv=p=0",
        str(distorted),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe packets failed: {(proc.stderr or '')[-500:]}")

    # Need fps to map pts → frame index
    fps_cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(distorted),
    ]
    fps_proc = subprocess.run(fps_cmd, capture_output=True, text=True)
    rate = (fps_proc.stdout or "30/1").strip() or "30/1"
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / max(1e-9, float(den))
    else:
        fps = float(rate)

    # packet lines: pts_time,size,flags  (pts_time may be N/A)
    pkt_sizes: list[tuple[int, int]] = []  # (frame_approx, size)
    for i, line in enumerate((proc.stdout or "").splitlines()):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            size = int(parts[1])
        except ValueError:
            continue
        pts = parts[0]
        if pts not in ("", "N/A"):
            try:
                frame_i = int(round(float(pts) * fps))
            except ValueError:
                frame_i = i
        else:
            frame_i = i
        pkt_sizes.append((frame_i, size))

    rows: list[dict[str, Any]] = []
    for seg in segments:
        sf = int(seg["start_frame"])
        ef = int(seg["end_frame"])
        total = sum(sz for fi, sz in pkt_sizes if sf <= fi < ef)
        # Fallback: if pts mapping empty, use packet index ranges
        if total == 0 and pkt_sizes:
            total = sum(
                sz for fi, sz in pkt_sizes if sf <= fi < ef
            )
        rows.append(
            {
                "index": seg["index"],
                "start_frame": sf,
                "end_frame": ef,
                "bytes": int(total),
                "bytes_per_frame": total / max(1, ef - sf),
            }
        )
    return rows


def _verify_zones_applied(
    *,
    segments: list[dict[str, Any]],
    qps: list[int],
    distorted: Path,
    encode_stderr: str = "",
) -> dict[str, Any]:
    """Check zones were accepted and size roughly follows QP (lower q → larger)."""
    print("-" * 72)
    print("ZONE VERIFY", flush=True)
    invalid = "Invalid value for zones" in (encode_stderr or "")
    if invalid:
        print("  FAIL: encoder reported 'Invalid value for zones' — overrides ignored", flush=True)

    sizes = _zone_output_sizes(distorted, segments)
    print(
        f"{'zone':>4}  {'frames':>13}  {'q':>3}  {'bytes':>10}  {'B/frame':>8}",
        flush=True,
    )
    paired: list[tuple[int, float, int]] = []
    for seg, qp, sz in zip(segments, qps, sizes):
        bpf = float(sz["bytes_per_frame"])
        paired.append((int(qp), bpf, int(seg["index"])))
        print(
            f"{seg['index']:4d}  {seg['start_frame']:6d}-{seg['end_frame']:<6d}  "
            f"{int(qp):3d}  {sz['bytes']:10d}  {bpf:8.1f}",
            flush=True,
        )

    # Spearman-ish check: among pairs with different QP, lower QP should usually be larger.
    checks = 0
    ok = 0
    for i in range(len(paired)):
        for j in range(i + 1, len(paired)):
            qi, bi, zi = paired[i]
            qj, bj, zj = paired[j]
            if qi == qj:
                continue
            checks += 1
            # expect: qi < qj ⇒ bi >= bj (more bits at lower QP)
            if qi < qj and bi >= bj * 0.95:
                ok += 1
            elif qi > qj and bi <= bj / 0.95:
                ok += 1
    ratio = (ok / checks) if checks else 0.0
    passed = (not invalid) and checks > 0 and ratio >= 0.6
    print(
        f"  qp↔size pair agreement: {ok}/{checks} ({ratio:.0%})  "
        f"{'PASS' if passed else 'WEAK/FAIL'}",
        flush=True,
    )
    if not passed and not invalid:
        print(
            "  hint: if all zones look similar, zones may not have applied; "
            "confirm zones= uses q= not crf=",
            flush=True,
        )
    return {
        "invalid_zones_error": invalid,
        "pair_checks": checks,
        "pair_ok": ok,
        "pair_ratio": ratio,
        "passed": passed,
        "sizes": sizes,
    }


def _trim_frame_range(
    src: Path,
    dst: Path,
    *,
    start_frame: int,
    end_frame: int,
    ffmpeg_bin: str,
) -> None:
    """Decode+trim [start_frame, end_frame) so ref/dist frame counts match for VMAF."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # end exclusive → between(n,start,end-1)
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


def _segment_vmaf(
    reference: Path,
    distorted: Path,
    segments: list[dict[str, Any]],
    qps: list[int],
    *,
    args: argparse.Namespace,
    also_base: bool,
) -> list[dict[str, Any]]:
    ffmpeg_bin = resolve_binary("ffmpeg", None)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="zones_seg_vmaf_") as tmp:
        tmp_dir = Path(tmp)
        for seg, qp in zip(segments, qps):
            sf = int(seg["start_frame"])
            ef = int(seg["end_frame"])
            ref_clip = tmp_dir / f"ref_{seg['index']}.mp4"
            dist_clip = tmp_dir / f"dist_{seg['index']}.mp4"
            t0 = time.monotonic()
            _trim_frame_range(reference, ref_clip, start_frame=sf, end_frame=ef, ffmpeg_bin=ffmpeg_bin)
            _trim_frame_range(distorted, dist_clip, start_frame=sf, end_frame=ef, ffmpeg_bin=ffmpeg_bin)
            vmaf_neg = compute_vmaf(
                str(ref_clip),
                str(dist_clip),
                n_subsample=args.vmaf_n_subsample,
                n_threads=args.vmaf_n_threads,
                vmaf_backend="docker",
                vmaf_docker_image="vmaf_ffmpeg",
                vmaf_docker_gpus=bool(args.gpu),
                vmaf_gpu_device=args.gpu_device if args.gpu else None,
                vmaf_gpu_prefer=bool(args.gpu),
                model=NEG_MODEL,
            )
            vmaf_base: Optional[float] = None
            vmaf_delta: Optional[float] = None
            if also_base:
                vmaf_base = compute_vmaf(
                    str(ref_clip),
                    str(dist_clip),
                    n_subsample=args.vmaf_n_subsample,
                    n_threads=args.vmaf_n_threads,
                    vmaf_backend="docker",
                    vmaf_docker_image="vmaf_ffmpeg",
                    vmaf_docker_gpus=bool(args.gpu),
                    vmaf_gpu_device=args.gpu_device if args.gpu else None,
                    vmaf_gpu_prefer=bool(args.gpu),
                    model=BASE_MODEL,
                )
                vmaf_delta = abs(float(vmaf_base) - float(vmaf_neg))
            elapsed = time.monotonic() - t0
            row = {
                "index": seg["index"],
                "start_frame": sf,
                "end_frame": ef,
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "frame_count": seg["frame_count"],
                "difficulty": seg["difficulty"],
                "motion": seg["motion"],
                "motion_p90": seg["motion_p90"],
                "texture": seg["texture"],
                "edge": seg["edge"],
                "noise": seg["noise"],
                "entropy": seg["entropy"],
                "flatness": seg["flatness"],
                "luma_mean": seg["luma_mean"],
                "hf_energy": seg["hf_energy"],
                "qp": int(qp),
                "vmaf_neg": float(vmaf_neg),
                "vmaf_base": None if vmaf_base is None else float(vmaf_base),
                "vmaf_delta": None if vmaf_delta is None else float(vmaf_delta),
                "score_sec": elapsed,
            }
            rows.append(row)
            _print_zone_features(row, live=True)
    return rows


def _print_zone_features(row: dict[str, Any], *, live: bool = False) -> None:
    """Print one zone's features including q= / vmaf_neg."""
    prefix = "  " if live else ""
    base_txt = ""
    if row.get("vmaf_base") is not None and row.get("vmaf_delta") is not None:
        base_txt = f"  vmaf_base={float(row['vmaf_base']):.2f}  vmaf_delta={float(row['vmaf_delta']):.2f}"
    vmaf_txt = (
        f"vmaf_neg={float(row['vmaf_neg']):.2f}{base_txt}"
        if row.get("vmaf_neg") is not None
        else "vmaf_neg=pending"
    )
    qp = row.get("qp", row.get("crf"))  # crf key kept as legacy alias in memory only
    print(
        f"{prefix}zone[{row['index']}]  frames={row['start_frame']}-{row['end_frame']}  "
        f"({row['start_sec']:.2f}s-{row['end_sec']:.2f}s)  "
        f"q={int(qp)}  {vmaf_txt}",
        flush=True,
    )
    print(
        f"{prefix}  features: difficulty={row['difficulty']:.4f}  "
        f"motion={row['motion']:.4f}  motion_p90={row['motion_p90']:.4f}  "
        f"texture={row['texture']:.4f}  edge={row['edge']:.4f}",
        flush=True,
    )
    print(
        f"{prefix}            noise={row['noise']:.4f}  entropy={row['entropy']:.4f}  "
        f"flatness={row['flatness']:.4f}  luma_mean={row['luma_mean']:.4f}  "
        f"hf_energy={row['hf_energy']:.4f}",
        flush=True,
    )


def _attach_zone_compression(
    rows: list[dict[str, Any]],
    *,
    reference: Path,
    distorted: Path,
) -> None:
    ref_sizes = _zone_output_sizes(reference, rows)
    out_sizes = _zone_output_sizes(distorted, rows)
    for row, rs, os_ in zip(rows, ref_sizes, out_sizes):
        rb = int(rs["bytes"])
        ob = int(os_["bytes"])
        row["size_in_bytes"] = rb
        row["size_out_bytes"] = ob
        row["compression_rate"] = (ob / rb) if rb > 0 else 0.0
        row["compression_ratio"] = (rb / ob) if ob > 0 else 0.0


def _print_zone_table(segment_rows: list[dict[str, Any]]) -> None:
    """Final readable table + per-zone feature blocks."""
    if not segment_rows:
        return
    has_base = any(r.get("vmaf_base") is not None for r in segment_rows)
    print("=" * 100)
    print("ZONE RESULTS — q= / VMAF / compression")
    print("=" * 100)
    hdr = (
        f"{'zone':>4}  {'frames':>13}  {'q':>3}  {'vmaf_neg':>8}  "
        f"{'rate':>8}  {'ratio':>7}  {'outMiB':>6}  {'diff':>6}"
    )
    if has_base:
        hdr = (
            f"{'zone':>4}  {'frames':>13}  {'q':>3}  {'vmaf_neg':>8}  "
            f"{'vmaf_base':>9}  {'rate':>8}  {'ratio':>7}  {'outMiB':>6}"
        )
    print(hdr)
    print("-" * 100)
    for r in segment_rows:
        qp = int(r.get("qp", r.get("crf", 0)))
        rate = float(r.get("compression_rate") or 0.0)
        ratio = float(r.get("compression_ratio") or 0.0)
        out_m = float(r.get("size_out_bytes") or 0) / (1024 * 1024)
        if has_base:
            print(
                f"{r['index']:4d}  {r['start_frame']:6d}-{r['end_frame']:<6d}  "
                f"{qp:3d}  {r['vmaf_neg']:8.2f}  "
                f"{float(r['vmaf_base']):9.2f}  {rate:8.4f}  {ratio:6.2f}x  "
                f"{out_m:6.2f}"
            )
        else:
            print(
                f"{r['index']:4d}  {r['start_frame']:6d}-{r['end_frame']:<6d}  "
                f"{qp:3d}  {r['vmaf_neg']:8.2f}  "
                f"{rate:8.4f}  {ratio:6.2f}x  {out_m:6.2f}  "
                f"{r['difficulty']:6.3f}"
            )
    print("-" * 100)
    print("rate = zone_out_bytes / zone_in_bytes   ratio = zone_in / zone_out")
    print("per-zone details:")
    for r in segment_rows:
        _print_zone_features(r, live=False)
        if r.get("compression_rate") is not None:
            print(
                f"  compression: rate={r['compression_rate']:.4f}  "
                f"ratio={r['compression_ratio']:.2f}x  "
                f"in={r['size_in_bytes']/(1024*1024):.2f}MiB  "
                f"out={r['size_out_bytes']/(1024*1024):.2f}MiB"
            )
    print("=" * 100)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    feat = _resolve_features(args, input_path)
    segments = _segments_from_features(feat)
    qps = _resolve_zone_qps(segments, args)
    zones = _build_zones_param(segments, qps)
    if ",crf=" in zones:
        raise SystemExit("internal error: zones must use q=, not crf=")
    base_params = _strip_zones(args.params)
    params = f"{base_params}:{zones}" if base_params else zones

    out = Path(args.output) if args.output else Path("work") / "zones_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json) if args.result_json else Path(str(out) + ".zones.json")

    print(f"input      : {input_path}")
    print(f"output     : {out}")
    print(f"base_crf   : {args.base_crf}  (global fallback; zones use q=)")
    print(f"segments   : {len(segments)}")
    print(f"seg QPs    : {qps}")
    print(f"zones      : {zones}")
    print(f"params     : {params}")
    print(f"preset     : {args.preset}  profile={args.profile}")
    print(f"vmaf thr   : {args.vmaf_threshold}  gpu={args.gpu}")
    print("-" * 72)
    print("NOTE: official zones= supports q= / b= only. crf= in zones is INVALID.")
    print("-" * 72)

    print("ZONE QP PLAN (before encode)")
    print("-" * 72)
    for seg, qp in zip(segments, qps):
        _print_zone_features(
            {
                **seg,
                "qp": int(qp),
                "vmaf_neg": None,
                "vmaf_base": None,
                "vmaf_delta": None,
            },
            live=False,
        )
    print("-" * 72)

    t0 = time.monotonic()
    enc = encode_hevc(
        str(input_path),
        str(out),
        preset=args.preset,
        params=params,
        codec_mode="RC",
        crf=int(args.base_crf),
        encoder="libx265",
        libx265_profile=args.profile,
        progress_reference_path=str(input_path),
        progress_label=f"zones/q= + baseCRF{args.base_crf}",
    )
    encode_sec = time.monotonic() - t0
    encode_stderr = getattr(enc, "stderr_tail", "") or getattr(enc, "stderr", "") or ""
    if not enc.ok:
        raise SystemExit(f"encode failed: {getattr(enc, 'error', None) or encode_stderr[-1000:]}")
    if "Invalid value for zones" in encode_stderr:
        print("WARNING: encoder said 'Invalid value for zones' — check zones string!", flush=True)

    verify: dict[str, Any] = {}
    if args.verify_zones:
        verify = _verify_zones_applied(
            segments=segments,
            qps=qps,
            distorted=out,
            encode_stderr=encode_stderr,
        )

    print("-" * 72)
    print("whole-file VMAF…", flush=True)
    t1 = time.monotonic()
    score = score_candidate(
        str(input_path),
        str(out),
        args.vmaf_threshold,
        vmaf_n_subsample=args.vmaf_n_subsample,
        vmaf_n_threads=args.vmaf_n_threads,
        vmaf_backend="docker",
        vmaf_docker_image="vmaf_ffmpeg",
        vmaf_docker_gpus=bool(args.gpu),
        vmaf_gpu_device=args.gpu_device if args.gpu else None,
        vmaf_gpu_prefer=bool(args.gpu),
        codec_mode="RC",
    )
    score_sec = time.monotonic() - t1

    in_size = input_path.stat().st_size
    out_size = out.stat().st_size
    print("-" * 72)
    print(f"encode_sec        : {encode_sec:.1f}")
    print(f"whole_score_sec   : {score_sec:.1f}")
    print(f"size_in           : {in_size / (1024 * 1024):.2f} MiB")
    print(f"size_out          : {out_size / (1024 * 1024):.2f} MiB")
    print(f"vmaf (NEG)        : {score.vmaf:.6f}")
    print(f"vmaf_base         : {score.vmaf_base}")
    print(f"vmaf_delta        : {score.vmaf_delta}")
    print(f"compression_rate  : {score.compression_rate:.6f}")
    print(f"compression_ratio : {score.compression_ratio:.4f}x")
    print(f"s_f               : {score.s_f:.6f}")
    print(f"reason            : {score.reason}")
    print(f"gates             : enc={score.passed_encoding_gates} delta={score.passed_vmaf_delta_gate}")

    segment_rows: list[dict[str, Any]] = []
    if not args.skip_segment_vmaf:
        print("-" * 72)
        print("per-segment VMAF (trim → libvmaf)…", flush=True)
        segment_rows = _segment_vmaf(
            input_path,
            out,
            segments,
            qps,
            args=args,
            also_base=bool(args.segment_base_vmaf),
        )
        if segment_rows:
            _attach_zone_compression(
                segment_rows, reference=input_path, distorted=out
            )
            _print_zone_table(segment_rows)
            neg_vals = [r["vmaf_neg"] for r in segment_rows]
            print(f"zone vmaf_neg min   : {min(neg_vals):.2f}")
            print(f"zone vmaf_neg max   : {max(neg_vals):.2f}")
            print(f"zone vmaf_neg mean  : {sum(neg_vals) / len(neg_vals):.2f}")
            weights = [max(1, r["end_frame"] - r["start_frame"]) for r in segment_rows]
            wsum = sum(weights)
            wmean = sum(v * w for v, w in zip(neg_vals, weights)) / wsum
            print(f"zone vmaf_neg wmean : {wmean:.2f}  (whole-file {score.vmaf:.2f})")
            print("per-zone q= + VMAF + compression:")
            for r in segment_rows:
                print(
                    f"  zone[{r['index']}]  q={int(r['qp'])}  "
                    f"vmaf_neg={r['vmaf_neg']:.2f}  "
                    f"rate={r['compression_rate']:.4f}  "
                    f"ratio={r['compression_ratio']:.2f}x  "
                    f"out={r['size_out_bytes']/(1024*1024):.2f}MiB"
                )

    payload = {
        "mode": "zones_qp",
        "input": str(input_path),
        "output": str(out),
        "base_crf": int(args.base_crf),
        "segment_qps": qps,
        "zones": zones,
        "params": params,
        "preset": args.preset,
        "profile": args.profile,
        "vmaf_threshold": int(args.vmaf_threshold),
        "encode_sec": encode_sec,
        "whole_score_sec": score_sec,
        "verify_zones": verify,
        "whole": {
            "vmaf_neg": float(score.vmaf),
            "vmaf_base": score.vmaf_base,
            "vmaf_delta": score.vmaf_delta,
            "compression_rate": float(score.compression_rate),
            "compression_ratio": float(score.compression_ratio),
            "s_f": float(score.s_f),
            "reason": score.reason,
            "passed_encoding_gates": bool(score.passed_encoding_gates),
            "passed_vmaf_delta_gate": bool(score.passed_vmaf_delta_gate),
        },
        "segments": segment_rows,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"result_json       : {result_path}")
    return 0 if score.s_f > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
