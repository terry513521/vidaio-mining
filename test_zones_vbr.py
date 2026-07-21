#!/usr/bin/env python3
"""Test x265 zones VBR: one encode, different bitrate multiplier (b=) per segment.

Separate from CRF zones (test_zones_crf.py) and from fleet/oracle search.
One continuous encode — no concat/splice.

x265 zone bitrate control is relative:
  global -b:v 4M  +  zones=...,b=1.4  → that zone targets ~1.4 × average bitrate.

Manual multipliers (recommended):
  python3 test_zones_vbr.py \\
    --input ../video/1.mp4 \\
    --bitrate 4M \\
    --segment-bs 0.7,1.2,1.4,1.0,1.2,1.3 \\
    --params "aq-mode=1:aq-strength=0.9:rd=6:ref=5:bframes=6:rc-lookahead=40:keyint=60:min-keyint=1:scenecut=40" \\
    --gpu

Or derive average bitrate from a compression-rate target:
  python3 test_zones_vbr.py \\
    --input ../video/1.mp4 \\
    --target-compression-rate 0.04 \\
    --segment-bs 0.7,1.2,1.4,1.0,1.2,1.3 \\
    --params "..." --gpu

Omit --segment-bs to be prompted for each zone after feature extract.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from compress_util import bitrate_for_compression_rate
from encoder import encode_hevc
from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary
from scoring import NEG_MODEL, BASE_MODEL, compute_vmaf, probe_video, score_candidate


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
    p.add_argument("--output", "-o", default="", help="Output mp4 (default: work/zones_vbr_test.mp4)")
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
        "--bitrate",
        "-b",
        default="",
        help="Global average bitrate for ABR (e.g. 4M, 2500k)",
    )
    p.add_argument(
        "--target-compression-rate",
        type=float,
        default=None,
        help="If set and --bitrate omitted: derive -b:v from source size × rate",
    )
    p.add_argument(
        "--segment-bs",
        default="",
        help="Manual bitrate multipliers, comma-separated, one per zone (e.g. 0.7,1.2,1.4)",
    )
    p.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prompt for each zone b= after features (default: on if --segment-bs omitted)",
    )
    p.add_argument(
        "--auto-b",
        action="store_true",
        help="Optional: auto b= from difficulty (not recommended; prefer manual)",
    )
    p.add_argument(
        "--b-span",
        type=float,
        default=0.6,
        help="Only with --auto-b: harder segments get higher b by up to this span around 1.0",
    )
    p.add_argument("--b-min", type=float, default=0.2)
    p.add_argument("--b-max", type=float, default=3.0)
    p.add_argument("--params", default="", help="Base -x265-params (zones appended)")
    p.add_argument("--preset", "-p", default="fast", choices=_X265_PRESETS)
    p.add_argument("--profile", default="main")
    p.add_argument("--twopass", action="store_true", help="libx265 2-pass ABR (often better for zones)")
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
) -> tuple[str, float]:
    explicit = str(bitrate or "").strip()
    if explicit:
        mbps = _parse_bitrate_mbps(explicit)
        if mbps is None or mbps <= 0:
            raise SystemExit(f"invalid --bitrate: {explicit}")
        return explicit, mbps
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
    br = bitrate_for_compression_rate(
        source_bytes=input_path.stat().st_size,
        duration_sec=duration,
        compression_rate=rate,
    )
    mbps = _parse_bitrate_mbps(br)
    if mbps is None or mbps <= 0:
        raise SystemExit(f"failed to derive bitrate from rate={rate}")
    return br, mbps


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
    print(f"features   : extracted {n_seg} segments in {elapsed:.1f}s", flush=True)
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
                "end_frame": end_f,
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
    if frame_count > 0 and out[-1]["end_frame"] < frame_count:
        out[-1]["end_frame"] = frame_count
        out[-1]["frame_count"] = out[-1]["end_frame"] - out[-1]["start_frame"]
    return out


def _parse_manual_bs(
    segment_bs: str,
    n: int,
    *,
    b_min: float,
    b_max: float,
) -> list[float]:
    vals = [float(x.strip()) for x in segment_bs.split(",") if x.strip()]
    if len(vals) != n:
        raise SystemExit(
            f"--segment-bs has {len(vals)} values but video has {n} zones. "
            f"Example: --segment-bs {','.join(['1.0'] * n)}"
        )
    return [max(b_min, min(b_max, v)) for v in vals]


def _auto_bs_from_difficulty(
    segments: list[dict[str, Any]],
    *,
    b_span: float,
    b_min: float,
    b_max: float,
) -> list[float]:
    diffs = [float(s["difficulty"]) for s in segments]
    d_min = min(diffs)
    d_max = max(diffs)
    span = max(0.0, float(b_span))
    out: list[float] = []
    for d in diffs:
        if d_max <= d_min + 1e-9:
            b = 1.0
        else:
            t = (d - d_min) / (d_max - d_min)
            # Harder → higher bitrate multiplier.
            b = 1.0 + span * (t - 0.5)
        out.append(round(max(b_min, min(b_max, b)), 3))
    return out


def _prompt_manual_bs(
    segments: list[dict[str, Any]],
    *,
    target_mbps: float,
    b_min: float,
    b_max: float,
) -> list[float]:
    if not sys.stdin.isatty():
        n = len(segments)
        raise SystemExit(
            "Need manual zone b= multipliers. Pass --segment-bs (non-interactive stdin).\n"
            f"  Example: --segment-bs {','.join(['1.0'] * n)}"
        )

    print("=" * 72)
    print("MANUAL ZONE BITRATE MULTIPLIER (b=) ENTRY")
    print(f"  {len(segments)} zones  |  default (Enter) = 1.0")
    print(f"  global average bitrate ≈ {target_mbps:.3f} Mbps")
    print(f"  zone target ≈ b × {target_mbps:.3f} Mbps")
    print(f"  allowed range: [{b_min}, {b_max}]")
    print("=" * 72)

    out: list[float] = []
    for seg in segments:
        print("-" * 72)
        _print_zone_features(
            {
                **seg,
                "b": 1.0,
                "approx_mbps": target_mbps,
                "vmaf_neg": None,
                "vmaf_base": None,
                "vmaf_delta": None,
            },
            live=False,
        )
        while True:
            raw = input(
                f"  → b= for zone[{seg['index']}] "
                f"(Enter=1.0, range {b_min}-{b_max}): "
            ).strip()
            if raw == "":
                b = 1.0
                break
            try:
                b = float(raw)
            except ValueError:
                print("    invalid number, try again")
                continue
            if b < b_min or b > b_max:
                print(f"    out of range [{b_min}, {b_max}], try again")
                continue
            break
        out.append(round(b, 3))
        print(
            f"  zone[{seg['index']}] b={b:.3f}  (~{b * target_mbps:.3f} Mbps)",
            flush=True,
        )

    print("-" * 72)
    print(f"manual b=   : {out}")
    print(f"re-run tip  : --segment-bs {','.join(f'{b:.3g}' for b in out)}")
    print("-" * 72)
    return out


def _resolve_zone_bs(
    segments: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    target_mbps: float,
) -> list[float]:
    n = len(segments)
    if args.segment_bs.strip():
        print("b mode     : manual (--segment-bs)", flush=True)
        return _parse_manual_bs(
            args.segment_bs,
            n,
            b_min=args.b_min,
            b_max=args.b_max,
        )

    if args.auto_b:
        print("b mode     : auto (--auto-b from difficulty)", flush=True)
        return _auto_bs_from_difficulty(
            segments,
            b_span=args.b_span,
            b_min=args.b_min,
            b_max=args.b_max,
        )

    interactive = True if args.interactive is None else bool(args.interactive)
    if not interactive:
        raise SystemExit(
            "Pass --segment-bs for manual zone multipliers, or omit it for interactive prompts,\n"
            "or use --auto-b for difficulty-based multipliers.\n"
            f"  Example: --segment-bs {','.join(['1.0'] * n)}"
        )
    print("b mode     : manual (interactive)", flush=True)
    return _prompt_manual_bs(
        segments,
        target_mbps=target_mbps,
        b_min=args.b_min,
        b_max=args.b_max,
    )


def _strip_zones(params: str) -> str:
    parts = [p for p in (params or "").split(":") if p and not p.startswith("zones=")]
    return ":".join(parts)


def _build_zones_param(segments: list[dict[str, Any]], bs: list[float]) -> str:
    """x265 zones: start,end,b=N — end exclusive; zones abut on boundaries."""
    chunks: list[str] = []
    for seg, b in zip(segments, bs):
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        if end_f <= start_f:
            continue
        chunks.append(f"{start_f},{end_f},b={float(b):.4g}")
    if not chunks:
        raise SystemExit("empty zones string")
    return "zones=" + "/".join(chunks)


def _trim_frame_range(
    src: Path,
    dst: Path,
    *,
    start_frame: int,
    end_frame: int,
    ffmpeg_bin: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
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
    bs: list[float],
    *,
    target_mbps: float,
    args: argparse.Namespace,
    also_base: bool,
) -> list[dict[str, Any]]:
    ffmpeg_bin = resolve_binary("ffmpeg", None)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="zones_vbr_seg_vmaf_") as tmp:
        tmp_dir = Path(tmp)
        for seg, b in zip(segments, bs):
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
                "b": float(b),
                "approx_mbps": float(b) * float(target_mbps),
                "vmaf_neg": float(vmaf_neg),
                "vmaf_base": None if vmaf_base is None else float(vmaf_base),
                "vmaf_delta": None if vmaf_delta is None else float(vmaf_delta),
                "score_sec": elapsed,
            }
            rows.append(row)
            _print_zone_features(row, live=True)
    return rows


def _print_zone_features(row: dict[str, Any], *, live: bool = False) -> None:
    prefix = "  " if live else ""
    base_txt = ""
    if row.get("vmaf_base") is not None and row.get("vmaf_delta") is not None:
        base_txt = f"  vmaf_base={float(row['vmaf_base']):.2f}  vmaf_delta={float(row['vmaf_delta']):.2f}"
    vmaf_txt = (
        f"vmaf_neg={float(row['vmaf_neg']):.2f}{base_txt}"
        if row.get("vmaf_neg") is not None
        else "vmaf_neg=pending"
    )
    b = float(row.get("b", 1.0) or 1.0)
    approx = row.get("approx_mbps")
    approx_txt = f"  ~{float(approx):.3f}Mbps" if approx is not None else ""
    print(
        f"{prefix}zone[{row['index']}]  frames={row['start_frame']}-{row['end_frame']}  "
        f"({row['start_sec']:.2f}s-{row['end_sec']:.2f}s)  "
        f"b={b:.3f}{approx_txt}  {vmaf_txt}",
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


def _print_zone_table(segment_rows: list[dict[str, Any]]) -> None:
    if not segment_rows:
        return
    print("=" * 88)
    print("ZONE RESULTS — b= / approx Mbps / VMAF / FEATURES")
    print("=" * 88)
    print(
        f"{'zone':>4}  {'frames':>13}  {'b':>5}  {'~Mbps':>7}  {'vmaf_neg':>8}  "
        f"{'diff':>6}  {'motion':>7}  {'tex':>6}"
    )
    print("-" * 88)
    for r in segment_rows:
        print(
            f"{r['index']:4d}  {r['start_frame']:6d}-{r['end_frame']:<6d}  "
            f"{float(r['b']):5.2f}  {float(r['approx_mbps']):7.3f}  {r['vmaf_neg']:8.2f}  "
            f"{r['difficulty']:6.3f}  {r['motion']:7.4f}  {r['texture']:6.3f}"
        )
    print("-" * 88)
    print("per-zone details:")
    for r in segment_rows:
        _print_zone_features(r, live=False)
    print("=" * 88)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    bitrate, target_mbps = _resolve_bitrate(
        input_path,
        bitrate=args.bitrate,
        target_compression_rate=args.target_compression_rate,
    )
    feat = _resolve_features(args, input_path)
    segments = _segments_from_features(feat)
    bs = _resolve_zone_bs(segments, args, target_mbps=target_mbps)
    zones = _build_zones_param(segments, bs)
    base_params = _strip_zones(args.params)
    params = f"{base_params}:{zones}" if base_params else zones

    out = Path(args.output) if args.output else Path("work") / "zones_vbr_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json) if args.result_json else Path(str(out) + ".zones.json")

    print(f"input      : {input_path}")
    print(f"output     : {out}")
    print(f"bitrate    : {bitrate}  ({target_mbps:.3f} Mbps average)")
    print(f"segments   : {len(segments)}")
    print(f"seg b=     : {bs}")
    print(f"approx Mbps: {[round(b * target_mbps, 3) for b in bs]}")
    print(f"zones      : {zones}")
    print(f"params     : {params}")
    print(f"preset     : {args.preset}  profile={args.profile}  twopass={bool(args.twopass)}")
    print(f"vmaf thr   : {args.vmaf_threshold}  gpu={args.gpu}")
    print("-" * 72)

    print("ZONE VBR PLAN (before encode)")
    print("-" * 72)
    for seg, b in zip(segments, bs):
        _print_zone_features(
            {
                **seg,
                "b": float(b),
                "approx_mbps": float(b) * target_mbps,
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
        codec_mode="ABR",
        crf=None,
        bitrate=bitrate,
        encoder="libx265",
        libx265_profile=args.profile,
        twopass=bool(args.twopass),
        progress_reference_path=str(input_path),
        progress_label=f"zones/VBR{bitrate}",
    )
    encode_sec = time.monotonic() - t0
    if not enc.ok:
        raise SystemExit(f"encode failed: {enc.error or enc.stderr[-1000:]}")

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
        codec_mode="ABR",
        target_bitrate_mbps=target_mbps,
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
    if score.validation_errors:
        print(f"validation_errors : {score.validation_errors}")

    segment_rows: list[dict[str, Any]] = []
    if not args.skip_segment_vmaf:
        print("-" * 72)
        print("per-segment VMAF (trim → libvmaf)…", flush=True)
        segment_rows = _segment_vmaf(
            input_path,
            out,
            segments,
            bs,
            target_mbps=target_mbps,
            args=args,
            also_base=bool(args.segment_base_vmaf),
        )
        if segment_rows:
            _print_zone_table(segment_rows)
            neg_vals = [r["vmaf_neg"] for r in segment_rows]
            print(f"zone vmaf_neg min   : {min(neg_vals):.2f}")
            print(f"zone vmaf_neg max   : {max(neg_vals):.2f}")
            print(f"zone vmaf_neg mean  : {sum(neg_vals) / len(neg_vals):.2f}")
            weights = [max(1, r["end_frame"] - r["start_frame"]) for r in segment_rows]
            wsum = sum(weights)
            wmean = sum(v * w for v, w in zip(neg_vals, weights)) / wsum
            print(f"zone vmaf_neg wmean : {wmean:.2f}  (whole-file {score.vmaf:.2f})")
            print("per-zone b= + VMAF :")
            for r in segment_rows:
                print(
                    f"  zone[{r['index']}]  b={float(r['b']):.3f}  "
                    f"~{float(r['approx_mbps']):.3f}Mbps  "
                    f"vmaf_neg={r['vmaf_neg']:.2f}  "
                    f"difficulty={r['difficulty']:.3f}"
                )

    payload = {
        "mode": "zones_vbr",
        "input": str(input_path),
        "output": str(out),
        "bitrate": bitrate,
        "target_mbps": target_mbps,
        "segment_bs": bs,
        "approx_mbps": [round(b * target_mbps, 4) for b in bs],
        "zones": zones,
        "params": params,
        "preset": args.preset,
        "profile": args.profile,
        "twopass": bool(args.twopass),
        "vmaf_threshold": int(args.vmaf_threshold),
        "encode_sec": encode_sec,
        "whole_score_sec": score_sec,
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
            "validation_errors": list(score.validation_errors or []),
        },
        "segments": segment_rows,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"result_json       : {result_path}")
    return 0 if score.s_f > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
