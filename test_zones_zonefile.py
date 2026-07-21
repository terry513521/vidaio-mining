#!/usr/bin/env python3
"""Generate / try an x265 zonefile (per-zone --crf / --aq-strength / --rd / --ref).

zonefile is richer than zones=q=/b=, but it is CLI-oriented. Through ffmpeg
libx265, ``zonefile=`` is usually rejected. This script:

  1. Builds a zonefile from video_features segments
  2. Tries encode via ffmpeg -x265-params zonefile=... (reports if unsupported)
  3. If native ``x265`` is on PATH, encodes with --zonefile (full support)

Examples:
  # Write zonefile only
  python3 test_zones_zonefile.py --input ../video/1.mp4 --print-only \\
    --zone-opts "crf=22,aq-strength=0.8" "crf=28,aq-strength=1.0" ...

  # Try encode (ffmpeg first; native x265 if available)
  python3 test_zones_zonefile.py --input ../video/1.mp4 --base-crf 28 \\
    --segment-qps 22,28,26,24,30,28 --gpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="work/zones_zonefile_test.mp4")
    p.add_argument("--zonefile-out", default="work/zones_zonefile.txt")
    p.add_argument("--features", default="")
    p.add_argument("--use-cached-features", action="store_true")
    p.add_argument("--base-crf", type=int, default=28)
    p.add_argument(
        "--segment-qps",
        default="",
        help="Comma QPs/CRFs for --crf in each zonefile line (same count as segments)",
    )
    p.add_argument(
        "--segment-aq",
        default="",
        help="Optional comma aq-strength per zone",
    )
    p.add_argument(
        "--zone-extra",
        default="",
        help="Extra flags appended to every zone line (e.g. '--rd 6 --ref 5')",
    )
    p.add_argument("--params", default="aq-mode=1:keyint=60:min-keyint=1:scenecut=40")
    p.add_argument("--preset", default="fast")
    p.add_argument("--print-only", action="store_true", help="Only write/print zonefile")
    p.add_argument("--prefer-native-x265", action="store_true")
    return p.parse_args()


def _load_or_extract(input_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.features or args.use_cached_features:
        path = Path(args.features) if args.features else Path("video_features") / f"{input_path.stem}.json"
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"features   : re-extracting {input_path} …", flush=True)
    data = extract_features(input_path)
    out = Path("video_features") / f"{input_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"features   : saved {out}", flush=True)
    return data


def _segments(feat: dict[str, Any]) -> list[dict[str, Any]]:
    meta = feat.get("meta") or {}
    fps = float(meta.get("fps") or 30.0)
    frame_count = int(meta.get("frame_count") or 0)
    out = []
    for i, seg in enumerate(feat.get("segments") or []):
        start_f = int(round(float(seg["start_sec"]) * fps))
        end_f = int(round(float(seg["end_sec"]) * fps))
        if frame_count:
            end_f = min(end_f, frame_count)
        out.append({"index": i, "start_frame": start_f, "end_frame": max(start_f + 1, end_f), **seg})
    if frame_count and out:
        out[-1]["end_frame"] = frame_count
    return out


def _build_zonefile(
    segments: list[dict[str, Any]],
    qps: list[int],
    aqs: list[Optional[float]],
    extra: str,
) -> str:
    lines = ["# x265 zonefile: <start_frame> --crf N [--aq-strength X] ..."]
    for seg, qp, aq in zip(segments, qps, aqs):
        parts = [str(int(seg["start_frame"])), "--crf", str(int(qp))]
        if aq is not None:
            parts.extend(["--aq-strength", f"{float(aq):.4g}"])
        if extra.strip():
            parts.append(extra.strip())
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _try_ffmpeg_zonefile(
    input_path: Path,
    output: Path,
    zonefile: Path,
    *,
    base_crf: int,
    preset: str,
    params: str,
) -> tuple[bool, str]:
    ffmpeg = resolve_binary("ffmpeg", None)
    # zonefile is typically CLI-only; expect failure via -x265-params
    x265 = f"{params}:zonefile={zonefile}" if params else f"zonefile={zonefile}"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "info",
        "-i", str(input_path),
        "-an", "-c:v", "libx265", "-preset", preset, "-pix_fmt", "yuv420p",
        "-crf", str(base_crf),
        "-x265-params", x265,
        str(output),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = (p.stderr or "") + (p.stdout or "")
    ok = p.returncode == 0 and output.is_file() and output.stat().st_size > 0
    unsupported = any(
        s in text.lower()
        for s in ("unknown option", "invalid", "zonefile", "no such")
    ) and ("zonefile" in text.lower() or "unknown option" in text.lower())
    return ok and not unsupported, text[-3000:]


def _try_native_x265(
    input_path: Path,
    output: Path,
    zonefile: Path,
    *,
    base_crf: int,
    preset: str,
) -> tuple[bool, str]:
    x265 = shutil.which("x265")
    if not x265:
        return False, "native x265 not found on PATH (apt install x265 / build from source)"
    # Need yuv input typically; use ffmpeg pipe
    ffmpeg = resolve_binary("ffmpeg", None)
    # Probe size
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(input_path)],
        capture_output=True, text=True,
    )
    # Simpler: ffmpeg decode to y4m pipe into x265
    out_hevc = output.with_suffix(".hevc")
    cmd_ff = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path), "-f", "yuv4mpegpipe", "-strict", "-1", "-",
    ]
    cmd_x = [
        x265, "--y4m", "--preset", preset, "--crf", str(base_crf),
        "--zonefile", str(zonefile), "-o", str(out_hevc), "-",
    ]
    ff = subprocess.Popen(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    xx = subprocess.run(cmd_x, stdin=ff.stdout, capture_output=True, text=True)
    if ff.stdout:
        ff.stdout.close()
    ff.wait()
    text = (xx.stderr or "") + (xx.stdout or "")
    if xx.returncode != 0 or not out_hevc.is_file():
        return False, text[-3000:] or "x265 encode failed"
    # Remux to mp4
    remux = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(out_hevc), "-c", "copy", str(output)],
        capture_output=True, text=True,
    )
    ok = remux.returncode == 0 and output.is_file()
    return ok, text[-2000:]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"missing input: {input_path}")

    feat = _load_or_extract(input_path, args)
    segments = _segments(feat)
    n = len(segments)
    if not n:
        raise SystemExit("no segments")

    if args.segment_qps.strip():
        qps = [int(x) for x in args.segment_qps.split(",") if x.strip()]
    else:
        qps = [int(args.base_crf)] * n
    if len(qps) != n:
        raise SystemExit(f"--segment-qps length {len(qps)} != segments {n}")

    if args.segment_aq.strip():
        aqs_f = [float(x) for x in args.segment_aq.split(",") if x.strip()]
        if len(aqs_f) != n:
            raise SystemExit(f"--segment-aq length {len(aqs_f)} != segments {n}")
        aqs: list[Optional[float]] = list(aqs_f)
    else:
        aqs = [None] * n

    text = _build_zonefile(segments, qps, aqs, args.zone_extra)
    zonefile = Path(args.zonefile_out)
    zonefile.parent.mkdir(parents=True, exist_ok=True)
    zonefile.write_text(text, encoding="utf-8")
    print(f"zonefile   : {zonefile}")
    print("---- zonefile contents ----")
    print(text.rstrip())
    print("---------------------------")

    if args.print_only:
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.prefer_native_x265:
        ok, log = _try_native_x265(
            input_path, out, zonefile, base_crf=args.base_crf, preset=args.preset
        )
        print("backend    : native x265")
        print(log[-1500:])
        print(f"ok={ok} output={out}")
        return 0 if ok else 1

    print("backend    : trying ffmpeg libx265 zonefile= …", flush=True)
    ok, log = _try_ffmpeg_zonefile(
        input_path, out, zonefile,
        base_crf=args.base_crf, preset=args.preset, params=args.params,
    )
    if ok:
        print("ffmpeg zonefile path worked")
        print(f"output     : {out}")
        return 0

    print("ffmpeg zonefile unsupported/failed (expected — zonefile is CLI-only).")
    print(log[-1200:])
    print("falling back to native x265 if available…", flush=True)
    ok2, log2 = _try_native_x265(
        input_path, out, zonefile, base_crf=args.base_crf, preset=args.preset
    )
    print(log2[-1500:])
    if ok2:
        print(f"native x265 ok → {out}")
        return 0
    print(
        "Install native x265 CLI to use zonefile, or stick to zones=q=/b= via "
        "test_zones_crf.py / test_zones_vbr.py."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
