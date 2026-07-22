#!/usr/bin/env python3
"""Per-zone HEVC encode + whole-file / per-zone VMAF (NEG + optional base).

Default ``--zone-apply segments``: encode each zone with **full** zonefile params
(``--qp``/``--crf``, ``--aq-strength``, ``--rd``, ``--ref``, …) via ffmpeg
``libx265``, then concat. This is required for true per-zone AQ/analysis on
Ubuntu x265/libx265 3.5 (native ``--zonefile`` ignores those overrides).

Each segment encode uses **scene-as-one-GOP**: ``keyint=min-keyint=frame_count``,
``scenecut=0`` (overrides ``--x265-args`` keyint/scenecut).

**Prefer ``--crf`` per zone** when using AQ: under ``--qp`` (CQP), x265 ignores
``aq-strength`` / ``aq-mode`` (identical bitstream). CRF enables AQ.

Faster alternative ``--zone-apply zones-q``: native ``x265 --zones …,q=`` only
(per-zone QP; AQ/rd/ref become one global set; ``--crf`` in file → force ``q=``).

Examples:
  # CRF + per-zone AQ (recommended; segments + GPU are defaults)
  python3 test_zones_zonefile_score.py \\
    --input ../video/1.mp4 \\
    --zonefile work/zones_zonefile.txt \\
    --base-crf 34 --segment-base-vmaf

  # Parallel encode + VMAF
  python3 test_zones_zonefile_score.py \\
    --input ../video/2.mp4 --zonefile work/zones_zonefile.txt \\
    --zone-workers 3 --vmaf-workers 6 --segment-base-vmaf

  # Fast QP-only path
  python3 test_zones_zonefile_score.py \\
    --input ../video/1.mp4 --zonefile work/zones_zonefile.txt \\
    --zone-apply zones-q
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from extract_video_features import extract_features
from ffmpeg_tools import resolve_binary
from scoring import NEG_MODEL, BASE_MODEL, compute_vmaf, score_candidate
from segment_crf_aq_grid_sweep import apply_scene_as_gop


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

# Zonefile keys that can be lifted to global CLI when present (zones=q= cannot vary these).
_GLOBAL_LIFT_KEYS = (
    "aq-mode",
    "aq-strength",
    "rd",
    "ref",
    "rdoq-level",
    "me",
    "subme",
    "merange",
    "max-merge",
    "rskip",
    "rskip-edge-threshold",
    "limit-tu",
    "rdpenalty",
    "dynamic-rd",
    "nr-intra",
    "nr-inter",
    "radl",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default="work/zones_zonefile_score.mp4")
    p.add_argument(
        "--zonefile",
        default="",
        help="Existing zonefile path (if omitted, generate from --segment-crfs)",
    )
    p.add_argument(
        "--zonefile-out",
        default="work/zones_zonefile.txt",
        help="Where to write a generated zonefile",
    )
    p.add_argument("--features", default="")
    p.add_argument("--use-cached-features", action="store_true")
    p.add_argument(
        "--save-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--features-dir", default="video_features")
    p.add_argument("--base-crf", type=float, default=28.0, help="Global x265 --crf fallback (outside zones)")
    p.add_argument(
        "--segment-crfs",
        default="",
        help="Comma values for generated zonefile --crf (mapped to zones=q= at encode)",
    )
    p.add_argument(
        "--segment-qps",
        default="",
        help="Comma force-QPs for generated zones (preferred; writes --qp in zonefile)",
    )
    p.add_argument("--segment-aq", default="", help="Optional comma aq-strength per zone (globalized if constant)")
    p.add_argument(
        "--segment-presets",
        default="",
        help="Optional comma x265 presets per zone for generated zonefile (segments mode)",
    )
    p.add_argument(
        "--zone-extra",
        default="",
        help="Extra flags appended to every generated zone line",
    )
    p.add_argument("--preset", "-p", default="fast", choices=_X265_PRESETS,
                   help="Default x265 preset (overridden by --preset on zonefile lines in segments mode)")
    p.add_argument(
        "--x265-args",
        default="--bframes 8 --rc-lookahead 40",
        help=(
            "Extra global x265 CLI args. In segments mode, keyint/min-keyint/scenecut "
            "are overridden per zone (scene-as-one-GOP)."
        ),
    )
    p.add_argument(
        "--zone-apply",
        choices=["segments", "zones-q", "zonefile"],
        default="segments",
        help=(
            "segments= per-zone ffmpeg libx265 encode+concat (FULL params: aq/rd/ref/…). "
            "zones-q= native x265 --zones q= only (fast; AQ/rd global). "
            "zonefile= broken on x265 3.5 (debug)."
        ),
    )
    p.add_argument(
        "--zone-workers",
        type=int,
        default=1,
        help="Parallel zone encodes for --zone-apply segments (1=sequential)",
    )
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPU for whole-file VMAF when available (default: on)",
    )
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--vmaf-n-threads", type=int, default=40)
    p.add_argument(
        "--vmaf-workers",
        type=int,
        default=1,
        help="Parallel per-zone VMAF scores (uses CPU when > 1)",
    )
    p.add_argument(
        "--segment-vmaf-cpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force CPU libvmaf for per-zone VMAF (default: auto when --vmaf-workers > 1)",
    )
    p.add_argument("--skip-segment-vmaf", action="store_true")
    p.add_argument(
        "--segment-base-vmaf",
        action="store_true",
        help="Also compute base-model VMAF + |base-neg| per zone",
    )
    p.add_argument("--result-json", default="")
    p.add_argument(
        "--keep-hevc",
        action="store_true",
        help="Keep intermediate .hevc bitstream next to output",
    )
    return p.parse_args()


def _resolve_features(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    features_dir = Path(args.features_dir)
    if args.features or args.use_cached_features:
        path = Path(args.features) if args.features else features_dir / f"{input_path.stem}.json"
        if not path.is_file():
            raise SystemExit(f"features not found: {path}")
        print(f"features   : cached {path}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
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
        raise SystemExit("no usable segments")
    if frame_count > 0 and out[-1]["end_frame"] < frame_count:
        out[-1]["end_frame"] = frame_count
        out[-1]["frame_count"] = out[-1]["end_frame"] - out[-1]["start_frame"]
    return out


def _sanitize_zonefile(src: Path, dst: Path) -> Path:
    """Rewrite zonefile without blank/comment lines.

    x265 3.5 ``parseZoneFile`` treats bare ``\\n`` as a zone row, then segfaults
    on ``strchr(line, ' ')`` when the line has no spaces. Comments (#) are OK,
    but blank lines between them are not.
    """
    kept: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        kept.append(s)
    if not kept:
        raise SystemExit(f"zonefile has no usable zone lines: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dst


def _parse_zonefile(path: Path) -> list[dict[str, Any]]:
    """Parse zonefile lines → [{start_frame, opts: {crf, aq-strength, ...}, raw}]."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = shlex.split(s)
        if not toks:
            continue
        try:
            start = int(toks[0])
        except ValueError as exc:
            raise SystemExit(f"bad zonefile line (need startFrame …): {line!r}") from exc
        opts: dict[str, Any] = {}
        i = 1
        while i < len(toks):
            tok = toks[i]
            if not tok.startswith("--"):
                i += 1
                continue
            key = tok[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                val: Any = toks[i + 1]
                # numeric when possible
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                opts[key] = val
                i += 2
            else:
                opts[key] = True
                i += 1
        rows.append({"start_frame": start, "opts": opts, "raw": s})
    if not rows:
        raise SystemExit(f"no zones parsed from {path}")
    return rows


def _normalize_preset(name: Any, *, context: str = "") -> str:
    p = str(name).lower().strip()
    if p not in _X265_PRESETS:
        where = f" ({context})" if context else ""
        raise SystemExit(f"invalid x265 preset {name!r}{where}; choose from {_X265_PRESETS}")
    return p


def _resolve_zone_preset(z: dict[str, Any], default: str) -> str:
    opts = z.get("zone_opts") or {}
    p = opts.get("preset")
    if p is None:
        p = z.get("zone_preset")
    if p is None:
        return _normalize_preset(default, context=f"zone[{z.get('index')}] default")
    return _normalize_preset(p, context=f"zone[{z.get('index')}]")


def _attach_zone_opts(
    segments: list[dict[str, Any]],
    zone_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map feature segments to zonefile rows.

    If counts match → **1:1 by index** (zonefile line i → segment i).
    This is what you want for ``--zone-apply segments`` and avoids wrong
    params when zonefile startFrames were written for a different video.

    If counts differ → fall back to startFrame coverage (x265 zonefile semantics)
    and print a warning.
    """
    out: list[dict[str, Any]] = []

    def _row_from_zone(seg: dict[str, Any], z: dict[str, Any]) -> dict[str, Any]:
        opts = dict(z["opts"])
        qp = opts.get("qp")
        crf = opts.get("crf")
        bitrate = opts.get("bitrate")
        if qp is not None and crf is not None:
            print(
                f"WARNING    : zone start={z['start_frame']} has both --qp and --crf; "
                f"using --qp (CQP; AQ ignored).",
                flush=True,
            )
        if qp is not None:
            rc_mode = "qp"
            applied_q = int(round(float(qp)))
            applied_crf = None
        elif crf is not None:
            rc_mode = "crf"
            applied_crf = float(crf)
            applied_q = int(round(applied_crf))
        elif bitrate is not None:
            rc_mode = "bitrate"
            applied_q = None
            applied_crf = None
        else:
            rc_mode = "fallback"
            applied_q = None
            applied_crf = None
        return {
            **seg,
            "zone_start_frame": int(z["start_frame"]),
            "zone_opts": opts,
            "zone_raw": z["raw"],
            "crf": crf,
            "qp": qp,
            "rc_mode": rc_mode,
            "applied_q": applied_q,
            "applied_crf": applied_crf,
            "aq_strength": opts.get("aq-strength"),
            "aq_mode": opts.get("aq-mode"),
            "rd": opts.get("rd"),
            "ref": opts.get("ref"),
            "zone_preset": opts.get("preset"),
        }

    if len(zone_rows) == len(segments):
        # Prefer index pairing; rewrite zone_start_frame to this video's segment start
        # so plans/logs match the frames actually encoded.
        for seg, z in zip(segments, zone_rows):
            row = _row_from_zone(seg, z)
            row["zone_start_frame"] = int(seg["start_frame"])
            # Keep original raw for debugging, but note file start may differ
            file_start = int(z["start_frame"])
            if file_start != int(seg["start_frame"]):
                row["zone_raw"] = (
                    f"{z['raw']}  # file_start={file_start} → seg_start={seg['start_frame']}"
                )
            out.append(row)
        return out

    print(
        f"WARNING    : zonefile has {len(zone_rows)} lines but video has "
        f"{len(segments)} segments — mapping by startFrame coverage "
        f"(reorder/regenerate zonefile for 1:1).",
        flush=True,
    )
    starts = sorted(z["start_frame"] for z in zone_rows)
    by_start = {z["start_frame"]: z for z in zone_rows}
    for seg in segments:
        sf = int(seg["start_frame"])
        active = starts[0]
        for st in starts:
            if st <= sf:
                active = st
            else:
                break
        out.append(_row_from_zone(seg, by_start[active]))
    return out


def _build_zonefile(
    segments: list[dict[str, Any]],
    values: list[float],
    aqs: list[Optional[float]],
    extra: str,
    *,
    use_qp: bool,
    presets: Optional[list[Optional[str]]] = None,
) -> str:
    flag = "--qp" if use_qp else "--crf"
    lines = [f"# x265 zonefile (reference). Encode uses --zones q= (see --zone-apply)."]
    preset_list = presets if presets is not None else [None] * len(segments)
    for seg, val, aq, pr in zip(segments, values, aqs, preset_list):
        parts = [str(int(seg["start_frame"])), flag, f"{float(val):g}"]
        if pr is not None and str(pr).strip():
            parts.extend(["--preset", _normalize_preset(pr, context="generated zonefile")])
        if aq is not None:
            parts.extend(["--aq-strength", f"{float(aq):.4g}"])
        if extra.strip():
            parts.extend(shlex.split(extra))
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _zone_qp_list(zones: list[dict[str, Any]], *, base_crf: float) -> list[int]:
    """Numeric RC targets for display / zones-q (``--qp`` preferred, else ``--crf``)."""
    qps: list[int] = []
    for z in zones:
        if z.get("qp") is not None:
            qps.append(int(round(float(z["qp"]))))
        elif z.get("crf") is not None:
            qps.append(int(round(float(z["crf"]))))
        elif z.get("applied_crf") is not None:
            qps.append(int(round(float(z["applied_crf"]))))
        else:
            qps.append(int(round(float(base_crf))))
    return qps


def _zone_rc_label(z: dict[str, Any]) -> str:
    """Human-readable per-zone rate-control label (crf= / q= / …)."""
    mode = z.get("rc_mode")
    if mode == "crf" or (z.get("crf") is not None and z.get("qp") is None):
        v = z.get("applied_crf", z.get("crf"))
        return f"crf={float(v):g}" if v is not None else "crf=?"
    if mode == "qp" or z.get("qp") is not None:
        v = z.get("applied_q", z.get("qp"))
        return f"q={int(v)}" if v is not None else "q=?"
    if mode == "bitrate" or (z.get("zone_opts") or {}).get("bitrate") is not None:
        br = (z.get("zone_opts") or {}).get("bitrate")
        return f"bitrate={br}k" if br is not None else "bitrate=?"
    if z.get("applied_q") is not None:
        return f"crf={z['applied_q']}(fallback)"
    return "rc=?"


def _build_zones_q_param(zones: list[dict[str, Any]], qps: list[int]) -> str:
    """Official zones=start,end,q=N — end exclusive, abutting ranges."""
    chunks: list[str] = []
    for z, qp in zip(zones, qps):
        start_f = int(z["start_frame"])
        end_f = int(z["end_frame"])
        if end_f <= start_f:
            continue
        chunks.append(f"{start_f},{end_f},q={int(qp)}")
    if not chunks:
        raise SystemExit("empty zones= string")
    return "/".join(chunks)


def _lift_global_zone_opts(zones: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return (global_cli_args, warnings) from zone opts that are constant across zones."""
    warnings: list[str] = []
    global_args: list[str] = []
    if not zones:
        return global_args, warnings

    for key in _GLOBAL_LIFT_KEYS:
        vals = []
        for z in zones:
            opts = z.get("zone_opts") or {}
            if key in opts:
                vals.append(opts[key])
        if not vals:
            continue
        # Prefer majority value when zones disagree (zones= cannot vary these).
        counts: dict[str, int] = {}
        for v in vals:
            counts[str(v)] = counts.get(str(v), 0) + 1
        best_s = max(counts.items(), key=lambda kv: kv[1])[0]
        chosen = next(v for v in vals if str(v) == best_s)
        if len(counts) > 1:
            warnings.append(
                f"per-zone {key} differs {vals}; zones=q= cannot vary it — "
                f"using majority ({chosen})"
            )
        global_args.extend([f"--{key}", f"{chosen}"])
    return global_args, warnings


def _probe_fps(path: Path) -> float:
    ffprobe = resolve_binary("ffprobe", None)
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    rate = (proc.stdout or "30/1").strip() or "30/1"
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / max(1e-9, float(den))
    return float(rate)


def _remux_hevc_to_mp4(hevc: Path, output: Path, *, fps: float) -> None:
    mp4box = shutil.which("MP4Box")
    if mp4box:
        remux = subprocess.run(
            [mp4box, "-quiet", "-add", f"{hevc}:fps={fps:g}", "-new", str(output)],
            capture_output=True,
            text=True,
        )
        if remux.returncode != 0 or not output.is_file():
            raise SystemExit(
                f"MP4Box remux failed: {(remux.stderr or remux.stdout or '')[-800:]}"
            )
        return
    ffmpeg = resolve_binary("ffmpeg", None)
    remux = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts", "-framerate", f"{fps:g}",
            "-i", str(hevc), "-c", "copy", "-tag:v", "hvc1", str(output),
        ],
        capture_output=True,
        text=True,
    )
    if remux.returncode != 0 or not output.is_file():
        raise SystemExit(f"remux failed: {(remux.stderr or '')[-800:]}")
    print(
        "WARNING: MP4Box not found; ffmpeg remux may break VMAF timestamps. "
        "Install gpac (MP4Box) for reliable scoring.",
        flush=True,
    )


def _x265_cli_args_to_params(x265_args: str) -> list[str]:
    """Convert ``--keyint 60 --bframes 6`` → ``keyint=60:bframes=6`` pieces."""
    toks = shlex.split(x265_args or "")
    parts: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            parts.append(f"{key}={toks[i + 1]}")
            i += 2
        else:
            parts.append(f"{key}=1")
            i += 1
    return parts


def _zone_opts_to_x265_params(opts: dict[str, Any]) -> tuple[list[str], Optional[float]]:
    """Build -x265-params pieces + optional ffmpeg -crf from one zone's opts.

    Returns (param_parts, crf_or_none). If qp set → CQP via qp=; elif crf → -crf.
    """
    parts: list[str] = []
    crf: Optional[float] = None
    skip = {"crf", "qp", "bitrate", "preset"}  # handled specially
    if "qp" in opts and opts["qp"] is not None:
        parts.append(f"qp={int(round(float(opts['qp'])))}")
    elif "crf" in opts and opts["crf"] is not None:
        crf = float(opts["crf"])
    elif "bitrate" in opts and opts["bitrate"] is not None:
        # ABR kbps — ffmpeg uses -b:v; keep as hint in params too
        parts.append(f"bitrate={int(round(float(opts['bitrate'])))}")

    for key, val in opts.items():
        if key in skip:
            continue
        if val is True:
            parts.append(f"{key}=1")
        elif val is False:
            parts.append(f"{key}=0")
        else:
            parts.append(f"{key}={val}")
    return parts, crf


def _encode_one_zone_segment(
    z: dict[str, Any],
    *,
    input_path: Path,
    clip: Path,
    ffmpeg: str,
    base_params: list[str],
    base_crf: float,
    preset: str,
) -> dict[str, Any]:
    """Encode one zone clip; return metadata for merging into ``zones``."""
    sf = int(z["start_frame"])
    ef = int(z["end_frame"])
    last = max(sf, ef - 1)
    opts = dict(z.get("zone_opts") or {})
    if "qp" not in opts and z.get("qp") is not None:
        opts["qp"] = z["qp"]
    if "crf" not in opts and z.get("crf") is not None:
        opts["crf"] = z["crf"]
    if "aq-strength" not in opts and z.get("aq_strength") is not None:
        opts["aq-strength"] = z["aq_strength"]

    zone_preset = _resolve_zone_preset(z, preset)
    zparts, zcrf = _zone_opts_to_x265_params(opts)
    merged: dict[str, str] = {}
    for p in base_params + zparts:
        if "=" in p:
            k, v = p.split("=", 1)
            merged[k] = v
        else:
            merged[p] = "1"
    # One GOP for the whole zone clip (I at start; no mid-zone scenecuts).
    frame_count = max(1, int(ef) - int(sf))
    merged = apply_scene_as_gop(merged, frame_count=frame_count)
    param_str = ":".join(f"{k}={v}" for k, v in merged.items())

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"select=between(n\\,{sf}\\,{last}),setpts=PTS-STARTPTS",
        "-an",
        "-c:v", "libx265",
        "-preset", zone_preset,
        "-pix_fmt", "yuv420p",
        "-tag:v", "hvc1",
        "-x265-params", param_str,
    ]

    rc_mode = z.get("rc_mode")
    applied_q = z.get("applied_q")
    applied_crf = z.get("applied_crf")
    if "qp" in merged:
        rc_mode = "qp"
        applied_q = int(merged["qp"])
        applied_crf = None
    elif "bitrate" in merged:
        cmd.extend(["-b:v", f"{merged['bitrate']}k"])
        rc_mode = "bitrate"
    else:
        crf_use = zcrf if zcrf is not None else float(base_crf)
        cmd.extend(["-crf", f"{crf_use:g}"])
        rc_mode = "crf" if zcrf is not None else "fallback"
        applied_crf = float(crf_use)
        applied_q = int(round(float(crf_use)))
    cmd.append(str(clip))

    rc = _zone_rc_label({
        **z,
        "rc_mode": rc_mode,
        "applied_q": applied_q,
        "applied_crf": applied_crf,
        "qp": opts.get("qp"),
        "crf": opts.get("crf"),
    })
    print(
        f"encode     : zone[{z['index']}] frames={sf}-{ef}  {rc}  preset={zone_preset}  "
        f"params={param_str[:100]}{'…' if len(param_str)>100 else ''}",
        flush=True,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not clip.is_file() or clip.stat().st_size <= 0:
        raise RuntimeError(
            f"segment encode failed zone[{z['index']}]:\n{(proc.stderr or '')[-1500:]}"
        )

    warning = None
    if rc_mode == "qp" and opts.get("aq-strength") is not None:
        warning = (
            f"zone[{z['index']}] uses --qp (CQP): "
            f"aq-strength={opts.get('aq-strength')} is ignored by x265. "
            f"Use --crf for AQ."
        )

    return {
        "index": int(z["index"]),
        "clip": clip,
        "stderr": proc.stderr or "",
        "rc_mode": rc_mode,
        "applied_q": applied_q,
        "applied_crf": applied_crf,
        "applied_aq": opts.get("aq-strength"),
        "applied_aq_mode": opts.get("aq-mode"),
        "applied_preset": zone_preset,
        "warning": warning,
    }


def _encode_segments_libx265(
    input_path: Path,
    output: Path,
    zones: list[dict[str, Any]],
    *,
    base_crf: float,
    preset: str,
    x265_args: str,
    zone_workers: int = 1,
    profile: str = "main",
) -> tuple[float, str, str]:
    """Encode each zone with its full zonefile params (ffmpeg libx265), then concat.

    This is the only reliable way on x265/libx265 3.5 to vary aq-strength/rd/ref
    per zone (native --zonefile ignores those overrides).
    """
    ffmpeg = resolve_binary("ffmpeg", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_params = _x265_cli_args_to_params(x265_args)
    # Per-zone scene-as-one-GOP is applied in _encode_one_zone_segment.
    print(
        "GOP        : scene-as-one per zone "
        "(keyint=min-keyint=frame_count, scenecut=0)",
        flush=True,
    )

    workers = max(1, min(int(zone_workers), len(zones)))
    if workers > 1:
        print(f"parallel   : {workers} zone encode workers", flush=True)

    t0 = time.monotonic()
    logs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="zone_seg_enc_") as tmp:
        tmp_dir = Path(tmp)

        def _run_zone(z: dict[str, Any]) -> dict[str, Any]:
            clip = tmp_dir / f"z{z['index']:02d}.mp4"
            return _encode_one_zone_segment(
                z,
                input_path=input_path,
                clip=clip,
                ffmpeg=ffmpeg,
                base_params=base_params,
                base_crf=base_crf,
                preset=preset,
            )

        results: list[dict[str, Any]] = []
        if workers == 1:
            for z in zones:
                results.append(_run_zone(z))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_run_zone, z) for z in zones]
                for fut in as_completed(futs):
                    try:
                        results.append(fut.result())
                    except RuntimeError as exc:
                        raise SystemExit(str(exc)) from exc

        results.sort(key=lambda r: int(r["index"]))
        by_index = {int(r["index"]): r for r in results}
        clip_paths: list[Path] = []
        for z in zones:
            r = by_index[int(z["index"])]
            z["rc_mode"] = r["rc_mode"]
            z["applied_q"] = r["applied_q"]
            z["applied_crf"] = r["applied_crf"]
            z["applied_aq"] = r["applied_aq"]
            z["applied_aq_mode"] = r["applied_aq_mode"]
            z["applied_preset"] = r.get("applied_preset")
            logs.append(r["stderr"])
            if r.get("warning"):
                print(f"WARNING    : {r['warning']}", flush=True)
            clip_paths.append(r["clip"])

        # Concat demuxer (stream copy)
        list_file = tmp_dir / "concat.txt"
        list_file.write_text(
            "".join(f"file '{c.resolve()}'\n" for c in clip_paths),
            encoding="utf-8",
        )
        concat_cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy",
            str(output),
        ]
        print(f"concat     : {len(clip_paths)} zone clips → {output}", flush=True)
        cproc = subprocess.run(concat_cmd, capture_output=True, text=True)
        logs.append(cproc.stderr or "")
        if cproc.returncode != 0 or not output.is_file():
            raise SystemExit(f"concat failed: {(cproc.stderr or '')[-800:]}")

    return time.monotonic() - t0, "\n".join(logs)[-3000:], "segments"


def _encode_native_x265(
    input_path: Path,
    output: Path,
    *,
    base_crf: float,
    preset: str,
    x265_args: str,
    keep_hevc: bool,
    zone_apply: str,
    zonefile: Path,
    zones_q_param: str,
    global_from_zones: list[str],
) -> tuple[float, str, str]:
    """Encode with working --zones q= (default) or legacy --zonefile.

    Returns (encode_sec, log_tail, apply_mode_used).
    """
    x265 = shutil.which("x265")
    if not x265:
        raise SystemExit("native x265 not found (apt install x265)")
    ffmpeg = resolve_binary("ffmpeg", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    hevc = output.with_suffix(".hevc")
    if hevc.exists():
        hevc.unlink()
    fps = _probe_fps(input_path)

    cmd_ff = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path), "-an", "-f", "yuv4mpegpipe", "-strict", "-1", "-",
    ]
    cmd_x = [x265]
    if x265_args.strip():
        cmd_x.extend(shlex.split(x265_args))
    cmd_x.extend(global_from_zones)
    cmd_x.extend(["--y4m", "--preset", preset, "--crf", f"{float(base_crf):g}"])

    apply_used = zone_apply
    if zone_apply == "zones-q":
        cmd_x.extend(["--zones", zones_q_param])
    else:
        cmd_x.extend(["--zonefile", str(zonefile)])
        print(
            "WARNING: --zone-apply zonefile: --crf/--qp in zonefile are ignored on "
            "x265 3.5; prefer default zones-q",
            flush=True,
        )

    cmd_x.extend(["-o", str(hevc), "-"])

    print(f"encode     : ffmpeg | {' '.join(cmd_x)}", flush=True)
    t0 = time.monotonic()
    ff = subprocess.Popen(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ff.stdout is not None
    xx = subprocess.run(cmd_x, stdin=ff.stdout, capture_output=True, text=True)
    ff.stdout.close()
    ff_err = (ff.stderr.read() if ff.stderr else b"")
    if isinstance(ff_err, bytes):
        ff_err = ff_err.decode("utf-8", errors="replace")
    ff.wait()
    log = (xx.stderr or "") + (xx.stdout or "") + ff_err
    if xx.returncode != 0 or not hevc.is_file() or hevc.stat().st_size <= 0:
        raise SystemExit(f"x265 encode failed:\n{log[-2500:]}")

    _remux_hevc_to_mp4(hevc, output, fps=fps)
    if not keep_hevc:
        try:
            hevc.unlink()
        except OSError:
            pass
    return time.monotonic() - t0, log[-3000:], apply_used


def _zone_packet_bytes(path: Path, segments: list[dict[str, Any]]) -> list[int]:
    """Sum video packet bytes per segment via pts→frame mapping."""
    ffprobe_bin = resolve_binary("ffprobe", None)
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,size",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe packets failed: {(proc.stderr or '')[-500:]}")

    fps_proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
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

    out: list[int] = []
    for seg in segments:
        sf = int(seg["start_frame"])
        ef = int(seg["end_frame"])
        total = sum(sz for fi, sz in pkt_sizes if sf <= fi < ef)
        out.append(int(total))
    return out


def _attach_zone_compression(
    rows: list[dict[str, Any]],
    *,
    reference: Path,
    distorted: Path,
) -> list[dict[str, Any]]:
    """Add per-zone size_in/out + compression_rate/ratio (packet bytes)."""
    ref_bytes = _zone_packet_bytes(reference, rows)
    out_bytes = _zone_packet_bytes(distorted, rows)
    for row, rb, ob in zip(rows, ref_bytes, out_bytes):
        rate = (ob / rb) if rb > 0 else 0.0
        ratio = (rb / ob) if ob > 0 else 0.0
        row["size_in_bytes"] = int(rb)
        row["size_out_bytes"] = int(ob)
        row["compression_rate"] = float(rate)
        row["compression_ratio"] = float(ratio)
    return rows


def _trim_frame_range(
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


def _segment_vmaf_threads(total_threads: int, workers: int) -> int:
    """Share libvmaf threads across parallel zone scores (OOM guard)."""
    workers = max(1, int(workers))
    raw = max(2, int(total_threads) // workers)
    return max(2, min(8, raw))


def _segment_vmaf_compute_kwargs(
    args: argparse.Namespace,
    *,
    workers: int,
) -> dict[str, Any]:
    use_cpu = (
        bool(args.segment_vmaf_cpu)
        if args.segment_vmaf_cpu is not None
        else int(workers) > 1
    )
    n_threads = _segment_vmaf_threads(args.vmaf_n_threads, workers)
    return {
        "n_subsample": args.vmaf_n_subsample,
        "n_threads": n_threads,
        "vmaf_backend": "docker",
        "vmaf_docker_image": "vmaf_ffmpeg",
        "vmaf_docker_gpus": False if use_cpu else bool(args.gpu),
        "vmaf_gpu_device": args.gpu_device if args.gpu and not use_cpu else None,
        "vmaf_gpu_prefer": bool(args.gpu) and not use_cpu,
        "_use_cpu": use_cpu,
        "_n_threads": n_threads,
    }


def _score_one_zone_vmaf(
    z: dict[str, Any],
    *,
    reference: Path,
    distorted: Path,
    tmp_dir: Path,
    ffmpeg_bin: str,
    vmaf_kwargs: dict[str, Any],
    also_base: bool,
) -> dict[str, Any]:
    sf = int(z["start_frame"])
    ef = int(z["end_frame"])
    ref_clip = tmp_dir / f"ref_{z['index']}.mp4"
    dist_clip = tmp_dir / f"dist_{z['index']}.mp4"
    t0 = time.monotonic()
    _trim_frame_range(reference, ref_clip, start_frame=sf, end_frame=ef, ffmpeg_bin=ffmpeg_bin)
    _trim_frame_range(distorted, dist_clip, start_frame=sf, end_frame=ef, ffmpeg_bin=ffmpeg_bin)
    kw = {k: v for k, v in vmaf_kwargs.items() if not str(k).startswith("_")}
    vmaf_neg = float(
        compute_vmaf(
            str(ref_clip),
            str(dist_clip),
            model=NEG_MODEL,
            **kw,
        )
    )
    vmaf_base: Optional[float] = None
    vmaf_delta: Optional[float] = None
    if also_base:
        vmaf_base = float(
            compute_vmaf(
                str(ref_clip),
                str(dist_clip),
                model=BASE_MODEL,
                **kw,
            )
        )
        vmaf_delta = abs(vmaf_base - vmaf_neg)
    elapsed = time.monotonic() - t0
    row = {
        **{k: z[k] for k in (
            "index", "start_frame", "end_frame", "start_sec", "end_sec",
            "frame_count", "difficulty", "motion", "motion_p90", "texture",
            "edge", "noise", "entropy", "flatness", "luma_mean", "hf_energy",
            "zone_start_frame", "zone_opts", "zone_raw", "crf", "qp",
            "rc_mode", "applied_q", "applied_crf", "applied_aq", "zonefile_aq",
            "aq_strength", "aq_mode", "rd", "ref",
        ) if k in z},
        "vmaf_neg": vmaf_neg,
        "vmaf_base": vmaf_base,
        "vmaf_delta": vmaf_delta,
        "score_sec": elapsed,
    }
    return row


def _print_segment_vmaf_row(z: dict[str, Any], row: dict[str, Any]) -> None:
    sf = int(z["start_frame"])
    ef = int(z["end_frame"])
    q_txt = _zone_rc_label(z)
    aq_show = z.get("applied_aq", z.get("aq_strength"))
    aq_txt = f" aq={aq_show}" if aq_show is not None else ""
    if (
        z.get("applied_aq") is not None
        and z.get("zonefile_aq") is not None
        and float(z["zonefile_aq"]) != float(z["applied_aq"])
    ):
        aq_txt += f"(file={z['zonefile_aq']})"
    base_txt = ""
    vmaf_base = row.get("vmaf_base")
    vmaf_delta = row.get("vmaf_delta")
    if vmaf_base is not None and vmaf_delta is not None:
        base_txt = f"  vmaf_base={vmaf_base:.2f}  delta={vmaf_delta:.2f}"
    print(
        f"  zone[{z['index']}]  frames={sf}-{ef}  {q_txt}{aq_txt}  "
        f"vmaf_neg={row['vmaf_neg']:.2f}{base_txt}",
        flush=True,
    )
    print(
        f"    features: difficulty={z['difficulty']:.4f}  motion={z['motion']:.4f}  "
        f"texture={z['texture']:.4f}  edge={z['edge']:.4f}",
        flush=True,
    )


def _segment_vmaf(
    reference: Path,
    distorted: Path,
    zones: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ffmpeg_bin = resolve_binary("ffmpeg", None)
    also_base = bool(args.segment_base_vmaf)
    workers = max(1, min(int(args.vmaf_workers), len(zones)))
    vmaf_kwargs = _segment_vmaf_compute_kwargs(args, workers=workers)
    use_cpu = bool(vmaf_kwargs.pop("_use_cpu"))
    per_threads = int(vmaf_kwargs.pop("_n_threads"))

    if workers > 1:
        print(
            f"parallel   : {workers} zone VMAF workers  "
            f"(CPU={use_cpu}  threads/zone={per_threads})",
            flush=True,
        )
    elif use_cpu:
        print(f"segment_vmaf: CPU libvmaf  threads/zone={per_threads}", flush=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="zonefile_seg_vmaf_") as tmp:
        tmp_dir = Path(tmp)

        def _run_one(z: dict[str, Any]) -> dict[str, Any]:
            return _score_one_zone_vmaf(
                z,
                reference=reference,
                distorted=distorted,
                tmp_dir=tmp_dir,
                ffmpeg_bin=ffmpeg_bin,
                vmaf_kwargs=dict(vmaf_kwargs),
                also_base=also_base,
            )

        if workers == 1:
            for z in zones:
                row = _run_one(z)
                rows.append(row)
                _print_segment_vmaf_row(z, row)
        else:
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_run_one, z): z for z in zones}
                for fut in as_completed(futs):
                    z = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        raise SystemExit(
                            f"segment VMAF failed zone[{z['index']}]: {exc}"
                        ) from exc
                    results.append(row)
            results.sort(key=lambda r: int(r["index"]))
            by_index = {int(r["index"]): r for r in results}
            for z in zones:
                row = by_index[int(z["index"])]
                rows.append(row)
                _print_segment_vmaf_row(z, row)
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    has_base = any(r.get("vmaf_base") is not None for r in rows)
    print("=" * 110)
    print("ZONE RESULTS — applied CRF/QP + AQ + VMAF + compression")
    print("=" * 110)
    hdr = (
        f"{'zone':>4}  {'frames':>13}  {'rc':>8}  {'aq':>5}  "
        f"{'vmaf_neg':>8}  {'rate':>8}  {'ratio':>7}  "
        f"{'inMiB':>6}  {'outMiB':>6}"
    )
    if has_base:
        hdr = (
            f"{'zone':>4}  {'frames':>13}  {'rc':>8}  {'aq':>5}  "
            f"{'vmaf_neg':>8}  {'base':>7}  {'rate':>8}  {'ratio':>7}  "
            f"{'outMiB':>6}"
        )
    print(hdr)
    print("-" * 110)
    for r in rows:
        q = _zone_rc_label(r)
        aq = r.get("applied_aq", r.get("aq_strength"))
        aq_s = f"{float(aq):.2f}" if aq is not None else "-"
        fr = f"{r['start_frame']}-{r['end_frame']}"
        rate = float(r.get("compression_rate") or 0.0)
        ratio = float(r.get("compression_ratio") or 0.0)
        in_m = float(r.get("size_in_bytes") or 0) / (1024 * 1024)
        out_m = float(r.get("size_out_bytes") or 0) / (1024 * 1024)
        if has_base:
            print(
                f"{r['index']:>4}  {fr:>13}  {q:>8}  {aq_s:>5}  "
                f"{float(r.get('vmaf_neg') or float('nan')):8.2f}  "
                f"{float(r.get('vmaf_base') or float('nan')):7.2f}  "
                f"{rate:8.4f}  {ratio:6.2f}x  {out_m:6.2f}"
            )
        else:
            print(
                f"{r['index']:>4}  {fr:>13}  {q:>8}  {aq_s:>5}  "
                f"{float(r.get('vmaf_neg') or float('nan')):8.2f}  "
                f"{rate:8.4f}  {ratio:6.2f}x  "
                f"{in_m:6.2f}  {out_m:6.2f}"
            )
    print("=" * 110)
    print(
        "segments mode: crf/qp + aq (+ rd/ref/…) are per-zone.  "
        "zones-q mode: only q= per-zone.  AQ requires --crf (ignored under --qp)."
    )
    print("rate=out/in  ratio=in/out")
    print("=" * 110)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")
    if int(args.zone_workers) < 1:
        raise SystemExit("--zone-workers must be >= 1")
    if int(args.vmaf_workers) < 1:
        raise SystemExit("--vmaf-workers must be >= 1")
    if args.zone_apply != "segments" and not shutil.which("x265"):
        raise SystemExit("x265 CLI not found — install with: apt install x265")

    feat = _resolve_features(args, input_path)
    segments = _segments_from_features(feat)

    if args.zonefile:
        zonefile_src = Path(args.zonefile)
        if not zonefile_src.is_file():
            raise SystemExit(f"zonefile not found: {zonefile_src}")
    else:
        use_qp = bool(args.segment_qps.strip())
        raw = (args.segment_qps or args.segment_crfs or "").strip()
        if not raw:
            raise SystemExit("provide --zonefile or --segment-qps / --segment-crfs")
        vals = [float(x) for x in raw.split(",") if x.strip()]
        if len(vals) != len(segments):
            raise SystemExit(
                f"segment values length {len(vals)} != segments {len(segments)}"
            )
        if args.segment_aq.strip():
            aqs_f = [float(x) for x in args.segment_aq.split(",") if x.strip()]
            if len(aqs_f) != len(segments):
                raise SystemExit(
                    f"--segment-aq length {len(aqs_f)} != segments {len(segments)}"
                )
            aqs: list[Optional[float]] = list(aqs_f)
        else:
            aqs = [None] * len(segments)
        if args.segment_presets.strip():
            prs = [x.strip() for x in args.segment_presets.split(",") if x.strip()]
            if len(prs) != len(segments):
                raise SystemExit(
                    f"--segment-presets length {len(prs)} != segments {len(segments)}"
                )
            presets: list[Optional[str]] = [_normalize_preset(p) for p in prs]
        else:
            presets = [None] * len(segments)
        text = _build_zonefile(
            segments, vals, aqs, args.zone_extra, use_qp=use_qp, presets=presets
        )
        zonefile_src = Path(args.zonefile_out)
        zonefile_src.parent.mkdir(parents=True, exist_ok=True)
        zonefile_src.write_text(text, encoding="utf-8")

    # Sanitize for humans / optional --zone-apply zonefile (blanks segfault x265 3.5).
    zonefile = Path(args.zonefile_out).with_suffix(".clean.txt")
    if zonefile.resolve() == zonefile_src.resolve():
        zonefile = zonefile_src.parent / (zonefile_src.stem + ".clean.txt")
    _sanitize_zonefile(zonefile_src, zonefile)
    print(f"zonefile   : {zonefile_src}  (clean → {zonefile})", flush=True)

    zone_rows = _parse_zonefile(zonefile)
    zones = _attach_zone_opts(segments, zone_rows)
    qps = _zone_qp_list(zones, base_crf=args.base_crf)
    for z, qp in zip(zones, qps):
        if z.get("applied_q") is None:
            z["applied_q"] = int(qp)
        if z.get("rc_mode") in (None, "fallback") and z.get("qp") is None and z.get("crf") is None:
            z["rc_mode"] = "fallback"
            z["applied_crf"] = float(qp)
    zones_q = _build_zones_q_param(zones, qps)
    global_from_zones, lift_warnings = _lift_global_zone_opts(zones)

    # If zone-extra was only on generated file, globals may already be in zone opts.
    # Also allow --zone-extra globals when generating via CLI without per-line parse of extras into opts:
    if args.zone_extra.strip() and not global_from_zones:
        # parse --aq-mode etc from zone-extra into global args
        extra_toks = shlex.split(args.zone_extra)
        global_from_zones.extend(extra_toks)

    # What the encode actually uses for AQ
    applied_aq: Optional[float] = None
    applied_aq_mode: Optional[Any] = None
    if args.zone_apply != "segments":
        for i, tok in enumerate(global_from_zones):
            if tok == "--aq-strength" and i + 1 < len(global_from_zones):
                try:
                    applied_aq = float(global_from_zones[i + 1])
                except ValueError:
                    pass
            if tok == "--aq-mode" and i + 1 < len(global_from_zones):
                applied_aq_mode = global_from_zones[i + 1]
        for z in zones:
            z["applied_aq"] = applied_aq
            z["applied_aq_mode"] = applied_aq_mode
            z["zonefile_aq"] = z.get("aq_strength")
    else:
        for z in zones:
            z["zonefile_aq"] = z.get("aq_strength")
            z["applied_aq"] = z.get("aq_strength")  # truly per-zone in segments mode
            z["applied_aq_mode"] = (z.get("zone_opts") or {}).get("aq-mode", z.get("aq_mode"))

    for z in zones:
        z["applied_preset"] = _resolve_zone_preset(z, args.preset)
    preset_note = args.preset
    zone_presets = sorted({z["applied_preset"] for z in zones})
    if len(zone_presets) > 1:
        preset_note = f"{args.preset} default; per-zone {zone_presets}"
    elif zone_presets and zone_presets[0] != args.preset:
        preset_note = zone_presets[0]

    out = Path(args.output)
    print(f"input      : {input_path}")
    print(f"output     : {out}")
    print(f"base_crf   : {args.base_crf}  (fallback; per-zone uses --crf/--qp from file)")
    print(f"zone-apply : {args.zone_apply}")
    rc_modes = [z.get("rc_mode") or "?" for z in zones]
    n_crf = sum(1 for m in rc_modes if m == "crf")
    n_qp = sum(1 for m in rc_modes if m == "qp")
    if n_crf and not n_qp:
        print(f"seg CRFs   : {[z.get('applied_crf', z.get('crf', q)) for z, q in zip(zones, qps)]}")
    elif n_qp and not n_crf:
        print(f"seg QPs    : {qps}")
    else:
        print(f"seg targets: {[_zone_rc_label(z) for z in zones]}")
    if args.zone_apply == "zones-q":
        print(f"zones=     : {zones_q}")
        if global_from_zones:
            print(f"global+    : {' '.join(str(x) for x in global_from_zones)}")
        for w in lift_warnings:
            print(f"WARNING    : {w}", flush=True)
    elif args.zone_apply == "segments":
        print(
            "NOTE       : segments mode = each zone encoded with its FULL zonefile "
            "params (aq/rd/ref/…) via ffmpeg libx265, then concat",
            flush=True,
        )
        if int(args.zone_workers) > 1:
            w = max(1, min(int(args.zone_workers), len(zones)))
            print(f"NOTE       : --zone-workers {w} (parallel encode, sequential concat)", flush=True)
    if n_crf and args.zone_apply == "zones-q":
        print(
            "NOTE       : zonefile --crf values are applied as force-QP q= "
            "(not CRF). Same numbers ≠ CRF quality. Use --zone-apply segments for real CRF+AQ.",
            flush=True,
        )
    elif n_crf and args.zone_apply == "segments":
        print(
            "NOTE       : CRF mode — real ffmpeg -crf per zone; AQ/rd/ref apply.",
            flush=True,
        )
    if n_qp and args.zone_apply == "segments":
        print(
            "NOTE       : QP/CQP mode — aq-strength is ignored by x265; use --crf for AQ.",
            flush=True,
        )
    print(f"segments   : {len(segments)}")
    print(f"preset     : {preset_note}")
    print(f"x265-args  : {args.x265_args}")
    print(f"vmaf thr   : {args.vmaf_threshold}  gpu={args.gpu}")
    print("-" * 72)
    print("ZONE PLAN (applied)")
    print("-" * 72)
    for z in zones:
        if args.zone_apply == "segments":
            aq_note = (
                f"  aq={z.get('applied_aq')}" if z.get("applied_aq") is not None else ""
            )
            pr_note = ""
            if z.get("applied_preset") and z.get("applied_preset") != args.preset:
                pr_note = f"  preset={z['applied_preset']}"
            print(
                f"  zone[{z['index']}] frames={z['start_frame']}-{z['end_frame']}  "
                f"{_zone_rc_label(z)}{aq_note}{pr_note}  "
                f"raw={z['zone_raw']}"
            )
        else:
            aq_note = ""
            if z.get("applied_aq") is not None:
                zf_aq = z.get("zonefile_aq")
                aq_note = f"  aq_applied={z['applied_aq']}"
                if zf_aq is not None and float(zf_aq) != float(z["applied_aq"]):
                    aq_note += f" (zonefile had {zf_aq}, not per-zone)"
            print(
                f"  zone[{z['index']}] frames={z['start_frame']}-{z['end_frame']}  "
                f"q={z['applied_q']}{aq_note}  raw={z['zone_raw']}"
            )
    print("-" * 72)

    if args.zone_apply == "segments":
        encode_sec, enc_log, apply_used = _encode_segments_libx265(
            input_path,
            out,
            zones,
            base_crf=args.base_crf,
            preset=args.preset,
            x265_args=args.x265_args,
            zone_workers=max(1, int(args.zone_workers)),
        )
    else:
        encode_sec, enc_log, apply_used = _encode_native_x265(
            input_path,
            out,
            base_crf=args.base_crf,
            preset=args.preset,
            x265_args=args.x265_args,
            keep_hevc=bool(args.keep_hevc),
            zone_apply=args.zone_apply,
            zonefile=zonefile,
            zones_q_param=zones_q,
            global_from_zones=global_from_zones,
        )
    if re.search(r"invalid|error|failed", enc_log, re.I):
        print(f"x265 log   : …{enc_log[-800:]}", flush=True)
    print(f"encode_sec : {encode_sec:.1f}  apply={apply_used}", flush=True)

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
        segment_rows = _segment_vmaf(input_path, out, zones, args=args)
    else:
        # Still report per-zone size even when skipping VMAF.
        segment_rows = [{**z} for z in zones]

    print("-" * 72)
    print("per-zone compression (packet bytes)…", flush=True)
    _attach_zone_compression(segment_rows, reference=input_path, distorted=out)
    for r in segment_rows:
        print(
            f"  zone[{r['index']}]  rate={r['compression_rate']:.4f}  "
            f"ratio={r['compression_ratio']:.2f}x  "
            f"in={r['size_in_bytes']/(1024*1024):.2f}MiB  "
            f"out={r['size_out_bytes']/(1024*1024):.2f}MiB",
            flush=True,
        )

    if segment_rows and not args.skip_segment_vmaf:
        _print_table(segment_rows)
        neg_vals = [r["vmaf_neg"] for r in segment_rows if r.get("vmaf_neg") is not None]
        if neg_vals:
            print(f"zone vmaf_neg min   : {min(neg_vals):.2f}")
            print(f"zone vmaf_neg max   : {max(neg_vals):.2f}")
            print(f"zone vmaf_neg mean  : {sum(neg_vals) / len(neg_vals):.2f}")
            weights = [max(1, r["end_frame"] - r["start_frame"]) for r in segment_rows]
            wmean = sum(v * w for v, w in zip(neg_vals, weights)) / sum(weights)
            print(f"zone vmaf_neg wmean : {wmean:.2f}  (whole-file {score.vmaf:.2f})")
        rates = [float(r["compression_rate"]) for r in segment_rows]
        print(f"zone rate min/max   : {min(rates):.4f} / {max(rates):.4f}")
        print(
            f"zone rate wmean     : "
            f"{sum(r['compression_rate'] * max(1, r['end_frame']-r['start_frame']) for r in segment_rows) / sum(max(1, r['end_frame']-r['start_frame']) for r in segment_rows):.4f}"
            f"  (whole-file {score.compression_rate:.4f})"
        )

    result_path = (
        Path(args.result_json) if args.result_json else Path(str(out) + ".zonefile.json")
    )
    payload = {
        "mode": "zones_q_native_x265",
        "zone_apply": apply_used,
        "input": str(input_path),
        "output": str(out),
        "zonefile": str(zonefile),
        "zones_q": zones_q,
        "segment_qps": qps,
        "global_from_zones": global_from_zones,
        "base_crf": float(args.base_crf),
        "zone_workers": max(1, int(args.zone_workers)),
        "vmaf_workers": max(1, int(args.vmaf_workers)),
        "segment_vmaf_cpu": (
            bool(args.segment_vmaf_cpu)
            if args.segment_vmaf_cpu is not None
            else int(args.vmaf_workers) > 1
        ),
        "preset": args.preset,
        "x265_args": args.x265_args,
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
        },
        "segments": segment_rows,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"result_json       : {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
