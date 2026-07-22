#!/usr/bin/env python3
"""Grid-search CRF × aq-strength on pre-segmented clips (PySceneDetect splits).

Reads segment mp4s from ``segmented videos/<stem>/segXX_fSTART-END.mp4``, encodes
each clip with libx265 across a CRF × aq-strength grid, scores VMAF (NEG + base)
against the segment clip, and writes every trial.

Default grid:
  CRF          [22, 38] step 1     → 17 values
  aq-strength  [0.3, 2.3] step 0.1 → 21 values
  Total        357 trials / segment

Default workers:
  18 total → 3 videos in parallel × 6 encode+score workers each

Compression uses **source packet bytes** from the matching raw video (not the
lossless segment file size), matching competition scoring.

Example:
  python3 segment_crf_aq_grid_sweep.py \\
    --segmented-dir "../segmented videos" \\
    --raw-dir "../raw videos" \\
    --workers 18 --video-workers 3 --preset fast --resume

  python3 segment_crf_aq_grid_sweep.py --limit 1 --gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from compress_util import measure_compression
from encoder import encode_hevc
from ffmpeg_tools import resolve_binary
from interp_search import format_x265_params, parse_x265_params
from scoring import ScoreResult, score_candidate
from test_crf_aq_sweep import (
    DEFAULT_BASE_PARAMS,
    _build_float_grid,
    _build_int_grid,
    _completed_keys,
    _gpu_score_lock,
)

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

SEG_RE = re.compile(r"^seg(?P<idx>\d+)_f(?P<start>\d+)-(?P<end>\d+)\.mp4$", re.I)

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

CSV_FIELDS = [
    "video_stem",
    "segment_index",
    "start_frame",
    "end_frame",
    "crf",
    "aq_strength",
    "vmaf_base",
    "vmaf_neg",
    "vmaf_delta",
    "compression_rate",
    "compression_ratio",
    "s_f",
    "gates_ok",
    "passed_encoding_gates",
    "passed_vmaf_delta_gate",
    "reason",
    "encode_ok",
    "encode_sec",
    "score_sec",
    "size_in_bytes",
    "size_out_bytes",
    "params",
    "segment_path",
    "output_path",
    "error",
]

_write_lock = threading.Lock()


@dataclass
class TrialRow:
    video_stem: str
    segment_index: int
    start_frame: int
    end_frame: int
    crf: int
    aq_strength: float
    vmaf_base: Optional[float]
    vmaf_neg: float
    vmaf_delta: Optional[float]
    compression_rate: float
    compression_ratio: float
    s_f: float
    gates_ok: bool
    passed_encoding_gates: bool
    passed_vmaf_delta_gate: bool
    reason: str
    encode_ok: bool
    encode_sec: float
    score_sec: float
    size_in_bytes: int
    size_out_bytes: int
    params: str
    segment_path: str
    output_path: str
    error: str


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    with _write_lock:
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k) for k in CSV_FIELDS})


def _discover_video_dirs(segmented_dir: Path) -> list[Path]:
    dirs = sorted(
        p for p in segmented_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    out: list[Path] = []
    for d in dirs:
        if any(SEG_RE.match(p.name) for p in d.glob("seg*.mp4")):
            out.append(d)
    return out


def _parse_segments(video_dir: Path) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    for path in sorted(video_dir.glob("seg*.mp4")):
        m = SEG_RE.match(path.name)
        if not m:
            continue
        start_f = int(m.group("start"))
        end_f = int(m.group("end"))
        if end_f <= start_f:
            continue
        segs.append(
            {
                "index": int(m.group("idx")),
                "start_frame": start_f,
                "end_frame": end_f,
                "frame_count": end_f - start_f,
                "path": path,
            }
        )
    segs.sort(key=lambda s: (int(s["index"]), int(s["start_frame"])))
    return segs


def apply_scene_as_gop(params: dict[str, str], *, frame_count: int) -> dict[str, str]:
    """Force one GOP for the whole segment: keyint=min-keyint=N, scenecut=0."""
    n = max(1, int(frame_count))
    out = dict(params)
    out["keyint"] = str(n)
    out["min-keyint"] = str(n)
    out["scenecut"] = "0"
    return out


def _find_raw_video(stem: str, raw_dir: Path) -> Optional[Path]:
    for ext in (".mp4", ".mov", ".mkv", ".webm"):
        p = raw_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    matches = sorted(raw_dir.glob(f"{stem}.*"))
    for p in matches:
        if p.is_file():
            return p
    return None


def _source_segment_packet_bytes(
    source: Path,
    segments: list[dict[str, Any]],
    *,
    ffprobe_bin: Optional[str] = None,
) -> dict[int, int]:
    """Sum source video packet bytes per segment (competition size-in)."""
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


def _run_trial(
    *,
    video_stem: str,
    seg: dict[str, Any],
    crf: int,
    aq: float,
    out_path: Path,
    base_params: dict[str, str],
    preset: str,
    profile: str,
    vmaf_threshold: int,
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    keep_encode: bool,
    source_segment_bytes: int,
) -> TrialRow:
    seg_path: Path = seg["path"]
    params = apply_scene_as_gop(
        dict(base_params),
        frame_count=int(seg.get("frame_count") or (int(seg["end_frame"]) - int(seg["start_frame"]))),
    )
    params["aq-strength"] = f"{round(float(aq), 3):g}"
    params_str = format_x265_params(params)
    src_bytes = max(0, int(source_segment_bytes))

    t0 = time.monotonic()
    enc = encode_hevc(
        str(seg_path),
        str(out_path),
        preset=preset,
        params=params_str,
        codec_mode="RC",
        crf=int(crf),
        encoder="libx265",
        libx265_profile=profile,
        progress_reference_path=str(seg_path),
        progress_reference_bytes=src_bytes if src_bytes > 0 else None,
        progress_label=f"{video_stem[:8]} seg{seg['index']} CRF{crf}/aq{aq:.1f}",
    )
    encode_sec = time.monotonic() - t0

    if not enc.ok:
        return TrialRow(
            video_stem=video_stem,
            segment_index=int(seg["index"]),
            start_frame=int(seg["start_frame"]),
            end_frame=int(seg["end_frame"]),
            crf=int(crf),
            aq_strength=round(float(aq), 3),
            vmaf_base=None,
            vmaf_neg=0.0,
            vmaf_delta=None,
            compression_rate=1.0,
            compression_ratio=1.0,
            s_f=0.0,
            gates_ok=False,
            passed_encoding_gates=False,
            passed_vmaf_delta_gate=False,
            reason="encode_failed",
            encode_ok=False,
            encode_sec=encode_sec,
            score_sec=0.0,
            size_in_bytes=src_bytes,
            size_out_bytes=0,
            params=params_str,
            segment_path=str(seg_path),
            output_path=str(out_path),
            error=(enc.stderr_tail or "")[-400:],
        )

    size_out = out_path.stat().st_size if out_path.is_file() else 0
    if src_bytes > 0:
        rate, _ratio = measure_compression(
            str(seg_path),
            str(out_path),
            reference_bytes=src_bytes,
        )
        rate_override: Optional[float] = float(rate)
    else:
        rate_override = None

    t1 = time.monotonic()

    def _score() -> ScoreResult:
        return score_candidate(
            str(seg_path),
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
    return TrialRow(
        video_stem=video_stem,
        segment_index=int(seg["index"]),
        start_frame=int(seg["start_frame"]),
        end_frame=int(seg["end_frame"]),
        crf=int(crf),
        aq_strength=round(float(aq), 3),
        vmaf_base=None if score.vmaf_base is None else float(score.vmaf_base),
        vmaf_neg=float(score.vmaf),
        vmaf_delta=None if score.vmaf_delta is None else float(score.vmaf_delta),
        compression_rate=float(score.compression_rate),
        compression_ratio=float(score.compression_ratio),
        s_f=float(score.s_f),
        gates_ok=gates_ok,
        passed_encoding_gates=bool(score.passed_encoding_gates),
        passed_vmaf_delta_gate=bool(score.passed_vmaf_delta_gate),
        reason=str(score.reason),
        encode_ok=True,
        encode_sec=encode_sec,
        score_sec=score_sec,
        size_in_bytes=src_bytes,
        size_out_bytes=int(size_out),
        params=params_str,
        segment_path=str(seg_path),
        output_path=str(out_path),
        error="",
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _best_row(rows: list[dict[str, Any]], *, gated: bool) -> Optional[dict[str, Any]]:
    cand = [r for r in rows if r.get("encode_ok")]
    if gated:
        cand = [r for r in cand if r.get("gates_ok")]
    if not cand:
        return None
    return max(cand, key=lambda r: float(r.get("s_f") or 0.0))


def _process_one_video(
    video_dir: Path,
    *,
    raw_dir: Path,
    work_root: Path,
    grid: list[tuple[int, float]],
    base_params: dict[str, str],
    preset: str,
    profile: str,
    vmaf_threshold: int,
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    keep_encode: bool,
    resume: bool,
    force: bool,
    workers_per_video: int,
) -> dict[str, Any]:
    stem = video_dir.name
    segs = _parse_segments(video_dir)
    if not segs:
        return {"video_stem": stem, "ok": False, "error": "no seg*.mp4 found"}

    raw = _find_raw_video(stem, raw_dir)
    if raw is None:
        return {
            "video_stem": stem,
            "ok": False,
            "error": f"raw video not found under {raw_dir}",
        }

    video_work = work_root / stem
    if force and video_work.exists():
        import shutil

        shutil.rmtree(video_work)
    video_work.mkdir(parents=True, exist_ok=True)

    print(f"[{stem}] probing source packet bytes from {raw.name} …", flush=True)
    src_bytes = _source_segment_packet_bytes(raw, segs)
    for seg in segs:
        idx = int(seg["index"])
        print(
            f"[{stem}] seg[{idx}] frames={seg['start_frame']}-{seg['end_frame']} "
            f"source_pkt={src_bytes.get(idx, 0) / 1e6:.2f}MB  clip={seg['path'].name}",
            flush=True,
        )

    jobs: list[tuple[dict[str, Any], int, float, Path, Path]] = []
    for seg in segs:
        seg_dir = video_work / f"segment_{int(seg['index']):02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        encodes_dir = seg_dir / "encodes"
        encodes_dir.mkdir(parents=True, exist_ok=True)
        trials_path = seg_dir / "trials.jsonl"
        csv_path = seg_dir / "results.csv"
        if force:
            for p in (trials_path, csv_path, seg_dir / "summary.json"):
                if p.exists():
                    p.unlink()
        done = _completed_keys(trials_path) if resume else set()
        for crf, aq in grid:
            key = (int(crf), round(float(aq), 4))
            if key in done:
                continue
            out_path = encodes_dir / f"crf{crf}_aq{aq:.1f}.mp4"
            jobs.append((seg, int(crf), float(aq), out_path, trials_path))

    n_skip = len(grid) * len(segs) - len(jobs)
    print(
        f"[{stem}] segments={len(segs)} grid={len(grid)} "
        f"todo={len(jobs)} skipped={n_skip} workers={workers_per_video}",
        flush=True,
    )

    t0 = time.monotonic()
    completed = 0

    def _job(item: tuple[dict[str, Any], int, float, Path, Path]) -> TrialRow:
        seg, crf, aq, out_path, trials_path = item
        row = _run_trial(
            video_stem=stem,
            seg=seg,
            crf=crf,
            aq=aq,
            out_path=out_path,
            base_params=base_params,
            preset=preset,
            profile=profile,
            vmaf_threshold=vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=vmaf_n_subsample,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            keep_encode=keep_encode,
            source_segment_bytes=int(src_bytes.get(int(seg["index"]), 0)),
        )
        payload = asdict(row)
        _append_jsonl(trials_path, payload)
        _append_csv_row(trials_path.parent / "results.csv", payload)
        return row

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers_per_video)) as pool:
            futs = {pool.submit(_job, j): j for j in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                completed += 1
                elapsed = time.monotonic() - t0
                rate = completed / max(elapsed, 1e-6)
                eta = (len(jobs) - completed) / max(rate, 1e-9)
                print(
                    f"[{stem}] [{completed}/{len(jobs)}] "
                    f"seg={row.segment_index} crf={row.crf} aq={row.aq_strength:.1f}  "
                    f"vmaf_neg={row.vmaf_neg:.2f} vmaf_base="
                    f"{'—' if row.vmaf_base is None else f'{row.vmaf_base:.2f}'}  "
                    f"ratio={row.compression_ratio:.2f}x rate={row.compression_rate:.4f}  "
                    f"s_f={row.s_f:.4f} gates={row.gates_ok}  "
                    f"ETA={eta / 60:.1f}m  {row.reason}",
                    flush=True,
                )

    # Per-segment + video summaries
    segment_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for seg in segs:
        seg_dir = video_work / f"segment_{int(seg['index']):02d}"
        rows = _load_rows(seg_dir / "trials.jsonl")
        all_rows.extend(rows)
        best = _best_row(rows, gated=False)
        best_g = _best_row(rows, gated=True)
        summary = {
            "video_stem": stem,
            "segment_index": int(seg["index"]),
            "start_frame": int(seg["start_frame"]),
            "end_frame": int(seg["end_frame"]),
            "segment_path": str(seg["path"]),
            "source_packet_bytes": int(src_bytes.get(int(seg["index"]), 0)),
            "n_trials": len(rows),
            "n_ok": sum(1 for r in rows if r.get("encode_ok")),
            "best_s_f": best,
            "best_gated_s_f": best_g,
        }
        (seg_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        segment_summaries.append(summary)
        if best is not None:
            print(
                f"[{stem}] seg[{seg['index']}] best: "
                f"crf={best['crf']} aq={best['aq_strength']} "
                f"s_f={best['s_f']:.4f} vmaf_neg={best['vmaf_neg']:.2f} "
                f"ratio={best['compression_ratio']:.2f}x",
                flush=True,
            )

    video_summary = {
        "video_stem": stem,
        "raw_video": str(raw),
        "segmented_dir": str(video_dir),
        "work_dir": str(video_work),
        "n_segments": len(segs),
        "grid_points_per_segment": len(grid),
        "n_trials_logged": len(all_rows),
        "wall_sec": time.monotonic() - t0,
        "segments": segment_summaries,
    }
    (video_work / "summary.json").write_text(
        json.dumps(video_summary, indent=2), encoding="utf-8"
    )
    with (video_work / "all_trials.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "video_stem": stem,
        "ok": True,
        "n_segments": len(segs),
        "n_trials": len(all_rows),
        "n_todo_this_run": len(jobs),
        "work_dir": str(video_work),
        "wall_sec": time.monotonic() - t0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--segmented-dir",
        type=Path,
        default=WORKSPACE / "segmented videos",
        help="Root of per-video segment folders (default: ../segmented videos)",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=WORKSPACE / "raw videos",
        help="Raw source videos for packet-byte size-in (default: ../raw videos)",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "work" / "segment_crf_aq_grid",
        help="Output root (default: work/segment_crf_aq_grid)",
    )
    p.add_argument("--limit", type=int, default=0, help="Max videos (0=all)")
    p.add_argument(
        "--videos",
        default="",
        help="Comma-separated video stems to include (default: all)",
    )
    p.add_argument("--crf-min", type=int, default=22)
    p.add_argument("--crf-max", type=int, default=38)
    p.add_argument("--crf-step", type=int, default=1)
    p.add_argument("--aq-min", type=float, default=0.3)
    p.add_argument("--aq-max", type=float, default=2.3)
    p.add_argument("--aq-step", type=float, default=0.1)
    p.add_argument(
        "--params",
        default=DEFAULT_BASE_PARAMS,
        help="Fixed libx265 params (aq-strength overwritten per trial)",
    )
    p.add_argument("--preset", "-p", default="fast", choices=_X265_PRESETS)
    p.add_argument("--profile", default="main")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument("--gpu", action="store_true", help="Use Docker libvmaf_cuda")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument(
        "--vmaf-n-threads",
        type=int,
        default=0,
        help="libvmaf threads/job (0=auto from worker count)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=18,
        help="Total encode+score workers (default: 18)",
    )
    p.add_argument(
        "--video-workers",
        type=int,
        default=3,
        help="Videos processed in parallel (default: 3 → 6 workers/video with --workers 18)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="Wipe per-video work dir first")
    p.add_argument(
        "--keep-encodes",
        action="store_true",
        help="Keep per-trial mp4s (default: delete after score)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    segmented_dir: Path = args.segmented_dir
    raw_dir: Path = args.raw_dir
    work_root: Path = args.work_dir

    if not segmented_dir.is_dir():
        raise SystemExit(f"segmented-dir not found: {segmented_dir}")
    if not raw_dir.is_dir():
        raise SystemExit(f"raw-dir not found: {raw_dir}")

    video_dirs = _discover_video_dirs(segmented_dir)
    if args.videos.strip():
        want = {s.strip() for s in args.videos.split(",") if s.strip()}
        video_dirs = [d for d in video_dirs if d.name in want]
    if args.limit > 0:
        video_dirs = video_dirs[: int(args.limit)]
    if not video_dirs:
        raise SystemExit(f"no segment folders found under {segmented_dir}")

    total_workers = max(1, int(args.workers))
    video_workers = max(1, min(int(args.video_workers), len(video_dirs), total_workers))
    workers_per_video = max(1, total_workers // video_workers)

    vmaf_n_threads = int(args.vmaf_n_threads)
    if vmaf_n_threads <= 0:
        vmaf_n_threads = max(2, min(6, 48 // total_workers))

    crfs = _build_int_grid(args.crf_min, args.crf_max, args.crf_step)
    aqs = _build_float_grid(args.aq_min, args.aq_max, args.aq_step)
    grid = [(crf, aq) for crf in crfs for aq in aqs]
    base_params = parse_x265_params(args.params)
    base_params.pop("aq-strength", None)

    work_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"segmented  : {segmented_dir}")
    print(f"raw        : {raw_dir}")
    print(f"work_dir   : {work_root}")
    print(f"videos     : {len(video_dirs)}")
    print(
        f"grid/seg   : {len(grid)} points  "
        f"(CRF {crfs[0]}..{crfs[-1]} step {args.crf_step}, "
        f"AQ {aqs[0]}..{aqs[-1]} step {args.aq_step})"
    )
    print(
        f"workers    : total={total_workers}  "
        f"video_parallel={video_workers}  per_video={workers_per_video}"
    )
    print(f"preset     : {args.preset}")
    print("GOP        : scene-as-one (keyint=min-keyint=frame_count, scenecut=0)")
    print(
        f"vmaf       : thr={args.vmaf_threshold} "
        f"backend={'GPU' if args.gpu else 'CPU'} threads/job={vmaf_n_threads}"
    )
    print("comp_ratio : source packet bytes / encoded size")
    print("saved      : crf, aq, vmaf_base, vmaf_neg, rate, ratio, s_f → trials.jsonl + results.csv")
    print("=" * 88)

    results: list[dict[str, Any]] = []
    t_wall0 = time.monotonic()

    def _run(video_dir: Path) -> dict[str, Any]:
        return _process_one_video(
            video_dir,
            raw_dir=raw_dir,
            work_root=work_root,
            grid=grid,
            base_params=base_params,
            preset=args.preset,
            profile=args.profile,
            vmaf_threshold=args.vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=args.vmaf_n_subsample,
            use_gpu=bool(args.gpu),
            gpu_device=args.gpu_device,
            keep_encode=bool(args.keep_encodes),
            resume=bool(args.resume),
            force=bool(args.force),
            workers_per_video=workers_per_video,
        )

    with ThreadPoolExecutor(max_workers=video_workers) as pool:
        futs = {pool.submit(_run, d): d for d in video_dirs}
        done = 0
        for fut in as_completed(futs):
            video_dir = futs[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as exc:
                row = {"video_stem": video_dir.name, "ok": False, "error": str(exc)}
                print(f"FAIL {video_dir.name}: {exc}", flush=True)
            results.append(row)
            tag = "ok" if row.get("ok") else f"FAIL {row.get('error')}"
            print(
                f"=== video [{done}/{len(video_dirs)}] {row.get('video_stem')} {tag} ===",
                flush=True,
            )

    # Global flatten
    all_path = work_root / "all_trials.jsonl"
    with all_path.open("w", encoding="utf-8") as f:
        for video_dir in video_dirs:
            p = work_root / video_dir.name / "all_trials.jsonl"
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    f.write(line.rstrip() + "\n")

    summary = {
        "segmented_dir": str(segmented_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "work_dir": str(work_root.resolve()),
        "grid": {
            "crf_min": args.crf_min,
            "crf_max": args.crf_max,
            "crf_step": args.crf_step,
            "aq_min": args.aq_min,
            "aq_max": args.aq_max,
            "aq_step": args.aq_step,
            "n_points": len(grid),
        },
        "workers": total_workers,
        "video_workers": video_workers,
        "workers_per_video": workers_per_video,
        "preset": args.preset,
        "vmaf_threshold": args.vmaf_threshold,
        "videos": results,
        "wall_sec": time.monotonic() - t_wall0,
        "all_trials_jsonl": str(all_path),
    }
    summary_path = work_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r.get("ok"))
    print("=" * 88)
    print(f"done       : {n_ok}/{len(results)} videos")
    print(f"summary    : {summary_path}")
    print(f"all_trials : {all_path}")
    print(f"wall_sec   : {time.monotonic() - t_wall0:.1f}")
    print("=" * 88)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
