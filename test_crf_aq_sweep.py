#!/usr/bin/env python3
"""Full-grid CRF × aq-strength sweep for one video (parallel encode + VMAF).

Default grid:
  CRF          [28, 38] step 1   → 11 values
  aq-strength  [0.4, 2.4] step 0.2 → 11 values
  Total        121 trials

Each trial encodes with libx265 CRF + fixed other x265 params (only aq-strength
varies), then scores VMAF NEG / base / s_f. Results are appended live to
JSONL + CSV under --work-dir.

Example:
  python3 test_crf_aq_sweep.py \\
    --input ../raw\\ videos/d7cbca62-b96c-4370-804f-23a930ea3455.mp4 \\
    --workers 10 --preset slow

  # Resume + plot 3D surfaces
  python3 test_crf_aq_sweep.py -i video.mp4 --workers 10 --resume --plot --plot-html

  # Optional GPU VMAF (default is CPU libvmaf)
  python3 test_crf_aq_sweep.py -i video.mp4 --workers 10 --gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from encoder import encode_hevc
from interp_search import format_x265_params, parse_x265_params, propose_feature_x265_params
from scoring import ScoreResult, probe_video, score_candidate


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

DEFAULT_BASE_PARAMS = (
    "aq-mode=2:rd=6:ref=6:rdoq-level=2:me=hex:subme=2:merange=57:"
    "max-merge=3:bframes=8:rc-lookahead=40:keyint=60:min-keyint=1:scenecut=40"
)

_write_lock = threading.Lock()
_gpu_score_lock = threading.Lock()


@dataclass
class TrialResult:
    trial_idx: int
    crf: int
    aq_strength: float
    params: str
    encode_ok: bool
    encode_sec: float
    score_sec: float
    vmaf_neg: float
    vmaf_base: Optional[float]
    vmaf_delta: Optional[float]
    compression_rate: float
    compression_ratio: float
    s_f: float
    reason: str
    gates_ok: bool
    passed_encoding_gates: bool
    passed_vmaf_delta_gate: bool
    size_out_bytes: int
    output_path: str
    error: str


def _build_float_grid(min_val: float, max_val: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if max_val < min_val:
        raise ValueError(f"max must be >= min, got [{min_val}, {max_val}]")
    out: list[float] = []
    v = float(min_val)
    while v <= float(max_val) + 1e-9:
        out.append(round(v, 4))
        v += float(step)
    return out


def _build_int_grid(min_val: int, max_val: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if max_val < min_val:
        raise ValueError(f"max must be >= min, got [{min_val}, {max_val}]")
    return list(range(int(min_val), int(max_val) + 1, int(step)))


def _load_features(input_path: Path, features_dir: Path) -> Optional[dict[str, Any]]:
    path = features_dir / f"{input_path.stem}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_fps(input_path: Path) -> float:
    probe = probe_video(str(input_path))
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fps_text = video_stream.get("r_frame_rate") or "0/1"
    try:
        n, d = fps_text.split("/")
        return float(n) / float(d)
    except Exception:
        return 30.0


def _resolve_base_params(args: argparse.Namespace, input_path: Path) -> dict[str, str]:
    if args.params.strip():
        return parse_x265_params(args.params)
    feat = _load_features(input_path, Path(args.features_dir))
    if feat is not None:
        fps = _probe_fps(input_path)
        params, reasons = propose_feature_x265_params(feat, fps=fps, quality_pack=False)
        print("base params: feature-derived", flush=True)
        for r in reasons:
            print(f"  - {r}", flush=True)
        return {k: str(v) for k, v in params.items()}
    print(f"base params: default pack ({DEFAULT_BASE_PARAMS})", flush=True)
    return parse_x265_params(DEFAULT_BASE_PARAMS)


def _completed_keys(trials_path: Path) -> set[tuple[int, float]]:
    done: set[tuple[int, float]] = set()
    if not trials_path.is_file():
        return done
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("encode_ok", False):
            # allow retry of failed encodes on resume
            continue
        crf = int(row["crf"])
        aq = round(float(row["aq_strength"]), 4)
        done.add((crf, aq))
    return done


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_csv(path: Path, row: TrialResult, *, write_header: bool) -> None:
    fieldnames = [
        "trial_idx",
        "crf",
        "aq_strength",
        "compression_ratio",
        "compression_rate",
        "vmaf_neg",
        "vmaf_base",
        "vmaf_delta",
        "s_f",
        "reason",
        "gates_ok",
        "encode_sec",
        "score_sec",
        "size_out_bytes",
        "params",
        "output_path",
        "error",
    ]
    with _write_lock:
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file or write_header:
                w.writeheader()
            w.writerow({k: getattr(row, k) for k in fieldnames})


def _run_one(
    *,
    trial_idx: int,
    crf: int,
    aq: float,
    input_path: Path,
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
) -> TrialResult:
    params = dict(base_params)
    params["aq-strength"] = f"{round(float(aq), 3):g}"
    params_str = format_x265_params(params)

    t0 = time.monotonic()
    enc = encode_hevc(
        str(input_path),
        str(out_path),
        preset=preset,
        params=params_str,
        codec_mode="RC",
        crf=int(crf),
        bitrate=None,
        encoder="libx265",
        preprocess=preprocess,
        libx265_profile=profile,
        progress_reference_path=str(input_path),
        progress_label=f"CRF{crf}/aq{aq:.1f}",
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

    def _score() -> ScoreResult:
        return score_candidate(
            str(input_path),
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
            target_bitrate_mbps=None,
        )

    # Serialize GPU VMAF so 10 workers don't contend / OOM the CUDA scorer.
    if use_gpu:
        with _gpu_score_lock:
            score = _score()
    else:
        score = _score()
    score_sec = time.monotonic() - t1

    size_out = out_path.stat().st_size if out_path.is_file() else 0
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
    p.add_argument("--input", "-i", required=True, help="Source video path")
    p.add_argument(
        "--work-dir",
        default="",
        help="Output directory (default: work/crf_aq_sweep/<stem>)",
    )
    p.add_argument("--crf-min", type=int, default=28)
    p.add_argument("--crf-max", type=int, default=38)
    p.add_argument("--crf-step", type=int, default=1)
    p.add_argument("--aq-min", type=float, default=0.4)
    p.add_argument("--aq-max", type=float, default=2.4)
    p.add_argument("--aq-step", type=float, default=0.2)
    p.add_argument(
        "--params",
        default="",
        help="Fixed libx265 params (aq-strength overwritten per trial). "
        "Default: feature-derived if available, else a slow analysis pack.",
    )
    p.add_argument("--features-dir", default="video_features")
    p.add_argument("--preset", "-p", default="slow", choices=_X265_PRESETS)
    p.add_argument("--profile", default="main")
    p.add_argument("--preprocess", default="none")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use Docker libvmaf_cuda for VMAF (default: CPU libvmaf)",
    )
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument(
        "--vmaf-n-threads",
        type=int,
        default=6,
        help="libvmaf CPU threads per score job (default: 6; use ~4 with --gpu)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel encode+score workers (default: 10)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip CRF/AQ pairs already present (encode_ok) in trials.jsonl",
    )
    p.add_argument(
        "--keep-encodes",
        action="store_true",
        help="Keep per-trial mp4s under work-dir/encodes/ (default: delete after score)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete existing trials.jsonl / results.csv and start fresh",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="After sweep, render 3× 3D surfaces (CRF × AQ × s_f / ratio / vmaf)",
    )
    p.add_argument(
        "--plot-html",
        action="store_true",
        help="With --plot, also write interactive plotly HTML",
    )
    p.add_argument(
        "--plot-gates-only",
        action="store_true",
        help="With --plot, only include trials that passed gates",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    workers = max(1, int(args.workers))
    crfs = _build_int_grid(args.crf_min, args.crf_max, args.crf_step)
    aqs = _build_float_grid(args.aq_min, args.aq_max, args.aq_step)
    grid = [(crf, aq) for crf in crfs for aq in aqs]

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("work") / "crf_aq_sweep" / input_path.stem
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    encodes_dir = work_dir / "encodes"
    encodes_dir.mkdir(parents=True, exist_ok=True)

    trials_path = work_dir / "trials.jsonl"
    csv_path = work_dir / "results.csv"
    summary_path = work_dir / "summary.json"

    if args.force:
        for p in (trials_path, csv_path, summary_path):
            if p.exists():
                p.unlink()

    done = _completed_keys(trials_path) if args.resume else set()
    todo = [(crf, aq) for crf, aq in grid if (int(crf), round(float(aq), 4)) not in done]

    base_params = _resolve_base_params(args, input_path)
    # Ensure aq-strength is not sticky from base; each trial sets it.
    base_params.pop("aq-strength", None)
    preprocess = None if str(args.preprocess).lower().strip() in {"", "none"} else str(
        args.preprocess
    ).lower().strip()

    print("=" * 88)
    print(f"input      : {input_path}")
    print(f"work_dir   : {work_dir}")
    print(f"grid       : CRF {crfs[0]}..{crfs[-1]} step {args.crf_step}  "
          f"× AQ {aqs[0]}..{aqs[-1]} step {args.aq_step}")
    print(f"points     : {len(grid)} total, {len(todo)} to run, {len(done)} skipped")
    print(f"workers    : {workers}")
    print(f"preset     : {args.preset}  profile={args.profile}")
    print(f"params     : {format_x265_params(base_params)}  (+ aq-strength per trial)")
    print(f"preprocess : {preprocess or 'none'}")
    print(f"vmaf       : thr={args.vmaf_threshold} backend={'GPU' if args.gpu else 'CPU'} "
          f"threads/job={args.vmaf_n_threads}")
    print(f"keep_encodes: {bool(args.keep_encodes)}")
    if args.gpu and workers > 1:
        print(
            "NOTE       : GPU VMAF is serialized (one at a time); encodes stay parallel",
            flush=True,
        )
    print("=" * 88)

    if not todo:
        print("nothing to do (all grid points already in trials.jsonl)", flush=True)
        return 0

    results: list[TrialResult] = []
    t_wall0 = time.monotonic()
    completed = 0

    def _job(item: tuple[int, int, float]) -> TrialResult:
        idx, crf, aq = item
        out = encodes_dir / f"crf{crf}_aq{aq:.1f}.mp4"
        return _run_one(
            trial_idx=idx,
            crf=crf,
            aq=aq,
            input_path=input_path,
            out_path=out,
            base_params=base_params,
            preset=args.preset,
            profile=args.profile,
            preprocess=preprocess,
            vmaf_threshold=args.vmaf_threshold,
            vmaf_n_threads=args.vmaf_n_threads,
            vmaf_n_subsample=args.vmaf_n_subsample,
            use_gpu=bool(args.gpu),
            gpu_device=args.gpu_device,
            keep_encode=bool(args.keep_encodes),
        )

    # Stable trial indices across resume: index into full grid
    key_to_idx = {(int(c), round(float(a), 4)): i for i, (c, a) in enumerate(grid)}
    jobs = [
        (key_to_idx[(int(crf), round(float(aq), 4))], int(crf), float(aq))
        for crf, aq in todo
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            trial = fut.result()
            results.append(trial)
            _append_jsonl(trials_path, asdict(trial))
            _append_csv(csv_path, trial, write_header=False)
            completed += 1
            elapsed = time.monotonic() - t_wall0
            rate = completed / max(elapsed, 1e-6)
            eta = (len(todo) - completed) / max(rate, 1e-9)
            print(
                f"[{completed}/{len(todo)}] "
                f"crf={trial.crf} aq={trial.aq_strength:.1f}  "
                f"vmaf={trial.vmaf_neg:.2f}  ratio={trial.compression_ratio:.2f}x  "
                f"s_f={trial.s_f:.4f}  gates={trial.gates_ok}  "
                f"enc={trial.encode_sec:.0f}s score={trial.score_sec:.0f}s  "
                f"ETA={eta/60:.1f}m  reason={trial.reason}",
                flush=True,
            )

    # Merge resume history for summary
    all_rows: list[dict[str, Any]] = []
    if trials_path.is_file():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    ok_rows = [r for r in all_rows if r.get("encode_ok")]
    best = max(ok_rows, key=lambda r: float(r.get("s_f") or 0.0), default=None)
    best_gated = max(
        (r for r in ok_rows if r.get("gates_ok")),
        key=lambda r: float(r.get("s_f") or 0.0),
        default=None,
    )

    summary = {
        "input": str(input_path),
        "work_dir": str(work_dir),
        "grid": {
            "crf_min": args.crf_min,
            "crf_max": args.crf_max,
            "crf_step": args.crf_step,
            "aq_min": args.aq_min,
            "aq_max": args.aq_max,
            "aq_step": args.aq_step,
            "n_points": len(grid),
        },
        "preset": args.preset,
        "base_params": format_x265_params(base_params),
        "vmaf_threshold": args.vmaf_threshold,
        "workers": workers,
        "n_trials_logged": len(all_rows),
        "n_ok": len(ok_rows),
        "wall_sec_this_run": time.monotonic() - t_wall0,
        "best_s_f": best,
        "best_gated_s_f": best_gated,
        "trials_jsonl": str(trials_path),
        "results_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 88)
    print(f"done       : {len(todo)} trials this run, {len(all_rows)} logged total")
    print(f"trials     : {trials_path}")
    print(f"csv        : {csv_path}")
    print(f"summary    : {summary_path}")
    if best is not None:
        print(
            f"best s_f   : crf={best['crf']} aq={best['aq_strength']} "
            f"s_f={best['s_f']:.4f} vmaf={best['vmaf_neg']:.2f} "
            f"ratio={best['compression_ratio']:.2f}x gates={best.get('gates_ok')}"
        )
    if best_gated is not None and best_gated is not best:
        print(
            f"best gated : crf={best_gated['crf']} aq={best_gated['aq_strength']} "
            f"s_f={best_gated['s_f']:.4f} vmaf={best_gated['vmaf_neg']:.2f} "
            f"ratio={best_gated['compression_ratio']:.2f}x"
        )
    print("=" * 88)

    if args.plot:
        try:
            from plot_crf_aq_sweep import plot_sweep_3d

            title = f"{input_path.name}  CRF×AQ sweep"
            plot_outputs = plot_sweep_3d(
                work_dir=work_dir,
                gates_only=bool(args.plot_gates_only),
                html=bool(args.plot_html),
                title=title,
            )
            for kind, path in plot_outputs.items():
                print(f"plot {kind:5s}: {path}")
        except Exception as exc:
            print(f"plot failed: {exc}", flush=True)
            raise SystemExit(1) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
