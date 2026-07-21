#!/usr/bin/env python3
"""Build per-video oracle labels for CRF/x265 from explicit trial sweeps.

Protocol:
  Stage 0: derive feature-based x265 baseline params per video.
  Stage 1: sweep CRF x aq-strength while keeping all other x265 params fixed.
  Stage 2: optional coordinate refinement over selected x265 knobs.

Writes:
  - <work_dir>/<video_stem>/trials.jsonl
  - <work_dir>/<video_stem>/oracle.json
  - <work_dir>/oracle_summary.jsonl

Resume: skips videos that already have a valid oracle.json (use --force to redo).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoder import encode_hevc
from interp_search import format_x265_params, parse_x265_params, propose_feature_x265_params
from scoring import ScoreResult, probe_video, score_candidate


DEFAULT_AQ_MIN = 0.6
DEFAULT_AQ_MAX = 1.4
DEFAULT_AQ_STEP = 0.2
DEFAULT_CRF_MIN = 26
DEFAULT_CRF_MAX = 38
DEFAULT_CRF_STEP = 2
DEFAULT_CRF_FINE_RADIUS = 1  # ±1 CRF at step 1 around best coarse (e.g. 30 between 28/32)
DEFAULT_AQ_FINE_RADIUS = 0.3
DEFAULT_AQ_FINE_STEP = 0.3
STAGE2_CANDIDATES: dict[str, tuple[Any, ...]] = {
    "aq-mode": (1, 2, 3),
    "rd": (4, 5, 6),
    "ref": (4, 5, 6),
    "bframes": (4, 6, 8, 10, 12),
    "rc-lookahead": (20, 30, 40, 50, 60),
}


@dataclass
class Trial:
    stage: str
    trial_idx: int
    input_path: str
    output_path: str
    crf: int
    aq_strength: float
    params: str
    preprocess: Optional[str]
    encode_ok: bool
    encode_sec: float
    score_sec: float
    s_f: float
    vmaf_neg: float
    vmaf_base: Optional[float]
    vmaf_delta: Optional[float]
    compression_rate: float
    reason: str
    gates_ok: bool
    passed_encoding_gates: bool
    passed_vmaf_delta_gate: bool
    validation_errors: list[str]
    error: str


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _build_float_grid(min_val: float, max_val: float, step: float) -> list[float]:
    """Inclusive grid from min to max at fixed step."""
    if step <= 0:
        raise ValueError(f"grid step must be > 0, got {step}")
    if max_val < min_val:
        raise ValueError(f"grid max must be >= min, got [{min_val}, {max_val}]")
    out: list[float] = []
    v = float(min_val)
    while v <= float(max_val) + 1e-9:
        out.append(round(v, 4))
        v += float(step)
    return out


def _resolve_aq_strengths(args: argparse.Namespace) -> list[float]:
    explicit = _parse_float_list(getattr(args, "aq_strengths", "") or "")
    if explicit:
        return explicit
    return _build_float_grid(args.aq_min, args.aq_max, args.aq_step)


def _load_features(input_path: Path, features_dir: Optional[Path]) -> dict[str, Any]:
    candidates: list[Path] = []
    if features_dir:
        candidates.append(features_dir / f"{input_path.stem}.json")
    candidates.append(Path("video_features") / f"{input_path.stem}.json")
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if "noise_level_norm" in data or "motion_level" in data:
            return data
        for key in ("global", "features"):
            nested = data.get(key)
            if isinstance(nested, dict) and (
                "noise_level_norm" in nested or "motion_level" in nested
            ):
                return dict(nested)
    return {}


def _trial_better(a: Trial, b: Trial) -> bool:
    if a.gates_ok and not b.gates_ok:
        return True
    if not a.gates_ok and b.gates_ok:
        return False
    if a.s_f > b.s_f + 1e-9:
        return True
    if abs(a.s_f - b.s_f) <= 1e-9:
        return a.vmaf_neg > b.vmaf_neg + 1e-9
    return False


def _score_once(
    *,
    trial_idx: int,
    stage: str,
    input_path: Path,
    output_path: Path,
    crf: int,
    params: str,
    preprocess: Optional[str],
    vmaf_threshold: int,
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    preset: str,
    profile: str,
) -> Trial:
    t0 = time.monotonic()
    enc = encode_hevc(
        str(input_path),
        str(output_path),
        preset=preset,
        params=params,
        codec_mode="RC",
        crf=int(crf),
        bitrate=None,
        encoder="libx265",
        preprocess=preprocess,
        libx265_profile=profile,
        progress_reference_path=str(input_path),
        progress_label=f"{stage} CRF={crf}",
    )
    encode_sec = time.monotonic() - t0
    aq_strength = float(parse_x265_params(params).get("aq-strength", "1.0"))
    if not enc.ok:
        return Trial(
            stage=stage,
            trial_idx=trial_idx,
            input_path=str(input_path),
            output_path=str(output_path),
            crf=int(crf),
            aq_strength=aq_strength,
            params=params,
            preprocess=preprocess,
            encode_ok=False,
            encode_sec=encode_sec,
            score_sec=0.0,
            s_f=0.0,
            vmaf_neg=0.0,
            vmaf_base=None,
            vmaf_delta=None,
            compression_rate=1.0,
            reason="encode_failed",
            gates_ok=False,
            passed_encoding_gates=False,
            passed_vmaf_delta_gate=False,
            validation_errors=[],
            error=enc.stderr_tail[-400:],
        )

    t1 = time.monotonic()
    score: ScoreResult = score_candidate(
        str(input_path),
        str(output_path),
        vmaf_threshold,
        vmaf_n_subsample=vmaf_n_subsample,
        vmaf_n_threads=vmaf_n_threads,
        vmaf_backend="docker",
        vmaf_docker_image="vmaf_ffmpeg",
        vmaf_docker_gpus=use_gpu,
        vmaf_gpu_device=gpu_device if use_gpu else None,
        vmaf_gpu_prefer=bool(use_gpu),
        codec_mode="RC",
        target_bitrate_mbps=None,
    )
    score_sec = time.monotonic() - t1
    gates_ok = bool(score.passed_encoding_gates and score.passed_vmaf_delta_gate)
    return Trial(
        stage=stage,
        trial_idx=trial_idx,
        input_path=str(input_path),
        output_path=str(output_path),
        crf=int(crf),
        aq_strength=aq_strength,
        params=params,
        preprocess=preprocess,
        encode_ok=True,
        encode_sec=encode_sec,
        score_sec=score_sec,
        s_f=float(score.s_f),
        vmaf_neg=float(score.vmaf),
        vmaf_base=None if score.vmaf_base is None else float(score.vmaf_base),
        vmaf_delta=None if score.vmaf_delta is None else float(score.vmaf_delta),
        compression_rate=float(score.compression_rate),
        reason=str(score.reason),
        gates_ok=gates_ok,
        passed_encoding_gates=bool(score.passed_encoding_gates),
        passed_vmaf_delta_gate=bool(score.passed_vmaf_delta_gate),
        validation_errors=list(score.validation_errors or []),
        error="",
    )


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _stage1_crf_aq_search(
    *,
    input_path: Path,
    work_dir: Path,
    base_params: dict[str, Any],
    vmaf_threshold: int,
    crf_min: int,
    crf_max: int,
    coarse_step: int,
    fine_radius: int,
    aq_fine_radius: float,
    aq_fine_step: float,
    aq_strengths: list[float],
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    preset: str,
    profile: str,
    preprocess: Optional[str],
    trials_path: Path,
    trial_start_idx: int,
) -> tuple[list[Trial], int]:
    tried: set[tuple[int, float]] = set()
    rows: list[Trial] = []
    trial_idx = trial_start_idx

    def run_point(crf: int, aq: float, stage: str) -> None:
        nonlocal trial_idx
        key = (int(crf), round(float(aq), 4))
        if key in tried:
            return
        tried.add(key)
        p = dict(base_params)
        p["aq-strength"] = round(float(aq), 3)
        params_str = format_x265_params(p)
        out = work_dir / "encodes" / f"{stage}_crf{crf}_aq{aq:.3f}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        trial = _score_once(
            trial_idx=trial_idx,
            stage=stage,
            input_path=input_path,
            output_path=out,
            crf=crf,
            params=params_str,
            preprocess=preprocess,
            vmaf_threshold=vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=vmaf_n_subsample,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            preset=preset,
            profile=profile,
        )
        rows.append(trial)
        _write_jsonl(trials_path, asdict(trial))
        trial_idx += 1
        print(
            f"[{stage}] crf={crf} aq={aq:.2f} s_f={trial.s_f:.4f} "
            f"vmaf={trial.vmaf_neg:.2f} gates={trial.gates_ok} reason={trial.reason}",
            flush=True,
        )

    coarse_crfs = list(range(crf_min, crf_max + 1, max(1, coarse_step)))
    if coarse_crfs[-1] != crf_max:
        coarse_crfs.append(crf_max)
    for crf in coarse_crfs:
        for aq in aq_strengths:
            run_point(crf, aq, "stage1_coarse")

    if fine_radius <= 0 and aq_fine_radius <= 0:
        return rows, trial_idx

    best_coarse = _best_trial(rows)
    if best_coarse is None:
        return rows, trial_idx

    center_crf = int(best_coarse.crf)
    center_aq = float(best_coarse.aq_strength)

    fine_crfs: list[int] = []
    if fine_radius > 0:
        fine_min = max(crf_min, center_crf - fine_radius)
        fine_max = min(crf_max, center_crf + fine_radius)
        fine_crfs = list(range(fine_min, fine_max + 1))

    fine_aqs: list[float] = []
    if aq_fine_radius > 0 and aq_fine_step > 0:
        aq_lo = max(0.1, center_aq - aq_fine_radius)
        aq_hi = min(2.0, center_aq + aq_fine_radius)
        fine_aqs = _build_float_grid(aq_lo, aq_hi, aq_fine_step)
        if not fine_aqs:
            fine_aqs = [center_aq]

    if not fine_crfs:
        fine_crfs = [center_crf]
    if not fine_aqs:
        fine_aqs = [center_aq]

    for crf in fine_crfs:
        for aq in fine_aqs:
            run_point(crf, aq, "stage1_fine")
    return rows, trial_idx


def _stage2_refine(
    *,
    input_path: Path,
    work_dir: Path,
    start_trial: Trial,
    vmaf_threshold: int,
    rounds: int,
    vmaf_n_threads: int,
    vmaf_n_subsample: int,
    use_gpu: bool,
    gpu_device: int,
    preset: str,
    profile: str,
    preprocess: Optional[str],
    trials_path: Path,
    trial_start_idx: int,
) -> tuple[list[Trial], Trial, int]:
    rows: list[Trial] = []
    trial_idx = trial_start_idx
    incumbent = start_trial
    current = parse_x265_params(incumbent.params)
    fixed_crf = int(incumbent.crf)

    for rnd in range(max(0, rounds)):
        improved = False
        for key, candidates in STAGE2_CANDIDATES.items():
            best_local = incumbent
            best_local_params = dict(current)
            for cand in candidates:
                cand_params = dict(current)
                cand_params[key] = str(cand)
                params_str = ":".join(f"{k}={v}" for k, v in cand_params.items())
                out = work_dir / "encodes" / f"stage2_r{rnd}_{key}_{cand}.mp4"
                trial = _score_once(
                    trial_idx=trial_idx,
                    stage=f"stage2_r{rnd}_{key}",
                    input_path=input_path,
                    output_path=out,
                    crf=fixed_crf,
                    params=params_str,
                    preprocess=preprocess,
                    vmaf_threshold=vmaf_threshold,
                    vmaf_n_threads=vmaf_n_threads,
                    vmaf_n_subsample=vmaf_n_subsample,
                    use_gpu=use_gpu,
                    gpu_device=gpu_device,
                    preset=preset,
                    profile=profile,
                )
                rows.append(trial)
                _write_jsonl(trials_path, asdict(trial))
                trial_idx += 1
                if _trial_better(trial, best_local):
                    best_local = trial
                    best_local_params = dict(cand_params)
            if _trial_better(best_local, incumbent):
                incumbent = best_local
                current = best_local_params
                improved = True
                print(
                    f"[stage2] improved via {key}: s_f={incumbent.s_f:.4f} "
                    f"vmaf={incumbent.vmaf_neg:.2f} params={incumbent.params}",
                    flush=True,
                )
        if not improved:
            break
    return rows, incumbent, trial_idx


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    for p in args.input:
        out.append(Path(p))
    if args.inputs_file:
        for line in Path(args.inputs_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(Path(s))
    if args.video_dir and args.video_ids:
        base = Path(args.video_dir)
        for part in str(args.video_ids).split(","):
            token = part.strip()
            if not token:
                continue
            out.append(base / f"{token}.mp4")
    if args.all_videos:
        base = Path(args.video_dir or "../video")
        if not base.is_dir():
            raise SystemExit(f"--all-videos: video dir not found: {base}")
        exts = ("*.mp4", "*.mkv", "*.mov", "*.webm")
        discovered: list[Path] = []
        for ext in exts:
            discovered.extend(base.glob(ext))
        discovered = sorted(discovered, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
        if args.max_videos > 0:
            discovered = discovered[: args.max_videos]
        out.extend(discovered)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _best_trial(rows: list[Trial]) -> Optional[Trial]:
    if not rows:
        return None
    best = rows[0]
    for r in rows[1:]:
        if _trial_better(r, best):
            best = r
    return best


def _min_crf_at_threshold(rows: list[Trial], threshold: int) -> Optional[dict[str, Any]]:
    good = [r for r in rows if r.gates_ok and r.vmaf_neg >= float(threshold)]
    if not good:
        return None
    min_crf = min(r.crf for r in good)
    tied = [r for r in good if r.crf == min_crf]
    pick = max(tied, key=lambda r: (r.s_f, r.vmaf_neg))
    return {
        "crf": int(pick.crf),
        "aq_strength": float(pick.aq_strength),
        "params": pick.params,
        "s_f": float(pick.s_f),
        "vmaf_neg": float(pick.vmaf_neg),
    }


def _oracle_is_complete(oracle: dict[str, Any], *, threshold: int) -> bool:
    """True when a saved oracle.json looks finished enough to resume from."""
    if int(oracle.get("threshold", -1)) != int(threshold):
        return False
    best = oracle.get("oracle_best_sf")
    if not isinstance(best, dict):
        return False
    if best.get("params") in (None, ""):
        return False
    try:
        int(best["crf"])
    except (KeyError, TypeError, ValueError):
        return False
    if int(oracle.get("trial_count") or 0) <= 0:
        return False
    return True


def _load_completed_oracle(
    oracle_path: Path,
    *,
    threshold: int,
) -> Optional[dict[str, Any]]:
    if not oracle_path.is_file():
        return None
    try:
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(oracle, dict) or not _oracle_is_complete(oracle, threshold=threshold):
        return None
    return oracle


def _rebuild_summary(
    work_root: Path,
    inputs: list[Path],
    summary_path: Path,
    *,
    threshold: int,
) -> int:
    """Rewrite oracle_summary.jsonl from completed per-video oracle.json files."""
    rows: list[dict[str, Any]] = []
    for input_path in inputs:
        oracle_path = work_root / input_path.stem / "oracle.json"
        oracle = _load_completed_oracle(oracle_path, threshold=threshold)
        if oracle is not None:
            rows.append(oracle)
    with summary_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return len(rows)


def _run_video_oracle(
    *,
    input_path: Path,
    args: argparse.Namespace,
    strengths: list[float],
    work_root: Path,
) -> Optional[dict[str, Any]]:
    if not input_path.is_file():
        print(f"[skip] missing input: {input_path}")
        return None
    stem = input_path.stem
    video_work = work_root / stem
    video_work.mkdir(parents=True, exist_ok=True)
    oracle_path = video_work / "oracle.json"
    if not args.force:
        existing = _load_completed_oracle(oracle_path, threshold=args.vmaf_threshold)
        if existing is not None:
            best = existing.get("oracle_best_sf") or {}
            print(
                f"[skip] {stem}: already done "
                f"(crf={best.get('crf')} aq={best.get('aq_strength')} "
                f"s_f={best.get('s_f')})",
                flush=True,
            )
            return existing

    trials_path = video_work / "trials.jsonl"
    if trials_path.exists():
        trials_path.unlink()

    features = _load_features(input_path, Path(args.features_dir) if args.features_dir else None)
    probe = probe_video(str(input_path))
    video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    fps_text = video_stream.get("r_frame_rate") or "0/1"
    try:
        n, d = fps_text.split("/")
        fps = float(n) / float(d)
    except Exception:
        fps = 30.0

    base_params, base_reasons = propose_feature_x265_params(features, fps=fps, quality_pack=False)
    base_params_str = format_x265_params(base_params)
    preprocess = None if str(args.preprocess).lower().strip() in {"", "none"} else str(args.preprocess).lower().strip()

    print("=" * 88)
    print(f"video={input_path} threshold={args.vmaf_threshold} preset={args.preset}")
    print(f"stage0 feature params: {base_params_str}")
    print("reasons:")
    for r in base_reasons:
        print(f"  - {r}")

    stage1_rows, next_idx = _stage1_crf_aq_search(
        input_path=input_path,
        work_dir=video_work,
        base_params=base_params,
        vmaf_threshold=args.vmaf_threshold,
        crf_min=args.crf_min,
        crf_max=args.crf_max,
        coarse_step=args.coarse_step,
        fine_radius=args.fine_radius,
        aq_fine_radius=args.aq_fine_radius,
        aq_fine_step=args.aq_fine_step,
        aq_strengths=strengths,
        vmaf_n_threads=args.vmaf_n_threads,
        vmaf_n_subsample=args.vmaf_n_subsample,
        use_gpu=bool(args.gpu),
        gpu_device=args.gpu_device,
        preset=args.preset,
        profile=args.profile,
        preprocess=preprocess,
        trials_path=trials_path,
        trial_start_idx=0,
    )
    best_stage1 = _best_trial(stage1_rows)
    if best_stage1 is None:
        print(f"[warn] no stage1 trials for {input_path}")
        return None

    stage2_rows: list[Trial] = []
    best_final = best_stage1
    if args.stage2_rounds > 0:
        stage2_rows, best_final, _ = _stage2_refine(
            input_path=input_path,
            work_dir=video_work,
            start_trial=best_stage1,
            vmaf_threshold=args.vmaf_threshold,
            rounds=args.stage2_rounds,
            vmaf_n_threads=args.vmaf_n_threads,
            vmaf_n_subsample=args.vmaf_n_subsample,
            use_gpu=bool(args.gpu),
            gpu_device=args.gpu_device,
            preset=args.preset,
            profile=args.profile,
            preprocess=preprocess,
            trials_path=trials_path,
            trial_start_idx=next_idx,
        )

    all_rows = [*stage1_rows, *stage2_rows]
    oracle = {
        "video": str(input_path),
        "video_stem": stem,
        "threshold": int(args.vmaf_threshold),
        "preset": args.preset,
        "profile": args.profile,
        "preprocess": preprocess,
        "features": features,
        "feature_params": base_params_str,
        "feature_reasons": base_reasons,
        "stage1_best": asdict(best_stage1),
        "stage2_best": asdict(best_final),
        "oracle_best_sf": {
            "crf": int(best_final.crf),
            "aq_strength": float(best_final.aq_strength),
            "params": best_final.params,
            "s_f": float(best_final.s_f),
            "vmaf_neg": float(best_final.vmaf_neg),
            "vmaf_base": best_final.vmaf_base,
            "vmaf_delta": best_final.vmaf_delta,
            "compression_rate": float(best_final.compression_rate),
            "gates_ok": bool(best_final.gates_ok),
        },
        "oracle_crf_at_threshold": _min_crf_at_threshold(all_rows, args.vmaf_threshold),
        "trial_count": len(all_rows),
        "stage1_trial_count": len(stage1_rows),
        "stage2_trial_count": len(stage2_rows),
    }
    oracle_path.write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    print(
        f"[oracle] {stem}: best crf={best_final.crf} aq={best_final.aq_strength:.2f} "
        f"s_f={best_final.s_f:.4f} vmaf={best_final.vmaf_neg:.2f} trials={len(all_rows)}",
        flush=True,
    )
    return oracle


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", default=[], help="Input video path (repeatable)")
    p.add_argument("--inputs-file", default="", help="Optional file with one video path per line")
    p.add_argument("--video-dir", default="", help="Optional dir for --video-ids")
    p.add_argument("--video-ids", default="", help="Comma-separated ids (e.g. 1,2,3) with --video-dir")
    p.add_argument(
        "--all-videos",
        action="store_true",
        help="Run on all videos in --video-dir (or ../video by default)",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap when using --all-videos (0 = no cap)",
    )
    p.add_argument("--features-dir", default="", help="Optional features directory override")
    p.add_argument("--work-dir", default="work/dataset_oracle", help="Output work directory")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument("--crf-min", type=int, default=DEFAULT_CRF_MIN, help="CRF grid min (default: 26)")
    p.add_argument("--crf-max", type=int, default=DEFAULT_CRF_MAX, help="CRF grid max (default: 38)")
    p.add_argument(
        "--coarse-step",
        type=int,
        default=DEFAULT_CRF_STEP,
        help="CRF grid step (default: 2 → 26,28,...,38)",
    )
    p.add_argument(
        "--fine-radius",
        type=int,
        default=DEFAULT_CRF_FINE_RADIUS,
        help="After coarse grid, refine ±N CRF (step 1) around best point (default: 1 → e.g. 30)",
    )
    p.add_argument(
        "--aq-fine-radius",
        type=float,
        default=DEFAULT_AQ_FINE_RADIUS,
        help="After coarse grid, refine ±aq around best point (default: 0.3)",
    )
    p.add_argument(
        "--aq-fine-step",
        type=float,
        default=DEFAULT_AQ_FINE_STEP,
        help="aq-strength step in fine refine (default: 0.3)",
    )
    p.add_argument("--aq-min", type=float, default=DEFAULT_AQ_MIN, help="aq-strength grid min")
    p.add_argument("--aq-max", type=float, default=DEFAULT_AQ_MAX, help="aq-strength grid max")
    p.add_argument("--aq-step", type=float, default=DEFAULT_AQ_STEP, help="aq-strength grid step")
    p.add_argument(
        "--aq-strengths",
        default="",
        help="Optional comma-separated aq-strength override (else aq-min/max/step)",
    )
    p.add_argument("--stage2-rounds", type=int, default=2, help="Coordinate-refine rounds (0 disables)")
    p.add_argument("--preset", default="fast")
    p.add_argument("--profile", default="main")
    p.add_argument("--preprocess", default="none", help="Named preprocess preset or none")
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Prefer Docker GPU VMAF; fall back to CPU when GPU is busy",
    )
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-threads", type=int, default=40)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="Parallel videos to process")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run videos even when a completed oracle.json already exists",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    inputs = _resolve_inputs(args)
    if not inputs:
        raise SystemExit("no inputs: pass --input or --inputs-file or --video-dir+--video-ids")
    strengths = _resolve_aq_strengths(args)
    crf_grid = list(range(args.crf_min, args.crf_max + 1, max(1, args.coarse_step)))
    if crf_grid and crf_grid[-1] != args.crf_max:
        crf_grid.append(args.crf_max)
    print(
        f"[grid] coarse CRF={crf_grid}  aq-strength={strengths}  "
        f"coarse_points={len(crf_grid) * len(strengths)}",
        flush=True,
    )
    if args.fine_radius > 0 or args.aq_fine_radius > 0:
        print(
            f"[grid] fine refine: CRF ±{args.fine_radius} step 1, "
            f"aq ±{args.aq_fine_radius} step {args.aq_fine_step} around best coarse",
            flush=True,
        )
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    summary_path = work_root / "oracle_summary.jsonl"

    workers = max(1, int(args.workers))
    if workers == 1:
        for input_path in inputs:
            _run_video_oracle(
                input_path=input_path,
                args=args,
                strengths=strengths,
                work_root=work_root,
            )
    else:
        print(f"[parallel] workers={workers} videos={len(inputs)}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _run_video_oracle,
                    input_path=input_path,
                    args=args,
                    strengths=strengths,
                    work_root=work_root,
                )
                for input_path in inputs
            ]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    n_summary = _rebuild_summary(
        work_root,
        inputs,
        summary_path,
        threshold=args.vmaf_threshold,
    )
    print(f"summary: {summary_path} ({n_summary} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
