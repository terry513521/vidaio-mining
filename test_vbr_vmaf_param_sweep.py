#!/usr/bin/env python3
"""Sweep libx265 params one-at-a-time at fixed VBR to find VMAF sensitivity.

Fixes target compression rate (→ bitrate), encodes the full video for each
variant, scores dual VMAF, and ranks which x265-params move vmaf_neg most.

Only one parameter changes per trial; all others stay at the baseline pack.

Example:
  python3 test_vbr_vmaf_param_sweep.py \\
    --input ../video/1.mp4 \\
    --target-compression-rate 0.04 \\
    --params "aq-mode=2:aq-strength=1:rd=6:ref=6:bframes=8:rc-lookahead=50:keyint=60:min-keyint=1:scenecut=50" \\
    --preset fast --gpu --vmaf-threshold 89
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from compress_util import bitrate_for_compression_rate
from encoder import encode_hevc
from interp_search import parse_x265_params
from logutil import log
from param_tune import get_param, set_param
from scoring import ScoreResult, probe_video, score_candidate
from vbr_mode import build_vbr_mode_params

# Knobs known to affect perceptual quality / VMAF at fixed bitrate (VBR path).
SWEEP_KEYS: tuple[str, ...] = (
    "aq-mode",
    "aq-strength",
    "rd",
    "ref",
    "bframes",
    "rc-lookahead",
)

# Default grids (override per key with --grid aq-strength=0.8,1.0,1.2).
DEFAULT_GRIDS: dict[str, list[Any]] = {
    "aq-mode": [1, 2, 3],
    "aq-strength": [0.4, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4],
    "rd": [3, 4, 5, 6],
    "ref": [4, 6, 8, 12, 16],
    "bframes": [4, 6, 8, 12, 16],
    "rc-lookahead": [20, 40, 50, 60, 80],
}

QUICK_GRIDS: dict[str, list[Any]] = {
    "aq-mode": [1, 2, 3],
    "aq-strength": [0.8, 1.0, 1.2, 1.6, 2.0],
    "rd": [4, 5, 6],
    "ref": [4, 6, 8, 12],
    "bframes": [6, 8, 12],
    "rc-lookahead": [40, 50, 60],
}


@dataclass
class TrialRow:
    param: str
    value: str
    vmaf_neg: float
    vmaf_base: Optional[float]
    vmaf_delta: Optional[float]
    s_f: float
    compression_rate: float
    delta_vmaf_vs_baseline: float
    encode_sec: float
    score_sec: float
    output_path: str
    params: str
    ok: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, help="Source video")
    p.add_argument(
        "--target-compression-rate",
        type=float,
        required=True,
        help="Target output_size / input_size (e.g. 0.04)",
    )
    p.add_argument(
        "--params",
        default="",
        help="Baseline x265-params (default: vbr_mode pack from features)",
    )
    p.add_argument("--preset", default="fast", help="libx265 preset")
    p.add_argument("--profile", default="main", help="libx265 profile")
    p.add_argument("--vmaf-threshold", type=int, default=89, choices=[85, 89, 93])
    p.add_argument("--gpu", action="store_true", help="Docker libvmaf_cuda")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-threads", type=int, default=16)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument(
        "--work-dir",
        default="",
        help="Output dir (default: work/vbr_vmaf_sweep/<stem>)",
    )
    p.add_argument(
        "--keys",
        default="",
        help="Comma-separated subset of params to sweep (default: all)",
    )
    p.add_argument(
        "--grid",
        action="append",
        default=[],
        help="Override grid, e.g. aq-strength=0.8,1.0,1.2",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller default grids",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Reuse work_dir/baseline.json if present",
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


def _resolve_bitrate(input_path: Path, target_rate: float) -> str:
    rate = float(target_rate)
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
    return bitrate_for_compression_rate(
        source_bytes=input_path.stat().st_size,
        duration_sec=duration,
        compression_rate=rate,
    )


def _load_features(input_path: Path) -> dict[str, Any]:
    path = Path("video_features") / f"{input_path.stem}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    if "noise_level_norm" in data or "motion_level" in data:
        return data
    nested = data.get("features")
    if isinstance(nested, dict):
        return dict(nested)
    return {}


def _parse_grid_overrides(raw_items: list[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for item in raw_items:
        if "=" not in item:
            raise SystemExit(f"invalid --grid {item!r}; use key=v1,v2,v3")
        key, values_text = item.split("=", 1)
        key = key.strip()
        values: list[Any] = []
        for token in values_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                if "." in token:
                    values.append(float(token))
                else:
                    values.append(int(token))
            except ValueError:
                values.append(token)
        if values:
            out[key] = values
    return out


def _baseline_params(args: argparse.Namespace, features: dict[str, Any]) -> str:
    if str(args.params or "").strip():
        return str(args.params).strip()
    params, _, _ = build_vbr_mode_params(features)
    return params


def _encode_score(
    *,
    input_path: Path,
    out: Path,
    bitrate: str,
    params: str,
    args: argparse.Namespace,
    label: str,
) -> tuple[Optional[ScoreResult], float, float, str]:
    target_mbps = _parse_bitrate_mbps(bitrate) or 0.0
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
        preprocess="none",
        libx265_profile=args.profile,
        progress_reference_path=str(input_path),
        progress_label=label,
        progress_interval_sec=15.0,
    )
    encode_sec = time.monotonic() - t0
    if not enc.ok:
        err = enc.stderr_tail or "encode failed"
        return None, encode_sec, 0.0, err[-500:]

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
        codec_mode="ABR",
        target_bitrate_mbps=target_mbps,
    )
    return score, encode_sec, time.monotonic() - t1, ""


def _trial_row(
    *,
    param: str,
    value: str,
    params: str,
    score: Optional[ScoreResult],
    baseline_vmaf: float,
    encode_sec: float,
    score_sec: float,
    output_path: Path,
    note: str = "",
) -> TrialRow:
    if score is None:
        return TrialRow(
            param=param,
            value=value,
            vmaf_neg=0.0,
            vmaf_base=None,
            vmaf_delta=None,
            s_f=0.0,
            compression_rate=1.0,
            delta_vmaf_vs_baseline=0.0,
            encode_sec=encode_sec,
            score_sec=score_sec,
            output_path=str(output_path),
            params=params,
            ok=False,
            note=note,
        )
    vmaf = float(score.vmaf or 0.0)
    return TrialRow(
        param=param,
        value=value,
        vmaf_neg=vmaf,
        vmaf_base=score.vmaf_base,
        vmaf_delta=score.vmaf_delta,
        s_f=float(score.s_f or 0.0),
        compression_rate=float(score.compression_rate or 1.0),
        delta_vmaf_vs_baseline=vmaf - baseline_vmaf,
        encode_sec=encode_sec,
        score_sec=score_sec,
        output_path=str(output_path),
        params=params,
        ok=vmaf > 0 and bool(score.passed_encoding_gates),
        note=note,
    )


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    features = _load_features(input_path)
    baseline = _baseline_params(args, features)
    bitrate = _resolve_bitrate(input_path, args.target_compression_rate)
    target_mbps = _parse_bitrate_mbps(bitrate) or 0.0

    work_dir = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path("work") / "vbr_vmaf_sweep" / input_path.stem
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    grids = QUICK_GRIDS if args.quick else DEFAULT_GRIDS
    grids = {**grids, **_parse_grid_overrides(args.grid)}

    keys = SWEEP_KEYS
    if args.keys.strip():
        keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())

    log(f"input={input_path}")
    log(f"target_compression_rate={args.target_compression_rate} → bitrate={bitrate} ({target_mbps:.3f} Mbps)")
    log(f"baseline params={baseline}")
    log(f"work_dir={work_dir}")

    baseline_path = work_dir / "baseline.mp4"
    baseline_meta = work_dir / "baseline.json"

    baseline_vmaf = 0.0
    if args.skip_baseline and baseline_meta.is_file():
        saved = json.loads(baseline_meta.read_text(encoding="utf-8"))
        baseline_vmaf = float(saved.get("vmaf_neg") or 0.0)
        log(f"reuse baseline vmaf_neg={baseline_vmaf:.4f} from {baseline_meta}")
    else:
        log("--- baseline ---")
        score, enc_sec, sc_sec, err = _encode_score(
            input_path=input_path,
            out=baseline_path,
            bitrate=bitrate,
            params=baseline,
            args=args,
            label="[baseline]",
        )
        row = _trial_row(
            param="baseline",
            value="baseline",
            params=baseline,
            score=score,
            baseline_vmaf=0.0,
            encode_sec=enc_sec,
            score_sec=sc_sec,
            output_path=baseline_path,
            note=err,
        )
        if not row.ok:
            raise SystemExit(f"baseline encode/score failed: {err or row.note}")
        baseline_vmaf = row.vmaf_neg
        _save_json(baseline_meta, asdict(row))
        log(
            f"baseline vmaf_neg={row.vmaf_neg:.4f} s_f={row.s_f:.4f} "
            f"rate={row.compression_rate:.4f} encode={enc_sec:.1f}s score={sc_sec:.1f}s"
        )

    rows: list[TrialRow] = []
    parsed_base = parse_x265_params(baseline)

    for key in keys:
        if key not in grids:
            log(f"skip unknown key {key!r}")
            continue
        base_val = get_param(baseline, key, parsed_base.get(key))
        candidates = grids[key]
        log(f"--- sweep {key} (baseline={base_val}) ---")
        for cand in candidates:
            cand_text = str(cand)
            if base_val is not None and str(base_val) == cand_text:
                log(f"  skip {key}={cand_text} (baseline value)")
                continue
            trial_params = set_param(baseline, key, cand)
            safe = re.sub(r"[^a-zA-Z0-9._+-]+", "_", cand_text)
            out = work_dir / f"{key}={safe}.mp4"
            label = f"[sweep {key}={cand_text}]"
            log(f"  trial {key}={cand_text}")
            score, enc_sec, sc_sec, err = _encode_score(
                input_path=input_path,
                out=out,
                bitrate=bitrate,
                params=trial_params,
                args=args,
                label=label,
            )
            row = _trial_row(
                param=key,
                value=cand_text,
                params=trial_params,
                score=score,
                baseline_vmaf=baseline_vmaf,
                encode_sec=enc_sec,
                score_sec=sc_sec,
                output_path=out,
                note=err,
            )
            rows.append(row)
            if row.ok:
                log(
                    f"    vmaf_neg={row.vmaf_neg:.4f} Δ={row.delta_vmaf_vs_baseline:+.4f} "
                    f"s_f={row.s_f:.4f} encode={enc_sec:.1f}s"
                )
            else:
                log(f"    FAILED: {err or 'score failed'}")

    # Rank parameters by max absolute VMAF swing vs baseline.
    impact: dict[str, dict[str, Any]] = {}
    for key in keys:
        sub = [r for r in rows if r.param == key and r.ok]
        if not sub:
            continue
        best = max(sub, key=lambda r: r.vmaf_neg)
        worst = min(sub, key=lambda r: r.vmaf_neg)
        max_swing = max(abs(r.delta_vmaf_vs_baseline) for r in sub)
        impact[key] = {
            "max_abs_delta_vmaf": max_swing,
            "best_value": best.value,
            "best_vmaf_neg": best.vmaf_neg,
            "worst_value": worst.value,
            "worst_vmaf_neg": worst.vmaf_neg,
            "trials": len(sub),
        }

    ranking = sorted(
        impact.items(),
        key=lambda item: float(item[1]["max_abs_delta_vmaf"]),
        reverse=True,
    )

    results = {
        "input": str(input_path),
        "target_compression_rate": args.target_compression_rate,
        "bitrate": bitrate,
        "baseline_params": baseline,
        "baseline_vmaf_neg": baseline_vmaf,
        "trials": [asdict(r) for r in rows],
        "impact_by_param": impact,
        "impact_ranking": [
            {"param": name, **stats} for name, stats in ranking
        ],
    }
    out_json = work_dir / "results.json"
    out_csv = work_dir / "results.csv"
    _save_json(out_json, results)

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(rows[0]).keys()) if rows else list(asdict(
                _trial_row(
                    param="",
                    value="",
                    params="",
                    score=None,
                    baseline_vmaf=0.0,
                    encode_sec=0.0,
                    score_sec=0.0,
                    output_path=Path("."),
                )
            ).keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("\n" + "=" * 72)
    print("VMAF parameter impact ranking (max |Δ vmaf_neg| vs baseline)")
    print("=" * 72)
    print(f"baseline vmaf_neg = {baseline_vmaf:.4f}")
    print(f"bitrate = {bitrate}  target_rate = {args.target_compression_rate}")
    print()
    for name, stats in ranking:
        print(
            f"  {name:14s}  max|Δ|={stats['max_abs_delta_vmaf']:.3f}  "
            f"best {stats['best_value']} → {stats['best_vmaf_neg']:.3f}  "
            f"worst {stats['worst_value']} → {stats['worst_vmaf_neg']:.3f}"
        )
    print()
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
