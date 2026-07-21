#!/usr/bin/env python3
"""Sweep libaom (AV1) params one-at-a-time at fixed VBR to find VMAF sensitivity.

Fixes target compression rate → kbps, encodes with aomenc, scores dual VMAF.
Uses the locally built aomenc from /root/workspace/aom/build by default.

Example:
  python3 test_av1_vmaf_param_sweep.py \\
    --input ../video/1.mp4 \\
    --target-compression-rate 0.04 \\
    --quick --passes 1 --cpu-used 6 --gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from compress_util import bitrate_for_compression_rate
from logutil import log
from scoring import ScoreResult, probe_video, score_candidate

DEFAULT_AOMENC = Path("/root/workspace/aom/build/aomenc")

# libaom knobs that affect perceptual quality at fixed VBR bitrate.
SWEEP_KEYS: tuple[str, ...] = (
    "aq-mode",
    "sharpness",
    "deltaq-mode",
    "cpu-used",
    "min-q",
    "max-q",
)

DEFAULT_GRIDS: dict[str, list[Any]] = {
    "aq-mode": [0, 1, 2, 3],
    "sharpness": [0, 1, 2, 3, 4, 5],
    "deltaq-mode": [0, 1],
    "cpu-used": [4, 6, 8],
    "min-q": [0, 4, 8],
    "max-q": [48, 55, 63],
}

QUICK_GRIDS: dict[str, list[Any]] = {
    "aq-mode": [0, 1, 2],
    "sharpness": [0, 2, 4],
    "deltaq-mode": [0, 1],
    "cpu-used": [6, 8],
    "min-q": [0],
    "max-q": [63],
}

# Baseline aomenc flags (VBR, good quality profile).
BASELINE_FLAGS: list[str] = [
    "--good",
    "--end-usage=vbr",
    "--row-mt=1",
]


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
    aomenc_flags: list[str]
    ok: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True, help="Source video (mp4)")
    p.add_argument(
        "--target-compression-rate",
        type=float,
        required=True,
        help="Target output_size / input_size (e.g. 0.04)",
    )
    p.add_argument(
        "--aomenc",
        default=str(DEFAULT_AOMENC),
        help=f"path to aomenc (default: {DEFAULT_AOMENC})",
    )
    p.add_argument("--threads", type=int, default=16, help="aomenc --threads")
    p.add_argument("--passes", type=int, default=1, help="aomenc --passes (1=faster sweep)")
    p.add_argument("--cpu-used", type=int, default=6, help="Baseline --cpu-used")
    p.add_argument("--vmaf-threshold", type=int, default=89, choices=[85, 89, 93])
    p.add_argument("--gpu", action="store_true", help="Docker libvmaf_cuda for scoring")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-threads", type=int, default=16)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Encode only first N frames (0=all)")
    p.add_argument(
        "--y4m-path",
        default="",
        help="Reuse existing y4m (otherwise pipe from ffmpeg each encode)",
    )
    p.add_argument("--work-dir", default="", help="Output dir")
    p.add_argument("--keys", default="", help="Comma-separated params to sweep")
    p.add_argument("--grid", action="append", default=[], help="key=v1,v2,...")
    p.add_argument("--quick", action="store_true", help="Smaller grids")
    p.add_argument("--skip-baseline", action="store_true")
    return p.parse_args()


def _resolve_bitrate_kbps(input_path: Path, target_rate: float) -> int:
    rate = float(target_rate)
    if not (0.0 < rate < 1.0):
        raise SystemExit(f"target_compression_rate must be in (0, 1), got {rate}")
    probe = probe_video(str(input_path))
    fmt = probe.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise SystemExit(f"unable to probe duration for {input_path}")
    bitrate_str = bitrate_for_compression_rate(
        source_bytes=input_path.stat().st_size,
        duration_sec=duration,
        compression_rate=rate,
    )
    mbps = _parse_bitrate_mbps(bitrate_str)
    if mbps is None or mbps <= 0:
        raise SystemExit(f"invalid derived bitrate: {bitrate_str!r}")
    return max(1, int(round(mbps * 1000)))


def _parse_bitrate_mbps(value: str) -> Optional[float]:
    text = str(value or "").strip().lower().replace(" ", "")
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


def _parse_grid_overrides(raw_items: list[str]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for item in raw_items:
        key, values_text = item.split("=", 1)
        key = key.strip()
        values: list[Any] = []
        for token in values_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token) if "." not in token else float(token))
            except ValueError:
                values.append(token)
        if values:
            out[key] = values
    return out


def _baseline_flags(args: argparse.Namespace, target_kbps: int) -> list[str]:
    flags = list(BASELINE_FLAGS)
    flags.extend(
        [
            f"--target-bitrate={target_kbps}",
            f"--threads={args.threads}",
            f"--passes={args.passes}",
            f"--cpu-used={args.cpu_used}",
        ]
    )
    if int(args.passes) >= 1 and "deltaq-mode=1" not in " ".join(flags):
        # deltaq-mode=1 needs tpl model; enable by default for baseline good mode
        flags.append("--enable-tpl-model=1")
    return flags


def _with_override(flags: list[str], key: str, value: Any) -> list[str]:
    prefix = f"--{key}="
    out = [f for f in flags if not f.startswith(prefix)]
    out.append(f"--{key}={value}")
    if key == "deltaq-mode" and str(value) == "1":
        if not any(f.startswith("--enable-tpl-model=") for f in out):
            out.append("--enable-tpl-model=1")
    return out


def _base_val(flags: list[str], key: str) -> Optional[str]:
    prefix = f"--{key}="
    for f in flags:
        if f.startswith(prefix):
            return f.split("=", 1)[1]
    return None


def _ivf_to_mp4(ivf_path: Path, mp4_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(ivf_path),
        "-c",
        "copy",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  remux failed: {(proc.stderr or proc.stdout or '')[-400:]}")
        return False
    return mp4_path.is_file()


def _encode_aomenc(
    *,
    aomenc: Path,
    input_path: Path,
    y4m_path: Optional[Path],
    ivf_out: Path,
    flags: list[str],
    limit: int,
) -> tuple[bool, float, str]:
    ivf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(aomenc), *flags, "-o", str(ivf_out)]

    t0 = time.monotonic()
    if y4m_path is not None and y4m_path.is_file():
        cmd.append(str(y4m_path))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    else:
        ff_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
        if limit and limit > 0:
            ff_cmd.extend(["-frames:v", str(limit)])
        ff_cmd.extend(["-f", "yuv4mpegpipe", "-"])
        cmd.append("-")
        proc = subprocess.run(
            ff_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            return False, time.monotonic() - t0, err[-500:]
        enc = subprocess.run(cmd, input=proc.stdout, capture_output=True, text=True)
        proc = enc

    elapsed = time.monotonic() - t0
    if proc.returncode != 0 or not ivf_out.is_file():
        err = (proc.stderr or proc.stdout or "aomenc failed")[-800:]
        return False, elapsed, err
    return True, elapsed, ""


def _score_mp4(
    *,
    input_path: Path,
    mp4_path: Path,
    target_kbps: int,
    args: argparse.Namespace,
) -> tuple[Optional[ScoreResult], float]:
    t0 = time.monotonic()
    score = score_candidate(
        str(input_path),
        str(mp4_path),
        args.vmaf_threshold,
        vmaf_n_subsample=args.vmaf_n_subsample,
        vmaf_n_threads=args.vmaf_n_threads,
        vmaf_backend="docker",
        vmaf_docker_image="vmaf_ffmpeg",
        vmaf_docker_gpus=bool(args.gpu),
        vmaf_gpu_device=args.gpu_device if args.gpu else None,
        codec_mode="ABR",
        target_bitrate_mbps=target_kbps / 1000.0,
    )
    return score, time.monotonic() - t0


def _trial_row(
    *,
    param: str,
    value: str,
    flags: list[str],
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
            aomenc_flags=list(flags),
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
        aomenc_flags=list(flags),
        ok=vmaf > 0 and bool(score.passed_encoding_gates),
        note=note,
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    aomenc = Path(args.aomenc).expanduser().resolve()
    if not aomenc.is_file():
        raise SystemExit(f"aomenc not found: {aomenc} (build libaom first)")

    target_kbps = _resolve_bitrate_kbps(input_path, args.target_compression_rate)
    work_dir = (
        Path(args.work_dir).expanduser()
        if args.work_dir
        else Path("work") / "av1_vmaf_sweep" / input_path.stem
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    y4m_path = Path(args.y4m_path).expanduser() if args.y4m_path else None
    grids = QUICK_GRIDS if args.quick else DEFAULT_GRIDS
    grids = {**grids, **_parse_grid_overrides(args.grid)}
    keys = SWEEP_KEYS
    if args.keys.strip():
        keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())

    baseline_flags = _baseline_flags(args, target_kbps)
    log(f"input={input_path}")
    log(f"target_compression_rate={args.target_compression_rate} → target-bitrate={target_kbps} kbps")
    log(f"aomenc={aomenc}")
    log(f"baseline flags={' '.join(baseline_flags)}")
    log(f"work_dir={work_dir}")

    baseline_ivf = work_dir / "baseline.ivf"
    baseline_mp4 = work_dir / "baseline.mp4"
    baseline_meta = work_dir / "baseline.json"
    baseline_vmaf = 0.0

    if args.skip_baseline and baseline_meta.is_file() and baseline_mp4.is_file():
        saved = json.loads(baseline_meta.read_text(encoding="utf-8"))
        baseline_vmaf = float(saved.get("vmaf_neg") or 0.0)
        log(f"reuse baseline vmaf_neg={baseline_vmaf:.4f}")
    else:
        log("--- baseline ---")
        ok, enc_sec, err = _encode_aomenc(
            aomenc=aomenc,
            input_path=input_path,
            y4m_path=y4m_path,
            ivf_out=baseline_ivf,
            flags=baseline_flags,
            limit=args.limit,
        )
        if not ok or not _ivf_to_mp4(baseline_ivf, baseline_mp4):
            raise SystemExit(f"baseline encode failed: {err}")
        score, sc_sec = _score_mp4(
            input_path=input_path,
            mp4_path=baseline_mp4,
            target_kbps=target_kbps,
            args=args,
        )
        row = _trial_row(
            param="baseline",
            value="baseline",
            flags=baseline_flags,
            score=score,
            baseline_vmaf=0.0,
            encode_sec=enc_sec,
            score_sec=sc_sec,
            output_path=baseline_mp4,
            note=err,
        )
        if not row.ok:
            raise SystemExit(f"baseline score failed: {row.note}")
        baseline_vmaf = row.vmaf_neg
        baseline_meta.write_text(json.dumps(asdict(row), indent=2) + "\n", encoding="utf-8")
        log(
            f"baseline vmaf_neg={row.vmaf_neg:.4f} s_f={row.s_f:.4f} "
            f"rate={row.compression_rate:.4f} encode={enc_sec:.1f}s score={sc_sec:.1f}s"
        )

    rows: list[TrialRow] = []
    for key in keys:
        if key not in grids:
            log(f"skip unknown key {key!r}")
            continue
        base_val = _base_val(baseline_flags, key)
        log(f"--- sweep {key} (baseline={base_val}) ---")
        for cand in grids[key]:
            cand_text = str(cand)
            if base_val is not None and base_val == cand_text:
                log(f"  skip {key}={cand_text} (baseline)")
                continue
            flags = _with_override(baseline_flags, key, cand)
            safe = re.sub(r"[^a-zA-Z0-9._+-]+", "_", cand_text)
            ivf_out = work_dir / f"{key}={safe}.ivf"
            mp4_out = work_dir / f"{key}={safe}.mp4"
            log(f"  trial {key}={cand_text}")
            ok, enc_sec, err = _encode_aomenc(
                aomenc=aomenc,
                input_path=input_path,
                y4m_path=y4m_path,
                ivf_out=ivf_out,
                flags=flags,
                limit=args.limit,
            )
            if not ok or not _ivf_to_mp4(ivf_out, mp4_out):
                rows.append(
                    _trial_row(
                        param=key,
                        value=cand_text,
                        flags=flags,
                        score=None,
                        baseline_vmaf=baseline_vmaf,
                        encode_sec=enc_sec,
                        score_sec=0.0,
                        output_path=mp4_out,
                        note=err or "remux failed",
                    )
                )
                log(f"    FAILED: {err}")
                continue
            score, sc_sec = _score_mp4(
                input_path=input_path,
                mp4_path=mp4_out,
                target_kbps=target_kbps,
                args=args,
            )
            row = _trial_row(
                param=key,
                value=cand_text,
                flags=flags,
                score=score,
                baseline_vmaf=baseline_vmaf,
                encode_sec=enc_sec,
                score_sec=sc_sec,
                output_path=mp4_out,
                note=err,
            )
            rows.append(row)
            if row.ok:
                log(
                    f"    vmaf_neg={row.vmaf_neg:.4f} Δ={row.delta_vmaf_vs_baseline:+.4f} "
                    f"s_f={row.s_f:.4f} encode={enc_sec:.1f}s"
                )
            else:
                log("    score failed")

    impact: dict[str, dict[str, Any]] = {}
    for key in keys:
        sub = [r for r in rows if r.param == key and r.ok]
        if not sub:
            continue
        best = max(sub, key=lambda r: r.vmaf_neg)
        worst = min(sub, key=lambda r: r.vmaf_neg)
        impact[key] = {
            "max_abs_delta_vmaf": max(abs(r.delta_vmaf_vs_baseline) for r in sub),
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
        "codec": "av1",
        "aomenc": str(aomenc),
        "target_compression_rate": args.target_compression_rate,
        "target_bitrate_kbps": target_kbps,
        "baseline_flags": baseline_flags,
        "baseline_vmaf_neg": baseline_vmaf,
        "trials": [asdict(r) for r in rows],
        "impact_by_param": impact,
        "impact_ranking": [{"param": n, **s} for n, s in ranking],
    }
    out_json = work_dir / "results.json"
    out_csv = work_dir / "results.csv"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    print("\n" + "=" * 72)
    print("AV1 VMAF parameter impact ranking (max |Δ vmaf_neg| vs baseline)")
    print("=" * 72)
    print(f"baseline vmaf_neg = {baseline_vmaf:.4f}")
    print(f"target-bitrate = {target_kbps} kbps  rate = {args.target_compression_rate}")
    for name, stats in ranking:
        print(
            f"  {name:14s}  max|Δ|={stats['max_abs_delta_vmaf']:.3f}  "
            f"best {stats['best_value']} → {stats['best_vmaf_neg']:.3f}  "
            f"worst {stats['worst_value']} → {stats['worst_vmaf_neg']:.3f}"
        )
    print(f"\nWrote {out_json}\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
