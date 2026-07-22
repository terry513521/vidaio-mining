#!/usr/bin/env python3
"""Recompute compression_rate / ratio / s_f on an existing segment sweep.

Does **not** re-encode or re-run VMAF. Uses stored ``size_out_bytes`` + source
video packet bytes (same formula as ``test_zones_zonefile_score``).

Example:
  python3 scripts/refine_segment_sweep_compression.py \\
    --work-dir work/crf_aq_segment_sweep/d7cbca62-b96c-4370-804f-23a930ea3455 \\
    --input "../raw videos/d7cbca62-b96c-4370-804f-23a930ea3455.mp4"

  # dry-run (print only)
  python3 scripts/refine_segment_sweep_compression.py \\
    --work-dir work/crf_aq_segment_sweep/... --input video.mp4 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import calculate_compression_score
from test_crf_aq_segment_sweep import (
    _load_segment_rows,
    _source_segment_packet_bytes,
)


CSV_FIELDS = [
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
]


def _size_out(row: dict[str, Any]) -> int:
    try:
        n = int(row.get("size_out_bytes") or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n
    # Fallback: encode file still on disk
    out = row.get("output_path")
    if out:
        p = Path(str(out))
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return int(p.stat().st_size)
    return 0


def _refine_row(
    row: dict[str, Any],
    *,
    source_bytes: int,
    vmaf_threshold: float,
) -> tuple[dict[str, Any], bool]:
    """Return (updated_row, changed)."""
    if not row.get("encode_ok"):
        return row, False

    out_bytes = _size_out(row)
    if out_bytes <= 0 or source_bytes <= 0:
        return row, False

    # Competition semantics (same as compress_util.measure_compression)
    if out_bytes >= source_bytes:
        rate = 1.0
        ratio = 1.0
    else:
        rate = out_bytes / source_bytes
        ratio = source_bytes / out_bytes

    vmaf = float(row.get("vmaf_neg") or 0.0)
    s_f, _c, _q, reason = calculate_compression_score(
        vmaf_score=vmaf,
        compression_rate=rate,
        vmaf_threshold=float(vmaf_threshold),
    )

    old_rate = float(row.get("compression_rate") or 0.0)
    old_ratio = float(row.get("compression_ratio") or 0.0)
    old_sf = float(row.get("s_f") or 0.0)
    changed = (
        abs(old_rate - rate) > 1e-9
        or abs(old_ratio - ratio) > 1e-6
        or abs(old_sf - s_f) > 1e-9
        or str(row.get("reason") or "") != reason
    )

    out = dict(row)
    out["size_out_bytes"] = int(out_bytes)
    out["size_in_bytes"] = int(source_bytes)
    out["compression_rate"] = float(rate)
    out["compression_ratio"] = float(ratio)
    out["s_f"] = float(s_f)
    out["reason"] = reason
    # Keep gates_* as-is (encoding / VMAF-delta gates don't depend on size-in)
    return out, changed


def _write_trials(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in CSV_FIELDS})


def _best_row(rows: list[dict[str, Any]], *, gated: bool) -> Optional[dict[str, Any]]:
    cand = [r for r in rows if r.get("encode_ok")]
    if gated:
        cand = [r for r in cand if r.get("gates_ok")]
    if not cand:
        return None
    return max(cand, key=lambda r: float(r.get("s_f") or 0.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Segment sweep root (contains segment_XX/)",
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Original source video (for packet-byte size-in)",
    )
    p.add_argument("--vmaf-threshold", type=float, default=85.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes only; do not rewrite files",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write *.bak before overwrite",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work_dir: Path = args.work_dir
    input_path: Path = args.input
    if not work_dir.is_dir():
        raise SystemExit(f"work-dir not found: {work_dir}")
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    seg_dirs = sorted(work_dir.glob("segment_*"))
    seg_dirs = [d for d in seg_dirs if d.is_dir()]
    if not seg_dirs:
        raise SystemExit(f"no segment_* dirs under {work_dir}")

    # Build segment stubs from trials (start/end frames)
    segments: list[dict[str, Any]] = []
    for seg_dir in seg_dirs:
        trials = _load_segment_rows(seg_dir / "trials.jsonl")
        if not trials:
            continue
        sample = trials[0]
        idx = int(sample.get("segment_index", seg_dir.name.split("_")[-1]))
        segments.append(
            {
                "index": idx,
                "start_frame": int(sample["start_frame"]),
                "end_frame": int(sample["end_frame"]),
            }
        )

    print(f"input      : {input_path}")
    print(f"work_dir   : {work_dir}")
    print(f"segments   : {len(segments)}")
    print("probing source packet bytes …", flush=True)
    src_bytes = _source_segment_packet_bytes(input_path, segments)
    for seg in segments:
        idx = int(seg["index"])
        print(
            f"  seg[{idx}] frames={seg['start_frame']}-{seg['end_frame']}  "
            f"source_pkt={src_bytes.get(idx, 0)/1e6:.2f}MB",
            flush=True,
        )

    total_changed = 0
    total_rows = 0
    segment_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for seg in segments:
        idx = int(seg["index"])
        seg_dir = work_dir / f"segment_{idx:02d}"
        trials_path = seg_dir / "trials.jsonl"
        rows = _load_segment_rows(trials_path)
        source_b = int(src_bytes.get(idx, 0))
        new_rows: list[dict[str, Any]] = []
        n_changed = 0
        for row in rows:
            total_rows += 1
            updated, changed = _refine_row(
                row,
                source_bytes=source_b,
                vmaf_threshold=args.vmaf_threshold,
            )
            if changed:
                n_changed += 1
                total_changed += 1
            new_rows.append(updated)

        best = _best_row(new_rows, gated=False)
        best_gated = _best_row(new_rows, gated=True)
        print(
            f"seg[{idx}] rows={len(rows)} changed={n_changed}  "
            f"best_s_f={float(best['s_f']):.4f} ratio={float(best['compression_ratio']):.2f}x "
            f"crf={best['crf']} aq={best['aq_strength']}"
            if best
            else f"seg[{idx}] rows={len(rows)} changed={n_changed} (no ok trials)",
            flush=True,
        )

        if not args.dry_run:
            if not args.no_backup and trials_path.is_file():
                shutil.copy2(trials_path, trials_path.with_suffix(".jsonl.bak"))
            _write_trials(trials_path, new_rows)
            csv_path = seg_dir / "results.csv"
            if csv_path.is_file() and not args.no_backup:
                shutil.copy2(csv_path, csv_path.with_suffix(".csv.bak"))
            _write_csv(csv_path, new_rows)

            ok_rows = [r for r in new_rows if r.get("encode_ok")]
            seg_summary = {
                "segment_index": idx,
                "start_frame": int(seg["start_frame"]),
                "end_frame": int(seg["end_frame"]),
                "source_packet_bytes": source_b,
                "n_trials": len(new_rows),
                "n_ok": len(ok_rows),
                "best_s_f": best,
                "best_gated_s_f": best_gated,
                "trials_jsonl": str(trials_path),
                "compression_refined": True,
            }
            (seg_dir / "summary.json").write_text(
                json.dumps(seg_summary, indent=2), encoding="utf-8"
            )
            segment_summaries.append(seg_summary)
            all_rows.extend(new_rows)
        else:
            all_rows.extend(new_rows)

    if not args.dry_run:
        summary = {
            "input": str(input_path.resolve()),
            "work_dir": str(work_dir.resolve()),
            "mode": "per_segment",
            "compression_refined": True,
            "compression_formula": "source_packet_bytes / encoded_size",
            "vmaf_threshold": float(args.vmaf_threshold),
            "segments": segment_summaries,
            "n_trials_total": total_rows,
            "n_trials_changed": total_changed,
        }
        summary_path = work_dir / "summary.json"
        if summary_path.is_file() and not args.no_backup:
            shutil.copy2(summary_path, summary_path.with_suffix(".json.bak"))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        all_path = work_dir / "all_trials.jsonl"
        if all_path.is_file() and not args.no_backup:
            shutil.copy2(all_path, all_path.with_suffix(".jsonl.bak"))
        with all_path.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("=" * 72)
    print(f"rows       : {total_rows}")
    print(f"changed    : {total_changed}")
    print(f"mode       : {'dry-run' if args.dry_run else 'wrote files (+ .bak)'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
