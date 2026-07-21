#!/usr/bin/env python3
"""Convert dataset_oracle outputs into ML training tables.

Reads per-video:
  - trials.jsonl  (all encode/score trials)
  - oracle.json   (oracle labels + features, when complete)

Writes:
  - trials.csv / trials.parquet       one row per trial
  - oracle_labels.csv / .parquet      one row per completed video
  - dataset_manifest.json             build metadata

Example (after dataset_oracle.py):
  python3 scripts/build_ml_table.py --work-dir work/dataset_oracle
  python3 scripts/build_ml_table.py --work-dir work/dataset_oracle --format both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp_search import parse_x265_params

X265_PARAM_KEYS = (
    "aq-mode",
    "aq-strength",
    "rd",
    "ref",
    "bframes",
    "rc-lookahead",
    "keyint",
    "min-keyint",
    "scenecut",
)

FEATURE_KEYS = (
    "segment_count",
    "cut_count",
    "cut_rate",
    "hard_fraction",
    "worst_difficulty",
    "difficulty_mean",
    "difficulty_p90",
    "duration_weighted_difficulty",
    "motion_mean",
    "motion_std",
    "motion_p90",
    "texture",
    "texture_lbp",
    "texture_std",
    "entropy",
    "edge_density",
    "noise_level",
    "high_freq_energy",
    "flatness",
    "luma_mean",
    "luma_std",
    "sat_mean",
    "chroma_std",
    "cut_density",
    "volatility",
    "duration",
    "fps",
    "width",
    "height",
    "short_side",
    "pixels",
    "motion_level",
    "texture_level",
    "noise_level_norm",
    "edge_level",
    "cut_level",
)


def _load_features(stem: str, features_dir: Optional[Path]) -> dict[str, Any]:
    candidates: list[Path] = []
    if features_dir:
        candidates.append(features_dir / f"{stem}.json")
    candidates.append(ROOT / "video_features" / f"{stem}.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if "noise_level_norm" in data or "motion_level" in data:
            return _canonical_features(data)
        for key in ("global", "features"):
            nested = data.get(key)
            if isinstance(nested, dict) and (
                "noise_level_norm" in nested or "motion_level" in nested
            ):
                return _canonical_features(dict(nested))
    return {}


def _canonical_features(raw: dict[str, Any]) -> dict[str, Any]:
    """Map video_features / oracle feature dicts onto stable ML column names."""
    out = dict(raw)
    alias_pairs = (
        ("motion_mean", "mean_motion"),
        ("texture", "mean_texture"),
        ("noise_level", "mean_noise"),
        ("edge_density", "mean_edge"),
        ("entropy", "mean_entropy"),
        ("high_freq_energy", "mean_hf_energy"),
        ("flatness", "mean_flatness"),
        ("luma_mean", "mean_luma"),
        ("difficulty_mean", "mean_difficulty"),
    )
    for canonical, alias in alias_pairs:
        if out.get(canonical) is None and out.get(alias) is not None:
            out[canonical] = out[alias]
    width = out.get("width")
    height = out.get("height")
    if width and height:
        try:
            w, h = int(width), int(height)
            out.setdefault("short_side", min(w, h))
            out.setdefault("pixels", w * h)
        except (TypeError, ValueError):
            pass
    if out.get("cut_density") is None and out.get("cut_rate") is not None:
        out["cut_density"] = out["cut_rate"]
    if out.get("cut_level") is None and out.get("cut_rate") is not None:
        try:
            out["cut_level"] = min(float(out["cut_rate"]) / 0.5, 1.0)
        except (TypeError, ValueError):
            pass
    return out


def _sanitize_cell(value: Any, *, max_len: int = 240) -> Any:
    if not isinstance(value, str):
        return value
    one_line = " ".join(value.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _discover_video_dirs(work_dir: Path) -> list[Path]:
    out: list[Path] = []
    if not work_dir.is_dir():
        return out
    for child in sorted(work_dir.iterdir(), key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name)):
        if not child.is_dir():
            continue
        if (child / "trials.jsonl").is_file() or (child / "oracle.json").is_file():
            out.append(child)
    return out


def _param_columns(params_str: str) -> dict[str, Any]:
    parsed = parse_x265_params(params_str or "")
    out: dict[str, Any] = {}
    for key in X265_PARAM_KEYS:
        col = key.replace("-", "_")
        raw = parsed.get(key)
        if raw is None:
            out[col] = None
            continue
        if key in {"aq-mode", "rd", "ref", "bframes", "rc-lookahead", "keyint", "min-keyint", "scenecut"}:
            try:
                out[col] = int(float(raw))
            except (TypeError, ValueError):
                out[col] = raw
        elif key == "aq-strength":
            try:
                out[col] = float(raw)
            except (TypeError, ValueError):
                out[col] = raw
        else:
            out[col] = raw
    return out


def _feature_columns(features: Optional[dict[str, Any]]) -> dict[str, Any]:
    f = _canonical_features(features or {})
    return {f"feat_{k}": f.get(k) for k in FEATURE_KEYS}


def _oracle_label_row(video_dir: Path, oracle: dict[str, Any]) -> dict[str, Any]:
    best = oracle.get("oracle_best_sf") or {}
    at_thr = oracle.get("oracle_crf_at_threshold") or {}
    row: dict[str, Any] = {
        "video_stem": oracle.get("video_stem") or video_dir.name,
        "video_path": oracle.get("video"),
        "threshold": oracle.get("threshold"),
        "preset": oracle.get("preset"),
        "profile": oracle.get("profile"),
        "preprocess": oracle.get("preprocess"),
        "feature_params": oracle.get("feature_params"),
        "trial_count": oracle.get("trial_count"),
        "stage1_trial_count": oracle.get("stage1_trial_count"),
        "stage2_trial_count": oracle.get("stage2_trial_count"),
        "oracle_crf": best.get("crf"),
        "oracle_aq_strength": best.get("aq_strength"),
        "oracle_params": best.get("params"),
        "oracle_s_f": best.get("s_f"),
        "oracle_vmaf_neg": best.get("vmaf_neg"),
        "oracle_vmaf_base": best.get("vmaf_base"),
        "oracle_vmaf_delta": best.get("vmaf_delta"),
        "oracle_compression_rate": best.get("compression_rate"),
        "oracle_gates_ok": best.get("gates_ok"),
        "crf_at_threshold": at_thr.get("crf"),
        "aq_at_threshold": at_thr.get("aq_strength"),
        "params_at_threshold": at_thr.get("params"),
        "s_f_at_threshold": at_thr.get("s_f"),
        "vmaf_neg_at_threshold": at_thr.get("vmaf_neg"),
    }
    row.update(_feature_columns(oracle.get("features") if isinstance(oracle.get("features"), dict) else {}))
    row.update(_param_columns(str(best.get("params") or oracle.get("feature_params") or "")))
    return row


def _trial_row(
    *,
    video_stem: str,
    trial: dict[str, Any],
    oracle: Optional[dict[str, Any]],
) -> dict[str, Any]:
    best = (oracle or {}).get("oracle_best_sf") or {}
    at_thr = (oracle or {}).get("oracle_crf_at_threshold") or {}
    features = oracle.get("features") if oracle and isinstance(oracle.get("features"), dict) else {}

    crf = trial.get("crf")
    aq = trial.get("aq_strength")
    params = str(trial.get("params") or "")

    is_best = False
    is_at_threshold = False
    if best:
        try:
            is_best = int(crf) == int(best.get("crf")) and abs(float(aq) - float(best.get("aq_strength"))) < 1e-6
        except (TypeError, ValueError):
            is_best = params == str(best.get("params") or "")
    if at_thr:
        try:
            is_at_threshold = int(crf) == int(at_thr.get("crf")) and abs(float(aq) - float(at_thr.get("aq_strength"))) < 1e-6
        except (TypeError, ValueError):
            is_at_threshold = params == str(at_thr.get("params") or "")

    row: dict[str, Any] = {
        "video_stem": video_stem,
        "video_path": trial.get("input_path") or (oracle or {}).get("video"),
        "threshold": (oracle or {}).get("threshold"),
        "preset": (oracle or {}).get("preset"),
        "profile": (oracle or {}).get("profile"),
        "preprocess": trial.get("preprocess") or (oracle or {}).get("preprocess"),
        "stage": trial.get("stage"),
        "trial_idx": trial.get("trial_idx"),
        "crf": crf,
        "aq_strength": aq,
        "params": params,
        "encode_ok": trial.get("encode_ok"),
        "encode_sec": trial.get("encode_sec"),
        "score_sec": trial.get("score_sec"),
        "s_f": trial.get("s_f"),
        "vmaf_neg": trial.get("vmaf_neg"),
        "vmaf_base": trial.get("vmaf_base"),
        "vmaf_delta": trial.get("vmaf_delta"),
        "compression_rate": trial.get("compression_rate"),
        "reason": trial.get("reason"),
        "gates_ok": trial.get("gates_ok"),
        "passed_encoding_gates": trial.get("passed_encoding_gates"),
        "passed_vmaf_delta_gate": trial.get("passed_vmaf_delta_gate"),
        "is_oracle_best_sf": is_best,
        "is_oracle_crf_at_threshold": is_at_threshold,
        "error": trial.get("error"),
    }
    row.update(_param_columns(params))
    row.update(_feature_columns(features))
    return row


def _ordered_fieldnames(rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in preferred:
        if any(key in r for r in rows):
            out.append(key)
            seen.add(key)
    extras = sorted({k for r in rows for k in r.keys()} - seen)
    return out + extras


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _ordered_fieldnames(rows, preferred)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Parquet output requires pyarrow. Install with: pip install pyarrow\n"
            f"({exc})"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pq.write_table(pa.table({}), path)
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_tables(
    work_dir: Path,
    *,
    gates_only: bool = False,
    include_failed: bool = True,
    features_dir: Optional[Path] = None,
    default_threshold: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trial_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    video_dirs = _discover_video_dirs(work_dir)
    feature_cache: dict[str, dict[str, Any]] = {}

    for video_dir in video_dirs:
        stem = video_dir.name
        oracle = _read_json(video_dir / "oracle.json")
        trials = _read_jsonl(video_dir / "trials.jsonl")

        if oracle is not None:
            label_rows.append(_oracle_label_row(video_dir, oracle))
            features = _canonical_features(
                oracle.get("features") if isinstance(oracle.get("features"), dict) else {}
            )
        else:
            features = feature_cache.get(stem)
            if features is None:
                features = _load_features(stem, features_dir)
                feature_cache[stem] = features

        for trial in trials:
            if not include_failed and not trial.get("encode_ok", False):
                continue
            if gates_only and not trial.get("gates_ok", False):
                continue
            row = _trial_row(video_stem=stem, trial=trial, oracle=oracle)
            if row.get("threshold") is None and default_threshold is not None:
                row["threshold"] = default_threshold
            if not any(row.get(f"feat_{k}") is not None for k in FEATURE_KEYS):
                row.update(_feature_columns(features))
            if row.get("error"):
                row["error"] = _sanitize_cell(row["error"])
            trial_rows.append(row)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "videos_seen": len(video_dirs),
        "videos_with_oracle": len(label_rows),
        "trial_rows": len(trial_rows),
        "label_rows": len(label_rows),
        "gates_only": gates_only,
        "include_failed": include_failed,
    }
    return trial_rows, label_rows, manifest


TRIAL_PREFERRED = (
    "video_stem",
    "video_path",
    "threshold",
    "stage",
    "trial_idx",
    "crf",
    "aq_strength",
    "s_f",
    "vmaf_neg",
    "vmaf_base",
    "vmaf_delta",
    "compression_rate",
    "gates_ok",
    "passed_encoding_gates",
    "passed_vmaf_delta_gate",
    "is_oracle_best_sf",
    "is_oracle_crf_at_threshold",
    "encode_ok",
    "encode_sec",
    "score_sec",
    "reason",
    "params",
    "aq_mode",
    "rd",
    "ref",
    "bframes",
    "rc_lookahead",
    "keyint",
    "scenecut",
    "preset",
    "profile",
    "preprocess",
)

LABEL_PREFERRED = (
    "video_stem",
    "video_path",
    "threshold",
    "oracle_crf",
    "oracle_aq_strength",
    "oracle_s_f",
    "oracle_vmaf_neg",
    "oracle_vmaf_base",
    "oracle_vmaf_delta",
    "oracle_compression_rate",
    "oracle_gates_ok",
    "oracle_params",
    "crf_at_threshold",
    "aq_at_threshold",
    "s_f_at_threshold",
    "vmaf_neg_at_threshold",
    "params_at_threshold",
    "feature_params",
    "trial_count",
    "stage1_trial_count",
    "stage2_trial_count",
    "preset",
    "profile",
    "preprocess",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        default="work/dataset_oracle",
        help="dataset_oracle output directory",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Where to write tables (default: <work-dir>/ml_tables)",
    )
    p.add_argument(
        "--format",
        choices=("csv", "parquet", "both"),
        default="csv",
        help="Output format",
    )
    p.add_argument(
        "--trials-out",
        default="trials",
        help="Trials table basename (extension added by format)",
    )
    p.add_argument(
        "--labels-out",
        default="oracle_labels",
        help="Oracle labels table basename",
    )
    p.add_argument(
        "--gates-only",
        action="store_true",
        help="Keep only trials that passed all gates",
    )
    p.add_argument(
        "--features-dir",
        default="",
        help="Optional video_features directory (default: repo video_features/)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="VMAF threshold label when oracle.json is not finished yet (e.g. 85)",
    )
    p.add_argument(
        "--include-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include encode_failed trials in trials table",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir) if args.output_dir else work_dir / "ml_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(args.features_dir) if args.features_dir else None
    default_threshold = int(args.threshold) if int(args.threshold) > 0 else None

    trial_rows, label_rows, manifest = build_tables(
        work_dir,
        gates_only=bool(args.gates_only),
        include_failed=bool(args.include_failed),
        features_dir=features_dir,
        default_threshold=default_threshold,
    )

    formats: list[str]
    if args.format == "both":
        formats = ["csv", "parquet"]
    else:
        formats = [args.format]

    written: list[str] = []
    for fmt in formats:
        ext = ".csv" if fmt == "csv" else ".parquet"
        trials_path = out_dir / f"{args.trials_out}{ext}"
        labels_path = out_dir / f"{args.labels_out}{ext}"
        if fmt == "csv":
            _write_csv(trials_path, trial_rows, TRIAL_PREFERRED)
            _write_csv(labels_path, label_rows, LABEL_PREFERRED)
        else:
            _write_parquet(trials_path, trial_rows)
            _write_parquet(labels_path, label_rows)
        written.extend([str(trials_path), str(labels_path)])

    manifest["output_dir"] = str(out_dir)
    manifest["written_files"] = written
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"work_dir       : {work_dir}")
    print(f"output_dir     : {out_dir}")
    print(f"videos_seen    : {manifest['videos_seen']}")
    print(f"with_oracle    : {manifest['videos_with_oracle']}")
    print(f"trial_rows     : {manifest['trial_rows']}")
    print(f"label_rows     : {manifest['label_rows']}")
    for path in written:
        print(f"wrote          : {path}")
    print(f"manifest       : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
