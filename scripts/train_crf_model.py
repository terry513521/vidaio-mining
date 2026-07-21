#!/usr/bin/env python3
"""Train Stage-1 CRF model (crf_at_threshold) from ML table rows."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols = sorted({k for r in rows for k in r.keys() if k.startswith("f_")})
    # include threshold conditioning
    if "threshold" not in cols:
        cols = ["threshold", *cols]
    return cols


def _to_xy(rows: list[dict[str, Any]], feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(r.get(c) or 0.0) for c in feature_cols] for r in rows],
        dtype=np.float32,
    )
    y = np.array([float(r["target_crf_at_threshold"]) for r in rows], dtype=np.float32)
    return x, y


def _kfold_indices(n: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    folds = np.array_split(idx, max(2, min(k, n)))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(folds)):
        val = folds[i]
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        out.append((train, val))
    return out


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def _train_model(x_train: np.ndarray, y_train: np.ndarray):
    try:
        import lightgbm as lgb  # type: ignore

        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            objective="mae",
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model, "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=42,
        )
        model.fit(x_train, y_train)
        return model, "sklearn_hist_gbr"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-jsonl",
        default="work/ml/train_rows.jsonl",
        help="Training rows JSONL from build_ml_table.py",
    )
    p.add_argument("--out-model", default="work/ml/models/crf_stage1.pkl")
    p.add_argument("--out-metadata", default="work/ml/models/crf_stage1_meta.json")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--crf-min", type=float, default=22.0)
    p.add_argument("--crf-max", type=float, default=42.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(Path(args.train_jsonl))
    rows = [r for r in rows if r.get("target_crf_at_threshold") is not None]
    if len(rows) < 8:
        raise SystemExit(f"need at least 8 labeled rows, got {len(rows)}")

    feature_cols = _feature_columns(rows)
    x, y = _to_xy(rows, feature_cols)
    splits = _kfold_indices(len(rows), args.kfold)
    fold_metrics: list[dict[str, float]] = []

    for i, (tr_idx, va_idx) in enumerate(splits, start=1):
        model, backend = _train_model(x[tr_idx], y[tr_idx])
        pred = np.asarray(model.predict(x[va_idx]), dtype=np.float32)
        pred = np.clip(np.round(pred), args.crf_min, args.crf_max)
        m = {
            "fold": float(i),
            "mae": _mae(y[va_idx], pred),
            "rmse": _rmse(y[va_idx], pred),
        }
        fold_metrics.append(m)
        print(f"fold={i} mae={m['mae']:.3f} rmse={m['rmse']:.3f}")

    model, backend = _train_model(x, y)
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    with out_model.open("wb") as f:
        pickle.dump(model, f)

    metrics = {
        "rows": len(rows),
        "features": feature_cols,
        "backend": backend,
        "cv_mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
        "cv_rmse_mean": float(np.mean([m["rmse"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
        "target": "target_crf_at_threshold",
    }
    out_meta = Path(args.out_metadata)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"saved model={out_model} backend={backend} "
        f"cv_mae={metrics['cv_mae_mean']:.3f} cv_rmse={metrics['cv_rmse_mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
