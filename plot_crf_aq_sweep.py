#!/usr/bin/env python3
"""Plot 3D surfaces from CRF × aq-strength sweep results.

Reads ``trials.jsonl`` or ``results.csv`` under a sweep work directory and
renders three 3D graphs (CRF × aq-strength × metric):

  1. final score (s_f)
  2. compression ratio
  3. VMAF NEG

Outputs:
  - ``crf_aq_3d.png``  — matplotlib 1×3 panel (static)
  - ``crf_aq_3d.html`` — plotly interactive (optional, --html)

Example:
  python3 plot_crf_aq_sweep.py --work-dir work/crf_aq_sweep/d7cbca62-b96c-4370-804f-23a930ea3455
  python3 plot_crf_aq_sweep.py --trials work/crf_aq_sweep/video/trials.jsonl --html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def load_sweep_rows(
    *,
    work_dir: Optional[Path] = None,
    trials_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    gates_only: bool = False,
) -> list[dict[str, Any]]:
    if trials_path is None and work_dir is not None:
        trials_path = work_dir / "trials.jsonl"
    if csv_path is None and work_dir is not None:
        csv_path = work_dir / "results.csv"

    raw: list[dict[str, Any]] = []
    if trials_path and trials_path.is_file():
        raw = _load_rows_from_jsonl(trials_path)
    elif csv_path and csv_path.is_file():
        raw = _load_rows_from_csv(csv_path)
    else:
        raise SystemExit(
            "need --work-dir with trials.jsonl/results.csv, or --trials / --csv"
        )

    rows: list[dict[str, Any]] = []
    for r in raw:
        ok = r.get("encode_ok")
        if isinstance(ok, str):
            ok = ok.lower() in {"1", "true", "yes"}
        elif ok is None:
            ok = True
        if not ok:
            continue
        if gates_only and not r.get("gates_ok"):
            continue
        try:
            rows.append(
                {
                    "crf": int(float(r["crf"])),
                    "aq_strength": round(float(r["aq_strength"]), 4),
                    "s_f": float(r["s_f"]),
                    "compression_ratio": float(r["compression_ratio"]),
                    "vmaf_neg": float(r["vmaf_neg"]),
                    "gates_ok": bool(r.get("gates_ok")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        raise SystemExit("no successful trials to plot")
    return rows


def build_grid(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    crfs = sorted({int(r["crf"]) for r in rows})
    aqs = sorted({float(r["aq_strength"]) for r in rows})
    crf_to_i = {c: i for i, c in enumerate(crfs)}
    aq_to_j = {a: j for j, a in enumerate(aqs)}

    shape = (len(crfs), len(aqs))
    z_maps = {
        "s_f": np.full(shape, np.nan),
        "compression_ratio": np.full(shape, np.nan),
        "vmaf_neg": np.full(shape, np.nan),
    }
    for r in rows:
        i = crf_to_i[int(r["crf"])]
        j = aq_to_j[float(r["aq_strength"])]
        for key in z_maps:
            z_maps[key][i, j] = float(r[key])

    X, Y = np.meshgrid(np.array(crfs, dtype=float), np.array(aqs, dtype=float))
    # meshgrid above gives (n_aq, n_crf); transpose Z to match
    for key in z_maps:
        z_maps[key] = z_maps[key].T  # now (n_aq, n_crf) to align with X,Y

    return X, Y, z_maps


def _best_point(rows: list[dict[str, Any]], metric: str) -> Optional[dict[str, Any]]:
    valid = [r for r in rows if math.isfinite(float(r.get(metric, float("nan"))))]
    if not valid:
        return None
    return max(valid, key=lambda r: float(r[metric]))


def plot_matplotlib(
    X: np.ndarray,
    Y: np.ndarray,
    z_maps: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    out_png: Path,
    *,
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("s_f", "final score (s_f)", "plasma"),
        ("compression_ratio", "compression ratio", "viridis"),
        ("vmaf_neg", "VMAF NEG", "coolwarm"),
    ]

    fig = plt.figure(figsize=(20, 6))
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    for idx, (key, zlabel, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        Z = z_maps[key]
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        ax.set_xlabel("CRF")
        ax.set_ylabel("aq-strength")
        ax.set_zlabel(zlabel)
        ax.set_title(zlabel)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)

        best = _best_point(rows, key)
        if best is not None:
            ax.scatter(
                [float(best["crf"])],
                [float(best["aq_strength"])],
                [float(best[key])],
                color="red",
                s=60,
                depthshade=True,
                label="best",
            )
            ax.text(
                float(best["crf"]),
                float(best["aq_strength"]),
                float(best[key]),
                f"  {best['crf']}/{best['aq_strength']:.1f}",
                color="red",
                fontsize=8,
            )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_plotly_html(
    X: np.ndarray,
    Y: np.ndarray,
    z_maps: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    out_html: Path,
    *,
    title: str = "",
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [
        ("s_f", "final score (s_f)", "Plasma"),
        ("compression_ratio", "compression ratio", "Viridis"),
        ("vmaf_neg", "VMAF NEG", "RdBu"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.05,
    )

    for col, (key, zlabel, colorscale) in enumerate(panels, start=1):
        Z = z_maps[key]
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                colorbar=dict(title=zlabel, len=0.75, x=0.28 + 0.33 * (col - 1)),
                name=zlabel,
                showscale=True,
            ),
            row=1,
            col=col,
        )
        best = _best_point(rows, key)
        if best is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[float(best["crf"])],
                    y=[float(best["aq_strength"])],
                    z=[float(best[key])],
                    mode="markers+text",
                    marker=dict(color="red", size=5),
                    text=[f"best {best['crf']}/{best['aq_strength']:.1f}"],
                    textposition="top center",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=title or "CRF × aq-strength sweep",
        height=620,
        width=1800,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    for col in range(1, 4):
        scene_name = "scene" if col == 1 else f"scene{col}"
        fig.update_layout(
            {
                scene_name: dict(
                    xaxis_title="CRF",
                    yaxis_title="aq-strength",
                    zaxis_title=panels[col - 1][1],
                )
            }
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))


def plot_sweep_3d(
    *,
    work_dir: Optional[Path] = None,
    trials_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    gates_only: bool = False,
    html: bool = False,
    title: str = "",
) -> dict[str, str]:
    """Build 3× 3D surface plots; return output paths."""
    rows = load_sweep_rows(
        work_dir=work_dir,
        trials_path=trials_path,
        csv_path=csv_path,
        gates_only=gates_only,
    )
    X, Y, z_maps = build_grid(rows)

    if out_dir is None:
        if work_dir is not None:
            out_dir = work_dir
        elif trials_path is not None:
            out_dir = trials_path.parent
        else:
            out_dir = Path(".")

    out_png = out_dir / "crf_aq_3d.png"
    plot_matplotlib(X, Y, z_maps, rows, out_png, title=title)

    outputs = {"png": str(out_png)}
    if html:
        out_html = out_dir / "crf_aq_3d.html"
        plot_plotly_html(X, Y, z_maps, rows, out_html, title=title)
        outputs["html"] = str(out_html)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        default="",
        help="Sweep output dir containing trials.jsonl / results.csv",
    )
    p.add_argument("--trials", default="", help="Path to trials.jsonl")
    p.add_argument("--csv", default="", help="Path to results.csv")
    p.add_argument(
        "--out-dir",
        default="",
        help="Where to write plots (default: work-dir or trials parent)",
    )
    p.add_argument(
        "--gates-only",
        action="store_true",
        help="Plot only trials that passed encoding + VMAF delta gates",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="Also write interactive plotly HTML",
    )
    p.add_argument("--title", default="", help="Optional figure title")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir) if args.work_dir else None
    trials_path = Path(args.trials) if args.trials else None
    csv_path = Path(args.csv) if args.csv else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    outputs = plot_sweep_3d(
        work_dir=work_dir,
        trials_path=trials_path,
        csv_path=csv_path,
        out_dir=out_dir,
        gates_only=bool(args.gates_only),
        html=bool(args.html),
        title=args.title,
    )
    for kind, path in outputs.items():
        print(f"{kind:5s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
