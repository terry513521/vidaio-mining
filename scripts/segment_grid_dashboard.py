#!/usr/bin/env python3
"""Build an interactive HTML dashboard from segment CRF×AQ grid results.

Reads every ``trials.jsonl`` under a work root (new grid or legacy segment
sweep) and writes a self-contained Plotly dashboard with:

  - summary stats + best CRF/AQ per segment
  - heatmaps: s_f, VMAF NEG, compression ratio  (CRF × aq)
  - full trials table (all saved fields)

Example:
  python3 scripts/segment_grid_dashboard.py \\
    --work-dir work/segment_crf_aq_grid

  python3 scripts/segment_grid_dashboard.py \\
    --work-dir work/crf_aq_segment_sweep/d7cbca62-b96c-4370-804f-23a930ea3455 \\
    --open
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _discover_rows(work_dir: Path) -> list[dict[str, Any]]:
    """Load trials from flat or nested work layouts."""
    rows: list[dict[str, Any]] = []

    # Preferred: flattened all_trials.jsonl
    for cand in (
        work_dir / "all_trials.jsonl",
        *sorted(work_dir.glob("*/all_trials.jsonl")),
    ):
        if cand.is_file():
            for r in _load_jsonl(cand):
                if "video_stem" not in r:
                    r["video_stem"] = cand.parent.name if cand.parent != work_dir else work_dir.name
                rows.append(r)
            if rows:
                return rows

    # Per-segment trials.jsonl (legacy + new)
    files = sorted(work_dir.rglob("trials.jsonl"))
    for path in files:
        # segment_XX/trials.jsonl → parent name; video/segment_XX → video stem
        parts = path.relative_to(work_dir).parts
        video_stem = work_dir.name
        if len(parts) >= 2 and parts[0].startswith("segment_"):
            video_stem = work_dir.name
        elif len(parts) >= 2:
            video_stem = parts[0]
        for r in _load_jsonl(path):
            r.setdefault("video_stem", video_stem)
            if "segment_index" not in r:
                # infer from path segment_XX
                for part in parts:
                    if part.startswith("segment_"):
                        try:
                            r["segment_index"] = int(part.split("_")[1])
                        except ValueError:
                            pass
            rows.append(r)
    return rows


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _ok(row: dict[str, Any]) -> bool:
    v = row.get("encode_ok")
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes"}
    if v is None:
        return True
    return bool(v)


def _best(rows: list[dict[str, Any]], *, gated: bool = False) -> Optional[dict[str, Any]]:
    cand = [r for r in rows if _ok(r)]
    if gated:
        cand = [r for r in cand if r.get("gates_ok")]
    if not cand:
        return None
    return max(cand, key=lambda r: _f(r, "s_f"))


def _heatmap_payload(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    ok = [r for r in rows if _ok(r)]
    crfs = sorted({int(_f(r, "crf")) for r in ok})
    aqs = sorted({round(_f(r, "aq_strength"), 4) for r in ok})
    z = [[None for _ in crfs] for _ in aqs]
    for r in ok:
        ci = crfs.index(int(_f(r, "crf")))
        ai = aqs.index(round(_f(r, "aq_strength"), 4))
        z[ai][ci] = _f(r, metric)
    return {
        "crfs": crfs,
        "aqs": aqs,
        "z": z,
        "metric": metric,
    }


def build_dashboard(rows: list[dict[str, Any]], *, title: str, source: str) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    ok_rows = [r for r in rows if _ok(r)]
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in ok_rows:
        stem = str(r.get("video_stem") or "video")
        seg = int(r.get("segment_index") or 0)
        by_key[(stem, seg)].append(r)

    keys = sorted(by_key.keys(), key=lambda k: (k[0], k[1]))
    summary_rows = []
    for stem, seg in keys:
        group = by_key[(stem, seg)]
        best = _best(group)
        best_g = _best(group, gated=True)
        summary_rows.append(
            {
                "video": stem[:12],
                "seg": seg,
                "n": len(group),
                "best_crf": best["crf"] if best else None,
                "best_aq": best["aq_strength"] if best else None,
                "best_s_f": round(_f(best, "s_f"), 4) if best else None,
                "best_vmaf": round(_f(best, "vmaf_neg"), 2) if best else None,
                "best_ratio": round(_f(best, "compression_ratio"), 2) if best else None,
                "gated_crf": best_g["crf"] if best_g else None,
                "gated_aq": best_g["aq_strength"] if best_g else None,
                "gated_s_f": round(_f(best_g, "s_f"), 4) if best_g else None,
            }
        )

    # One figure per segment: 3 heatmaps
    figures_html: list[str] = []
    for stem, seg in keys:
        group = by_key[(stem, seg)]
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["final score (s_f)", "VMAF NEG", "compression ratio"],
            horizontal_spacing=0.08,
        )
        metrics = [
            ("s_f", "Plasma"),
            ("vmaf_neg", "RdBu"),
            ("compression_ratio", "Viridis"),
        ]
        for col, (metric, colorscale) in enumerate(metrics, start=1):
            hm = _heatmap_payload(group, metric)
            fig.add_trace(
                go.Heatmap(
                    x=hm["crfs"],
                    y=hm["aqs"],
                    z=hm["z"],
                    colorscale=colorscale,
                    colorbar=dict(title=metric, len=0.75, x=0.28 + 0.33 * (col - 1)),
                    hovertemplate="CRF=%{x}<br>aq=%{y}<br>" + metric + "=%{z:.4f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        best = _best(group)
        if best is not None:
            for col in range(1, 4):
                metric = metrics[col - 1][0]
                fig.add_trace(
                    go.Scatter(
                        x=[best["crf"]],
                        y=[best["aq_strength"]],
                        mode="markers+text",
                        marker=dict(color="red", size=10, symbol="x"),
                        text=[f"best {_f(best, metric):.3f}"],
                        textposition="top center",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_layout(
            title=f"{stem} · segment {seg} · n={len(group)} trials",
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_xaxes(title_text="CRF")
        fig.update_yaxes(title_text="aq-strength")
        figures_html.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Full trials table (cap display rows in HTML for browser; all still in JSON embed)
    table_cols = [
        "video_stem",
        "segment_index",
        "crf",
        "aq_strength",
        "vmaf_neg",
        "vmaf_base",
        "compression_rate",
        "compression_ratio",
        "s_f",
        "gates_ok",
        "reason",
    ]
    table_data = []
    for r in sorted(
        ok_rows,
        key=lambda x: (
            str(x.get("video_stem") or ""),
            int(x.get("segment_index") or 0),
            int(_f(x, "crf")),
            round(_f(x, "aq_strength"), 4),
        ),
    ):
        table_data.append({c: r.get(c) for c in table_cols})

    summary_json = json.dumps(summary_rows)
    table_json = json.dumps(table_data)

    sections = "\n".join(
        f'<section class="panel"><div class="fig">{html}</div></section>'
        for html in figures_html
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #0f1115;
    --panel: #171a21;
    --text: #e8eaed;
    --muted: #9aa0a6;
    --line: #2a2f3a;
    --accent: #7aa2f7;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  header {{
    padding: 20px 28px 12px;
    border-bottom: 1px solid var(--line);
  }}
  h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    padding: 16px 28px;
  }}
  .stat {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .stat .label {{ color: var(--muted); font-size: 12px; }}
  .stat .value {{ font-size: 22px; margin-top: 4px; color: var(--accent); }}
  .panel {{
    margin: 0 28px 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 8px 4px;
  }}
  h2 {{
    margin: 20px 28px 8px;
    font-size: 16px;
    font-weight: 600;
  }}
  table {{
    width: calc(100% - 56px);
    margin: 0 28px 28px;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    padding: 6px 8px;
    text-align: left;
    white-space: nowrap;
  }}
  th {{ color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }}
  tr:hover td {{ background: #1c212b; }}
  .table-wrap {{ max-height: 480px; overflow: auto; margin-bottom: 28px; }}
  input.filter {{
    margin: 0 28px 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--text);
    padding: 8px 10px;
    border-radius: 6px;
    width: min(420px, calc(100% - 56px));
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">Source: {source} · encode_ok trials only in heatmaps/table · red × = best s_f</div>
</header>
<div class="stats">
  <div class="stat"><div class="label">Trials (ok)</div><div class="value" id="n-ok">0</div></div>
  <div class="stat"><div class="label">Segments</div><div class="value" id="n-seg">0</div></div>
  <div class="stat"><div class="label">Videos</div><div class="value" id="n-vid">0</div></div>
  <div class="stat"><div class="label">Best s_f (any)</div><div class="value" id="best-sf">—</div></div>
</div>

<h2>Best CRF / AQ per segment</h2>
<div class="table-wrap">
<table id="summary-table">
  <thead>
    <tr>
      <th>video</th><th>seg</th><th>n</th>
      <th>best CRF</th><th>best AQ</th><th>s_f</th><th>VMAF</th><th>ratio</th>
      <th>gated CRF</th><th>gated AQ</th><th>gated s_f</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>

<h2>CRF × aq heatmaps</h2>
{sections}

<h2>All trials</h2>
<input class="filter" id="filter" placeholder="Filter table (video, seg, crf, reason…)"/>
<div class="table-wrap">
<table id="trials-table">
  <thead>
    <tr>
      <th>video</th><th>seg</th><th>CRF</th><th>AQ</th>
      <th>vmaf_neg</th><th>vmaf_base</th>
      <th>rate</th><th>ratio</th><th>s_f</th>
      <th>gates</th><th>reason</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>

<script>
const SUMMARY = {summary_json};
const TRIALS = {table_json};

document.getElementById('n-ok').textContent = String(TRIALS.length);
document.getElementById('n-seg').textContent = String(SUMMARY.length);
document.getElementById('n-vid').textContent = String(new Set(SUMMARY.map(r => r.video)).size);
const bestSf = Math.max(...SUMMARY.map(r => r.best_s_f || 0), 0);
document.getElementById('best-sf').textContent = bestSf ? bestSf.toFixed(4) : '—';

const sumBody = document.querySelector('#summary-table tbody');
for (const r of SUMMARY) {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${{r.video}}</td><td>${{r.seg}}</td><td>${{r.n}}</td>
    <td>${{r.best_crf ?? ''}}</td><td>${{r.best_aq ?? ''}}</td>
    <td>${{r.best_s_f ?? ''}}</td><td>${{r.best_vmaf ?? ''}}</td><td>${{r.best_ratio ?? ''}}</td>
    <td>${{r.gated_crf ?? ''}}</td><td>${{r.gated_aq ?? ''}}</td><td>${{r.gated_s_f ?? ''}}</td>`;
  sumBody.appendChild(tr);
}}

const trialBody = document.querySelector('#trials-table tbody');
function renderTrials(filter) {{
  trialBody.innerHTML = '';
  const q = (filter || '').toLowerCase();
  for (const r of TRIALS) {{
    const cells = [
      r.video_stem, r.segment_index, r.crf, r.aq_strength,
      r.vmaf_neg, r.vmaf_base, r.compression_rate, r.compression_ratio,
      r.s_f, r.gates_ok, r.reason
    ];
    const text = cells.join(' ').toLowerCase();
    if (q && !text.includes(q)) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = cells.map(c => {{
      let v = c;
      if (typeof c === 'number') {{
        v = Number.isInteger(c) ? String(c) : c.toFixed(4);
      }}
      if (v === null || v === undefined) v = '';
      return `<td>${{v}}</td>`;
    }}).join('');
    trialBody.appendChild(tr);
  }}
}}
renderTrials('');
document.getElementById('filter').addEventListener('input', (e) => renderTrials(e.target.value));
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Sweep root (segment_crf_aq_grid or a single-video segment sweep dir)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path (default: <work-dir>/dashboard.html)",
    )
    p.add_argument("--title", default="Segment CRF × AQ grid dashboard")
    p.add_argument("--open", action="store_true", help="Open in browser")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work_dir: Path = args.work_dir
    if not work_dir.is_dir():
        raise SystemExit(f"work-dir not found: {work_dir}")

    rows = _discover_rows(work_dir)
    if not rows:
        raise SystemExit(f"no trials.jsonl found under {work_dir}")

    out = args.out or (work_dir / "dashboard.html")
    html = build_dashboard(
        rows,
        title=args.title,
        source=str(work_dir.resolve()),
    )
    out.write_text(html, encoding="utf-8")
    n_ok = sum(1 for r in rows if _ok(r))
    print(f"rows       : {len(rows)} total, {n_ok} encode_ok")
    print(f"dashboard  : {out.resolve()}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
