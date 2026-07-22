#!/usr/bin/env python3
"""Adaptive aq-strength interval search at fixed CRF (interpolated + expand).

For each segment and CRF in [22, 38] (default), find the AQ interval where
both vmaf_neg > threshold and compression_ratio > min_ratio, using:

  Phase 1 — interpolated bracketing (start aq=0.2, 1.3; probe 2.6 when both pass)
  Phase 2 — binary search toward boundaries, or ±0.1 expand when [AQ_MIN, AQ_MAX] is valid

Trials are saved in the **same layout as** ``segment_crf_aq_grid_sweep.py``:
  segment_XX/trials.jsonl, results.csv, summary.json
  <video>/summary.json, all_trials.jsonl

Per-CRF pass interval metadata: segment_XX/crf_YY/pass_interval.json

Example:
  # Continue from an existing grid sweep (reuse encodes/VMAF, adaptive search only):
  python3 segment_crf_aq_adaptive_sweep.py \\
    --segmented-dir "../segmented videos" \\
    --raw-dir "../raw videos" \\
    --videos 0317ca2b-0f03-4bfe-a236-3c790534aa5a \\
    --workers 6 --resume

  # Re-run adaptive search for CRFs not yet finished (default reuses work/segment_crf_aq_grid):
  python3 segment_crf_aq_adaptive_sweep.py --resume --limit 1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from interp_search import parse_x265_params, propose_feature_x265_params
from segment_crf_aq_grid_sweep import (
    _append_csv_row,
    _append_jsonl,
    _best_row,
    _discover_video_dirs,
    _find_raw_video,
    _load_rows,
    _parse_segments,
    _run_trial,
    _source_segment_packet_bytes,
)
from test_crf_aq_sweep import DEFAULT_BASE_PARAMS, _completed_keys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_REUSE_ROOTS = (
    ROOT / "work" / "segment_crf_aq_grid",
    ROOT / "work" / "crf_aq_segment_sweep",
)

AQ_MIN = 0.2
AQ_MAX = 2.6
AQ_EXPAND_STEP = 0.1
AQ_START_LOW = 0.2
AQ_START_MID = 1.3
MAX_AQ_PROBES = 32

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


DEFAULT_FEATURES_DIR = ROOT / "video_features"


def _load_video_features(stem: str, features_dir: Path) -> Optional[dict[str, Any]]:
    path = features_dir / f"{stem}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_feat_dict_for_params(
    seg_row: dict[str, Any],
    *,
    meta: dict[str, Any],
    global_: dict[str, Any],
) -> dict[str, Any]:
    """Map per-segment ML features → propose_feature_x265_params input."""
    from feature_extractor import (
        _EDGE_DENSITY_MID,
        _MOTION_P90_MID,
        _NOISE_LEVEL_MID,
        _TEXTURE_MID,
        soft_level,
    )

    fps = float(meta.get("fps") or global_.get("fps") or 30.0)
    motion = float(seg_row.get("motion_p90") or seg_row.get("motion") or 0.0)
    texture = float(seg_row.get("texture") or 0.0)
    noise = float(seg_row.get("noise") or 0.0)
    edge = float(seg_row.get("edge") or 0.0)
    difficulty = float(seg_row.get("difficulty") or 0.0)
    duration = float(seg_row.get("duration") or max(0.1, 1.0))
    luma = float(seg_row.get("luma_mean") or global_.get("mean_luma") or 0.5)
    if luma <= 1.5:
        luma *= 255.0
    return {
        "motion_level": soft_level(motion, _MOTION_P90_MID),
        "texture_level": soft_level(texture, _TEXTURE_MID),
        "noise_level_norm": soft_level(noise, _NOISE_LEVEL_MID),
        "edge_level": soft_level(edge, _EDGE_DENSITY_MID),
        "cut_level": 0.0,
        "flatness": float(seg_row.get("flatness") or global_.get("mean_flatness") or 0.0),
        "entropy": float(seg_row.get("entropy") or global_.get("mean_entropy") or 0.5),
        "luma_mean": luma,
        "worst_difficulty": difficulty,
        "hard_fraction": 1.0 if difficulty >= 0.45 else 0.0,
        "volatility": 0.0,
        "segment_count": 1.0,
        "duration": max(duration, 0.1),
        "fps": fps,
        "texture": texture,
        "motion_p90": motion,
        "noise": noise,
        "edge": edge,
    }


def _resolve_segment_base_params(
    stem: str,
    seg: dict[str, Any],
    *,
    use_feature_params: bool,
    features_dir: Path,
    fixed_base_params: dict[str, str],
) -> tuple[dict[str, str], str]:
    """Return (x265 params without crf/aq-strength, source label)."""
    if not use_feature_params:
        return dict(fixed_base_params), "fixed"

    feat_json = _load_video_features(stem, features_dir)
    if feat_json is None:
        return dict(fixed_base_params), "fixed_fallback"

    meta = feat_json.get("meta") or {}
    global_ = feat_json.get("global") or {}
    seg_idx = int(seg["index"])
    seg_row: Optional[dict[str, Any]] = None
    for row in feat_json.get("segments") or []:
        if isinstance(row, dict) and int(row.get("index", -1)) == seg_idx:
            seg_row = row
            break
    if seg_row is None:
        return dict(fixed_base_params), "fixed_fallback"

    feat_dict = _segment_feat_dict_for_params(seg_row, meta=meta, global_=global_)
    fps = float(feat_dict.get("fps") or 30.0)
    params, _reasons = propose_feature_x265_params(feat_dict, fps=fps, quality_pack=False)
    params.pop("crf", None)
    params.pop("aq-strength", None)
    return {k: str(v) for k, v in params.items()}, "features"


def _trial_key(crf: int, aq: float) -> tuple[int, float]:
    return (int(crf), _round_aq(aq))


def _build_trial_cache(paths: list[Path]) -> dict[tuple[int, float], dict[str, Any]]:
    """Merge trials from one or more trials.jsonl files; last row wins per (crf, aq)."""
    cache: dict[tuple[int, float], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _load_rows(path):
            key = _trial_key(int(float(row.get("crf") or 0)), float(row.get("aq_strength") or 0))
            cache[key] = row
    return cache


def _trial_sources_for_segment(
    *,
    trials_path: Path,
    video_stem: str,
    segment_index: int,
    reuse_roots: list[Path],
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for root in reuse_roots:
        p = root / video_stem / f"segment_{int(segment_index):02d}" / "trials.jsonl"
        key = str(p.resolve())
        if p.is_file() and key not in seen:
            paths.append(p)
            seen.add(key)
    local_key = str(trials_path.resolve())
    if trials_path.is_file() and local_key not in seen:
        paths.append(trials_path)
    elif not trials_path.is_file():
        paths.append(trials_path)
    return paths


def _logged_keys(trials_path: Path) -> set[tuple[int, float]]:
    """All (crf, aq) already present in trials.jsonl (any encode outcome)."""
    keys: set[tuple[int, float]] = set()
    for row in _load_rows(trials_path):
        keys.add(
            _trial_key(int(float(row.get("crf") or 0)), float(row.get("aq_strength") or 0))
        )
    return keys


def _sync_trial_cache_to_disk(
    trials_path: Path,
    csv_path: Path,
    cache: dict[tuple[int, float], dict[str, Any]],
) -> int:
    """Append any cached (crf, aq) rows missing from local trials.jsonl (all encode outcomes)."""
    logged = _logged_keys(trials_path)
    n = 0
    for key in sorted(cache.keys()):
        if key in logged:
            continue
        row = cache[key]
        _append_jsonl(trials_path, row)
        _append_csv_row(csv_path, row)
        logged.add(key)
        n += 1
    return n


def _load_pass_interval(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _round_aq(aq: float) -> float:
    return round(float(aq), 1)


def _clamp_aq(aq: float) -> float:
    return _round_aq(max(AQ_MIN, min(AQ_MAX, float(aq))))


def _bisect_gap_reached(lo: float, hi: float, *, step: float = AQ_EXPAND_STEP) -> bool:
    return float(hi) - float(lo) <= float(step) + 1e-9


def _untested_mid(
    lo: float,
    hi: float,
    tested: dict[float, dict[str, Any]],
    *,
    step: float = AQ_EXPAND_STEP,
) -> Optional[float]:
    """Next AQ probe by bisection on the AQ grid between lo and hi (exclusive)."""
    lo_f, hi_f = float(lo), float(hi)
    if _bisect_gap_reached(lo_f, hi_f, step=step):
        return None
    mid = _clamp_aq((lo_f + hi_f) / 2.0)
    if mid > lo_f + 1e-9 and mid < hi_f - 1e-9 and mid not in tested:
        return mid
    untested: list[float] = []
    v = _round_aq(lo_f + step)
    while v < hi_f - 1e-9:
        if v not in tested:
            untested.append(v)
        v = _round_aq(v + step)
    if not untested:
        return None
    return untested[len(untested) // 2]


def _trial_ok(row: dict[str, Any]) -> bool:
    v = row.get("encode_ok")
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes"}
    if v is None:
        return False
    return bool(v)


def _metrics(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row.get("vmaf_neg") or 0.0),
        float(row.get("compression_ratio") or 0.0),
    )


def _is_valid(
    row: dict[str, Any],
    *,
    vmaf_thr: float,
    ratio_thr: float,
) -> bool:
    if not _trial_ok(row):
        return False
    vmaf, ratio = _metrics(row)
    return vmaf > vmaf_thr and ratio > ratio_thr


def _trial_state(
    row: dict[str, Any],
    *,
    vmaf_thr: float,
    ratio_thr: float,
) -> dict[str, Any]:
    vmaf, ratio = _metrics(row)
    valid = _is_valid(row, vmaf_thr=vmaf_thr, ratio_thr=ratio_thr)
    reason: list[str] = []
    if not _trial_ok(row):
        reason.append("encode_failed")
    else:
        if vmaf <= vmaf_thr:
            reason.append(f"vmaf<={vmaf_thr:g}")
        if ratio <= ratio_thr:
            reason.append(f"ratio<={ratio_thr:g}")
    return {
        "aq_strength": float(row.get("aq_strength") or 0.0),
        "vmaf_neg": vmaf,
        "vmaf_base": row.get("vmaf_base"),
        "compression_rate": float(row.get("compression_rate") or 0.0),
        "compression_ratio": ratio,
        "s_f": float(row.get("s_f") or 0.0),
        "gates_ok": bool(row.get("gates_ok")),
        "valid": valid,
        "invalid_reason": ",".join(reason) if reason else None,
    }


def _interpolate_aq(
    aq_lo: float,
    metric_lo: float,
    aq_hi: float,
    metric_hi: float,
    target: float,
) -> Optional[float]:
    if abs(metric_hi - metric_lo) < 1e-9:
        return None
    aq = aq_lo + (target - metric_lo) * (aq_hi - aq_lo) / (metric_hi - metric_lo)
    return _clamp_aq(aq)


def _sorted_tested(tested: dict[float, dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    return sorted(tested.items(), key=lambda t: t[0])


def _valid_aqs(
    tested: dict[float, dict[str, Any]],
    *,
    vmaf_thr: float,
    ratio_thr: float,
) -> list[float]:
    return sorted(
        aq
        for aq, row in tested.items()
        if _is_valid(row, vmaf_thr=vmaf_thr, ratio_thr=ratio_thr)
    )


class AqIntervalSearcher:
    """Plan aq probes until pass interval is bracketed and expanded."""

    def __init__(
        self,
        *,
        vmaf_thr: float,
        ratio_thr: float,
        expand_step: float = AQ_EXPAND_STEP,
        max_probes: int = MAX_AQ_PROBES,
    ) -> None:
        self.vmaf_thr = float(vmaf_thr)
        self.ratio_thr = float(ratio_thr)
        self.expand_step = float(expand_step)
        self.max_probes = int(max_probes)
        self.tested: dict[float, dict[str, Any]] = {}
        self.phase = "seed"
        self.seed_queue: list[float] = [_round_aq(AQ_START_LOW), _round_aq(AQ_START_MID)]
        self.bracket_lo: Optional[tuple[float, float]] = None  # (invalid, valid)
        self.bracket_hi: Optional[tuple[float, float]] = None  # (valid, invalid)
        self.pass_lo: Optional[float] = None
        self.pass_hi: Optional[float] = None
        self.expand_lo_done = False
        self.expand_hi_done = False
        self.empty_interval = False
        self._checked_top = False
        self.expand_mode = "bisect"  # "bisect" or "linear" (±expand_step when full range valid)
        self.expand_lo_range: Optional[tuple[float, float]] = None
        self.expand_hi_range: Optional[tuple[float, float]] = None

    def ingest(self, row: dict[str, Any]) -> None:
        if not _trial_ok(row):
            aq = _round_aq(float(row.get("aq_strength") or 0.0))
            self.tested[aq] = row
            return
        aq = _round_aq(float(row.get("aq_strength") or 0.0))
        self.tested[aq] = row

    def _row(self, aq: float) -> Optional[dict[str, Any]]:
        return self.tested.get(_round_aq(aq))

    def _valid(self, aq: float) -> Optional[bool]:
        row = self._row(aq)
        if row is None:
            return None
        return _is_valid(row, vmaf_thr=self.vmaf_thr, ratio_thr=self.ratio_thr)

    def _update_pass_bounds_from_valid(self) -> None:
        valids = _valid_aqs(
            self.tested, vmaf_thr=self.vmaf_thr, ratio_thr=self.ratio_thr
        )
        if valids:
            self.pass_lo = valids[0]
            self.pass_hi = valids[-1]

    def _is_full_pass_range(self) -> bool:
        return (
            self.pass_lo is not None
            and self.pass_hi is not None
            and self.pass_lo <= AQ_MIN + 1e-9
            and self.pass_hi >= AQ_MAX - 1e-9
        )

    def _setup_brackets_after_seeds(self) -> None:
        v02 = self._valid(AQ_START_LOW)
        v13 = self._valid(AQ_START_MID)
        if v02 is None or v13 is None:
            return

        if v02 and v13:
            self.pass_lo = AQ_START_LOW
            self.pass_hi = AQ_START_MID
            self.phase = "probe_top_valid"
            return

        if not v02 and v13:
            self.phase = "bracket_lo"
            self.bracket_lo = (AQ_START_LOW, AQ_START_MID)
            return

        if v02 and not v13:
            self.phase = "bracket_hi"
            self.bracket_hi = (AQ_START_LOW, AQ_START_MID)
            return

        # both invalid
        self.phase = "probe_top"
        if not self._checked_top:
            self._checked_top = True
            if _round_aq(AQ_MAX) not in self.tested:
                self.seed_queue.append(_round_aq(AQ_MAX))

    def _after_probe_top(self) -> None:
        v_top = self._valid(AQ_MAX)
        if v_top is None:
            return
        if not v_top:
            self.empty_interval = True
            self.phase = "done"
            return
        self.phase = "bracket_lo"
        self.bracket_lo = (AQ_START_MID, AQ_MAX)
        if self.pass_hi is None:
            self.pass_hi = AQ_MAX

    def _after_probe_top_valid(self) -> None:
        """Both seeds valid: after probing AQ_MAX, bisect or linear-expand."""
        v_top = self._valid(AQ_MAX)
        if v_top is None:
            return
        self._update_pass_bounds_from_valid()
        if v_top:
            self.pass_lo = AQ_MIN
            self.pass_hi = AQ_MAX
            self.expand_mode = "linear"
        else:
            self.pass_lo = AQ_START_LOW
            self.pass_hi = AQ_START_MID
            self.expand_mode = "bisect"
        self._begin_expand()

    def _interpolate_next(
        self,
        bracket: tuple[float, float],
        *,
        target_metric: str,
        target_value: float,
    ) -> Optional[float]:
        aq_a, aq_b = bracket
        row_a = self._row(aq_a)
        row_b = self._row(aq_b)
        if row_a is None or row_b is None:
            return _clamp_aq((aq_a + aq_b) / 2.0)
        if target_metric == "vmaf":
            m_a, m_b = _metrics(row_a)[0], _metrics(row_b)[0]
        else:
            m_a, m_b = _metrics(row_a)[1], _metrics(row_b)[1]
        cand = _interpolate_aq(aq_a, m_a, aq_b, m_b, target_value)
        if cand is None:
            return _clamp_aq((aq_a + aq_b) / 2.0)
        return cand

    def _bracket_resolved(
        self,
        bracket: tuple[float, float],
        *,
        lo_valid: bool,
    ) -> bool:
        """One side valid, one invalid at endpoints."""
        aq_a, aq_b = bracket
        v_a = self._valid(aq_a)
        v_b = self._valid(aq_b)
        if v_a is None or v_b is None:
            return False
        if lo_valid:
            return (not v_a) and v_b
        return v_a and (not v_b)

    def next_aq(self) -> Optional[float]:
        if len(self.tested) >= self.max_probes:
            self.phase = "done"
            return None
        if self.phase == "done" or self.empty_interval:
            return None

        if self.phase == "seed":
            while self.seed_queue:
                aq = _round_aq(self.seed_queue.pop(0))
                if aq not in self.tested:
                    return aq
            self._setup_brackets_after_seeds()
            return self.next_aq()

        if self.phase == "probe_top":
            self._after_probe_top()
            return self.next_aq()

        if self.phase == "probe_top_valid":
            if _round_aq(AQ_MAX) not in self.tested:
                return _round_aq(AQ_MAX)
            self._after_probe_top_valid()
            return self.next_aq()

        if self.phase == "bracket_lo" and self.bracket_lo is not None:
            aq_a, aq_b = self.bracket_lo
            if self._bracket_resolved(self.bracket_lo, lo_valid=True):
                self.pass_lo = aq_b
                self.phase = "bracket_hi" if self.bracket_hi else "expand"
                if self.phase == "expand":
                    self._begin_expand()
                return self.next_aq()
            cand = self._interpolate_next(
                self.bracket_lo,
                target_metric="vmaf",
                target_value=self.vmaf_thr,
            )
            if cand is not None and cand in self.tested:
                self.pass_lo = aq_b if self._valid(aq_b) else aq_a
                self.phase = "bracket_hi" if self.bracket_hi else "expand"
                if self.phase == "expand":
                    self._begin_expand()
                return self.next_aq()
            if cand is not None and cand not in self.tested:
                return cand
            bisect = _untested_mid(aq_a, aq_b, self.tested, step=self.expand_step)
            if bisect is not None:
                return bisect
            self.phase = "expand"
            self._begin_expand()
            return self.next_aq()

        if self.phase == "bracket_hi" and self.bracket_hi is not None:
            aq_a, aq_b = self.bracket_hi
            v_a = self._valid(aq_a)
            v_b = self._valid(aq_b)
            if v_a and v_b:
                # Both ends pass — extrapolate/interpolate ratio crossing above aq_b.
                row_a = self._row(aq_a)
                row_b = self._row(aq_b)
                if row_a and row_b:
                    _, r_a = _metrics(row_a)
                    _, r_b = _metrics(row_b)
                    cand = _interpolate_aq(aq_a, r_a, aq_b, r_b, self.ratio_thr)
                    if cand is not None:
                        if cand <= aq_b + 1e-9 and cand not in self.tested:
                            return cand
                        if cand > aq_b:
                            nxt = _clamp_aq(cand)
                            if nxt not in self.tested:
                                self.bracket_hi = (aq_b, min(AQ_MAX, _clamp_aq(aq_b + 1.1)))
                                return nxt
                            if not self._valid(AQ_MAX) and AQ_MAX not in self.tested:
                                return AQ_MAX
                self.pass_hi = aq_b
                self.phase = "expand"
                self._begin_expand()
                return self.next_aq()

            if self._bracket_resolved(self.bracket_hi, lo_valid=False):
                self.pass_hi = aq_a
                self.phase = "expand"
                self._begin_expand()
                return self.next_aq()
            cand = self._interpolate_next(
                self.bracket_hi,
                target_metric="ratio",
                target_value=self.ratio_thr,
            )
            if cand is not None and cand in self.tested:
                self.pass_hi = aq_a if self._valid(aq_a) else aq_b
                self.phase = "expand"
                self._begin_expand()
                return self.next_aq()
            if cand is not None and cand not in self.tested:
                return cand
            bisect = _untested_mid(aq_a, aq_b, self.tested, step=self.expand_step)
            if bisect is not None:
                return bisect
            self.phase = "expand"
            self._begin_expand()
            return self.next_aq()

        if self.phase == "expand":
            return self._next_expand()

        return None

    def _begin_expand(self) -> None:
        self._update_pass_bounds_from_valid()
        if self.pass_lo is None and self.pass_hi is None:
            self.empty_interval = True
            self.phase = "done"
            return
        if self._is_full_pass_range():
            self.expand_mode = "linear"
        self.phase = "expand"
        self.expand_lo_done = False
        self.expand_hi_done = False
        if self.expand_mode == "linear":
            self.expand_lo_range = None
            self.expand_hi_range = None
            return
        if self.pass_lo is not None and self.pass_lo > AQ_MIN + 1e-9:
            self.expand_lo_range = (AQ_MIN, float(self.pass_lo))
        else:
            self.expand_lo_done = True
            self.expand_lo_range = None
        if self.pass_hi is not None and self.pass_hi < AQ_MAX - 1e-9:
            self.expand_hi_range = (float(self.pass_hi), AQ_MAX)
        else:
            self.expand_hi_done = True
            self.expand_hi_range = None

    def _next_expand(self) -> Optional[float]:
        if self.expand_mode == "linear":
            return self._next_expand_linear()
        return self._next_expand_bisect()

    def _next_expand_linear(self) -> Optional[float]:
        """±expand_step outside confirmed pass bounds (used when [AQ_MIN, AQ_MAX] is valid)."""
        self._update_pass_bounds_from_valid()
        if self.pass_lo is None or self.pass_hi is None:
            self.phase = "done"
            return None

        if not self.expand_lo_done:
            below = _clamp_aq(self.pass_lo - self.expand_step)
            if below < self.pass_lo - 1e-9 and below not in self.tested:
                return below
            self.expand_lo_done = True

        if not self.expand_hi_done:
            above = _clamp_aq(self.pass_hi + self.expand_step)
            if above > self.pass_hi + 1e-9 and above not in self.tested:
                return above
            self.expand_hi_done = True

        self.phase = "done"
        return None

    def _next_expand_bisect(self) -> Optional[float]:
        self._update_pass_bounds_from_valid()
        if self.pass_lo is None or self.pass_hi is None:
            self.phase = "done"
            return None

        if not self.expand_lo_done and self.expand_lo_range is not None:
            lo, hi = self.expand_lo_range
            cand = _untested_mid(lo, hi, self.tested, step=self.expand_step)
            if cand is not None:
                return cand
            self.expand_lo_done = True

        if not self.expand_hi_done and self.expand_hi_range is not None:
            lo, hi = self.expand_hi_range
            cand = _untested_mid(lo, hi, self.tested, step=self.expand_step)
            if cand is not None:
                return cand
            self.expand_hi_done = True

        self.phase = "done"
        return None

    def apply_expand_result(self, aq: float) -> None:
        """After an expand probe: update bounds (bisect or linear)."""
        if self.expand_mode == "linear":
            self._apply_expand_result_linear(aq)
        else:
            self._apply_expand_result_bisect(aq)

    def _apply_expand_result_linear(self, aq: float) -> None:
        aq = _round_aq(aq)
        if self._valid(aq):
            if self.pass_lo is not None and aq < self.pass_lo:
                self.pass_lo = aq
                self.expand_lo_done = False
                return
            if self.pass_hi is not None and aq > self.pass_hi:
                self.pass_hi = aq
                self.expand_hi_done = False
                return
        if self.pass_lo is not None and aq < self.pass_lo:
            self.expand_lo_done = True
        if self.pass_hi is not None and aq > self.pass_hi:
            self.expand_hi_done = True

    def _apply_expand_result_bisect(self, aq: float) -> None:
        """Shrink binary-search range toward threshold boundary."""
        aq = _round_aq(aq)
        valid = self._valid(aq)

        if (
            self.expand_lo_range is not None
            and not self.expand_lo_done
            and self.expand_lo_range[0] - 1e-9 <= aq <= self.expand_lo_range[1] + 1e-9
        ):
            lo, hi = self.expand_lo_range
            if valid:
                self.expand_lo_range = (lo, aq)
                self.pass_lo = aq
            else:
                self.expand_lo_range = (aq, hi)
            if _bisect_gap_reached(*self.expand_lo_range, step=self.expand_step):
                self.expand_lo_done = True
            return

        if (
            self.expand_hi_range is not None
            and not self.expand_hi_done
            and self.expand_hi_range[0] - 1e-9 <= aq <= self.expand_hi_range[1] + 1e-9
        ):
            lo, hi = self.expand_hi_range
            if valid:
                self.expand_hi_range = (aq, hi)
                self.pass_hi = aq
            else:
                self.expand_hi_range = (lo, aq)
            if _bisect_gap_reached(*self.expand_hi_range, step=self.expand_step):
                self.expand_hi_done = True
            return

    def result(self) -> dict[str, Any]:
        self._update_pass_bounds_from_valid()
        trials = [
            _trial_state(
                row,
                vmaf_thr=self.vmaf_thr,
                ratio_thr=self.ratio_thr,
            )
            for _aq, row in _sorted_tested(self.tested)
        ]
        valid_trials = [t for t in trials if t["valid"]]
        if self.empty_interval or not valid_trials:
            interval = None
        else:
            lo = self.pass_lo if self.pass_lo is not None else valid_trials[0]["aq_strength"]
            hi = self.pass_hi if self.pass_hi is not None else valid_trials[-1]["aq_strength"]
            interval = {"lo": float(lo), "hi": float(hi)}

        best_sf: Optional[dict[str, Any]] = None
        if valid_trials:
            best_sf = max(valid_trials, key=lambda t: float(t["s_f"]))

        return {
            "vmaf_threshold": self.vmaf_thr,
            "min_compression_ratio": self.ratio_thr,
            "pass_interval": interval,
            "n_probes": len(self.tested),
            "n_valid": len(valid_trials),
            "empty": self.empty_interval or not valid_trials,
            "best_valid_s_f": best_sf,
            "trials": trials,
        }


def _encode_aq(
    *,
    crf: int,
    aq: float,
    seg: dict[str, Any],
    video_stem: str,
    trials_path: Path,
    encodes_dir: Path,
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
    trial_cache: dict[tuple[int, float], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Run encode+score or return cached trial. Second value: True if bypassed."""
    key = _trial_key(crf, aq)
    cached = trial_cache.get(key)
    if cached is not None and _trial_ok(cached):
        return cached, True

    out_path = encodes_dir / f"crf{crf}_aq{aq:.1f}.mp4"
    row = _run_trial(
        video_stem=video_stem,
        seg=seg,
        crf=int(crf),
        aq=float(aq),
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
        source_segment_bytes=source_segment_bytes,
    )
    payload = asdict(row)
    _append_jsonl(trials_path, payload)
    _append_csv_row(trials_path.parent / "results.csv", payload)
    trial_cache[key] = payload
    return payload, False


def _run_adaptive_crf(
    *,
    crf: int,
    seg: dict[str, Any],
    video_stem: str,
    trials_path: Path,
    encodes_dir: Path,
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
    trial_cache: dict[tuple[int, float], dict[str, Any]],
    vmaf_thr: float,
    ratio_thr: float,
    expand_step: float,
    max_probes: int,
) -> tuple[dict[str, Any], int, int]:
    crf_dir = trials_path.parent / f"crf_{int(crf):02d}"
    crf_dir.mkdir(parents=True, exist_ok=True)

    searcher = AqIntervalSearcher(
        vmaf_thr=vmaf_thr,
        ratio_thr=ratio_thr,
        expand_step=expand_step,
        max_probes=max_probes,
    )
    for row in trial_cache.values():
        if int(float(row.get("crf") or 0)) != int(crf):
            continue
        searcher.ingest(row)

    n_new = 0
    n_bypass = 0
    while True:
        aq = searcher.next_aq()
        if aq is None:
            break
        payload, bypassed = _encode_aq(
            crf=int(crf),
            aq=float(aq),
            seg=seg,
            video_stem=video_stem,
            trials_path=trials_path,
            encodes_dir=encodes_dir,
            base_params=base_params,
            preset=preset,
            profile=profile,
            vmaf_threshold=vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=vmaf_n_subsample,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            keep_encode=keep_encode,
            source_segment_bytes=source_segment_bytes,
            trial_cache=trial_cache,
        )
        if bypassed:
            n_bypass += 1
        else:
            n_new += 1
        phase_before = searcher.phase
        searcher.ingest(payload)
        if phase_before == "expand":
            searcher.apply_expand_result(aq)
        st = _trial_state(payload, vmaf_thr=vmaf_thr, ratio_thr=ratio_thr)
        tag = "cache" if bypassed else "encode"
        print(
            f"  [{video_stem[:8]}] seg={seg['index']} CRF{crf} aq={aq:.1f}  "
            f"neg={st['vmaf_neg']:.2f} ratio={st['compression_ratio']:.2f}x  "
            f"s_f={st['s_f']:.4f} valid={st['valid']}  "
            f"phase={searcher.phase} probes={len(searcher.tested)}  [{tag}]",
            flush=True,
        )

    summary = searcher.result()
    summary["crf"] = int(crf)
    summary["segment_index"] = int(seg["index"])
    summary["n_bypassed"] = n_bypass
    (crf_dir / "pass_interval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, n_new, n_bypass


def _process_one_segment(
    seg: dict[str, Any],
    *,
    stem: str,
    video_work: Path,
    src_bytes: dict[int, int],
    crfs: list[int],
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
    vmaf_thr: float,
    ratio_thr: float,
    expand_step: float,
    max_probes: int,
    reuse_roots: list[Path],
    use_feature_params: bool,
    features_dir: Path,
) -> dict[str, Any]:
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

    seg_base_params, param_src = _resolve_segment_base_params(
        stem,
        seg,
        use_feature_params=use_feature_params,
        features_dir=features_dir,
        fixed_base_params=base_params,
    )
    if param_src == "features":
        print(
            f"[{stem}] seg={seg['index']} x265 params: feature-derived "
            f"(aq-mode={seg_base_params.get('aq-mode')}, "
            f"ref={seg_base_params.get('ref')}, bframes={seg_base_params.get('bframes')})",
            flush=True,
        )
    elif use_feature_params and param_src == "fixed_fallback":
        print(
            f"[{stem}] seg={seg['index']} x265 params: fixed fallback "
            f"(no segment features in {features_dir.name})",
            flush=True,
        )

    sources = _trial_sources_for_segment(
        trials_path=trials_path,
        video_stem=stem,
        segment_index=int(seg["index"]),
        reuse_roots=reuse_roots,
    )
    trial_cache = _build_trial_cache(sources)
    if trial_cache:
        print(
            f"[{stem}] seg={seg['index']} trial cache: {len(trial_cache)} "
            f"(crf,aq) from {len(sources)} source(s)",
            flush=True,
        )

    crf_intervals: list[dict[str, Any]] = []
    n_new = 0
    n_bypass = 0

    for crf in crfs:
        pass_interval_path = seg_dir / f"crf_{int(crf):02d}" / "pass_interval.json"
        if resume:
            existing = _load_pass_interval(pass_interval_path)
            if existing is not None:
                print(
                    f"[{stem}] seg={seg['index']} CRF{crf} resume: "
                    f"skip search (existing {pass_interval_path.name})",
                    flush=True,
                )
                crf_intervals.append(existing)
                continue

        print(
            f"[{stem}] seg={seg['index']} CRF={crf} adaptive aq …",
            flush=True,
        )
        interval_summary, crf_new, crf_bypass = _run_adaptive_crf(
            crf=int(crf),
            seg=seg,
            video_stem=stem,
            trials_path=trials_path,
            encodes_dir=encodes_dir,
            base_params=seg_base_params,
            preset=preset,
            profile=profile,
            vmaf_threshold=vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=vmaf_n_subsample,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            keep_encode=keep_encode,
            source_segment_bytes=int(src_bytes.get(int(seg["index"]), 0)),
            trial_cache=trial_cache,
            vmaf_thr=vmaf_thr,
            ratio_thr=ratio_thr,
            expand_step=expand_step,
            max_probes=max_probes,
        )
        n_new += crf_new
        n_bypass += crf_bypass
        crf_intervals.append(interval_summary)
        pi = interval_summary.get("pass_interval")
        if pi:
            print(
                f"[{stem}] seg={seg['index']} CRF{crf} pass_interval "
                f"[{pi['lo']:.1f}, {pi['hi']:.1f}] "
                f"probes={interval_summary.get('n_probes')}",
                flush=True,
            )
        elif interval_summary.get("empty"):
            print(
                f"[{stem}] seg={seg['index']} CRF{crf} — no valid aq interval",
                flush=True,
            )

    n_synced = _sync_trial_cache_to_disk(trials_path, csv_path, trial_cache)
    if n_synced:
        print(
            f"[{stem}] seg={seg['index']} synced {n_synced} trials to "
            f"{trials_path.name} (includes below-threshold / reused)",
            flush=True,
        )

    rows = _load_rows(trials_path)
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
        "search_method": "adaptive_aq_interval",
        "pass_intervals_by_crf": {
            str(c["crf"]): c.get("pass_interval") for c in crf_intervals
        },
    }
    (seg_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if best is not None:
        print(
            f"[{stem}] seg[{seg['index']}] best: "
            f"crf={best['crf']} aq={best['aq_strength']} "
            f"s_f={best['s_f']:.4f} vmaf_neg={best['vmaf_neg']:.2f} "
            f"ratio={best['compression_ratio']:.2f}x",
            flush=True,
        )

    return {
        "summary": summary,
        "rows": rows,
        "n_new": n_new,
        "n_bypass": n_bypass,
    }


def _process_one_video(
    video_dir: Path,
    *,
    raw_dir: Path,
    work_root: Path,
    crfs: list[int],
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
    vmaf_thr: float,
    ratio_thr: float,
    expand_step: float,
    max_probes: int,
    reuse_roots: list[Path],
    use_feature_params: bool,
    features_dir: Path,
    segment_workers: int,
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

    t0 = time.monotonic()
    total_new = 0
    total_bypass = 0
    segment_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    seg_workers = max(1, min(int(segment_workers), len(segs)))
    print(
        f"[{stem}] processing {len(segs)} segment(s) with {seg_workers} parallel worker(s) …",
        flush=True,
    )

    def _run_seg(seg: dict[str, Any]) -> dict[str, Any]:
        return _process_one_segment(
            seg,
            stem=stem,
            video_work=video_work,
            src_bytes=src_bytes,
            crfs=crfs,
            base_params=base_params,
            preset=preset,
            profile=profile,
            vmaf_threshold=vmaf_threshold,
            vmaf_n_threads=vmaf_n_threads,
            vmaf_n_subsample=vmaf_n_subsample,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            keep_encode=keep_encode,
            resume=resume,
            force=force,
            vmaf_thr=vmaf_thr,
            ratio_thr=ratio_thr,
            expand_step=expand_step,
            max_probes=max_probes,
            reuse_roots=reuse_roots,
            use_feature_params=use_feature_params,
            features_dir=features_dir,
        )

    if seg_workers == 1:
        seg_results = [_run_seg(seg) for seg in segs]
    else:
        with ThreadPoolExecutor(max_workers=seg_workers) as pool:
            seg_results = list(pool.map(_run_seg, segs))

    for out in seg_results:
        total_new += int(out["n_new"])
        total_bypass += int(out["n_bypass"])
        segment_summaries.append(out["summary"])
        all_rows.extend(out["rows"])

    segment_summaries.sort(key=lambda s: int(s["segment_index"]))

    video_summary = {
        "video_stem": stem,
        "raw_video": str(raw),
        "segmented_dir": str(video_dir),
        "work_dir": str(video_work),
        "n_segments": len(segs),
        "search_method": "adaptive_aq_interval",
        "crf_range": [crfs[0], crfs[-1]],
        "aq_range": [AQ_MIN, AQ_MAX],
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
        "n_new_trials": total_new,
        "n_bypassed_trials": total_bypass,
        "work_dir": str(video_work),
        "wall_sec": time.monotonic() - t0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmented-dir", type=Path, default=WORKSPACE / "segmented videos")
    p.add_argument("--raw-dir", type=Path, default=WORKSPACE / "raw videos")
    p.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "work" / "segment_crf_aq_adaptive",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--videos", default="")
    p.add_argument("--crf-min", type=int, default=22)
    p.add_argument("--crf-max", type=int, default=38)
    p.add_argument("--params", default="", help="Fixed libx265 params (overrides --feature-params)")
    p.add_argument(
        "--feature-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per-segment x265 knobs from video_features/ (default: on; CRF/aq still swept)",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory with <video_stem>.json feature files",
    )
    p.add_argument("--preset", "-p", default="fast", choices=_X265_PRESETS)
    p.add_argument("--profile", default="main")
    p.add_argument("--vmaf-threshold", type=int, default=85, choices=[85, 89, 93])
    p.add_argument(
        "--valid-vmaf",
        type=float,
        default=None,
        help="Pass threshold for vmaf_neg (default: same as --vmaf-threshold)",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=1.25,
        help="Pass threshold for compression_ratio (default: 1.25)",
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--vmaf-n-subsample", type=int, default=1)
    p.add_argument("--vmaf-n-threads", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel segments per video (encode+score threads; default 6)",
    )
    p.add_argument(
        "--video-workers",
        type=int,
        default=1,
        help="Videos in parallel (default 1 = one video at a time)",
    )
    p.add_argument(
        "--reuse-dirs",
        default="",
        help=(
            "Comma-separated extra work roots to reuse trials.jsonl from "
            f"(default: {DEFAULT_REUSE_ROOTS[0].name})"
        ),
    )
    p.add_argument(
        "--no-reuse-grid",
        action="store_true",
        help="Do not load prior trials from grid sweep work dirs",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse prior trials (grid + local), skip CRFs that already have "
            "pass_interval.json; all probed trials are saved to trials.jsonl "
            "(including below VMAF/ratio thresholds)"
        ),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-encodes", action="store_true")
    p.add_argument(
        "--expand-step",
        type=float,
        default=AQ_EXPAND_STEP,
        help="Minimum AQ gap to stop binary search (default 0.1)",
    )
    p.add_argument("--max-aq-probes", type=int, default=MAX_AQ_PROBES)
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
        raise SystemExit(f"no segment folders under {segmented_dir}")

    crfs = list(range(int(args.crf_min), int(args.crf_max) + 1))
    use_feature_params = bool(args.feature_params) and not bool(args.params.strip())
    if args.params.strip():
        base_params = parse_x265_params(args.params)
    else:
        base_params = parse_x265_params(DEFAULT_BASE_PARAMS)
    base_params.pop("aq-strength", None)
    features_dir: Path = args.features_dir

    vmaf_thr = float(args.valid_vmaf if args.valid_vmaf is not None else args.vmaf_threshold)
    ratio_thr = float(args.min_ratio)

    total_workers = max(1, int(args.workers))
    video_workers = max(1, min(int(args.video_workers), len(video_dirs)))
    segment_workers = max(1, total_workers // video_workers)

    vmaf_n_threads = int(args.vmaf_n_threads)
    if vmaf_n_threads <= 0:
        vmaf_n_threads = max(2, min(6, 48 // segment_workers))

    if args.no_reuse_grid:
        reuse_roots: list[Path] = []
    elif args.reuse_dirs.strip():
        reuse_roots = [Path(p.strip()) for p in args.reuse_dirs.split(",") if p.strip()]
    else:
        reuse_roots = list(DEFAULT_REUSE_ROOTS)

    work_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"segmented  : {segmented_dir}")
    print(f"raw        : {raw_dir}")
    print(f"work_dir   : {work_root}")
    print(f"videos     : {len(video_dirs)}")
    print(f"CRF        : {crfs[0]}..{crfs[-1]} ({len(crfs)} values, all)")
    print(f"AQ range   : [{AQ_MIN}, {AQ_MAX}]  start {AQ_START_LOW} → {AQ_START_MID}")
    print(f"valid      : vmaf_neg > {vmaf_thr:g}  AND  ratio > {ratio_thr:g}")
    print(f"method     : interpolated bracket + probe-top + bisect/linear expand")
    print(
        f"workers    : videos={'serial' if video_workers == 1 else video_workers}  "
        f"segments_parallel={segment_workers}"
    )
    print(f"preset     : {args.preset}")
    if use_feature_params:
        print(f"x265 params: per-segment from {features_dir}")
    else:
        print(f"x265 params: fixed ({args.params or DEFAULT_BASE_PARAMS})")
    if use_feature_params and reuse_roots:
        print(
            "warning    : grid reuse may mismatch if grid used fixed --params "
            "(use --no-reuse-grid to re-encode with feature params)"
        )
    if reuse_roots:
        print(f"reuse      : {', '.join(str(r) for r in reuse_roots)}")
    else:
        print("reuse      : disabled")
    print(f"resume     : {bool(args.resume)}")
    print("save       : trials.jsonl + results.csv + summary.json (grid-compatible)")
    print("=" * 88)

    results: list[dict[str, Any]] = []
    if video_workers == 1:
        for vd in video_dirs:
            results.append(
                _process_one_video(
                    vd,
                    raw_dir=raw_dir,
                    work_root=work_root,
                    crfs=crfs,
                    base_params=base_params,
                    preset=str(args.preset),
                    profile=str(args.profile),
                    vmaf_threshold=int(args.vmaf_threshold),
                    vmaf_n_threads=vmaf_n_threads,
                    vmaf_n_subsample=int(args.vmaf_n_subsample),
                    use_gpu=bool(args.gpu),
                    gpu_device=int(args.gpu_device),
                    keep_encode=bool(args.keep_encodes),
                    resume=bool(args.resume),
                    force=bool(args.force),
                    vmaf_thr=vmaf_thr,
                    ratio_thr=ratio_thr,
                    expand_step=float(args.expand_step),
                    max_probes=int(args.max_aq_probes),
                    reuse_roots=reuse_roots,
                    use_feature_params=use_feature_params,
                    features_dir=features_dir,
                    segment_workers=segment_workers,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=video_workers) as pool:
            futs = {
                pool.submit(
                    _process_one_video,
                    vd,
                    raw_dir=raw_dir,
                    work_root=work_root,
                    crfs=crfs,
                    base_params=base_params,
                    preset=str(args.preset),
                    profile=str(args.profile),
                    vmaf_threshold=int(args.vmaf_threshold),
                    vmaf_n_threads=vmaf_n_threads,
                    vmaf_n_subsample=int(args.vmaf_n_subsample),
                    use_gpu=bool(args.gpu),
                    gpu_device=int(args.gpu_device),
                    keep_encode=bool(args.keep_encodes),
                    resume=bool(args.resume),
                    force=bool(args.force),
                    vmaf_thr=vmaf_thr,
                    ratio_thr=ratio_thr,
                    expand_step=float(args.expand_step),
                    max_probes=int(args.max_aq_probes),
                    reuse_roots=reuse_roots,
                    use_feature_params=use_feature_params,
                    features_dir=features_dir,
                    segment_workers=segment_workers,
                ): vd
                for vd in video_dirs
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"Done: {ok_n}/{len(results)} videos")
    for r in sorted(results, key=lambda x: str(x.get("video_stem", ""))):
        if r.get("ok"):
            print(
                f"  {r['video_stem']}: +{r.get('n_new_trials', 0)} new  "
                f"bypassed={r.get('n_bypassed_trials', 0)}  "
                f"{r.get('wall_sec', 0):.0f}s  {r.get('work_dir')}"
            )
        else:
            print(f"  {r.get('video_stem')}: FAILED — {r.get('error')}")
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
