"""Sequential libx265 param tuning after CRF search (maximize s_f).

Order: aq-mode → aq-strength → rd → ref → bframes → rc-lookahead.
After each key, optionally try CRF+1 when VMAF headroom ≥ threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from interp_search import format_x265_params, parse_x265_params

# Tune these knobs only; leave keyint/min-keyint/scenecut/etc. fixed.
PARAM_TUNE_KEYS: tuple[str, ...] = (
    "aq-mode",
    "aq-strength",
    "rd",
    "ref",
    "bframes",
    "rc-lookahead",
)

_AQ_STRENGTH_MIN = 0.8
_AQ_STRENGTH_MAX = 1.4


@dataclass
class ParamTuneState:
    crf: int
    params: str
    s_f: float
    vmaf: float
    path: str = ""
    trials: int = 0
    no_improve_streak: int = 0
    history: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def set_param(params: str, key: str, value: Any) -> str:
    """Return params with ``key`` set to ``value`` (colon-joined)."""
    parsed = parse_x265_params(params)
    if isinstance(value, float):
        parsed[key] = f"{value:g}"
    else:
        parsed[key] = str(value)
    # Preserve original key order when possible: rebuild from parsed dict.
    return ":".join(f"{k}={v}" for k, v in parsed.items())


def get_param(params: str, key: str, default: Any = None) -> Optional[str]:
    parsed = parse_x265_params(params)
    return parsed.get(key, None if default is None else str(default))


def feature_guided_candidates(
    key: str,
    baseline: Any,
    features: Optional[dict[str, Any]] = None,
) -> list[Any]:
    """Discrete candidates around the feature/baseline value for one knob."""
    f = features or {}
    motion = float(f.get("motion_level", f.get("motion", 0.4)) or 0.4)
    texture = float(f.get("texture_level", f.get("texture", 0.5)) or 0.5)
    noise = float(f.get("noise_level_norm", f.get("noise", 0.3)) or 0.3)

    if key == "aq-mode":
        try:
            base = int(float(baseline))
        except (TypeError, ValueError):
            base = 2
        # Prefer baseline first; include 1/2/3 nearby.
        ordered = [base]
        for cand in (1, 2, 3):
            if cand not in ordered:
                ordered.append(cand)
        # High texture/noise: try mode 1 earlier.
        if texture >= 0.75 or noise >= 0.45:
            ordered = [1] + [c for c in ordered if c != 1]
        return ordered[:3]

    if key == "aq-strength":
        try:
            base = float(baseline)
        except (TypeError, ValueError):
            base = 1.0
        deltas = [0.0, -0.1, 0.1, -0.2, 0.2]
        if texture >= 0.7:
            deltas = [0.0, 0.1, 0.2, -0.1]
        elif noise >= 0.45:
            deltas = [0.0, -0.1, -0.2, 0.1]
        out: list[float] = []
        for d in deltas:
            v = round(min(_AQ_STRENGTH_MAX, max(_AQ_STRENGTH_MIN, base + d)), 2)
            if v not in out:
                out.append(v)
        return out[:4]

    if key == "rd":
        try:
            base = int(float(baseline))
        except (TypeError, ValueError):
            base = 5
        cands = [base, base + 1, base - 1]
        return [c for c in cands if 3 <= c <= 6]

    if key == "ref":
        try:
            base = int(float(baseline))
        except (TypeError, ValueError):
            base = 4
        cands = [base, base + 1, base - 1]
        if motion >= 0.55:
            cands = [base, base + 1, base + 2, base - 1]
        out_i: list[int] = []
        for c in cands:
            if 2 <= c <= 6 and c not in out_i:
                out_i.append(c)
        return out_i[:4]

    if key == "bframes":
        try:
            base = int(float(baseline))
        except (TypeError, ValueError):
            base = 4
        step = 2
        cands = [base, base + step, base - step]
        if motion >= 0.55:
            cands = [base, base + step, base + 2 * step]
        out_b: list[int] = []
        for c in cands:
            if 2 <= c <= 8 and c not in out_b:
                out_b.append(c)
        return out_b[:4]

    if key == "rc-lookahead":
        try:
            base = int(float(baseline))
        except (TypeError, ValueError):
            base = 40
        deltas = [0, 10, -10, 20]
        if motion >= 0.55:
            deltas = [0, 10, 20, -10]
        out_l: list[int] = []
        for d in deltas:
            v = min(80, max(20, base + d))
            # Keep even-ish steps common for x265.
            v = int(round(v / 5.0) * 5)
            if v not in out_l:
                out_l.append(v)
        return out_l[:4]

    return [baseline]


def should_try_crf_bump(
    vmaf: float,
    *,
    vmaf_threshold: float,
    headroom: float = 2.0,
) -> bool:
    return float(vmaf) - float(vmaf_threshold) >= float(headroom)


def _same_param_value(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


@dataclass
class TuneTrialResult:
    ok: bool
    crf: int
    params: str
    s_f: float = 0.0
    vmaf: float = 0.0
    path: str = ""
    reason: str = ""


EvaluateFn = Callable[[int, str], TuneTrialResult]


def run_param_tune_loop(
    *,
    initial_crf: int,
    initial_params: str,
    initial_s_f: float,
    initial_vmaf: float,
    initial_path: str,
    features: Optional[dict[str, Any]],
    evaluate: EvaluateFn,
    vmaf_threshold: float,
    crf_max: int,
    max_trials: int = 25,
    no_improve_stop: int = 10,
    vmaf_headroom: float = 2.0,
    keys: Sequence[str] = PARAM_TUNE_KEYS,
    max_rounds: int = 3,
) -> ParamTuneState:
    """Multi-round sequential param tune; maximize ``s_f``.

    ``evaluate(crf, params)`` must encode+score and return ``TuneTrialResult``.
    The initial (crf, params) is assumed already scored and counts as trial 0
    (not against the budget).
    """
    state = ParamTuneState(
        crf=int(initial_crf),
        params=str(initial_params),
        s_f=float(initial_s_f),
        vmaf=float(initial_vmaf),
        path=str(initial_path or ""),
    )
    seen: set[tuple[int, str]] = {(state.crf, state.params)}

    def _accept(trial: TuneTrialResult, *, label: str) -> bool:
        state.trials += 1
        state.history.append(
            {
                "trial": state.trials,
                "label": label,
                "crf": trial.crf,
                "params": trial.params,
                "s_f": trial.s_f,
                "vmaf": trial.vmaf,
                "ok": trial.ok,
                "reason": trial.reason,
            }
        )
        if not trial.ok:
            state.no_improve_streak += 1
            return False
        if trial.s_f > state.s_f + 1e-9:
            state.crf = trial.crf
            state.params = trial.params
            state.s_f = trial.s_f
            state.vmaf = trial.vmaf
            state.path = trial.path or state.path
            state.no_improve_streak = 0
            return True
        state.no_improve_streak += 1
        return False

    def _budget_ok() -> bool:
        return state.trials < max_trials and state.no_improve_streak < no_improve_stop

    def _try(crf: int, params: str, label: str) -> bool:
        key = (int(crf), str(params))
        if key in seen:
            return False
        if not _budget_ok():
            return False
        seen.add(key)
        trial = evaluate(int(crf), str(params))
        trial.crf = int(crf)
        trial.params = str(params)
        return _accept(trial, label=label)

    def _maybe_crf_bump(label_prefix: str) -> None:
        if not _budget_ok():
            return
        if not should_try_crf_bump(
            state.vmaf,
            vmaf_threshold=vmaf_threshold,
            headroom=vmaf_headroom,
        ):
            return
        new_crf = int(state.crf) + 1
        if new_crf > int(crf_max):
            return
        _try(new_crf, state.params, f"{label_prefix}:crf+1")

    for round_idx in range(1, max(1, int(max_rounds)) + 1):
        if not _budget_ok():
            break
        round_best_s_f = state.s_f
        for key in keys:
            if not _budget_ok():
                break
            current_val = get_param(state.params, key)
            cands = feature_guided_candidates(key, current_val, features)
            for cand in cands:
                if not _budget_ok():
                    break
                # Skip the value already active (already scored at this CRF).
                if _same_param_value(cand, current_val):
                    continue
                new_params = set_param(state.params, key, cand)
                _try(state.crf, new_params, f"r{round_idx}:{key}={cand}")
            # After finishing this key's candidates, optional CRF bump.
            _maybe_crf_bump(f"r{round_idx}:{key}")
        # Outer plateau: no s_f gain across a full key pass.
        if state.s_f <= round_best_s_f + 1e-9:
            break

    return state


def params_dict_for_save(params: str) -> dict[str, str]:
    """Stable dict form for JSON (also keeps colon string via caller)."""
    return parse_x265_params(params)


def format_params_dict(d: dict[str, Any]) -> str:
    return format_x265_params(d)
