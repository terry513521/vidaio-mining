"""VBR-mode pack + preprocess arms + param ladder (proxy search).

Goal: find a suitable preprocess + libx265 pack that clears VMAF_neg, then stop.

Search order (fixed bitrate, proxy encodes), stopping early once
``vmaf_neg > vmaf_threshold`` on the saved best state:
  1. Preprocess arms — ``none`` first, then other candidates
  2. aq-strength walk (default 0.2 … 2.6 step 0.2)
  3. rd ∈ {4, 5, 6}
  4. bframes ∈ {6, 8, 12}
  5. rc-lookahead ∈ {40, 50, 60}

Pack freeze: ref=6, keyint=60, min-keyint=1, scenecut=50; aq-mode from features.
While still below threshold, accept a trial only when it is feasible and improves
``s_f``. As soon as the best state clears the threshold, finish and keep it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from crf_mode import select_crf_mode_aq_mode
from interp_search import format_x265_params
from logutil import log
from param_tune import get_param, set_param

VBR_MODE_REF = 6
VBR_MODE_RD_DEFAULT = 6
VBR_MODE_BFRAMES_DEFAULT = 8
VBR_MODE_AQ_STRENGTH_DEFAULT = 1.0
VBR_MODE_LOOKAHEAD_DEFAULT = 50
VBR_MODE_KEYINT = 60
VBR_MODE_MIN_KEYINT = 1
VBR_MODE_SCENECUT = 50

VBR_MODE_AQ_MIN = 0.2
VBR_MODE_AQ_MAX = 2.6
VBR_MODE_AQ_STEP = 0.2
VBR_MODE_RD_SWEEP: tuple[int, ...] = (4, 5, 6)
VBR_MODE_BFRAMES_SWEEP: tuple[int, ...] = (6, 8, 12)
VBR_MODE_LOOKAHEAD_SWEEP: tuple[int, ...] = (40, 50, 60)
VBR_MODE_PROXY_SECONDS_PER_SCENE = 2.0


def build_vbr_mode_params(
    features: Optional[dict[str, Any]] = None,
    *,
    aq_mode: Optional[int] = None,
    aq_strength: float = VBR_MODE_AQ_STRENGTH_DEFAULT,
    rd: int = VBR_MODE_RD_DEFAULT,
    bframes: int = VBR_MODE_BFRAMES_DEFAULT,
    rc_lookahead: int = VBR_MODE_LOOKAHEAD_DEFAULT,
    aq_mode_reason: Optional[str] = None,
) -> tuple[str, int, str]:
    """Build colon-joined ``-x265-params`` for VBR mode.

    Returns ``(params_str, aq_mode, reason)``.
    """
    if aq_mode is None:
        aq_mode, reason = select_crf_mode_aq_mode(features)
    else:
        aq_mode = int(aq_mode)
        reason = aq_mode_reason or f"aq-mode={aq_mode} (explicit)"

    params = {
        "aq-mode": aq_mode,
        "aq-strength": round(float(aq_strength), 2),
        "rd": int(rd),
        "ref": VBR_MODE_REF,
        "bframes": int(bframes),
        "rc-lookahead": int(rc_lookahead),
        "keyint": VBR_MODE_KEYINT,
        "min-keyint": VBR_MODE_MIN_KEYINT,
        "scenecut": VBR_MODE_SCENECUT,
    }
    return format_x265_params(params), aq_mode, reason


def aq_strength_grid(
    *,
    aq_min: float = VBR_MODE_AQ_MIN,
    aq_max: float = VBR_MODE_AQ_MAX,
    aq_step: float = VBR_MODE_AQ_STEP,
) -> list[float]:
    """Inclusive aq-strength grid from min to max."""
    step = float(aq_step)
    if step <= 0:
        raise ValueError("aq_step must be > 0")
    lo = float(aq_min)
    hi = float(aq_max)
    if lo > hi:
        raise ValueError("aq_min must be <= aq_max")
    out: list[float] = []
    aq = round(lo, 2)
    while aq <= hi + 1e-9:
        out.append(round(aq, 2))
        aq = round(aq + step, 2)
    return out


@dataclass
class VbrModeTrial:
    preprocess: Optional[str]
    aq_strength: float
    rd: int
    bframes: int
    rc_lookahead: int
    params: str
    s_f: float = 0.0
    vmaf: float = 0.0
    compression_rate: float = 1.0
    compression_ratio: float = 1.0
    path: str = ""
    ok: bool = False
    feasible: bool = False
    reason: str = ""
    label: str = ""


@dataclass
class VbrModeState:
    preprocess: Optional[str]
    aq_strength: float
    rd: int
    bframes: int
    rc_lookahead: int
    params: str
    s_f: float
    vmaf: float
    compression_rate: float = 1.0
    compression_ratio: float = 1.0
    path: str = ""
    aq_mode: int = 2
    preprocess_reason: str = ""
    trials: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""


# evaluate(preprocess, params) -> VbrModeTrial
EvaluateVbrModeFn = Callable[[Optional[str], str], VbrModeTrial]


def is_vbr_mode_feasible(trial: VbrModeTrial, *, vmaf_threshold: float) -> bool:
    if not trial.ok:
        return False
    if float(trial.vmaf) <= float(vmaf_threshold):
        return False
    if float(trial.s_f) <= 0:
        return False
    return True


def state_clears_vmaf_threshold(
    state: VbrModeState, *, vmaf_threshold: float
) -> bool:
    """True when saved best has VMAF_neg strictly above threshold."""
    return float(state.vmaf) > float(vmaf_threshold) and float(state.s_f) > 0


def should_accept_vbr_trial(
    state: VbrModeState,
    trial: VbrModeTrial,
    *,
    vmaf_threshold: float,
) -> bool:
    """Whether ``trial`` should replace the saved best.

    - Ignore failed / non-positive scores.
    - First ok result always seeds state.
    - First trial that clears ``vmaf > threshold`` always wins (search goal).
    - While still below threshold, prefer higher ``s_f`` (tie-break: higher vmaf).
    """
    if not trial.ok or float(trial.s_f) <= 0:
        return False
    thr = float(vmaf_threshold)
    trial_clears = float(trial.vmaf) > thr
    state_clears = state_clears_vmaf_threshold(state, vmaf_threshold=thr)
    if float(state.s_f) <= 0:
        return True
    if trial_clears and not state_clears:
        return True
    if trial_clears and state_clears:
        return float(trial.s_f) > float(state.s_f) + 1e-9
    if not trial_clears and not state_clears:
        if float(trial.s_f) > float(state.s_f) + 1e-9:
            return True
        if (
            abs(float(trial.s_f) - float(state.s_f)) <= 1e-9
            and float(trial.vmaf) > float(state.vmaf) + 1e-9
        ):
            return True
        return False
    return False


def _apply_trial_to_state(state: VbrModeState, trial: VbrModeTrial) -> None:
    state.preprocess = trial.preprocess
    state.params = str(trial.params)
    state.s_f = float(trial.s_f)
    state.vmaf = float(trial.vmaf)
    state.compression_rate = float(trial.compression_rate)
    state.compression_ratio = float(trial.compression_ratio)
    state.path = trial.path or state.path
    state.aq_strength = float(trial.aq_strength)
    state.rd = int(trial.rd)
    state.bframes = int(trial.bframes)
    state.rc_lookahead = int(trial.rc_lookahead)


def _knobs_from_params(params: str) -> tuple[float, int, int, int, int]:
    def _f(key: str, default: float) -> float:
        raw = get_param(params, key, str(default))
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float(default)

    def _i(key: str, default: int) -> int:
        return int(_f(key, float(default)))

    return (
        round(_f("aq-strength", VBR_MODE_AQ_STRENGTH_DEFAULT), 2),
        _i("rd", VBR_MODE_RD_DEFAULT),
        _i("bframes", VBR_MODE_BFRAMES_DEFAULT),
        _i("rc-lookahead", VBR_MODE_LOOKAHEAD_DEFAULT),
        _i("aq-mode", 2),
    )


def run_vbr_preprocess_arms(
    *,
    candidates: Sequence[Optional[str]],
    baseline_params: str,
    evaluate: EvaluateVbrModeFn,
    vmaf_threshold: float,
    job_id: str = "",
) -> VbrModeState:
    """Run preprocess arms: none first, then others; keep best + method."""
    aq, rd, bf, la, aq_mode = _knobs_from_params(baseline_params)
    ordered: list[Optional[str]] = []
    for cand in [None, *candidates]:
        if cand not in ordered:
            ordered.append(cand)

    prefix = f"  [{job_id}] " if job_id else "  "
    state = VbrModeState(
        preprocess=None,
        aq_strength=aq,
        rd=rd,
        bframes=bf,
        rc_lookahead=la,
        params=str(baseline_params),
        s_f=-1.0,
        vmaf=0.0,
        aq_mode=aq_mode,
        preprocess_reason="pending",
    )

    def _record(trial: VbrModeTrial, *, label: str) -> None:
        state.trials += 1
        trial.label = label
        trial.feasible = is_vbr_mode_feasible(trial, vmaf_threshold=vmaf_threshold)
        state.history.append(
            {
                "trial": state.trials,
                "label": label,
                "preprocess": trial.preprocess,
                "aq_strength": trial.aq_strength,
                "rd": trial.rd,
                "bframes": trial.bframes,
                "rc_lookahead": trial.rc_lookahead,
                "params": trial.params,
                "s_f": trial.s_f,
                "vmaf": trial.vmaf,
                "compression_rate": trial.compression_rate,
                "ok": trial.ok,
                "feasible": trial.feasible,
                "reason": trial.reason,
            }
        )

    def _maybe_accept(trial: VbrModeTrial, *, label: str) -> bool:
        _record(trial, label=label)
        if not should_accept_vbr_trial(
            state, trial, vmaf_threshold=vmaf_threshold
        ):
            if not trial.ok:
                log(
                    f"{prefix}vbr-mode skip {label}: "
                    f"{trial.reason or 'not ok'} "
                    f"s_f={trial.s_f:.4f} vmaf={trial.vmaf:.2f}"
                )
            return False
        _apply_trial_to_state(state, trial)
        tag = trial.preprocess or "none"
        state.preprocess_reason = f"best preprocess={tag} s_f={trial.s_f:.4f}"
        gate = "clears" if trial.feasible else "below-thr"
        log(
            f"{prefix}vbr-mode accept {label} ({gate}): preprocess={tag} "
            f"s_f={trial.s_f:.4f} vmaf={trial.vmaf:.2f} "
            f"C={trial.compression_rate:.4f}"
        )
        return True

    for cand in ordered:
        tag = cand or "none"
        label = f"pp={tag}"
        trial = evaluate(cand, baseline_params)
        trial.preprocess = cand
        trial.aq_strength = aq
        trial.rd = rd
        trial.bframes = bf
        trial.rc_lookahead = la
        trial.params = baseline_params
        _maybe_accept(trial, label=label)
        if state_clears_vmaf_threshold(state, vmaf_threshold=vmaf_threshold):
            state.stopped_early = True
            state.stop_reason = (
                f"vmaf {state.vmaf:.2f} > threshold {float(vmaf_threshold):g}; "
                f"preprocess locked to {state.preprocess or 'none'}"
            )
            log(f"{prefix}vbr-mode preprocess stop: {state.stop_reason}")
            break

    if state.s_f < 0:
        state.preprocess_reason = "all preprocess arms failed"
        state.s_f = 0.0
    return state


def run_vbr_param_ladder(
    *,
    preprocess: Optional[str],
    initial_params: str,
    initial_s_f: float,
    initial_vmaf: float,
    initial_path: str,
    initial_compression_rate: float,
    initial_compression_ratio: float,
    evaluate: EvaluateVbrModeFn,
    vmaf_threshold: float,
    aq_min: float = VBR_MODE_AQ_MIN,
    aq_max: float = VBR_MODE_AQ_MAX,
    aq_step: float = VBR_MODE_AQ_STEP,
    rd_sweep: Sequence[int] = VBR_MODE_RD_SWEEP,
    bframes_sweep: Sequence[int] = VBR_MODE_BFRAMES_SWEEP,
    lookahead_sweep: Sequence[int] = VBR_MODE_LOOKAHEAD_SWEEP,
    job_id: str = "",
) -> VbrModeState:
    """Sequential aq → rd → bframes → lookahead until VMAF clears threshold."""
    aq0, rd0, bf0, la0, aq_mode = _knobs_from_params(initial_params)
    state = VbrModeState(
        preprocess=preprocess,
        aq_strength=aq0,
        rd=rd0,
        bframes=bf0,
        rc_lookahead=la0,
        params=str(initial_params),
        s_f=float(initial_s_f),
        vmaf=float(initial_vmaf),
        compression_rate=float(initial_compression_rate),
        compression_ratio=float(initial_compression_ratio),
        path=str(initial_path or ""),
        aq_mode=aq_mode,
        preprocess_reason=f"locked preprocess={preprocess or 'none'}",
    )
    prefix = f"  [{job_id}] " if job_id else "  "
    seen: set[str] = {str(initial_params)}

    def _mark_done(reason: str) -> None:
        state.stopped_early = True
        state.stop_reason = reason
        log(f"{prefix}vbr-mode ladder stop: {reason}")

    if state_clears_vmaf_threshold(state, vmaf_threshold=vmaf_threshold):
        _mark_done(
            f"vmaf {state.vmaf:.2f} > threshold {float(vmaf_threshold):g}; "
            "skip param ladder"
        )
        return state

    def _params_for(
        *,
        aq: float,
        rd: int,
        bframes: int,
        lookahead: int,
    ) -> str:
        p = set_param(initial_params, "aq-mode", aq_mode)
        p = set_param(p, "aq-strength", round(float(aq), 2))
        p = set_param(p, "rd", int(rd))
        p = set_param(p, "ref", VBR_MODE_REF)
        p = set_param(p, "bframes", int(bframes))
        p = set_param(p, "rc-lookahead", int(lookahead))
        p = set_param(p, "keyint", VBR_MODE_KEYINT)
        p = set_param(p, "min-keyint", VBR_MODE_MIN_KEYINT)
        p = set_param(p, "scenecut", VBR_MODE_SCENECUT)
        return p

    def _record(trial: VbrModeTrial, *, label: str) -> None:
        state.trials += 1
        trial.label = label
        trial.feasible = is_vbr_mode_feasible(trial, vmaf_threshold=vmaf_threshold)
        state.history.append(
            {
                "trial": state.trials,
                "label": label,
                "preprocess": trial.preprocess,
                "aq_strength": trial.aq_strength,
                "rd": trial.rd,
                "bframes": trial.bframes,
                "rc_lookahead": trial.rc_lookahead,
                "params": trial.params,
                "s_f": trial.s_f,
                "vmaf": trial.vmaf,
                "compression_rate": trial.compression_rate,
                "ok": trial.ok,
                "feasible": trial.feasible,
                "reason": trial.reason,
            }
        )

    def _maybe_accept(trial: VbrModeTrial, *, label: str) -> bool:
        _record(trial, label=label)
        if not should_accept_vbr_trial(
            state, trial, vmaf_threshold=vmaf_threshold
        ):
            return False
        _apply_trial_to_state(state, trial)
        gate = "clears" if trial.feasible else "below-thr"
        log(
            f"{prefix}vbr-mode accept {label} ({gate}): "
            f"aq={trial.aq_strength:g} rd={trial.rd} bf={trial.bframes} "
            f"la={trial.rc_lookahead} s_f={trial.s_f:.4f} vmaf={trial.vmaf:.2f}"
        )
        return True

    def _try(aq: float, rd: int, bframes: int, lookahead: int, label: str) -> bool:
        """Evaluate one pack; return True if search should stop."""
        params = _params_for(aq=aq, rd=rd, bframes=bframes, lookahead=lookahead)
        if params in seen:
            return False
        seen.add(params)
        trial = evaluate(preprocess, params)
        trial.preprocess = preprocess
        trial.aq_strength = float(aq)
        trial.rd = int(rd)
        trial.bframes = int(bframes)
        trial.rc_lookahead = int(lookahead)
        trial.params = params
        _maybe_accept(trial, label=label)
        if state_clears_vmaf_threshold(state, vmaf_threshold=vmaf_threshold):
            _mark_done(
                f"vmaf {state.vmaf:.2f} > threshold {float(vmaf_threshold):g} "
                f"after {label}"
            )
            return True
        return False

    # Stage 1: aq-strength grid (skip current best aq to avoid duplicate).
    for aq in aq_strength_grid(aq_min=aq_min, aq_max=aq_max, aq_step=aq_step):
        if abs(aq - state.aq_strength) < 1e-9:
            continue
        if _try(aq, state.rd, state.bframes, state.rc_lookahead, f"aq={aq:g}"):
            return state

    # Stage 2: rd
    for rd in rd_sweep:
        if int(rd) == int(state.rd):
            continue
        if _try(
            state.aq_strength, int(rd), state.bframes, state.rc_lookahead, f"rd={int(rd)}"
        ):
            return state

    # Stage 3: bframes
    for bf in bframes_sweep:
        if int(bf) == int(state.bframes):
            continue
        if _try(
            state.aq_strength, state.rd, int(bf), state.rc_lookahead, f"bf={int(bf)}"
        ):
            return state

    # Stage 4: lookahead
    for la in lookahead_sweep:
        if int(la) == int(state.rc_lookahead):
            continue
        if _try(
            state.aq_strength, state.rd, state.bframes, int(la), f"la={int(la)}"
        ):
            return state

    return state
