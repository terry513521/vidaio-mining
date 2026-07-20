"""CRF-mode pack + Phase B/C (aq-strength × CRF compensation, then lookahead).

Phase 0 — fixed encode pack (ref/rd/bframes/la) + feature aq-mode rule
Phase A — interpolated CRF search (caller; uses this pack)
Phase B — walk aq-strength; compensate CRF from measurements:
            VMAF ≤ thr        → CRF− (quality recovery)
            VMAF > thr+headroom → CRF+ (spend headroom on size)
Phase C — try rc-lookahead 40 and 60 on the global best
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from interp_search import format_x265_params, parse_x265_params
from logutil import log
from param_tune import get_param, set_param


# Fixed knobs for CRF mode (Phase A + B + C).
# ref is capped at 6: libx265 rejects ref>=8 (often ref>6) on 4K Main —
# "NumPocTotalCurr non-compliant" → Cannot open encoder / NONE profile.
CRF_MODE_REF = 6
CRF_MODE_RD = 6
CRF_MODE_BFRAMES = 12
CRF_MODE_AQ_STRENGTH_DEFAULT = 1.0
CRF_MODE_LOOKAHEAD_DEFAULT = 50
CRF_MODE_KEYINT = 60
CRF_MODE_MIN_KEYINT = 1
CRF_MODE_SCENECUT = 50

CRF_MODE_AQ_MIN = 0.2
CRF_MODE_AQ_MAX = 2.4
CRF_MODE_AQ_STEP = 0.2
CRF_MODE_VMAF_HEADROOM = 2.0
CRF_MODE_CRF_COMPENSATE_STEPS = 2  # try ±1 .. ±K from base_crf
CRF_MODE_MAX_COMPRESSION_RATE: Optional[float] = None  # None = no size-rate gate
CRF_MODE_LOOKAHEAD_SWEEP = (40, 60)


def select_crf_mode_aq_mode(features: Optional[dict[str, Any]]) -> tuple[int, str]:
    """Pick aq-mode from soft features.

    aq_mode = 1 when high texture, low noise, few cuts; else 2.
    """
    f = features or {}
    texture = float(f.get("texture_level", f.get("texture", 0.0)) or 0.0)
    noise = float(f.get("noise_level_norm", f.get("noise", 0.0)) or 0.0)
    cut_count = int(float(f.get("cut_count", 0) or 0))

    if texture > 0.80 and noise < 0.15 and cut_count < 3:
        return 1, (
            f"texture_level={texture:.2f}>0.80, noise_level_norm={noise:.2f}<0.15, "
            f"cut_count={cut_count}<3 → aq-mode=1"
        )
    return 2, (
        f"texture_level={texture:.2f}, noise_level_norm={noise:.2f}, "
        f"cut_count={cut_count} → aq-mode=2"
    )


def build_crf_mode_params(
    features: Optional[dict[str, Any]] = None,
    *,
    aq_mode: Optional[int] = None,
    aq_strength: float = CRF_MODE_AQ_STRENGTH_DEFAULT,
    rc_lookahead: int = CRF_MODE_LOOKAHEAD_DEFAULT,
    aq_mode_reason: Optional[str] = None,
) -> tuple[str, int, str]:
    """Build colon-joined ``-x265-params`` for CRF mode.

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
        "rd": CRF_MODE_RD,
        "ref": CRF_MODE_REF,
        "bframes": CRF_MODE_BFRAMES,
        "rc-lookahead": int(rc_lookahead),
        "keyint": CRF_MODE_KEYINT,
        "min-keyint": CRF_MODE_MIN_KEYINT,
        "scenecut": CRF_MODE_SCENECUT,
    }
    return format_x265_params(params), aq_mode, reason


def aq_strength_candidates(
    *,
    aq_min: float = CRF_MODE_AQ_MIN,
    aq_max: float = CRF_MODE_AQ_MAX,
    aq_step: float = CRF_MODE_AQ_STEP,
    baseline: float = CRF_MODE_AQ_STRENGTH_DEFAULT,
) -> list[float]:
    """Stronger then weaker AQ values around baseline (baseline itself omitted)."""
    step = float(aq_step)
    if step <= 0:
        raise ValueError("aq_step must be > 0")
    lo = float(aq_min)
    hi = float(aq_max)
    base = round(float(baseline), 2)

    stronger: list[float] = []
    aq = round(base + step, 2)
    while aq <= hi + 1e-9:
        stronger.append(round(aq, 2))
        aq = round(aq + step, 2)

    weaker: list[float] = []
    aq = round(base - step, 2)
    while aq >= lo - 1e-9:
        weaker.append(round(aq, 2))
        aq = round(aq - step, 2)

    return stronger + weaker


@dataclass
class CrfModeTrial:
    crf: float
    aq_strength: float
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
class CrfModeState:
    crf: float
    aq_strength: float
    rc_lookahead: int
    params: str
    s_f: float
    vmaf: float
    compression_rate: float = 1.0
    compression_ratio: float = 1.0
    path: str = ""
    aq_mode: int = 2
    trials: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


EvaluateCrfModeFn = Callable[[float, str], CrfModeTrial]


def is_crf_mode_feasible(
    trial: CrfModeTrial,
    *,
    vmaf_threshold: float,
    max_compression_rate: Optional[float] = CRF_MODE_MAX_COMPRESSION_RATE,
) -> bool:
    """Feasibility for accepting a Phase B/C challenger.

    Requires encode/score ok and VMAF_neg above threshold. Optional
    ``max_compression_rate`` size gate is off by default.
    """
    if not trial.ok:
        return False
    if float(trial.vmaf) <= float(vmaf_threshold):
        return False
    if max_compression_rate is not None and float(trial.compression_rate) >= float(
        max_compression_rate
    ):
        return False
    return True


def run_crf_mode_phase_bc(
    *,
    base_crf: float,
    initial_params: str,
    initial_s_f: float,
    initial_vmaf: float,
    initial_path: str,
    initial_compression_rate: float,
    initial_compression_ratio: float,
    evaluate: EvaluateCrfModeFn,
    vmaf_threshold: float,
    crf_min: float = 1.0,
    crf_max: float = 51.0,
    aq_min: float = CRF_MODE_AQ_MIN,
    aq_max: float = CRF_MODE_AQ_MAX,
    aq_step: float = CRF_MODE_AQ_STEP,
    vmaf_headroom: float = CRF_MODE_VMAF_HEADROOM,
    crf_compensate_steps: int = CRF_MODE_CRF_COMPENSATE_STEPS,
    max_compression_rate: Optional[float] = CRF_MODE_MAX_COMPRESSION_RATE,
    lookahead_sweep: Sequence[int] = CRF_MODE_LOOKAHEAD_SWEEP,
    default_lookahead: int = CRF_MODE_LOOKAHEAD_DEFAULT,
    default_aq: float = CRF_MODE_AQ_STRENGTH_DEFAULT,
    job_id: str = "",
) -> CrfModeState:
    """Phase B (AQ walk + CRF± compensation) then Phase C (lookahead 40/60)."""
    aq_mode_raw = get_param(initial_params, "aq-mode", "2")
    try:
        aq_mode = int(float(aq_mode_raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        aq_mode = 2

    la0 = get_param(initial_params, "rc-lookahead", str(default_lookahead))
    try:
        initial_la = int(float(la0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        initial_la = int(default_lookahead)

    aq0 = get_param(initial_params, "aq-strength", str(default_aq))
    try:
        initial_aq = round(float(aq0), 2)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        initial_aq = float(default_aq)

    state = CrfModeState(
        crf=float(base_crf),
        aq_strength=initial_aq,
        rc_lookahead=initial_la,
        params=str(initial_params),
        s_f=float(initial_s_f),
        vmaf=float(initial_vmaf),
        compression_rate=float(initial_compression_rate),
        compression_ratio=float(initial_compression_ratio),
        path=str(initial_path or ""),
        aq_mode=aq_mode,
    )
    prefix = f"  [{job_id}] " if job_id else "  "
    seen: set[tuple[float, str]] = {(round(float(base_crf), 3), str(initial_params))}

    def _record(trial: CrfModeTrial, *, label: str) -> None:
        state.trials += 1
        trial.label = label
        trial.feasible = is_crf_mode_feasible(
            trial,
            vmaf_threshold=vmaf_threshold,
            max_compression_rate=max_compression_rate,
        )
        state.history.append(
            {
                "trial": state.trials,
                "label": label,
                "crf": trial.crf,
                "aq_strength": trial.aq_strength,
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

    def _maybe_accept(trial: CrfModeTrial, *, label: str) -> bool:
        _record(trial, label=label)
        if not trial.feasible:
            return False
        if trial.s_f > state.s_f + 1e-9:
            state.crf = float(trial.crf)
            state.aq_strength = float(trial.aq_strength)
            state.rc_lookahead = int(trial.rc_lookahead)
            state.params = str(trial.params)
            state.s_f = float(trial.s_f)
            state.vmaf = float(trial.vmaf)
            state.compression_rate = float(trial.compression_rate)
            state.compression_ratio = float(trial.compression_ratio)
            state.path = trial.path or state.path
            log(
                f"{prefix}crf-mode accept {label}: CRF={trial.crf:g} "
                f"aq={trial.aq_strength:g} la={trial.rc_lookahead} "
                f"s_f={trial.s_f:.4f} vmaf={trial.vmaf:.2f} "
                f"C={trial.compression_rate:.4f}"
            )
            return True
        return False

    def _params_for(aq: float, lookahead: int) -> str:
        p = set_param(initial_params, "aq-strength", round(float(aq), 2))
        p = set_param(p, "rc-lookahead", int(lookahead))
        # Keep frozen pack keys even if initial_params drifted.
        p = set_param(p, "aq-mode", aq_mode)
        p = set_param(p, "rd", CRF_MODE_RD)
        p = set_param(p, "ref", CRF_MODE_REF)
        p = set_param(p, "bframes", CRF_MODE_BFRAMES)
        return p

    def _try(crf: float, aq: float, lookahead: int, label: str) -> CrfModeTrial:
        crf_r = round(float(crf), 3)
        if crf_r > float(crf_max) + 1e-9 or crf_r < float(crf_min) - 1e-9:
            trial = CrfModeTrial(
                crf=crf_r,
                aq_strength=float(aq),
                rc_lookahead=int(lookahead),
                params=_params_for(aq, lookahead),
                reason="crf out of range",
                label=label,
            )
            _record(trial, label=label)
            return trial
        params = _params_for(aq, lookahead)
        key = (crf_r, params)
        if key in seen:
            trial = CrfModeTrial(
                crf=crf_r,
                aq_strength=float(aq),
                rc_lookahead=int(lookahead),
                params=params,
                reason="cache-skip",
                label=label,
            )
            # Do not count pure skips against trial budget messaging.
            return trial
        seen.add(key)
        trial = evaluate(crf_r, params)
        trial.crf = crf_r
        trial.aq_strength = float(aq)
        trial.rc_lookahead = int(lookahead)
        trial.params = params
        _maybe_accept(trial, label=label)
        return trial

    def _crf_compensate(s0: CrfModeTrial, aq: float, lookahead: int, label_prefix: str) -> None:
        """Measurement-driven CRF± from the AQ trial at base_crf.

        - VMAF ≤ threshold → lower CRF (recover quality)
        - VMAF > threshold + headroom → raise CRF (spend headroom on size)
        """
        if not s0.ok:
            return
        thr = float(vmaf_threshold)
        head = float(vmaf_headroom)
        steps = max(1, int(crf_compensate_steps))
        vmaf0 = float(s0.vmaf)

        if vmaf0 <= thr:
            # Quality broke at this AQ — walk CRF down.
            for step in range(1, steps + 1):
                s = _try(
                    base_crf - float(step),
                    aq,
                    lookahead,
                    f"{label_prefix}:crf-{step}",
                )
                if s.ok and float(s.vmaf) > thr:
                    break
            return

        if vmaf0 > thr + head:
            # Spare quality — walk CRF up while headroom remains.
            for step in range(1, steps + 1):
                s = _try(
                    base_crf + float(step),
                    aq,
                    lookahead,
                    f"{label_prefix}:crf+{step}",
                )
                if not s.ok or float(s.vmaf) <= thr + head:
                    break

    # --- Phase B ---
    aq_list = aq_strength_candidates(
        aq_min=aq_min, aq_max=aq_max, aq_step=aq_step, baseline=default_aq
    )
    log(
        f"{prefix}crf-mode Phase B: base_crf={base_crf:g} aq_mode={aq_mode} "
        f"la={default_lookahead} aq walk={aq_list} "
        f"(compensate ±{max(1, int(crf_compensate_steps))} CRF)"
    )
    for aq in aq_list:
        label = f"aq={aq:g}"
        s0 = _try(base_crf, aq, default_lookahead, label)
        _crf_compensate(s0, aq, default_lookahead, label)

    # --- Phase C ---
    sweep = [int(x) for x in lookahead_sweep if int(x) != int(default_lookahead)]
    if sweep:
        log(
            f"{prefix}crf-mode Phase C: best CRF={state.crf:g} aq={state.aq_strength:g} "
            f"try la={sweep}"
        )
    for la in sweep:
        _try(state.crf, state.aq_strength, la, f"la={la}")

    log(
        f"{prefix}crf-mode done: trials={state.trials} "
        f"best CRF={state.crf:g} aq={state.aq_strength:g} la={state.rc_lookahead} "
        f"s_f={state.s_f:.4f} vmaf={state.vmaf:.2f} C={state.compression_rate:.4f}"
    )
    return state


def merge_crf_mode_pack_into_recipe_params(
    features: Optional[dict[str, Any]],
    *,
    params_override: Optional[str] = None,
) -> tuple[str, int, str]:
    """Phase 0 helper: build pack; overlay may only add unknown keys.

    Frozen pack + Phase A defaults (aq-strength, rc-lookahead, aq-mode rule)
    always win over ``libx265_params``.
    """
    params, aq_mode, reason = build_crf_mode_params(features)
    if params_override:
        base = parse_x265_params(params)
        over = parse_x265_params(params_override)
        extras = {k: v for k, v in over.items() if k not in base}
        if extras:
            base.update(extras)
            params = format_x265_params(base)
            reason = f"{reason}; overlay extras={sorted(extras)}"
    return params, aq_mode, reason
