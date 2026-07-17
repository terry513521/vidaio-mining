"""Interpolated + answer-based CQ / CRF search helpers.

Round 1: feature-seeded CQ band around a content/threshold prior
Round 2+: propose unused CQs from measured answers, ranked by predicted s_f
         in the band between the best point and the VMAF cliff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from scoring import calculate_compression_score


@dataclass(frozen=True)
class CqObservation:
    cq: int
    vmaf: float
    compression_rate: float
    compression_ratio: float
    s_f: float
    encode_ok: bool


@dataclass(frozen=True)
class NvencOverrides:
    """Per-trial NVENC knob overrides (only fields set are applied)."""

    nvenc_tune: Optional[str] = None
    nvenc_rc: Optional[str] = None
    nvenc_multipass: Optional[str] = None
    nvenc_spatial_aq: Optional[bool] = None
    nvenc_temporal_aq: Optional[bool] = None
    nvenc_aq_strength: Optional[int] = None
    nvenc_rc_lookahead: Optional[int] = None
    nvenc_bf: Optional[int] = None
    nvenc_gop: Optional[int] = None
    nvenc_b_ref_mode: Optional[str] = None
    preprocess: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "nvenc_tune",
            "nvenc_rc",
            "nvenc_multipass",
            "nvenc_spatial_aq",
            "nvenc_temporal_aq",
            "nvenc_aq_strength",
            "nvenc_rc_lookahead",
            "nvenc_bf",
            "nvenc_gop",
            "nvenc_b_ref_mode",
            "preprocess",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out

    def suffix(self) -> str:
        parts: list[str] = []
        if self.nvenc_multipass is not None:
            parts.append(f"mp{self.nvenc_multipass[:4]}")
        if self.nvenc_aq_strength is not None:
            parts.append(f"aq{self.nvenc_aq_strength}")
        if self.nvenc_temporal_aq is False:
            parts.append("notaq")
        if self.nvenc_temporal_aq is True:
            parts.append("taq")
        if self.nvenc_spatial_aq is False:
            parts.append("nosaq")
        if self.nvenc_spatial_aq is True:
            parts.append("saq")
        if self.nvenc_tune is not None:
            parts.append(f"t{self.nvenc_tune}")
        if self.nvenc_rc is not None:
            parts.append(f"rc{self.nvenc_rc}")
        if self.nvenc_rc_lookahead is not None:
            parts.append(f"la{self.nvenc_rc_lookahead}")
        if self.nvenc_bf is not None:
            parts.append(f"bf{self.nvenc_bf}")
        if self.nvenc_gop is not None:
            parts.append(f"g{self.nvenc_gop}")
        if self.nvenc_b_ref_mode is not None:
            parts.append(f"bref{self.nvenc_b_ref_mode[:3]}")
        if self.preprocess is not None:
            parts.append(f"pp{self.preprocess}")
        return "_".join(parts) if parts else "base"


@dataclass(frozen=True)
class Round2TrialSpec:
    cq: int
    reason: str
    nvenc: NvencOverrides = NvencOverrides()
    predicted_s_f: float = 0.0


@dataclass(frozen=True)
class CqProposal:
    cq: int
    predicted_s_f: float
    reason: str


def round1_cqs(crf_min: int, crf_max: int, count: int) -> list[int]:
    """Evenly spaced integer CQs across the search range (inclusive).

    Prefer ``round1_feature_cqs`` when features are available; this linspace
    remains as a no-feature fallback.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if crf_min > crf_max:
        raise ValueError("crf_min must be <= crf_max")
    if count == 1:
        return [int(round((crf_min + crf_max) / 2.0))]

    span = crf_max - crf_min
    out: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        cq = int(round(crf_min + span * i / (count - 1)))
        cq = min(max(cq, crf_min), crf_max)
        if cq not in seen:
            seen.add(cq)
            out.append(cq)

    probe = crf_min
    while len(out) < count and probe <= crf_max:
        if probe not in seen:
            seen.add(probe)
            out.append(probe)
        probe += 1
    return sorted(out)


def cq_seed_from_features(
    features: Optional[dict[str, Any]],
    *,
    vmaf_threshold: float,
    crf_min: int,
    crf_max: int,
    crf_start: Optional[int] = None,
) -> tuple[int, str]:
    """Pick a CRF/CQ search seed from content features.

    Hard rule: this only chooses the *starting* probe. Measured sample VMAF
    still decides the final CRF. The seed should:

    * span a real content range (hard ≪ easy), not park everyone near ``crf_max``
    * prefer slightly safe (lower CRF) when uncertain so search can climb
    * leave room above/below for ab-av1-style bisection

    Explicit ``crf_start`` overrides. Returns ``(seed, reason)``.
    """
    lo, hi = (crf_min, crf_max) if crf_min <= crf_max else (crf_max, crf_min)
    if crf_start is not None:
        seed = min(max(int(crf_start), lo), hi)
        return seed, f"crf_start={crf_start}"

    thr = float(vmaf_threshold)
    span = max(1, hi - lo)
    # Mid-band bases (not cliff tops). Higher VMAF gate → lower base.
    if thr >= 93:
        base_frac = 0.38
        soft_lo_frac, soft_hi_frac = 0.12, 0.58
        base_tag = "thr>=93"
    elif thr >= 89:
        base_frac = 0.45
        soft_lo_frac, soft_hi_frac = 0.15, 0.65
        base_tag = "thr>=89"
    else:
        base_frac = 0.50
        soft_lo_frac, soft_hi_frac = 0.18, 0.72
        base_tag = "thr<89"

    base = lo + span * base_frac
    soft_lo = int(round(lo + span * soft_lo_frac))
    soft_hi = int(round(lo + span * soft_hi_frac))
    soft_lo = min(max(soft_lo, lo), hi)
    soft_hi = min(max(soft_hi, soft_lo), hi)

    lvl = _feature_levels(features)
    motion = lvl["motion"]
    texture = lvl["texture"]
    edge = lvl["edge"]
    noise = lvl["noise"]
    cuts = lvl["cuts"]
    mashup = _feature_mashup(features)

    # 0 = easy, 1 = hard. Texture/motion dominate for 4K fleet content.
    difficulty = (
        0.35 * texture
        + 0.25 * motion
        + 0.15 * edge
        + 0.15 * mashup
        + 0.10 * cuts
    )
    difficulty = min(max(difficulty, 0.0), 1.0)

    # Map difficulty onto a wide bias: easy climbs, hard drops hard.
    # difficulty 0.40 → ~0 bias; 0 → +~0.28*span; 1 → -~0.42*span
    bias = (0.40 - difficulty) * span * 0.70

    # Extra floors for the signals that previously under-moved the seed.
    if texture >= 0.85:
        bias -= 5.0
    elif texture >= 0.70:
        bias -= 3.0
    elif texture >= 0.55:
        bias -= 1.5

    if motion >= 0.65:
        bias -= 3.0
    elif motion >= 0.50:
        bias -= 1.5

    if mashup >= 0.60:
        bias -= 3.0
    elif mashup >= 0.45:
        bias -= 1.5

    if edge >= 0.65 and texture >= 0.55:
        bias -= 1.0

    # Grain: VMAF-NEG is sensitive → prefer a bit more quality (lower CRF).
    if noise >= 0.55:
        bias -= 1.5
    elif noise >= 0.45:
        bias -= 0.5

    # Easy / clean content: allow climbing toward compression.
    if difficulty <= 0.35 and mashup < 0.35:
        bias += 2.0

    seed = int(round(base + bias))
    seed = min(max(seed, soft_lo), soft_hi)
    seed = min(max(seed, lo), hi)
    reason = (
        f"{base_tag} base={base:.1f} difficulty={difficulty:.2f} "
        f"tex={texture:.2f} mot={motion:.2f} mashup={mashup:.2f} "
        f"bias={bias:+.1f} soft=[{soft_lo},{soft_hi}] → CQ {seed}"
    )
    return seed, reason


def _feature_mashup(features: Optional[dict[str, Any]]) -> float:
    f = features or {}
    worst = float(f.get("worst_difficulty", 0.0) or 0.0)
    hard_frac = float(f.get("hard_fraction", 0.0) or 0.0)
    volatility = float(f.get("volatility", 0.0) or 0.0)
    return 0.50 * worst + 0.30 * volatility + 0.20 * hard_frac


def _proxy_probe_offsets(
    features: Optional[dict[str, Any]],
    *,
    count: int,
    spread: int,
) -> tuple[list[int], str]:
    """Feature-asymmetric CRF offsets around the rule seed (count=2 is specialized)."""
    spread = max(1, int(spread))
    if count <= 1:
        return [0], "center"
    if count != 2:
        return [(i - 1) * spread for i in range(count)], "upward_band"

    lvl = _feature_levels(features)
    motion = lvl["motion"]
    noise = lvl["noise"]
    mashup = _feature_mashup(features)
    composite = (
        0.30 * motion
        + 0.25 * lvl["texture"]
        + 0.20 * lvl["edge"]
        + 0.15 * lvl["cuts"]
        + 0.10 * noise
    )

    # Very hard mashup: stay conservative (proxy→full drop can be large).
    if mashup >= 0.60:
        return [-spread, 0], "hard_down_band"
    if mashup >= 0.55 or motion >= 0.52:
        if mashup >= 0.55:
            return [-spread, 0], "hard_down_band"
        # Motion high but mashup already pulled seed down — probe upward.
        return [0, spread], "hard_up_band"
    if noise >= 0.50:
        return [-1, 0], "noise_down_band"
    if composite < 0.45:
        return [0, spread], "easy_up_band"
    return [-1, spread], "default_bracket"


def round1_feature_cqs(
    features: Optional[dict[str, Any]],
    *,
    count: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    spread: int = 2,
    crf_start: Optional[int] = None,
) -> tuple[list[int], int, str]:
    """Feature-seeded Round-1 CQ candidates with upward (cliff) bias.

    Places one safety probe below the seed, then climbs toward ``crf_max``
    so Round 1 actually samples near the VMAF gate instead of mid-band.

    Returns ``(candidates, seed, reason)``. Falls back to linspace when
    ``features`` is empty/None and ``crf_start`` is unset.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if crf_min > crf_max:
        raise ValueError("crf_min must be <= crf_max")

    if not features and crf_start is None:
        cqs = round1_cqs(crf_min, crf_max, count)
        mid = cqs[len(cqs) // 2] if cqs else crf_min
        return cqs, mid, "linspace_fallback(no_features)"

    seed, reason = cq_seed_from_features(
        features,
        vmaf_threshold=vmaf_threshold,
        crf_min=crf_min,
        crf_max=crf_max,
        crf_start=crf_start,
    )
    spread = max(1, int(spread))
    center = min(max(int(seed), crf_min), crf_max)
    if count == 1:
        return [center], center, reason

    offsets, band_tag = _proxy_probe_offsets(features, count=count, spread=spread)

    ordered: list[int] = []
    seen: set[int] = set()
    for offset in offsets:
        cq = min(max(center + offset, crf_min), crf_max)
        if cq not in seen:
            seen.add(cq)
            ordered.append(cq)

    # Fill gaps if clamping removed uniqueness (seed near bounds).
    step = 1
    while len(ordered) < count and step <= (crf_max - crf_min + 1):
        for side in (step, -step):
            cand = center + side
            if crf_min <= cand <= crf_max and cand not in seen:
                seen.add(cand)
                ordered.append(cand)
                if len(ordered) >= count:
                    break
        step += 1

    return sorted(ordered[:count]), center, reason + f"; {band_tag}"


def next_serial_cq_probe(
    observations: Sequence[CqObservation],
    *,
    round_idx: int,
    max_rounds: int,
    probe_plan: Sequence[int],
    features: Optional[dict[str, Any]],
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    spread: int,
    crf_start: Optional[int],
    used: set[int],
) -> tuple[Optional[int], str]:
    """Pick exactly one CQ to probe this round (fleet / serial search).

    Round 1..len(probe_plan) walks the feature band one CQ at a time.
    Later rounds use interpolated proposals (count=1). Returns (None, reason)
    when no further probes are needed.
    """
    if round_idx < 1 or round_idx > max_rounds:
        return None, "max_rounds_reached"

    plan = [int(c) for c in probe_plan]
    if round_idx <= len(plan):
        cq = plan[round_idx - 1]
        if cq not in used:
            return cq, f"probe_plan[{round_idx}]"
        # Skip duplicates in plan (clamping collisions).
        for c in plan[round_idx - 1 :]:
            if c not in used:
                return c, f"probe_plan[{round_idx}](dedup)"
        return None, "probe_plan_exhausted"

    proposals = propose_round2_details(
        observations,
        count=1,
        crf_min=crf_min,
        crf_max=crf_max,
        vmaf_threshold=float(vmaf_threshold),
        used=used,
    )
    if proposals:
        p = proposals[0]
        return p.cq, f"interp_{p.reason}"

    # Fallback: feature band if we have no observations yet.
    if not observations:
        cqs, _, reason = round1_feature_cqs(
            features,
            count=max_rounds,
            crf_min=crf_min,
            crf_max=crf_max,
            vmaf_threshold=float(vmaf_threshold),
            spread=max(1, spread),
            crf_start=crf_start,
        )
        for c in cqs:
            if c not in used:
                return c, f"fallback_{reason}"
    return None, "no_candidates"


def estimate_primary_x265_crf(
    observations: Sequence[CqObservation],
    *,
    nvenc_cq_min: int,
    nvenc_cq_max: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
) -> tuple[Optional[int], str]:
    """Single best x265 CRF from NVENC probe curve (VMAF-anchored match)."""
    props = propose_vmaf_anchored_crfs(
        observations,
        count=1,
        nvenc_cq_min=nvenc_cq_min,
        nvenc_cq_max=nvenc_cq_max,
        crf_min=crf_min,
        crf_max=crf_max,
        vmaf_threshold=float(vmaf_threshold),
        spread=2,
    )
    if not props:
        return None, "no_observations"
    p = props[0]
    return p.crf, p.reason


def interpolate_cq_for_vmaf(
    observations: Sequence[CqObservation],
    target_vmaf: float,
) -> Optional[float]:
    """Linear-interpolate CQ where VMAF crosses ``target_vmaf``."""
    pts = sorted(
        (o for o in observations if o.encode_ok and o.vmaf > 0),
        key=lambda o: o.cq,
    )
    if len(pts) < 2:
        return float(pts[0].cq) if pts else None

    for left, right in zip(pts, pts[1:]):
        lo_v, hi_v = left.vmaf, right.vmaf
        if (lo_v - target_vmaf) * (hi_v - target_vmaf) <= 0 and lo_v != hi_v:
            t = (target_vmaf - lo_v) / (hi_v - lo_v)
            return left.cq + t * (right.cq - left.cq)

    closest = sorted(pts, key=lambda o: abs(o.vmaf - target_vmaf))[:2]
    closest = sorted(closest, key=lambda o: o.cq)
    a, b = closest[0], closest[1]
    if a.vmaf == b.vmaf:
        return float(a.cq)
    t = (target_vmaf - a.vmaf) / (b.vmaf - a.vmaf)
    return a.cq + t * (b.cq - a.cq)


def estimate_interp_crf(
    observations: Sequence[CqObservation],
    *,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
) -> tuple[Optional[int], str]:
    """Pick x265 CRF from proxy observations via VMAF interpolation."""
    pts = [o for o in observations if o.encode_ok and o.vmaf > 0]
    if not pts:
        return None, "no_observations"
    if len(pts) == 1:
        o = pts[0]
        if o.vmaf >= float(vmaf_threshold):
            crf = min(crf_max, int(o.cq) + 1)
            return crf, f"single_probe_vmaf={o.vmaf:.1f}>={vmaf_threshold}"
        crf = max(crf_min, int(o.cq) - 1)
        return crf, f"single_probe_vmaf={o.vmaf:.1f}<{vmaf_threshold}"
    cq_star = interpolate_cq_for_vmaf(pts, float(vmaf_threshold))
    if cq_star is None:
        best = max(pts, key=lambda o: o.s_f)
        return int(best.cq), "best_s_f_fallback"
    crf = int(round(cq_star))
    crf = min(max(crf, crf_min), crf_max)
    return crf, f"interp_vmaf={vmaf_threshold} crf*={cq_star:.2f}"


def _clamp_rule_crf(
    crf: int,
    *,
    seed: int,
    candidates: Sequence[int],
    crf_min: int,
    crf_max: int,
    max_nudge: int = 1,
) -> int:
    """Keep final CRF within the probed candidate band (+/- nudge)."""
    del seed  # anchor is the probe plan, not seed alone
    cand_lo = min(int(c) for c in candidates)
    cand_hi = max(int(c) for c in candidates)
    nudge = max(0, int(max_nudge))
    lo = max(crf_min, cand_lo - nudge)
    hi = min(crf_max, cand_hi + nudge)
    return min(max(int(crf), lo), hi)


def pick_rule_anchored_crf(
    observations: Sequence[CqObservation],
    *,
    seed: int,
    candidates: Sequence[int],
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    max_nudge: int = 1,
    proxy_vmaf_margin: float = 2.0,
    mashup_push_ceiling: float = 0.55,
    features: Optional[dict[str, Any]] = None,
) -> tuple[Optional[int], str]:
    """Confirm or nudge a rule-seeded CRF using proxy VMAF (bounded adjustment).

    Targets proxy VMAF near ``vmaf_threshold + proxy_vmaf_margin`` so the
    full-file encode lands closer to the gate. Skips upward pushes on very
    hard mashups (large proxy→full VMAF drop).
    """
    pts = sorted(
        (o for o in observations if o.encode_ok and o.vmaf > 0),
        key=lambda o: o.cq,
    )
    if not pts:
        return None, "no_observations"

    thr = float(vmaf_threshold)
    proxy_target = thr + float(proxy_vmaf_margin)
    seed_i = int(seed)
    cands = [int(c) for c in candidates]
    nudge = max(0, int(max_nudge))
    mashup = _feature_mashup(features)
    push_ok = mashup < float(mashup_push_ceiling)

    def clamp(crf: int) -> int:
        return _clamp_rule_crf(
            crf,
            seed=seed_i,
            candidates=cands,
            crf_min=crf_min,
            crf_max=crf_max,
            max_nudge=nudge,
        )

    if len(pts) == 1:
        obs = pts[0]
        if obs.vmaf >= thr:
            if push_ok and obs.vmaf >= proxy_target + 1.0 and nudge > 0:
                crf = clamp(min(crf_max, obs.cq + nudge))
                return (
                    crf,
                    f"single_pass_margin+{nudge} vmaf={obs.vmaf:.1f} "
                    f"target={proxy_target:.1f}",
                )
            crf = clamp(min(crf_max, obs.cq + (nudge if push_ok and obs.vmaf >= proxy_target else 0)))
            if crf > obs.cq:
                return crf, f"single_pass_nudge+{crf - obs.cq} vmaf={obs.vmaf:.1f}"
            return crf, f"single_pass_keep vmaf={obs.vmaf:.1f}"
        crf = clamp(max(crf_min, obs.cq - nudge))
        return crf, f"single_fail_nudge-{nudge} vmaf={obs.vmaf:.1f}"

    passing = [o for o in pts if o.vmaf >= thr]
    failing = [o for o in pts if o.vmaf < thr]

    if passing and not failing:
        best_sf = max(passing, key=lambda o: (o.s_f, o.cq))
        highest = max(passing, key=lambda o: (o.cq, o.s_f))
        min_vmaf = min(o.vmaf for o in passing)
        margin_room = min_vmaf - proxy_target
        pick = highest if push_ok and margin_room >= 0 else best_sf
        if push_ok and margin_room >= 1.0 and nudge > 0:
            crf = clamp(min(crf_max, pick.cq + nudge))
            return (
                crf,
                f"all_pass_margin+{nudge} target={proxy_target:.1f} "
                f"pick={pick.cq} min_vmaf={min_vmaf:.1f}",
            )
        if push_ok and margin_room >= 0:
            crf = clamp(pick.cq)
            return (
                crf,
                f"all_pass_high_crf pick={pick.cq} min_vmaf={min_vmaf:.1f} "
                f"target={proxy_target:.1f}",
            )
        crf = clamp(best_sf.cq)
        return (
            crf,
            f"all_pass_best_s_f pick={best_sf.cq} s_f={best_sf.s_f:.4f} "
            f"mashup={mashup:.2f}",
        )

    if failing and not passing:
        lowest = min(pts, key=lambda o: o.cq)
        crf = clamp(max(crf_min, lowest.cq - nudge))
        return (
            crf,
            f"all_fail_nudge-{nudge} from={lowest.cq} vmaf={lowest.vmaf:.1f}",
        )

    low = min(pts, key=lambda o: o.cq)
    high = max(pts, key=lambda o: o.cq)
    best_pass = max(passing, key=lambda o: (o.s_f, o.cq))

    if (
        push_ok
        and low in passing
        and high in failing
        and low.vmaf >= proxy_target - 0.5
        and high.vmaf >= thr - 1.0
    ):
        crf = clamp(high.cq)
        return (
            crf,
            f"bracket_lean_up high={high.cq} vmaf={high.vmaf:.1f} "
            f"low={low.cq}@{low.vmaf:.1f}",
        )

    cq_star = interpolate_cq_for_vmaf(pts, thr)
    if cq_star is not None:
        interp = int(round(cq_star))
        interp = min(max(interp, min(cands)), max(cands))
        passing_cqs = {o.cq for o in passing}
        if interp in passing_cqs:
            interp_obs = next(o for o in passing if o.cq == interp)
            pick = interp_obs if interp_obs.s_f >= best_pass.s_f else best_pass
        else:
            pick = best_pass
    else:
        pick = best_pass

    crf = clamp(pick.cq)
    return crf, f"bracket pick={pick.cq} vmaf={pick.vmaf:.1f} s_f={pick.s_f:.4f}"


def _predicted_s_f(
    vmaf: float,
    compression_rate: float,
    vmaf_threshold: float,
) -> float:
    s_f, _, _, _ = calculate_compression_score(
        vmaf_score=vmaf,
        compression_rate=compression_rate,
        vmaf_threshold=vmaf_threshold,
    )
    return s_f


def _interp_rate(cq: float, a: CqObservation, b: CqObservation) -> float:
    if a.cq == b.cq:
        return a.compression_rate
    t = (cq - a.cq) / (b.cq - a.cq)
    return a.compression_rate + t * (b.compression_rate - a.compression_rate)


def _interp_vmaf(cq: float, a: CqObservation, b: CqObservation) -> float:
    if a.cq == b.cq:
        return a.vmaf
    t = (cq - a.cq) / (b.cq - a.cq)
    return a.vmaf + t * (b.vmaf - a.vmaf)


def predict_s_f_at_cq(
    cq: int,
    observations: Sequence[CqObservation],
    vmaf_threshold: float,
) -> Optional[float]:
    """Predict s_f at ``cq`` by interpolating VMAF + compression_rate, then formula."""
    pts = sorted(
        (o for o in observations if o.encode_ok and o.compression_rate > 0),
        key=lambda o: o.cq,
    )
    if not pts:
        return None
    if cq <= pts[0].cq:
        a = pts[0]
        return _predicted_s_f(a.vmaf, a.compression_rate, vmaf_threshold)
    if cq >= pts[-1].cq:
        a = pts[-1]
        return _predicted_s_f(a.vmaf, a.compression_rate, vmaf_threshold)
    for left, right in zip(pts, pts[1:]):
        if left.cq <= cq <= right.cq:
            vmaf = _interp_vmaf(cq, left, right)
            rate = _interp_rate(cq, left, right)
            return _predicted_s_f(vmaf, rate, vmaf_threshold)
    return None


def _pick_anchor(pts: Sequence[CqObservation], vmaf_threshold: float) -> CqObservation:
    """Best measured point to refine around."""
    positive = [o for o in pts if o.s_f > 0]
    if positive:
        return max(positive, key=lambda o: (o.s_f, o.vmaf, -o.cq))
    hard_cutoff = float(vmaf_threshold) - 5.0
    feasible = [o for o in pts if o.vmaf >= hard_cutoff]
    if feasible:
        return max(feasible, key=lambda o: (o.vmaf, -o.cq))
    return max(pts, key=lambda o: (o.vmaf, -o.cq))


def _sweet_band(
    pts: Sequence[CqObservation],
    anchor: CqObservation,
    vmaf_threshold: float,
) -> tuple[int, int]:
    """CQ window where s_f can still improve: near peak, toward the VMAF cliff.

    Prefer exploring *higher* CQ (more compression) while VMAF stays viable.
    Avoid deep quality-overshoot (much lower CQ than anchor) and known-fail CQs.
    """
    threshold = float(vmaf_threshold)
    hard_cutoff = threshold - 5.0
    ordered = sorted(pts, key=lambda o: o.cq)

    # First measured CQ at/above threshold with soft/fail on the right side of anchor.
    soft_edge = None
    hard_edge = None
    for o in ordered:
        if o.cq <= anchor.cq:
            continue
        if soft_edge is None and o.vmaf < threshold:
            soft_edge = o.cq
        if hard_edge is None and o.vmaf < hard_cutoff:
            hard_edge = o.cq
            break

    # Lower bound: don't wander far into the flat high-quality / low-ratio region.
    lo = max(anchor.cq - 1, min(o.cq for o in ordered))
    # Upper bound: up to just before hard fail; allow one step into soft zone for bracketing.
    if hard_edge is not None:
        hi = hard_edge
    elif soft_edge is not None:
        hi = soft_edge
    else:
        hi = max(o.cq for o in ordered)

    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def propose_round2_cqs(
    observations: Sequence[CqObservation],
    *,
    count: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    used: set[int],
) -> list[int]:
    """Choose up to ``count`` new CQs ranked by predicted s_f from Round-1 answers."""
    proposals = propose_round2_details(
        observations,
        count=count,
        crf_min=crf_min,
        crf_max=crf_max,
        vmaf_threshold=vmaf_threshold,
        used=used,
    )
    return [p.cq for p in proposals]


def propose_round2_details(
    observations: Sequence[CqObservation],
    *,
    count: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    used: set[int],
) -> list[CqProposal]:
    """Like ``propose_round2_cqs`` but includes predicted s_f + reason for logging."""
    pts = [o for o in observations if o.encode_ok]
    if not pts:
        fallback = [
            c
            for c in round1_cqs(crf_min, crf_max, count + len(used))
            if c not in used
        ][:count]
        return [CqProposal(cq=c, predicted_s_f=0.0, reason="fallback_linspace") for c in fallback]

    threshold = float(vmaf_threshold)
    hard_cutoff = threshold - 5.0
    anchor = _pick_anchor(pts, threshold)
    band_lo, band_hi = _sweet_band(pts, anchor, threshold)
    band_lo = max(crf_min, band_lo)
    band_hi = min(crf_max, band_hi)
    cq_vmaf = interpolate_cq_for_vmaf(pts, threshold)
    cq_vmaf_i = int(round(cq_vmaf)) if cq_vmaf is not None else None

    def _biased_score(cq: int, pred: float) -> tuple[float, str]:
        reasons: list[str] = ["pred_s_f"]
        score = pred
        if band_lo <= cq <= band_hi:
            score += 0.02
            reasons.append("sweet_band")
        if cq_vmaf_i is not None and abs(cq - cq_vmaf_i) <= 1:
            score += 0.015
            reasons.append("near_cq_vmaf*")
        # Prefer climbing CQ (more compression) only while prediction stays healthy.
        if cq == anchor.cq + 1 and pred >= 0.5 * max(anchor.s_f, 1e-9):
            score += 0.02
            reasons.append("best+1")
        if cq < anchor.cq - 1:
            score -= 0.10 * (anchor.cq - cq)
            reasons.append("overshoot_penalty")
        # Strong penalty once VMAF interp is below hard cutoff.
        # Soft-zone predictions stay allowed but naturally rank low via pred_s_f.
        right = [o for o in pts if o.cq >= cq]
        if right and max(o.vmaf for o in right) < hard_cutoff and cq > anchor.cq + 1:
            score -= 0.25
            reasons.append("known_fail_region")
        return score, "+".join(reasons)

    ranked: list[tuple[float, CqProposal]] = []
    for cq in range(crf_min, crf_max + 1):
        if cq in used:
            continue
        pred = predict_s_f_at_cq(cq, pts, threshold)
        if pred is None:
            continue
        bias_score, reason = _biased_score(cq, pred)
        ranked.append((bias_score, CqProposal(cq=cq, predicted_s_f=pred, reason=reason)))

    ranked.sort(key=lambda item: (item[0], item[1].predicted_s_f, -item[1].cq), reverse=True)

    # Only chase CQs that can plausibly match the current best.
    # Reject deep quality-overshoot (< best-1): Round 1 already mapped that side.
    peak = max(anchor.s_f, 0.0)
    min_pred = peak * 0.95 if peak > 0 else 0.0

    promising: list[CqProposal] = []
    for _, prop in ranked:
        if prop.cq < anchor.cq - 1:
            continue  # known worse / flat-VMAF region
        keep = (
            prop.predicted_s_f >= min_pred
            or prop.cq == anchor.cq + 1  # one cliff probe
            or (
                cq_vmaf_i is not None
                and prop.cq == cq_vmaf_i
                and prop.cq >= anchor.cq - 1
            )
        )
        if keep:
            promising.append(prop)

    if not promising:
        # Fall back to nearest unused neighbors around the best answer.
        for cq, tag in (
            (anchor.cq - 1, "fallback_best-1"),
            (anchor.cq + 1, "fallback_best+1"),
            (cq_vmaf_i, "fallback_cq_vmaf*"),
        ):
            if cq is None or cq in used or not (crf_min <= cq <= crf_max):
                continue
            pred = predict_s_f_at_cq(cq, pts, threshold) or 0.0
            promising.append(CqProposal(cq=cq, predicted_s_f=pred, reason=tag))

    out: list[CqProposal] = []
    seen: set[int] = set()
    for prop in promising:
        if prop.cq in seen or prop.cq in used:
            continue
        if not (crf_min <= prop.cq <= crf_max):
            continue
        seen.add(prop.cq)
        out.append(prop)
        if len(out) >= count:
            break

    out.sort(key=lambda p: p.cq)
    return out


@dataclass(frozen=True)
class HandoffCq:
    """A CQ value handed off from Round 1 to a second encoder."""

    cq: int
    reason: str
    predicted_s_f: float


@dataclass(frozen=True)
class CrfProposal:
    """An x265 CRF candidate with provenance for logging."""

    crf: int
    reason: str
    target_vmaf: float


def propose_handoff_cqs(
    observations: Sequence[CqObservation],
    *,
    count: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
) -> list[HandoffCq]:
    """CQ set to hand off from Round 1 (measured best + higher interp).

    Kept for logging / debugging; refine now prefers
    ``propose_vmaf_anchored_crfs`` instead of CQ+offset mapping.
    """
    if count < 1:
        return []
    pts = [o for o in observations if o.encode_ok]
    if not pts:
        return []

    anchor = _pick_anchor(pts, float(vmaf_threshold))
    out: list[HandoffCq] = [
        HandoffCq(cq=anchor.cq, reason="measured_best", predicted_s_f=anchor.s_f)
    ]
    seen: set[int] = {anchor.cq}

    remaining = count - 1
    if remaining > 0:
        proposals = propose_round2_details(
            pts,
            count=remaining,
            crf_min=crf_min,
            crf_max=crf_max,
            vmaf_threshold=float(vmaf_threshold),
            used=set(seen),
        )
        for prop in proposals:
            if prop.cq in seen:
                continue
            seen.add(prop.cq)
            out.append(
                HandoffCq(
                    cq=prop.cq,
                    reason=f"interp_{prop.reason}",
                    predicted_s_f=prop.predicted_s_f,
                )
            )
            if len(out) >= count:
                break

    return out[:count]


def _range_map(value: float, src_lo: int, src_hi: int, dst_lo: int, dst_hi: int) -> int:
    """Map ``value`` from [src_lo, src_hi] into [dst_lo, dst_hi] by relative position."""
    span_src = max(1e-9, float(src_hi - src_lo))
    span_dst = float(dst_hi - dst_lo)
    frac = (float(value) - float(src_lo)) / span_src
    frac = min(max(frac, 0.0), 1.0)
    return int(round(dst_lo + frac * span_dst))


def propose_vmaf_anchored_crfs(
    observations: Sequence[CqObservation],
    *,
    count: int,
    nvenc_cq_min: int,
    nvenc_cq_max: int,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    spread: int = 2,
) -> list[CrfProposal]:
    """Propose x265 CRFs anchored on measured NVENC VMAF, not CQ+offset.

    CQ and CRF are not codec-equivalent. Instead:
      1. Read the NVENC VMAF curve (best s_f + headroom to the gate).
      2. Pick target VMAFs: match the NVENC winner, then push toward the gate.
      3. Find the NVENC CQ at each target VMAF, then map that CQ's *relative*
         position in the NVENC search band into the x265 CRF band.

    This matched CQ 33 (neg≈90.6) → CRF ≈31 on the recorded run, instead of
    the broken fixed offset CQ 33 → CRF 27 (neg≈95).
    """
    if count < 1:
        return []
    pts = [o for o in observations if o.encode_ok and o.vmaf > 0]
    if not pts:
        return []

    thr = float(vmaf_threshold)
    anchor = _pick_anchor(pts, thr)
    headroom = float(anchor.vmaf) - thr

    # Target VMAFs: match winner, step down toward the gate, then near-gate.
    targets: list[tuple[float, str]] = [
        (float(anchor.vmaf), f"match_nvenc CQ{anchor.cq} V={anchor.vmaf:.1f}"),
    ]
    if headroom > 2.0:
        mid_v = max(thr + 1.0, float(anchor.vmaf) - 2.0)
        if abs(mid_v - anchor.vmaf) >= 0.75:
            targets.append((mid_v, f"push_compression V≈{mid_v:.1f}"))
    cliff_v = thr + 2.0
    if cliff_v < float(anchor.vmaf) - 0.5:
        targets.append((cliff_v, f"near_gate V≈{cliff_v:.1f}"))

    spread = max(1, int(spread))
    lo_c, hi_c = (
        (nvenc_cq_min, nvenc_cq_max)
        if nvenc_cq_min <= nvenc_cq_max
        else (nvenc_cq_max, nvenc_cq_min)
    )
    lo_r, hi_r = (crf_min, crf_max) if crf_min <= crf_max else (crf_max, crf_min)

    proposals: list[CrfProposal] = []
    seen: set[int] = set()

    def _add(crf: int, reason: str, target_vmaf: float) -> None:
        crf = min(max(int(crf), lo_r), hi_r)
        if crf in seen:
            return
        seen.add(crf)
        proposals.append(CrfProposal(crf=crf, reason=reason, target_vmaf=target_vmaf))

    for target_v, tag in targets:
        cq_at = interpolate_cq_for_vmaf(pts, target_v)
        if cq_at is None:
            cq_at = float(anchor.cq)
        crf = _range_map(cq_at, lo_c, hi_c, lo_r, hi_r)
        _add(crf, f"vmaf_anchor {tag} CQ≈{cq_at:.1f}→CRF{crf}", target_v)
        if len(proposals) >= count:
            break

    # Expand around the primary (match) CRF with upward bias toward the cliff.
    if proposals and len(proposals) < count:
        primary = proposals[0].crf
        for i in range(count * 2):
            # Prefer higher CRF (more compression) when filling.
            offset = ((i // 2) + 1) * spread if i % 2 == 0 else -((i // 2) + 1) * spread
            # First extras: +spread, -spread/2-ish via smaller steps once spread filled.
            if i == 0:
                offset = spread
            elif i == 1:
                offset = -spread
            _add(
                primary + offset,
                f"band_around CRF{primary}{offset:+d}",
                float(anchor.vmaf),
            )
            if len(proposals) >= count:
                break

    proposals.sort(key=lambda p: p.crf)
    return proposals[:count]


def map_cq_to_crf(cq: int, offset: int, crf_min: int, crf_max: int) -> int:
    """Legacy fixed-offset CQ→CRF map (kept for tests / explicit overrides).

    Prefer ``propose_vmaf_anchored_crfs`` for refine — CQ and CRF are not
    codec-equivalent.
    """
    lo, hi = (crf_min, crf_max) if crf_min <= crf_max else (crf_max, crf_min)
    return max(lo, min(hi, int(cq) + int(offset)))


def map_handoff_cqs_to_crfs(
    handoff: Sequence[HandoffCq],
    *,
    offset: int,
    crf_min: int,
    crf_max: int,
) -> list[tuple[int, HandoffCq]]:
    """Legacy CQ+offset mapper. Prefer ``propose_vmaf_anchored_crfs``."""
    out: list[tuple[int, HandoffCq]] = []
    seen: set[int] = set()
    for hc in handoff:
        crf = map_cq_to_crf(hc.cq, offset, crf_min, crf_max)
        if crf in seen:
            continue
        seen.add(crf)
        out.append((crf, hc))
    return out


def _feature_levels(features: Optional[dict[str, Any]]) -> dict[str, float]:
    """Normalized 0..1 levels for NVENC param rules."""
    f = features or {}
    if "motion_level" in f:
        return {
            "motion": float(f.get("motion_level", 0.0) or 0.0),
            "texture": float(f.get("texture_level", 0.0) or 0.0),
            "noise": float(f.get("noise_level_norm", 0.0) or 0.0),
            "edge": float(f.get("edge_level", 0.0) or 0.0),
            "cuts": float(f.get("cut_level", 0.0) or 0.0),
            "fps": float(f.get("fps", 30.0) or 30.0),
        }
    # Fallback for legacy summaries without normalized fields (soft midpoints).
    from feature_extractor import (
        _EDGE_DENSITY_MID,
        _MOTION_P90_MID,
        _NOISE_LEVEL_MID,
        _TEXTURE_MID,
        soft_level,
    )

    return {
        "motion": soft_level(float(f.get("motion_p90", 0.0) or 0.0), _MOTION_P90_MID),
        "texture": soft_level(float(f.get("texture", 0.0) or 0.0), _TEXTURE_MID),
        "noise": soft_level(float(f.get("noise_level", 0.0) or 0.0), _NOISE_LEVEL_MID),
        "edge": soft_level(float(f.get("edge_density", 0.0) or 0.0), _EDGE_DENSITY_MID),
        "cuts": min(float(f.get("cut_rate", 0.0) or 0.0) / 0.5, 1.0),
        "fps": float(f.get("fps", 30.0) or 30.0),
    }


def propose_feature_nvenc_baseline(
    features: Optional[dict[str, Any]],
    *,
    fps: Optional[float] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Derive NVENC baseline knobs from features. Never sets CQ/CRF.

    Returns (overrides_dict, reason_lines). Only keys that should be applied.
    """
    lvl = _feature_levels(features)
    motion = lvl["motion"]
    texture = lvl["texture"]
    noise = lvl["noise"]
    edge = lvl["edge"]
    cuts = lvl["cuts"]
    fps_v = float(fps if fps is not None else lvl["fps"])

    out: dict[str, Any] = {}
    reasons: list[str] = []

    # Noise wins AQ conflicts: grain burns bits and hurts VMAF-NEG.
    # Soft-norm mid≈0.5 on the calibrated corpus; thresholds sit around mid.
    if noise >= 0.55:
        out["nvenc_aq_strength"] = 4
        out["nvenc_temporal_aq"] = False
        out["nvenc_spatial_aq"] = True  # keep mild spatial AQ with low strength
        reasons.append(f"noise={noise:.2f} → aq=4, temporal_aq=off")
    elif noise >= 0.50:
        out["nvenc_aq_strength"] = 5
        out["nvenc_temporal_aq"] = False
        reasons.append(f"noise={noise:.2f} → aq=5, temporal_aq=off")
    elif texture >= 0.50 or edge >= 0.50:
        out["nvenc_spatial_aq"] = True
        out["nvenc_aq_strength"] = 12 if max(texture, edge) >= 0.55 else 10
        reasons.append(
            f"texture={texture:.2f}/edge={edge:.2f} → "
            f"spatial_aq=on, aq={out['nvenc_aq_strength']}"
        )
    else:
        out["nvenc_spatial_aq"] = True
        out["nvenc_aq_strength"] = 8
        out["nvenc_temporal_aq"] = True
        reasons.append("balanced → aq=8, spatial/temporal_aq=on")

    # Motion: B-frames + lookahead (+ temporal AQ unless noise already forced it off).
    # Soft levels in this corpus span ~0.46..0.54, so "high" is above mid (0.52).
    if motion >= 0.48:
        bf = 4 if motion >= 0.52 else 3
        la = 32 if fps_v >= 50 else 20
        out["nvenc_bf"] = bf
        out["nvenc_rc_lookahead"] = la
        out["nvenc_b_ref_mode"] = "middle"
        if "nvenc_temporal_aq" not in out:
            out["nvenc_temporal_aq"] = True
        reasons.append(f"motion={motion:.2f} → bf={bf}, lookahead={la}, b_ref=middle")
    elif motion < 0.45 and noise < 0.48:
        # Near-static: fewer B-frames, low lookahead.
        out["nvenc_bf"] = 2
        out["nvenc_rc_lookahead"] = 8
        out["nvenc_b_ref_mode"] = "disabled"
        reasons.append(f"motion={motion:.2f} → bf=2, lookahead=8")

    # GOP from average segment length (frames), clamped.
    # Short mashup segments → shorter GOP; long scenes → up to 2*fps.
    f = features or {}
    seg_n = max(1.0, float(f.get("segment_count", 0.0) or 0.0))
    duration = float(f.get("duration", 0.0) or 0.0)
    if duration <= 0 and fps_v > 0:
        # Fallback if duration missing
        duration = 0.0
    if duration > 0 and fps_v > 0:
        avg_seg_sec = duration / seg_n
        gop_raw = int(round(avg_seg_sec * fps_v))
        gop_min = max(24, int(round(fps_v)))          # ~1s
        gop_max = max(gop_min, int(round(2.0 * fps_v)))  # ~2s
        gop = min(max(gop_raw, gop_min), gop_max)
        out["nvenc_gop"] = gop
        reasons.append(
            f"segments={int(seg_n)} avg={avg_seg_sec:.2f}s → "
            f"gop={gop} (raw={gop_raw}, clamp {gop_min}..{gop_max})"
        )
        # Cut-heavy: ensure enough lookahead for scene decisions.
        if cuts >= 0.35:
            la = max(int(out.get("nvenc_rc_lookahead", 0) or 0), 20)
            out["nvenc_rc_lookahead"] = la
            reasons.append(f"cuts={cuts:.2f} → lookahead>={la}")

    return out, reasons


def _feature_aq_mode(
    *,
    texture: float,
    edge: float,
    motion: float,
    mashup: float,
    noise: float,
) -> tuple[int, str]:
    """Pick x265 ``aq-mode`` from content features.

    Fleet eval: aq-mode=1 suits high-texture / hard mashups (v1–v3); aq-mode=2
    suits lower-detail sources (v4–v5) for better compression at similar VMAF.
    """
    if noise >= 0.50:
        return 1, f"noise={noise:.2f} → aq-mode=1"
    if texture >= 0.87 or edge >= 0.62 or mashup >= 0.52:
        return (
            1,
            f"texture={texture:.2f}/edge={edge:.2f}/mashup={mashup:.2f} → aq-mode=1",
        )
    if texture < 0.86 and edge < 0.55:
        return 2, f"texture={texture:.2f}/edge={edge:.2f} → aq-mode=2"
    if motion >= 0.68 and texture >= 0.84:
        return 1, f"motion={motion:.2f}/texture={texture:.2f} → aq-mode=1"
    return 2, "balanced → aq-mode=2"


_AQ_STRENGTH_MIN = 0.8
_AQ_STRENGTH_MAX = 1.4


def _feature_aq_strength(
    features: Optional[dict[str, Any]],
    *,
    texture: float,
    edge: float,
    motion: float,
    noise: float,
) -> tuple[float, list[str]]:
    """Additive aq-strength rules from video features (clamped)."""
    f = features or {}
    flatness = min(max(float(f.get("flatness", 0.0) or 0.0), 0.0), 1.0)
    entropy = min(max(float(f.get("entropy", 0.0) or 0.0), 0.0), 1.0)
    luma_mean = float(f.get("luma_mean", 128.0) or 128.0)

    strength = 1.0
    reasons: list[str] = []

    if texture >= 0.75:
        strength += 0.35
        reasons.append(f"very_high_texture={texture:.2f} +0.35")
    elif texture >= 0.55:
        strength += 0.20
        reasons.append(f"high_texture={texture:.2f} +0.20")
    elif texture < 0.30:
        strength -= 0.20
        reasons.append(f"very_low_texture={texture:.2f} -0.20")
    elif texture < 0.40:
        strength -= 0.10
        reasons.append(f"low_texture={texture:.2f} -0.10")

    if texture < 0.75 and edge >= 0.62:
        strength += 0.05
        reasons.append(f"high_edge={edge:.2f} +0.05")
    if entropy >= 0.65 and texture < 0.75:
        strength += 0.05
        reasons.append(f"high_entropy={entropy:.2f} +0.05")

    if noise >= 0.55:
        strength -= 0.30
        reasons.append(f"very_high_noise={noise:.2f} -0.30")
    elif noise >= 0.50:
        strength -= 0.20
        reasons.append(f"high_noise={noise:.2f} -0.20")
    elif noise >= 0.40:
        strength -= 0.10
        reasons.append(f"medium_noise={noise:.2f} -0.10")
    elif noise < 0.25:
        strength += 0.10
        reasons.append(f"very_low_noise={noise:.2f} +0.10")

    if motion >= 0.68:
        strength += 0.10
        reasons.append(f"very_high_motion={motion:.2f} +0.10")
    elif motion >= 0.52:
        strength += 0.05
        reasons.append(f"high_motion={motion:.2f} +0.05")

    if flatness >= 0.70:
        strength -= 0.20
        reasons.append(f"extremely_flat={flatness:.2f} -0.20")
    elif flatness >= 0.55:
        strength -= 0.10
        reasons.append(f"large_flat={flatness:.2f} -0.10")

    if noise < 0.50:
        if luma_mean < 60:
            strength += 0.20
            reasons.append(f"very_dark_luma={luma_mean:.0f} +0.20")
        elif luma_mean < 90:
            strength += 0.10
            reasons.append(f"dark_luma={luma_mean:.0f} +0.10")

    pre_archetype = strength
    if motion >= 0.52 and texture >= 0.55:
        strength = max(strength, 1.2)
        if strength > pre_archetype:
            reasons.append("sports_like → floor 1.2")
    if texture >= 0.87 and edge >= 0.60:
        pre_forest = strength
        strength = max(strength, 1.3)
        if strength > pre_forest:
            reasons.append("forest_like → floor 1.3")
    if flatness >= 0.60 and motion < 0.45 and texture < 0.55:
        strength = min(strength, 1.0)
        reasons.append("anime_flat → cap 1.0")
    if flatness >= 0.55 and motion < 0.40 and edge < 0.35:
        strength = min(strength, 1.0)
        reasons.append("screen_like → cap 1.0")
    if luma_mean < 90 and noise >= 0.45:
        strength = min(strength, 1.0)
        reasons.append("dark_noisy → cap 1.0")
    elif luma_mean < 90 and noise < 0.40:
        pre_dark = strength
        strength = max(strength, 1.15)
        if strength > pre_dark:
            reasons.append("dark_clean → floor 1.15")

    strength = round(min(max(strength, _AQ_STRENGTH_MIN), _AQ_STRENGTH_MAX), 2)
    reasons.insert(0, f"aq-strength={strength:g} (base 1.0)")
    return strength, reasons


def propose_feature_x265_params(
    features: Optional[dict[str, Any]],
    *,
    fps: Optional[float] = None,
    quality_pack: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Derive libx265 ``-x265-params`` knobs from features. Never sets CRF.

    Returns (params_dict, reason_lines). Values are suitable for a colon-joined
    ``-x265-params`` string via ``format_x265_params``.
    """
    lvl = _feature_levels(features)
    motion = lvl["motion"]
    texture = lvl["texture"]
    noise = lvl["noise"]
    edge = lvl["edge"]
    cuts = lvl["cuts"]
    fps_v = float(fps if fps is not None else lvl["fps"])
    f = features or {}
    worst = float(f.get("worst_difficulty", 0.0) or 0.0)
    hard_frac = float(f.get("hard_fraction", 0.0) or 0.0)
    volatility = float(f.get("volatility", 0.0) or 0.0)
    mashup = 0.50 * worst + 0.30 * volatility + 0.20 * hard_frac

    aq_mode, aq_mode_reason = _feature_aq_mode(
        texture=texture,
        edge=edge,
        motion=motion,
        mashup=mashup,
        noise=noise,
    )
    aq_strength, aq_strength_reasons = _feature_aq_strength(
        features,
        texture=texture,
        edge=edge,
        motion=motion,
        noise=noise,
    )

    out: dict[str, Any] = {
        "aq-mode": aq_mode,
        "aq-strength": aq_strength,
        "rd": 6 if quality_pack else 5,
        "ref": 5 if quality_pack else 4,
        "bframes": 6,
        "rc-lookahead": 40,
        "keyint": 48,
        "min-keyint": 1,
        "scenecut": 40,
    }
    reasons: list[str] = [aq_mode_reason, *aq_strength_reasons]

    # Motion: B-frames + lookahead + refs.
    if motion >= 0.52:
        out["bframes"] = 8 if quality_pack else 7
        out["rc-lookahead"] = 60 if fps_v >= 50 else 50
        out["ref"] = max(int(out["ref"]), 5)
        reasons.append(
            f"motion={motion:.2f} → bframes={out['bframes']}, "
            f"lookahead={out['rc-lookahead']}, ref={out['ref']}"
        )
    elif motion >= 0.48:
        out["bframes"] = 6
        out["rc-lookahead"] = 40 if fps_v < 50 else 50
        reasons.append(
            f"motion={motion:.2f} → bframes=6, lookahead={out['rc-lookahead']}"
        )
    elif motion < 0.45 and noise < 0.48:
        out["bframes"] = 4
        out["rc-lookahead"] = 20
        reasons.append(f"motion={motion:.2f} → bframes=4, lookahead=20")

    # Hard mashup: spend a bit more on RD / refs for efficiency.
    if mashup >= 0.60 or quality_pack:
        out["rd"] = max(int(out["rd"]), 6)
        out["ref"] = max(int(out["ref"]), 5)
        if quality_pack:
            out["me"] = "umh"
            out["subme"] = 7
        reasons.append(
            f"mashup={mashup:.2f} → rd={out['rd']}, ref={out['ref']}"
            + (" +me/subme" if quality_pack else "")
        )

    # GOP / scenecut from average segment length (same idea as NVENC).
    gop = gop_from_segments(features, fps=fps_v)
    if gop is not None:
        out["keyint"] = gop
        out["min-keyint"] = 1
        reasons.append(f"segments → keyint={gop}")
    if cuts >= 0.35:
        out["scenecut"] = 60
        out["rc-lookahead"] = max(int(out["rc-lookahead"]), 40)
        reasons.append(f"cuts={cuts:.2f} → scenecut=60, lookahead>={out['rc-lookahead']}")
    elif cuts >= 0.20:
        out["scenecut"] = 50
        reasons.append(f"cuts={cuts:.2f} → scenecut=50")

    return out, reasons


def format_x265_params(params: dict[str, Any]) -> str:
    """Colon-join an x265-params dict. Rejects any CRF key."""
    banned = {"crf", "qp", "bitrate", "vbv-maxrate", "vbv-bufsize"}
    parts: list[str] = []
    for key, val in params.items():
        k = str(key).strip()
        if k.lower() in banned:
            raise ValueError(f"format_x265_params must not set rate-control key {k!r}")
        if isinstance(val, bool):
            parts.append(f"{k}={1 if val else 0}")
        elif isinstance(val, float):
            parts.append(f"{k}={val:g}")
        else:
            parts.append(f"{k}={val}")
    return ":".join(parts)


def parse_x265_params(params: str) -> dict[str, str]:
    """Parse a colon-joined ``-x265-params`` string into key/value pairs."""
    out: dict[str, str] = {}
    for part in str(params or "").split(":"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
        else:
            out[part] = "1"
    return out


def merge_x265_params(base: str, override: str) -> str:
    """Merge two x265-params strings; ``override`` wins on duplicate keys."""
    override = str(override or "").strip()
    if not override:
        return str(base or "").strip()
    base = str(base or "").strip()
    if not base:
        return override
    merged = {**parse_x265_params(base), **parse_x265_params(override)}
    return ":".join(f"{key}={value}" for key, value in merged.items())


def gop_from_segments(
    features: Optional[dict[str, Any]],
    *,
    fps: Optional[float] = None,
) -> Optional[int]:
    """Average-segment GOP in frames, clamped to [fps, 2*fps] (min 24)."""
    f = features or {}
    fps_v = float(fps if fps is not None else (f.get("fps") or 30.0))
    seg_n = max(1.0, float(f.get("segment_count", 0.0) or 0.0))
    duration = float(f.get("duration", 0.0) or 0.0)
    if duration <= 0 or fps_v <= 0:
        return None
    avg_seg_sec = duration / seg_n
    gop_raw = int(round(avg_seg_sec * fps_v))
    gop_min = max(24, int(round(fps_v)))
    gop_max = max(gop_min, int(round(2.0 * fps_v)))
    return min(max(gop_raw, gop_min), gop_max)


def apply_feature_nvenc_baseline(
    req: Any,
    features: Optional[dict[str, Any]],
) -> list[str]:
    """Mutate ``req`` NVENC fields from features. Returns human-readable change log.

    No-ops when encoder is not hevc_nvenc, feature baseline is disabled, or
    features are empty. Never touches CQ/CRF search bounds.
    """
    if getattr(req, "encoder", "") != "hevc_nvenc":
        return []
    if not getattr(req, "nvenc_feature_baseline", True):
        return ["feature baseline disabled"]
    if not features:
        return ["no features — keeping request NVENC defaults"]

    overrides, reasons = propose_feature_nvenc_baseline(features)
    changes: list[str] = []
    for key, new_val in overrides.items():
        old_val = getattr(req, key, None)
        if old_val == new_val:
            continue
        setattr(req, key, new_val)
        changes.append(f"{key}: {old_val!r} → {new_val!r}")

    if not changes:
        return reasons + ["(no field changes vs request defaults)"]
    return reasons + changes


def _nvenc_param_variants(
    baseline: dict[str, Any],
    features: Optional[dict[str, Any]] = None,
) -> list[tuple[NvencOverrides, str]]:
    """Feature-ranked NVENC tweaks at Round-1 best CQ (never proposes CQ).

    Features only choose *which* params to try; CQ remains a pure search variable.
    """
    multipass = str(baseline.get("nvenc_multipass", "qres"))
    spatial = bool(baseline.get("nvenc_spatial_aq", True))
    temporal = bool(baseline.get("nvenc_temporal_aq", True))
    aq = int(baseline.get("nvenc_aq_strength", 8))
    lookahead = int(baseline.get("nvenc_rc_lookahead", 0) or 0)
    bf = int(baseline.get("nvenc_bf", 0) or 0)
    gop = baseline.get("nvenc_gop")
    gop_i = int(gop) if gop is not None else None
    b_ref = str(baseline.get("nvenc_b_ref_mode", "disabled") or "disabled")

    lvl = _feature_levels(features)
    motion, texture, noise, edge, cuts, fps = (
        lvl["motion"],
        lvl["texture"],
        lvl["noise"],
        lvl["edge"],
        lvl["cuts"],
        lvl["fps"],
    )

    scored: list[tuple[float, NvencOverrides, str]] = []

    def add(priority: float, ov: NvencOverrides, reason: str) -> None:
        scored.append((priority, ov, reason))

    # Always useful: better multipass at fixed CQ.
    if multipass != "fullres":
        add(0.55, NvencOverrides(nvenc_multipass="fullres"), "feat_multipass_fullres")

    # High noise → lower AQ / temporal AQ off (bits spent on grain hurt NEG/s_f).
    if noise >= 0.4:
        aq_down = max(1, aq - 4)
        if aq_down != aq:
            add(0.95 + 0.05 * noise, NvencOverrides(nvenc_aq_strength=aq_down), f"feat_noise_aq_{aq_down}")
        if temporal:
            add(0.85 + 0.1 * noise, NvencOverrides(nvenc_temporal_aq=False), "feat_noise_taq_off")

    # High texture / edges (and not noise-dominated) → stronger spatial AQ.
    if (texture >= 0.4 or edge >= 0.4) and noise < 0.55:
        aq_up = min(15, aq + 4)
        if aq_up != aq:
            add(
                0.8 + 0.15 * max(texture, edge),
                NvencOverrides(nvenc_aq_strength=aq_up, nvenc_spatial_aq=True),
                f"feat_texture_aq_{aq_up}",
            )
        if not spatial:
            add(0.7, NvencOverrides(nvenc_spatial_aq=True), "feat_texture_saq_on")

    # High motion → temporal AQ, B-frames, lookahead.
    if motion >= 0.4:
        if not temporal and noise < 0.4:
            add(0.88, NvencOverrides(nvenc_temporal_aq=True), "feat_motion_taq_on")
        bf_target = 4 if motion >= 0.7 else 3
        if bf < bf_target:
            add(0.75 + 0.1 * motion, NvencOverrides(nvenc_bf=bf_target), f"feat_motion_bf_{bf_target}")
        la_target = 32 if fps >= 50 else 20
        if lookahead < la_target:
            add(
                0.7 + 0.1 * motion,
                NvencOverrides(nvenc_rc_lookahead=la_target),
                f"feat_motion_la_{la_target}",
            )
        if bf_target > 0 and b_ref == "disabled":
            add(0.6, NvencOverrides(nvenc_b_ref_mode="middle"), "feat_motion_bref_middle")
        elif b_ref == "middle":
            add(0.25, NvencOverrides(nvenc_b_ref_mode="each"), "feat_bref_each")

    # Frequent cuts → ensure GOP not longer than ~1s; otherwise trust baseline.
    if cuts >= 0.35:
        gop_target = max(24, int(round(fps)))
        if gop_i is None or gop_i > gop_target:
            add(0.78 + 0.1 * cuts, NvencOverrides(nvenc_gop=gop_target), f"feat_cuts_gop_{gop_target}")
        la_target = max(20, lookahead)
        if lookahead < la_target:
            add(0.65, NvencOverrides(nvenc_rc_lookahead=la_target), f"feat_cuts_la_{la_target}")
    else:
        # Mild refine: try 2*fps if baseline used a shorter clamp.
        gop_alt = gop_from_segments(features, fps=fps)
        if gop_alt is not None and gop_i is not None and gop_i < gop_alt:
            add(0.35, NvencOverrides(nvenc_gop=gop_alt), f"feat_gop_alt_{gop_alt}")

    # Generic fallbacks when features are weak / flat.
    # RC stays fixed as request nvenc_rc (typically vbr); do not propose vbr_hq —
    # p1..p7 presets reject legacy VBR_HQ on this ffmpeg.
    if temporal and noise < 0.4:
        add(0.3, NvencOverrides(nvenc_temporal_aq=False), "feat_taq_off")
    if spatial and texture < 0.35 and noise < 0.4:
        add(0.25, NvencOverrides(nvenc_spatial_aq=False), "feat_saq_off")

    scored.sort(key=lambda x: (-x[0], x[2]))
    seen: set[str] = set()
    unique: list[tuple[NvencOverrides, str]] = []
    for _pri, ov, reason in scored:
        key = ov.suffix()
        if key == "base" or key in seen:
            continue
        seen.add(key)
        unique.append((ov, reason))
    return unique


def propose_round2_mixed(
    observations: Sequence[CqObservation],
    *,
    crf_min: int,
    crf_max: int,
    vmaf_threshold: float,
    used: set[int],
    param_trials: int,
    cq_trials: int,
    baseline_nvenc: dict[str, Any],
    features: Optional[dict[str, Any]] = None,
    preprocess_trial: Optional[str] = None,
) -> list[Round2TrialSpec]:
    """Round 2: lock best CQ, tune NVENC knobs from features, then refine nearby CQs.

    If ``preprocess_trial`` is set, one measured light-denoise trial is added at
    the locked best CQ (kept only if it improves s_f without tripping gates).
    """
    pts = [o for o in observations if o.encode_ok]
    if not pts:
        return []

    anchor = _pick_anchor(pts, float(vmaf_threshold))
    best_cq = anchor.cq
    specs: list[Round2TrialSpec] = []

    if preprocess_trial:
        specs.append(
            Round2TrialSpec(
                cq=best_cq,
                nvenc=NvencOverrides(preprocess=preprocess_trial),
                reason=f"preprocess_experiment:{preprocess_trial}",
                predicted_s_f=anchor.s_f,
            )
        )

    for ov, reason in _nvenc_param_variants(baseline_nvenc, features)[: max(0, param_trials)]:
        specs.append(
            Round2TrialSpec(
                cq=best_cq,
                nvenc=ov,
                reason=reason,
                predicted_s_f=anchor.s_f,
            )
        )

    cq_props = propose_round2_details(
        observations,
        count=max(0, cq_trials),
        crf_min=crf_min,
        crf_max=crf_max,
        vmaf_threshold=vmaf_threshold,
        used=used,
    )
    for prop in cq_props:
        specs.append(
            Round2TrialSpec(
                cq=prop.cq,
                reason=f"refine_cq:{prop.reason}",
                predicted_s_f=prop.predicted_s_f,
            )
        )

    return specs


def observations_from_trials(trials: Sequence[object]) -> list[CqObservation]:
    """Build observations from TrialResult-like objects."""
    out: list[CqObservation] = []
    for trial in trials:
        crf = getattr(trial, "crf", None)
        if crf is None:
            continue
        score = getattr(trial, "score", None)
        encode_ok = bool(getattr(trial, "encode_ok", False))
        if score is None:
            continue
        out.append(
            CqObservation(
                cq=int(crf),
                vmaf=float(getattr(score, "vmaf", 0.0) or 0.0),
                compression_rate=float(getattr(score, "compression_rate", 1.0) or 1.0),
                compression_ratio=float(getattr(score, "compression_ratio", 1.0) or 1.0),
                s_f=float(getattr(score, "s_f", 0.0) or 0.0),
                encode_ok=encode_ok,
            )
        )
    return out
