"""HEVC recipe priors for mashup-style challenges.

Features nudge the CRF seed and (when enabled) set non-CRF ``-x265-params``.
Segment-aware summary fields preferred:
  cut_rate, hard_fraction, worst_difficulty, volatility

Default strategy: one medium preset + three quality-biased CRF candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class HevcRecipe:
    name: str
    preset: str
    params: str
    crf_start: int


# Default encode recipe: medium is a good speed/quality tradeoff for parallel trials.
DEFAULT_MEDIUM = HevcRecipe(
    name="default_medium",
    preset="medium",
    params="aq-mode=3:rd=5:ref=4:bframes=6:rc-lookahead=40:keyint=48:min-keyint=1:scenecut=40",
    crf_start=28,
)

# Optional slower pack if max_recipes > 1.
DEFAULT_QUALITY = HevcRecipe(
    name="default_quality",
    preset="slow",
    params="aq-mode=3:rd=6:ref=5:bframes=8:rc-lookahead=60:me=umh:subme=7:keyint=48:min-keyint=1:scenecut=40",
    crf_start=28,
)


def crf_seed_for_threshold(vmaf_threshold: int) -> int:
    if vmaf_threshold >= 93:
        return 14
    if vmaf_threshold >= 89:
        return 16
    return 18


def adjust_crf_for_volatility(seed: int, features: dict[str, Any]) -> int:
    """Bias CRF safer on jumpy / hard mashups."""
    volatility = float(features.get("volatility", 0.0))
    cut_density = float(features.get("cut_density", 0.0))
    cut_rate = float(features.get("cut_rate", 0.0))
    hard_fraction = float(features.get("hard_fraction", 0.0))
    worst = float(features.get("worst_difficulty", 0.0))
    texture = float(features.get("texture", features.get("texture_lbp", 0.0)))

    bias = 0
    if worst >= 0.7 or hard_fraction >= 0.45 or cut_rate >= 0.35 or volatility >= 0.55:
        bias = -3
    elif worst >= 0.55 or hard_fraction >= 0.30 or cut_rate >= 0.20 or volatility >= 0.35:
        bias = -2
    elif worst >= 0.40 or cut_density >= 0.15 or volatility >= 0.20:
        bias = -1

    if texture >= 1.2 or hard_fraction >= 0.5:
        bias -= 1

    return max(8, seed + bias)


def candidate_crfs(
    seed: int,
    crf_min: int,
    crf_max: int,
    *,
    count: int = 3,
    spread: int = 2,
) -> list[int]:
    """Pick ``count`` quality-biased CRFs ending at ``seed``.

    Default for count=3, spread=2: ``[seed-4, seed-2, seed]`` (clamped).
    If clamping collapses values near the edges, fill with nearest unused CRFs.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if crf_min > crf_max:
        raise ValueError("crf_min must be <= crf_max")

    center = min(max(int(seed), crf_min), crf_max)
    spread = max(1, int(spread))

    if count == 1:
        return [center]

    # Prefer quality (lower CRF). The highest candidate is the feature-biased
    # seed; the others move toward visually safer encodes.
    offsets = [-(count - 1 - i) * spread for i in range(count)]

    ordered: list[int] = []
    seen: set[int] = set()
    for offset in offsets:
        crf = min(max(center + offset, crf_min), crf_max)
        if crf not in seen:
            seen.add(crf)
            ordered.append(crf)

    # Fill gaps if clamping removed uniqueness (seed near bounds).
    step = 1
    while len(ordered) < count and step <= (crf_max - crf_min + 1):
        for side in (-step, step):
            cand = center + side
            if crf_min <= cand <= crf_max and cand not in seen:
                seen.add(cand)
                ordered.append(cand)
                if len(ordered) >= count:
                    break
        step += 1

    return sorted(ordered[:count])


def _feature_x265_params_string(
    features: Optional[dict[str, Any]],
    *,
    quality_pack: bool = False,
) -> tuple[str, list[str]]:
    """Build ``-x265-params`` from features (no CRF). Falls back to defaults."""
    from interp_search import format_x265_params, propose_feature_x265_params

    if not features:
        base = DEFAULT_QUALITY.params if quality_pack else DEFAULT_MEDIUM.params
        return base, ["no features — static default params"]
    params, reasons = propose_feature_x265_params(
        features, quality_pack=quality_pack
    )
    return format_x265_params(params), reasons


def select_recipes(
    features: dict[str, Any],
    vmaf_threshold: int,
    max_recipes: int = 1,
    preset: str = "medium",
    *,
    feature_baseline: bool = True,
    params_override: Optional[str] = None,
) -> list[HevcRecipe]:
    from interp_search import merge_x265_params

    seed = adjust_crf_for_volatility(crf_seed_for_threshold(vmaf_threshold), features)
    preset = (preset or "medium").lower().strip()

    if feature_baseline:
        primary_params, _ = _feature_x265_params_string(features, quality_pack=False)
        quality_params, _ = _feature_x265_params_string(features, quality_pack=True)
    else:
        primary_params = DEFAULT_MEDIUM.params
        quality_params = DEFAULT_QUALITY.params

    if params_override:
        primary_params = merge_x265_params(primary_params, params_override)
        quality_params = merge_x265_params(quality_params, params_override)

    packs = [
        HevcRecipe(
            name=f"default_{preset}",
            preset=preset,
            params=primary_params,
            crf_start=seed,
        ),
    ]
    if max_recipes >= 2:
        packs.append(
            HevcRecipe(
                name=DEFAULT_QUALITY.name,
                preset=DEFAULT_QUALITY.preset,
                params=quality_params,
                crf_start=seed,
            )
        )
    return packs[: max(1, max_recipes)]


def describe_feature_x265_baseline(
    features: Optional[dict[str, Any]],
    *,
    quality_pack: bool = False,
) -> list[str]:
    """Human-readable feature → x265-params lines for logging."""
    params_str, reasons = _feature_x265_params_string(
        features, quality_pack=quality_pack
    )
    return reasons + [f"x265-params={params_str}"]


# Feature-driven preprocess (VMAF NEG survey): denoise, bilateral, mild
# sharpen, slight contrast. Soft-norm mid ≈ 0.5 = average for this corpus.
_PREPROCESS_NOISE_MED = 0.55
_PREPROCESS_NOISE_LIGHT = 0.52  # ~p75; avoid firing denoise on average clips
_PREPROCESS_MOTION_TEMPORAL = 0.55
_PREPROCESS_TEXTURE_TEMPORAL = 0.45
_PREPROCESS_EDGE_SHARPEN = 0.60
_PREPROCESS_NOISE_CLEAN = 0.48
_PREPROCESS_FLAT_CONTRAST = 0.55


def propose_preprocess_from_features(
    features: Optional[dict[str, Any]],
) -> tuple[str, str]:
    """Pick one survey-aligned preprocess preset from content features.

    Returns ``(preset_name, reason)``. Prefers mild denoise on noisy clips,
    bilateral on noisy+textured, very mild sharpen on edge-rich clean clips,
    slight contrast on flat clean clips; otherwise ``none``.
    """
    f = features or {}
    noise = float(f.get("noise_level_norm", f.get("noise", 0.0)) or 0.0)
    motion = float(f.get("motion_level", f.get("motion", 0.0)) or 0.0)
    texture = float(f.get("texture_level", f.get("texture", 0.5)) or 0.5)
    edge = float(f.get("edge_level", f.get("edge_density", 0.0)) or 0.0)
    flatness = float(f.get("flatness", 0.0) or 0.0)

    if noise >= _PREPROCESS_NOISE_MED:
        if motion >= _PREPROCESS_MOTION_TEMPORAL and texture < _PREPROCESS_TEXTURE_TEMPORAL:
            return (
                "atadenoise_light",
                f"noise={noise:.2f}/motion={motion:.2f}/texture={texture:.2f} "
                f"→ atadenoise_light",
            )
        if texture >= 0.55:
            return (
                "bilateral_light",
                f"noise={noise:.2f}/texture={texture:.2f} → bilateral_light",
            )
        return (
            "hqdn3d_med",
            f"noise={noise:.2f} ≥ {_PREPROCESS_NOISE_MED:.2f} → hqdn3d_med",
        )
    if noise >= _PREPROCESS_NOISE_LIGHT:
        return (
            "hqdn3d_light",
            f"noise={noise:.2f} ≥ {_PREPROCESS_NOISE_LIGHT:.2f} → hqdn3d_light",
        )
    # Clean: edge-rich → very mild sharpen; flat → slight contrast.
    if noise < _PREPROCESS_NOISE_CLEAN and edge >= _PREPROCESS_EDGE_SHARPEN:
        return (
            "unsharp_mild",
            f"noise={noise:.2f}/edge={edge:.2f} → unsharp_mild",
        )
    if noise < _PREPROCESS_NOISE_CLEAN and flatness >= _PREPROCESS_FLAT_CONTRAST:
        return (
            "contrast_mild",
            f"noise={noise:.2f}/flatness={flatness:.2f} → contrast_mild",
        )
    return ("none", f"noise={noise:.2f} edge={edge:.2f} → none")


def survey_preprocess_candidates(
    features: Optional[dict[str, Any]] = None,
    *,
    sweep: bool = False,
    brave: bool = False,
) -> tuple[list[Optional[str]], str]:
    """Build ordered preprocess candidates for VBR A/B or full survey sweep.

    When ``brave`` is True, try the micro-enhancement set tuned for higher
    VMAF NEG under the dual-model delta gate. When ``sweep`` is True, try the
    standard VMAF-NEG survey set. Otherwise return ``[none, feature_pick]``
    (deduped) for a cheap A/B.
    """
    from encoder import BRAVE_PREPROCESS_SWEEP, SURVEY_PREPROCESS_SWEEP

    if brave or sweep:
        names = list(BRAVE_PREPROCESS_SWEEP if brave else SURVEY_PREPROCESS_SWEEP)
        # Also include stronger denoise variants when features look noisy.
        primary, primary_reason = propose_preprocess_from_features(features)
        if primary not in names and primary != "none":
            names.append(primary)
        ordered: list[Optional[str]] = []
        seen: set[str] = set()
        for name in names:
            key = str(name).lower().strip()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalize_preprocess_name(key))
        label = "brave sweep" if brave else "survey sweep"
        return ordered, f"{label} ({primary_reason})"

    primary, reason = propose_preprocess_from_features(features)
    primary_n = normalize_preprocess_name(primary)
    ordered = [None]
    if primary_n is not None:
        ordered.append(primary_n)
    return ordered, reason


def normalize_preprocess_name(preset: Optional[str]) -> Optional[str]:
    """Map preset name to encode value: ``None`` means no filter."""
    if preset is None:
        return None
    key = str(preset).lower().strip()
    if key in {"", "none"}:
        return None
    return key


@dataclass(frozen=True)
class PreprocessScoreView:
    """Minimal score view for A/B winner selection (testable without ffmpeg)."""

    s_f: float
    vmaf: float
    gates_ok: bool


def preprocess_trial_better(
    challenger: PreprocessScoreView,
    incumbent: PreprocessScoreView,
) -> bool:
    """True when ``challenger`` should replace ``incumbent``.

    Rejects challengers that fail gates. Prefers higher ``s_f``, then higher
    VMAF when both ``s_f`` are zero/tied within epsilon.
    """
    if not challenger.gates_ok:
        return False
    if not incumbent.gates_ok:
        return True
    if challenger.s_f > incumbent.s_f + 1e-9:
        return True
    if abs(challenger.s_f - incumbent.s_f) <= 1e-9:
        return float(challenger.vmaf) > float(incumbent.vmaf) + 1e-9
    return False


def choose_preprocess_ab_winner(
    *,
    none_score: PreprocessScoreView,
    denoise_score: PreprocessScoreView,
    denoise_preset: str,
) -> tuple[Optional[str], str]:
    """Pick ``none`` vs one challenger after fixed-bitrate A/B.

    Returns ``(winning_preprocess_or_None, reason)``.
    """
    return choose_best_preprocess(
        [
            (None, none_score),
            (normalize_preprocess_name(denoise_preset), denoise_score),
        ]
    )


def choose_best_preprocess(
    trials: list[tuple[Optional[str], PreprocessScoreView]],
) -> tuple[Optional[str], str]:
    """Pick the best preprocess among scored trials (gates → s_f → VMAF)."""
    if not trials:
        return None, "no preprocess trials"
    best_name, best_score = trials[0]
    for name, score in trials[1:]:
        if preprocess_trial_better(score, best_score):
            best_name, best_score = name, score
    label = best_name or "none"
    usable = [(n or "none", s) for n, s in trials if s.gates_ok]
    detail = ", ".join(f"{n}:s_f={s.s_f:.4f}/vmaf={s.vmaf:.2f}" for n, s in usable) or "none usable"
    if not best_score.gates_ok:
        return None, f"preprocess sweep: no candidate passed gates ({detail})"
    return best_name, f"preprocess keep {label} ({detail})"


def resolve_vbr_preprocess(
    *,
    explicit: Optional[str],
    preprocess_auto: bool,
    features: Optional[dict[str, Any]],
    preprocess_sweep: bool = False,
    preprocess_brave: bool = False,
) -> tuple[Optional[str], str, list[Optional[str]]]:
    """Resolve preprocess for VBR.

    Returns ``(primary_preset_or_None, reason, candidates)``.
    ``candidates`` is what to encode when A/B or sweep is enabled (may be just
    ``[primary]`` when no comparison is needed).
    Explicit ``request.preprocess`` always wins over auto (single candidate).
    """
    if explicit is not None:
        key = str(explicit).lower().strip() or "none"
        if key in {"sweep", "brave"}:
            cands, reason = survey_preprocess_candidates(
                features,
                sweep=key == "sweep",
                brave=key == "brave",
            )
            primary = cands[0] if cands else None
            return primary, reason, cands
        normalized = normalize_preprocess_name(key)
        return (
            normalized,
            f"explicit preprocess={key}",
            [normalized],
        )
    if preprocess_brave or preprocess_sweep:
        cands, reason = survey_preprocess_candidates(
            features,
            sweep=preprocess_sweep and not preprocess_brave,
            brave=preprocess_brave,
        )
        primary = next((c for c in cands if c is not None), None)
        # Prefer feature primary as first non-none when present in list.
        feat_name, _ = propose_preprocess_from_features(features)
        feat_n = normalize_preprocess_name(feat_name)
        if feat_n is not None and feat_n in cands:
            # Move feature pick after none for evaluation order.
            rest = [c for c in cands if c not in (None, feat_n)]
            cands = [None, feat_n, *rest]
        return feat_n if feat_n is not None else primary, reason, cands
    if not preprocess_auto:
        return None, "preprocess_auto=false → none", [None]
    cands, reason = survey_preprocess_candidates(features, sweep=False)
    primary = cands[-1] if len(cands) > 1 else None
    return primary, reason, cands
