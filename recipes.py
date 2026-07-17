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
