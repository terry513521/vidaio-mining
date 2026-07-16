"""HEVC recipe priors for mashup-style challenges.

Features only nudge the CRF seed. Segment-aware summary fields preferred:
  cut_rate, hard_fraction, worst_difficulty, volatility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HevcRecipe:
    name: str
    preset: str
    params: str
    crf_start: int


DEFAULT_QUALITY = HevcRecipe(
    name="default_quality",
    preset="slow",
    params="aq-mode=3:rd=6:ref=5:bframes=8:rc-lookahead=60:me=umh:subme=7:keyint=48:min-keyint=1:scenecut=40",
    crf_start=28,
)

DEFAULT_BALANCE = HevcRecipe(
    name="default_balance",
    preset="medium",
    params="aq-mode=3:rd=5:ref=4:bframes=6:rc-lookahead=40:keyint=48:min-keyint=1:scenecut=40",
    crf_start=28,
)


def crf_seed_for_threshold(vmaf_threshold: int) -> int:
    if vmaf_threshold >= 93:
        return 24
    if vmaf_threshold >= 89:
        return 26
    return 28


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

    return max(18, seed + bias)


def select_recipes(
    features: dict[str, Any],
    vmaf_threshold: int,
    max_recipes: int = 2,
) -> list[HevcRecipe]:
    seed = adjust_crf_for_volatility(crf_seed_for_threshold(vmaf_threshold), features)

    packs = [
        HevcRecipe(
            name=DEFAULT_QUALITY.name,
            preset=DEFAULT_QUALITY.preset,
            params=DEFAULT_QUALITY.params,
            crf_start=seed,
        ),
        HevcRecipe(
            name=DEFAULT_BALANCE.name,
            preset=DEFAULT_BALANCE.preset,
            params=DEFAULT_BALANCE.params,
            crf_start=min(seed + 1, 40),
        ),
    ]
    return packs[: max(1, max_recipes)]
