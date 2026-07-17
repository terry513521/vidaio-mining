"""Python port of ab-av1 interpolated CRF search on scene samples."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from logutil import log


@dataclass
class CrfAttempt:
    crf: float
    q: int
    mean_vmaf: float
    per_sample_vmaf: list[float]
    encode_percent: float = 100.0
    encode_ok: bool = True
    error: str = ""


@dataclass
class CrfSearchResult:
    ok: bool
    crf: Optional[float]
    mean_vmaf: float
    encode_percent: float = 100.0
    attempts: list[CrfAttempt] = field(default_factory=list)
    reason: str = ""


class QualityConverter:
    """Map CRF ↔ integer q (low q = higher quality for x265)."""

    def __init__(self, *, crf_increment: float = 0.1, high_crf_means_hq: bool = False):
        self.crf_increment = max(0.001, float(crf_increment))
        self.high_crf_means_hq = high_crf_means_hq

    def q(self, crf: float) -> int:
        value = round(float(crf) / self.crf_increment)
        return -value if self.high_crf_means_hq else value

    def crf(self, q: int) -> float:
        pos_q = -q if self.high_crf_means_hq else q
        return round(pos_q * self.crf_increment, 3)

    def min_max_q(self, min_crf: float, max_crf: float) -> tuple[int, int]:
        if self.high_crf_means_hq:
            return self.q(max_crf), self.q(min_crf)
        return self.q(min_crf), self.q(max_crf)


def encoded_percent_size(sample_bytes: int, encoded_bytes: int) -> float:
    """Mirror ab-av1 ``EncodeResults::encoded_percent_size``."""
    if sample_bytes <= 0:
        return 100.0
    return encoded_bytes * 100.0 / float(sample_bytes)


def vmaf_lerp_q(
    min_vmaf: float,
    worse_q: int,
    worse_vmaf: float,
    better_q: int,
    better_vmaf: float,
) -> int:
    """Interpolate q toward ``min_vmaf`` between a too-low and too-high sample."""
    vmaf_diff = better_vmaf - worse_vmaf
    if vmaf_diff <= 1e-6:
        return (worse_q + better_q) // 2
    vmaf_factor = (min_vmaf - worse_vmaf) / vmaf_diff
    q_diff = worse_q - better_q
    lerp = int(round(worse_q - q_diff * vmaf_factor))
    return max(better_q + 1, min(worse_q - 1, lerp))


def _higher_tolerance(run: int, crf_increment: float, thorough: bool) -> float:
    if thorough:
        return 0.05
    return max(0.1, min(1.0, crf_increment) * (2 ** (run - 1)) * 0.1)


def _near_threshold(score: float, min_vmaf: float, band: float) -> bool:
    """True when VMAF is close enough that ±1 CRF steps beat interpolation."""
    return abs(float(score) - float(min_vmaf)) <= max(0.0, float(band))


def _sample_small_enough(encode_percent: float, max_encoded_percent: float) -> bool:
    return encode_percent <= max_encoded_percent


EvaluateFn = Callable[[float], CrfAttempt]


def search_crf(
    evaluate: EvaluateFn,
    *,
    min_vmaf: float,
    crf_min: float,
    crf_max: float,
    crf_increment: float = 0.1,
    max_encoded_percent: float = 80.0,
    thorough: bool = False,
    deadline: Optional[float] = None,
    max_runs: int = 16,
    initial_crf: Optional[float] = None,
    near_vmaf_band: float = 2.0,
) -> CrfSearchResult:
    """CRF search: coarse lerp when far, ±1 CRF steps when near the VMAF target.

    ``initial_crf`` (optional) seeds the first probe instead of the mid-band
    midpoint — useful when features / ``crf_start`` give a prior near the cliff.

    When ``|vmaf - min_vmaf| <= near_vmaf_band``, skip interpolation and nudge
    CRF by one increment (more stable on noisy full-file dual VMAF).
    """
    q_conv = QualityConverter(crf_increment=crf_increment)
    min_q, max_q = q_conv.min_max_q(crf_min, crf_max)
    if min_q >= max_q:
        return CrfSearchResult(False, None, 0.0, reason="invalid crf range")

    default_span = 46.0 - 10.0  # x265 defaults in ab-av1
    cut_on_iter2 = (crf_max - crf_min) > default_span * 0.5
    step = max(1, int(round(1.0 / max(crf_increment, 1e-6))))
    if initial_crf is not None:
        seed_crf = min(max(float(initial_crf), float(crf_min)), float(crf_max))
        q = q_conv.q(seed_crf)
        q = min(max(q, min_q), max_q)
    else:
        q = (min_q + max_q) // 2
    attempts: list[CrfAttempt] = []

    for run in range(1, max_runs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            return _finish_on_deadline(
                attempts,
                min_vmaf=min_vmaf,
                max_encoded_percent=max_encoded_percent,
                reason="deadline",
            )

        crf = q_conv.crf(q)
        attempt = evaluate(crf)
        attempt.q = q
        attempts.append(attempt)
        if not attempt.encode_ok or attempt.mean_vmaf <= 0:
            log(f"  crf-search CRF {crf:.1f}: encode/score failed ({attempt.error})")
            if q > min_q:
                q = max(min_q, q - step)
                continue
            return CrfSearchResult(
                False,
                None,
                0.0,
                attempts=attempts,
                reason=attempt.error or "evaluate failed",
            )

        score = attempt.mean_vmaf
        size_pct = attempt.encode_percent
        small_enough = _sample_small_enough(size_pct, max_encoded_percent)
        tol = _higher_tolerance(run, crf_increment, thorough)
        near = _near_threshold(score, min_vmaf, near_vmaf_band)
        log(
            f"  crf-search run {run}: CRF {crf:.1f} mean_vmaf={score:.2f} "
            f"size={size_pct:.1f}% (target VMAF {min_vmaf:.1f}, max {max_encoded_percent:.0f}%)"
            + (" [near→step]" if near else "")
        )

        if score > min_vmaf:
            if small_enough and score < min_vmaf + tol:
                return _done(attempt, attempts, reason="target_vmaf")
            u_bound = min((a for a in attempts if a.q > q), key=lambda a: a.q, default=None)
            if u_bound is not None and u_bound.q == q + step:
                if not small_enough:
                    return _no_good_crf(attempts, last=attempt)
                return _done(attempt, attempts, reason="bracket_upper")
            # Near target: nudge CRF up one step instead of interpolating.
            if near and q + step <= max_q:
                q = q + step
                continue
            if u_bound is not None:
                q = vmaf_lerp_q(
                    min_vmaf,
                    u_bound.q,
                    u_bound.mean_vmaf,
                    q,
                    score,
                )
            elif q == max_q:
                if not small_enough:
                    return _no_good_crf(attempts, last=attempt)
                return _done(attempt, attempts, reason="max_crf")
            elif cut_on_iter2 and run == 1 and q + step < max_q:
                q = int(round(q * 0.4 + max_q * 0.6))
            else:
                q = max_q
        else:
            if not small_enough or q == min_q:
                return _no_good_crf(attempts, last=attempt)
            l_bound = max((a for a in attempts if a.q < q), key=lambda a: a.q, default=None)
            if l_bound is not None and l_bound.q + step == q:
                if not _sample_small_enough(l_bound.encode_percent, max_encoded_percent):
                    return _no_good_crf(attempts, last=attempt)
                return _done(l_bound, attempts, reason="bracket_lower")
            # Near target but under: nudge CRF down one step (more quality).
            if near and q - step >= min_q:
                q = q - step
                continue
            if l_bound is not None:
                q = vmaf_lerp_q(
                    min_vmaf,
                    q,
                    score,
                    l_bound.q,
                    l_bound.mean_vmaf,
                )
            elif cut_on_iter2 and run == 1 and q > min_q + step:
                q = int(round(q * 0.4 + min_q * 0.6))
            else:
                q = min_q

    best = _best_attempt(attempts, min_vmaf, max_encoded_percent)
    if best is None:
        return CrfSearchResult(False, None, 0.0, attempts=attempts, reason="no_attempts")
    return CrfSearchResult(
        best.mean_vmaf >= min_vmaf - 0.5
        and _sample_small_enough(best.encode_percent, max_encoded_percent),
        best.crf,
        best.mean_vmaf,
        encode_percent=best.encode_percent,
        attempts=attempts,
        reason="max_runs",
    )


def _done(attempt: CrfAttempt, attempts: list[CrfAttempt], *, reason: str) -> CrfSearchResult:
    return CrfSearchResult(
        True,
        attempt.crf,
        attempt.mean_vmaf,
        encode_percent=attempt.encode_percent,
        attempts=attempts,
        reason=reason,
    )


def _no_good_crf(attempts: list[CrfAttempt], *, last: CrfAttempt) -> CrfSearchResult:
    return CrfSearchResult(
        False,
        last.crf,
        last.mean_vmaf,
        encode_percent=last.encode_percent,
        attempts=attempts,
        reason="no_good_crf",
    )


def _best_attempt(
    attempts: list[CrfAttempt],
    min_vmaf: float,
    max_encoded_percent: float,
) -> Optional[CrfAttempt]:
    if not attempts:
        return None
    passing = [
        a
        for a in attempts
        if a.encode_ok
        and a.mean_vmaf >= min_vmaf
        and _sample_small_enough(a.encode_percent, max_encoded_percent)
    ]
    if passing:
        return max(passing, key=lambda a: a.crf)
    return max(attempts, key=lambda a: a.mean_vmaf)


def _finish_on_deadline(
    attempts: list[CrfAttempt],
    *,
    min_vmaf: float,
    max_encoded_percent: float,
    reason: str,
) -> CrfSearchResult:
    best = _best_attempt(attempts, min_vmaf, max_encoded_percent)
    if best is None:
        return CrfSearchResult(False, None, 0.0, attempts=attempts, reason=reason)
    return CrfSearchResult(
        best.encode_ok
        and best.mean_vmaf >= min_vmaf
        and _sample_small_enough(best.encode_percent, max_encoded_percent),
        best.crf,
        best.mean_vmaf,
        encode_percent=best.encode_percent,
        attempts=attempts,
        reason=reason,
    )


def round_crf_for_encode(crf: float) -> int:
    """Round search CRF to an integer suitable for ffmpeg -crf."""
    return int(round(float(crf)))
