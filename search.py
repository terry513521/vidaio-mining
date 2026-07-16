"""CRF / VBR search on the full input (mashup-aware).

CRF search walks quality via VMAF interpolation toward `vmaf_threshold`
(ab-av1 style), then picks the trial with best Vidaio s_f.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from encoder import encode_hevc
from recipes import HevcRecipe, select_recipes
from request import CompressionRequest
from scoring import ScoreResult, probe_video, score_candidate


@dataclass
class TrialResult:
    recipe: str
    mode: str
    crf: Optional[int]
    bitrate: Optional[str]
    path: str
    score: ScoreResult
    encode_ok: bool
    encode_error: str = ""
    stage: str = "search"  # proxy | final | full
    measured_bitrate_mbps: Optional[float] = None
    rejected_reason: Optional[str] = None


@dataclass
class SearchResult:
    best: Optional[TrialResult]
    trials: list[TrialResult] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    recipes: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    output_path: Optional[str] = None
    strategy: str = "full_search"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "output_path": self.output_path,
            "elapsed_sec": self.elapsed_sec,
            "features": self.features,
            "recipes": self.recipes,
            "best": None
            if self.best is None
            else {
                "recipe": self.best.recipe,
                "mode": self.best.mode,
                "crf": self.best.crf,
                "bitrate": self.best.bitrate,
                "measured_bitrate_mbps": self.best.measured_bitrate_mbps,
                "path": self.best.path,
                "stage": self.best.stage,
                "s_f": self.best.score.s_f,
                "vmaf": self.best.score.vmaf,
                "compression_rate": self.best.score.compression_rate,
                "compression_ratio": self.best.score.compression_ratio,
                "reason": self.best.score.reason,
                "rejected_reason": self.best.rejected_reason,
                "validation_errors": self.best.score.validation_errors,
            },
            "trials": [
                {
                    "recipe": t.recipe,
                    "mode": t.mode,
                    "crf": t.crf,
                    "bitrate": t.bitrate,
                    "measured_bitrate_mbps": t.measured_bitrate_mbps,
                    "path": t.path,
                    "stage": t.stage,
                    "encode_ok": t.encode_ok,
                    "encode_error": t.encode_error,
                    "s_f": t.score.s_f,
                    "vmaf": t.score.vmaf,
                    "compression_rate": t.score.compression_rate,
                    "reason": t.score.reason,
                    "rejected_reason": t.rejected_reason,
                }
                for t in self.trials
            ],
        }


def _deadline_left(deadline: float) -> float:
    return max(0.0, deadline - time.time())


def _parse_bitrate_mbps(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    multipliers = {"k": 1 / 1000.0, "m": 1.0, "g": 1000.0}
    suffix = text[-1]
    if suffix in multipliers:
        return float(text[:-1]) * multipliers[suffix]
    return float(text)


def _format_bitrate_mbps(mbps: float) -> str:
    if mbps >= 1.0:
        return f"{mbps:.3f}M"
    return f"{mbps * 1000.0:.0f}k"


def _measured_bitrate_mbps(path: str, ffprobe_bin: Optional[str]) -> Optional[float]:
    probe = probe_video(path, ffprobe_bin)
    video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    bit_rate = video.get("bit_rate") or (probe.get("format") or {}).get("bit_rate")
    if not bit_rate:
        return None
    return float(bit_rate) / 1_000_000.0


def _score(
    reference_path: str,
    distorted_path: str,
    req: CompressionRequest,
    *,
    vmaf_n_subsample: Optional[int] = None,
) -> ScoreResult:
    return score_candidate(
        reference_path,
        distorted_path,
        req.vmaf_threshold,
        ffmpeg_bin=req.ffmpeg_bin,
        ffprobe_bin=req.ffprobe_bin,
        vmaf_n_subsample=vmaf_n_subsample if vmaf_n_subsample is not None else req.vmaf_n_subsample,
        vmaf_n_threads=req.vmaf_n_threads,
    )


def _failed_score(stderr: str) -> ScoreResult:
    return ScoreResult(
        s_f=0.0,
        vmaf=0.0,
        compression_rate=1.0,
        compression_ratio=1.0,
        compression_component=0.0,
        quality_component=0.0,
        reason="encode_failed",
        validation_errors=[stderr],
    )


def _encode_and_score(
    req: CompressionRequest,
    recipe: HevcRecipe,
    out_path: Path,
    *,
    reference_path: str,
    crf: Optional[int],
    bitrate: Optional[str],
    timeout: float,
    stage: str,
    ss: Optional[float] = None,
    t: Optional[float] = None,
    vmaf_n_subsample: Optional[int] = None,
) -> TrialResult:
    enc = encode_hevc(
        req.input_path,
        str(out_path),
        preset=recipe.preset,
        params=recipe.params,
        codec_mode=req.codec_mode,
        crf=crf,
        bitrate=bitrate,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout if timeout > 0 else None,
        ss=ss,
        t=t,
    )

    if not enc.ok:
        return TrialResult(
            recipe=recipe.name,
            mode=req.codec_mode,
            crf=crf,
            bitrate=bitrate,
            path=str(out_path),
            score=_failed_score(enc.stderr_tail),
            encode_ok=False,
            encode_error=enc.stderr_tail,
            stage=stage,
        )

    score = _score(reference_path, str(out_path), req, vmaf_n_subsample=vmaf_n_subsample)
    return TrialResult(
        recipe=recipe.name,
        mode=req.codec_mode,
        crf=crf,
        bitrate=bitrate,
        path=str(out_path),
        score=score,
        encode_ok=True,
        stage=stage,
    )


def _vmaf_lerp_crf(
    target_vmaf: float,
    worse: TrialResult,
    better: TrialResult,
) -> Optional[int]:
    """Interpolate CRF between a failing (worse/higher CRF) and passing (better/lower CRF) trial.

    Mirrors ab-av1 `vmaf_lerp_q`: estimate the CRF whose VMAF crosses `target_vmaf`.
    """
    if worse.crf is None or better.crf is None:
        return None
    if not (worse.crf > better.crf):
        return None
    if worse.score.vmaf >= better.score.vmaf:
        return None

    vmaf_diff = better.score.vmaf - worse.score.vmaf
    if vmaf_diff <= 1e-6:
        mid = (better.crf + worse.crf) // 2
        return mid if better.crf < mid < worse.crf else None

    factor = (target_vmaf - worse.score.vmaf) / vmaf_diff
    lerp = int(round(worse.crf - (worse.crf - better.crf) * factor))
    lo = better.crf + 1
    hi = worse.crf - 1
    if lo > hi:
        return None
    return max(lo, min(hi, lerp))


def search_crf_on_source(
    req: CompressionRequest,
    recipe: HevcRecipe,
    work_dir: Path,
    deadline: float,
    trials: list[TrialResult],
    *,
    reference_path: str,
    stage: str,
    ss: Optional[float],
    t: Optional[float],
    vmaf_n_subsample: int,
    prefix: str,
) -> Optional[TrialResult]:
    """VMAF-interpolated CRF walk toward threshold; keep max s_f.

    1) Seed from recipe / request CRF
    2) Walk CRF by lerping VMAF toward `vmaf_threshold` (not by lerping s_f)
    3) Among all successful trials, return the one with highest s_f
    4) Local ±1 CRF polish around that winner
    """

    crf_lo = req.crf_min
    crf_hi = req.crf_max
    start = req.crf_start if req.crf_start is not None else recipe.crf_start
    start = min(max(int(start), crf_lo), crf_hi)
    step_budget = max(1, req.max_search_steps)
    threshold = float(req.vmaf_threshold)

    by_crf: dict[int, TrialResult] = {}
    local_best: Optional[TrialResult] = None
    run = 0

    def consider(crf: int) -> Optional[TrialResult]:
        nonlocal local_best
        crf = int(crf)
        if crf < req.crf_min or crf > req.crf_max:
            return None
        if crf in by_crf:
            return by_crf[crf]
        if _deadline_left(deadline) < 5:
            return None

        out_path = work_dir / f"{prefix}_{recipe.name}_crf{crf}.mp4"
        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            reference_path=reference_path,
            crf=crf,
            bitrate=None,
            timeout=_deadline_left(deadline),
            stage=stage,
            ss=ss,
            t=t,
            vmaf_n_subsample=vmaf_n_subsample,
        )
        by_crf[crf] = trial
        trials.append(trial)

        if trial.encode_ok and (
            local_best is None or trial.score.s_f > local_best.score.s_f
        ):
            local_best = trial
        return trial

    def ok_trials() -> list[TrialResult]:
        return [t for t in by_crf.values() if t.encode_ok and t.crf is not None]

    def passing() -> list[TrialResult]:
        return [t for t in ok_trials() if t.score.vmaf >= threshold]

    def failing() -> list[TrialResult]:
        return [t for t in ok_trials() if t.score.vmaf < threshold]

    def next_crf(last: TrialResult) -> Optional[int]:
        """Choose next CRF using VMAF lerp / edge jumps (ab-av1-inspired)."""
        assert last.crf is not None
        seen = set(by_crf)

        if last.score.vmaf >= threshold:
            # Quality OK → push worse quality (higher CRF) toward the cliff
            upper = min(
                (t for t in failing() if t.crf is not None and t.crf > last.crf),
                key=lambda t: t.crf or 0,
                default=None,
            )
            if upper is not None and upper.crf is not None:
                if upper.crf == last.crf + 1:
                    return None
                cand = _vmaf_lerp_crf(threshold, worse=upper, better=last)
                if cand is not None and cand not in seen:
                    return cand
            if last.crf >= crf_hi:
                return None
            # First-iter cut toward high CRF (like ab-av1 40/60)
            if run == 1 and last.crf + 1 < crf_hi:
                cand = int(round(last.crf * 0.4 + crf_hi * 0.6))
            else:
                cand = last.crf + max(1, (crf_hi - last.crf + 1) // 2)
            cand = min(max(cand, last.crf + 1), crf_hi)
            return cand if cand not in seen else (
                last.crf + 1 if last.crf + 1 <= crf_hi and last.crf + 1 not in seen else None
            )

        # Quality too low → push better quality (lower CRF)
        lower = max(
            (t for t in passing() if t.crf is not None and t.crf < last.crf),
            key=lambda t: t.crf or 0,
            default=None,
        )
        if lower is not None and lower.crf is not None:
            if lower.crf + 1 == last.crf:
                return None
            cand = _vmaf_lerp_crf(threshold, worse=last, better=lower)
            if cand is not None and cand not in seen:
                return cand
        if last.crf <= crf_lo:
            return None
        if run == 1 and last.crf > crf_lo + 1:
            cand = int(round(last.crf * 0.4 + crf_lo * 0.6))
        else:
            cand = last.crf - max(1, (last.crf - crf_lo + 1) // 2)
        cand = max(min(cand, last.crf - 1), crf_lo)
        return cand if cand not in seen else (
            last.crf - 1 if last.crf - 1 >= crf_lo and last.crf - 1 not in seen else None
        )

    last = consider(start)
    run = 1

    while (
        last is not None
        and last.encode_ok
        and last.crf is not None
        and len(by_crf) < step_budget
        and _deadline_left(deadline) > 5
    ):
        cand = next_crf(last)
        if cand is None:
            break
        trial = consider(cand)
        run += 1
        if trial is None or not trial.encode_ok:
            break
        last = trial

        # Near-threshold stop: still keep walking a bit for s_f, but if we sit
        # just above threshold with no room between pass/fail, local polish next.
        pass_list = passing()
        fail_list = failing()
        if pass_list and fail_list:
            best_pass = max(pass_list, key=lambda t: t.crf or 0)
            worst_fail = min(fail_list, key=lambda t: t.crf or 0)
            if (
                best_pass.crf is not None
                and worst_fail.crf is not None
                and worst_fail.crf <= best_pass.crf + 1
            ):
                break

    # Local ±1 around s_f winner (and around threshold-edge passer)
    polish: set[int] = set()
    if local_best is not None and local_best.crf is not None:
        polish.update({local_best.crf - 1, local_best.crf + 1})
    pass_list = passing()
    if pass_list:
        edge = max(pass_list, key=lambda t: t.crf or 0)
        if edge.crf is not None:
            polish.update({edge.crf - 1, edge.crf + 1})
    for crf in sorted(polish):
        if len(by_crf) >= step_budget + 2:
            break
        if _deadline_left(deadline) < 5:
            break
        consider(crf)

    return local_best


def search_vbr_on_source(
    req: CompressionRequest,
    recipe: HevcRecipe,
    work_dir: Path,
    deadline: float,
    trials: list[TrialResult],
    *,
    reference_path: str,
    stage: str,
    ss: Optional[float],
    t: Optional[float],
    vmaf_n_subsample: int,
    prefix: str,
) -> Optional[TrialResult]:
    """Bracket-search average bitrate under cap while keeping VMAF threshold."""
    target_mbps = _parse_bitrate_mbps(req.target_bitrate)
    if target_mbps is None:
        raise ValueError("target_bitrate is required for VBR mode")

    cap_mbps = target_mbps * req.vbr_max_ratio_to_target
    lo = max(req.vbr_min_mbps_floor, min(0.8 * target_mbps, cap_mbps))
    hi = cap_mbps
    step_budget = max(1, req.max_search_steps)
    seen: set[float] = set()
    local_best: Optional[TrialResult] = None

    def consider(mbps: float) -> None:
        nonlocal lo, hi, local_best
        mbps = round(max(req.vbr_min_mbps_floor, min(mbps, cap_mbps)), 3)
        if mbps in seen or _deadline_left(deadline) < 5:
            return
        seen.add(mbps)
        bitrate = _format_bitrate_mbps(mbps)
        out_path = work_dir / f"{prefix}_{recipe.name}_vbr_{str(mbps).replace('.', '_')}M.mp4"
        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            reference_path=reference_path,
            crf=None,
            bitrate=bitrate,
            timeout=_deadline_left(deadline),
            stage=stage,
            ss=ss,
            t=t,
            vmaf_n_subsample=vmaf_n_subsample,
        )

        if trial.encode_ok:
            measured_mbps = _measured_bitrate_mbps(str(out_path), req.ffprobe_bin)
            trial.measured_bitrate_mbps = measured_mbps

            if measured_mbps is None:
                trial.rejected_reason = "bitrate_missing"
                trial.score.s_f = 0.0
                trial.score.reason = (
                    f"bitrate_missing (requested {bitrate}, cap {cap_mbps:.3f} Mbps)"
                )
                print(f"  Reject VBR trial {bitrate}: measured bitrate missing")
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))
            elif measured_mbps > cap_mbps:
                trial.rejected_reason = "bitrate_above_cap"
                trial.score.s_f = 0.0
                trial.score.reason = (
                    f"bitrate_above_cap (measured {measured_mbps:.3f} Mbps > "
                    f"cap {cap_mbps:.3f} Mbps = {req.vbr_max_ratio_to_target:.2f}x "
                    f"target {target_mbps:.3f})"
                )
                print(
                    f"  Reject VBR trial {bitrate}: "
                    f"measured {measured_mbps:.3f} Mbps > cap {cap_mbps:.3f} Mbps"
                )
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))
            elif trial.score.vmaf < req.vmaf_threshold:
                trial.rejected_reason = "vmaf_below_threshold"
                print(
                    f"  Reject VBR trial {bitrate}: "
                    f"VMAF {trial.score.vmaf:.2f} < threshold {req.vmaf_threshold}"
                )
                lo = min(cap_mbps, mbps + 0.05)
            else:
                # Valid under cap + VMAF threshold — keep best s_f
                if local_best is None or trial.score.s_f > local_best.score.s_f:
                    local_best = trial
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))

        trials.append(trial)

    # Seed probes
    consider(min(target_mbps, cap_mbps))
    consider(0.9 * min(target_mbps, cap_mbps))

    while len(seen) < step_budget and lo <= hi and _deadline_left(deadline) > 5:
        mid = round((lo + hi) / 2.0, 3)
        if mid in seen:
            probe = round(mid + 0.1, 3)
            if probe <= hi and probe not in seen:
                mid = probe
            else:
                break
        consider(mid)

    return local_best


def run_search(req: CompressionRequest, features: dict[str, float]) -> SearchResult:
    started = time.time()
    deadline = started + req.time_budget_sec

    work_dir = Path(req.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    recipes = select_recipes(features, req.vmaf_threshold, max_recipes=req.max_recipes)
    trials: list[TrialResult] = []
    strategy = "full_search" if req.codec_mode == "CRF" else "vbr_search"
    final_best: Optional[TrialResult] = None

    print(f"  Full-file {req.codec_mode} search ({len(recipes)} recipe(s))")

    for recipe in recipes:
        if _deadline_left(deadline) < 5:
            break
        if req.codec_mode == "CRF":
            best_for_recipe = search_crf_on_source(
                req,
                recipe,
                work_dir,
                deadline,
                trials,
                reference_path=req.input_path,
                stage="full",
                ss=None,
                t=None,
                vmaf_n_subsample=req.vmaf_n_subsample,
                prefix="full",
            )
        else:
            best_for_recipe = search_vbr_on_source(
                req,
                recipe,
                work_dir,
                deadline,
                trials,
                reference_path=req.input_path,
                stage="full",
                ss=None,
                t=None,
                vmaf_n_subsample=req.vmaf_n_subsample,
                prefix="full",
            )
        if best_for_recipe is not None and (
            final_best is None or best_for_recipe.score.s_f > final_best.score.s_f
        ):
            final_best = best_for_recipe

    best = final_best
    output_path = None
    if best is not None and best.encode_ok and best.score.s_f > 0:
        output_path = req.output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best.path, output_path)

        if not req.keep_candidates:
            for trial in trials:
                p = Path(trial.path)
                if not p.is_file():
                    continue
                if Path(output_path).resolve() == p.resolve():
                    continue
                try:
                    p.unlink()
                except OSError:
                    pass

    return SearchResult(
        best=best,
        trials=trials,
        features=features,
        recipes=[r.name for r in recipes],
        elapsed_sec=time.time() - started,
        output_path=output_path,
        strategy=strategy,
    )
