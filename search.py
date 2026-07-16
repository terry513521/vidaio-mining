"""CRF / VBR search on the full input (mashup-aware).

Default CRF strategy (two-phase):
  1) Build a short proxy from ~2.5s mid-windows of each segment
  2) Encode 3 CRF candidates on the proxy in parallel (bitrate-ratio s_f)
  3) Encode the full file once at the winning CRF for the true s_f
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from encoder import encode_hevc
from logutil import log
from proxy import build_proxy_reference, select_proxy_windows
from recipes import HevcRecipe, candidate_crfs, select_recipes
from request import CompressionRequest
from scoring import ScoreResult, calculate_compression_score, probe_video, score_candidate


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
    proxy_path: Optional[str] = None
    proxy_windows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "output_path": self.output_path,
            "elapsed_sec": self.elapsed_sec,
            "features": self.features,
            "recipes": self.recipes,
            "proxy_path": self.proxy_path,
            "proxy_windows": self.proxy_windows,
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
    compression_rate_override: Optional[float] = None,
) -> ScoreResult:
    return score_candidate(
        reference_path,
        distorted_path,
        req.vmaf_threshold,
        ffmpeg_bin=req.ffmpeg_bin,
        ffprobe_bin=req.ffprobe_bin,
        vmaf_n_subsample=vmaf_n_subsample if vmaf_n_subsample is not None else req.vmaf_n_subsample,
        vmaf_n_threads=req.vmaf_n_threads,
        vmaf_backend=req.vmaf_backend,
        vmaf_docker_image=req.vmaf_docker_image,
        vmaf_docker_gpus=req.vmaf_docker_gpus,
        compression_rate_override=compression_rate_override,
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


def _is_better_trial(candidate: TrialResult, incumbent: Optional[TrialResult]) -> bool:
    """Prefer higher s_f; on ties prefer higher VMAF, then lower CRF."""
    if not candidate.encode_ok:
        return False
    if incumbent is None or not incumbent.encode_ok:
        return True
    if candidate.score.s_f != incumbent.score.s_f:
        return candidate.score.s_f > incumbent.score.s_f
    if candidate.score.vmaf != incumbent.score.vmaf:
        return candidate.score.vmaf > incumbent.score.vmaf
    cand_crf = candidate.crf if candidate.crf is not None else 10**9
    inc_crf = incumbent.crf if incumbent.crf is not None else 10**9
    return cand_crf < inc_crf


def _encode_and_score(
    req: CompressionRequest,
    recipe: HevcRecipe,
    out_path: Path,
    *,
    input_path: str,
    reference_path: str,
    crf: Optional[int],
    bitrate: Optional[str],
    timeout: float,
    stage: str,
    ss: Optional[float] = None,
    t: Optional[float] = None,
    vmaf_n_subsample: Optional[int] = None,
    compression_rate_override: Optional[float] = None,
) -> TrialResult:
    enc = encode_hevc(
        input_path,
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

    score = _score(
        reference_path,
        str(out_path),
        req,
        vmaf_n_subsample=vmaf_n_subsample,
        compression_rate_override=compression_rate_override,
    )
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


def _parallel_crf_trials(
    req: CompressionRequest,
    recipe: HevcRecipe,
    work_dir: Path,
    deadline: float,
    trials: list[TrialResult],
    *,
    input_path: str,
    reference_path: str,
    stage: str,
    prefix: str,
    source_bitrate_mbps: Optional[float] = None,
) -> Optional[TrialResult]:
    """Encode ``crf_candidates`` in parallel; return max-s_f trial."""
    start = req.crf_start if req.crf_start is not None else recipe.crf_start
    candidates = candidate_crfs(
        start,
        req.crf_min,
        req.crf_max,
        count=max(1, req.crf_candidates),
        spread=max(1, req.crf_spread),
    )
    timeout = _deadline_left(deadline)
    workers = max(1, min(req.max_workers, len(candidates)))

    log(
        f"  Parallel CRF ({stage}): recipe={recipe.name} preset={recipe.preset} "
        f"seed={start} candidates={candidates} workers={workers}"
    )

    if timeout < 5:
        log(f"  Skipping {stage} CRF search: time budget exhausted")
        return None

    def _run_one(crf: int) -> TrialResult:
        out_path = work_dir / f"{prefix}_{recipe.name}_crf{crf}.mp4"
        log(f"  → [{stage}] encoding CRF {crf} ...")

        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            input_path=input_path,
            reference_path=reference_path,
            crf=crf,
            bitrate=None,
            timeout=timeout,
            stage=stage,
            vmaf_n_subsample=req.vmaf_n_subsample,
        )

        if trial.encode_ok and source_bitrate_mbps is not None and stage == "proxy":
            measured = _measured_bitrate_mbps(str(out_path), req.ffprobe_bin)
            trial.measured_bitrate_mbps = measured
            if measured is not None and source_bitrate_mbps > 0:
                rate_override = measured / source_bitrate_mbps
                s_f, c_comp, q_comp, reason = calculate_compression_score(
                    vmaf_score=trial.score.vmaf,
                    compression_rate=rate_override,
                    vmaf_threshold=float(req.vmaf_threshold),
                )
                trial.score = ScoreResult(
                    s_f=s_f,
                    vmaf=trial.score.vmaf,
                    compression_rate=rate_override,
                    compression_ratio=1.0 / max(rate_override, 1e-9),
                    compression_component=c_comp,
                    quality_component=q_comp,
                    reason=f"proxy_bitrate_ratio:{reason}",
                    validation_errors=[],
                )

        if trial.encode_ok:
            log(
                f"  ← [{stage}] CRF {crf}: vmaf={trial.score.vmaf:.2f} "
                f"s_f={trial.score.s_f:.4f} rate={trial.score.compression_rate:.3f}"
            )
        else:
            log(f"  ← [{stage}] CRF {crf}: encode failed")
        return trial

    local_best: Optional[TrialResult] = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, crf): crf for crf in candidates}
        for fut in as_completed(futures):
            crf = futures[fut]
            try:
                trial = fut.result()
            except Exception as exc:
                out_path = work_dir / f"{prefix}_{recipe.name}_crf{crf}.mp4"
                trial = TrialResult(
                    recipe=recipe.name,
                    mode=req.codec_mode,
                    crf=crf,
                    bitrate=None,
                    path=str(out_path),
                    score=_failed_score(str(exc)),
                    encode_ok=False,
                    encode_error=str(exc),
                    stage=stage,
                )
                log(f"  ← [{stage}] CRF {crf}: exception {exc}")
            trials.append(trial)
            if _is_better_trial(trial, local_best):
                local_best = trial

    return local_best


def search_crf_two_phase(
    req: CompressionRequest,
    recipe: HevcRecipe,
    work_dir: Path,
    deadline: float,
    trials: list[TrialResult],
    *,
    segments: list[dict[str, Any]],
) -> tuple[Optional[TrialResult], Optional[str], list[dict[str, Any]]]:
    """Proxy CRF selection → one full-file encode at the winning CRF."""
    windows = select_proxy_windows(
        segments,
        seconds_per_segment=req.proxy_seconds_per_segment,
        max_seconds=req.proxy_max_seconds,
        min_window_seconds=req.proxy_min_window_seconds,
    )
    proxy_meta = [
        {
            "segment_index": w.segment_index,
            "start_sec": w.start_sec,
            "duration_sec": w.duration_sec,
            "difficulty": w.difficulty,
        }
        for w in windows
    ]

    if not windows:
        log("  Proxy: no windows — falling back to full-file parallel CRF")
        best = _parallel_crf_trials(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=req.input_path,
            reference_path=req.input_path,
            stage="full",
            prefix="full",
        )
        return best, None, []

    proxy_path = work_dir / "proxy_reference.mp4"
    log(
        f"  Building proxy: {len(windows)} window(s), "
        f"~{sum(w.duration_sec for w in windows):.1f}s "
        f"(target {req.proxy_seconds_per_segment}s/seg, cap {req.proxy_max_seconds}s)"
    )
    built = build_proxy_reference(
        req.input_path,
        str(proxy_path),
        windows,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=_deadline_left(deadline),
    )
    if not built.ok:
        log(f"  Proxy build failed: {built.error[:300]} — falling back to full-file")
        best = _parallel_crf_trials(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=req.input_path,
            reference_path=req.input_path,
            stage="full",
            prefix="full",
        )
        return best, None, proxy_meta

    log(f"  Proxy ready: {built.path} ({built.total_seconds:.1f}s)")

    source_bitrate = _measured_bitrate_mbps(req.input_path, req.ffprobe_bin)
    if source_bitrate is None:
        log("  Warning: could not probe source bitrate; proxy s_f uses file-size rate")

    proxy_best = _parallel_crf_trials(
        req,
        recipe,
        work_dir,
        deadline,
        trials,
        input_path=built.path,
        reference_path=built.path,
        stage="proxy",
        prefix="proxy",
        source_bitrate_mbps=source_bitrate,
    )

    if proxy_best is None or proxy_best.crf is None:
        log("  Proxy search found no winner — falling back to full-file")
        best = _parallel_crf_trials(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=req.input_path,
            reference_path=req.input_path,
            stage="full",
            prefix="full",
        )
        return best, built.path, proxy_meta

    chosen_crf = proxy_best.crf
    log(
        f"  Proxy best CRF={chosen_crf} "
        f"vmaf={proxy_best.score.vmaf:.2f} s_f={proxy_best.score.s_f:.4f} "
        f"→ one full-file encode"
    )
    if proxy_best.score.vmaf < req.vmaf_threshold:
        log(
            f"  Warning: proxy VMAF {proxy_best.score.vmaf:.2f} < "
            f"threshold {req.vmaf_threshold}; full encode may still fail"
        )

    if _deadline_left(deadline) < 5:
        log("  Skipping final encode: time budget exhausted")
        return proxy_best, built.path, proxy_meta

    out_path = work_dir / f"final_{recipe.name}_crf{chosen_crf}.mp4"
    log(f"  → [final] encoding CRF {chosen_crf} ...")
    final_best = _encode_and_score(
        req,
        recipe,
        out_path,
        input_path=req.input_path,
        reference_path=req.input_path,
        crf=chosen_crf,
        bitrate=None,
        timeout=_deadline_left(deadline),
        stage="final",
        vmaf_n_subsample=req.vmaf_n_subsample,
    )
    trials.append(final_best)
    if final_best.encode_ok:
        log(
            f"  ← [final] CRF {chosen_crf}: vmaf={final_best.score.vmaf:.2f} "
            f"s_f={final_best.score.s_f:.4f} rate={final_best.score.compression_rate:.3f}"
        )
    else:
        log(f"  ← [final] CRF {chosen_crf}: encode failed")

    return final_best if final_best.encode_ok else proxy_best, built.path, proxy_meta


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
    """Legacy full-file parallel CRF (used when proxy is disabled)."""
    return _parallel_crf_trials(
        req,
        recipe,
        work_dir,
        deadline,
        trials,
        input_path=req.input_path,
        reference_path=reference_path,
        stage=stage,
        prefix=prefix,
    )


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
            input_path=req.input_path,
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
                log(f"  Reject VBR trial {bitrate}: measured bitrate missing")
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))
            elif measured_mbps > cap_mbps:
                trial.rejected_reason = "bitrate_above_cap"
                trial.score.s_f = 0.0
                trial.score.reason = (
                    f"bitrate_above_cap (measured {measured_mbps:.3f} Mbps > "
                    f"cap {cap_mbps:.3f} Mbps = {req.vbr_max_ratio_to_target:.2f}x "
                    f"target {target_mbps:.3f})"
                )
                log(
                    f"  Reject VBR trial {bitrate}: "
                    f"measured {measured_mbps:.3f} Mbps > cap {cap_mbps:.3f} Mbps"
                )
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))
            elif trial.score.vmaf < req.vmaf_threshold:
                trial.rejected_reason = "vmaf_below_threshold"
                log(
                    f"  Reject VBR trial {bitrate}: "
                    f"VMAF {trial.score.vmaf:.2f} < threshold {req.vmaf_threshold}"
                )
                lo = min(cap_mbps, mbps + 0.05)
            else:
                if local_best is None or trial.score.s_f > local_best.score.s_f:
                    local_best = trial
                hi = min(hi, max(req.vbr_min_mbps_floor, mbps - 0.05))

        trials.append(trial)

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


def run_search(
    req: CompressionRequest,
    features: dict[str, float],
    segments: Optional[list[dict[str, Any]]] = None,
) -> SearchResult:
    started = time.time()
    deadline = started + req.time_budget_sec

    work_dir = Path(req.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    recipes = select_recipes(
        features,
        req.vmaf_threshold,
        max_recipes=req.max_recipes,
        preset=req.preset,
    )
    trials: list[TrialResult] = []
    final_best: Optional[TrialResult] = None
    proxy_path: Optional[str] = None
    proxy_windows: list[dict[str, Any]] = []

    use_proxy = bool(req.use_proxy and req.codec_mode == "CRF" and segments)
    strategy = "proxy_then_full" if use_proxy else (
        "parallel_crf" if req.codec_mode == "CRF" else "vbr_search"
    )

    log(
        f"  {strategy} search "
        f"({len(recipes)} recipe(s), workers={req.max_workers})"
    )

    for recipe in recipes:
        if _deadline_left(deadline) < 5:
            break
        if req.codec_mode == "CRF" and use_proxy:
            best_for_recipe, proxy_path, proxy_windows = search_crf_two_phase(
                req,
                recipe,
                work_dir,
                deadline,
                trials,
                segments=segments or [],
            )
        elif req.codec_mode == "CRF":
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
        proxy_path=proxy_path,
        proxy_windows=proxy_windows,
    )
