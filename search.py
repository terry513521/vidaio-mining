"""RC / ABR search on the full input (mashup-aware).

Default RC strategy (two-phase):
  1) Build a short proxy from ~2.5s mid-windows of each segment
  2) Encode 3 CQ/CRF candidates on the proxy in parallel (bitrate-ratio s_f)
  3) Encode the full file once at the winning CQ for the true s_f
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from encoder import encode_hevc
from interp_search import (
    NvencOverrides,
    Round2TrialSpec,
    estimate_primary_x265_crf,
    interpolate_cq_for_vmaf,
    next_serial_cq_probe,
    observations_from_trials,
    propose_round2_details,
    propose_round2_mixed,
    propose_vmaf_anchored_crfs,
    round1_feature_cqs,
)
from logutil import log
from proxy import build_proxy_reference, select_proxy_windows
from recipes import (
    HevcRecipe,
    candidate_crfs,
    describe_feature_x265_baseline,
    select_recipes,
)
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
    stage: str = "search"  # proxy | final | full | libx265_refine
    encoder: str = "libx265"
    measured_bitrate_mbps: Optional[float] = None
    rejected_reason: Optional[str] = None
    encode_sec: float = 0.0
    score_sec: float = 0.0
    elapsed_sec: float = 0.0
    nvenc_overrides: dict[str, Any] = field(default_factory=dict)


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
            "best": self._best_dict(),
            "trials": [
                {
                    "recipe": t.recipe,
                    "mode": t.mode,
                    "encoder": t.encoder,
                    "crf": t.crf,
                    "bitrate": t.bitrate,
                    "measured_bitrate_mbps": t.measured_bitrate_mbps,
                    "path": t.path,
                    "stage": t.stage,
                    "encode_ok": t.encode_ok,
                    "encode_error": t.encode_error,
                    "s_f": t.score.s_f,
                    "vmaf": t.score.vmaf,
                    "vmaf_base": t.score.vmaf_base,
                    "vmaf_delta": t.score.vmaf_delta,
                    "passed_encoding_gates": t.score.passed_encoding_gates,
                    "passed_vmaf_delta_gate": t.score.passed_vmaf_delta_gate,
                    "compression_rate": t.score.compression_rate,
                    "reason": t.score.reason,
                    "rejected_reason": t.rejected_reason,
                    "encode_sec": t.encode_sec,
                    "score_sec": t.score_sec,
                    "elapsed_sec": t.elapsed_sec,
                    "nvenc_overrides": t.nvenc_overrides,
                }
                for t in self.trials
            ],
        }

    def to_final_dict(self) -> dict[str, Any]:
        """Final delivered result only — no probe/trial history."""
        return {
            "strategy": self.strategy,
            "output_path": self.output_path,
            "elapsed_sec": self.elapsed_sec,
            "features": self.features,
            "recipes": self.recipes,
            "best": self._best_dict(),
        }

    def _best_dict(self) -> Optional[dict[str, Any]]:
        if self.best is None:
            return None
        return {
            "recipe": self.best.recipe,
            "mode": self.best.mode,
            "encoder": self.best.encoder,
            "crf": self.best.crf,
            "bitrate": self.best.bitrate,
            "measured_bitrate_mbps": self.best.measured_bitrate_mbps,
            "path": self.best.path,
            "stage": self.best.stage,
            "s_f": self.best.score.s_f,
            "vmaf": self.best.score.vmaf,
            "vmaf_base": self.best.score.vmaf_base,
            "vmaf_delta": self.best.score.vmaf_delta,
            "passed_encoding_gates": self.best.score.passed_encoding_gates,
            "passed_vmaf_delta_gate": self.best.score.passed_vmaf_delta_gate,
            "compression_rate": self.best.score.compression_rate,
            "compression_ratio": self.best.score.compression_ratio,
            "reason": self.best.score.reason,
            "rejected_reason": self.best.rejected_reason,
            "validation_errors": self.best.score.validation_errors,
            "encode_sec": self.best.encode_sec,
            "score_sec": self.best.score_sec,
            "elapsed_sec": self.best.elapsed_sec,
            "nvenc_overrides": self.best.nvenc_overrides,
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


def _bitrate_log(mbps: Optional[float]) -> str:
    if mbps is None:
        return "bitrate=?"
    return f"bitrate={mbps:.2f}Mbps"


def _score(
    reference_path: str,
    distorted_path: str,
    req: CompressionRequest,
    *,
    vmaf_n_subsample: Optional[int] = None,
    compression_rate_override: Optional[float] = None,
    timeout: Optional[float] = None,
    vmaf_docker_gpus: Optional[bool] = None,
    vmaf_gpu_device: Optional[int] = None,
    pair_gates: bool = True,
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
        vmaf_docker_gpus=(
            req.vmaf_docker_gpus if vmaf_docker_gpus is None else vmaf_docker_gpus
        ),
        vmaf_gpu_device=vmaf_gpu_device,
        compression_rate_override=compression_rate_override,
        codec_mode=req.codec_mode,
        target_bitrate_mbps=_parse_bitrate_mbps(req.target_bitrate),
        timeout=timeout,
        pair_gates=pair_gates,
    )


_PROXY_PROBE_STAGES = frozenset({"proxy", "sla_proxy_probe", "sla_scene_crf_search"})


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


def _resolve_nvenc(
    req: CompressionRequest,
    overrides: Optional[NvencOverrides] = None,
) -> dict[str, Any]:
    ov = overrides or NvencOverrides()
    return {
        "nvenc_tune": ov.nvenc_tune if ov.nvenc_tune is not None else req.nvenc_tune,
        "nvenc_rc": ov.nvenc_rc if ov.nvenc_rc is not None else req.nvenc_rc,
        "nvenc_multipass": (
            ov.nvenc_multipass if ov.nvenc_multipass is not None else req.nvenc_multipass
        ),
        "nvenc_spatial_aq": (
            ov.nvenc_spatial_aq if ov.nvenc_spatial_aq is not None else req.nvenc_spatial_aq
        ),
        "nvenc_temporal_aq": (
            ov.nvenc_temporal_aq if ov.nvenc_temporal_aq is not None else req.nvenc_temporal_aq
        ),
        "nvenc_aq_strength": (
            ov.nvenc_aq_strength if ov.nvenc_aq_strength is not None else req.nvenc_aq_strength
        ),
        "nvenc_rc_lookahead": (
            ov.nvenc_rc_lookahead
            if ov.nvenc_rc_lookahead is not None
            else req.nvenc_rc_lookahead
        ),
        "nvenc_bf": ov.nvenc_bf if ov.nvenc_bf is not None else req.nvenc_bf,
        "nvenc_gop": ov.nvenc_gop if ov.nvenc_gop is not None else req.nvenc_gop,
        "nvenc_b_ref_mode": (
            ov.nvenc_b_ref_mode if ov.nvenc_b_ref_mode is not None else req.nvenc_b_ref_mode
        ),
        "preprocess": ov.preprocess if ov.preprocess is not None else req.preprocess,
        "nvenc_gpu": req.nvenc_gpu,
        "nvenc_hwaccel": req.nvenc_hwaccel,
    }


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
    nvenc_overrides: Optional[NvencOverrides] = None,
    encoder: Optional[str] = None,
    preset: Optional[str] = None,
    vmaf_docker_gpus: Optional[bool] = None,
    progress_reference_path: Optional[str] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> TrialResult:
    t0 = time.time()
    enc_name = (encoder or req.encoder).lower().strip()
    if enc_name in {"x265"}:
        enc_name = "libx265"
    if enc_name in {"nvenc", "nvenc_hevc"}:
        enc_name = "hevc_nvenc"
    enc_preset = preset if preset is not None else recipe.preset
    # Never carry NVENC-only overrides into a libx265 refine encode.
    active_overrides = None if enc_name == "libx265" else nvenc_overrides
    nv = _resolve_nvenc(req, active_overrides)
    enc = encode_hevc(
        input_path,
        str(out_path),
        preset=enc_preset,
        params=recipe.params,
        codec_mode=req.codec_mode,
        crf=crf,
        bitrate=bitrate,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout if timeout > 0 else None,
        ss=ss,
        t=t,
        encoder=enc_name,
        preprocess=nv["preprocess"] if enc_name == "hevc_nvenc" else req.preprocess,
        libx265_profile=req.libx265_profile if enc_name == "libx265" else None,
        nvenc_tune=nv["nvenc_tune"],
        nvenc_rc=nv["nvenc_rc"],
        nvenc_multipass=nv["nvenc_multipass"],
        nvenc_spatial_aq=nv["nvenc_spatial_aq"],
        nvenc_temporal_aq=nv["nvenc_temporal_aq"],
        nvenc_aq_strength=nv["nvenc_aq_strength"],
        nvenc_rc_lookahead=nv["nvenc_rc_lookahead"],
        nvenc_bf=nv["nvenc_bf"],
        nvenc_gop=nv["nvenc_gop"],
        nvenc_b_ref_mode=nv["nvenc_b_ref_mode"],
        nvenc_gpu=nv["nvenc_gpu"],
        nvenc_hwaccel=nv["nvenc_hwaccel"],
        progress_reference_path=progress_reference_path,
        progress_label=progress_label,
        progress_interval_sec=progress_interval_sec,
    )
    encode_sec = time.time() - t0

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
            encoder=enc_name,
            encode_sec=encode_sec,
            score_sec=0.0,
            elapsed_sec=encode_sec,
        )

    t1 = time.time()
    score = _score(
        reference_path,
        str(out_path),
        req,
        vmaf_n_subsample=vmaf_n_subsample,
        compression_rate_override=compression_rate_override,
        timeout=max(0.1, timeout - encode_sec),
        vmaf_docker_gpus=vmaf_docker_gpus,
        pair_gates=stage not in _PROXY_PROBE_STAGES,
    )
    score_sec = time.time() - t1
    elapsed_sec = encode_sec + score_sec
    measured_mbps = _measured_bitrate_mbps(str(out_path), req.ffprobe_bin)
    ov_dict = (active_overrides or NvencOverrides()).to_dict()
    return TrialResult(
        recipe=recipe.name,
        mode=req.codec_mode,
        crf=crf,
        bitrate=bitrate,
        measured_bitrate_mbps=measured_mbps,
        path=str(out_path),
        score=score,
        encode_ok=True,
        stage=stage,
        encoder=enc_name,
        encode_sec=encode_sec,
        score_sec=score_sec,
        elapsed_sec=elapsed_sec,
        nvenc_overrides=ov_dict,
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
    candidates: Optional[list[int]] = None,
) -> Optional[TrialResult]:
    """Encode CQ/CRF candidates in parallel; return max-s_f trial."""
    start = req.crf_start if req.crf_start is not None else recipe.crf_start
    if candidates is None:
        candidates = candidate_crfs(
            start,
            req.crf_min,
            req.crf_max,
            count=max(1, req.crf_candidates),
            spread=max(1, req.crf_spread),
        )
    else:
        candidates = sorted({int(c) for c in candidates})
    timeout = _deadline_left(deadline)
    workers = max(1, min(req.max_workers, len(candidates))) if candidates else 1
    if req.serial_cq_search:
        workers = 1

    log(
        f"  Parallel CRF ({stage}): encoder={req.encoder} recipe={recipe.name} "
        f"preset={recipe.preset} "
        f"seed={start} candidates={candidates} workers={workers}"
    )

    if not candidates:
        log(f"  Skipping {stage} CRF search: empty candidate list")
        return None

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
                    vmaf_base=trial.score.vmaf_base,
                    vmaf_delta=trial.score.vmaf_delta,
                    passed_encoding_gates=trial.score.passed_encoding_gates,
                    passed_vmaf_delta_gate=trial.score.passed_vmaf_delta_gate,
                )

        if trial.encode_ok:
            log(
                f"  ← [{stage}] CRF {crf}: neg={trial.score.vmaf:.2f} "
                f"base={trial.score.vmaf_base:.2f} "
                f"delta={trial.score.vmaf_delta:.2f} "
                f"s_f={trial.score.s_f:.4f} "
                f"compression_ratio={trial.score.compression_ratio:.2f}x "
                f"{_bitrate_log(trial.measured_bitrate_mbps)} "
                f"encode={trial.encode_sec:.1f}s score={trial.score_sec:.1f}s "
                f"total={trial.elapsed_sec:.1f}s"
            )
        else:
            log(
                f"  ← [{stage}] CRF {crf}: encode failed "
                f"encode={trial.encode_sec:.1f}s total={trial.elapsed_sec:.1f}s"
            )
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
                    encoder=req.encoder,
                )
                log(f"  ← [{stage}] CRF {crf}: exception {exc}")
            trials.append(trial)
            if _is_better_trial(trial, local_best):
                local_best = trial

    return local_best


def _parallel_round2_trials(
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
    trial_specs: list[Round2TrialSpec],
) -> Optional[TrialResult]:
    """Run Round-2 mixed trials (NVENC tune at best CQ + CQ refine)."""
    if not trial_specs:
        return None

    timeout = _deadline_left(deadline)
    workers = max(1, min(req.max_workers, len(trial_specs)))
    log(
        f"  Parallel round2 ({stage}): encoder={req.encoder} recipe={recipe.name} "
        f"preset={recipe.preset} trials={len(trial_specs)} workers={workers}"
    )
    if timeout < 5:
        log(f"  Skipping {stage}: time budget exhausted")
        return None

    def _run_one(spec: Round2TrialSpec) -> TrialResult:
        suffix = spec.nvenc.suffix()
        out_path = work_dir / f"{prefix}_{recipe.name}_crf{spec.cq}_{suffix}.mp4"
        label = f"CQ {spec.cq}"
        if spec.nvenc.to_dict():
            label += f" nvenc={spec.nvenc.to_dict()}"
        log(f"  → [{stage}] {label} ({spec.reason}) ...")

        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            input_path=input_path,
            reference_path=reference_path,
            crf=spec.cq,
            bitrate=None,
            timeout=timeout,
            stage=stage,
            vmaf_n_subsample=req.vmaf_n_subsample,
            nvenc_overrides=spec.nvenc,
        )

        if trial.encode_ok:
            log(
                f"  ← [{stage}] {label}: neg={trial.score.vmaf:.2f} "
                f"base={trial.score.vmaf_base:.2f} "
                f"delta={trial.score.vmaf_delta:.2f} "
                f"s_f={trial.score.s_f:.4f} "
                f"compression_ratio={trial.score.compression_ratio:.2f}x "
                f"{_bitrate_log(trial.measured_bitrate_mbps)} "
                f"encode={trial.encode_sec:.1f}s score={trial.score_sec:.1f}s "
                f"total={trial.elapsed_sec:.1f}s"
            )
        else:
            log(
                f"  ← [{stage}] {label}: encode failed "
                f"encode={trial.encode_sec:.1f}s total={trial.elapsed_sec:.1f}s"
            )
        return trial

    local_best: Optional[TrialResult] = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, spec): spec for spec in trial_specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                trial = fut.result()
            except Exception as exc:
                suffix = spec.nvenc.suffix()
                out_path = work_dir / f"{prefix}_{recipe.name}_crf{spec.cq}_{suffix}.mp4"
                trial = TrialResult(
                    recipe=recipe.name,
                    mode=req.codec_mode,
                    crf=spec.cq,
                    bitrate=None,
                    path=str(out_path),
                    score=_failed_score(str(exc)),
                    encode_ok=False,
                    encode_error=str(exc),
                    stage=stage,
                    encoder=req.encoder,
                    nvenc_overrides=spec.nvenc.to_dict(),
                )
                log(f"  ← [{stage}] CQ {spec.cq}: exception {exc}")
            trials.append(trial)
            if _is_better_trial(trial, local_best):
                local_best = trial

    return local_best


def _serial_cq_probe_search(
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
    features: Optional[dict[str, Any]] = None,
    source_bitrate_mbps: Optional[float] = None,
) -> Optional[TrialResult]:
    """One CQ probe per round (no parallel CQ sweep on the same video)."""
    max_rounds = max(1, req.crf_candidates)
    probe_plan, seed, seed_reason = round1_feature_cqs(
        features,
        count=max_rounds,
        crf_min=req.crf_min,
        crf_max=req.crf_max,
        vmaf_threshold=float(req.vmaf_threshold),
        spread=max(1, req.crf_spread),
        crf_start=req.crf_start,
    )
    log(
        f"  serial_cq search: seed CQ {seed} ({seed_reason}) "
        f"plan={probe_plan} rounds={max_rounds}"
    )
    used: set[int] = set()
    local_best: Optional[TrialResult] = None

    for round_idx in range(1, max_rounds + 1):
        if _deadline_left(deadline) < 5:
            log(f"  serial_cq round {round_idx}: time budget exhausted")
            break
        round_trials = [
            t for t in trials if t.crf is not None and t.stage.startswith(stage)
        ]
        obs = observations_from_trials(round_trials)
        cq, cq_reason = next_serial_cq_probe(
            obs,
            round_idx=round_idx,
            max_rounds=max_rounds,
            probe_plan=probe_plan,
            features=features,
            crf_min=req.crf_min,
            crf_max=req.crf_max,
            vmaf_threshold=float(req.vmaf_threshold),
            spread=max(1, req.crf_spread),
            crf_start=req.crf_start,
            used=used,
        )
        if cq is None:
            log(f"  serial_cq round {round_idx}: no CQ ({cq_reason})")
            break
        used.add(cq)
        log(f"  serial_cq round {round_idx}/{max_rounds}: CQ {cq} ({cq_reason})")
        round_prefix = f"{prefix}_r{round_idx}"
        best = _parallel_crf_trials(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=input_path,
            reference_path=reference_path,
            stage=f"{stage}_r{round_idx}",
            prefix=round_prefix,
            source_bitrate_mbps=source_bitrate_mbps,
            candidates=[cq],
        )
        if best is not None and _is_better_trial(best, local_best):
            local_best = best
    return local_best


def _interp_answer_crf_search(
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
    features: Optional[dict[str, Any]] = None,
) -> Optional[TrialResult]:
    """Two-round interpolated + answer-based CQ search."""
    count = max(1, req.crf_candidates)
    rounds = max(1, req.search_rounds)
    # NVENC RC → x265 handoff: run only Round 1, then map CQ answers to x265 CRFs.
    if req.nvenc_x265_handoff and req.encoder == "hevc_nvenc":
        if rounds > 1:
            log(
                "  interp_answer: NVENC Round 2 suppressed "
                "(RC handoff maps Round-1 CQ answers to x265 CRFs)"
            )
        rounds = 1
    used: set[int] = set()
    local_best: Optional[TrialResult] = None

    for round_idx in range(1, rounds + 1):
        if _deadline_left(deadline) < 5:
            log(f"  interp_answer round {round_idx}: time budget exhausted")
            break

        if round_idx == 1:
            candidates, seed, seed_reason = round1_feature_cqs(
                features,
                count=count,
                crf_min=req.crf_min,
                crf_max=req.crf_max,
                vmaf_threshold=float(req.vmaf_threshold),
                spread=max(1, req.crf_spread),
                crf_start=req.crf_start,
            )
            log(
                f"  interp_answer round 1/{rounds}: feature seed CQ {seed} "
                f"({seed_reason}) spread={req.crf_spread} → {candidates}"
            )
        else:
            round_trials = [
                t
                for t in trials
                if t.crf is not None and t.stage.startswith(stage)
            ]
            obs = observations_from_trials(round_trials)
            cq_star = interpolate_cq_for_vmaf(obs, float(req.vmaf_threshold))
            best_so_far = None
            positive = [o for o in obs if o.s_f > 0]
            if positive:
                best_so_far = max(positive, key=lambda o: (o.s_f, o.vmaf, -o.cq))

            round_prefix = f"{prefix}_r{round_idx}"
            if req.encoder == "hevc_nvenc":
                param_n = max(0, req.round2_nvenc_param_trials)
                cq_n = max(0, req.round2_cq_trials)
                trial_specs = propose_round2_mixed(
                    obs,
                    crf_min=req.crf_min,
                    crf_max=req.crf_max,
                    vmaf_threshold=float(req.vmaf_threshold),
                    used=used,
                    param_trials=param_n,
                    cq_trials=cq_n,
                    baseline_nvenc={
                        "nvenc_tune": req.nvenc_tune,
                        "nvenc_rc": req.nvenc_rc,
                        "nvenc_multipass": req.nvenc_multipass,
                        "nvenc_spatial_aq": req.nvenc_spatial_aq,
                        "nvenc_temporal_aq": req.nvenc_temporal_aq,
                        "nvenc_aq_strength": req.nvenc_aq_strength,
                        "nvenc_rc_lookahead": req.nvenc_rc_lookahead,
                        "nvenc_bf": req.nvenc_bf,
                        "nvenc_gop": req.nvenc_gop,
                        "nvenc_b_ref_mode": req.nvenc_b_ref_mode,
                    },
                    features=features,
                    preprocess_trial=(
                        req.round2_preprocess if req.round2_preprocess_trial else None
                    ),
                )
                if best_so_far is not None and cq_star is not None:
                    log(
                        f"  interp_answer round {round_idx}/{rounds}: "
                        f"best_cq={best_so_far.cq} cq_vmaf*={cq_star:.2f} "
                        f"→ {param_n} nvenc tune + {cq_n} cq refine"
                    )
                elif best_so_far is not None:
                    log(
                        f"  interp_answer round {round_idx}/{rounds}: "
                        f"best_cq={best_so_far.cq} "
                        f"→ {param_n} nvenc tune + {cq_n} cq refine"
                    )
                else:
                    log(f"  interp_answer round {round_idx}/{rounds}:")
                for spec in trial_specs:
                    log(
                        f"    trial CQ {spec.cq} "
                        f"nvenc={spec.nvenc.to_dict() or 'default'} "
                        f"pred_s_f={spec.predicted_s_f:.4f} ({spec.reason})"
                    )
                if not trial_specs:
                    log(f"  interp_answer round {round_idx}: no new trials")
                    break
                used.update(spec.cq for spec in trial_specs)
                best = _parallel_round2_trials(
                    req,
                    recipe,
                    work_dir,
                    deadline,
                    trials,
                    input_path=input_path,
                    reference_path=reference_path,
                    stage=f"{stage}_r{round_idx}",
                    prefix=round_prefix,
                    trial_specs=trial_specs,
                )
                if best is not None and _is_better_trial(best, local_best):
                    local_best = best
                continue

            proposals = propose_round2_details(
                obs,
                count=count,
                crf_min=req.crf_min,
                crf_max=req.crf_max,
                vmaf_threshold=float(req.vmaf_threshold),
                used=used,
            )
            candidates = [p.cq for p in proposals]
            if best_so_far is not None and cq_star is not None:
                log(
                    f"  interp_answer round {round_idx}/{rounds}: "
                    f"cq_vmaf*={cq_star:.2f} "
                    f"best_so_far=CQ{best_so_far.cq}/s_f={best_so_far.s_f:.4f}"
                )
            elif cq_star is not None:
                log(
                    f"  interp_answer round {round_idx}/{rounds}: "
                    f"cq_vmaf*={cq_star:.2f}"
                )
            elif best_so_far is not None:
                log(
                    f"  interp_answer round {round_idx}/{rounds}: "
                    f"best_so_far=CQ{best_so_far.cq}/s_f={best_so_far.s_f:.4f}"
                )
            else:
                log(f"  interp_answer round {round_idx}/{rounds}:")
            for p in proposals:
                log(
                    f"    next CQ {p.cq}: pred_s_f={p.predicted_s_f:.4f} ({p.reason})"
                )
            log(f"    candidates → {candidates}")

            if not candidates:
                log(f"  interp_answer round {round_idx}: no new candidates")
                break

            used.update(candidates)
            best = _parallel_crf_trials(
                req,
                recipe,
                work_dir,
                deadline,
                trials,
                input_path=input_path,
                reference_path=reference_path,
                stage=f"{stage}_r{round_idx}",
                prefix=round_prefix,
                source_bitrate_mbps=source_bitrate_mbps,
                candidates=candidates,
            )
            if best is not None and _is_better_trial(best, local_best):
                local_best = best
            continue

        if not candidates:
            log(f"  interp_answer round {round_idx}: no new candidates")
            break

        used.update(candidates)
        round_prefix = f"{prefix}_r{round_idx}"
        best = _parallel_crf_trials(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=input_path,
            reference_path=reference_path,
            stage=f"{stage}_r{round_idx}",
            prefix=round_prefix,
            source_bitrate_mbps=source_bitrate_mbps,
            candidates=candidates,
        )
        if best is not None and _is_better_trial(best, local_best):
            local_best = best

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
            f"  ← [final] CRF {chosen_crf}: neg={final_best.score.vmaf:.2f} "
            f"base={final_best.score.vmaf_base:.2f} "
            f"delta={final_best.score.vmaf_delta:.2f} "
            f"s_f={final_best.score.s_f:.4f} "
            f"compression_ratio={final_best.score.compression_ratio:.2f}x "
            f"{_bitrate_log(final_best.measured_bitrate_mbps)} "
            f"encode={final_best.encode_sec:.1f}s score={final_best.score_sec:.1f}s "
            f"total={final_best.elapsed_sec:.1f}s"
        )
    else:
        log(
            f"  ← [final] CRF {chosen_crf}: encode failed "
            f"encode={final_best.encode_sec:.1f}s total={final_best.elapsed_sec:.1f}s"
        )

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
    features: Optional[dict[str, Any]] = None,
) -> Optional[TrialResult]:
    """Full-file CRF/CQ search (grid or interpolated answer-based)."""
    # ss/t reserved for legacy callers; current full search always uses whole file.
    _ = (ss, t, vmaf_n_subsample)
    if req.search_strategy == "interp_answer":
        if req.serial_cq_search:
            return _serial_cq_probe_search(
                req,
                recipe,
                work_dir,
                deadline,
                trials,
                input_path=req.input_path,
                reference_path=reference_path,
                stage=stage,
                prefix=prefix,
                features=features,
            )
        return _interp_answer_crf_search(
            req,
            recipe,
            work_dir,
            deadline,
            trials,
            input_path=req.input_path,
            reference_path=reference_path,
            stage=stage,
            prefix=prefix,
            features=features,
        )
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


def _vbr_bitrate_grid(lo_mbps: float, hi_mbps: float, count: int) -> list[float]:
    """Evenly spaced bitrate candidates in Mbps (inclusive)."""
    lo = float(lo_mbps)
    hi = float(hi_mbps)
    n = max(1, int(count))
    if hi < lo:
        lo, hi = hi, lo
    if n == 1:
        return [round((lo + hi) / 2.0, 3)]
    span = hi - lo
    out: list[float] = []
    seen: set[float] = set()
    for i in range(n):
        mbps = round(lo + span * i / (n - 1), 3)
        if mbps not in seen:
            seen.add(mbps)
            out.append(mbps)
    return out


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
    """Search average bitrate targets; pick the trial with highest s_f.

    Round 1: linspace bitrates in [vbr_min_mbps_floor, target * max_ratio].
    Round 2: refine near the best so-far (lower if VMAF still clears, else nudge up).
    """
    target_mbps = _parse_bitrate_mbps(req.target_bitrate)
    if target_mbps is None:
        raise ValueError("target_bitrate is required for ABR mode")

    cap_mbps = target_mbps * req.vbr_max_ratio_to_target
    floor_mbps = max(req.vbr_min_mbps_floor, 0.05)
    if floor_mbps > cap_mbps:
        floor_mbps = cap_mbps

    rounds = max(1, req.search_rounds if req.search_strategy == "interp_answer" else 1)
    count = max(1, req.crf_candidates)
    used: set[float] = set()
    local_best: Optional[TrialResult] = None

    def _run_bitrate(mbps: float, round_idx: int) -> TrialResult:
        mbps = round(max(floor_mbps, min(mbps, cap_mbps)), 3)
        bitrate = _format_bitrate_mbps(mbps)
        out_path = (
            work_dir
            / f"{prefix}_r{round_idx}_{recipe.name}_vbr_{str(mbps).replace('.', '_')}M.mp4"
        )
        log(f"  → [{stage}_r{round_idx}] encoding VBR {bitrate} ...")
        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            input_path=req.input_path,
            reference_path=reference_path,
            crf=None,
            bitrate=bitrate,
            timeout=_deadline_left(deadline),
            stage=f"{stage}_r{round_idx}",
            ss=ss,
            t=t,
            vmaf_n_subsample=vmaf_n_subsample,
        )
        if trial.encode_ok:
            log(
                f"  ← [{stage}_r{round_idx}] VBR {bitrate}: "
                f"neg={trial.score.vmaf:.2f} "
                f"base={trial.score.vmaf_base:.2f} "
                f"delta={trial.score.vmaf_delta:.2f} "
                f"s_f={trial.score.s_f:.4f} "
                f"compression_ratio={trial.score.compression_ratio:.2f}x "
                f"{_bitrate_log(trial.measured_bitrate_mbps)} "
                f"encode={trial.encode_sec:.1f}s score={trial.score_sec:.1f}s "
                f"total={trial.elapsed_sec:.1f}s"
            )
        else:
            log(
                f"  ← [{stage}_r{round_idx}] VBR {bitrate}: encode failed "
                f"encode={trial.encode_sec:.1f}s total={trial.elapsed_sec:.1f}s"
            )
        return trial

    for round_idx in range(1, rounds + 1):
        if _deadline_left(deadline) < 5:
            log(f"  VBR round {round_idx}: time budget exhausted")
            break

        if round_idx == 1:
            candidates = _vbr_bitrate_grid(floor_mbps, cap_mbps, count)
            log(
                f"  VBR round 1/{rounds}: linspace "
                f"[{floor_mbps:.3f},{cap_mbps:.3f}] Mbps → {candidates}"
            )
        else:
            # Refine around best: prefer lower bitrate if still above threshold.
            if local_best is None or local_best.bitrate is None:
                log(f"  VBR round {round_idx}: no anchor — stop")
                break
            anchor = _parse_bitrate_mbps(local_best.bitrate) or cap_mbps
            if local_best.score.vmaf >= req.vmaf_threshold:
                # Push lower: between floor and anchor.
                lo = floor_mbps
                hi = max(floor_mbps, anchor * 0.92)
            elif local_best.score.vmaf >= req.vmaf_threshold - 5:
                # Soft zone: nudge up slightly.
                lo = anchor
                hi = min(cap_mbps, anchor * 1.15)
            else:
                lo = min(cap_mbps, anchor * 1.1)
                hi = min(cap_mbps, anchor * 1.35)
            raw = _vbr_bitrate_grid(lo, hi, count)
            candidates = [c for c in raw if c not in used]
            log(
                f"  VBR round {round_idx}/{rounds}: refine around "
                f"{anchor:.3f} Mbps (neg={local_best.score.vmaf:.2f}) → {candidates}"
            )

        candidates = [c for c in candidates if c not in used]
        if not candidates:
            log(f"  VBR round {round_idx}: no new candidates")
            break

        used.update(candidates)
        workers = max(1, min(req.max_workers, len(candidates)))
        timeout = _deadline_left(deadline)
        if timeout < 5:
            break

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_bitrate, mbps, round_idx): mbps for mbps in candidates
            }
            for fut in as_completed(futures):
                mbps = futures[fut]
                try:
                    trial = fut.result()
                except Exception as exc:
                    trial = TrialResult(
                        recipe=recipe.name,
                        mode=req.codec_mode,
                        crf=None,
                        bitrate=_format_bitrate_mbps(mbps),
                        path="",
                        score=_failed_score(str(exc)),
                        encode_ok=False,
                        encode_error=str(exc),
                        stage=f"{stage}_r{round_idx}",
                        encoder=req.encoder,
                    )
                    log(f"  ← [{stage}_r{round_idx}] VBR {mbps}: exception {exc}")
                trials.append(trial)
                if _is_better_trial(trial, local_best):
                    local_best = trial

    return local_best


def abr_refine_bitrate_candidates(
    anchor_mbps: float,
    *,
    floor_mbps: float,
    cap_mbps: float,
    count: int = 3,
) -> list[float]:
    """Quality-oriented ABR band at or below the NVENC measured operating point.

    Default ratios: 0.85×, 0.92×, 1.0× of the clamped anchor.
    """
    floor = max(0.05, float(floor_mbps))
    cap = max(floor, float(cap_mbps))
    anchor = min(max(float(anchor_mbps), floor), cap)
    n = max(1, int(count))
    if n == 1:
        return [round(anchor, 3)]

    # Prefer equal/lower bitrates than the NVENC operating point.
    ratios = [0.85, 0.92, 1.0]
    if n == 2:
        ratios = [0.92, 1.0]
    elif n > 3:
        # Extend downward evenly toward floor for larger grids.
        extra = n - 3
        low_ratio = max(floor / anchor, 0.70) if anchor > 0 else 0.70
        step = (0.85 - low_ratio) / (extra + 1)
        ratios = [low_ratio + step * (i + 1) for i in range(extra)] + ratios

    out: list[float] = []
    seen: set[float] = set()
    for ratio in ratios[:n]:
        mbps = round(min(max(anchor * ratio, floor), cap), 3)
        if mbps not in seen:
            seen.add(mbps)
            out.append(mbps)
    # Fill if clamping collapsed uniqueness.
    fill = floor
    while len(out) < n and fill <= cap + 1e-9:
        mbps = round(fill, 3)
        if mbps not in seen:
            seen.add(mbps)
            out.append(mbps)
        fill += max((cap - floor) / max(n, 1), 0.05)
    return sorted(out)[:n]


def refine_search_deadline(req: CompressionRequest, overall_deadline: float) -> float:
    """Return NVENC-search deadline with refine time reserved when enabled."""
    if not req.refine_with_libx265_enabled:
        return overall_deadline
    reserve = max(0.0, float(req.libx265_refine_time_sec))
    # Keep at least a few seconds for search so refine never starves Round 1 entirely.
    left = _deadline_left(overall_deadline)
    search_budget = max(5.0, left - reserve)
    return time.time() + search_budget


def affordable_refine_candidates(
    budget_sec: float,
    requested: int,
    workers: int,
    *,
    sec_per_candidate: float,
) -> int:
    """How many x265 candidates the remaining budget can realistically support.

    Assumes ``workers`` candidates run in parallel, each taking roughly
    ``sec_per_candidate``. Always returns at least 1 (callers apply their own
    minimum-budget skip guard before calling this).
    """
    requested = max(1, int(requested))
    workers = max(1, int(workers))
    per = max(1e-6, float(sec_per_candidate))
    batches = max(1, int(budget_sec // per))
    affordable = workers * batches
    return max(1, min(requested, affordable))


def refine_with_libx265(
    req: CompressionRequest,
    nvenc_best: TrialResult,
    work_dir: Path,
    deadline: float,
    trials: list[TrialResult],
    *,
    features: dict[str, float],
) -> Optional[TrialResult]:
    """Run a small measurement-driven libx265 refine; return best refine trial."""
    if not req.refine_with_libx265_enabled:
        return None
    if nvenc_best is None or not nvenc_best.encode_ok:
        log("  libx265 refine: skip (no valid NVENC winner)")
        return None

    timeout = _deadline_left(deadline)
    min_budget = max(5.0, float(req.libx265_refine_min_budget_sec))
    if timeout < min_budget:
        log(
            f"  libx265 refine: skip (only {timeout:.1f}s left, "
            f"need >= {min_budget:.1f}s)"
        )
        return None

    refine_recipes = select_recipes(
        features,
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )
    recipe = refine_recipes[0]
    count = max(1, req.libx265_refine_candidates)
    workers = max(1, min(req.libx265_refine_max_workers, count))
    stage = "libx265_refine"

    if req.libx265_feature_baseline:
        for line in describe_feature_x265_baseline(features):
            log(f"  libx265 feature baseline: {line}")
    log(
        f"  libx265 refine: preset={recipe.preset} workers={workers} "
        f"budget={timeout:.1f}s mode={req.codec_mode} "
        f"nvenc_s_f={nvenc_best.score.s_f:.4f}"
    )

    local_best: Optional[TrialResult] = None

    if req.is_abr:
        target_mbps = _parse_bitrate_mbps(req.target_bitrate)
        if target_mbps is None:
            log("  libx265 refine: skip (target_bitrate missing)")
            return None
        cap_mbps = target_mbps * req.vbr_max_ratio_to_target
        floor_mbps = max(req.vbr_min_mbps_floor, 0.05)
        if floor_mbps > cap_mbps:
            floor_mbps = cap_mbps

        measured = nvenc_best.measured_bitrate_mbps
        requested = _parse_bitrate_mbps(nvenc_best.bitrate) or target_mbps
        # Stay under the validator 1.10× gate; prefer measured operating point.
        soft_cap = min(cap_mbps, target_mbps * 1.05)
        if measured is not None and measured > 0:
            anchor = min(measured, soft_cap)
        else:
            anchor = min(requested, soft_cap)
        affordable = affordable_refine_candidates(
            timeout,
            count,
            workers,
            sec_per_candidate=req.libx265_refine_sec_per_candidate,
        )
        if affordable < count:
            log(
                f"  libx265 refine ABR: budget {timeout:.1f}s supports "
                f"{affordable}/{count} candidate(s) — reducing"
            )
        candidates = abr_refine_bitrate_candidates(
            anchor,
            floor_mbps=floor_mbps,
            cap_mbps=soft_cap,
            count=affordable,
        )
        workers = max(1, min(workers, len(candidates)))
        log(
            f"  libx265 refine ABR: anchor={anchor:.3f} Mbps "
            f"(measured={measured}) → {candidates}"
        )

        def _run_abr(mbps: float) -> TrialResult:
            bitrate = _format_bitrate_mbps(mbps)
            out_path = (
                work_dir
                / f"refine_{recipe.name}_vbr_{str(mbps).replace('.', '_')}M.mp4"
            )
            log(f"  → [{stage}] encoding libx265 ABR {bitrate} ...")
            trial = _encode_and_score(
                req,
                recipe,
                out_path,
                input_path=req.input_path,
                reference_path=req.input_path,
                crf=None,
                bitrate=bitrate,
                timeout=_deadline_left(deadline),
                stage=stage,
                vmaf_n_subsample=req.vmaf_n_subsample,
                encoder="libx265",
                preset=recipe.preset,
            )
            if trial.encode_ok:
                log(
                    f"  ← [{stage}] ABR {bitrate}: neg={trial.score.vmaf:.2f} "
                    f"base={trial.score.vmaf_base:.2f} "
                    f"delta={trial.score.vmaf_delta:.2f} "
                    f"s_f={trial.score.s_f:.4f} "
                    f"compression_ratio={trial.score.compression_ratio:.2f}x "
                    f"{_bitrate_log(trial.measured_bitrate_mbps)} "
                    f"encode={trial.encode_sec:.1f}s score={trial.score_sec:.1f}s"
                )
            else:
                log(
                    f"  ← [{stage}] ABR {bitrate}: encode failed "
                    f"encode={trial.encode_sec:.1f}s"
                )
            return trial

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_abr, mbps): mbps for mbps in candidates}
            for fut in as_completed(futures):
                mbps = futures[fut]
                try:
                    trial = fut.result()
                except Exception as exc:
                    trial = TrialResult(
                        recipe=recipe.name,
                        mode=req.codec_mode,
                        crf=None,
                        bitrate=_format_bitrate_mbps(mbps),
                        path="",
                        score=_failed_score(str(exc)),
                        encode_ok=False,
                        encode_error=str(exc),
                        stage=stage,
                        encoder="libx265",
                    )
                    log(f"  ← [{stage}] ABR {mbps}: exception {exc}")
                trials.append(trial)
                if _is_better_trial(trial, local_best):
                    local_best = trial
        return local_best

    # RC: VMAF-anchored x265 CRFs from NVENC Round-1 curve (not CQ+offset).
    candidates: list[int] = []
    if req.nvenc_x265_handoff:
        round1_trials = [
            t
            for t in trials
            if t.crf is not None
            and t.encoder == "hevc_nvenc"
            and t.stage.startswith("full")
        ]
        obs = observations_from_trials(round1_trials)
        if req.serial_cq_search:
            crf, reason = estimate_primary_x265_crf(
                obs,
                nvenc_cq_min=req.crf_min,
                nvenc_cq_max=req.crf_max,
                crf_min=req.x265_crf_floor,
                crf_max=req.x265_crf_ceiling,
                vmaf_threshold=float(req.vmaf_threshold),
            )
            if crf is not None:
                workers = 1
                log(f"  libx265 refine RC (serial): CRF {crf} ({reason})")
                candidates = [crf]
        else:
            proposals = propose_vmaf_anchored_crfs(
                obs,
                count=count,
                nvenc_cq_min=req.crf_min,
                nvenc_cq_max=req.crf_max,
                crf_min=req.x265_crf_floor,
                crf_max=req.x265_crf_ceiling,
                vmaf_threshold=float(req.vmaf_threshold),
                spread=max(1, req.libx265_refine_crf_spread),
            )
            if proposals:
                affordable = affordable_refine_candidates(
                    timeout,
                    len(proposals),
                    workers,
                    sec_per_candidate=req.libx265_refine_sec_per_candidate,
                )
                if affordable < len(proposals):
                    # Prefer higher CRF (more compression) when trimming budget.
                    proposals = sorted(proposals, key=lambda p: p.crf, reverse=True)[
                        :affordable
                    ]
                    proposals = sorted(proposals, key=lambda p: p.crf)
                    log(
                        f"  libx265 refine RC: budget {timeout:.1f}s supports "
                        f"{affordable} candidate(s) — keeping higher-CRF probes"
                    )
                workers = max(1, min(workers, len(proposals)))
                for prop in proposals:
                    log(
                        f"  libx265 refine RC: CRF {prop.crf} "
                        f"(target_vmaf≈{prop.target_vmaf:.1f}, {prop.reason})"
                    )
                candidates = [p.crf for p in proposals]
            else:
                log("  libx265 refine RC: no Round-1 observations — using CRF seed")

    if not candidates:
        seed = req.crf_start if req.crf_start is not None else recipe.crf_start
        candidates = candidate_crfs(
            seed,
            req.crf_min,
            req.crf_max,
            count=count,
            spread=max(1, req.libx265_refine_crf_spread),
        )
        log(f"  libx265 refine RC: seed={seed} → CRF {candidates}")

    def _run_rc(crf: int) -> TrialResult:
        out_path = work_dir / f"refine_{recipe.name}_crf{crf}.mp4"
        log(f"  → [{stage}] encoding libx265 CRF {crf} ...")
        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            input_path=req.input_path,
            reference_path=req.input_path,
            crf=crf,
            bitrate=None,
            timeout=_deadline_left(deadline),
            stage=stage,
            vmaf_n_subsample=req.vmaf_n_subsample,
            encoder="libx265",
            preset=recipe.preset,
        )
        if trial.encode_ok:
            log(
                f"  ← [{stage}] CRF {crf}: neg={trial.score.vmaf:.2f} "
                f"base={trial.score.vmaf_base:.2f} "
                f"delta={trial.score.vmaf_delta:.2f} "
                f"s_f={trial.score.s_f:.4f} "
                f"compression_ratio={trial.score.compression_ratio:.2f}x "
                f"{_bitrate_log(trial.measured_bitrate_mbps)} "
                f"encode={trial.encode_sec:.1f}s score={trial.score_sec:.1f}s"
            )
        else:
            log(
                f"  ← [{stage}] CRF {crf}: encode failed "
                f"encode={trial.encode_sec:.1f}s"
            )
        return trial

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_rc, crf): crf for crf in candidates}
        for fut in as_completed(futures):
            crf = futures[fut]
            try:
                trial = fut.result()
            except Exception as exc:
                trial = TrialResult(
                    recipe=recipe.name,
                    mode=req.codec_mode,
                    crf=crf,
                    bitrate=None,
                    path="",
                    score=_failed_score(str(exc)),
                    encode_ok=False,
                    encode_error=str(exc),
                    stage=stage,
                    encoder="libx265",
                )
                log(f"  ← [{stage}] CRF {crf}: exception {exc}")
            trials.append(trial)
            if _is_better_trial(trial, local_best):
                local_best = trial
    return local_best


def run_search(
    req: CompressionRequest,
    features: dict[str, float],
    segments: Optional[list[dict[str, Any]]] = None,
) -> SearchResult:
    started = time.time()
    overall_deadline = started + req.time_budget_sec
    search_deadline = refine_search_deadline(req, overall_deadline)

    work_dir = Path(req.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    recipes = select_recipes(
        features,
        req.vmaf_threshold,
        max_recipes=req.max_recipes,
        preset=req.preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )
    trials: list[TrialResult] = []
    final_best: Optional[TrialResult] = None
    proxy_path: Optional[str] = None
    proxy_windows: list[dict[str, Any]] = []

    use_proxy = bool(req.use_proxy and not req.is_abr and segments)
    if use_proxy:
        strategy = "proxy_then_full"
    elif not req.is_abr:
        strategy = (
            "interp_answer"
            if req.search_strategy == "interp_answer"
            else "parallel_crf"
        )
    else:
        strategy = (
            "vbr_interp_answer"
            if req.search_strategy == "interp_answer"
            else "vbr_search"
        )
    if req.refine_with_libx265_enabled:
        strategy = f"{strategy}+libx265_refine"

    log(
        f"  {strategy} search "
        f"({len(recipes)} recipe(s), workers={req.max_workers})"
    )
    if req.refine_with_libx265_enabled:
        log(
            f"  reserved {req.libx265_refine_time_sec:.1f}s for libx265 refine "
            f"(search budget={_deadline_left(search_deadline):.1f}s)"
        )

    for recipe in recipes:
        if _deadline_left(search_deadline) < 5:
            break
        if not req.is_abr and use_proxy:
            best_for_recipe, proxy_path, proxy_windows = search_crf_two_phase(
                req,
                recipe,
                work_dir,
                search_deadline,
                trials,
                segments=segments or [],
            )
        elif not req.is_abr:
            best_for_recipe = search_crf_on_source(
                req,
                recipe,
                work_dir,
                search_deadline,
                trials,
                reference_path=req.input_path,
                stage="full",
                ss=None,
                t=None,
                vmaf_n_subsample=req.vmaf_n_subsample,
                prefix="full",
                features=features,
            )
        else:
            best_for_recipe = search_vbr_on_source(
                req,
                recipe,
                work_dir,
                search_deadline,
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

    if req.refine_with_libx265_enabled and final_best is not None:
        refine_best = refine_with_libx265(
            req,
            final_best,
            work_dir,
            overall_deadline,
            trials,
            features=features,
        )
        if refine_best is not None and _is_better_trial(refine_best, final_best):
            log(
                f"  libx265 refine wins: s_f={refine_best.score.s_f:.4f} "
                f"(nvenc s_f={final_best.score.s_f:.4f})"
            )
            final_best = refine_best
        elif refine_best is not None:
            log(
                f"  keeping NVENC winner: s_f={final_best.score.s_f:.4f} "
                f"(refine best s_f={refine_best.score.s_f:.4f})"
            )
        else:
            log("  libx265 refine produced no valid trial — keeping NVENC winner")

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
