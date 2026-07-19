"""Fleet batch search: N videos in parallel, one CQ probe per video per wave.

Real-world constraint: GPU runs ~5 encodes at once (one per video). Each video
gets a single CQ trial per wave — no parallel CQ sweep on one file. After probe
rounds, estimate one x265 CRF per video and run final encodes in parallel waves.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any, Optional

from compress_util import format_compression, measure_compression
from crf_search import (
    CrfAttempt,
    CrfSearchResult,
    encoded_percent_size,
    round_crf_for_encode,
    search_crf,
)
from encoder import encode_hevc
from fleet_io import download_to_path, seconds_left, timeout_from_deadline, upload_presigned_put
from interp_search import (
    apply_feature_nvenc_baseline,
    cq_seed_from_features,
    pick_rule_anchored_crf,
    estimate_primary_x265_crf,
    next_serial_cq_probe,
    observations_from_trials,
    round1_feature_cqs,
)
from logutil import log
from param_tune import TuneTrialResult, run_param_tune_loop
from proxy import build_proxy_reference, select_proxy_windows
from recipes import HevcRecipe, describe_feature_x265_baseline, select_recipes
from request import CompressionRequest
from scene_detect import SceneSpan
from scoring import ScoreResult, score_candidate, validate_hevc_output
from search import (
    SearchResult,
    TrialResult,
    _encode_and_score,
    _failed_score,
    _is_better_trial,
    _measured_bitrate_mbps,
    _parallel_crf_trials,
    _parse_bitrate_mbps,
)


@dataclass
class FleetVideoJob:
    """One video in a fleet batch."""

    job_id: str
    input_path: str
    output_path: str
    work_dir: str
    input_url: str = ""
    upload_url: str = ""
    crf: Optional[int] = None
    libx265_params: Optional[str] = None
    features: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    scenes: list[SceneSpan] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)
    probe_plan: list[int] = field(default_factory=list)
    probe_seed: int = 0
    used_cqs: set[int] = field(default_factory=set)
    round_idx: int = 0
    probe_best: Optional[TrialResult] = None
    nvenc_best: Optional[TrialResult] = None  # legacy fleet batch path
    final_best: Optional[TrialResult] = None
    recipe: Optional[HevcRecipe] = None
    proxy_path: Optional[str] = None
    chosen_crf: Optional[int] = None
    chosen_bitrate: Optional[str] = None
    best_params: Optional[str] = None
    vmaf_threshold: Optional[int] = None
    param_tune_trials: int = 0
    param_tune_history: list[dict[str, Any]] = field(default_factory=list)
    uploaded: bool = False
    use_gpu: bool = False
    error: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)


def _fleet_sla_strategy(job: FleetVideoJob) -> str:
    if job.chosen_bitrate:
        if job.param_tune_trials > 0:
            return "fleet_sla_x265_vbr_param_tune"
        return "fleet_sla_x265_fixed_vbr"
    if job.param_tune_trials > 0:
        return "fleet_sla_x265_crf_param_tune"
    return "fleet_sla_x265_full_crf"


def assign_fleet_gpu_slots(
    jobs: list[FleetVideoJob],
    slots: int,
) -> None:
    """Mark the first ``slots`` jobs for GPU VMAF; encode is always libx265."""
    gpu_slots = max(0, int(slots))
    for index, job in enumerate(jobs):
        job.use_gpu = index < gpu_slots


def fleet_job_final_payload(
    job: FleetVideoJob,
    *,
    elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Final delivered result for one video (no probe/trial history)."""
    best = job.final_best
    strategy = _fleet_sla_strategy(job)
    result = SearchResult(
        best=best,
        trials=[],
        features=job.features,
        recipes=[job.recipe.name] if job.recipe else [],
        elapsed_sec=elapsed_sec,
        output_path=job.output_path if best and job.uploaded else None,
        strategy=strategy,
    )
    payload = result.to_final_dict()
    payload["job_id"] = job.job_id
    payload["input_path"] = job.input_path
    payload["uploaded"] = job.uploaded
    payload["error"] = job.error
    payload["use_gpu"] = job.use_gpu
    payload["stage_timings"] = dict(job.stage_timings)
    payload["chosen_crf"] = job.chosen_crf
    payload["chosen_bitrate"] = job.chosen_bitrate
    payload["target_bitrate"] = job.chosen_bitrate
    payload["libx265_params"] = job.best_params or (
        job.recipe.params if job.recipe else None
    )
    payload["vmaf_threshold"] = job.vmaf_threshold
    payload["features"] = job.features
    payload["param_tune_trials"] = job.param_tune_trials
    return payload


def save_best_encode_config(job: FleetVideoJob) -> Path:
    """Write best CRF/bitrate + x265 params + features into the per-video work dir."""
    path = Path(job.work_dir) / "best.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    best = job.final_best or job.probe_best
    payload: dict[str, Any] = {
        "job_id": job.job_id,
        "input_path": job.input_path,
        "vmaf_threshold": job.vmaf_threshold,
        "crf": job.chosen_crf if job.chosen_crf is not None else (best.crf if best else None),
        "bitrate": job.chosen_bitrate
        or (best.bitrate if best else None),
        "target_bitrate": job.chosen_bitrate,
        "libx265_params": job.best_params
        or (job.recipe.params if job.recipe else None),
        "features": job.features,
        "s_f": best.score.s_f if best else None,
        "vmaf": best.score.vmaf if best else None,
        "vmaf_base": best.score.vmaf_base if best else None,
        "compression_rate": best.score.compression_rate if best else None,
        "param_tune_trials": job.param_tune_trials,
        "param_tune_history": job.param_tune_history,
        "error": job.error or None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_fleet_job_result(
    job: FleetVideoJob,
    *,
    elapsed_sec: float = 0.0,
    run_id: Optional[str] = None,
    results_db_path: Optional[str] = None,
) -> Path:
    """Write ``work_dir/result.json`` and mirror into SQLite results DB."""
    path = Path(job.work_dir) / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = fleet_job_final_payload(job, elapsed_sec=elapsed_sec)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        save_best_encode_config(job)
    except Exception as exc:  # noqa: BLE001
        log(f"  [{job.job_id}] best.json write failed: {exc}")
    if run_id is not None or results_db_path is not None:
        try:
            from results_db import DEFAULT_DB_PATH, upsert_result

            db_path = results_db_path or str(DEFAULT_DB_PATH)
            upsert_result(payload, db_path=db_path, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            log(f"  [{job.job_id}] results DB write failed: {exc}")
    return path


def _job_request(template: CompressionRequest, job: FleetVideoJob) -> CompressionRequest:
    """Per-video request cloned from fleet template."""
    from dataclasses import replace

    req = replace(
        template,
        input_path=job.input_path,
        output_path=job.output_path,
        work_dir=job.work_dir,
        serial_cq_search=True,
        max_workers=1,
        crf=job.crf if job.crf is not None else template.crf,
        libx265_params=(
            job.libx265_params
            if job.libx265_params is not None
            else template.libx265_params
        ),
    )
    if req.encoder == "hevc_nvenc" and req.nvenc_feature_baseline and job.features:
        apply_feature_nvenc_baseline(req, job.features)
    return req


def _init_job(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    extractor: Any = None,
    deadline: Optional[float] = None,
) -> None:
    if not job.features:
        if extractor is not None and hasattr(extractor, "extract_for_path"):
            job.features, job.segments = extractor.extract_for_path(job.input_path)
        else:
            from feature_extractor import HEVCFeatureExtractor

            full = HEVCFeatureExtractor(job.input_path).extract_full(
                deadline=deadline,
            )
            job.features = full["summary"]
            job.segments = full.get("segments") or []
    if not job.probe_plan:
        plan, seed, reason = round1_feature_cqs(
            job.features,
            count=max(1, template.crf_candidates),
            crf_min=template.crf_min,
            crf_max=template.crf_max,
            vmaf_threshold=float(template.vmaf_threshold),
            spread=max(1, template.crf_spread),
            crf_start=template.crf_start,
        )
        job.probe_plan = plan
        log(
            f"  [{job.job_id}] probe plan seed={seed} ({reason}) → {plan}"
        )
    if job.recipe is None:
        job.recipe = select_recipes(
            job.features,
            template.vmaf_threshold,
            max_recipes=1,
            preset=template.preset,
            feature_baseline=bool(template.libx265_feature_baseline),
            params_override=template.libx265_params,
        )[0]
    Path(job.work_dir).mkdir(parents=True, exist_ok=True)


def _probe_wave(
    template: CompressionRequest,
    jobs: list[FleetVideoJob],
    *,
    deadline: float,
    wave_idx: int,
    max_rounds: int,
) -> None:
    """Run one CQ probe per active job in parallel (fleet_batch_size)."""
    batch_size = max(1, template.fleet_batch_size)
    tasks: list[tuple[FleetVideoJob, int, str]] = []

    for job in jobs:
        if job.round_idx >= max_rounds:
            continue
        round_idx = job.round_idx + 1
        obs = observations_from_trials(job.trials)
        cq, reason = next_serial_cq_probe(
            obs,
            round_idx=round_idx,
            max_rounds=max_rounds,
            probe_plan=job.probe_plan,
            features=job.features,
            crf_min=template.crf_min,
            crf_max=template.crf_max,
            vmaf_threshold=float(template.vmaf_threshold),
            spread=max(1, template.crf_spread),
            crf_start=template.crf_start,
            used=job.used_cqs,
        )
        if cq is None:
            job.round_idx = max_rounds
            continue
        job.used_cqs.add(cq)
        tasks.append((job, cq, reason, round_idx))

    if not tasks:
        return

    log(
        f"  fleet wave {wave_idx}: {len(tasks)} video(s), "
        f"1 CQ each, batch_size={batch_size}"
    )

    def _run_probe(
        item: tuple[FleetVideoJob, int, str, int],
    ) -> tuple[FleetVideoJob, Optional[TrialResult]]:
        job, cq, reason, round_idx = item
        req = _job_request(template, job)
        recipe = job.recipe
        assert recipe is not None
        log(f"  → [{job.job_id}] wave {wave_idx} CQ {cq} ({reason})")
        best = _parallel_crf_trials(
            req,
            recipe,
            Path(job.work_dir),
            deadline,
            job.trials,
            input_path=job.input_path,
            reference_path=job.input_path,
            stage="full",
            prefix=f"full_r{round_idx}",
            candidates=[cq],
        )
        job.round_idx += 1
        if best is not None:
            job.nvenc_best = (
                best
                if job.nvenc_best is None or _is_better_trial(best, job.nvenc_best)
                else job.nvenc_best
            )
        return job, best

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_run_probe, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                log(f"  fleet wave probe failed: {exc}")


def _final_x265_wave(
    template: CompressionRequest,
    jobs: list[FleetVideoJob],
    *,
    deadline: float,
) -> None:
    """Estimate one CRF per video and encode final outputs in parallel waves."""
    if not template.refine_with_libx265_enabled:
        for job in jobs:
            job.final_best = job.nvenc_best
        return

    batch_size = max(1, template.fleet_batch_size)
    tasks: list[tuple[FleetVideoJob, int, str]] = []

    for job in jobs:
        if job.nvenc_best is None or not job.nvenc_best.encode_ok:
            continue
        obs = observations_from_trials(job.trials)
        crf, reason = estimate_primary_x265_crf(
            obs,
            nvenc_cq_min=template.crf_min,
            nvenc_cq_max=template.crf_max,
            crf_min=template.x265_crf_floor,
            crf_max=template.x265_crf_ceiling,
            vmaf_threshold=float(template.vmaf_threshold),
        )
        if crf is None:
            job.final_best = job.nvenc_best
            continue
        tasks.append((job, crf, reason))

    if not tasks:
        return

    log(f"  fleet final x265: {len(tasks)} video(s), 1 CRF each")

    def _run_final(item: tuple[FleetVideoJob, int, str]) -> None:
        job, crf, reason = item
        req = _job_request(template, job)
        refine_recipes = select_recipes(
            job.features,
            template.vmaf_threshold,
            max_recipes=1,
            preset=template.libx265_refine_preset,
            feature_baseline=bool(template.libx265_feature_baseline),
            params_override=template.libx265_params,
        )
        recipe = refine_recipes[0]
        if template.libx265_feature_baseline:
            for line in describe_feature_x265_baseline(job.features):
                log(f"  [{job.job_id}] x265 baseline: {line}")
        log(f"  → [{job.job_id}] final libx265 CRF {crf} ({reason})")
        out_path = Path(job.work_dir) / f"final_{recipe.name}_crf{crf}.mp4"
        trial = _encode_and_score(
            req,
            recipe,
            out_path,
            input_path=job.input_path,
            reference_path=job.input_path,
            crf=crf,
            bitrate=None,
            timeout=max(5.0, deadline - time.time()),
            stage="libx265_final",
            vmaf_n_subsample=template.vmaf_n_subsample,
            encoder="libx265",
            preset=recipe.preset,
        )
        job.trials.append(trial)
        if trial.encode_ok:
            log(
                f"  ← [{job.job_id}] CRF {crf}: s_f={trial.score.s_f:.4f} "
                f"neg={trial.score.vmaf:.2f}"
            )
        if job.nvenc_best is not None and _is_better_trial(trial, job.nvenc_best):
            job.final_best = trial
        else:
            job.final_best = job.nvenc_best

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(_run_final, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                log(f"  fleet final encode failed: {exc}")


def _copy_outputs(template: CompressionRequest, jobs: list[FleetVideoJob]) -> None:
    for job in jobs:
        best = job.final_best or job.nvenc_best
        if best is None or not best.encode_ok or best.score.s_f <= 0:
            continue
        out = Path(job.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best.path, out)


def run_fleet_batch_search(
    template: CompressionRequest,
    jobs: list[FleetVideoJob],
    *,
    extractor: Optional[Any] = None,
) -> list[SearchResult]:
    """Fleet NVENC probe waves → estimated x265 final encodes."""
    from search import refine_search_deadline

    started = time.time()
    overall_deadline = started + template.time_budget_sec
    # Reserve refine budget before probes (same contract as single-video search).
    probe_deadline = refine_search_deadline(template, overall_deadline)
    max_rounds = max(1, template.crf_candidates)

    template.serial_cq_search = True
    log(
        f"Fleet batch: {len(jobs)} video(s), batch_size={template.fleet_batch_size}, "
        f"probe_rounds={max_rounds}"
    )

    for job in jobs:
        _init_job(template, job, extractor=extractor, deadline=probe_deadline)

    for wave in range(1, max_rounds + 1):
        if probe_deadline - time.time() < 10:
            log("  fleet: probe budget exhausted")
            break
        active = [j for j in jobs if j.round_idx < max_rounds]
        if not active:
            break
        _probe_wave(
            template,
            active,
            deadline=probe_deadline,
            wave_idx=wave,
            max_rounds=max_rounds,
        )

    if template.refine_with_libx265_enabled and time.time() < overall_deadline:
        _final_x265_wave(template, jobs, deadline=overall_deadline)
    else:
        for job in jobs:
            job.final_best = job.nvenc_best

    _copy_outputs(template, jobs)

    results: list[SearchResult] = []
    strategy = "fleet_serial_cq"
    if template.refine_with_libx265_enabled:
        strategy += "+x265_final"

    for job in jobs:
        best = job.final_best or job.nvenc_best
        results.append(
            SearchResult(
                best=best,
                trials=job.trials,
                features=job.features,
                recipes=[job.recipe.name] if job.recipe else [],
                elapsed_sec=time.time() - started,
                output_path=job.output_path if best and best.encode_ok else None,
                strategy=strategy,
            )
        )
    return results


def load_fleet_jobs(
    manifest_path: str,
    *,
    output_dir: str = "output/fleet",
    work_root: str = "work_fleet",
    limit: int = 0,
) -> list[FleetVideoJob]:
    """Load jobs from a manifest (one input path per line, or input\\toutput)."""
    path = Path(manifest_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    jobs: list[FleetVideoJob] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        inp = parts[0].strip()
        if len(parts) > 1 and parts[1].strip() and not parts[1].startswith("http"):
            out = parts[1].strip()
        else:
            stem = Path(inp).stem
            out = str(Path(output_dir) / f"{stem}_fleet.mp4")
        job_id = Path(inp).stem[:12] or f"v{i}"
        jobs.append(
            FleetVideoJob(
                job_id=job_id,
                input_path=inp,
                output_path=out,
                work_dir=str(Path(work_root) / job_id),
            )
        )
        if limit > 0 and len(jobs) >= limit:
            break
    return jobs


def _threshold_label(req: CompressionRequest) -> str:
    """Directory name for the active VMAF threshold (e.g. ``85``)."""
    return str(int(req.vmaf_threshold))


def _output_path_for_threshold(
    output_path: str,
    *,
    threshold: str,
    safe_id: str,
) -> str:
    """Nest delivered encodes under ``…/<threshold>/`` when not already nested."""
    path = Path(output_path)
    if path.parent.name == threshold:
        return str(path)
    return str(path.parent / threshold / path.name)


def fleet_jobs_from_request(req: CompressionRequest) -> list[FleetVideoJob]:
    """Materialize fleet jobs from the request JSON contract (local or remote).

    Work/output paths are nested by ``vmaf_threshold`` so 85/89/93 runs do not
    overwrite each other::

        work_fleet/<threshold>/<job_id>/
        output/fleet/<threshold>/<job_id>.mp4
    """
    jobs: list[FleetVideoJob] = []
    thr = _threshold_label(req)
    root = Path(req.work_dir) / thr
    for index, item in enumerate(req.jobs):
        job_id = item["id"] or f"video-{index + 1}"
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
        job_dir = root / safe_id
        if item.get("input_path"):
            input_path = item["input_path"]
            raw_out = item.get("output_path") or str(
                Path("output/fleet") / f"{safe_id}.mp4"
            )
            output_path = _output_path_for_threshold(
                raw_out, threshold=thr, safe_id=safe_id
            )
            jobs.append(
                FleetVideoJob(
                    job_id=job_id,
                    input_path=input_path,
                    output_path=output_path,
                    work_dir=str(job_dir),
                    crf=item.get("crf"),
                    libx265_params=item.get("libx265_params"),
                    vmaf_threshold=int(req.vmaf_threshold),
                )
            )
        else:
            jobs.append(
                FleetVideoJob(
                    job_id=job_id,
                    input_url=item["input_url"],
                    upload_url=item["upload_url"],
                    input_path=str(job_dir / "input.mp4"),
                    output_path=str(job_dir / "output.mp4"),
                    work_dir=str(job_dir),
                    crf=item.get("crf"),
                    libx265_params=item.get("libx265_params"),
                    vmaf_threshold=int(req.vmaf_threshold),
                )
            )
    return jobs


def _job_needs_download(job: FleetVideoJob, req: CompressionRequest) -> bool:
    return bool(job.input_url) and not req.skip_transfer


def _job_needs_upload(job: FleetVideoJob, req: CompressionRequest) -> bool:
    return bool(job.upload_url) and not req.skip_transfer


def _score_sla_final(
    req: CompressionRequest,
    job: FleetVideoJob,
    trial: TrialResult,
    *,
    deadline: float,
) -> TrialResult:
    """Score a full-file final encode: real size ratio + dual VMAF vs source."""
    started = time.monotonic()
    timeout = timeout_from_deadline(deadline, minimum=0.1)
    log(
        f"  [{job.job_id}] scoring {trial.encoder} "
        f"{format_compression(job.input_path, trial.path)} …"
    )
    score = score_candidate(
        job.input_path,
        trial.path,
        req.vmaf_threshold,
        ffmpeg_bin=req.ffmpeg_bin,
        ffprobe_bin=req.ffprobe_bin,
        vmaf_n_subsample=req.vmaf_n_subsample,
        vmaf_n_threads=req.vmaf_n_threads,
        vmaf_backend=req.vmaf_backend,
        vmaf_docker_image=req.vmaf_docker_image,
        vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
        codec_mode=req.codec_mode,
        target_bitrate_mbps=_parse_bitrate_mbps(req.target_bitrate)
        if req.is_abr
        else None,
        timeout=timeout,
    )
    score_sec = time.monotonic() - started
    measured = _measured_bitrate_mbps(trial.path, req.ffprobe_bin)
    encode_ok = trial.encode_ok and not score.validation_errors
    return TrialResult(
        recipe=trial.recipe,
        mode=trial.mode,
        crf=trial.crf,
        bitrate=trial.bitrate,
        path=trial.path,
        score=score,
        encode_ok=encode_ok,
        encode_error=trial.encode_error
        or ("; ".join(score.validation_errors) if score.validation_errors else ""),
        stage="sla_final",
        encoder=trial.encoder,
        measured_bitrate_mbps=measured,
        encode_sec=trial.encode_sec,
        score_sec=score_sec,
        elapsed_sec=trial.encode_sec + score_sec,
        nvenc_overrides=dict(trial.nvenc_overrides or {}),
    )


def _scenes_from_segments(segments: list[dict[str, Any]]) -> list[SceneSpan]:
    """Build CRF sample spans from feature-extractor cut segments."""
    scenes: list[SceneSpan] = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", 0.0))
        if end <= start:
            duration = float(seg.get("duration", 0.0))
            end = start + duration
        if end <= start:
            continue
        scenes.append(
            SceneSpan(
                index=int(seg.get("index", i)),
                start_sec=start,
                end_sec=end,
            )
        )
    return scenes


def _extract_sla_features(
    job: FleetVideoJob,
    sample_frames: int,
    *,
    deadline: float,
) -> None:
    # One decode pass: cut detection + per-segment features.
    del sample_frames
    from feature_extractor import HEVCFeatureExtractor

    full = HEVCFeatureExtractor(job.input_path).extract_full(deadline=deadline)
    job.features = full["summary"]
    job.segments = full.get("segments") or []
    job.scenes = _scenes_from_segments(job.segments)


def _prepare_sla_job(
    req: CompressionRequest,
    job: FleetVideoJob,
    *,
    download_deadline: float,
    preparation_deadline: float,
) -> None:
    started = time.monotonic()
    Path(job.work_dir).mkdir(parents=True, exist_ok=True)
    if _job_needs_download(job, req):
        transfer = download_to_path(
            job.input_url,
            job.input_path,
            deadline=download_deadline,
        )
        job.stage_timings["download"] = transfer.elapsed_sec
        if not transfer.ok:
            job.error = f"download failed: {transfer.error}"
            return
    else:
        job.stage_timings["download"] = 0.0
        src = Path(job.input_path)
        if not src.is_file():
            job.error = f"local input not found: {job.input_path}"
            return
        job.input_path = str(src.resolve())

    feature_started = time.monotonic()
    try:
        _extract_sla_features(
            job,
            req.sample_frames,
            deadline=preparation_deadline,
        )
    except Exception as exc:
        job.error = f"feature extraction failed: {exc}"
        return
    if not job.features:
        job.error = "feature extraction returned no features"
        return
    job.stage_timings["features"] = time.monotonic() - feature_started
    job.stage_timings["prepare"] = time.monotonic() - started


def _run_fixed_crf_probe(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    probe_deadline: float,
) -> None:
    """One full-file encode at request.crf + dual VMAF. No CRF search."""
    if job.error:
        return
    left = seconds_left(probe_deadline)
    if left < 30.0:
        job.error = f"insufficient budget for fixed CRF encode (left={left:.1f}s)"
        return

    req = _job_request(template, job)
    req.encoder = "libx265"
    req.max_workers = 1
    fixed_crf = int(req.crf)  # type: ignore[arg-type]
    recipe = select_recipes(
        job.features or {},
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )[0]
    job.recipe = recipe
    job.stage_timings["scene_samples"] = 0.0
    job.probe_seed = fixed_crf
    job.chosen_crf = fixed_crf
    job.probe_plan = [fixed_crf]

    work_dir = Path(job.work_dir)
    out = work_dir / f"full_crf{fixed_crf}.mp4"
    preset = req.libx265_refine_preset or "fast"
    fleet_n = max(1, int(template.fleet_batch_size))
    vmaf_threads = max(2, int(req.vmaf_n_threads) // fleet_n)

    log(
        f"  [{job.job_id}] fixed CRF {fixed_crf}: dual VMAF, preset={preset}, "
        f"feature_baseline={bool(req.libx265_feature_baseline)}, "
        f"params={'set' if req.libx265_params else 'default/feature'}"
    )
    if recipe.params:
        log(f"  [{job.job_id}] x265-params={recipe.params}")

    probe_started = time.monotonic()
    enc = encode_hevc(
        job.input_path,
        str(out),
        preset=preset,
        params=recipe.params,
        codec_mode=req.codec_mode,
        crf=fixed_crf,
        bitrate=None,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout_from_deadline(probe_deadline),
        encoder="libx265",
        preprocess=req.preprocess,
        libx265_profile=req.libx265_profile,
        progress_reference_path=job.input_path,
        progress_label=f"[{job.job_id}] fixed CRF{fixed_crf}",
        progress_interval_sec=15.0,
    )
    if not enc.ok:
        job.error = f"fixed CRF encode failed: {enc.stderr_tail}"
        job.stage_timings["crf_search"] = time.monotonic() - probe_started
        return

    score = score_candidate(
        job.input_path,
        str(out),
        req.vmaf_threshold,
        ffmpeg_bin=req.ffmpeg_bin,
        ffprobe_bin=req.ffprobe_bin,
        vmaf_n_subsample=req.vmaf_n_subsample,
        vmaf_n_threads=vmaf_threads,
        vmaf_backend=req.vmaf_backend,
        vmaf_docker_image=req.vmaf_docker_image,
        vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
        codec_mode=req.codec_mode,
        timeout=timeout_from_deadline(probe_deadline),
    )
    job.stage_timings["crf_search"] = time.monotonic() - probe_started
    if (
        score.vmaf <= 0
        or not score.passed_encoding_gates
        or not score.passed_vmaf_delta_gate
    ):
        job.error = f"fixed CRF score failed: {score.reason}"
        return

    base_txt = f"{score.vmaf_base:.2f}" if score.vmaf_base is not None else "n/a"
    log(
        f"  [{job.job_id}] fixed CRF {fixed_crf} → "
        f"vmaf_neg={score.vmaf:.2f}, base={base_txt}, "
        f"rate={score.compression_rate:.4f}, s_f={score.s_f:.4f}"
    )
    job.probe_best = TrialResult(
        recipe=recipe.name,
        mode=req.codec_mode,
        crf=fixed_crf,
        bitrate=None,
        path=str(out.resolve()),
        score=score,
        encode_ok=True,
        stage="sla_fixed_crf",
        encoder="libx265",
        encode_sec=job.stage_timings.get("crf_search", 0.0),
        score_sec=0.0,
        elapsed_sec=job.stage_timings.get("crf_search", 0.0),
    )
    job.nvenc_best = job.probe_best
    job.best_params = recipe.params


def _run_fixed_bitrate_probe(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    probe_deadline: float,
) -> None:
    """One full-file encode at target_bitrate + dual VMAF, then param tune."""
    if job.error:
        return
    left = seconds_left(probe_deadline)
    if left < 30.0:
        job.error = f"insufficient budget for fixed bitrate encode (left={left:.1f}s)"
        return

    req = _job_request(template, job)
    req.encoder = "libx265"
    req.max_workers = 1
    target_bitrate = str(req.target_bitrate or "").strip()
    if not target_bitrate:
        job.error = "target_bitrate is required for ABR/VBR fleet mode"
        return
    target_mbps = _parse_bitrate_mbps(target_bitrate)
    if target_mbps is None or target_mbps <= 0:
        job.error = f"invalid target_bitrate: {target_bitrate!r}"
        return

    recipe = select_recipes(
        job.features or {},
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )[0]
    job.recipe = recipe
    job.stage_timings["scene_samples"] = 0.0
    job.chosen_bitrate = target_bitrate
    job.chosen_crf = None
    job.probe_seed = 0
    job.probe_plan = []

    work_dir = Path(job.work_dir)
    safe_label = target_bitrate.replace("/", "_").replace(" ", "")
    out = work_dir / f"full_vbr_{safe_label}.mp4"
    preset = req.libx265_refine_preset or "fast"
    fleet_n = max(1, int(template.fleet_batch_size))
    vmaf_threads = max(2, int(req.vmaf_n_threads) // fleet_n)
    min_attempt_sec = min(120.0, max(45.0, float(template.probe_min_budget_sec) * 0.15))

    log(
        f"  [{job.job_id}] fixed VBR {target_bitrate}: dual VMAF, preset={preset}, "
        f"feature_baseline={bool(req.libx265_feature_baseline)}, "
        f"params={'set' if req.libx265_params else 'default/feature'}"
    )
    if recipe.params:
        log(f"  [{job.job_id}] x265-params={recipe.params}")

    probe_started = time.monotonic()
    enc = encode_hevc(
        job.input_path,
        str(out),
        preset=preset,
        params=recipe.params,
        codec_mode="ABR",
        crf=None,
        bitrate=target_bitrate,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout_from_deadline(probe_deadline),
        encoder="libx265",
        preprocess=req.preprocess,
        libx265_profile=req.libx265_profile,
        progress_reference_path=job.input_path,
        progress_label=f"[{job.job_id}] fixed VBR {target_bitrate}",
        progress_interval_sec=15.0,
    )
    if not enc.ok:
        job.error = f"fixed VBR encode failed: {enc.stderr_tail}"
        job.stage_timings["crf_search"] = time.monotonic() - probe_started
        return

    score = score_candidate(
        job.input_path,
        str(out),
        req.vmaf_threshold,
        ffmpeg_bin=req.ffmpeg_bin,
        ffprobe_bin=req.ffprobe_bin,
        vmaf_n_subsample=req.vmaf_n_subsample,
        vmaf_n_threads=vmaf_threads,
        vmaf_backend=req.vmaf_backend,
        vmaf_docker_image=req.vmaf_docker_image,
        vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
        codec_mode="ABR",
        target_bitrate_mbps=target_mbps,
        timeout=timeout_from_deadline(probe_deadline),
    )
    job.stage_timings["crf_search"] = time.monotonic() - probe_started
    if (
        score.vmaf <= 0
        or not score.passed_encoding_gates
        or not score.passed_vmaf_delta_gate
    ):
        job.error = f"fixed VBR score failed: {score.reason}"
        return

    base_txt = f"{score.vmaf_base:.2f}" if score.vmaf_base is not None else "n/a"
    log(
        f"  [{job.job_id}] fixed VBR {target_bitrate} → "
        f"vmaf_neg={score.vmaf:.2f}, base={base_txt}, "
        f"rate={score.compression_rate:.4f}, s_f={score.s_f:.4f}"
    )
    job.probe_best = TrialResult(
        recipe=recipe.name,
        mode="ABR",
        crf=None,
        bitrate=target_bitrate,
        path=str(out.resolve()),
        score=score,
        encode_ok=True,
        stage="sla_fixed_vbr",
        encoder="libx265",
        encode_sec=job.stage_timings.get("crf_search", 0.0),
        score_sec=0.0,
        elapsed_sec=job.stage_timings.get("crf_search", 0.0),
    )
    job.nvenc_best = job.probe_best
    job.best_params = recipe.params

    _run_param_tune_after_bitrate(
        template,
        job,
        req=req,
        preset=preset,
        vmaf_threads=vmaf_threads,
        probe_deadline=probe_deadline,
        min_attempt_sec=min_attempt_sec,
    )


def _run_param_tune_after_crf(
    _template: CompressionRequest,
    job: FleetVideoJob,
    *,
    req: CompressionRequest,
    preset: str,
    vmaf_threads: int,
    probe_deadline: float,
    min_attempt_sec: float,
) -> None:
    """Sequential x265 param tune + optional CRF+1, maximizing s_f."""
    if not bool(req.param_tune):
        return
    if job.error or job.probe_best is None or job.recipe is None:
        return
    probe = job.probe_best
    if (
        not probe.encode_ok
        or probe.score.s_f <= 0
        or not probe.score.passed_encoding_gates
        or not probe.score.passed_vmaf_delta_gate
    ):
        log(f"  [{job.job_id}] skip param tune: CRF probe score not usable")
        return

    work_dir = Path(job.work_dir)
    initial_params = job.best_params or job.recipe.params
    job.best_params = initial_params
    started = time.monotonic()
    trial_counter = {"n": 0}
    score_cache: dict[tuple[int, str], ScoreResult] = {
        (int(probe.crf or 0), initial_params): probe.score
    }
    path_cache: dict[tuple[int, str], str] = {
        (int(probe.crf or 0), initial_params): str(probe.path)
    }

    log(
        f"  [{job.job_id}] param tune: start CRF={probe.crf} s_f={probe.score.s_f:.4f} "
        f"vmaf={probe.score.vmaf:.2f} max_trials={req.param_tune_max_trials} "
        f"no_improve_stop={req.param_tune_no_improve_stop}"
    )

    def evaluate(crf: int, params: str) -> TuneTrialResult:
        left = seconds_left(probe_deadline)
        if left < min_attempt_sec:
            return TuneTrialResult(
                ok=False,
                crf=crf,
                params=params,
                reason="probe budget exhausted",
            )
        cache_key = (int(crf), str(params))
        if cache_key in score_cache:
            score = score_cache[cache_key]
            return TuneTrialResult(
                ok=True,
                crf=crf,
                params=params,
                s_f=float(score.s_f),
                vmaf=float(score.vmaf),
                path=path_cache[cache_key],
                reason="cache",
            )
        trial_counter["n"] += 1
        out = work_dir / f"tune_t{trial_counter['n']}_crf{crf}.mp4"
        enc = encode_hevc(
            job.input_path,
            str(out),
            preset=preset,
            params=params,
            codec_mode=req.codec_mode,
            crf=int(crf),
            bitrate=None,
            ffmpeg_bin=req.ffmpeg_bin,
            timeout=timeout_from_deadline(probe_deadline),
            encoder="libx265",
            preprocess=req.preprocess,
            libx265_profile=req.libx265_profile,
            progress_reference_path=job.input_path,
            progress_label=f"[{job.job_id}] tune CRF{crf} t{trial_counter['n']}",
            progress_interval_sec=15.0,
        )
        if not enc.ok:
            return TuneTrialResult(
                ok=False,
                crf=crf,
                params=params,
                reason=enc.stderr_tail or "encode failed",
            )
        score = score_candidate(
            job.input_path,
            str(out),
            req.vmaf_threshold,
            ffmpeg_bin=req.ffmpeg_bin,
            ffprobe_bin=req.ffprobe_bin,
            vmaf_n_subsample=req.vmaf_n_subsample,
            vmaf_n_threads=vmaf_threads,
            vmaf_backend=req.vmaf_backend,
            vmaf_docker_image=req.vmaf_docker_image,
            vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
            codec_mode=req.codec_mode,
            timeout=timeout_from_deadline(probe_deadline),
        )
        if (
            score.vmaf <= 0
            or not score.passed_encoding_gates
            or not score.passed_vmaf_delta_gate
            or score.s_f <= 0
        ):
            return TuneTrialResult(
                ok=False,
                crf=crf,
                params=params,
                s_f=float(score.s_f or 0.0),
                vmaf=float(score.vmaf or 0.0),
                path=str(out),
                reason=score.reason or "score gates failed",
            )
        path_cache[cache_key] = str(out.resolve())
        score_cache[cache_key] = score
        log(
            f"  [{job.job_id}] tune t{trial_counter['n']}: CRF={crf} "
            f"s_f={score.s_f:.4f} vmaf={score.vmaf:.2f} rate={score.compression_rate:.4f}"
        )
        return TuneTrialResult(
            ok=True,
            crf=crf,
            params=params,
            s_f=float(score.s_f),
            vmaf=float(score.vmaf),
            path=path_cache[cache_key],
            reason="ok",
        )

    state = run_param_tune_loop(
        initial_crf=int(probe.crf or job.chosen_crf or 0),
        initial_params=initial_params,
        initial_s_f=float(probe.score.s_f),
        initial_vmaf=float(probe.score.vmaf),
        initial_path=str(probe.path),
        features=job.features,
        evaluate=evaluate,
        vmaf_threshold=float(req.vmaf_threshold),
        crf_max=int(req.x265_crf_ceiling),
        max_trials=int(req.param_tune_max_trials),
        no_improve_stop=int(req.param_tune_no_improve_stop),
        vmaf_headroom=float(req.param_tune_vmaf_headroom),
        max_rounds=int(req.param_tune_max_rounds),
        allow_crf_bump=True,
    )
    job.stage_timings["param_tune"] = time.monotonic() - started
    job.param_tune_trials = int(state.trials)
    job.param_tune_history = list(state.history)
    job.chosen_crf = int(state.crf)
    job.best_params = state.params
    job.recipe = dc_replace(job.recipe, params=state.params)

    best_key = (int(state.crf), str(state.params))
    best_score = score_cache.get(best_key, probe.score)
    best_path = path_cache.get(best_key, state.path or probe.path)
    job.probe_best = TrialResult(
        recipe=job.recipe.name,
        mode=req.codec_mode,
        crf=int(state.crf),
        bitrate=None,
        path=best_path,
        score=best_score,
        encode_ok=True,
        stage="sla_param_tune",
        encoder="libx265",
    )
    job.nvenc_best = job.probe_best
    log(
        f"  [{job.job_id}] param tune done: trials={state.trials} "
        f"no_improve_streak={state.no_improve_streak} "
        f"best CRF={state.crf} s_f={state.s_f:.4f} vmaf={state.vmaf:.2f}"
    )
    log(f"  [{job.job_id}] best x265-params={state.params}")


def _run_param_tune_after_bitrate(
    _template: CompressionRequest,
    job: FleetVideoJob,
    *,
    req: CompressionRequest,
    preset: str,
    vmaf_threads: int,
    probe_deadline: float,
    min_attempt_sec: float,
) -> None:
    """Sequential x265 param tune at fixed bitrate (no CRF/bitrate bump)."""
    if not bool(req.param_tune):
        return
    if job.error or job.probe_best is None or job.recipe is None:
        return
    probe = job.probe_best
    bitrate = job.chosen_bitrate or probe.bitrate
    if not bitrate:
        log(f"  [{job.job_id}] skip param tune: missing bitrate")
        return
    if (
        not probe.encode_ok
        or probe.score.s_f <= 0
        or not probe.score.passed_encoding_gates
        or not probe.score.passed_vmaf_delta_gate
    ):
        log(f"  [{job.job_id}] skip param tune: VBR probe score not usable")
        return

    work_dir = Path(job.work_dir)
    initial_params = job.best_params or job.recipe.params
    job.best_params = initial_params
    target_mbps = _parse_bitrate_mbps(bitrate)
    started = time.monotonic()
    trial_counter = {"n": 0}
    score_cache: dict[tuple[str, str], ScoreResult] = {
        (str(bitrate), initial_params): probe.score
    }
    path_cache: dict[tuple[str, str], str] = {
        (str(bitrate), initial_params): str(probe.path)
    }

    log(
        f"  [{job.job_id}] param tune: start bitrate={bitrate} "
        f"s_f={probe.score.s_f:.4f} vmaf={probe.score.vmaf:.2f} "
        f"max_trials={req.param_tune_max_trials} "
        f"no_improve_stop={req.param_tune_no_improve_stop}"
    )

    def evaluate(_crf: int, params: str) -> TuneTrialResult:
        left = seconds_left(probe_deadline)
        if left < min_attempt_sec:
            return TuneTrialResult(
                ok=False,
                crf=0,
                params=params,
                bitrate=bitrate,
                reason="probe budget exhausted",
            )
        cache_key = (str(bitrate), str(params))
        if cache_key in score_cache:
            score = score_cache[cache_key]
            return TuneTrialResult(
                ok=True,
                crf=0,
                params=params,
                bitrate=bitrate,
                s_f=float(score.s_f),
                vmaf=float(score.vmaf),
                path=path_cache[cache_key],
                reason="cache",
            )
        trial_counter["n"] += 1
        out = work_dir / f"tune_t{trial_counter['n']}_vbr.mp4"
        enc = encode_hevc(
            job.input_path,
            str(out),
            preset=preset,
            params=params,
            codec_mode="ABR",
            crf=None,
            bitrate=bitrate,
            ffmpeg_bin=req.ffmpeg_bin,
            timeout=timeout_from_deadline(probe_deadline),
            encoder="libx265",
            preprocess=req.preprocess,
            libx265_profile=req.libx265_profile,
            progress_reference_path=job.input_path,
            progress_label=(
                f"[{job.job_id}] tune VBR {bitrate} t{trial_counter['n']}"
            ),
            progress_interval_sec=15.0,
        )
        if not enc.ok:
            return TuneTrialResult(
                ok=False,
                crf=0,
                params=params,
                bitrate=bitrate,
                reason=enc.stderr_tail or "encode failed",
            )
        score = score_candidate(
            job.input_path,
            str(out),
            req.vmaf_threshold,
            ffmpeg_bin=req.ffmpeg_bin,
            ffprobe_bin=req.ffprobe_bin,
            vmaf_n_subsample=req.vmaf_n_subsample,
            vmaf_n_threads=vmaf_threads,
            vmaf_backend=req.vmaf_backend,
            vmaf_docker_image=req.vmaf_docker_image,
            vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
            codec_mode="ABR",
            target_bitrate_mbps=target_mbps,
            timeout=timeout_from_deadline(probe_deadline),
        )
        if (
            score.vmaf <= 0
            or not score.passed_encoding_gates
            or not score.passed_vmaf_delta_gate
            or score.s_f <= 0
        ):
            return TuneTrialResult(
                ok=False,
                crf=0,
                params=params,
                bitrate=bitrate,
                s_f=float(score.s_f or 0.0),
                vmaf=float(score.vmaf or 0.0),
                path=str(out),
                reason=score.reason or "score gates failed",
            )
        path_cache[cache_key] = str(out.resolve())
        score_cache[cache_key] = score
        log(
            f"  [{job.job_id}] tune t{trial_counter['n']}: bitrate={bitrate} "
            f"s_f={score.s_f:.4f} vmaf={score.vmaf:.2f} rate={score.compression_rate:.4f}"
        )
        return TuneTrialResult(
            ok=True,
            crf=0,
            params=params,
            bitrate=bitrate,
            s_f=float(score.s_f),
            vmaf=float(score.vmaf),
            path=path_cache[cache_key],
            reason="ok",
        )

    state = run_param_tune_loop(
        initial_crf=0,
        initial_params=initial_params,
        initial_s_f=float(probe.score.s_f),
        initial_vmaf=float(probe.score.vmaf),
        initial_path=str(probe.path),
        features=job.features,
        evaluate=evaluate,
        vmaf_threshold=float(req.vmaf_threshold),
        crf_max=int(req.x265_crf_ceiling),
        max_trials=int(req.param_tune_max_trials),
        no_improve_stop=int(req.param_tune_no_improve_stop),
        vmaf_headroom=float(req.param_tune_vmaf_headroom),
        max_rounds=int(req.param_tune_max_rounds),
        bitrate=str(bitrate),
        allow_crf_bump=False,
    )
    job.stage_timings["param_tune"] = time.monotonic() - started
    job.param_tune_trials = int(state.trials)
    job.param_tune_history = list(state.history)
    job.chosen_bitrate = str(state.bitrate or bitrate)
    job.best_params = state.params
    job.recipe = dc_replace(job.recipe, params=state.params)

    best_key = (str(job.chosen_bitrate), str(state.params))
    best_score = score_cache.get(best_key, probe.score)
    best_path = path_cache.get(best_key, state.path or probe.path)
    job.probe_best = TrialResult(
        recipe=job.recipe.name,
        mode="ABR",
        crf=None,
        bitrate=job.chosen_bitrate,
        path=best_path,
        score=best_score,
        encode_ok=True,
        stage="sla_param_tune",
        encoder="libx265",
    )
    job.nvenc_best = job.probe_best
    log(
        f"  [{job.job_id}] param tune done: trials={state.trials} "
        f"no_improve_streak={state.no_improve_streak} "
        f"best bitrate={job.chosen_bitrate} s_f={state.s_f:.4f} vmaf={state.vmaf:.2f}"
    )
    log(f"  [{job.job_id}] best x265-params={state.params}")


def _run_scene_crf_probe(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    probe_deadline: float,
) -> None:
    """Full-file CRF search, or fixed CRF when ``request.crf`` is set."""
    if template.crf is not None or job.crf is not None:
        return _run_fixed_crf_probe(template, job, probe_deadline=probe_deadline)

    if job.error or seconds_left(probe_deadline) < template.probe_min_budget_sec:
        if not job.error:
            left = seconds_left(probe_deadline)
            job.error = (
                f"insufficient probe budget "
                f"(left={left:.1f}s < probe_min={template.probe_min_budget_sec:.1f}s)"
            )
        return

    if float(job.features.get("duration") or 0.0) <= 0:
        job.error = "feature extraction produced empty features (deadline/decode)"
        return

    req = _job_request(template, job)
    req.encoder = "libx265"
    req.max_workers = 1
    crf_min = float(req.x265_crf_floor)
    crf_max = float(req.x265_crf_ceiling)
    recipe = select_recipes(
        job.features,
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )[0]
    job.recipe = recipe
    job.stage_timings["scene_samples"] = 0.0

    work_dir = Path(job.work_dir)
    min_vmaf = float(req.vmaf_threshold)
    seed_crf, seed_reason = cq_seed_from_features(
        job.features,
        vmaf_threshold=min_vmaf,
        crf_min=int(crf_min),
        crf_max=int(crf_max),
        crf_start=req.crf_start,
    )
    job.probe_seed = seed_crf
    try:
        source_bytes = max(1, Path(job.input_path).stat().st_size)
    except OSError:
        source_bytes = 1

    probe_preset = req.libx265_refine_preset or "fast"
    fleet_n = max(1, int(template.fleet_batch_size))
    probe_vmaf_threads = max(2, int(req.vmaf_n_threads) // fleet_n)
    # Per-attempt floor for one full encode + dual VMAF — not the whole probe window.
    min_attempt_sec = min(120.0, max(45.0, float(template.probe_min_budget_sec) * 0.15))

    log(
        f"  [{job.job_id}] full-file CRF search: dual VMAF, preset={probe_preset}, "
        f"target VMAF {min_vmaf:.0f}, max size {req.crf_search_max_encoded_percent:.0f}%, "
        f"CRF {crf_min:.0f}..{crf_max:.0f}, seed={seed_crf} ({seed_reason})"
    )

    probe_started = time.monotonic()
    encoded_cache: dict[float, str] = {}
    scored_cache: dict[float, ScoreResult] = {}

    def evaluate(crf: float) -> CrfAttempt:
        left = seconds_left(probe_deadline)
        if left < min_attempt_sec:
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=0.0,
                per_sample_vmaf=[],
                encode_ok=False,
                error="probe budget exhausted",
            )
        crf_key = float(round_crf_for_encode(crf))
        enc_path = encoded_cache.get(crf_key)
        if enc_path is None:
            out = work_dir / f"full_crf{crf_key:.0f}.mp4"
            enc = encode_hevc(
                job.input_path,
                str(out),
                preset=probe_preset,
                params=recipe.params,
                codec_mode=req.codec_mode,
                crf=int(crf_key),
                bitrate=None,
                ffmpeg_bin=req.ffmpeg_bin,
                timeout=timeout_from_deadline(probe_deadline),
                encoder="libx265",
                preprocess=req.preprocess,
                libx265_profile=req.libx265_profile,
                progress_reference_path=job.input_path,
                progress_label=f"[{job.job_id}] probe CRF{int(crf_key)}",
                progress_interval_sec=15.0,
            )
            if not enc.ok:
                return CrfAttempt(
                    crf=crf,
                    q=0,
                    mean_vmaf=0.0,
                    per_sample_vmaf=[],
                    encode_ok=False,
                    error=enc.stderr_tail,
                )
            enc_path = str(out)
            encoded_cache[crf_key] = enc_path
        try:
            encoded_bytes = Path(enc_path).stat().st_size
        except OSError:
            encoded_bytes = 0

        score = scored_cache.get(crf_key)
        if score is None:
            score = score_candidate(
                job.input_path,
                enc_path,
                req.vmaf_threshold,
                ffmpeg_bin=req.ffmpeg_bin,
                ffprobe_bin=req.ffprobe_bin,
                vmaf_n_subsample=req.vmaf_n_subsample,
                vmaf_n_threads=probe_vmaf_threads,
                vmaf_backend=req.vmaf_backend,
                vmaf_docker_image=req.vmaf_docker_image,
                vmaf_docker_gpus=bool(job.use_gpu and req.vmaf_docker_gpus),
                codec_mode=req.codec_mode,
                timeout=timeout_from_deadline(probe_deadline),
            )
            scored_cache[crf_key] = score
        if (
            score.vmaf <= 0
            or not score.passed_encoding_gates
            or not score.passed_vmaf_delta_gate
        ):
            return CrfAttempt(
                crf=crf,
                q=0,
                mean_vmaf=0.0,
                per_sample_vmaf=[],
                encode_ok=False,
                error=score.reason,
            )
        encode_percent = encoded_percent_size(source_bytes, encoded_bytes)
        return CrfAttempt(
            crf=crf,
            q=0,
            mean_vmaf=float(score.vmaf),
            per_sample_vmaf=[float(score.vmaf)],
            encode_percent=encode_percent,
            encode_ok=True,
        )

    search: CrfSearchResult = search_crf(
        evaluate,
        min_vmaf=min_vmaf,
        crf_min=crf_min,
        crf_max=crf_max,
        crf_increment=float(req.crf_search_increment),
        max_encoded_percent=float(req.crf_search_max_encoded_percent),
        thorough=bool(req.crf_search_thorough),
        deadline=probe_deadline,
        max_runs=int(req.crf_search_max_runs),
        initial_crf=float(seed_crf),
        near_vmaf_band=float(req.crf_search_near_vmaf_band),
    )
    job.stage_timings["crf_search"] = time.monotonic() - probe_started

    if search.crf is not None:
        final_crf = round_crf_for_encode(search.crf)
        reason = search.reason
        if not search.ok:
            reason = f"fallback_{search.reason}"
            log(
                f"  [{job.job_id}] CRF search incomplete ({search.reason}); "
                f"using CRF {final_crf} (vmaf={search.mean_vmaf:.2f})"
            )
    else:
        final_crf = int(seed_crf)
        reason = f"seed_fallback_{search.reason}"
        log(
            f"  [{job.job_id}] CRF search failed ({search.reason}); "
            f"falling back to seed CRF {final_crf}"
        )

    job.chosen_crf = final_crf
    job.probe_plan = [final_crf]
    crf_key = float(final_crf)
    enc_path = encoded_cache.get(crf_key) or str(work_dir / f"full_crf{final_crf}.mp4")
    score = scored_cache.get(crf_key) or ScoreResult(
        s_f=0.0,
        vmaf=search.mean_vmaf,
        compression_rate=1.0,
        compression_ratio=1.0,
        compression_component=0.0,
        quality_component=0.0,
        reason=reason,
        validation_errors=[],
        passed_encoding_gates=True,
        passed_vmaf_delta_gate=False,
    )
    base_txt = (
        f"{score.vmaf_base:.2f}" if score.vmaf_base is not None else "n/a"
    )
    log(
        f"  [{job.job_id}] full-file CRF search → CRF {final_crf} "
        f"(vmaf_neg={score.vmaf:.2f}, base={base_txt}, "
        f"size={search.encode_percent:.1f}%, {reason})"
    )

    job.probe_best = TrialResult(
        recipe=recipe.name,
        mode=req.codec_mode,
        crf=final_crf,
        bitrate=None,
        path=enc_path,
        score=score,
        encode_ok=True,
        stage="sla_full_crf_search",
        encoder="libx265",
    )
    job.nvenc_best = job.probe_best
    job.best_params = recipe.params
    job.chosen_crf = final_crf

    _run_param_tune_after_crf(
        template,
        job,
        req=req,
        preset=probe_preset,
        vmaf_threads=probe_vmaf_threads,
        probe_deadline=probe_deadline,
        min_attempt_sec=min_attempt_sec,
    )


def _run_sla_probe(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    probe_deadline: float,
) -> None:
    if template.is_abr:
        return _run_fixed_bitrate_probe(template, job, probe_deadline=probe_deadline)

    if template.scene_crf_search:
        return _run_scene_crf_probe(template, job, probe_deadline=probe_deadline)

    if job.error or seconds_left(probe_deadline) < template.probe_min_budget_sec:
        if not job.error:
            job.error = "insufficient probe budget"
        return
    req = _job_request(template, job)
    req.encoder = "libx265"
    req.max_workers = 1
    crf_min = req.x265_crf_floor
    crf_max = req.x265_crf_ceiling
    recipe = select_recipes(
        job.features,
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=req.libx265_params,
    )[0]
    job.recipe = recipe

    probe_count = max(2, int(template.crf_candidates))
    crfs, seed, seed_reason = round1_feature_cqs(
        job.features,
        count=probe_count,
        crf_min=crf_min,
        crf_max=crf_max,
        vmaf_threshold=float(req.vmaf_threshold),
        spread=max(1, req.crf_spread),
        crf_start=req.crf_start,
    )
    job.probe_plan = crfs
    job.probe_seed = seed
    log(
        f"  [{job.job_id}] rule proxy plan CRF {crfs} "
        f"(seed={seed}, {seed_reason})"
    )

    windows = select_proxy_windows(
        job.segments,
        seconds_per_segment=req.proxy_seconds_per_segment,
        max_seconds=req.proxy_max_seconds,
        min_window_seconds=req.proxy_min_window_seconds,
    )
    proxy_path = Path(job.work_dir) / "probe_reference.mp4"
    proxy_started = time.monotonic()
    built = build_proxy_reference(
        job.input_path,
        str(proxy_path),
        windows,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout_from_deadline(probe_deadline),
        deadline=probe_deadline,
    )
    job.stage_timings["proxy"] = time.monotonic() - proxy_started
    if not built.ok:
        job.error = f"proxy failed: {built.error}"
        return
    job.proxy_path = built.path

    probe_started = time.monotonic()
    for crf in crfs:
        if seconds_left(probe_deadline) < template.probe_min_budget_sec:
            log(f"  [{job.job_id}] proxy probe budget exhausted after CRF {crf}")
            break
        trial = _encode_and_score(
            req,
            recipe,
            Path(job.work_dir) / f"probe_crf{crf}.mp4",
            input_path=built.path,
            reference_path=built.path,
            crf=crf,
            bitrate=None,
            timeout=timeout_from_deadline(probe_deadline),
            stage="sla_proxy_probe",
            vmaf_n_subsample=req.vmaf_n_subsample,
            encoder="libx265",
            preset=req.libx265_refine_preset,
            vmaf_docker_gpus=False,
            progress_reference_path=built.path,
            progress_label=f"[{job.job_id}] probe libx265 CRF{crf}",
            progress_interval_sec=10.0,
        )
        job.trials.append(trial)
        job.used_cqs.add(crf)
        if not trial.encode_ok or trial.score.vmaf <= 0:
            log(
                f"  [{job.job_id}] proxy CRF {crf} failed: "
                f"{trial.encode_error or trial.score.reason}"
            )
            continue
        if job.probe_best is None or _is_better_trial(trial, job.probe_best):
            job.probe_best = trial
        proxy_rate, proxy_ratio = measure_compression(built.path, trial.path)
        log(
            f"  [{job.job_id}] proxy CRF {crf}: neg={trial.score.vmaf:.2f} "
            f"s_f={trial.score.s_f:.4f} "
            f"proxy_rate={proxy_rate:.4f} proxy_ratio={proxy_ratio:.2f}x"
        )

    job.stage_timings["probe"] = time.monotonic() - probe_started
    if job.probe_best is None:
        job.error = "all proxy probes failed"
        return
    job.nvenc_best = job.probe_best  # legacy alias


def _encode_sla_candidate(
    req: CompressionRequest,
    job: FleetVideoJob,
    *,
    encoder_name: str,
    preset: str,
    recipe: HevcRecipe,
    crf: Optional[int],
    output_path: Path,
    final_deadline: float,
    bitrate: Optional[str] = None,
) -> TrialResult:
    """Full-file encode only; VMAF/scoring happens once on the chosen winner."""
    started = time.monotonic()
    use_abr = req.is_abr or bool(bitrate)
    if use_abr:
        if not bitrate:
            raise ValueError("bitrate is required for ABR SLA final encode")
        progress_label = f"[{job.job_id}] {encoder_name} VBR {bitrate}"
        enc_crf = None
        enc_bitrate = bitrate
        codec_mode = "ABR"
    else:
        if crf is None:
            raise ValueError("crf is required for RC SLA final encode")
        progress_label = f"[{job.job_id}] {encoder_name} CRF{crf}"
        enc_crf = int(crf)
        enc_bitrate = None
        codec_mode = req.codec_mode
    result = encode_hevc(
        job.input_path,
        str(output_path),
        preset=preset,
        params=recipe.params,
        codec_mode=codec_mode,
        crf=enc_crf,
        bitrate=enc_bitrate,
        ffmpeg_bin=req.ffmpeg_bin,
        timeout=timeout_from_deadline(final_deadline, minimum=0.1),
        encoder=encoder_name,
        preprocess=req.preprocess,
        libx265_profile=req.libx265_profile if encoder_name == "libx265" else None,
        nvenc_tune=req.nvenc_tune,
        nvenc_rc=req.nvenc_rc,
        nvenc_multipass=req.nvenc_multipass,
        nvenc_spatial_aq=req.nvenc_spatial_aq,
        nvenc_temporal_aq=req.nvenc_temporal_aq,
        nvenc_aq_strength=req.nvenc_aq_strength,
        nvenc_rc_lookahead=req.nvenc_rc_lookahead,
        nvenc_bf=req.nvenc_bf,
        nvenc_gop=req.nvenc_gop,
        nvenc_b_ref_mode=req.nvenc_b_ref_mode,
        nvenc_gpu=req.nvenc_gpu,
        nvenc_hwaccel=req.nvenc_hwaccel,
        progress_reference_path=job.input_path,
        progress_label=progress_label,
        progress_interval_sec=15.0,
    )
    elapsed = time.monotonic() - started
    if not result.ok:
        return TrialResult(
            recipe=recipe.name,
            mode=codec_mode,
            crf=enc_crf,
            bitrate=enc_bitrate,
            path=str(output_path),
            score=_failed_score(result.stderr_tail),
            encode_ok=False,
            encode_error=result.stderr_tail,
            stage="sla_final",
            encoder=encoder_name,
            encode_sec=elapsed,
            elapsed_sec=elapsed,
        )
    validation = validate_hevc_output(
        str(output_path),
        req.ffprobe_bin,
        timeout=timeout_from_deadline(final_deadline, minimum=0.1),
    )
    return TrialResult(
        recipe=recipe.name,
        mode=codec_mode,
        crf=enc_crf,
        bitrate=enc_bitrate,
        path=str(output_path),
        score=_failed_score("; ".join(validation.errors))
        if not validation.ok
        else ScoreResult(
            s_f=0.0,
            vmaf=0.0,
            compression_rate=1.0,
            compression_ratio=1.0,
            compression_component=0.0,
            quality_component=0.0,
            reason="awaiting_full_score",
            validation_errors=[],
            passed_encoding_gates=True,
            passed_vmaf_delta_gate=False,
        ),
        encode_ok=validation.ok,
        encode_error="; ".join(validation.errors),
        stage="sla_final",
        encoder=encoder_name,
        encode_sec=elapsed,
        elapsed_sec=elapsed,
    )


def _finalize_and_upload_sla_job(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    final_deadline: float,
    overall_deadline: float,
) -> None:
    if job.error or job.probe_best is None:
        return

    req = _job_request(template, job)
    req.encoder = "libx265"
    params_override = job.best_params if job.best_params is not None else req.libx265_params
    x265_recipe = select_recipes(
        job.features,
        req.vmaf_threshold,
        max_recipes=1,
        preset=req.libx265_refine_preset,
        feature_baseline=bool(req.libx265_feature_baseline),
        params_override=params_override,
    )[0]
    if job.best_params:
        x265_recipe = dc_replace(x265_recipe, params=job.best_params)

    bitrate: Optional[str] = None
    crf: Optional[int] = None
    if template.is_abr or job.chosen_bitrate:
        bitrate = job.chosen_bitrate or req.target_bitrate
        if not bitrate:
            job.error = "target_bitrate is required for ABR/VBR finalize"
            return
        reason = (
            "vbr_param_tune" if job.param_tune_trials > 0 else "fixed_vbr"
        )
        log(f"  [{job.job_id}] final libx265 VBR {bitrate} ({reason})")
    elif job.chosen_crf is not None:
        crf = job.chosen_crf
        reason = (
            "fixed_crf"
            if template.crf is not None or job.crf is not None
            else (
                "crf_param_tune"
                if job.param_tune_trials > 0
                else "full_crf_search"
            )
        )
        log(f"  [{job.job_id}] final libx265 CRF {crf} ({reason})")
    else:
        proxy_trials = [t for t in job.trials if t.stage == "sla_proxy_probe" and t.encode_ok]
        observations = observations_from_trials(proxy_trials)
        crf, reason = pick_rule_anchored_crf(
            observations,
            seed=job.probe_seed,
            candidates=job.probe_plan,
            crf_min=template.x265_crf_floor,
            crf_max=template.x265_crf_ceiling,
            vmaf_threshold=float(template.vmaf_threshold),
            proxy_vmaf_margin=float(template.proxy_vmaf_margin),
            mashup_push_ceiling=float(template.proxy_mashup_push_ceiling),
            features=job.features,
        )
        if crf is None:
            job.error = "unable to estimate x265 CRF from probe"
            return
        log(f"  [{job.job_id}] final libx265 CRF {crf} ({reason})")

    # Reuse the full-file probe encode when it already has evaluation dual-VMAF.
    probe = job.probe_best
    if bitrate is not None:
        can_reuse_probe = (
            probe is not None
            and probe.encode_ok
            and str(probe.bitrate or "") == str(bitrate)
            and probe.score.vmaf_base is not None
            and probe.score.passed_vmaf_delta_gate
            and Path(probe.path).is_file()
        )
    else:
        can_reuse_probe = (
            probe is not None
            and probe.encode_ok
            and probe.crf == crf
            and probe.score.vmaf_base is not None
            and probe.score.passed_vmaf_delta_gate
            and Path(probe.path).is_file()
        )
    final_started = time.monotonic()
    if can_reuse_probe:
        final_path = Path(job.work_dir) / "final_x265.mp4"
        shutil.copy2(probe.path, final_path)
        chosen = TrialResult(
            recipe=probe.recipe,
            mode=probe.mode,
            crf=crf if bitrate is None else None,
            bitrate=bitrate if bitrate is not None else probe.bitrate,
            path=str(final_path),
            score=probe.score,
            encode_ok=True,
            stage="sla_final",
            encoder="libx265",
            encode_sec=probe.encode_sec,
            score_sec=probe.score_sec,
            elapsed_sec=probe.elapsed_sec,
        )
        job.trials.append(chosen)
        job.stage_timings["final_encode"] = time.monotonic() - final_started
        job.stage_timings["final_score"] = 0.0
        job.final_best = chosen
        log(
            f"  [{job.job_id}] reused probe encode as final "
            f"neg={chosen.score.vmaf:.2f} "
            f"base={chosen.score.vmaf_base:.2f} "
            f"rate={chosen.score.compression_rate:.4f} "
            f"s_f={chosen.score.s_f:.4f}"
        )
    else:
        chosen = _encode_sla_candidate(
            req,
            job,
            encoder_name="libx265",
            preset=req.libx265_refine_preset,
            recipe=x265_recipe,
            crf=crf,
            bitrate=bitrate,
            output_path=Path(job.work_dir) / "final_x265.mp4",
            final_deadline=final_deadline,
        )
        job.trials.append(chosen)
        job.stage_timings["final_encode"] = time.monotonic() - final_started

        if not chosen.encode_ok:
            job.error = "libx265 final encode failed"
            return

        # One full dual-VMAF + real size compression against the source file.
        score_started = time.monotonic()
        if overall_deadline is not None and seconds_left(overall_deadline) < 5.0:
            job.error = "insufficient budget for final dual-VMAF score"
            return
        scored = _score_sla_final(req, job, chosen, deadline=overall_deadline)
        job.stage_timings["final_score"] = time.monotonic() - score_started
        job.trials.append(scored)
        job.final_best = scored
        log(
            f"  [{job.job_id}] final score encoder={scored.encoder} "
            f"neg={scored.score.vmaf:.2f} "
            f"rate={scored.score.compression_rate:.4f} "
            f"ratio={scored.score.compression_ratio:.2f}x "
            f"s_f={scored.score.s_f:.4f} ({scored.score_sec:.1f}s)"
        )
        if scored.score.s_f <= 0 and scored.score.validation_errors:
            job.error = f"final score failed: {scored.score.reason}"
            return
        chosen = scored

    out = Path(job.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job.final_best.path, out)
    if _job_needs_upload(job, template):
        upload_started = time.monotonic()
        uploaded = upload_presigned_put(
            job.output_path,
            job.upload_url,
            deadline=overall_deadline,
        )
        job.stage_timings["upload"] = time.monotonic() - upload_started
        job.uploaded = uploaded.ok
        if not uploaded.ok:
            job.error = f"upload failed: {uploaded.error}"
    else:
        job.stage_timings["upload"] = 0.0
        job.uploaded = True  # local deliver counts as success
        log(f"  [{job.job_id}] wrote local output → {job.output_path}")


def _run_sla_job(
    template: CompressionRequest,
    job: FleetVideoJob,
    *,
    download_deadline: Optional[float],
    preparation_deadline: Optional[float],
    probe_deadline: Optional[float],
    final_deadline: Optional[float],
    overall_deadline: Optional[float],
    run_id: Optional[str] = None,
    results_db_path: Optional[str] = None,
) -> None:
    """Full per-video SLA path so slow downloads do not block ready peers."""
    job_started = time.monotonic()
    try:
        _prepare_sla_job(
            template,
            job,
            download_deadline=download_deadline,
            preparation_deadline=preparation_deadline,
        )
        if job.error:
            return
        _run_sla_probe(template, job, probe_deadline=probe_deadline)
        if job.error:
            return
        _finalize_and_upload_sla_job(
            template,
            job,
            final_deadline=final_deadline,
            overall_deadline=overall_deadline,
        )
    finally:
        result_path = save_fleet_job_result(
            job,
            elapsed_sec=time.monotonic() - job_started,
            run_id=run_id,
            results_db_path=results_db_path,
        )
        log(f"  [{job.job_id}] wrote final result → {result_path}")


def run_fleet_sla(
    template: CompressionRequest,
    jobs: list[FleetVideoJob],
    *,
    run_id: Optional[str] = None,
    results_db_path: Optional[str] = None,
) -> list[SearchResult]:
    """Fleet SLA: libx265 CRF or fixed-VBR search → param tune → dual VMAF → results.

    Jobs are processed in waves of ``fleet_batch_size``. When
    ``time_budget_sec > 0``, each wave gets that budget. When
    ``time_budget_sec == 0``, there is no wall-clock deadline.
    """
    if template.skip_transfer or all(not j.input_url for j in jobs):
        template.skip_transfer = True
        template.download_reserve_sec = 0.0
        template.upload_reserve_sec = 0.0

    run_started = time.monotonic()
    unlimited = float(template.time_budget_sec) <= 0
    final_reserve = 0.0 if unlimited else max(0.0, float(template.final_encode_reserve_sec))
    upload_reserve = 0.0 if unlimited else max(0.0, float(template.upload_reserve_sec))
    probe_min = 0.0 if unlimited else max(0.0, float(template.probe_min_budget_sec))
    final_score_reserve_sec = 0.0
    if not unlimited:
        prepare_floor = 0.0 if final_reserve <= 0 and probe_min <= 0 else 45.0
        available = float(template.time_budget_sec) - upload_reserve
        if prepare_floor > 0 and final_reserve + probe_min + prepare_floor > available:
            final_reserve = max(0.0, available - probe_min - prepare_floor)
            if final_reserve + probe_min + prepare_floor > available:
                probe_min = max(0.0, available - final_reserve - prepare_floor)
            log(
                f"  budget clamp: final_reserve={final_reserve:.0f}s "
                f"probe_min={probe_min:.0f}s prepare_floor={prepare_floor:.0f}s "
                f"(from time_budget={template.time_budget_sec:.0f}s)"
            )
        if final_reserve > 0:
            final_score_reserve_sec = min(40.0, max(20.0, final_reserve * 0.35))
    template.final_encode_reserve_sec = final_reserve
    template.probe_min_budget_sec = probe_min

    workers = max(1, min(template.fleet_batch_size, len(jobs)))
    mode = "local" if template.skip_transfer else "http"
    wave_count = (len(jobs) + workers - 1) // workers
    budget_txt = "unlimited" if unlimited else f"{template.time_budget_sec:.0f}s"
    log(
        f"Fleet SLA ({mode}): jobs={len(jobs)} workers={workers} "
        f"waves={wave_count} gpu_slots={template.fleet_gpu_slots} "
        f"deadline_per_wave={budget_txt} "
        f"final_reserve={final_reserve:.0f}s "
        f"probe_min={probe_min:.0f}s "
        f"score_reserve={final_score_reserve_sec:.0f}s "
        f"upload_reserve={upload_reserve:.0f}s"
    )

    for wave_idx in range(wave_count):
        wave = jobs[wave_idx * workers : (wave_idx + 1) * workers]
        assign_fleet_gpu_slots(wave, template.fleet_gpu_slots)
        if unlimited:
            overall_deadline = None
            probe_deadline = None
            download_deadline = None
            preparation_deadline = None
            final_deadline = None
            probe_window_txt = "unlimited"
        else:
            wave_started = time.monotonic()
            overall_deadline = wave_started + float(template.time_budget_sec)
            probe_deadline = overall_deadline - (final_reserve + upload_reserve)
            download_deadline = min(
                wave_started + max(0.0, template.download_reserve_sec),
                probe_deadline - probe_min if probe_min > 0 else probe_deadline,
            )
            preparation_deadline = (
                probe_deadline - probe_min if probe_min > 0 else probe_deadline
            )
            final_deadline = overall_deadline - upload_reserve - final_score_reserve_sec
            probe_window_txt = f"{max(0.0, probe_deadline - wave_started):.0f}s"
        gpu_jobs = [j.job_id for j in wave if j.use_gpu]
        cpu_jobs = [j.job_id for j in wave if not j.use_gpu]
        log(
            f"  wave {wave_idx + 1}/{wave_count}: jobs={[j.job_id for j in wave]} "
            f"gpu={gpu_jobs or '-'} cpu={cpu_jobs or '-'} "
            f"probe_window={probe_window_txt}"
        )
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [
                pool.submit(
                    _run_sla_job,
                    template,
                    job,
                    download_deadline=download_deadline,
                    preparation_deadline=preparation_deadline,
                    probe_deadline=probe_deadline,
                    final_deadline=final_deadline,
                    overall_deadline=overall_deadline,
                    run_id=run_id,
                    results_db_path=results_db_path,
                )
                for job in wave
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    log(f"  fleet SLA job crashed: {exc}")

    elapsed = time.monotonic() - run_started
    results: list[SearchResult] = []
    for job in jobs:
        best = job.final_best
        if job.error:
            log(f"  [{job.job_id}] failed: {job.error}")
        else:
            log(
                f"  [{job.job_id}] uploaded={job.uploaded} "
                f"encoder={best.encoder if best else '?'} timings={job.stage_timings}"
            )
        results.append(
            SearchResult(
                best=best,
                trials=[],
                features=job.features,
                recipes=[job.recipe.name] if job.recipe else [],
                elapsed_sec=elapsed,
                output_path=job.output_path if best and job.uploaded else None,
                strategy=_fleet_sla_strategy(job),
            )
        )
    return results
