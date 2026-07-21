#!/usr/bin/env python3
"""Single-page dashboard: original vs compressed video + VM resource monitor."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import unquote, urlparse, parse_qs

import psutil

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
STATIC = Path(__file__).resolve().parent / "static"
COMPRESSION_DIR = ROOT / "s3_videos" / "compression"
VIDEO_DIR = WORKSPACE / "video"
REQUEST_JSON = ROOT / "request.json"
BATCH_RESULTS = ROOT / "work_fleet" / "batch_results.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_compression_log import (  # noqa: E402
    filter_scores,
    parse_log,
    parse_log_lines,
    parse_uid_args,
)
from vbr_mode import build_vbr_mode_params  # noqa: E402

_DEFAULT_LIBX265_PARAMS = build_vbr_mode_params({})[0]

HOST = os.environ.get("WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEB_PORT", "8082"))
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "vidaio_vidaio")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sn85-validators")
# Optional override. When unset, the dashboard scans the workspace for validator runs.
WANDB_RUN = os.environ.get("WANDB_RUN", "").strip()
WANDB_RUN_SCAN_LIMIT = int(os.environ.get("WANDB_RUN_SCAN_LIMIT", "20"))

_fetch_lock = threading.Lock()
_FETCH_LOG_MAX = 120_000
_fetch_status: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "total": 0,
    "done": 0,
    "skipped": 0,
    "failed": 0,
    "current": None,
    "wandb_run": None,
    "message": "",
    "errors": [],
    "log": "",
}


def _append_fetch_log(line: str) -> None:
    with _fetch_lock:
        existing = str(_fetch_status.get("log") or "")
        chunk = line if line.endswith("\n") else f"{line}\n"
        merged = existing + chunk
        if len(merged) > _FETCH_LOG_MAX:
            merged = merged[-_FETCH_LOG_MAX:]
        _fetch_status["log"] = merged


_compress_lock = threading.Lock()
_COMPRESS_LOG_MAX = 200_000
_compress_proc: subprocess.Popen[str] | None = None
_compress_stop_requested = False
_compress_status: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "message": "",
    "videos": [],
    "total": 0,
    "exit_code": None,
    "errors": [],
    "params": {},
    "log": "",
}

_analyze_lock = threading.Lock()
_analyze_status: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "message": "",
    "wandb_run": None,
    "errors": [],
    "params": {},
    "result": None,
    "has_cache": False,
}
# In-memory score rows from the last WandB fetch (Search reuses these).
_analyze_score_cache: list[Any] = []
_ANALYZE_CACHE_PATH = ROOT / "wandb_output.log"


def _append_compress_log(line: str) -> None:
    with _compress_lock:
        existing = str(_compress_status.get("log") or "")
        chunk = line if line.endswith("\n") else f"{line}\n"
        merged = existing + chunk
        if len(merged) > _COMPRESS_LOG_MAX:
            merged = merged[-_COMPRESS_LOG_MAX:]
        _compress_status["log"] = merged
        _compress_status["log_lines"] = merged.count("\n")


def _stream_subprocess_output(
    proc: subprocess.Popen[str],
    append: Any,
) -> int:
    """Read subprocess stdout line-by-line until the process exits."""
    assert proc.stdout is not None

    def _reader() -> None:
        for line in proc.stdout:
            append(line.rstrip("\r\n"))

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    return_code = proc.wait()
    reader.join(timeout=30.0)
    return return_code


def _set_compress_proc(proc: subprocess.Popen[str] | None) -> None:
    global _compress_proc
    with _compress_lock:
        _compress_proc = proc


def _compress_should_stop() -> bool:
    with _compress_lock:
        return bool(_compress_stop_requested)


def _terminate_compress_proc(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except OSError:
                pass


def _dashboard_compress_pids() -> list[int]:
    """PIDs for dashboard-started main_batch.py runs."""
    marker = str(ROOT / "work_fleet" / "dashboard_request.json")
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        text = " ".join(str(part) for part in cmdline)
        if "main_batch.py" not in text:
            continue
        if marker not in text and "work_fleet/dashboard_request.json" not in text:
            continue
        pid = proc.info.get("pid")
        if isinstance(pid, int):
            pids.append(pid)
    return pids


def _terminate_pid_tree(pid: int) -> None:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            return
    gone, alive = psutil.wait_procs([proc], timeout=5)
    if alive:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


def stop_compression() -> dict[str, Any]:
    global _compress_stop_requested
    with _compress_lock:
        if not _compress_status["running"]:
            killed = False
            for pid in _dashboard_compress_pids():
                _terminate_pid_tree(pid)
                killed = True
            if killed:
                _append_compress_log("Stop requested — terminated orphaned compression subprocess")
                _compress_status["running"] = False
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
                _compress_status["exit_code"] = -1
                return {"stopped": True, "status": dict(_compress_status)}
            return {"stopped": False, "status": dict(_compress_status)}
        _compress_stop_requested = True
        proc = _compress_proc
    killed = False
    if proc is not None and proc.poll() is None:
        _append_compress_log("Stop requested — terminating compression…")
        _terminate_compress_proc(proc)
        killed = True
    else:
        for pid in _dashboard_compress_pids():
            _append_compress_log(f"Stop requested — terminating compression (pid {pid})…")
            _terminate_pid_tree(pid)
            killed = True
        if not killed:
            _append_compress_log("Stop requested — waiting for compression to start…")
    if killed:
        with _compress_lock:
            if _compress_status["running"]:
                _compress_status["running"] = False
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
                _compress_status["exit_code"] = -1
    return {"stopped": True, "status": compress_status()}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    return path.stat().st_size


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def _resolve_under(base: Path, rel: str) -> Path | None:
    text = str(rel).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (base / text.lstrip("./")).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def _safe_media_file(raw: str | Path | None) -> Path | None:
    """Resolve a media path under the mining repo or workspace."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (ROOT / text.lstrip("./")).resolve()
    else:
        path = path.resolve()
    for base in (ROOT.resolve(), WORKSPACE.resolve()):
        try:
            path.relative_to(base)
        except ValueError:
            continue
        if path.is_file():
            return path
        return None
    return None


def _load_jobs() -> list[dict[str, Any]]:
    data = _read_json(REQUEST_JSON)
    if not isinstance(data, dict):
        return []
    jobs = data.get("jobs") or []
    return [job for job in jobs if isinstance(job, dict)]


def _load_jobs_by_input() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for job in _load_jobs():
        input_path = job.get("input_path") or ""
        name = Path(str(input_path)).name
        if name:
            mapping[name] = job
        job_id = str(job.get("id") or "").strip()
        if job_id:
            mapping[job_id] = job
            # Also map common aliases: job v1 <-> file 1.mp4
            if job_id.startswith("v") and job_id[1:].isdigit():
                mapping[f"{job_id[1:]}.mp4"] = job
    return mapping


def _original_path_for_job(job: dict[str, Any]) -> Path | None:
    return _safe_media_file(job.get("input_path"))


def _load_batch_by_job() -> dict[str, dict[str, Any]]:
    """Load final encoding results.

    Encoding writes:
      1. per-video ``work_fleet/<threshold>/<job_id>/result.json``
         (or legacy ``work_fleet/<job_id>/result.json``)
      2. summary ``work_fleet/batch_results.json`` (fleet-wide elapsed_sec)

    Prefer per-video result.json when present. When multiple thresholds exist
    for the same job_id, prefer the threshold from ``request.json``.
    """
    out: dict[str, dict[str, Any]] = {}
    preferred_thr: int | None = None
    shared = _shared_request_params()
    if "vmaf_threshold" in shared:
        try:
            preferred_thr = int(shared["vmaf_threshold"])
        except (TypeError, ValueError):
            preferred_thr = None

    data = _read_json(BATCH_RESULTS)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("job_id"):
                out[str(item["job_id"])] = item

    work_root = ROOT / "work_fleet"
    if work_root.is_dir():
        paths = list(work_root.glob("*/*/result.json")) + list(
            work_root.glob("*/result.json")
        )
        # Prefer preferred-threshold results last so they win.
        def _sort_key(path: Path) -> tuple[int, str]:
            parent = path.parent.parent.name
            try:
                thr = int(parent)
            except ValueError:
                thr = -1
            prefer = 1 if preferred_thr is not None and thr == preferred_thr else 0
            return (prefer, str(path))

        for path in sorted(paths, key=_sort_key):
            # Skip legacy false-positives: work_fleet/<thr>/result.json
            if path.parent.parent == work_root and path.parent.name.isdigit():
                continue
            item = _read_json(path)
            if isinstance(item, dict) and item.get("job_id"):
                job_id = str(item["job_id"])
                item_thr = item.get("vmaf_threshold")
                existing = out.get(job_id)
                if existing is None:
                    out[job_id] = item
                    continue
                if preferred_thr is not None:
                    try:
                        if int(item_thr) == preferred_thr:
                            out[job_id] = item
                    except (TypeError, ValueError):
                        pass
                else:
                    out[job_id] = item
    return out


_REQUEST_PARAM_KEYS = (
    "vmaf_threshold",
    "codec_mode",
    "encoder",
    "preset",
    "target_bitrate",
    "nvenc_rc",
    "nvenc_multipass",
    "nvenc_tune",
    "nvenc_feature_baseline",
    "libx265_refine",
    "libx265_refine_preset",
    "libx265_feature_baseline",
    "libx265_profile",
    "libx265_tune",
    "libx265_params",
    "libx265_crf_min",
    "libx265_crf_max",
    "crf_start",
    "crf",
    "crf_candidates",
    "crf_min",
    "crf_max",
    "crf_spread",
    "param_tune",
    "param_tune_max_trials",
    "param_tune_no_improve_stop",
    "param_tune_vmaf_headroom",
    "param_tune_max_rounds",
    "crf_mode_tune",
    "crf_mode_pack",
    "crf_mode_aq_min",
    "crf_mode_aq_max",
    "crf_mode_aq_step",
    "crf_mode_vmaf_headroom",
    "crf_mode_compensate_steps",
    "crf_mode_max_compression_rate",
    "crf_mode_lookahead_default",
    "crf_mode_lookahead_sweep",
    "time_budget_sec",
    "final_encode_reserve_sec",
    "probe_min_budget_sec",
    "use_proxy",
    "proxy_seconds_per_segment",
    "proxy_max_seconds",
    "proxy_vmaf_margin",
    "proxy_mashup_push_ceiling",
    "sample_frames",
    "sample_seconds_per_scene",
    "vmaf_n_subsample",
    "vmaf_n_threads",
    "vmaf_docker_gpus",
    "fleet_batch_size",
    "fleet_gpu_slots",
    "work_dir",
)

_ENCODE_PARAM_KEYS = (
    "encoder",
    "preset",
    "libx265_profile",
    "libx265_tune",
    "libx265_params",
    "libx265_feature_baseline",
    "libx265_crf_min",
    "libx265_crf_max",
    "libx265_refine_preset",
    "crf",
    "crf_candidates",
    "crf_min",
    "crf_max",
    "crf_spread",
    "nvenc_rc",
    "nvenc_multipass",
    "nvenc_tune",
    "nvenc_feature_baseline",
    "time_budget_sec",
    "final_encode_reserve_sec",
    "use_proxy",
    "proxy_seconds_per_segment",
    "proxy_max_seconds",
    "proxy_vmaf_margin",
    "proxy_mashup_push_ceiling",
    "vmaf_n_threads",
    "preprocess",
    "preprocess_auto",
    "preprocess_ab",
    "preprocess_sweep",
    "vbr_mode_tune",
    "param_tune",
    "crf_mode_tune",
    "scene_crf_search",
    "vbr_fallback_to_crf",
)

# Dashboard test path: one full-file encode at fixed libx265-params, no search/tune.
_DASHBOARD_RUN_DEFAULTS: dict[str, Any] = {
    "preprocess": "none",
    "preprocess_auto": False,
    "preprocess_ab": False,
    "preprocess_sweep": False,
    "vbr_mode_tune": False,
    "param_tune": False,
    "libx265_feature_baseline": False,
    "crf_mode_tune": False,
    "scene_crf_search": False,
    "vbr_fallback_to_crf": False,
}


def _apply_dashboard_run_defaults(base: dict[str, Any]) -> None:
    for key, value in _DASHBOARD_RUN_DEFAULTS.items():
        base[key] = value


def _shared_request_params() -> dict[str, Any]:
    data = _read_json(REQUEST_JSON)
    if not isinstance(data, dict):
        return {}
    return {key: data[key] for key in _REQUEST_PARAM_KEYS if key in data}


def encode_defaults() -> dict[str, Any]:
    data = _read_json(REQUEST_JSON)
    if not isinstance(data, dict):
        return {}
    out = {key: data[key] for key in _ENCODE_PARAM_KEYS if key in data}
    if "profile" in data and "libx265_profile" not in out:
        out["libx265_profile"] = data["profile"]
    if "x265_params" in data and "libx265_params" not in out:
        out["libx265_params"] = data["x265_params"]
    # UI uses crf_min/max; fleet request.json often uses libx265_crf_min/max.
    if "crf_min" not in out and "libx265_crf_min" in out:
        out["crf_min"] = out["libx265_crf_min"]
    if "crf_max" not in out and "libx265_crf_max" in out:
        out["crf_max"] = out["libx265_crf_max"]
    if data.get("target_bitrate"):
        out["target_bitrate"] = data["target_bitrate"]
    if data.get("target_compression_rate") is not None:
        out["target_compression_rate"] = data["target_compression_rate"]
    if not out.get("libx265_params"):
        out["libx265_params"] = _DEFAULT_LIBX265_PARAMS
    for key, default in (
        ("crf_min", 8),
        ("crf_max", 40),
        ("crf_candidates", 3),
        ("crf_spread", 2),
        ("vmaf_n_threads", 10),
    ):
        if key not in out:
            out[key] = default
    out["vmaf_n_threads"] = 10
    out["run_no_preprocess"] = True
    out["run_full_video"] = True
    out["libx265_feature_baseline"] = False
    return out


def _merge_encode_params(base: dict[str, Any], encode_params: dict[str, Any] | None) -> None:
    if not encode_params:
        return
    for key, value in encode_params.items():
        if key not in _ENCODE_PARAM_KEYS:
            continue
        if value is None or value == "":
            base.pop(key, None)
            continue
        if key in {
            "libx265_feature_baseline",
            "nvenc_feature_baseline",
            "use_proxy",
            "preprocess_auto",
            "preprocess_ab",
            "preprocess_sweep",
            "vbr_mode_tune",
            "param_tune",
            "crf_mode_tune",
            "scene_crf_search",
            "vbr_fallback_to_crf",
        }:
            base[key] = bool(value)
        elif key in {
            "libx265_crf_min",
            "libx265_crf_max",
            "crf",
            "crf_candidates",
            "crf_min",
            "crf_max",
            "crf_spread",
            "vmaf_n_threads",
            "time_budget_sec",
            "final_encode_reserve_sec",
            "proxy_seconds_per_segment",
            "proxy_max_seconds",
            "proxy_vmaf_margin",
            "proxy_mashup_push_ceiling",
        }:
            if isinstance(value, bool):
                continue
            try:
                if key in {
                    "time_budget_sec",
                    "final_encode_reserve_sec",
                    "proxy_seconds_per_segment",
                    "proxy_max_seconds",
                    "proxy_vmaf_margin",
                    "proxy_mashup_push_ceiling",
                }:
                    parsed: int | float = float(value)
                else:
                    parsed = int(value)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, float) and parsed != parsed:
                continue
            if key in {"crf_candidates", "crf_spread", "vmaf_n_threads"} and parsed < 1:
                continue
            base[key] = parsed
        else:
            base[key] = str(value).strip() if isinstance(value, str) else value


def _request_params_for_job(job: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    params = dict(shared)
    # Per-video job fields from request.json jobs[]
    for key in (
        "id",
        "input_path",
        "output_path",
        "input_url",
        "upload_url",
        "crf",
        "libx265_params",
    ):
        if key in job and job[key] not in (None, ""):
            params[key] = job[key]
    if "x265_params" in job and "libx265_params" not in params and job["x265_params"]:
        params["libx265_params"] = job["x265_params"]
    return params


def _encode_params_from_result(batch: dict[str, Any] | None) -> dict[str, Any]:
    if not batch:
        return {}
    best = batch.get("best") or {}
    return {
        "strategy": batch.get("strategy"),
        "recipe": best.get("recipe"),
        "mode": best.get("mode"),
        "encoder": best.get("encoder"),
        "crf": best.get("crf"),
        "bitrate": best.get("bitrate"),
        "stage": best.get("stage"),
        "use_gpu": batch.get("use_gpu"),
        "uploaded": batch.get("uploaded"),
        "error": batch.get("error") or None,
        "nvenc_overrides": best.get("nvenc_overrides") or {},
        "stage_timings": batch.get("stage_timings") or {},
    }


def _compressed_path_for_job(job: dict[str, Any], batch: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    output_path = job.get("output_path")
    if output_path:
        p = _safe_media_file(str(output_path))
        if p:
            candidates.append(p)
        # Also try unresolved path under ROOT (may not exist yet).
        unresolved = _resolve_under(ROOT, str(output_path))
        if unresolved:
            candidates.append(unresolved)

    if batch:
        best = batch.get("best") or {}
        for key in ("path", "output_path"):
            raw = best.get(key) if key == "path" else batch.get(key)
            if not raw:
                continue
            p = _safe_media_file(str(raw))
            if p:
                candidates.append(p)

    job_id = str(job.get("id") or "")
    thr = None
    if batch and batch.get("vmaf_threshold") is not None:
        try:
            thr = str(int(batch["vmaf_threshold"]))
        except (TypeError, ValueError):
            thr = None
    elif job.get("vmaf_threshold") is not None:
        try:
            thr = str(int(job["vmaf_threshold"]))
        except (TypeError, ValueError):
            thr = None
    else:
        shared = _shared_request_params()
        if "vmaf_threshold" in shared:
            try:
                thr = str(int(shared["vmaf_threshold"]))
            except (TypeError, ValueError):
                thr = None

    if job_id:
        for name in ("final_x265.mp4", "final_nvenc_fallback.mp4"):
            if thr:
                candidates.append(ROOT / "work_fleet" / thr / job_id / name)
            candidates.append(ROOT / "work_fleet" / job_id / name)
        # Fleet CLI writes output/fleet/<thr>/<job_id>.mp4
        if thr:
            candidates.append(ROOT / "output" / "fleet" / thr / f"{job_id}.mp4")
        candidates.append(ROOT / "output" / "fleet" / f"{job_id}.mp4")

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def _catalog_entry(
    *,
    name: str,
    original: Path,
    job: dict[str, Any] | None,
    batch: dict[str, Any] | None,
    shared_request: dict[str, Any],
) -> dict[str, Any]:
    job = job or {}
    job_id = str(job.get("id") or "") or None
    compressed = _compressed_path_for_job(job, batch) if job else None
    best = (batch or {}).get("best") or {}
    result_mtime = _result_mtime_for_job(job_id, batch, compressed)
    return {
        "name": name,
        "job_id": job_id,
        "original_size": _file_size(original),
        "original_size_human": _fmt_bytes(_file_size(original)),
        "compressed_size": _file_size(compressed),
        "compressed_size_human": _fmt_bytes(_file_size(compressed)),
        "has_compressed": compressed is not None,
        "final_score": best.get("s_f"),
        "vmaf": best.get("vmaf"),
        "compression_rate": best.get("compression_rate"),
        "compression_ratio": best.get("compression_ratio"),
        "avg_bitrate_mbps": best.get("measured_bitrate_mbps"),
        "elapsed_sec": (batch or {}).get("elapsed_sec"),
        "strategy": (batch or {}).get("strategy"),
        "chosen_crf": (batch or {}).get("chosen_crf") or best.get("crf"),
        "error": (batch or {}).get("error") or None,
        "result_mtime": result_mtime,
        "result_mtime_iso": (
            datetime.fromtimestamp(result_mtime, tz=timezone.utc).isoformat()
            if result_mtime is not None
            else None
        ),
        "request_params": _request_params_for_job(job, shared_request) if job else shared_request,
        "encode_params": _encode_params_from_result(batch),
    }


def _result_mtime_for_job(
    job_id: str | None,
    batch: dict[str, Any] | None,
    compressed: Path | None,
) -> float | None:
    """Newest mtime among compressed output and work_fleet result.json."""
    mtimes: list[float] = []
    if compressed is not None and compressed.is_file():
        try:
            mtimes.append(compressed.stat().st_mtime)
        except OSError:
            pass
    if job_id:
        work = ROOT / "work_fleet"
        candidates: list[Path] = [work / job_id / "result.json"]
        if batch and batch.get("vmaf_threshold") is not None:
            try:
                thr = str(int(batch["vmaf_threshold"]))
                candidates.insert(0, work / thr / job_id / "result.json")
            except (TypeError, ValueError):
                pass
        if work.is_dir():
            candidates.extend(work.glob(f"*/{job_id}/result.json"))
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve() if path.exists() else path
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.is_file():
                try:
                    mtimes.append(path.stat().st_mtime)
                except OSError:
                    pass
    return max(mtimes) if mtimes else None


def build_catalog() -> list[dict[str, Any]]:
    jobs_by_input = _load_jobs_by_input()
    batch_by_job = _load_batch_by_job()
    shared_request = _shared_request_params()
    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    # Prefer request.json jobs (fleet inputs under /workspace/video, etc.).
    for job in _load_jobs():
        original = _original_path_for_job(job)
        if original is None:
            continue
        name = original.name
        if name in seen_names:
            continue
        seen_names.add(name)
        job_id = str(job.get("id") or "")
        batch = batch_by_job.get(job_id) if job_id else None
        entries.append(
            _catalog_entry(
                name=name,
                original=original,
                job=job,
                batch=batch,
                shared_request=shared_request,
            )
        )

    # Also include WandB challenge downloads if present.
    if COMPRESSION_DIR.is_dir():
        for original in sorted(COMPRESSION_DIR.glob("*.mp4")):
            if original.name in seen_names:
                continue
            seen_names.add(original.name)
            job = jobs_by_input.get(original.name, {})
            job_id = str(job.get("id") or "")
            batch = batch_by_job.get(job_id) if job_id else None
            entries.append(
                _catalog_entry(
                    name=original.name,
                    original=original,
                    job=job or None,
                    batch=batch,
                    shared_request=shared_request,
                )
            )

    # Newest results first; unfinished / no-mtime entries last.
    def _sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
        mtime = item.get("result_mtime")
        if mtime is None:
            return (1, 0.0, item["name"].lower())
        return (0, -float(mtime), item["name"].lower())

    entries.sort(key=_sort_key)
    return entries


def _gpu_stats() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            mem_used = float(parts[3])
            mem_total = float(parts[4])
            mem_pct = (mem_used / mem_total * 100) if mem_total else 0
        except ValueError:
            mem_used = mem_total = mem_pct = None
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "utilization_pct": _safe_float(parts[2]),
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_pct": mem_pct,
                "temperature_c": _safe_float(parts[5]),
                "power_w": _safe_float(parts[6]) if len(parts) > 6 else None,
            }
        )
    return gpus


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def system_stats() -> dict[str, Any]:
    cpu_pct = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    load1 = load5 = load15 = None
    if hasattr(os, "getloadavg"):
        load1, load5, load15 = os.getloadavg()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": cpu_pct,
        "cpu_count": psutil.cpu_count(logical=True),
        "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        "memory": {
            "total": mem.total,
            "used": mem.used,
            "available": mem.available,
            "percent": mem.percent,
            "total_human": _fmt_bytes(mem.total),
            "used_human": _fmt_bytes(mem.used),
            "available_human": _fmt_bytes(mem.available),
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
            "total_human": _fmt_bytes(disk.total),
            "used_human": _fmt_bytes(disk.used),
            "free_human": _fmt_bytes(disk.free),
        },
        "gpus": _gpu_stats(),
    }


def _find_original(name: str) -> Path | None:
    safe = Path(unquote(name)).name
    path = COMPRESSION_DIR / safe
    if path.is_file():
        return path
    jobs = _load_jobs_by_input()
    job = jobs.get(safe)
    if job:
        found = _original_path_for_job(job)
        if found is not None:
            return found
    # Direct lookup under workspace/video (fleet inputs).
    direct = VIDEO_DIR / safe
    if direct.is_file():
        return direct
    return None


def _find_compressed(name: str) -> Path | None:
    safe = Path(unquote(name)).name
    jobs = _load_jobs_by_input()
    job = jobs.get(safe)
    if not job:
        # Try job id form when media name is 1.mp4 but job is v1
        stem = Path(safe).stem
        if stem.isdigit():
            job = jobs.get(f"v{stem}")
        elif safe.startswith("v"):
            job = jobs.get(safe)
    if not job:
        return None
    batch = _load_batch_by_job().get(str(job.get("id") or ""))
    return _compressed_path_for_job(job, batch)


def _wandb_graphql(query: str, timeout: float = 30.0) -> dict[str, Any]:
    payload = json.dumps({"query": query}).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "vidaio-dashboard/1.0"}
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("errors"):
        raise RuntimeError(f"WandB GraphQL error: {data['errors']}")
    return data


def _wandb_list_validator_runs(limit: int | None = None) -> list[dict[str, Any]]:
    scan_limit = limit or WANDB_RUN_SCAN_LIMIT
    query = (
        f'query {{ project(name:"{WANDB_PROJECT}", entityName:"{WANDB_ENTITY}") '
        f'{{ runs(first:{scan_limit}, order:"-heartbeatAt") '
        f'{{ edges {{ node {{ name displayName heartbeatAt createdAt state }} }} }} }} }}'
    )
    data = _wandb_graphql(query)
    runs: list[dict[str, Any]] = []
    edges = (((data.get("data") or {}).get("project") or {}).get("runs") or {}).get("edges") or []
    for edge in edges:
        node = edge.get("node") or {}
        name = str(node.get("name") or "").strip()
        if name.startswith("validator-"):
            runs.append(node)
    return runs


def _wandb_run_output_log_url(run_name: str) -> str:
    query = (
        f'query {{ project(name:"{WANDB_PROJECT}", entityName:"{WANDB_ENTITY}") '
        f'{{ run(name:"{run_name}") {{ files {{ edges {{ node {{ name directUrl }} }} }} }} }} }}'
    )
    data = _wandb_graphql(query)
    run = (((data.get("data") or {}).get("project") or {}).get("run") or {})
    edges = ((run.get("files") or {}).get("edges") or [])
    for edge in edges:
        node = edge["node"]
        if str(node.get("name", "")).endswith("output.log"):
            return str(node["directUrl"])
    raise RuntimeError(f"WandB output.log not found for run {run_name!r}")


def _download_wandb_log_text(log_url: str, timeout: float = 120.0) -> str:
    req = urllib.request.Request(log_url, headers={"User-Agent": "vidaio-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _wandb_discover_output_log() -> tuple[str, str, str]:
    """Find the current validator output.log in the WandB workspace.

    Prefers the newest heartbeat among validator runs that contain compression
    challenge URLs or score_compressions lines (not the historically largest
    log, which often has expired S3 links).

    Set WANDB_RUN to force a specific run.
    """
    if WANDB_RUN:
        log_url = _wandb_run_output_log_url(WANDB_RUN)
        log_text = _download_wandb_log_text(log_url)
        return WANDB_RUN, log_url, log_text

    runs = _wandb_list_validator_runs()
    if not runs:
        raise RuntimeError(
            f"No validator runs found in WandB workspace {WANDB_ENTITY}/{WANDB_PROJECT}"
        )

    # API returns runs ordered by -heartbeatAt (newest first).
    best_fallback: tuple[str, str, str, str, int, int] | None = None
    # (run, url, text, heartbeat, url_count, score_count)

    for run in runs:
        run_name = str(run.get("name") or "")
        heartbeat = str(run.get("heartbeatAt") or "")
        try:
            log_url = _wandb_run_output_log_url(run_name)
            log_text = _download_wandb_log_text(log_url)
            url_count = len(_extract_compression_urls(log_text))
            score_count = log_text.count("score_compressions")
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
            _append_fetch_log(f"Skipped {run_name}: {exc}")
            continue

        _append_fetch_log(
            f"Checked {run_name}: {url_count} compression URL(s), "
            f"{score_count} score line(s), heartbeat {heartbeat or '—'}"
        )

        if url_count <= 0 and score_count <= 0:
            continue

        # Newest run with live challenge URLs wins immediately.
        if url_count > 0:
            return run_name, log_url, log_text

        # Otherwise keep newest score-only run as fallback (analyze still works).
        if best_fallback is None:
            best_fallback = (
                run_name,
                log_url,
                log_text,
                heartbeat,
                url_count,
                score_count,
            )

    if best_fallback is not None:
        return best_fallback[0], best_fallback[1], best_fallback[2]

    raise RuntimeError(
        "No validator output.log with compression scores/URLs found in WandB workspace"
    )


def _extract_compression_urls(log_text: str) -> list[str]:
    """Extract challenge video URLs; keep the latest signed URL per object path."""
    by_path: dict[str, str] = {}
    order: list[str] = []
    for line in log_text.splitlines():
        low = line.lower()
        if "compression" not in low and "score_compressions" not in low:
            continue
        for url in re.findall(r"https://s3\.us-east-005\.backblazeb2\.com/[^\s'\"\],]+", line):
            url = url.rstrip(".,)")
            if ".mp4" not in url or "/vidaiosubnet/" not in url:
                continue
            path = urlparse(url).path
            if path not in by_path:
                order.append(path)
            # Prefer later (fresher) signed URLs for the same object.
            by_path[path] = url
    return [by_path[path] for path in order]


def _download_file(url: str, dest: Path, timeout: float = 600.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "vidaio-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def fetch_compression_challenges_sync() -> dict[str, Any]:
    global _fetch_status
    with _fetch_lock:
        if _fetch_status["running"]:
            return dict(_fetch_status)
        _fetch_status = {
            "running": True,
            "phase": "fetching_log",
            "total": 0,
            "done": 0,
            "skipped": 0,
            "failed": 0,
            "current": None,
            "wandb_run": None,
            "message": "Fetching WandB log…",
            "errors": [],
            "log": "",
        }

    try:
        _append_fetch_log(
            f"Scanning WandB workspace {WANDB_ENTITY}/{WANDB_PROJECT} for validator runs..."
        )
        run_name, log_url, log_text = _wandb_discover_output_log()
        with _fetch_lock:
            _fetch_status["wandb_run"] = run_name
        _append_fetch_log(f"Using validator run: {run_name}")
        _append_fetch_log(f"Found output.log: {log_url.split('?')[0]}")
        _append_fetch_log(f"Downloaded WandB log ({len(log_text):,} bytes)")

        urls = _extract_compression_urls(log_text)
        if not urls:
            raise RuntimeError("No compression challenge URLs found in WandB output.log")
        COMPRESSION_DIR.mkdir(parents=True, exist_ok=True)
        url_list_path = ROOT / "s3_videos" / "compression_urls.txt"
        url_list_path.write_text("\n".join(urls) + "\n")
        _append_fetch_log(f"Extracted {len(urls)} compression URLs -> {url_list_path}")

        with _fetch_lock:
            _fetch_status["phase"] = "downloading"
            _fetch_status["total"] = len(urls)
            _fetch_status["message"] = f"Found {len(urls)} compression challenge videos"

        for index, url in enumerate(urls, start=1):
            name = Path(urlparse(url).path).name
            dest = COMPRESSION_DIR / name
            with _fetch_lock:
                _fetch_status["current"] = name
                _fetch_status["message"] = f"[{index}/{len(urls)}] {name}"

            if dest.is_file() and dest.stat().st_size > 1000:
                with _fetch_lock:
                    _fetch_status["skipped"] += 1
                _append_fetch_log(f"[{index}/{len(urls)}] already local: {name}")
                continue

            try:
                _append_fetch_log(f"[{index}/{len(urls)}] downloading: {name}")
                _download_file(url, dest)
                with _fetch_lock:
                    _fetch_status["done"] += 1
                _append_fetch_log(f"[{index}/{len(urls)}] downloaded: {name}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                with _fetch_lock:
                    _fetch_status["failed"] += 1
                    _fetch_status["errors"].append(f"{name}: {exc}")
                _append_fetch_log(f"[{index}/{len(urls)}] failed: {name}: {exc}")

        with _fetch_lock:
            _fetch_status["running"] = False
            _fetch_status["phase"] = "done"
            _fetch_status["current"] = None
            downloaded = _fetch_status["done"]
            skipped = _fetch_status["skipped"]
            failed = _fetch_status["failed"]
            if downloaded == 0 and skipped:
                _fetch_status["message"] = (
                    f"Finished: all {skipped} videos already exist locally, {failed} failed"
                )
            else:
                _fetch_status["message"] = (
                    f"Finished: {downloaded} downloaded, {skipped} already local, {failed} failed"
                )
        return dict(_fetch_status)
    except Exception as exc:  # noqa: BLE001
        _append_fetch_log(f"ERROR: {exc}")
        with _fetch_lock:
            _fetch_status["running"] = False
            _fetch_status["phase"] = "error"
            _fetch_status["message"] = str(exc)
            _fetch_status["errors"].append(str(exc))
        return dict(_fetch_status)


def _start_fetch_thread() -> dict[str, Any]:
    with _fetch_lock:
        if _fetch_status["running"]:
            return {"started": False, "status": dict(_fetch_status)}
    thread = threading.Thread(target=fetch_compression_challenges_sync, daemon=True)
    thread.start()
    return {"started": True, "status": dict(_fetch_status)}


def fetch_status() -> dict[str, Any]:
    with _fetch_lock:
        return dict(_fetch_status)


def _normalize_dashboard_codec_mode(mode: str) -> str:
    key = (mode or "crf").lower().strip()
    if key in {"crf", "rc", "cq"}:
        return "RC"
    if key in {"vbr", "abr", "bitrate"}:
        return "ABR"
    raise ValueError(f"codec_mode must be crf or vbr, got {mode!r}")


def _build_jobs_from_videos(video_names: list[str]) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for index, name in enumerate(video_names, start=1):
        safe_name = Path(name).name
        src = COMPRESSION_DIR / safe_name
        if not src.is_file():
            raise FileNotFoundError(f"video not found: {safe_name}")
        job_id = f"v{index}"
        jobs.append(
            {
                "id": job_id,
                "input_path": f"./s3_videos/compression/{safe_name}",
                "output_path": f"./output/fleet/{job_id}.mp4",
            }
        )
    return jobs


def _write_dashboard_request(
    *,
    codec_mode: str,
    vmaf_threshold: int,
    target_bitrate: str | None,
    target_compression_rate: float | None,
    video_names: list[str],
    encode_params: dict[str, Any] | None = None,
) -> Path:
    base = _read_json(REQUEST_JSON)
    if not isinstance(base, dict):
        base = {}
    jobs = _build_jobs_from_videos(video_names)
    base["skip_transfer"] = True
    base["jobs"] = jobs
    base["vmaf_threshold"] = vmaf_threshold
    base["codec_mode"] = _normalize_dashboard_codec_mode(codec_mode)
    if base["codec_mode"] == "ABR":
        if not str(target_bitrate or "").strip() and target_compression_rate is None:
            raise ValueError(
                "target_bitrate or target_compression_rate is required for vbr mode"
            )
        if str(target_bitrate or "").strip():
            base["target_bitrate"] = str(target_bitrate).strip()
        else:
            base.pop("target_bitrate", None)
        if target_compression_rate is not None:
            base["target_compression_rate"] = float(target_compression_rate)
        else:
            base.pop("target_compression_rate", None)
    else:
        base.pop("target_bitrate", None)
        base.pop("target_compression_rate", None)
    _apply_dashboard_run_defaults(base)
    _merge_encode_params(base, encode_params)

    req_path = ROOT / "work_fleet" / "dashboard_request.json"
    req_path.parent.mkdir(parents=True, exist_ok=True)
    req_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return req_path


def run_compression_sync(
    *,
    codec_mode: str,
    vmaf_threshold: int,
    target_bitrate: str | None,
    target_compression_rate: float | None,
    video_names: list[str],
    encode_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global _compress_status, _compress_stop_requested
    with _compress_lock:
        if _compress_status["running"]:
            return dict(_compress_status)
        _compress_stop_requested = False
        _compress_status = {
            "running": True,
            "phase": "preparing",
            "message": "Preparing compression request…",
            "videos": list(video_names),
            "total": len(video_names),
            "exit_code": None,
            "errors": [],
            "params": {
                "codec_mode": codec_mode,
                "vmaf_threshold": vmaf_threshold,
                "target_bitrate": target_bitrate,
                "target_compression_rate": target_compression_rate,
                "encode_params": encode_params or {},
            },
            "log": "",
        }

    try:
        req_path = _write_dashboard_request(
            codec_mode=codec_mode,
            vmaf_threshold=vmaf_threshold,
            target_bitrate=target_bitrate,
            target_compression_rate=target_compression_rate,
            video_names=video_names,
            encode_params=encode_params,
        )
        if _compress_should_stop():
            _append_compress_log("Stopped by user")
            with _compress_lock:
                _compress_status["running"] = False
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
                _compress_status["exit_code"] = -1
            return dict(_compress_status)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "main_batch.py"),
            "-r",
            str(req_path),
            "--local",
            "--force",
            "--work-root",
            "work_fleet",
            "--output-dir",
            "output/fleet",
            "--results",
            "work_fleet/batch_results.json",
            f"--limit={len(video_names)}",
        ]
        with _compress_lock:
            _compress_status["phase"] = "running"
            _compress_status["message"] = (
                f"Compressing {len(video_names)} video(s) "
                f"({codec_mode}, VMAF {vmaf_threshold})"
            )
            _compress_status["log_lines"] = 0
        _append_compress_log(f"$ {' '.join(cmd)}")

        if _compress_should_stop():
            _append_compress_log("Stopped by user")
            with _compress_lock:
                _compress_status["running"] = False
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
                _compress_status["exit_code"] = -1
            return dict(_compress_status)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        _set_compress_proc(proc)
        try:
            return_code = _stream_subprocess_output(proc, _append_compress_log)
        finally:
            _set_compress_proc(None)

        with _compress_lock:
            stopped = bool(_compress_stop_requested)
            _compress_status["running"] = False
            _compress_status["exit_code"] = return_code
            if stopped or return_code in (-signal.SIGTERM, -signal.SIGKILL):
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
            elif return_code == 0:
                _compress_status["phase"] = "done"
                _compress_status["message"] = f"Compression finished for {len(video_names)} video(s)"
            else:
                _compress_status["phase"] = "error"
                log_text = str(_compress_status.get("log") or "").strip()
                _compress_status["message"] = (
                    log_text.splitlines()[-1] if log_text else "compression failed"
                )
                if log_text:
                    _compress_status["errors"].append(log_text[-2000:])
        return dict(_compress_status)
    except Exception as exc:  # noqa: BLE001
        _append_compress_log(f"ERROR: {exc}")
        with _compress_lock:
            _compress_status["running"] = False
            if _compress_stop_requested:
                _compress_status["phase"] = "stopped"
                _compress_status["message"] = "Compression stopped by user"
            else:
                _compress_status["phase"] = "error"
                _compress_status["message"] = str(exc)
                _compress_status["errors"].append(str(exc))
        return dict(_compress_status)
    finally:
        _set_compress_proc(None)


def _start_compress_thread(
    *,
    codec_mode: str,
    vmaf_threshold: int,
    target_bitrate: str | None,
    target_compression_rate: float | None,
    video_names: list[str],
    encode_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _compress_lock:
        if _compress_status["running"]:
            return {"started": False, "status": dict(_compress_status)}
    thread = threading.Thread(
        target=run_compression_sync,
        kwargs={
            "codec_mode": codec_mode,
            "vmaf_threshold": vmaf_threshold,
            "target_bitrate": target_bitrate,
            "target_compression_rate": target_compression_rate,
            "video_names": video_names,
            "encode_params": encode_params,
        },
        daemon=True,
    )
    thread.start()
    return {"started": True, "status": dict(_compress_status)}


def compress_status() -> dict[str, Any]:
    with _compress_lock:
        return dict(_compress_status)


def _allowed_log_roots() -> list[Path]:
    return [ROOT.resolve(), ROOT.parent.resolve()]


def _safe_log_path(raw: str) -> Path | None:
    text = (raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    for base in _allowed_log_roots():
        try:
            path.relative_to(base)
            break
        except ValueError:
            continue
    else:
        return None
    if not path.is_file():
        return None
    return path


def discover_log_paths() -> list[dict[str, Any]]:
    candidates: list[Path] = [
        ROOT.parent / "files_output (3).log",
        ROOT / "wandb_output.log",
        *sorted(ROOT.parent.glob("files_output*.log")),
        *sorted(ROOT.glob("*.log")),
        *sorted(ROOT.glob("wandb*/**/output.log")),
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_file():
            continue
        if _safe_log_path(key) is None:
            continue
        seen.add(key)
        out.append(
            {
                "path": key,
                "name": resolved.name,
                "size": resolved.stat().st_size,
                "size_human": _fmt_bytes(resolved.stat().st_size),
                "mtime": datetime.fromtimestamp(
                    resolved.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    out.sort(key=lambda item: item["mtime"], reverse=True)
    return out


def analyze_validator_log(
    *,
    path: Path | None = None,
    text: str | None = None,
    scores: list[Any] | None = None,
    codec: str | None = None,
    mode: str | None = None,
    vmaf_threshold: float | None = None,
    uids: list[int] | None = None,
    failures_only: bool = False,
) -> dict[str, Any]:
    if scores is not None:
        source = "cache"
        source_path = str(_ANALYZE_CACHE_PATH) if _ANALYZE_CACHE_PATH.is_file() else None
        all_scores = scores
    elif text is not None:
        source = "paste"
        source_path = None
        _challenges, all_scores = parse_log_lines(text.splitlines())
    elif path is not None:
        source = "file"
        source_path = str(path)
        _challenges, all_scores = parse_log(str(path))
    else:
        raise ValueError("path, text, or scores is required")

    rows = filter_scores(
        all_scores,
        codec=codec or None,
        mode=mode or None,
        vmaf_threshold=vmaf_threshold,
        include_failures=failures_only,
        uids=uids or None,
    )

    # Group: challenge -> uid -> videos (uids ranked by mean final desc)
    by_chal: dict[tuple[str, str, float], dict[int, list[Any]]] = {}
    chal_order: list[tuple[str, str, float]] = []
    for row in rows:
        key = (row.codec, row.mode, float(row.vmaf_threshold))
        if key not in by_chal:
            by_chal[key] = {}
            chal_order.append(key)
        by_chal[key].setdefault(row.uid, []).append(row)

    challenges: list[dict[str, Any]] = []
    for codec_name, mode_name, thr in chal_order:
        uid_map = by_chal[(codec_name, mode_name, thr)]
        ranked_uids = sorted(
            uid_map.keys(),
            key=lambda u: (-mean(x.final for x in uid_map[u]), u),
        )
        uid_blocks: list[dict[str, Any]] = []
        for uid in ranked_uids:
            uid_rows = sorted(uid_map[uid], key=lambda r: (-r.final, r.video_idx or 0))
            uid_blocks.append(
                {
                    "uid": uid,
                    "videos": len(uid_rows),
                    "mean_final": mean(r.final for r in uid_rows),
                    "max_final": max(r.final for r in uid_rows),
                    "mean_vmaf_neg": mean(r.vmaf_neg for r in uid_rows),
                    "mean_rate": mean(r.compression_rate for r in uid_rows),
                    "successes": sum(1 for r in uid_rows if r.success),
                    "rows": [asdict(r) for r in uid_rows],
                }
            )
        challenges.append(
            {
                "codec": codec_name,
                "mode": mode_name,
                "vmaf_threshold": thr,
                "uids": uid_blocks,
                "top_mean_final": uid_blocks[0]["mean_final"] if uid_blocks else 0.0,
            }
        )

    # Challenges with higher top mean_final first (then thr / codec for stability).
    challenges.sort(
        key=lambda c: (
            -float(c.get("top_mean_final") or 0.0),
            -float(c.get("vmaf_threshold") or 0.0),
            str(c.get("codec") or ""),
            str(c.get("mode") or ""),
        )
    )

    return {
        "source": source,
        "path": source_path,
        "total_scores": len(all_scores),
        "filtered": len(rows),
        "codec_filter": codec or None,
        "mode_filter": mode or None,
        "vmaf_threshold_filter": vmaf_threshold,
        "uid_filter": uids or None,
        "failures_only": failures_only,
        "challenges": challenges,
    }


def _parse_analyze_filters(body: dict[str, Any]) -> dict[str, Any]:
    codec = str(body.get("codec") or "").strip().lower() or None
    if codec == "all":
        codec = None
    mode = str(body.get("mode") or body.get("codec_mode") or "").strip().lower() or None
    if mode in {"", "all"}:
        mode = None
    thr_raw = body.get("vmaf_threshold")
    vmaf_threshold: float | None = None
    if thr_raw is not None and str(thr_raw).strip() not in {"", "all"}:
        vmaf_threshold = float(thr_raw)
    failures_only = bool(body.get("failures_only"))
    uid_raw = body.get("uids") or body.get("uid")
    if isinstance(uid_raw, list):
        uid_parts = [str(x) for x in uid_raw]
    elif uid_raw is None or uid_raw == "":
        uid_parts = []
    else:
        uid_parts = [str(uid_raw)]
    uids = parse_uid_args(uid_parts) if uid_parts else []
    return {
        "codec": codec,
        "mode": mode,
        "vmaf_threshold": vmaf_threshold,
        "uids": uids or None,
        "failures_only": failures_only,
    }


def analyze_status() -> dict[str, Any]:
    with _analyze_lock:
        status = dict(_analyze_status)
        status["has_cache"] = bool(_analyze_score_cache) or _ANALYZE_CACHE_PATH.is_file()
        return status


def clear_analyze_status() -> dict[str, Any]:
    global _analyze_status
    with _analyze_lock:
        if _analyze_status.get("running"):
            return {"cleared": False, "status": dict(_analyze_status)}
        _analyze_status = {
            "running": False,
            "phase": "idle",
            "message": "",
            "wandb_run": None,
            "errors": [],
            "params": {},
            "result": None,
            "has_cache": bool(_analyze_score_cache) or _ANALYZE_CACHE_PATH.is_file(),
        }
        return {"cleared": True, "status": dict(_analyze_status)}


def _ensure_score_cache() -> tuple[list[Any], str | None]:
    """Return cached ScoreRows, loading from wandb_output.log if needed."""
    global _analyze_score_cache
    with _analyze_lock:
        if _analyze_score_cache:
            return list(_analyze_score_cache), _analyze_status.get("wandb_run")
        wandb_run = _analyze_status.get("wandb_run")
    if _ANALYZE_CACHE_PATH.is_file():
        _challenges, scores = parse_log(str(_ANALYZE_CACHE_PATH))
        with _analyze_lock:
            _analyze_score_cache = scores
            _analyze_status["has_cache"] = True
        return list(scores), wandb_run
    raise RuntimeError("No cached log yet — click Analyze from WandB first")


def search_analyze_cache(
    *,
    codec: str | None,
    mode: str | None,
    vmaf_threshold: float | None,
    uids: list[int] | None,
    failures_only: bool,
) -> dict[str, Any]:
    """Re-filter the cached WandB log without downloading again."""
    scores, wandb_run = _ensure_score_cache()
    result = analyze_validator_log(
        scores=scores,
        codec=codec,
        mode=mode,
        vmaf_threshold=vmaf_threshold,
        uids=uids,
        failures_only=failures_only,
    )
    if wandb_run:
        result["wandb_run"] = wandb_run
    result["cached_path"] = str(_ANALYZE_CACHE_PATH)
    with _analyze_lock:
        _analyze_status["phase"] = "done"
        _analyze_status["result"] = result
        _analyze_status["params"] = {
            "codec": codec,
            "mode": mode,
            "vmaf_threshold": vmaf_threshold,
            "uids": uids or [],
            "failures_only": failures_only,
        }
        _analyze_status["message"] = (
            f"{result['filtered']} / {result['total_scores']} scores"
            + (f" from {wandb_run}" if wandb_run else " (cached log)")
        )
        _analyze_status["has_cache"] = True
        return dict(_analyze_status)


def run_analyze_wandb_sync(
    *,
    codec: str | None,
    mode: str | None,
    vmaf_threshold: float | None,
    uids: list[int] | None,
    failures_only: bool,
) -> dict[str, Any]:
    global _analyze_status, _analyze_score_cache
    with _analyze_lock:
        if _analyze_status["running"]:
            return dict(_analyze_status)
        _analyze_status = {
            "running": True,
            "phase": "fetching_log",
            "message": f"Fetching WandB log from {WANDB_ENTITY}/{WANDB_PROJECT}…",
            "wandb_run": None,
            "errors": [],
            "params": {
                "codec": codec,
                "mode": mode,
                "vmaf_threshold": vmaf_threshold,
                "uids": uids or [],
                "failures_only": failures_only,
            },
            "result": None,
            "has_cache": False,
        }

    try:
        run_name, log_url, log_text = _wandb_discover_output_log()
        _ANALYZE_CACHE_PATH.write_text(log_text, encoding="utf-8")
        with _analyze_lock:
            _analyze_status["wandb_run"] = run_name
            _analyze_status["phase"] = "analyzing"
            _analyze_status["message"] = (
                f"Analyzing {run_name} ({len(log_text):,} bytes)…"
            )

        _challenges, scores = parse_log_lines(log_text.splitlines())
        with _analyze_lock:
            _analyze_score_cache = scores
            _analyze_status["has_cache"] = True

        result = analyze_validator_log(
            scores=scores,
            codec=codec,
            mode=mode,
            vmaf_threshold=vmaf_threshold,
            uids=uids,
            failures_only=failures_only,
        )
        result["wandb_run"] = run_name
        result["log_url"] = log_url.split("?")[0]
        result["cached_path"] = str(_ANALYZE_CACHE_PATH)

        with _analyze_lock:
            _analyze_status["running"] = False
            _analyze_status["phase"] = "done"
            _analyze_status["result"] = result
            _analyze_status["message"] = (
                f"{result['filtered']} / {result['total_scores']} scores "
                f"from {run_name}"
            )
            _analyze_status["has_cache"] = True
        return dict(_analyze_status)
    except Exception as exc:  # noqa: BLE001
        with _analyze_lock:
            _analyze_status["running"] = False
            _analyze_status["phase"] = "error"
            _analyze_status["message"] = str(exc)
            _analyze_status["errors"].append(str(exc))
            _analyze_status["result"] = None
        return dict(_analyze_status)


def _start_analyze_wandb_thread(
    *,
    codec: str | None,
    mode: str | None,
    vmaf_threshold: float | None,
    uids: list[int] | None,
    failures_only: bool,
) -> dict[str, Any]:
    with _analyze_lock:
        if _analyze_status["running"]:
            return {"started": False, "status": dict(_analyze_status)}
    thread = threading.Thread(
        target=run_analyze_wandb_sync,
        kwargs={
            "codec": codec,
            "mode": mode,
            "vmaf_threshold": vmaf_threshold,
            "uids": uids,
            "failures_only": failures_only,
        },
        daemon=True,
    )
    thread.start()
    with _analyze_lock:
        return {"started": True, "status": dict(_analyze_status)}


class Handler(BaseHTTPRequestHandler):
    server_version = "VidaioDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: int = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(
        self,
        path: Path,
        *,
        download_name: str | None = None,
        as_attachment: bool = False,
    ) -> None:
        if not path.is_file():
            self._send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, file_size - 1)
                if start <= end:
                    status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        headers: dict[str, str] = {
            "Accept-Ranges": "bytes",
        }
        if as_attachment:
            headers["Content-Disposition"] = f'attachment; filename="{download_name or path.name}"'
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()

        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_HEAD(self) -> None:
        self.do_GET(head_only=True)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/fetch-challenges":
            result = _start_fetch_thread()
            self._send_json(result)
            return
        if parsed.path == "/api/compress":
            body = self._read_json_body()
            codec_mode = str(body.get("codec_mode") or "crf")
            vmaf_threshold = int(body.get("vmaf_threshold") or 85)
            target_bitrate = body.get("target_bitrate")
            target_compression_rate_raw = body.get("target_compression_rate")
            videos = body.get("videos") or []
            if vmaf_threshold not in (85, 89, 93):
                self._send_json(
                    {"error": "vmaf_threshold must be 85, 89, or 93"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not isinstance(videos, list) or not videos:
                self._send_json(
                    {"error": "videos must be a non-empty list"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            video_names = [Path(str(v)).name for v in videos if str(v).strip()]
            if not video_names:
                self._send_json({"error": "no valid video names"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                _normalize_dashboard_codec_mode(codec_mode)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if codec_mode.lower() in {"vbr", "abr", "bitrate"}:
                target_bitrate_text = str(target_bitrate or "").strip()
                target_compression_rate: float | None = None
                if target_compression_rate_raw not in (None, ""):
                    try:
                        target_compression_rate = float(target_compression_rate_raw)
                    except (TypeError, ValueError):
                        self._send_json(
                            {"error": "target_compression_rate must be a number in (0, 1)"},
                            HTTPStatus.BAD_REQUEST,
                        )
                        return
                    if not (0.0 < target_compression_rate < 1.0):
                        self._send_json(
                            {"error": "target_compression_rate must be in (0, 1)"},
                            HTTPStatus.BAD_REQUEST,
                        )
                        return
                if not target_bitrate_text and target_compression_rate is None:
                    self._send_json(
                        {
                            "error": (
                                "target_bitrate or target_compression_rate "
                                "is required for vbr mode"
                            )
                        },
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                target_bitrate = target_bitrate_text or None
            else:
                target_bitrate = None
                target_compression_rate = None
            result = _start_compress_thread(
                codec_mode=codec_mode,
                vmaf_threshold=vmaf_threshold,
                target_bitrate=str(target_bitrate) if target_bitrate else None,
                target_compression_rate=target_compression_rate,
                video_names=video_names,
                encode_params=body.get("encode_params")
                if isinstance(body.get("encode_params"), dict)
                else None,
            )
            self._send_json(result)
            return
        if parsed.path == "/api/compress-stop":
            self._send_json(stop_compression())
            return
        if parsed.path == "/api/analyze-log":
            body = self._read_json_body()
            try:
                filters = _parse_analyze_filters(body)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            source = str(body.get("source") or "wandb").strip().lower()
            paste = body.get("text")
            if isinstance(paste, str) and paste.strip():
                source = "paste"

            if source in {"wandb", "fetch", "current"}:
                result = _start_analyze_wandb_thread(
                    codec=filters["codec"],
                    mode=filters["mode"],
                    vmaf_threshold=filters["vmaf_threshold"],
                    uids=filters["uids"],
                    failures_only=filters["failures_only"],
                )
                self._send_json(result)
                return

            if source == "paste" and isinstance(paste, str) and paste.strip():
                try:
                    payload = analyze_validator_log(
                        text=paste,
                        codec=filters["codec"],
                        mode=filters["mode"],
                        vmaf_threshold=filters["vmaf_threshold"],
                        uids=filters["uids"],
                        failures_only=filters["failures_only"],
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(payload)
                return

            self._send_json(
                {"error": "source must be wandb (default) or provide text"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if parsed.path == "/api/analyze-search":
            body = self._read_json_body()
            try:
                filters = _parse_analyze_filters(body)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            try:
                status = search_analyze_cache(
                    codec=filters["codec"],
                    mode=filters["mode"],
                    vmaf_threshold=filters["vmaf_threshold"],
                    uids=filters["uids"],
                    failures_only=filters["failures_only"],
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "status": status})
            return

        if parsed.path == "/api/analyze-clear":
            self._send_json(clear_analyze_status())
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self, head_only: bool = False) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index = STATIC / "index.html"
            if not index.is_file():
                self._send_json({"error": "index.html missing"}, HTTPStatus.NOT_FOUND)
                return
            body = index.read_bytes()
            if head_only:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            self._send_bytes(body, "text/html; charset=utf-8")
            return

        if path == "/api/videos":
            payload = {"videos": build_catalog()}
            if head_only:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            self._send_json(payload)
            return

        if path == "/api/system":
            payload = system_stats()
            if head_only:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return
            self._send_json(payload)
            return

        if path == "/api/fetch-status":
            self._send_json(fetch_status())
            return

        if path == "/api/compress-status":
            self._send_json(compress_status())
            return

        if path == "/api/analyze-status":
            self._send_json(analyze_status())
            return

        if path == "/api/encode-defaults":
            self._send_json({"defaults": encode_defaults()})
            return

        if path == "/api/analyze-logs":
            self._send_json({"logs": discover_log_paths()})
            return

        media_match = re.fullmatch(r"/media/(original|compressed)/([^/]+)", path)
        if media_match:
            kind, name = media_match.groups()
            file_path = _find_original(name) if kind == "original" else _find_compressed(name)
            if file_path is None:
                self._send_json({"error": "video not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_file(file_path)
            return

        download_match = re.fullmatch(r"/download/(original|compressed)/([^/]+)", path)
        if download_match:
            kind, name = download_match.groups()
            file_path = _find_original(name) if kind == "original" else _find_compressed(name)
            if file_path is None:
                self._send_json({"error": "video not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_file(file_path, download_name=file_path.name, as_attachment=True)
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Vidaio dashboard: http://{HOST}:{PORT}")
    print(f"Compression dir: {COMPRESSION_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
