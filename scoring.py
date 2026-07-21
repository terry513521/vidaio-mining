"""VMAF + Vidaio-style compression score."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from encode_mode import is_abr_mode
from ffmpeg_tools import resolve_binary

FRAME_TOLERANCE = 5
FPS_TOLERANCE = 0.3
MAX_VMAF_MODEL_DELTA = 3.0
NEG_MODEL = "version=vmaf_v0.6.1neg"
BASE_MODEL = "version=vmaf_v0.6.1"


def calculate_compression_score(
    vmaf_score: float,
    compression_rate: float,
    vmaf_threshold: float,
    compression_weight: float = 0.70,
    quality_weight: float = 0.30,
    soft_threshold_margin: float = 5.0,
) -> tuple[float, float, float, str]:
    """Mirror vidaio-subnet services/scoring/scoring_function.py."""

    if abs(compression_weight + quality_weight - 1.0) > 0.01:
        raise ValueError("Weights must sum to 1.0")

    hard_cutoff = vmaf_threshold - soft_threshold_margin

    if compression_rate >= 0.80:
        ratio = 1 / compression_rate if compression_rate > 0 else 1.0
        return (
            0.0,
            0.0,
            0.0,
            f"No meaningful compression (ratio: {ratio:.2f}x, rate: {compression_rate:.2f})",
        )

    if vmaf_score < hard_cutoff:
        return 0.0, 0.0, 0.0, f"VMAF {vmaf_score:.2f} below hard cutoff ({hard_cutoff:.2f})"

    compression_ratio = 1 / compression_rate
    normalization_factor = 1.12

    if vmaf_score < vmaf_threshold:
        soft_zone_position = (vmaf_score - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_zone_position**2)

        if compression_ratio <= 20:
            compression_component = ((compression_ratio - 1) / 19) ** 1.5
        else:
            compression_component = 1.0 + 0.3 * math.log(compression_ratio / 20)

        final_score = (compression_component * quality_factor) / normalization_factor
        return (
            min(1.0, final_score),
            compression_component,
            quality_factor,
            f"VMAF {vmaf_score:.2f} in soft zone",
        )

    vmaf_excess = vmaf_score - vmaf_threshold
    max_vmaf_excess = max(1e-6, 100 - vmaf_threshold)
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / max_vmaf_excess)

    if compression_ratio <= 20:
        compression_component = ((compression_ratio - 1.25) / 18.75) ** 0.9
    else:
        compression_component = 1.0 + 0.1 * math.log(compression_ratio / 20)

    final_score = (
        compression_weight * compression_component + quality_weight * quality_component
    ) / normalization_factor

    return min(1.0, final_score), compression_component, quality_component, "success"


@dataclass
class EncodeValidation:
    ok: bool
    errors: list[str]
    probe: dict[str, Any]


def probe_video(
    path: str,
    ffprobe_bin: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    ffprobe = resolve_binary("ffprobe", ffprobe_bin)
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def validate_hevc_output(
    path: str,
    ffprobe_bin: Optional[str] = None,
    timeout: Optional[float] = None,
) -> EncodeValidation:
    errors: list[str] = []
    try:
        probe = probe_video(path, ffprobe_bin, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return EncodeValidation(False, [str(exc)], {})

    video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        return EncodeValidation(False, ["No video stream"], probe)

    codec = (video.get("codec_name") or "").lower()
    if codec not in {"hevc", "h265"}:
        errors.append(f"codec must be hevc, got {codec}")

    pix_fmt = video.get("pix_fmt")
    if pix_fmt != "yuv420p":
        errors.append(f"pix_fmt must be yuv420p, got {pix_fmt}")

    sar = video.get("sample_aspect_ratio") or "1:1"
    if sar not in {"1:1", "1/1", "N/A", "0:1"}:
        # 0:1 often means unspecified; accept with note
        if sar not in {"0:1"}:
            errors.append(f"sample_aspect_ratio should be 1:1, got {sar}")

    fmt_name = (probe.get("format") or {}).get("format_name", "")
    if "mp4" not in fmt_name and "mov" not in fmt_name:
        errors.append(f"container should be mp4, got {fmt_name}")

    return EncodeValidation(ok=len(errors) == 0, errors=errors, probe=probe)


def _parse_fps(value: str | None) -> float:
    try:
        numerator, denominator = (value or "0/1").split("/")
        denominator_f = float(denominator)
        return float(numerator) / denominator_f if denominator_f else 0.0
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _frame_count(
    path: str,
    video: dict[str, Any],
    ffprobe_bin: Optional[str],
    timeout: Optional[float] = None,
) -> int:
    try:
        count = int(video.get("nb_frames") or 0)
        if count > 0:
            return count
    except (TypeError, ValueError):
        pass

    ffprobe = resolve_binary("ffprobe", ffprobe_bin)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=min(120.0, timeout) if timeout is not None else 120.0,
    )
    try:
        return int(result.stdout.strip())
    except (TypeError, ValueError):
        return 0


def _video_info(
    path: str,
    ffprobe_bin: Optional[str],
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    probe = probe_video(path, ffprobe_bin, timeout=timeout)
    video = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video is None:
        raise RuntimeError(f"No video stream in {path}")
    fmt = probe.get("format") or {}
    bit_rate = video.get("bit_rate") or fmt.get("bit_rate")
    try:
        bit_rate_mbps = float(bit_rate) / 1_000_000.0 if bit_rate else None
    except (TypeError, ValueError):
        bit_rate_mbps = None
    return {
        "codec": (video.get("codec_name") or "").lower(),
        "profile": video.get("profile") or "",
        "sar": video.get("sample_aspect_ratio") or "1:1",
        "pix_fmt": video.get("pix_fmt") or "",
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "frames": _frame_count(path, video, ffprobe_bin, timeout=timeout),
        "container": fmt.get("format_name") or "",
        "color_space": video.get("color_space"),
        "color_primaries": video.get("color_primaries"),
        "color_transfer": video.get("color_transfer"),
        "bit_rate_mbps": bit_rate_mbps,
        "probe": probe,
    }


def validate_validator_gates(
    reference_path: str,
    distorted_path: str,
    *,
    ffprobe_bin: Optional[str] = None,
    codec_mode: str = "RC",
    target_bitrate_mbps: Optional[float] = None,
    timeout: Optional[float] = None,
) -> EncodeValidation:
    """Mirror the compression validator's HEVC encoding hard gates."""
    errors: list[str] = []
    try:
        ref = _video_info(reference_path, ffprobe_bin, timeout=timeout)
        dist = _video_info(distorted_path, ffprobe_bin, timeout=timeout)
    except Exception as exc:
        return EncodeValidation(False, [str(exc)], {})

    if dist["codec"] not in {"hevc", "h265"}:
        errors.append(f"codec must be hevc, got {dist['codec']}")
    if dist["width"] != ref["width"] or dist["height"] != ref["height"]:
        errors.append(
            f"resolution must be {ref['width']}x{ref['height']}, "
            f"got {dist['width']}x{dist['height']}"
        )
    if abs(dist["fps"] - ref["fps"]) > FPS_TOLERANCE:
        errors.append(f"fps must be ~{ref['fps']:.3f}, got {dist['fps']:.3f}")
    if abs(dist["frames"] - ref["frames"]) > FRAME_TOLERANCE:
        errors.append(
            f"frame count {dist['frames']} vs ref {ref['frames']} "
            f"(tolerance {FRAME_TOLERANCE})"
        )
    if "ivf" in dist["container"].lower() or not (
        "mp4" in dist["container"].lower() or "mov" in dist["container"].lower()
    ):
        errors.append(f"container must be MP4, got {dist['container']}")
    if dist["profile"] not in {"Main", "Main 10"}:
        errors.append(f"HEVC profile must be Main/Main 10, got {dist['profile']}")
    if dist["sar"] not in {"1:1", "1/1"}:
        errors.append(f"SAR must be 1:1, got {dist['sar']}")
    if dist["pix_fmt"] != "yuv420p":
        errors.append(f"pix_fmt must be yuv420p, got {dist['pix_fmt']}")

    for key in ("color_space", "color_primaries", "color_transfer"):
        ref_value = ref.get(key)
        dist_value = dist.get(key)
        if ref_value and dist_value and ref_value != dist_value:
            errors.append(f"{key} mismatch: ref={ref_value} dist={dist_value}")

    if is_abr_mode(codec_mode):
        measured = dist.get("bit_rate_mbps")
        if measured is None:
            errors.append("Bitrate information missing for ABR mode")
        elif target_bitrate_mbps is not None and measured > target_bitrate_mbps * 1.10:
            errors.append(
                f"Bitrate mismatch for ABR: {measured:.2f} Mbps exceeds "
                f"{target_bitrate_mbps * 1.10:.2f} Mbps"
            )

    return EncodeValidation(
        ok=not errors,
        errors=errors,
        probe={"reference": ref["probe"], "distorted": dist["probe"]},
    )


def _lavfi_relative_path(path: str) -> str:
    """Return a cwd-relative path with forward slashes (no drive letter).

    Absolute Windows paths like ``C:/...`` break libvmaf filter parsing because
    ``:`` separates filter options. Escaping is unreliable across FFmpeg builds,
    so prefer a relative path when possible.
    """
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, os.getcwd())
    except ValueError:
        # Different drives on Windows — fall back to escaped absolute form.
        return abs_path.replace("\\", "/").replace(":", "\\:")
    if rel.startswith(".."):
        # Keep inside cwd tree when caller placed tmp elsewhere.
        return abs_path.replace("\\", "/").replace(":", "\\:")
    return rel.replace("\\", "/")


def _parse_vmaf_json(log_path: str) -> float:
    data = json.loads(open(log_path, encoding="utf-8").read())

    # Prefer pooled harmonic mean if present; else harmonic-mean the frames ourselves.
    pooled = data.get("pooled_metrics") or {}
    for key in ("vmaf", "vmaf_neg", "vmaf_float", "vmaf_float_neg"):
        if key in pooled and "harmonic_mean" in pooled[key]:
            return float(pooled[key]["harmonic_mean"])
        if key in pooled and "mean" in pooled[key]:
            # fallback if pool option ignored
            frames = data.get("frames") or []
            scores = [
                float(f["metrics"][key])
                for f in frames
                if "metrics" in f and key in f["metrics"]
            ]
            if scores:
                return len(scores) / sum(1.0 / max(s, 1e-6) for s in scores)
            return float(pooled[key]["mean"])

    frames = data.get("frames") or []
    scores = []
    for frame in frames:
        metrics = frame.get("metrics") or {}
        for key in ("vmaf", "vmaf_neg", "vmaf_float", "vmaf_float_neg"):
            if key in metrics:
                scores.append(float(metrics[key]))
                break

    if not scores:
        raise RuntimeError(f"No VMAF scores in log: keys={list(data.keys())}")

    return len(scores) / sum(1.0 / max(s, 1e-6) for s in scores)


def _compute_vmaf_native(
    reference_path: str,
    distorted_path: str,
    *,
    ffmpeg_bin: Optional[str],
    n_subsample: int,
    n_threads: int,
    model: str,
    timeout: Optional[float],
) -> float:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)

    # Temp under cwd so log_path can be relative and avoid Windows drive-letter ':'
    with tempfile.TemporaryDirectory(prefix="vmaf_", dir=".") as tmp:
        log_path = os.path.abspath(os.path.join(tmp, "vmaf.json"))
        lavfi_log = _lavfi_relative_path(log_path)
        # main=distorted, reference=second input (libvmaf convention)
        filter_graph = (
            f"[0:v]setpts=PTS-STARTPTS,setsar=1[main];"
            f"[1:v]setpts=PTS-STARTPTS,setsar=1[ref];"
            f"[main][ref]libvmaf="
            f"model={model}:"
            f"log_fmt=json:"
            f"log_path={lavfi_log}:"
            f"pool=harmonic_mean:"
            f"n_threads={n_threads}:"
            f"n_subsample={n_subsample}"
        )

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-i",
            distorted_path,
            "-i",
            reference_path,
            "-lavfi",
            filter_graph,
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0 or not os.path.isfile(log_path):
            raise RuntimeError(
                "VMAF computation failed (native):\n"
                + (result.stderr[-2000:] if result.stderr else "no stderr")
            )
        return _parse_vmaf_json(log_path)


def _default_vmaf_model_path(model: str) -> Optional[str]:
    """Resolve NEG/base VMAF JSON from the subnet checkout."""
    is_neg = "neg" in model.lower()
    filename = "vmaf_v0.6.1neg.json" if is_neg else "vmaf_v0.6.1.json"
    env_name = "VMAF_NEG_MODEL_PATH" if is_neg else "VMAF_BASE_MODEL_PATH"
    candidates = [
        os.environ.get(env_name, "").strip(),
        os.environ.get("VMAF_MODEL_PATH", "").strip() if is_neg else "",
        os.path.join("/root/terry/vidaio-subnet/vmaf/model", filename),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "vidaio-subnet",
            "vmaf",
            "model",
            filename,
        ),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def _libvmaf_model_option(model: str, model_path: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Return (libvmaf model=... value, host path to mount or None)."""
    if model_path and os.path.isfile(model_path):
        return f"path={os.path.abspath(model_path)}", os.path.abspath(model_path)
    # Local eval images may not register either model as a built-in. Mount the
    # same JSON model files used by the subnet checkout.
    if "version=" in model and "path=" not in model:
        fallback = _default_vmaf_model_path(model)
        if fallback:
            return f"path={fallback}", fallback
    return model, None


_GPU_DOCKER_FALLBACK_MARKERS = (
    "driver/library version mismatch",
    "driver not loaded",
    "nvidia-container-cli",
    "could not select device driver",
    "NVIDIA Container Toolkit",
    "nvidia-container-runtime",
    "CUDA_ERROR_NOT_PERMITTED",
    "CUDA_ERROR_NO_DEVICE",
    "cuCtxCreate",
    "cuInit",
)
_DOCKER_TRANSIENT_MARKERS = (
    "Cannot connect to the Docker daemon",
    "error during connect",
    "connection refused",
    "client connection is closing",
    "context canceled",
    "Error waiting for container: Canceled",
    "received signal 15",
    "Is the docker daemon running",
)
_GPU_DEVICE_LOCKS: dict[int, threading.Lock] = {}
_GPU_DEVICE_LOCKS_GUARD = threading.Lock()


def _gpu_device_lock(device: Optional[int]) -> threading.Lock:
    """Serialize concurrent libvmaf_cuda jobs that share the same GPU."""
    key = 0 if device is None else int(device)
    with _GPU_DEVICE_LOCKS_GUARD:
        lock = _GPU_DEVICE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _GPU_DEVICE_LOCKS[key] = lock
        return lock


def _stderr_has_marker(stderr: str, markers: tuple[str, ...]) -> bool:
    text = stderr or ""
    return any(marker in text for marker in markers)


def _wait_for_docker(timeout_sec: float = 120.0) -> bool:
    """Block until ``docker info`` succeeds or timeout."""
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(2.0)
    return False


def _docker_vmaf_attempt(
    reference_path: str,
    distorted_path: str,
    *,
    docker_image: str,
    attempt_gpu: bool,
    n_subsample: int,
    n_threads: int,
    model: str,
    model_path: Optional[str] = None,
    timeout: Optional[float] = None,
    gpu_device: Optional[int] = None,
    gpu_lock_mode: str = "none",
) -> tuple[Optional[float], str, bool]:
    """Run one docker VMAF attempt. Returns (score, stderr, wants_cpu_fallback)."""
    dist_path = os.path.abspath(distorted_path)
    ref_path = os.path.abspath(reference_path)
    dist_dir = os.path.dirname(dist_path)
    ref_dir = os.path.dirname(ref_path)
    model_opt, mount_model = _libvmaf_model_option(model, model_path)

    with tempfile.TemporaryDirectory(prefix="vmaf_docker_") as tmp:
        log_path = os.path.join(tmp, "vmaf.json")
        volumes = {dist_dir, ref_dir, tmp}
        if mount_model:
            volumes.add(os.path.dirname(mount_model))
        vol_args: list[str] = []
        for vol in sorted(volumes):
            vol_args.extend(["-v", f"{vol}:{vol}"])

        if attempt_gpu:
            filter_graph = (
                f"[0:v]format=yuv420p,hwupload_cuda[dis];"
                f"[1:v]format=yuv420p,hwupload_cuda[ref];"
                f"[dis][ref]libvmaf_cuda="
                f"n_subsample={n_subsample}:"
                f"model='{model_opt}':"
                f"log_fmt=json:"
                f"log_path={log_path}"
            )
        else:
            filter_graph = (
                f"[0:v]setpts=PTS-STARTPTS,setsar=1[main];"
                f"[1:v]setpts=PTS-STARTPTS,setsar=1[ref];"
                f"[main][ref]libvmaf="
                f"model={model_opt}:"
                f"log_fmt=json:"
                f"log_path={log_path}:"
                f"pool=harmonic_mean:"
                f"n_threads={n_threads}:"
                f"n_subsample={n_subsample}"
            )

        cmd = ["docker", "run", "--rm", "--entrypoint", "ffmpeg"]
        if attempt_gpu:
            gpu_spec = (
                f"device={int(gpu_device)}"
                if gpu_device is not None
                else "all"
            )
            cmd.extend(
                [
                    "--gpus",
                    gpu_spec,
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,video,utility",
                ]
            )
        cmd.extend(vol_args)
        cmd.extend(
            [
                docker_image,
                "-hide_banner",
                "-i",
                dist_path,
                "-i",
                ref_path,
                "-lavfi",
                filter_graph,
                "-f",
                "null",
                "-",
            ]
        )

        gpu_lock = None
        acquired = False
        if attempt_gpu and gpu_lock_mode in {"blocking", "held"}:
            gpu_lock = _gpu_device_lock(gpu_device)
            if gpu_lock_mode == "blocking":
                gpu_lock.acquire()
                acquired = True

        max_docker_tries = 4
        result = None
        last_stderr = ""
        try:
            for docker_try in range(1, max_docker_tries + 1):
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
                last_stderr = (result.stderr or "") + (result.stdout or "")
                if result.returncode == 0 and os.path.isfile(log_path):
                    return _parse_vmaf_json(log_path), last_stderr, False

                if _stderr_has_marker(last_stderr, _DOCKER_TRANSIENT_MARKERS):
                    if docker_try < max_docker_tries:
                        print(
                            f"[vmaf] docker transient failure "
                            f"(try {docker_try}/{max_docker_tries}); waiting for daemon…",
                            flush=True,
                        )
                        _wait_for_docker(timeout_sec=120.0)
                        time.sleep(2.0)
                        continue
                    break

                if (
                    attempt_gpu
                    and _stderr_has_marker(last_stderr, _GPU_DOCKER_FALLBACK_MARKERS)
                ):
                    return None, last_stderr, True

                raise RuntimeError(
                    "VMAF computation failed (docker):\n"
                    + (last_stderr[-2000:] if last_stderr else "no stderr")
                    + "\nHint: use an image with libvmaf, e.g.\n"
                    "  --vmaf-docker-image vmaf_ffmpeg\n"
                    "and ensure NEG model exists at vidaio-subnet/vmaf/model/vmaf_v0.6.1neg.json"
                )
        finally:
            if gpu_lock is not None and acquired:
                gpu_lock.release()

        if (
            attempt_gpu
            and result is not None
            and _stderr_has_marker(last_stderr, _GPU_DOCKER_FALLBACK_MARKERS)
        ):
            return None, last_stderr, True

        raise RuntimeError(
            "VMAF computation failed (docker):\n"
            + (last_stderr[-2000:] if last_stderr else "no stderr")
            + "\nHint: use an image with libvmaf, e.g.\n"
            "  --vmaf-docker-image vmaf_ffmpeg\n"
            "and ensure NEG model exists at vidaio-subnet/vmaf/model/vmaf_v0.6.1neg.json"
        )


def _compute_vmaf_docker(
    reference_path: str,
    distorted_path: str,
    *,
    docker_image: str,
    use_gpus: bool,
    n_subsample: int,
    n_threads: int,
    model: str,
    model_path: Optional[str] = None,
    timeout: Optional[float] = None,
    gpu_device: Optional[int] = None,
    gpu_prefer: bool = False,
) -> float:
    """Run VMAF inside a Docker image that provides ffmpeg+libvmaf.

    Uses ``--entrypoint ffmpeg`` so both validator images (ENTRYPOINT ffmpeg)
    and local eval images (python entrypoint) work.

    When ``gpu_prefer=True``, take the GPU only if it is free (non-blocking
    lock); otherwise score on CPU so parallel workers are not stalled.

    When ``use_gpus=True`` without ``gpu_prefer``, keep the prior behavior:
    block on the GPU lock, then fall back to CPU only on GPU driver errors.
    """
    if use_gpus and gpu_prefer:
        dev = 0 if gpu_device is None else int(gpu_device)
        lock = _gpu_device_lock(gpu_device)
        if lock.acquire(blocking=False):
            try:
                score, stderr, wants_cpu = _docker_vmaf_attempt(
                    reference_path,
                    distorted_path,
                    docker_image=docker_image,
                    attempt_gpu=True,
                    n_subsample=n_subsample,
                    n_threads=n_threads,
                    model=model,
                    model_path=model_path,
                    timeout=timeout,
                    gpu_device=gpu_device,
                    gpu_lock_mode="held",
                )
                if score is not None:
                    return score
                if wants_cpu:
                    print(
                        f"[vmaf] GPU device {dev} unavailable; using CPU libvmaf",
                        flush=True,
                    )
            finally:
                lock.release()
        else:
            print(
                f"[vmaf] GPU device {dev} busy; using CPU libvmaf",
                flush=True,
            )
        score, _, _ = _docker_vmaf_attempt(
            reference_path,
            distorted_path,
            docker_image=docker_image,
            attempt_gpu=False,
            n_subsample=n_subsample,
            n_threads=n_threads,
            model=model,
            model_path=model_path,
            timeout=timeout,
            gpu_device=gpu_device,
            gpu_lock_mode="none",
        )
        if score is None:
            raise RuntimeError("VMAF computation failed (docker CPU fallback)")
        return score

    gpu_attempts = [True, False] if use_gpus else [False]
    last_stderr = ""

    for attempt_gpu in gpu_attempts:
        try:
            score, last_stderr, wants_cpu = _docker_vmaf_attempt(
                reference_path,
                distorted_path,
                docker_image=docker_image,
                attempt_gpu=attempt_gpu,
                n_subsample=n_subsample,
                n_threads=n_threads,
                model=model,
                model_path=model_path,
                timeout=timeout,
                gpu_device=gpu_device,
                gpu_lock_mode="blocking" if attempt_gpu else "none",
            )
            if score is not None:
                return score
            if attempt_gpu and wants_cpu:
                continue
        except RuntimeError:
            if attempt_gpu and gpu_attempts[-1] is False:
                raise
            if not attempt_gpu:
                raise
            continue

    raise RuntimeError(
        "VMAF computation failed (docker):\n"
        + (last_stderr[-2000:] if last_stderr else "no stderr")
    )


def compute_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    ffmpeg_bin: Optional[str] = None,
    n_subsample: int = 1,
    n_threads: int = 4,
    model: str = "version=vmaf_v0.6.1neg",
    vmaf_backend: str = "docker",
    vmaf_docker_image: str = "vmaf_ffmpeg",
    vmaf_docker_gpus: bool = False,
    vmaf_gpu_device: Optional[int] = None,
    vmaf_gpu_prefer: bool = False,
    timeout: Optional[float] = None,
) -> float:
    backend = (vmaf_backend or "docker").lower().strip()
    if backend == "native":
        return _compute_vmaf_native(
            reference_path,
            distorted_path,
            ffmpeg_bin=ffmpeg_bin,
            n_subsample=n_subsample,
            n_threads=n_threads,
            model=model,
            timeout=timeout,
        )
    if backend == "docker":
        return _compute_vmaf_docker(
            reference_path,
            distorted_path,
            docker_image=vmaf_docker_image or "vmaf_ffmpeg",
            use_gpus=bool(vmaf_docker_gpus),
            n_subsample=n_subsample,
            n_threads=n_threads,
            model=model,
            timeout=timeout,
            gpu_device=vmaf_gpu_device,
            gpu_prefer=bool(vmaf_gpu_prefer),
        )
    raise ValueError(f"Unknown vmaf_backend: {vmaf_backend!r}")


@dataclass
class ScoreResult:
    s_f: float
    vmaf: float
    compression_rate: float
    compression_ratio: float
    compression_component: float
    quality_component: float
    reason: str
    validation_errors: list[str]
    vmaf_base: Optional[float] = None
    vmaf_delta: Optional[float] = None
    passed_encoding_gates: bool = False
    passed_vmaf_delta_gate: bool = False

    @property
    def ok(self) -> bool:
        return self.s_f > 0 and not self.validation_errors


def score_candidate(
    reference_path: str,
    distorted_path: str,
    vmaf_threshold: float,
    *,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
    vmaf_n_subsample: int = 1,
    vmaf_n_threads: int = 4,
    vmaf_backend: str = "docker",
    vmaf_docker_image: str = "vmaf_ffmpeg",
    vmaf_docker_gpus: bool = False,
    vmaf_gpu_device: Optional[int] = None,
    vmaf_gpu_prefer: bool = False,
    compression_rate_override: Optional[float] = None,
    codec_mode: str = "RC",
    target_bitrate_mbps: Optional[float] = None,
    timeout: Optional[float] = None,
    pair_gates: bool = True,
    neg_only: bool = False,
) -> ScoreResult:
    started = time.monotonic()

    def remaining() -> Optional[float]:
        if timeout is None:
            return None
        left = float(timeout) - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("candidate scoring deadline exhausted")
        return left

    try:
        if pair_gates:
            validation = validate_validator_gates(
                reference_path,
                distorted_path,
                ffprobe_bin=ffprobe_bin,
                codec_mode=codec_mode,
                target_bitrate_mbps=target_bitrate_mbps,
                timeout=remaining(),
            )
        else:
            validation = validate_hevc_output(
                distorted_path,
                ffprobe_bin=ffprobe_bin,
                timeout=remaining(),
            )
        if not validation.ok:
            return ScoreResult(
                s_f=0.0,
                vmaf=0.0,
                compression_rate=1.0,
                compression_ratio=1.0,
                compression_component=0.0,
                quality_component=0.0,
                reason="encoding_gate_failed",
                validation_errors=validation.errors,
            )

        if compression_rate_override is not None:
            compression_rate = float(compression_rate_override)
        else:
            original_size = os.path.getsize(reference_path)
            compressed_size = os.path.getsize(distorted_path)
            # The validator clamps non-compressing outputs to a rate of 1.0.
            compression_rate = (
                compressed_size / original_size
                if original_size > 0 and compressed_size < original_size
                else 1.0
            )

        vmaf_neg = compute_vmaf(
            reference_path,
            distorted_path,
            ffmpeg_bin=ffmpeg_bin,
            n_subsample=vmaf_n_subsample,
            n_threads=vmaf_n_threads,
            vmaf_backend=vmaf_backend,
            vmaf_docker_image=vmaf_docker_image,
            vmaf_docker_gpus=vmaf_docker_gpus,
            vmaf_gpu_device=vmaf_gpu_device,
            vmaf_gpu_prefer=vmaf_gpu_prefer,
            model=NEG_MODEL,
            timeout=remaining(),
        )
        if neg_only:
            s_f, c_comp, q_comp, reason = calculate_compression_score(
                vmaf_score=vmaf_neg,
                compression_rate=compression_rate,
                vmaf_threshold=vmaf_threshold,
            )
            return ScoreResult(
                s_f=s_f,
                vmaf=vmaf_neg,
                compression_rate=compression_rate,
                compression_ratio=1.0 / max(compression_rate, 1e-9),
                compression_component=c_comp,
                quality_component=q_comp,
                reason=reason,
                validation_errors=[],
                vmaf_base=None,
                vmaf_delta=None,
                passed_encoding_gates=True,
                passed_vmaf_delta_gate=True,
            )

        vmaf_base = compute_vmaf(
            reference_path,
            distorted_path,
            ffmpeg_bin=ffmpeg_bin,
            n_subsample=vmaf_n_subsample,
            n_threads=vmaf_n_threads,
            vmaf_backend=vmaf_backend,
            vmaf_docker_image=vmaf_docker_image,
            vmaf_docker_gpus=vmaf_docker_gpus,
            vmaf_gpu_device=vmaf_gpu_device,
            vmaf_gpu_prefer=vmaf_gpu_prefer,
            model=BASE_MODEL,
            timeout=remaining(),
        )
        vmaf_delta = abs(vmaf_base - vmaf_neg)

        if vmaf_delta > MAX_VMAF_MODEL_DELTA:
            return ScoreResult(
                s_f=0.0,
                vmaf=vmaf_neg,
                compression_rate=compression_rate,
                compression_ratio=1.0 / max(compression_rate, 1e-9),
                compression_component=0.0,
                quality_component=0.0,
                reason=(
                    f"VMAF model delta {vmaf_delta:.2f} > "
                    f"{MAX_VMAF_MODEL_DELTA:.2f} (enhancement/sharpening detected)"
                ),
                validation_errors=[],
                vmaf_base=vmaf_base,
                vmaf_delta=vmaf_delta,
                passed_encoding_gates=True,
                passed_vmaf_delta_gate=False,
            )

        s_f, c_comp, q_comp, reason = calculate_compression_score(
            vmaf_score=vmaf_neg,
            compression_rate=compression_rate,
            vmaf_threshold=vmaf_threshold,
        )

        return ScoreResult(
            s_f=s_f,
            vmaf=vmaf_neg,
            compression_rate=compression_rate,
            compression_ratio=1.0 / max(compression_rate, 1e-9),
            compression_component=c_comp,
            quality_component=q_comp,
            reason=reason,
            validation_errors=[],
            vmaf_base=vmaf_base,
            vmaf_delta=vmaf_delta,
            passed_encoding_gates=True,
            passed_vmaf_delta_gate=True,
        )
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        return ScoreResult(
            s_f=0.0,
            vmaf=0.0,
            compression_rate=1.0,
            compression_ratio=1.0,
            compression_component=0.0,
            quality_component=0.0,
            reason=f"scoring_timeout: {exc}",
            validation_errors=[str(exc)],
        )
