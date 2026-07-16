"""VMAF + Vidaio-style compression score."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

from ffmpeg_tools import resolve_binary


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


def probe_video(path: str, ffprobe_bin: Optional[str] = None) -> dict[str, Any]:
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def validate_hevc_output(path: str, ffprobe_bin: Optional[str] = None) -> EncodeValidation:
    errors: list[str] = []
    try:
        probe = probe_video(path, ffprobe_bin)
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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(log_path):
            raise RuntimeError(
                "VMAF computation failed (native):\n"
                + (result.stderr[-2000:] if result.stderr else "no stderr")
            )
        return _parse_vmaf_json(log_path)


def _default_vmaf_model_path() -> Optional[str]:
    """Prefer the subnet NEG model file when present on the host."""
    candidates = [
        os.environ.get("VMAF_MODEL_PATH", "").strip(),
        "/root/terry/vidaio-subnet/vmaf/model/vmaf_v0.6.1neg.json",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "vidaio-subnet",
            "vmaf",
            "model",
            "vmaf_v0.6.1neg.json",
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
    # If caller passed version=neg but image lacks built-ins, fall back to host file.
    if "neg" in model and "path=" not in model:
        fallback = _default_vmaf_model_path()
        if fallback:
            return f"path={fallback}", fallback
    return model, None


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
) -> float:
    """Run VMAF inside a Docker image that provides ffmpeg+libvmaf.

    Uses ``--entrypoint ffmpeg`` so both validator images (ENTRYPOINT ffmpeg)
    and local eval images (python entrypoint) work.
    """
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

        if use_gpus:
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

        # Force ffmpeg entrypoint: validator image uses ENTRYPOINT ["ffmpeg"],
        # while local eval images (e.g. vidaio-compression-eval) may default to python.
        cmd = ["docker", "run", "--rm", "--entrypoint", "ffmpeg"]
        if use_gpus:
            cmd.extend(["--gpus", "all"])
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

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.isfile(log_path):
            raise RuntimeError(
                "VMAF computation failed (docker):\n"
                + (result.stderr[-2000:] if result.stderr else "no stderr")
                + "\nHint: use an image with libvmaf, e.g.\n"
                "  --vmaf-docker-image vidaio-compression-eval\n"
                "and ensure NEG model exists at vidaio-subnet/vmaf/model/vmaf_v0.6.1neg.json"
            )
        return _parse_vmaf_json(log_path)


def compute_vmaf(
    reference_path: str,
    distorted_path: str,
    *,
    ffmpeg_bin: Optional[str] = None,
    n_subsample: int = 1,
    n_threads: int = 4,
    model: str = "version=vmaf_v0.6.1neg",
    vmaf_backend: str = "docker",
    vmaf_docker_image: str = "vidaio-compression-eval",
    vmaf_docker_gpus: bool = False,
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
        )
    if backend == "docker":
        return _compute_vmaf_docker(
            reference_path,
            distorted_path,
            docker_image=vmaf_docker_image or "vidaio-compression-eval",
            use_gpus=bool(vmaf_docker_gpus),
            n_subsample=n_subsample,
            n_threads=n_threads,
            model=model,
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
    vmaf_docker_image: str = "vidaio-compression-eval",
    vmaf_docker_gpus: bool = False,
    compression_rate_override: Optional[float] = None,
) -> ScoreResult:
    validation = validate_hevc_output(distorted_path, ffprobe_bin)
    if not validation.ok:
        return ScoreResult(
            s_f=0.0,
            vmaf=0.0,
            compression_rate=1.0,
            compression_ratio=1.0,
            compression_component=0.0,
            quality_component=0.0,
            reason="validation_failed",
            validation_errors=validation.errors,
        )

    if compression_rate_override is not None:
        compression_rate = float(compression_rate_override)
    else:
        original_size = os.path.getsize(reference_path)
        compressed_size = os.path.getsize(distorted_path)
        compression_rate = compressed_size / max(original_size, 1)

    vmaf = compute_vmaf(
        reference_path,
        distorted_path,
        ffmpeg_bin=ffmpeg_bin,
        n_subsample=vmaf_n_subsample,
        n_threads=vmaf_n_threads,
        vmaf_backend=vmaf_backend,
        vmaf_docker_image=vmaf_docker_image,
        vmaf_docker_gpus=vmaf_docker_gpus,
    )

    s_f, c_comp, q_comp, reason = calculate_compression_score(
        vmaf_score=vmaf,
        compression_rate=compression_rate,
        vmaf_threshold=vmaf_threshold,
    )

    return ScoreResult(
        s_f=s_f,
        vmaf=vmaf,
        compression_rate=compression_rate,
        compression_ratio=1.0 / max(compression_rate, 1e-9),
        compression_component=c_comp,
        quality_component=q_comp,
        reason=reason,
        validation_errors=[],
    )
