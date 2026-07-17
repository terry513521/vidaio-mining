"""HEVC encode helpers: libx265 (CPU) or hevc_nvenc (GPU)."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from compress_util import format_compression
from encode_mode import is_abr_mode, is_rc_mode, normalize_codec_mode
from ffmpeg_tools import resolve_binary
from logutil import log

# Map common libx265 preset names onto NVENC p1..p7.
_X265_TO_NVENC_PRESET = {
    "ultrafast": "p1",
    "superfast": "p1",
    "veryfast": "p2",
    "faster": "p3",
    "fast": "p4",
    "medium": "p5",
    "slow": "p6",
    "slower": "p7",
    "veryslow": "p7",
    "placebo": "p7",
}

_NVENC_PRESETS = {f"p{i}" for i in range(1, 8)}
_NVENC_TUNES = {"hq", "ll", "ull", "lossless"}
_NVENC_RC = {"vbr", "vbr_hq", "cbr", "cbr_hq", "constqp"}
_NVENC_CQ_RC = {"vbr", "vbr_hq"}  # CQ mode: let -cq drive quality, no bitrate cap
_NVENC_B_REF = {"disabled", "each", "middle"}
_NVENC_MULTIPASS = {"disabled", "qres", "fullres"}

# Safe preprocess presets. Denoise-only by design: enhancement filters
# (unsharp/CLAHE/gamma/contrast) are deliberately excluded because they widen
# base-vs-NEG VMAF and trip the validator delta gate (see arXiv:2107.04510).
_PREPROCESS_FILTERS = {
    "none": None,
    "hqdn3d_light": "hqdn3d=1.5:1.5:6:6",
    "hqdn3d_med": "hqdn3d=3:2:8:8",
    "atadenoise_light": "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04",
}


def resolve_preprocess_vf(preprocess: Optional[str]) -> Optional[str]:
    """Map a safe preprocess preset name to an ffmpeg filter string.

    Returns None for 'none'/empty. Raises on unknown presets. Raw filter
    strings are rejected to keep this denoise-only.
    """
    if preprocess is None:
        return None
    key = str(preprocess).lower().strip()
    if key in {"", "none"}:
        return None
    if key not in _PREPROCESS_FILTERS:
        raise ValueError(
            f"preprocess must be one of {sorted(_PREPROCESS_FILTERS)}, got {preprocess!r}"
        )
    return _PREPROCESS_FILTERS[key]


def normalize_nvenc_preset(preset: str) -> str:
    """Accept p1..p7 or libx265-style names; return NVENC pN."""
    value = (preset or "p5").lower().strip()
    if value in _NVENC_PRESETS:
        return value
    if value in _X265_TO_NVENC_PRESET:
        return _X265_TO_NVENC_PRESET[value]
    raise ValueError(
        f"Unknown NVENC preset {preset!r}; use p1..p7 or a libx265 preset name"
    )


@dataclass
class EncodeResult:
    ok: bool
    output_path: str
    returncode: int
    stderr_tail: str
    cmd: list[str]


def encode_hevc(
    input_path: str,
    output_path: str,
    *,
    preset: str,
    params: str,
    codec_mode: str = "RC",
    crf: Optional[int] = None,
    bitrate: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    timeout: Optional[float] = None,
    ss: Optional[float] = None,
    t: Optional[float] = None,
    encoder: str = "libx265",
    preprocess: Optional[str] = None,
    libx265_profile: Optional[str] = None,
    nvenc_tune: str = "hq",
    nvenc_rc: str = "vbr",
    nvenc_multipass: str = "qres",
    nvenc_spatial_aq: bool = True,
    nvenc_temporal_aq: bool = True,
    nvenc_aq_strength: int = 8,
    nvenc_rc_lookahead: int = 0,
    nvenc_bf: int = 0,
    nvenc_gop: Optional[int] = None,
    nvenc_b_ref_mode: str = "disabled",
    nvenc_gpu: int = 0,
    nvenc_hwaccel: bool = False,
    progress_reference_path: Optional[str] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    """Encode full file or a time window (ss/t) with libx265 or hevc_nvenc."""
    progress_kwargs = {
        "progress_reference_path": progress_reference_path,
        "progress_label": progress_label,
        "progress_interval_sec": progress_interval_sec,
    }
    enc = (encoder or "libx265").lower().strip()
    if enc in {"libx265", "x265"}:
        return _encode_libx265(
            input_path,
            output_path,
            preset=preset,
            params=params,
            codec_mode=codec_mode,
            crf=crf,
            bitrate=bitrate,
            ffmpeg_bin=ffmpeg_bin,
            timeout=timeout,
            ss=ss,
            t=t,
            preprocess=preprocess,
            libx265_profile=libx265_profile,
            **progress_kwargs,
        )
    if enc in {"hevc_nvenc", "nvenc", "nvenc_hevc"}:
        return _encode_hevc_nvenc(
            input_path,
            output_path,
            preset=preset,
            codec_mode=codec_mode,
            crf=crf,
            bitrate=bitrate,
            ffmpeg_bin=ffmpeg_bin,
            timeout=timeout,
            ss=ss,
            t=t,
            preprocess=preprocess,
            nvenc_tune=nvenc_tune,
            nvenc_rc=nvenc_rc,
            nvenc_multipass=nvenc_multipass,
            nvenc_spatial_aq=nvenc_spatial_aq,
            nvenc_temporal_aq=nvenc_temporal_aq,
            nvenc_aq_strength=nvenc_aq_strength,
            nvenc_rc_lookahead=nvenc_rc_lookahead,
            nvenc_bf=nvenc_bf,
            nvenc_gop=nvenc_gop,
            nvenc_b_ref_mode=nvenc_b_ref_mode,
            nvenc_gpu=nvenc_gpu,
            nvenc_hwaccel=nvenc_hwaccel,
            **progress_kwargs,
        )
    raise ValueError(f"Unsupported encoder: {encoder!r} (use libx265 or hevc_nvenc)")


def _run_ffmpeg(
    cmd: list[str],
    output_path: str,
    timeout: Optional[float],
    *,
    progress_reference_path: Optional[str] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    if progress_reference_path and progress_label:
        return _run_ffmpeg_with_progress(
            cmd,
            output_path,
            timeout,
            progress_reference_path=progress_reference_path,
            progress_label=progress_label,
            progress_interval_sec=progress_interval_sec,
        )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (
            (exc.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return EncodeResult(
            ok=False,
            output_path=output_path,
            returncode=-1,
            stderr_tail=f"timeout after {timeout}s\n{stderr[-1500:]}",
            cmd=cmd,
        )

    return EncodeResult(
        ok=result.returncode == 0,
        output_path=output_path,
        returncode=result.returncode,
        stderr_tail=(result.stderr or "")[-2000:],
        cmd=cmd,
    )


def _run_ffmpeg_with_progress(
    cmd: list[str],
    output_path: str,
    timeout: Optional[float],
    *,
    progress_reference_path: str,
    progress_label: str,
    progress_interval_sec: float,
) -> EncodeResult:
    started = time.monotonic()
    last_progress = 0.0
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    stderr_tail = ""
    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed > timeout:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if proc.stderr is not None:
                    stderr_tail = proc.stderr.read() or ""
                return EncodeResult(
                    ok=False,
                    output_path=output_path,
                    returncode=-1,
                    stderr_tail=f"timeout after {timeout}s\n{stderr_tail[-1500:]}",
                    cmd=cmd,
                )
            if elapsed - last_progress >= progress_interval_sec:
                last_progress = elapsed
                if os.path.isfile(output_path):
                    log(
                        f"  {progress_label} … {elapsed:.0f}s "
                        f"{format_compression(progress_reference_path, output_path)}"
                    )
            time.sleep(1.0)
        if proc.stderr is not None:
            stderr_tail = proc.stderr.read() or ""
    finally:
        if proc.poll() is None:
            proc.kill()

    elapsed = time.monotonic() - started
    ok = proc.returncode == 0
    if ok:
        log(
            f"  {progress_label} encode done {elapsed:.1f}s "
            f"{format_compression(progress_reference_path, output_path)}"
        )
    return EncodeResult(
        ok=ok,
        output_path=output_path,
        returncode=int(proc.returncode or 0),
        stderr_tail=stderr_tail[-2000:],
        cmd=cmd,
    )


def _encode_libx265(
    input_path: str,
    output_path: str,
    *,
    preset: str,
    params: str,
    codec_mode: str,
    crf: Optional[int],
    bitrate: Optional[str],
    ffmpeg_bin: Optional[str],
    timeout: Optional[float],
    ss: Optional[float],
    t: Optional[float],
    preprocess: Optional[str] = None,
    libx265_profile: Optional[str] = None,
    progress_reference_path: Optional[str] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
    cmd = [ffmpeg, "-y", "-hide_banner"]

    # Accurate seek for proxy windows: -ss after -i is slower but frame-accurate.
    if ss is not None:
        cmd.extend(["-ss", str(ss)])

    cmd.extend(["-i", input_path])

    if t is not None:
        cmd.extend(["-t", str(t)])

    pre_vf = resolve_preprocess_vf(preprocess)
    vf = f"{pre_vf},setsar=1" if pre_vf else "setsar=1"
    cmd.extend(
        [
            "-vf",
            vf,
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-an",
            "-movflags",
            "+faststart",
        ]
    )

    profile = (libx265_profile or "main").lower().strip()
    if profile:
        cmd.extend(["-profile:v", profile])

    mode = normalize_codec_mode(codec_mode)
    if is_rc_mode(mode):
        if crf is None:
            raise ValueError("crf is required for RC mode")
        cmd.extend(["-crf", str(crf)])
    elif is_abr_mode(mode):
        if not bitrate:
            raise ValueError("bitrate is required for ABR mode")
        cmd.extend(["-b:v", bitrate])
    else:
        raise ValueError(f"Unsupported codec_mode: {codec_mode}")

    if params:
        cmd.extend(["-x265-params", params])

    cmd.append(output_path)
    return _run_ffmpeg(
        cmd,
        output_path,
        timeout,
        progress_reference_path=progress_reference_path,
        progress_label=progress_label,
        progress_interval_sec=progress_interval_sec,
    )


def _encode_hevc_nvenc(
    input_path: str,
    output_path: str,
    *,
    preset: str,
    codec_mode: str,
    crf: Optional[int],
    bitrate: Optional[str],
    ffmpeg_bin: Optional[str],
    timeout: Optional[float],
    ss: Optional[float],
    t: Optional[float],
    preprocess: Optional[str] = None,
    nvenc_tune: str,
    nvenc_rc: str,
    nvenc_multipass: str,
    nvenc_spatial_aq: bool,
    nvenc_temporal_aq: bool,
    nvenc_aq_strength: int,
    nvenc_rc_lookahead: int,
    nvenc_bf: int,
    nvenc_gop: Optional[int],
    nvenc_b_ref_mode: str,
    nvenc_gpu: int,
    nvenc_hwaccel: bool,
    progress_reference_path: Optional[str] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
    nv_preset = normalize_nvenc_preset(preset)
    tune = (nvenc_tune or "hq").lower().strip()
    rc = (nvenc_rc or "vbr").lower().strip()
    multipass = (nvenc_multipass or "qres").lower().strip()
    b_ref = (nvenc_b_ref_mode or "disabled").lower().strip()

    if tune not in _NVENC_TUNES:
        raise ValueError(f"nvenc_tune must be one of {sorted(_NVENC_TUNES)}, got {tune!r}")
    if rc not in _NVENC_RC:
        raise ValueError(f"nvenc_rc must be one of {sorted(_NVENC_RC)}, got {rc!r}")
    if multipass not in _NVENC_MULTIPASS:
        raise ValueError(
            f"nvenc_multipass must be one of {sorted(_NVENC_MULTIPASS)}, got {multipass!r}"
        )
    if b_ref not in _NVENC_B_REF:
        raise ValueError(f"nvenc_b_ref_mode must be one of {sorted(_NVENC_B_REF)}, got {b_ref!r}")

    cmd = [ffmpeg, "-y", "-hide_banner"]

    if nvenc_hwaccel:
        # Decode on GPU when possible; fall back still works if CUDA decode fails.
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

    if ss is not None:
        cmd.extend(["-ss", str(ss)])

    cmd.extend(["-i", input_path])

    if t is not None:
        cmd.extend(["-t", str(t)])

    # Keep SAR/pix_fmt compatible with validator gates.
    # With CUDA frames, convert back to system memory via hwdownload before encode options
    # that expect yuv420p — simplest path: always force yuv420p via vf.
    pre_vf = resolve_preprocess_vf(preprocess)
    if nvenc_hwaccel:
        # CPU denoise needs frames in system memory: run after hwdownload.
        base_vf = "hwdownload,format=nv12,setsar=1,format=yuv420p"
        vf = f"{base_vf},{pre_vf}" if pre_vf else base_vf
    else:
        vf = f"{pre_vf},setsar=1" if pre_vf else "setsar=1"

    cmd.extend(
        [
            "-vf",
            vf,
            "-c:v",
            "hevc_nvenc",
            "-preset",
            nv_preset,
            "-tune",
            tune,
            "-rc",
            rc,
            "-multipass",
            multipass,
            "-gpu",
            str(int(nvenc_gpu)),
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-an",
            "-movflags",
            "+faststart",
        ]
    )

    if nvenc_spatial_aq:
        cmd.extend(["-spatial-aq", "1", "-aq-strength", str(int(nvenc_aq_strength))])
    if nvenc_temporal_aq:
        cmd.extend(["-temporal-aq", "1"])

    if int(nvenc_rc_lookahead) > 0:
        cmd.extend(["-rc-lookahead", str(int(nvenc_rc_lookahead))])
    if int(nvenc_bf) > 0:
        cmd.extend(["-bf", str(int(nvenc_bf))])
        if b_ref != "disabled":
            cmd.extend(["-b_ref_mode", b_ref])
    if nvenc_gop is not None and int(nvenc_gop) > 0:
        cmd.extend(["-g", str(int(nvenc_gop))])

    mode = normalize_codec_mode(codec_mode)
    if is_rc_mode(mode):
        # NVENC has no true CRF; CQ is the closest constant-quality control.
        if crf is None:
            raise ValueError("crf is required for RC mode (mapped to NVENC -cq)")
        cmd.extend(["-cq", str(crf)])
        if rc in _NVENC_CQ_RC:
            # Let CQ drive quality; avoid forcing a high average bitrate.
            cmd.extend(["-b:v", "0"])
    elif is_abr_mode(mode):
        if not bitrate:
            raise ValueError("bitrate is required for ABR mode")
        # Average-bitrate: -b:v target. VBR modes need maxrate/bufsize so NVENC
        # tracks the average (bare -b:v alone often overshoots).
        cmd.extend(["-b:v", bitrate])
        if rc in {"vbr", "vbr_hq"}:
            mbps = _parse_bitrate_to_mbps(bitrate)
            if mbps is not None and mbps > 0:
                maxrate = f"{mbps * 1.2:.3f}M"
                bufsize = f"{mbps * 2.0:.3f}M"
                cmd.extend(["-maxrate", maxrate, "-bufsize", bufsize])
    else:
        raise ValueError(f"Unsupported codec_mode: {codec_mode}")

    cmd.append(output_path)
    return _run_ffmpeg(
        cmd,
        output_path,
        timeout,
        progress_reference_path=progress_reference_path,
        progress_label=progress_label,
        progress_interval_sec=progress_interval_sec,
    )


def _parse_bitrate_to_mbps(value: str) -> Optional[float]:
    text = str(value).strip().lower()
    if not text:
        return None
    multipliers = {"k": 1 / 1000.0, "m": 1.0, "g": 1000.0}
    suffix = text[-1]
    try:
        if suffix in multipliers:
            return float(text[:-1]) * multipliers[suffix]
        return float(text)
    except ValueError:
        return None


def extract_proxy_reference(
    input_path: str,
    output_path: str,
    *,
    ss: float,
    t: float,
    ffmpeg_bin: Optional[str] = None,
    timeout: Optional[float] = None,
) -> EncodeResult:
    """Extract a time window used as VMAF reference for proxy search.

    Prefer stream-copy so the proxy reference matches source bits for that window.
    Fall back to a near-lossless re-encode if copy fails (e.g. mid-GOP cut).
    """
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)

    def _run(cmd: list[str]) -> EncodeResult:
        return _run_ffmpeg(cmd, output_path, timeout)

    copy_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-ss",
        str(ss),
        "-i",
        input_path,
        "-t",
        str(t),
        "-c",
        "copy",
        "-an",
        output_path,
    ]
    copied = _run(copy_cmd)
    if copied.ok:
        return copied

    # Fallback: high-quality rewrap (same resolution / duration window)
    reencode_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-ss",
        str(ss),
        "-i",
        input_path,
        "-t",
        str(t),
        "-vf",
        "setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]
    return _run(reencode_cmd)
