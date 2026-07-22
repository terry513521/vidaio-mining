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

# Preprocess presets for VBR experiments (VMAF NEG survey + brave micro set).
# Mild enhancement is allowed but must pass the dual-VMAF delta gate;
# fleet/CLI A/B or sweep keeps only winners that clear gates.
#
# Brave insight from thr=85 / rate≈0.04 sweeps: unsharp_mild / contrast_mild
# raise VMAF NEG but overshoot |base−neg|≤3. Dial strength down + try
# denoise→micro-sharpen combos (NEG paper) and edge-aware restore filters.
_PREPROCESS_FILTERS = {
    "none": None,
    # Gentle denoising
    "hqdn3d_light": "hqdn3d=1.5:1.5:6:6",
    "hqdn3d_med": "hqdn3d=3:2:8:8",
    # Prefer temporal over spatial — free bits on grain without mushing edges
    "hqdn3d_temporal": "hqdn3d=0.8:0.6:5:4",
    "atadenoise_light": "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04",
    "vaguedenoiser_light": "vaguedenoiser=threshold=1.5:method=soft:nsteps=4",
    # Bilateral (edge-preserving smooth)
    "bilateral_light": "bilateral=sigmaS=1.5:sigmaR=0.08",
    # Very mild sharpening (survey); often fails NEG delta — keep only if A/B wins
    "unsharp_mild": "unsharp=5:5:0.35:5:5:0.0",
    # Micro / nano sharpen — same direction as mild, sized for delta≤3
    "unsharp_micro": "unsharp=5:5:0.18:5:5:0.0",
    "unsharp_nano": "unsharp=3:3:0.10:3:3:0.0",
    # Slight contrast (survey) + micro/nano dial-backs
    "contrast_mild": "eq=contrast=1.05:brightness=0.0:saturation=1.0",
    "contrast_micro": "eq=contrast=1.02:brightness=0.0:saturation=1.015",
    "contrast_nano": "eq=contrast=1.012:brightness=0.0:saturation=1.008",
    # Contrast-adaptive sharpen (often gentler delta than global unsharp)
    "cas_micro": "cas=0.18",
    "cas_nano": "cas=0.10",
    # Smooth flats, slight outline restore (classic smartblur “restore”)
    "smartblur_restore": "smartblur=1.0:-0.55:20:0.6:-0.35:20",
    # Paper-style: mild denoise then micro edge restore
    "denoise_unsharp": "hqdn3d=1.0:0.8:4:3,unsharp=5:5:0.16:5:5:0.0",
    "bilateral_unsharp": "bilateral=sigmaS=1.2:sigmaR=0.06,unsharp=5:5:0.14:5:5:0.0",
    "temporal_cas": "hqdn3d=0.8:0.6:5:4,cas=0.12",
}

# Full survey candidate list for preprocess_sweep (includes none).
SURVEY_PREPROCESS_SWEEP: tuple[str, ...] = (
    "none",
    "hqdn3d_light",
    "bilateral_light",
    "unsharp_mild",
    "contrast_mild",
)

# Aggressive micro-enhancement sweep aimed at higher VMAF NEG under delta≤3.
BRAVE_PREPROCESS_SWEEP: tuple[str, ...] = (
    "none",
    "unsharp_micro",
    "unsharp_nano",
    "contrast_micro",
    "contrast_nano",
    "cas_micro",
    "cas_nano",
    "smartblur_restore",
    "denoise_unsharp",
    "bilateral_unsharp",
    "hqdn3d_temporal",
    "temporal_cas",
    # vaguedenoiser_light omitted from default brave sweep — too slow on mashups
)


def resolve_preprocess_vf(preprocess: Optional[str]) -> Optional[str]:
    """Map a preprocess preset name to an ffmpeg filter string.

    Returns None for 'none'/empty. Raises on unknown presets. Raw filter
    strings are rejected — use named presets only.
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
    twopass: bool = False,
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
    progress_reference_bytes: Optional[int] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    """Encode full file or a time window (ss/t) with libx265 or hevc_nvenc."""
    progress_kwargs = {
        "progress_reference_path": progress_reference_path,
        "progress_reference_bytes": progress_reference_bytes,
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
            twopass=twopass,
            **progress_kwargs,
        )
    if twopass:
        raise ValueError("twopass is only supported for libx265")
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
    progress_reference_bytes: Optional[int] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    if progress_reference_path and progress_label:
        return _run_ffmpeg_with_progress(
            cmd,
            output_path,
            timeout,
            progress_reference_path=progress_reference_path,
            progress_reference_bytes=progress_reference_bytes,
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
    progress_reference_bytes: Optional[int] = None,
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
                        f"{format_compression(progress_reference_path, output_path, reference_bytes=progress_reference_bytes)}"
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
            f"{format_compression(progress_reference_path, output_path, reference_bytes=progress_reference_bytes)}"
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
    twopass: bool = False,
    progress_reference_path: Optional[str] = None,
    progress_reference_bytes: Optional[int] = None,
    progress_label: Optional[str] = None,
    progress_interval_sec: float = 15.0,
) -> EncodeResult:
    mode = normalize_codec_mode(codec_mode)
    if twopass and not is_abr_mode(mode):
        raise ValueError("twopass requires ABR mode with -b:v")

    def build_cmd(*, pass_n: Optional[int], dest: str) -> list[str]:
        ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
        cmd = [ffmpeg, "-y", "-hide_banner"]
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
            ]
        )
        if pass_n != 1:
            cmd.extend(["-movflags", "+faststart"])

        profile = (libx265_profile or "main").lower().strip()
        if profile:
            cmd.extend(["-profile:v", profile])

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

        x265_params = params or ""
        if pass_n is not None:
            # libx265 2-pass via ffmpeg -pass; also stamp pass=N in x265-params.
            pass_kv = f"pass={pass_n}"
            x265_params = f"{x265_params}:{pass_kv}" if x265_params else pass_kv
            cmd.extend(["-pass", str(pass_n)])
        if x265_params:
            cmd.extend(["-x265-params", x265_params])
        cmd.append(dest)
        return cmd

    if not twopass:
        return _run_ffmpeg(
            build_cmd(pass_n=None, dest=output_path),
            output_path,
            timeout,
            progress_reference_path=progress_reference_path,
            progress_reference_bytes=progress_reference_bytes,
            progress_label=progress_label,
            progress_interval_sec=progress_interval_sec,
        )

    # Pass 1 → null sink; pass 2 → real output. Stats land in ffmpeg2pass-0.log.
    pass1 = build_cmd(pass_n=1, dest="/dev/null")
    pass1 = pass1[:-1] + ["-f", "null", "/dev/null"]

    label1 = f"{progress_label} pass1" if progress_label else "libx265 pass1"
    r1 = _run_ffmpeg(
        pass1,
        "/dev/null",
        timeout,
        progress_reference_path=progress_reference_path,
        progress_reference_bytes=progress_reference_bytes,
        progress_label=label1,
        progress_interval_sec=progress_interval_sec,
    )
    if not r1.ok:
        return r1

    label2 = f"{progress_label} pass2" if progress_label else "libx265 pass2"
    return _run_ffmpeg(
        build_cmd(pass_n=2, dest=output_path),
        output_path,
        timeout,
        progress_reference_path=progress_reference_path,
        progress_reference_bytes=progress_reference_bytes,
        progress_label=label2,
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
    progress_reference_bytes: Optional[int] = None,
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
        progress_reference_bytes=progress_reference_bytes,
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
