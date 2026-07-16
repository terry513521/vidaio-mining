"""HEVC (libx265) encode helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional

from ffmpeg_tools import resolve_binary


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
    codec_mode: str = "CRF",
    crf: Optional[int] = None,
    bitrate: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
    timeout: Optional[float] = None,
    ss: Optional[float] = None,
    t: Optional[float] = None,
) -> EncodeResult:
    """Encode full file or a time window (ss/t) with libx265."""
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
    ]

    # Accurate seek for proxy windows: -ss after -i is slower but frame-accurate.
    if ss is not None:
        cmd.extend(["-ss", str(ss)])

    cmd.extend(["-i", input_path])

    if t is not None:
        cmd.extend(["-t", str(t)])

    cmd.extend(
        [
            "-vf",
            "setsar=1",
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

    mode = codec_mode.upper()
    if mode == "CRF":
        if crf is None:
            raise ValueError("crf is required for CRF mode")
        cmd.extend(["-crf", str(crf)])
    elif mode == "VBR":
        if not bitrate:
            raise ValueError("bitrate is required for VBR mode")
        cmd.extend(["-b:v", bitrate])
    else:
        raise ValueError(f"Unsupported codec_mode: {codec_mode}")

    if params:
        cmd.extend(["-x265-params", params])

    cmd.append(output_path)

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
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
