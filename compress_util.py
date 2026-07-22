"""File-size compression helpers for logging and scoring."""

from __future__ import annotations

import os
from typing import Optional


def measure_compression(
    reference_path: str,
    output_path: str,
    *,
    reference_bytes: Optional[int] = None,
) -> tuple[float, float]:
    """Return ``(compression_rate, compression_ratio)`` using validator semantics.

    ``compression_rate = out / in``, ``compression_ratio = in / out``.

    Prefer ``reference_bytes`` (e.g. source packet bytes for a segment) when the
    VMAF reference file is a lossless re-encode and must not be used as size-in.
    """
    if not os.path.isfile(output_path):
        return 1.0, 1.0
    if reference_bytes is not None:
        original_size = int(reference_bytes)
    elif os.path.isfile(reference_path):
        original_size = os.path.getsize(reference_path)
    else:
        return 1.0, 1.0
    compressed_size = os.path.getsize(output_path)
    if original_size <= 0 or compressed_size >= original_size:
        return 1.0, 1.0
    rate = compressed_size / original_size
    return rate, 1.0 / rate


def format_compression(
    reference_path: str,
    output_path: str,
    *,
    reference_bytes: Optional[int] = None,
) -> str:
    """Human-readable compression summary for logs."""
    rate, ratio = measure_compression(
        reference_path, output_path, reference_bytes=reference_bytes
    )
    if not os.path.isfile(output_path):
        return "rate=? ratio=?"
    out_mb = os.path.getsize(output_path) / (1024 * 1024)
    return f"size={out_mb:.1f}MiB rate={rate:.4f} ratio={ratio:.2f}x"


def format_bitrate_mbps(mbps: float) -> str:
    """Format a bitrate for ffmpeg ``-b:v`` (e.g. ``1.250M`` / ``800k``)."""
    if mbps >= 1.0:
        return f"{mbps:.3f}M"
    return f"{max(1.0, mbps * 1000.0):.0f}k"


def bitrate_for_compression_rate(
    *,
    source_bytes: int,
    duration_sec: float,
    compression_rate: float,
) -> str:
    """Derive ``-b:v`` so expected size ≈ ``compression_rate * source_bytes``.

    Uses ``bits = rate * bytes * 8`` then ``bps = bits / duration``.
    """
    if source_bytes <= 0:
        raise ValueError(f"source_bytes must be > 0, got {source_bytes}")
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if not (0.0 < float(compression_rate) < 1.0):
        raise ValueError(
            f"compression_rate must be in (0, 1), got {compression_rate!r}"
        )
    mbps = (
        (float(compression_rate) * float(source_bytes) * 8.0)
        / float(duration_sec)
        / 1_000_000.0
    )
    if mbps <= 0:
        raise ValueError(f"derived bitrate non-positive ({mbps})")
    return format_bitrate_mbps(mbps)


def parse_bitrate_mbps(value: Optional[str]) -> Optional[float]:
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
