"""File-size compression helpers for logging and scoring."""

from __future__ import annotations

import os


def measure_compression(reference_path: str, output_path: str) -> tuple[float, float]:
    """Return ``(compression_rate, compression_ratio)`` using validator semantics."""
    if not os.path.isfile(reference_path) or not os.path.isfile(output_path):
        return 1.0, 1.0
    original_size = os.path.getsize(reference_path)
    compressed_size = os.path.getsize(output_path)
    if original_size <= 0 or compressed_size >= original_size:
        return 1.0, 1.0
    rate = compressed_size / original_size
    return rate, 1.0 / rate


def format_compression(reference_path: str, output_path: str) -> str:
    """Human-readable compression summary for logs."""
    rate, ratio = measure_compression(reference_path, output_path)
    if not os.path.isfile(output_path):
        return "rate=? ratio=?"
    out_mb = os.path.getsize(output_path) / (1024 * 1024)
    return f"size={out_mb:.1f}MiB rate={rate:.4f} ratio={ratio:.2f}x"
