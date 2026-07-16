"""Locate ffmpeg / ffprobe binaries."""

from __future__ import annotations

import os
import shutil
from typing import Optional


_FALLBACKS = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]


def _looks_like_exe(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def resolve_binary(name: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if not _looks_like_exe(explicit):
            raise FileNotFoundError(f"{name} not found at {explicit}")
        return explicit

    found = shutil.which(name)
    if found:
        return found

    exe = f"{name}.exe" if os.name == "nt" else name
    for folder in _FALLBACKS:
        candidate = os.path.join(folder, exe)
        if _looks_like_exe(candidate):
            return candidate

    raise FileNotFoundError(
        f"{name} not found. Add FFmpeg to PATH or pass ffmpeg_bin/ffprobe_bin in the request."
    )
