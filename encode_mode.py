"""Encode mode normalization: RC (quality) vs ABR (average bitrate)."""

from __future__ import annotations

# RC = constant quality (libx265 -crf, NVENC -cq + nvenc_rc).
RC_ALIASES = frozenset({"RC", "CRF", "CQ"})
# ABR = target average bitrate (libx265/NVENC -b:v + nvenc_rc).
ABR_ALIASES = frozenset({"ABR", "VBR", "BITRATE", "AVG_BITRATE", "AVGBITRATE"})

_NVENC_ABR_RC = frozenset({"vbr", "vbr_hq", "cbr", "cbr_hq"})


def normalize_codec_mode(mode: str) -> str:
    """Return canonical mode: ``RC`` or ``ABR``."""
    key = (mode or "RC").upper().strip()
    if key in RC_ALIASES:
        return "RC"
    if key in ABR_ALIASES:
        return "ABR"
    raise ValueError(
        f"codec_mode must be RC (constant quality) or ABR (average bitrate), "
        f"got {mode!r}. Aliases: RC={sorted(RC_ALIASES)}, ABR={sorted(ABR_ALIASES)}"
    )


def is_rc_mode(mode: str) -> bool:
    return normalize_codec_mode(mode) == "RC"


def is_abr_mode(mode: str) -> bool:
    return normalize_codec_mode(mode) == "ABR"


def nvenc_rc_ok_for_abr(nvenc_rc: str) -> bool:
    return (nvenc_rc or "vbr").lower().strip() in _NVENC_ABR_RC
