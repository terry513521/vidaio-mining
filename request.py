"""Compression challenge request body (dict / JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CompressionRequest:
    """Parameters a validator-like caller would send to the compressor."""

    input_path: str
    output_path: str = "compressed.mp4"

    # Vidaio-style challenge knobs
    vmaf_threshold: int = 89  # 85 | 89 | 93
    codec: str = "hevc"  # HEVC-only for now
    codec_mode: str = "CRF"  # CRF | VBR
    target_bitrate: Optional[str] = None  # e.g. "5M" when VBR

    # Search / runtime
    time_budget_sec: float = 600.0
    max_search_steps: int = 8
    max_recipes: int = 2
    max_workers: int = 1
    vbr_max_ratio_to_target: float = 1.1
    vbr_min_mbps_floor: float = 0.5
    crf_min: int = 18
    crf_max: int = 40
    crf_start: Optional[int] = None  # optional seed; else recipe default

    # Feature / VMAF
    sample_frames: int = 60
    vmaf_n_subsample: int = 1
    vmaf_n_threads: int = 4

    work_dir: str = "work"
    keep_candidates: bool = False

    ffmpeg_bin: Optional[str] = None
    ffprobe_bin: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.codec = self.codec.lower().strip()
        self.codec_mode = self.codec_mode.upper().strip()

        if self.codec not in {"hevc", "h265", "x265"}:
            raise ValueError(f"Only HEVC is supported currently, got codec={self.codec!r}")

        self.codec = "hevc"

        if self.codec_mode not in {"CRF", "VBR"}:
            raise ValueError(f"codec_mode must be CRF or VBR, got {self.codec_mode!r}")

        if self.vmaf_threshold not in {85, 89, 93}:
            # Allow custom, but warn via normalization to nearest typical set
            if not (0 < self.vmaf_threshold <= 100):
                raise ValueError(f"vmaf_threshold out of range: {self.vmaf_threshold}")

        if self.codec_mode == "VBR" and not self.target_bitrate:
            raise ValueError("target_bitrate is required when codec_mode=VBR")

        if self.crf_min > self.crf_max:
            raise ValueError("crf_min must be <= crf_max")
        if self.vbr_max_ratio_to_target <= 0:
            raise ValueError("vbr_max_ratio_to_target must be > 0")
        if self.vbr_min_mbps_floor <= 0:
            raise ValueError("vbr_min_mbps_floor must be > 0")

        self.input_path = str(Path(self.input_path).expanduser())
        self.output_path = str(Path(self.output_path).expanduser())
        self.work_dir = str(Path(self.work_dir).expanduser())

    def ensure_input_exists(self) -> None:
        path = Path(self.input_path)
        if not path.is_file():
            raise FileNotFoundError(f"input_path not found: {self.input_path}")
        self.input_path = str(path.resolve())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressionRequest":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if "extra" in payload and isinstance(payload["extra"], dict):
            payload["extra"] = {**payload["extra"], **extra}
        else:
            payload["extra"] = extra
        return cls(**payload)

    @classmethod
    def from_json(cls, path_or_text: str) -> "CompressionRequest":
        p = Path(path_or_text)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = json.loads(path_or_text)
        if not isinstance(data, dict):
            raise ValueError("Request JSON must be an object")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
