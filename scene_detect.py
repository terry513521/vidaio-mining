"""Scene detection via pyscenedetect (AdaptiveDetector)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from logutil import log


@dataclass(frozen=True)
class SceneSpan:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_segment_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration": self.duration,
        }


@dataclass
class SceneDetectResult:
    ok: bool
    scenes: list[SceneSpan]
    detect_sec: float = 0.0
    error: str = ""


def detect_scenes(
    video_path: str,
    *,
    downscale: int = 1,
    min_scene_len_sec: float = 0.4,
) -> SceneDetectResult:
    """Return scene spans in seconds. Falls back to one full-span scene on failure."""
    started = time.monotonic()
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
    except ImportError as exc:
        return SceneDetectResult(
            ok=False,
            scenes=[],
            detect_sec=time.monotonic() - started,
            error=f"scenedetect not installed: {exc}",
        )

    video_to_detect = video_path
    temp_downscaled = None
    try:
        if downscale is not None and int(downscale) > 1:
            temp_downscaled = _downscale_for_detect(video_path, int(downscale))
            if temp_downscaled:
                video_to_detect = temp_downscaled

        video = open_video(video_to_detect)
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(min_scene_len=int(min_scene_len_sec * 30)))
        manager.detect_scenes(video=video)
        scene_list = manager.get_scene_list()
        spans = [
            SceneSpan(
                index=i,
                start_sec=float(start.get_seconds()),
                end_sec=float(end.get_seconds()),
            )
            for i, (start, end) in enumerate(scene_list)
        ]
        if not spans:
            spans = [_whole_file_span(video_path)]
        log(f"  scenedetect: {len(spans)} scene(s) in {time.monotonic() - started:.1f}s")
        return SceneDetectResult(
            ok=True,
            scenes=spans,
            detect_sec=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  scenedetect failed: {exc}")
        return SceneDetectResult(
            ok=False,
            scenes=[_whole_file_span(video_path)],
            detect_sec=time.monotonic() - started,
            error=str(exc),
        )
    finally:
        if temp_downscaled:
            try:
                import os

                os.remove(temp_downscaled)
            except OSError:
                pass


def _whole_file_span(video_path: str) -> SceneSpan:
    from scoring import probe_video

    probe = probe_video(video_path)
    fmt = probe.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        duration = 30.0
    return SceneSpan(index=0, start_sec=0.0, end_sec=duration)


def _downscale_for_detect(input_path: str, factor: int) -> Optional[str]:
    import subprocess
    import tempfile
    from pathlib import Path

    from ffmpeg_tools import resolve_binary

    out = Path(tempfile.gettempdir()) / f"scene_detect_ds_{Path(input_path).stem}.mkv"
    if out.is_file():
        return str(out)
    ffmpeg = resolve_binary("ffmpeg", None)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i",
        input_path,
        "-vf",
        f"scale=-1:ih/{factor}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-an",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not out.is_file():
        return None
    return str(out)
