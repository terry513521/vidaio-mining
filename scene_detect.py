"""Scene detection via PySceneDetect (ContentDetector / AdaptiveDetector)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from logutil import log

DetectorName = Literal["content", "adaptive"]


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
    detector: str = ""

    @property
    def spans(self) -> list[tuple[float, float]]:
        """``(start_sec, end_sec)`` tuples for ``HEVCFeatureExtractor(scene_spans=…)``."""
        return [(s.start_sec, s.end_sec) for s in self.scenes]


def detect_scenes(
    video_path: str,
    *,
    detector: DetectorName = "content",
    threshold: Optional[float] = None,
    min_scene_len_sec: float = 0.4,
    downscale: int = 1,
) -> SceneDetectResult:
    """Detect scene spans with PySceneDetect.

    Defaults to ``ContentDetector`` (HSV/content-change cuts), which is the
    usual choice for hard-cut mashups. ``adaptive`` uses ``AdaptiveDetector``.
    """
    started = time.monotonic()
    try:
        from scenedetect import (
            AdaptiveDetector,
            ContentDetector,
            SceneManager,
            open_video,
        )
    except ImportError as exc:
        return SceneDetectResult(
            ok=False,
            scenes=[],
            detect_sec=time.monotonic() - started,
            error=f"scenedetect not installed: {exc}",
            detector=detector,
        )

    video_to_detect = video_path
    temp_downscaled = None
    try:
        if downscale is not None and int(downscale) > 1:
            temp_downscaled = _downscale_for_detect(video_path, int(downscale))
            if temp_downscaled:
                video_to_detect = temp_downscaled

        video = open_video(video_to_detect)
        fps = float(getattr(video, "frame_rate", None) or 30.0)
        min_len_frames = max(1, int(round(float(min_scene_len_sec) * max(fps, 1e-6))))

        manager = SceneManager()
        det_name = str(detector or "content").lower().strip()
        if det_name in {"adaptive", "adapt"}:
            # AdaptiveDetector: threshold is optional; use library default if None.
            kwargs: dict[str, Any] = {"min_scene_len": min_len_frames}
            if threshold is not None:
                kwargs["adaptive_threshold"] = float(threshold)
            manager.add_detector(AdaptiveDetector(**kwargs))
            det_name = "adaptive"
        else:
            # ContentDetector: default threshold 27 works well for hard cuts.
            thr = 27.0 if threshold is None else float(threshold)
            manager.add_detector(
                ContentDetector(threshold=thr, min_scene_len=min_len_frames)
            )
            det_name = "content"

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
        log(
            f"  scenedetect[{det_name}]: {len(spans)} scene(s) "
            f"in {time.monotonic() - started:.1f}s"
        )
        return SceneDetectResult(
            ok=True,
            scenes=spans,
            detect_sec=time.monotonic() - started,
            detector=det_name,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  scenedetect failed: {exc}")
        return SceneDetectResult(
            ok=False,
            scenes=[_whole_file_span(video_path)],
            detect_sec=time.monotonic() - started,
            error=str(exc),
            detector=str(detector),
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
