"""Duration-based sample clips from detected scenes, then concat for CRF search."""

from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ffmpeg_tools import resolve_binary
from logutil import log
from scene_detect import SceneSpan
from scoring import _video_info


@dataclass(frozen=True)
class SceneSample:
    scene_index: int
    path: str
    start_sec: float
    frame_count: int
    duration_sec: float = 0.0
    sample_bytes: int = 0


@dataclass
class SampleExtractResult:
    ok: bool
    samples: list[SceneSample]
    total_frames: int
    extract_sec: float = 0.0
    error: str = ""
    concat_path: str = ""
    concat_bytes: int = 0


def ab_av1_sample_count(
    duration_sec: float,
    *,
    samples: Optional[int] = None,
    sample_every_sec: float = 720.0,
    min_samples: int = 1,
) -> int:
    """Mirror ab-av1 ``Sample::sample_count`` (legacy helper)."""
    if samples is not None and int(samples) > 0:
        count = int(samples)
    else:
        every = max(1.0, float(sample_every_sec))
        count = int(math.ceil(float(duration_sec) / every))
    return max(int(min_samples), count, 1)


def ab_av1_sample_start_sec(
    duration_sec: float,
    sample_idx: int,
    sample_count: int,
    sample_duration_sec: float,
) -> float:
    """Mirror ab-av1 ``sample_encode::sample`` start positioning."""
    if sample_count <= 0:
        return 0.0
    sample_n = sample_idx + 1
    gap = max(0.0, duration_sec - sample_duration_sec * sample_count)
    return gap / float(sample_count + 1) * sample_n + sample_duration_sec * sample_idx


def select_scenes_for_sampling(
    scenes: list[SceneSpan],
    *,
    max_scenes: int,
) -> list[SceneSpan]:
    """Legacy helper: keep up to ``max_scenes`` scenes, spread across timeline."""
    if not scenes or max_scenes <= 0:
        return []
    if len(scenes) <= max_scenes:
        return list(scenes)
    if max_scenes == 1:
        mid = len(scenes) // 2
        return [scenes[mid]]
    step = (len(scenes) - 1) / float(max_scenes - 1)
    picked: list[SceneSpan] = []
    used: set[int] = set()
    for i in range(max_scenes):
        idx = int(round(i * step))
        idx = max(0, min(len(scenes) - 1, idx))
        if idx in used:
            continue
        used.add(idx)
        picked.append(scenes[idx])
    return picked


def resolve_sample_plan(
    duration_sec: float,
    scene_count: int,
    *,
    sample_seconds: float = 3.0,
    fps: float = 30.0,
    samples_override: Optional[int] = None,
    sample_every_sec: float = 720.0,
    min_samples: int = 1,
    scene_max_samples: int = 12,
) -> tuple[int, float, int]:
    """Legacy helper. Fleet path samples every scene instead."""
    desired = ab_av1_sample_count(
        duration_sec,
        samples=samples_override,
        sample_every_sec=sample_every_sec,
        min_samples=min_samples,
    )
    count = min(int(scene_max_samples), desired, max(1, scene_count))
    fps_v = max(1.0, float(fps))
    seconds = max(1.0 / fps_v, float(sample_seconds))
    frames = max(1, int(round(seconds * fps_v)))
    if seconds * count >= duration_sec * 0.85:
        count = 1
    return count, seconds, frames


def concat_scene_samples(
    samples: list[SceneSample],
    output_path: str,
    *,
    ffmpeg_bin: Optional[str] = None,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> tuple[bool, str, int]:
    """Concatenate scene clips into one VMAF reference. Returns (ok, path, bytes)."""
    if not samples:
        return False, "", 0
    if len(samples) == 1:
        path = samples[0].path
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        return True, path, size

    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.parent / f"{out.stem}_concat.txt"
    list_file.write_text(
        "".join(f"file '{Path(s.path).resolve()}'\n" for s in samples),
        encoding="utf-8",
    )

    def command_timeout() -> Optional[float]:
        if deadline is None:
            return timeout
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired("sample_concat", 0)
        return min(timeout, left) if timeout is not None else left

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-an",
            str(out),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=command_timeout()
        )
        if result.returncode != 0 or not out.is_file():
            return False, (result.stderr or "")[-1200:] or "concat failed", 0
        return True, str(out.resolve()), out.stat().st_size
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout: {exc}", 0
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except OSError:
            pass


def extract_scene_samples(
    input_path: str,
    scenes: list[SceneSpan],
    output_dir: str,
    *,
    sample_seconds: float = 3.0,
    floor_start_to_sec: bool = True,
    ffmpeg_bin: Optional[str] = None,
    ffprobe_bin: Optional[str] = None,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    # Legacy kwargs ignored (kept so old call sites do not break).
    max_scenes: int = 12,
    samples_override: Optional[int] = None,
    sample_every_sec: float = 720.0,
    min_samples: int = 1,
) -> SampleExtractResult:
    """Extract a fixed-duration midpoint clip from every scene, then concat.

    No scene picking / min-sample heuristics — every input scene is sampled.
    The concat mashup is the CRF-search VMAF reference.
    """
    del max_scenes, samples_override, sample_every_sec, min_samples
    started = time.monotonic()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scenes:
        return SampleExtractResult(
            ok=False,
            samples=[],
            total_frames=0,
            extract_sec=time.monotonic() - started,
            error="no scenes to sample",
        )

    try:
        info = _video_info(input_path, ffprobe_bin, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return SampleExtractResult(
            ok=False,
            samples=[],
            total_frames=0,
            extract_sec=time.monotonic() - started,
            error=str(exc),
        )

    fps = max(1.0, float(info.get("fps") or 30.0))
    sample_dur_sec = max(1.0 / fps, float(sample_seconds))
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
    samples: list[SceneSample] = []
    floor_sec = floor_start_to_sec and sample_dur_sec >= 2.0

    def command_timeout() -> Optional[float]:
        if deadline is None:
            return timeout
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired("sample_extract", 0)
        return min(timeout, left) if timeout is not None else left

    try:
        for scene in scenes:
            take_sec = min(sample_dur_sec, max(scene.duration, 1.0 / fps))
            take_frames = max(1, int(round(take_sec * fps)))
            center = scene.start_sec + scene.duration / 2.0
            start_sec = max(scene.start_sec, center - take_sec / 2.0)
            end_limit = max(scene.start_sec, scene.end_sec - take_sec)
            start_sec = min(start_sec, end_limit)
            if floor_sec:
                start_sec = float(int(start_sec))

            out_path = out_dir / f"scene_{scene.index:03d}_{take_sec:.1f}s.mkv"
            # Re-encode to a clean yuv420p clip (not -c:v copy). Stream-copy
            # mid-GOP cuts produce flaky VMAF references on hard 4K content.
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-ss",
                f"{start_sec:.3f}",
                "-i",
                input_path,
                "-t",
                f"{take_sec:.3f}",
                "-vf",
                "setsar=1,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-sn",
                str(out_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=command_timeout()
            )
            if result.returncode != 0 or not out_path.is_file():
                return SampleExtractResult(
                    ok=False,
                    samples=samples,
                    total_frames=sum(s.frame_count for s in samples),
                    extract_sec=time.monotonic() - started,
                    error=(result.stderr or "")[-1200:]
                    or f"failed scene {scene.index}",
                )
            size = out_path.stat().st_size
            if size <= 1024:
                return SampleExtractResult(
                    ok=False,
                    samples=samples,
                    total_frames=sum(s.frame_count for s in samples),
                    extract_sec=time.monotonic() - started,
                    error=f"sample too small for scene {scene.index}",
                )
            samples.append(
                SceneSample(
                    scene_index=scene.index,
                    path=str(out_path.resolve()),
                    start_sec=start_sec,
                    frame_count=take_frames,
                    duration_sec=take_sec,
                    sample_bytes=size,
                )
            )
    except subprocess.TimeoutExpired as exc:
        return SampleExtractResult(
            ok=False,
            samples=samples,
            total_frames=sum(s.frame_count for s in samples),
            extract_sec=time.monotonic() - started,
            error=f"timeout: {exc}",
        )

    concat_path = str((out_dir / "probe_reference.mkv").resolve())
    ok_concat, concat_or_err, concat_bytes = concat_scene_samples(
        samples,
        concat_path,
        ffmpeg_bin=ffmpeg_bin,
        timeout=timeout,
        deadline=deadline,
    )
    if not ok_concat:
        return SampleExtractResult(
            ok=False,
            samples=samples,
            total_frames=sum(s.frame_count for s in samples),
            extract_sec=time.monotonic() - started,
            error=f"concat failed: {concat_or_err}",
        )

    total_frames = sum(s.frame_count for s in samples)
    total_sec = sum(s.duration_sec for s in samples)
    log(
        f"  scene samples: {len(samples)} clip(s) → concat, "
        f"{sample_dur_sec:.1f}s/scene, {total_sec:.1f}s total "
        f"({total_frames} frames) in {time.monotonic() - started:.1f}s"
    )
    return SampleExtractResult(
        ok=True,
        samples=samples,
        total_frames=total_frames,
        extract_sec=time.monotonic() - started,
        concat_path=concat_or_err,
        concat_bytes=concat_bytes,
    )
