"""Segment-aware proxy builder for two-phase CRF search.

Builds a short mashup proxy by taking ~2.5s from the middle of each segment,
optionally capped to ``proxy_max_seconds``. The proxy is a lossless yuv420p
re-encode so it is a clean VMAF reference; compression_rate is estimated via
bitrate ratio against the full source.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ffmpeg_tools import resolve_binary


@dataclass(frozen=True)
class ProxyWindow:
    segment_index: int
    start_sec: float
    duration_sec: float
    difficulty: float


@dataclass
class ProxyBuildResult:
    ok: bool
    path: str
    windows: list[ProxyWindow]
    total_seconds: float
    error: str = ""


def select_proxy_windows(
    segments: list[dict[str, Any]],
    *,
    seconds_per_segment: float = 2.5,
    max_seconds: float = 15.0,
    min_window_seconds: float = 0.5,
) -> list[ProxyWindow]:
    """Pick a mid-segment window from each segment, then cap total duration.

    When the sum exceeds ``max_seconds``, keep the hardest segments first and
    scale remaining window lengths so the total fits the budget.
    """
    if not segments:
        return []

    raw: list[ProxyWindow] = []
    for seg in segments:
        try:
            idx = int(seg.get("index", len(raw)))
            start = float(seg["start_sec"])
            end = float(seg["end_sec"])
            seg_dur = max(0.0, end - start)
            difficulty = float(seg.get("difficulty", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        if seg_dur < min_window_seconds:
            # Tiny segments: take the whole thing if it has any duration.
            if seg_dur <= 1e-3:
                continue
            win_dur = seg_dur
            win_start = start
        else:
            win_dur = min(seconds_per_segment, seg_dur)
            # Center the window inside the segment, stay clear of cut edges.
            margin = max(0.0, (seg_dur - win_dur) / 2.0)
            win_start = start + margin

        raw.append(
            ProxyWindow(
                segment_index=idx,
                start_sec=win_start,
                duration_sec=win_dur,
                difficulty=difficulty,
            )
        )

    if not raw:
        return []

    total = sum(w.duration_sec for w in raw)
    if total <= max_seconds:
        return sorted(raw, key=lambda w: w.start_sec)

    # Prefer hard segments when trimming to the duration budget.
    ranked = sorted(raw, key=lambda w: w.difficulty, reverse=True)
    kept: list[ProxyWindow] = []
    used = 0.0
    for win in ranked:
        remaining = max_seconds - used
        if remaining < min_window_seconds:
            break
        take = min(win.duration_sec, remaining)
        if take < min_window_seconds and win.duration_sec >= min_window_seconds:
            continue
        # Keep the same mid-point centering when shortening.
        shrink = win.duration_sec - take
        new_start = win.start_sec + shrink / 2.0
        kept.append(
            ProxyWindow(
                segment_index=win.segment_index,
                start_sec=new_start,
                duration_sec=take,
                difficulty=win.difficulty,
            )
        )
        used += take

    return sorted(kept, key=lambda w: w.start_sec)


def build_proxy_reference(
    input_path: str,
    output_path: str,
    windows: list[ProxyWindow],
    *,
    ffmpeg_bin: Optional[str] = None,
    timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> ProxyBuildResult:
    """Lossless yuv420p concat of the selected windows (VMAF-safe reference)."""
    if not windows:
        return ProxyBuildResult(
            ok=False,
            path=output_path,
            windows=[],
            total_seconds=0.0,
            error="no proxy windows",
        )

    ffmpeg = resolve_binary("ffmpeg", ffmpeg_bin)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / f"{out.stem}_parts"
    work.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []

    def command_timeout() -> Optional[float]:
        if deadline is None:
            return timeout
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired("proxy", 0)
        return min(timeout, left) if timeout is not None else left

    try:
        for i, win in enumerate(windows):
            part = work / f"part_{i:03d}.mp4"
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-ss",
                f"{win.start_sec:.3f}",
                "-i",
                input_path,
                "-t",
                f"{win.duration_sec:.3f}",
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
                "-movflags",
                "+faststart",
                str(part),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=command_timeout()
            )
            if result.returncode != 0 or not part.is_file():
                return ProxyBuildResult(
                    ok=False,
                    path=output_path,
                    windows=windows,
                    total_seconds=sum(w.duration_sec for w in windows),
                    error=(result.stderr or "")[-1500:] or f"failed to extract part {i}",
                )
            part_paths.append(part)

        list_file = work / "concat.txt"
        list_file.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in part_paths),
            encoding="utf-8",
        )
        concat_cmd = [
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
            "-movflags",
            "+faststart",
            str(out),
        ]
        result = subprocess.run(
            concat_cmd, capture_output=True, text=True, timeout=command_timeout()
        )
        if result.returncode != 0 or not out.is_file():
            return ProxyBuildResult(
                ok=False,
                path=output_path,
                windows=windows,
                total_seconds=sum(w.duration_sec for w in windows),
                error=(result.stderr or "")[-1500:] or "concat failed",
            )
    except subprocess.TimeoutExpired as exc:
        return ProxyBuildResult(
            ok=False,
            path=output_path,
            windows=windows,
            total_seconds=sum(w.duration_sec for w in windows),
            error=f"timeout after {timeout}s: {exc}",
        )
    finally:
        # Best-effort cleanup of intermediate parts; keep final proxy.
        for part in part_paths:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            list_path = work / "concat.txt"
            list_path.unlink(missing_ok=True)
            work.rmdir()
        except OSError:
            pass

    return ProxyBuildResult(
        ok=True,
        path=str(out.resolve()),
        windows=windows,
        total_seconds=sum(w.duration_sec for w in windows),
    )
