"""Segment-aware video feature extraction for interleaved mashup clips.

Pipeline:
  1) One sequential decode pass: detect cuts + cache downscaled gray/sat
  2) Extract per-segment features from the cache (no MP4 seeks)
  3) Build clip-level summary (cut_rate, worst/hard difficulty, etc.)

`extract()` returns a flat summary dict compatible with recipes / search.
`extract_full()` returns {segments, summary, meta}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


@dataclass
class SegmentFeatures:
    index: int
    start_sec: float
    end_sec: float
    duration: float
    frame_count: int

    motion_mean: float
    motion_std: float
    motion_p90: float
    motion_max: float

    texture: float
    texture_std: float
    edge_density: float
    entropy: float
    high_freq_energy: float

    noise_level: float
    flatness: float

    luma_mean: float
    luma_std: float
    sat_mean: float

    difficulty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _CachedFrame:
    index: int
    gray: np.ndarray  # uint8 HxW
    sat: np.ndarray  # uint8 HxW


class HEVCFeatureExtractor:
    """Mashup-aware extractor: segment first, then summarize."""

    def __init__(
        self,
        video_path: str,
        *,
        # Cut detection
        analysis_scale: float = 0.25,
        analysis_fps: float = 10.0,
        hist_bins: int = 16,
        cut_hist_threshold: float = 0.45,
        cut_diff_threshold: float = 0.18,
        min_segment_sec: float = 0.40,
        # Per-segment sampling
        samples_per_segment: int = 8,
        max_segments: int = 64,
        # Difficulty weights
        hard_difficulty_threshold: float = 0.45,
    ):
        self.video_path = video_path
        self.analysis_scale = analysis_scale
        self.analysis_fps = analysis_fps
        self.hist_bins = hist_bins
        self.cut_hist_threshold = cut_hist_threshold
        self.cut_diff_threshold = cut_diff_threshold
        self.min_segment_sec = min_segment_sec
        self.samples_per_segment = samples_per_segment
        self.max_segments = max_segments
        self.hard_difficulty_threshold = hard_difficulty_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> dict[str, float]:
        """Flat clip summary used by recipes / search."""
        return self.extract_full()["summary"]

    def extract_full(self) -> dict[str, Any]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 1e-6:
            fps = 30.0
        duration = frame_count / fps if frame_count > 0 else 0.0

        meta = {
            "path": self.video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

        cut_frames, cache, decoded = self._scan_cuts_and_cache(cap, fps)
        cap.release()

        if decoded <= 1:
            empty = self._empty_summary()
            return {"meta": meta, "segments": [], "summary": empty, "cut_times_sec": []}

        # Prefer actual decoded length when CAP_PROP_FRAME_COUNT is wrong/missing
        if frame_count <= 1 or abs(decoded - frame_count) > max(2, int(0.02 * max(frame_count, 1))):
            frame_count = decoded
            meta["frame_count"] = frame_count
            meta["duration"] = frame_count / fps
            duration = meta["duration"]

        segments_idx = self._frames_to_segments(cut_frames, frame_count, fps)
        if len(segments_idx) > self.max_segments:
            segments_idx = self._limit_segments(segments_idx, self.max_segments)

        segment_feats: list[SegmentFeatures] = []
        for i, (start_f, end_f) in enumerate(segments_idx):
            seg = self._analyze_segment_from_cache(cache, i, start_f, end_f, fps)
            segment_feats.append(seg)

        summary = self._summarize(segment_feats, duration, fps)
        return {
            "meta": meta,
            "cut_times_sec": [float(f / fps) for f in cut_frames if 0 < f < frame_count],
            "segments": [s.to_dict() for s in segment_feats],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Sequential scan: cuts + frame cache
    # ------------------------------------------------------------------

    def _scan_cuts_and_cache(
        self,
        cap: cv2.VideoCapture,
        fps: float,
    ) -> tuple[list[int], list[_CachedFrame], int]:
        """Forward-only decode. Sample every `step` frames for cuts + cache.

        Returns (cut_frames, cache, decoded_frame_count).
        """
        step = max(1, int(round(fps / max(self.analysis_fps, 1e-6))))
        scale = self.analysis_scale

        prev_gray: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        cut_frames: list[int] = []
        cache: list[_CachedFrame] = []

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % step == 0:
                if scale < 0.999:
                    small = cv2.resize(
                        frame,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = frame

                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                sat = hsv[:, :, 1]

                hist = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_gray is not None and prev_hist is not None:
                    diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                    hist_delta = float(np.linalg.norm(hist - prev_hist, ord=1) / 2.0)
                    if hist_delta >= self.cut_hist_threshold or diff >= self.cut_diff_threshold:
                        cut_frames.append(int(idx))

                prev_gray = gray
                prev_hist = hist
                cache.append(_CachedFrame(index=idx, gray=gray, sat=sat))

            idx += 1

        return cut_frames, cache, idx

    def _frames_to_segments(
        self,
        cut_frames: list[int],
        frame_count: int,
        fps: float,
    ) -> list[tuple[int, int]]:
        bounds = [0] + sorted(set(cut_frames)) + [frame_count]
        min_frames = max(1, int(round(self.min_segment_sec * fps)))

        raw: list[tuple[int, int]] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 1:
                raw.append((a, b))

        if not raw:
            return [(0, frame_count)]

        # Merge segments that are too short into the previous one
        merged: list[tuple[int, int]] = [raw[0]]
        for start, end in raw[1:]:
            prev_start, prev_end = merged[-1]
            if (end - start) < min_frames:
                merged[-1] = (prev_start, end)
            elif (prev_end - prev_start) < min_frames:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        # Final pass: if first segment still tiny, merge forward
        if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_frames:
            merged[1] = (merged[0][0], merged[1][1])
            merged = merged[1:]

        return merged

    @staticmethod
    def _limit_segments(
        segments: list[tuple[int, int]],
        max_segments: int,
    ) -> list[tuple[int, int]]:
        if len(segments) <= max_segments:
            return segments
        # Evenly merge adjacent segments
        n = len(segments)
        group = int(np.ceil(n / max_segments))
        out: list[tuple[int, int]] = []
        for i in range(0, n, group):
            chunk = segments[i : i + group]
            out.append((chunk[0][0], chunk[-1][1]))
        return out

    # ------------------------------------------------------------------
    # Per-segment features (from cache)
    # ------------------------------------------------------------------

    def _analyze_segment_from_cache(
        self,
        cache: list[_CachedFrame],
        index: int,
        start_f: int,
        end_f: int,
        fps: float,
    ) -> SegmentFeatures:
        n_frames = max(1, end_f - start_f)
        start_sec = start_f / fps
        end_sec = end_f / fps
        duration = end_sec - start_sec

        in_seg = [c for c in cache if start_f <= c.index < end_f]
        if not in_seg and cache:
            # Fallback: nearest cached frame to segment midpoint
            mid = (start_f + end_f - 1) // 2
            nearest = min(cache, key=lambda c: abs(c.index - mid))
            in_seg = [nearest]

        sample_n = min(self.samples_per_segment, len(in_seg)) if in_seg else 0
        if sample_n <= 0:
            return self._empty_segment(index, start_sec, end_sec, duration, n_frames)

        if sample_n == len(in_seg):
            samples = in_seg
        else:
            pick = np.linspace(0, len(in_seg) - 1, sample_n).astype(int)
            samples = [in_seg[i] for i in pick]

        return self._features_from_samples(
            samples,
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            duration=duration,
            n_frames=n_frames,
        )

    def _features_from_samples(
        self,
        samples: list[_CachedFrame],
        *,
        index: int,
        start_sec: float,
        end_sec: float,
        duration: float,
        n_frames: int,
    ) -> SegmentFeatures:
        motion_vals: list[float] = []
        texture_vals: list[float] = []
        edge_vals: list[float] = []
        entropy_vals: list[float] = []
        hf_vals: list[float] = []
        noise_vals: list[float] = []
        flat_vals: list[float] = []
        luma_vals: list[float] = []
        luma_std_vals: list[float] = []
        sat_vals: list[float] = []

        prev_gray: Optional[np.ndarray] = None

        for sample in samples:
            gray = sample.gray
            sat = sample.sat

            if prev_gray is not None:
                motion_vals.append(float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0))
            prev_gray = gray

            # Texture: LBP variance normalized but NOT hard-capped at 1 for discrimination
            lbp = local_binary_pattern(gray, 8, 1, method="uniform")
            texture_vals.append(float(np.var(lbp) / 20.0))

            edges = cv2.Canny(gray, 100, 200)
            edge_vals.append(float(np.mean(edges > 0)))

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_vals.append(float(entropy / 8.0))

            # High-frequency energy via Laplacian variance (cheap proxy for DCT HF)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            hf_vals.append(float(min(np.var(lap) / 500.0, 5.0)))

            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            residual = gray.astype(np.float64) - blur.astype(np.float64)
            noise_vals.append(float(np.std(residual) / 255.0))

            # Flatness: fraction of low-gradient pixels
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            flat_vals.append(float(np.mean(mag < 15.0)))

            luma_vals.append(float(np.mean(gray) / 255.0))
            luma_std_vals.append(float(np.std(gray) / 255.0))
            sat_vals.append(float(np.mean(sat) / 255.0))

        motion_arr = np.asarray(motion_vals, dtype=np.float64) if motion_vals else np.array([0.0])
        texture_arr = np.asarray(texture_vals, dtype=np.float64) if texture_vals else np.array([0.0])

        motion_mean = float(np.mean(motion_arr))
        motion_std = float(np.std(motion_arr)) if len(motion_arr) > 1 else 0.0
        motion_p90 = float(np.percentile(motion_arr, 90))
        motion_max = float(np.max(motion_arr))

        texture = float(np.mean(texture_arr))
        texture_std = float(np.std(texture_arr)) if len(texture_arr) > 1 else 0.0
        edge_density = float(np.mean(edge_vals)) if edge_vals else 0.0
        entropy = float(np.mean(entropy_vals)) if entropy_vals else 0.0
        high_freq_energy = float(np.mean(hf_vals)) if hf_vals else 0.0
        noise_level = float(np.mean(noise_vals)) if noise_vals else 0.0
        flatness = float(np.mean(flat_vals)) if flat_vals else 0.0
        luma_mean = float(np.mean(luma_vals)) if luma_vals else 0.0
        luma_std = float(np.mean(luma_std_vals)) if luma_std_vals else 0.0
        sat_mean = float(np.mean(sat_vals)) if sat_vals else 0.0

        difficulty = self._segment_difficulty(
            motion_p90=motion_p90,
            texture=texture,
            entropy=entropy,
            noise_level=noise_level,
            edge_density=edge_density,
            high_freq_energy=high_freq_energy,
            flatness=flatness,
        )

        return SegmentFeatures(
            index=index,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration=float(duration),
            frame_count=int(n_frames),
            motion_mean=motion_mean,
            motion_std=motion_std,
            motion_p90=motion_p90,
            motion_max=motion_max,
            texture=texture,
            texture_std=texture_std,
            edge_density=edge_density,
            entropy=entropy,
            high_freq_energy=high_freq_energy,
            noise_level=noise_level,
            flatness=flatness,
            luma_mean=luma_mean,
            luma_std=luma_std,
            sat_mean=sat_mean,
            difficulty=difficulty,
        )

    def _empty_segment(
        self,
        index: int,
        start_sec: float,
        end_sec: float,
        duration: float,
        n_frames: int,
    ) -> SegmentFeatures:
        return SegmentFeatures(
            index=index,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration=float(duration),
            frame_count=int(n_frames),
            motion_mean=0.0,
            motion_std=0.0,
            motion_p90=0.0,
            motion_max=0.0,
            texture=0.0,
            texture_std=0.0,
            edge_density=0.0,
            entropy=0.0,
            high_freq_energy=0.0,
            noise_level=0.0,
            flatness=0.0,
            luma_mean=0.0,
            luma_std=0.0,
            sat_mean=0.0,
            difficulty=0.0,
        )

    @staticmethod
    def _segment_difficulty(
        *,
        motion_p90: float,
        texture: float,
        entropy: float,
        noise_level: float,
        edge_density: float,
        high_freq_energy: float,
        flatness: float,
    ) -> float:
        # Normalize roughly into [0, 1] contributions
        m = min(motion_p90 / 0.25, 1.0)
        t = min(texture / 2.0, 1.0)
        e = min(entropy, 1.0)
        n = min(noise_level / 0.08, 1.0)
        ed = min(edge_density / 0.15, 1.0)
        hf = min(high_freq_energy / 2.0, 1.0)
        # Flatness reduces difficulty
        flat_penalty = 1.0 - min(max(flatness, 0.0), 1.0)

        score = (
            0.25 * m
            + 0.20 * t
            + 0.15 * e
            + 0.15 * n
            + 0.10 * ed
            + 0.10 * hf
            + 0.05 * flat_penalty
        )
        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Clip summary
    # ------------------------------------------------------------------

    def _summarize(
        self,
        segments: list[SegmentFeatures],
        duration: float,
        fps: float,
    ) -> dict[str, float]:
        if not segments:
            return self._empty_summary()

        durations = np.asarray([s.duration for s in segments], dtype=np.float64)
        diffs = np.asarray([s.difficulty for s in segments], dtype=np.float64)
        total_dur = float(np.sum(durations)) if float(np.sum(durations)) > 0 else 1.0
        weights = durations / total_dur

        def wmean(vals: list[float]) -> float:
            arr = np.asarray(vals, dtype=np.float64)
            return float(np.sum(arr * weights))

        cut_count = max(0, len(segments) - 1)
        cut_rate = float(cut_count / max(duration, 1e-6))  # cuts per second

        hard_mask = diffs >= self.hard_difficulty_threshold
        hard_fraction = float(np.sum(durations[hard_mask]) / total_dur)
        worst_difficulty = float(np.max(diffs))
        difficulty_mean = float(np.mean(diffs))
        difficulty_p90 = float(np.percentile(diffs, 90))
        duration_weighted_difficulty = float(np.sum(diffs * weights))

        motion_mean = wmean([s.motion_mean for s in segments])
        motion_std = float(np.std([s.motion_mean for s in segments])) if len(segments) > 1 else 0.0
        motion_p90 = float(np.max([s.motion_p90 for s in segments]))
        texture_mean = wmean([s.texture for s in segments])
        texture_std = float(np.std([s.texture for s in segments])) if len(segments) > 1 else 0.0
        entropy = wmean([s.entropy for s in segments])
        edge_density = wmean([s.edge_density for s in segments])
        noise_level = wmean([s.noise_level for s in segments])
        high_freq_energy = wmean([s.high_freq_energy for s in segments])
        flatness = wmean([s.flatness for s in segments])
        luma_mean = wmean([s.luma_mean for s in segments])
        luma_std = wmean([s.luma_std for s in segments])
        sat_mean = wmean([s.sat_mean for s in segments])

        # Legacy-compatible mashup signals for recipes
        cut_density = min(cut_rate / 0.5, 1.0)  # ~0.5 cuts/sec -> 1.0
        volatility = float(
            min(
                1.0,
                0.35 * min(motion_std / 0.08, 1.0)
                + 0.25 * min(texture_std / 0.5, 1.0)
                + 0.25 * min(cut_density, 1.0)
                + 0.15 * worst_difficulty,
            )
        )

        return {
            # Segment-aware summary (primary)
            "segment_count": float(len(segments)),
            "cut_count": float(cut_count),
            "cut_rate": cut_rate,
            "hard_fraction": hard_fraction,
            "worst_difficulty": worst_difficulty,
            "difficulty_mean": difficulty_mean,
            "difficulty_p90": difficulty_p90,
            "duration_weighted_difficulty": duration_weighted_difficulty,
            # Content aggregates (duration-weighted)
            "motion_mean": motion_mean,
            "motion_std": motion_std,
            "motion_p90": motion_p90,
            "texture": texture_mean,
            "texture_lbp": texture_mean,  # legacy alias for recipes
            "texture_std": texture_std,
            "entropy": entropy,
            "edge_density": edge_density,
            "noise_level": noise_level,
            "high_freq_energy": high_freq_energy,
            "flatness": flatness,
            "luma_mean": luma_mean,
            "luma_std": luma_std,
            "sat_mean": sat_mean,
            # Mashup flags (legacy + new)
            "cut_density": cut_density,
            "volatility": volatility,
            "duration": float(duration),
            "fps": float(fps),
        }

    @staticmethod
    def _empty_summary() -> dict[str, float]:
        keys = [
            "segment_count",
            "cut_count",
            "cut_rate",
            "hard_fraction",
            "worst_difficulty",
            "difficulty_mean",
            "difficulty_p90",
            "duration_weighted_difficulty",
            "motion_mean",
            "motion_std",
            "motion_p90",
            "texture",
            "texture_lbp",
            "texture_std",
            "entropy",
            "edge_density",
            "noise_level",
            "high_freq_energy",
            "flatness",
            "luma_mean",
            "luma_std",
            "sat_mean",
            "cut_density",
            "volatility",
            "duration",
            "fps",
        ]
        return {k: 0.0 for k in keys}
