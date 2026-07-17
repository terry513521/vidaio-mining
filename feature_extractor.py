"""Segment-aware video feature extraction for interleaved mashup clips.

Pipeline:
  1) One sequential decode pass: detect cuts + cache downscaled frames
  2) Per-frame motion (consecutive frames); noise/texture on detail-scale crops
  3) Per-segment features from the cache (no MP4 seeks)
  4) Clip-level summary (cut_rate, worst/hard difficulty, etc.)

`extract()` returns a flat summary dict compatible with recipes / search.
`extract_full()` returns {segments, summary, meta}.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional
import time

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# Reference short side for resolution-normalized detail analysis (~540p).
_DETAIL_SHORT_SIDE = 540.0

# Soft-norm midpoints (level = x / (x + mid) → mid maps to 0.5).
# Calibrated on s3_videos/compression (n=30, p50) via calibrate_feature_refs.py.
# Soft curve avoids hard-clamp saturation that pinned every mashup at 1.0.
_MOTION_P90_MID = 0.145
_NOISE_LEVEL_MID = 0.048
_TEXTURE_MID = 4.22
_EDGE_DENSITY_MID = 0.0475
_HF_ENERGY_MID = 4.99

# Backward-compatible aliases (older hard-clamp refs). Prefer *_MID + soft_level().
_MOTION_P90_REF = _MOTION_P90_MID
_NOISE_LEVEL_REF = _NOISE_LEVEL_MID
_TEXTURE_REF = _TEXTURE_MID
_EDGE_DENSITY_REF = _EDGE_DENSITY_MID
_HF_ENERGY_REF = _HF_ENERGY_MID


def soft_level(value: float, mid: float) -> float:
    """Saturating 0..1 map: ``mid`` → 0.5, asymptote → 1 (no hard clamp)."""
    x = max(0.0, float(value))
    m = float(mid)
    if m <= 0.0:
        return 0.0
    return float(x / (x + m))

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
    chroma_std: float

    difficulty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _CachedFrame:
    index: int
    gray: np.ndarray  # uint8 HxW — cut / coarse analysis
    sat: np.ndarray  # uint8 HxW
    detail_gray: np.ndarray  # uint8 — detail-scale crop for texture/noise
    motion_inst: float  # max consecutive-frame motion in the cache window
    noise_level: float  # temporal MAD in flat regions (detail scale)


class HEVCFeatureExtractor:
    """Mashup-aware extractor: segment first, then summarize."""

    def __init__(
        self,
        video_path: str,
        *,
        # Cut detection (coarse scale)
        analysis_scale: float = 0.25,
        analysis_fps: float = 10.0,
        hist_bins: int = 16,
        cut_hist_threshold: float = 0.45,
        cut_diff_threshold: float = 0.18,
        min_segment_sec: float = 0.40,
        # Detail analysis: short side scaled toward _DETAIL_SHORT_SIDE px
        detail_short_side: float = _DETAIL_SHORT_SIDE,
        # Per-segment sampling
        samples_per_segment: int = 8,
        max_segments: int = 64,
        hard_difficulty_threshold: float = 0.45,
    ):
        self.video_path = video_path
        self.analysis_scale = analysis_scale
        self.analysis_fps = analysis_fps
        self.hist_bins = hist_bins
        self.cut_hist_threshold = cut_hist_threshold
        self.cut_diff_threshold = cut_diff_threshold
        self.min_segment_sec = min_segment_sec
        self.detail_short_side = detail_short_side
        self.samples_per_segment = samples_per_segment
        self.max_segments = max_segments
        self.hard_difficulty_threshold = hard_difficulty_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> dict[str, float]:
        """Flat clip summary used by recipes / search."""
        return self.extract_full()["summary"]

    def extract_full(
        self,
        *,
        deadline: Optional[float] = None,
        scene_spans: Optional[list[tuple[float, float]]] = None,
    ) -> dict[str, Any]:
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

        detail_scale = self._detail_scale(width, height)
        texture_comp = self._texture_compensation(detail_scale)

        meta = {
            "path": self.video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "detail_scale": detail_scale,
            "texture_compensation": texture_comp,
            "scene_spans_provided": bool(scene_spans),
        }

        cut_frames, cache, decoded = self._scan_cuts_and_cache(
            cap, fps, width, height, detail_scale, deadline=deadline
        )
        cap.release()

        if decoded <= 1:
            empty = self._empty_summary()
            return {"meta": meta, "segments": [], "summary": empty, "cut_times_sec": []}

        if frame_count <= 1 or abs(decoded - frame_count) > max(
            2, int(0.02 * max(frame_count, 1))
        ):
            frame_count = decoded
            meta["frame_count"] = frame_count
            meta["duration"] = frame_count / fps
            duration = meta["duration"]

        if scene_spans:
            segments_idx = self._spans_to_segments(scene_spans, frame_count, fps)
            cut_times = [float(s) for s, _e in scene_spans[1:]]
        else:
            segments_idx = self._frames_to_segments(cut_frames, frame_count, fps)
            cut_times = [float(f / fps) for f in cut_frames if 0 < f < frame_count]

        if len(segments_idx) > self.max_segments:
            segments_idx = self._limit_segments(segments_idx, self.max_segments)

        segment_feats: list[SegmentFeatures] = []
        for i, (start_f, end_f) in enumerate(segments_idx):
            seg = self._analyze_segment_from_cache(
                cache, i, start_f, end_f, fps, texture_comp
            )
            segment_feats.append(seg)

        summary = self._summarize(segment_feats, duration, fps, width, height)
        return {
            "meta": meta,
            "cut_times_sec": cut_times,
            "segments": [s.to_dict() for s in segment_feats],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------

    def _detail_scale(self, width: int, height: int) -> float:
        short = min(max(width, 1), max(height, 1))
        return min(1.0, self.detail_short_side / float(short))

    @staticmethod
    def _texture_compensation(detail_scale: float) -> float:
        # Texture variance shrinks ~scale^2 when downsampling.
        return 1.0 / max(detail_scale * detail_scale, 1e-6)

    @staticmethod
    def _resize_gray(frame: np.ndarray, scale: float) -> np.ndarray:
        if scale >= 0.999:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _estimate_noise_level(
        detail_gray: np.ndarray,
        prev_detail: Optional[np.ndarray],
    ) -> float:
        """Temporal MAD in flat (low-gradient) regions at detail scale."""
        gx = cv2.Sobel(detail_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(detail_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        flat_mask = mag < 20.0

        if prev_detail is not None and np.count_nonzero(flat_mask) > 64:
            td = np.abs(
                detail_gray.astype(np.float32) - prev_detail.astype(np.float32)
            )
            vals = td[flat_mask]
            return float(np.median(vals) / 255.0 * 1.4826)

        blur = cv2.GaussianBlur(detail_gray, (5, 5), 0)
        residual = np.abs(detail_gray.astype(np.float32) - blur.astype(np.float32))
        if np.count_nonzero(flat_mask) > 64:
            vals = residual[flat_mask]
            return float(np.median(vals) / 255.0 * 1.4826)
        return float(np.median(residual) / 255.0 * 1.4826)

    # ------------------------------------------------------------------
    # Sequential scan: cuts + frame cache
    # ------------------------------------------------------------------

    def _scan_cuts_and_cache(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        width: int,
        height: int,
        detail_scale: float,
        *,
        deadline: Optional[float] = None,
    ) -> tuple[list[int], list[_CachedFrame], int]:
        """Forward-only decode. Per-frame motion; cache at analysis_fps."""
        step = max(1, int(round(fps / max(self.analysis_fps, 1e-6))))
        coarse_scale = self.analysis_scale

        prev_coarse: Optional[np.ndarray] = None
        prev_hist: Optional[np.ndarray] = None
        prev_detail: Optional[np.ndarray] = None
        prev_motion_gray: Optional[np.ndarray] = None

        cut_frames: list[int] = []
        cache: list[_CachedFrame] = []
        motion_window: list[float] = []

        idx = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            ok, frame = cap.read()
            if not ok:
                break

            coarse = self._resize_gray(frame, coarse_scale)
            if prev_motion_gray is not None:
                inst = float(np.mean(cv2.absdiff(coarse, prev_motion_gray)) / 255.0)
                motion_window.append(inst)
            prev_motion_gray = coarse

            if idx % step == 0:
                if prev_coarse is not None and prev_hist is not None:
                    diff = float(np.mean(cv2.absdiff(coarse, prev_coarse)) / 255.0)
                    hist = cv2.calcHist([coarse], [0], None, [self.hist_bins], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    hist_delta = float(np.linalg.norm(hist - prev_hist, ord=1) / 2.0)
                    if (
                        hist_delta >= self.cut_hist_threshold
                        or diff >= self.cut_diff_threshold
                    ):
                        cut_frames.append(int(idx))
                else:
                    hist = cv2.calcHist([coarse], [0], None, [self.hist_bins], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()

                hsv = cv2.cvtColor(
                    cv2.resize(frame, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA)
                    if coarse_scale < 0.999
                    else frame,
                    cv2.COLOR_BGR2HSV,
                )
                sat = hsv[:, :, 1]
                detail = self._resize_gray(frame, detail_scale)
                noise = self._estimate_noise_level(detail, prev_detail)
                prev_detail = detail

                motion_inst = float(max(motion_window)) if motion_window else 0.0
                motion_window.clear()

                cache.append(
                    _CachedFrame(
                        index=idx,
                        gray=coarse,
                        sat=sat,
                        detail_gray=detail,
                        motion_inst=motion_inst,
                        noise_level=noise,
                    )
                )
                prev_coarse = coarse
                prev_hist = hist

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

        merged: list[tuple[int, int]] = [raw[0]]
        for start, end in raw[1:]:
            prev_start, prev_end = merged[-1]
            if (end - start) < min_frames:
                merged[-1] = (prev_start, end)
            elif (prev_end - prev_start) < min_frames:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_frames:
            merged[1] = (merged[0][0], merged[1][1])
            merged = merged[1:]

        return merged

    def _spans_to_segments(
        self,
        scene_spans: list[tuple[float, float]],
        frame_count: int,
        fps: float,
    ) -> list[tuple[int, int]]:
        """Map external scene spans (seconds) onto frame ranges."""
        fps_v = max(1e-6, float(fps))
        out: list[tuple[int, int]] = []
        for start_sec, end_sec in scene_spans:
            start_f = int(max(0, round(float(start_sec) * fps_v)))
            end_f = int(min(frame_count, round(float(end_sec) * fps_v)))
            if end_f <= start_f:
                end_f = min(frame_count, start_f + 1)
            out.append((start_f, end_f))
        return out if out else [(0, max(1, frame_count))]

    @staticmethod
    @staticmethod
    def _limit_segments(
        segments: list[tuple[int, int]],
        max_segments: int,
    ) -> list[tuple[int, int]]:
        if len(segments) <= max_segments:
            return segments
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
        texture_comp: float,
    ) -> SegmentFeatures:
        n_frames = max(1, end_f - start_f)
        start_sec = start_f / fps
        end_sec = end_f / fps
        duration = end_sec - start_sec

        in_seg = [c for c in cache if start_f <= c.index < end_f]
        if not in_seg and cache:
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
            texture_comp=texture_comp,
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
        texture_comp: float,
    ) -> SegmentFeatures:
        motion_vals = [s.motion_inst for s in samples]
        texture_vals: list[float] = []
        edge_vals: list[float] = []
        entropy_vals: list[float] = []
        hf_vals: list[float] = []
        noise_vals = [s.noise_level for s in samples]
        flat_vals: list[float] = []
        luma_vals: list[float] = []
        luma_std_vals: list[float] = []
        sat_vals: list[float] = []
        chroma_std_vals: list[float] = []

        for sample in samples:
            gray = sample.gray
            detail = sample.detail_gray
            sat = sample.sat

            lbp = local_binary_pattern(detail, 8, 1, method="uniform")
            texture_vals.append(float(np.var(lbp) / 20.0 * texture_comp))

            edges = cv2.Canny(detail, 100, 200)
            edge_vals.append(float(np.mean(edges > 0)))

            hist = cv2.calcHist([detail], [0], None, [256], [0, 256])
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_vals.append(float(entropy / 8.0))

            lap = cv2.Laplacian(detail, cv2.CV_64F)
            hf_vals.append(float(min(np.var(lap) / 500.0 * texture_comp, 5.0)))

            gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            flat_vals.append(float(np.mean(mag < 15.0)))

            luma_vals.append(float(np.mean(detail) / 255.0))
            luma_std_vals.append(float(np.std(detail) / 255.0))
            sat_vals.append(float(np.mean(sat) / 255.0))
            chroma_std_vals.append(float(np.std(sat) / 255.0))

        motion_arr = np.asarray(motion_vals, dtype=np.float64)
        texture_arr = np.asarray(texture_vals, dtype=np.float64)

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
        chroma_std = float(np.mean(chroma_std_vals)) if chroma_std_vals else 0.0

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
            chroma_std=chroma_std,
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
            chroma_std=0.0,
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
        m = soft_level(motion_p90, _MOTION_P90_MID)
        t = soft_level(texture, _TEXTURE_MID)
        e = min(entropy, 1.0)
        n = soft_level(noise_level, _NOISE_LEVEL_MID)
        ed = soft_level(edge_density, _EDGE_DENSITY_MID)
        hf = soft_level(high_freq_energy, _HF_ENERGY_MID)
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
        width: int,
        height: int,
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
        cut_rate = float(cut_count / max(duration, 1e-6))

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
        chroma_std = wmean([s.chroma_std for s in segments])

        cut_density = min(cut_rate / 0.5, 1.0)
        volatility = float(
            min(
                1.0,
                0.35 * min(motion_std / 0.08, 1.0)
                + 0.25 * min(texture_std / 0.5, 1.0)
                + 0.25 * min(cut_density, 1.0)
                + 0.15 * worst_difficulty,
            )
        )

        short_side = float(min(width, height))
        pixels = float(max(width * height, 1))

        return {
            "segment_count": float(len(segments)),
            "cut_count": float(cut_count),
            "cut_rate": cut_rate,
            "hard_fraction": hard_fraction,
            "worst_difficulty": worst_difficulty,
            "difficulty_mean": difficulty_mean,
            "difficulty_p90": difficulty_p90,
            "duration_weighted_difficulty": duration_weighted_difficulty,
            "motion_mean": motion_mean,
            "motion_std": motion_std,
            "motion_p90": motion_p90,
            "texture": texture_mean,
            "texture_lbp": texture_mean,
            "texture_std": texture_std,
            "entropy": entropy,
            "edge_density": edge_density,
            "noise_level": noise_level,
            "high_freq_energy": high_freq_energy,
            "flatness": flatness,
            "luma_mean": luma_mean,
            "luma_std": luma_std,
            "sat_mean": sat_mean,
            "chroma_std": chroma_std,
            "cut_density": cut_density,
            "volatility": volatility,
            "duration": float(duration),
            "fps": float(fps),
            "width": float(width),
            "height": float(height),
            "short_side": short_side,
            "pixels": pixels,
            # Soft-normalized 0..1 levels for param rules / logging
            "motion_level": soft_level(motion_p90, _MOTION_P90_MID),
            "texture_level": soft_level(texture_mean, _TEXTURE_MID),
            "noise_level_norm": soft_level(noise_level, _NOISE_LEVEL_MID),
            "edge_level": soft_level(edge_density, _EDGE_DENSITY_MID),
            "cut_level": cut_density,
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
            "chroma_std",
            "cut_density",
            "volatility",
            "duration",
            "fps",
            "width",
            "height",
            "short_side",
            "pixels",
            "motion_level",
            "texture_level",
            "noise_level_norm",
            "edge_level",
            "cut_level",
        ]
        return {k: 0.0 for k in keys}


def format_feature_report(full: dict[str, Any]) -> str:
    """Human-readable feature dump for sanity checks."""
    meta = full.get("meta") or {}
    summary = full.get("summary") or {}
    lines = [
        f"path: {meta.get('path', '?')}",
        f"resolution: {int(meta.get('width', 0))}x{int(meta.get('height', 0))} "
        f"@ {meta.get('fps', 0):.3f} fps, {meta.get('duration', 0):.2f}s",
        f"detail_scale: {meta.get('detail_scale', 0):.4f}  "
        f"texture_comp: {meta.get('texture_compensation', 0):.2f}",
        f"segments: {int(summary.get('segment_count', 0))}  "
        f"cuts: {int(summary.get('cut_count', 0))}  "
        f"cut_rate: {summary.get('cut_rate', 0):.4f}/s",
        "",
        "Raw metrics:",
    ]
    for key in (
        "motion_mean",
        "motion_p90",
        "texture",
        "edge_density",
        "noise_level",
        "high_freq_energy",
        "entropy",
        "chroma_std",
        "luma_mean",
        "sat_mean",
        "difficulty_mean",
        "worst_difficulty",
    ):
        if key in summary:
            lines.append(f"  {key:22} {summary[key]:.6f}")

    lines.append("")
    lines.append("Normalized levels (0..1, for NVENC rules):")
    for key in (
        "motion_level",
        "texture_level",
        "noise_level_norm",
        "edge_level",
        "cut_level",
    ):
        if key in summary:
            lines.append(f"  {key:22} {summary[key]:.4f}")

    segs = full.get("segments") or []
    if segs:
        lines.append("")
        lines.append(f"Per-segment ({len(segs)}):")
        for seg in segs[:8]:
            lines.append(
                f"  #{int(seg['index']):02d} "
                f"{seg['start_sec']:.1f}-{seg['end_sec']:.1f}s "
                f"motion_p90={seg['motion_p90']:.4f} "
                f"noise={seg['noise_level']:.5f} "
                f"texture={seg['texture']:.3f} "
                f"diff={seg['difficulty']:.3f}"
            )
        if len(segs) > 8:
            lines.append(f"  ... +{len(segs) - 8} more")

    return "\n".join(lines)


def extract_quick_features(
    video_path: str,
    *,
    sample_frames: int = 16,
    deadline: Optional[float] = None,
) -> dict[str, Any]:
    """Bounded uniform-frame feature prior for the 180-second fleet path.

    This intentionally avoids the full sequential cut scan. It supplies the
    normalized signals used by CQ/x265 baselines plus one whole-clip proxy
    segment.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    count = max(2, int(sample_frames))
    indices = np.linspace(0, max(0, frame_count - 1), count, dtype=int)
    grays: list[np.ndarray] = []
    try:
        for index in indices:
            if deadline is not None and time.monotonic() >= deadline:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 320.0 / max(1, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if not grays:
        raise RuntimeError(f"Cannot sample frames: {video_path}")

    motions = [
        float(np.mean(cv2.absdiff(a, b))) / 255.0
        for a, b in zip(grays, grays[1:])
    ]
    textures = [
        float(np.sqrt(max(0.0, cv2.Laplacian(g, cv2.CV_64F).var())))
        for g in grays
    ]
    edges = [
        float(np.mean(cv2.Canny(g, 80, 160) > 0))
        for g in grays
    ]
    noises = []
    for gray in grays:
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        noises.append(float(np.mean(cv2.absdiff(gray, smooth))) / 255.0)
    hist_diffs = []
    for left, right in zip(grays, grays[1:]):
        h1 = cv2.calcHist([left], [0], None, [16], [0, 256])
        h2 = cv2.calcHist([right], [0], None, [16], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hist_diffs.append(float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)))

    motion_p90 = float(np.percentile(motions, 90)) if motions else 0.0
    texture = float(np.mean(textures))
    edge_density = float(np.mean(edges))
    noise = float(np.mean(noises))
    cut_fraction = (
        float(np.mean(np.asarray(hist_diffs) > 0.45)) if hist_diffs else 0.0
    )
    motion_level = soft_level(motion_p90, _MOTION_P90_MID)
    texture_level = soft_level(texture, _TEXTURE_MID)
    edge_level = soft_level(edge_density, _EDGE_DENSITY_MID)
    noise_level = soft_level(noise, _NOISE_LEVEL_MID)
    difficulty = float(
        0.35 * motion_level
        + 0.30 * texture_level
        + 0.20 * noise_level
        + 0.15 * cut_fraction
    )
    summary = {
        "fps": fps,
        "duration": duration,
        "segment_count": 1,
        "motion_p90": motion_p90,
        "motion_level": motion_level,
        "texture": texture,
        "texture_level": texture_level,
        "edge_density": edge_density,
        "edge_level": edge_level,
        "noise_level": noise,
        "noise_level_norm": noise_level,
        "cut_rate": cut_fraction,
        "cut_level": cut_fraction,
        "hard_fraction": 1.0 if difficulty >= 0.45 else 0.0,
        "worst_difficulty": difficulty,
        "difficulty_p90": difficulty,
        "volatility": cut_fraction,
        "avg_segment_duration": duration,
    }
    segment = {
        "index": 0,
        "start_sec": 0.0,
        "end_sec": duration,
        "duration": duration,
        "difficulty": difficulty,
    }
    return {
        "meta": {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "sampled_frames": len(grays),
            "quick": True,
        },
        "segments": [segment] if duration > 0 else [],
        "summary": summary,
    }
