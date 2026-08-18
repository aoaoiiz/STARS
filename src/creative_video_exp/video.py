from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SamplingConfig


@dataclass
class SparseFrameBatch:
    frames: np.ndarray
    selected_indices: list[int]
    bin_ids: list[int]
    source_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseFrameSampler:
    def __init__(self, config: SamplingConfig):
        self.config = config

    def sample(
        self,
        video_path: str,
        video_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SparseFrameBatch:
        frames, source_kind, load_metadata = self._load_frames(video_path)
        indices, bin_ids = self._select_indices(len(frames))
        sampled = frames[indices]
        return SparseFrameBatch(
            frames=sampled,
            selected_indices=indices,
            bin_ids=bin_ids,
            source_kind=source_kind,
            metadata={
                "original_frame_count": int(len(frames)),
                "sampling_strategy": "binwise",
                "num_bins": self.config.num_bins,
                "frames_per_bin": self.config.frames_per_bin,
                **load_metadata,
            },
        )

    def _load_frames(
        self,
        video_path: str,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        if not video_path:
            raise FileNotFoundError("Video path is empty.")
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video input not found: {path}")
        if path.is_dir():
            frames = _load_image_dir(path, self.config.image_size)
            if not len(frames):
                raise RuntimeError(f"No supported image frames found in: {path}")
            return frames, "image_dir", {
                "decode_backend": "image_directory",
                "decode_strategy": "preextracted_frames",
                "fallback_used": False,
                "fallback_reason": "",
            }
        if path.suffix.lower() == ".npz":
            frames = _load_npz(path, self.config.image_size)
            if not len(frames):
                raise RuntimeError(f"No valid frame array found in: {path}")
            return frames, "npz", {
                "decode_backend": "numpy",
                "decode_strategy": "preextracted_frames",
                "fallback_used": False,
                "fallback_reason": "",
            }
        frames, decode_strategy, failure_reason = _load_video_with_pyav(
            path,
            self.config.image_size,
            max_frames=max(self.config.max_frames * 2, self.config.num_bins * 2),
        )
        if len(frames):
            return frames, "video", {
                "decode_backend": "pyav",
                "decode_strategy": decode_strategy,
                "fallback_used": decode_strategy != "duration_seek_uniform",
                "fallback_reason": failure_reason,
            }
        failure_reason = failure_reason or "PyAV returned no frames"
        raise RuntimeError(f"Unable to decode video input {path}: {failure_reason}")

    def _select_indices(self, total_frames: int) -> tuple[list[int], list[int]]:
        if total_frames <= 0:
            raise ValueError("Cannot sample from an empty frame sequence.")

        indices: list[int] = []
        bin_ids: list[int] = []
        bins = min(self.config.num_bins, total_frames)
        for bin_id in range(bins):
            start = math.floor(bin_id * total_frames / bins)
            end = math.floor((bin_id + 1) * total_frames / bins)
            end = max(end, start + 1)
            span = list(range(start, min(end, total_frames)))
            picks = _even_picks(span, self.config.frames_per_bin)
            for pick in picks:
                indices.append(pick)
                bin_ids.append(bin_id)

        deduped: list[int] = []
        deduped_bins: list[int] = []
        seen: set[int] = set()
        for idx, bin_id in zip(indices, bin_ids):
            if idx not in seen:
                deduped.append(idx)
                deduped_bins.append(bin_id)
                seen.add(idx)

        if self.config.max_frames > 0 and len(deduped) > self.config.max_frames:
            keep = _even_picks(list(range(len(deduped))), self.config.max_frames)
            deduped = [deduped[i] for i in keep]
            deduped_bins = [deduped_bins[i] for i in keep]

        return deduped, deduped_bins

def _even_picks(values: list[int], count: int) -> list[int]:
    if not values:
        return []
    count = max(1, min(count, len(values)))
    if count == 1:
        return [values[len(values) // 2]]
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def _load_image_dir(path: Path, size: int) -> np.ndarray:
    image_paths = sorted(
        item
        for item in path.iterdir()
        if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    frames = [_load_image(image_path, size) for image_path in image_paths]
    return np.stack(frames, axis=0) if frames else np.empty((0, size, size, 3), dtype=np.uint8)


def _load_npz(path: Path, size: int) -> np.ndarray:
    payload = np.load(path)
    key = "frames" if "frames" in payload else payload.files[0]
    frames = payload[key]
    if frames.ndim != 4:
        return np.empty((0, size, size, 3), dtype=np.uint8)
    if frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    return _resize_frames(_to_uint8(frames), size)


def _load_video_with_pyav(
    path: Path,
    size: int,
    max_frames: int,
) -> tuple[np.ndarray, str, str]:
    try:
        import av
    except ImportError:
        return np.empty((0, size, size, 3), dtype=np.uint8), "pyav_unavailable", "PyAV is not installed"

    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        target_count = max(1, max_frames)
        decoded = _seek_sample_pyav(container, stream, size, target_count)
        strategy = "duration_seek_uniform"
        fallback_reason = ""
        if not decoded:
            container.close()
            container = av.open(str(path))
            stream = container.streams.video[0]
            decoded = _early_sample_pyav(container, stream, size, target_count)
            strategy = "early_sequential_fallback"
            fallback_reason = "stream duration was unavailable or temporal seeking failed"
        container.close()
        frames = (
            np.stack(decoded, axis=0)
            if decoded
            else np.empty((0, size, size, 3), dtype=np.uint8)
        )
        return frames, strategy, fallback_reason
    except Exception as exc:
        return (
            np.empty((0, size, size, 3), dtype=np.uint8),
            "pyav_failed",
            f"{type(exc).__name__}: {exc}",
        )


def _seek_sample_pyav(container: Any, stream: Any, size: int, target_count: int) -> list[np.ndarray]:
    if not stream.duration:
        return []
    decoded: list[np.ndarray] = []
    targets = np.linspace(0, max(0, int(stream.duration) - 1), target_count)
    for target in targets:
        try:
            container.seek(int(target), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                decoded.append(_resize_frame_array(frame.to_ndarray(format="rgb24"), size))
                break
        except Exception:
            return []
    return decoded


def _early_sample_pyav(container: Any, stream: Any, size: int, target_count: int) -> list[np.ndarray]:
    decoded: list[np.ndarray] = []
    max_decoded_frames = max(90, target_count * 20)
    stride = max(1, max_decoded_frames // target_count)
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx % stride == 0:
            decoded.append(_resize_frame_array(frame.to_ndarray(format="rgb24"), size))
        if len(decoded) >= target_count or frame_idx >= max_decoded_frames:
            break
    return decoded


def _load_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(image, dtype=np.uint8)


def _resize_frame_array(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame.astype(np.uint8)
    return np.asarray(Image.fromarray(frame).convert("RGB").resize((size, size)), dtype=np.uint8)


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    resized = [
        np.asarray(Image.fromarray(frame).convert("RGB").resize((size, size)), dtype=np.uint8)
        for frame in frames
    ]
    return np.stack(resized, axis=0)


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype == np.uint8:
        return frames
    if frames.max(initial=0) <= 1.0:
        frames = frames * 255.0
    return np.clip(frames, 0, 255).astype(np.uint8)
