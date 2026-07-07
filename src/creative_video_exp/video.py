from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SamplingConfig
from .utils import stable_hash_int


@dataclass
class SparseFrameBatch:
    frames: np.ndarray
    selected_indices: list[int]
    bin_ids: list[int]
    source_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseFrameSampler:
    """Bin-wise sparse sampling with deterministic fallback for smoke tests."""

    def __init__(self, config: SamplingConfig):
        self.config = config

    def sample(
        self,
        video_path: str,
        video_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SparseFrameBatch:
        metadata = metadata or {}
        frames, source_kind = self._load_frames(video_path, video_id, metadata)
        indices, bin_ids = self._select_indices(len(frames))
        sampled = frames[indices]
        return SparseFrameBatch(
            frames=sampled,
            selected_indices=indices,
            bin_ids=bin_ids,
            source_kind=source_kind,
            metadata={
                "original_frame_count": int(len(frames)),
                "num_bins": self.config.num_bins,
                "frames_per_bin": self.config.frames_per_bin,
            },
        )

    def _load_frames(
        self,
        video_path: str,
        video_id: str,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, str]:
        if video_path:
            path = Path(video_path)
            if path.exists():
                if path.is_dir():
                    frames = _load_image_dir(path, self.config.synthetic_size)
                    if len(frames):
                        return frames, "image_dir"
                if path.suffix.lower() == ".npz":
                    frames = _load_npz(path, self.config.synthetic_size)
                    if len(frames):
                        return frames, "npz"
                frames = _load_video_with_pyav(
                    path,
                    self.config.synthetic_size,
                    max_frames=max(self.config.max_frames * 2, self.config.num_bins * 2),
                )
                if len(frames):
                    return frames, "video"
                frames = _load_video_with_torchvision(path, self.config.synthetic_size)
                if len(frames):
                    return frames, "video"

        return _make_synthetic_video(
            video_id=video_id,
            metadata=metadata,
            frame_count=max(self.config.max_frames * 2, self.config.num_bins * 4),
            size=self.config.synthetic_size,
        ), "synthetic"

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


def _load_video_with_torchvision(path: Path, size: int) -> np.ndarray:
    try:
        from torchvision.io import read_video

        frames, _, _ = read_video(str(path), pts_unit="sec")
        if hasattr(frames, "numpy"):
            frames = frames.numpy()
        if frames.ndim != 4:
            return np.empty((0, size, size, 3), dtype=np.uint8)
        return _resize_frames(_to_uint8(frames), size)
    except Exception:
        return np.empty((0, size, size, 3), dtype=np.uint8)


def _load_video_with_pyav(path: Path, size: int, max_frames: int) -> np.ndarray:
    try:
        import av
    except ImportError:
        return np.empty((0, size, size, 3), dtype=np.uint8)

    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        target_count = max(1, max_frames)
        decoded = _seek_sample_pyav(container, stream, size, target_count)
        if not decoded:
            container.close()
            container = av.open(str(path))
            stream = container.streams.video[0]
            decoded = _early_sample_pyav(container, stream, size, target_count)
        container.close()
        return np.stack(decoded, axis=0) if decoded else np.empty((0, size, size, 3), dtype=np.uint8)
    except Exception:
        return np.empty((0, size, size, 3), dtype=np.uint8)


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


def _make_synthetic_video(
    video_id: str,
    metadata: dict[str, Any],
    frame_count: int,
    size: int,
) -> np.ndarray:
    seed = stable_hash_int(video_id or str(metadata), 2**32)
    rng = np.random.default_rng(seed)
    base = _palette_from_text(" ".join(str(value) for value in metadata.values()))
    yy, xx = np.mgrid[0:size, 0:size]
    frames = []
    for frame_idx in range(frame_count):
        phase = frame_idx / max(1, frame_count - 1)
        wave = 0.5 + 0.5 * np.sin((xx / size * 3.0 + phase * 2.0) * np.pi)
        ring = 0.5 + 0.5 * np.cos((yy / size * 2.0 - phase) * np.pi)
        noise = rng.normal(0, 8, size=(size, size, 3))
        frame = np.zeros((size, size, 3), dtype=np.float32)
        frame[..., 0] = base[0] * (0.65 + 0.35 * wave)
        frame[..., 1] = base[1] * (0.7 + 0.3 * ring)
        frame[..., 2] = base[2] * (0.6 + 0.4 * (1.0 - wave))
        square = int(size * (0.18 + 0.08 * math.sin(phase * math.pi)))
        x0 = int((size - square) * phase)
        y0 = int((size - square) * (1.0 - phase))
        accent = np.roll(base, 1) + rng.integers(-20, 20, size=3)
        frame[y0 : y0 + square, x0 : x0 + square] = accent
        frames.append(np.clip(frame + noise, 0, 255).astype(np.uint8))
    return np.stack(frames, axis=0)


def _palette_from_text(text: str) -> np.ndarray:
    if "咖啡" in text or "coffee" in text.lower():
        return np.array([162, 108, 62], dtype=np.float32)
    if "鞋" in text or "跑" in text or "sport" in text.lower():
        return np.array([72, 132, 190], dtype=np.float32)
    if "灯" in text or "书" in text or "home" in text.lower():
        return np.array([224, 198, 126], dtype=np.float32)
    digest = stable_hash_int(text or "default")
    return np.array(
        [80 + digest % 140, 80 + (digest // 17) % 140, 80 + (digest // 31) % 140],
        dtype=np.float32,
    )
