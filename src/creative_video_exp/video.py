from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
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


    def __init__(self, config: SamplingConfig):
        self.config = config

    def sample(
        self,
        video_path: str,
        video_id: str,
        metadata: dict[str, Any] | None = None,
        time_window_sec: tuple[float, float] | None = None,
    ) -> SparseFrameBatch:
        metadata = metadata or {}
        time_window_sec = _validate_time_window_sec(time_window_sec)
        frames, source_kind, load_metadata = self._load_frames(
            video_path,
            video_id,
            metadata,
            time_window_sec=time_window_sec,
        )
        if self.config.strategy != "binwise":
            raise ValueError("STARS requires bin-wise temporal sampling.")
        indices, bin_ids = self._select_indices(len(frames))
        sampled = frames[indices]
        audit_started = time.perf_counter()
        candidate_records = _candidate_frame_records(
            frames,
            records=load_metadata.get("decoded_candidate_frame_records"),
            origin=str(load_metadata.get("decode_backend", source_kind)),
        )
        sampled_records = []
        for sample_order, (candidate_pool_index, bin_id) in enumerate(
            zip(indices, bin_ids)
        ):
            record = dict(candidate_records[candidate_pool_index])
            record.update(
                {
                    "sample_order": sample_order,
                    "candidate_pool_index": candidate_pool_index,
                    "bin_id": bin_id,
                }
            )
            sampled_records.append(record)
        decoded_sequence_hash = _frame_sequence_sha256(frames)
        sampled_sequence_hash = _frame_sequence_sha256(sampled)
        sampling_audit_seconds = time.perf_counter() - audit_started
        return SparseFrameBatch(
            frames=sampled,
            selected_indices=indices,
            bin_ids=bin_ids,
            source_kind=source_kind,
            metadata={
                **load_metadata,
                "original_frame_count": int(len(frames)),
                "decoded_candidate_frame_count": int(len(frames)),
                "selected_candidate_pool_indices": list(indices),
                "sampled_frame_count": int(len(sampled)),
                "decoded_candidate_frame_records": candidate_records,
                "sampled_frame_records": sampled_records,
                "decoded_candidate_frame_sequence_sha256": decoded_sequence_hash,
                "sampled_frame_sequence_sha256": sampled_sequence_hash,
                "sampling_audit_seconds": sampling_audit_seconds,
                "sampling_strategy": self.config.strategy,
                "num_bins": self.config.num_bins,
                "frames_per_bin": self.config.frames_per_bin,
            },
        )

    def _load_frames(
        self,
        video_path: str,
        video_id: str,
        metadata: dict[str, Any],
        *,
        time_window_sec: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        video_failure_reason = ""
        if video_path:
            path = Path(video_path)
            if path.exists():
                if path.is_dir():
                    if time_window_sec is not None:
                        raise ValueError(
                            "Time-window sampling requires a timestamped video file, "
                            "not a pre-extracted image directory."
                        )
                    frames = _load_image_dir(path, self.config.synthetic_size)
                    if len(frames):
                        return frames, "image_dir", {
                            "decode_backend": "image_directory",
                            "decode_strategy": "preextracted_frames",
                            "fallback_used": False,
                            "fallback_reason": "",
                            **_empty_stream_metadata(),
                            "decoded_candidate_frame_records": [],
                        }
                if path.suffix.lower() == ".npz":
                    if time_window_sec is not None:
                        raise ValueError(
                            "Time-window sampling requires a timestamped video file, "
                            "not a frame archive."
                        )
                    frames = _load_npz(path, self.config.synthetic_size)
                    if len(frames):
                        return frames, "npz", {
                            "decode_backend": "numpy",
                            "decode_strategy": "preextracted_frames",
                            "fallback_used": False,
                            "fallback_reason": "",
                            **_empty_stream_metadata(),
                            "decoded_candidate_frame_records": [],
                        }
                if time_window_sec is None:
                    pyav_result = _load_video_with_pyav(
                        path,
                        self.config.synthetic_size,
                        max_frames=max(
                            self.config.max_frames * 2,
                            self.config.num_bins * 2,
                        ),
                    )
                else:
                    pyav_result = _load_video_with_pyav(
                        path,
                        self.config.synthetic_size,
                        max_frames=max(
                            self.config.max_frames * 2,
                            self.config.num_bins * 2,
                        ),
                        time_window_sec=time_window_sec,
                    )
                if len(pyav_result) == 3:
                    frames, decode_strategy, fallback_reason = pyav_result
                    pyav_metadata = {
                        **_empty_stream_metadata(),
                        "decoded_candidate_frame_records": (
                            _candidate_frame_records(frames, origin="pyav")
                        ),
                    }
                else:
                    (
                        frames,
                        decode_strategy,
                        fallback_reason,
                        pyav_metadata,
                    ) = pyav_result
                if len(frames):
                    return frames, "video", {
                        "decode_backend": "pyav",
                        "decode_strategy": decode_strategy,
                        "fallback_used": decode_strategy != "duration_seek_uniform",
                        "fallback_reason": fallback_reason,
                        **pyav_metadata,
                    }
                video_failure_reason = fallback_reason or "PyAV returned no frames"
                if self.config.allow_full_decode_fallback and time_window_sec is None:
                    frames = _load_video_with_torchvision(path, self.config.synthetic_size)
                    if len(frames):
                        return frames, "video", {
                            "decode_backend": "torchvision",
                            "decode_strategy": "torchvision_full_decode_fallback",
                            "fallback_used": True,
                            "fallback_reason": video_failure_reason,
                            **_empty_stream_metadata(),
                            "decoded_candidate_frame_records": [],
                        }

        if time_window_sec is not None:
            raise RuntimeError(
                "Time-window sampling could not decode frames strictly within "
                f"{time_window_sec[0]:.6f}-{time_window_sec[1]:.6f} seconds: "
                f"{video_failure_reason or 'video path is missing'}."
            )

        frames = _make_synthetic_video(
            video_id=video_id,
            metadata=metadata,
            frame_count=max(self.config.max_frames * 2, self.config.num_bins * 4),
            size=self.config.synthetic_size,
        )
        return frames, "synthetic", {
            "decode_backend": "synthetic",
            "decode_strategy": "synthetic_fallback",
            "fallback_used": True,
            "fallback_reason": (
                video_failure_reason
                or "video path missing or all enabled real-video decoders failed"
            ),
            **_empty_stream_metadata(),
            "decoded_candidate_frame_records": [],
        }

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


def _load_video_with_pyav(
    path: Path,
    size: int,
    max_frames: int,
    time_window_sec: tuple[float, float] | None = None,
) -> tuple[np.ndarray, str, str, dict[str, Any]]:
    try:
        import av
    except ImportError:
        return (
            np.empty((0, size, size, 3), dtype=np.uint8),
            "pyav_unavailable",
            "PyAV is not installed",
            {
                **_empty_stream_metadata(),
                "decoded_candidate_frame_records": [],
            },
        )

    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream_metadata = _pyav_stream_metadata(stream)
        target_count = max(1, max_frames)
        if time_window_sec is None:
            decoded, frame_records = _seek_sample_pyav(
                container,
                stream,
                size,
                target_count,
            )
        else:
            decoded, frame_records = _seek_sample_pyav(
                container,
                stream,
                size,
                target_count,
                time_window_sec=time_window_sec,
            )
        strategy = "duration_seek_uniform"
        fallback_reason = ""
        if not decoded:
            container.close()
            container = av.open(str(path))
            stream = container.streams.video[0]
            stream_metadata = _pyav_stream_metadata(stream)
            if time_window_sec is None:
                decoded, frame_records = _early_sample_pyav(
                    container,
                    stream,
                    size,
                    target_count,
                )
            else:
                decoded, frame_records = _early_sample_pyav(
                    container,
                    stream,
                    size,
                    target_count,
                    time_window_sec=time_window_sec,
                )
            strategy = "early_sequential_fallback"
            fallback_reason = "stream duration was unavailable or temporal seeking failed"
        container.close()
        frames = (
            np.stack(decoded, axis=0)
            if decoded
            else np.empty((0, size, size, 3), dtype=np.uint8)
        )
        return (
            frames,
            strategy,
            fallback_reason,
            {
                **stream_metadata,
                **(
                    {
                        "requested_time_window_seconds": [
                            float(time_window_sec[0]),
                            float(time_window_sec[1]),
                        ]
                    }
                    if time_window_sec is not None
                    else {}
                ),
                "temporal_sampling_rule": (
                    (
                        "uniform_window_pts_decode_to_target_with_window_tail_fallback"
                        if time_window_sec is not None
                        else "uniform_stream_pts_decode_to_target_with_eof_tail_fallback"
                    )
                    if strategy == "duration_seek_uniform"
                    else None
                ),
                "decoded_candidate_frame_records": frame_records,
            },
        )
    except Exception as exc:
        return (
            np.empty((0, size, size, 3), dtype=np.uint8),
            "pyav_failed",
            f"{type(exc).__name__}: {exc}",
            {
                **_empty_stream_metadata(),
                "decoded_candidate_frame_records": [],
            },
        )


def _seek_sample_pyav(
    container: Any,
    stream: Any,
    size: int,
    target_count: int,
    time_window_sec: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if time_window_sec is not None:
        return _seek_sample_pyav_windowed(
            container,
            stream,
            size,
            target_count,
            time_window_sec,
        )
    duration = _optional_int(getattr(stream, "duration", None))
    if duration is None or duration <= 0:
        return [], []
    start = _optional_int(getattr(stream, "start_time", None))
    start = 0 if start is None else start
    target_count = max(1, int(target_count))
    if target_count == 1:
        targets = [start]
    else:
        span = duration - 1
        targets = list(
            dict.fromkeys(
                start + (span * target_index) // (target_count - 1)
                for target_index in range(target_count)
            )
        )
    decoded: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    last_selected_pts: int | None = None
    for target in targets:
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
            selected_frame = None
            selected_pts = None
            last_before_target = None
            last_before_target_pts = None
            decoded_frames_after_seek = 0
            for frame in container.decode(stream):
                decoded_frames_after_seek += 1
                frame_pts = _optional_int(getattr(frame, "pts", None))
                if frame_pts is None:
                    continue
                if last_selected_pts is not None and frame_pts <= last_selected_pts:
                    continue
                if frame_pts >= target:
                    selected_frame = frame
                    selected_pts = frame_pts
                    break
                if (
                    last_before_target_pts is None
                    or frame_pts > last_before_target_pts
                ):
                    last_before_target = frame
                    last_before_target_pts = frame_pts
            seek_resolution = "at_or_after_target"
            if selected_frame is None and last_before_target is not None:
                selected_frame = last_before_target
                selected_pts = last_before_target_pts
                seek_resolution = "last_before_eof"
            if selected_frame is None or selected_pts is None:
                continue
            resized = _resize_frame_array(
                selected_frame.to_ndarray(format="rgb24"),
                size,
            )
            decoded.append(resized)
            records.append(
                _pyav_frame_record(
                    selected_frame,
                    resized,
                    candidate_pool_index=len(decoded) - 1,
                    seek_target_pts=target,
                    origin="pyav_duration_seek",
                    stream=stream,
                    seek_resolution=seek_resolution,
                    decoded_frames_after_seek=decoded_frames_after_seek,
                )
            )
            last_selected_pts = selected_pts
        except Exception:
            return [], []
    return decoded, records


def _seek_sample_pyav_windowed(
    container: Any,
    stream: Any,
    size: int,
    target_count: int,
    time_window_sec: tuple[float, float],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    bounds = _time_window_pts_bounds(stream, time_window_sec)
    if bounds is None:
        return [], []
    window_start_pts, window_end_pts = bounds
    target_count = max(1, int(target_count))
    if target_count == 1:
        targets = [window_start_pts]
    else:
        span = window_end_pts - window_start_pts - 1
        targets = list(
            dict.fromkeys(
                window_start_pts + (span * target_index) // (target_count - 1)
                for target_index in range(target_count)
            )
        )

    decoded: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    last_selected_pts: int | None = None
    for target in targets:
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
            selected_frame = None
            selected_pts = None
            last_before_target = None
            last_before_target_pts = None
            decoded_frames_after_seek = 0
            reached_window_end = False
            for frame in container.decode(stream):
                decoded_frames_after_seek += 1
                frame_pts = _optional_int(getattr(frame, "pts", None))
                if frame_pts is None:
                    continue
                if frame_pts < window_start_pts:
                    continue
                if frame_pts >= window_end_pts:
                    reached_window_end = True
                    break
                if last_selected_pts is not None and frame_pts <= last_selected_pts:
                    continue
                if frame_pts >= target:
                    selected_frame = frame
                    selected_pts = frame_pts
                    break
                if (
                    last_before_target_pts is None
                    or frame_pts > last_before_target_pts
                ):
                    last_before_target = frame
                    last_before_target_pts = frame_pts
            seek_resolution = "at_or_after_target"
            if selected_frame is None and last_before_target is not None:
                selected_frame = last_before_target
                selected_pts = last_before_target_pts
                seek_resolution = (
                    "last_before_window_end"
                    if reached_window_end
                    else "last_before_eof"
                )
            if selected_frame is None or selected_pts is None:
                continue
            resized = _resize_frame_array(
                selected_frame.to_ndarray(format="rgb24"),
                size,
            )
            decoded.append(resized)
            records.append(
                _pyav_frame_record(
                    selected_frame,
                    resized,
                    candidate_pool_index=len(decoded) - 1,
                    seek_target_pts=target,
                    origin="pyav_duration_seek_windowed",
                    stream=stream,
                    seek_resolution=seek_resolution,
                    decoded_frames_after_seek=decoded_frames_after_seek,
                )
            )
            last_selected_pts = selected_pts
        except Exception:
            return [], []
    return decoded, records


def _early_sample_pyav(
    container: Any,
    stream: Any,
    size: int,
    target_count: int,
    time_window_sec: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if time_window_sec is not None:
        return _early_sample_pyav_windowed(
            container,
            stream,
            size,
            target_count,
            time_window_sec,
        )
    decoded: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    max_decoded_frames = max(90, target_count * 20)
    stride = max(1, max_decoded_frames // target_count)
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx % stride == 0:
            resized = _resize_frame_array(
                frame.to_ndarray(format="rgb24"),
                size,
            )
            decoded.append(resized)
            records.append(
                _pyav_frame_record(
                    frame,
                    resized,
                    candidate_pool_index=len(decoded) - 1,
                    seek_target_pts=None,
                    origin="pyav_early_sequential_fallback",
                    stream=stream,
                )
            )
        if len(decoded) >= target_count or frame_idx >= max_decoded_frames:
            break
    return decoded, records


def _early_sample_pyav_windowed(
    container: Any,
    stream: Any,
    size: int,
    target_count: int,
    time_window_sec: tuple[float, float],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    bounds = _time_window_pts_bounds(stream, time_window_sec)
    if bounds is None:
        return [], []
    window_start_pts, window_end_pts = bounds
    try:
        container.seek(
            window_start_pts,
            stream=stream,
            backward=True,
            any_frame=False,
        )
    except Exception:
        return [], []

    decoded: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    max_decoded_frames = max(90, target_count * 20)
    stride = max(1, max_decoded_frames // target_count)
    in_window_index = 0
    for frame in container.decode(stream):
        frame_pts = _optional_int(getattr(frame, "pts", None))
        if frame_pts is None or frame_pts < window_start_pts:
            continue
        if frame_pts >= window_end_pts:
            break
        if in_window_index % stride == 0:
            resized = _resize_frame_array(
                frame.to_ndarray(format="rgb24"),
                size,
            )
            records.append(
                _pyav_frame_record(
                    frame,
                    resized,
                    candidate_pool_index=len(decoded),
                    seek_target_pts=None,
                    origin="pyav_window_sequential_fallback",
                    stream=stream,
                    seek_resolution="within_requested_window",
                )
            )
            decoded.append(resized)
        in_window_index += 1
        if len(decoded) >= target_count or in_window_index >= max_decoded_frames:
            break
    return decoded, records


def _candidate_frame_records(
    frames: np.ndarray,
    *,
    records: Any = None,
    origin: str,
) -> list[dict[str, Any]]:
    supplied = records if isinstance(records, list) else []
    output = []
    for candidate_pool_index, frame in enumerate(frames):
        record = (
            dict(supplied[candidate_pool_index])
            if candidate_pool_index < len(supplied)
            and isinstance(supplied[candidate_pool_index], dict)
            else {}
        )
        record.update(
            {
                "candidate_pool_index": candidate_pool_index,
                "origin": str(record.get("origin") or origin),
                "seek_target_pts": record.get("seek_target_pts"),
                "decoded_pts": record.get("decoded_pts"),
                "decoded_dts": record.get("decoded_dts"),
                "decoded_time_seconds": record.get("decoded_time_seconds"),
                "time_base": record.get("time_base"),
                "seek_resolution": record.get("seek_resolution"),
                "decoded_frames_after_seek": record.get(
                    "decoded_frames_after_seek"
                ),
                "target_to_decoded_offset_pts": record.get(
                    "target_to_decoded_offset_pts"
                ),
                "post_resize_rgb_sha256": _frame_sha256(frame),
            }
        )
        output.append(record)
    return output


def _pyav_frame_record(
    frame: Any,
    resized: np.ndarray,
    *,
    candidate_pool_index: int,
    seek_target_pts: int | None,
    origin: str,
    stream: Any,
    seek_resolution: str | None = None,
    decoded_frames_after_seek: int | None = None,
) -> dict[str, Any]:
    frame_time = getattr(frame, "time", None)
    time_base = getattr(frame, "time_base", None) or getattr(
        stream,
        "time_base",
        None,
    )
    decoded_pts = _optional_int(getattr(frame, "pts", None))
    target_offset = None
    if decoded_pts is not None and seek_target_pts is not None:
        target_offset = decoded_pts - seek_target_pts
    return {
        "candidate_pool_index": int(candidate_pool_index),
        "origin": origin,
        "seek_target_pts": seek_target_pts,
        "decoded_pts": decoded_pts,
        "decoded_dts": _optional_int(getattr(frame, "dts", None)),
        "decoded_time_seconds": _optional_float(frame_time),
        "time_base": _fraction_string(time_base),
        "seek_resolution": seek_resolution,
        "decoded_frames_after_seek": decoded_frames_after_seek,
        "target_to_decoded_offset_pts": target_offset,
        "post_resize_rgb_sha256": None,
    }


def _pyav_stream_metadata(stream: Any) -> dict[str, Any]:
    duration_pts = _optional_int(getattr(stream, "duration", None))
    time_base = getattr(stream, "time_base", None)
    duration_seconds = None
    if duration_pts is not None and time_base is not None:
        try:
            duration_seconds = float(duration_pts * time_base)
        except (TypeError, ValueError, OverflowError):
            duration_seconds = None
    reported_frames = _optional_int(getattr(stream, "frames", None))
    if reported_frames is not None and reported_frames <= 0:
        reported_frames = None
    return {
        "source_stream_frame_count_reported": reported_frames,
        "source_stream_duration_pts": duration_pts,
        "source_stream_duration_seconds": duration_seconds,
        "source_stream_start_time_pts": _optional_int(
            getattr(stream, "start_time", None)
        ),
        "source_stream_time_base": _fraction_string(time_base),
        "source_stream_average_rate": _fraction_string(
            getattr(stream, "average_rate", None)
        ),
    }


def _empty_stream_metadata() -> dict[str, None]:
    return {
        "source_stream_frame_count_reported": None,
        "source_stream_duration_pts": None,
        "source_stream_duration_seconds": None,
        "source_stream_start_time_pts": None,
        "source_stream_time_base": None,
        "source_stream_average_rate": None,
        "temporal_sampling_rule": None,
    }


def _validate_time_window_sec(
    value: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("time_window_sec must be a two-item (start, end) pair.")
    start = _optional_float(value[0])
    end = _optional_float(value[1])
    if start is None or end is None:
        raise ValueError("time_window_sec values must be finite numbers.")
    if start < 0.0 or end <= start:
        raise ValueError(
            "time_window_sec must satisfy 0 <= start < end."
        )
    return float(start), float(end)


def _time_window_pts_bounds(
    stream: Any,
    time_window_sec: tuple[float, float],
) -> tuple[int, int] | None:
    start_sec, end_sec = _validate_time_window_sec(time_window_sec) or (0.0, 0.0)
    time_base = getattr(stream, "time_base", None)
    if time_base is None:
        return None
    try:
        time_base_fraction = Fraction(time_base)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if time_base_fraction <= 0:
        return None

    stream_start = _optional_int(getattr(stream, "start_time", None))
    stream_start = 0 if stream_start is None else stream_start
    start_offset = math.ceil(Fraction(str(start_sec)) / time_base_fraction)
    end_offset = math.ceil(Fraction(str(end_sec)) / time_base_fraction)
    window_start = stream_start + int(start_offset)
    window_end = stream_start + int(end_offset)

    stream_duration = _optional_int(getattr(stream, "duration", None))
    if stream_duration is not None and stream_duration > 0:
        stream_end = stream_start + stream_duration
        window_start = min(window_start, stream_end)
        window_end = min(window_end, stream_end)
    if window_end <= window_start:
        return None
    return window_start, window_end


def _frame_sha256(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frame_sequence_sha256(frames: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(frames)).encode("ascii"))
    digest.update(b"\0")
    for frame in frames:
        digest.update(bytes.fromhex(_frame_sha256(frame)))
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _fraction_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        fraction = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return str(value)
    return f"{fraction.numerator}/{fraction.denominator}"


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
