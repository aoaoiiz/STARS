from __future__ import annotations

import tempfile
import unittest
import sys
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import SamplingConfig
from creative_video_exp.representations import StatsFrameEncoder
from creative_video_exp.video import (
    SparseFrameSampler,
    _frame_sha256,
    _seek_sample_pyav,
)


class _FakeFrame:
    def __init__(self, value: int, pts: int | None, time_base: Fraction):
        self.value = value
        self.pts = pts
        self.dts = None if pts is None else pts - 1
        self.time_base = time_base
        self.time = None if pts is None else float(pts * time_base)

    def to_ndarray(self, format: str) -> np.ndarray:
        if format != "rgb24":
            raise AssertionError(format)
        return np.full((4, 4, 3), self.value, dtype=np.uint8)


class _FakeStream:
    def __init__(self, duration: int = 100, start_time: int | None = 0):
        self.duration = duration
        self.frames = 1000
        self.start_time = start_time
        self.time_base = Fraction(1, 10)
        self.average_rate = Fraction(30, 1)


class _FakeContainer:
    def __init__(self, sequences: dict[int, list[int | None]] | None = None):
        self.targets: list[int] = []
        self.sequences = sequences or {}
        self.decoded_counts: dict[int, int] = {}

    def seek(self, target: int, **kwargs) -> None:
        self.targets.append(target)

    def decode(self, stream):
        target = self.targets[-1]
        values = self.sequences.get(target, [target])
        for pts in values:
            self.decoded_counts[target] = self.decoded_counts.get(target, 0) + 1
            value = 0 if pts is None else pts % 256
            yield _FakeFrame(value, pts, stream.time_base)


class SamplingProvenanceTests(unittest.TestCase):
    def test_seek_decodes_past_keyframe_to_each_uniform_target(self) -> None:
        container = _FakeContainer(
            {
                target: [target - 10, target - 1, target, target + 1]
                for target in (0, 33, 66, 99)
            }
        )
        stream = _FakeStream()
        frames, records = _seek_sample_pyav(
            container,
            stream,
            size=4,
            target_count=4,
        )

        self.assertEqual(container.targets, [0, 33, 66, 99])
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [0, 33, 66, 99])
        self.assertEqual([row["decoded_pts"] for row in records], [0, 33, 66, 99])
        self.assertEqual([row["seek_target_pts"] for row in records], [0, 33, 66, 99])
        self.assertEqual(records[1]["decoded_time_seconds"], 3.3)
        self.assertEqual(records[1]["time_base"], "1/10")
        self.assertEqual(
            [row["seek_resolution"] for row in records],
            ["at_or_after_target"] * 4,
        )
        self.assertEqual(
            [row["decoded_frames_after_seek"] for row in records],
            [3, 3, 3, 3],
        )
        self.assertEqual(
            [row["target_to_decoded_offset_pts"] for row in records],
            [0, 0, 0, 0],
        )
        self.assertEqual(container.decoded_counts, {0: 3, 33: 3, 66: 3, 99: 3})
        self.assertIsNone(records[1]["post_resize_rgb_sha256"])

    def test_seek_targets_include_nonzero_stream_start_time(self) -> None:
        container = _FakeContainer()
        frames, records = _seek_sample_pyav(
            container,
            _FakeStream(duration=100, start_time=1000),
            size=4,
            target_count=4,
        )

        self.assertEqual(container.targets, [1000, 1033, 1066, 1099])
        self.assertEqual([row["decoded_pts"] for row in records], container.targets)
        self.assertEqual(len(frames), 4)

    def test_seek_deduplicates_short_stream_targets_and_output_pts(self) -> None:
        container = _FakeContainer(
            {
                0: [0],
                1: [0, 1],
                2: [1, 2],
            }
        )
        frames, records = _seek_sample_pyav(
            container,
            _FakeStream(duration=3),
            size=4,
            target_count=5,
        )

        self.assertEqual(container.targets, [0, 1, 2])
        self.assertEqual([row["decoded_pts"] for row in records], [0, 1, 2])
        self.assertEqual(len(frames), 3)

    def test_seek_uses_last_available_frame_only_at_eof(self) -> None:
        container = _FakeContainer({0: [0], 99: [80, 98, 95]})
        _, records = _seek_sample_pyav(
            container,
            _FakeStream(),
            size=4,
            target_count=2,
        )

        self.assertEqual([row["decoded_pts"] for row in records], [0, 98])
        self.assertEqual(records[1]["seek_resolution"], "last_before_eof")
        self.assertEqual(records[1]["target_to_decoded_offset_pts"], -1)

    def test_seek_skips_frames_without_pts(self) -> None:
        container = _FakeContainer({0: [None, None, 0]})
        _, records = _seek_sample_pyav(
            container,
            _FakeStream(duration=1),
            size=4,
            target_count=4,
        )

        self.assertEqual([row["decoded_pts"] for row in records], [0])
        self.assertEqual(records[0]["decoded_frames_after_seek"], 3)

    def test_batch_metadata_uses_candidate_pool_names_without_changing_selection(self) -> None:
        frames = np.stack(
            [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(32)],
            axis=0,
        )
        records = [
            {
                "candidate_pool_index": index,
                "origin": "pyav_duration_seek",
                "seek_target_pts": index * 10,
                "decoded_pts": index * 10,
                "decoded_dts": index * 10 - 1,
                "decoded_time_seconds": float(index),
                "time_base": "1/10",
                "post_resize_rgb_sha256": _frame_sha256(frame),
            }
            for index, frame in enumerate(frames)
        ]
        sampler = SparseFrameSampler(
            SamplingConfig(
                strategy="binwise",
                num_bins=8,
                frames_per_bin=2,
                max_frames=16,
                synthetic_size=4,
            )
        )

        with tempfile.TemporaryDirectory() as raw_dir:
            video_path = Path(raw_dir) / "video.mp4"
            video_path.write_bytes(b"fixture")
            with patch(
                "creative_video_exp.video._load_video_with_pyav",
                return_value=(
                    frames,
                    "duration_seek_uniform",
                    "",
                    {
                        "source_stream_frame_count_reported": 1000,
                        "source_stream_duration_pts": 320,
                        "source_stream_duration_seconds": 32.0,
                        "source_stream_start_time_pts": 0,
                        "source_stream_time_base": "1/10",
                        "source_stream_average_rate": "30/1",
                        "decoded_candidate_frame_records": records,
                    },
                ),
            ):
                batch = sampler.sample(str(video_path), "fixture")

        expected = [0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31]
        self.assertEqual(batch.selected_indices, expected)
        self.assertEqual(batch.metadata["selected_candidate_pool_indices"], expected)
        self.assertEqual(batch.metadata["original_frame_count"], 32)
        self.assertEqual(batch.metadata["decoded_candidate_frame_count"], 32)
        self.assertEqual(batch.metadata["sampled_frame_count"], 16)
        self.assertGreaterEqual(batch.metadata["sampling_audit_seconds"], 0.0)
        self.assertEqual(
            [row["candidate_pool_index"] for row in batch.metadata["sampled_frame_records"]],
            expected,
        )
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in batch.frames],
            expected,
        )

    def test_non_temporal_sources_use_null_time_fields_and_verifiable_hashes(self) -> None:
        sampler = SparseFrameSampler(
            SamplingConfig(
                strategy="binwise",
                num_bins=2,
                frames_per_bin=1,
                max_frames=2,
                synthetic_size=4,
            )
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.fromarray(np.full((4, 4, 3), 25, dtype=np.uint8)).save(
                image_dir / "a.png"
            )
            Image.fromarray(np.full((4, 4, 3), 75, dtype=np.uint8)).save(
                image_dir / "b.png"
            )
            image_batch = sampler.sample(str(image_dir), "images")

            npz_path = root / "frames.npz"
            np.savez(
                npz_path,
                frames=np.stack(
                    [
                        np.full((4, 4, 3), 10, dtype=np.uint8),
                        np.full((4, 4, 3), 20, dtype=np.uint8),
                    ],
                    axis=0,
                ),
            )
            npz_batch = sampler.sample(str(npz_path), "npz")
            synthetic_batch = sampler.sample("", "synthetic")

        for batch in (image_batch, npz_batch, synthetic_batch):
            self.assertIsNone(batch.metadata["source_stream_frame_count_reported"])
            self.assertIsNone(batch.metadata["source_stream_duration_seconds"])
            self.assertTrue(batch.metadata["decoded_candidate_frame_records"])
            self.assertGreaterEqual(batch.metadata["sampling_audit_seconds"], 0.0)
            for row in batch.metadata["decoded_candidate_frame_records"]:
                self.assertIsNone(row["seek_target_pts"])
                self.assertIsNone(row["decoded_pts"])
                self.assertIsNone(row["decoded_dts"])
                self.assertIsNone(row["decoded_time_seconds"])
                self.assertIsNone(row["time_base"])
                self.assertEqual(len(row["post_resize_rgb_sha256"]), 64)

    def test_keyframe_source_alias_is_named_as_candidate_pool_index(self) -> None:
        sampler = SparseFrameSampler(
            SamplingConfig(
                strategy="binwise",
                num_bins=2,
                frames_per_bin=1,
                max_frames=2,
                synthetic_size=4,
            )
        )
        batch = sampler.sample("", "representation")
        representation = StatsFrameEncoder().encode(batch)
        self.assertEqual(
            representation.keyframe_candidate_pool_indices,
            representation.keyframe_source_indices,
        )
        self.assertTrue(representation.keyframe_candidate_pool_indices)


if __name__ == "__main__":
    unittest.main()
