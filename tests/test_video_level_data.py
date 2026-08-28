from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from creative_video_exp.checkpoint_identity import (
    CHECKPOINT_MANIFEST_SCHEMA,
    LOCAL_CHECKPOINT_KIND,
)
from creative_video_exp.config import DataConfig
from creative_video_exp.data import load_samples
from creative_video_exp.semantic_points import (
    FORMAL_REFERENCE_PROTOCOL,
    formal_reference_contract_issue,
)
from run_server_matrix import build_config


def _provenance() -> dict:
    return {
        "protocol": FORMAL_REFERENCE_PROTOCOL,
        "reference_usage": "evaluation_only",
        "reference_enters_generation": False,
        "reference_enters_reward": False,
        "reference_enters_candidate_selection": False,
    }


class VideoLevelDataTest(unittest.TestCase):
    def _load(self, rows: list[dict], sample_unit: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"samples": rows}), encoding="utf-8")
            return load_samples(
                DataConfig(
                    name="fixture",
                    annotation_path=str(path),
                    sample_unit=sample_unit,
                )
            )

    def test_annotation_row_loading_remains_available(self) -> None:
        rows = [
            {"video_id": "v1", "video_path": "/videos/v1.mp4", "question": "Q1", "answer": "A1"},
            {"video_id": "v1", "video_path": "/videos/v1.mp4", "question": "Q2", "answer": "A2"},
        ]
        samples = self._load(rows, "annotation_row")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].source_annotation_row_count, 1)
        self.assertEqual(samples[0].question, "Q1")

    def test_unique_video_preserves_order_pairs_and_provenance(self) -> None:
        rows = [
            {
                "video_id": "v1",
                "video_path": "/videos/v1.mp4",
                "duration": 30,
                "caption": "Shared caption",
                "question": "Q1",
                "answer": "A1",
                "category": "reasoning",
                "semantic_points": ["Evidence context: Q1", "Expected answer: A1"],
                "semantic_point_provenance": _provenance(),
            },
            {
                "video_id": "v2",
                "video_path": "/videos/v2.mp4",
                "question": "Q3",
                "answer": "A3",
            },
            {
                "video_id": "v1",
                "video_path": "/videos/v1.mp4",
                "duration": 30.0,
                "caption": "Additional caption",
                "question": "Q2",
                "answer": "A2",
                "category": "detail",
                "semantic_points": ["Evidence context: Q2", "Expected answer: A2"],
                "semantic_point_provenance": _provenance(),
            },
            {
                "video_id": "v1",
                "video_path": "/videos/v1.mp4",
                "duration": 30,
                "caption": "Shared caption",
                "question": "Q1",
                "answer": "A1",
                "category": "secondary",
                "semantic_points": ["Evidence context: Q1", "Expected answer: A1"],
                "semantic_point_provenance": _provenance(),
            },
        ]
        samples = self._load(rows, "unique_video")
        self.assertEqual([sample.video_id for sample in samples], ["v1", "v2"])
        first = samples[0]
        self.assertEqual(first.source_annotation_row_count, 3)
        self.assertEqual(len(first.source_row_fingerprints), 3)
        self.assertEqual(len(first.evaluation_references), 2)
        self.assertEqual(first.question, "Q1\nQ2")
        self.assertEqual(first.answer, "A1\nA2")
        self.assertEqual(first.caption, "Shared caption\nAdditional caption")
        self.assertEqual(first.category, "reasoning | detail | secondary")
        self.assertEqual(
            first.semantic_points,
            [
                "Evidence context: Q1",
                "Expected answer: A1",
                "Evidence context: Q2",
                "Expected answer: A2",
            ],
        )
        self.assertEqual(
            first.raw["semantic_point_provenance"]["qa_reference_count"],
            2,
        )
        self.assertEqual(formal_reference_contract_issue(first.raw), "")

    def test_unique_video_falls_back_to_video_id_without_path(self) -> None:
        samples = self._load(
            [
                {"video_id": "v1", "question": "Q1", "answer": "A1"},
                {"video_id": "v1", "question": "Q2", "answer": "A2"},
            ],
            "unique_video",
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].source_annotation_row_count, 2)

    def test_conflicting_path_id_and_duration_are_rejected(self) -> None:
        cases = [
            [
                {"video_id": "v1", "video_path": "/videos/a.mp4"},
                {"video_id": "v1", "video_path": "/videos/b.mp4"},
            ],
            [
                {"video_id": "v1", "video_path": "/videos/a.mp4"},
                {"video_id": "v2", "video_path": "/videos/a.mp4"},
            ],
            [
                {"video_id": "v1", "video_path": "/videos/a.mp4", "duration": 10},
                {"video_id": "v1", "video_path": "/videos/a.mp4", "duration": 12},
            ],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self._load(rows, "unique_video")

    def test_formal_contract_accepts_multiple_qa_pairs(self) -> None:
        row = {
            "semantic_points": [
                "Evidence context: Q1",
                "Expected answer: A1",
                "Evidence context: Q2",
                "Expected answer: A2",
            ],
            "semantic_point_provenance": {**_provenance(), "qa_reference_count": 2},
        }
        self.assertEqual(formal_reference_contract_issue(row), "")
        row["semantic_points"] = row["semantic_points"][:-1]
        self.assertIn("even number", formal_reference_contract_issue(row))

    def test_server_matrix_builds_unique_video_config(self) -> None:
        def identity(model_id: str) -> dict:
            return {
                "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
                "kind": LOCAL_CHECKPOINT_KIND,
                "model_id": model_id,
                "revision": "fixture",
                "identity_sha256": "0" * 64,
                "file_count": 1,
                "total_bytes": 1,
            }

        args = SimpleNamespace(
            output_root="outputs/fixture",
        )
        environment = {
            "LONGBENCH_ANNOTATION_PATH": "/annotations/manifest.json",
            "LONGBENCH_VIDEO_ROOT": "/videos",
            "LLAVA_VIDEO_ENDPOINT_URL": "http://127.0.0.1:8010/v1/chat/completions",
            "REWARD_VISION_MODEL_PATH": "/models/siglip2",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = build_config(
                "longvideobench",
                "llava_video_qwen2",
                args,
                model_identity=identity("LLaVA-Video-7B-Qwen2"),
                model_manifest_path="generator.json",
                reward_identity=identity("google/siglip2-so400m-patch14-384"),
                reward_manifest_path="reward.json",
            )
        self.assertEqual(config["data"]["sample_unit"], "unique_video")
        self.assertEqual(config["data"]["offset"], 0)
        self.assertEqual(config["data"]["limit"], 0)


if __name__ == "__main__":
    unittest.main()
