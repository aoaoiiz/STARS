from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import GenerationConfig, ModelEndpointConfig
from creative_video_exp.data import VideoSample
from creative_video_exp.modeling import (
    OpenAICompatibleScriptGenerationModel,
    _parse_single_script_candidate,
)
from creative_video_exp.reward import validate_candidate
from creative_video_exp.video import SparseFrameBatch


class ContractFieldRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GenerationConfig(
            num_candidates=1,
            segments=2,
            target_duration_sec=30,
        )
        self.sample = VideoSample(video_id="field-recovery")

    def parse(self, payload: dict[str, object]):
        return _parse_single_script_candidate(
            json.dumps(payload),
            self.sample,
            self.config,
            None,
            1,
        )

    def test_aliases_placeholders_and_unique_boundaries_are_recovered(self) -> None:
        payload = {
            "candidate_id": "candidate",
            "variant": "visual_story",
            "segments": [
                {
                    "end_time": 15,
                    "description": "A person enters the visible room.",
                    "key_point": "visible entrance",
                    "tags": ["salient_point"],
                },
                {
                    "voiceover": "The final view summarizes the visible sequence.",
                    "on_screen_text": "",
                    "tags": ["summary"],
                },
            ],
        }

        candidate, error = self.parse(payload)

        self.assertEqual(error, "")
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(
            [(segment.start, segment.end) for segment in candidate.timeline],
            [(0.0, 15.0), (15.0, 30.0)],
        )
        self.assertEqual(candidate.timeline[0].narration, payload["segments"][0]["description"])
        self.assertEqual(candidate.timeline[0].on_screen_text, "")
        self.assertEqual(candidate.timeline[1].salient_point, "")
        self.assertIn("timeline[1].on_screen_text", candidate.missing_required_fields)
        self.assertIn("timeline[2].salient_point", candidate.missing_required_fields)
        self.assertNotIn("timeline[1].start", candidate.missing_required_fields)
        self.assertNotIn("timeline[2].start", candidate.missing_required_fields)
        self.assertNotIn("timeline[2].end", candidate.missing_required_fields)
        self.assertIn("required_fields", validate_candidate(candidate, self.config))
        rules = {item["rule"] for item in candidate.field_recoveries}
        self.assertIn("field_alias_v1", rules)
        self.assertIn("missing_field_placeholder_v1", rules)
        self.assertIn("control_boundary_endpoint_v1", rules)
        self.assertIn("adjacent_boundary_propagation_v1", rules)
        self.assertEqual(
            candidate.unresolved_required_fields,
            candidate.missing_required_fields,
        )
        self.assertEqual(len(candidate.normalized_payload_sha256), 64)
        second, second_error = self.parse(payload)
        self.assertEqual(second_error, "")
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(
            second.normalized_payload_sha256,
            candidate.normalized_payload_sha256,
        )

    def test_conflicting_alias_values_are_rejected(self) -> None:
        candidate, error = self.parse(
            {
                "variant": "visual_story",
                "timeline": [
                    {
                        "start": 0,
                        "end": 15,
                        "narration": "First narration.",
                        "description": "Conflicting narration.",
                        "on_screen_text": "",
                        "salient_point": "",
                        "control_tags": [],
                    },
                    {
                        "start": 15,
                        "end": 30,
                        "narration": "Second narration.",
                        "on_screen_text": "",
                        "salient_point": "",
                        "control_tags": ["summary"],
                    },
                ],
            }
        )

        self.assertIsNone(candidate)
        self.assertIn("conflicting values", error)

    def test_conflicting_adjacent_boundaries_are_rejected(self) -> None:
        candidate, error = self.parse(
            {
                "variant": "visual_story",
                "timeline": [
                    {
                        "start": 0,
                        "end": 14,
                        "narration": "First narration.",
                        "on_screen_text": "",
                        "salient_point": "",
                        "control_tags": [],
                    },
                    {
                        "start": 15,
                        "end": 30,
                        "narration": "Second narration.",
                        "on_screen_text": "",
                        "salient_point": "",
                        "control_tags": ["summary"],
                    },
                ],
            }
        )

        self.assertIsNone(candidate)
        self.assertIn("adjacent timeline boundaries conflict", error)

    def test_missing_narration_is_not_generated(self) -> None:
        candidate, error = self.parse(
            {
                "variant": "visual_story",
                "timeline": [
                    {
                        "start": 0,
                        "end": 15,
                        "on_screen_text": "Visible title",
                        "salient_point": "visible event",
                        "control_tags": [],
                    },
                    {
                        "start": 15,
                        "end": 30,
                        "narration": "The final view summarizes the sequence.",
                        "on_screen_text": "",
                        "salient_point": "",
                        "control_tags": ["summary"],
                    },
                ],
            }
        )

        self.assertEqual(error, "")
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.timeline[0].narration, "")
        self.assertIn("timeline[1].narration", candidate.missing_required_fields)
        self.assertNotIn(
            "timeline[1].narration",
            {item["path"] for item in candidate.field_recoveries},
        )
        self.assertIn("required_fields", validate_candidate(candidate, self.config))

    def test_generation_provenance_records_recovery_without_semantic_repair(self) -> None:
        endpoint = ModelEndpointConfig(
            id="local-vlm",
            name="Local VLM",
            role="main_multimodal_generator",
            provider="openai_compatible",
            adapter="chat_completions_multimodal",
            endpoint_url="http://127.0.0.1:9999/v1/chat/completions",
            max_frames=1,
            retry_count=0,
            seed=42,
        )
        model = OpenAICompatibleScriptGenerationModel(endpoint)
        config = GenerationConfig(
            num_candidates=1,
            segments=1,
            target_duration_sec=30,
            parse_retry_count=0,
        )
        content = json.dumps(
            {
                "variant": "visual_story",
                "segments": [
                    {
                        "description": "A person crosses the visible room.",
                        "tags": ["summary"],
                    }
                ],
            }
        )
        response = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
        batch = SparseFrameBatch(
            frames=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            selected_indices=[0],
            bin_ids=[0],
            source_kind="video",
        )

        with patch(
            "creative_video_exp.modeling._post_chat_completion",
            return_value=response,
        ):
            candidates = model.generate(self.sample, batch, config)

        self.assertEqual(len(candidates), 1)
        provenance = candidates[0].generation_provenance
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertTrue(provenance["field_recovery_applied"])
        self.assertFalse(provenance["semantic_repair_applied"])
        self.assertIn(
            "timeline[1].on_screen_text",
            provenance["unresolved_required_fields"],
        )
        self.assertEqual(len(provenance["normalized_payload_sha256"]), 64)
        self.assertEqual(
            provenance["normalized_payload_sha256"],
            model.model_report["last_generation_trace"][0]["attempts"][0][
                "normalized_payload_sha256"
            ],
        )


if __name__ == "__main__":
    unittest.main()
