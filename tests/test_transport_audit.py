from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.config import GenerationConfig, ModelEndpointConfig
from creative_video_exp.data import VideoSample
from creative_video_exp.modeling import OpenAICompatibleScriptGenerationModel
from creative_video_exp.video import SparseFrameBatch


class _HTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TransportAuditTest(unittest.TestCase):
    def test_hidden_transport_retry_is_visible_without_changing_semantic_attempt(self) -> None:
        endpoint = self._endpoint(retry_count=1)
        model = OpenAICompatibleScriptGenerationModel(endpoint)
        response = self._response(
            self._valid_candidate(),
            model="gpt-5-2026-08-07",
            server_metrics={"peak_allocated_mib": 1200.0},
        )

        with patch(
            "creative_video_exp.modeling.urllib.request.urlopen",
            side_effect=[
                urllib.error.URLError("transient connection failure"),
                _HTTPResponse(response),
            ],
        ) as urlopen, patch("creative_video_exp.modeling.time.sleep", return_value=None):
            candidates = model.generate(
                VideoSample(video_id="transport-retry"),
                self._batch(),
                self._config(parse_retry_count=0),
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(urlopen.call_count, 2)


        request_payloads = [
            json.loads(call.args[0].data.decode("utf-8"))
            for call in urlopen.call_args_list
        ]
        self.assertEqual(request_payloads[0], request_payloads[1])

        slot = model.model_report["last_generation_trace"][0]
        self.assertEqual(slot["request_count"], 1)
        self.assertEqual(slot["transport_request_count"], 2)
        self.assertEqual(len(slot["attempts"]), 1)
        semantic_attempt = slot["attempts"][0]
        self.assertEqual(semantic_attempt["status"], "accepted")
        self.assertEqual(semantic_attempt["transport_request_count"], 2)
        self.assertEqual(
            [item["status"] for item in semantic_attempt["transport_attempts"]],
            ["error", "success"],
        )
        self.assertEqual(
            semantic_attempt["transport_attempts"][0]["usage"],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            semantic_attempt["transport_attempts"][1]["usage"],
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(model.model_report["generation_semantic_attempts"], 1)
        self.assertEqual(model.model_report["generation_transport_requests"], 2)
        self.assertEqual(model.model_report["generation_requests"], 2)
        self.assertEqual(model.model_report["transport_error_count"], 1)
        self.assertEqual(
            slot["server_metrics"]["returned_model_id"],
            "gpt-5-2026-08-07",
        )

    def test_exhausted_transport_retries_remain_endpoint_infrastructure_error(self) -> None:
        endpoint = self._endpoint(retry_count=2)
        model = OpenAICompatibleScriptGenerationModel(endpoint)

        with patch(
            "creative_video_exp.modeling.urllib.request.urlopen",
            side_effect=urllib.error.URLError("server unavailable"),
        ) as urlopen, patch("creative_video_exp.modeling.time.sleep", return_value=None):
            candidates = model.generate(
                VideoSample(video_id="transport-failure"),
                self._batch(),
                self._config(parse_retry_count=0),
            )

        self.assertEqual(candidates, [])
        self.assertEqual(urlopen.call_count, 3)
        slot = model.model_report["last_generation_trace"][0]
        self.assertEqual(slot["terminal_status"], "failed")
        self.assertEqual(slot["terminal_reason"], "bounded_endpoint_attempts_exhausted")
        self.assertEqual(slot["request_count"], 1)
        self.assertEqual(slot["transport_request_count"], 3)
        attempt = slot["attempts"][0]
        self.assertEqual(attempt["status"], "endpoint_error")
        self.assertTrue(attempt["infrastructure_error"])
        self.assertEqual(attempt["usage"], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        self.assertEqual(len(attempt["transport_attempts"]), 3)
        self.assertTrue(
            all(item["status"] == "error" for item in attempt["transport_attempts"])
        )
        self.assertTrue(
            all(item["error"] for item in attempt["transport_attempts"])
        )
        self.assertEqual(model.model_report["generation_attempts"], 1)
        self.assertEqual(model.model_report["generation_requests"], 3)
        self.assertEqual(model.model_report["transport_error_count"], 3)

    def test_slot_server_metrics_use_max_peaks_across_all_semantic_attempts(self) -> None:
        endpoint = self._endpoint(retry_count=0)
        model = OpenAICompatibleScriptGenerationModel(endpoint)
        invalid_candidate = self._valid_candidate()
        invalid_candidate["candidate_id"] = "invalid"
        invalid_candidate["timeline"][0]["control_tags"] = ["Summary"]
        invalid = self._response(
            invalid_candidate,
            model="returned-checkpoint",
            server_metrics={
                "peak_allocated_mib": 900.0,
                "peak_reserved_mib": 1100.0,
                "degenerate_repetition_early_stop": True,
            },
        )
        valid = self._response(
            self._valid_candidate(),
            model="returned-checkpoint",
            server_metrics={
                "peak_allocated_mib": 700.0,
                "peak_reserved_mib": 1300.0,
                "degenerate_repetition_early_stop": False,
                "returned_model_id": "conflicting-server-field",
            },
        )

        with patch(
            "creative_video_exp.modeling._post_chat_completion",
            side_effect=[invalid, valid],
        ):
            candidates = model.generate(
                VideoSample(video_id="peak-aggregation"),
                self._batch(),
                self._config(parse_retry_count=1),
            )

        self.assertEqual(len(candidates), 1)
        slot = model.model_report["last_generation_trace"][0]
        metrics = slot["server_metrics"]
        self.assertEqual(metrics["peak_allocated_mib"], 900.0)
        self.assertEqual(metrics["peak_reserved_mib"], 1300.0)
        self.assertTrue(metrics["degenerate_repetition_early_stop"])
        self.assertEqual(metrics["server_metrics_response_count"], 2)
        self.assertEqual(metrics["returned_model_id"], "returned-checkpoint")
        self.assertTrue(metrics["returned_model_id_conflict"])
        self.assertEqual(
            metrics["server_metrics_returned_model_id"],
            "conflicting-server-field",
        )
        self.assertEqual(
            model.model_report["last_slot_server_metrics"],
            [metrics],
        )
        self.assertEqual(
            model.model_report["last_candidate_server_metrics"],
            [metrics],
        )

    @staticmethod
    def _endpoint(retry_count: int) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            id="audit-vlm",
            name="Audit VLM",
            role="main_multimodal_generator",
            provider="openai_compatible",
            adapter="chat_completions_multimodal",
            endpoint_url="http://127.0.0.1:9999/v1/chat/completions",
            max_frames=1,
            retry_count=retry_count,
            seed=42,
        )

    @staticmethod
    def _config(parse_retry_count: int) -> GenerationConfig:
        return GenerationConfig(
            num_candidates=1,
            segments=1,
            target_duration_sec=6,
            parse_retry_count=parse_retry_count,
        )

    @staticmethod
    def _batch() -> SparseFrameBatch:
        return SparseFrameBatch(
            frames=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            selected_indices=[0],
            bin_ids=[0],
            source_kind="video",
            metadata={"sampling_strategy": "binwise"},
        )

    @staticmethod
    def _valid_candidate() -> dict:
        return {
            "candidate_id": "candidate",
            "variant": "visual_story",
            "timeline": [
                {
                    "start": 0,
                    "end": 6,
                    "narration": "A person walks through the visible room.",
                    "on_screen_text": "",
                    "salient_point": "visible movement through the room",
                    "control_tags": ["summary"],
                }
            ],
        }

    @staticmethod
    def _response(
        content: dict,
        model: str,
        server_metrics: dict,
    ) -> dict:
        return {
            "model": model,
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "server_metrics": server_metrics,
        }


if __name__ == "__main__":
    unittest.main()
