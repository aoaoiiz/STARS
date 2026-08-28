from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
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
    build_api_model_identity,
)
from creative_video_exp.config import ExperimentConfig
from run_experiment import _validate_formal_config
from run_server_matrix import (
    _query_local_service_checkpoint_identity,
    build_config,
)


class _HTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _identity(model_id: str, marker: str) -> dict:
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "kind": LOCAL_CHECKPOINT_KIND,
        "model_id": model_id,
        "revision": "fixture",
        "identity_sha256": marker * 64,
        "file_count": 1,
        "total_bytes": 1,
    }


def _load_config(payload: dict) -> ExperimentConfig:
    with tempfile.TemporaryDirectory() as raw_dir:
        path = Path(raw_dir) / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return ExperimentConfig.from_file(path)


class FormalConfigContractTest(unittest.TestCase):
    def setUp(self) -> None:
        args = SimpleNamespace(output_root="outputs/fixture")
        environment = {
            "LONGBENCH_ANNOTATION_PATH": "/annotations/manifest.json",
            "LONGBENCH_VIDEO_ROOT": "/videos",
            "LLAVA_VIDEO_ENDPOINT_URL": "http://127.0.0.1:8010/v1/chat/completions",
            "REWARD_VISION_MODEL_PATH": "/models/siglip2",
        }
        with patch.dict(os.environ, environment, clear=False):
            self.payload = build_config(
                "longvideobench",
                "llava_video_qwen2",
                args,
                model_identity=_identity("LLaVA-Video-7B-Qwen2", "a"),
                model_manifest_path="generator.json",
                reward_identity=_identity(
                    "google/siglip2-so400m-patch14-384",
                    "b",
                ),
                reward_manifest_path="reward.json",
            )

    def test_matrix_config_satisfies_the_contract(self) -> None:
        _validate_formal_config(_load_config(self.payload))

    def test_seed_mutation_is_rejected(self) -> None:
        payload = deepcopy(self.payload)
        payload["seed"] = 99
        with self.assertRaisesRegex(RuntimeError, "seed=42"):
            _validate_formal_config(_load_config(payload))

    def test_reward_mutation_is_rejected(self) -> None:
        payload = deepcopy(self.payload)
        payload["reward"]["alignment_weight"] = 0.4
        with self.assertRaisesRegex(RuntimeError, "alignment_weight"):
            _validate_formal_config(_load_config(payload))

    def test_generation_and_evaluation_mutations_are_rejected(self) -> None:
        payload = deepcopy(self.payload)
        payload["models"]["endpoints"][0]["temperature"] = 1.7
        with self.assertRaisesRegex(RuntimeError, "temperature=0.3"):
            _validate_formal_config(_load_config(payload))
        payload = deepcopy(self.payload)
        payload["evaluation"]["semantic_point_similarity_threshold"] = 0.99
        with self.assertRaisesRegex(RuntimeError, "similarity_threshold=0.50"):
            _validate_formal_config(_load_config(payload))

    def test_live_service_protocol_is_audited(self) -> None:
        identity = _identity("LLaVA-Video-7B-Qwen2", "c")
        payload = {
            "data": [
                {
                    "id": "LLaVA-Video-7B-Qwen2",
                    "checkpoint_identity": identity,
                    "service_protocol": {
                        "max_frames": 16,
                        "model_name": "llava_qwen",
                    },
                }
            ]
        }
        with patch(
            "run_server_matrix.urllib.request.urlopen",
            return_value=_HTTPResponse(payload),
        ):
            observed = _query_local_service_checkpoint_identity(
                "http://127.0.0.1:8010/v1/chat/completions",
                api_key_env="",
                expected_model_id="LLaVA-Video-7B-Qwen2",
                expected_service_protocol={
                    "max_frames": 16,
                    "model_name": "llava_qwen",
                },
            )
        self.assertEqual(observed, identity)

        payload["data"][0]["service_protocol"]["max_frames"] = 8
        with patch(
            "run_server_matrix.urllib.request.urlopen",
            return_value=_HTTPResponse(payload),
        ), self.assertRaisesRegex(RuntimeError, "service protocol"):
            _query_local_service_checkpoint_identity(
                "http://127.0.0.1:8010/v1/chat/completions",
                api_key_env="",
                expected_model_id="LLaVA-Video-7B-Qwen2",
                expected_service_protocol={
                    "max_frames": 16,
                    "model_name": "llava_qwen",
                },
            )

    def test_external_api_generator_is_rejected(self) -> None:
        payload = deepcopy(self.payload)
        endpoint = payload["models"]["endpoints"][0]
        endpoint["name"] = "external-api-model"
        endpoint["provider"] = "openai"
        endpoint["adapter"] = "openai_responses_multimodal"
        endpoint["checkpoint_identity"] = build_api_model_identity(
            "external-api-model"
        )
        with self.assertRaisesRegex(RuntimeError, "three local STARS backbones"):
            _validate_formal_config(_load_config(payload))


if __name__ == "__main__":
    unittest.main()
