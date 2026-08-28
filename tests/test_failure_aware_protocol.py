from __future__ import annotations

from copy import deepcopy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from creative_video_exp.checkpoint_identity import build_api_model_identity
from creative_video_exp.config import (
    ExperimentConfig,
    GenerationConfig,
    ModelEndpointConfig,
)
from creative_video_exp.data import VideoSample
from creative_video_exp.failure_aware_postprocessing import (
    analyze_failure_aware_candidate_pool,
)
from creative_video_exp.modeling import OpenAICompatibleScriptGenerationModel
from creative_video_exp.video import SparseFrameBatch
from run_experiment import (
    _build_slot_outcomes,
    _failure_aware_analysis_issue,
    _generation_failure_audit,
    _slot_accounting_issue,
    _validate_formal_checkpoint_identities,
    _validate_runtime_generation_checkpoint_identity,
    _validate_runtime_reward_checkpoint_identity,
)


class FailureAwareGenerationProtocolTest(unittest.TestCase):


    def test_exhausted_middle_slot_continues_later_fixed_slots(self) -> None:
        model = OpenAICompatibleScriptGenerationModel(self._endpoint())
        sample = VideoSample(video_id="failure-aware-middle-slot")
        batch = self._batch()
        config = self._generation_config()

        responses = [
            self._response(self._valid_candidate("slot-1", "A person enters the room.")),
            *[
                self._response(self._invalid_cjk_candidate(f"slot-2-attempt-{attempt}"))
                for attempt in range(1, 9)
            ],
            self._response(self._valid_candidate("slot-3", "A person sits beside the table.")),
            self._response(self._valid_candidate("slot-4", "A person leaves through the doorway.")),
        ]

        with patch(
            "creative_video_exp.modeling._post_chat_completion",
            side_effect=responses,
        ) as mocked:
            candidates = model.generate(sample, batch, config)

        self.assertEqual(mocked.call_count, 11)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            [
                candidate.generation_provenance["candidate_index"]
                for candidate in candidates
            ],
            [1, 3, 4],
        )

        trace = model.model_report["last_generation_trace"]
        self.assertEqual([slot["candidate_index"] for slot in trace], [1, 2, 3, 4])
        self.assertEqual([slot["accepted"] for slot in trace], [True, False, True, True])
        self.assertEqual([len(slot["attempts"]) for slot in trace], [1, 8, 1, 1])

        failed_slot = trace[1]
        self.assertTrue(
            all(
                attempt["status"] == "parse_rejected"
                for attempt in failed_slot["attempts"]
            )
        )
        self.assertTrue(
            all("欢迎" in attempt["raw_response"] for attempt in failed_slot["attempts"])
        )
        self.assertTrue(
            all(
                "English semantic-field contract" in attempt["parse_error"]
                for attempt in failed_slot["attempts"]
            )
        )

        accepted_text = "\n".join(candidate.text for candidate in candidates)
        self.assertNotIn("欢迎", accepted_text)
        self.assertNotIn("Welcome", accepted_text)
        self.assertTrue(
            all(
                candidate.generation_provenance["semantic_repair_applied"] is False
                for candidate in candidates
            )
        )

        base_hashes = {
            slot["attempts"][0]["prompt_sha256"]
            for slot in trace
        }
        retry_hashes = {
            attempt["prompt_sha256"]
            for attempt in failed_slot["attempts"][1:]
        }
        self.assertEqual(len(base_hashes), 1)
        self.assertEqual(len(retry_hashes), 1)
        self.assertNotEqual(next(iter(base_hashes)), next(iter(retry_hashes)))
        self.assertEqual(model.model_report["generation_requests"], 11)
        self.assertEqual(model.model_report["parsed_candidates"], 3)

        scored_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_index": candidate.generation_provenance["candidate_index"],
            }
            for candidate in candidates
        ]
        outcomes = _build_slot_outcomes(trace, scored_rows, expected_slots=4)
        self.assertEqual(
            [outcome["terminal_status"] for outcome in outcomes],
            ["valid", "failed", "valid", "valid"],
        )
        self.assertIsNone(outcomes[1]["candidate_id"])
        self.assertEqual(outcomes[1]["request_count"], 8)
        self.assertTrue(
            all("欢迎" in attempt["raw_response"] for attempt in outcomes[1]["attempts"])
        )

    def test_zero_valid_candidates_records_all_four_exhausted_slots(self) -> None:
        model = OpenAICompatibleScriptGenerationModel(self._endpoint())
        sample = VideoSample(video_id="failure-aware-zero-valid")
        batch = self._batch()
        config = self._generation_config()
        responses = [
            self._response(
                self._invalid_cjk_candidate(
                    f"slot-{slot}-attempt-{attempt}"
                )
            )
            for slot in range(1, 5)
            for attempt in range(1, 9)
        ]

        with patch(
            "creative_video_exp.modeling._post_chat_completion",
            side_effect=responses,
        ) as mocked:
            candidates = model.generate(sample, batch, config)

        self.assertEqual(candidates, [])
        self.assertEqual(mocked.call_count, 32)

        trace = model.model_report["last_generation_trace"]
        self.assertEqual([slot["candidate_index"] for slot in trace], [1, 2, 3, 4])
        self.assertTrue(all(slot["accepted"] is False for slot in trace))
        self.assertTrue(all(len(slot["attempts"]) == 8 for slot in trace))
        self.assertTrue(
            all(
                attempt["status"] == "parse_rejected"
                and "欢迎" in attempt["raw_response"]
                for slot in trace
                for attempt in slot["attempts"]
            )
        )
        self.assertEqual(model.model_report["generation_requests"], 32)
        self.assertEqual(model.model_report["generation_attempts"], 32)
        self.assertEqual(model.model_report["parse_failures"], 32)
        self.assertEqual(model.model_report["parsed_candidates"], 0)
        self.assertEqual(model.model_report["last_candidate_generation_seconds"], [])
        self.assertEqual(model.model_report["last_candidate_usage"], [])
        self.assertEqual(
            model.model_report["last_generation_usage"],
            {"input_tokens": 320, "output_tokens": 640, "total_tokens": 960},
        )
        outcomes = _build_slot_outcomes(trace, [], expected_slots=4)
        self.assertEqual(len(outcomes), 4)
        self.assertTrue(
            all(outcome["terminal_status"] == "failed" for outcome in outcomes)
        )
        self.assertTrue(all(outcome["candidate_id"] is None for outcome in outcomes))
        self.assertTrue(all(outcome["request_count"] == 8 for outcome in outcomes))

    @staticmethod
    def _endpoint() -> ModelEndpointConfig:
        return ModelEndpointConfig(
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
    def _generation_config() -> GenerationConfig:
        return GenerationConfig(
            num_candidates=4,
            segments=1,
            target_duration_sec=6,
            parse_retry_count=7,
        )

    @staticmethod
    def _valid_candidate(candidate_id: str, narration: str) -> str:
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "variant": "visual_story",
                "timeline": [
                    {
                        "start": 0,
                        "end": 6,
                        "narration": narration,
                        "on_screen_text": "",
                        "salient_point": "visible physical action",
                        "control_tags": ["summary"],
                    }
                ],
            }
        )

    @staticmethod
    def _invalid_cjk_candidate(candidate_id: str) -> str:
        return json.dumps(
            {
                "candidate_id": candidate_id,
                "variant": "visual_story",
                "timeline": [
                    {
                        "start": 0,
                        "end": 6,
                        "narration": "A visible sign reads 欢迎 beside the doorway.",
                        "on_screen_text": "",
                        "salient_point": "visible doorway",
                        "control_tags": ["summary"],
                    }
                ],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _response(content: str) -> dict:
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "server_metrics": {"degenerate_repetition_early_stop": False},
        }


class FailureAwareCandidatePoolAnalysisTest(unittest.TestCase):


    def test_direct_equals_best1_in_both_success_and_failure_states(self) -> None:
        success = self._row(
            "success-c1",
            {
                1: self._candidate("success-c1-c1", 1, reward=0.3),
                2: None,
                3: self._candidate("success-c1-c3", 3, reward=0.5),
                4: None,
            },
        )
        failure = self._row(
            "failure-c1",
            {
                1: None,
                2: self._candidate("failure-c1-c2", 2, reward=0.4),
                3: None,
                4: self._candidate("failure-c1-c4", 4, reward=0.8),
            },
        )

        summary, selections = analyze_failure_aware_candidate_pool(
            [success, failure],
            self._reward_config(),
            bootstrap_replicates=0,
        )

        for selection in selections:
            self.assertEqual(
                selection["methods"]["direct_generation_c1"],
                selection["methods"]["best_of_1"],
            )

        success_direct = selections[0]["methods"]["direct_generation_c1"]
        self.assertEqual(success_direct["selection_status"], "success")
        self.assertEqual(success_direct["candidate_index"], 1)
        self.assertEqual(success_direct["candidate_id"], "success-c1-c1")

        failed_direct = selections[1]["methods"]["direct_generation_c1"]
        self.assertEqual(failed_direct["selection_status"], "failure")
        self.assertIsNone(failed_direct["candidate_index"])
        self.assertIsNone(failed_direct["candidate_id"])
        self.assertIsNone(failed_direct["conditional_metrics"])
        self.assertEqual(failed_direct["effective_metrics"]["reward"], 0.0)
        self.assertEqual(failed_direct["effective_metrics"]["rep"], 1.0)

        invariant = summary["baseline_best_of_1_invariant"]
        self.assertTrue(invariant["identical_candidate_ids"])
        self.assertTrue(invariant["identical_failure_states"])
        self.assertEqual(
            summary["methods"]["direct_generation_c1"],
            summary["methods"]["best_of_1"],
        )

    def test_sparse_best_of_n_uses_original_requested_prefixes(self) -> None:
        row = self._row(
            "sparse-prefix",
            {
                1: self._candidate("c1", 1, reward=0.2),
                2: None,
                3: self._candidate("c3", 3, reward=0.7),
                4: self._candidate("c4", 4, reward=0.6),
            },
        )

        summary, selections = analyze_failure_aware_candidate_pool(
            [row],
            self._reward_config(),
            bootstrap_replicates=0,
        )
        methods = selections[0]["methods"]

        self.assertEqual(methods["best_of_1"]["valid_slot_indices"], [1])
        self.assertEqual(methods["best_of_1"]["candidate_id"], "c1")
        self.assertEqual(methods["best_of_2"]["valid_slot_indices"], [1])
        self.assertEqual(methods["best_of_2"]["candidate_id"], "c1")
        self.assertFalse(methods["best_of_2"]["exact_prefix_complete"])
        self.assertEqual(methods["stars_best_of_4"]["valid_slot_indices"], [1, 3, 4])
        self.assertEqual(methods["stars_best_of_4"]["candidate_id"], "c3")
        self.assertEqual(methods["stars_best_of_4"]["candidate_index"], 3)

        rewards = [
            methods[name]["effective_metrics"]["reward"]
            for name in ("best_of_1", "best_of_2", "stars_best_of_4")
        ]
        self.assertEqual(rewards, [0.2, 0.2, 0.7])
        self.assertTrue(summary["nested_reward_monotonicity"]["verified"])
        self.assertEqual(summary["valid_pool_size_histogram"], {"3": 1})

    def test_effective_metrics_penalize_method_failure_on_fixed_manifest(self) -> None:
        success = self._row(
            "effective-success",
            {
                1: self._candidate(
                    "effective-success-c1",
                    1,
                    reward=0.8,
                    mv_align=0.6,
                    csr=1.0,
                    spc=0.4,
                    cider=0.2,
                    rep=0.25,
                ),
                2: None,
                3: None,
                4: None,
            },
        )
        zero_valid = self._row(
            "effective-failure",
            {1: None, 2: None, 3: None, 4: None},
        )

        summary, _ = analyze_failure_aware_candidate_pool(
            [success, zero_valid],
            self._reward_config(),
            bootstrap_replicates=0,
        )
        best1 = summary["methods"]["best_of_1"]

        self.assertEqual(best1["successful_sample_count"], 1)
        self.assertEqual(best1["failed_sample_count"], 1)
        self.assertEqual(best1["selection_success_rate"], 0.5)
        self.assertEqual(best1["conditional_means"]["reward"], 0.8)
        self.assertEqual(best1["conditional_means"]["mv_align"], 0.6)
        self.assertEqual(best1["conditional_means"]["rep"], 0.25)
        self.assertEqual(best1["effective_means"]["reward"], 0.4)
        self.assertEqual(best1["effective_means"]["mv_align"], 0.3)
        self.assertEqual(best1["effective_means"]["csr"], 0.5)
        self.assertEqual(best1["effective_means"]["semantic_point_coverage"], 0.2)
        self.assertEqual(best1["effective_means"]["cider_lite"], 0.1)
        self.assertEqual(best1["effective_means"]["rep"], 0.625)
        self.assertEqual(summary["failure_penalties"]["reward"], 0.0)
        self.assertEqual(summary["failure_penalties"]["rep"], 1.0)

    def test_zero_valid_pool_is_a_complete_audited_failure_outcome(self) -> None:
        row = self._row(
            "zero-valid",
            {1: None, 2: None, 3: None, 4: None},
            failed_attempts=8,
        )

        summary, selections = analyze_failure_aware_candidate_pool(
            [row],
            self._reward_config(),
            bootstrap_replicates=0,
        )

        self.assertEqual(summary["requested_sample_count"], 1)
        self.assertEqual(summary["requested_slot_count"], 4)
        self.assertEqual(summary["valid_slot_count"], 0)
        self.assertEqual(summary["failed_slot_count"], 4)
        self.assertEqual(summary["any_valid_sample_count"], 0)
        self.assertEqual(summary["valid_pool_size_histogram"], {"0": 1})

        for method in summary["methods"].values():
            self.assertEqual(method["successful_sample_count"], 0)
            self.assertEqual(method["failed_sample_count"], 1)
            self.assertEqual(method["selection_success_rate"], 0.0)
            self.assertTrue(
                all(value is None for value in method["conditional_means"].values())
            )
            self.assertEqual(method["effective_means"]["reward"], 0.0)
            self.assertEqual(method["effective_means"]["rep"], 1.0)

        stars = selections[0]["methods"]["stars_best_of_4"]
        self.assertEqual(stars["selection_status"], "failure")
        self.assertEqual(stars["valid_slot_indices"], [])
        self.assertEqual(stars["efficiency"]["request_count"], 32.0)
        self.assertEqual(stars["efficiency"]["input_tokens"], 320.0)
        self.assertEqual(stars["efficiency"]["output_tokens"], 640.0)
        upper = selections[0]["metric_wise_best_of_4_upper_bound"]
        self.assertEqual(upper["selection_status"], "failure")
        self.assertEqual(upper["effective_pool_size"], 0)
        self.assertIsNone(upper["metrics"])
        self.assertEqual(upper["effective_metrics"]["reward"], 0.0)
        self.assertEqual(upper["effective_metrics"]["rep"], 1.0)

    def test_three_of_four_sparse_pool_remains_analyzable(self) -> None:
        row = self._row(
            "sparse-slot4-failed",
            {
                1: self._candidate("sparse-c1", 1, reward=0.4),
                2: self._candidate("sparse-c2", 2, reward=0.6),
                3: self._candidate("sparse-c3", 3, reward=0.7),
                4: None,
            },
            failed_attempts=8,
        )

        summary, selections = analyze_failure_aware_candidate_pool(
            [row],
            self._reward_config(),
            bootstrap_replicates=0,
        )

        self.assertEqual(summary["candidate_slot_success_rate"], 0.75)
        self.assertEqual(summary["full_pool_rate"], 0.0)
        self.assertEqual(summary["any_valid_sample_rate"], 1.0)
        stars_summary = summary["methods"]["stars_best_of_4"]
        self.assertEqual(stars_summary["selection_success_rate"], 1.0)
        self.assertEqual(stars_summary["exact_prefix_completion_rate"], 0.0)
        self.assertEqual(stars_summary["mean_valid_slots_in_prefix"], 3.0)
        self.assertEqual(stars_summary["valid_slots_in_prefix_histogram"], {"3": 1})

        stars = selections[0]["methods"]["stars_best_of_4"]
        self.assertEqual(stars["selection_status"], "success")
        self.assertEqual(stars["valid_slot_indices"], [1, 2, 3])
        self.assertEqual(stars["candidate_id"], "sparse-c3")
        self.assertFalse(stars["exact_prefix_complete"])
        upper = selections[0]["metric_wise_best_of_4_upper_bound"]
        self.assertEqual(upper["selection_status"], "success")
        self.assertEqual(upper["effective_pool_size"], 3)

    def test_run_gate_accepts_three_of_four_and_zero_of_four_as_accounted(self) -> None:
        partial = self._row(
            "sparse-accounted",
            {
                1: self._candidate("sparse-accounted-c1", 1, reward=0.4),
                2: self._candidate("sparse-accounted-c2", 2, reward=0.6),
                3: self._candidate("sparse-accounted-c3", 3, reward=0.7),
                4: None,
            },
            failed_attempts=8,
        )
        zero_valid = self._row(
            "zero-valid-accounted",
            {1: None, 2: None, 3: None, 4: None},
            failed_attempts=8,
        )

        with tempfile.TemporaryDirectory() as raw_dir:
            audit = _generation_failure_audit(
                failure_path=Path(raw_dir) / "generation_failures.jsonl",
                results=[partial, zero_valid],
                requested_sample_count=2,
                expected_sample_keys=[
                    "sparse-accounted",
                    "zero-valid-accounted",
                ],
                expected_protocol_fingerprint="failure-aware-test-fingerprint",
            )

        self.assertTrue(audit["evidence_eligible"])
        self.assertEqual(audit["completed_num_samples"], 2)
        self.assertEqual(audit["fully_accounted_sample_count"], 2)
        self.assertEqual(audit["slot_accounting_completion_rate"], 1.0)
        self.assertEqual(audit["requested_candidate_slot_count"], 8)
        self.assertEqual(audit["accounted_candidate_slot_count"], 8)
        self.assertEqual(audit["valid_candidate_slot_count"], 3)
        self.assertEqual(audit["failed_candidate_slot_count"], 5)
        self.assertEqual(audit["full_candidate_pool_rate"], 0.0)
        self.assertEqual(audit["samples_with_any_valid_candidate_rate"], 0.5)
        self.assertEqual(audit["generation_success_rate"], 0.5)
        self.assertEqual(audit["unresolved_generation_failure_count"], 0)
        self.assertEqual(audit["slot_accounting_issues"], [])

    def test_run_gate_rejects_a_missing_terminal_slot(self) -> None:
        incomplete = self._row(
            "missing-terminal-slot",
            {
                1: self._candidate("missing-terminal-slot-c1", 1, reward=0.5),
                2: None,
                3: None,
                4: None,
            },
        )
        incomplete["slot_outcomes"].pop()

        issue = _slot_accounting_issue(incomplete, expected_slots=4)
        self.assertIn("expected 4 slot outcomes", issue)
        with tempfile.TemporaryDirectory() as raw_dir:
            audit = _generation_failure_audit(
                failure_path=Path(raw_dir) / "generation_failures.jsonl",
                results=[incomplete],
                requested_sample_count=1,
                expected_sample_keys=["missing-terminal-slot"],
                expected_protocol_fingerprint="failure-aware-test-fingerprint",
            )

        self.assertFalse(audit["evidence_eligible"])
        self.assertEqual(audit["fully_accounted_sample_count"], 0)
        self.assertEqual(audit["unresolved_generation_failure_count"], 1)
        self.assertEqual(len(audit["slot_accounting_issues"]), 1)

    def test_runner_terminal_audit_rejects_malformed_complete_rows(self) -> None:
        valid = self._row(
            "strict-terminal-row",
            {
                index: self._candidate(
                    f"strict-terminal-row-c{index}", index, reward=0.5
                )
                for index in range(1, 5)
            },
        )
        self.assertEqual(_slot_accounting_issue(valid, expected_slots=4), "")

        mutations = []
        declared = deepcopy(valid)
        declared["valid_candidate_slots"] = 3
        mutations.append((declared, "valid_candidate_slots"))

        no_reason = deepcopy(valid)
        no_reason["slot_outcomes"][0]["terminal_reason"] = ""
        mutations.append((no_reason, "terminal_reason"))

        wrong_join = deepcopy(valid)
        wrong_join["slot_outcomes"][0]["candidate_id"] = "wrong-candidate"
        mutations.append((wrong_join, "join exactly"))

        bad_usage = deepcopy(valid)
        bad_usage["slot_outcomes"][0]["usage"]["total_tokens"] += 1
        mutations.append((bad_usage, "usage does not match"))

        bad_latency = deepcopy(valid)
        bad_latency["slot_outcomes"][0]["request_seconds"] += 1.0
        mutations.append((bad_latency, "request_seconds"))

        exhausted = self._row(
            "strict-failed-row",
            {1: None, 2: None, 3: None, 4: None},
            failed_attempts=8,
        )
        exhausted["slot_outcomes"][0]["attempts"].pop()
        exhausted["slot_outcomes"][0]["request_count"] = 7
        exhausted["slot_outcomes"][0]["request_seconds"] = 7 * 0.25
        exhausted["slot_outcomes"][0]["transport_request_count"] = 7
        exhausted["slot_outcomes"][0]["transport_request_seconds"] = 7 * 0.25
        exhausted["slot_outcomes"][0]["usage"] = {
            "input_tokens": 70,
            "output_tokens": 140,
            "total_tokens": 210,
        }
        mutations.append((exhausted, "fixed 8-attempt bound"))

        for row, expected_message in mutations:
            with self.subTest(expected_message=expected_message):
                self.assertIn(
                    expected_message,
                    _slot_accounting_issue(row, expected_slots=4),
                )

    def test_any_nested_transport_error_makes_formal_run_ineligible(self) -> None:
        row = self._row(
            "transport-retry-infra",
            {
                index: self._candidate(
                    f"transport-retry-infra-c{index}", index, reward=0.5
                )
                for index in range(1, 5)
            },
        )
        attempt = row["slot_outcomes"][0]["attempts"][0]
        attempt["transport_attempts"] = [
            {
                "transport_attempt": 1,
                "status": "error",
                "request_seconds": 0.04,
                "backoff_seconds_after": 0.01,
                "error_type": "TimeoutError",
                "error": "timed out",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
            {
                "transport_attempt": 2,
                "status": "success",
                "request_seconds": 0.20,
                "backoff_seconds_after": 0.0,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            },
        ]
        attempt["transport_request_count"] = 2
        attempt["transport_request_seconds"] = 0.24
        attempt["transport_backoff_seconds"] = 0.01
        attempt["request_seconds"] = 0.25
        slot = row["slot_outcomes"][0]
        slot["transport_request_count"] = 2
        slot["transport_request_seconds"] = 0.24
        slot["transport_backoff_seconds"] = 0.01
        slot["request_seconds"] = 0.25
        self.assertEqual(_slot_accounting_issue(row, expected_slots=4), "")

        with tempfile.TemporaryDirectory() as raw_dir:
            audit = _generation_failure_audit(
                failure_path=Path(raw_dir) / "generation_failures.jsonl",
                results=[row],
                requested_sample_count=1,
                expected_sample_keys=["transport-retry-infra"],
                expected_protocol_fingerprint="failure-aware-test-fingerprint",
            )

        self.assertFalse(audit["evidence_eligible"])
        self.assertFalse(audit["formal_infrastructure_valid"])
        self.assertEqual(audit["transport_error_attempt_count"], 1)
        self.assertEqual(audit["semantic_endpoint_error_attempt_count"], 0)
        self.assertEqual(audit["infrastructure_issue_count"], 1)
        self.assertEqual(
            audit["infrastructure_issue_sample_keys"],
            ["transport-retry-infra"],
        )

    def test_canonical_analysis_must_close_before_final_eligibility(self) -> None:
        row = self._row(
            "canonical-close",
            {
                index: self._candidate(
                    f"canonical-close-c{index}", index, reward=index / 10
                )
                for index in range(1, 5)
            },
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            audit = _generation_failure_audit(
                failure_path=Path(raw_dir) / "generation_failures.jsonl",
                results=[row],
                requested_sample_count=1,
                expected_sample_keys=["canonical-close"],
                expected_protocol_fingerprint="failure-aware-test-fingerprint",
            )
        summary, selections = analyze_failure_aware_candidate_pool(
            [row], self._reward_config(), bootstrap_replicates=0
        )
        self.assertEqual(
            _failure_aware_analysis_issue(
                summary,
                selections,
                [row],
                audit,
                "failure-aware-test-fingerprint",
            ),
            "",
        )
        tampered = deepcopy(summary)
        tampered["valid_candidate_slot_count"] = 3
        self.assertIn(
            "valid_candidate_slot_count",
            _failure_aware_analysis_issue(
                tampered,
                selections,
                [row],
                audit,
                "failure-aware-test-fingerprint",
            ),
        )

    def test_formal_checkpoint_identity_guard_and_disabled_exception(self) -> None:
        disabled = ExperimentConfig()
        _validate_formal_checkpoint_identities(disabled)

        local_identity = {
            "schema_version": "starsv18_checkpoint_manifest_v1",
            "kind": "local_checkpoint",
            "model_id": "Local-Video",
            "revision": "immutable-revision",
            "identity_sha256": "a" * 64,
            "file_count": 1,
            "total_bytes": 1,
        }
        reward_identity = {
            **local_identity,
            "model_id": "Local-Reward",
            "identity_sha256": "b" * 64,
        }
        formal = deepcopy(disabled)
        formal.models.mode = "server_full_matrix"
        formal.models.active_video_model = "video"
        formal.models.active_reward_vision_model = "reward"
        formal.models.endpoints = [
            ModelEndpointConfig(
                id="video",
                name="Local-Video",
                role="main_multimodal_generator",
                provider="openai_compatible",
                checkpoint_identity=local_identity,
            ),
            ModelEndpointConfig(
                id="reward",
                name="Local-Reward",
                role="frozen_vision_text_reward_encoder",
                provider="huggingface_local",
                checkpoint_identity=reward_identity,
            ),
        ]
        _validate_formal_checkpoint_identities(formal)

        api = deepcopy(formal)
        api.models.endpoints[0].provider = "openai"
        api.models.endpoints[0].name = "gpt-5-2026-08-07"
        api.models.endpoints[0].checkpoint_identity = build_api_model_identity(
            "gpt-5-2026-08-07"
        )
        _validate_formal_checkpoint_identities(api)

        missing = deepcopy(formal)
        missing.models.endpoints[0].checkpoint_identity = {}
        with self.assertRaisesRegex(RuntimeError, "invalid checkpoint_identity"):
            _validate_formal_checkpoint_identities(missing)

        mismatch = deepcopy(api)
        mismatch.models.endpoints[0].name = "gpt-5-different"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_formal_checkpoint_identities(mismatch)

    def test_runtime_checkpoint_identities_are_bound_to_observed_models(self) -> None:
        disabled = ExperimentConfig()
        _validate_runtime_reward_checkpoint_identity(disabled, {})
        _validate_runtime_generation_checkpoint_identity(disabled, [])

        local_identity = {
            "schema_version": "starsv18_checkpoint_manifest_v1",
            "kind": "local_checkpoint",
            "model_id": "Local-Video",
            "revision": "immutable-video-revision",
            "identity_sha256": "a" * 64,
            "file_count": 1,
            "total_bytes": 1,
        }
        reward_identity = {
            **local_identity,
            "model_id": "Local-Reward",
            "revision": "immutable-reward-revision",
            "identity_sha256": "b" * 64,
        }
        formal = deepcopy(disabled)
        formal.models.mode = "server_full_matrix"
        formal.models.active_video_model = "video"
        formal.models.active_reward_vision_model = "reward"
        formal.models.endpoints = [
            ModelEndpointConfig(
                id="video",
                name="Local-Video",
                role="main_multimodal_generator",
                provider="openai_compatible",
                checkpoint_identity=local_identity,
            ),
            ModelEndpointConfig(
                id="reward",
                name="Local-Reward",
                role="frozen_vision_text_reward_encoder",
                provider="huggingface_local",
                checkpoint_identity=reward_identity,
            ),
        ]
        reward_report = {
            "checkpoint_identity": deepcopy(reward_identity),
            "checkpoint_identity_sha256": reward_identity["identity_sha256"],
        }
        _validate_runtime_reward_checkpoint_identity(formal, reward_report)

        local_server_metrics = {
            "returned_model_id": "Local-Video",
            "checkpoint_identity_sha256": local_identity["identity_sha256"],
            "checkpoint_model_id": local_identity["model_id"],
            "checkpoint_revision": local_identity["revision"],
        }
        local_trace = [
            {
                "candidate_index": 1,
                "attempts": [
                    {
                        "status": "parse_rejected",
                        "server_metrics": deepcopy(local_server_metrics),
                    },
                    {
                        "status": "accepted",
                        "server_metrics": deepcopy(local_server_metrics),
                    },
                ],
            }
        ]
        _validate_runtime_generation_checkpoint_identity(formal, local_trace)

        bad_reward = deepcopy(reward_report)
        bad_reward["checkpoint_identity_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "reward checkpoint fingerprint"):
            _validate_runtime_reward_checkpoint_identity(formal, bad_reward)

        bad_local_trace = deepcopy(local_trace)
        bad_local_trace[0]["attempts"][0]["server_metrics"][
            "checkpoint_revision"
        ] = "different-revision"
        with self.assertRaisesRegex(RuntimeError, "local generation checkpoint"):
            _validate_runtime_generation_checkpoint_identity(
                formal,
                bad_local_trace,
            )

        api = deepcopy(formal)
        api.models.endpoints[0].provider = "openai"
        api.models.endpoints[0].name = "gpt-5-2026-08-07"
        api.models.endpoints[0].checkpoint_identity = build_api_model_identity(
            "gpt-5-2026-08-07"
        )
        _validate_runtime_generation_checkpoint_identity(
            api,
            [
                {
                    "candidate_index": 1,
                    "attempts": [
                        {
                            "status": "accepted",
                            "server_metrics": {
                                "returned_model_id": "gpt-5-2026-08-07"
                            },
                        }
                    ],
                }
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "returned model identity"):
            _validate_runtime_generation_checkpoint_identity(
                api,
                [
                    {
                        "candidate_index": 1,
                        "attempts": [
                            {
                                "status": "accepted",
                                "server_metrics": {
                                    "returned_model_id": "gpt-5-wrong"
                                },
                            }
                        ],
                    }
                ],
            )

    @staticmethod
    def _reward_config() -> dict[str, float]:
        return {
            "alignment_weight": 0.35,
            "readability_weight": 0.18,
            "rhythm_weight": 0.18,
            "control_weight": 0.23,
            "risk_weight": 0.06,
        }

    @classmethod
    def _row(
        cls,
        sample_key: str,
        candidates_by_slot: dict[int, dict | None],
        failed_attempts: int = 1,
    ) -> dict:
        candidates = [
            candidate
            for candidate in candidates_by_slot.values()
            if candidate is not None
        ]
        slots = [
            cls._slot_outcome(
                index,
                candidates_by_slot[index],
                failed_attempts=failed_attempts,
            )
            for index in range(1, 5)
        ]
        valid_count = len(candidates)
        return {
            "experiment_version": "starsv20",
            "sample_key": sample_key,
            "video_id": sample_key,
            "protocol_fingerprint": "failure-aware-test-fingerprint",
            "generation_source": "multimodal_model",
            "requested_candidate_slots": 4,
            "valid_candidate_slots": valid_count,
            "failed_candidate_slots": 4 - valid_count,
            "slot_outcomes": slots,
            "candidates": candidates,
        }

    @staticmethod
    def _slot_outcome(
        candidate_index: int,
        candidate: dict | None,
        failed_attempts: int,
    ) -> dict:
        valid = candidate is not None
        request_count = 1 if valid else failed_attempts
        attempts = [
            {
                "attempt": attempt,
                "status": "accepted" if valid else "parse_rejected",
                "request_seconds": 0.25,
                "raw_response": (
                    "{\"timeline\": []}"
                    if valid
                    else '{"narration":"欢迎"}'
                ),
                "parse_error": (
                    "" if valid else "English semantic-field contract violation"
                ),
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
                "transport_attempts": [
                    {
                        "transport_attempt": 1,
                        "status": "success",
                        "request_seconds": 0.25,
                        "backoff_seconds_after": 0.0,
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "total_tokens": 30,
                        },
                    }
                ],
                "transport_request_count": 1,
                "transport_request_seconds": 0.25,
                "transport_backoff_seconds": 0.0,
            }
            for attempt in range(1, request_count + 1)
        ]
        input_tokens = request_count * 10
        output_tokens = request_count * 20
        return {
            "candidate_index": candidate_index,
            "terminal_status": "valid" if valid else "failed",
            "candidate_id": candidate["candidate_id"] if valid else None,
            "terminal_reason": (
                "accepted_valid_candidate"
                if valid
                else "bounded_validation_attempts_exhausted"
            ),
            "request_count": request_count,
            "request_seconds": request_count * 0.25,
            "transport_request_count": request_count,
            "transport_request_seconds": request_count * 0.25,
            "transport_backoff_seconds": 0.0,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "attempts": attempts,
        }

    @staticmethod
    def _candidate(
        candidate_id: str,
        candidate_index: int,
        reward: float,
        mv_align: float | None = None,
        csr: float = 1.0,
        spc: float = 0.25,
        cider: float = 0.2,
        rep: float = 0.1,
    ) -> dict:
        alignment = reward if mv_align is None else mv_align
        return {
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "reward": {
                "total": reward,
                "alignment": alignment,
                "readability": reward,
                "rhythm": reward,
                "control": csr,
                "risk": reward,
                "violations": [] if csr == 1.0 else ["test_control_violation"],
                "control_violations": (
                    [] if csr == 1.0 else ["test_control_violation"]
                ),
            },
            "quality": {
                "semantic_point_coverage": spc,
                "cider_lite": cider,
                "repetition_rate": rep,
            },
            "generation_provenance": {
                "candidate_index": candidate_index,
                "parse_status": "valid_json_candidate",
                "semantic_repair_applied": False,
                "wrapped_free_text": False,
                "surrounding_free_text": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
