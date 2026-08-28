from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.failure_aware_postprocessing import (
    analyze_failure_aware_candidate_pool,
    write_failure_aware_candidate_pool_analysis,
)


REWARD_CONFIG = {
    "alignment_weight": 6.0 / 17.0,
    "readability_weight": 3.0 / 17.0,
    "rhythm_weight": 3.0 / 17.0,
    "control_weight": 4.0 / 17.0,
    "risk_weight": 1.0 / 17.0,
}


def _candidate(
    sample: str,
    index: int,
    reward: float,
    *,
    alignment: float | None = None,
    csr: float = 1.0,
    spc: float = 0.5,
    cider: float = 0.4,
    rep: float = 0.2,
) -> dict:
    return {
        "candidate_id": f"{sample}_c{index}",
        "candidate_index": index,
        "variant": "visual_story",
        "reward": {
            "total": reward,
            "alignment": reward if alignment is None else alignment,
            "readability": reward,
            "rhythm": reward,
            "control": csr,
            "risk": 1.0,
            "violations": [] if csr == 1.0 else ["test_violation"],
        },
        "quality": {
            "semantic_point_coverage": spc,
            "cider_lite": cider,
            "repetition_rate": rep,
        },
        "generation_provenance": {
            "parse_status": "valid_json_candidate",
            "semantic_repair_applied": False,
            "wrapped_free_text": False,
            "surrounding_free_text": False,
        },
    }


def _row(
    sample: str,
    statuses: list[str],
    rewards: dict[int, float] | None = None,
    candidates: dict[int, dict] | None = None,
) -> dict:
    rewards = rewards or {}
    by_index = dict(candidates or {})
    for index, reward in rewards.items():
        by_index[index] = _candidate(sample, index, reward)
    slot_outcomes = []
    for index, status in enumerate(statuses, start=1):
        valid = status == "valid"
        slot_outcomes.append(
            {
                "candidate_index": index,
                "terminal_status": status,
                "candidate_id": by_index[index]["candidate_id"] if valid else None,
                "terminal_reason": "accepted" if valid else "retry_budget_exhausted",
                "request_count": 1,
                "request_seconds": float(index),
                "usage": {
                    "input_tokens": 10 * index,
                    "output_tokens": 5 * index,
                    "total_tokens": 15 * index,
                },
                "attempts": [{"attempt": 1}],
            }
        )
    valid_count = sum(status == "valid" for status in statuses)
    return {
        "sample_key": sample,
        "video_id": sample,
        "protocol_fingerprint": "frozen-fingerprint",
        "requested_candidate_slots": 4,
        "valid_candidate_slots": valid_count,
        "failed_candidate_slots": 4 - valid_count,
        "slot_outcomes": slot_outcomes,
        "candidates": [by_index[index] for index in sorted(by_index)],
    }


class FailureAwareCandidatePoolTest(unittest.TestCase):
    def test_prefix_selection_keeps_failures_and_direct_equals_best1(self) -> None:
        complete = _row(
            "complete",
            ["valid"] * 4,
            rewards={1: 0.2, 2: 0.8, 3: 0.5, 4: 0.9},
        )
        partial = _row(
            "partial",
            ["failed", "valid", "failed", "valid"],
            rewards={2: 0.4, 4: 0.7},
        )

        summary, selections = analyze_failure_aware_candidate_pool(
            [complete, partial],
            REWARD_CONFIG,
            bootstrap_replicates=20,
        )

        self.assertEqual(summary["requested_sample_count"], 2)
        self.assertEqual(summary["requested_candidate_slot_count"], 8)
        self.assertEqual(summary["valid_candidate_slot_count"], 6)
        self.assertEqual(summary["failed_candidate_slot_count"], 2)
        self.assertEqual(summary["full_candidate_pool_sample_count"], 1)
        self.assertEqual(summary["full_candidate_pool_sample_rate"], 0.5)
        self.assertEqual(summary["any_valid_candidate_sample_count"], 2)
        self.assertEqual(summary["any_valid_candidate_sample_rate"], 1.0)
        self.assertEqual(summary["requested_slot_count"], 8)
        self.assertEqual(summary["valid_slot_count"], 6)
        self.assertEqual(summary["failed_slot_count"], 2)
        self.assertEqual(summary["valid_pool_size_histogram"], {"2": 1, "4": 1})

        direct = summary["methods"]["direct_generation_c1"]
        best_one = summary["methods"]["best_of_1"]
        self.assertEqual(direct, best_one)
        self.assertEqual(direct["successful_sample_count"], 1)
        self.assertEqual(direct["selection_success_rate"], 0.5)
        self.assertEqual(direct["conditional_means"]["reward"], 0.2)
        self.assertEqual(direct["effective_means"]["reward"], 0.1)
        self.assertEqual(direct["effective_means"]["rep"], 0.6)

        best_two = summary["methods"]["best_of_2"]
        stars = summary["methods"]["stars_best_of_4"]
        self.assertEqual(best_two["selection_success_rate"], 1.0)
        self.assertEqual(best_two["effective_means"]["reward"], 0.6)
        self.assertEqual(stars["effective_means"]["reward"], 0.8)
        self.assertLessEqual(
            best_one["effective_means"]["reward"],
            best_two["effective_means"]["reward"],
        )
        self.assertLessEqual(
            best_two["effective_means"]["reward"],
            stars["effective_means"]["reward"],
        )

        partial_methods = selections[1]["methods"]
        self.assertEqual(partial_methods["direct_generation_c1"], partial_methods["best_of_1"])
        self.assertEqual(
            partial_methods["best_of_1"]["selection_status"], "failure"
        )
        self.assertIsNone(partial_methods["best_of_1"]["candidate_id"])
        self.assertIsNone(partial_methods["best_of_1"]["conditional_metrics"])
        self.assertEqual(
            partial_methods["best_of_1"]["effective_metrics"]["reward"], 0.0
        )
        self.assertEqual(
            partial_methods["best_of_1"]["effective_metrics"]["rep"], 1.0
        )
        self.assertEqual(
            partial_methods["best_of_2"]["candidate_id"], "partial_c2"
        )
        self.assertEqual(
            partial_methods["stars_best_of_4"]["candidate_id"], "partial_c4"
        )

    def test_zero_valid_pool_is_explicit_failure_with_worst_effective_penalty(self) -> None:
        row = _row("zero", ["failed"] * 4)
        summary, selections = analyze_failure_aware_candidate_pool(
            [row], REWARD_CONFIG, bootstrap_replicates=0
        )

        self.assertEqual(summary["any_valid_sample_count"], 0)
        self.assertEqual(summary["full_pool_sample_count"], 0)
        for method in summary["methods"].values():
            self.assertEqual(method["successful_sample_count"], 0)
            self.assertIsNone(method["conditional_means"]["reward"])
            self.assertEqual(method["effective_means"]["reward"], 0.0)
            self.assertEqual(method["effective_means"]["rep"], 1.0)

        upper = selections[0]["metric_wise_best_of_4_upper_bound"]
        self.assertEqual(upper["selection_status"], "failure")
        self.assertEqual(upper["effective_pool_size"], 0)
        self.assertIsNone(upper["metrics"])
        self.assertEqual(upper["effective_metrics"]["reward"], 0.0)
        self.assertEqual(upper["effective_metrics"]["rep"], 1.0)
        self.assertTrue(
            all(
                value is None
                for value in upper["candidate_source_by_reward_component"].values()
            )
        )

    def test_metric_wise_upper_bound_uses_only_valid_slots_and_reports_pool_size(self) -> None:
        candidates = {
            1: _candidate(
                "upper", 1, 0.7, alignment=0.1, spc=0.8, cider=0.3, rep=0.4
            ),
            3: _candidate(
                "upper", 3, 0.6, alignment=0.9, spc=0.2, cider=0.7, rep=0.1
            ),
        }
        row = _row(
            "upper", ["valid", "failed", "valid", "failed"], candidates=candidates
        )
        raw_reward_config = {
            key: value * 17.0 for key, value in REWARD_CONFIG.items()
        }
        summary, selections = analyze_failure_aware_candidate_pool(
            [row], raw_reward_config, bootstrap_replicates=0
        )

        upper = selections[0]["metric_wise_best_of_4_upper_bound"]
        self.assertEqual(upper["effective_pool_size"], 2)
        self.assertEqual(upper["metrics"]["reward"], 0.858824)
        self.assertIsNone(upper["candidate_source_by_metric"]["reward"])
        self.assertEqual(
            upper["reward_component_maxima"],
            {
                "alignment": 0.9,
                "readability": 0.7,
                "rhythm": 0.7,
                "control": 1.0,
                "risk": 1.0,
            },
        )
        self.assertEqual(
            upper["candidate_source_by_reward_component"],
            {
                "alignment": "upper_c3",
                "readability": "upper_c1",
                "rhythm": "upper_c1",
                "control": "upper_c1",
                "risk": "upper_c1",
            },
        )
        self.assertEqual(upper["metrics"]["mv_align"], 0.9)
        self.assertEqual(upper["candidate_source_by_metric"]["mv_align"], "upper_c3")
        self.assertEqual(upper["metrics"]["rep"], 0.1)
        self.assertEqual(upper["candidate_source_by_metric"]["rep"], "upper_c3")
        upper_summary = summary["metric_wise_best_of_4_upper_bound"]
        self.assertEqual(upper_summary["mean_effective_pool_size"], 2.0)
        self.assertEqual(upper_summary["effective_pool_size_histogram"], {"2": 1})
        self.assertGreater(
            upper_summary["effective_means"]["reward"],
            summary["methods"]["stars_best_of_4"]["effective_means"]["reward"],
        )

    def test_efficiency_includes_failed_slots_and_exhausted_cost(self) -> None:
        row = _row(
            "cost", ["failed", "valid", "failed", "valid"], rewards={2: 0.4, 4: 0.7}
        )
        for slot, transport_count in zip(
            row["slot_outcomes"], [3, 2, 1, 4]
        ):
            slot["transport_request_count"] = transport_count
        summary, _ = analyze_failure_aware_candidate_pool(
            [row], REWARD_CONFIG, bootstrap_replicates=0
        )
        best_one = summary["methods"]["best_of_1"][
            "efficiency_all_attempted_samples"
        ]
        best_four = summary["methods"]["stars_best_of_4"][
            "efficiency_all_attempted_samples"
        ]
        self.assertEqual(best_one["request_seconds"]["total"], 1.0)
        self.assertEqual(best_one["total_tokens"]["total"], 15.0)
        self.assertEqual(best_one["request_count"]["total"], 3.0)
        self.assertEqual(best_one["generation_requests"]["total"], 3.0)
        self.assertEqual(best_one["transport_request_count"]["total"], 3.0)
        self.assertEqual(best_one["semantic_attempt_count"]["total"], 1.0)
        self.assertEqual(best_four["request_seconds"]["total"], 10.0)
        self.assertEqual(best_four["total_tokens"]["total"], 150.0)
        self.assertEqual(best_four["request_count"]["total"], 10.0)
        self.assertEqual(best_four["generation_requests"]["total"], 10.0)
        self.assertEqual(
            best_four["transport_request_count"]["total"], 10.0
        )
        self.assertEqual(best_four["semantic_attempt_count"]["total"], 4.0)

    def test_reward_counterfactuals_use_all_valid_slots_without_complete_case_filter(self) -> None:
        candidates = {
            1: _candidate("ablation", 1, 0.9, alignment=0.1),
            2: _candidate("ablation", 2, 0.8, alignment=1.0),
            4: _candidate("ablation", 4, 0.2, alignment=0.2),
        }
        candidates[1]["reward"].update(
            {"readability": 1.0, "rhythm": 1.0, "control": 1.0, "risk": 1.0}
        )
        candidates[2]["reward"].update(
            {"readability": 0.0, "rhythm": 0.0, "control": 0.0, "risk": 0.0}
        )
        row = _row(
            "ablation",
            ["valid", "valid", "failed", "valid"],
            candidates=candidates,
        )
        summary, selections = analyze_failure_aware_candidate_pool(
            [row], REWARD_CONFIG, bootstrap_replicates=0
        )

        self.assertIn("leave_one_out__alignment", summary["methods"])
        self.assertIn("single_component__alignment", summary["methods"])
        self.assertIn(
            "weight_sensitivity__alignment__x0_8", summary["methods"]
        )
        self.assertEqual(
            summary["methods"]["single_component__alignment"][
                "successful_sample_count"
            ],
            1,
        )
        self.assertEqual(
            selections[0]["methods"]["single_component__alignment"][
                "candidate_id"
            ],
            "ablation_c2",
        )
        self.assertEqual(
            summary["counterfactual_selection_switch_rate"][
                "single_component__alignment"
            ],
            1.0,
        )
        self.assertEqual(
            summary["component_diagnostics"]["alignment"][
                "valid_candidate_count"
            ],
            3,
        )
        self.assertEqual(
            summary["full_reward_external_metric_correlations"][
                "reward_vs_mv_align_spearman"
            ]["valid_candidate_count"],
            3,
        )

    def test_invalid_slot_contracts_are_rejected(self) -> None:
        row = _row("invalid", ["valid"] * 4, rewards={index: 0.5 for index in range(1, 5)})
        row["slot_outcomes"][3]["candidate_index"] = 3
        with self.assertRaisesRegex(RuntimeError, "indices must be exactly"):
            analyze_failure_aware_candidate_pool(
                [row], REWARD_CONFIG, bootstrap_replicates=0
            )

        row = _row("invalid", ["failed"] * 4)
        row["slot_outcomes"][0]["candidate_id"] = "invented"
        with self.assertRaisesRegex(RuntimeError, "must not contain"):
            analyze_failure_aware_candidate_pool(
                [row], REWARD_CONFIG, bootstrap_replicates=0
            )

    def test_valid_candidate_without_spc_is_rejected_not_zero_imputed(self) -> None:
        row = _row("missing-spc", ["valid", "failed", "failed", "failed"], rewards={1: 0.5})
        row["candidates"][0]["quality"]["semantic_point_coverage"] = None
        with self.assertRaisesRegex(RuntimeError, "missing its evaluation-only SPC"):
            analyze_failure_aware_candidate_pool(
                [row], REWARD_CONFIG, bootstrap_replicates=0
            )

    def test_writer_emits_distinct_conditional_and_effective_artifacts(self) -> None:
        row = _row("write", ["failed"] * 4)
        summary, selections = analyze_failure_aware_candidate_pool(
            [row], REWARD_CONFIG, bootstrap_replicates=0
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_failure_aware_candidate_pool_analysis(tmp, summary, selections)
            root = Path(tmp)
            analysis = json.loads(
                (root / "failure_aware_candidate_pool_analysis.json").read_text()
            )
            self.assertEqual(
                analysis["analysis_protocol"],
                "failure_aware_fixed_requested_slots_v1",
            )
            self.assertTrue(
                (root / "failure_aware_candidate_pool_selections.jsonl").exists()
            )
            header = (
                root / "failure_aware_main_summary.csv"
            ).read_text().splitlines()[0]
            self.assertIn("conditional_reward", header)
            self.assertIn("effective_reward", header)
            self.assertTrue(
                (root / "failure_aware_candidate_pool_summary.md").exists()
            )
            self.assertTrue(
                (root / "failure_aware_reward_leave_one_out.csv").exists()
            )
            self.assertTrue(
                (root / "failure_aware_reward_single_component.csv").exists()
            )
            self.assertTrue(
                (root / "failure_aware_reward_weight_sensitivity.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
