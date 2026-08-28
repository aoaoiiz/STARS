from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SCRIPTS_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compile_revision_evidence import (
    _audit_diagnostic_invariants,
    _write_csr_pool_diagnostics,
    _write_generation_failure_attempts,
    _write_metric_agreement_regret,
    _write_selector_misses,
)


def _candidate(
    index: int,
    total: float,
    violations: list[str],
    *,
    alignment: float,
    spc: float,
    cider: float,
    rep: float,
) -> dict:
    return {
        "candidate_id": f"c{index}",
        "candidate_index": index,
        "reward": {
            "total": total,
            "alignment": alignment,
            "readability": total,
            "rhythm": total,
            "control": 1.0 if not violations else 0.8,
            "risk": total,
            "control_violations": violations,
        },
        "quality": {
            "semantic_point_coverage": spc,
            "cider_lite": cider,
            "repetition_rate": rep,
        },
    }


def _attempt(status: str, number: int, raw_response: str) -> dict:
    return {
        "attempt": number,
        "status": status,
        "prompt_kind": "base_generation" if number == 1 else "validation_retry",
        "prompt_sha256": f"prompt-{number}",
        "request_seed": number,
        "parse_error": "invalid JSON" if status == "parse_rejected" else "",
        "infrastructure_error": False,
        "request_seconds": 0.1,
        "transport_request_count": 1,
        "transport_request_seconds": 0.1,
        "transport_backoff_seconds": 0.0,
        "transport_attempts": [
            {
                "status": "success",
                "request_seconds": 0.1,
                "backoff_seconds_after": 0.0,
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        "raw_response": raw_response,
    }


def _fixture_run() -> dict:
    candidates = [
        _candidate(
            1,
            0.92,
            ["summary_position"],
            alignment=1.00,
            spc=0.60,
            cider=0.20,
            rep=0.30,
        ),
        _candidate(
            2,
            0.88,
            [],
            alignment=0.80,
            spc=0.80,
            cider=0.30,
            rep=0.20,
        ),
        _candidate(
            3,
            0.70,
            [],
            alignment=0.50,
            spc=0.70,
            cider=0.50,
            rep=0.10,
        ),
    ]
    result = {
        "sample_key": "sample-1",
        "video_id": "video-1",
        "candidates": candidates,
        "slot_outcomes": [
            {
                "candidate_index": 1,
                "terminal_status": "valid",
                "terminal_reason": "accepted_valid_candidate",
                "attempts": [
                    _attempt("parse_rejected", 1, "not-json"),
                    _attempt("accepted", 2, "{}"),
                ],
            },
            {
                "candidate_index": 2,
                "terminal_status": "valid",
                "terminal_reason": "accepted_valid_candidate",
                "attempts": [_attempt("accepted", 1, "{}")],
            },
            {
                "candidate_index": 3,
                "terminal_status": "valid",
                "terminal_reason": "accepted_valid_candidate",
                "attempts": [_attempt("accepted", 1, "{}")],
            },
            {
                "candidate_index": 4,
                "terminal_status": "failed",
                "terminal_reason": "parse_retry_exhausted",
                "attempts": [_attempt("parse_rejected", 1, "still-not-json")],
            },
        ],
    }
    selection = {
        "sample_key": "sample-1",
        "methods": {
            "stars_best_of_4": {
                "selection_status": "success",
                "candidate_index": 1,
                "candidate_id": "c1",
                "conditional_metrics": {
                    "reward": 0.92,
                    "mv_align": 1.00,
                    "csr": 0.0,
                    "semantic_point_coverage": 0.60,
                    "cider_lite": 0.20,
                    "rep": 0.30,
                },
            }
        },
        "metric_wise_best_of_4_upper_bound": {
            "selection_status": "success",
            "metrics": {"csr": 1.0},
        },
    }
    return {
        "condition": "fixture",
        "dataset": "dataset",
        "model": "model",
        "audit": {
            "result_rows": [result],
            "selection_rows": [selection],
            "failure_records": [{"sample_key": "sample-1"}],
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RevisionEvidenceDiagnosticsTest(unittest.TestCase):
    def test_diagnostic_tables_report_pool_miss_regret_and_failures(self) -> None:
        run = _fixture_run()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csr_path = root / "csr.csv"
            misses_path = root / "misses.csv"
            regret_path = root / "regret.csv"
            failures_path = root / "failures.csv"
            _write_csr_pool_diagnostics(csr_path, [run])
            _write_selector_misses(misses_path, [run])
            _write_metric_agreement_regret(regret_path, [run])
            _write_generation_failure_attempts(failures_path, [run])

            csr_rows = _read_csv(csr_path)
            self.assertEqual(len(csr_rows), 1)
            self.assertEqual(csr_rows[0]["compliant_candidate_count"], "2")
            self.assertEqual(
                csr_rows[0]["selector_missed_compliant_candidate"], "True"
            )
            self.assertEqual(csr_rows[0]["stars_candidate_id"], "c1")
            self.assertEqual(csr_rows[0]["best_compliant_candidate_id"], "c2")
            self.assertEqual(
                csr_rows[0]["inactive_control_categories"],
                '["key_information_order"]',
            )
            self.assertEqual(csr_rows[0]["minimum_control_violation_count"], "0")
            self.assertIn(
                '"c1": ["summary_position"]',
                csr_rows[0]["candidate_violation_signatures"],
            )

            miss_rows = _read_csv(misses_path)
            self.assertEqual(len(miss_rows), 1)
            self.assertEqual(miss_rows[0]["stars_candidate_id"], "c1")
            self.assertEqual(miss_rows[0]["best_compliant_candidate_id"], "c2")
            self.assertAlmostEqual(
                float(miss_rows[0]["stars_reward_minus_best_compliant_reward"]),
                0.04,
            )
            self.assertAlmostEqual(
                float(miss_rows[0]["reward_delta_reconciliation_error"]),
                0.0,
            )

            regret_rows = _read_csv(regret_path)
            self.assertEqual(len(regret_rows), 6)
            by_metric = {row["metric"]: row for row in regret_rows}
            self.assertEqual(by_metric["reward"]["metric_value_regret"], "0.0")
            self.assertEqual(by_metric["csr"]["metric_value_regret"], "1.0")
            self.assertAlmostEqual(
                float(by_metric["rep"]["metric_value_regret"]), 0.20
            )

            failure_rows = _read_csv(failures_path)
            self.assertEqual(len(failure_rows), 2)
            self.assertEqual(
                {row["attempt_scope"] for row in failure_rows},
                {"retry_then_accepted", "terminal_slot_failure"},
            )
            self.assertTrue(
                all(len(row["raw_response_sha256"]) == 64 for row in failure_rows)
            )

    def test_diagnostic_invariant_rejects_upper_csr_without_pool_support(self) -> None:
        run = _fixture_run()
        results = run["audit"]["result_rows"]
        selections = run["audit"]["selection_rows"]
        self.assertEqual(_audit_diagnostic_invariants(results, selections), [])
        selections[0]["metric_wise_best_of_4_upper_bound"]["metrics"]["csr"] = 0.0
        failures = _audit_diagnostic_invariants(results, selections)
        self.assertTrue(any("candidate-pool compliance" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
