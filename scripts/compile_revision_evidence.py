from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from creative_video_exp.checkpoint_identity import (
    API_MODEL_KIND,
    LOCAL_CHECKPOINT_KIND,
    checkpoint_identity_summary,
    validate_checkpoint_manifest,
    validate_checkpoint_identity,
)
from creative_video_exp.config import DataConfig
from creative_video_exp.data import load_samples
from creative_video_exp.provenance import SOURCE_FILES
from creative_video_exp.reward import CONTROL_CATEGORIES
from creative_video_exp.utils import sha256_file, sha256_json, stable_hash_int


PAPER_METRICS = [
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
]
METHODS = [
    "direct_generation_c1",
    "best_of_1",
    "best_of_2",
    "stars_best_of_4",
]
FORMAL_REWARD_WEIGHTS = {
    "alignment": 6.0 / 17.0,
    "readability": 3.0 / 17.0,
    "rhythm": 3.0 / 17.0,
    "control": 4.0 / 17.0,
    "risk": 1.0 / 17.0,
}
FORMAL_PROMPT_VERSION = "starsv19_visual_only_contract_guided_v2"
FORMAL_PRE_SCORE_PROCESSING = (
    "json_envelope_schema_canonicalization_and_timeline_duration_normalization"
)
FORMAL_PARSE_RETRY_COUNT = 7
FORMAL_OUTPUT_LANGUAGE = "English"
RUNNER_VARIANT = "visual_story"
SLOTS_PER_SAMPLE = 4
FORMAL_SLOT_FAILURE_POLICY = "retain_invalid_slot_and_continue"
FORMAL_METHOD_FAILURE_AGGREGATION = "conditional_and_failure_aware_effective"
CRITICAL_SOURCE_FILES = set(SOURCE_FILES)
FAILURE_AWARE_ANALYSIS = (
    "candidate_pool_analysis/failure_aware_candidate_pool_analysis.json"
)
FAILURE_AWARE_SELECTIONS = (
    "candidate_pool_analysis/failure_aware_candidate_pool_selections.jsonl"
)
EFFICIENCY_KEYS = [
    "mean_online_baseline_seconds",
    "median_online_baseline_seconds",
    "p95_online_baseline_seconds",
    "mean_online_best_of_2_seconds",
    "median_online_best_of_2_seconds",
    "p95_online_best_of_2_seconds",
    "mean_online_best_of_4_seconds",
    "median_online_best_of_4_seconds",
    "p95_online_best_of_4_seconds",
    "mean_best_of_1_input_tokens",
    "mean_best_of_2_input_tokens",
    "mean_best_of_4_input_tokens",
    "mean_best_of_1_output_tokens",
    "mean_best_of_2_output_tokens",
    "mean_best_of_4_output_tokens",
    "mean_best_of_1_total_tokens",
    "mean_best_of_2_total_tokens",
    "mean_best_of_4_total_tokens",
    "mean_best_of_1_generation_requests",
    "mean_best_of_2_generation_requests",
    "mean_best_of_4_generation_requests",
    "mean_best_of_1_semantic_attempts",
    "mean_best_of_2_semantic_attempts",
    "mean_best_of_4_semantic_attempts",
    "mean_best_of_1_transport_requests",
    "mean_best_of_2_transport_requests",
    "mean_best_of_4_transport_requests",
    "mean_best_of_1_semantic_request_seconds",
    "mean_best_of_2_semantic_request_seconds",
    "mean_best_of_4_semantic_request_seconds",
    "mean_best_of_1_transport_request_seconds",
    "mean_best_of_2_transport_request_seconds",
    "mean_best_of_4_transport_request_seconds",
    "mean_best_of_1_transport_backoff_seconds",
    "mean_best_of_2_transport_backoff_seconds",
    "mean_best_of_4_transport_backoff_seconds",
    "mean_reward_encoding_seconds",
    "mean_reward_scoring_seconds",
    "mean_vlm_server_peak_allocated_mib",
    "mean_reward_runner_peak_allocated_mib",
]
EFFICIENCY_PREFIXES = {
    "best_of_1": 1,
    "best_of_2": 2,
    "best_of_4": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile English reviewer-facing conditional, failure-aware effective, "
            "slot-completion, and integrity evidence from STARS matrices."
        )
    )
    parser.add_argument(
        "--matrix",
        action="append",
        required=True,
        help="Condition and matrix root in LABEL=PATH form; repeat as needed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=0,
        help=(
            "Optional expected row count for every run. The default 0 derives the "
            "expected count from protocol metadata."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: list[dict[str, Any]] = []
    for specification in args.matrix:
        if "=" not in specification:
            raise ValueError("Each --matrix value must use LABEL=PATH syntax.")
        condition, raw_path = specification.split("=", 1)
        root = Path(raw_path).expanduser().resolve()
        if not condition.strip() or not root.exists():
            raise ValueError(f"Invalid matrix specification: `{specification}`.")
        runs.extend(_load_runs(condition.strip(), root, args.expected_samples))
    if not runs:
        raise RuntimeError("No auditable STARS metrics.json files were found.")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"Evidence output path is not a directory: `{output}`.")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite a non-empty evidence directory: `{output}`. "
            "Use a new output directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_main_table(output / "main_candidate_and_upper_bound.csv", runs)
    _write_method_success_table(output / "method_success.csv", runs)
    _write_slot_completion_table(output / "slot_completion.csv", runs)
    _write_efficiency_table(output / "efficiency.csv", runs)
    _write_group_table(output / "reward_leave_one_out.csv", runs, "leave_one_out")
    _write_group_table(output / "reward_single_component.csv", runs, "single_component")
    _write_group_table(output / "reward_weight_sensitivity.csv", runs, "weight_sensitivity")
    _write_diagnostics_table(output / "reward_component_diagnostics.csv", runs)
    _write_csr_pool_diagnostics(output / "csr_pool_diagnostics.csv", runs)
    _write_selector_misses(output / "selector_misses.csv", runs)
    _write_metric_agreement_regret(output / "metric_agreement_regret.csv", runs)
    _write_generation_failure_attempts(
        output / "generation_failure_attempts.csv", runs
    )
    _write_integrity_table(output / "protocol_integrity.csv", runs)
    (output / "reviewer_evidence_index.md").write_text(
        _markdown_index(runs), encoding="utf-8"
    )
    print(f"Compiled fixed-manifest runs: {len(runs)}")
    print(f"Evidence directory: {output}")


def _load_runs(condition: str, root: Path, expected_samples: int) -> list[dict[str, Any]]:
    runs = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("config", {}).get("experiment_version") != "starsv20":
            continue
        analysis_path = metrics_path.parent / FAILURE_AWARE_ANALYSIS
        selections_path = metrics_path.parent / FAILURE_AWARE_SELECTIONS
        if not analysis_path.exists():
            raise RuntimeError(f"Missing failure-aware candidate-pool analysis: {analysis_path}")
        if not selections_path.exists():
            raise RuntimeError(f"Missing failure-aware per-sample selections: {selections_path}")
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        selections = _read_jsonl(selections_path)
        audit = _validate_run(
            metrics_path,
            metrics,
            analysis,
            selections,
            expected_samples,
        )
        runs.append(
            {
                "condition": condition,
                "dataset": metrics.get("config", {}).get("data", {}).get("name", ""),
                "model": metrics.get("config", {}).get("models", {}).get(
                    "active_video_model", ""
                ),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "analysis": analysis,
                "audit": audit,
            }
        )
    return runs


def _validate_run(
    path: Path,
    metrics: dict[str, Any],
    analysis: dict[str, Any],
    selections: list[dict[str, Any]],
    expected_samples: int,
) -> dict[str, Any]:
    failures: list[str] = []
    generation_config = metrics.get("config", {}).get("generation", {})
    requested_samples = int(metrics.get("requested_num_samples", 0))
    if generation_config.get("prompt_version") != FORMAL_PROMPT_VERSION:
        failures.append("prompt_version does not match the STARS protocol")
    if (
        generation_config.get("pre_score_processing")
        != FORMAL_PRE_SCORE_PROCESSING
    ):
        failures.append("pre_score_processing does not match the STARS protocol")
    if generation_config.get("parse_retry_count") != FORMAL_PARSE_RETRY_COUNT:
        failures.append("parse_retry_count does not match the STARS protocol")
    if generation_config.get("output_language") != FORMAL_OUTPUT_LANGUAGE:
        failures.append("output_language is not exactly English")
    if (
        generation_config.get("candidate_slot_failure_policy")
        != FORMAL_SLOT_FAILURE_POLICY
    ):
        failures.append("candidate-slot failure policy is not frozen")
    if (
        generation_config.get("method_failure_aggregation")
        != FORMAL_METHOD_FAILURE_AGGREGATION
    ):
        failures.append("method failure aggregation is not frozen")
    if expected_samples > 0 and requested_samples != expected_samples:
        failures.append(
            f"requested_num_samples={requested_samples} instead of {expected_samples}"
        )
    if metrics.get("evidence_eligible") is not True:
        failures.append("run_status does not certify complete fixed-slot auditing")
    if metrics.get("formal_infrastructure_valid") is not True:
        failures.append("formal infrastructure audit is not valid")
    if int(metrics.get("infrastructure_issue_count", -1)) != 0:
        failures.append("formal run contains infrastructure issues")
    if int(metrics.get("transport_error_attempt_count", -1)) != 0:
        failures.append("formal run contains hidden transport errors")
    if int(metrics.get("semantic_endpoint_error_attempt_count", -1)) != 0:
        failures.append("formal run contains endpoint-error semantic attempts")

    protocol = metrics.get("protocol", {})
    scientific_payload = protocol.get("scientific_payload", {})
    recorded_protocol_fingerprint = str(protocol.get("protocol_fingerprint", ""))
    if not scientific_payload or recorded_protocol_fingerprint != sha256_json(
        scientific_payload
    ):
        failures.append("protocol fingerprint does not match its scientific payload")
    source_hashes = scientific_payload.get("source_hashes", {})
    for relative_path in sorted(CRITICAL_SOURCE_FILES):
        source_path = PROJECT_ROOT / relative_path
        recorded_hash = str(source_hashes.get(relative_path, ""))
        if not source_path.is_file() or not recorded_hash:
            failures.append(f"critical source provenance is missing for {relative_path}")
        elif sha256_file(source_path) != recorded_hash:
            failures.append(f"critical source hash differs for {relative_path}")
    identity_failures, generation_identity, reward_identity = (
        _audit_model_identity_protocol(metrics, scientific_payload)
    )
    failures.extend(identity_failures)
    configured_weights = protocol.get("reward_definition", {}).get(
        "configured_weights", {}
    )
    if configured_weights != FORMAL_REWARD_WEIGHTS:
        failures.append(
            "formal Reward must contain exactly the disclosed normalized "
            "Alignment/Readability/Rhythm/Control/Risk weights"
        )
    if abs(sum(configured_weights.values()) - 1.0) > 1e-12:
        failures.append("formal Reward weights do not sum to 1.0")
    formal_data = scientific_payload.get("data", {})
    failures.extend(_audit_formal_data_protocol(formal_data))
    configured_data = metrics.get("config", {}).get("data", {})
    for field in ("sample_unit", "offset", "limit"):
        if configured_data.get(field) != formal_data.get(field):
            failures.append(
                f"metrics config data.{field} differs from the protocol manifest"
            )
    formal_long_video = scientific_payload.get("long_video", {})
    configured_long_video = metrics.get("config", {}).get("long_video", {})
    if configured_long_video != formal_long_video:
        failures.append(
            "metrics config long_video differs from the protocol manifest"
        )
    if formal_long_video.get("enabled") is not False:
        failures.append(
            "formal main-run evidence must keep the separate long-video test disabled"
        )
    spc_definition = protocol.get("evaluation_definition", {}).get(
        "semantic_point_coverage", {}
    )
    if spc_definition.get("reference_enters_generation") is not False:
        failures.append("SPC references are not isolated from generation")
    if spc_definition.get("reference_enters_reward") is not False:
        failures.append("SPC references are not isolated from Reward")
    pool_protocol = protocol.get("candidate_pool_protocol", {})
    variant_policy = pool_protocol.get("variant_policy", {})
    expected_variant_policy = {
        "owner": "runner",
        "fixed_value": RUNNER_VARIANT,
        "used_by_reward_or_evaluation": False,
        "raw_model_value_preserved_in_attempt_trace": True,
        "canonicalization_record": "fixed_runner_owned_variant",
    }
    if variant_policy != expected_variant_policy:
        failures.append("runner-owned variant policy is missing or altered")
    if pool_protocol.get("failed_sample_replacement") is not False:
        failures.append("failed-sample replacement is not explicitly forbidden")
    if int(pool_protocol.get("requested_slots_per_sample", 0)) != SLOTS_PER_SAMPLE:
        failures.append("protocol does not declare four requested slots per sample")
    if pool_protocol.get("slot_failure_policy") != "continue_remaining_slots":
        failures.append("protocol does not continue and audit all remaining slots")

    results_path = path.parent / "results.jsonl"
    rows: list[dict[str, Any]] = []
    if not results_path.exists():
        failures.append("results.jsonl is missing")
    else:
        try:
            rows = _read_jsonl(results_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"results.jsonl cannot be audited: {exc}")
    if len(rows) != requested_samples:
        failures.append(
            f"results.jsonl has {len(rows)} rows for {requested_samples} requested samples"
        )
    sample_keys = [str(row.get("sample_key", "")) for row in rows]
    if any(not key for key in sample_keys):
        failures.append("a requested sample row is missing sample_key")
    if len(set(sample_keys)) != len(sample_keys):
        failures.append("duplicate sample_key rows indicate replacement or mixed results")

    valid_slots = 0
    failed_slots = 0
    full_pool_samples = 0
    any_valid_samples = 0
    serialized_candidates: list[dict[str, Any]] = []
    for row in rows:
        row_issues, row_counts, candidates = _audit_result_row(row, generation_config)
        failures.extend(
            f"sample {row.get('sample_key', '<missing>')}: {issue}"
            for issue in row_issues
        )
        valid_slots += row_counts["valid"]
        failed_slots += row_counts["failed"]
        full_pool_samples += int(row_counts["valid"] == SLOTS_PER_SAMPLE)
        any_valid_samples += int(row_counts["valid"] > 0)
        serialized_candidates.extend(candidates)

    requested_slots = requested_samples * SLOTS_PER_SAMPLE
    if valid_slots + failed_slots != requested_slots:
        failures.append("valid and failed terminal slots do not cover all 4N requested slots")
    if len(serialized_candidates) != valid_slots:
        failures.append("serialized valid candidates do not join one-to-one with valid slots")

    expected_sample_keys = _expected_fixed_sample_keys(metrics, protocol, failures)
    if expected_sample_keys and set(sample_keys) != expected_sample_keys:
        missing = sorted(expected_sample_keys - set(sample_keys))[:5]
        unexpected = sorted(set(sample_keys) - expected_sample_keys)[:5]
        failures.append(
            "results.jsonl does not match the configured fixed-manifest sample set; "
            f"missing={missing}, unexpected={unexpected}"
        )

    protocol_fingerprint = str(protocol.get("protocol_fingerprint", ""))
    run_status_path = path.parent / "run_status.json"
    if not run_status_path.is_file():
        failures.append("final run_status.json is missing")
    else:
        try:
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            run_status = {}
            failures.append(f"run_status.json cannot be audited: {exc}")
        if run_status.get("status") != "eligible":
            failures.append("run_status.json is not in final eligible state")
        if run_status.get("evidence_eligible") is not True:
            failures.append("run_status.json does not certify evidence eligibility")
        if run_status.get("formal_infrastructure_valid") is not True:
            failures.append("run_status.json does not certify infrastructure validity")
        if str(run_status.get("protocol_fingerprint", "")) != protocol_fingerprint:
            failures.append("run_status.json has a mixed protocol fingerprint")
    for row in rows:
        if row.get("experiment_version") != "starsv20":
            failures.append("a result row has an incompatible experiment version")
        if str(row.get("protocol_fingerprint", "")) != protocol_fingerprint:
            failures.append("a result row has a mixed protocol fingerprint")
    failures.extend(
        _audit_observed_model_identities(
            rows,
            metrics,
            generation_identity,
            reward_identity,
        )
    )

    analysis_expected = {
        "experiment_version": "starsv20",
        "protocol_fingerprint": protocol_fingerprint,
        "requested_sample_count": requested_samples,
        "requested_slots_per_sample": SLOTS_PER_SAMPLE,
        "requested_candidate_slot_count": requested_slots,
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "full_candidate_pool_sample_count": full_pool_samples,
        "any_valid_candidate_sample_count": any_valid_samples,
    }
    for key, expected in analysis_expected.items():
        if analysis.get(key) != expected:
            failures.append(
                f"failure-aware analysis {key}={analysis.get(key)!r}, expected {expected!r}"
            )
    _check_rate(
        failures,
        "full_candidate_pool_sample_rate",
        analysis.get("full_candidate_pool_sample_rate"),
        full_pool_samples,
        requested_samples,
    )
    _check_rate(
        failures,
        "any_valid_candidate_sample_rate",
        analysis.get("any_valid_candidate_sample_rate"),
        any_valid_samples,
        requested_samples,
    )

    if len(selections) != requested_samples:
        failures.append("failure-aware selections do not contain exactly N sample rows")
    selection_keys = [str(row.get("sample_key", "")) for row in selections]
    if set(selection_keys) != set(sample_keys) or len(set(selection_keys)) != len(selection_keys):
        failures.append("failure-aware selections do not match the fixed result sample keys")
    counterfactual_specs = _counterfactual_method_specs()
    expected_method_groups = {
        **{method: "nested_pool" for method in METHODS},
        **{method: group for method, (_, group) in counterfactual_specs.items()},
    }
    if analysis.get("method_groups") != expected_method_groups:
        failures.append("failure-aware analysis method_groups is missing or altered")

    result_by_key = {str(row.get("sample_key")): row for row in rows}
    failure_records_path = path.parent / "generation_failures.jsonl"
    try:
        failure_records = (
            _read_jsonl(failure_records_path)
            if failure_records_path.exists()
            else []
        )
    except (OSError, json.JSONDecodeError) as exc:
        failure_records = []
        failures.append(f"generation failure records cannot be audited: {exc}")
    failures.extend(
        _audit_failure_records(
            failure_records,
            result_by_key,
            protocol_fingerprint,
        )
    )
    if int(metrics.get("generation_failure_records", 0)) != len(failure_records):
        failures.append("metrics generation_failure_records differs from the file")

    for selection in selections:
        failures.extend(
            _audit_selection_row(
                selection,
                result_by_key.get(str(selection.get("sample_key", ""))),
                protocol_fingerprint,
                counterfactual_specs,
            )
        )

    methods = analysis.get("methods", {})
    for method in METHODS:
        failures.extend(_audit_method_summary(method, methods.get(method), requested_samples))
        failures.extend(
            _audit_method_summary_against_selections(
                method, methods.get(method), selections
            )
        )
    for method in counterfactual_specs:
        failures.extend(_audit_method_summary(method, methods.get(method), requested_samples))
        failures.extend(
            _audit_method_summary_against_selections(
                method, methods.get(method), selections
            )
        )
    upper = analysis.get("metric_wise_best_of_4_upper_bound")
    failures.extend(_audit_upper_bound(upper, requested_samples))
    failures.extend(_audit_upper_summary_against_selections(upper, selections))
    stars_summary = methods.get("stars_best_of_4", {})
    for scope in ("conditional_means", "effective_means"):
        if isinstance(upper, dict):
            upper_reward = upper.get(scope, {}).get("reward")
            stars_reward = stars_summary.get(scope, {}).get("reward")
            if (
                upper_reward is not None
                and stars_reward is not None
                and float(upper_reward) + 1e-6 < float(stars_reward)
            ):
                failures.append(
                    f"Metric-wise Best@4 {scope} Reward is below STARS"
                )
    invariant = analysis.get("baseline_best_of_1_invariant", {})
    if not all(
        invariant.get(key)
        for key in (
            "identical_candidate_ids",
            "identical_failure_states",
            "identical_conditional_metrics",
            "identical_effective_metrics",
        )
    ):
        failures.append("Direct Generation and Best-of-1 are not identical")
    if not analysis.get("nested_reward_monotonicity", {}).get("verified"):
        failures.append("nested failure-aware Reward monotonicity was not verified")

    efficiency_audit, efficiency_failures = _audit_failure_aware_efficiency(
        rows=rows,
        metrics=metrics,
        analysis=analysis,
    )
    failures.extend(efficiency_failures)

    metric_cross_checks = {
        "requested_candidate_slot_count": requested_slots,
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "full_candidate_pool_sample_count": full_pool_samples,
        "any_valid_candidate_sample_count": any_valid_samples,
    }
    for key, expected in metric_cross_checks.items():
        if int(metrics.get(key, -1)) != expected:
            failures.append(f"metrics {key} does not match audited results")
    _check_rate(
        failures,
        "candidate_slot_success_rate",
        metrics.get("candidate_slot_success_rate"),
        valid_slots,
        requested_slots,
    )
    _check_rate(
        failures,
        "full_candidate_pool_sample_rate",
        metrics.get("full_candidate_pool_sample_rate"),
        full_pool_samples,
        requested_samples,
    )
    _check_rate(
        failures,
        "any_valid_candidate_sample_rate",
        metrics.get("any_valid_candidate_sample_rate"),
        any_valid_samples,
        requested_samples,
    )

    for candidate in serialized_candidates:
        if candidate.get("variant") != RUNNER_VARIANT:
            failures.append("a valid serialized candidate does not use variant=visual_story")
            break
        provenance = candidate.get("generation_provenance", {})
        if provenance.get("variant_policy") != "runner_owned_fixed_visual_story":
            failures.append("a valid candidate is missing runner-owned variant provenance")
            break
        if provenance.get("semantic_repair_applied") is not False:
            failures.append("a valid candidate has semantic repair provenance")
            break
        timeline_normalization = provenance.get("timeline_normalization", {})
        if (
            timeline_normalization.get("policy")
            != "positive_duration_proportional_v1"
            or timeline_normalization.get("status")
            not in {"applied", "unchanged", "skipped"}
        ):
            failures.append(
                "a valid candidate is missing deterministic timeline normalization provenance"
            )
            break
        if candidate.get("timeline_normalization") != timeline_normalization:
            failures.append(
                "candidate timeline normalization does not match its provenance"
            )
            break
        timestamp_violation = "timestamp_validity" in candidate.get(
            "reward", {}
        ).get("control_violations", candidate.get("reward", {}).get("violations", []))
        if timeline_normalization.get("status") == "skipped":
            if not timestamp_violation:
                failures.append(
                    "a skipped timeline normalization lacks a timestamp violation"
                )
                break
        elif timestamp_violation:
            failures.append(
                "an applied or unchanged timeline normalization has a timestamp violation"
            )
            break
    if int(metrics.get("semantic_repair_candidate_count", 0)) != 0:
        failures.append("semantic repair candidates were detected")
    if int(metrics.get("wrapped_free_text_candidate_count", 0)) != 0:
        failures.append("wrapped free-text candidates were detected")
    if int(metrics.get("surrounding_free_text_candidate_count", 0)) != 0:
        failures.append("surrounding free-text candidates were detected")
    failures.extend(_audit_diagnostic_invariants(rows, selections))

    if failures:
        raise RuntimeError(f"Protocol integrity failure in {path}: " + "; ".join(failures))
    return {
        "requested_sample_count": requested_samples,
        "requested_slot_count": requested_slots,
        "valid_slot_count": valid_slots,
        "failed_slot_count": failed_slots,
        "full_pool_sample_count": full_pool_samples,
        "any_valid_sample_count": any_valid_samples,
        "efficiency": efficiency_audit,
        "result_rows": rows,
        "selection_rows": selections,
        "failure_records": failure_records,
    }


def _audit_model_identity_protocol(
    metrics: dict[str, Any],
    scientific_payload: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    payload_models = scientific_payload.get("models", {})
    config_models = metrics.get("config", {}).get("models", {})
    if payload_models.get("mode") != "server_full_matrix":
        failures.append("formal evidence does not use models.mode=server_full_matrix")
    active_generation_id = str(payload_models.get("active_video_model", ""))
    active_reward_id = str(payload_models.get("active_reward_vision_model", ""))
    payload_endpoints = {
        str(endpoint.get("id", "")): endpoint
        for endpoint in payload_models.get("endpoints", [])
        if isinstance(endpoint, dict) and endpoint.get("id")
    }
    config_endpoints = {
        str(endpoint.get("id", "")): endpoint
        for endpoint in config_models.get("endpoints", [])
        if isinstance(endpoint, dict) and endpoint.get("id")
    }
    identities: dict[str, dict[str, Any]] = {}
    for role, endpoint_id in (
        ("generation", active_generation_id),
        ("reward", active_reward_id),
    ):
        endpoint = payload_endpoints.get(endpoint_id)
        config_endpoint = config_endpoints.get(endpoint_id)
        if endpoint is None or config_endpoint is None:
            failures.append(f"formal {role} endpoint is missing from config/protocol")
            identities[role] = {}
            continue
        identity = endpoint.get("checkpoint_identity", {})
        try:
            validate_checkpoint_identity(identity)
        except (TypeError, ValueError) as exc:
            failures.append(f"formal {role} checkpoint identity is invalid: {exc}")
            identities[role] = {}
            continue
        if config_endpoint.get("checkpoint_identity") != identity:
            failures.append(f"formal {role} checkpoint identity differs in config/protocol")
        if identity.get("model_id") != endpoint.get("name"):
            failures.append(f"formal {role} checkpoint model_id differs from endpoint name")
        expected_kind = (
            API_MODEL_KIND if endpoint.get("provider") == "openai" else LOCAL_CHECKPOINT_KIND
        )
        if identity.get("kind") != expected_kind:
            failures.append(
                f"formal {role} checkpoint identity kind is not {expected_kind}"
            )
        if expected_kind == LOCAL_CHECKPOINT_KIND:
            checkpoint_manifest = endpoint.get("checkpoint_manifest", {})
            try:
                validate_checkpoint_manifest(checkpoint_manifest)
                if checkpoint_identity_summary(checkpoint_manifest) != identity:
                    failures.append(
                        f"formal {role} sidecar does not match checkpoint identity"
                    )
            except (TypeError, ValueError) as exc:
                failures.append(f"formal {role} checkpoint sidecar is invalid: {exc}")
        elif endpoint.get("checkpoint_manifest") not in ({}, None):
            failures.append("API generation endpoint unexpectedly contains a local sidecar")
        identities[role] = identity
    return failures, identities.get("generation", {}), identities.get("reward", {})


def _audit_observed_model_identities(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    generation_identity: dict[str, Any],
    reward_identity: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not generation_identity or not reward_identity:
        return failures

    for row in rows:
        sample_key = str(row.get("sample_key", "<missing>"))
        for outcome in row.get("slot_outcomes", []):
            slot_index = int(outcome.get("candidate_index", 0))
            for attempt_index, attempt in enumerate(
                outcome.get("attempts", []), start=1
            ):
                if attempt.get("status") == "endpoint_error":
                    continue
                server_metrics = attempt.get("server_metrics", {})
                if not isinstance(server_metrics, dict):
                    failures.append(
                        f"sample {sample_key} slot {slot_index} attempt {attempt_index} "
                        "has no model-identity server metrics"
                    )
                    continue
                if server_metrics.get("returned_model_id_conflict") is True:
                    failures.append(
                        f"sample {sample_key} slot {slot_index} attempt {attempt_index} "
                        "reported conflicting model identities"
                    )
                returned_model_id = str(server_metrics.get("returned_model_id", ""))
                if returned_model_id != str(generation_identity.get("model_id", "")):
                    failures.append(
                        f"sample {sample_key} slot {slot_index} attempt {attempt_index} "
                        "returned model id differs from the bound identity"
                    )
                if generation_identity.get("kind") == LOCAL_CHECKPOINT_KIND:
                    observed = {
                        "identity_sha256": server_metrics.get(
                            "checkpoint_identity_sha256"
                        ),
                        "model_id": server_metrics.get("checkpoint_model_id"),
                        "revision": server_metrics.get("checkpoint_revision"),
                    }
                    expected = {
                        "identity_sha256": generation_identity.get("identity_sha256"),
                        "model_id": generation_identity.get("model_id"),
                        "revision": generation_identity.get("revision"),
                    }
                    if observed != expected:
                        failures.append(
                            f"sample {sample_key} slot {slot_index} attempt {attempt_index} "
                            "local checkpoint fingerprint differs from the bound identity"
                        )

    reward_runtime = metrics.get("model_runtime", {}).get("video_model", {})
    observed_reward_identity = reward_runtime.get("checkpoint_identity", {})
    if observed_reward_identity != reward_identity:
        failures.append(
            "runtime reward checkpoint identity differs from the protocol-bound identity"
        )
    if reward_runtime.get("checkpoint_identity_sha256") != reward_identity.get(
        "identity_sha256"
    ):
        failures.append("runtime reward checkpoint fingerprint is missing or altered")
    return failures


def _audit_result_row(
    row: dict[str, Any],
    generation_config: dict[str, Any],
) -> tuple[list[str], dict[str, int], list[dict[str, Any]]]:
    failures: list[str] = []
    source_count = row.get("source_annotation_row_count")
    fingerprints = row.get("source_row_fingerprints")
    try:
        source_count = int(source_count)
    except (TypeError, ValueError):
        source_count = 0
    if source_count <= 0:
        failures.append("source_annotation_row_count is missing or invalid")
    if not isinstance(fingerprints, list) or len(fingerprints) != source_count:
        failures.append(
            "source_row_fingerprints does not match source_annotation_row_count"
        )
        fingerprints = []
    elif any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in fingerprints
    ):
        failures.append("source_row_fingerprints contains an invalid SHA256 value")
    references = row.get("evaluation_only_references")
    if not isinstance(references, list) or not references:
        failures.append("evaluation_only_references is missing or empty")
    else:
        reference_fingerprints = [
            value
            for reference in references
            if isinstance(reference, dict)
            for value in reference.get("source_row_fingerprints", [])
        ]
        if Counter(reference_fingerprints) != Counter(fingerprints):
            failures.append(
                "evaluation-only references do not account for all source rows"
            )
    expected_sample_key = _sample_key_payload(
        {
            "video_id": row.get("video_id", ""),
            "video_path": row.get("video_path", ""),
            "source_row_fingerprints": fingerprints,
        }
    )
    if str(row.get("sample_key", "")) != expected_sample_key:
        failures.append("sample_key does not commit to the video-level source rows")
    sampling = row.get("sampling", {})
    if not isinstance(sampling, dict):
        failures.append("sampling audit metadata is missing")
    else:
        source_kind = str(sampling.get("source_kind", ""))
        decode_strategy = str(sampling.get("decode_strategy", ""))
        if source_kind == "synthetic":
            failures.append("sampling used synthetic frames")
        if sampling.get("fallback_used") is not False:
            failures.append("sampling used a fallback decoder")
        if decode_strategy not in {"duration_seek_uniform", "preextracted_frames"}:
            failures.append(
                f"sampling used non-uniform decode strategy `{decode_strategy or 'missing'}`"
            )
    if int(row.get("requested_candidate_slots", 0)) != SLOTS_PER_SAMPLE:
        failures.append("requested_candidate_slots is not four")
    outcomes = row.get("slot_outcomes", [])
    if not isinstance(outcomes, list) or len(outcomes) != SLOTS_PER_SAMPLE:
        return [*failures, "slot_outcomes is not an exact four-item list"], {
            "valid": 0,
            "failed": 0,
        }, []
    indices = [int(outcome.get("candidate_index", 0)) for outcome in outcomes]
    if indices != list(range(1, SLOTS_PER_SAMPLE + 1)):
        failures.append(f"slot indices are {indices}, expected [1, 2, 3, 4]")
    candidates = row.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
        failures.append("candidates is not a list")
    by_index = {int(candidate.get("candidate_index", 0)): candidate for candidate in candidates}
    if len(by_index) != len(candidates):
        failures.append("valid candidate indices are missing or duplicated")
    valid_count = 0
    failed_count = 0
    max_attempts = int(generation_config.get("parse_retry_count", 0)) + 1
    templates = row.get("prompt_templates", {})
    base_hash = str(templates.get("base_generation", ""))
    retry_hash = str(templates.get("validation_retry", ""))
    if not base_hash or not retry_hash or base_hash == retry_hash:
        failures.append("row lacks two distinct frozen prompt-template hashes")
    for outcome in outcomes:
        index = int(outcome.get("candidate_index", 0))
        status = outcome.get("terminal_status")
        attempts = outcome.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            failures.append(f"slot {index} has no auditable attempts")
            attempts = []
        if int(outcome.get("request_count", -1)) != len(attempts):
            failures.append(f"slot {index} request_count does not match attempts")
        if len(attempts) > max_attempts:
            failures.append(f"slot {index} exceeds the bounded attempt count")
        failures.extend(_audit_attempts(index, attempts, base_hash, retry_hash))
        observed_seconds = sum(
            float(attempt.get("request_seconds", 0.0)) for attempt in attempts
        )
        try:
            reported_seconds = float(outcome.get("request_seconds", -1.0))
        except (TypeError, ValueError):
            reported_seconds = -1.0
        if reported_seconds < 0.0 or abs(reported_seconds - observed_seconds) > 1e-4:
            failures.append(f"slot {index} request_seconds does not match attempts")
        observed_usage = {
            key: sum(int(attempt.get("usage", {}).get(key, 0)) for attempt in attempts)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        reported_usage = outcome.get("usage", {})
        if any(int(reported_usage.get(key, -1)) != value for key, value in observed_usage.items()):
            failures.append(f"slot {index} usage does not match attempts")
        observed_transport_count = sum(
            len(attempt.get("transport_attempts", [])) for attempt in attempts
        )
        observed_transport_seconds = sum(
            float(transport.get("request_seconds", 0.0))
            for attempt in attempts
            for transport in attempt.get("transport_attempts", [])
        )
        observed_transport_backoff = sum(
            float(transport.get("backoff_seconds_after", 0.0))
            for attempt in attempts
            for transport in attempt.get("transport_attempts", [])
        )
        if int(outcome.get("transport_request_count", -1)) != observed_transport_count:
            failures.append(f"slot {index} transport_request_count is inconsistent")
        try:
            reported_transport_seconds = float(
                outcome.get("transport_request_seconds", -1.0)
            )
            reported_transport_backoff = float(
                outcome.get("transport_backoff_seconds", -1.0)
            )
        except (TypeError, ValueError):
            reported_transport_seconds = -1.0
            reported_transport_backoff = -1.0
        if (
            not _finite_nonnegative(reported_transport_seconds)
            or abs(reported_transport_seconds - observed_transport_seconds) > 1e-4
        ):
            failures.append(f"slot {index} transport_request_seconds is inconsistent")
        if (
            not _finite_nonnegative(reported_transport_backoff)
            or abs(reported_transport_backoff - observed_transport_backoff) > 1e-4
        ):
            failures.append(f"slot {index} transport_backoff_seconds is inconsistent")
        candidate = by_index.get(index)
        if status == "valid":
            valid_count += 1
            candidate_id = str(outcome.get("candidate_id") or "")
            if candidate is None or candidate.get("candidate_id") != candidate_id:
                failures.append(f"valid slot {index} does not join to one scored candidate")
            elif not isinstance(candidate.get("reward"), dict) or not isinstance(
                candidate.get("quality"), dict
            ):
                failures.append(f"valid slot {index} is missing Reward or quality metrics")
            if outcome.get("terminal_reason") != "accepted_valid_candidate":
                failures.append(f"valid slot {index} has a failure terminal_reason")
            if sum(attempt.get("status") == "accepted" for attempt in attempts) != 1:
                failures.append(f"valid slot {index} lacks one terminal accepted attempt")
            elif attempts[-1].get("status") != "accepted":
                failures.append(f"valid slot {index} continued after acceptance")
        elif status == "failed":
            failed_count += 1
            if candidate is not None or outcome.get("candidate_id") is not None:
                failures.append(f"failed slot {index} was promoted into a candidate")
            if not str(outcome.get("terminal_reason") or ""):
                failures.append(f"failed slot {index} lacks a terminal_reason")
            if len(attempts) != max_attempts:
                failures.append(f"failed slot {index} did not exhaust the fixed bound")
            if any(attempt.get("status") == "accepted" for attempt in attempts):
                failures.append(f"failed slot {index} contains an accepted attempt")
        else:
            failures.append(f"slot {index} has invalid terminal_status={status!r}")
    if set(by_index) != {
        int(outcome.get("candidate_index", 0))
        for outcome in outcomes
        if outcome.get("terminal_status") == "valid"
    }:
        failures.append("candidate list and valid terminal slots disagree")
    return failures, {"valid": valid_count, "failed": failed_count}, candidates


def _audit_attempts(
    slot_index: int,
    attempts: list[dict[str, Any]],
    base_hash: str,
    retry_hash: str,
) -> list[str]:
    failures: list[str] = []
    for position, attempt in enumerate(attempts, start=1):
        expected_kind = "base_generation" if position == 1 else "validation_retry"
        if attempt.get("prompt_kind") != expected_kind:
            failures.append(f"slot {slot_index} attempt {position} has wrong prompt_kind")
        if not str(attempt.get("prompt_sha256") or ""):
            failures.append(f"slot {slot_index} attempt {position} lacks prompt hash")
        expected_hash = base_hash if position == 1 else retry_hash
        if attempt.get("prompt_sha256") != expected_hash:
            failures.append(f"slot {slot_index} attempt {position} has wrong prompt hash")
        if attempt.get("request_seed") is None:
            failures.append(f"slot {slot_index} attempt {position} lacks request seed")
        if float(attempt.get("request_seconds", -1.0)) < 0.0:
            failures.append(f"slot {slot_index} attempt {position} lacks nonnegative latency")
        status = attempt.get("status")
        if status not in {"accepted", "parse_rejected", "endpoint_error"}:
            failures.append(f"slot {slot_index} attempt {position} has invalid status")
        if status == "endpoint_error" or attempt.get("infrastructure_error") is True:
            failures.append(
                f"slot {slot_index} attempt {position} is an infrastructure failure"
            )
        if status in {"accepted", "parse_rejected"} and "raw_response" not in attempt:
            failures.append(f"slot {slot_index} attempt {position} lacks raw response")
        if status == "parse_rejected" and not str(attempt.get("parse_error") or ""):
            failures.append(f"slot {slot_index} rejected attempt lacks reason")
        usage = attempt.get("usage", {})
        if status != "endpoint_error" and not all(
            key in usage for key in ("input_tokens", "output_tokens", "total_tokens")
        ):
            failures.append(f"slot {slot_index} attempt {position} lacks token audit")
        transports = attempt.get("transport_attempts", [])
        if not isinstance(transports, list) or not transports:
            failures.append(
                f"slot {slot_index} attempt {position} lacks transport_attempts"
            )
            continue
        if int(attempt.get("transport_request_count", -1)) != len(transports):
            failures.append(
                f"slot {slot_index} attempt {position} transport count is inconsistent"
            )
        transport_seconds = 0.0
        transport_backoff = 0.0
        for transport_position, transport in enumerate(transports, start=1):
            if not isinstance(transport, dict):
                failures.append(
                    f"slot {slot_index} attempt {position} has a malformed transport record"
                )
                continue
            if int(transport.get("transport_attempt", -1)) != transport_position:
                failures.append(
                    f"slot {slot_index} attempt {position} transport indices are not sequential"
                )
            if transport.get("status") != "success":
                failures.append(
                    f"slot {slot_index} attempt {position} contains a transport error"
                )
            try:
                seconds = float(transport.get("request_seconds", -1.0))
                backoff = float(transport.get("backoff_seconds_after", -1.0))
            except (TypeError, ValueError):
                seconds = -1.0
                backoff = -1.0
            if not _finite_nonnegative(seconds) or not _finite_nonnegative(backoff):
                failures.append(
                    f"slot {slot_index} attempt {position} has invalid transport timing"
                )
            transport_seconds += max(0.0, seconds)
            transport_backoff += max(0.0, backoff)
            transport_usage = transport.get("usage", {})
            try:
                input_tokens = int(transport_usage.get("input_tokens", -1))
                output_tokens = int(transport_usage.get("output_tokens", -1))
                total_tokens = int(transport_usage.get("total_tokens", -1))
            except (TypeError, ValueError):
                input_tokens = output_tokens = total_tokens = -1
            if (
                min(input_tokens, output_tokens, total_tokens) < 0
                or total_tokens != input_tokens + output_tokens
            ):
                failures.append(
                    f"slot {slot_index} attempt {position} has invalid transport usage"
                )
        try:
            declared_seconds = float(
                attempt.get("transport_request_seconds", -1.0)
            )
            declared_backoff = float(
                attempt.get("transport_backoff_seconds", -1.0)
            )
        except (TypeError, ValueError):
            declared_seconds = -1.0
            declared_backoff = -1.0
        if (
            not _finite_nonnegative(declared_seconds)
            or abs(declared_seconds - transport_seconds) > 1e-4
        ):
            failures.append(
                f"slot {slot_index} attempt {position} transport seconds are inconsistent"
            )
        if (
            not _finite_nonnegative(declared_backoff)
            or abs(declared_backoff - transport_backoff) > 1e-4
        ):
            failures.append(
                f"slot {slot_index} attempt {position} transport backoff is inconsistent"
            )
    return failures


def _finite_nonnegative(value: float) -> bool:
    return value >= 0.0 and value != float("inf") and value == value


def _audit_failure_aware_efficiency(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:


    failures: list[str] = []
    measures = (
        "semantic_attempts",
        "transport_requests",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "semantic_request_seconds",
        "transport_request_seconds",
        "transport_backoff_seconds",
    )
    values = {
        method: {measure: [] for measure in measures}
        for method in EFFICIENCY_PREFIXES
    }
    online_values = {method: [] for method in EFFICIENCY_PREFIXES}

    for row in rows:
        sample_key = str(row.get("sample_key", "<missing>"))
        slots = row.get("slot_outcomes", [])
        if not isinstance(slots, list) or len(slots) != SLOTS_PER_SAMPLE:
            failures.append(
                f"sample {sample_key}: efficiency audit requires four terminal slots"
            )
            continue
        slots = sorted(slots, key=lambda slot: int(slot.get("candidate_index", 0)))
        row_efficiency = row.get("efficiency", {})
        if not isinstance(row_efficiency, dict):
            failures.append(f"sample {sample_key}: efficiency record is missing")
            continue

        reward_by_slot = row_efficiency.get("reward_scoring_seconds_by_slot", [])
        if not isinstance(reward_by_slot, list) or len(reward_by_slot) != SLOTS_PER_SAMPLE:
            failures.append(
                f"sample {sample_key}: reward_scoring_seconds_by_slot is not length four"
            )
            reward_by_slot = [0.0] * SLOTS_PER_SAMPLE
        try:
            sampling_seconds = float(row_efficiency["sampling_seconds"])
            reward_encoding_seconds = float(row_efficiency["reward_encoding_seconds"])
            selection_n2_seconds = float(row_efficiency["selection_n2_seconds"])
            selection_n4_seconds = float(row_efficiency["selection_n4_seconds"])
            reward_by_slot = [float(value) for value in reward_by_slot]
        except (KeyError, TypeError, ValueError):
            failures.append(
                f"sample {sample_key}: auditable pipeline-latency components are missing"
            )
            sampling_seconds = reward_encoding_seconds = 0.0
            selection_n2_seconds = selection_n4_seconds = 0.0
            reward_by_slot = [0.0] * SLOTS_PER_SAMPLE

        row_prefixes: dict[str, dict[str, float]] = {}
        for method, prefix_size in EFFICIENCY_PREFIXES.items():
            prefix_slots = slots[:prefix_size]
            attempts = [
                attempt
                for slot in prefix_slots
                for attempt in slot.get("attempts", [])
            ]
            transports = [
                transport
                for attempt in attempts
                for transport in attempt.get("transport_attempts", [])
            ]
            try:
                prefix = {
                    "semantic_attempts": float(len(attempts)),
                    "transport_requests": float(len(transports)),
                    "input_tokens": float(
                        sum(int(attempt["usage"]["input_tokens"]) for attempt in attempts)
                    ),
                    "output_tokens": float(
                        sum(int(attempt["usage"]["output_tokens"]) for attempt in attempts)
                    ),
                    "total_tokens": float(
                        sum(int(attempt["usage"]["total_tokens"]) for attempt in attempts)
                    ),
                    "semantic_request_seconds": float(
                        sum(float(attempt["request_seconds"]) for attempt in attempts)
                    ),
                    "transport_request_seconds": float(
                        sum(float(item["request_seconds"]) for item in transports)
                    ),
                    "transport_backoff_seconds": float(
                        sum(float(item["backoff_seconds_after"]) for item in transports)
                    ),
                }
            except (KeyError, TypeError, ValueError):
                failures.append(
                    f"sample {sample_key}: raw prefix efficiency trace is malformed"
                )
                prefix = {measure: 0.0 for measure in measures}
            row_prefixes[method] = prefix
            for measure in measures:
                values[method][measure].append(prefix[measure])

            prefix_label = prefix_size
            row_checks = {
                f"best_of_{prefix_label}_semantic_attempts": prefix[
                    "semantic_attempts"
                ],
                f"best_of_{prefix_label}_generation_requests": prefix[
                    "transport_requests"
                ],
                f"best_of_{prefix_label}_input_tokens": prefix["input_tokens"],
                f"best_of_{prefix_label}_output_tokens": prefix["output_tokens"],
                f"best_of_{prefix_label}_total_tokens": prefix["total_tokens"],
            }
            for key, expected in row_checks.items():
                if not _numeric_matches(row_efficiency.get(key), expected):
                    failures.append(
                        f"sample {sample_key}: efficiency.{key} differs from raw slots"
                    )

            if prefix_size == 1:
                online_key = "online_baseline_seconds"
                online_seconds = sampling_seconds + prefix["semantic_request_seconds"]
            else:
                online_key = f"online_best_of_{prefix_size}_seconds"
                selection_seconds = (
                    selection_n2_seconds if prefix_size == 2 else selection_n4_seconds
                )
                online_seconds = (
                    sampling_seconds
                    + prefix["semantic_request_seconds"]
                    + reward_encoding_seconds
                    + sum(reward_by_slot[:prefix_size])
                    + selection_seconds
                )
            online_seconds = round(online_seconds, 6)
            online_values[method].append(online_seconds)
            if not _numeric_matches(row_efficiency.get(online_key), online_seconds):
                failures.append(
                    f"sample {sample_key}: efficiency.{online_key} differs from raw slots"
                )

        full = row_prefixes["best_of_4"]
        full_checks = {
            "generation_requests": full["transport_requests"],
            "generation_transport_requests": full["transport_requests"],
            "generation_semantic_attempts": full["semantic_attempts"],
            "generation_input_tokens": full["input_tokens"],
            "generation_output_tokens": full["output_tokens"],
            "generation_total_tokens": full["total_tokens"],
        }
        for key, expected in full_checks.items():
            if not _numeric_matches(row_efficiency.get(key), expected):
                failures.append(
                    f"sample {sample_key}: efficiency.{key} differs from raw slots"
                )

    summaries = {
        method: {
            measure: _efficiency_stats(raw_values)
            for measure, raw_values in method_values.items()
        }
        for method, method_values in values.items()
    }
    online_summaries = {
        method: _efficiency_stats(raw_values)
        for method, raw_values in online_values.items()
    }
    failures.extend(
        _crosscheck_efficiency_metrics(metrics, summaries, online_summaries)
    )
    failures.extend(_crosscheck_efficiency_analysis(analysis, summaries))

    exported = {key: metrics.get(key, 0.0) for key in EFFICIENCY_KEYS}
    for method, prefix_size in EFFICIENCY_PREFIXES.items():
        summary = summaries[method]
        exported.update(
            {
                f"mean_best_of_{prefix_size}_input_tokens": summary[
                    "input_tokens"
                ]["mean"],
                f"mean_best_of_{prefix_size}_output_tokens": summary[
                    "output_tokens"
                ]["mean"],
                f"mean_best_of_{prefix_size}_total_tokens": summary[
                    "total_tokens"
                ]["mean"],
                f"mean_best_of_{prefix_size}_generation_requests": summary[
                    "transport_requests"
                ]["mean"],
                f"mean_best_of_{prefix_size}_semantic_attempts": summary[
                    "semantic_attempts"
                ]["mean"],
                f"mean_best_of_{prefix_size}_transport_requests": summary[
                    "transport_requests"
                ]["mean"],
                f"mean_best_of_{prefix_size}_semantic_request_seconds": summary[
                    "semantic_request_seconds"
                ]["mean"],
                f"mean_best_of_{prefix_size}_transport_request_seconds": summary[
                    "transport_request_seconds"
                ]["mean"],
                f"mean_best_of_{prefix_size}_transport_backoff_seconds": summary[
                    "transport_backoff_seconds"
                ]["mean"],
            }
        )
        online_stem = (
            "online_baseline_seconds"
            if prefix_size == 1
            else f"online_best_of_{prefix_size}_seconds"
        )
        for statistic in ("mean", "median", "p95"):
            exported[f"{statistic}_{online_stem}"] = online_summaries[method][
                statistic
            ]
    return exported, failures


def _crosscheck_efficiency_metrics(
    metrics: dict[str, Any],
    summaries: dict[str, dict[str, dict[str, float]]],
    online_summaries: dict[str, dict[str, float]],
) -> list[str]:
    failures: list[str] = []
    for method, prefix_size in EFFICIENCY_PREFIXES.items():
        summary = summaries[method]
        for measure in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "transport_requests",
        ):
            metric_suffix = (
                "generation_requests"
                if measure == "transport_requests"
                else measure
            )
            for statistic in ("mean", "median", "p95"):
                key = f"{statistic}_best_of_{prefix_size}_{metric_suffix}"
                if not _numeric_matches(metrics.get(key), summary[measure][statistic]):
                    failures.append(f"metrics.{key} differs from raw fixed slots")
        online_stem = (
            "online_baseline_seconds"
            if prefix_size == 1
            else f"online_best_of_{prefix_size}_seconds"
        )
        for statistic in ("mean", "median", "p95"):
            key = f"{statistic}_{online_stem}"
            if not _numeric_matches(
                metrics.get(key), online_summaries[method][statistic]
            ):
                failures.append(f"metrics.{key} differs from raw fixed slots")
    return failures


def _crosscheck_efficiency_analysis(
    analysis: dict[str, Any],
    summaries: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    failures: list[str] = []
    method_names = {
        "best_of_1": ("direct_generation_c1", "best_of_1"),
        "best_of_2": ("best_of_2",),
        "best_of_4": ("stars_best_of_4",),
    }
    key_map = {
        "semantic_attempts": ("semantic_attempt_count",),
        "transport_requests": (
            "transport_request_count",
            "generation_requests",
            "request_count",
        ),
        "semantic_request_seconds": ("request_seconds",),
        "input_tokens": ("input_tokens",),
        "output_tokens": ("output_tokens",),
        "total_tokens": ("total_tokens",),
    }
    methods = analysis.get("methods", {})
    for prefix_method, aliases in method_names.items():
        expected = summaries[prefix_method]
        for method in aliases:
            observed = methods.get(method, {}).get(
                "efficiency_all_attempted_samples", {}
            )
            for source_measure, analysis_keys in key_map.items():
                for analysis_key in analysis_keys:
                    stats = observed.get(analysis_key, {})
                    for statistic in ("mean", "median", "p95", "total"):
                        if not _numeric_matches(
                            stats.get(statistic),
                            expected[source_measure][statistic],
                        ):
                            failures.append(
                                f"failure-aware {method} efficiency.{analysis_key}."
                                f"{statistic} differs from raw fixed slots"
                            )
    return failures


def _efficiency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "median", "p95", "total")}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    rank = (count - 1) * 0.95
    lower = int(rank)
    upper = min(count - 1, lower + 1)
    fraction = rank - lower
    p95 = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return {
        "mean": round(sum(ordered) / count, 6),
        "median": round(median, 6),
        "p95": round(p95, 6),
        "total": round(sum(ordered), 6),
    }


def _numeric_matches(observed: Any, expected: float) -> bool:
    try:
        return abs(float(observed) - float(expected)) <= 1e-5
    except (TypeError, ValueError):
        return False


def _audit_failure_records(
    records: list[dict[str, Any]],
    result_by_key: dict[str, dict[str, Any]],
    protocol_fingerprint: str,
) -> list[str]:
    failures: list[str] = []
    expected_keys = {
        sample_key
        for sample_key, result in result_by_key.items()
        if any(
            slot.get("terminal_status") == "failed"
            for slot in result.get("slot_outcomes", [])
        )
    }
    record_keys = [str(record.get("sample_key", "")) for record in records]
    if len(set(record_keys)) != len(record_keys):
        failures.append("generation failure records contain duplicate sample keys")
    if set(record_keys) != expected_keys:
        failures.append("generation failure records do not match samples with failed slots")
    for record in records:
        sample_key = str(record.get("sample_key", ""))
        result = result_by_key.get(sample_key)
        if result is None:
            continue
        if record.get("experiment_version") != "starsv20":
            failures.append("a generation failure record has the wrong version")
        if str(record.get("protocol_fingerprint", "")) != protocol_fingerprint:
            failures.append("a generation failure record has a mixed fingerprint")
        if record.get("video_id") != result.get("video_id"):
            failures.append("a generation failure record has the wrong video_id")
        expected_failed = [
            int(slot["candidate_index"])
            for slot in result.get("slot_outcomes", [])
            if slot.get("terminal_status") == "failed"
        ]
        if record.get("failed_candidate_slot_indices") != expected_failed:
            failures.append("a generation failure record has the wrong failed slots")
        candidate_ids = [
            candidate.get("candidate_id") for candidate in result.get("candidates", [])
        ]
        if int(record.get("parsed_candidates", -1)) != len(candidate_ids):
            failures.append("a generation failure record has the wrong valid count")
        if record.get("candidate_ids") != candidate_ids:
            failures.append("a generation failure record has the wrong candidate IDs")
        trace = record.get("model_report", {}).get("last_generation_trace", [])
        if len(trace) != SLOTS_PER_SAMPLE:
            failures.append("a generation failure record lacks all four slot traces")
            continue
        result_outcomes = result.get("slot_outcomes", [])
        for source, outcome in zip(trace, result_outcomes):
            if int(source.get("candidate_index", 0)) != int(
                outcome.get("candidate_index", -1)
            ) or source.get("attempts", []) != outcome.get("attempts", []):
                failures.append("a failure record trace differs from results.jsonl")
                break
    return failures


def _audit_selection_row(
    row: dict[str, Any],
    result: dict[str, Any] | None,
    protocol_fingerprint: str,
    counterfactual_specs: dict[str, tuple[dict[str, float], str]],
) -> list[str]:
    failures: list[str] = []
    if not str(row.get("sample_key") or ""):
        failures.append("a failure-aware selection row lacks sample_key")
    if result is None:
        return [*failures, "a selection row has no matching fixed result row"]
    if row.get("video_id") != result.get("video_id"):
        failures.append("a selection row video_id differs from its result row")
    if str(row.get("protocol_fingerprint", "")) != protocol_fingerprint:
        failures.append("a selection row has a mixed protocol fingerprint")
    requested_slots = row.get("requested_slots", [])
    result_slots = result.get("slot_outcomes", [])
    if not isinstance(requested_slots, list) or len(requested_slots) != SLOTS_PER_SAMPLE:
        failures.append("a selection row does not audit four requested slots")
    else:
        indices = [int(slot.get("candidate_index", 0)) for slot in requested_slots]
        if indices != [1, 2, 3, 4]:
            failures.append("a selection row has missing or reordered requested slots")
        if any(slot.get("terminal_status") not in {"valid", "failed"} for slot in requested_slots):
            failures.append("a selection row contains a nonterminal slot")
        for selection_slot, result_slot in zip(requested_slots, result_slots):
            for key in ("candidate_index", "terminal_status", "candidate_id"):
                if selection_slot.get(key) != result_slot.get(key):
                    failures.append(
                        f"a selection row requested slot disagrees on {key}"
                    )
                    break
            semantic_attempts = len(result_slot.get("attempts", []))
            transport_requests = sum(
                len(attempt.get("transport_attempts", []))
                for attempt in result_slot.get("attempts", [])
            )
            expected_usage = {
                key: sum(
                    int(attempt.get("usage", {}).get(key, 0))
                    for attempt in result_slot.get("attempts", [])
                )
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            slot_efficiency_checks = {
                "request_count": semantic_attempts,
                "semantic_attempt_count": semantic_attempts,
                "generation_requests": transport_requests,
                "transport_request_count": transport_requests,
                "request_seconds": sum(
                    float(attempt.get("request_seconds", 0.0))
                    for attempt in result_slot.get("attempts", [])
                ),
            }
            for key, expected in slot_efficiency_checks.items():
                if not _numeric_matches(selection_slot.get(key), expected):
                    failures.append(
                        f"a selection row requested slot disagrees on {key}"
                    )
            if selection_slot.get("usage") != expected_usage:
                failures.append("a selection row requested slot disagrees on usage")
    candidates = {
        int(candidate.get("candidate_index", 0)): candidate
        for candidate in result.get("candidates", [])
    }
    for candidate in candidates.values():
        reward = candidate.get("reward", {})
        try:
            recomputed_total = sum(
                FORMAL_REWARD_WEIGHTS[component] * float(reward[component])
                for component in FORMAL_REWARD_WEIGHTS
            )
        except (KeyError, TypeError, ValueError):
            failures.append("a candidate lacks complete numeric Reward components")
            continue
        if not _numeric_matches(reward.get("total"), recomputed_total):
            failures.append("a candidate Reward total differs from its weighted components")
    methods = row.get("methods", {})
    expected_prefixes = {
        "direct_generation_c1": 1,
        "best_of_1": 1,
        "best_of_2": 2,
        "stars_best_of_4": 4,
    }
    for method in METHODS:
        payload = methods.get(method)
        if not isinstance(payload, dict):
            failures.append(f"selection row lacks method {method}")
            continue
        status = payload.get("selection_status")
        if status not in {"success", "failure"}:
            failures.append(f"selection row {method} has invalid selection_status")
        prefix_size = expected_prefixes[method]
        if int(payload.get("requested_prefix_size", 0)) != prefix_size:
            failures.append(f"selection row {method} has wrong requested prefix")
        valid_indices = sorted(index for index in candidates if index <= prefix_size)
        if payload.get("valid_slot_indices") != valid_indices:
            failures.append(f"selection row {method} has wrong valid-slot indices")
        if int(payload.get("valid_slot_count", -1)) != len(valid_indices):
            failures.append(f"selection row {method} has wrong valid-slot count")
        if bool(payload.get("exact_prefix_complete")) != (
            len(valid_indices) == prefix_size
        ):
            failures.append(f"selection row {method} has wrong prefix-completion flag")
        expected_candidate = (
            max(valid_indices, key=lambda index: float(candidates[index]["reward"]["total"]))
            if valid_indices
            else None
        )
        expected_status = "success" if expected_candidate is not None else "failure"
        if status != expected_status:
            failures.append(f"selection row {method} has wrong success/failure state")
        if payload.get("candidate_index") != expected_candidate:
            failures.append(f"selection row {method} selects the wrong candidate index")
        expected_id = (
            candidates[expected_candidate]["candidate_id"]
            if expected_candidate is not None
            else None
        )
        if payload.get("candidate_id") != expected_id:
            failures.append(f"selection row {method} selects the wrong candidate ID")
        conditional = payload.get("conditional_metrics")
        effective = payload.get("effective_metrics")
        if status == "success" and not isinstance(conditional, dict):
            failures.append(f"successful selection {method} lacks conditional metrics")
        if status == "failure" and conditional is not None:
            failures.append(f"failed selection {method} has conditional metrics")
        expected_conditional = (
            _paper_metrics_from_candidate(candidates[expected_candidate])
            if expected_candidate is not None
            else None
        )
        if not _metric_dict_equal(conditional, expected_conditional):
            failures.append(f"selection row {method} has incorrect conditional metrics")
        expected_effective = expected_conditional or _failure_penalty_metrics()
        if not isinstance(effective, dict) or any(
            metric not in effective for metric in PAPER_METRICS
        ):
            failures.append(f"selection {method} lacks complete effective metrics")
        elif not _metric_dict_equal(effective, expected_effective):
            failures.append(f"selection row {method} has incorrect effective metrics")

    upper = row.get("metric_wise_best_of_4_upper_bound", {})
    valid_indices = sorted(candidates)
    expected_upper_status = "success" if valid_indices else "failure"
    if upper.get("selection_status") != expected_upper_status:
        failures.append("selection row Metric-wise Best@4 has wrong failure state")
    if int(upper.get("effective_pool_size", -1)) != len(valid_indices):
        failures.append("selection row Metric-wise Best@4 has wrong effective pool size")
    if not valid_indices:
        if upper.get("metrics") is not None:
            failures.append("failed Metric-wise Best@4 has conditional metrics")
        if not _metric_dict_equal(upper.get("effective_metrics"), _failure_penalty_metrics()):
            failures.append("failed Metric-wise Best@4 has wrong effective penalties")
        expected_sources = {metric: None for metric in PAPER_METRICS}
        expected_component_values = {
            component: None for component in FORMAL_REWARD_WEIGHTS
        }
        if upper.get("candidate_source_by_metric") != expected_sources:
            failures.append("failed Metric-wise Best@4 has fabricated metric sources")
        if upper.get("reward_component_maxima") != expected_component_values:
            failures.append("failed Metric-wise Best@4 has fabricated reward maxima")
        if (
            upper.get("candidate_source_by_reward_component")
            != expected_component_values
        ):
            failures.append("failed Metric-wise Best@4 has fabricated reward sources")
    elif not isinstance(upper.get("metrics"), dict):
        failures.append("successful Metric-wise Best@4 lacks metrics")
    else:
        ordered_candidates = [candidates[index] for index in valid_indices]
        component_maxima: dict[str, float] = {}
        component_sources: dict[str, str] = {}
        for component in FORMAL_REWARD_WEIGHTS:
            source = max(
                ordered_candidates,
                key=lambda candidate: float(
                    candidate.get("reward", {}).get(component, 0.0)
                ),
            )
            component_maxima[component] = float(
                source.get("reward", {}).get(component, 0.0)
            )
            component_sources[component] = str(source["candidate_id"])
        denominator = sum(FORMAL_REWARD_WEIGHTS.values())
        expected_reward = sum(
            FORMAL_REWARD_WEIGHTS[component] * component_maxima[component]
            for component in FORMAL_REWARD_WEIGHTS
        ) / denominator
        candidate_metrics = [
            _paper_metrics_from_candidate(candidate)
            for candidate in ordered_candidates
        ]
        expected_metrics = {"reward": expected_reward}
        expected_sources: dict[str, str | None] = {"reward": None}
        for metric in PAPER_METRICS:
            if metric in {"reward", "rep"}:
                continue
            source_index = max(
                range(len(candidate_metrics)),
                key=lambda index: candidate_metrics[index][metric],
            )
            expected_metrics[metric] = candidate_metrics[source_index][metric]
            expected_sources[metric] = str(
                ordered_candidates[source_index]["candidate_id"]
            )
        source_index = min(
            range(len(candidate_metrics)),
            key=lambda index: candidate_metrics[index]["rep"],
        )
        expected_metrics["rep"] = candidate_metrics[source_index]["rep"]
        expected_sources["rep"] = str(
            ordered_candidates[source_index]["candidate_id"]
        )
        if not _metric_dict_equal(upper.get("metrics"), expected_metrics):
            failures.append("Metric-wise Best@4 has incorrect component-wise metrics")
        if not _metric_dict_equal(upper.get("effective_metrics"), expected_metrics):
            failures.append("Metric-wise Best@4 has incorrect effective metrics")
        if upper.get("candidate_source_by_metric") != expected_sources:
            failures.append("Metric-wise Best@4 has incorrect metric sources")
        observed_maxima = upper.get("reward_component_maxima", {})
        if any(
            component not in observed_maxima
            or not _numeric_matches(observed_maxima[component], value)
            for component, value in component_maxima.items()
        ):
            failures.append("Metric-wise Best@4 has incorrect reward maxima")
        if upper.get("candidate_source_by_reward_component") != component_sources:
            failures.append("Metric-wise Best@4 has incorrect reward sources")
    if upper.get("single_realizable_candidate") is not False:
        failures.append("Metric-wise Best@4 must be marked non-realizable")
    if upper.get("deployable") is not False:
        failures.append("Metric-wise Best@4 must be marked non-deployable")

    valid_indices = sorted(candidates)
    for method, (weights, _) in counterfactual_specs.items():
        payload = methods.get(method)
        if not isinstance(payload, dict):
            failures.append(f"selection row lacks counterfactual method {method}")
            continue
        expected_candidate = (
            max(
                valid_indices,
                key=lambda index: _counterfactual_reward(candidates[index], weights),
            )
            if valid_indices
            else None
        )
        expected_status = "success" if expected_candidate is not None else "failure"
        if payload.get("selection_status") != expected_status:
            failures.append(f"selection row {method} has wrong failure state")
        if int(payload.get("requested_prefix_size", 0)) != SLOTS_PER_SAMPLE:
            failures.append(f"selection row {method} has wrong requested pool size")
        if payload.get("valid_slot_indices") != valid_indices:
            failures.append(f"selection row {method} has wrong valid-slot indices")
        if int(payload.get("valid_slot_count", -1)) != len(valid_indices):
            failures.append(f"selection row {method} has wrong valid-slot count")
        if payload.get("candidate_index") != expected_candidate:
            failures.append(f"selection row {method} selects the wrong candidate")
        expected_id = (
            candidates[expected_candidate]["candidate_id"]
            if expected_candidate is not None
            else None
        )
        if payload.get("candidate_id") != expected_id:
            failures.append(f"selection row {method} selects the wrong candidate ID")
        expected_conditional = (
            _paper_metrics_from_candidate(candidates[expected_candidate])
            if expected_candidate is not None
            else None
        )
        if not _metric_dict_equal(
            payload.get("conditional_metrics"), expected_conditional
        ):
            failures.append(f"selection row {method} has wrong conditional metrics")
        expected_effective = expected_conditional or _failure_penalty_metrics()
        if not _metric_dict_equal(payload.get("effective_metrics"), expected_effective):
            failures.append(f"selection row {method} has wrong effective metrics")
        expected_score = (
            _counterfactual_reward(candidates[expected_candidate], weights)
            if expected_candidate is not None
            else None
        )
        observed_score = payload.get("selection_score")
        if expected_score is None:
            if observed_score is not None:
                failures.append(f"failed selection row {method} has a selection score")
        else:
            try:
                score_matches = abs(float(observed_score) - expected_score) <= 1e-6
            except (TypeError, ValueError):
                score_matches = False
            if not score_matches:
                failures.append(f"selection row {method} has wrong selection score")
    return failures


def _paper_metrics_from_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    reward = candidate.get("reward", {})
    quality = candidate.get("quality", {})
    violations = reward.get("control_violations", reward.get("violations", []))
    return {
        "reward": float(reward["total"]),
        "mv_align": float(reward["alignment"]),
        "csr": 0.0 if violations else 1.0,
        "semantic_point_coverage": float(quality["semantic_point_coverage"]),
        "cider_lite": float(quality["cider_lite"]),
        "rep": float(quality["repetition_rate"]),
    }


def _failure_penalty_metrics() -> dict[str, float]:
    return {
        "reward": 0.0,
        "mv_align": 0.0,
        "csr": 0.0,
        "semantic_point_coverage": 0.0,
        "cider_lite": 0.0,
        "rep": 1.0,
    }


def _metric_dict_equal(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is expected
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    try:
        return all(
            metric in observed
            and abs(float(observed[metric]) - float(expected[metric])) <= 1e-6
            for metric in PAPER_METRICS
        )
    except (TypeError, ValueError):
        return False


def _counterfactual_method_specs() -> dict[str, tuple[dict[str, float], str]]:
    specifications: dict[str, tuple[dict[str, float], str]] = {}
    for component, weight in FORMAL_REWARD_WEIGHTS.items():
        leave_one_out = dict(FORMAL_REWARD_WEIGHTS)
        leave_one_out[component] = 0.0
        specifications[f"leave_one_out__{component}"] = (
            leave_one_out,
            "leave_one_out",
        )
        single = {name: 0.0 for name in FORMAL_REWARD_WEIGHTS}
        single[component] = 1.0
        specifications[f"single_component__{component}"] = (
            single,
            "single_component",
        )
        for factor, label in ((0.8, "0_8"), (1.2, "1_2")):
            perturbed = dict(FORMAL_REWARD_WEIGHTS)
            perturbed[component] = weight * factor
            specifications[f"weight_sensitivity__{component}__x{label}"] = (
                perturbed,
                "weight_sensitivity",
            )
    return specifications


def _counterfactual_reward(
    candidate: dict[str, Any], weights: dict[str, float]
) -> float:
    denominator = sum(weight for weight in weights.values() if weight > 0.0)
    reward = candidate.get("reward", {})
    return sum(
        weight * float(reward.get(component, 0.0))
        for component, weight in weights.items()
        if weight > 0.0
    ) / denominator


def _audit_method_summary(
    method: str,
    payload: Any,
    requested_samples: int,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return [f"failure-aware analysis lacks method {method}"]
    successful = int(payload.get("successful_sample_count", -1))
    failed = int(payload.get("failed_sample_count", -1))
    if int(payload.get("requested_sample_count", -1)) != requested_samples:
        failures.append(f"{method} requested_sample_count is inconsistent")
    if successful < 0 or failed < 0 or successful + failed != requested_samples:
        failures.append(f"{method} success/failure counts do not cover N samples")
    _check_rate(
        failures,
        f"{method}.selection_success_rate",
        payload.get("selection_success_rate"),
        max(0, successful),
        requested_samples,
    )
    conditional = payload.get("conditional_means")
    effective = payload.get("effective_means")
    counts = payload.get("conditional_valid_sample_counts")
    if not isinstance(conditional, dict) or not isinstance(effective, dict):
        failures.append(f"{method} lacks conditional/effective means")
    else:
        for metric in PAPER_METRICS:
            if metric not in conditional or metric not in effective:
                failures.append(f"{method} lacks {metric} conditional/effective value")
    if not isinstance(counts, dict) or any(
        int(counts.get(metric, -1)) != successful for metric in PAPER_METRICS
    ):
        failures.append(f"{method} conditional denominators do not equal successful samples")
    exact_count = int(payload.get("exact_prefix_completion_count", -1))
    if exact_count < 0 or exact_count > requested_samples:
        failures.append(f"{method} has invalid exact-prefix completion count")
    _check_rate(
        failures,
        f"{method}.exact_prefix_completion_rate",
        payload.get("exact_prefix_completion_rate"),
        max(0, exact_count),
        requested_samples,
    )
    histogram = payload.get("valid_slots_in_prefix_histogram")
    if not isinstance(histogram, dict) or sum(int(value) for value in histogram.values()) != requested_samples:
        failures.append(f"{method} valid-slot histogram does not cover N samples")
    for key in (
        "conditional_cluster_bootstrap_ci95",
        "effective_cluster_bootstrap_ci95",
    ):
        ci = payload.get(key)
        if not isinstance(ci, dict) or any(metric not in ci for metric in PAPER_METRICS):
            failures.append(f"{method} lacks complete {key}")
    return failures


def _audit_method_summary_against_selections(
    method: str,
    payload: Any,
    selections: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    outcomes = [row.get("methods", {}).get(method, {}) for row in selections]
    failures: list[str] = []
    successful = sum(item.get("selection_status") == "success" for item in outcomes)
    exact = sum(bool(item.get("exact_prefix_complete")) for item in outcomes)
    valid_counts = [int(item.get("valid_slot_count", -1)) for item in outcomes]
    if int(payload.get("successful_sample_count", -1)) != successful:
        failures.append(f"{method} summary success count differs from selections")
    if int(payload.get("exact_prefix_completion_count", -1)) != exact:
        failures.append(f"{method} summary prefix count differs from selections")
    histogram = {
        str(value): valid_counts.count(value) for value in sorted(set(valid_counts))
    }
    if payload.get("valid_slots_in_prefix_histogram") != histogram:
        failures.append(f"{method} summary valid-slot histogram differs from selections")

    conditional_rows = [
        item.get("conditional_metrics")
        for item in outcomes
        if item.get("selection_status") == "success"
    ]
    effective_rows = [item.get("effective_metrics") for item in outcomes]
    for metric in PAPER_METRICS:
        conditional_values = [float(row[metric]) for row in conditional_rows]
        effective_values = [float(row[metric]) for row in effective_rows]
        conditional_mean = (
            sum(conditional_values) / len(conditional_values)
            if conditional_values
            else None
        )
        effective_mean = sum(effective_values) / max(1, len(effective_values))
        observed_conditional = payload.get("conditional_means", {}).get(metric)
        if conditional_mean is None:
            if observed_conditional is not None:
                failures.append(f"{method} conditional {metric} mean is fabricated")
        elif abs(float(observed_conditional) - conditional_mean) > 1e-6:
            failures.append(f"{method} conditional {metric} mean differs from selections")
        if abs(float(payload.get("effective_means", {}).get(metric, -9.0)) - effective_mean) > 1e-6:
            failures.append(f"{method} effective {metric} mean differs from selections")
    return failures


def _audit_upper_bound(payload: Any, requested_samples: int) -> list[str]:
    if not isinstance(payload, dict):
        return ["failure-aware analysis lacks Metric-wise Best@4 upper bound"]
    failures = _audit_method_summary(
        "metric_wise_best_of_4_upper_bound", payload, requested_samples
    )
    if "mean_effective_pool_size" not in payload or "effective_pool_size_histogram" not in payload:
        failures.append("Metric-wise Best@4 lacks effective-pool-size disclosure")
    return failures


def _audit_upper_summary_against_selections(
    payload: Any,
    selections: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    outcomes = [
        row.get("metric_wise_best_of_4_upper_bound", {}) for row in selections
    ]
    failures: list[str] = []
    successes = sum(item.get("selection_status") == "success" for item in outcomes)
    pool_sizes = [int(item.get("effective_pool_size", -1)) for item in outcomes]
    if int(payload.get("successful_sample_count", -1)) != successes:
        failures.append("Metric-wise Best@4 success count differs from selections")
    histogram = {
        str(value): pool_sizes.count(value) for value in sorted(set(pool_sizes))
    }
    if payload.get("effective_pool_size_histogram") != histogram:
        failures.append("Metric-wise Best@4 pool histogram differs from selections")
    expected_mean_pool = sum(pool_sizes) / max(1, len(pool_sizes))
    if abs(float(payload.get("mean_effective_pool_size", -1.0)) - expected_mean_pool) > 1e-6:
        failures.append("Metric-wise Best@4 mean pool size differs from selections")
    conditional_rows = [
        item.get("metrics")
        for item in outcomes
        if item.get("selection_status") == "success"
    ]
    effective_rows = [item.get("effective_metrics") for item in outcomes]
    for metric in PAPER_METRICS:
        conditional_values = [float(row[metric]) for row in conditional_rows]
        effective_values = [float(row[metric]) for row in effective_rows]
        conditional_mean = (
            sum(conditional_values) / len(conditional_values)
            if conditional_values
            else None
        )
        observed_conditional = payload.get("conditional_means", {}).get(metric)
        if conditional_mean is None:
            if observed_conditional is not None:
                failures.append(f"Metric-wise Best@4 {metric} mean is fabricated")
        elif abs(float(observed_conditional) - conditional_mean) > 1e-6:
            failures.append(f"Metric-wise Best@4 {metric} conditional mean differs")
        effective_mean = sum(effective_values) / max(1, len(effective_values))
        if abs(float(payload.get("effective_means", {}).get(metric, -9.0)) - effective_mean) > 1e-6:
            failures.append(f"Metric-wise Best@4 {metric} effective mean differs")
    return failures


def _check_rate(
    failures: list[str],
    label: str,
    observed: Any,
    numerator: int,
    denominator: int,
) -> None:
    expected = numerator / max(1, denominator)
    try:
        value = float(observed)
    except (TypeError, ValueError):
        failures.append(f"{label} is missing or nonnumeric")
        return
    if abs(value - expected) > 1e-6:
        failures.append(f"{label}={value}, expected {expected}")


def _expected_fixed_sample_keys(
    metrics: dict[str, Any],
    protocol: dict[str, Any],
    failures: list[str],
) -> set[str]:


    annotation_path = Path(str(protocol.get("annotation_path", ""))).expanduser()
    if not annotation_path.is_file():
        failures.append(
            "the configured annotation manifest is unavailable for replacement audit"
        )
        return set()
    recorded_hash = str(
        protocol.get("scientific_payload", {})
        .get("data", {})
        .get("annotation_sha256", "")
    )
    observed_hash = sha256_file(annotation_path)
    if not recorded_hash or observed_hash != recorded_hash:
        failures.append("the annotation manifest hash differs from the protocol manifest")
        return set()

    data_payload = metrics.get("config", {}).get("data", {})
    try:
        data_config = DataConfig(**data_payload)
        samples = load_samples(data_config, project_root=PROJECT_ROOT)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        failures.append(f"the fixed sample set cannot be reconstructed: {exc}")
        return set()

    keys = {_sample_key(sample) for sample in samples}
    if len(keys) != len(samples):
        failures.append("the configured fixed manifest produces duplicate sample keys")
    requested_samples = int(metrics.get("requested_num_samples", 0))
    if len(samples) != requested_samples:
        failures.append(
            "the configured manifest slice does not contain the declared requested N"
        )
    return keys


def _sample_key(sample: Any) -> str:
    payload = {
        "video_id": sample.video_id,
        "video_path": sample.video_path,
        "source_row_fingerprints": sample.source_row_fingerprints,
    }
    return _sample_key_payload(payload)


def _sample_key_payload(payload: dict[str, Any]) -> str:
    video_id = str(payload.get("video_id", ""))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{video_id}::{stable_hash_int(text, 10**12):012d}"


def _audit_formal_data_protocol(formal_data: Any) -> list[str]:
    failures: list[str] = []
    if not isinstance(formal_data, dict):
        return ["formal data protocol is missing"]
    if formal_data.get("sample_unit") != "unique_video":
        failures.append(
            "STARS evidence must use data.sample_unit=`unique_video`"
        )
    try:
        formal_offset = int(formal_data.get("offset", -1))
        formal_limit = int(formal_data.get("limit", -1))
    except (TypeError, ValueError):
        formal_offset = -1
        formal_limit = -1
    if formal_offset != 0 or formal_limit != 0:
        failures.append(
            "STARS evidence requires data.offset=0 and data.limit=0"
        )
    return failures


def _candidate_control_violations(
    candidate: dict[str, Any],
) -> tuple[str, list[str]]:
    reward = candidate.get("reward")
    if not isinstance(reward, dict):
        return "missing_reward", []
    if "control_violations" in reward:
        raw = reward.get("control_violations")
        source = "control_violations"
    elif "violations" in reward:
        raw = reward.get("violations")
        source = "legacy_violations"
    else:
        return "missing_control_violations", []
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        return f"malformed_{source}", []
    return source, list(raw)


def _audit_diagnostic_invariants(
    results: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    selection_by_key = {
        str(row.get("sample_key", "")): row for row in selections
    }
    allowed_violations = set(CONTROL_CATEGORIES)
    for result in results:
        sample_key = str(result.get("sample_key", "<missing>"))
        candidates = result.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        compliant_ids: set[str] = set()
        candidate_by_id: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            candidate_by_id[candidate_id] = candidate
            status, violations = _candidate_control_violations(candidate)
            if status not in {"control_violations", "legacy_violations"}:
                failures.append(
                    f"sample {sample_key}: candidate {candidate_id or '<missing>'} "
                    "lacks auditable control violations"
                )
                continue
            if len(set(violations)) != len(violations):
                failures.append(
                    f"sample {sample_key}: candidate {candidate_id} has duplicate "
                    "control violations"
                )
            unknown = sorted(set(violations) - allowed_violations)
            if unknown:
                failures.append(
                    f"sample {sample_key}: candidate {candidate_id} has unknown "
                    f"control violations {unknown}"
                )
            if not violations:
                compliant_ids.add(candidate_id)

        selection = selection_by_key.get(sample_key, {})
        stars = selection.get("methods", {}).get("stars_best_of_4", {})
        if stars.get("selection_status") == "success":
            selected_id = str(stars.get("candidate_id", ""))
            selected = candidate_by_id.get(selected_id)
            if selected is None:
                failures.append(
                    f"sample {sample_key}: STARS diagnostic selection cannot be joined"
                )
            else:
                status, selected_violations = _candidate_control_violations(selected)
                if status in {"control_violations", "legacy_violations"}:
                    expected_csr = 0.0 if selected_violations else 1.0
                    observed_csr = stars.get("conditional_metrics", {}).get("csr")
                    if not _numeric_matches(observed_csr, expected_csr):
                        failures.append(
                            f"sample {sample_key}: STARS CSR differs from selected "
                            "candidate control violations"
                        )

        upper = selection.get("metric_wise_best_of_4_upper_bound", {})
        if upper.get("selection_status") == "success":
            expected_upper_csr = 1.0 if compliant_ids else 0.0
            observed_upper_csr = upper.get("metrics", {}).get("csr")
            if not _numeric_matches(observed_upper_csr, expected_upper_csr):
                failures.append(
                    f"sample {sample_key}: Metric-wise Best@4 CSR differs from "
                    "candidate-pool compliance availability"
                )
    return failures


def _run_result_selection_pairs(
    run: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results = run.get("audit", {}).get("result_rows", [])
    selections = run.get("audit", {}).get("selection_rows", [])
    selection_by_key = {
        str(row.get("sample_key", "")): row for row in selections
    }
    return [
        (result, selection_by_key.get(str(result.get("sample_key", "")), {}))
        for result in results
    ]


def _candidate_maps(
    result: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_index: dict[int, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        try:
            index = int(candidate.get("candidate_index", 0))
        except (TypeError, ValueError):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        by_index[index] = candidate
        by_id[candidate_id] = candidate
    return by_index, by_id


def _known_compliant_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    compliant: list[dict[str, Any]] = []
    unknown_count = 0
    violation_counts: dict[str, int] = {}
    for candidate in candidates:
        status, violations = _candidate_control_violations(candidate)
        if status not in {"control_violations", "legacy_violations"}:
            unknown_count += 1
            continue
        if not violations:
            compliant.append(candidate)
        for violation in violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
    return compliant, unknown_count, violation_counts


def _candidate_metric(candidate: dict[str, Any], metric: str) -> float:
    return _paper_metrics_from_candidate(candidate)[metric]


def _best_compliant_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compliant, _, _ = _known_compliant_candidates(candidates)
    if not compliant:
        return []
    best_value = max(float(candidate["reward"]["total"]) for candidate in compliant)
    return [
        candidate
        for candidate in compliant
        if abs(float(candidate["reward"]["total"]) - best_value) <= 1e-12
    ]


def _selected_candidate(
    selection: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = selection.get("methods", {}).get("stars_best_of_4", {})
    if payload.get("selection_status") != "success":
        return payload, None
    return payload, candidates_by_id.get(str(payload.get("candidate_id", "")))


def _write_csr_pool_diagnostics(
    path: Path, runs: list[dict[str, Any]]
) -> None:
    header = [
        "condition",
        "dataset",
        "model",
        "sample_key",
        "video_id",
        "valid_candidate_count",
        "known_control_audit_candidate_count",
        "unknown_control_audit_candidate_count",
        "compliant_candidate_count",
        "pool_has_compliant_candidate",
        "pool_control_audit_status",
        "active_control_categories",
        "inactive_control_categories",
        "candidate_violation_counts",
        "candidate_violation_signatures",
        "pool_violation_union",
        "pool_violation_intersection",
        "minimum_control_violation_count",
        "minimum_violation_candidate_ids",
        "stars_selection_status",
        "stars_candidate_index",
        "stars_candidate_id",
        "stars_reward",
        "stars_csr",
        "stars_control_audit_status",
        "stars_control_violation_count",
        "stars_control_violations",
        "best_compliant_candidate_index",
        "best_compliant_candidate_id",
        "best_compliant_reward",
        "best_compliant_tie_count",
        "best_compliant_candidate_ids",
        "selector_missed_compliant_candidate",
        "stars_reward_minus_best_compliant_reward",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for run in runs:
            for result, selection in _run_result_selection_pairs(run):
                by_index, by_id = _candidate_maps(result)
                candidates = [by_index[index] for index in sorted(by_index)]
                compliant, unknown_count, violation_counts = (
                    _known_compliant_candidates(candidates)
                )
                control_signatures: dict[str, list[str] | None] = {}
                known_violation_sets: list[set[str]] = []
                for candidate in candidates:
                    status, violations = _candidate_control_violations(candidate)
                    candidate_id = str(candidate.get("candidate_id", ""))
                    if status in {"control_violations", "legacy_violations"}:
                        control_signatures[candidate_id] = violations
                        known_violation_sets.append(set(violations))
                    else:
                        control_signatures[candidate_id] = None
                violation_union = sorted(
                    set().union(*known_violation_sets) if known_violation_sets else set()
                )
                violation_intersection = sorted(
                    set.intersection(*known_violation_sets)
                    if known_violation_sets
                    else set()
                )
                minimum_violation_count: int | str = ""
                minimum_violation_ids: list[str] = []
                if not unknown_count and candidates:
                    minimum_violation_count = min(
                        len(control_signatures[str(candidate.get("candidate_id", ""))] or [])
                        for candidate in candidates
                    )
                    minimum_violation_ids = [
                        str(candidate.get("candidate_id", ""))
                        for candidate in candidates
                        if len(
                            control_signatures[
                                str(candidate.get("candidate_id", ""))
                            ]
                            or []
                        )
                        == minimum_violation_count
                    ]
                configured_salient_points = (
                    run.get("metrics", {})
                    .get("config", {})
                    .get("generation", {})
                    .get("salient_points", [])
                )
                inactive_categories = (
                    [] if configured_salient_points else ["key_information_order"]
                )
                active_categories = [
                    category
                    for category in CONTROL_CATEGORIES
                    if category not in inactive_categories
                ]
                best_compliant = _best_compliant_candidates(candidates)
                best = best_compliant[0] if best_compliant else None
                stars_payload, stars = _selected_candidate(selection, by_id)
                stars_status, stars_violations = (
                    _candidate_control_violations(stars)
                    if stars is not None
                    else ("unavailable", [])
                )
                stars_control_known = stars_status in {
                    "control_violations",
                    "legacy_violations",
                }
                control_status = (
                    "no_valid_candidate"
                    if not candidates
                    else "incomplete_control_audit"
                    if unknown_count
                    else "complete"
                )
                pool_has_compliant: bool | str = (
                    "" if unknown_count else bool(compliant)
                )
                missed: bool | str = ""
                if stars is not None and not unknown_count and stars_status in {
                    "control_violations",
                    "legacy_violations",
                }:
                    missed = bool(compliant) and bool(stars_violations)
                stars_reward = (
                    float(stars.get("reward", {}).get("total"))
                    if stars is not None
                    else ""
                )
                best_reward = (
                    float(best.get("reward", {}).get("total"))
                    if best is not None
                    else ""
                )
                reward_delta = (
                    float(stars_reward) - float(best_reward)
                    if stars_reward != "" and best_reward != ""
                    else ""
                )
                writer.writerow(
                    _identity(run)
                    + [
                        result.get("sample_key", ""),
                        result.get("video_id", ""),
                        len(candidates),
                        len(candidates) - unknown_count,
                        unknown_count,
                        len(compliant),
                        pool_has_compliant,
                        control_status,
                        json.dumps(active_categories, ensure_ascii=False),
                        json.dumps(inactive_categories, ensure_ascii=False),
                        json.dumps(violation_counts, ensure_ascii=False, sort_keys=True),
                        json.dumps(
                            control_signatures,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(violation_union, ensure_ascii=False),
                        json.dumps(violation_intersection, ensure_ascii=False),
                        minimum_violation_count,
                        json.dumps(minimum_violation_ids, ensure_ascii=False),
                        stars_payload.get("selection_status", "missing"),
                        stars.get("candidate_index", "") if stars else "",
                        stars.get("candidate_id", "") if stars else "",
                        stars_reward,
                        (
                            0.0 if stars_violations else 1.0
                            if stars is not None and stars_control_known
                            else ""
                        ),
                        stars_status,
                        len(stars_violations) if stars_control_known else "",
                        (
                            json.dumps(stars_violations, ensure_ascii=False)
                            if stars_control_known
                            else ""
                        ),
                        best.get("candidate_index", "") if best else "",
                        best.get("candidate_id", "") if best else "",
                        best_reward,
                        len(best_compliant),
                        json.dumps(
                            [candidate.get("candidate_id", "") for candidate in best_compliant],
                            ensure_ascii=False,
                        ),
                        missed,
                        reward_delta,
                    ]
                )


def _write_selector_misses(path: Path, runs: list[dict[str, Any]]) -> None:
    reward_components = list(FORMAL_REWARD_WEIGHTS)
    comparison_metrics = ["semantic_point_coverage", "cider_lite", "rep"]
    header = [
        "condition",
        "dataset",
        "model",
        "sample_key",
        "video_id",
        "valid_candidate_count",
        "compliant_candidate_count",
        "stars_candidate_index",
        "stars_candidate_id",
        "stars_control_violations",
        "best_compliant_candidate_index",
        "best_compliant_candidate_id",
        "best_compliant_tie_count",
        "best_compliant_candidate_ids",
        "stars_reward",
        "best_compliant_reward",
        "stars_reward_minus_best_compliant_reward",
    ]
    for component in reward_components:
        header.extend(
            [
                f"stars_{component}",
                f"best_compliant_{component}",
                f"stars_minus_best_compliant_{component}",
                f"{component}_weight",
                f"weighted_stars_minus_best_compliant_{component}",
            ]
        )
    header.extend(
        [
            "weighted_component_delta_sum",
            "reward_delta_reconciliation_error",
        ]
    )
    for metric in comparison_metrics:
        header.extend(
            [
                f"stars_{metric}",
                f"best_compliant_{metric}",
                f"stars_minus_best_compliant_{metric}",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for run in runs:
            for result, selection in _run_result_selection_pairs(run):
                by_index, by_id = _candidate_maps(result)
                candidates = [by_index[index] for index in sorted(by_index)]
                compliant, unknown_count, _ = _known_compliant_candidates(candidates)
                stars_payload, stars = _selected_candidate(selection, by_id)
                if stars is None or unknown_count or not compliant:
                    continue
                _, stars_violations = _candidate_control_violations(stars)
                if not stars_violations:
                    continue
                best_candidates = _best_compliant_candidates(candidates)
                best = best_candidates[0]
                stars_reward = float(stars["reward"]["total"])
                best_reward = float(best["reward"]["total"])
                row = _identity(run) + [
                    result.get("sample_key", ""),
                    result.get("video_id", ""),
                    len(candidates),
                    len(compliant),
                    stars.get("candidate_index", stars_payload.get("candidate_index", "")),
                    stars.get("candidate_id", stars_payload.get("candidate_id", "")),
                    json.dumps(stars_violations, ensure_ascii=False),
                    best.get("candidate_index", ""),
                    best.get("candidate_id", ""),
                    len(best_candidates),
                    json.dumps(
                        [candidate.get("candidate_id", "") for candidate in best_candidates],
                        ensure_ascii=False,
                    ),
                    stars_reward,
                    best_reward,
                    stars_reward - best_reward,
                ]
                for component in reward_components:
                    stars_value = float(stars["reward"][component])
                    best_value = float(best["reward"][component])
                    component_delta = stars_value - best_value
                    weight = FORMAL_REWARD_WEIGHTS[component]
                    row.extend(
                        [
                            stars_value,
                            best_value,
                            component_delta,
                            weight,
                            weight * component_delta,
                        ]
                    )
                weighted_delta_sum = sum(
                    FORMAL_REWARD_WEIGHTS[component]
                    * (
                        float(stars["reward"][component])
                        - float(best["reward"][component])
                    )
                    for component in reward_components
                )
                reward_delta = stars_reward - best_reward
                row.extend(
                    [
                        weighted_delta_sum,
                        reward_delta - weighted_delta_sum,
                    ]
                )
                stars_metrics = _paper_metrics_from_candidate(stars)
                best_metrics = _paper_metrics_from_candidate(best)
                for metric in comparison_metrics:
                    stars_value = stars_metrics[metric]
                    best_value = best_metrics[metric]
                    row.extend([stars_value, best_value, stars_value - best_value])
                writer.writerow(row)


def _write_metric_agreement_regret(
    path: Path, runs: list[dict[str, Any]]
) -> None:
    header = [
        "condition",
        "dataset",
        "model",
        "sample_key",
        "video_id",
        "metric",
        "optimization_direction",
        "comparison_target",
        "availability_status",
        "valid_candidate_count",
        "stars_selection_status",
        "stars_candidate_index",
        "stars_candidate_id",
        "stars_metric_value",
        "best_pool_metric_value",
        "metric_value_regret",
        "stars_is_metric_value_optimal",
        "first_optimal_candidate_index",
        "first_optimal_candidate_id",
        "optimal_candidate_count",
        "optimal_candidate_ids",
        "stars_matches_first_optimal_candidate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for run in runs:
            for result, selection in _run_result_selection_pairs(run):
                by_index, by_id = _candidate_maps(result)
                candidates = [by_index[index] for index in sorted(by_index)]
                stars_payload, stars = _selected_candidate(selection, by_id)
                for metric in PAPER_METRICS:
                    minimize = metric == "rep"
                    direction = "minimize" if minimize else "maximize"
                    if not candidates:
                        writer.writerow(
                            _identity(run)
                            + [
                                result.get("sample_key", ""),
                                result.get("video_id", ""),
                                metric,
                                direction,
                                "best_real_candidate_in_valid_pool",
                                "no_valid_candidate",
                                0,
                                stars_payload.get("selection_status", "missing"),
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                0,
                                "[]",
                                "",
                            ]
                        )
                        continue
                    values = [
                        (candidate, _candidate_metric(candidate, metric))
                        for candidate in candidates
                    ]
                    best_value = (
                        min(value for _, value in values)
                        if minimize
                        else max(value for _, value in values)
                    )
                    optimal = [
                        candidate
                        for candidate, value in values
                        if abs(value - best_value) <= 1e-12
                    ]
                    first = optimal[0]
                    if stars is None:
                        status = "stars_selection_unavailable"
                        stars_value: float | str = ""
                        regret: float | str = ""
                        value_optimal: bool | str = ""
                        id_matches: bool | str = ""
                    else:
                        status = "available"
                        stars_value = _candidate_metric(stars, metric)
                        regret = (
                            stars_value - best_value
                            if minimize
                            else best_value - stars_value
                        )
                        if abs(float(regret)) <= 1e-12:
                            regret = 0.0
                        value_optimal = float(regret) == 0.0
                        id_matches = stars.get("candidate_id") == first.get("candidate_id")
                    writer.writerow(
                        _identity(run)
                        + [
                            result.get("sample_key", ""),
                            result.get("video_id", ""),
                            metric,
                            direction,
                            "best_real_candidate_in_valid_pool",
                            status,
                            len(candidates),
                            stars_payload.get("selection_status", "missing"),
                            stars.get("candidate_index", "") if stars else "",
                            stars.get("candidate_id", "") if stars else "",
                            stars_value,
                            best_value,
                            regret,
                            value_optimal,
                            first.get("candidate_index", ""),
                            first.get("candidate_id", ""),
                            len(optimal),
                            json.dumps(
                                [candidate.get("candidate_id", "") for candidate in optimal],
                                ensure_ascii=False,
                            ),
                            id_matches,
                        ]
                    )


def _raw_response_digest(attempt: dict[str, Any]) -> tuple[str, str]:
    if "raw_response" not in attempt:
        return "missing", ""
    value = attempt.get("raw_response")
    if value is None:
        payload = b"null"
        status = "present_null"
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        status = "present_text"
    else:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        status = "present_json"
    return status, hashlib.sha256(payload).hexdigest()


def _write_generation_failure_attempts(
    path: Path, runs: list[dict[str, Any]]
) -> None:
    header = [
        "condition",
        "dataset",
        "model",
        "sample_key",
        "video_id",
        "candidate_index",
        "slot_terminal_status",
        "slot_terminal_reason",
        "attempt_scope",
        "attempt_position",
        "recorded_attempt_number",
        "attempt_status",
        "prompt_kind",
        "prompt_sha256",
        "request_seed",
        "parse_error",
        "infrastructure_error",
        "request_seconds",
        "transport_request_count",
        "transport_request_seconds",
        "transport_backoff_seconds",
        "transport_status_counts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "raw_response_status",
        "raw_response_sha256",
        "sample_failure_record_present",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for run in runs:
            failure_keys = {
                str(record.get("sample_key", ""))
                for record in run.get("audit", {}).get("failure_records", [])
            }
            for result in run.get("audit", {}).get("result_rows", []):
                sample_key = str(result.get("sample_key", ""))
                for slot in result.get("slot_outcomes", []):
                    terminal_status = str(slot.get("terminal_status", ""))
                    scope = (
                        "terminal_slot_failure"
                        if terminal_status == "failed"
                        else "retry_then_accepted"
                    )
                    for position, attempt in enumerate(
                        slot.get("attempts", []), start=1
                    ):
                        if attempt.get("status") == "accepted":
                            continue
                        transports = attempt.get("transport_attempts", [])
                        transport_status_counts: dict[str, int] = {}
                        for transport in transports:
                            status = str(transport.get("status", "missing"))
                            transport_status_counts[status] = (
                                transport_status_counts.get(status, 0) + 1
                            )
                        usage = attempt.get("usage", {})
                        raw_status, raw_digest = _raw_response_digest(attempt)
                        writer.writerow(
                            _identity(run)
                            + [
                                sample_key,
                                result.get("video_id", ""),
                                slot.get("candidate_index", ""),
                                terminal_status,
                                slot.get("terminal_reason", ""),
                                scope,
                                position,
                                attempt.get("attempt", ""),
                                attempt.get("status", ""),
                                attempt.get("prompt_kind", ""),
                                attempt.get("prompt_sha256", ""),
                                attempt.get("request_seed", ""),
                                attempt.get("parse_error", ""),
                                attempt.get("infrastructure_error", ""),
                                attempt.get("request_seconds", ""),
                                attempt.get("transport_request_count", len(transports)),
                                attempt.get("transport_request_seconds", ""),
                                attempt.get("transport_backoff_seconds", ""),
                                json.dumps(
                                    transport_status_counts,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                usage.get("input_tokens", ""),
                                usage.get("output_tokens", ""),
                                usage.get("total_tokens", ""),
                                raw_status,
                                raw_digest,
                                sample_key in failure_keys,
                            ]
                        )


def _write_main_table(path: Path, runs: list[dict[str, Any]]) -> None:
    metric_columns = [
        column
        for metric in PAPER_METRICS
        for column in (f"conditional_{metric}", f"effective_{metric}")
    ]
    header = [
        "condition",
        "dataset",
        "model",
        "method",
        "requested_sample_count",
        "successful_sample_count",
        "failed_sample_count",
        "selection_success_rate",
        "exact_prefix_completion_rate",
        "mean_valid_slots_in_prefix",
        *metric_columns,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for run in runs:
            analysis = run["analysis"]
            rows = [(method, analysis["methods"][method]) for method in METHODS]
            rows.append(
                (
                    "metric_wise_best_of_4_non_deployable_upper_bound",
                    analysis["metric_wise_best_of_4_upper_bound"],
                )
            )
            for method, payload in rows:
                conditional = payload["conditional_means"]
                effective = payload["effective_means"]
                values = [
                    value
                    for metric in PAPER_METRICS
                    for value in (conditional.get(metric), effective.get(metric))
                ]
                writer.writerow(
                    _identity(run)
                    + [
                        method,
                        payload["requested_sample_count"],
                        payload["successful_sample_count"],
                        payload["failed_sample_count"],
                        payload["selection_success_rate"],
                        payload.get("exact_prefix_completion_rate", ""),
                        payload.get("mean_valid_slots_in_prefix", ""),
                        *values,
                    ]
                )


def _write_method_success_table(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "dataset",
                "model",
                "method",
                "requested_sample_count",
                "successful_sample_count",
                "failed_sample_count",
                "selection_success_rate",
                "exact_prefix_completion_count",
                "exact_prefix_completion_rate",
                "mean_valid_slots_in_prefix",
                "valid_slots_in_prefix_histogram",
            ]
        )
        for run in runs:
            for method in METHODS:
                payload = run["analysis"]["methods"][method]
                writer.writerow(
                    _identity(run)
                    + [
                        method,
                        payload["requested_sample_count"],
                        payload["successful_sample_count"],
                        payload["failed_sample_count"],
                        payload["selection_success_rate"],
                        payload["exact_prefix_completion_count"],
                        payload["exact_prefix_completion_rate"],
                        payload["mean_valid_slots_in_prefix"],
                        json.dumps(payload["valid_slots_in_prefix_histogram"], sort_keys=True),
                    ]
                )


def _write_slot_completion_table(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "dataset",
                "model",
                "requested_sample_count",
                "requested_slots_per_sample",
                "requested_candidate_slot_count",
                "valid_candidate_slot_count",
                "failed_candidate_slot_count",
                "candidate_slot_success_rate",
                "full_candidate_pool_sample_count",
                "full_candidate_pool_sample_rate",
                "any_valid_candidate_sample_count",
                "any_valid_candidate_sample_rate",
                "valid_pool_size_histogram",
            ]
        )
        for run in runs:
            analysis = run["analysis"]
            writer.writerow(
                _identity(run)
                + [
                    analysis["requested_sample_count"],
                    analysis["requested_slots_per_sample"],
                    analysis["requested_candidate_slot_count"],
                    analysis["valid_candidate_slot_count"],
                    analysis["failed_candidate_slot_count"],
                    analysis["candidate_slot_success_rate"],
                    analysis["full_candidate_pool_sample_count"],
                    analysis["full_candidate_pool_sample_rate"],
                    analysis["any_valid_candidate_sample_count"],
                    analysis["any_valid_candidate_sample_rate"],
                    json.dumps(analysis["valid_pool_size_histogram"], sort_keys=True),
                ]
            )


def _write_efficiency_table(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "dataset", "model", *EFFICIENCY_KEYS])
        for run in runs:
            audited = run["audit"]["efficiency"]
            writer.writerow(
                _identity(run) + [audited.get(key, 0.0) for key in EFFICIENCY_KEYS]
            )


def _write_group_table(path: Path, runs: list[dict[str, Any]], group: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "dataset",
                "model",
                "method",
                "selection_success_rate",
                *[f"conditional_{metric}" for metric in PAPER_METRICS],
                *[f"effective_{metric}" for metric in PAPER_METRICS],
                "selection_switch_rate",
            ]
        )
        for run in runs:
            analysis = run["analysis"]
            for method, method_group in analysis.get("method_groups", {}).items():
                if method_group != group:
                    continue
                payload = analysis["methods"][method]
                writer.writerow(
                    _identity(run)
                    + [
                        method,
                        payload["selection_success_rate"],
                        *[payload["conditional_means"].get(key) for key in PAPER_METRICS],
                        *[payload["effective_means"].get(key) for key in PAPER_METRICS],
                        analysis.get("counterfactual_selection_switch_rate", {}).get(method, 0.0),
                    ]
                )


def _write_diagnostics_table(path: Path, runs: list[dict[str, Any]]) -> None:
    keys = [
        "mean",
        "std",
        "min",
        "max",
        "zero_rate",
        "saturation_at_one_rate",
        "leave_one_out_selection_switch_rate",
        "single_component_selection_switch_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "dataset", "model", "component", *keys])
        for run in runs:
            for component, payload in run["analysis"].get(
                "component_diagnostics", {}
            ).items():
                writer.writerow(
                    _identity(run) + [component] + [payload.get(key, 0.0) for key in keys]
                )


def _write_integrity_table(path: Path, runs: list[dict[str, Any]]) -> None:
    keys = [
        "requested_num_samples",
        "requested_candidate_slot_count",
        "valid_candidate_slot_count",
        "failed_candidate_slot_count",
        "candidate_slot_success_rate",
        "full_candidate_pool_sample_count",
        "full_candidate_pool_sample_rate",
        "any_valid_candidate_sample_count",
        "any_valid_candidate_sample_rate",
        "evidence_eligible",
        "generation_attempt_records",
        "generation_parse_rejections",
        "generation_rejection_reason_counts",
        "generation_endpoint_errors",
        "generation_validation_retry_prompt_attempts",
        "generation_degenerate_repetition_early_stops",
        "semantic_repair_candidate_count",
        "wrapped_free_text_candidate_count",
        "json_envelope_normalized_candidate_count",
        "schema_envelope_normalized_candidate_count",
        "schema_canonicalized_candidate_count",
        "model_authored_controls_ignored_candidate_count",
        "schema_normalization_counts",
        "timeline_normalization_status_counts",
        "timeline_normalization_skip_reason_counts",
        "surrounding_free_text_candidate_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "dataset", "model", *keys, "protocol_fingerprint"])
        for run in runs:
            metrics = run["metrics"]
            writer.writerow(
                _identity(run)
                + [metrics.get(key, 0.0) for key in keys]
                + [metrics.get("protocol", {}).get("protocol_fingerprint", "")]
            )


def _identity(run: dict[str, Any]) -> list[str]:
    return [str(run["condition"]), str(run["dataset"]), str(run["model"])]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _markdown_index(runs: list[dict[str, Any]]) -> str:
    conditions = sorted({str(run["condition"]) for run in runs})
    return "\n".join(
        [
            "# STARS failure-aware evidence index",
            "",
            f"Validated fixed-manifest runs: **{len(runs)}**.",
            f"Conditions: **{', '.join(conditions)}**.",
            "",
            "## Primary evidence",
            "",
            "- `slot_completion.csv` proves that every fixed sample has exactly four terminal requested slots and discloses full-pool and valid-slot rates.",
            "- `method_success.csv` reports method-specific availability and exact-prefix completion for Direct/Best-of-1, Best-of-2, and STARS.",
            "- `main_candidate_and_upper_bound.csv` separates conditional quality among successful selections from failure-aware effective metrics over every requested sample.",
            "- `csr_pool_diagnostics.csv` distinguishes pools without a compliant candidate from cases in which STARS selected a noncompliant candidate despite a compliant alternative.",
            "- `selector_misses.csv` records only those compliance opportunities missed by the Reward selector and compares the selected candidate with the highest-Reward compliant alternative.",
            "- `metric_agreement_regret.csv` reports per-sample agreement and nonnegative value regret against the best real candidate in each valid pool for every paper metric.",
            "- `generation_failure_attempts.csv` records every rejected semantic attempt, including retries that later succeeded and attempts belonging to terminally failed slots; raw responses are represented only by SHA-256 digests.",
            "- Effective failures use 0 for Reward, MV-Align, CSR, SPC, and CIDEr-lite, and 1 for Rep; they are never presented as generated scripts.",
            "- `protocol_integrity.csv` reports retries, terminal failures, canonicalization, and the shared protocol fingerprint.",
            "",
            "## Interpretation constraints",
            "",
            "- In Best-of-K, K is the requested ordered slot-prefix size (1, 2, or 4), not a promise that K valid candidates were produced; the separate fixed-manifest sample count is n.",
            "- Conditional means must always be accompanied by method success and valid-slot rates; they are not all-sample estimates.",
            "- Effective means retain the fixed-manifest denominator and conservatively penalize method-level output failure.",
            "- Failed slots are never translated, repaired, replaced, or silently omitted, and all raw attempts remain auditable.",
            "- Direct Generation and Best-of-1 share C1 and therefore have identical success, candidate identity, conditional metrics, and effective metrics.",
            "- Metric-wise Best@4 is pool-conditioned, non-deployable, and may combine metrics from different valid slots.",
            "- Reward is optimized by STARS and is not independent validation evidence.",
            "- CIDEr-lite and SPC use held-out annotation references only for evaluation.",
        ]
    ) + "\n"


if __name__ == "__main__":
    main()
