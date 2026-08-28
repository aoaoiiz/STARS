from __future__ import annotations

from typing import Any

import numpy as np


def summarize_results(results: list[dict[str, Any]], preference_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [
        candidate
        for result in results
        for candidate in result.get("candidates", [])
    ]
    best_rows = [result["best_candidate"] for result in results if result.get("best_candidate")]
    if not candidate_rows:
        summary = {
            "num_samples": len(results),
            "num_candidates": 0,
            "preference_pairs": len(preference_pairs),
            "selected_output_success_count": 0,
            "selected_output_success_rate": 0.0,
            "source_kind_counts": _counts(
                result.get("sampling", {}).get("source_kind", "unknown")
                for result in results
            ),
            "generation_source_counts": _counts(
                result.get("generation_source", "unknown") for result in results
            ),
            "multimodal_generation_rate": _generation_rate(
                results, "multimodal_model"
            ),
            "real_video_rate": _real_video_rate(results),
            "mean_best_reward": 0.0,
            "control_success_rate_best": 0.0,
            "best_mean_semantic_point_coverage": None,
            "best_mean_cider_lite": 0.0,
            "best_mean_repetition_rate": 0.0,
        }
        summary.update(_efficiency_summary(results))
        summary.update(_candidate_pool_integrity_summary(results))
        return summary

    all_violations = [candidate["reward"]["violations"] for candidate in candidate_rows]
    best_violations = [candidate["reward"]["violations"] for candidate in best_rows]
    summary = {
        "num_samples": len(results),
        "num_candidates": len(candidate_rows),
        "selected_output_success_count": len(best_rows),
        "selected_output_success_rate": round(
            len(best_rows) / max(1, len(results)), 6
        ),
        "preference_pairs": len(preference_pairs),
        "source_kind_counts": _counts(
            result.get("sampling", {}).get("source_kind", "unknown") for result in results
        ),
        "decode_strategy_counts": _counts(
            result.get("sampling", {}).get("decode_strategy", "unknown") for result in results
        ),
        "sampling_fallback_count": sum(
            1 for result in results if result.get("sampling", {}).get("fallback_used", False)
        ),
        "sampling_fallback_rate": _mean(
            1.0 if result.get("sampling", {}).get("fallback_used", False) else 0.0
            for result in results
        ),
        "generation_source_counts": _counts(
            result.get("generation_source", "unknown") for result in results
        ),
        "multimodal_generation_rate": _generation_rate(results, "multimodal_model"),
        "control_success_rate_all": _success_rate(all_violations),
        "control_success_rate_best": _success_rate(best_violations),
        "control_violation_counts_all": _counts(
            violation for violations in all_violations for violation in violations
        ),
        "control_violation_counts_best": _counts(
            violation for violations in best_violations for violation in violations
        ),
        "constraint_violation_rate_all": 1.0 - _success_rate(all_violations),
        "constraint_violation_rate_best": 1.0 - _success_rate(best_violations),
        "mean_candidate_reward": _mean(candidate["reward"]["total"] for candidate in candidate_rows),
        "mean_best_reward": _mean(candidate["reward"]["total"] for candidate in best_rows),
        "mean_alignment": _mean(candidate["reward"]["alignment"] for candidate in candidate_rows),
        "mean_best_alignment": _mean(candidate["reward"]["alignment"] for candidate in best_rows),
        "mean_readability": _mean(candidate["reward"]["readability"] for candidate in candidate_rows),
        "mean_rhythm": _mean(candidate["reward"]["rhythm"] for candidate in candidate_rows),
        "mean_control": _mean(candidate["reward"]["control"] for candidate in candidate_rows),
        "mean_risk": _mean(candidate["reward"]["risk"] for candidate in candidate_rows),
        "risk_violation_rate_all": _mean(
            1.0 if candidate["reward"].get("risk_terms_detected", []) else 0.0
            for candidate in candidate_rows
        ),
        "risk_violation_rate_best": _mean(
            1.0 if candidate["reward"].get("risk_terms_detected", []) else 0.0
            for candidate in best_rows
        ),
        "real_video_rate": _real_video_rate(results),
        "mean_bleu_1": _quality_mean(candidate_rows, "bleu_1"),
        "mean_bleu_2": _quality_mean(candidate_rows, "bleu_2"),
        "mean_bleu_4": _quality_mean(candidate_rows, "bleu_4"),
        "mean_rouge_l": _quality_mean(candidate_rows, "rouge_l"),
        "mean_meteor": _quality_mean(candidate_rows, "meteor"),
        "mean_semantic_point_coverage": _optional_quality_mean(
            candidate_rows, "semantic_point_coverage"
        ),
        "mean_cider_lite": _quality_mean(candidate_rows, "cider_lite"),
        "mean_distinct_1": _quality_mean(candidate_rows, "distinct_1"),
        "mean_distinct_2": _quality_mean(candidate_rows, "distinct_2"),
        "mean_repetition_rate": _quality_mean(candidate_rows, "repetition_rate"),
        "mean_length_tokens": _quality_mean(candidate_rows, "length_tokens"),
        "mean_reference_token_coverage": _quality_mean(candidate_rows, "reference_token_coverage"),
        "answer_hit_rate": _quality_mean(candidate_rows, "answer_hit"),
        "mean_english_token_rate": _quality_mean(candidate_rows, "english_token_rate"),
        "non_latin_on_screen_text_candidate_rate_all": _quality_mean(
            candidate_rows, "on_screen_text_has_non_latin"
        ),
        "mean_on_screen_text_non_latin_letter_rate_all": _quality_mean(
            candidate_rows, "on_screen_text_non_latin_letter_rate"
        ),
        "best_mean_bleu_4": _quality_mean(best_rows, "bleu_4"),
        "best_mean_bleu_1": _quality_mean(best_rows, "bleu_1"),
        "best_mean_bleu_2": _quality_mean(best_rows, "bleu_2"),
        "best_mean_rouge_l": _quality_mean(best_rows, "rouge_l"),
        "best_mean_meteor": _quality_mean(best_rows, "meteor"),
        "best_mean_semantic_point_coverage": _optional_quality_mean(
            best_rows, "semantic_point_coverage"
        ),
        "best_mean_cider_lite": _quality_mean(best_rows, "cider_lite"),
        "best_mean_distinct_1": _quality_mean(best_rows, "distinct_1"),
        "best_mean_distinct_2": _quality_mean(best_rows, "distinct_2"),
        "best_mean_repetition_rate": _quality_mean(best_rows, "repetition_rate"),
        "best_mean_length_tokens": _quality_mean(best_rows, "length_tokens"),
        "best_mean_reference_token_coverage": _quality_mean(
            best_rows, "reference_token_coverage"
        ),
        "best_answer_hit_rate": _quality_mean(best_rows, "answer_hit"),
        "best_mean_english_token_rate": _quality_mean(best_rows, "english_token_rate"),
        "non_latin_on_screen_text_candidate_rate_best": _quality_mean(
            best_rows, "on_screen_text_has_non_latin"
        ),
        "mean_on_screen_text_non_latin_letter_rate_best": _quality_mean(
            best_rows, "on_screen_text_non_latin_letter_rate"
        ),
        "semantic_point_valid_sample_count": sum(
            1
            for candidate in best_rows
            if candidate.get("quality", {}).get(
                "semantic_point_reference_available", 0.0
            )
            == 1.0
        ),
    }
    summary["semantic_point_valid_sample_rate"] = round(
        summary["semantic_point_valid_sample_count"] / max(1, len(best_rows)),
        6,
    )
    summary.update(
        {
            "selected_control_success_rate": summary["control_success_rate_best"],
            "selected_constraint_violation_rate": summary[
                "constraint_violation_rate_best"
            ],
            "selected_mean_reward": summary["mean_best_reward"],
            "selected_mean_alignment": summary["mean_best_alignment"],
            "selected_mean_semantic_point_coverage": summary[
                "best_mean_semantic_point_coverage"
            ],
            "selected_mean_cider_lite": summary["best_mean_cider_lite"],
            "selected_mean_repetition_rate": summary["best_mean_repetition_rate"],
            "selected_answer_hit_rate": summary["best_answer_hit_rate"],
            "selected_mean_english_token_rate": summary[
                "best_mean_english_token_rate"
            ],
        }
    )
    summary.update(_efficiency_summary(results))
    summary.update(_candidate_pool_integrity_summary(results))
    return summary


def _success_rate(violation_lists: list[list[str]]) -> float:
    if not violation_lists:
        return 0.0
    return float(np.mean([1.0 if not violations else 0.0 for violations in violation_lists]))


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(float(np.mean(values)), 6)


def _quality_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        row.get("quality", {}).get(key)
        for row in rows
        if row.get("quality", {}).get(key) is not None
    ]
    return _mean(values)


def _optional_quality_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        row.get("quality", {}).get(key)
        for row in rows
        if row.get("quality", {}).get(key) is not None
    ]
    return _mean(values) if values else None


def _real_video_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    real = [
        1.0 if result.get("sampling", {}).get("source_kind") != "synthetic" else 0.0
        for result in results
    ]
    return round(float(np.mean(real)), 6)


def _generation_rate(results: list[dict[str, Any]], source: str) -> float:
    if not results:
        return 0.0
    return round(
        float(np.mean([1.0 if result.get("generation_source") == source else 0.0 for result in results])),
        6,
    )


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _efficiency_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "sampling_seconds",
        "reward_encoding_seconds",
        "generation_seconds",
        "generation_requests",
        "generation_input_tokens",
        "generation_output_tokens",
        "generation_total_tokens",
        "generation_seconds_per_candidate",
        "reward_scoring_seconds",
        "reward_scoring_seconds_per_candidate",
        "metric_evaluation_seconds",
        "total_pipeline_seconds",
        "reward_runner_peak_allocated_mib",
        "reward_runner_peak_reserved_mib",
        "online_baseline_seconds",
        "online_best_of_2_seconds",
        "online_best_of_4_seconds",
        "best_of_1_input_tokens",
        "best_of_2_input_tokens",
        "best_of_4_input_tokens",
        "best_of_1_output_tokens",
        "best_of_2_output_tokens",
        "best_of_4_output_tokens",
        "best_of_1_total_tokens",
        "best_of_2_total_tokens",
        "best_of_4_total_tokens",
        "best_of_1_generation_requests",
        "best_of_2_generation_requests",
        "best_of_4_generation_requests",
        "selection_n2_seconds",
        "selection_n4_seconds",
        "vlm_server_peak_allocated_mib",
        "vlm_server_peak_reserved_mib",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            float(result.get("efficiency", {}).get(key, 0.0))
            for result in results
            if key in result.get("efficiency", {})
        ]
        if not values:
            continue
        summary[f"mean_{key}"] = round(float(np.mean(values)), 6)
        summary[f"median_{key}"] = round(float(np.median(values)), 6)
        summary[f"p95_{key}"] = round(float(np.percentile(values, 95)), 6)
    return summary


def _candidate_pool_integrity_summary(results: list[dict[str, Any]]) -> dict[str, float | int]:
    slot_outcomes = [
        slot
        for result in results
        for slot in result.get(
            "slot_outcomes", result.get("generation_trace", [])
        )
    ]
    attempts = [
        attempt
        for slot in slot_outcomes
        for attempt in slot.get("attempts", [])
    ]
    parse_rejections = sum(
        1 for attempt in attempts if attempt.get("status") == "parse_rejected"
    )
    accepted_attempts = sum(
        1 for attempt in attempts if attempt.get("status") == "accepted"
    )
    rejection_reason_counts = _counts(
        attempt.get("parse_error", "unknown parse rejection")
        for attempt in attempts
        if attempt.get("status") == "parse_rejected"
    )
    endpoint_errors = sum(
        1 for attempt in attempts if attempt.get("status") == "endpoint_error"
    )
    retry_prompt_attempts = sum(
        1 for attempt in attempts if attempt.get("prompt_kind") == "validation_retry"
    )
    degenerate_repetition_early_stops = sum(
        1
        for attempt in attempts
        if attempt.get("server_metrics", {}).get(
            "degenerate_repetition_early_stop"
        )
        is True
    )
    retried_samples = sum(
        1
        for result in results
        if any(
            len(slot.get("attempts", [])) > 1
            for slot in result.get(
                "slot_outcomes", result.get("generation_trace", [])
            )
        )
    )
    model_candidates = [
        candidate
        for result in results
        if result.get("generation_source") == "multimodal_model"
        for candidate in result.get("candidates", [])
    ]
    semantic_repairs = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get("semantic_repair_applied")
        is not False
    )
    wrapped = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get("wrapped_free_text")
        is not False
    )
    normalized_json_envelopes = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get(
            "json_envelope_normalized"
        )
        is True
    )
    normalized_schema_envelopes = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get(
            "schema_envelope_normalized"
        )
        is True
    )
    schema_canonicalized = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get(
            "schema_canonicalized"
        )
        is True
    )
    ignored_model_controls = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get(
            "model_authored_control_keys_ignored"
        )
    )
    schema_normalization_counts = _counts(
        normalization
        for candidate in model_candidates
        for normalization in candidate.get("generation_provenance", {}).get(
            "schema_normalizations", []
        )
    )
    timeline_normalization_status_counts = _counts(
        str(
            candidate.get("generation_provenance", {})
            .get("timeline_normalization", {})
            .get("status", "missing")
        )
        for candidate in model_candidates
    )
    timeline_normalization_skip_reason_counts = _counts(
        str(normalization.get("reason", "missing"))
        for candidate in model_candidates
        for normalization in [
            candidate.get("generation_provenance", {}).get(
                "timeline_normalization", {}
            )
        ]
        if normalization.get("status") == "skipped"
    )
    surrounding_free_text = sum(
        1
        for candidate in model_candidates
        if candidate.get("generation_provenance", {}).get(
            "surrounding_free_text"
        )
        is not False
    )
    valid_slots = sum(
        1 for slot in slot_outcomes if slot.get("terminal_status") == "valid"
    )
    failed_slots = sum(
        1 for slot in slot_outcomes if slot.get("terminal_status") == "failed"
    )
    full_pool_samples = sum(
        1
        for result in results
        if result.get("requested_candidate_slots")
        and result.get("valid_candidate_slots")
        == result.get("requested_candidate_slots")
    )
    return {
        "requested_candidate_slot_records": len(slot_outcomes),
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "valid_candidate_slot_rate": round(
            valid_slots / max(1, valid_slots + failed_slots), 6
        ),
        "full_candidate_pool_count": full_pool_samples,
        "full_candidate_pool_rate": round(
            full_pool_samples / max(1, len(results)), 6
        ),
        "generation_attempt_records": len(attempts),
        "generation_parse_rejections": parse_rejections,
        "generation_valid_candidate_rate": round(
            accepted_attempts / max(1, accepted_attempts + parse_rejections), 6
        ),
        "generation_rejection_reason_counts": rejection_reason_counts,
        "generation_endpoint_errors": endpoint_errors,
        "generation_validation_retry_prompt_attempts": retry_prompt_attempts,
        "generation_degenerate_repetition_early_stops": (
            degenerate_repetition_early_stops
        ),
        "samples_with_generation_retry": retried_samples,
        "semantic_repair_candidate_count": semantic_repairs,
        "wrapped_free_text_candidate_count": wrapped,
        "json_envelope_normalized_candidate_count": normalized_json_envelopes,
        "schema_envelope_normalized_candidate_count": normalized_schema_envelopes,
        "schema_canonicalized_candidate_count": schema_canonicalized,
        "model_authored_controls_ignored_candidate_count": ignored_model_controls,
        "schema_normalization_counts": schema_normalization_counts,
        "timeline_normalization_status_counts": (
            timeline_normalization_status_counts
        ),
        "timeline_normalization_skip_reason_counts": (
            timeline_normalization_skip_reason_counts
        ),
        "surrounding_free_text_candidate_count": surrounding_free_text,
    }
