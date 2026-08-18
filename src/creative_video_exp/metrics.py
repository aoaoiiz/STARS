from __future__ import annotations

from typing import Any, Iterable

import numpy as np


PAPER_METRICS = (
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [
        candidate
        for result in results
        for candidate in result.get("candidates", [])
    ]
    selected_rows = [
        result["best_candidate"]
        for result in results
        if result.get("best_candidate") is not None
    ]
    selected_metrics = [_candidate_metrics(row) for row in selected_rows]
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "num_candidates": len(candidate_rows),
        "selected_output_success_count": len(selected_rows),
        "selected_output_success_rate": round(
            len(selected_rows) / max(1, len(results)),
            6,
        ),
        "real_video_rate": _mean(
            float(result.get("sampling", {}).get("source_kind") != "synthetic")
            for result in results
        ),
        "multimodal_generation_rate": _mean(
            float(result.get("generation_source") == "multimodal_model")
            for result in results
        ),
        "source_kind_counts": _counts(
            result.get("sampling", {}).get("source_kind", "unknown")
            for result in results
        ),
        "generation_source_counts": _counts(
            result.get("generation_source", "unknown") for result in results
        ),
        "selected_metrics": {
            metric: _optional_mean(
                row.get(metric) for row in selected_metrics
            )
            for metric in PAPER_METRICS
        },
    }
    summary.update(_efficiency_summary(results))
    summary.update(_candidate_pool_integrity_summary(results))
    return summary


def _candidate_metrics(candidate: dict[str, Any]) -> dict[str, float | None]:
    reward = candidate.get("reward", {})
    quality = candidate.get("quality", {})
    violations = reward.get("violations", reward.get("control_violations", []))
    return {
        "reward": float(reward.get("total", 0.0)),
        "mv_align": float(reward.get("alignment", 0.0)),
        "csr": float(not violations),
        "semantic_point_coverage": _optional_float(
            quality.get("semantic_point_coverage")
        ),
        "cider_lite": float(quality.get("cider_lite", 0.0)),
        "rep": float(quality.get("repetition_rate", 0.0)),
    }


def _efficiency_summary(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "sampling_seconds",
        "reward_encoding_seconds",
        "generation_seconds",
        "generation_requests",
        "generation_input_tokens",
        "generation_output_tokens",
        "generation_total_tokens",
        "reward_scoring_seconds",
        "metric_evaluation_seconds",
        "direct_generation_seconds",
        "stars_seconds",
        "direct_generation_input_tokens",
        "stars_input_tokens",
        "direct_generation_output_tokens",
        "stars_output_tokens",
        "direct_generation_total_tokens",
        "stars_total_tokens",
        "direct_generation_requests",
        "stars_generation_requests",
        "total_pipeline_seconds",
        "vlm_server_peak_allocated_mib",
        "vlm_server_peak_reserved_mib",
        "reward_runner_peak_allocated_mib",
        "reward_runner_peak_reserved_mib",
    )
    summary: dict[str, float] = {}
    for key in keys:
        values = [
            float(result["efficiency"][key])
            for result in results
            if key in result.get("efficiency", {})
        ]
        if not values:
            continue
        summary[f"mean_{key}"] = round(float(np.mean(values)), 6)
        summary[f"median_{key}"] = round(float(np.median(values)), 6)
        summary[f"p95_{key}"] = round(float(np.percentile(values, 95)), 6)
    return summary


def _candidate_pool_integrity_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    slots = [
        slot
        for result in results
        for slot in result.get("slot_outcomes", [])
    ]
    attempts = [
        attempt
        for slot in slots
        for attempt in slot.get("attempts", [])
    ]
    valid_slots = sum(slot.get("terminal_status") == "valid" for slot in slots)
    failed_slots = sum(slot.get("terminal_status") == "failed" for slot in slots)
    full_pools = sum(
        int(result.get("valid_candidate_slots", 0))
        == int(result.get("requested_candidate_slots", 0))
        for result in results
    )
    return {
        "requested_candidate_slot_count": len(slots),
        "valid_candidate_slot_count": valid_slots,
        "failed_candidate_slot_count": failed_slots,
        "candidate_slot_success_rate": round(
            valid_slots / max(1, len(slots)),
            6,
        ),
        "full_candidate_pool_sample_count": full_pools,
        "full_candidate_pool_sample_rate": round(
            full_pools / max(1, len(results)),
            6,
        ),
        "generation_attempt_count": len(attempts),
        "generation_parse_rejection_count": sum(
            attempt.get("status") == "parse_rejected" for attempt in attempts
        ),
        "generation_endpoint_error_count": sum(
            attempt.get("status") == "endpoint_error" for attempt in attempts
        ),
        "generation_validation_retry_count": sum(
            attempt.get("prompt_kind") == "validation_retry"
            for attempt in attempts
        ),
    }


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(float(np.mean(present)), 6)


def _mean(values: Iterable[float]) -> float:
    present = list(values)
    if not present:
        return 0.0
    return round(float(np.mean(present)), 6)


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
