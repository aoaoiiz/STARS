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
        return {"num_samples": 0, "num_candidates": 0}

    all_violations = [candidate["reward"]["violations"] for candidate in candidate_rows]
    best_violations = [candidate["reward"]["violations"] for candidate in best_rows]
    return {
        "num_samples": len(results),
        "num_candidates": len(candidate_rows),
        "preference_pairs": len(preference_pairs),
        "source_kind_counts": _counts(
            result.get("sampling", {}).get("source_kind", "unknown") for result in results
        ),
        "generation_source_counts": _counts(
            result.get("generation_source", "unknown") for result in results
        ),
        "multimodal_generation_rate": _generation_rate(results, "multimodal_model"),
        "control_success_rate_all": _success_rate(all_violations),
        "control_success_rate_best": _success_rate(best_violations),
        "constraint_violation_rate_all": 1.0 - _success_rate(all_violations),
        "constraint_violation_rate_best": 1.0 - _success_rate(best_violations),
        "mean_candidate_reward": _mean(candidate["reward"]["total"] for candidate in candidate_rows),
        "mean_best_reward": _mean(candidate["reward"]["total"] for candidate in best_rows),
        "mean_alignment": _mean(candidate["reward"]["alignment"] for candidate in candidate_rows),
        "mean_best_alignment": _mean(candidate["reward"]["alignment"] for candidate in best_rows),
        "mean_readability": _mean(candidate["reward"]["readability"] for candidate in candidate_rows),
        "mean_rhythm": _mean(candidate["reward"]["rhythm"] for candidate in candidate_rows),
        "mean_text_reasoning": _mean(
            candidate["reward"].get("text_reasoning", 0.0) for candidate in candidate_rows
        ),
        "real_video_rate": _real_video_rate(results),
        "mean_bleu_1": _quality_mean(candidate_rows, "bleu_1"),
        "mean_bleu_2": _quality_mean(candidate_rows, "bleu_2"),
        "mean_bleu_4": _quality_mean(candidate_rows, "bleu_4"),
        "mean_rouge_l": _quality_mean(candidate_rows, "rouge_l"),
        "mean_meteor": _quality_mean(candidate_rows, "meteor"),
        "mean_cider_lite": _quality_mean(candidate_rows, "cider_lite"),
        "mean_distinct_1": _quality_mean(candidate_rows, "distinct_1"),
        "mean_distinct_2": _quality_mean(candidate_rows, "distinct_2"),
        "mean_repetition_rate": _quality_mean(candidate_rows, "repetition_rate"),
        "mean_length_tokens": _quality_mean(candidate_rows, "length_tokens"),
        "mean_reference_token_coverage": _quality_mean(candidate_rows, "reference_token_coverage"),
        "answer_hit_rate": _quality_mean(candidate_rows, "answer_hit"),
        "selling_point_coverage": _quality_mean(candidate_rows, "selling_point_coverage"),
        "best_mean_bleu_4": _quality_mean(best_rows, "bleu_4"),
        "best_mean_rouge_l": _quality_mean(best_rows, "rouge_l"),
        "best_mean_meteor": _quality_mean(best_rows, "meteor"),
        "best_mean_cider_lite": _quality_mean(best_rows, "cider_lite"),
    }


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
    return _mean(row.get("quality", {}).get(key, 0.0) for row in rows)


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
