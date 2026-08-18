from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .utils import write_json, write_jsonl


PAPER_METRICS = (
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
)
HIGHER_IS_BETTER = {
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
}
LOWER_IS_BETTER = {"rep"}
METHOD_PREFIXES = {
    "direct_generation_c1": 1,
    "stars_best_of_4": 4,
}
USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")
EFFICIENCY_KEYS = (
    "online_seconds",
    "generation_requests",
    "semantic_attempts",
    "generation_seconds",
    "reward_encoding_seconds",
    "reward_scoring_seconds",
    *USAGE_KEYS,
)


def analyze_failure_aware_candidate_pool(
    results: list[dict[str, Any]],
    reward_config: dict[str, float] | None = None,
    pool_size: int = 4,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not results:
        raise ValueError("No requested-sample rows were supplied.")
    if pool_size != 4:
        raise ValueError("STARS requires four requested candidate slots.")
    fingerprints = {
        str(result.get("protocol_fingerprint", "")) for result in results
    }
    if "" in fingerprints or len(fingerprints) != 1:
        raise RuntimeError(
            "All rows must share one non-empty protocol fingerprint."
        )
    sample_keys = [_sample_key(result) for result in results]
    if len(sample_keys) != len(set(sample_keys)):
        raise RuntimeError("Result rows must have unique sample keys.")
    rng = np.random.default_rng(seed)
    cluster_ids: list[str] = []
    selections: list[dict[str, Any]] = []
    method_conditional: dict[str, list[dict[str, float] | None]] = {
        method: [] for method in METHOD_PREFIXES
    }
    method_effective: dict[str, list[dict[str, float]]] = {
        method: [] for method in METHOD_PREFIXES
    }
    method_success: dict[str, list[float]] = {
        method: [] for method in METHOD_PREFIXES
    }
    method_exact_completion: dict[str, list[float]] = {
        method: [] for method in METHOD_PREFIXES
    }
    method_valid_counts: dict[str, list[int]] = {
        method: [] for method in METHOD_PREFIXES
    }
    method_efficiency: dict[str, list[dict[str, float]]] = {
        method: [] for method in METHOD_PREFIXES
    }
    upper_conditional: list[dict[str, float] | None] = []
    upper_effective: list[dict[str, float]] = []
    upper_success: list[float] = []
    upper_pool_sizes: list[int] = []

    for result in results:
        slots = _ordered_slot_outcomes(result, pool_size)
        video_id = str(result.get("video_id", _sample_key(result)))
        cluster_ids.append(video_id)
        selection_row: dict[str, Any] = {
            "sample_key": _sample_key(result),
            "video_id": video_id,
            "protocol_fingerprint": result["protocol_fingerprint"],
            "requested_slots": [_slot_view(slot) for slot in slots],
            "methods": {},
        }
        for method, prefix_size in METHOD_PREFIXES.items():
            prefix_slots = slots[:prefix_size]
            valid_slots = [
                slot
                for slot in prefix_slots
                if slot["terminal_status"] == "valid"
            ]
            if method == "direct_generation_c1":
                selected_slot = valid_slots[0] if valid_slots else None
            else:
                selected_slot = (
                    max(
                        valid_slots,
                        key=lambda slot: _reward(slot["candidate"]),
                    )
                    if valid_slots
                    else None
                )
            conditional = (
                _candidate_metrics(selected_slot["candidate"])
                if selected_slot is not None
                else None
            )
            effective = _effective_metrics(conditional)
            efficiency = _method_efficiency(
                result,
                prefix_slots,
                method,
            )
            outcome = {
                "selection_status": (
                    "success" if selected_slot is not None else "failure"
                ),
                "requested_prefix_size": prefix_size,
                "valid_slot_count": len(valid_slots),
                "valid_slot_indices": [
                    int(slot["candidate_index"]) for slot in valid_slots
                ],
                "exact_prefix_complete": len(valid_slots) == prefix_size,
                "candidate_id": (
                    str(selected_slot["candidate_id"])
                    if selected_slot is not None
                    else None
                ),
                "candidate_index": (
                    int(selected_slot["candidate_index"])
                    if selected_slot is not None
                    else None
                ),
                "selection_score": (
                    _reward(selected_slot["candidate"])
                    if selected_slot is not None
                    else None
                ),
                "conditional_metrics": _round_metrics(conditional),
                "effective_metrics": _round_metrics(effective),
                "efficiency": {
                    key: round(float(value), 6)
                    for key, value in efficiency.items()
                },
            }
            selection_row["methods"][method] = outcome
            method_conditional[method].append(conditional)
            method_effective[method].append(effective)
            method_success[method].append(float(selected_slot is not None))
            method_exact_completion[method].append(
                float(len(valid_slots) == prefix_size)
            )
            method_valid_counts[method].append(len(valid_slots))
            method_efficiency[method].append(efficiency)

        direct = selection_row["methods"]["direct_generation_c1"]
        c1 = slots[0]
        expected_c1 = (
            c1.get("candidate_id")
            if c1.get("terminal_status") == "valid"
            else None
        )
        if direct["candidate_id"] != expected_c1:
            raise AssertionError("Direct Generation must return C1.")
        direct_reward = direct["effective_metrics"]["reward"]
        stars_reward = selection_row["methods"]["stars_best_of_4"][
            "effective_metrics"
        ]["reward"]
        if direct_reward > stars_reward + 1e-12:
            raise AssertionError(
                "STARS Reward must not be lower than Direct Generation Reward."
            )

        valid_slots = [
            slot for slot in slots if slot["terminal_status"] == "valid"
        ]
        if valid_slots:
            upper_metrics, upper_sources = _metric_wise_upper_bound(valid_slots)
        else:
            upper_metrics = None
            upper_sources = {metric: None for metric in PAPER_METRICS}
        upper_effective_metrics = _effective_metrics(upper_metrics)
        upper_pool_size = len(valid_slots)
        upper_conditional.append(upper_metrics)
        upper_effective.append(upper_effective_metrics)
        upper_success.append(float(bool(valid_slots)))
        upper_pool_sizes.append(upper_pool_size)
        selection_row["metric_wise_best_of_4_upper_bound"] = {
            "selection_status": "success" if valid_slots else "failure",
            "effective_pool_size": upper_pool_size,
            "metrics": _round_metrics(upper_metrics),
            "effective_metrics": _round_metrics(upper_effective_metrics),
            "candidate_source_by_metric": upper_sources,
            "single_realizable_candidate": False,
            "deployable": False,
        }
        selections.append(selection_row)

    methods = {
        method: _summarize_method(
            conditional_rows=method_conditional[method],
            effective_rows=method_effective[method],
            successes=method_success[method],
            exact_completions=method_exact_completion[method],
            valid_counts=method_valid_counts[method],
            efficiency_rows=method_efficiency[method],
            clusters=cluster_ids,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        for method in METHOD_PREFIXES
    }
    upper_summary = _summarize_method(
        conditional_rows=upper_conditional,
        effective_rows=upper_effective,
        successes=upper_success,
        exact_completions=[
            float(size == pool_size) for size in upper_pool_sizes
        ],
        valid_counts=upper_pool_sizes,
        efficiency_rows=method_efficiency["stars_best_of_4"],
        clusters=cluster_ids,
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
    )
    upper_summary["mean_effective_pool_size"] = round(
        float(np.mean(upper_pool_sizes)),
        6,
    )
    upper_summary["effective_pool_size_histogram"] = _histogram(
        upper_pool_sizes
    )
    valid_slot_count = sum(
        slot["terminal_status"] == "valid"
        for result in results
        for slot in _ordered_slot_outcomes(result, pool_size)
    )
    full_pool_count = sum(size == pool_size for size in upper_pool_sizes)
    any_valid_count = sum(size > 0 for size in upper_pool_sizes)
    summary = {
        "experiment_version": "stars",
        "analysis_protocol": "failure_aware_fixed_requested_slots_v1",
        "protocol_fingerprint": next(iter(fingerprints)),
        "requested_sample_count": len(results),
        "requested_slots_per_sample": pool_size,
        "requested_candidate_slot_count": len(results) * pool_size,
        "valid_candidate_slot_count": valid_slot_count,
        "failed_candidate_slot_count": len(results) * pool_size
        - valid_slot_count,
        "candidate_slot_success_rate": round(
            valid_slot_count / (len(results) * pool_size),
            6,
        ),
        "full_candidate_pool_sample_count": full_pool_count,
        "full_candidate_pool_sample_rate": round(
            full_pool_count / len(results),
            6,
        ),
        "any_valid_candidate_sample_count": any_valid_count,
        "any_valid_candidate_sample_rate": round(
            any_valid_count / len(results),
            6,
        ),
        "methods": methods,
        "metric_wise_best_of_4_upper_bound_definition": {
            "name": "Metric-wise Best@4 Upper Bound",
            "pool": "valid candidates among C1-C4",
            "selection": "independent best candidate for each metric",
            "single_realizable_candidate": False,
            "deployable": False,
        },
        "metric_wise_best_of_4_upper_bound": upper_summary,
    }
    _verify_summary(summary)
    return summary, selections


def write_failure_aware_candidate_pool_analysis(
    output_dir: str | Path,
    summary: dict[str, Any],
    selections: list[dict[str, Any]],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "failure_aware_candidate_pool_analysis.json",
        summary,
    )
    write_jsonl(output / "failure_aware_selections.jsonl", selections)
    _write_main_csv(output / "failure_aware_main_results.csv", summary)
    _write_efficiency_csv(output / "failure_aware_efficiency.csv", summary)
    (output / "failure_aware_summary.md").write_text(
        _markdown_summary(summary),
        encoding="utf-8",
    )


def _ordered_slot_outcomes(
    result: dict[str, Any],
    pool_size: int,
) -> list[dict[str, Any]]:
    raw_slots = result.get("slot_outcomes")
    if not isinstance(raw_slots, list) or len(raw_slots) != pool_size:
        raise RuntimeError(
            f"Sample {_sample_key(result)!r} must contain {pool_size} slot outcomes."
        )
    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("The candidates field must be a list.")
    candidates_by_index: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("Every scored candidate must be an object.")
        index = int(candidate.get("candidate_index", 0))
        if index in candidates_by_index or not 1 <= index <= pool_size:
            raise RuntimeError("Candidate slot indices must be unique and in C1-C4.")
        candidates_by_index[index] = candidate
    slots: list[dict[str, Any]] = []
    for expected_index, raw_slot in enumerate(raw_slots, start=1):
        index = int(raw_slot.get("candidate_index", 0))
        if index != expected_index:
            raise RuntimeError("Slot outcomes must be ordered C1-C4.")
        status = str(raw_slot.get("terminal_status", ""))
        if status not in {"valid", "failed"}:
            raise RuntimeError("Every slot must have a valid or failed status.")
        candidate = candidates_by_index.get(index)
        if (status == "valid") != (candidate is not None):
            raise RuntimeError(
                "Valid slot outcomes must join to exactly one scored candidate."
            )
        candidate_id = raw_slot.get("candidate_id")
        if candidate is not None and candidate_id != candidate.get("candidate_id"):
            raise RuntimeError("Slot and candidate identifiers do not match.")
        attempts = raw_slot.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError("Every slot must contain an auditable attempt trace.")
        if any(not isinstance(attempt, dict) for attempt in attempts):
            raise RuntimeError("Every generation attempt must be an object.")
        if "request_seconds" in raw_slot:
            request_seconds = _required_nonnegative_number(
                raw_slot,
                "request_seconds",
                f"slot C{index}",
            )
        else:
            request_seconds = sum(
                _required_nonnegative_number(
                    attempt,
                    "request_seconds",
                    f"slot C{index} attempt {position}",
                )
                for position, attempt in enumerate(attempts, start=1)
            )
        usage = _usage(raw_slot.get("usage", {}))
        slot = dict(raw_slot)
        slot.update(
            {
                "candidate_index": index,
                "candidate_id": candidate_id,
                "terminal_status": status,
                "candidate": candidate,
                "request_count": _nonnegative_int(
                    raw_slot.get("request_count", len(attempts)),
                    f"slot C{index}.request_count",
                ),
                "transport_request_count": _nonnegative_int(
                    raw_slot.get("transport_request_count", len(attempts)),
                    f"slot C{index}.transport_request_count",
                ),
                "request_seconds": request_seconds,
                "usage": usage,
            }
        )
        slots.append(slot)
    if set(candidates_by_index) != {
        int(slot["candidate_index"])
        for slot in slots
        if slot["terminal_status"] == "valid"
    }:
        raise RuntimeError("Scored candidates and valid slots do not match.")
    return slots


def _slot_view(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(slot["candidate_index"]),
        "terminal_status": slot["terminal_status"],
        "candidate_id": slot.get("candidate_id"),
        "terminal_reason": slot.get("terminal_reason", ""),
        "semantic_attempts": int(slot["request_count"]),
        "generation_requests": int(slot["transport_request_count"]),
        "request_seconds": round(float(slot["request_seconds"]), 6),
        "usage": dict(slot["usage"]),
    }


def _candidate_metrics(candidate: dict[str, Any]) -> dict[str, float]:
    reward = _required_mapping(candidate, "reward", "candidate")
    quality = _required_mapping(candidate, "quality", "candidate")
    violations = reward.get("violations")
    if not isinstance(violations, list):
        raise RuntimeError("candidate.reward.violations must be a list.")
    return {
        "reward": _required_unit_number(reward, "total", "candidate.reward"),
        "mv_align": _required_unit_number(
            reward,
            "alignment",
            "candidate.reward",
        ),
        "csr": float(not violations),
        "semantic_point_coverage": _required_unit_number(
            quality,
            "semantic_point_coverage",
            "candidate.quality",
        ),
        "cider_lite": _required_unit_number(
            quality,
            "cider_lite",
            "candidate.quality",
        ),
        "rep": _required_unit_number(
            quality,
            "repetition_rate",
            "candidate.quality",
        ),
    }


def _effective_metrics(
    metrics: dict[str, float] | None,
) -> dict[str, float]:
    if metrics is not None:
        return dict(metrics)
    return {
        metric: 1.0 if metric in LOWER_IS_BETTER else 0.0
        for metric in PAPER_METRICS
    }


def _metric_wise_upper_bound(
    valid_slots: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, str]]:
    metrics_by_slot = [
        (slot, _candidate_metrics(slot["candidate"])) for slot in valid_slots
    ]
    metrics: dict[str, float] = {}
    sources: dict[str, str] = {}
    for metric in PAPER_METRICS:
        key = lambda pair, name=metric: pair[1][name]
        if metric in LOWER_IS_BETTER:
            source, values = min(metrics_by_slot, key=key)
        else:
            source, values = max(metrics_by_slot, key=key)
        metrics[metric] = values[metric]
        sources[metric] = str(source["candidate_id"])
    return metrics, sources


def _method_efficiency(
    result: dict[str, Any],
    slots: list[dict[str, Any]],
    method: str,
) -> dict[str, float]:
    run_efficiency = _required_mapping(result, "efficiency", "result")
    is_stars = method == "stars_best_of_4"
    online_key = "stars_seconds" if is_stars else "direct_generation_seconds"
    return {
        "online_seconds": _required_nonnegative_number(
            run_efficiency,
            online_key,
            "result.efficiency",
        ),
        "generation_requests": float(
            sum(int(slot["transport_request_count"]) for slot in slots)
        ),
        "semantic_attempts": float(
            sum(int(slot["request_count"]) for slot in slots)
        ),
        "generation_seconds": float(
            sum(float(slot["request_seconds"]) for slot in slots)
        ),
        "reward_encoding_seconds": float(
            _required_nonnegative_number(
                run_efficiency,
                "reward_encoding_seconds",
                "result.efficiency",
            )
            if is_stars
            else 0.0
        ),
        "reward_scoring_seconds": float(
            _required_nonnegative_number(
                run_efficiency,
                "reward_scoring_seconds",
                "result.efficiency",
            )
            if is_stars
            else 0.0
        ),
        **{
            key: float(sum(slot["usage"][key] for slot in slots))
            for key in USAGE_KEYS
        },
    }


def _summarize_method(
    conditional_rows: list[dict[str, float] | None],
    effective_rows: list[dict[str, float]],
    successes: list[float],
    exact_completions: list[float],
    valid_counts: list[int],
    efficiency_rows: list[dict[str, float]],
    clusters: list[str],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    successful_indices = [
        index for index, row in enumerate(conditional_rows) if row is not None
    ]
    conditional_means: dict[str, float | None] = {}
    conditional_ci95: dict[str, list[float] | None] = {}
    effective_means: dict[str, float] = {}
    effective_ci95: dict[str, list[float]] = {}
    for metric in PAPER_METRICS:
        conditional_values = [
            float(conditional_rows[index][metric])
            for index in successful_indices
        ]
        conditional_clusters = [clusters[index] for index in successful_indices]
        conditional_means[metric] = (
            round(float(np.mean(conditional_values)), 6)
            if conditional_values
            else None
        )
        conditional_ci95[metric] = (
            _cluster_bootstrap_mean_ci(
                conditional_values,
                conditional_clusters,
                bootstrap_replicates,
                rng,
            )
            if conditional_values
            else None
        )
        effective_values = [float(row[metric]) for row in effective_rows]
        effective_means[metric] = round(
            float(np.mean(effective_values)),
            6,
        )
        effective_ci95[metric] = _cluster_bootstrap_mean_ci(
            effective_values,
            clusters,
            bootstrap_replicates,
            rng,
        )
    requested = len(successes)
    successful = int(sum(successes))
    return {
        "requested_sample_count": requested,
        "successful_sample_count": successful,
        "failed_sample_count": requested - successful,
        "selection_success_rate": round(
            successful / max(1, requested),
            6,
        ),
        "exact_prefix_completion_count": int(sum(exact_completions)),
        "exact_prefix_completion_rate": round(
            float(np.mean(exact_completions)),
            6,
        ),
        "mean_valid_slots_in_prefix": round(float(np.mean(valid_counts)), 6),
        "valid_slots_in_prefix_histogram": _histogram(valid_counts),
        "conditional_means": conditional_means,
        "conditional_cluster_bootstrap_ci95": conditional_ci95,
        "effective_means": effective_means,
        "effective_cluster_bootstrap_ci95": effective_ci95,
        "efficiency_all_attempted_samples": _summarize_efficiency(
            efficiency_rows
        ),
    }


def _summarize_efficiency(
    rows: list[dict[str, float]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in EFFICIENCY_KEYS:
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": round(float(np.mean(values)), 6),
            "median": round(float(np.median(values)), 6),
            "p95": round(float(np.percentile(values, 95)), 6),
            "total": round(float(np.sum(values)), 6),
        }
    return summary


def _cluster_bootstrap_mean_ci(
    values: Iterable[float],
    clusters: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) != len(clusters):
        raise ValueError(
            "Bootstrap values and cluster identifiers must have equal length."
        )
    if len(array) == 0:
        return [0.0, 0.0]
    unique_clusters = sorted(set(clusters))
    if replicates <= 0 or len(unique_clusters) == 1:
        mean = round(float(array.mean()), 6)
        return [mean, mean]
    indices_by_cluster = {
        cluster: np.asarray(
            [
                index
                for index, observed in enumerate(clusters)
                if observed == cluster
            ],
            dtype=np.int64,
        )
        for cluster in unique_clusters
    }
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        indices = np.concatenate(
            [indices_by_cluster[str(cluster)] for cluster in sampled]
        )
        draws[replicate] = float(array[indices].mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def _reward(candidate: dict[str, Any]) -> float:
    reward = _required_mapping(candidate, "reward", "candidate")
    return _required_unit_number(reward, "total", "candidate.reward")


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{context}.{key} must be an object.")
    return value


def _required_number(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> float:
    if key not in payload:
        raise RuntimeError(f"Missing required field {context}.{key}.")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context}.{key} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{context}.{key} must be finite.")
    return number


def _required_nonnegative_number(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> float:
    number = _required_number(payload, key, context)
    if number < 0.0:
        raise RuntimeError(f"{context}.{key} must be non-negative.")
    return number


def _required_unit_number(
    payload: dict[str, Any],
    key: str,
    context: str,
) -> float:
    number = _required_number(payload, key, context)
    if not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{context}.{key} must be within [0, 1].")
    return number


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} must be a non-negative integer.")
    return value


def _sample_key(row: dict[str, Any]) -> str:
    value = str(row.get("sample_key") or row.get("video_id") or "").strip()
    if not value:
        raise RuntimeError("Every result row must have a sample key or video ID.")
    return value


def _usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise RuntimeError("Slot usage must be an object.")
    usage: dict[str, int] = {}
    for key in USAGE_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                "Slot usage must contain integer input, output, and total tokens."
            )
        usage[key] = value
    if any(value < 0 for value in usage.values()):
        raise RuntimeError("Token counts must be non-negative.")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise RuntimeError("Total tokens must equal input plus output tokens.")
    return usage


def _round_metrics(
    metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(int(item) for item in values).items())
    }


def _verify_summary(summary: dict[str, Any]) -> None:
    direct = summary["methods"]["direct_generation_c1"]
    stars = summary["methods"]["stars_best_of_4"]
    if (
        direct["effective_means"]["reward"]
        > stars["effective_means"]["reward"] + 1e-12
    ):
        raise AssertionError(
            "Aggregate STARS Reward must not be lower than Direct Generation Reward."
        )
    upper = summary["metric_wise_best_of_4_upper_bound"]
    if (
        upper["conditional_means"]["reward"]
        != stars["conditional_means"]["reward"]
        or upper["effective_means"]["reward"]
        != stars["effective_means"]["reward"]
    ):
        raise AssertionError(
            "Metric-wise Best@4 Reward must equal STARS Reward."
        )


def _write_main_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "requested_samples",
                "successful_samples",
                "selection_success_rate",
                "complete_requested_prefix_rate",
                *[f"conditional_{metric}" for metric in PAPER_METRICS],
                *[f"effective_{metric}" for metric in PAPER_METRICS],
            ]
        )
        for method in METHOD_PREFIXES:
            payload = summary["methods"][method]
            writer.writerow(
                [
                    method,
                    payload["requested_sample_count"],
                    payload["successful_sample_count"],
                    payload["selection_success_rate"],
                    payload["exact_prefix_completion_rate"],
                    *[
                        payload["conditional_means"][metric]
                        for metric in PAPER_METRICS
                    ],
                    *[
                        payload["effective_means"][metric]
                        for metric in PAPER_METRICS
                    ],
                ]
            )
        upper = summary["metric_wise_best_of_4_upper_bound"]
        writer.writerow(
            [
                "metric_wise_best_of_4_upper_bound",
                upper["requested_sample_count"],
                upper["successful_sample_count"],
                upper["selection_success_rate"],
                upper["exact_prefix_completion_rate"],
                *[
                    upper["conditional_means"][metric]
                    for metric in PAPER_METRICS
                ],
                *[
                    upper["effective_means"][metric]
                    for metric in PAPER_METRICS
                ],
            ]
        )


def _write_efficiency_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                *[f"mean_{key}" for key in EFFICIENCY_KEYS],
            ]
        )
        for method in METHOD_PREFIXES:
            efficiency = summary["methods"][method][
                "efficiency_all_attempted_samples"
            ]
            writer.writerow(
                [method, *[efficiency[key]["mean"] for key in EFFICIENCY_KEYS]]
            )


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# STARS failure-aware candidate-pool analysis",
        "",
        (
            f"Requested samples: **{summary['requested_sample_count']}**; "
            f"valid slots: **{summary['valid_candidate_slot_count']}**; "
            f"failed slots: **{summary['failed_candidate_slot_count']}**."
        ),
        "",
        (
            "Metric-wise Best@4 Upper Bound selects the best valid C1-C4 "
            "candidate independently for each metric; its values may come "
            "from different candidates and do not represent one deployable output."
        ),
        "",
        "## Effective means over all requested samples",
        "",
        (
            "Failed selections contribute 0 to higher-is-better metrics and "
            "1 to repetition."
        ),
        "",
        "| Method | Success | Complete requested prefix | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "direct_generation_c1": "Direct Generation (C1)",
        "stars_best_of_4": "STARS",
    }
    for method, label in labels.items():
        payload = summary["methods"][method]
        lines.append(
            _metric_markdown_row(
                label,
                payload,
                payload["effective_means"],
            )
        )
    upper = summary["metric_wise_best_of_4_upper_bound"]
    lines.append(
        _metric_markdown_row(
            "Metric-wise Best@4 Upper Bound",
            upper,
            upper["effective_means"],
        )
    )
    lines.extend(
        [
            "",
            "## Conditional means over successful selections only",
            "",
            "| Method | Success | Complete requested prefix | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, label in labels.items():
        payload = summary["methods"][method]
        lines.append(
            _metric_markdown_row(
                label,
                payload,
                payload["conditional_means"],
            )
        )
    lines.append(
        _metric_markdown_row(
            "Metric-wise Best@4 Upper Bound",
            upper,
            upper["conditional_means"],
        )
    )
    lines.extend(
        [
            "",
            "## Mean efficiency over all requested samples",
            "",
            "| Method | Accounted online latency (s) | Requests | Input tokens | Output tokens | Total tokens | Reward encoding (s) | Reward scoring (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, label in labels.items():
        efficiency = summary["methods"][method][
            "efficiency_all_attempted_samples"
        ]
        lines.append(
            f"| {label} | {_efficiency_mean(efficiency, 'online_seconds')} | "
            f"{_efficiency_mean(efficiency, 'generation_requests')} | "
            f"{_efficiency_mean(efficiency, 'input_tokens')} | "
            f"{_efficiency_mean(efficiency, 'output_tokens')} | "
            f"{_efficiency_mean(efficiency, 'total_tokens')} | "
            f"{_efficiency_mean(efficiency, 'reward_encoding_seconds')} | "
            f"{_efficiency_mean(efficiency, 'reward_scoring_seconds')} |"
        )
    return "\n".join(lines) + "\n"


def _metric_markdown_row(
    label: str,
    payload: dict[str, Any],
    metrics: dict[str, float | None],
) -> str:
    return (
        f"| {label} | {payload['selection_success_rate']:.4f} | "
        f"{payload['exact_prefix_completion_rate']:.4f} | "
        f"{_format_optional(metrics['reward'])} | "
        f"{_format_optional(metrics['mv_align'])} | "
        f"{_format_optional(metrics['csr'])} | "
        f"{_format_optional(metrics['semantic_point_coverage'])} | "
        f"{_format_optional(metrics['cider_lite'])} | "
        f"{_format_optional(metrics['rep'])} |"
    )


def _efficiency_mean(payload: dict[str, Any], key: str) -> str:
    value = _required_mapping(payload, key, "efficiency").get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"efficiency.{key}.mean must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"efficiency.{key}.mean must be finite.")
    return f"{number:.4f}"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
