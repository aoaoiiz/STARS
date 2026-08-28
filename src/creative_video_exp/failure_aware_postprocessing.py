from __future__ import annotations

import csv
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .utils import write_json, write_jsonl


PAPER_METRICS = [
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
]
HIGHER_IS_BETTER = {
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
}
LOWER_IS_BETTER = {"rep"}
PREFIX_METHODS = {
    "direct_generation_c1": 1,
    "best_of_1": 1,
    "best_of_2": 2,
    "stars_best_of_4": 4,
}
USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")
EFFICIENCY_KEYS = (
    "request_count",
    "generation_requests",
    "transport_request_count",
    "semantic_attempt_count",
    "request_seconds",
    *USAGE_KEYS,
)
REWARD_COMPONENT_WEIGHTS = {
    "alignment": "alignment_weight",
    "readability": "readability_weight",
    "rhythm": "rhythm_weight",
    "control": "control_weight",
    "risk": "risk_weight",
}


def analyze_failure_aware_candidate_pool(
    results: list[dict[str, Any]],
    reward_config: dict[str, float],
    pool_size: int = 4,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:


    if not results:
        raise ValueError("No requested-sample rows were supplied.")
    if pool_size != 4:
        raise ValueError("STARS failure-aware analysis requires four requested slots.")
    weights = _reward_weights(reward_config)
    active_components = [name for name, weight in weights.items() if weight > 0.0]
    counterfactual_specs = _counterfactual_method_specs(weights)
    method_groups = {
        **{method: "nested_pool" for method in PREFIX_METHODS},
        **{
            method: specification[1]
            for method, specification in counterfactual_specs.items()
        },
    }
    method_names = list(method_groups)

    fingerprints = {str(row.get("protocol_fingerprint", "")) for row in results}
    if "" in fingerprints or len(fingerprints) != 1:
        raise RuntimeError(
            "All requested-sample rows must share one non-empty protocol_fingerprint."
        )
    sample_keys = [_sample_key(row) for row in results]
    if len(set(sample_keys)) != len(sample_keys):
        raise RuntimeError("Requested-sample rows must have unique sample keys.")

    rng = np.random.default_rng(seed)
    cluster_ids: list[str] = []
    selections: list[dict[str, Any]] = []
    method_conditional: dict[str, list[dict[str, float] | None]] = {
        method: [] for method in method_names
    }
    method_effective: dict[str, list[dict[str, float]]] = {
        method: [] for method in method_names
    }
    method_success: dict[str, list[float]] = {
        method: [] for method in method_names
    }
    method_exact_completion: dict[str, list[float]] = {
        method: [] for method in method_names
    }
    method_valid_counts: dict[str, list[int]] = {
        method: [] for method in method_names
    }
    method_efficiency: dict[str, list[dict[str, float]]] = {
        method: [] for method in method_names
    }
    method_candidate_ids: dict[str, list[str | None]] = {
        method: [] for method in method_names
    }
    upper_conditional: list[dict[str, float] | None] = []
    upper_effective: list[dict[str, float]] = []
    upper_success: list[float] = []
    upper_pool_sizes: list[int] = []
    all_pool_sizes: list[int] = []
    all_component_values: dict[str, list[float]] = {
        component: [] for component in active_components
    }
    reward_vs_metrics: dict[str, tuple[list[float], list[float]]] = {
        metric: ([], [])
        for metric in [
            "mv_align",
            "csr",
            "semantic_point_coverage",
            "cider_lite",
            "rep",
        ]
    }

    for result in results:
        slots = _ordered_slot_outcomes(result, pool_size)
        video_id = str(result.get("video_id", _sample_key(result)))
        cluster_ids.append(video_id)
        all_pool_sizes.append(sum(slot["terminal_status"] == "valid" for slot in slots))

        requested_slots_view = [
            {
                "candidate_index": slot["candidate_index"],
                "terminal_status": slot["terminal_status"],
                "candidate_id": slot["candidate_id"],
                "terminal_reason": slot["terminal_reason"],
                "request_count": slot["request_count"],
                "generation_requests": slot["transport_request_count"],
                "transport_request_count": slot["transport_request_count"],
                "semantic_attempt_count": slot["request_count"],
                "request_seconds": round(slot["request_seconds"], 6),
                "usage": dict(slot["usage"]),
            }
            for slot in slots
        ]
        selection_row: dict[str, Any] = {
            "sample_key": _sample_key(result),
            "video_id": video_id,
            "protocol_fingerprint": result["protocol_fingerprint"],
            "requested_slots": requested_slots_view,
            "methods": {},
        }

        computed_by_prefix: dict[int, dict[str, Any]] = {}
        for prefix_size in sorted(set(PREFIX_METHODS.values())):
            prefix_slots = slots[:prefix_size]
            valid_slots = [
                slot for slot in prefix_slots if slot["terminal_status"] == "valid"
            ]
            selected_slot = (
                max(valid_slots, key=lambda slot: _full_reward(slot["candidate"]))
                if valid_slots
                else None
            )
            conditional = (
                _candidate_metrics(selected_slot["candidate"])
                if selected_slot is not None
                else None
            )
            effective = _effective_metrics(conditional)
            computed_by_prefix[prefix_size] = {
                "selection_status": "success" if selected_slot else "failure",
                "requested_prefix_size": prefix_size,
                "valid_slot_count": len(valid_slots),
                "valid_slot_indices": [slot["candidate_index"] for slot in valid_slots],
                "exact_prefix_complete": len(valid_slots) == prefix_size,
                "candidate_id": (
                    str(selected_slot["candidate_id"]) if selected_slot else None
                ),
                "candidate_index": (
                    int(selected_slot["candidate_index"]) if selected_slot else None
                ),
                "selection_score": (
                    _full_reward(selected_slot["candidate"])
                    if selected_slot
                    else None
                ),
                "conditional_metrics": conditional,
                "effective_metrics": effective,
                "efficiency": _prefix_efficiency(prefix_slots),
            }

        valid_full_slots = [
            slot for slot in slots if slot["terminal_status"] == "valid"
        ]
        outcomes_by_method = {
            method: computed_by_prefix[prefix_size]
            for method, prefix_size in PREFIX_METHODS.items()
        }
        for method, (counterfactual_weights, _) in counterfactual_specs.items():
            selected_slot = (
                max(
                    valid_full_slots,
                    key=lambda slot, active=counterfactual_weights: _variant_reward(
                        slot["candidate"], active
                    ),
                )
                if valid_full_slots
                else None
            )
            conditional = (
                _candidate_metrics(selected_slot["candidate"])
                if selected_slot is not None
                else None
            )
            outcomes_by_method[method] = {
                "selection_status": "success" if selected_slot else "failure",
                "requested_prefix_size": pool_size,
                "valid_slot_count": len(valid_full_slots),
                "valid_slot_indices": [
                    slot["candidate_index"] for slot in valid_full_slots
                ],
                "exact_prefix_complete": len(valid_full_slots) == pool_size,
                "candidate_id": (
                    str(selected_slot["candidate_id"]) if selected_slot else None
                ),
                "candidate_index": (
                    int(selected_slot["candidate_index"]) if selected_slot else None
                ),
                "selection_score": (
                    _variant_reward(selected_slot["candidate"], counterfactual_weights)
                    if selected_slot
                    else None
                ),
                "conditional_metrics": conditional,
                "effective_metrics": _effective_metrics(conditional),
                "efficiency": _prefix_efficiency(slots),
            }

        for method in method_names:
            computed = outcomes_by_method[method]
            outcome = _rounded_outcome(computed)
            selection_row["methods"][method] = outcome
            method_conditional[method].append(computed["conditional_metrics"])
            method_effective[method].append(computed["effective_metrics"])
            method_success[method].append(
                float(computed["selection_status"] == "success")
            )
            method_exact_completion[method].append(
                float(computed["exact_prefix_complete"])
            )
            method_valid_counts[method].append(int(computed["valid_slot_count"]))
            method_efficiency[method].append(computed["efficiency"])
            method_candidate_ids[method].append(computed["candidate_id"])

        direct_view = selection_row["methods"]["direct_generation_c1"]
        best_one_view = selection_row["methods"]["best_of_1"]
        if direct_view != best_one_view:
            raise AssertionError(
                "Direct Generation and Best-of-1 must be identical, including failure state."
            )

        nested_effective_rewards = [
            computed_by_prefix[size]["effective_metrics"]["reward"]
            for size in (1, 2, 4)
        ]
        if any(
            left > right + 1e-12
            for left, right in zip(
                nested_effective_rewards, nested_effective_rewards[1:]
            )
        ):
            raise AssertionError(
                "Failure-aware Best-of-1/2/4 effective Reward must be monotonic."
            )

        stars_slot = (
            max(valid_full_slots, key=lambda slot: _full_reward(slot["candidate"]))
            if valid_full_slots
            else None
        )
        if stars_slot is None:
            upper_metrics = None
            upper_sources = {metric: None for metric in PAPER_METRICS}
            upper_reward_component_maxima = {
                component: None for component in weights
            }
            upper_reward_component_sources = {
                component: None for component in weights
            }
        else:
            (
                upper_metrics,
                upper_sources,
                upper_reward_component_maxima,
                upper_reward_component_sources,
            ) = _metric_wise_upper_bound(
                valid_full_slots, weights
            )
        upper_effective_metrics = _effective_metrics(upper_metrics)
        upper_pool_size = len(valid_full_slots)
        upper_pool_sizes.append(upper_pool_size)
        upper_conditional.append(upper_metrics)
        upper_effective.append(upper_effective_metrics)
        upper_success.append(float(upper_pool_size > 0))
        selection_row["metric_wise_best_of_4_upper_bound"] = {
            "selection_status": "success" if upper_pool_size > 0 else "failure",
            "effective_pool_size": upper_pool_size,
            "metrics": _round_metrics(upper_metrics),
            "effective_metrics": _round_metrics(upper_effective_metrics),
            "candidate_source_by_metric": upper_sources,
            "reward_component_maxima": {
                component: (
                    None if value is None else round(float(value), 6)
                )
                for component, value in upper_reward_component_maxima.items()
            },
            "candidate_source_by_reward_component": (
                upper_reward_component_sources
            ),
            "single_realizable_candidate": False,
            "deployable": False,
        }
        selections.append(selection_row)

        for slot in valid_full_slots:
            candidate = slot["candidate"]
            candidate_metrics = _candidate_metrics(candidate)
            for component in active_components:
                all_component_values[component].append(
                    float(candidate.get("reward", {}).get(component, 0.0))
                )
            for metric, (reward_values, metric_values) in reward_vs_metrics.items():
                reward_values.append(candidate_metrics["reward"])
                metric_values.append(candidate_metrics[metric])

    methods_summary = {
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
        for method in method_names
    }


    methods_summary["direct_generation_c1"] = deepcopy(
        methods_summary["best_of_1"]
    )
    upper_summary = _summarize_method(
        conditional_rows=upper_conditional,
        effective_rows=upper_effective,
        successes=upper_success,
        exact_completions=[float(size == pool_size) for size in upper_pool_sizes],
        valid_counts=upper_pool_sizes,
        efficiency_rows=method_efficiency["stars_best_of_4"],
        clusters=cluster_ids,
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
    )
    upper_summary["mean_effective_pool_size"] = round(
        float(np.mean(upper_pool_sizes)), 6
    )
    upper_summary["effective_pool_size_histogram"] = _histogram(upper_pool_sizes)
    metric_agreement, metric_regret = _metric_wise_diagnostics(
        stars_conditional=method_conditional["stars_best_of_4"],
        stars_effective=method_effective["stars_best_of_4"],
        upper_conditional=upper_conditional,
        upper_effective=upper_effective,
        clusters=cluster_ids,
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
    )
    stars_ids = method_candidate_ids["stars_best_of_4"]
    switch_audit: dict[str, dict[str, float | int | None]] = {}
    switch_rates: dict[str, float | None] = {}
    for method, ids in method_candidate_ids.items():
        if method_groups[method] == "nested_pool":
            continue
        comparable = [
            candidate_id != stars_id
            for candidate_id, stars_id in zip(ids, stars_ids)
            if candidate_id is not None and stars_id is not None
        ]
        conditional_rate = (
            round(float(np.mean(comparable)), 6) if comparable else None
        )
        all_requested_rate = round(
            sum(
                candidate_id is not None
                and stars_id is not None
                and candidate_id != stars_id
                for candidate_id, stars_id in zip(ids, stars_ids)
            )
            / len(results),
            6,
        )
        switch_rates[method] = conditional_rate
        switch_audit[method] = {
            "conditional_rate": conditional_rate,
            "conditional_comparable_sample_count": len(comparable),
            "all_requested_rate_with_failures_as_no_switch": all_requested_rate,
            "requested_sample_count": len(results),
        }

    component_diagnostics: dict[str, Any] = {}
    for component, values in all_component_values.items():
        if not values:
            component_diagnostics[component] = {
                "valid_candidate_count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "zero_rate": None,
                "saturation_at_one_rate": None,
                "leave_one_out_selection_switch_rate": switch_rates.get(
                    f"leave_one_out__{component}"
                ),
                "single_component_selection_switch_rate": switch_rates.get(
                    f"single_component__{component}"
                ),
            }
            continue
        array = np.asarray(values, dtype=np.float64)
        component_diagnostics[component] = {
            "valid_candidate_count": len(values),
            "mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6),
            "min": round(float(array.min()), 6),
            "max": round(float(array.max()), 6),
            "zero_rate": round(float(np.mean(array <= 1e-12)), 6),
            "saturation_at_one_rate": round(
                float(np.mean(array >= 1.0 - 1e-12)), 6
            ),
            "leave_one_out_selection_switch_rate": switch_rates.get(
                f"leave_one_out__{component}"
            ),
            "single_component_selection_switch_rate": switch_rates.get(
                f"single_component__{component}"
            ),
        }

    correlations: dict[str, dict[str, float | int]] = {}
    for metric, (reward_values, metric_values) in reward_vs_metrics.items():
        correlations[f"reward_vs_{metric}_spearman"] = {
            "value": round(_spearman(reward_values, metric_values), 6),
            "valid_candidate_count": len(reward_values),
        }
        if metric == "rep":
            correlations["reward_vs_negative_rep_spearman"] = {
                "value": round(
                    _spearman(reward_values, [-value for value in metric_values]),
                    6,
                ),
                "valid_candidate_count": len(reward_values),
            }

    requested_sample_count = len(results)
    valid_slot_count = sum(all_pool_sizes)
    requested_slot_count = requested_sample_count * pool_size
    full_pool_count = sum(size == pool_size for size in all_pool_sizes)
    any_valid_count = sum(size > 0 for size in all_pool_sizes)
    summary = {
        "experiment_version": "starsv20",
        "analysis_protocol": "failure_aware_fixed_requested_slots_v1",
        "protocol_fingerprint": next(iter(fingerprints)),
        "requested_sample_count": requested_sample_count,
        "requested_slots_per_sample": pool_size,
        "requested_candidate_slot_count": requested_slot_count,
        "valid_candidate_slot_count": valid_slot_count,
        "failed_candidate_slot_count": requested_slot_count - valid_slot_count,
        "candidate_slot_success_rate": round(
            valid_slot_count / requested_slot_count, 6
        ),
        "full_candidate_pool_sample_count": full_pool_count,
        "full_candidate_pool_sample_rate": round(
            full_pool_count / requested_sample_count, 6
        ),
        "any_valid_candidate_sample_count": any_valid_count,
        "any_valid_candidate_sample_rate": round(
            any_valid_count / requested_sample_count, 6
        ),

        "requested_slot_count": requested_slot_count,
        "valid_slot_count": valid_slot_count,
        "failed_slot_count": requested_slot_count - valid_slot_count,
        "full_pool_sample_count": full_pool_count,
        "full_pool_rate": round(full_pool_count / requested_sample_count, 6),
        "any_valid_sample_count": any_valid_count,
        "any_valid_sample_rate": round(any_valid_count / requested_sample_count, 6),
        "valid_pool_size_histogram": _histogram(all_pool_sizes),
        "candidate_pool_definition": {
            "unit_of_N": "requested candidate slots, not guaranteed-valid candidates",
            "ordering": "fixed candidate_index 1 through 4",
            "best_of_N": "argmax full Reward over valid candidates among requested slots 1..N",
            "no_valid_candidate": "explicit method failure",
            "failed_slot_selection_score": "negative infinity; failed slots are never selected",
            "conditional_estimand": "quality among samples with a valid selected output",
            "effective_estimand": (
                "all fixed-manifest samples; failed method outcomes receive 0 for "
                "higher-is-better metrics and 1 for Rep"
            ),
        },
        "metric_scale": {metric: "0-1" for metric in PAPER_METRICS},
        "failure_penalties": {
            **{metric: 0.0 for metric in sorted(HIGHER_IS_BETTER)},
            "rep": 1.0,
        },
        "reward_weights": {
            "configured": {
                key: round(value, 8) for key, value in weights.items()
            },
            "normalized_over_active_components": {
                key: round(value, 8)
                for key, value in _normalized_weights(weights).items()
            },
        },
        "baseline_best_of_1_invariant": {
            "definition": (
                "both inspect only requested slot C1 and share its success/failure state"
            ),
            "identical_candidate_ids": True,
            "identical_failure_states": True,
            "identical_conditional_metrics": True,
            "identical_effective_metrics": True,
        },
        "nested_reward_monotonicity": {
            "expected": (
                "Effective Reward(Best-of-1) <= Effective Reward(Best-of-2) "
                "<= Effective Reward(Best-of-4)"
            ),
            "verified": True,
        },
        "methods": methods_summary,
        "method_groups": method_groups,
        "counterfactual_reward_definitions": {
            method: {
                "group": group,
                "selection_weights": {
                    key: round(value, 8)
                    for key, value in counterfactual_weights.items()
                },
                "selection_score_normalization": (
                    "weighted sum divided by the sum of positive counterfactual weights"
                ),
                "reported_reward": "original frozen full Reward of the selected candidate",
                "valid_slots_only": True,
                "zero_valid_outcome": "explicit method failure",
            }
            for method, (counterfactual_weights, group) in counterfactual_specs.items()
        },
        "metric_wise_best_of_4_upper_bound_definition": {
            "name": "Metric-wise Best@4-attempts (non-deployable upper bound)",
            "valid_slots_only": True,
            "reports_effective_pool_size": True,
            "single_realizable_candidate": False,
            "reward_column": (
                "normalized original-weight combination of the valid pool's "
                "per-component Reward maxima"
            ),
            "reward_component_source_audit": (
                "candidate_source_by_reward_component"
            ),
        },
        "metric_wise_best_of_4_upper_bound": upper_summary,
        "metric_wise_selection_agreement_rate": metric_agreement,
        "metric_wise_regret": metric_regret,
        "counterfactual_selection_switch_rate": switch_rates,
        "counterfactual_selection_switch_audit": switch_audit,
        "component_diagnostics": component_diagnostics,
        "full_reward_external_metric_correlations": correlations,
        "uncertainty": {
            "method": "cluster bootstrap by video_id",
            "replicates": bootstrap_replicates,
            "seed": seed,
        },
    }
    _verify_summary_invariants(summary)
    return summary, selections


def write_failure_aware_candidate_pool_analysis(
    output_dir: str | Path,
    summary: dict[str, Any],
    selections: list[dict[str, Any]],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "failure_aware_candidate_pool_analysis.json", summary)
    write_jsonl(
        output / "failure_aware_candidate_pool_selections.jsonl", selections
    )
    _write_main_csv(output / "failure_aware_main_summary.csv", summary)
    _write_counterfactual_csv(
        output / "failure_aware_reward_leave_one_out.csv",
        summary,
        "leave_one_out",
    )
    _write_counterfactual_csv(
        output / "failure_aware_reward_single_component.csv",
        summary,
        "single_component",
    )
    _write_counterfactual_csv(
        output / "failure_aware_reward_weight_sensitivity.csv",
        summary,
        "weight_sensitivity",
    )
    _write_component_diagnostics_csv(
        output / "failure_aware_reward_component_diagnostics.csv", summary
    )
    (output / "failure_aware_candidate_pool_summary.md").write_text(
        _markdown_summary(summary), encoding="utf-8"
    )


def _ordered_slot_outcomes(
    result: dict[str, Any], pool_size: int
) -> list[dict[str, Any]]:
    raw_slots = [dict(slot) for slot in result.get("slot_outcomes", [])]
    if len(raw_slots) != pool_size:
        raise RuntimeError(
            f"Sample `{_sample_key(result)}` has {len(raw_slots)} slot outcomes; "
            f"expected exactly {pool_size}."
        )
    raw_slots.sort(key=lambda slot: int(slot.get("candidate_index", -1)))
    indices = [int(slot.get("candidate_index", -1)) for slot in raw_slots]
    if indices != list(range(1, pool_size + 1)):
        raise RuntimeError(
            f"Sample `{_sample_key(result)}` slot indices must be exactly 1..{pool_size}; "
            f"got {indices}."
        )

    candidates = [dict(candidate) for candidate in result.get("candidates", [])]
    candidates_by_index: dict[int, dict[str, Any]] = {}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        index = int(candidate.get("candidate_index", -1))
        candidate_id = str(candidate.get("candidate_id", ""))
        if index in candidates_by_index or not candidate_id or candidate_id in candidates_by_id:
            raise RuntimeError(
                f"Sample `{_sample_key(result)}` has duplicate or empty candidate identity."
            )
        candidates_by_index[index] = candidate
        candidates_by_id[candidate_id] = candidate

    normalized: list[dict[str, Any]] = []
    joined_candidate_ids: set[str] = set()
    for slot in raw_slots:
        index = int(slot["candidate_index"])
        status = str(slot.get("terminal_status", ""))
        if status not in {"valid", "failed"}:
            raise RuntimeError(
                f"Sample `{_sample_key(result)}` slot {index} has invalid terminal_status `{status}`."
            )
        terminal_reason = str(slot.get("terminal_reason", "")).strip()
        if not terminal_reason:
            raise RuntimeError(
                f"Sample `{_sample_key(result)}` slot {index} is missing terminal_reason."
            )
        attempts = list(slot.get("attempts", []))
        request_count = int(slot.get("request_count", -1))
        if request_count < 1 or request_count != len(attempts):
            raise RuntimeError(
                f"Sample `{_sample_key(result)}` slot {index} request_count does not "
                "match its non-empty attempt trace."
            )
        transport_request_count = int(
            slot.get("transport_request_count", request_count)
        )
        if transport_request_count < 0:
            raise RuntimeError("Slot transport_request_count must be non-negative.")
        if transport_request_count < request_count:
            raise RuntimeError(
                "Slot transport_request_count cannot be smaller than its semantic "
                "attempt count."
            )
        request_seconds = float(slot.get("request_seconds", -1.0))
        if request_seconds < 0.0:
            raise RuntimeError("Slot request_seconds must be non-negative.")
        usage = {
            key: int(slot.get("usage", {}).get(key, 0)) for key in USAGE_KEYS
        }
        if any(value < 0 for value in usage.values()):
            raise RuntimeError("Slot token usage must be non-negative.")
        if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
            raise RuntimeError("Slot total_tokens must equal input_tokens + output_tokens.")

        raw_candidate_id = slot.get("candidate_id")
        candidate_id = None if raw_candidate_id is None else str(raw_candidate_id)
        candidate: dict[str, Any] | None = None
        if status == "valid":
            if not candidate_id:
                raise RuntimeError("A valid slot must identify one scored candidate.")
            candidate = candidates_by_index.get(index)
            if candidate is None or str(candidate.get("candidate_id")) != candidate_id:
                raise RuntimeError(
                    f"Sample `{_sample_key(result)}` valid slot {index} cannot be joined "
                    "to exactly one scored candidate by index and ID."
                )
            _validate_scored_candidate(candidate)
            joined_candidate_ids.add(candidate_id)
        elif candidate_id is not None or index in candidates_by_index:
            raise RuntimeError(
                f"Sample `{_sample_key(result)}` failed slot {index} must not contain "
                "or reference a synthetic candidate."
            )

        normalized.append(
            {
                "candidate_index": index,
                "terminal_status": status,
                "candidate_id": candidate_id,
                "terminal_reason": terminal_reason,
                "request_count": request_count,
                "transport_request_count": transport_request_count,
                "request_seconds": request_seconds,
                "usage": usage,
                "attempts": attempts,
                "candidate": candidate,
            }
        )

    if joined_candidate_ids != set(candidates_by_id):
        raise RuntimeError(
            f"Sample `{_sample_key(result)}` has scored candidates not represented by valid slots."
        )
    valid_count = sum(slot["terminal_status"] == "valid" for slot in normalized)
    declared_requested = int(result.get("requested_candidate_slots", -1))
    declared_valid = int(result.get("valid_candidate_slots", -1))
    declared_failed = int(result.get("failed_candidate_slots", -1))
    if declared_requested != pool_size:
        raise RuntimeError("requested_candidate_slots must equal four.")
    if declared_valid != valid_count or declared_failed != pool_size - valid_count:
        raise RuntimeError(
            "Row-level valid_candidate_slots/failed_candidate_slots do not match slot outcomes."
        )
    return normalized


def _validate_scored_candidate(candidate: dict[str, Any]) -> None:
    metrics = _candidate_metrics(candidate)
    for metric, value in metrics.items():
        if not np.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
            raise RuntimeError(
                f"Candidate metric `{metric}` must be finite and lie on the 0-1 scale."
            )
    provenance = candidate.get("generation_provenance", {})
    if provenance and (
        provenance.get("parse_status") != "valid_json_candidate"
        or provenance.get("semantic_repair_applied") is not False
        or provenance.get("wrapped_free_text") is not False
        or provenance.get("surrounding_free_text") is not False
        or (
            "timeline_normalization" in provenance
            and (
                provenance.get("timeline_normalization", {}).get("policy")
                != "positive_duration_proportional_v1"
                or provenance.get("timeline_normalization", {}).get("status")
                not in {"applied", "unchanged", "skipped"}
            )
        )
    ):
        raise RuntimeError(
            f"Candidate `{candidate.get('candidate_id')}` violates the structured-candidate gate."
        )


def _candidate_metrics(candidate: dict[str, Any]) -> dict[str, float]:
    reward = candidate.get("reward", {})
    quality = candidate.get("quality", {})
    semantic_point_coverage = quality.get("semantic_point_coverage")
    if semantic_point_coverage is None:
        raise RuntimeError(
            "A valid formal candidate is missing its evaluation-only SPC score."
        )
    control_violations = reward.get(
        "control_violations", reward.get("violations", [])
    )
    return {
        "reward": float(reward.get("total", 0.0)),
        "mv_align": float(reward.get("alignment", 0.0)),
        "csr": 1.0 if not control_violations else 0.0,
        "semantic_point_coverage": float(semantic_point_coverage),
        "cider_lite": float(quality.get("cider_lite", 0.0)),
        "rep": float(quality.get("repetition_rate", 0.0)),
    }


def _effective_metrics(
    conditional: dict[str, float] | None,
) -> dict[str, float]:
    if conditional is not None:
        return dict(conditional)
    return {
        **{metric: 0.0 for metric in HIGHER_IS_BETTER},
        "rep": 1.0,
    }


def _metric_wise_upper_bound(
    valid_slots: list[dict[str, Any]],
    weights: dict[str, float],
) -> tuple[
    dict[str, float],
    dict[str, str | None],
    dict[str, float],
    dict[str, str],
]:
    candidate_metrics = [
        _candidate_metrics(slot["candidate"]) for slot in valid_slots
    ]
    reward_component_maxima: dict[str, float] = {}
    reward_component_sources: dict[str, str] = {}
    for component in weights:
        index = int(
            np.argmax(
                [
                    float(slot["candidate"].get("reward", {}).get(component, 0.0))
                    for slot in valid_slots
                ]
            )
        )
        reward_component_maxima[component] = float(
            valid_slots[index]["candidate"]
            .get("reward", {})
            .get(component, 0.0)
        )
        reward_component_sources[component] = str(
            valid_slots[index]["candidate_id"]
        )

    denominator = sum(weight for weight in weights.values() if weight > 0.0)
    reward_upper_bound = sum(
        weight * reward_component_maxima[component]
        for component, weight in weights.items()
        if weight > 0.0
    ) / denominator
    metrics = {"reward": reward_upper_bound}
    sources: dict[str, str | None] = {"reward": None}
    for metric in sorted(HIGHER_IS_BETTER - {"reward"}):
        index = int(np.argmax([row[metric] for row in candidate_metrics]))
        metrics[metric] = candidate_metrics[index][metric]
        sources[metric] = str(valid_slots[index]["candidate_id"])
    index = int(np.argmin([row["rep"] for row in candidate_metrics]))
    metrics["rep"] = candidate_metrics[index]["rep"]
    sources["rep"] = str(valid_slots[index]["candidate_id"])
    return (
        metrics,
        sources,
        reward_component_maxima,
        reward_component_sources,
    )


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
    successful_count = int(sum(successes))
    conditional_means: dict[str, float | None] = {}
    conditional_counts: dict[str, int] = {}
    conditional_ci: dict[str, list[float] | None] = {}
    effective_means: dict[str, float] = {}
    effective_ci: dict[str, list[float]] = {}
    for metric in PAPER_METRICS:
        conditional = [
            (row[metric], cluster)
            for row, cluster in zip(conditional_rows, clusters)
            if row is not None
        ]
        conditional_counts[metric] = len(conditional)
        if conditional:
            values = [value for value, _ in conditional]
            metric_clusters = [cluster for _, cluster in conditional]
            conditional_means[metric] = round(float(np.mean(values)), 6)
            conditional_ci[metric] = _cluster_bootstrap_mean_ci(
                values, metric_clusters, bootstrap_replicates, rng
            )
        else:
            conditional_means[metric] = None
            conditional_ci[metric] = None
        effective_values = [row[metric] for row in effective_rows]
        effective_means[metric] = round(float(np.mean(effective_values)), 6)
        effective_ci[metric] = _cluster_bootstrap_mean_ci(
            effective_values, clusters, bootstrap_replicates, rng
        )

    return {
        "requested_sample_count": len(successes),
        "successful_sample_count": successful_count,
        "failed_sample_count": len(successes) - successful_count,
        "selection_success_rate": round(float(np.mean(successes)), 6),
        "selection_success_rate_ci95": _cluster_bootstrap_mean_ci(
            successes, clusters, bootstrap_replicates, rng
        ),
        "exact_prefix_completion_count": int(sum(exact_completions)),
        "exact_prefix_completion_rate": round(
            float(np.mean(exact_completions)), 6
        ),
        "exact_prefix_completion_rate_ci95": _cluster_bootstrap_mean_ci(
            exact_completions, clusters, bootstrap_replicates, rng
        ),
        "mean_valid_slots_in_prefix": round(float(np.mean(valid_counts)), 6),
        "valid_slots_in_prefix_histogram": _histogram(valid_counts),
        "conditional_means": conditional_means,
        "conditional_valid_sample_counts": conditional_counts,
        "conditional_cluster_bootstrap_ci95": conditional_ci,
        "effective_means": effective_means,
        "effective_cluster_bootstrap_ci95": effective_ci,
        "efficiency_all_attempted_samples": _summarize_efficiency(efficiency_rows),
    }


def _metric_wise_diagnostics(
    stars_conditional: list[dict[str, float] | None],
    stars_effective: list[dict[str, float]],
    upper_conditional: list[dict[str, float] | None],
    upper_effective: list[dict[str, float]],
    clusters: list[str],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    agreement: dict[str, Any] = {}
    regret: dict[str, Any] = {}
    for metric in PAPER_METRICS:
        conditional_pairs = [
            (stars[metric], upper[metric], cluster)
            for stars, upper, cluster in zip(
                stars_conditional, upper_conditional, clusters
            )
            if stars is not None and upper is not None
        ]
        conditional_agreements = [
            float(abs(stars - upper) <= 1e-12)
            for stars, upper, _ in conditional_pairs
        ]
        conditional_regrets = [
            max(
                0.0,
                (stars - upper) if metric in LOWER_IS_BETTER else (upper - stars),
            )
            for stars, upper, _ in conditional_pairs
        ]
        conditional_clusters = [cluster for _, _, cluster in conditional_pairs]

        effective_agreements = [
            float(abs(stars[metric] - upper[metric]) <= 1e-12)
            for stars, upper in zip(stars_effective, upper_effective)
        ]
        effective_regrets = [
            max(
                0.0,
                (stars[metric] - upper[metric])
                if metric in LOWER_IS_BETTER
                else (upper[metric] - stars[metric]),
            )
            for stars, upper in zip(stars_effective, upper_effective)
        ]
        agreement[metric] = {
            "conditional_rate": (
                round(float(np.mean(conditional_agreements)), 6)
                if conditional_agreements
                else None
            ),
            "conditional_valid_sample_count": len(conditional_agreements),
            "effective_rate": round(float(np.mean(effective_agreements)), 6),
            "effective_requested_sample_count": len(effective_agreements),
        }
        regret[metric] = {
            "conditional_mean": (
                round(float(np.mean(conditional_regrets)), 6)
                if conditional_regrets
                else None
            ),
            "conditional_valid_sample_count": len(conditional_regrets),
            "conditional_cluster_bootstrap_ci95": (
                _cluster_bootstrap_mean_ci(
                    conditional_regrets,
                    conditional_clusters,
                    bootstrap_replicates,
                    rng,
                )
                if conditional_regrets
                else None
            ),
            "effective_mean": round(float(np.mean(effective_regrets)), 6),
            "effective_requested_sample_count": len(effective_regrets),
            "effective_cluster_bootstrap_ci95": _cluster_bootstrap_mean_ci(
                effective_regrets,
                clusters,
                bootstrap_replicates,
                rng,
            ),
        }
    return agreement, regret


def _prefix_efficiency(slots: list[dict[str, Any]]) -> dict[str, float]:
    semantic_attempt_count = float(
        sum(slot["request_count"] for slot in slots)
    )
    transport_request_count = float(
        sum(slot["transport_request_count"] for slot in slots)
    )
    return {


        "request_count": transport_request_count,
        "generation_requests": transport_request_count,
        "transport_request_count": transport_request_count,
        "semantic_attempt_count": semantic_attempt_count,
        "request_seconds": float(sum(slot["request_seconds"] for slot in slots)),
        **{
            key: float(sum(slot["usage"][key] for slot in slots))
            for key in USAGE_KEYS
        },
    }


def _summarize_efficiency(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in EFFICIENCY_KEYS:
        values = [row[key] for row in rows]
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
        raise ValueError("Bootstrap values and cluster identifiers must have equal length.")
    if len(array) == 0:
        return [0.0, 0.0]
    unique_clusters = sorted(set(clusters))
    if replicates <= 0 or len(unique_clusters) == 1:
        mean = round(float(array.mean()), 6)
        return [mean, mean]
    indices_by_cluster = {
        cluster: np.asarray(
            [index for index, value in enumerate(clusters) if value == cluster],
            dtype=np.int64,
        )
        for cluster in unique_clusters
    }
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate(
            [indices_by_cluster[str(cluster)] for cluster in sampled]
        )
        draws[replicate] = float(array[indices].mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def _full_reward(candidate: dict[str, Any]) -> float:
    return float(candidate.get("reward", {}).get("total", 0.0))


def _sample_key(row: dict[str, Any]) -> str:
    value = str(row.get("sample_key") or row.get("video_id") or "").strip()
    if not value:
        raise RuntimeError("Every requested-sample row must have a sample_key or video_id.")
    return value


def _reward_weights(reward_config: dict[str, float]) -> dict[str, float]:
    weights = {
        component: float(reward_config.get(config_key, 0.0))
        for component, config_key in REWARD_COMPONENT_WEIGHTS.items()
    }
    if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0.0:
        raise ValueError("Formal Reward weights must be non-negative and non-empty.")
    return weights


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    denominator = sum(value for value in weights.values() if value > 0.0)
    return {
        key: (value / denominator if value > 0.0 else 0.0)
        for key, value in weights.items()
    }


def _counterfactual_method_specs(
    weights: dict[str, float],
) -> dict[str, tuple[dict[str, float], str]]:
    specifications: dict[str, tuple[dict[str, float], str]] = {}
    for component, weight in weights.items():
        if weight <= 0.0:
            continue
        leave_one_out = dict(weights)
        leave_one_out[component] = 0.0
        specifications[f"leave_one_out__{component}"] = (
            leave_one_out,
            "leave_one_out",
        )

        single_component = {name: 0.0 for name in weights}
        single_component[component] = 1.0
        specifications[f"single_component__{component}"] = (
            single_component,
            "single_component",
        )

        for factor, label in ((0.8, "0_8"), (1.2, "1_2")):
            perturbed = dict(weights)
            perturbed[component] = weight * factor
            specifications[f"weight_sensitivity__{component}__x{label}"] = (
                perturbed,
                "weight_sensitivity",
            )
    return specifications


def _variant_reward(
    candidate: dict[str, Any], weights: dict[str, float]
) -> float:
    denominator = sum(weight for weight in weights.values() if weight > 0.0)
    if denominator <= 0.0:
        raise ValueError("Counterfactual Reward has no active component weights.")
    reward = candidate.get("reward", {})
    numerator = sum(
        weight * float(reward.get(component, 0.0))
        for component, weight in weights.items()
        if weight > 0.0
    )
    return numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(np.asarray(left, dtype=np.float64))
    right_ranks = _average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_ranks) <= 1e-12 or np.std(right_ranks) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _round_metrics(
    metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _rounded_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        **outcome,
        "selection_score": (
            None
            if outcome.get("selection_score") is None
            else round(float(outcome["selection_score"]), 6)
        ),
        "conditional_metrics": _round_metrics(outcome["conditional_metrics"]),
        "effective_metrics": _round_metrics(outcome["effective_metrics"]),
        "efficiency": {
            key: round(float(value), 6)
            for key, value in outcome["efficiency"].items()
        },
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(int(item) for item in values).items())
    }


def _verify_summary_invariants(summary: dict[str, Any]) -> None:
    direct = summary["methods"]["direct_generation_c1"]
    best_one = summary["methods"]["best_of_1"]
    keys = [
        "successful_sample_count",
        "failed_sample_count",
        "selection_success_rate",
        "exact_prefix_completion_count",
        "exact_prefix_completion_rate",
        "conditional_means",
        "effective_means",
        "valid_slots_in_prefix_histogram",
        "efficiency_all_attempted_samples",
    ]
    if any(direct[key] != best_one[key] for key in keys):
        raise AssertionError("Direct Generation and Best-of-1 summary invariants failed.")
    rewards = [
        summary["methods"][method]["effective_means"]["reward"]
        for method in ("best_of_1", "best_of_2", "stars_best_of_4")
    ]
    if any(left > right + 1e-12 for left, right in zip(rewards, rewards[1:])):
        raise AssertionError("Aggregate failure-aware effective Reward is not nested.")


def _write_main_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "requested_samples",
                "successful_samples",
                "selection_success_rate",
                "exact_prefix_completion_rate",
                "mean_valid_slots_in_prefix",
                *[f"conditional_{metric}" for metric in PAPER_METRICS],
                *[f"effective_{metric}" for metric in PAPER_METRICS],
            ]
        )
        for method in PREFIX_METHODS:
            payload = summary["methods"][method]
            writer.writerow(
                [
                    method,
                    payload["requested_sample_count"],
                    payload["successful_sample_count"],
                    payload["selection_success_rate"],
                    payload["exact_prefix_completion_rate"],
                    payload["mean_valid_slots_in_prefix"],
                    *[payload["conditional_means"][metric] for metric in PAPER_METRICS],
                    *[payload["effective_means"][metric] for metric in PAPER_METRICS],
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
                upper["mean_valid_slots_in_prefix"],
                *[upper["conditional_means"][metric] for metric in PAPER_METRICS],
                *[upper["effective_means"][metric] for metric in PAPER_METRICS],
            ]
        )


def _write_counterfactual_csv(
    path: Path, summary: dict[str, Any], group: str
) -> None:
    methods = [
        method
        for method, method_group in summary["method_groups"].items()
        if method_group == group
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "requested_samples",
                "successful_samples",
                "selection_success_rate",
                "full_pool_rate",
                "conditional_selection_switch_rate",
                *[f"conditional_{metric}" for metric in PAPER_METRICS],
                *[f"effective_{metric}" for metric in PAPER_METRICS],
            ]
        )
        for method in sorted(methods):
            payload = summary["methods"][method]
            switch = summary["counterfactual_selection_switch_audit"][method]
            writer.writerow(
                [
                    method,
                    payload["requested_sample_count"],
                    payload["successful_sample_count"],
                    payload["selection_success_rate"],
                    payload["exact_prefix_completion_rate"],
                    switch["conditional_rate"],
                    *[
                        payload["conditional_means"][metric]
                        for metric in PAPER_METRICS
                    ],
                    *[payload["effective_means"][metric] for metric in PAPER_METRICS],
                ]
            )


def _write_component_diagnostics_csv(
    path: Path, summary: dict[str, Any]
) -> None:
    diagnostics = summary["component_diagnostics"]
    keys = sorted(next(iter(diagnostics.values()))) if diagnostics else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", *keys])
        for component, payload in diagnostics.items():
            writer.writerow([component, *[payload.get(key) for key in keys]])


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# STARS failure-aware candidate-pool analysis",
        "",
        (
            f"Requested samples: **{summary['requested_sample_count']}**; requested slots: "
            f"**{summary['requested_candidate_slot_count']}**; valid slots: "
            f"**{summary['valid_candidate_slot_count']}**; failed slots: "
            f"**{summary['failed_candidate_slot_count']}**."
        ),
        "",
        "Conditional means describe successful method outcomes. Effective means retain every fixed-manifest sample; failures receive 0 for higher-is-better metrics and 1 for Rep.",
        "",
        "| Method | Success | Full prefix | Conditional Reward | Effective Reward | Conditional Rep | Effective Rep |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "direct_generation_c1": "Direct Generation (C1)",
        "best_of_1": "Best-of-1 (= Direct Generation)",
        "best_of_2": "Best-of-2 attempts",
        "stars_best_of_4": "STARS (Best-of-4 attempts)",
    }
    for method, label in labels.items():
        payload = summary["methods"][method]
        conditional = payload["conditional_means"]
        effective = payload["effective_means"]
        conditional_reward = conditional["reward"]
        conditional_rep = conditional["rep"]
        lines.append(
            f"| {label} | {payload['selection_success_rate']:.4f} | "
            f"{payload['exact_prefix_completion_rate']:.4f} | "
            f"{_format_optional(conditional_reward)} | {effective['reward']:.4f} | "
            f"{_format_optional(conditional_rep)} | {effective['rep']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- N counts requested candidate slots under the frozen retry budget, not guaranteed-valid outputs.",
            "- Failed slots are never scored or selected and are not replaced by synthetic candidates.",
            "- Direct Generation and Best-of-1 share both output and failure state.",
            "- Latency, requests, and token estimates include rejected and exhausted attempts.",
            "- Rep=1 on a failed method outcome is an explicit worst-case effective penalty, not a claim that nonexistent text literally repeats.",
            "- Metric-wise Best@4 uses only valid slots, reports effective pool size, and is non-deployable.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
