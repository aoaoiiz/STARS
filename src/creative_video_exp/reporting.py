from __future__ import annotations

from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "requested_num_samples",
    "completed_num_samples",
    "requested_candidate_slot_count",
    "valid_candidate_slot_count",
    "failed_candidate_slot_count",
    "candidate_slot_success_rate",
    "full_candidate_pool_sample_count",
    "full_candidate_pool_sample_rate",
    "unresolved_generation_failure_count",
    "evidence_eligible",
    "num_samples",
    "num_candidates",
    "selected_output_success_count",
    "selected_output_success_rate",
    "real_video_rate",
    "multimodal_generation_rate",
)
PAPER_METRICS = (
    "reward",
    "mv_align",
    "csr",
    "semantic_point_coverage",
    "cider_lite",
    "rep",
)
METHOD_LABELS = {
    "direct_generation_c1": "Direct Generation (C1)",
    "stars_best_of_4": "STARS",
}


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    print("\n=== STARS Summary ===")
    for key in SUMMARY_KEYS:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    print("\nEffective means over all requested samples:")
    for label, success, values in _method_rows(metrics, "effective_means"):
        print(f"{label}: success={success}, metrics={values}")
    print("\nConditional means over successful selections only:")
    for label, success, values in _method_rows(metrics, "conditional_means"):
        print(f"{label}: success={success}, metrics={values}")
    print("\nMean accounted online efficiency per requested sample:")
    for label, values in _efficiency_rows(metrics):
        print(f"{label}: {values}")
    if "source_kind_counts" in metrics:
        print(f"source_kind_counts: {metrics['source_kind_counts']}")
    if "generation_source_counts" in metrics:
        print(f"generation_source_counts: {metrics['generation_source_counts']}")


def write_markdown_report(path: str | Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# STARS Metrics",
        "",
        "## Run summary",
        "",
        "| Field | Value |",
        "| --- | ---: |",
    ]
    for key in SUMMARY_KEYS:
        if key in metrics:
            lines.append(f"| `{key}` | {metrics[key]} |")
    lines.extend(
        [
            "",
            (
                "Metric-wise Best@4 Upper Bound selects the best valid C1-C4 "
                "candidate independently for each metric; its values may come "
                "from different candidates and do not represent one deployable output."
            ),
            "",
            "## Effective means over all requested samples",
            "",
            "Failed selections contribute 0 to higher-is-better metrics and 1 to repetition.",
            "",
            "| Method | Success | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, success, values in _method_rows(metrics, "effective_means"):
        lines.append(_metric_row(label, success, values))
    lines.extend(
        [
            "",
            "## Conditional means over successful selections only",
            "",
            "| Method | Success | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, success, values in _method_rows(metrics, "conditional_means"):
        lines.append(_metric_row(label, success, values))
    lines.extend(
        [
            "",
            "## Mean efficiency per requested sample",
            "",
            "Latency follows the accounted online-time definition in the README.",
            "",
            "| Method | Accounted online latency (s) | Input tokens | Output tokens | Total tokens | Requests |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, values in _efficiency_rows(metrics):
        lines.append(
            f"| {label} | {_format_value(values['latency'])} | "
            f"{_format_value(values['input_tokens'])} | "
            f"{_format_value(values['output_tokens'])} | "
            f"{_format_value(values['total_tokens'])} | "
            f"{_format_value(values['requests'])} |"
        )
    lines.extend(
        [
            "",
            "| Reward stage | Mean seconds per requested sample |",
            "| --- | ---: |",
            f"| Encoding | {_format_value(metrics['mean_reward_encoding_seconds'])} |",
            f"| Candidate scoring | {_format_value(metrics['mean_reward_scoring_seconds'])} |",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_rows(
    metrics: dict[str, Any],
    means_key: str,
) -> list[tuple[str, float, dict[str, float | None]]]:
    success_rates = metrics["method_output_success_rates"]
    means = metrics[
        "method_effective_means"
        if means_key == "effective_means"
        else "method_conditional_means"
    ]
    rows = [
        (
            label,
            float(success_rates[method]),
            means[method],
        )
        for method, label in METHOD_LABELS.items()
    ]
    upper = metrics["metric_wise_best_of_4_upper_bound"]
    rows.append(
        (
            "Metric-wise Best@4 Upper Bound",
            float(upper["selection_success_rate"]),
            upper[means_key],
        )
    )
    return rows


def _efficiency_rows(
    metrics: dict[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    return [
        (
            "Direct Generation (C1)",
            {
                "latency": float(metrics["mean_direct_generation_seconds"]),
                "input_tokens": float(
                    metrics["mean_direct_generation_input_tokens"]
                ),
                "output_tokens": float(
                    metrics["mean_direct_generation_output_tokens"]
                ),
                "total_tokens": float(
                    metrics["mean_direct_generation_total_tokens"]
                ),
                "requests": float(metrics["mean_direct_generation_requests"]),
            },
        ),
        (
            "STARS",
            {
                "latency": float(metrics["mean_stars_seconds"]),
                "input_tokens": float(metrics["mean_stars_input_tokens"]),
                "output_tokens": float(metrics["mean_stars_output_tokens"]),
                "total_tokens": float(metrics["mean_stars_total_tokens"]),
                "requests": float(metrics["mean_stars_generation_requests"]),
            },
        ),
    ]


def _metric_row(
    label: str,
    success: float,
    metrics: dict[str, float | None],
) -> str:
    values = " | ".join(_format_value(metrics[key]) for key in PAPER_METRICS)
    return f"| {label} | {_format_value(success)} | {values} |"


def _format_value(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
