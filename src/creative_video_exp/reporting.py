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
    "mean_direct_generation_seconds",
    "mean_stars_seconds",
    "mean_direct_generation_total_tokens",
    "mean_stars_total_tokens",
    "mean_reward_encoding_seconds",
    "mean_reward_scoring_seconds",
    "mean_vlm_server_peak_allocated_mib",
    "mean_reward_runner_peak_allocated_mib",
)


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    print("\n=== STARS Summary ===")
    for key in SUMMARY_KEYS:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    for key, value in metrics.get("selected_metrics", {}).items():
        print(f"selected_{key}: {value}")
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
            "## Selected-output metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in metrics.get("selected_metrics", {}).items():
        lines.append(f"| `{key}` | {value} |")
    methods = metrics.get("method_conditional_means", {})
    if methods:
        lines.extend(
            [
                "",
                "## Failure-aware method results",
                "",
                "| Method | Reward | MV-Align | CSR | SPC | CIDEr-lite | Rep |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in ("direct_generation_c1", "stars_best_of_4"):
            values = methods.get(method)
            if not values:
                continue
            lines.append(
                f"| `{method}` | {values.get('reward')} | "
                f"{values.get('mv_align')} | {values.get('csr')} | "
                f"{values.get('semantic_point_coverage')} | "
                f"{values.get('cider_lite')} | {values.get('rep')} |"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
