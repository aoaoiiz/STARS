from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import atomic_write_text


SUMMARY_KEYS = [
    "requested_num_samples",
    "completed_num_samples",
    "fully_accounted_sample_count",
    "requested_candidate_slot_count",
    "valid_candidate_slot_count",
    "failed_candidate_slot_count",
    "candidate_slot_success_rate",
    "full_candidate_pool_sample_count",
    "full_candidate_pool_sample_rate",
    "any_valid_candidate_sample_count",
    "any_valid_candidate_sample_rate",
    "generation_success_rate",
    "unresolved_generation_failure_count",
    "evidence_eligible",
    "num_samples",
    "num_candidates",
    "preference_pairs",
    "real_video_rate",
    "multimodal_generation_rate",
    "control_success_rate_best",
    "control_violation_counts_best",
    "constraint_violation_rate_best",
    "mean_best_reward",
    "mean_best_alignment",
    "best_mean_bleu_4",
    "best_mean_rouge_l",
    "best_mean_meteor",
    "best_mean_semantic_point_coverage",
    "semantic_point_valid_sample_count",
    "semantic_point_valid_sample_rate",
    "best_mean_cider_lite",
    "best_mean_distinct_1",
    "best_mean_distinct_2",
    "best_mean_repetition_rate",
    "best_answer_hit_rate",
    "best_mean_english_token_rate",
    "non_latin_on_screen_text_candidate_rate_best",
    "mean_on_screen_text_non_latin_letter_rate_best",
    "generation_validation_retry_prompt_attempts",
    "generation_degenerate_repetition_early_stops",
    "failed_generation_attempt_records",
    "failed_generation_repetition_early_stops",
    "failed_generation_total_tokens",
    "failed_generation_seconds",
    "mean_generation_seconds",
    "mean_reward_encoding_seconds",
    "mean_reward_scoring_seconds",
    "mean_total_pipeline_seconds",
    "mean_reward_runner_peak_allocated_mib",
    "mean_online_baseline_seconds",
    "mean_online_best_of_2_seconds",
    "mean_online_best_of_4_seconds",
    "mean_best_of_1_total_tokens",
    "mean_best_of_2_total_tokens",
    "mean_best_of_4_total_tokens",
    "mean_best_of_1_generation_requests",
    "mean_best_of_2_generation_requests",
    "mean_best_of_4_generation_requests",
    "risk_violation_rate_best",
    "mean_vlm_server_peak_allocated_mib",
    "mean_vlm_server_peak_reserved_mib",
]


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    print("\n=== Metrics Summary ===")
    for key in SUMMARY_KEYS:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    if "source_kind_counts" in metrics:
        print(f"source_kind_counts: {metrics['source_kind_counts']}")
    if "decode_strategy_counts" in metrics:
        print(f"decode_strategy_counts: {metrics['decode_strategy_counts']}")
    if "sampling_fallback_count" in metrics:
        print(f"sampling_fallback_count: {metrics['sampling_fallback_count']}")
        print(f"sampling_fallback_rate: {metrics['sampling_fallback_rate']}")
    if "generation_source_counts" in metrics:
        print(f"generation_source_counts: {metrics['generation_source_counts']}")
    runtime = metrics.get("model_runtime", {})
    if runtime:
        print("\n=== Model Runtime ===")
        print(f"video_model: {runtime.get('video_model')}")
        print(f"script_generation_model: {runtime.get('script_generation_model')}")


def write_markdown_report(path: str | Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Experiment Metrics",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in SUMMARY_KEYS:
        if key in metrics:
            lines.append(f"| `{key}` | {metrics[key]} |")
    if "source_kind_counts" in metrics:
        lines.append(f"| `source_kind_counts` | `{metrics['source_kind_counts']}` |")
    if "decode_strategy_counts" in metrics:
        lines.append(f"| `decode_strategy_counts` | `{metrics['decode_strategy_counts']}` |")
    if "sampling_fallback_count" in metrics:
        lines.append(f"| `sampling_fallback_count` | {metrics['sampling_fallback_count']} |")
        lines.append(f"| `sampling_fallback_rate` | {metrics['sampling_fallback_rate']} |")
    if "generation_source_counts" in metrics:
        lines.append(f"| `generation_source_counts` | `{metrics['generation_source_counts']}` |")

    lines.extend(["", "## Model Runtime", ""])
    runtime = metrics.get("model_runtime", {})
    for key, value in runtime.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Notes", ""])
    lines.append("- `real_video_rate` below 1.0 means some samples used deterministic synthetic frames because local video files were not found or not extracted.")
    lines.append("- `multimodal_generation_rate` below 1.0 means some samples used the local rule fallback instead of a server VLM.")
    lines.append("- BLEU/ROUGE/METEOR are lexical weak-reference metrics computed against available dataset text fields; for final paper results, report them together with control, safety, and human/LLM preference metrics.")
    lines.append("- `cider_lite` is a dependency-free CIDEr-style n-gram F1 proxy, not the official COCO CIDEr implementation.")
    lines.append("- `best_mean_semantic_point_coverage` is SPC: thresholded semantic recall over evaluation-only annotation points using the frozen reward encoder text tower.")
    lines.append("- Samples without semantic-point references have SPC=null and are excluded from the SPC mean; valid sample count and rate are reported explicitly.")
    lines.append("- Reward is the deployment-time selection objective and is not treated as independent external evidence.")
    lines.append("- Direct Generation and Best-of-1 must be identical because both return C1 from the same ordered pool.")
    lines.append("- `evidence_eligible` requires every fixed-manifest sample and every requested C1-C4 slot to have an auditable terminal status; an explicitly failed slot does not make the run incomplete.")
    lines.append("- Conditional method metrics summarize samples with a valid selected output; failure-aware effective metrics retain all requested samples, assign 0 to failed higher-is-better metrics, and assign 1 to failed Rep.")
    lines.append("- Formal candidate-pool confidence intervals are computed by cluster bootstrap over video_id.")

    atomic_write_text(path, "\n".join(lines) + "\n")
