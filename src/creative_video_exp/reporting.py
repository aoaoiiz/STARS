from __future__ import annotations

from pathlib import Path
from typing import Any


SUMMARY_KEYS = [
    "num_samples",
    "num_candidates",
    "preference_pairs",
    "real_video_rate",
    "multimodal_generation_rate",
    "control_success_rate_best",
    "constraint_violation_rate_best",
    "mean_best_reward",
    "mean_best_alignment",
    "mean_bleu_4",
    "mean_rouge_l",
    "mean_meteor",
    "mean_cider_lite",
    "mean_distinct_1",
    "mean_distinct_2",
    "mean_repetition_rate",
    "answer_hit_rate",
    "selling_point_coverage",
]


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    print("\n=== Metrics Summary ===")
    for key in SUMMARY_KEYS:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    if "source_kind_counts" in metrics:
        print(f"source_kind_counts: {metrics['source_kind_counts']}")
    if "generation_source_counts" in metrics:
        print(f"generation_source_counts: {metrics['generation_source_counts']}")
    runtime = metrics.get("model_runtime", {})
    if runtime:
        print("\n=== Model Runtime ===")
        print(f"video_model: {runtime.get('video_model')}")
        print(f"script_generation_model: {runtime.get('script_generation_model')}")
        print(f"text_reward_model: {runtime.get('text_reward_model')}")


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
    lines.append("- `cider_lite` is a dependency-free CIDEr-style n-gram F1 proxy for local smoke tests, not the official COCO CIDEr implementation.")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
