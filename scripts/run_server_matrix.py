from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    "mean_text_reasoning",
    "mean_bleu_4",
    "mean_rouge_l",
    "mean_meteor",
    "mean_cider_lite",
    "answer_hit_rate",
    "selling_point_coverage",
]

DATASETS = {
    "longvideobench": {
        "name": "longvideobench",
        "annotation_env": "LONGBENCH_ANNOTATION_PATH",
        "fallback_annotation_env": "LONGVIDEOBENCH_ANNOTATION_PATH",
        "video_root_env": "LONGBENCH_VIDEO_ROOT",
        "fallback_video_root_env": "LONGVIDEOBENCH_VIDEO_ROOT",
        "default_annotation": "",
        "default_video_root": "",
        "video_search_dirs": ["videos", "video", "clips", "all_videos", ""],
        "selling_points": ["视觉证据", "时间线线索", "答案支撑"],
    },
    "cgbench": {
        "name": "cg-bench",
        "annotation_env": "CGBENCH_ANNOTATION_PATH",
        "video_root_env": "CGBENCH_VIDEO_ROOT",
        "default_annotation": "",
        "default_video_root": "",
        "video_search_dirs": ["videos", "video", "clips", "all_videos", "clue_video", "clue_videos", ""],
        "selling_points": ["视觉证据", "关键信息", "行动引导"],
    },
    "videomme": {
        "name": "video-mme",
        "annotation_env": "VIDEOMME_ANNOTATION_PATH",
        "video_root_env": "VIDEOMME_VIDEO_ROOT",
        "default_annotation": "",
        "default_video_root": "",
        "video_search_dirs": ["videos", "video", "clips", "data", ""],
        "selling_points": ["视觉证据", "字幕线索", "答案支撑"],
    },
}

MODELS = {
    "llava_video_qwen2": {
        "id": "llava_video_7b_qwen2",
        "name": "LLaVA-Video-7B-Qwen2",
        "role": "main_multimodal_generator_and_aligner",
        "endpoint_env": "LLAVA_VIDEO_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
    },
    "internvl25_8b": {
        "id": "internvl25_8b",
        "name": "InternVL2.5-8B",
        "role": "baseline_multimodal_generator_and_aligner",
        "endpoint_env": "INTERNVL25_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
    },
    "videollama2_7b_16f": {
        "id": "videollama2_7b_16f",
        "name": "VideoLLaMA2-7B-16F",
        "role": "baseline_multimodal_generator_and_aligner",
        "endpoint_env": "VIDEOLLAMA2_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full server experiment matrix.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("EXPERIMENT_LIMIT", "0")))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("EXPERIMENT_OFFSET", "0")))
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-root", default=os.environ.get("EXPERIMENT_OUTPUT_ROOT", "outputs/server_matrix"))
    parser.add_argument("--config-dir", default=os.environ.get("EXPERIMENT_CONFIG_DIR", "outputs/server_matrix_configs"))
    parser.add_argument("--text-reward", choices=["deepseek", "llama33", "none"], default=os.environ.get("TEXT_REWARD_KIND", "none"))
    parser.add_argument("--no-strict-video", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-ranker", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset_key in args.datasets:
        for model_key in args.models:
            config = build_config(dataset_key, model_key, args)
            config_path = config_dir / f"{dataset_key}__{model_key}.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [
                sys.executable,
                "scripts/run_experiment.py",
                "--config",
                str(config_path),
                "--strict-model",
            ]
            if not args.no_strict_video:
                command.append("--strict-video")
            if not args.no_resume:
                command.append("--resume")
            if args.no_ranker:
                command.append("--no-ranker")

            print(f"\n>>> {dataset_key} / {model_key}")
            print(" ".join(command))
            if args.dry_run:
                rows.append({"dataset": dataset_key, "model": model_key, "config_path": str(config_path)})
                continue

            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            metrics_path = PROJECT_ROOT / config["output_dir"] / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "dataset": dataset_key,
                "model": model_key,
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
            }
            row.update({key: metrics.get(key, 0.0) for key in SUMMARY_KEYS})
            rows.append(row)

    summary_dir = PROJECT_ROOT / args.output_root
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (summary_dir / "summary.md").write_text(_markdown_summary(rows), encoding="utf-8")
    print(f"\nSummary written to {summary_dir}")


def build_config(dataset_key: str, model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    dataset = DATASETS[dataset_key]
    model = MODELS[model_key]
    annotation_path = _env(
        dataset["annotation_env"],
        dataset.get("fallback_annotation_env", ""),
        default=dataset["default_annotation"],
    )
    video_root = _env(
        dataset["video_root_env"],
        dataset.get("fallback_video_root_env", ""),
        default=dataset["default_video_root"],
    )
    if not annotation_path:
        raise RuntimeError(
            f"Missing annotation path for {dataset_key}. Set {dataset['annotation_env']}."
        )
    if not video_root:
        raise RuntimeError(
            f"Missing video root for {dataset_key}. Set {dataset['video_root_env']}."
        )
    endpoint_url = _env(model["endpoint_env"], "VLM_ENDPOINT_URL", default="")
    if not endpoint_url:
        raise RuntimeError(
            f"Missing endpoint for {model_key}. Set {model['endpoint_env']} or VLM_ENDPOINT_URL."
        )

    text_reward = _text_reward_endpoint(args.text_reward)
    if text_reward and not text_reward["endpoint_url"]:
        raise RuntimeError(
            "Missing text-only reward endpoint. Set TEXT_REWARD_ENDPOINT_URL "
            "or run with --text-reward none."
        )
    output_dir = f"{args.output_root}/{dataset_key}/{model_key}"
    return {
        "seed": 42,
        "data": {
            "name": dataset["name"],
            "annotation_path": annotation_path,
            "video_root": video_root,
            "video_search_dirs": dataset["video_search_dirs"],
            "offset": args.offset,
            "limit": args.limit,
        },
        "sampling": {
            "num_bins": 8,
            "frames_per_bin": 2,
            "max_frames": args.max_frames,
            "synthetic_size": args.image_size,
        },
        "generation": {
            "num_candidates": args.num_candidates,
            "target_duration_sec": 30,
            "segments": 5,
            "cta_position": "late",
            "pace": "medium",
            "information_density": "medium",
            "selling_points": dataset["selling_points"],
            "risk_terms": ["绝对", "第一", "治愈", "稳赚", "包治", "最低价"],
        },
        "reward": {
            "alignment_weight": 0.3,
            "readability_weight": 0.15,
            "rhythm_weight": 0.15,
            "control_weight": 0.2,
            "risk_weight": 0.05,
            "text_reasoning_weight": 0.15 if text_reward else 0.0,
            "preference_margin": 0.08,
        },
        "models": {
            "mode": "server_full_matrix",
            "active_video_model": model["id"],
            "active_text_reward_model": text_reward["id"] if text_reward else "",
            "allow_heavy_model_load": True,
            "endpoints": [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "role": model["role"],
                    "provider": "openai_compatible",
                    "adapter": model["adapter"],
                    "enabled": True,
                    "endpoint_url": endpoint_url,
                    "api_key_env": "VLM_API_KEY",
                    "max_frames": args.max_frames,
                    "max_new_tokens": 1200,
                    "temperature": float(os.environ.get("VLM_TEMPERATURE", "0.3")),
                    "request_timeout_sec": 300,
                    "retry_count": 2,
                    "notes": "Server-hosted OpenAI-compatible multimodal VLM endpoint.",
                },
                *([text_reward] if text_reward else []),
            ],
        },
        "output_dir": output_dir,
    }


def _text_reward_endpoint(kind: str) -> dict[str, Any] | None:
    if kind == "none":
        return None
    if kind == "llama33":
        return {
            "id": "llama33_70b_8bit_text_reward",
            "name": os.environ.get("LLAMA33_TEXT_REWARD_MODEL", "Llama-3.3-70B 8-bit"),
            "role": "text_only_reward_reasoning",
            "provider": "openai_compatible_text",
            "adapter": "chat_completions_text_reward",
            "enabled": True,
            "endpoint_url": _env("LLAMA33_TEXT_REWARD_ENDPOINT_URL", "TEXT_REWARD_ENDPOINT_URL", default=""),
            "api_key_env": "TEXT_REWARD_API_KEY",
            "max_new_tokens": 768,
            "temperature": 0.0,
            "request_timeout_sec": 240,
            "retry_count": 2,
            "quantization": "8bit",
            "notes": "Text-only reward/reasoning module; it never receives video frames.",
        }
    return {
        "id": "deepseek_r1_text_reward",
        "name": os.environ.get("DEEPSEEK_TEXT_REWARD_MODEL", "DeepSeek-R1"),
        "role": "text_only_reward_reasoning",
        "provider": "openai_compatible_text",
        "adapter": "chat_completions_text_reward",
        "enabled": True,
        "endpoint_url": _env("DEEPSEEK_TEXT_REWARD_ENDPOINT_URL", "TEXT_REWARD_ENDPOINT_URL", default=""),
        "api_key_env": "TEXT_REWARD_API_KEY",
        "max_new_tokens": 768,
        "temperature": 0.0,
        "request_timeout_sec": 240,
        "retry_count": 2,
        "notes": "Text-only reward/reasoning module; it never receives video frames.",
    }


def _env(primary: str, fallback: str = "", default: str = "") -> str:
    value = os.environ.get(primary, "")
    if not value and fallback:
        value = os.environ.get(fallback, "")
    return value or default


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Server Experiment Matrix",
        "",
        "| Dataset | Model | " + " | ".join(SUMMARY_KEYS) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(SUMMARY_KEYS)) + " |",
    ]
    for row in rows:
        values = [str(row.get(key, "")) for key in SUMMARY_KEYS]
        lines.append(f"| {row['dataset']} | {row['model']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
