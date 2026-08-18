from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from creative_video_exp.checkpoint_identity import (
    LOCAL_CHECKPOINT_KIND,
    checkpoint_identity_summary,
    load_checkpoint_manifest,
    validate_checkpoint_identity,
)

FORMAL_REWARD_WEIGHTS = {
    "alignment_weight": 6.0 / 17.0,
    "readability_weight": 3.0 / 17.0,
    "rhythm_weight": 3.0 / 17.0,
    "control_weight": 4.0 / 17.0,
    "risk_weight": 1.0 / 17.0,
}

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
    "best_mean_repetition_rate",
    "best_mean_english_token_rate",
    "non_latin_on_screen_text_candidate_rate_best",
    "mean_on_screen_text_non_latin_letter_rate_best",
    "generation_validation_retry_prompt_attempts",
    "generation_degenerate_repetition_early_stops",
    "failed_generation_attempt_records",
    "failed_generation_repetition_early_stops",
    "failed_generation_total_tokens",
    "failed_generation_seconds",
    "best_answer_hit_rate",
    "mean_generation_seconds",
    "mean_generation_input_tokens",
    "mean_generation_output_tokens",
    "mean_generation_total_tokens",
    "mean_reward_encoding_seconds",
    "mean_reward_scoring_seconds",
    "mean_total_pipeline_seconds",
    "mean_reward_runner_peak_allocated_mib",
    "mean_online_baseline_seconds",
    "mean_online_best_of_4_seconds",
    "mean_best_of_1_total_tokens",
    "mean_best_of_4_total_tokens",
    "mean_best_of_1_generation_requests",
    "mean_best_of_4_generation_requests",
    "risk_violation_rate_best",
    "mean_vlm_server_peak_allocated_mib",
    "mean_vlm_server_peak_reserved_mib",
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
        "salient_points": [],
    },
    "cgbench": {
        "name": "cg-bench",
        "annotation_env": "CGBENCH_ANNOTATION_PATH",
        "video_root_env": "CGBENCH_VIDEO_ROOT",
        "default_annotation": "",
        "default_video_root": "",
        "video_search_dirs": ["videos", "video", "clips", "all_videos", "clue_video", "clue_videos", ""],
        "salient_points": [],
    },
    "videomme": {
        "name": "video-mme",
        "annotation_env": "VIDEOMME_ANNOTATION_PATH",
        "video_root_env": "VIDEOMME_VIDEO_ROOT",
        "default_annotation": "",
        "default_video_root": "",
        "video_search_dirs": ["videos", "video", "clips", "data", ""],
        "salient_points": [],
    },
}

MODELS = {
    "llava_video_qwen2": {
        "id": "llava_video_7b_qwen2",
        "name": "LLaVA-Video-7B-Qwen2",
        "role": "main_multimodal_generator",
        "endpoint_env": "LLAVA_VIDEO_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
        "checkpoint_manifest_env": "LLAVA_VIDEO_CHECKPOINT_MANIFEST",
    },
    "internvl25_8b": {
        "id": "internvl25_8b",
        "name": "InternVL2.5-8B",
        "role": "baseline_multimodal_generator",
        "endpoint_env": "INTERNVL25_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
        "checkpoint_manifest_env": "INTERNVL25_CHECKPOINT_MANIFEST",
    },
    "videollama2_7b_16f": {
        "id": "videollama2_7b_16f",
        "name": "VideoLLaMA2-7B-16F",
        "role": "baseline_multimodal_generator",
        "endpoint_env": "VIDEOLLAMA2_ENDPOINT_URL",
        "adapter": "chat_completions_multimodal",
        "checkpoint_manifest_env": "VIDEOLLAMA2_CHECKPOINT_MANIFEST",
    },
}
DEFAULT_MODELS = ["llava_video_qwen2", "internvl25_8b", "videollama2_7b_16f"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full STARS experiment matrix.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODELS),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("EXPERIMENT_OUTPUT_ROOT", "outputs/full"),
    )
    parser.add_argument(
        "--config-dir",
        default=os.environ.get("EXPERIMENT_CONFIG_DIR", "outputs/full_configs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = PROJECT_ROOT / args.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model_bindings, reward_binding = _prepare_checkpoint_identities(args.models)

    for dataset_key in args.datasets:
        for model_key in args.models:
            config = build_config(
                dataset_key,
                model_key,
                args,
                model_identity=model_bindings[model_key]["identity"],
                model_manifest_path=model_bindings[model_key]["manifest_path"],
                reward_identity=reward_binding["identity"],
                reward_manifest_path=reward_binding["manifest_path"],
            )
            config_path = config_dir / f"{dataset_key}__{model_key}.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [
                sys.executable,
                "scripts/run_experiment.py",
                "--config",
                str(config_path),
                "--resume",
            ]

            print(f"\n>>> {dataset_key} / {model_key}")
            print(" ".join(command))
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


def build_config(
    dataset_key: str,
    model_key: str,
    args: argparse.Namespace,
    *,
    model_identity: dict[str, Any] | None = None,
    model_manifest_path: str = "",
    reward_identity: dict[str, Any] | None = None,
    reward_manifest_path: str = "",
) -> dict[str, Any]:
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
    endpoint_url = _env(
        model["endpoint_env"],
        "VLM_ENDPOINT_URL",
        default=model.get("default_endpoint", ""),
    )
    if not endpoint_url:
        raise RuntimeError(
            f"Missing endpoint for {model_key}. Set {model['endpoint_env']} or VLM_ENDPOINT_URL."
        )
    api_key_env = model.get("api_key_env", "VLM_API_KEY")
    output_dir = f"{args.output_root}/{dataset_key}/{model_key}"
    if model_identity is None:
        raise RuntimeError(
            f"Formal local model identity was not resolved for {model_key}."
        )
    validate_checkpoint_identity(model_identity)
    reward_vision = _reward_vision_endpoint(
        reward_identity,
        checkpoint_manifest_path=reward_manifest_path,
    )
    return {
        "experiment_version": "stars",
        "seed": 42,
        "data": {
            "name": dataset["name"],
            "annotation_path": annotation_path,
            "video_root": video_root,
            "video_search_dirs": dataset["video_search_dirs"],
        },
        "sampling": {
            "num_bins": 8,
            "frames_per_bin": 2,
            "max_frames": 16,
            "image_size": 224,
        },
        "generation": {
            "num_candidates": 4,
            "input_protocol": "visual_only",
            "prompt_version": "stars_visual_only_fixed_validation_retry_v2",
            "candidate_generation_protocol": "independent_single_candidate_calls",
            "parse_retry_count": 7,
            "pre_score_processing": "json_envelope_and_schema_canonicalization_only",
            "candidate_slot_failure_policy": "retain_invalid_slot_and_continue",
            "method_failure_aggregation": "conditional_and_failure_aware_effective",
            "target_duration_sec": 30,
            "segments": 5,
            "output_language": "English",
            "summary_position": "late",
            "pace": "medium",
            "information_density": "medium",
            "target_words_per_segment": 12,
            "min_words_per_segment": 6,
            "max_words_per_segment": 18,
            "salient_points": dataset["salient_points"],
            "risk_terms": [
                "absolute",
                "guaranteed",
                "cure",
                "risk-free",
                "lowest price",
                "number one",
                "best",
            ],
        },
        "reward": {
            **FORMAL_REWARD_WEIGHTS,
            "visual_grounding_balance": 0.5,
            "text_anchor_semantic_balance": 0.7,
        },
        "evaluation": {
            "semantic_point_coverage_enabled": True,
            "semantic_point_reference_policy": "annotation_only",
            "semantic_point_encoder": "active_reward_encoder_text_tower",
            "semantic_point_similarity_threshold": 0.50,
            "semantic_point_text_fields": [
                "narration",
                "on_screen_text",
                "salient_point",
            ],
        },
        "models": {
            "mode": "server_full_matrix",
            "active_video_model": model["id"],
            "active_reward_vision_model": reward_vision["id"],
            "endpoints": [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "role": model["role"],
                    "provider": model.get("provider", "openai_compatible"),
                    "adapter": model["adapter"],
                    "enabled": True,
                    "endpoint_url": endpoint_url,
                    "api_key_env": api_key_env,
                    "max_frames": 16,
                    "max_new_tokens": 900,
                    "temperature": float(os.environ.get("VLM_TEMPERATURE", "0.3")),
                    "request_timeout_sec": 300,
                    "retry_count": 2,
                    "seed": 42,
                    "checkpoint_identity": model_identity,
                    "checkpoint_manifest_path": model_manifest_path,
                    "notes": "Server-hosted OpenAI-compatible multimodal VLM endpoint.",
                },
                reward_vision,
            ],
        },
        "output_dir": output_dir,
    }


def _prepare_checkpoint_identities(
    model_keys: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reward_manifest_path = os.environ.get(
        "REWARD_VISION_CHECKPOINT_MANIFEST", ""
    ).strip()
    if not reward_manifest_path:
        raise RuntimeError(
            "Set REWARD_VISION_CHECKPOINT_MANIFEST to the formal SigLIP2 sidecar."
        )
    reward_model_path = os.environ.get("REWARD_VISION_MODEL_PATH", "").strip()
    if not reward_model_path:
        raise RuntimeError("Set REWARD_VISION_MODEL_PATH to the local SigLIP2 checkpoint.")
    reward_manifest = load_checkpoint_manifest(
        reward_manifest_path,
        model_path=reward_model_path,
        verify_files=True,
    )
    reward_identity = checkpoint_identity_summary(reward_manifest)
    if reward_identity["model_id"] != "google/siglip2-so400m-patch14-384":
        raise RuntimeError("Reward checkpoint sidecar identifies the wrong model.")

    bindings: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        model = MODELS[model_key]
        manifest_env = str(model.get("checkpoint_manifest_env", ""))
        manifest_path = os.environ.get(manifest_env, "").strip()
        if not manifest_path:
            raise RuntimeError(
                f"Set {manifest_env} to the checkpoint sidecar for {model_key}."
            )
        expected_manifest = load_checkpoint_manifest(manifest_path)
        expected_identity = checkpoint_identity_summary(expected_manifest)
        if expected_identity["model_id"] != model["name"]:
            raise RuntimeError(
                f"{manifest_env} model_id does not equal {model['name']!r}."
            )
        endpoint_url = _env(
            model["endpoint_env"],
            "VLM_ENDPOINT_URL",
            default=model.get("default_endpoint", ""),
        )
        if not endpoint_url:
            raise RuntimeError(
                f"Missing endpoint for {model_key}. Set {model['endpoint_env']}."
            )
        observed_identity = _query_local_service_checkpoint_identity(
            endpoint_url,
            api_key_env=model.get("api_key_env", "VLM_API_KEY"),
            expected_model_id=model["name"],
        )
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"Live service checkpoint identity differs from {manifest_env}."
            )
        bindings[model_key] = {
            "identity": expected_identity,
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        }
    return bindings, {
        "identity": reward_identity,
        "manifest_path": str(Path(reward_manifest_path).expanduser().resolve()),
    }


def _query_local_service_checkpoint_identity(
    endpoint_url: str,
    *,
    api_key_env: str,
    expected_model_id: str,
) -> dict[str, Any]:
    parts = urllib.parse.urlsplit(endpoint_url)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    models_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, f"{path}/models", "", "")
    )
    headers: dict[str, str] = {}
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not audit the live model service at {models_url}: {exc}"
        ) from exc
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    matches = [
        item
        for item in rows
        if isinstance(item, dict) and item.get("id") == expected_model_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Service must advertise exactly one {expected_model_id!r} model identity."
        )
    identity = matches[0].get("checkpoint_identity", {})
    validate_checkpoint_identity(identity)
    if identity.get("kind") != LOCAL_CHECKPOINT_KIND:
        raise RuntimeError("Local service returned a non-local checkpoint identity.")
    if identity.get("model_id") != expected_model_id:
        raise RuntimeError("Service checkpoint identity model_id is inconsistent.")
    return identity


def _reward_vision_endpoint(
    checkpoint_identity: dict[str, Any] | None = None,
    *,
    checkpoint_manifest_path: str = "",
) -> dict[str, Any]:
    if checkpoint_identity is None:
        raise RuntimeError("Formal reward checkpoint identity was not resolved.")
    validate_checkpoint_identity(checkpoint_identity)
    return {
        "id": "siglip2_so400m_patch14_384_reward",
        "name": "google/siglip2-so400m-patch14-384",
        "role": "frozen_vision_text_reward_encoder",
        "provider": "huggingface_local",
        "adapter": "siglip2_frame_encoder",
        "enabled": True,
        "local_path": os.environ["REWARD_VISION_MODEL_PATH"],
        "device_map": os.environ.get("REWARD_VISION_DEVICE", "cuda:0"),
        "dtype": os.environ.get("REWARD_VISION_DTYPE", "bfloat16"),
        "trust_remote_code": False,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_manifest_path": checkpoint_manifest_path,
        "notes": (
            "Frozen independent SigLIP2 encoder for frame embeddings, text embeddings, "
            "coarse visual tags, and formal alignment reward."
        ),
    }


def _env(primary: str, fallback: str = "", default: str = "") -> str:
    value = os.environ.get(primary, "")
    if not value and fallback:
        value = os.environ.get(fallback, "")
    return value or default


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# STARS Experiment Matrix",
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
